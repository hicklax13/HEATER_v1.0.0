"""SFH D guard (2026-05-20): every bootstrap phase must write refresh_log
on the error path.

Background:
  When a `_bootstrap_*` function caught an Exception and just logged-and-
  returned an error string, refresh_log was never updated. The next
  bootstrap saw a stale last_refresh timestamp from the last *successful*
  run and the operator's Data Status dashboard kept showing the phase as
  healthy. We discovered six phases stuck at refresh_log timestamps
  7–17 days old despite running daily (ecr_consensus, catcher_framing,
  umpire_tendencies, pvb_splits, draft_results, plus injury_writeback
  via a different mechanism).

This guard prevents regression of the pattern. It AST-scans
`src/data_bootstrap.py` and asserts that every outermost ``except``
handler in a ``_bootstrap_*`` function either:

  (a) Calls ``update_refresh_log`` / ``update_refresh_log_auto``, or
  (b) Re-raises (failure propagates upward, parent records it), or
  (c) Catches ``ImportError`` specifically — these are the established
      "skip if optional dep not installed" patterns. They could also
      write status="skipped" but doing so on every bootstrap is noisy
      for missing-dep cases; we exempt them explicitly.

Companion to ``test_no_unguarded_update_refresh_log.py`` (which guards
the SUCCESS path uses ``update_refresh_log_auto`` not bare
``update_refresh_log(..., "success")``). Together they cover both
write-side gaps in the refresh_log contract.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_REFRESH_LOG_FUNCS = {"update_refresh_log", "update_refresh_log_auto"}


def _calls_refresh_log(node: ast.AST) -> bool:
    """True if ``node`` (or any descendant) calls update_refresh_log[_auto]."""
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            f = child.func
            name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
            if name in _REFRESH_LOG_FUNCS:
                return True
    return False


def _has_raise(handler: ast.ExceptHandler) -> bool:
    """True if the handler re-raises (so parent gets to record)."""
    for child in ast.walk(handler):
        if isinstance(child, ast.Raise):
            return True
    return False


def _catches_only_import_error(handler: ast.ExceptHandler) -> bool:
    """True if this except handler catches ImportError only — the
    'optional dep not installed' pattern, exempt from the guard."""
    if handler.type is None:
        return False
    if isinstance(handler.type, ast.Name) and handler.type.id == "ImportError":
        return True
    if isinstance(handler.type, ast.Tuple):
        return all(isinstance(e, ast.Name) and e.id == "ImportError" for e in handler.type.elts)
    return False


def _outer_except_handlers(func: ast.FunctionDef) -> list[tuple[ast.ExceptHandler, int]]:
    """Return (handler, lineno) for every except handler whose Try is a
    direct child of the function body (i.e. the outermost wrapping try)."""
    out: list[tuple[ast.ExceptHandler, int]] = []
    for stmt in func.body:
        if isinstance(stmt, ast.Try):
            for h in stmt.handlers:
                out.append((h, h.lineno))
    return out


# 2026-05-20 SFH MED-1 follow-up: widened the prefix set so the guard also
# covers `_enrich_pitcher_positions` (and any future `_enrich_*` /
# `_refresh_*` / `_merge_*` / `_sync_*` orchestrator-called phases that
# write to refresh_log on the success path). If you add a new orchestrator-
# called phase that follows a different naming convention, add it here.
_PHASE_PREFIXES = ("_bootstrap_", "_enrich_", "_refresh_", "_merge_", "_sync_")


def test_every_outermost_except_in_bootstrap_phase_records_refresh_log():
    """Outermost except in each orchestrator-called phase function must
    record refresh_log on failure, or re-raise, or be an ImportError
    dep-check. Phase functions are identified by name prefix (see
    _PHASE_PREFIXES)."""
    src = pathlib.Path("src/data_bootstrap.py").read_text()
    tree = ast.parse(src)

    failures: list[str] = []
    phase_count = 0

    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name.startswith(_PHASE_PREFIXES)):
            continue
        phase_count += 1
        for handler, lineno in _outer_except_handlers(node):
            if _calls_refresh_log(handler):
                continue
            if _has_raise(handler):
                continue
            if _catches_only_import_error(handler):
                continue
            failures.append(
                f"  {node.name} (function def L{node.lineno}) has outer "
                f"except L{lineno} that does NOT call update_refresh_log, "
                f"raise, or catch ImportError-only → refresh_log will be "
                f"silently stale on failure"
            )

    # Sentinel: catches the case where someone renames every _bootstrap_* off
    # the prefix and this test silently degrades to "no phases scanned, all
    # green." Align with the test_data_bootstrap mock fixture sentinel.
    assert phase_count >= 25, (
        f"AST scan found only {phase_count} `_bootstrap_*` functions; "
        f"expected >= 25 (most-recent count was ~29). Either a phase was "
        f"renamed off the prefix (the guard now covers less code — bad) "
        f"or the expected count is stale (bump it after a phase rename)."
    )

    if failures:
        msg = (
            f"\n{len(failures)} phase function(s) have outer except handlers "
            f"that silently skip refresh_log on failure — operator dashboards "
            f"will show stale-but-healthy for these. Add "
            f'`update_refresh_log(<source>, "error", message=str(exc)[:200])` '
            f"or re-raise:\n\n" + "\n".join(failures)
        )
        pytest.fail(msg)

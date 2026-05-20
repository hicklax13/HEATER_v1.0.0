"""SFH M1 guard (2026-05-20): _bootstrap_pvb_splits must commit per-batter.

Background:
  pvb_splits runs in a ThreadPoolExecutor(max_workers=3) group alongside
  umpire_tendencies and catcher_framing. All three open write connections
  to the same SQLite DB. Without per-batter commits, pvb_splits holds a
  single write transaction across the entire 50-batter loop — including
  ~1-3s pybaseball Statcast fetches PER batter (network-bound). The held
  lock easily exceeds the 60s busy_timeout that PR #69 set, causing the
  other two phases to abort with "database is locked" and silently leave
  refresh_log stale at last week's "success".

  Root cause: ONE conn.commit() at the bottom of the function instead of
  one per batter iteration. PR #69's busy_timeout bump was a band-aid;
  this guard pins the actual fix.

This test AST-scans `_bootstrap_pvb_splits` and asserts that the only
`conn.commit()` call inside the body is INSIDE the outer batter
``for idx, (_, hitter) in enumerate(hitter_sample.iterrows())`` loop —
NOT at the end of the wrapping ``try:`` block.

Regression prevention: a future refactor that moves the commit back to
end-of-function (or removes it entirely) fails this guard.
"""

from __future__ import annotations

import ast
import pathlib

import pytest


def _find_function(tree: ast.AST, name: str) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _is_conn_commit_call(node: ast.AST) -> bool:
    """True if node is `conn.commit()` syntax."""
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    if not isinstance(f, ast.Attribute):
        return False
    if f.attr != "commit":
        return False
    val = f.value
    return isinstance(val, ast.Name) and val.id == "conn"


def _find_batter_loop(func: ast.FunctionDef) -> ast.For | None:
    """Return the outer batter For loop (iterating hitter_sample.iterrows())."""
    for node in ast.walk(func):
        if not isinstance(node, ast.For):
            continue
        # Look for: for ... in enumerate(hitter_sample.iterrows()):
        it = node.iter
        if isinstance(it, ast.Call):
            f = it.func
            if isinstance(f, ast.Name) and f.id == "enumerate":
                args = it.args
                if (
                    args
                    and isinstance(args[0], ast.Call)
                    and isinstance(args[0].func, ast.Attribute)
                    and args[0].func.attr == "iterrows"
                ):
                    inner = args[0].func.value
                    if isinstance(inner, ast.Name) and inner.id == "hitter_sample":
                        return node
    return None


def _commit_calls_in_subtree(node: ast.AST) -> list[ast.Call]:
    """All `conn.commit()` Call nodes anywhere under node."""
    return [n for n in ast.walk(node) if _is_conn_commit_call(n)]


def test_pvb_splits_commits_inside_batter_loop():
    """Regression guard for SFH M1: pvb_splits must commit per-batter so the
    write lock releases between Statcast fetches, not after all 50 batters."""
    src = pathlib.Path("src/data_bootstrap.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    func = _find_function(tree, "_bootstrap_pvb_splits")
    assert func is not None, (
        "_bootstrap_pvb_splits function not found in src/data_bootstrap.py — renamed? Update this guard accordingly."
    )

    batter_loop = _find_batter_loop(func)
    assert batter_loop is not None, (
        "Could not locate the outer batter For loop "
        "(`for idx, (_, hitter) in enumerate(hitter_sample.iterrows())`). "
        "If it was renamed, update _find_batter_loop accordingly."
    )

    commits_in_loop = _commit_calls_in_subtree(batter_loop)
    assert len(commits_in_loop) >= 1, (
        "No `conn.commit()` found INSIDE the outer batter loop. "
        "SFH M1 fix requires the commit to live inside this loop so the "
        "write lock releases between batters — otherwise parallel writers "
        "(umpire_tendencies, catcher_framing) abort with 'database is "
        "locked'."
    )

    # Every conn.commit() call inside the function should be inside the
    # batter loop. A bare commit AFTER the loop (the pre-PR pattern) would
    # mean we still hold the lock across batters — which is the bug.
    all_commits = _commit_calls_in_subtree(func)
    loop_commit_lines = {c.lineno for c in commits_in_loop}
    outside_commits = [c for c in all_commits if c.lineno not in loop_commit_lines]
    assert not outside_commits, (
        f"Found {len(outside_commits)} conn.commit() call(s) OUTSIDE the "
        f"batter loop at line(s) {[c.lineno for c in outside_commits]}. "
        f"SFH M1 fix forbids these — they'd hold the write lock across "
        f"the entire phase and starve parallel writers. Move the commit "
        f"inside the batter loop body."
    )


def test_pvb_splits_function_still_exists():
    """Sentinel: if pvb_splits is renamed/removed, surface that explicitly
    rather than silently passing because _find_function returned None and
    the assertion just never reaches the commit-position check."""
    src = pathlib.Path("src/data_bootstrap.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    func = _find_function(tree, "_bootstrap_pvb_splits")
    if func is None:
        pytest.fail(
            "_bootstrap_pvb_splits not in src/data_bootstrap.py — either it "
            "was renamed (update both guards in this file) or removed (then "
            "delete this test file)."
        )

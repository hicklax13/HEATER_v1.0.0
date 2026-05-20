"""SFH M2 guard (2026-05-20): pybaseball batting_stats must not be called
with pos= kwarg.

Background:
  pybaseball 2.2.7's `batting_stats(year, qual=0, pos="c")` raises
  `TypeError: unexpected keyword argument 'pos'` — the kwarg is passed
  through to `FangraphsDataTable.fetch()` which rejects it. This silently
  broke the FanGraphs-batting-stats fallback inside Tier 1 of
  _bootstrap_catcher_framing for an unknown number of bootstrap runs;
  Tier 2 (browser-headers Savant scrape) covered for it.

This AST guard scans src/data_bootstrap.py for any call to batting_stats
and asserts that none of them pass pos= as a kwarg. A future re-add
of the kwarg (e.g., when copy-pasting old code) fails the test.
"""

from __future__ import annotations

import ast
import pathlib


def _find_batting_stats_calls(tree: ast.AST) -> list[ast.Call]:
    """Return all `batting_stats(...)` Call nodes in the tree."""
    out: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        # Match both `batting_stats(...)` and `pybaseball.batting_stats(...)`.
        name = None
        if isinstance(f, ast.Name):
            name = f.id
        elif isinstance(f, ast.Attribute):
            name = f.attr
        if name == "batting_stats":
            out.append(node)
    return out


def test_no_pos_kwarg_on_batting_stats():
    """SFH M2: pybaseball.batting_stats no longer accepts pos=. Any call
    that passes it raises TypeError and the fallback path silently dies."""
    src = pathlib.Path("src/data_bootstrap.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    calls = _find_batting_stats_calls(tree)
    # Sentinel: confirm we actually found the call site (catches the case
    # where the call is renamed / removed and this test silently degrades
    # to "no calls scanned, all green").
    assert calls, (
        "No batting_stats(...) call found in src/data_bootstrap.py. "
        "If the FanGraphs fallback path was removed, delete this guard."
    )

    violations = []
    for call in calls:
        for kw in call.keywords:
            if kw.arg == "pos":
                violations.append(
                    f"L{call.lineno}: batting_stats(...) called with pos= kwarg "
                    f"— pybaseball 2.2.7 rejects this with TypeError. "
                    f"Drop the kwarg and filter the returned DataFrame by "
                    f"position column post-hoc."
                )

    assert not violations, "\n".join(violations)

"""Structural guards: the weekly add budget is canonical (10, from league_rules).

Phase 0 of the Pitcher Streaming Analyzer plan. ``src/optimizer/streaming.py``
shipped with ``WEEKLY_ADDS_BUDGET = 7`` while FourzynBurn allows 10 adds+trades
per matchup week — its last-add penalty fired three adds early and its
budget-exhausted guard cut streaming off at 7. The canonical limit now lives in
``src.league_rules.WEEKLY_TRANSACTION_LIMIT`` and every consumer derives from it.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def test_league_rules_canonical_limit():
    from src.league_rules import WEEKLY_TRANSACTION_LIMIT

    assert WEEKLY_TRANSACTION_LIMIT == 10


def test_streaming_budget_sourced_from_league_rules():
    from src.league_rules import WEEKLY_TRANSACTION_LIMIT
    from src.optimizer.streaming import WEEKLY_ADDS_BUDGET

    assert WEEKLY_ADDS_BUDGET == WEEKLY_TRANSACTION_LIMIT, (
        "streaming.WEEKLY_ADDS_BUDGET must mirror the canonical "
        "league_rules.WEEKLY_TRANSACTION_LIMIT (10), not the stale 7"
    )


def test_streaming_no_inline_budget_literal():
    """The 7 literal must not survive as the budget definition."""
    src = (_REPO / "src" / "optimizer" / "streaming.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for tgt in targets:
                if isinstance(tgt, ast.Name) and tgt.id == "WEEKLY_ADDS_BUDGET":
                    assert not isinstance(node.value, ast.Constant), (
                        "WEEKLY_ADDS_BUDGET must be derived from league_rules.WEEKLY_TRANSACTION_LIMIT, not a literal"
                    )


def test_transactions_remaining_default_uses_canonical_limit():
    from src.league_rules import WEEKLY_TRANSACTION_LIMIT, get_transactions_remaining

    assert get_transactions_remaining(0) == WEEKLY_TRANSACTION_LIMIT
    assert get_transactions_remaining(WEEKLY_TRANSACTION_LIMIT) == 0


def test_rank_streaming_candidates_honors_ten_add_budget():
    """adds_used=7 must still stream (old budget of 7 returned [])."""
    from src.optimizer.streaming import rank_streaming_candidates

    pitchers = [
        {
            "player_name": "Streamer A",
            "team": "SEA",
            "k": 7.0,
            "w": 0.6,
            "era": 3.20,
            "whip": 1.10,
            "ip": 5.8,
        }
    ]
    assert rank_streaming_candidates(pitchers, adds_used_this_week=7), (
        "7 adds used of a 10-add budget must NOT exhaust streaming"
    )
    assert rank_streaming_candidates(pitchers, adds_used_this_week=10) == []

    # Last-add penalty fires at 9-of-10 used, not 6-of-7.
    full = rank_streaming_candidates(pitchers, adds_used_this_week=0)
    last = rank_streaming_candidates(pitchers, adds_used_this_week=9)
    assert last and full
    assert abs(last[0]["net_value"] - full[0]["net_value"] * 0.5) < 1e-6


def test_rank_streaming_candidates_budget_override():
    """An explicit weekly_adds_budget kwarg overrides the canonical default."""
    from src.optimizer.streaming import rank_streaming_candidates

    pitchers = [
        {
            "player_name": "Streamer A",
            "team": "SEA",
            "k": 7.0,
            "w": 0.6,
            "era": 3.20,
            "whip": 1.10,
            "ip": 5.8,
        }
    ]
    assert rank_streaming_candidates(pitchers, adds_used_this_week=7, weekly_adds_budget=7) == []
    assert rank_streaming_candidates(pitchers, adds_used_this_week=7, weekly_adds_budget=12)


def test_stream_analyzer_does_not_import_streaming_budget():
    """stream_analyzer must source the budget from league_rules, never streaming."""
    path = _REPO / "src" / "optimizer" / "stream_analyzer.py"
    assert path.exists(), "src/optimizer/stream_analyzer.py must exist (Phase 0 skeleton)"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and "streaming" in node.module:
            names = {a.name for a in node.names}
            assert "WEEKLY_ADDS_BUDGET" not in names, (
                "stream_analyzer.py must not import WEEKLY_ADDS_BUDGET from "
                "streaming — use league_rules.WEEKLY_TRANSACTION_LIMIT"
            )

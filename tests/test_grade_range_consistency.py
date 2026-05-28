"""Structural-invariant guard: grade_range["grade"] always equals result["grade"].

Bug E (2026-05-23 validation): the pre-fix engine showed `grade = B`
(MC-overridden) alongside `grade_range = F → F → D` (computed from Phase
1 surplus pre-MC). Users saw two contradictory grade signals next to
each other in the UI.

After Bug A's fix (Phase 1 SGP is the grade authority, MC no longer
overrides), both `grade` and `grade_range["grade"]` are derived from the
SAME Phase 1 surplus_sgp value, so they must match. This test locks
that invariant.

If MC override behavior is ever re-introduced, this test will fail —
forcing the implementer to also update grade_range so the displayed
signals stay consistent.
"""

from __future__ import annotations

import pandas as pd

from src.engine.output.trade_evaluator import evaluate_trade


def _minimal_pool() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "A",
                "player_name": "A",
                "is_hitter": 1,
                "positions": "OF",
                "r": 80,
                "hr": 25,
                "rbi": 80,
                "sb": 10,
                "h": 130,
                "ab": 500,
                "bb": 55,
                "hbp": 5,
                "sf": 5,
                "pa": 570,
                "avg": 0.260,
                "obp": 0.330,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "ip": 0,
                "er": 0,
                "bb_allowed": 0,
                "h_allowed": 0,
                "era": 0,
                "whip": 0,
            },
            {
                "player_id": 2,
                "name": "B",
                "player_name": "B",
                "is_hitter": 1,
                "positions": "OF",
                "r": 75,
                "hr": 20,
                "rbi": 75,
                "sb": 8,
                "h": 125,
                "ab": 500,
                "bb": 55,
                "hbp": 5,
                "sf": 5,
                "pa": 570,
                "avg": 0.250,
                "obp": 0.330,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "ip": 0,
                "er": 0,
                "bb_allowed": 0,
                "h_allowed": 0,
                "era": 0,
                "whip": 0,
            },
        ]
    )


def test_grade_matches_grade_range_center_phase1_only() -> None:
    """grade and grade_range['grade'] must match when MC is disabled."""
    pool = _minimal_pool()
    result = evaluate_trade(
        giving_ids=[1],
        receiving_ids=[2],
        user_roster_ids=[1],
        player_pool=pool,
        enable_mc=False,
        enable_context=False,
        enable_game_theory=False,
        apply_ytd_blend=False,
    )
    assert result["grade"] == result["grade_range"]["grade"], (
        f"Bug E regression: result['grade']={result['grade']!r} but "
        f"result['grade_range']['grade']={result['grade_range']['grade']!r}. "
        f"They MUST match since both derive from Phase 1 surplus_sgp."
    )


def test_grade_matches_grade_range_center_with_mc() -> None:
    """Even when MC runs, grade still matches grade_range['grade'] — both
    are Phase 1 derived. MC contributes mc_grade only, never overriding grade."""
    pool = _minimal_pool()
    result = evaluate_trade(
        giving_ids=[1],
        receiving_ids=[2],
        user_roster_ids=[1],
        player_pool=pool,
        enable_mc=True,
        n_sims=500,  # small for test speed
        enable_context=False,
        enable_game_theory=False,
        apply_ytd_blend=False,
    )
    # Phase 1 grade and grade_range center MUST match
    assert result["grade"] == result["grade_range"]["grade"], (
        f"Bug E regression with MC enabled: "
        f"grade={result['grade']!r} but grade_range center={result['grade_range']['grade']!r}"
    )
    # MC's own grader output is exposed under mc_grade — may or may not differ
    assert "mc_grade" in result


def test_grade_range_low_le_center_le_high() -> None:
    """Ordering invariant: low <= center <= high (in grade-value terms)."""
    pool = _minimal_pool()
    result = evaluate_trade(
        giving_ids=[1],
        receiving_ids=[2],
        user_roster_ids=[1],
        player_pool=pool,
        enable_mc=False,
        enable_context=False,
        enable_game_theory=False,
        apply_ytd_blend=False,
    )
    gr = result["grade_range"]
    # Map letter grades to numeric for comparison (higher = better)
    grade_order = ["F", "D", "C-", "C", "C+", "B-", "B", "B+", "A-", "A", "A+"]
    low_idx = grade_order.index(gr["grade_low"])
    center_idx = grade_order.index(gr["grade"])
    high_idx = grade_order.index(gr["grade_high"])
    assert low_idx <= center_idx <= high_idx, (
        f"grade_range ordering violated: "
        f"low={gr['grade_low']!r}({low_idx}) center={gr['grade']!r}({center_idx}) "
        f"high={gr['grade_high']!r}({high_idx})"
    )


def test_grade_range_confidence_field_valid() -> None:
    """confidence is one of {high, medium, low}."""
    pool = _minimal_pool()
    result = evaluate_trade(
        giving_ids=[1],
        receiving_ids=[2],
        user_roster_ids=[1],
        player_pool=pool,
        enable_mc=False,
        enable_context=False,
        enable_game_theory=False,
        apply_ytd_blend=False,
    )
    assert result["grade_range"]["confidence"] in ("high", "medium", "low")

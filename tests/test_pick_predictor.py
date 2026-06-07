# tests/test_pick_predictor.py
"""Tests for pick predictor survival curves."""

from __future__ import annotations

import pytest

from src.pick_predictor import (
    POSITION_SHAPE_PARAMS,
    _future_user_picks,
    compute_survival_curve,
    weibull_survival,
)


def test_weibull_survival_decreases():
    s1 = weibull_survival(5, 1.0, 30.0)
    s2 = weibull_survival(15, 1.0, 30.0)
    assert s1 > s2


def test_weibull_survival_bounds():
    s = weibull_survival(1, 1.0, 75.0)
    assert 0.0 <= s <= 1.0


def test_weibull_shape_above_one():
    s_early = weibull_survival(5, 1.4, 30.0)
    s_late = weibull_survival(20, 1.4, 30.0)
    assert s_late < s_early


def test_weibull_zero_scale():
    s = weibull_survival(10, 1.0, 0.0)
    assert s == 1.0


def test_position_shape_params_complete():
    for pos in ["C", "SS", "2B", "3B", "1B", "OF", "SP", "RP"]:
        assert pos in POSITION_SHAPE_PARAMS


def test_curve_monotonically_decreasing():
    curve = compute_survival_curve(
        player_adp=50.0,
        player_positions="SS",
        current_pick=10,
        user_team_index=0,
        num_teams=12,
        num_rounds=23,
    )
    probs = [pt["p_available"] for pt in curve]
    for i in range(1, len(probs)):
        assert probs[i] <= probs[i - 1] + 0.01


def test_curve_length():
    curve = compute_survival_curve(
        player_adp=50.0,
        player_positions="1B",
        current_pick=10,
        user_team_index=0,
        num_teams=12,
        num_rounds=23,
    )
    assert len(curve) > 0
    assert all("pick" in pt and "round" in pt and "p_available" in pt for pt in curve)


def test_curve_past_adp_low_survival():
    curve = compute_survival_curve(
        player_adp=5.0,
        player_positions="OF",
        current_pick=20,
        user_team_index=0,
        num_teams=12,
        num_rounds=23,
    )
    if curve:
        assert curve[0]["p_available"] < 0.15


def test_future_picks_excludes_current_on_the_clock_pick():
    """DE-C4: the user's CURRENT on-the-clock pick is not a future point.

    current_pick is 0-indexed (matches DraftState.current_pick). When the
    user is on the clock, overall = current_pick + 1 maps back to the
    current pick — it must NOT appear in the future-pick list.
    """
    # current_pick=0 puts team 0 on the clock (round 0, pos 0).
    picks = _future_user_picks(current_pick=0, user_team_index=0, num_teams=12, num_rounds=23)
    pick_numbers = [p["pick"] for p in picks]
    # The current pick (1-indexed overall = 1) must be excluded.
    assert 1 not in pick_numbers
    # First real future pick is the user's round-2 snake pick (overall 24).
    assert picks[0]["pick"] == 24
    # Every future pick must be strictly after the current pick.
    assert all(p["pick"] > 1 for p in picks)


def test_future_picks_unaffected_when_not_on_the_clock():
    """When the user is NOT on the clock, the count of future picks is the
    same as before (skipping overall=current_pick+1 only ever removes the
    on-the-clock pick, never a legitimate future pick)."""
    # current_pick=5: team 5 is on the clock, not team 0.
    picks = _future_user_picks(current_pick=5, user_team_index=0, num_teams=12, num_rounds=23)
    # User (team 0) round-2 pick is overall 24; all 22 remaining picks > 5.
    assert all(p["pick"] > 5 for p in picks)
    assert picks[0]["pick"] == 24
    # 23 rounds, user picks once per round; rounds 2..23 remain ahead = 22.
    assert len(picks) == 22

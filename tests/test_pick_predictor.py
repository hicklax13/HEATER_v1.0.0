# tests/test_pick_predictor.py
"""Tests for pick predictor survival curves."""

from __future__ import annotations

import pytest

from src.pick_predictor import (
    POSITION_SHAPE_PARAMS,
    compute_survival_curve,
    weibull_survival,
)


def test_weibull_survival_decreases():
    s1 = weibull_survival(5, 20.0, 1.0, 30.0)
    s2 = weibull_survival(15, 20.0, 1.0, 30.0)
    assert s1 > s2


def test_weibull_survival_bounds():
    s = weibull_survival(1, 50.0, 1.0, 75.0)
    assert 0.0 <= s <= 1.0


def test_weibull_shape_above_one():
    s_early = weibull_survival(5, 20.0, 1.4, 30.0)
    s_late = weibull_survival(20, 20.0, 1.4, 30.0)
    assert s_late < s_early


def test_weibull_zero_scale():
    s = weibull_survival(10, 20.0, 1.0, 0.0)
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

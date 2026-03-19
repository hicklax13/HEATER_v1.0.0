# tests/test_power_rankings.py
"""Tests for league power rankings."""
from __future__ import annotations

from src.power_rankings import (
    compute_roster_quality,
    compute_category_balance_score,
    compute_schedule_strength_index,
    compute_injury_exposure,
    compute_momentum,
    compute_power_rating,
    compute_power_rankings,
    bootstrap_confidence_interval,
    WEIGHTS,
)


def test_roster_quality_bounds():
    assert 0.0 <= compute_roster_quality(50.0, 100.0) <= 1.0
    assert compute_roster_quality(100.0, 100.0) == 1.0
    assert compute_roster_quality(0.0, 100.0) == 0.0


def test_perfect_balance_near_one():
    zscores = {cat: 1.0 for cat in ["R", "HR", "RBI", "SB", "AVG", "OBP",
                                      "W", "SV", "K", "ERA", "WHIP", "L"]}
    score = compute_category_balance_score(zscores)
    assert score > 0.9  # Perfect balance (all equal) → near 1.0


def test_punt_balance_low():
    zscores = {"R": 5.0, "HR": 5.0, "RBI": 5.0, "SB": 0.0, "AVG": 0.0, "OBP": 0.0,
               "W": 0.0, "SV": 0.0, "K": 0.0, "ERA": 0.0, "WHIP": 0.0, "L": 0.0}
    score = compute_category_balance_score(zscores)
    assert score < 0.6


def test_healthy_team_risk_near_zero():
    tvs = [10.0, 8.0, 6.0, 5.0, 4.0]
    health = [0.95, 0.95, 0.95, 0.95, 0.95]
    risk = compute_injury_exposure(tvs, health)
    assert risk < 0.1


def test_fragile_team_risk_high():
    tvs = [10.0, 8.0, 6.0]
    health = [0.50, 0.40, 0.30]
    risk = compute_injury_exposure(tvs, health)
    assert risk > 0.5


def test_momentum_default():
    assert compute_momentum() == 1.0


def test_momentum_hot():
    m = compute_momentum(recent_win_rate=0.80, season_win_rate=0.50)
    assert m > 1.0


def test_weights_sum_to_one():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 0.001


def test_all_teams_present():
    data = [{"team_name": f"Team {i}", "roster_quality": 0.5} for i in range(12)]
    df = compute_power_rankings(data)
    assert len(df) == 12


def test_sorted_descending():
    data = [
        {"team_name": "Best", "roster_quality": 1.0, "category_balance": 1.0},
        {"team_name": "Worst", "roster_quality": 0.1, "category_balance": 0.1},
    ]
    df = compute_power_rankings(data)
    assert df.iloc[0]["team_name"] == "Best"
    assert df.iloc[0]["rank"] == 1


def test_bootstrap_p5_less_than_p95():
    p5, p95 = bootstrap_confidence_interval(50.0)
    assert p5 < p95


def test_power_rating_range():
    rating = compute_power_rating(
        roster_quality=0.5, category_balance=0.5,
        schedule_strength=0.5, injury_exposure=0.0, momentum=1.0,
    )
    assert 0 <= rating <= 100

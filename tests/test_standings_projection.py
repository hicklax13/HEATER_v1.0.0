# tests/test_standings_projection.py
"""Tests for projected season standings simulation."""
from __future__ import annotations

from src.standings_projection import (
    compute_category_win_probability,
    generate_round_robin_schedule,
    simulate_season,
    INVERSE_CATS,
)


def _make_team_totals(n_teams: int = 4) -> dict[str, dict[str, float]]:
    """Create simple team totals for testing."""
    teams = {}
    for i in range(n_teams):
        teams[f"Team {i+1}"] = {
            "R": 5.0 + i * 0.5, "HR": 2.0 + i * 0.2,
            "RBI": 5.5 + i * 0.5, "SB": 1.0 + i * 0.1,
            "AVG": 0.260 + i * 0.005, "OBP": 0.330 + i * 0.005,
            "W": 1.0, "L": 1.0, "SV": 0.5, "K": 8.0,
            "ERA": 4.00 - i * 0.2, "WHIP": 1.25 - i * 0.02,
        }
    return teams


def test_round_robin_covers_all_matchups():
    names = ["A", "B", "C", "D"]
    schedule = generate_round_robin_schedule(names, n_weeks=6)
    assert len(schedule) == 6
    # Each week should have N/2 = 2 matchups
    for week in schedule:
        assert len(week) <= 2


def test_round_robin_no_self_play():
    names = ["A", "B", "C", "D"]
    schedule = generate_round_robin_schedule(names, n_weeks=10)
    for week in schedule:
        for a, b in week:
            assert a != b


def test_category_win_prob_symmetry():
    p = compute_category_win_probability(5.0, 5.0, 1.0, 1.0)
    assert abs(p - 0.5) < 0.01  # Equal teams ≈ 50%


def test_equal_teams_near_50():
    p = compute_category_win_probability(10.0, 10.0, 2.0, 2.0)
    assert 0.45 < p < 0.55


def test_inverse_stat_lower_wins():
    """For ERA, lower team_a should have higher win probability."""
    p = compute_category_win_probability(3.00, 5.00, 0.5, 0.5, is_inverse=True)
    assert p > 0.5  # 3.00 ERA beats 5.00 ERA


def test_simulate_season_total_games():
    teams = _make_team_totals(4)
    df = simulate_season(teams, n_sims=50, seed=42)
    assert len(df) == 4
    # Each team plays every week
    for _, row in df.iterrows():
        total = row["mean_wins"] + row["mean_losses"] + row["mean_ties"]
        assert total > 0


def test_simulate_ci_ordering():
    teams = _make_team_totals(4)
    df = simulate_season(teams, n_sims=100, seed=42)
    for _, row in df.iterrows():
        assert row["win_p5"] <= row["win_p95"]


def test_simulate_playoff_pct_range():
    teams = _make_team_totals(4)
    df = simulate_season(teams, n_sims=100, seed=42)
    for _, row in df.iterrows():
        assert 0.0 <= row["playoff_pct"] <= 100.0


def test_simulate_best_team_highest_wins():
    """Strong team should generally have highest mean wins."""
    teams = {
        "Weak": {"R": 3.0, "HR": 1.0, "RBI": 3.0, "SB": 0.5, "AVG": 0.230, "OBP": 0.300,
                  "W": 0.5, "L": 1.5, "SV": 0.3, "K": 5.0, "ERA": 5.00, "WHIP": 1.45},
        "Strong": {"R": 8.0, "HR": 3.5, "RBI": 8.0, "SB": 2.0, "AVG": 0.290, "OBP": 0.370,
                    "W": 1.8, "L": 0.5, "SV": 1.0, "K": 12.0, "ERA": 3.00, "WHIP": 1.05},
    }
    df = simulate_season(teams, n_sims=500, seed=42)
    best = df.iloc[0]["team_name"]
    assert best == "Strong"

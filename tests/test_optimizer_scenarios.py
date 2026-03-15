"""Tests for stochastic lineup optimization scenario generator.

Covers scenario generation, mean-variance adjustments, CVaR constraint
building, player variance estimation, and per-scenario lineup values.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.optimizer.scenario_generator import (
    ALL_CATS,
    build_cvar_constraints,
    compute_scenario_lineup_values,
    estimate_player_variance,
    generate_stat_scenarios,
    mean_variance_adjustments,
)

# ── Fixtures ─────────────────────────────────────────────────────────


def _sample_hitter(hr=25, rbi=80, r=70, sb=10, avg=0.270) -> dict:
    """A typical hitter projection."""
    return {
        "r": r,
        "hr": hr,
        "rbi": rbi,
        "sb": sb,
        "avg": avg,
        "w": 0,
        "sv": 0,
        "k": 0,
        "era": 0.0,
        "whip": 0.0,
    }


def _sample_pitcher(w=10, sv=0, k=180, era=3.50, whip=1.15) -> dict:
    """A typical starting pitcher projection."""
    return {
        "r": 0,
        "hr": 0,
        "rbi": 0,
        "sb": 0,
        "avg": 0.0,
        "w": w,
        "sv": sv,
        "k": k,
        "era": era,
        "whip": whip,
    }


def _three_player_roster() -> list[dict]:
    """A 3-player roster: 2 hitters + 1 pitcher."""
    return [
        _sample_hitter(hr=30, rbi=90, r=80, sb=5, avg=0.280),
        _sample_hitter(hr=10, rbi=50, r=60, sb=25, avg=0.260),
        _sample_pitcher(w=12, sv=0, k=200, era=3.20, whip=1.10),
    ]


def _equal_weights() -> dict[str, float]:
    """Equal category weights for all 10 categories."""
    return {cat: 1.0 for cat in ALL_CATS}


# ── Scenario generation tests ────────────────────────────────────────


class TestGenerateStatScenarios:
    def test_scenarios_shape(self):
        """Output shape is (n_scenarios, n_players, 10)."""
        roster = _three_player_roster()
        scenarios = generate_stat_scenarios(roster, n_scenarios=100, seed=42)
        assert scenarios.shape == (100, 3, 10)

    def test_scenarios_deterministic_seed(self):
        """Same seed produces identical scenarios."""
        roster = _three_player_roster()
        s1 = generate_stat_scenarios(roster, n_scenarios=50, seed=99)
        s2 = generate_stat_scenarios(roster, n_scenarios=50, seed=99)
        np.testing.assert_array_equal(s1, s2)

    def test_scenarios_different_seed(self):
        """Different seeds produce different scenarios."""
        roster = _three_player_roster()
        s1 = generate_stat_scenarios(roster, n_scenarios=50, seed=1)
        s2 = generate_stat_scenarios(roster, n_scenarios=50, seed=2)
        assert not np.allclose(s1, s2)

    def test_scenarios_mean_close_to_projection(self):
        """Mean across scenarios approximately equals projected stats (within 10%)."""
        roster = [_sample_hitter(hr=30, rbi=90, r=80, sb=10, avg=0.270)]
        scenarios = generate_stat_scenarios(roster, n_scenarios=5000, seed=42)
        means = scenarios[:, 0, :].mean(axis=0)

        # Check counting stats are within 10% of projection
        hr_idx = ALL_CATS.index("hr")
        rbi_idx = ALL_CATS.index("rbi")
        r_idx = ALL_CATS.index("r")

        assert abs(means[hr_idx] - 30) / 30 < 0.10
        assert abs(means[rbi_idx] - 90) / 90 < 0.10
        assert abs(means[r_idx] - 80) / 80 < 0.10

    def test_scenarios_variance_positive(self):
        """All generated stat columns have positive variance across scenarios."""
        roster = _three_player_roster()
        scenarios = generate_stat_scenarios(roster, n_scenarios=200, seed=42)

        for p in range(3):
            player_scenarios = scenarios[:, p, :]
            variances = player_scenarios.var(axis=0)
            # Only check categories with non-zero projections
            for j, cat in enumerate(ALL_CATS):
                proj = _three_player_roster()[p].get(cat, 0.0)
                if abs(proj) > 0.01:
                    assert variances[j] > 0, f"Player {p}, cat {cat} has zero variance"

    def test_scenarios_correlation_hr_rbi(self):
        """HR and RBI are positively correlated across scenarios."""
        roster = [_sample_hitter(hr=30, rbi=90)]
        scenarios = generate_stat_scenarios(roster, n_scenarios=2000, seed=42)

        hr_idx = ALL_CATS.index("hr")
        rbi_idx = ALL_CATS.index("rbi")

        hr_vals = scenarios[:, 0, hr_idx]
        rbi_vals = scenarios[:, 0, rbi_idx]
        corr = np.corrcoef(hr_vals, rbi_vals)[0, 1]

        assert corr > 0.3, f"HR-RBI correlation {corr:.3f} should be positive"

    def test_scenarios_correlation_sb_hr(self):
        """SB and HR are negatively correlated across scenarios."""
        # Use a player with non-trivial values in both
        roster = [_sample_hitter(hr=25, sb=20)]
        scenarios = generate_stat_scenarios(roster, n_scenarios=2000, seed=42)

        hr_idx = ALL_CATS.index("hr")
        sb_idx = ALL_CATS.index("sb")

        hr_vals = scenarios[:, 0, hr_idx]
        sb_vals = scenarios[:, 0, sb_idx]
        corr = np.corrcoef(hr_vals, sb_vals)[0, 1]

        assert corr < 0.0, f"SB-HR correlation {corr:.3f} should be negative"


# ── Mean-variance adjustment tests ───────────────────────────────────


class TestMeanVarianceAdjustments:
    def test_mean_variance_penalizes_high_variance(self):
        """Player with higher variance gets a more negative adjustment."""
        # High-variance slugger vs low-variance contact hitter
        roster = [
            _sample_hitter(hr=40, rbi=100, r=90, sb=2, avg=0.240),  # big counting stats -> big CV
            _sample_hitter(hr=5, rbi=30, r=40, sb=3, avg=0.280),  # small counting stats -> small CV
        ]
        adj = mean_variance_adjustments(roster, lambda_risk=0.15)

        # Slugger should have more negative (larger magnitude) penalty
        assert adj[0] < adj[1], "High-variance player should be penalized more"

    def test_mean_variance_stable_player_no_penalty(self):
        """Low-variance player gets a small (near-zero) penalty."""
        roster = [_sample_hitter(hr=2, rbi=10, r=15, sb=1, avg=0.260)]
        adj = mean_variance_adjustments(roster, lambda_risk=0.15)

        # Penalty should be small in magnitude (within a few SGP points)
        assert abs(adj[0]) < 5.0, f"Penalty {adj[0]} too large for low-variance player"

    def test_mean_variance_lambda_zero_no_effect(self):
        """lambda=0 produces zero adjustments for all players."""
        roster = _three_player_roster()
        adj = mean_variance_adjustments(roster, lambda_risk=0.0)

        for idx in adj:
            assert adj[idx] == 0.0, f"Player {idx} should have zero adjustment"


# ── CVaR constraint tests ────────────────────────────────────────────


class TestBuildCvarConstraints:
    def test_cvar_constraints_structure(self):
        """Output has correct keys."""
        roster = _three_player_roster()
        scenarios = generate_stat_scenarios(roster, n_scenarios=50, seed=42)
        weights = _equal_weights()

        result = build_cvar_constraints(scenarios, [0, 1, 2], weights, alpha=0.05)

        assert "n_scenarios" in result
        assert "scenario_values" in result
        assert "alpha" in result
        assert "eta_name" in result

    def test_cvar_constraints_n_scenarios(self):
        """scenario_values list has correct length."""
        roster = _three_player_roster()
        n = 75
        scenarios = generate_stat_scenarios(roster, n_scenarios=n, seed=42)
        weights = _equal_weights()

        result = build_cvar_constraints(scenarios, [0, 1, 2], weights, alpha=0.05)

        assert result["n_scenarios"] == n
        assert len(result["scenario_values"]) == n

    def test_cvar_alpha_preserved(self):
        """Alpha parameter is passed through correctly."""
        roster = _three_player_roster()
        scenarios = generate_stat_scenarios(roster, n_scenarios=20, seed=42)
        weights = _equal_weights()

        result = build_cvar_constraints(scenarios, [0, 1], weights, alpha=0.10)
        assert result["alpha"] == 0.10


# ── Player variance tests ────────────────────────────────────────────


class TestEstimatePlayerVariance:
    def test_player_variance_hitter_vs_pitcher(self):
        """Hitters and pitchers produce different variance profiles."""
        hitter_stats = {
            "r": 80,
            "hr": 30,
            "rbi": 90,
            "sb": 10,
            "avg": 0.270,
            "w": 0,
            "sv": 0,
            "k": 0,
            "era": 0.0,
            "whip": 0.0,
        }
        pitcher_stats = {
            "r": 0,
            "hr": 0,
            "rbi": 0,
            "sb": 0,
            "avg": 0.0,
            "w": 12,
            "sv": 0,
            "k": 200,
            "era": 3.20,
            "whip": 1.10,
        }

        hitter_var = estimate_player_variance(hitter_stats, is_hitter=True)
        pitcher_var = estimate_player_variance(pitcher_stats, is_hitter=False)

        # Both should be positive and non-equal
        assert hitter_var > 0
        assert pitcher_var > 0
        assert hitter_var != pitcher_var

    def test_player_variance_always_positive(self):
        """Variance is always positive, even for zero-stat players."""
        zero_stats = {cat: 0.0 for cat in ALL_CATS}
        var = estimate_player_variance(zero_stats, is_hitter=True)
        # Rate stats (avg) have fixed std, so variance > 0 even for zero-stat player
        assert var > 0


# ── Scenario lineup values tests ─────────────────────────────────────


class TestComputeScenarioLineupValues:
    def test_scenario_lineup_values_shape(self):
        """Output shape is (n_scenarios,)."""
        roster = _three_player_roster()
        scenarios = generate_stat_scenarios(roster, n_scenarios=100, seed=42)
        assignments = {0: 1.0, 1: 1.0}  # Start players 0 and 1
        weights = _equal_weights()

        values = compute_scenario_lineup_values(scenarios, assignments, weights)
        assert values.shape == (100,)

    def test_scenario_lineup_values_positive(self):
        """Reasonable lineup produces mostly positive scenario values."""
        roster = _three_player_roster()
        scenarios = generate_stat_scenarios(roster, n_scenarios=500, seed=42)
        assignments = {0: 1.0, 1: 1.0, 2: 1.0}
        # Weight counting stats positively, inverse stats positively too
        # (sign flip handled internally)
        weights = _equal_weights()

        values = compute_scenario_lineup_values(scenarios, assignments, weights)

        # The majority of scenarios should produce positive values
        # since we have reasonable player projections
        positive_frac = (values > 0).mean()
        assert positive_frac > 0.5, f"Only {positive_frac:.1%} of scenarios positive"

    def test_scenario_lineup_values_empty(self):
        """Empty assignments produce zero-valued array."""
        roster = _three_player_roster()
        scenarios = generate_stat_scenarios(roster, n_scenarios=50, seed=42)
        assignments: dict[int, float] = {}
        weights = _equal_weights()

        values = compute_scenario_lineup_values(scenarios, assignments, weights)
        assert values.shape == (50,)
        np.testing.assert_array_equal(values, np.zeros(50))


# ── Empty roster edge case ───────────────────────────────────────────


class TestEmptyRoster:
    def test_empty_roster_scenarios(self):
        """Empty roster produces shape (N, 0, 10)."""
        scenarios = generate_stat_scenarios([], n_scenarios=100, seed=42)
        assert scenarios.shape == (100, 0, 10)

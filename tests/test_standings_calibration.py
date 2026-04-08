"""Tests for R1 (10K MC sims) and R2 (calibrated weekly tau) in standings_engine."""

from __future__ import annotations

import inspect

import pytest

from src.standings_engine import (
    ALL_CATEGORIES,
    CALIBRATED_WEEKLY_TAU,
    compute_category_win_probabilities,
    simulate_season_enhanced,
)

# ── R1: Default n_sims is 10000 ──────────────────────────────────────


class TestDefaultNSims:
    """Verify both MC functions default to 10000 simulations."""

    def test_simulate_season_enhanced_default_n_sims(self) -> None:
        """simulate_season_enhanced default n_sims should be 10000."""
        sig = inspect.signature(simulate_season_enhanced)
        assert sig.parameters["n_sims"].default == 10000

    def test_compute_category_win_probabilities_default_n_sims(self) -> None:
        """compute_category_win_probabilities default n_sims should be 10000."""
        sig = inspect.signature(compute_category_win_probabilities)
        assert sig.parameters["n_sims"].default == 10000

    def test_simulation_produces_results_with_10k_sims(self) -> None:
        """simulate_season_enhanced should produce valid results at 10K sims."""
        standings = {
            "Team A": {"W": 5, "L": 3, "T": 0},
            "Team B": {"W": 3, "L": 5, "T": 0},
            "Team C": {"W": 4, "L": 4, "T": 0},
            "Team D": {"W": 6, "L": 2, "T": 0},
        }
        weekly_totals = {
            "Team A": {"R": 30, "HR": 8, "RBI": 28, "SB": 4, "AVG": 0.265, "OBP": 0.335,
                        "W": 3, "L": 2, "SV": 2, "K": 35, "ERA": 3.80, "WHIP": 1.22},
            "Team B": {"R": 25, "HR": 6, "RBI": 24, "SB": 3, "AVG": 0.250, "OBP": 0.320,
                        "W": 2, "L": 3, "SV": 1, "K": 30, "ERA": 4.20, "WHIP": 1.35},
            "Team C": {"R": 28, "HR": 7, "RBI": 26, "SB": 5, "AVG": 0.258, "OBP": 0.328,
                        "W": 3, "L": 2, "SV": 3, "K": 32, "ERA": 3.95, "WHIP": 1.28},
            "Team D": {"R": 32, "HR": 9, "RBI": 30, "SB": 2, "AVG": 0.270, "OBP": 0.340,
                        "W": 4, "L": 2, "SV": 2, "K": 38, "ERA": 3.50, "WHIP": 1.18},
        }
        schedule = {
            9: [("Team A", "Team B"), ("Team C", "Team D")],
            10: [("Team A", "Team C"), ("Team B", "Team D")],
        }

        result = simulate_season_enhanced(
            current_standings=standings,
            team_weekly_totals=weekly_totals,
            full_schedule=schedule,
            current_week=9,
            # n_sims omitted — should use 10000 default
        )

        assert "projected_records" in result
        assert "playoff_probability" in result
        assert len(result["projected_records"]) == 4
        # With 10K sims, playoff probs should be stable (non-NaN)
        for team, prob in result["playoff_probability"].items():
            assert 0.0 <= prob <= 1.0, f"{team} playoff prob out of range: {prob}"


# ── R2: Calibrated weekly tau ────────────────────────────────────────


class TestCalibratedWeeklyTau:
    """Verify CALIBRATED_WEEKLY_TAU has correct properties."""

    def test_has_all_12_categories(self) -> None:
        """CALIBRATED_WEEKLY_TAU must contain all 12 league categories."""
        for cat in ALL_CATEGORIES:
            assert cat in CALIBRATED_WEEKLY_TAU, f"Missing category: {cat}"
        assert len(CALIBRATED_WEEKLY_TAU) == 12

    def test_k_variance_less_than_r_variance(self) -> None:
        """K (strikeouts) is the most stable counting stat per FanGraphs research."""
        assert CALIBRATED_WEEKLY_TAU["K"] < CALIBRATED_WEEKLY_TAU["R"]

    def test_all_values_positive(self) -> None:
        """All tau values must be strictly positive."""
        for cat, val in CALIBRATED_WEEKLY_TAU.items():
            assert val > 0, f"{cat} tau is non-positive: {val}"

    def test_rate_stats_have_small_tau(self) -> None:
        """Rate stat taus should be much smaller than counting stat taus."""
        rate_cats = ("AVG", "OBP", "ERA", "WHIP")
        counting_cats = ("R", "HR", "RBI", "SB", "W", "L", "SV", "K")
        max_rate = max(CALIBRATED_WEEKLY_TAU[c] for c in rate_cats)
        min_counting = min(CALIBRATED_WEEKLY_TAU[c] for c in counting_cats)
        assert max_rate < min_counting, (
            f"Rate stat max tau ({max_rate}) should be < counting stat min ({min_counting})"
        )

    def test_specific_calibrated_values(self) -> None:
        """Spot-check specific FanGraphs-derived values."""
        assert CALIBRATED_WEEKLY_TAU["R"] == 1.6
        assert CALIBRATED_WEEKLY_TAU["HR"] == 2.1
        assert CALIBRATED_WEEKLY_TAU["K"] == 1.2
        assert CALIBRATED_WEEKLY_TAU["ERA"] == 0.50
        assert CALIBRATED_WEEKLY_TAU["WHIP"] == 0.05

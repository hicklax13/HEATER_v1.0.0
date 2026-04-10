"""Tests for the sigmoid K-value calibration framework.

Covers:
  - simulate_h2h_week: blowout win, close loss, mixed, tied, empty
  - score_urgency_weights: high/low urgency correctness, perfect, equal
  - calibrate_sigmoid_k: result validity, bounds, determinism
  - recommend_k_values: output format and content
"""

from __future__ import annotations

import pytest

from src.optimizer.sigmoid_calibrator import (
    CalibrationResult,
    calibrate_sigmoid_k,
    recommend_k_values,
    score_urgency_weights,
    simulate_h2h_week,
)

# ── TestSimulateH2HWeek ─────────────────────────────────────────────


class TestSimulateH2HWeek:
    """Verify H2H category win/loss/tie simulation."""

    def test_blowout_win_all_categories(self):
        """Winning all 12 categories should return 12-0-0."""
        my = {
            "R": 50,
            "HR": 15,
            "RBI": 45,
            "SB": 10,
            "AVG": 0.300,
            "OBP": 0.380,
            "W": 7,
            "L": 1,
            "SV": 5,
            "K": 60,
            "ERA": 2.00,
            "WHIP": 0.90,
        }
        opp = {
            "R": 20,
            "HR": 3,
            "RBI": 15,
            "SB": 1,
            "AVG": 0.220,
            "OBP": 0.280,
            "W": 2,
            "L": 6,
            "SV": 0,
            "K": 25,
            "ERA": 5.00,
            "WHIP": 1.50,
        }
        wins, losses, ties = simulate_h2h_week(my, opp)
        assert wins == 12
        assert losses == 0
        assert ties == 0

    def test_close_loss_inverse_stat_handling(self):
        """Losing ERA/WHIP (inverse) when my values are higher."""
        # I'm losing: ERA and WHIP are higher (worse) than opponent
        my = {
            "R": 30,
            "HR": 6,
            "RBI": 25,
            "SB": 3,
            "AVG": 0.250,
            "OBP": 0.320,
            "W": 3,
            "L": 4,
            "SV": 1,
            "K": 35,
            "ERA": 4.50,
            "WHIP": 1.35,
        }
        opp = {
            "R": 35,
            "HR": 9,
            "RBI": 32,
            "SB": 5,
            "AVG": 0.270,
            "OBP": 0.345,
            "W": 5,
            "L": 2,
            "SV": 3,
            "K": 50,
            "ERA": 3.00,
            "WHIP": 1.05,
        }
        wins, losses, ties = simulate_h2h_week(my, opp)
        # All normal stats: opponent wins
        # L: my 4 > opp 2, but L is inverse so lower is better -> opponent wins
        # ERA: my 4.50 > opp 3.00 -> opponent wins (lower ERA is better)
        # WHIP: my 1.35 > opp 1.05 -> opponent wins
        assert losses == 12
        assert wins == 0
        assert ties == 0

    def test_mixed_results(self):
        """Mixed scenario with some wins, losses, and ties."""
        my = {
            "R": 40,
            "HR": 10,
            "RBI": 35,
            "SB": 5,
            "AVG": 0.270,
            "OBP": 0.340,
            "W": 4,
            "L": 3,
            "SV": 2,
            "K": 45,
            "ERA": 3.50,
            "WHIP": 1.15,
        }
        opp = {
            "R": 35,
            "HR": 10,
            "RBI": 38,
            "SB": 5,
            "AVG": 0.265,
            "OBP": 0.340,
            "W": 4,
            "L": 4,
            "SV": 3,
            "K": 42,
            "ERA": 3.80,
            "WHIP": 1.20,
        }
        wins, losses, ties = simulate_h2h_week(my, opp)
        total = wins + losses + ties
        assert total == 12
        assert wins > 0  # At minimum: R, AVG, K, L(inverse), ERA(inverse), WHIP(inverse)
        assert ties > 0  # HR and SB and OBP are tied

    def test_tied_all_categories(self):
        """Identical totals should return 0-0-12."""
        totals = {
            "R": 33,
            "HR": 7,
            "RBI": 30,
            "SB": 4,
            "AVG": 0.265,
            "OBP": 0.335,
            "W": 4,
            "L": 3,
            "SV": 2,
            "K": 42,
            "ERA": 3.40,
            "WHIP": 1.15,
        }
        wins, losses, ties = simulate_h2h_week(totals, totals)
        assert wins == 0
        assert losses == 0
        assert ties == 12

    def test_empty_totals(self):
        """Empty dicts should be handled gracefully (all ties at 0)."""
        wins, losses, ties = simulate_h2h_week({}, {})
        total = wins + losses + ties
        assert total == 12
        assert ties == 12  # All zeros = all tied


# ── TestScoreUrgencyWeights ──────────────────────────────────────────


class TestScoreUrgencyWeights:
    """Verify urgency weight scoring accuracy."""

    def test_high_urgency_on_losing_categories(self):
        """High urgency on categories I'm losing should score well."""
        my = {
            "R": 20,
            "HR": 3,
            "RBI": 15,
            "SB": 1,
            "AVG": 0.220,
            "OBP": 0.280,
            "W": 2,
            "L": 6,
            "SV": 0,
            "K": 25,
            "ERA": 5.00,
            "WHIP": 1.50,
        }
        opp = {
            "R": 45,
            "HR": 14,
            "RBI": 42,
            "SB": 8,
            "AVG": 0.290,
            "OBP": 0.370,
            "W": 7,
            "L": 2,
            "SV": 5,
            "K": 60,
            "ERA": 2.50,
            "WHIP": 0.98,
        }
        # Good urgency: high (> 0.6) for all categories since I'm losing all
        urgency = {
            "R": 0.9,
            "HR": 0.85,
            "RBI": 0.9,
            "SB": 0.8,
            "AVG": 0.88,
            "OBP": 0.87,
            "W": 0.85,
            "L": 0.9,
            "SV": 0.82,
            "K": 0.88,
            "ERA": 0.92,
            "WHIP": 0.91,
        }
        score = score_urgency_weights(urgency, my, opp)
        assert score > 0.8  # Most should be correct

    def test_low_urgency_on_losing_categories_scores_poorly(self):
        """Low urgency on categories I'm losing should score poorly."""
        my = {
            "R": 20,
            "HR": 3,
            "RBI": 15,
            "SB": 1,
            "AVG": 0.220,
            "OBP": 0.280,
            "W": 2,
            "L": 6,
            "SV": 0,
            "K": 25,
            "ERA": 5.00,
            "WHIP": 1.50,
        }
        opp = {
            "R": 45,
            "HR": 14,
            "RBI": 42,
            "SB": 8,
            "AVG": 0.290,
            "OBP": 0.370,
            "W": 7,
            "L": 2,
            "SV": 5,
            "K": 60,
            "ERA": 2.50,
            "WHIP": 0.98,
        }
        # Bad urgency: low values when losing everything
        urgency = {
            "R": 0.1,
            "HR": 0.15,
            "RBI": 0.1,
            "SB": 0.2,
            "AVG": 0.12,
            "OBP": 0.13,
            "W": 0.15,
            "L": 0.1,
            "SV": 0.18,
            "K": 0.12,
            "ERA": 0.08,
            "WHIP": 0.09,
        }
        score = score_urgency_weights(urgency, my, opp)
        assert score < 0.2  # Almost all wrong

    def test_perfect_urgency(self):
        """Perfect urgency assignment: 1.0 for losing, 0.0 for winning."""
        my = {
            "R": 50,
            "HR": 15,
            "RBI": 45,
            "SB": 10,
            "AVG": 0.300,
            "OBP": 0.380,
            "W": 7,
            "L": 1,
            "SV": 5,
            "K": 60,
            "ERA": 2.00,
            "WHIP": 0.90,
        }
        opp = {
            "R": 20,
            "HR": 3,
            "RBI": 15,
            "SB": 1,
            "AVG": 0.220,
            "OBP": 0.280,
            "W": 2,
            "L": 6,
            "SV": 0,
            "K": 25,
            "ERA": 5.00,
            "WHIP": 1.50,
        }
        # Perfect: all 0.0 since I'm winning everything
        urgency = {
            "R": 0.0,
            "HR": 0.0,
            "RBI": 0.0,
            "SB": 0.0,
            "AVG": 0.0,
            "OBP": 0.0,
            "W": 0.0,
            "L": 0.0,
            "SV": 0.0,
            "K": 0.0,
            "ERA": 0.0,
            "WHIP": 0.0,
        }
        score = score_urgency_weights(urgency, my, opp)
        assert score == pytest.approx(1.0)

    def test_all_equal_urgency(self):
        """All 0.5 urgency with tied categories should score moderately."""
        totals = {
            "R": 33,
            "HR": 7,
            "RBI": 30,
            "SB": 4,
            "AVG": 0.265,
            "OBP": 0.335,
            "W": 4,
            "L": 3,
            "SV": 2,
            "K": 42,
            "ERA": 3.40,
            "WHIP": 1.15,
        }
        urgency = {cat: 0.5 for cat in totals}
        score = score_urgency_weights(urgency, totals, totals)
        # All tied + urgency 0.5 = all correct
        assert score == pytest.approx(1.0)


# ── TestCalibrateSigmoidK ────────────────────────────────────────────


class TestCalibrateSigmoidK:
    """Verify grid search calibration behavior."""

    def test_returns_valid_result(self):
        """Should return a CalibrationResult with valid k-values."""
        result = calibrate_sigmoid_k(
            counting_k_grid=[1.0, 2.0, 3.0],
            rate_k_grid=[2.0, 3.0, 4.0],
            n_scenarios=3,
        )
        assert isinstance(result, CalibrationResult)
        assert 1.0 <= result.counting_k <= 5.0
        assert 1.5 <= result.rate_k <= 5.0

    def test_k_values_within_bounds(self):
        """Best k-values should stay within the tested grid."""
        grid_ck = [1.5, 2.0, 2.5, 3.0]
        grid_rk = [2.0, 2.5, 3.0, 3.5]
        result = calibrate_sigmoid_k(
            counting_k_grid=grid_ck,
            rate_k_grid=grid_rk,
            n_scenarios=3,
        )
        assert result.counting_k in grid_ck
        assert result.rate_k in grid_rk

    def test_deterministic_results(self):
        """Same inputs should produce identical output (no randomness)."""
        kwargs = {
            "counting_k_grid": [1.0, 2.0, 3.0, 4.0],
            "rate_k_grid": [2.0, 3.0, 4.0],
            "n_scenarios": 3,
        }
        r1 = calibrate_sigmoid_k(**kwargs)
        r2 = calibrate_sigmoid_k(**kwargs)
        assert r1.counting_k == r2.counting_k
        assert r1.rate_k == r2.rate_k
        assert r1.avg_rank_correlation == pytest.approx(r2.avg_rank_correlation)
        assert r1.avg_rmse == pytest.approx(r2.avg_rmse)

    def test_result_fields_populated(self):
        """All CalibrationResult fields should be populated."""
        result = calibrate_sigmoid_k(
            counting_k_grid=[2.0],
            rate_k_grid=[3.0],
            n_scenarios=3,
        )
        assert isinstance(result.counting_k, float)
        assert isinstance(result.rate_k, float)
        assert isinstance(result.avg_win_rate, float)
        assert isinstance(result.avg_rank_correlation, float)
        assert isinstance(result.avg_rmse, float)
        assert isinstance(result.n_weeks_tested, int)
        assert result.n_weeks_tested > 0

    def test_full_grid_completes(self):
        """Full 9x8 grid should complete without errors."""
        result = calibrate_sigmoid_k(n_scenarios=5)
        assert isinstance(result, CalibrationResult)
        assert result.n_weeks_tested == 5

    def test_single_point_grid(self):
        """Grid with one point should return that point."""
        result = calibrate_sigmoid_k(
            counting_k_grid=[2.5],
            rate_k_grid=[3.5],
            n_scenarios=3,
        )
        assert result.counting_k == pytest.approx(2.5)
        assert result.rate_k == pytest.approx(3.5)


# ── TestRecommendKValues ─────────────────────────────────────────────


class TestRecommendKValues:
    """Verify human-readable recommendation output."""

    def test_returns_string_with_recommendation(self):
        """Should return a non-empty string with key sections."""
        text = recommend_k_values(
            counting_k_grid=[1.0, 2.0, 3.0],
            rate_k_grid=[2.0, 3.0, 4.0],
            n_scenarios=3,
        )
        assert isinstance(text, str)
        assert len(text) > 0
        assert "CALIBRATION RESULTS" in text

    def test_includes_current_vs_best_comparison(self):
        """Output should show current and best values side-by-side."""
        text = recommend_k_values(
            counting_k_grid=[1.0, 2.0, 3.0],
            rate_k_grid=[2.0, 3.0, 4.0],
            n_scenarios=3,
        )
        assert "Current:" in text
        assert "Best:" in text
        assert "COUNTING_STAT_K" in text
        assert "RATE_STAT_K" in text

    def test_includes_percentage_change(self):
        """Output should include % change information."""
        text = recommend_k_values(
            counting_k_grid=[1.0, 2.0, 3.0],
            rate_k_grid=[2.0, 3.0, 4.0],
            n_scenarios=3,
        )
        # The output should contain a percentage marker
        assert "%" in text

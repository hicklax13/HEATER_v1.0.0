"""Tests for the historical backtesting validator.

Validates optimizer predictions against actual MLB outcomes using
RMSE, Spearman rank correlation, and bust rate metrics.
"""

import pytest

from src.optimizer.backtest_validator import (
    compute_bust_rate,
    compute_projection_rmse,
    compute_rank_correlation,
    grade_lineup_quality,
)


class TestProjectionAccuracyMetrics:
    """Verify projection accuracy scoring functions."""

    def test_perfect_projection_has_zero_rmse(self):
        """If projected == actual, RMSE should be 0."""
        projected = {"R": 80, "HR": 25, "RBI": 75}
        actual = {"R": 80, "HR": 25, "RBI": 75}
        rmse = compute_projection_rmse(projected, actual)
        assert rmse == pytest.approx(0.0, abs=1e-6)

    def test_rmse_increases_with_error(self):
        """Larger prediction errors should produce higher RMSE."""
        actual = {"R": 80, "HR": 25, "RBI": 75}
        small_error = {"R": 82, "HR": 24, "RBI": 77}
        large_error = {"R": 100, "HR": 15, "RBI": 50}

        rmse_small = compute_projection_rmse(small_error, actual)
        rmse_large = compute_projection_rmse(large_error, actual)
        assert rmse_large > rmse_small

    def test_perfect_ranking_correlation_is_one(self):
        """If player rankings match actual, Spearman rho = 1.0."""
        projected_order = [101, 102, 103, 104, 105]
        actual_order = [101, 102, 103, 104, 105]
        rho = compute_rank_correlation(projected_order, actual_order)
        assert rho == pytest.approx(1.0, abs=0.01)

    def test_reversed_ranking_correlation_is_negative(self):
        """If rankings are perfectly reversed, Spearman rho = -1.0."""
        projected_order = [101, 102, 103, 104, 105]
        actual_order = [105, 104, 103, 102, 101]
        rho = compute_rank_correlation(projected_order, actual_order)
        assert rho == pytest.approx(-1.0, abs=0.01)

    def test_bust_rate_counts_overestimates(self):
        """Bust rate = fraction of players whose actual < 50% of projected."""
        projected = {101: 30, 102: 25, 103: 20, 104: 15}
        actual = {101: 28, 102: 10, 103: 8, 104: 14}
        # Player 102: 10 < 12.5 (50% of 25) -> bust
        # Player 103: 8 < 10 (50% of 20) -> bust
        # Bust rate: 2/4 = 0.50
        rate = compute_bust_rate(projected, actual, threshold=0.50)
        assert rate == pytest.approx(0.50, abs=0.01)

    def test_bust_rate_skips_zero_projected(self):
        """Players with projected <= 0 should not count as busts."""
        projected = {101: 0, 102: 0, 103: 20}
        actual = {101: 5, 102: 0, 103: 18}
        # Only player 103 is eligible (projected > 0), and 18 >= 10 -> not bust
        rate = compute_bust_rate(projected, actual, threshold=0.50)
        assert rate == pytest.approx(0.0, abs=0.01)


class TestLineupQualityGrading:
    """Verify lineup quality comparison against actual-optimal."""

    def test_perfect_lineup_gets_a_grade(self):
        """If optimizer lineup matches actual-optimal, grade is 'A'."""
        grade = grade_lineup_quality(
            optimizer_value=95.0,
            optimal_value=100.0,
            threshold_a=0.90,
            threshold_b=0.80,
        )
        assert grade == "A"

    def test_poor_lineup_gets_c_grade(self):
        """If optimizer captures < 80% of optimal value, grade is 'C'."""
        grade = grade_lineup_quality(
            optimizer_value=70.0,
            optimal_value=100.0,
            threshold_a=0.90,
            threshold_b=0.80,
        )
        assert grade == "C"

    def test_middle_lineup_gets_b_grade(self):
        """If optimizer captures between 80% and 90%, grade is 'B'."""
        grade = grade_lineup_quality(
            optimizer_value=85.0,
            optimal_value=100.0,
            threshold_a=0.90,
            threshold_b=0.80,
        )
        assert grade == "B"

    def test_zero_optimal_returns_a(self):
        """If optimal_value is 0, return 'A' as edge case."""
        grade = grade_lineup_quality(
            optimizer_value=50.0,
            optimal_value=0.0,
        )
        assert grade == "A"

"""Tests for Bayesian platoon adjustment (J2) and calibrated pitcher quality (J3)."""

from __future__ import annotations

import pytest

from src.optimizer.matchup_adjustments import (
    bayesian_platoon_adjustment,
    calibrated_pitcher_quality_mult,
)

# ── J2: Bayesian Platoon Adjustment ─────────────────────────────────


class TestBayesianPlatoonAdjustment:
    """Tests for bayesian_platoon_adjustment()."""

    def test_lhb_vs_rhp_no_individual_data_league_avg_boost(self):
        """LHB vs RHP with no individual data returns league-average boost (~1.086)."""
        result = bayesian_platoon_adjustment("L", "R")
        assert result == pytest.approx(1.086, abs=0.001)

    def test_rhb_vs_lhp_no_individual_data_smaller_boost(self):
        """RHB vs LHP with no data returns smaller boost (~1.061)."""
        result = bayesian_platoon_adjustment("R", "L")
        assert result == pytest.approx(1.061, abs=0.001)

    def test_same_side_lhb_vs_lhp_penalty(self):
        """Same-side matchup (LHB vs LHP) returns penalty (<1.0)."""
        result = bayesian_platoon_adjustment("L", "L")
        assert result == pytest.approx(1.0 - 0.086, abs=0.001)
        assert result < 1.0

    def test_same_side_rhb_vs_rhp_penalty(self):
        """Same-side matchup (RHB vs RHP) returns penalty (<1.0)."""
        result = bayesian_platoon_adjustment("R", "R")
        assert result == pytest.approx(1.0 - 0.061, abs=0.001)
        assert result < 1.0

    def test_200pa_individual_data_mostly_league_avg(self):
        """With 200 PA individual data for LHB: 80% league average, 20% individual."""
        # Individual split: .290 vs overall .270 = +7.4% individual advantage
        # League advantage for LHB vs RHP = +8.6%
        # Blend weight = 200/1000 = 0.20
        # Blended = 0.20 * 0.0741 + 0.80 * 0.086 = 0.01481 + 0.0688 = 0.08361
        result = bayesian_platoon_adjustment(
            "L", "R",
            individual_split_avg=0.290,
            individual_overall_avg=0.270,
            sample_pa=200,
        )
        individual_adv = (0.290 - 0.270) / 0.270  # ~0.0741
        expected = 1.0 + 0.20 * individual_adv + 0.80 * 0.086
        assert result == pytest.approx(expected, abs=0.002)
        # Should be mostly league average
        assert abs(result - 1.086) < abs(result - (1.0 + individual_adv))

    def test_1000pa_individual_data_50_50_blend(self):
        """With 1000 PA individual data for LHB: 50/50 blend."""
        # Individual split: .300 vs overall .270 = +11.1% individual advantage
        # League advantage for LHB vs RHP = +8.6%
        # Blend weight = 1000/1000 = 1.0
        # Blended = 1.0 * 0.1111 + 0.0 * 0.086 = 0.1111
        result = bayesian_platoon_adjustment(
            "L", "R",
            individual_split_avg=0.300,
            individual_overall_avg=0.270,
            sample_pa=1000,
        )
        individual_adv = (0.300 - 0.270) / 0.270  # ~0.1111
        expected = 1.0 + 1.0 * individual_adv + 0.0 * 0.086
        assert result == pytest.approx(expected, abs=0.002)

    def test_rhb_2200pa_full_individual_weight(self):
        """RHB at 2200 PA (stabilization point) uses full individual data."""
        result = bayesian_platoon_adjustment(
            "R", "L",
            individual_split_avg=0.310,
            individual_overall_avg=0.280,
            sample_pa=2200,
        )
        individual_adv = (0.310 - 0.280) / 0.280  # ~0.1071
        expected = 1.0 + individual_adv
        assert result == pytest.approx(expected, abs=0.002)

    def test_result_clamped_to_lower_bound(self):
        """Result clamped to 0.80 minimum."""
        # Extreme same-side with terrible individual split
        result = bayesian_platoon_adjustment(
            "L", "L",
            individual_split_avg=0.100,
            individual_overall_avg=0.270,
            sample_pa=2000,
        )
        assert result >= 0.80

    def test_result_clamped_to_upper_bound(self):
        """Result clamped to 1.20 maximum."""
        # Extreme advantage with great individual split
        result = bayesian_platoon_adjustment(
            "L", "R",
            individual_split_avg=0.400,
            individual_overall_avg=0.250,
            sample_pa=2000,
        )
        assert result <= 1.20

    def test_unknown_handedness_neutral(self):
        """Unknown handedness returns 1.0."""
        assert bayesian_platoon_adjustment("", "R") == 1.0
        assert bayesian_platoon_adjustment("L", "") == 1.0
        assert bayesian_platoon_adjustment("S", "R") == 1.0

    def test_zero_overall_avg_uses_league_default(self):
        """Zero overall average falls back to league default."""
        result = bayesian_platoon_adjustment(
            "L", "R",
            individual_split_avg=0.300,
            individual_overall_avg=0.0,
            sample_pa=500,
        )
        # Should use league default since overall_avg <= 0
        assert result == pytest.approx(1.086, abs=0.001)

    def test_none_split_avg_uses_league_default(self):
        """None individual split falls back to league default."""
        result = bayesian_platoon_adjustment(
            "R", "L",
            individual_split_avg=None,
            individual_overall_avg=0.270,
            sample_pa=500,
        )
        assert result == pytest.approx(1.061, abs=0.001)


# ── J3: Calibrated Pitcher Quality Multiplier ───────────────────────


class TestCalibratedPitcherQualityMult:
    """Tests for calibrated_pitcher_quality_mult()."""

    def test_league_avg_pitcher_neutral(self):
        """League-average pitcher (ERA 4.20) returns multiplier = 1.0."""
        result = calibrated_pitcher_quality_mult(4.20)
        assert result == pytest.approx(1.0, abs=0.001)

    def test_ace_pitcher_penalizes_hitters(self):
        """Ace (ERA 2.50) penalizes hitters (~0.87)."""
        result = calibrated_pitcher_quality_mult(2.50)
        # z = (2.50 - 4.20) / 0.80 = -2.125, clamped to -2.0
        # mult = 1.0 + (-2.0) * 0.075 = 0.85
        assert result < 1.0
        assert result == pytest.approx(0.85, abs=0.02)

    def test_bad_pitcher_boosts_hitters(self):
        """Bad pitcher (ERA 5.50) boosts hitters (~1.12)."""
        result = calibrated_pitcher_quality_mult(5.50)
        # z = (5.50 - 4.20) / 0.80 = 1.625
        # mult = 1.0 + 1.625 * 0.075 = 1.1219
        assert result > 1.0
        assert result == pytest.approx(1.12, abs=0.02)

    def test_result_clamped_to_lower_bound(self):
        """Result clamped to 0.85 minimum (extreme ace)."""
        result = calibrated_pitcher_quality_mult(1.50)
        # z = (1.50 - 4.20) / 0.80 = -3.375, clamped to -2.0
        # mult = 1.0 + (-2.0) * 0.075 = 0.85
        assert result >= 0.85
        assert result == pytest.approx(0.85, abs=0.001)

    def test_result_clamped_to_upper_bound(self):
        """Result clamped to 1.15 maximum (terrible pitcher)."""
        result = calibrated_pitcher_quality_mult(7.00)
        # z = (7.00 - 4.20) / 0.80 = 3.5, clamped to 2.0
        # mult = 1.0 + 2.0 * 0.075 = 1.15
        assert result <= 1.15
        assert result == pytest.approx(1.15, abs=0.001)

    def test_slightly_below_avg_pitcher(self):
        """Slightly below average (ERA 4.60) gives small boost."""
        result = calibrated_pitcher_quality_mult(4.60)
        # z = (4.60 - 4.20) / 0.80 = 0.5
        # mult = 1.0 + 0.5 * 0.075 = 1.0375
        assert result == pytest.approx(1.0375, abs=0.005)

    def test_slightly_above_avg_pitcher(self):
        """Slightly above average (ERA 3.80) gives small penalty."""
        result = calibrated_pitcher_quality_mult(3.80)
        # z = (3.80 - 4.20) / 0.80 = -0.5
        # mult = 1.0 + (-0.5) * 0.075 = 0.9625
        assert result == pytest.approx(0.9625, abs=0.005)

    def test_custom_league_parameters(self):
        """Custom league average and std dev work correctly."""
        result = calibrated_pitcher_quality_mult(3.50, league_avg_era=3.50, league_std_era=0.60)
        assert result == pytest.approx(1.0, abs=0.001)

    def test_near_zero_std_protected(self):
        """Near-zero std dev does not cause division by zero."""
        result = calibrated_pitcher_quality_mult(4.20, league_std_era=0.0)
        # Falls back to max(0.01, 0.0) = 0.01
        # z = 0 / 0.01 = 0.0
        assert result == pytest.approx(1.0, abs=0.001)

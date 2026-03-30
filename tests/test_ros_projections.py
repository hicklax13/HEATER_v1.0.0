"""Tests for the integrated Bayesian ROS projection updater."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bayesian import (
    STABILIZATION_POINTS,
    BayesianUpdater,
    _blend_with_projection,
    _build_marcel_prior,
    _compute_age,
)

# ---------------------------------------------------------------------------
# _compute_age
# ---------------------------------------------------------------------------


class TestComputeAge:
    def test_normal_birth_date(self):
        assert _compute_age("1997-05-15", 2026) == 29

    def test_none_returns_none(self):
        assert _compute_age(None) is None

    def test_empty_string(self):
        assert _compute_age("") is None

    def test_invalid_format(self):
        assert _compute_age("not-a-date") is None

    def test_unreasonable_year(self):
        assert _compute_age("1960-01-01") is None


# ---------------------------------------------------------------------------
# _build_marcel_prior
# ---------------------------------------------------------------------------


class TestBuildMarcelPrior:
    def _make_stats(self, years_data):
        """Build stats_by_year dict from {year: {col: val}} mapping."""
        result = {}
        for year, data in years_data.items():
            df = pd.DataFrame([data])
            df.index = [data.get("player_id", 1)]
            result[year] = df
        return result

    def test_three_years_weighted(self):
        """Marcel 5/4/3 weighting across 3 seasons."""
        stats = self._make_stats(
            {
                2024: {"player_id": 1, "pa": 500, "r": 75, "hr": 25, "rbi": 80, "sb": 10, "avg": 0.270, "obp": 0.340},
                2025: {"player_id": 1, "pa": 600, "r": 90, "hr": 30, "rbi": 95, "sb": 15, "avg": 0.280, "obp": 0.350},
                2026: {"player_id": 1, "pa": 50, "r": 8, "hr": 3, "rbi": 10, "sb": 2, "avg": 0.260, "obp": 0.320},
            }
        )
        result = _build_marcel_prior(1, stats, ["r", "hr", "rbi", "sb"], ["avg", "obp"], "pa")

        # Total weighted volume: 50*5 + 600*4 + 500*3 = 250 + 2400 + 1500 = 4150
        assert result["total_weighted_vol"] == pytest.approx(4150.0)

        # HR rate: (3*5 + 30*4 + 25*3) / 4150 = (15+120+75)/4150 = 210/4150
        assert result["hr_rate"] == pytest.approx(210 / 4150, rel=1e-3)

    def test_single_year(self):
        """Player with only one year of data."""
        stats = self._make_stats(
            {2025: {"player_id": 1, "pa": 400, "r": 60, "hr": 20, "rbi": 65, "sb": 8, "avg": 0.265, "obp": 0.330}}
        )
        result = _build_marcel_prior(1, stats, ["r", "hr"], ["avg"], "pa")

        assert result["total_weighted_vol"] == pytest.approx(400 * 4)  # 2025 weight = 4
        assert result["hr_rate"] == pytest.approx(20 / 400)

    def test_no_history(self):
        """Player with no historical stats returns empty dict."""
        result = _build_marcel_prior(999, {}, ["r", "hr"], ["avg"], "pa")
        assert result == {}

    def test_pitcher_ip_based(self):
        """Pitcher stats use IP as volume."""
        stats = self._make_stats(
            {2025: {"player_id": 1, "ip": 180, "w": 12, "l": 6, "sv": 0, "k": 200, "era": 3.50, "whip": 1.15}}
        )
        result = _build_marcel_prior(1, stats, ["w", "k"], ["era", "whip"], "ip", is_rate_ip=True)

        assert result["k_rate"] == pytest.approx(200 / 180, rel=1e-3)
        assert result["era"] == pytest.approx(3.50, rel=1e-3)


# ---------------------------------------------------------------------------
# _blend_with_projection
# ---------------------------------------------------------------------------


class TestBlendWithProjection:
    def test_no_history_uses_projection(self):
        """With no Marcel history, blend returns 100% projection."""
        marcel = {}  # no history
        proj = pd.Series({"pa": 550, "r": 80, "hr": 25, "avg": 0.265, "obp": 0.335})
        result = _blend_with_projection(marcel, proj, ["r", "hr"], ["avg", "obp"], "pa")

        assert result["avg"] == pytest.approx(0.265)
        assert result["hr_rate"] == pytest.approx(25 / 550, rel=1e-3)

    def test_full_history_blends(self):
        """With substantial history, blend shifts toward Marcel."""
        marcel = {
            "total_weighted_vol": 1500.0,  # above threshold → reliability = 1500/1200 capped at 1.0
            "hr_rate": 0.040,
            "avg": 0.280,
        }
        proj = pd.Series({"pa": 600, "hr": 20, "avg": 0.260})
        result = _blend_with_projection(marcel, proj, ["hr"], ["avg"], "pa")

        # reliability = min(1.0, 1500/1200) = 1.0 → 100% Marcel
        assert result["avg"] == pytest.approx(0.280, rel=1e-3)
        assert result["hr_rate"] == pytest.approx(0.040, rel=1e-3)

    def test_partial_history_blends(self):
        """With partial history, blend is proportional."""
        marcel = {
            "total_weighted_vol": 600.0,  # reliability = 600/1200 = 0.5
            "hr_rate": 0.050,
            "avg": 0.290,
        }
        proj = pd.Series({"pa": 500, "hr": 20, "avg": 0.250})
        result = _blend_with_projection(marcel, proj, ["hr"], ["avg"], "pa")

        # reliability = 0.5 → 50% Marcel, 50% projection
        assert result["avg"] == pytest.approx(0.5 * 0.290 + 0.5 * 0.250, rel=1e-3)
        proj_hr_rate = 20 / 500
        assert result["hr_rate"] == pytest.approx(0.5 * 0.050 + 0.5 * proj_hr_rate, rel=1e-3)


# ---------------------------------------------------------------------------
# Regression behavior (via BayesianUpdater)
# ---------------------------------------------------------------------------


class TestRegressionBehavior:
    def test_tiny_sample_barely_shifts(self):
        """12 PA should give ~1.3% weight to observed AVG."""
        updater = BayesianUpdater()
        prior = 0.265
        observed = 0.000  # Durbin's 0-for-12
        result = updater.regressed_rate_with_prior(observed, 12, prior, STABILIZATION_POINTS["avg"])

        # At 12 PA, stab=910: (0.000*12 + 0.265*910)/(12+910) = 241.15/922 = 0.2615
        expected = (observed * 12 + prior * 910) / (12 + 910)
        assert result == pytest.approx(expected, rel=1e-4)
        # Should be very close to prior
        assert abs(result - prior) < 0.005

    def test_large_sample_shifts_toward_observed(self):
        """500 PA should give ~35% weight to observed AVG."""
        updater = BayesianUpdater()
        prior = 0.265
        observed = 0.300
        result = updater.regressed_rate_with_prior(observed, 500, prior, STABILIZATION_POINTS["avg"])

        weight = 500 / (500 + 910)
        expected = weight * observed + (1 - weight) * prior
        assert result == pytest.approx(expected, rel=1e-4)
        assert result > prior  # Should shift toward .300

    def test_era_regression(self):
        """ERA regression: small IP sample stays near prior."""
        updater = BayesianUpdater()
        prior_era = 3.80
        observed_era = 1.00  # Great 5 IP
        result = updater.regressed_rate_with_prior(observed_era, 5, prior_era, STABILIZATION_POINTS["era"])

        # 5 IP, stab=70: mostly prior
        assert result > 3.0  # Should stay close to 3.80, not drop to 1.00

    def test_zero_sample_returns_prior(self):
        """Zero PA/IP should return the prior exactly."""
        updater = BayesianUpdater()
        result = updater.regressed_rate_with_prior(0.400, 0, 0.265, 910)
        assert result == pytest.approx(0.265)


# ---------------------------------------------------------------------------
# Age adjustment
# ---------------------------------------------------------------------------


class TestAgeAdjustment:
    def test_peak_age_no_adjustment(self):
        updater = BayesianUpdater()
        assert updater.age_adjustment(27, "hr") == pytest.approx(1.0)

    def test_old_hitter_decline(self):
        """35-year-old power hitter should decline."""
        updater = BayesianUpdater()
        mult = updater.age_adjustment(35, "hr")
        # 8 years past peak 27, decline = 0.015/year → 1.0 - 8*0.015 = 0.88
        assert mult == pytest.approx(0.88, rel=1e-2)

    def test_old_speed_declines_faster(self):
        """Speed declines faster than power."""
        updater = BayesianUpdater()
        speed_mult = updater.age_adjustment(33, "sb")
        power_mult = updater.age_adjustment(33, "hr")
        assert speed_mult < power_mult

    def test_floor_at_50_percent(self):
        """Age multiplier floors at 0.5 for extreme ages."""
        updater = BayesianUpdater()
        # power peak=27, decline=0.015/yr → floor hit at 27 + 34 = age 61
        mult = updater.age_adjustment(65, "hr")
        assert mult == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Counting stat ROS projection math
# ---------------------------------------------------------------------------


class TestROSProjectionMath:
    def test_ros_equals_remaining(self):
        """ROS counting stats = updated_rate * remaining_PA."""
        rate = 0.04  # HR rate
        total_pa = 600
        observed_pa = 50
        observed_hr = 3

        remaining_pa = total_pa - observed_pa
        full_season_hr = observed_hr + rate * remaining_pa
        ros_hr = full_season_hr - observed_hr

        assert ros_hr == pytest.approx(rate * remaining_pa)

    def test_ros_never_negative(self):
        """ROS should never go negative even with overperformance."""
        rate = 0.03
        total_pa = 500
        observed_pa = 500  # season over
        remaining_pa = max(0, total_pa - observed_pa)

        ros = int(rate * remaining_pa)
        assert ros == 0

"""Tests for I3 Ratio Protection Calculator and J4 Comprehensive Weather Model."""

from __future__ import annotations

import pytest

from src.optimizer.pivot_advisor import compute_ratio_protection
from src.optimizer.matchup_adjustments import (
    weather_rain_adjustment,
    weather_wind_hr_adjustment,
)


# ── I3: Ratio Protection Calculator ─────────────────────────────────


class TestComputeRatioProtection:
    """Tests for compute_ratio_protection()."""

    def test_era_risk_positive_when_pitcher_era_higher(self):
        """ERA risk should be positive when pitcher ERA > current team ERA."""
        result = compute_ratio_protection(
            current_era=3.00,
            current_whip=1.10,
            banked_ip=30.0,
            pitcher_proj_era=5.00,
            pitcher_proj_ip=6.0,
        )
        assert result["era_risk"] > 0, "ERA risk should be positive when pitcher ERA exceeds team ERA"
        assert result["era_after"] > 3.00, "ERA after should be higher than current ERA"

    def test_era_risk_negative_when_pitcher_era_lower(self):
        """ERA risk should be negative (improving) when pitcher ERA < current team ERA."""
        result = compute_ratio_protection(
            current_era=4.50,
            current_whip=1.30,
            banked_ip=30.0,
            pitcher_proj_era=2.50,
            pitcher_proj_ip=6.0,
        )
        assert result["era_risk"] < 0, "ERA risk should be negative when pitcher ERA is below team ERA"

    def test_bench_recommendation_when_risk_exceeds_half_lead(self):
        """Should recommend BENCH when ERA risk > 50% of the ERA lead."""
        result = compute_ratio_protection(
            current_era=3.00,
            current_whip=1.10,
            banked_ip=20.0,
            pitcher_proj_era=6.00,
            pitcher_proj_ip=6.0,
            era_lead=0.50,  # Leading by 0.50 ERA
        )
        # pitcher ERA is 6.00 vs team 3.00 with only 20 IP banked
        # era_after = (3.00*20 + 6.00*6) / 26 = 96/26 = 3.692
        # era_risk = 3.692 - 3.00 = 0.692, which > 0.50 * 0.5 = 0.25
        assert result["recommend"] == "BENCH", f"Expected BENCH, got {result['recommend']}"

    def test_start_recommendation_when_risk_is_small(self):
        """Should recommend START when risk is negligible."""
        result = compute_ratio_protection(
            current_era=3.50,
            current_whip=1.20,
            banked_ip=50.0,
            pitcher_proj_era=3.60,
            pitcher_proj_ip=6.0,
            era_lead=0.0,
        )
        # era_after = (3.50*50 + 3.60*6) / 56 = 196.6/56 = 3.511
        # era_risk = 0.011 — very small
        assert result["recommend"] == "START", f"Expected START, got {result['recommend']}"

    def test_risky_recommendation_for_high_era_risk_no_lead(self):
        """Should recommend RISKY when ERA risk > 0.30 without a specific lead."""
        result = compute_ratio_protection(
            current_era=3.00,
            current_whip=1.10,
            banked_ip=10.0,
            pitcher_proj_era=7.00,
            pitcher_proj_ip=6.0,
            era_lead=0.0,
        )
        # era_after = (3.00*10 + 7.00*6) / 16 = 72/16 = 4.50
        # era_risk = 1.50 > 0.30
        assert result["recommend"] == "RISKY", f"Expected RISKY, got {result['recommend']}"

    def test_zero_banked_ip_returns_start(self):
        """Should return START with zero risk when no IP banked."""
        result = compute_ratio_protection(
            current_era=0.0,
            current_whip=0.0,
            banked_ip=0.0,
            pitcher_proj_era=4.00,
            pitcher_proj_ip=0.0,
        )
        assert result["recommend"] == "START"
        assert result["era_risk"] == 0.0

    def test_whip_risk_computed(self):
        """WHIP risk should be computed when pitcher WHIP > current."""
        result = compute_ratio_protection(
            current_era=3.00,
            current_whip=1.00,
            banked_ip=30.0,
            pitcher_proj_era=3.50,
            pitcher_proj_whip=1.60,
            pitcher_proj_ip=6.0,
        )
        assert result["whip_risk"] > 0, "WHIP risk should be positive when pitcher WHIP exceeds team WHIP"
        assert result["whip_after"] > 1.00

    def test_bench_on_whip_lead_risk(self):
        """Should recommend BENCH when WHIP risk exceeds 50% of WHIP lead."""
        result = compute_ratio_protection(
            current_era=3.00,
            current_whip=1.00,
            banked_ip=15.0,
            pitcher_proj_era=3.00,  # ERA is fine
            pitcher_proj_whip=1.80,
            pitcher_proj_ip=6.0,
            era_lead=0.0,
            whip_lead=0.20,
        )
        # whip_after = (1.00*15 + 1.80*6) / 21 = 25.8/21 = 1.229
        # whip_risk = 0.229 > 0.20 * 0.5 = 0.10
        assert result["recommend"] == "BENCH"


# ── J4: Comprehensive Weather Model ─────────────────────────────────


class TestWeatherRainAdjustment:
    """Tests for weather_rain_adjustment()."""

    def test_no_rain_returns_neutral(self):
        """No rain (0%) should return 1.0 for both K and BB."""
        result = weather_rain_adjustment(0.0)
        assert result["k_mult"] == 1.0
        assert result["bb_mult"] == 1.0

    def test_negative_precip_returns_neutral(self):
        """Negative precip should return neutral."""
        result = weather_rain_adjustment(-5.0)
        assert result["k_mult"] == 1.0
        assert result["bb_mult"] == 1.0

    def test_rain_50_pct_k_mult_approx_090(self):
        """Rain 50% (capped at 40%) should give K mult ~0.899."""
        result = weather_rain_adjustment(50.0)
        assert abs(result["k_mult"] - 0.899) < 0.01, f"K mult should be ~0.899, got {result['k_mult']}"

    def test_rain_50_pct_bb_mult_approx_110(self):
        """Rain 50% should give BB mult ~1.096."""
        result = weather_rain_adjustment(50.0)
        assert abs(result["bb_mult"] - 1.096) < 0.01, f"BB mult should be ~1.096, got {result['bb_mult']}"

    def test_rain_20_pct_partial_effect(self):
        """Rain 20% should give half the full effect (rain_factor=0.5)."""
        result = weather_rain_adjustment(20.0)
        expected_k = 1.0 - 0.101 * 0.5  # 0.9495
        expected_bb = 1.0 + 0.096 * 0.5  # 1.048
        assert abs(result["k_mult"] - expected_k) < 0.001
        assert abs(result["bb_mult"] - expected_bb) < 0.001

    def test_rain_40_pct_full_effect(self):
        """Rain at exactly 40% should give full effect."""
        result = weather_rain_adjustment(40.0)
        assert abs(result["k_mult"] - 0.899) < 0.001
        assert abs(result["bb_mult"] - 1.096) < 0.001


class TestWeatherWindHRAdjustment:
    """Tests for weather_wind_hr_adjustment()."""

    def test_low_wind_returns_neutral(self):
        """Wind <5 mph should return 1.0 regardless of direction."""
        assert weather_wind_hr_adjustment(3.0, wind_out=True) == 1.0
        assert weather_wind_hr_adjustment(4.9, wind_out=False) == 1.0
        assert weather_wind_hr_adjustment(0.0) == 1.0

    def test_wind_out_15_mph_hr_mult_approx_140(self):
        """Wind out at 15 mph should give HR mult ~1.40 (capped at 1.20)."""
        result = weather_wind_hr_adjustment(15.0, wind_out=True)
        # (15-5) * 0.04 = 0.40, capped at 0.20 -> 1.20
        assert abs(result - 1.20) < 0.01, f"HR mult should be ~1.20, got {result}"

    def test_wind_out_10_mph(self):
        """Wind out at 10 mph -> (10-5)*0.04 = 0.20 -> 1.20."""
        result = weather_wind_hr_adjustment(10.0, wind_out=True)
        assert abs(result - 1.20) < 0.01

    def test_wind_out_7_mph(self):
        """Wind out at 7 mph -> (7-5)*0.04 = 0.08 -> 1.08."""
        result = weather_wind_hr_adjustment(7.0, wind_out=True)
        assert abs(result - 1.08) < 0.01

    def test_wind_in_15_mph_hr_mult_approx_085(self):
        """Wind in at 15 mph should give HR mult ~0.70, floored at 0.85."""
        result = weather_wind_hr_adjustment(15.0, wind_out=False)
        # (15-5) * 0.03 = 0.30, 1.0 - 0.30 = 0.70, floor 0.85
        assert abs(result - 0.85) < 0.01, f"HR mult should be ~0.85, got {result}"

    def test_wind_in_8_mph(self):
        """Wind in at 8 mph -> 1.0 - (8-5)*0.03 = 0.91."""
        result = weather_wind_hr_adjustment(8.0, wind_out=False)
        assert abs(result - 0.91) < 0.01

    def test_wind_out_cap_at_20_pct(self):
        """Wind out multiplier should cap at +20% (1.20)."""
        result = weather_wind_hr_adjustment(30.0, wind_out=True)
        assert abs(result - 1.20) < 0.001, "Should cap at 1.20"

    def test_wind_in_floor_at_85_pct(self):
        """Wind in multiplier should floor at 0.85."""
        result = weather_wind_hr_adjustment(30.0, wind_out=False)
        assert abs(result - 0.85) < 0.001, "Should floor at 0.85"

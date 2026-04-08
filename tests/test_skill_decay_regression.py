"""Tests for L2 (K% Skill Decay) and M4 (Regression Alerts)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.alerts import generate_regression_alerts
from src.closer_monitor import compute_skill_decay

# ─── L2: K% Skill Decay ─────────────────────────────────────────────


class TestComputeSkillDecay:
    """Tests for compute_skill_decay()."""

    def test_critical_on_large_k_drop(self):
        """K% drop of 10 points triggers CRITICAL."""
        result = compute_skill_decay(
            season_k_pct=28.0,
            recent_k_pct=18.0,
            season_kbb_pct=18.0,
            recent_kbb_pct=12.0,
        )
        assert result["severity"] == "CRITICAL"
        assert result["k_pct_drop"] == 10.0
        assert "K% dropped 10.0 pts" in result["message"]

    def test_warning_on_moderate_k_drop(self):
        """K% drop of 6 points triggers WARNING."""
        result = compute_skill_decay(
            season_k_pct=25.0,
            recent_k_pct=19.0,
            season_kbb_pct=15.0,
            recent_kbb_pct=12.0,
        )
        assert result["severity"] == "WARNING"
        assert result["k_pct_drop"] == 6.0
        assert "K% declining" in result["message"]

    def test_critical_on_low_kbb(self):
        """K-BB% < 10 triggers CRITICAL regardless of K% drop."""
        result = compute_skill_decay(
            season_k_pct=22.0,
            recent_k_pct=21.0,  # Only 1pt drop — not enough for K% trigger
            season_kbb_pct=12.0,
            recent_kbb_pct=8.5,
        )
        assert result["severity"] == "CRITICAL"
        assert result["kbb_warning"] is True
        assert "K-BB% at 8.5%" in result["message"]

    def test_none_when_no_decline(self):
        """No decline produces NONE severity."""
        result = compute_skill_decay(
            season_k_pct=25.0,
            recent_k_pct=26.0,  # Actually improved
            season_kbb_pct=15.0,
            recent_kbb_pct=16.0,
        )
        assert result["severity"] == "NONE"
        assert result["message"] == ""
        assert result["kbb_warning"] is False

    def test_exact_threshold_8_is_critical(self):
        """K% drop of exactly 8.0 hits the CRITICAL threshold."""
        result = compute_skill_decay(
            season_k_pct=30.0,
            recent_k_pct=22.0,
            season_kbb_pct=20.0,
            recent_kbb_pct=14.0,
        )
        assert result["severity"] == "CRITICAL"

    def test_just_below_warning_is_none(self):
        """K% drop of 4.9 is below WARNING threshold."""
        result = compute_skill_decay(
            season_k_pct=25.0,
            recent_k_pct=20.1,
            season_kbb_pct=15.0,
            recent_kbb_pct=12.0,
        )
        assert result["severity"] == "NONE"

    def test_kbb_exactly_10_is_not_warning(self):
        """K-BB% at exactly 10.0 does NOT trigger kbb_warning (< 10 required)."""
        result = compute_skill_decay(
            season_k_pct=22.0,
            recent_k_pct=22.0,
            season_kbb_pct=12.0,
            recent_kbb_pct=10.0,
        )
        assert result["kbb_warning"] is False


# ─── M4: Regression Alerts ──────────────────────────────────────────


class TestGenerateRegressionAlerts:
    """Tests for generate_regression_alerts()."""

    def _make_roster(self, rows: list[dict]) -> pd.DataFrame:
        """Helper to build a roster DataFrame from dicts."""
        return pd.DataFrame(rows)

    def test_buy_low_when_xwoba_exceeds_actual(self):
        """BUY_LOW when xwOBA >> actual wOBA (gap > 0.030)."""
        roster = self._make_roster(
            [
                {
                    "name": "Player A",
                    "obp": 0.300,  # wOBA approx = 0.345
                    "xwoba": 0.400,  # delta = 0.055
                    "ytd_pa": 100,
                }
            ]
        )
        alerts = generate_regression_alerts(roster)
        assert len(alerts) == 1
        assert alerts[0]["alert_type"] == "BUY_LOW"
        assert alerts[0]["player_name"] == "Player A"
        assert alerts[0]["divergence_sd"] > 1.5

    def test_sell_high_when_actual_exceeds_xwoba(self):
        """SELL_HIGH when actual wOBA >> xwOBA."""
        roster = self._make_roster(
            [
                {
                    "name": "Player B",
                    "obp": 0.380,  # wOBA approx = 0.437
                    "xwoba": 0.380,  # delta = -0.057
                    "ytd_pa": 100,
                }
            ]
        )
        alerts = generate_regression_alerts(roster)
        assert len(alerts) == 1
        assert alerts[0]["alert_type"] == "SELL_HIGH"
        assert alerts[0]["player_name"] == "Player B"

    def test_no_alert_when_gap_small(self):
        """No alert when gap < 0.030 (1.5 * 0.020 SD)."""
        roster = self._make_roster(
            [
                {
                    "name": "Player C",
                    "obp": 0.340,  # wOBA approx = 0.391
                    "xwoba": 0.400,  # delta = 0.009 (< 0.030)
                    "ytd_pa": 100,
                }
            ]
        )
        alerts = generate_regression_alerts(roster)
        assert len(alerts) == 0

    def test_no_alert_when_pa_below_min(self):
        """No alert when PA < 50."""
        roster = self._make_roster(
            [
                {
                    "name": "Player D",
                    "obp": 0.250,
                    "xwoba": 0.400,
                    "ytd_pa": 30,
                }
            ]
        )
        alerts = generate_regression_alerts(roster)
        assert len(alerts) == 0

    def test_alerts_sorted_by_divergence(self):
        """Alerts are sorted by divergence magnitude (descending)."""
        roster = self._make_roster(
            [
                {
                    "name": "Small Gap",
                    "obp": 0.310,  # wOBA ~ 0.3565, delta = 0.0435
                    "xwoba": 0.400,
                    "ytd_pa": 100,
                },
                {
                    "name": "Big Gap",
                    "obp": 0.260,  # wOBA ~ 0.299, delta = 0.101
                    "xwoba": 0.400,
                    "ytd_pa": 100,
                },
            ]
        )
        alerts = generate_regression_alerts(roster)
        assert len(alerts) == 2
        assert alerts[0]["player_name"] == "Big Gap"
        assert alerts[1]["player_name"] == "Small Gap"
        assert alerts[0]["divergence_sd"] > alerts[1]["divergence_sd"]

    def test_no_alert_when_xwoba_zero(self):
        """No alert when xwOBA is 0 or missing."""
        roster = self._make_roster(
            [
                {
                    "name": "No Statcast",
                    "obp": 0.350,
                    "xwoba": 0,
                    "ytd_pa": 100,
                }
            ]
        )
        alerts = generate_regression_alerts(roster)
        assert len(alerts) == 0

    def test_empty_roster(self):
        """Empty roster returns no alerts."""
        roster = pd.DataFrame()
        alerts = generate_regression_alerts(roster)
        assert alerts == []

    def test_uses_player_name_fallback(self):
        """Falls back to player_name column if name is missing."""
        roster = self._make_roster(
            [
                {
                    "player_name": "Fallback Name",
                    "obp": 0.280,
                    "xwoba": 0.400,
                    "ytd_pa": 100,
                }
            ]
        )
        alerts = generate_regression_alerts(roster)
        assert len(alerts) == 1
        assert alerts[0]["player_name"] == "Fallback Name"

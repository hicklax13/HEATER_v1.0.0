"""Test injury risk model: health scores, age risk, workload flags, adjustments, badges."""

import pandas as pd
import pytest

from src.injury_model import (
    age_risk_adjustment,
    apply_injury_adjustment,
    compute_health_score,
    get_injury_badge,
    workload_flag,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def full_health_seasons():
    """3 seasons of 162/162 games."""
    return [162, 162, 162], [162, 162, 162]


@pytest.fixture
def partial_health_seasons():
    """3 seasons with varying games played."""
    return [120, 140, 100], [162, 162, 162]


# ---------------------------------------------------------------------------
# compute_health_score
# ---------------------------------------------------------------------------


class TestHealthScore:
    def test_health_score_full_health(self, full_health_seasons):
        games_played, games_available = full_health_seasons
        score = compute_health_score(games_played, games_available)
        assert score == pytest.approx(1.0)

    def test_health_score_partial(self, partial_health_seasons):
        games_played, games_available = partial_health_seasons
        score = compute_health_score(games_played, games_available)
        expected = (120 / 162 + 140 / 162 + 100 / 162) / 3
        assert score == pytest.approx(expected)

    def test_health_score_missing_seasons(self):
        """Only 1 season provided; missing years padded with 0.85 default."""
        games_played = [140]
        games_available = [162]
        score = compute_health_score(games_played, games_available)
        expected = (140 / 162 + 0.85 + 0.85) / 3
        assert score == pytest.approx(expected)

    def test_health_score_empty(self):
        """Empty lists return 0.85 (league average)."""
        score = compute_health_score([], [])
        assert score == pytest.approx(0.85)

    def test_health_score_zero_available(self):
        """Zero games available in one season handled gracefully."""
        games_played = [0, 162, 162]
        games_available = [0, 162, 162]
        score = compute_health_score(games_played, games_available)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# age_risk_adjustment
# ---------------------------------------------------------------------------


class TestAgeRisk:
    def test_age_risk_young_hitter(self):
        """Age 25 hitter returns 1.0 (no penalty)."""
        factor = age_risk_adjustment(25, is_pitcher=False)
        assert factor == pytest.approx(1.0)

    def test_age_risk_old_hitter(self):
        """Age 35 hitter: 1.0 - (35-30)*0.02 = 0.90."""
        factor = age_risk_adjustment(35, is_pitcher=False)
        assert factor == pytest.approx(0.90)

    def test_age_risk_old_pitcher(self):
        """Age 35 pitcher: 1.0 - (35-28)*0.03 = 0.79."""
        factor = age_risk_adjustment(35, is_pitcher=True)
        assert factor == pytest.approx(0.79)

    def test_age_risk_floor(self):
        """Very old player never drops below 0.5."""
        factor = age_risk_adjustment(50, is_pitcher=True)
        assert factor >= 0.5


# ---------------------------------------------------------------------------
# workload_flag
# ---------------------------------------------------------------------------


class TestWorkloadFlag:
    def test_workload_flag_over(self):
        """200 IP vs 150 previous → flagged (50 > 40 threshold)."""
        result = workload_flag(ip_current=200.0, ip_previous=150.0)
        assert result is True

    def test_workload_flag_under(self):
        """170 IP vs 160 previous → not flagged (10 < 40 threshold)."""
        result = workload_flag(ip_current=170.0, ip_previous=160.0)
        assert result is False

    def test_workload_flag_none(self):
        """None values → not flagged."""
        assert workload_flag(ip_current=None, ip_previous=150.0) is False
        assert workload_flag(ip_current=200.0, ip_previous=None) is False
        assert workload_flag(ip_current=None, ip_previous=None) is False


# ---------------------------------------------------------------------------
# apply_injury_adjustment
# ---------------------------------------------------------------------------


class TestApplyInjuryAdjustment:
    def test_apply_reduces_counting_stats(self):
        """health_score=0.8 reduces counting stats by 20%."""
        projections = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "pa": 600,
                    "ab": 550,
                    "h": 160,
                    "r": 100,
                    "hr": 40,
                    "rbi": 110,
                    "sb": 10,
                    "avg": 0.291,
                    "ip": 0,
                    "w": 0,
                    "sv": 0,
                    "k": 0,
                    "era": 0,
                    "whip": 0,
                    "er": 0,
                    "bb_allowed": 0,
                    "h_allowed": 0,
                }
            ]
        )
        health = pd.DataFrame([{"player_id": 1, "health_score": 0.8}])
        adjusted = apply_injury_adjustment(projections, health)
        row = adjusted.iloc[0]
        # Counting stats should be reduced
        assert row["r"] < 100
        assert row["hr"] < 40
        assert row["rbi"] < 110
        assert row["sb"] < 10

    def test_full_health_no_change(self):
        """health_score=1.0 should leave stats unchanged."""
        projections = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "pa": 600,
                    "ab": 550,
                    "h": 160,
                    "r": 100,
                    "hr": 40,
                    "rbi": 110,
                    "sb": 10,
                    "avg": 0.291,
                    "ip": 0,
                    "w": 0,
                    "sv": 0,
                    "k": 0,
                    "era": 0,
                    "whip": 0,
                    "er": 0,
                    "bb_allowed": 0,
                    "h_allowed": 0,
                }
            ]
        )
        health = pd.DataFrame([{"player_id": 1, "health_score": 1.0}])
        adjusted = apply_injury_adjustment(projections, health)
        row = adjusted.iloc[0]
        assert row["r"] == pytest.approx(100, abs=1)
        assert row["hr"] == pytest.approx(40, abs=1)


# ---------------------------------------------------------------------------
# get_injury_badge
# ---------------------------------------------------------------------------


class TestInjuryBadge:
    def test_injury_badge_low(self):
        icon, label = get_injury_badge(0.95)
        assert "#84cc16" in icon  # green dot
        assert label == "Low Risk"

    def test_injury_badge_moderate(self):
        icon, label = get_injury_badge(0.80)
        assert "#fb923c" in icon  # amber dot
        assert label == "Moderate Risk"

    def test_injury_badge_high(self):
        icon, label = get_injury_badge(0.60)
        assert "#f43f5e" in icon  # red dot
        assert label == "High Risk"

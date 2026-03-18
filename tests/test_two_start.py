"""Tests for the Two-Start Pitcher Planner (src/two_start.py).

Covers:
  - identify_two_start_pitchers with mocked schedule data
  - compute_pitcher_matchup_score with various pitcher/opponent combos
  - rate_stat_damage with hand-calculated expected values
  - fetch_team_batting_stats with fallback behavior
  - Edge cases: empty schedule, no two-start pitchers, zero IP
"""

from __future__ import annotations

from datetime import UTC
from unittest.mock import MagicMock, patch

import pytest

from src.two_start import (
    _confidence_tier,
    _normalize,
    clear_team_batting_cache,
    compute_pitcher_matchup_score,
    fetch_team_batting_stats,
    identify_two_start_pitchers,
    rate_stat_damage,
)

# ── rate_stat_damage tests ───────────────────────────────────────────


class TestRateStatDamage:
    """Tests for rate_stat_damage with hand-calculated values."""

    def test_pitcher_matches_team_no_change(self):
        """When pitcher ERA/WHIP equals team ERA/WHIP, damage is zero."""
        result = rate_stat_damage(
            pitcher_era=4.00,
            pitcher_whip=1.25,
            pitcher_ip=6.0,
            team_era=4.00,
            team_whip=1.25,
            team_ip=54.0,
        )
        assert result["era_change"] == 0.0
        assert result["whip_change"] == 0.0

    def test_bad_pitcher_raises_era(self):
        """Pitcher with higher ERA than team raises team ERA."""
        # era_change = (6.00 - 4.00) * 6 / (54 + 6) = 2.0 * 6 / 60 = 0.2
        result = rate_stat_damage(
            pitcher_era=6.00,
            pitcher_whip=1.25,
            pitcher_ip=6.0,
            team_era=4.00,
            team_whip=1.25,
            team_ip=54.0,
        )
        assert result["era_change"] == pytest.approx(0.2, abs=1e-6)
        assert result["whip_change"] == pytest.approx(0.0, abs=1e-6)

    def test_good_pitcher_lowers_era(self):
        """Pitcher with lower ERA than team lowers team ERA."""
        # era_change = (2.50 - 4.00) * 5 / (55 + 5) = -1.5 * 5 / 60 = -0.125
        result = rate_stat_damage(
            pitcher_era=2.50,
            pitcher_whip=1.25,
            pitcher_ip=5.0,
            team_era=4.00,
            team_whip=1.25,
            team_ip=55.0,
        )
        assert result["era_change"] == pytest.approx(-0.125, abs=1e-6)

    def test_whip_change_calculation(self):
        """WHIP change follows same formula as ERA change."""
        # whip_change = (1.50 - 1.20) * 6 / (54 + 6) = 0.30 * 6 / 60 = 0.03
        result = rate_stat_damage(
            pitcher_era=4.00,
            pitcher_whip=1.50,
            pitcher_ip=6.0,
            team_era=4.00,
            team_whip=1.20,
            team_ip=54.0,
        )
        assert result["whip_change"] == pytest.approx(0.03, abs=1e-6)
        assert result["era_change"] == pytest.approx(0.0, abs=1e-6)

    def test_both_stats_change(self):
        """Both ERA and WHIP can change simultaneously."""
        # era_change = (5.00 - 3.50) * 5 / (45 + 5) = 1.5 * 5 / 50 = 0.15
        # whip_change = (1.40 - 1.10) * 5 / (45 + 5) = 0.30 * 5 / 50 = 0.03
        result = rate_stat_damage(
            pitcher_era=5.00,
            pitcher_whip=1.40,
            pitcher_ip=5.0,
            team_era=3.50,
            team_whip=1.10,
            team_ip=45.0,
        )
        assert result["era_change"] == pytest.approx(0.15, abs=1e-6)
        assert result["whip_change"] == pytest.approx(0.03, abs=1e-6)

    def test_zero_total_ip(self):
        """Zero team IP + zero pitcher IP returns zero change."""
        result = rate_stat_damage(
            pitcher_era=4.00,
            pitcher_whip=1.25,
            pitcher_ip=0.0,
            team_era=4.00,
            team_whip=1.25,
            team_ip=0.0,
        )
        assert result["era_change"] == 0.0
        assert result["whip_change"] == 0.0

    def test_zero_pitcher_ip(self):
        """Zero pitcher IP means zero change (numerator is zero)."""
        result = rate_stat_damage(
            pitcher_era=6.00,
            pitcher_whip=2.00,
            pitcher_ip=0.0,
            team_era=4.00,
            team_whip=1.25,
            team_ip=55.0,
        )
        assert result["era_change"] == pytest.approx(0.0, abs=1e-6)
        assert result["whip_change"] == pytest.approx(0.0, abs=1e-6)

    def test_large_team_ip_dilution(self):
        """Large team IP dilutes the pitcher's impact."""
        # era_change = (6.00 - 4.00) * 5 / (500 + 5) = 10 / 505 ~ 0.0198
        result = rate_stat_damage(
            pitcher_era=6.00,
            pitcher_whip=1.25,
            pitcher_ip=5.0,
            team_era=4.00,
            team_whip=1.25,
            team_ip=500.0,
        )
        assert result["era_change"] == pytest.approx(10.0 / 505.0, abs=1e-6)


# ── compute_pitcher_matchup_score tests ──────────────────────────────


class TestPitcherMatchupScore:
    """Tests for compute_pitcher_matchup_score."""

    def test_league_average_pitcher_default_opponent(self):
        """League-average pitcher vs league-average opponent near 5.0."""
        score = compute_pitcher_matchup_score(
            pitcher_stats={},
            opponent_team_stats=None,
            park_factor=1.0,
            is_home=True,
        )
        assert 3.0 <= score <= 7.0

    def test_elite_pitcher_high_score(self):
        """Elite pitcher stats should produce a high score."""
        score = compute_pitcher_matchup_score(
            pitcher_stats={"k_bb_pct": 0.28, "xfip": 2.80, "csw_pct": 0.33},
            opponent_team_stats={"wrc_plus": 85.0, "k_pct": 0.27},
            park_factor=0.93,
            is_home=True,
        )
        assert score >= 6.0

    def test_bad_pitcher_low_score(self):
        """Poor pitcher stats should produce a low score."""
        score = compute_pitcher_matchup_score(
            pitcher_stats={"k_bb_pct": -0.02, "xfip": 5.50, "csw_pct": 0.23},
            opponent_team_stats={"wrc_plus": 115.0, "k_pct": 0.18},
            park_factor=1.38,
            is_home=False,
        )
        assert score <= 4.0

    def test_home_better_than_away(self):
        """Home pitcher should score higher than away, all else equal."""
        home = compute_pitcher_matchup_score(
            pitcher_stats={"k_bb_pct": 0.12},
            opponent_team_stats=None,
            park_factor=1.0,
            is_home=True,
        )
        away = compute_pitcher_matchup_score(
            pitcher_stats={"k_bb_pct": 0.12},
            opponent_team_stats=None,
            park_factor=1.0,
            is_home=False,
        )
        assert home > away

    def test_pitcher_friendly_park_helps(self):
        """Pitcher-friendly park (low PF) should give higher score."""
        friendly = compute_pitcher_matchup_score(
            pitcher_stats={},
            opponent_team_stats=None,
            park_factor=0.88,
            is_home=True,
        )
        hostile = compute_pitcher_matchup_score(
            pitcher_stats={},
            opponent_team_stats=None,
            park_factor=1.38,
            is_home=True,
        )
        assert friendly > hostile

    def test_weak_opponent_helps(self):
        """Facing a weak-hitting team (low wRC+, high K%) helps."""
        weak_opp = compute_pitcher_matchup_score(
            pitcher_stats={},
            opponent_team_stats={"wrc_plus": 80.0, "k_pct": 0.28},
            park_factor=1.0,
            is_home=True,
        )
        strong_opp = compute_pitcher_matchup_score(
            pitcher_stats={},
            opponent_team_stats={"wrc_plus": 120.0, "k_pct": 0.18},
            park_factor=1.0,
            is_home=True,
        )
        assert weak_opp > strong_opp

    def test_score_clipped_to_range(self):
        """Score should always be between 0 and 10."""
        score = compute_pitcher_matchup_score(
            pitcher_stats={"k_bb_pct": 0.50, "xfip": 1.00, "csw_pct": 0.50},
            opponent_team_stats={"wrc_plus": 50.0, "k_pct": 0.40},
            park_factor=0.50,
            is_home=True,
        )
        assert 0.0 <= score <= 10.0

    def test_zero_wrc_plus_clamped(self):
        """Zero or negative wRC+ should not cause division errors."""
        score = compute_pitcher_matchup_score(
            pitcher_stats={},
            opponent_team_stats={"wrc_plus": 0.0, "k_pct": 0.22},
            park_factor=1.0,
            is_home=True,
        )
        assert 0.0 <= score <= 10.0

    def test_zero_k_pct_uses_default(self):
        """Zero opponent K% should fall back gracefully."""
        score = compute_pitcher_matchup_score(
            pitcher_stats={},
            opponent_team_stats={"wrc_plus": 100.0, "k_pct": 0.0},
            park_factor=1.0,
            is_home=True,
        )
        assert 0.0 <= score <= 10.0


# ── _normalize tests ─────────────────────────────────────────────────


class TestNormalize:
    """Tests for the _normalize helper."""

    def test_mid_value(self):
        assert _normalize(5.0, 0.0, 10.0) == pytest.approx(0.5)

    def test_at_lower_bound(self):
        assert _normalize(0.0, 0.0, 10.0) == pytest.approx(0.0)

    def test_at_upper_bound(self):
        assert _normalize(10.0, 0.0, 10.0) == pytest.approx(1.0)

    def test_below_lower_clipped(self):
        assert _normalize(-5.0, 0.0, 10.0) == 0.0

    def test_above_upper_clipped(self):
        assert _normalize(15.0, 0.0, 10.0) == 1.0

    def test_equal_bounds_returns_half(self):
        assert _normalize(5.0, 5.0, 5.0) == 0.5


# ── _confidence_tier tests ───────────────────────────────────────────


class TestConfidenceTier:
    """Tests for confidence tier assignment."""

    def test_invalid_date_returns_low(self):
        assert _confidence_tier("not-a-date") == "LOW"

    def test_empty_date_returns_low(self):
        assert _confidence_tier("") == "LOW"

    @patch("src.two_start.datetime")
    def test_tomorrow_is_high(self, mock_dt):
        """Game 1 day ahead should be HIGH confidence."""
        from datetime import datetime as real_dt
        from datetime import timezone

        mock_dt.now.return_value = real_dt(2026, 7, 1, tzinfo=UTC)
        mock_dt.strptime = real_dt.strptime
        assert _confidence_tier("2026-07-02") == "HIGH"

    @patch("src.two_start.datetime")
    def test_four_days_is_medium(self, mock_dt):
        """Game 4 days ahead should be MEDIUM confidence."""
        from datetime import datetime as real_dt
        from datetime import timezone

        mock_dt.now.return_value = real_dt(2026, 7, 1, tzinfo=UTC)
        mock_dt.strptime = real_dt.strptime
        assert _confidence_tier("2026-07-05") == "MEDIUM"

    @patch("src.two_start.datetime")
    def test_seven_days_is_low(self, mock_dt):
        """Game 7 days ahead should be LOW confidence."""
        from datetime import datetime as real_dt
        from datetime import timezone

        mock_dt.now.return_value = real_dt(2026, 7, 1, tzinfo=UTC)
        mock_dt.strptime = real_dt.strptime
        assert _confidence_tier("2026-07-08") == "LOW"


# ── fetch_team_batting_stats tests ───────────────────────────────────


class TestFetchTeamBattingStats:
    """Tests for fetch_team_batting_stats with fallback behavior."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_team_batting_cache()

    def test_returns_empty_when_statsapi_unavailable(self):
        """Without statsapi, returns empty dict (graceful fallback)."""
        with patch.dict("sys.modules", {"statsapi": None}):
            clear_team_batting_cache()
            result = fetch_team_batting_stats()
            assert isinstance(result, dict)

    def test_cache_returns_same_object(self):
        """Second call returns cached result without re-fetching."""
        with patch.dict("sys.modules", {"statsapi": None}):
            clear_team_batting_cache()
            first = fetch_team_batting_stats()
            second = fetch_team_batting_stats()
            assert first is second

    def test_clear_cache_forces_refresh(self):
        """Clearing cache allows a fresh fetch."""
        with patch.dict("sys.modules", {"statsapi": None}):
            clear_team_batting_cache()
            first = fetch_team_batting_stats()
            clear_team_batting_cache()
            # After clearing, should re-fetch (and get empty dict again)
            second = fetch_team_batting_stats()
            assert first is not second


# ── identify_two_start_pitchers tests ────────────────────────────────


class TestIdentifyTwoStartPitchers:
    """Tests for identify_two_start_pitchers with mocked schedule."""

    def _make_schedule(self, entries):
        """Helper to build a mock schedule list."""
        return [
            {
                "game_date": e.get("date", "2026-07-01"),
                "home_name": e.get("home", "New York Yankees"),
                "away_name": e.get("away", "Boston Red Sox"),
                "home_probable_pitcher": e.get("home_pitcher", ""),
                "away_probable_pitcher": e.get("away_pitcher", ""),
            }
            for e in entries
        ]

    @patch("src.two_start.fetch_team_batting_stats", return_value={})
    @patch("src.two_start.get_weekly_schedule")
    def test_empty_schedule_returns_empty(self, mock_sched, mock_batting):
        """Empty schedule returns empty list."""
        mock_sched.return_value = []
        result = identify_two_start_pitchers()
        assert result == []

    @patch("src.two_start.fetch_team_batting_stats", return_value={})
    @patch("src.two_start.get_weekly_schedule")
    def test_no_two_start_pitchers(self, mock_sched, mock_batting):
        """All pitchers with only 1 start are excluded."""
        mock_sched.return_value = self._make_schedule(
            [
                {"home_pitcher": "Ace Smith", "away_pitcher": "Joe Starter", "date": "2026-07-01"},
                {"home_pitcher": "Bob Righty", "away_pitcher": "Carl Lefty", "date": "2026-07-02"},
            ]
        )
        result = identify_two_start_pitchers()
        assert result == []

    @patch("src.two_start.fetch_team_batting_stats", return_value={})
    @patch("src.two_start.get_weekly_schedule")
    def test_finds_two_start_pitcher(self, mock_sched, mock_batting):
        """Pitcher appearing twice is identified."""
        mock_sched.return_value = self._make_schedule(
            [
                {"home_pitcher": "Ace Smith", "away_pitcher": "Joe Starter", "date": "2026-07-01"},
                {"home_pitcher": "Ace Smith", "away_pitcher": "Bob Righty", "date": "2026-07-05"},
            ]
        )
        result = identify_two_start_pitchers()
        assert len(result) == 1
        assert result[0]["pitcher_name"] == "Ace Smith"
        assert result[0]["num_starts"] == 2
        assert result[0]["team"] == "NYY"

    @patch("src.two_start.fetch_team_batting_stats", return_value={})
    @patch("src.two_start.get_weekly_schedule")
    def test_three_start_pitcher(self, mock_sched, mock_batting):
        """Pitcher appearing three times has num_starts=3."""
        mock_sched.return_value = self._make_schedule(
            [
                {"home_pitcher": "Iron Man", "date": "2026-07-01"},
                {"home_pitcher": "Iron Man", "date": "2026-07-03"},
                {"home_pitcher": "Iron Man", "date": "2026-07-06"},
            ]
        )
        result = identify_two_start_pitchers()
        assert len(result) == 1
        assert result[0]["num_starts"] == 3

    @patch("src.two_start.fetch_team_batting_stats", return_value={})
    @patch("src.two_start.get_weekly_schedule")
    def test_start_details_include_opponent(self, mock_sched, mock_batting):
        """Each start entry includes opponent abbreviation."""
        mock_sched.return_value = self._make_schedule(
            [
                {"home_pitcher": "Ace Smith", "away": "Boston Red Sox", "date": "2026-07-01"},
                {"home_pitcher": "Ace Smith", "away": "Tampa Bay Rays", "date": "2026-07-05"},
            ]
        )
        result = identify_two_start_pitchers()
        assert len(result) == 1
        opps = [s["opponent"] for s in result[0]["starts"]]
        assert "BOS" in opps
        assert "TB" in opps

    @patch("src.two_start.fetch_team_batting_stats", return_value={})
    @patch("src.two_start.get_weekly_schedule")
    def test_away_pitcher_tracked(self, mock_sched, mock_batting):
        """Away probable pitchers are tracked correctly."""
        mock_sched.return_value = self._make_schedule(
            [
                {"away_pitcher": "Road Warrior", "home": "Colorado Rockies", "date": "2026-07-01"},
                {"away_pitcher": "Road Warrior", "home": "San Diego Padres", "date": "2026-07-04"},
            ]
        )
        result = identify_two_start_pitchers()
        assert len(result) == 1
        assert result[0]["pitcher_name"] == "Road Warrior"
        # Away pitcher's team is the "away" team in schedule (Boston Red Sox default)
        assert result[0]["team"] == "BOS"

    @patch("src.two_start.fetch_team_batting_stats", return_value={})
    @patch("src.two_start.get_weekly_schedule")
    def test_result_has_rate_damage(self, mock_sched, mock_batting):
        """Result includes rate_damage dict."""
        mock_sched.return_value = self._make_schedule(
            [
                {"home_pitcher": "Ace Smith", "date": "2026-07-01"},
                {"home_pitcher": "Ace Smith", "date": "2026-07-05"},
            ]
        )
        result = identify_two_start_pitchers()
        assert "rate_damage_per_start" in result[0]
        assert "era_change" in result[0]["rate_damage_per_start"]
        assert "whip_change" in result[0]["rate_damage_per_start"]
        assert "rate_damage_weekly" in result[0]
        assert "era_change" in result[0]["rate_damage_weekly"]

    @patch("src.two_start.fetch_team_batting_stats", return_value={})
    @patch("src.two_start.get_weekly_schedule")
    def test_result_has_streaming_value(self, mock_sched, mock_batting):
        """Result includes streaming_value and two_start_value."""
        mock_sched.return_value = self._make_schedule(
            [
                {"home_pitcher": "Ace Smith", "date": "2026-07-01"},
                {"home_pitcher": "Ace Smith", "date": "2026-07-05"},
            ]
        )
        result = identify_two_start_pitchers()
        assert "streaming_value" in result[0]
        assert "net_value" in result[0]["streaming_value"]
        assert "two_start_value" in result[0]
        assert "recommendation" in result[0]["two_start_value"]

    @patch("src.two_start.fetch_team_batting_stats", return_value={})
    @patch("src.two_start.get_weekly_schedule")
    def test_sorted_by_matchup_score(self, mock_sched, mock_batting):
        """Results are sorted by avg_matchup_score descending."""
        # Two pitchers: one at pitcher-friendly park, one at Coors
        mock_sched.return_value = [
            {
                "game_date": "2026-07-01",
                "home_name": "Miami Marlins",
                "away_name": "New York Mets",
                "home_probable_pitcher": "Good Park",
                "away_probable_pitcher": "",
            },
            {
                "game_date": "2026-07-04",
                "home_name": "Miami Marlins",
                "away_name": "New York Mets",
                "home_probable_pitcher": "Good Park",
                "away_probable_pitcher": "",
            },
            {
                "game_date": "2026-07-01",
                "home_name": "Colorado Rockies",
                "away_name": "Atlanta Braves",
                "home_probable_pitcher": "Bad Park",
                "away_probable_pitcher": "",
            },
            {
                "game_date": "2026-07-04",
                "home_name": "Colorado Rockies",
                "away_name": "Atlanta Braves",
                "home_probable_pitcher": "Bad Park",
                "away_probable_pitcher": "",
            },
        ]
        result = identify_two_start_pitchers()
        assert len(result) == 2
        # Miami (0.88 PF) pitcher should score higher than Coors (1.38 PF)
        assert result[0]["pitcher_name"] == "Good Park"
        assert result[0]["avg_matchup_score"] >= result[1]["avg_matchup_score"]

    @patch("src.two_start.fetch_team_batting_stats", return_value={})
    @patch("src.two_start.get_weekly_schedule")
    def test_starts_include_confidence(self, mock_sched, mock_batting):
        """Each start includes a confidence tier string."""
        mock_sched.return_value = self._make_schedule(
            [
                {"home_pitcher": "Ace Smith", "date": "2026-07-01"},
                {"home_pitcher": "Ace Smith", "date": "2026-07-05"},
            ]
        )
        result = identify_two_start_pitchers()
        for start in result[0]["starts"]:
            assert start["confidence"] in ("HIGH", "MEDIUM", "LOW")

    @patch("src.two_start.fetch_team_batting_stats", return_value={})
    @patch("src.two_start.get_weekly_schedule")
    def test_starts_include_park_factor(self, mock_sched, mock_batting):
        """Each start includes park_factor from PARK_FACTORS."""
        mock_sched.return_value = [
            {
                "game_date": "2026-07-01",
                "home_name": "Colorado Rockies",
                "away_name": "Atlanta Braves",
                "home_probable_pitcher": "Coors Pitcher",
                "away_probable_pitcher": "",
            },
            {
                "game_date": "2026-07-04",
                "home_name": "Colorado Rockies",
                "away_name": "Atlanta Braves",
                "home_probable_pitcher": "Coors Pitcher",
                "away_probable_pitcher": "",
            },
        ]
        result = identify_two_start_pitchers()
        assert len(result) == 1
        for start in result[0]["starts"]:
            assert start["park_factor"] == pytest.approx(1.38)

    def test_import_failure_returns_empty(self):
        """When required modules are missing, returns empty list."""
        with patch.dict("sys.modules", {"src.data_bootstrap": None}):
            # Force reimport
            import importlib

            import src.two_start

            importlib.reload(src.two_start)
            result = src.two_start.identify_two_start_pitchers()
            assert result == []
            # Reload back to normal
            importlib.reload(src.two_start)

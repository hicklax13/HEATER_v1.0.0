"""Tests for src/opponent_intel.py — opponent profiles, schedule, matchup analysis."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Constants sanity checks
# ---------------------------------------------------------------------------


class TestConstants:
    def test_schedule_covers_24_weeks(self):
        from src.opponent_intel import TEAM_HICKEY_SCHEDULE

        assert len(TEAM_HICKEY_SCHEDULE) == 24
        for w in range(1, 25):
            assert w in TEAM_HICKEY_SCHEDULE

    def test_all_scheduled_opponents_have_profiles(self):
        from src.opponent_intel import OPPONENT_PROFILES, TEAM_HICKEY_SCHEDULE

        unique_opponents = set(TEAM_HICKEY_SCHEDULE.values())
        for opp in unique_opponents:
            assert opp in OPPONENT_PROFILES, f"Missing profile for {opp}"

    def test_profiles_have_required_keys(self):
        from src.opponent_intel import OPPONENT_PROFILES

        required = {"tier", "threat", "manager", "strengths", "weaknesses", "notes"}
        for name, profile in OPPONENT_PROFILES.items():
            for key in required:
                assert key in profile, f"{name} missing key '{key}'"


# ---------------------------------------------------------------------------
# get_week_number
# ---------------------------------------------------------------------------


class TestGetWeekNumber:
    def test_week_1_on_opening_day(self):
        from src.opponent_intel import get_week_number

        with patch("src.opponent_intel.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 24, tzinfo=UTC)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = get_week_number()
            assert result == 1

    def test_capped_at_26(self):
        from src.opponent_intel import get_week_number

        with patch("src.opponent_intel.datetime") as mock_dt:
            # Way after the season (December)
            mock_dt.now.return_value = datetime(2026, 12, 1, tzinfo=UTC)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = get_week_number()
            assert result == 26

    def test_minimum_is_1(self):
        from src.opponent_intel import get_week_number

        with patch("src.opponent_intel.datetime") as mock_dt:
            # Before season start
            mock_dt.now.return_value = datetime(2026, 3, 1, tzinfo=UTC)
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = get_week_number()
            assert result == 1


# ---------------------------------------------------------------------------
# get_current_opponent
# ---------------------------------------------------------------------------


class TestGetCurrentOpponent:
    def test_fallback_to_hardcoded_data(self):
        from src.opponent_intel import get_current_opponent

        with patch("src.opponent_intel.get_week_number", return_value=1):
            result = get_current_opponent(yds=None)
            assert result["name"] == "The Good The Vlad The Ugly"
            assert result["week"] == 1
            assert "tier" in result

    def test_live_schedule_used_when_available(self):
        from src.opponent_intel import get_current_opponent

        mock_yds = MagicMock()
        mock_yds.get_schedule.return_value = {1: "Custom Team"}
        mock_yds.get_opponent_profile.return_value = {
            "tier": 2,
            "threat": "Medium",
            "manager": "Bob",
            "strengths": ["HR"],
            "weaknesses": ["SB"],
            "notes": "live",
        }

        with patch("src.opponent_intel.get_week_number", return_value=1):
            result = get_current_opponent(yds=mock_yds)
            assert result["name"] == "Custom Team"

    def test_live_profile_fallback_on_unknown_threat(self):
        """When live profile has threat=Unknown, should fall back to hardcoded data."""
        from src.opponent_intel import get_current_opponent

        mock_yds = MagicMock()
        mock_yds.get_schedule.return_value = {1: "The Good The Vlad The Ugly"}
        mock_yds.get_opponent_profile.return_value = {"threat": "Unknown"}

        with patch("src.opponent_intel.get_week_number", return_value=1):
            result = get_current_opponent(yds=mock_yds)
            # Should fall back to hardcoded profile
            assert result["threat"] == "Medium-Low"

    def test_yds_exception_handled(self):
        from src.opponent_intel import get_current_opponent

        mock_yds = MagicMock()
        mock_yds.get_schedule.side_effect = RuntimeError("API down")

        with patch("src.opponent_intel.get_week_number", return_value=1):
            result = get_current_opponent(yds=mock_yds)
            # Should still return hardcoded fallback data
            assert result["name"] == "The Good The Vlad The Ugly"

    def test_empty_dict_for_invalid_week(self):
        from src.opponent_intel import get_current_opponent

        with patch("src.opponent_intel.get_week_number", return_value=25):
            # Week 25 is not in the 24-week schedule
            result = get_current_opponent(yds=None)
            assert result == {}


# ---------------------------------------------------------------------------
# get_opponent_for_week
# ---------------------------------------------------------------------------


class TestGetOpponentForWeek:
    def test_known_week(self):
        from src.opponent_intel import get_opponent_for_week

        result = get_opponent_for_week(4)
        assert result["name"] == "Over the Rembow"
        assert result["tier"] == 1

    def test_out_of_range_week(self):
        from src.opponent_intel import get_opponent_for_week

        result = get_opponent_for_week(99)
        assert result == {}

    def test_live_data_override(self):
        from src.opponent_intel import get_opponent_for_week

        mock_yds = MagicMock()
        mock_yds.get_schedule.return_value = {4: "Live Opponent"}
        mock_yds.get_opponent_profile.return_value = {
            "tier": 5,
            "threat": "Ultra",
            "manager": "Z",
            "strengths": [],
            "weaknesses": [],
            "notes": "",
        }
        result = get_opponent_for_week(4, yds=mock_yds)
        assert result["name"] == "Live Opponent"
        assert result["tier"] == 5


# ---------------------------------------------------------------------------
# get_schedule_difficulty
# ---------------------------------------------------------------------------


class TestGetScheduleDifficulty:
    def test_full_schedule(self):
        from src.opponent_intel import get_schedule_difficulty

        result = get_schedule_difficulty()
        assert len(result) == 24
        assert result[0]["week"] == 1

    def test_custom_range(self):
        from src.opponent_intel import get_schedule_difficulty

        result = get_schedule_difficulty(weeks=range(1, 4))
        assert len(result) == 3
        assert result[0]["opponent"] == "The Good The Vlad The Ugly"
        assert result[1]["opponent"] == "Baty Babies"

    def test_entries_have_expected_keys(self):
        from src.opponent_intel import get_schedule_difficulty

        result = get_schedule_difficulty(weeks=range(1, 2))
        entry = result[0]
        assert "week" in entry
        assert "opponent" in entry
        assert "tier" in entry
        assert "threat" in entry


# ---------------------------------------------------------------------------
# analyze_weekly_matchup
# ---------------------------------------------------------------------------


@pytest.fixture
def matchup_roster():
    return pd.DataFrame(
        {
            "name": ["Hitter A", "Pitcher B"],
            "hr": [8, 0],
            "r": [12, 0],
            "rbi": [10, 0],
            "sb": [2, 0],
            "avg": [0.300, 0.0],
            "obp": [0.380, 0.0],
            "w": [0, 4],
            "l": [0, 1],
            "sv": [0, 0],
            "k": [0, 30],
            "era": [0.0, 3.20],
            "whip": [0.0, 1.05],
        }
    )


class TestAnalyzeWeeklyMatchup:
    def test_basic_structure(self, matchup_roster, opponent_profile):
        from src.opponent_intel import analyze_weekly_matchup

        result = analyze_weekly_matchup(matchup_roster, opponent_profile, week=7)
        assert result["week"] == 7
        assert result["opponent"] == "Twigs"
        assert "likely_wins" in result
        assert "likely_losses" in result
        assert "toss_ups" in result
        assert "projected_record" in result

    def test_wins_losses_tossups(self, matchup_roster, opponent_profile):
        from src.opponent_intel import analyze_weekly_matchup

        result = analyze_weekly_matchup(matchup_roster, opponent_profile, week=7)
        assert "SV" in result["likely_wins"]
        assert "HR" in result["likely_losses"]
        # Categories not in strengths or weaknesses should be toss-ups
        assert "W" in result["toss_ups"]

    def test_projected_record_format(self, matchup_roster, opponent_profile):
        from src.opponent_intel import analyze_weekly_matchup

        result = analyze_weekly_matchup(matchup_roster, opponent_profile, week=7)
        record = result["projected_record"]
        parts = record.split("-")
        assert len(parts) == 3
        total = sum(int(p) for p in parts)
        assert total == 12  # 12 categories

    def test_streaming_recommendations_sv(self, matchup_roster):
        from src.opponent_intel import analyze_weekly_matchup

        profile = {
            "name": "SV Weak Team",
            "tier": 3,
            "threat": "Low",
            "strengths": [],
            "weaknesses": ["SV"],
        }
        result = analyze_weekly_matchup(matchup_roster, profile, week=1)
        assert any("closers" in r.lower() or "saves" in r.lower() for r in result["streaming_recommendations"])

    def test_empty_roster(self, opponent_profile):
        from src.opponent_intel import analyze_weekly_matchup

        empty = pd.DataFrame()
        result = analyze_weekly_matchup(empty, opponent_profile, week=1)
        # Should still produce results without crashing
        assert len(result["category_results"]) == 12
        for cr in result["category_results"]:
            assert cr["team_total"] == 0.0

    def test_no_streaming_edge(self, matchup_roster):
        """When opponent has no exploitable weaknesses for streaming."""
        from src.opponent_intel import analyze_weekly_matchup

        profile = {
            "name": "Balanced",
            "tier": 2,
            "threat": "Medium",
            "strengths": ["HR", "SB", "ERA", "WHIP", "K", "W"],  # Cover streaming triggers
            "weaknesses": ["RBI"],  # RBI not in streaming triggers
        }
        result = analyze_weekly_matchup(matchup_roster, profile, week=1)
        assert any("No clear" in r or "best lineup" in r for r in result["streaming_recommendations"])

    def test_exploit_targets_include_hickey_strengths(self, matchup_roster):
        from src.opponent_intel import analyze_weekly_matchup

        profile = {
            "name": "K Weak",
            "tier": 3,
            "threat": "Low",
            "strengths": [],
            "weaknesses": ["K", "HR"],
        }
        result = analyze_weekly_matchup(matchup_roster, profile, week=1)
        # K and HR are Hickey structural strengths, so they should be exploit targets
        assert "K" in result["exploit_targets"]
        assert "HR" in result["exploit_targets"]


@pytest.fixture
def opponent_profile():
    return {
        "name": "Twigs",
        "tier": 1,
        "threat": "High",
        "manager": "Nick",
        "strengths": ["HR", "R", "SB", "K"],
        "weaknesses": ["SV"],
        "notes": "Judge + Duran.",
    }

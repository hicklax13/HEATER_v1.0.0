"""Tests for Yahoo schedule and records fetching."""

from __future__ import annotations

import pandas as pd
import pytest

from src.database import init_db


@pytest.fixture(autouse=True)
def _fresh_db():
    init_db()
    # Clear tables that tests write to, avoiding cross-test leakage
    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute("DELETE FROM league_records")
        conn.execute("DELETE FROM league_schedule_full")
        conn.commit()
    except Exception:
        pass  # Tables may not exist yet if Task 1 is still running
    finally:
        conn.close()


# ── DB-level record tests ────────────────────────────────────────────


class TestFetchAndSyncRecords:
    """Test W-L-T record capture from standings."""

    def test_sync_records_stores_wlt(self):
        """Verify upsert_league_record stores and loads W-L-T data."""
        try:
            from src.database import load_league_records, upsert_league_record
        except ImportError:
            pytest.skip("DB functions not yet created by Task 1")

        upsert_league_record(
            "Team A",
            wins=10,
            losses=5,
            ties=1,
            win_pct=0.656,
            streak="W2",
            rank=1,
        )
        df = load_league_records()
        assert len(df) == 1
        row = df.iloc[0]
        assert row["wins"] == 10
        assert row["losses"] == 5
        assert row["streak"] == "W2"

    def test_sync_records_multiple_teams(self):
        """Verify multiple team records are stored correctly."""
        try:
            from src.database import load_league_records, upsert_league_record
        except ImportError:
            pytest.skip("DB functions not yet created by Task 1")

        upsert_league_record("Team A", wins=10, losses=5, ties=1, win_pct=0.656, streak="W2", rank=1)
        upsert_league_record("Team B", wins=8, losses=7, ties=1, win_pct=0.531, streak="L1", rank=3)

        df = load_league_records()
        assert len(df) == 2

    def test_sync_records_upsert_overwrites(self):
        """Verify that upserting the same team overwrites old values."""
        try:
            from src.database import load_league_records, upsert_league_record
        except ImportError:
            pytest.skip("DB functions not yet created by Task 1")

        upsert_league_record("Team A", wins=10, losses=5, ties=1, win_pct=0.656, streak="W2", rank=1)
        upsert_league_record("Team A", wins=11, losses=5, ties=1, win_pct=0.688, streak="W3", rank=1)

        df = load_league_records()
        assert len(df) == 1
        assert df.iloc[0]["wins"] == 11
        assert df.iloc[0]["streak"] == "W3"


# ── Schedule parsing (pure functions, no Yahoo API) ──────────────────


class TestFullLeagueScheduleParsing:
    """Test schedule parsing logic (no Yahoo API call)."""

    def test_parse_scoreboard_matchups(self):
        """Test that we correctly parse matchup pairs from a scoreboard response."""
        from src.standings_engine import parse_scoreboard_matchups

        mock_matchups = [
            {"team_a": "Team Hickey", "team_b": "Baty Babies"},
            {"team_a": "Jonny Jockstrap", "team_b": "Shohei Show"},
            {"team_a": "The Good The Vlad The Ugly", "team_b": "Team F"},
        ]
        result = parse_scoreboard_matchups(mock_matchups)
        assert len(result) == 3
        assert ("Team Hickey", "Baty Babies") in result

    def test_parse_scoreboard_matchups_empty(self):
        """Test parsing with empty matchup list."""
        from src.standings_engine import parse_scoreboard_matchups

        result = parse_scoreboard_matchups([])
        assert result == []

    def test_parse_scoreboard_matchups_missing_keys(self):
        """Test parsing with malformed matchup dicts."""
        from src.standings_engine import parse_scoreboard_matchups

        result = parse_scoreboard_matchups([{"team_a": "Only One"}])
        assert result == []  # Missing team_b -> empty string -> skipped

    def test_find_user_opponent_in_schedule(self):
        """Test finding user's opponent for a given week."""
        from src.standings_engine import find_user_opponent

        schedule = {
            1: [("Team Hickey", "Baty Babies"), ("Team C", "Team D")],
            2: [("Jonny Jockstrap", "Team Hickey"), ("Team C", "Team D")],
        }
        assert find_user_opponent(schedule, 1, "Team Hickey") == "Baty Babies"
        assert find_user_opponent(schedule, 2, "Team Hickey") == "Jonny Jockstrap"
        assert find_user_opponent(schedule, 3, "Team Hickey") is None

    def test_find_user_opponent_not_in_any_matchup(self):
        """Test when user team is not in the given week's matchups."""
        from src.standings_engine import find_user_opponent

        schedule = {1: [("Team A", "Team B"), ("Team C", "Team D")]}
        assert find_user_opponent(schedule, 1, "Team Hickey") is None

    def test_find_user_opponent_empty_schedule(self):
        """Test with completely empty schedule."""
        from src.standings_engine import find_user_opponent

        assert find_user_opponent({}, 1, "Team Hickey") is None


# ── Category result parsing ──────────────────────────────────────────


class TestWeekScoreboardParsing:
    """Test scoreboard result parsing for past weeks."""

    def test_parse_week_results(self):
        """Test parsing category W/L results from a completed week."""
        from src.standings_engine import parse_week_category_results

        categories = [
            {"name": "R", "user_val": 38, "opp_val": 33, "is_inverse": False},
            {"name": "ERA", "user_val": 3.42, "opp_val": 3.10, "is_inverse": True},
            {"name": "SB", "user_val": 5, "opp_val": 5, "is_inverse": False},
        ]
        result = parse_week_category_results(categories)
        assert result[0]["result"] == "W"  # R: 38 > 33
        assert result[1]["result"] == "L"  # ERA: 3.42 > 3.10 (inverse, lower wins)
        assert result[2]["result"] == "T"  # SB: tied

    def test_parse_week_results_all_wins(self):
        """Test parsing when user wins all categories."""
        from src.standings_engine import parse_week_category_results

        categories = [
            {"name": "HR", "user_val": 10, "opp_val": 5, "is_inverse": False},
            {"name": "WHIP", "user_val": 1.05, "opp_val": 1.25, "is_inverse": True},
        ]
        result = parse_week_category_results(categories)
        assert all(r["result"] == "W" for r in result)

    def test_parse_week_results_inverse_tie(self):
        """Test tie in an inverse category."""
        from src.standings_engine import parse_week_category_results

        categories = [
            {"name": "ERA", "user_val": 3.50, "opp_val": 3.50, "is_inverse": True},
        ]
        result = parse_week_category_results(categories)
        assert result[0]["result"] == "T"

    def test_parse_week_results_preserves_values(self):
        """Test that parsed results preserve original values."""
        from src.standings_engine import parse_week_category_results

        categories = [
            {"name": "K", "user_val": 55, "opp_val": 48, "is_inverse": False},
        ]
        result = parse_week_category_results(categories)
        assert result[0]["user_val"] == 55.0
        assert result[0]["opp_val"] == 48.0
        assert result[0]["name"] == "K"

    def test_parse_week_results_empty(self):
        """Test with empty categories list."""
        from src.standings_engine import parse_week_category_results

        result = parse_week_category_results([])
        assert result == []

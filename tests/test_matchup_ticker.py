"""Tests for matchup ticker — Yahoo API raw stats + UI rendering."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.yahoo_api import YahooFantasyClient

# ---------------------------------------------------------------------------
# _get_team_week_stats_raw
# ---------------------------------------------------------------------------


class TestGetTeamWeekStatsRaw:
    """Tests for the raw Yahoo REST API stats fetcher."""

    def _make_client(self):
        """Create a client with a mocked yfpy query."""
        client = YahooFantasyClient.__new__(YahooFantasyClient)
        client.league_id = "109662"
        client.game_code = "mlb"
        client.season = 2026
        mock_query = MagicMock()
        mock_query._yahoo_access_token_dict = {"access_token": "test_token_abc"}
        mock_query.game_id = 469
        mock_query.league_key = "469.l.109662"
        client._query = mock_query
        return client

    def test_parses_stats_correctly(self):
        """Parses Yahoo JSON response into category dict."""
        fake_json = {
            "fantasy_content": {
                "team": [
                    [{"team_key": "469.l.109662.t.9"}],
                    {
                        "team_stats": {
                            "stats": [
                                {"stat": {"stat_id": "7", "value": "19"}},
                                {"stat": {"stat_id": "12", "value": "6"}},
                                {"stat": {"stat_id": "13", "value": "14"}},
                                {"stat": {"stat_id": "16", "value": "3"}},
                                {"stat": {"stat_id": "3", "value": ".243"}},
                                {"stat": {"stat_id": "4", "value": ".364"}},
                                {"stat": {"stat_id": "28", "value": "2"}},
                                {"stat": {"stat_id": "29", "value": "0"}},
                                {"stat": {"stat_id": "32", "value": "0"}},
                                {"stat": {"stat_id": "42", "value": "30"}},
                                {"stat": {"stat_id": "26", "value": "1.38"}},
                                {"stat": {"stat_id": "27", "value": "1.15"}},
                            ]
                        },
                        "team_points": {"total": "4", "coverage_type": "week", "week": "1"},
                    },
                ]
            }
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake_json

        client = self._make_client()
        with patch("src.yahoo_api._requests.get", return_value=mock_resp):
            stats, points = client._get_team_week_stats_raw("469.l.109662.t.9", 1)

        assert stats["R"] == "19"
        assert stats["HR"] == "6"
        assert stats["AVG"] == ".243"
        assert stats["ERA"] == "1.38"
        assert stats["WHIP"] == "1.15"
        assert stats["K"] == "30"
        assert points == 4.0

    def test_returns_empty_on_http_error(self):
        """Returns empty stats on non-200 response."""
        mock_resp = MagicMock()
        mock_resp.status_code = 401

        client = self._make_client()
        with patch("src.yahoo_api._requests.get", return_value=mock_resp):
            stats, points = client._get_team_week_stats_raw("469.l.109662.t.9", 1)

        assert stats == {}
        assert points == 0.0

    def test_returns_empty_on_no_token(self):
        """Returns empty stats when no bearer token is available."""
        client = self._make_client()
        client._query._yahoo_access_token_dict = {}

        with patch("src.yahoo_api._AUTH_DIR", Path("/nonexistent")):
            stats, points = client._get_team_week_stats_raw("469.l.109662.t.9", 1)

        assert stats == {}
        assert points == 0.0

    def test_ignores_unknown_stat_ids(self):
        """Only maps known stat IDs, ignores display-only stats like H/AB (60)."""
        fake_json = {
            "fantasy_content": {
                "team": [
                    {
                        "team_stats": {
                            "stats": [
                                {"stat": {"stat_id": "60", "value": "33/136"}},
                                {"stat": {"stat_id": "50", "value": "26.0"}},
                                {"stat": {"stat_id": "7", "value": "19"}},
                                {"stat": {"stat_id": "999", "value": "unknown"}},
                            ]
                        },
                        "team_points": {"total": "2"},
                    },
                ]
            }
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake_json

        client = self._make_client()
        with patch("src.yahoo_api._requests.get", return_value=mock_resp):
            stats, points = client._get_team_week_stats_raw("469.l.109662.t.9", 1)

        assert "R" in stats
        assert "H/AB" not in stats
        assert "IP" not in stats
        assert "stat_999" not in stats
        assert points == 2.0


# ---------------------------------------------------------------------------
# get_current_matchup — win/loss/tie logic
# ---------------------------------------------------------------------------


class TestMatchupScoring:
    """Tests for category win/loss/tie computation."""

    def test_inverse_cat_lower_wins(self):
        """ERA/WHIP/L: lower value wins."""
        client = YahooFantasyClient.__new__(YahooFantasyClient)
        inverse = client._INVERSE_CATS
        assert "ERA" in inverse
        assert "WHIP" in inverse
        assert "L" in inverse
        assert "R" not in inverse
        assert "HR" not in inverse

    def test_all_cats_count(self):
        """12 scoring categories defined."""
        client = YahooFantasyClient.__new__(YahooFantasyClient)
        assert len(client._ALL_CATS) == 12

    def test_stat_id_map_covers_all_cats(self):
        """Every category in _ALL_CATS has a stat_id mapping."""
        client = YahooFantasyClient.__new__(YahooFantasyClient)
        mapped_cats = set(client._STAT_ID_MAP.values())
        for cat in client._ALL_CATS:
            assert cat in mapped_cats, f"{cat} missing from _STAT_ID_MAP"


class TestGetCurrentMatchup:
    """Tests for the full get_current_matchup() flow."""

    def _make_client_with_mocks(self, user_stats, opp_stats, status="midevent"):
        """Build a fully mocked client for matchup testing."""
        client = YahooFantasyClient.__new__(YahooFantasyClient)
        client.league_id = "109662"
        client.game_code = "mlb"
        client.season = 2026

        mock_query = MagicMock()
        mock_query._yahoo_access_token_dict = {"access_token": "tok"}
        mock_query.game_id = 469
        mock_query.league_key = "469.l.109662"

        # Mock game weeks — use a wide range so "today" always falls inside
        mock_week = MagicMock()
        mock_week.start = "2020-01-01"
        mock_week.end = "2099-12-31"
        mock_week.week = 1
        mock_query.get_game_weeks_by_game_id.return_value = [mock_week]

        # Mock scoreboard with one matchup
        mock_user_team = MagicMock()
        mock_user_team.team_key = "469.l.109662.t.9"
        mock_user_team.name = "Team Hickey"
        mock_opp_team = MagicMock()
        mock_opp_team.team_key = "469.l.109662.t.13"
        mock_opp_team.name = "Opponent"

        mock_team_wrapper_1 = MagicMock()
        mock_team_wrapper_1.team = mock_user_team
        mock_team_wrapper_2 = MagicMock()
        mock_team_wrapper_2.team = mock_opp_team

        mock_matchup = MagicMock()
        mock_matchup.teams = [mock_team_wrapper_1, mock_team_wrapper_2]
        mock_matchup.status = status

        mock_scoreboard = MagicMock()
        mock_scoreboard.matchups = [mock_matchup]
        mock_query.get_league_scoreboard_by_week.return_value = mock_scoreboard

        # Mock league teams for _get_user_team_key
        mock_user_entry = MagicMock()
        mock_user_entry.team = MagicMock()
        mock_user_entry.team.team_key = "469.l.109662.t.9"
        mock_user_entry.team.is_owned_by_current_login = True
        mock_query.get_league_teams.return_value = [mock_user_entry]

        client._query = mock_query

        # Mock _get_team_week_stats_raw
        def mock_raw_stats(team_key, week):
            if "t.9" in team_key:
                return user_stats, 0.0
            return opp_stats, 0.0

        client._get_team_week_stats_raw = mock_raw_stats
        return client

    def test_computes_wins_losses_ties(self):
        """Correctly counts W/L/T across all 12 categories."""
        user = {"R": "19", "HR": "6", "RBI": "14", "SB": "3", "AVG": ".243", "OBP": ".364",
                "W": "2", "L": "0", "SV": "0", "K": "30", "ERA": "1.38", "WHIP": "1.15"}
        opp = {"R": "21", "HR": "3", "RBI": "16", "SB": "3", "AVG": ".257", "OBP": ".383",
               "W": "2", "L": "2", "SV": "0", "K": "37", "ERA": "3.30", "WHIP": "1.35"}

        client = self._make_client_with_mocks(user, opp)
        result = client.get_current_matchup()

        assert result is not None
        assert result["wins"] == 4   # HR, L(inv), ERA(inv), WHIP(inv)
        assert result["losses"] == 5  # R, RBI, AVG, OBP, K
        assert result["ties"] == 3    # SB, W, SV
        assert result["user_name"] == "Team Hickey"
        assert result["opp_name"] == "Opponent"
        assert result["week"] == 1
        assert len(result["categories"]) == 12

    def test_preevent_all_dashes(self):
        """Pre-event matchup returns 0-0-0 with dash values."""
        user = {}
        opp = {}
        client = self._make_client_with_mocks(user, opp, status="preevent")
        result = client.get_current_matchup()

        assert result is not None
        assert result["wins"] == 0
        assert result["losses"] == 0
        assert result["ties"] == 0
        assert result["status"] == "preevent"

    def test_returns_none_when_not_authenticated(self):
        """Returns None when client has no query object."""
        client = YahooFantasyClient.__new__(YahooFantasyClient)
        client._query = None
        assert client.get_current_matchup() is None


# ---------------------------------------------------------------------------
# render_matchup_ticker (unit tests for data logic, not Streamlit rendering)
# ---------------------------------------------------------------------------


class TestMatchupTickerHelpers:
    """Tests for ticker display logic."""

    def test_score_color_winning(self):
        """Winning score should use green color."""
        from src.ui_shared import THEME

        # Wins > losses = green
        assert THEME["green"] == "#2d6a4f"

    def test_score_color_losing(self):
        """Losing score should use primary/red color."""
        from src.ui_shared import THEME

        assert THEME["primary"] == "#e63946"

    def test_all_categories_in_stat_map(self):
        """Verify stat ID map produces all 12 expected categories."""
        expected = {"R", "HR", "RBI", "SB", "AVG", "OBP", "W", "L", "SV", "K", "ERA", "WHIP"}
        actual = set(YahooFantasyClient._STAT_ID_MAP.values())
        assert actual == expected

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
        # Use real __init__ so instance attrs from LeagueConfig populate
        # (was previously class-body globals — moved to instance attrs in
        # Wave 8a/D2A-002).
        client = YahooFantasyClient(league_id="109662")
        inverse = client._inverse_cats
        assert "ERA" in inverse
        assert "WHIP" in inverse
        assert "L" in inverse
        assert "R" not in inverse
        assert "HR" not in inverse

    def test_all_cats_count(self):
        """12 scoring categories defined."""
        client = YahooFantasyClient(league_id="109662")
        assert len(client._all_cats) == 12

    def test_stat_id_map_covers_all_cats(self):
        """Every category in _all_cats has a stat_id mapping."""
        client = YahooFantasyClient(league_id="109662")
        mapped_cats = set(client._STAT_ID_MAP.values())
        for cat in client._all_cats:
            assert cat in mapped_cats, f"{cat} missing from _STAT_ID_MAP"


class TestGetCurrentMatchup:
    """Tests for the full get_current_matchup() flow."""

    def _make_client_with_mocks(self, user_stats, opp_stats, status="midevent"):
        """Build a fully mocked client for matchup testing."""
        client = YahooFantasyClient.__new__(YahooFantasyClient)
        client.league_id = "109662"
        client.game_code = "mlb"
        client.season = 2026
        # __init__ also populates these from LeagueConfig (Wave 8a/D2A-002).
        # Bypassed __init__ via __new__, so set them manually here.
        from src.valuation import LeagueConfig

        _lc = LeagueConfig()
        client._inverse_cats = set(_lc.inverse_stats)
        client._all_cats = list(_lc.all_categories)

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
        user = {
            "R": "19",
            "HR": "6",
            "RBI": "14",
            "SB": "3",
            "AVG": ".243",
            "OBP": ".364",
            "W": "2",
            "L": "0",
            "SV": "0",
            "K": "30",
            "ERA": "1.38",
            "WHIP": "1.15",
        }
        opp = {
            "R": "21",
            "HR": "3",
            "RBI": "16",
            "SB": "3",
            "AVG": ".257",
            "OBP": ".383",
            "W": "2",
            "L": "2",
            "SV": "0",
            "K": "37",
            "ERA": "3.30",
            "WHIP": "1.35",
        }

        client = self._make_client_with_mocks(user, opp)
        result = client.get_current_matchup()

        assert result is not None
        assert result["wins"] == 4  # HR, L(inv), ERA(inv), WHIP(inv)
        assert result["losses"] == 5  # R, RBI, AVG, OBP, K
        assert result["ties"] == 3  # SB, W, SV
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

    def test_get_all_team_matchups_covers_both_teams(self):
        """One scoreboard matchup yields a card for BOTH teams, keyed by name —
        the basis for per-team matchup population (all 12 members, not just the
        token owner)."""
        user = {
            "R": "19",
            "HR": "6",
            "RBI": "14",
            "SB": "3",
            "AVG": ".243",
            "OBP": ".364",
            "W": "2",
            "L": "0",
            "SV": "0",
            "K": "30",
            "ERA": "1.38",
            "WHIP": "1.15",
        }
        opp = {
            "R": "21",
            "HR": "3",
            "RBI": "16",
            "SB": "3",
            "AVG": ".257",
            "OBP": ".383",
            "W": "2",
            "L": "2",
            "SV": "0",
            "K": "37",
            "ERA": "3.30",
            "WHIP": "1.35",
        }
        client = self._make_client_with_mocks(user, opp)
        client._ensure_auth = lambda: True

        all_m = client.get_all_team_matchups()

        assert set(all_m.keys()) == {"Team Hickey", "Opponent"}
        assert all_m["Team Hickey"]["opp_name"] == "Opponent"
        assert all_m["Opponent"]["opp_name"] == "Team Hickey"
        # Symmetric perspectives: one team's wins are the other's losses.
        assert all_m["Team Hickey"]["wins"] == all_m["Opponent"]["losses"]
        assert all_m["Team Hickey"]["losses"] == all_m["Opponent"]["wins"]


# ---------------------------------------------------------------------------
# render_matchup_ticker (unit tests for data logic, not Streamlit rendering)
# ---------------------------------------------------------------------------


class TestMatchupTickerCachedData:
    """render_matchup_ticker renders from a cache-served dict without yahoo_connected."""

    _CACHE_MATCHUP = {
        "week": 7,
        "status": "midevent",
        "user_name": "Team Hickey",
        "opp_name": "Rival Squad",
        "wins": 5,
        "losses": 4,
        "ties": 3,
        "categories": [
            {"cat": "R", "you": 30, "opp": 25, "result": "WIN"},
            {"cat": "ERA", "you": 3.10, "opp": 4.20, "result": "WIN"},
        ],
    }

    def test_renders_team_names_from_cache_without_yahoo_connected(self):
        """Ticker renders using a pre-fetched matchup dict even when
        yahoo_connected is NOT set in session state — covers the read-only
        MULTI_USER member path where the scheduler writes the matchup to
        the SQLite cache and get_matchup() returns it."""
        from unittest.mock import MagicMock, patch

        import streamlit as st

        from src.ui_shared import render_matchup_ticker

        # Ensure yahoo_connected is absent from session state
        st.session_state.pop("yahoo_connected", None)

        rendered_html: list[str] = []

        def capture_markdown(html, **kwargs):
            rendered_html.append(html)

        with (
            patch.object(st, "markdown", side_effect=capture_markdown),
            patch.object(st, "expander", return_value=MagicMock(__enter__=lambda s: s, __exit__=lambda s, *a: None)),
        ):
            render_matchup_ticker(matchup_data=self._CACHE_MATCHUP)

        assert rendered_html, "render_matchup_ticker produced no output with cache-served matchup"
        combined = " ".join(rendered_html)
        assert "Rival Squad" in combined, "Opponent name missing from ticker HTML"
        assert "5-4-3" in combined, "Score string missing from ticker HTML"
        assert "Week 7" in combined, "Week number missing from ticker HTML"

    def test_renders_nothing_when_no_data_and_no_yahoo_connected(self):
        """When matchup_data=None and yahoo_connected is absent, no HTML is emitted."""
        from unittest.mock import MagicMock, patch

        import streamlit as st

        from src.ui_shared import render_matchup_ticker

        st.session_state.pop("yahoo_connected", None)
        # Ensure _fetch_matchup_data also returns nothing (no yahoo_client)
        st.session_state.pop("yahoo_client", None)
        st.session_state.pop("_matchup_ticker_data", None)

        rendered_html: list[str] = []

        def capture_markdown(html, **kwargs):
            rendered_html.append(html)

        # Also mock get_yahoo_data_service so the YDS fallback returns None.
        # The function does a local import, so patch the module it imports from.
        mock_yds = MagicMock()
        mock_yds.get_matchup.return_value = None

        with (
            patch.object(st, "markdown", side_effect=capture_markdown),
            patch("src.yahoo_data_service.YahooDataService.get_matchup", return_value=None),
            patch("src.yahoo_data_service.get_yahoo_data_service", return_value=mock_yds),
        ):
            render_matchup_ticker(matchup_data=None)

        ticker_html = [h for h in rendered_html if "matchup-ticker" in h]
        assert not ticker_html, "Ticker should not render when no matchup data is available"

    def test_live_session_unaffected_when_yahoo_connected(self):
        """When yahoo_connected IS set and data comes from yahoo_client, behavior unchanged."""
        from unittest.mock import MagicMock, patch

        import streamlit as st

        from src.ui_shared import render_matchup_ticker

        st.session_state["yahoo_connected"] = True

        rendered_html: list[str] = []

        def capture_markdown(html, **kwargs):
            rendered_html.append(html)

        with (
            patch.object(st, "markdown", side_effect=capture_markdown),
            patch.object(st, "expander", return_value=MagicMock(__enter__=lambda s: s, __exit__=lambda s, *a: None)),
        ):
            render_matchup_ticker(matchup_data=self._CACHE_MATCHUP)

        assert any("matchup-ticker" in h for h in rendered_html), "Ticker should render for live Yahoo session too"


class TestMatchupTickerHelpers:
    """Tests for ticker display logic."""

    def test_score_color_winning(self):
        """Winning score should use the Combustion green color (wins > losses)."""
        from src.ui_shared import THEME

        # render_matchup_ticker: w > lo -> score_color = THEME["green"]
        assert THEME["green"] == "#1f9d6b"

    def test_score_color_losing(self):
        """Losing score should use the primary (orange) color (losses > wins)."""
        from src.ui_shared import THEME

        # render_matchup_ticker: lo > w -> score_color = THEME["primary"]
        assert THEME["primary"] == "#ff6d00"

    def test_all_categories_in_stat_map(self):
        """Verify stat ID map produces all 12 expected categories."""
        expected = {"R", "HR", "RBI", "SB", "AVG", "OBP", "W", "L", "SV", "K", "ERA", "WHIP"}
        actual = set(YahooFantasyClient._STAT_ID_MAP.values())
        assert actual == expected

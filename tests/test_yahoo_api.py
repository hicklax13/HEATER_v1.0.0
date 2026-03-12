"""Tests for Yahoo Fantasy API client (yfpy-based rewrite)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.yahoo_api import (
    YFPY_AVAILABLE,
    YahooFantasyClient,
    build_oauth_url,
    exchange_code_for_token,
    validate_credentials,
)

# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def test_build_oauth_url():
    url = build_oauth_url("my_consumer_key_abc123")
    assert "client_id=my_consumer_key_abc123" in url
    assert "response_type=code" in url
    assert url.startswith("https://api.login.yahoo.com/oauth2/request_auth")


def test_build_oauth_url_custom_redirect():
    url = build_oauth_url("key123", redirect_uri="https://example.com/callback")
    assert "redirect_uri=https://example.com/callback" in url


def test_exchange_code_for_token_success():
    """Test successful token exchange with mocked HTTP response."""
    import json
    from unittest.mock import MagicMock

    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps(
        {"access_token": "tok123", "refresh_token": "ref456", "token_type": "bearer"}
    ).encode("utf-8")
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_response):
        result = exchange_code_for_token("key123", "secret456", "verifier789")
    assert result is not None
    assert result["access_token"] == "tok123"
    assert result["refresh_token"] == "ref456"


def test_exchange_code_for_token_failure():
    """Test token exchange returns None on network error."""
    with patch("urllib.request.urlopen", side_effect=Exception("Network error")):
        result = exchange_code_for_token("key123", "secret456", "bad_code")
    assert result is None


def test_validate_credentials_valid():
    assert validate_credentials("a" * 72, "b" * 32) is True


def test_validate_credentials_empty_key():
    assert validate_credentials("", "b" * 32) is False


def test_validate_credentials_empty_secret():
    assert validate_credentials("a" * 72, "") is False


def test_validate_credentials_none():
    assert validate_credentials(None, None) is False


def test_validate_credentials_too_short():
    assert validate_credentials("short", "tiny") is False


# ---------------------------------------------------------------------------
# Client initialization
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """Pre-configured YahooFantasyClient (not authenticated)."""
    return YahooFantasyClient(league_id="12345", game_code="mlb", season=2026)


def test_client_init(client):
    assert client.league_id == "12345"
    assert client.game_code == "mlb"
    assert client.season == 2026


def test_client_not_authenticated_initially(client):
    assert client.is_authenticated is False
    assert client._query is None


# ---------------------------------------------------------------------------
# Unauthenticated client returns safe defaults
# ---------------------------------------------------------------------------


def test_get_league_settings_unauthenticated(client):
    result = client.get_league_settings()
    assert result == {}


def test_get_league_standings_unauthenticated(client):
    result = client.get_league_standings()
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_get_all_rosters_unauthenticated(client):
    result = client.get_all_rosters()
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_get_team_roster_unauthenticated(client):
    result = client.get_team_roster("some.team.key")
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_get_free_agents_unauthenticated(client):
    result = client.get_free_agents()
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_get_league_transactions_unauthenticated(client):
    result = client.get_league_transactions()
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_get_draft_results_unauthenticated(client):
    result = client.get_draft_results()
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_sync_to_db_unauthenticated(client):
    result = client.sync_to_db()
    assert result == {}


# ---------------------------------------------------------------------------
# authenticate — yfpy not available
# ---------------------------------------------------------------------------


def test_authenticate_without_yfpy(client):
    """When YFPY_AVAILABLE is False, authenticate returns False."""
    with patch("src.yahoo_api.YFPY_AVAILABLE", False):
        result = client.authenticate("a" * 72, "b" * 32)
    assert result is False
    assert client.is_authenticated is False


def test_authenticate_invalid_credentials(client):
    """Invalid-format credentials are rejected before any API call."""
    result = client.authenticate("", "")
    assert result is False


# ---------------------------------------------------------------------------
# authenticate — yfpy available, mocked
# ---------------------------------------------------------------------------


def test_authenticate_success(client):
    """Successful auth sets _query and returns True."""
    mock_query = MagicMock()
    with (
        patch("src.yahoo_api.YFPY_AVAILABLE", True),
        patch("src.yahoo_api.YahooFantasySportsQuery", create=True, return_value=mock_query),
    ):
        result = client.authenticate("a" * 72, "b" * 32)
    assert result is True
    assert client.is_authenticated is True


def test_authenticate_api_failure(client):
    """When yfpy constructor raises, authenticate returns False."""
    with (
        patch("src.yahoo_api.YFPY_AVAILABLE", True),
        patch(
            "src.yahoo_api.YahooFantasySportsQuery",
            create=True,
            side_effect=Exception("OAuth failed"),
        ),
    ):
        result = client.authenticate("a" * 72, "b" * 32)
    assert result is False
    assert client.is_authenticated is False


# ---------------------------------------------------------------------------
# refresh_token
# ---------------------------------------------------------------------------


def test_refresh_token_unauthenticated(client):
    assert client.refresh_token() is False


def test_refresh_token_success(client):
    client._query = MagicMock()
    with patch("src.yahoo_api.YFPY_AVAILABLE", True):
        assert client.refresh_token() is True


def test_refresh_token_failure(client):
    client._query = MagicMock()
    client._query.get_league_metadata.side_effect = Exception("expired")
    with patch("src.yahoo_api.YFPY_AVAILABLE", True):
        assert client.refresh_token() is False


# ---------------------------------------------------------------------------
# get_league_settings — authenticated with mock
# ---------------------------------------------------------------------------


def test_get_league_settings_authenticated(client):
    mock_settings = MagicMock()
    mock_settings.name = "FourzynBurn"
    mock_settings.num_teams = 12
    mock_settings.draft_type = "live"
    mock_settings.roster_positions = []
    mock_settings.stat_categories = None

    client._query = MagicMock()
    client._query.get_league_settings.return_value = mock_settings

    result = client.get_league_settings()
    assert result["name"] == "FourzynBurn"
    assert result["num_teams"] == 12
    assert result["draft_type"] == "live"
    assert result["scoring_categories"] == []
    assert result["roster_positions"] == {}


def test_get_league_settings_with_categories(client):
    mock_stat = MagicMock()
    mock_stat.stat = MagicMock()
    mock_stat.stat.display_name = "HR"

    mock_cats = MagicMock()
    mock_cats.stats = [mock_stat]

    mock_settings = MagicMock()
    mock_settings.name = "TestLeague"
    mock_settings.num_teams = 10
    mock_settings.draft_type = "live"
    mock_settings.roster_positions = []
    mock_settings.stat_categories = mock_cats

    client._query = MagicMock()
    client._query.get_league_settings.return_value = mock_settings

    result = client.get_league_settings()
    assert "HR" in result["scoring_categories"]


def test_get_league_settings_with_roster_positions(client):
    mock_pos = MagicMock()
    mock_pos.position = "C"
    mock_pos.count = 1

    mock_settings = MagicMock()
    mock_settings.name = "TestLeague"
    mock_settings.num_teams = 12
    mock_settings.draft_type = "live"
    mock_settings.roster_positions = [mock_pos]
    mock_settings.stat_categories = None

    client._query = MagicMock()
    client._query.get_league_settings.return_value = mock_settings

    result = client.get_league_settings()
    assert result["roster_positions"]["C"] == 1


def test_get_league_settings_api_error(client):
    client._query = MagicMock()
    client._query.get_league_settings.side_effect = Exception("API error")

    result = client.get_league_settings()
    assert result == {}


# ---------------------------------------------------------------------------
# get_draft_results — authenticated with mock
# ---------------------------------------------------------------------------


def test_get_draft_results_empty(client):
    client._query = MagicMock()
    client._query.get_league_draft_results.return_value = []

    result = client.get_draft_results()
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_get_draft_results_with_picks(client):
    mock_pick = MagicMock()
    mock_pick.draft_result = mock_pick
    mock_pick.pick = 1
    mock_pick.round = 1
    mock_pick.team_name = "Team Hickey"
    mock_pick.team_key = "422.l.12345.t.1"
    mock_pick.player_key = "422.p.10566"
    mock_player = MagicMock()
    mock_player.name = MagicMock()
    mock_player.name.full = "Aaron Judge"
    mock_pick.player = mock_player

    client._query = MagicMock()
    client._query.get_league_draft_results.return_value = [mock_pick]

    result = client.get_draft_results()
    assert len(result) == 1
    assert result.iloc[0]["pick_number"] == 1
    assert result.iloc[0]["player_name"] == "Aaron Judge"
    assert result.iloc[0]["team_key"] == "422.l.12345.t.1"


def test_get_draft_results_sorted(client):
    picks = []
    for i, (pick_num, name) in enumerate([(3, "Player C"), (1, "Player A"), (2, "Player B")]):
        mp = MagicMock()
        mp.draft_result = mp
        mp.pick = pick_num
        mp.round = 1
        mp.team_name = f"Team {i}"
        mp.team_key = f"t.{i}"
        mp.player_key = f"p.{i}"
        mp.player = None
        picks.append(mp)

    client._query = MagicMock()
    client._query.get_league_draft_results.return_value = picks

    result = client.get_draft_results()
    assert list(result["pick_number"]) == [1, 2, 3]


def test_get_draft_results_api_error(client):
    client._query = MagicMock()
    client._query.get_league_draft_results.side_effect = Exception("API error")

    result = client.get_draft_results()
    assert isinstance(result, pd.DataFrame)
    assert result.empty


# ---------------------------------------------------------------------------
# get_league_standings — authenticated with mock
# ---------------------------------------------------------------------------


def test_get_league_standings_empty(client):
    mock_standings = MagicMock()
    mock_standings.teams = []

    client._query = MagicMock()
    client._query.get_league_standings.return_value = mock_standings

    result = client.get_league_standings()
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_get_league_standings_api_error(client):
    client._query = MagicMock()
    client._query.get_league_standings.side_effect = Exception("API error")

    result = client.get_league_standings()
    assert isinstance(result, pd.DataFrame)
    assert result.empty


# ---------------------------------------------------------------------------
# get_free_agents — authenticated with mock
# ---------------------------------------------------------------------------


def test_get_free_agents_empty(client):
    client._query = MagicMock()
    client._query.get_league_players.return_value = []

    result = client.get_free_agents()
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_get_free_agents_api_error(client):
    client._query = MagicMock()
    client._query.get_league_players.side_effect = Exception("API error")

    result = client.get_free_agents()
    assert isinstance(result, pd.DataFrame)
    assert result.empty


# ---------------------------------------------------------------------------
# get_league_transactions — authenticated with mock
# ---------------------------------------------------------------------------


def test_get_league_transactions_empty(client):
    client._query = MagicMock()
    client._query.get_league_transactions.return_value = []

    result = client.get_league_transactions()
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_get_league_transactions_api_error(client):
    client._query = MagicMock()
    client._query.get_league_transactions.side_effect = Exception("API error")

    result = client.get_league_transactions()
    assert isinstance(result, pd.DataFrame)
    assert result.empty

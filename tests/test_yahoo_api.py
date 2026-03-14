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
    # URL-encoded: https://example.com/callback → https%3A%2F%2Fexample.com%2Fcallback
    assert "redirect_uri=https%3A%2F%2Fexample.com%2Fcallback" in url


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
    # _resolve_game_key() calls get_game_key_by_season() which must
    # return a string that int() can parse (e.g. a Yahoo game key).
    mock_query.get_game_key_by_season.return_value = "449"
    with (
        patch("src.yahoo_api.YFPY_AVAILABLE", True),
        patch("src.yahoo_api.YahooFantasySportsQuery", create=True, return_value=mock_query),
    ):
        result = client.authenticate("a" * 72, "b" * 32)
    assert result is True
    assert client.is_authenticated is True
    # Verify game_id and league_key were set on the query object
    assert mock_query.game_id == 449
    assert mock_query.league_key == "449.l.12345"


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


# ---------------------------------------------------------------------------
# _safe_str helper
# ---------------------------------------------------------------------------


def test_safe_str_with_bytes():
    assert YahooFantasyClient._safe_str(b"Aaron Judge") == "Aaron Judge"


def test_safe_str_with_str():
    assert YahooFantasyClient._safe_str("Aaron Judge") == "Aaron Judge"


def test_safe_str_with_none():
    assert YahooFantasyClient._safe_str(None) == ""
    assert YahooFantasyClient._safe_str(None, "fallback") == "fallback"


# ---------------------------------------------------------------------------
# _extract_position helper
# ---------------------------------------------------------------------------


def test_extract_position_plain_string():
    assert YahooFantasyClient._extract_position("SS") == "SS"


def test_extract_position_bytes():
    assert YahooFantasyClient._extract_position(b"3B") == "3B"


def test_extract_position_model_object():
    """yfpy EligiblePosition model objects have a .position attribute."""
    mock_pos = MagicMock(spec=[])  # spec=[] prevents auto-creating attrs
    mock_pos.position = "CF"
    assert YahooFantasyClient._extract_position(mock_pos) == "CF"


def test_extract_position_model_object_bytes():
    """Position attribute itself may be bytes on Python 3.14."""
    mock_pos = MagicMock(spec=[])
    mock_pos.position = b"1B"
    assert YahooFantasyClient._extract_position(mock_pos) == "1B"


def test_extract_position_display_name_fallback():
    """Falls back to display_name if position attr is missing."""
    mock_pos = MagicMock(spec=[])
    mock_pos.display_name = "LF"
    assert YahooFantasyClient._extract_position(mock_pos) == "LF"


# ---------------------------------------------------------------------------
# BUG 3: get_free_agents bytes decode
# ---------------------------------------------------------------------------


def test_get_free_agents_bytes_player_name(client):
    """Player names from yfpy may be bytes on Python 3.14."""
    mock_player = MagicMock()
    mock_name = MagicMock()
    mock_name.full = b"Shohei Ohtani"  # bytes
    mock_player.name = mock_name
    mock_player.eligible_positions = ["DH", "SP"]
    mock_player.percent_owned = 99.5
    mock_player.player_id = "12345"
    mock_entry = MagicMock()
    mock_entry.player = mock_player

    client._query = MagicMock()
    client._query.get_league_players.return_value = [mock_entry]

    result = client.get_free_agents()
    assert len(result) == 1
    assert result.iloc[0]["player_name"] == "Shohei Ohtani"
    assert "DH" in result.iloc[0]["positions"]


def test_get_free_agents_position_model_objects(client):
    """eligible_positions may be yfpy model objects, not plain strings."""
    mock_player = MagicMock()
    mock_name = MagicMock()
    mock_name.full = "Mookie Betts"
    mock_player.name = mock_name
    mock_player.player_id = "99"
    mock_player.percent_owned = 85.0

    # Simulate yfpy EligiblePosition model objects
    mock_pos_ss = MagicMock(spec=[])
    mock_pos_ss.position = "SS"
    mock_pos_of = MagicMock(spec=[])
    mock_pos_of.position = "OF"
    mock_player.eligible_positions = [mock_pos_ss, mock_pos_of]

    mock_entry = MagicMock()
    mock_entry.player = mock_player

    client._query = MagicMock()
    client._query.get_league_players.return_value = [mock_entry]

    result = client.get_free_agents()
    assert len(result) == 1
    assert result.iloc[0]["positions"] == "SS/OF"


def test_get_free_agents_position_filter_with_model_objects(client):
    """Position filter should work even when positions are model objects."""
    mock_player = MagicMock()
    mock_name = MagicMock()
    mock_name.full = "Player A"
    mock_player.name = mock_name
    mock_player.player_id = "1"
    mock_player.percent_owned = 10.0

    mock_pos = MagicMock(spec=[])
    mock_pos.position = "C"
    mock_player.eligible_positions = [mock_pos]

    mock_entry = MagicMock()
    mock_entry.player = mock_player

    client._query = MagicMock()
    client._query.get_league_players.return_value = [mock_entry]

    # Filter for SS should exclude this C-only player
    result = client.get_free_agents(position="SS")
    assert result.empty

    # Filter for C should include this player
    result = client.get_free_agents(position="C")
    assert len(result) == 1


# ---------------------------------------------------------------------------
# BUG 4: get_league_transactions bytes decode
# ---------------------------------------------------------------------------


def test_get_league_transactions_bytes_names(client):
    """Player and team names from transactions may be bytes."""
    mock_player = MagicMock()
    mock_name_obj = MagicMock()
    mock_name_obj.full = b"Juan Soto"
    mock_player.name = mock_name_obj

    mock_tx_data = MagicMock()
    mock_tx_data.source_team_name = b"BUBBA CROSBY"
    mock_tx_data.destination_team_name = b"Team Hickey"
    mock_player.transaction_data = mock_tx_data

    mock_p_entry = MagicMock()
    mock_p_entry.player = mock_player

    mock_tx = MagicMock()
    mock_tx.transaction = mock_tx
    mock_tx.transaction_id = "tx123"
    mock_tx.type = "trade"
    mock_tx.timestamp = "1710000000"
    mock_tx.players = [mock_p_entry]

    client._query = MagicMock()
    client._query.get_league_transactions.return_value = [mock_tx]

    result = client.get_league_transactions()
    assert len(result) == 1
    assert result.iloc[0]["player_name"] == "Juan Soto"
    assert result.iloc[0]["team_from"] == "BUBBA CROSBY"
    assert result.iloc[0]["team_to"] == "Team Hickey"


# ---------------------------------------------------------------------------
# BUG 5: get_draft_results bytes decode
# ---------------------------------------------------------------------------


def test_get_draft_results_bytes_names(client):
    """Player and team names from draft results may be bytes."""
    mock_pick = MagicMock()
    mock_pick.draft_result = mock_pick
    mock_pick.pick = 1
    mock_pick.round = 1
    mock_pick.team_name = b"BUBBA CROSBY"  # bytes team name
    mock_pick.team_key = "469.l.109662.t.3"
    mock_pick.player_key = "469.p.10566"

    mock_player = MagicMock()
    mock_player.name = MagicMock()
    mock_player.name.full = b"Aaron Judge"  # bytes player name
    mock_pick.player = mock_player

    client._query = MagicMock()
    client._query.get_league_draft_results.return_value = [mock_pick]

    result = client.get_draft_results()
    assert len(result) == 1
    assert result.iloc[0]["player_name"] == "Aaron Judge"
    assert result.iloc[0]["team_name"] == "BUBBA CROSBY"


def test_get_draft_results_no_name_obj_bytes(client):
    """When player has no name obj, str(player) bytes should be handled."""
    mock_pick = MagicMock()
    mock_pick.draft_result = mock_pick
    mock_pick.pick = 5
    mock_pick.round = 1
    mock_pick.team_name = "Team Normal"
    mock_pick.team_key = "469.l.109662.t.1"
    mock_pick.player_key = "469.p.999"

    # player exists but has no .name attribute
    mock_player = MagicMock(spec=[])
    mock_pick.player = mock_player

    client._query = MagicMock()
    client._query.get_league_draft_results.return_value = [mock_pick]

    result = client.get_draft_results()
    assert len(result) == 1
    # Should not crash and should produce some string for the name
    assert isinstance(result.iloc[0]["player_name"], str)
    assert len(result.iloc[0]["player_name"]) > 0


# ---------------------------------------------------------------------------
# BUG 6: eligible_positions model objects in rosters
# ---------------------------------------------------------------------------


def test_get_all_rosters_position_model_objects(client):
    """Roster position extraction should handle yfpy model objects."""
    mock_team = MagicMock()
    mock_team.name = "Test Team"
    mock_team.team_key = "469.l.12345.t.1"

    mock_player = MagicMock()
    mock_player.name = MagicMock()
    mock_player.name.full = "Trea Turner"
    mock_player.player_id = "42"
    mock_player.status = "active"

    # Simulate yfpy EligiblePosition model objects
    mock_pos_ss = MagicMock(spec=[])
    mock_pos_ss.position = "SS"
    mock_pos_2b = MagicMock(spec=[])
    mock_pos_2b.position = "2B"
    mock_player.eligible_positions = [mock_pos_ss, mock_pos_2b]

    mock_p_entry = MagicMock()
    mock_p_entry.player = mock_player

    mock_roster = MagicMock()
    mock_roster.players = [mock_p_entry]

    client._query = MagicMock()
    client._query.get_league_teams.return_value = [mock_team]
    client._query.get_team_roster_by_week.return_value = mock_roster

    result = client.get_all_rosters()
    assert len(result) == 1
    assert result.iloc[0]["position"] == "SS/2B"


def test_get_team_roster_position_model_objects(client):
    """Team roster position extraction should handle yfpy model objects."""
    mock_player = MagicMock()
    mock_player.name = MagicMock()
    mock_player.name.full = "Francisco Lindor"
    mock_player.player_id = "55"
    mock_player.status = b"active"  # also test bytes status

    mock_pos = MagicMock(spec=[])
    mock_pos.position = b"SS"  # bytes position
    mock_player.eligible_positions = [mock_pos]

    mock_p_entry = MagicMock()
    mock_p_entry.player = mock_player

    mock_roster = MagicMock()
    mock_roster.players = [mock_p_entry]

    client._query = MagicMock()
    client._query.get_team_roster_by_week.return_value = mock_roster

    result = client.get_team_roster("469.l.12345.t.1")
    assert len(result) == 1
    assert result.iloc[0]["position"] == "SS"
    assert result.iloc[0]["status"] == "active"

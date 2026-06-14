"""TDD tests for YahooFantasyClient write methods: set_lineup and add_drop.

All tests are fully mocked — no real network calls.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.yahoo_api import YahooFantasyClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """Unauthenticated client."""
    return YahooFantasyClient(league_id="109662", game_code="mlb", season=2026)


@pytest.fixture
def authed_client():
    """Client with a mocked authenticated session."""
    c = YahooFantasyClient(league_id="109662", game_code="mlb", season=2026)
    mock_query = MagicMock()
    mock_query.game_id = 469
    mock_query.league_key = "469.l.109662"
    mock_query._yahoo_access_token_dict = {"access_token": "fake_token_abc123"}
    c._query = mock_query
    return c


# ---------------------------------------------------------------------------
# XML builder: _build_roster_xml
# ---------------------------------------------------------------------------


def test_build_roster_xml_structure(authed_client):
    """_build_roster_xml produces valid XML with player_key, position, and date."""
    assignments = [
        {"player_key": "469.p.10001", "position": "SS"},
        {"player_key": "469.p.10002", "position": "BN"},
    ]
    xml = authed_client._build_roster_xml(assignments, "2026-06-14")
    assert "<player_key>469.p.10001</player_key>" in xml
    assert "<position>SS</position>" in xml
    assert "<player_key>469.p.10002</player_key>" in xml
    assert "<position>BN</position>" in xml
    assert "2026-06-14" in xml


def test_build_roster_xml_root_element(authed_client):
    """_build_roster_xml has a <fantasy_content> or <roster> root."""
    xml = authed_client._build_roster_xml([{"player_key": "469.p.9999", "position": "1B"}], "2026-06-14")
    # Must be valid XML with a recognizable root
    assert xml.strip().startswith("<")
    # Must contain the player entry
    assert "469.p.9999" in xml


def test_build_roster_xml_multiple_players(authed_client):
    """All assignments appear in the XML output."""
    assignments = [{"player_key": f"469.p.{i}", "position": "BN"} for i in range(5)]
    xml = authed_client._build_roster_xml(assignments, "2026-06-15")
    for i in range(5):
        assert f"469.p.{i}" in xml


# ---------------------------------------------------------------------------
# XML builder: _build_transaction_xml
# ---------------------------------------------------------------------------


def test_build_transaction_xml_add_drop(authed_client):
    """_build_transaction_xml add+drop includes both player_keys and correct types."""
    xml = authed_client._build_transaction_xml(
        add_player_key="469.p.11111",
        drop_player_key="469.p.22222",
        team_key="469.l.109662.t.7",
    )
    assert "469.p.11111" in xml
    assert "469.p.22222" in xml
    assert "add" in xml.lower()
    assert "drop" in xml.lower()


def test_build_transaction_xml_add_only(authed_client):
    """_build_transaction_xml add-only omits the drop player_key."""
    xml = authed_client._build_transaction_xml(
        add_player_key="469.p.11111",
        drop_player_key=None,
        team_key="469.l.109662.t.7",
    )
    assert "469.p.11111" in xml
    # Should be an "add" type transaction
    assert "add" in xml.lower()


def test_build_transaction_xml_drop_only(authed_client):
    """_build_transaction_xml drop-only omits the add player_key."""
    xml = authed_client._build_transaction_xml(
        add_player_key=None,
        drop_player_key="469.p.22222",
        team_key="469.l.109662.t.7",
    )
    assert "469.p.22222" in xml
    # Should be a "drop" type transaction
    assert "drop" in xml.lower()


def test_build_transaction_xml_contains_league_key(authed_client):
    """_build_transaction_xml root or type element references the league."""
    xml = authed_client._build_transaction_xml(
        add_player_key="469.p.11111",
        drop_player_key="469.p.22222",
        team_key="469.l.109662.t.7",
    )
    # The XML body used by the POST need not embed league key (it goes in the URL),
    # but the team_key must appear so Yahoo knows which team's roster to update.
    assert "469.l.109662.t.7" in xml


# ---------------------------------------------------------------------------
# set_lineup — unauthenticated
# ---------------------------------------------------------------------------


def test_set_lineup_not_connected(client):
    """set_lineup returns ok=False with 'Not connected' when unauthenticated."""
    result = client.set_lineup([{"player_key": "469.p.1", "position": "SS"}], "2026-06-14")
    assert result["ok"] is False
    assert "not connected" in result["error"].lower()


# ---------------------------------------------------------------------------
# set_lineup — no bearer token
# ---------------------------------------------------------------------------


def test_set_lineup_no_token(authed_client):
    """set_lineup returns ok=False when bearer token is unavailable."""
    authed_client._query._yahoo_access_token_dict = None
    # Also make sure disk fallback finds nothing
    with patch("src.yahoo_api._AUTH_DIR") as mock_dir:
        mock_token_path = MagicMock()
        mock_token_path.exists.return_value = False
        mock_dir.__truediv__ = lambda self, other: mock_token_path
        result = authed_client.set_lineup([{"player_key": "469.p.1", "position": "SS"}], "2026-06-14")
    # Should fail gracefully (ok=False), not raise
    assert result["ok"] is False


# ---------------------------------------------------------------------------
# set_lineup — 200 OK → success
# ---------------------------------------------------------------------------


def test_set_lineup_success_200(authed_client):
    """set_lineup returns ok=True and applied count on HTTP 200."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()

    assignments = [
        {"player_key": "469.p.10001", "position": "SS"},
        {"player_key": "469.p.10002", "position": "BN"},
    ]

    with patch("src.yahoo_api._requests") as mock_requests:
        mock_requests.put.return_value = mock_resp
        result = authed_client.set_lineup(assignments, "2026-06-14")

    assert result["ok"] is True
    assert result["applied"] == 2


# ---------------------------------------------------------------------------
# set_lineup — 401 → not-authorized message
# ---------------------------------------------------------------------------


def test_set_lineup_401_returns_not_authorized(authed_client):
    """set_lineup maps HTTP 401 to the fspt-w not-authorized message."""
    mock_resp = MagicMock()
    mock_resp.status_code = 401

    with patch("src.yahoo_api._requests") as mock_requests:
        mock_requests.put.return_value = mock_resp
        result = authed_client.set_lineup([{"player_key": "469.p.1", "position": "SP"}], "2026-06-14")

    assert result["ok"] is False
    assert result["status"] == 401
    assert "fspt-w" in result["error"] or "write" in result["error"].lower()


def test_set_lineup_403_returns_not_authorized(authed_client):
    """set_lineup maps HTTP 403 to the not-authorized message."""
    mock_resp = MagicMock()
    mock_resp.status_code = 403

    with patch("src.yahoo_api._requests") as mock_requests:
        mock_requests.put.return_value = mock_resp
        result = authed_client.set_lineup([{"player_key": "469.p.1", "position": "SP"}], "2026-06-14")

    assert result["ok"] is False
    assert result["status"] == 403


# ---------------------------------------------------------------------------
# set_lineup — non-2xx other than 401/403
# ---------------------------------------------------------------------------


def test_set_lineup_500_returns_error(authed_client):
    """set_lineup returns ok=False with HTTP status on a server error."""
    mock_resp = MagicMock()
    mock_resp.status_code = 500

    with patch("src.yahoo_api._requests") as mock_requests:
        mock_requests.put.return_value = mock_resp
        result = authed_client.set_lineup([{"player_key": "469.p.1", "position": "SP"}], "2026-06-14")

    assert result["ok"] is False
    assert result.get("status") == 500


# ---------------------------------------------------------------------------
# set_lineup — network exception → graceful
# ---------------------------------------------------------------------------


def test_set_lineup_network_exception_does_not_raise(authed_client):
    """set_lineup catches network exceptions and returns ok=False, never raises."""
    with patch("src.yahoo_api._requests") as mock_requests:
        mock_requests.put.side_effect = ConnectionError("Network down")
        result = authed_client.set_lineup([{"player_key": "469.p.1", "position": "C"}], "2026-06-14")

    assert result["ok"] is False
    assert "error" in result


def test_set_lineup_timeout_does_not_raise(authed_client):
    """set_lineup catches timeout exceptions and returns ok=False, never raises."""
    import socket

    with patch("src.yahoo_api._requests") as mock_requests:
        mock_requests.put.side_effect = TimeoutError("timed out")
        result = authed_client.set_lineup([{"player_key": "469.p.1", "position": "C"}], "2026-06-14")

    assert result["ok"] is False


# ---------------------------------------------------------------------------
# add_drop — unauthenticated
# ---------------------------------------------------------------------------


def test_add_drop_not_connected(client):
    """add_drop returns ok=False with 'Not connected' when unauthenticated."""
    result = client.add_drop(add_player_key="469.p.1", drop_player_key="469.p.2")
    assert result["ok"] is False
    assert "not connected" in result["error"].lower()


# ---------------------------------------------------------------------------
# add_drop — 200 OK → success
# ---------------------------------------------------------------------------


def test_add_drop_success_200(authed_client):
    """add_drop returns ok=True on HTTP 200 (add+drop)."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()

    with patch("src.yahoo_api._requests") as mock_requests:
        mock_requests.post.return_value = mock_resp
        result = authed_client.add_drop(add_player_key="469.p.11111", drop_player_key="469.p.22222")

    assert result["ok"] is True


def test_add_drop_add_only_success(authed_client):
    """add_drop with add-only (drop_player_key=None) returns ok=True on 200."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()

    with patch("src.yahoo_api._requests") as mock_requests:
        mock_requests.post.return_value = mock_resp
        result = authed_client.add_drop(add_player_key="469.p.11111", drop_player_key=None)

    assert result["ok"] is True


def test_add_drop_drop_only_success(authed_client):
    """add_drop with drop-only (add_player_key=None) returns ok=True on 200."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()

    with patch("src.yahoo_api._requests") as mock_requests:
        mock_requests.post.return_value = mock_resp
        result = authed_client.add_drop(add_player_key=None, drop_player_key="469.p.22222")

    assert result["ok"] is True


# ---------------------------------------------------------------------------
# add_drop — both None → validation error
# ---------------------------------------------------------------------------


def test_add_drop_both_none_returns_error(authed_client):
    """add_drop with both keys None returns ok=False (no-op guard)."""
    result = authed_client.add_drop(add_player_key=None, drop_player_key=None)
    assert result["ok"] is False
    assert "error" in result


# ---------------------------------------------------------------------------
# add_drop — 401/403 → not-authorized message
# ---------------------------------------------------------------------------


def test_add_drop_401_returns_not_authorized(authed_client):
    """add_drop maps HTTP 401 to the fspt-w not-authorized message."""
    mock_resp = MagicMock()
    mock_resp.status_code = 401

    with patch("src.yahoo_api._requests") as mock_requests:
        mock_requests.post.return_value = mock_resp
        result = authed_client.add_drop(add_player_key="469.p.1", drop_player_key="469.p.2")

    assert result["ok"] is False
    assert result["status"] == 401
    assert "fspt-w" in result["error"] or "write" in result["error"].lower()


def test_add_drop_403_returns_not_authorized(authed_client):
    """add_drop maps HTTP 403 to the not-authorized message."""
    mock_resp = MagicMock()
    mock_resp.status_code = 403

    with patch("src.yahoo_api._requests") as mock_requests:
        mock_requests.post.return_value = mock_resp
        result = authed_client.add_drop(add_player_key="469.p.1", drop_player_key=None)

    assert result["ok"] is False
    assert result["status"] == 403


# ---------------------------------------------------------------------------
# add_drop — network exception → graceful
# ---------------------------------------------------------------------------


def test_add_drop_network_exception_does_not_raise(authed_client):
    """add_drop catches network exceptions and returns ok=False, never raises."""
    with patch("src.yahoo_api._requests") as mock_requests:
        mock_requests.post.side_effect = ConnectionError("Network down")
        result = authed_client.add_drop(add_player_key="469.p.1", drop_player_key="469.p.2")

    assert result["ok"] is False
    assert "error" in result


# ---------------------------------------------------------------------------
# Structured-result contract checks
# ---------------------------------------------------------------------------


def test_set_lineup_ok_result_has_applied_key(authed_client):
    """Successful set_lineup result always contains 'applied' key."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch("src.yahoo_api._requests") as mock_requests:
        mock_requests.put.return_value = mock_resp
        result = authed_client.set_lineup([{"player_key": "469.p.1", "position": "SP"}], "2026-06-14")

    assert "applied" in result


def test_set_lineup_error_result_has_error_key(authed_client):
    """Failed set_lineup result always contains 'error' key."""
    mock_resp = MagicMock()
    mock_resp.status_code = 503

    with patch("src.yahoo_api._requests") as mock_requests:
        mock_requests.put.return_value = mock_resp
        result = authed_client.set_lineup([{"player_key": "469.p.1", "position": "SP"}], "2026-06-14")

    assert "error" in result
    assert result["ok"] is False


def test_add_drop_ok_result_shape(authed_client):
    """Successful add_drop result has ok=True and no required extra keys."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch("src.yahoo_api._requests") as mock_requests:
        mock_requests.post.return_value = mock_resp
        result = authed_client.add_drop(add_player_key="469.p.1", drop_player_key="469.p.2")

    assert result["ok"] is True


def test_add_drop_error_result_has_error_key(authed_client):
    """Failed add_drop result always contains 'error' and ok=False."""
    mock_resp = MagicMock()
    mock_resp.status_code = 503

    with patch("src.yahoo_api._requests") as mock_requests:
        mock_requests.post.return_value = mock_resp
        result = authed_client.add_drop(add_player_key="469.p.1", drop_player_key="469.p.2")

    assert result["ok"] is False
    assert "error" in result


# ---------------------------------------------------------------------------
# URL construction sanity checks (PUT to team roster, POST to league txn)
# ---------------------------------------------------------------------------


def test_set_lineup_puts_to_team_roster_url(authed_client):
    """set_lineup calls PUT to the Yahoo team roster endpoint."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch("src.yahoo_api._requests") as mock_requests:
        mock_requests.put.return_value = mock_resp
        authed_client.set_lineup([{"player_key": "469.p.1", "position": "C"}], "2026-06-14")

    assert mock_requests.put.called
    call_url = mock_requests.put.call_args[0][0]
    assert "roster" in call_url
    assert "fantasysports.yahooapis.com" in call_url


def test_add_drop_posts_to_league_transactions_url(authed_client):
    """add_drop calls POST to the Yahoo league transactions endpoint."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch("src.yahoo_api._requests") as mock_requests:
        mock_requests.post.return_value = mock_resp
        authed_client.add_drop(add_player_key="469.p.1", drop_player_key="469.p.2")

    assert mock_requests.post.called
    call_url = mock_requests.post.call_args[0][0]
    assert "transactions" in call_url
    assert "fantasysports.yahooapis.com" in call_url

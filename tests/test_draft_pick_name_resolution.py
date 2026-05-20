"""SFH L8 guard (2026-05-20): draft pick names resolve via batch Yahoo lookup.

Background:
  yfpy's get_league_draft_results() returns pick entries with a
  player_key but does NOT expand the player resource. The
  get_draft_results loop fell back to "Player {player_key}"
  placeholders for ~70% of rounds 1-3 picks in today's bootstrap
  (visible in bootstrap.log as repeated 'Could not resolve player_id
  for draft pick' warnings). Those placeholders defeated downstream
  name-based player_id resolution and the undroppable flag (rounds 1-3
  player can't be dropped) wasn't getting set.

This file pins the fix:
  1. resolve_player_names_by_keys posts to Yahoo's batch Players API
     and returns {player_key: name}.
  2. get_draft_results uses it after the main yfpy loop — picks with
     resolved names get the real name; still-unresolved fall back to
     the legacy "Player {key}" string so DB NOT-NULL constraints
     don't break.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def yahoo_client():
    """Build a minimal YahooFantasyClient with mocked auth."""
    from src.yahoo_api import YahooFantasyClient

    client = YahooFantasyClient.__new__(YahooFantasyClient)
    client._ensure_auth = MagicMock(return_value=True)
    client._query = MagicMock()
    client._query._yahoo_access_token_dict = {"access_token": "fake-token"}
    client._query.game_id = 469
    client.league_id = "12345"
    return client


def _build_yfpy_pick(pick: int, rd: int, team_name: str, team_key: str, player_key: str):
    """Build a mock yfpy DraftResult entry where the player resource is NOT expanded."""
    pick_data = MagicMock()
    pick_data.pick = pick
    pick_data.round = rd
    pick_data.team_name = team_name
    pick_data.team_key = team_key
    pick_data.player_key = player_key
    # player attribute is missing/empty — the signature of unexpanded picks.
    pick_data.player = None

    entry = MagicMock()
    entry.draft_result = pick_data
    return entry


def test_resolve_player_names_by_keys_returns_mapping(yahoo_client):
    """L8: batch lookup returns {player_key: name} for resolved entries."""
    fake_resp = MagicMock()
    fake_resp.json.return_value = {
        "fantasy_content": {
            "players": {
                "count": 2,
                "0": {
                    "player": [
                        [
                            {"player_key": "469.p.10480"},
                            {"name": {"full": "Shohei Ohtani"}},
                        ]
                    ]
                },
                "1": {
                    "player": [
                        [
                            {"player_key": "469.p.9999"},
                            {"name": {"full": "Aaron Judge"}},
                        ]
                    ]
                },
            }
        }
    }
    with patch("src.yahoo_api._request_with_backoff", return_value=fake_resp):
        resolved = yahoo_client.resolve_player_names_by_keys(["469.p.10480", "469.p.9999"])
    assert resolved == {"469.p.10480": "Shohei Ohtani", "469.p.9999": "Aaron Judge"}


def test_resolve_player_names_empty_input(yahoo_client):
    """L8: empty input → empty output, no HTTP call."""
    with patch("src.yahoo_api._request_with_backoff") as mock_req:
        resolved = yahoo_client.resolve_player_names_by_keys([])
    assert resolved == {}
    mock_req.assert_not_called()


def test_resolve_player_names_batches_at_25(yahoo_client):
    """L8: more than 25 keys → multiple batched HTTP calls (Yahoo's per-call cap)."""
    fake_resp = MagicMock()
    fake_resp.json.return_value = {"fantasy_content": {"players": {"count": 0}}}
    keys = [f"469.p.{i}" for i in range(60)]  # 60 keys → 3 batches (25, 25, 10)
    with patch("src.yahoo_api._request_with_backoff", return_value=fake_resp) as mock_req:
        yahoo_client.resolve_player_names_by_keys(keys)
    # Note: _rate_limit() is also called once before each request, but the
    # batched API call count itself should be ceil(60/25) = 3.
    assert mock_req.call_count == 3


def test_get_draft_results_resolves_placeholders(yahoo_client):
    """L8 main scenario: unexpanded picks get real names via batch lookup,
    not 'Player 469.p.XXXX' placeholders."""
    picks = [
        _build_yfpy_pick(1, 1, "Team A", "469.l.1.t.1", "469.p.660271"),
        _build_yfpy_pick(2, 1, "Team B", "469.l.1.t.2", "469.p.592450"),
    ]
    yahoo_client._query.get_league_draft_results = MagicMock(return_value=picks)

    fake_resp = MagicMock()
    fake_resp.json.return_value = {
        "fantasy_content": {
            "players": {
                "count": 2,
                "0": {
                    "player": [
                        [
                            {"player_key": "469.p.660271"},
                            {"name": {"full": "Shohei Ohtani"}},
                        ]
                    ]
                },
                "1": {
                    "player": [
                        [
                            {"player_key": "469.p.592450"},
                            {"name": {"full": "Aaron Judge"}},
                        ]
                    ]
                },
            }
        }
    }
    with patch("src.yahoo_api._request_with_backoff", return_value=fake_resp):
        df = yahoo_client.get_draft_results()

    assert len(df) == 2
    names = sorted(df["player_name"].tolist())
    assert names == ["Aaron Judge", "Shohei Ohtani"]
    # Placeholder pattern must NOT appear in any name.
    assert not any(n.startswith("Player 469.p.") for n in df["player_name"])


def test_get_draft_results_falls_back_to_placeholder_when_resolve_fails(yahoo_client):
    """L8 regression guard: when batch resolve returns nothing for a key
    (Yahoo timeout, network failure, etc.), we still produce the legacy
    placeholder string so league_draft_picks INSERT (NOT NULL constraint
    on player_name) doesn't crash."""
    picks = [
        _build_yfpy_pick(1, 1, "Team A", "469.l.1.t.1", "469.p.99999"),
    ]
    yahoo_client._query.get_league_draft_results = MagicMock(return_value=picks)

    # Resolve returns empty (simulating Yahoo API failure for unknown key).
    with patch.object(yahoo_client, "resolve_player_names_by_keys", return_value={}):
        df = yahoo_client.get_draft_results()
    assert len(df) == 1
    assert df.iloc[0]["player_name"] == "Player 469.p.99999"


def test_get_draft_results_keeps_yfpy_resolved_names(yahoo_client):
    """L8 regression guard: when yfpy DID expand the player resource (the
    happy case), we use that name without re-fetching."""
    pick_data = MagicMock()
    pick_data.pick = 1
    pick_data.round = 1
    pick_data.team_name = "Team A"
    pick_data.team_key = "469.l.1.t.1"
    pick_data.player_key = "469.p.660271"
    name_obj = MagicMock()
    name_obj.full = "Shohei Ohtani"
    player_obj = MagicMock()
    player_obj.name = name_obj
    pick_data.player = player_obj

    entry = MagicMock()
    entry.draft_result = pick_data
    yahoo_client._query.get_league_draft_results = MagicMock(return_value=[entry])

    with patch("src.yahoo_api._request_with_backoff") as mock_req:
        df = yahoo_client.get_draft_results()
    assert len(df) == 1
    assert df.iloc[0]["player_name"] == "Shohei Ohtani"
    # No batch resolution call when yfpy already gave us the name.
    mock_req.assert_not_called()

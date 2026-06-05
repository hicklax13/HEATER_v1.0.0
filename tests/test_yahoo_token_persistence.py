"""Item #2 (2026-06-05): the refreshed Yahoo token must persist to the volume so it
survives its first hourly refresh.

yfpy constructs OAuth2 with ``store_file=False``, so it never writes the token itself.
After yfpy refreshes the access token (and any rotated refresh token) in memory, HEATER
must write it back to ``yahoo_token.json`` -- otherwise the next scheduler cycle / process
restart re-reads a stale token, re-refreshes an already-consumed token, and Yahoo
eventually rejects it. The token then "dies" ~1 hour after a paste (the live recurrence
seen 2026-06-05, where every short-TTL Data Freshness row went STALE while the 24h-TTL
settings/schedule rows kept the card falsely green).

These guard:
- ``persist_current_token()`` reads the LIVE oauth object, writes ATOMICALLY, refuses to
  write an incomplete token, falls back to yfpy's updated dict, and never clobbers a good
  file on failure.
- The scheduler persists the token each cycle so a restart re-reads the latest token.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.yahoo_api as yahoo_api
from src.yahoo_api import YahooFantasyClient


class _FakeOAuth:
    """Stand-in for yfpy's live ``query.oauth`` (a yahoo_oauth OAuth2 object)."""

    def __init__(self, access="", refresh="", token_time=0.0):
        self.access_token = access
        self.refresh_token = refresh
        self.token_time = token_time
        self.token_type = "bearer"
        self.guid = "GUID123"
        self.consumer_key = "CK"
        self.consumer_secret = "CS"


class _FakeQuery:
    def __init__(self, oauth=None, token_dict=None):
        if oauth is not None:
            self.oauth = oauth
        if token_dict is not None:
            self._yahoo_access_token_dict = token_dict


def _client_with(tmp_path, monkeypatch, query):
    monkeypatch.setattr(yahoo_api, "_AUTH_DIR", tmp_path)
    client = YahooFantasyClient(league_id="109662")
    client._query = query
    return client


def test_persist_current_token_writes_live_oauth_token(tmp_path, monkeypatch):
    oauth = _FakeOAuth(access="ACCESS_NEW", refresh="REFRESH_NEW", token_time=1780000000.0)
    client = _client_with(tmp_path, monkeypatch, _FakeQuery(oauth=oauth))

    assert client.persist_current_token() is True

    saved = json.loads((tmp_path / "yahoo_token.json").read_text(encoding="utf-8"))
    assert saved["access_token"] == "ACCESS_NEW"
    assert saved["refresh_token"] == "REFRESH_NEW"
    assert saved["token_time"] == 1780000000.0
    assert saved["consumer_key"] == "CK"
    assert saved["consumer_secret"] == "CS"
    assert saved["token_type"] == "bearer"
    # All 8 fields yfpy requires must be present so the file round-trips.
    for field in (
        "access_token",
        "consumer_key",
        "consumer_secret",
        "expires_in",
        "guid",
        "refresh_token",
        "token_time",
        "token_type",
    ):
        assert field in saved
    # Atomic write leaves no temp file behind.
    assert not list(tmp_path.glob("*.tmp"))


def test_persist_current_token_refuses_partial_token(tmp_path, monkeypatch):
    # A token without a refresh_token cannot be refreshed later -> refuse to persist it.
    oauth = _FakeOAuth(access="ACCESS_NEW", refresh="")
    client = _client_with(tmp_path, monkeypatch, _FakeQuery(oauth=oauth))

    assert client.persist_current_token() is False
    assert not (tmp_path / "yahoo_token.json").exists()


def test_persist_current_token_preserves_existing_file_on_failure(tmp_path, monkeypatch):
    good = {
        "access_token": "OLD",
        "refresh_token": "OLDR",
        "token_time": 1.0,
        "token_type": "bearer",
        "guid": "G",
        "consumer_key": "CK",
        "consumer_secret": "CS",
        "expires_in": 3600,
    }
    monkeypatch.setattr(yahoo_api, "_AUTH_DIR", tmp_path)
    (tmp_path / "yahoo_token.json").write_text(json.dumps(good), encoding="utf-8")
    client = YahooFantasyClient(league_id="109662")
    client._query = None  # nothing live to read

    assert client.persist_current_token() is False
    # The existing good token must be untouched (no partial/empty clobber).
    saved = json.loads((tmp_path / "yahoo_token.json").read_text(encoding="utf-8"))
    assert saved["access_token"] == "OLD"
    assert saved["refresh_token"] == "OLDR"


def test_persist_current_token_falls_back_to_token_dict(tmp_path, monkeypatch):
    # No live oauth object, but yfpy's updated dict carries the refreshed token.
    token_dict = {
        "access_token": "ACC",
        "refresh_token": "REF",
        "token_time": 42.0,
        "token_type": "bearer",
        "guid": "G",
        "consumer_key": "CK",
        "consumer_secret": "CS",
    }
    client = _client_with(tmp_path, monkeypatch, _FakeQuery(token_dict=token_dict))

    assert client.persist_current_token() is True
    saved = json.loads((tmp_path / "yahoo_token.json").read_text(encoding="utf-8"))
    assert saved["access_token"] == "ACC"
    assert saved["refresh_token"] == "REF"


def test_scheduler_persists_token_each_cycle():
    src = (Path(__file__).parent.parent / "src" / "scheduler.py").read_text(encoding="utf-8")
    assert "persist_current_token" in src, (
        "scheduler must persist the (possibly refreshed) Yahoo token each cycle so a "
        "restart / next cycle re-reads the latest token instead of re-refreshing a "
        "stale one until Yahoo rejects it (item #2, 2026-06-05)"
    )

"""Read-only MULTI_USER members never hold their own Yahoo client (the scheduler
is the sole connector), so `is_connected()` is always False for a member session.

Two consequences this guards against (2026-06-03 read-only-member data-gating fix):
1. The Data Freshness card screamed "Yahoo Offline" at every member even while the
   scheduler served live data. `connection_status()` reports DATA availability under
   MULTI_USER and session connection in v1 (byte-for-byte).
2. My Team gated the weekly matchup behind `is_connected()`, so it went dark for
   members. `get_matchup()` already falls back to the SQLite cache, so the gate must
   read the cache, not the (always-False) session connection.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.yahoo_data_service import YahooDataService

_MY_TEAM = Path(__file__).parent.parent / "pages" / "1_My_Team.py"


def test_connection_status_v1_reflects_session(monkeypatch):
    """Single-user v1: status mirrors the live session connection (unchanged)."""
    monkeypatch.delenv("MULTI_USER", raising=False)
    yds = YahooDataService(yahoo_client=None)
    monkeypatch.setattr(yds, "is_connected", lambda: True)
    assert yds.connection_status() == "connected"
    monkeypatch.setattr(yds, "is_connected", lambda: False)
    assert yds.connection_status() == "offline"


def test_connection_status_multiuser_reflects_data_not_session(monkeypatch):
    """MULTI_USER: a member is never connected, but live scheduler data → 'server'."""
    monkeypatch.setenv("MULTI_USER", "1")
    yds = YahooDataService(yahoo_client=None)
    monkeypatch.setattr(yds, "is_connected", lambda: False)  # members never connect
    monkeypatch.setattr(
        yds,
        "get_data_freshness",
        lambda: {"rosters": "Live (just now)", "standings": "Cached (5m ago)"},
    )
    assert yds.connection_status() == "server"
    monkeypatch.setattr(
        yds,
        "get_data_freshness",
        lambda: {"rosters": "Offline (DB)", "standings": "Offline (DB)"},
    )
    assert yds.connection_status() == "warming"


def test_my_team_matchup_reads_cache_not_session_connection():
    """My Team must not gate the matchup fetch behind is_connected() — members are
    never connected; get_matchup() already falls back to the SQLite cache."""
    src = _MY_TEAM.read_text(encoding="utf-8")
    assert "yds.get_matchup()" in src
    assert "if yds and yds.is_connected():" not in src

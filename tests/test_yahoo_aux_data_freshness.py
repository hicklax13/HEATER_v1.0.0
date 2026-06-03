"""Settings/Schedule/Matchup must be fetched + persisted so the scheduler can
keep them fresh and read-only members can read them (2026-06-03 goal: every
Data Freshness row genuinely Live + fed to pages).

Settings had NO persistence (`_fetch_settings` returned the live dict, db
fallback was `{}`), so under MULTI_USER members got nothing. This adds a JSON
store (modeled on league_matchup_cache) + write-through + DB fallback.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import load_league_settings, save_league_settings
from src.yahoo_data_service import YahooDataService, _get_state_store


def test_league_settings_roundtrip():
    save_league_settings({"categories": ["R", "HR"], "roster_positions": {"C": 1}})
    got = load_league_settings()
    assert got["categories"] == ["R", "HR"]
    assert got["roster_positions"]["C"] == 1


def test_load_settings_empty_when_never_saved_is_safe():
    # Overwrite with empty, then confirm a dict comes back (no crash).
    save_league_settings({})
    assert isinstance(load_league_settings(), dict)


def test_get_settings_serves_db_fallback_when_disconnected():
    """A read-only member (no live client) still gets settings from SQLite."""
    save_league_settings({"sentinel": 42})
    _get_state_store().clear()  # force the db-fallback path, not session cache
    yds = YahooDataService(yahoo_client=None)  # not connected
    assert yds.get_settings().get("sentinel") == 42


def test_refresh_yahoo_aux_logs_settings_and_schedule():
    """The scheduler's aux refresh writes a refresh_log row for settings +
    schedule so the Data Freshness card can report them Live."""
    from unittest.mock import MagicMock

    from src.data_bootstrap import _refresh_yahoo_aux
    from src.database import get_refresh_log_snapshot

    fake_yds = MagicMock()
    fake_yds.get_settings.return_value = {"categories": ["R", "HR"]}
    fake_yds.get_schedule.return_value = {1: "Opp A", 2: "Opp B"}

    fake_yds.sync_all_team_matchups.return_value = 12

    _refresh_yahoo_aux(yahoo_client=None, yds=fake_yds)

    snap = {e["source"]: e for e in get_refresh_log_snapshot()}
    assert snap.get("yahoo_settings", {}).get("status") == "success"
    assert snap.get("yahoo_schedule", {}).get("status") == "success"
    assert snap.get("yahoo_matchup", {}).get("status") == "success"


def test_sync_all_team_matchups_caches_each_team():
    """sync_all_team_matchups writes every team's matchup to the per-team cache
    so read-only members read their own from SQLite."""
    from unittest.mock import MagicMock

    from src.database import load_matchup_cache
    from src.yahoo_data_service import YahooDataService

    fake_client = MagicMock()
    fake_client.is_authenticated = True
    fake_client.get_all_team_matchups.return_value = {
        "Team Hickey": {
            "week": 11,
            "status": "midevent",
            "user_name": "Team Hickey",
            "opp_name": "Rivals",
            "wins": 3,
            "losses": 2,
            "ties": 1,
            "categories": [],
            "user_points": 0,
            "opp_points": 0,
        },
        "Rivals": {
            "week": 11,
            "status": "midevent",
            "user_name": "Rivals",
            "opp_name": "Team Hickey",
            "wins": 2,
            "losses": 3,
            "ties": 1,
            "categories": [],
            "user_points": 0,
            "opp_points": 0,
        },
    }
    yds = YahooDataService(yahoo_client=fake_client)

    n = yds.sync_all_team_matchups()

    assert n == 2
    cached = load_matchup_cache("Rivals", 11)
    assert cached is not None
    assert cached["opp_name"] == "Team Hickey"

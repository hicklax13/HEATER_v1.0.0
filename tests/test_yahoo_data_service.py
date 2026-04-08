"""Tests for YahooDataService — three-tier cache, write-through, fallbacks."""

import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.database as db_mod
from src.database import init_db
from src.yahoo_data_service import (
    DEFAULT_TTL,
    TTLConfig,
    YahooDataService,
    _CacheEntry,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def temp_db():
    """Redirect DB_PATH to a temp file and init schema."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    original = db_mod.DB_PATH
    db_mod.DB_PATH = Path(tmp.name)
    init_db()

    # Seed sample players
    conn = sqlite3.connect(tmp.name)
    conn.execute(
        "INSERT INTO players (player_id, name, team, positions, is_hitter) VALUES (1, 'Aaron Judge', 'NYY', 'OF', 1)"
    )
    conn.execute(
        "INSERT INTO players (player_id, name, team, positions, is_hitter) VALUES (2, 'Shohei Ohtani', 'LAD', 'DH', 1)"
    )
    conn.execute(
        "INSERT INTO players (player_id, name, team, positions, is_hitter) VALUES (3, 'Gerrit Cole', 'NYY', 'SP', 0)"
    )

    # Seed league_rosters
    conn.execute(
        "INSERT INTO league_rosters (team_name, team_index, player_id, roster_slot, is_user_team) "
        "VALUES ('Team Hickey', 0, 1, 'OF', 1)"
    )
    conn.execute(
        "INSERT INTO league_rosters (team_name, team_index, player_id, roster_slot, is_user_team) "
        "VALUES ('Team Hickey', 0, 2, 'Util', 1)"
    )
    conn.execute(
        "INSERT INTO league_rosters (team_name, team_index, player_id, roster_slot, is_user_team) "
        "VALUES ('Opponent A', 1, 3, 'SP', 0)"
    )

    # Seed league_standings
    conn.execute(
        "INSERT INTO league_standings (team_name, category, total, rank) VALUES ('Team Hickey', 'HR', 42.0, 2)"
    )
    conn.execute(
        "INSERT INTO league_standings (team_name, category, total, rank) VALUES ('Team Hickey', 'ERA', 3.45, 5)"
    )
    conn.execute("INSERT INTO league_standings (team_name, category, total, rank) VALUES ('Opponent A', 'HR', 38.0, 4)")
    conn.execute(
        "INSERT INTO league_standings (team_name, category, total, rank) VALUES ('Opponent A', 'ERA', 4.10, 10)"
    )

    # Seed league_teams (user team)
    conn.execute(
        "INSERT INTO league_teams (team_key, team_name, is_user_team) VALUES ('469.l.1.t.1', 'Team Hickey', 1)"
    )
    conn.execute("INSERT INTO league_teams (team_key, team_name, is_user_team) VALUES ('469.l.1.t.2', 'Opponent A', 0)")

    conn.commit()
    conn.close()
    yield tmp.name
    db_mod.DB_PATH = original
    try:
        os.unlink(tmp.name)
    except PermissionError:
        pass


@pytest.fixture(autouse=True)
def isolated_state_store(monkeypatch):
    """Each test gets a fresh dict as its state store.

    Streamlit is installed in this project, so _get_state_store() normally
    returns st.session_state which persists across tests. Patching it
    isolates each test's cache.
    """
    fresh_store: dict = {}
    monkeypatch.setattr(
        "src.yahoo_data_service._get_state_store",
        lambda: fresh_store,
    )
    yield fresh_store


@pytest.fixture
def mock_yahoo_client():
    """Create a mock Yahoo client with standard return values."""
    client = MagicMock()
    type(client).is_authenticated = PropertyMock(return_value=True)

    # Standings: wide-format DataFrame (team_name, team_key, rank, HR, ERA, ...)
    client.get_league_standings.return_value = pd.DataFrame(
        [
            {"team_name": "Team Hickey", "team_key": "469.l.1.t.1", "rank": 1, "HR": 50.0, "ERA": 3.20},
            {"team_name": "Opponent A", "team_key": "469.l.1.t.2", "rank": 5, "HR": 35.0, "ERA": 4.50},
        ]
    )

    # Rosters
    client.get_all_rosters.return_value = pd.DataFrame(
        [
            {"team_name": "Team Hickey", "team_key": "469.l.1.t.1", "player_name": "Aaron Judge", "position": "OF"},
            {"team_name": "Opponent A", "team_key": "469.l.1.t.2", "player_name": "Gerrit Cole", "position": "SP"},
        ]
    )

    # sync_to_db writes to DB and returns counts
    client.sync_to_db.return_value = {"standings": 24, "rosters": 276}

    # Matchup
    client.get_current_matchup.return_value = {
        "week": 2,
        "opponent": "Baty Babies",
        "user_stats": {"HR": 8, "ERA": 3.10},
        "opp_stats": {"HR": 5, "ERA": 4.20},
    }

    # Free agents
    client.get_all_free_agents.return_value = pd.DataFrame(
        [
            {"player_name": "Free Guy", "positions": "OF", "percent_owned": 15.0},
            {"player_name": "Bench Warmer", "positions": "1B", "percent_owned": 2.0},
        ]
    )

    # Transactions
    client.get_league_transactions.return_value = pd.DataFrame(
        [
            {"type": "add/drop", "player_name": "New Pickup", "team": "Team X"},
        ]
    )

    # Settings
    client.get_league_settings.return_value = {
        "num_teams": 12,
        "scoring_type": "head",
        "roster_positions": {"C": 1, "OF": 3},
    }

    # Draft results
    client.get_draft_results.return_value = pd.DataFrame(
        [
            {"player_name": "Aaron Judge", "pick": 1, "round": 1, "team_key": "469.l.1.t.1"},
        ]
    )

    return client


@pytest.fixture
def service(mock_yahoo_client):
    """Create a YahooDataService with mocked client."""
    return YahooDataService(yahoo_client=mock_yahoo_client)


@pytest.fixture
def offline_service():
    """Create a YahooDataService with no client (offline mode)."""
    return YahooDataService(yahoo_client=None)


# ---------------------------------------------------------------------------
# CacheEntry tests
# ---------------------------------------------------------------------------


class TestCacheEntry:
    def test_fresh_entry_is_not_stale(self):
        entry = _CacheEntry(value="data", ttl=60)
        assert not entry.is_stale
        assert entry.age_seconds < 1.0

    def test_expired_entry_is_stale(self):
        entry = _CacheEntry(value="data", ttl=0)
        # TTL=0 means immediately stale
        time.sleep(0.01)
        assert entry.is_stale

    def test_hits_counter(self):
        entry = _CacheEntry(value="data", ttl=60)
        assert entry.hits == 0
        entry.hits += 1
        assert entry.hits == 1


# ---------------------------------------------------------------------------
# TTLConfig tests
# ---------------------------------------------------------------------------


class TestTTLConfig:
    def test_default_values(self):
        cfg = TTLConfig()
        assert cfg.rosters == 1800
        assert cfg.standings == 1800
        assert cfg.matchup == 300
        assert cfg.free_agents == 3600
        assert cfg.transactions == 900
        assert cfg.settings == 86400

    def test_custom_values(self):
        cfg = TTLConfig(rosters=60, matchup=10)
        assert cfg.rosters == 60
        assert cfg.matchup == 10


# ---------------------------------------------------------------------------
# Service: connection state
# ---------------------------------------------------------------------------


class TestConnection:
    def test_connected_with_client(self, service, mock_yahoo_client):
        assert service.is_connected() is True

    def test_disconnected_without_client(self, offline_service):
        assert offline_service.is_connected() is False

    def test_disconnected_when_auth_fails(self, mock_yahoo_client):
        type(mock_yahoo_client).is_authenticated = PropertyMock(return_value=False)
        svc = YahooDataService(yahoo_client=mock_yahoo_client)
        assert svc.is_connected() is False

    def test_disconnected_when_auth_throws(self, mock_yahoo_client):
        type(mock_yahoo_client).is_authenticated = PropertyMock(side_effect=RuntimeError("broken"))
        svc = YahooDataService(yahoo_client=mock_yahoo_client)
        assert svc.is_connected() is False


# ---------------------------------------------------------------------------
# Service: standings
# ---------------------------------------------------------------------------


class TestGetStandings:
    def test_live_standings_from_yahoo(self, service, mock_yahoo_client):
        standings = service.get_standings()
        assert not standings.empty
        assert "team_name" in standings.columns
        mock_yahoo_client.get_league_standings.assert_called_once()

    def test_standings_cached_on_second_call(self, service, mock_yahoo_client):
        service.get_standings()
        service.get_standings()
        # Should only call Yahoo once; second call hits cache
        mock_yahoo_client.get_league_standings.assert_called_once()

    def test_standings_force_refresh_bypasses_cache(self, service, mock_yahoo_client):
        service.get_standings()
        service.get_standings(force_refresh=True)
        assert mock_yahoo_client.get_league_standings.call_count == 2

    def test_standings_fallback_to_db_when_offline(self, offline_service):
        standings = offline_service.get_standings()
        # Should return the seeded DB data
        assert not standings.empty
        assert "Team Hickey" in standings["team_name"].values

    def test_standings_fallback_on_yahoo_error(self, service, mock_yahoo_client):
        mock_yahoo_client.get_league_standings.side_effect = RuntimeError("API down")
        standings = service.get_standings()
        # Should fall back to DB
        assert not standings.empty


# ---------------------------------------------------------------------------
# Service: rosters
# ---------------------------------------------------------------------------


class TestGetRosters:
    def test_live_rosters_from_yahoo(self, service, mock_yahoo_client):
        rosters = service.get_rosters()
        # sync_to_db is called, then load from DB
        mock_yahoo_client.sync_to_db.assert_called_once()
        assert not rosters.empty

    def test_rosters_cached_on_second_call(self, service, mock_yahoo_client):
        service.get_rosters()
        service.get_rosters()
        mock_yahoo_client.sync_to_db.assert_called_once()

    def test_rosters_fallback_when_offline(self, offline_service):
        rosters = offline_service.get_rosters()
        # Should return seeded DB data
        assert not rosters.empty
        assert "Team Hickey" in rosters["team_name"].values


# ---------------------------------------------------------------------------
# Service: matchup
# ---------------------------------------------------------------------------


class TestGetMatchup:
    def test_live_matchup(self, service, mock_yahoo_client):
        matchup = service.get_matchup()
        assert matchup is not None
        assert matchup["week"] == 2
        assert matchup["opponent"] == "Baty Babies"
        mock_yahoo_client.get_current_matchup.assert_called_once()

    def test_matchup_cached(self, service, mock_yahoo_client):
        service.get_matchup()
        service.get_matchup()
        mock_yahoo_client.get_current_matchup.assert_called_once()

    def test_matchup_none_when_offline(self, offline_service):
        result = offline_service.get_matchup()
        assert result is None


# ---------------------------------------------------------------------------
# Service: free agents
# ---------------------------------------------------------------------------


class TestGetFreeAgents:
    def test_live_free_agents(self, service, mock_yahoo_client):
        fa = service.get_free_agents()
        assert not fa.empty
        assert "Free Guy" in fa["player_name"].values
        mock_yahoo_client.get_all_free_agents.assert_called_once()

    def test_free_agents_cached(self, service, mock_yahoo_client):
        service.get_free_agents()
        service.get_free_agents()
        mock_yahoo_client.get_all_free_agents.assert_called_once()


# ---------------------------------------------------------------------------
# Service: transactions
# ---------------------------------------------------------------------------


class TestGetTransactions:
    def test_live_transactions(self, service, mock_yahoo_client):
        txns = service.get_transactions()
        assert not txns.empty
        mock_yahoo_client.get_league_transactions.assert_called_once()

    def test_transactions_empty_when_offline(self, offline_service):
        txns = offline_service.get_transactions()
        assert txns.empty


# ---------------------------------------------------------------------------
# Service: settings
# ---------------------------------------------------------------------------


class TestGetSettings:
    def test_live_settings(self, service, mock_yahoo_client):
        settings = service.get_settings()
        assert settings["num_teams"] == 12
        mock_yahoo_client.get_league_settings.assert_called_once()

    def test_settings_cached(self, service, mock_yahoo_client):
        service.get_settings()
        service.get_settings()
        mock_yahoo_client.get_league_settings.assert_called_once()

    def test_settings_empty_when_offline(self, offline_service):
        settings = offline_service.get_settings()
        assert settings == {}


# ---------------------------------------------------------------------------
# Service: cache invalidation
# ---------------------------------------------------------------------------


class TestCacheInvalidation:
    def test_invalidate_clears_cache(self, service, mock_yahoo_client):
        service.get_standings()
        assert mock_yahoo_client.get_league_standings.call_count == 1
        service.invalidate("standings")
        service.get_standings()
        assert mock_yahoo_client.get_league_standings.call_count == 2

    def test_invalidate_nonexistent_key_is_safe(self, service):
        service.invalidate("nonexistent")  # should not raise


# ---------------------------------------------------------------------------
# Service: force refresh all
# ---------------------------------------------------------------------------


class TestForceRefreshAll:
    def test_refreshes_all_data_types(self, service, mock_yahoo_client):
        results = service.force_refresh_all()
        assert "rosters" in results
        assert "standings" in results
        assert "matchup" in results
        assert results["standings"] == "Refreshed"
        assert results["matchup"] == "Refreshed"


# ---------------------------------------------------------------------------
# Service: data freshness
# ---------------------------------------------------------------------------


class TestDataFreshness:
    def test_freshness_after_fetch(self, service, mock_yahoo_client):
        service.get_standings()
        freshness = service.get_data_freshness()
        assert "standings" in freshness
        assert "Live" in freshness["standings"] or "Cached" in freshness["standings"]

    def test_freshness_offline(self, offline_service):
        freshness = offline_service.get_data_freshness()
        assert freshness["standings"] == "Offline (DB)"


# ---------------------------------------------------------------------------
# Service: cache stats
# ---------------------------------------------------------------------------


class TestCacheStats:
    def test_stats_track_hits_and_misses(self, service, mock_yahoo_client):
        service.get_standings()  # miss
        service.get_standings()  # hit
        service.get_standings()  # hit
        stats = service.get_cache_stats()
        assert stats["total_hits"] >= 2
        assert stats["total_misses"] >= 1
        assert stats["hit_rate"] > 0

    def test_stats_track_errors(self, service, mock_yahoo_client):
        mock_yahoo_client.get_league_transactions.side_effect = RuntimeError("fail")
        service.get_transactions()
        stats = service.get_cache_stats()
        assert "transactions" in stats["errors"]


# ---------------------------------------------------------------------------
# Service: opponent profile
# ---------------------------------------------------------------------------


class TestOpponentProfile:
    def test_live_profile_from_standings(self, service, mock_yahoo_client):
        # Seed enough standings for profile generation
        profile = service.get_opponent_profile("Opponent A")
        assert "tier" in profile
        assert "threat" in profile
        assert "strengths" in profile
        assert "weaknesses" in profile

    def test_fallback_profile_when_offline(self, offline_service):
        profile = offline_service.get_opponent_profile("Unknown Team")
        assert profile["threat"] in ("Unknown", "High", "Medium", "Low", "Minimal")


# ---------------------------------------------------------------------------
# Service: write-through to SQLite
# ---------------------------------------------------------------------------


class TestWriteThrough:
    def test_standings_written_to_db(self, service, mock_yahoo_client):
        """After fetching standings from Yahoo, DB should have updated data."""
        service.get_standings()

        # Verify DB was updated via upsert_league_standing
        from src.database import load_league_standings

        db_standings = load_league_standings()
        # The Yahoo mock returns HR=50 for Team Hickey
        hickey_hr = db_standings[(db_standings["team_name"] == "Team Hickey") & (db_standings["category"] == "HR")]
        assert not hickey_hr.empty
        assert float(hickey_hr.iloc[0]["total"]) == 50.0

    def test_rosters_written_to_db(self, service, mock_yahoo_client):
        """After fetching rosters from Yahoo, DB should have updated data."""
        service.get_rosters()
        # sync_to_db was called, which writes to DB
        mock_yahoo_client.sync_to_db.assert_called_once()


# ---------------------------------------------------------------------------
# Service: _is_valid_data
# ---------------------------------------------------------------------------


class TestIsValidData:
    def test_none_is_invalid(self):
        assert YahooDataService._is_valid_data(None) is False

    def test_empty_df_is_invalid(self):
        assert YahooDataService._is_valid_data(pd.DataFrame()) is False

    def test_nonempty_df_is_valid(self):
        assert YahooDataService._is_valid_data(pd.DataFrame({"a": [1]})) is True

    def test_empty_dict_is_invalid(self):
        assert YahooDataService._is_valid_data({}) is False

    def test_nonempty_dict_is_valid(self):
        assert YahooDataService._is_valid_data({"a": 1}) is True

    def test_empty_list_is_invalid(self):
        assert YahooDataService._is_valid_data([]) is False

    def test_nonempty_list_is_valid(self):
        assert YahooDataService._is_valid_data([1]) is True

    def test_string_is_valid(self):
        assert YahooDataService._is_valid_data("hello") is True


# ---------------------------------------------------------------------------
# Database: league_schedule helpers
# ---------------------------------------------------------------------------


class TestLeagueSchedule:
    def test_upsert_and_load_schedule(self):
        from src.database import load_league_schedule, upsert_league_schedule

        upsert_league_schedule(1, "Team Hickey", "Opponent A")
        upsert_league_schedule(2, "Team Hickey", "Opponent B")

        schedule = load_league_schedule()
        assert schedule[1] == "Opponent A"
        assert schedule[2] == "Opponent B"

    def test_upsert_schedule_replaces(self):
        from src.database import load_league_schedule, upsert_league_schedule

        upsert_league_schedule(1, "Team Hickey", "Opponent A")
        upsert_league_schedule(1, "Team Hickey", "New Opponent")

        schedule = load_league_schedule()
        assert schedule[1] == "New Opponent"

    def test_load_empty_schedule(self):
        from src.database import load_league_schedule

        schedule = load_league_schedule()
        assert isinstance(schedule, dict)


# ---------------------------------------------------------------------------
# Singleton accessor (outside Streamlit)
# ---------------------------------------------------------------------------


class TestGetYahooDataService:
    def test_singleton_creation(self, isolated_state_store):
        from src.yahoo_data_service import get_yahoo_data_service

        svc1 = get_yahoo_data_service()
        svc2 = get_yahoo_data_service()
        assert svc1 is svc2

    def test_singleton_rewires_client(self, isolated_state_store, mock_yahoo_client):
        from src.yahoo_data_service import get_yahoo_data_service

        svc = get_yahoo_data_service()
        assert svc._client is None  # No yahoo_client in store

        # Simulate user connecting Yahoo
        isolated_state_store["yahoo_client"] = mock_yahoo_client
        svc2 = get_yahoo_data_service()
        assert svc2._client is mock_yahoo_client
        assert svc2 is svc  # Still the same singleton

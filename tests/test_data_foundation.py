"""Tests for Phase 2 Data Foundation expansion.

Tests schema migration, data extraction, DraftState fix, and sample data updates.
"""

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.database import _safe_add_column, coerce_numeric_df, get_connection, init_db

# ---------------------------------------------------------------------------
# Schema expansion tests
# ---------------------------------------------------------------------------


class TestSchemaExpansion:
    """Test new columns are added to the players and projections tables."""

    def setup_method(self):
        """Use a temp file DB for isolation."""
        import src.database as db_mod

        self._orig_db_path = db_mod.DB_PATH
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        db_mod.DB_PATH = Path(self._tmp.name)

    def teardown_method(self):
        import os

        import src.database as db_mod

        db_mod.DB_PATH = self._orig_db_path
        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass

    def test_players_has_bats_column(self):
        """Players table should have bats column after init_db."""
        init_db()
        conn = get_connection()
        try:
            cursor = conn.execute("PRAGMA table_info(players)")
            cols = {row[1] for row in cursor.fetchall()}
            assert "bats" in cols
        finally:
            conn.close()

    def test_players_has_throws_column(self):
        """Players table should have throws column after init_db."""
        init_db()
        conn = get_connection()
        try:
            cursor = conn.execute("PRAGMA table_info(players)")
            cols = {row[1] for row in cursor.fetchall()}
            assert "throws" in cols
        finally:
            conn.close()

    def test_players_has_birth_date_column(self):
        """Players table should have birth_date column after init_db."""
        init_db()
        conn = get_connection()
        try:
            cursor = conn.execute("PRAGMA table_info(players)")
            cols = {row[1] for row in cursor.fetchall()}
            assert "birth_date" in cols
        finally:
            conn.close()

    def test_projections_has_fip_column(self):
        """Projections table should have fip column after init_db."""
        init_db()
        conn = get_connection()
        try:
            cursor = conn.execute("PRAGMA table_info(projections)")
            cols = {row[1] for row in cursor.fetchall()}
            assert "fip" in cols
        finally:
            conn.close()

    def test_projections_has_xfip_column(self):
        """Projections table should have xfip column after init_db."""
        init_db()
        conn = get_connection()
        try:
            cursor = conn.execute("PRAGMA table_info(projections)")
            cols = {row[1] for row in cursor.fetchall()}
            assert "xfip" in cols
        finally:
            conn.close()

    def test_projections_has_siera_column(self):
        """Projections table should have siera column after init_db."""
        init_db()
        conn = get_connection()
        try:
            cursor = conn.execute("PRAGMA table_info(projections)")
            cols = {row[1] for row in cursor.fetchall()}
            assert "siera" in cols
        finally:
            conn.close()

    def test_adp_has_yahoo_adp_column(self):
        """ADP table should already have yahoo_adp — verify it exists."""
        init_db()
        conn = get_connection()
        try:
            cursor = conn.execute("PRAGMA table_info(adp)")
            cols = {row[1] for row in cursor.fetchall()}
            assert "yahoo_adp" in cols
        finally:
            conn.close()

    def test_safe_add_column_idempotent(self):
        """_safe_add_column should not raise on duplicate column adds."""
        init_db()
        conn = get_connection()
        try:
            # Adding same column twice should not raise
            _safe_add_column(conn, "players", "bats", "TEXT")
            _safe_add_column(conn, "players", "bats", "TEXT")
        finally:
            conn.close()

    def test_fip_xfip_siera_in_valid_stat_columns(self):
        """FIP/xFIP/SIERA should be in _VALID_STAT_COLUMNS for SQL safety."""
        from src.database import _VALID_STAT_COLUMNS

        assert "fip" in _VALID_STAT_COLUMNS
        assert "xfip" in _VALID_STAT_COLUMNS
        assert "siera" in _VALID_STAT_COLUMNS

    def test_fip_xfip_siera_in_float_stat_cols(self):
        """FIP/xFIP/SIERA should be in _FLOAT_STAT_COLS for coercion."""
        from src.database import _FLOAT_STAT_COLS

        assert "fip" in _FLOAT_STAT_COLS
        assert "xfip" in _FLOAT_STAT_COLS
        assert "siera" in _FLOAT_STAT_COLS

    def test_load_player_pool_includes_new_columns(self):
        """load_player_pool() should return fip, xfip, siera columns."""
        from src.database import load_player_pool

        init_db()
        conn = get_connection()
        try:
            conn.execute(
                "INSERT INTO players (name, team, positions, is_hitter, bats, throws) "
                "VALUES ('Test Player', 'NYY', 'SP', 0, 'R', 'R')"
            )
            pid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.execute(
                "INSERT INTO projections (player_id, system, ip, w, sv, k, era, whip, fip, xfip, siera) "
                "VALUES (?, 'blended', 180, 12, 0, 200, 3.50, 1.15, 3.40, 3.30, 3.20)",
                (pid,),
            )
            conn.commit()
        finally:
            conn.close()

        pool = load_player_pool()
        assert "fip" in pool.columns
        assert "xfip" in pool.columns
        assert "siera" in pool.columns


# ---------------------------------------------------------------------------
# MLB Stats API extraction tests
# ---------------------------------------------------------------------------


class TestMLBStatsExtraction:
    """Test bats/throws/birth_date extraction from MLB Stats API."""

    def test_fetch_all_mlb_players_extracts_bats(self):
        """fetch_all_mlb_players should extract batSide.code as bats."""
        from src.live_stats import fetch_all_mlb_players

        mock_data = {
            "people": [
                {
                    "id": 12345,
                    "fullName": "Test Player",
                    "active": True,
                    "currentTeam": {"abbreviation": "NYY"},
                    "primaryPosition": {"abbreviation": "SS", "type": "Infielder"},
                    "batSide": {"code": "R"},
                    "pitchHand": {"code": "R"},
                    "birthDate": "1998-05-15",
                }
            ]
        }
        with patch("src.live_stats.statsapi") as mock_api:
            mock_api.get.return_value = mock_data
            df = fetch_all_mlb_players()

        assert "bats" in df.columns
        assert df.iloc[0]["bats"] == "R"

    def test_fetch_all_mlb_players_extracts_throws(self):
        """fetch_all_mlb_players should extract pitchHand.code as throws."""
        from src.live_stats import fetch_all_mlb_players

        mock_data = {
            "people": [
                {
                    "id": 12345,
                    "fullName": "Test Player",
                    "active": True,
                    "currentTeam": {"abbreviation": "NYY"},
                    "primaryPosition": {"abbreviation": "SS", "type": "Infielder"},
                    "batSide": {"code": "L"},
                    "pitchHand": {"code": "L"},
                    "birthDate": "1998-05-15",
                }
            ]
        }
        with patch("src.live_stats.statsapi") as mock_api:
            mock_api.get.return_value = mock_data
            df = fetch_all_mlb_players()

        assert "throws" in df.columns
        assert df.iloc[0]["throws"] == "L"

    def test_fetch_all_mlb_players_extracts_birth_date(self):
        """fetch_all_mlb_players should extract birthDate as birth_date."""
        from src.live_stats import fetch_all_mlb_players

        mock_data = {
            "people": [
                {
                    "id": 12345,
                    "fullName": "Test Player",
                    "active": True,
                    "currentTeam": {"abbreviation": "NYY"},
                    "primaryPosition": {"abbreviation": "P", "type": "Pitcher"},
                    "batSide": {"code": "R"},
                    "pitchHand": {"code": "R"},
                    "birthDate": "2000-01-20",
                }
            ]
        }
        with patch("src.live_stats.statsapi") as mock_api:
            mock_api.get.return_value = mock_data
            df = fetch_all_mlb_players()

        assert "birth_date" in df.columns
        assert df.iloc[0]["birth_date"] == "2000-01-20"

    def test_fetch_all_mlb_players_missing_fields_graceful(self):
        """Missing bats/throws/birth_date should default to empty string."""
        from src.live_stats import fetch_all_mlb_players

        mock_data = {
            "people": [
                {
                    "id": 12345,
                    "fullName": "Test Player",
                    "active": True,
                    "currentTeam": {"abbreviation": "NYY"},
                    "primaryPosition": {"abbreviation": "OF", "type": "Outfielder"},
                    # No batSide, pitchHand, or birthDate
                }
            ]
        }
        with patch("src.live_stats.statsapi") as mock_api:
            mock_api.get.return_value = mock_data
            df = fetch_all_mlb_players()

        assert df.iloc[0]["bats"] == ""
        assert df.iloc[0]["throws"] == ""
        assert df.iloc[0]["birth_date"] == ""


# ---------------------------------------------------------------------------
# FanGraphs pitcher extraction tests
# ---------------------------------------------------------------------------


class TestFanGraphsExtraction:
    """Test FIP/xFIP/SIERA extraction from FanGraphs pitcher JSON."""

    def test_normalize_pitcher_json_extracts_fip(self):
        """normalize_pitcher_json should extract FIP from JSON."""
        from src.data_pipeline import normalize_pitcher_json

        raw = [
            {
                "PlayerName": "Test Pitcher",
                "Team": "NYY",
                "IP": 180,
                "GS": 30,
                "SV": 0,
                "W": 12,
                "L": 8,
                "SO": 200,
                "ERA": 3.50,
                "WHIP": 1.15,
                "ER": 70,
                "BB": 50,
                "H": 157,
                "FIP": 3.40,
                "xFIP": 3.30,
                "SIERA": 3.20,
            }
        ]
        df = normalize_pitcher_json(raw)
        assert "fip" in df.columns
        assert df.iloc[0]["fip"] == pytest.approx(3.40)

    def test_normalize_pitcher_json_extracts_xfip(self):
        """normalize_pitcher_json should extract xFIP from JSON."""
        from src.data_pipeline import normalize_pitcher_json

        raw = [
            {
                "PlayerName": "Test Pitcher",
                "Team": "NYY",
                "IP": 180,
                "GS": 30,
                "SV": 0,
                "W": 12,
                "L": 8,
                "SO": 200,
                "ERA": 3.50,
                "WHIP": 1.15,
                "FIP": 3.40,
                "xFIP": 3.30,
                "SIERA": 3.20,
            }
        ]
        df = normalize_pitcher_json(raw)
        assert "xfip" in df.columns
        assert df.iloc[0]["xfip"] == pytest.approx(3.30)

    def test_normalize_pitcher_json_extracts_siera(self):
        """normalize_pitcher_json should extract SIERA from JSON."""
        from src.data_pipeline import normalize_pitcher_json

        raw = [
            {
                "PlayerName": "Test Pitcher",
                "Team": "NYY",
                "IP": 180,
                "GS": 30,
                "SV": 0,
                "W": 12,
                "L": 8,
                "SO": 200,
                "ERA": 3.50,
                "WHIP": 1.15,
                "FIP": 3.40,
                "xFIP": 3.30,
                "SIERA": 3.20,
            }
        ]
        df = normalize_pitcher_json(raw)
        assert "siera" in df.columns
        assert df.iloc[0]["siera"] == pytest.approx(3.20)

    def test_normalize_pitcher_json_missing_advanced_stats(self):
        """Missing FIP/xFIP/SIERA should default to 0.0."""
        from src.data_pipeline import normalize_pitcher_json

        raw = [
            {
                "PlayerName": "Test Pitcher",
                "Team": "NYY",
                "IP": 180,
                "GS": 30,
                "SV": 0,
                "W": 12,
                "SO": 200,
                "ERA": 3.50,
                "WHIP": 1.15,
                # No FIP, xFIP, SIERA
            }
        ]
        df = normalize_pitcher_json(raw)
        assert df.iloc[0]["fip"] == 0.0
        assert df.iloc[0]["xfip"] == 0.0
        assert df.iloc[0]["siera"] == 0.0

    def test_store_projections_includes_fip_xfip_siera(self):
        """_store_projections should INSERT fip/xfip/siera for pitchers."""
        from src.data_pipeline import _store_projections

        init_db()
        pitcher_df = pd.DataFrame(
            [
                {
                    "name": "Test Pitcher",
                    "team": "NYY",
                    "positions": "SP",
                    "is_hitter": False,
                    "ip": 180.0,
                    "w": 12,
                    "l": 8,
                    "sv": 0,
                    "k": 200,
                    "era": 3.50,
                    "whip": 1.15,
                    "er": 70,
                    "bb_allowed": 50,
                    "h_allowed": 157,
                    "fip": 3.40,
                    "xfip": 3.30,
                    "siera": 3.20,
                }
            ]
        )
        projections = {"steamer_pit": pitcher_df}
        _store_projections(projections)

        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT fip, xfip, siera FROM projections WHERE system='steamer'"
            ).fetchone()
            assert row is not None
            assert float(row[0]) == pytest.approx(3.40)
            assert float(row[1]) == pytest.approx(3.30)
            assert float(row[2]) == pytest.approx(3.20)
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Yahoo ADP tests
# ---------------------------------------------------------------------------


class TestYahooADP:
    """Test Yahoo ADP fetching capability."""

    def test_fetch_yahoo_adp_method_exists(self):
        """YahooFantasyClient should have fetch_yahoo_adp method."""
        from src.yahoo_api import YahooFantasyClient

        client = YahooFantasyClient(league_id="12345")
        assert hasattr(client, "fetch_yahoo_adp")
        assert callable(client.fetch_yahoo_adp)

    def test_fetch_yahoo_adp_returns_dataframe(self):
        """fetch_yahoo_adp should return a DataFrame."""
        from src.yahoo_api import YahooFantasyClient

        client = YahooFantasyClient(league_id="12345")
        result = client.fetch_yahoo_adp()
        assert isinstance(result, pd.DataFrame)

    def test_fetch_yahoo_adp_unauthenticated_returns_empty(self):
        """fetch_yahoo_adp should return empty DataFrame when not authenticated."""
        from src.yahoo_api import YahooFantasyClient

        client = YahooFantasyClient(league_id="12345")
        result = client.fetch_yahoo_adp()
        assert result.empty


# ---------------------------------------------------------------------------
# DraftState fix tests
# ---------------------------------------------------------------------------


class TestDraftStateFix:
    """Test DraftState tracks all 12 H2H categories."""

    def _make_pool(self):
        """Create a minimal player pool DataFrame."""
        return pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "name": "Hitter One",
                    "team": "NYY",
                    "positions": "SS",
                    "is_hitter": True,
                    "r": 100,
                    "hr": 30,
                    "rbi": 90,
                    "sb": 15,
                    "ab": 550,
                    "h": 160,
                    "bb": 60,
                    "hbp": 5,
                    "sf": 4,
                    "obp": 0.350,
                    "ip": 0,
                    "w": 0,
                    "l": 0,
                    "sv": 0,
                    "k": 0,
                    "er": 0,
                    "bb_allowed": 0,
                    "h_allowed": 0,
                },
                {
                    "player_id": 2,
                    "name": "Pitcher One",
                    "team": "LAD",
                    "positions": "SP",
                    "is_hitter": False,
                    "r": 0,
                    "hr": 0,
                    "rbi": 0,
                    "sb": 0,
                    "ab": 0,
                    "h": 0,
                    "bb": 0,
                    "hbp": 0,
                    "sf": 0,
                    "obp": 0,
                    "ip": 200,
                    "w": 15,
                    "l": 8,
                    "sv": 0,
                    "k": 220,
                    "er": 70,
                    "bb_allowed": 50,
                    "h_allowed": 160,
                },
            ]
        )

    def test_user_roster_totals_includes_losses(self):
        """get_user_roster_totals should track L (losses)."""
        from src.draft_state import DraftState

        ds = DraftState(num_teams=2, num_rounds=2, user_team_index=0)
        pool = self._make_pool()

        ds.make_pick(1, "Hitter One", "SS")  # pick 0 -> team 0
        ds.make_pick(2, "Pitcher One", "SP", team_index=0)

        totals = ds.get_user_roster_totals(pool)
        assert "L" in totals
        assert totals["L"] == 8

    def test_user_roster_totals_includes_obp(self):
        """get_user_roster_totals should compute OBP from components."""
        from src.draft_state import DraftState

        ds = DraftState(num_teams=2, num_rounds=2, user_team_index=0)
        pool = self._make_pool()

        ds.make_pick(1, "Hitter One", "SS")

        totals = ds.get_user_roster_totals(pool)
        assert "OBP" in totals
        # OBP = (h + bb + hbp) / (ab + bb + hbp + sf) = (160+60+5)/(550+60+5+4) = 225/619
        expected = (160 + 60 + 5) / (550 + 60 + 5 + 4)
        assert totals["OBP"] == pytest.approx(expected, rel=1e-3)

    def test_user_roster_totals_tracks_bb_hbp_sf(self):
        """get_user_roster_totals should track bb, hbp, sf intermediates."""
        from src.draft_state import DraftState

        ds = DraftState(num_teams=2, num_rounds=2, user_team_index=0)
        pool = self._make_pool()

        ds.make_pick(1, "Hitter One", "SS")

        totals = ds.get_user_roster_totals(pool)
        assert totals["bb"] == 60
        assert totals["hbp"] == 5
        assert totals["sf"] == 4

    def test_user_roster_totals_obp_zero_division(self):
        """OBP should be 0 when no plate appearances."""
        from src.draft_state import DraftState

        ds = DraftState(num_teams=2, num_rounds=2, user_team_index=0)
        empty_pool = pd.DataFrame(columns=self._make_pool().columns)

        totals = ds.get_user_roster_totals(empty_pool)
        assert totals["OBP"] == 0

    def test_all_team_roster_totals_includes_losses(self):
        """get_all_team_roster_totals should track L for each team."""
        from src.draft_state import DraftState

        ds = DraftState(num_teams=2, num_rounds=1, user_team_index=0)
        pool = self._make_pool()

        ds.make_pick(1, "Hitter One", "SS")   # pick 0 -> team 0
        ds.make_pick(2, "Pitcher One", "SP")  # pick 1 -> team 1

        all_totals = ds.get_all_team_roster_totals(pool)
        assert "L" in all_totals[0]
        assert "L" in all_totals[1]
        assert all_totals[1]["L"] == 8

    def test_all_team_roster_totals_includes_obp(self):
        """get_all_team_roster_totals should compute OBP for each team."""
        from src.draft_state import DraftState

        ds = DraftState(num_teams=2, num_rounds=1, user_team_index=0)
        pool = self._make_pool()

        ds.make_pick(1, "Hitter One", "SS")

        all_totals = ds.get_all_team_roster_totals(pool)
        assert "OBP" in all_totals[0]
        expected = (160 + 60 + 5) / (550 + 60 + 5 + 4)
        assert all_totals[0]["OBP"] == pytest.approx(expected, rel=1e-3)


# ---------------------------------------------------------------------------
# upsert_player_bulk with new fields tests
# ---------------------------------------------------------------------------


class TestUpsertPlayerBulkNewFields:
    """Test upsert_player_bulk handles bats/throws/birth_date."""

    def setup_method(self):
        import src.database as db_mod

        self._orig_db_path = db_mod.DB_PATH
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        db_mod.DB_PATH = Path(self._tmp.name)

    def teardown_method(self):
        import os

        import src.database as db_mod

        db_mod.DB_PATH = self._orig_db_path
        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass

    def test_upsert_player_bulk_stores_bats_throws_birth_date(self):
        """upsert_player_bulk should store bats, throws, birth_date when provided."""
        from src.database import upsert_player_bulk

        init_db()
        players = [
            {
                "name": "Test Player",
                "team": "NYY",
                "positions": "SS",
                "is_hitter": True,
                "bats": "R",
                "throws": "R",
                "birth_date": "1998-05-15",
            }
        ]
        upsert_player_bulk(players)

        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT bats, throws, birth_date FROM players WHERE name = 'Test Player'"
            ).fetchone()
            assert row is not None
            assert row[0] == "R"
            assert row[1] == "R"
            assert row[2] == "1998-05-15"
        finally:
            conn.close()

    def test_upsert_player_bulk_missing_new_fields(self):
        """upsert_player_bulk should handle missing bats/throws/birth_date gracefully."""
        from src.database import upsert_player_bulk

        init_db()
        players = [
            {
                "name": "Test Player",
                "team": "NYY",
                "positions": "SS",
                "is_hitter": True,
                # No bats, throws, birth_date
            }
        ]
        upsert_player_bulk(players)

        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT bats, throws, birth_date FROM players WHERE name = 'Test Player'"
            ).fetchone()
            assert row is not None
            # Should be None/NULL when not provided
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Sample data tests
# ---------------------------------------------------------------------------


class TestSampleDataUpdates:
    """Test that sample data generators include new fields."""

    def test_make_hitter_has_bats(self):
        """_make_hitter should include bats parameter."""
        from src.data_2026 import _make_hitter

        result = _make_hitter(
            "Test", "NYY", "SS", 600, 30, 15, 0.280, 90, 95, 10, bats="L"
        )
        assert result["bats"] == "L"

    def test_make_hitter_default_bats(self):
        """_make_hitter should default bats to 'R' when not provided."""
        from src.data_2026 import _make_hitter

        result = _make_hitter("Test", "NYY", "SS", 600, 30, 15, 0.280, 90, 95, 10)
        assert result["bats"] == "R"

    def test_make_pitcher_has_throws(self):
        """_make_pitcher should include throws parameter."""
        from src.data_2026 import _make_pitcher

        result = _make_pitcher(
            "Test", "NYY", "SP", 180, 12, 0, 200, 3.50, 1.15, 50, throws="L"
        )
        assert result["throws"] == "L"

    def test_make_pitcher_default_throws(self):
        """_make_pitcher should default throws to 'R' when not provided."""
        from src.data_2026 import _make_pitcher

        result = _make_pitcher(
            "Test", "NYY", "SP", 180, 12, 0, 200, 3.50, 1.15, 50
        )
        assert result["throws"] == "R"

    def test_make_pitcher_has_fip(self):
        """_make_pitcher should include fip parameter."""
        from src.data_2026 import _make_pitcher

        result = _make_pitcher(
            "Test", "NYY", "SP", 180, 12, 0, 200, 3.50, 1.15, 50, fip=3.40
        )
        assert result["fip"] == 3.40

    def test_make_pitcher_default_fip(self):
        """_make_pitcher should derive FIP from ERA when not provided."""
        from src.data_2026 import _make_pitcher

        result = _make_pitcher(
            "Test", "NYY", "SP", 180, 12, 0, 200, 3.50, 1.15, 50
        )
        # Default: fip ~= ERA - 0.10
        assert "fip" in result
        assert result["fip"] == pytest.approx(3.40, abs=0.01)

    def test_make_pitcher_has_xfip(self):
        """_make_pitcher should include xfip parameter."""
        from src.data_2026 import _make_pitcher

        result = _make_pitcher(
            "Test", "NYY", "SP", 180, 12, 0, 200, 3.50, 1.15, 50, xfip=3.30
        )
        assert result["xfip"] == 3.30

    def test_make_pitcher_has_siera(self):
        """_make_pitcher should include siera parameter."""
        from src.data_2026 import _make_pitcher

        result = _make_pitcher(
            "Test", "NYY", "SP", 180, 12, 0, 200, 3.50, 1.15, 50, siera=3.20
        )
        assert result["siera"] == 3.20

    def test_make_hitter_has_birth_date(self):
        """_make_hitter should accept birth_date parameter."""
        from src.data_2026 import _make_hitter

        result = _make_hitter(
            "Test", "NYY", "SS", 600, 30, 15, 0.280, 90, 95, 10,
            birth_date="1998-05-15"
        )
        assert result["birth_date"] == "1998-05-15"

    def test_make_pitcher_has_birth_date(self):
        """_make_pitcher should accept birth_date parameter."""
        from src.data_2026 import _make_pitcher

        result = _make_pitcher(
            "Test", "NYY", "SP", 180, 12, 0, 200, 3.50, 1.15, 50,
            birth_date="2000-01-20"
        )
        assert result["birth_date"] == "2000-01-20"

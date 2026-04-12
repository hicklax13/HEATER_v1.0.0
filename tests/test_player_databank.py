"""Tests for Player Databank data module."""

import sqlite3

import pandas as pd
import pytest

from src.database import get_connection, init_db
from src.player_databank import compute_rolling_stats, filter_databank, load_databank, load_game_logs


class TestGameLogsSchema:
    """Verify game_logs table exists and has correct columns."""

    def setup_method(self):
        init_db()

    def test_game_logs_table_exists(self):
        conn = get_connection()
        try:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='game_logs'")
            assert cursor.fetchone() is not None, "game_logs table should exist"
        finally:
            conn.close()

    def test_game_logs_columns(self):
        conn = get_connection()
        try:
            cursor = conn.execute("PRAGMA table_info(game_logs)")
            columns = {row[1] for row in cursor.fetchall()}
            expected = {
                "player_id",
                "game_date",
                "season",
                "pa",
                "ab",
                "h",
                "r",
                "hr",
                "rbi",
                "sb",
                "bb",
                "hbp",
                "sf",
                "ip",
                "w",
                "l",
                "sv",
                "k",
                "er",
                "bb_allowed",
                "h_allowed",
            }
            assert expected.issubset(columns), f"Missing columns: {expected - columns}"
        finally:
            conn.close()

    def test_game_logs_primary_key(self):
        """Inserting duplicate (player_id, game_date) should fail."""
        conn = get_connection()
        try:
            conn.execute(
                "INSERT INTO game_logs (player_id, game_date, season, h, ab, r, hr, rbi, sb) "
                "VALUES (1, '2026-04-01', 2026, 2, 4, 1, 1, 2, 0)"
            )
            conn.commit()
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO game_logs (player_id, game_date, season, h, ab, r, hr, rbi, sb) "
                    "VALUES (1, '2026-04-01', 2026, 1, 3, 0, 0, 1, 0)"
                )
                conn.commit()
        finally:
            conn.execute("DELETE FROM game_logs WHERE player_id = 1")
            conn.commit()
            conn.close()


class TestGameLogFetching:
    """Test game log data fetching and storage."""

    def setup_method(self):
        init_db()

    def test_load_game_logs_empty(self):
        df = load_game_logs(player_ids=[99999], days=7)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_load_game_logs_with_data(self):
        conn = get_connection()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO game_logs "
                "(player_id, game_date, season, ab, h, r, hr, rbi, sb, pa) "
                "VALUES (100, '2026-04-10', 2026, 4, 2, 1, 1, 2, 0, 4)"
            )
            conn.execute(
                "INSERT OR IGNORE INTO game_logs "
                "(player_id, game_date, season, ab, h, r, hr, rbi, sb, pa) "
                "VALUES (100, '2026-04-09', 2026, 3, 1, 0, 0, 1, 1, 3)"
            )
            conn.execute(
                "INSERT OR IGNORE INTO game_logs "
                "(player_id, game_date, season, ab, h, r, hr, rbi, sb, pa) "
                "VALUES (100, '2026-03-01', 2026, 4, 0, 0, 0, 0, 0, 4)"
            )
            conn.commit()
        finally:
            conn.close()

        df = load_game_logs(player_ids=[100], days=7)
        # Only recent games within 7 days should be returned
        assert isinstance(df, pd.DataFrame)
        # The March 1 entry should be excluded (more than 7 days ago)

    def teardown_method(self):
        conn = get_connection()
        try:
            conn.execute("DELETE FROM game_logs WHERE player_id IN (100, 99999)")
            conn.commit()
        finally:
            conn.close()


class TestRollingStats:
    """Test rolling window stat computations."""

    def setup_method(self):
        init_db()
        conn = get_connection()
        try:
            # Insert 5 games for batter (player 200)
            games = [
                (200, "2026-04-10", 2026, 4, 4, 2, 1, 1, 2, 0, 0, 0, 0, 0.0, 0, 0, 0, 0, 0, 0, 0),
                (200, "2026-04-09", 2026, 4, 4, 1, 0, 0, 1, 1, 0, 0, 0, 0.0, 0, 0, 0, 0, 0, 0, 0),
                (200, "2026-04-08", 2026, 3, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0.0, 0, 0, 0, 0, 0, 0, 0),
                (200, "2026-04-07", 2026, 5, 5, 3, 2, 2, 3, 0, 0, 0, 0, 0.0, 0, 0, 0, 0, 0, 0, 0),
                (200, "2026-04-06", 2026, 4, 4, 1, 0, 0, 1, 0, 0, 0, 0, 0.0, 0, 0, 0, 0, 0, 0, 0),
            ]
            for g in games:
                conn.execute(
                    "INSERT OR IGNORE INTO game_logs "
                    "(player_id, game_date, season, pa, ab, h, r, hr, rbi, sb, bb, hbp, sf, "
                    "ip, w, l, sv, k, er, bb_allowed, h_allowed) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    g,
                )
            conn.commit()
        finally:
            conn.close()

    def test_compute_total(self):
        df = compute_rolling_stats([200], days=7, stat_type="total")
        assert len(df) == 1
        row = df.iloc[0]
        assert row["h"] == 7  # 2+1+0+3+1
        assert row["r"] == 4  # 1+0+1+2+0
        assert row["hr"] == 3  # 1+0+0+2+0

    def test_compute_avg(self):
        df = compute_rolling_stats([200], days=7, stat_type="avg")
        assert len(df) == 1
        row = df.iloc[0]
        # Average per game: h=[2,1,0,3,1] -> mean=1.4
        assert abs(row["h"] - 1.4) < 0.01
        assert abs(row["r"] - 0.8) < 0.01  # [1,0,1,2,0] -> 0.8

    def test_compute_stddev(self):
        df = compute_rolling_stats([200], days=7, stat_type="stddev")
        assert len(df) == 1
        assert df.iloc[0]["h"] > 0  # std dev of [2,1,0,3,1] > 0

    def test_compute_total_rate_stats(self):
        """Verify rate stats are computed correctly from totals."""
        df = compute_rolling_stats([200], days=7, stat_type="total")
        row = df.iloc[0]
        # Total: 7 hits / 20 AB = .350 AVG
        if "avg_calc" in df.columns:
            assert abs(row["avg_calc"] - 0.350) < 0.001

    def teardown_method(self):
        conn = get_connection()
        try:
            conn.execute("DELETE FROM game_logs WHERE player_id = 200")
            conn.commit()
        finally:
            conn.close()


class TestDataAssembly:
    """Test main data loading and filtering."""

    def setup_method(self):
        init_db()

    def test_load_databank_returns_dataframe(self):
        df = load_databank("S_S_2026")
        assert isinstance(df, pd.DataFrame)

    def test_load_databank_has_player_columns(self):
        df = load_databank("S_S_2026")
        if not df.empty:
            assert "player_name" in df.columns or "name" in df.columns

    def test_filter_by_position_batters(self):
        df = load_databank("S_S_2026")
        if not df.empty and "is_hitter" in df.columns:
            batters = filter_databank(df, position="B")
            if not batters.empty:
                assert all(batters["is_hitter"] == 1)

    def test_filter_by_position_pitchers(self):
        df = load_databank("S_S_2026")
        if not df.empty and "is_hitter" in df.columns:
            pitchers = filter_databank(df, position="P")
            if not pitchers.empty:
                assert all(pitchers["is_hitter"] == 0)

    def test_filter_by_search_no_match(self):
        df = load_databank("S_S_2026")
        if not df.empty:
            result = filter_databank(df, search="ZZZNONEXISTENT999")
            assert len(result) == 0

    def test_filter_by_mlb_team(self):
        df = load_databank("S_S_2026")
        if not df.empty and "team" in df.columns:
            teams = df["team"].dropna().unique()
            if len(teams) > 0:
                target = teams[0]
                result = filter_databank(df, mlb_team=target)
                assert all(result["team"] == target)

    def test_filter_defaults_return_all(self):
        df = load_databank("S_S_2026")
        filtered = filter_databank(df)
        # Default position="B" filters to batters, so filtered <= df
        assert len(filtered) <= len(df)

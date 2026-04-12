"""Tests for Player Databank data module."""

import sqlite3

import pandas as pd
import pytest

from src.database import get_connection, init_db


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

"""Test that all database tables are created correctly."""

import sqlite3
import sys
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.database as db_mod
from src.database import init_db, DB_PATH

import pytest


def _table_exists(cursor, table_name):
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    return cursor.fetchone() is not None


def _get_columns(cursor, table_name):
    cursor.execute(f"PRAGMA table_info({table_name})")
    return {row[1] for row in cursor.fetchall()}


@pytest.fixture(autouse=True)
def temp_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    original = db_mod.DB_PATH
    db_mod.DB_PATH = Path(tmp.name)
    yield tmp.name
    db_mod.DB_PATH = original
    try:
        os.unlink(tmp.name)
    except PermissionError:
        pass  # Windows may hold the file lock


def test_new_tables_exist(temp_db):
    """All 6 new tables should be created by init_db()."""
    init_db()
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    for table in [
        "season_stats",
        "ros_projections",
        "league_rosters",
        "league_standings",
        "park_factors",
        "refresh_log",
    ]:
        assert _table_exists(cursor, table), f"Table {table} not found"

    conn.close()


def test_season_stats_columns(temp_db):
    """season_stats table should have the expected columns."""
    init_db()
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    cols = _get_columns(cursor, "season_stats")
    expected = {
        "player_id", "season", "pa", "ab", "h", "r", "hr", "rbi",
        "sb", "avg", "ip", "w", "sv", "k", "era", "whip",
        "er", "bb_allowed", "h_allowed", "games_played", "last_updated",
    }
    assert expected.issubset(cols), f"Missing columns: {expected - cols}"

    conn.close()


def test_league_rosters_columns(temp_db):
    """league_rosters table should have the expected columns."""
    init_db()
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    cols = _get_columns(cursor, "league_rosters")
    expected = {"id", "team_name", "team_index", "player_id", "roster_slot", "is_user_team"}
    assert expected.issubset(cols), f"Missing columns: {expected - cols}"

    conn.close()


def test_refresh_log_columns(temp_db):
    """refresh_log table should have source, last_refresh, status columns."""
    init_db()
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()

    cols = _get_columns(cursor, "refresh_log")
    expected = {"source", "last_refresh", "status"}
    assert expected.issubset(cols), f"Missing columns: {expected - cols}"

    conn.close()

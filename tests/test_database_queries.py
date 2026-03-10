"""Test database query functions for in-season tables."""

import sqlite3
import sys
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.database as db_mod
from src.database import (
    init_db,
    upsert_season_stats,
    upsert_ros_projection,
    upsert_league_roster_entry,
    upsert_league_standing,
    update_refresh_log,
    get_refresh_status,
    load_season_stats,
    load_league_rosters,
    clear_league_rosters,
)

import pytest


@pytest.fixture(autouse=True)
def temp_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    original = db_mod.DB_PATH
    db_mod.DB_PATH = Path(tmp.name)
    init_db()
    yield tmp.name
    db_mod.DB_PATH = original
    try:
        os.unlink(tmp.name)
    except PermissionError:
        pass  # Windows may hold the file lock


def _insert_player(db_path, name="Test Player", team="NYY", positions="SS", is_hitter=1):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO players (name, team, positions, is_hitter) VALUES (?, ?, ?, ?)",
        (name, team, positions, is_hitter),
    )
    conn.commit()
    pid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return pid


def test_upsert_season_stats(temp_db):
    pid = _insert_player(temp_db)
    upsert_season_stats(pid, {"r": 50, "hr": 20, "rbi": 60, "ab": 400, "h": 110})
    df = load_season_stats()
    assert len(df) == 1
    assert df.iloc[0]["r"] == 50
    assert df.iloc[0]["hr"] == 20


def test_upsert_season_stats_updates(temp_db):
    pid = _insert_player(temp_db)
    upsert_season_stats(pid, {"r": 50, "hr": 20})
    upsert_season_stats(pid, {"r": 60, "hr": 25})
    df = load_season_stats()
    assert len(df) == 1
    assert df.iloc[0]["r"] == 60


def test_upsert_ros_projection(temp_db):
    pid = _insert_player(temp_db)
    upsert_ros_projection(pid, "steamer_ros", {"r": 30, "hr": 10})
    conn = sqlite3.connect(temp_db)
    row = conn.execute(
        "SELECT * FROM ros_projections WHERE player_id=?", (pid,)
    ).fetchone()
    conn.close()
    assert row is not None


def test_league_roster_operations(temp_db):
    pid = _insert_player(temp_db)
    upsert_league_roster_entry("Team Hickey", 0, pid, "SS", is_user_team=True)
    df = load_league_rosters()
    assert len(df) == 1
    assert df.iloc[0]["team_name"] == "Team Hickey"
    assert df.iloc[0]["is_user_team"] == 1

    clear_league_rosters()
    df = load_league_rosters()
    assert len(df) == 0


def test_refresh_log(temp_db):
    update_refresh_log("mlb_stats", "success")
    status = get_refresh_status("mlb_stats")
    assert status is not None
    assert status["status"] == "success"
    assert status["last_refresh"] is not None

    status2 = get_refresh_status("nonexistent")
    assert status2 is None

"""SF-19: sprint_speed must be SELECTed into the player pool from statcast_archive."""

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.database as db_mod
from src.database import _load_player_pool_impl, init_db


@pytest.fixture
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
        pass


def _seed(db_path: str, *, with_ros: bool, with_blended: bool) -> int:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO players (name, team, positions, is_hitter) VALUES (?, ?, ?, ?)",
        ("Speedy McTest", "NYY", "OF", 1),
    )
    pid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute(
        "INSERT INTO statcast_archive (player_id, season, sprint_speed) VALUES (?, ?, ?)",
        (pid, 2026, 27.5),
    )
    if with_ros:
        conn.execute(
            "INSERT INTO ros_projections (player_id, system, pa, ab, h, hr, sb) VALUES (?, 'bayes_ros', ?, ?, ?, ?, ?)",
            (pid, 600, 540, 150, 25, 30),
        )
    if with_blended:
        conn.execute(
            "INSERT INTO projections (player_id, system, pa, ab, h, hr, sb) VALUES (?, 'blended', ?, ?, ?, ?, ?)",
            (pid, 600, 540, 150, 25, 30),
        )
    conn.commit()
    conn.close()
    return pid


def test_sprint_speed_present_in_ros_path(temp_db):
    _seed(temp_db, with_ros=True, with_blended=False)
    pool = _load_player_pool_impl()
    assert "sprint_speed" in pool.columns
    row = pool.iloc[0]
    assert float(row["sprint_speed"]) == pytest.approx(27.5)


def test_sprint_speed_present_in_blended_path(temp_db):
    _seed(temp_db, with_ros=False, with_blended=True)
    pool = _load_player_pool_impl()
    assert "sprint_speed" in pool.columns
    row = pool.iloc[0]
    assert float(row["sprint_speed"]) == pytest.approx(27.5)


def test_sprint_speed_present_in_fallback_avg_path(temp_db):
    _seed(temp_db, with_ros=False, with_blended=False)
    conn = sqlite3.connect(temp_db)
    conn.execute(
        "INSERT INTO projections (player_id, system, pa, ab, h, hr, sb) VALUES (?, 'steamer', ?, ?, ?, ?, ?)",
        (1, 600, 540, 150, 25, 30),
    )
    conn.commit()
    conn.close()
    pool = _load_player_pool_impl()
    assert "sprint_speed" in pool.columns
    row = pool.iloc[0]
    assert float(row["sprint_speed"]) == pytest.approx(27.5)

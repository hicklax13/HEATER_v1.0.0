"""SF-20: rate-stat aggregation in projection-blending fallback must be component-weighted.

The fallback (no-blended) path averages projection systems. AVG/OBP/ERA/WHIP
are RATE stats — they must be aggregated as component sums, not simple means
across systems. We force the fallback path with a mock that makes the blended
query return empty.
"""

import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
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


def _force_avg_fallback(func):
    """Decorator: monkeypatch read_sql_query so the blended query returns empty,
    forcing the fallback AVG path to run.
    """
    orig = pd.read_sql_query

    def patched(query, conn_arg, **kw):
        if "AND proj.system = 'blended'" in query:
            return pd.DataFrame()
        return orig(query, conn_arg, **kw)

    def wrapper(*args, **kwargs):
        with patch("src.database.pd.read_sql_query", side_effect=patched):
            return func(*args, **kwargs)

    return wrapper


def _insert_hitter(db_path: str, name: str = "Test Hitter") -> int:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO players (name, team, positions, is_hitter) VALUES (?, ?, ?, ?)",
        (name, "NYY", "OF", 1),
    )
    pid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.commit()
    conn.close()
    return pid


def _insert_pitcher(db_path: str, name: str = "Test Pitcher") -> int:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO players (name, team, positions, is_hitter) VALUES (?, ?, ?, ?)",
        (name, "NYY", "SP", 0),
    )
    pid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.commit()
    conn.close()
    return pid


def _insert_proj(db_path, pid, system, **stats):
    conn = sqlite3.connect(db_path)
    cols = ["player_id", "system"] + list(stats.keys())
    vals = [pid, system] + list(stats.values())
    placeholders = ",".join(["?"] * len(cols))
    conn.execute(f"INSERT INTO projections ({','.join(cols)}) VALUES ({placeholders})", vals)
    conn.commit()
    conn.close()


@_force_avg_fallback
def _load_pool_via_avg_path():
    return _load_player_pool_impl()


def test_avg_uses_component_weighted_sum(temp_db):
    pid = _insert_hitter(temp_db)
    _insert_proj(temp_db, pid, "steamer", h=100, ab=400, bb=40, hbp=5, sf=5)
    _insert_proj(temp_db, pid, "zips", h=50, ab=100, bb=10, hbp=2, sf=1)

    pool = _load_pool_via_avg_path()
    row = pool[pool["player_id"] == pid].iloc[0]

    expected_avg = 150 / 500
    assert float(row["avg"]) == pytest.approx(expected_avg, abs=1e-6)
    simple_mean = ((100 / 400) + (50 / 100)) / 2
    assert abs(float(row["avg"]) - simple_mean) > 0.01


def test_obp_uses_component_weighted_sum(temp_db):
    pid = _insert_hitter(temp_db)
    _insert_proj(temp_db, pid, "steamer", h=100, ab=400, bb=40, hbp=5, sf=5)
    _insert_proj(temp_db, pid, "zips", h=50, ab=100, bb=10, hbp=2, sf=1)

    pool = _load_pool_via_avg_path()
    row = pool[pool["player_id"] == pid].iloc[0]

    expected_obp = (150 + 50 + 7) / (500 + 50 + 7 + 6)
    assert float(row["obp"]) == pytest.approx(expected_obp, abs=1e-6)


def test_era_uses_ip_weighted(temp_db):
    pid = _insert_pitcher(temp_db)
    _insert_proj(temp_db, pid, "steamer", er=70, ip=200.0, bb_allowed=50, h_allowed=180)
    _insert_proj(temp_db, pid, "zips", er=20, ip=20.0, bb_allowed=10, h_allowed=20)

    pool = _load_pool_via_avg_path()
    row = pool[pool["player_id"] == pid].iloc[0]

    expected_era = (70 + 20) * 9 / (200.0 + 20.0)
    assert float(row["era"]) == pytest.approx(expected_era, abs=1e-6)
    simple_mean = ((70 * 9 / 200.0) + (20 * 9 / 20.0)) / 2
    assert abs(float(row["era"]) - simple_mean) > 0.5


def test_whip_uses_ip_weighted(temp_db):
    pid = _insert_pitcher(temp_db)
    _insert_proj(temp_db, pid, "steamer", er=70, ip=200.0, bb_allowed=50, h_allowed=180)
    _insert_proj(temp_db, pid, "zips", er=20, ip=20.0, bb_allowed=10, h_allowed=20)

    pool = _load_pool_via_avg_path()
    row = pool[pool["player_id"] == pid].iloc[0]

    expected_whip = (50 + 10 + 180 + 20) / (200.0 + 20.0)
    assert float(row["whip"]) == pytest.approx(expected_whip, abs=1e-6)


def test_zero_denominator_safe_for_hitter(temp_db):
    pid = _insert_hitter(temp_db, name="No-PA Hitter")
    _insert_proj(temp_db, pid, "steamer", h=0, ab=0, bb=0, hbp=0, sf=0)

    pool = _load_pool_via_avg_path()
    row = pool[pool["player_id"] == pid].iloc[0]

    assert float(row["avg"]) == 0
    assert float(row["obp"]) == 0


def test_zero_ip_pitcher_safe(temp_db):
    pid = _insert_pitcher(temp_db, name="No-IP Pitcher")
    _insert_proj(temp_db, pid, "steamer", er=0, ip=0.0, bb_allowed=0, h_allowed=0)

    pool = _load_pool_via_avg_path()
    row = pool[pool["player_id"] == pid].iloc[0]

    assert float(row["era"]) == 0
    assert float(row["whip"]) == 0

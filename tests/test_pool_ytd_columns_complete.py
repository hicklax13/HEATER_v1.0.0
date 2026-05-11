"""C1 followup: player pool must surface all ytd_* columns needed for weighted rate-stat aggregation.

Mirrors the SF-19 sprint_speed pattern: full init_db() schema + INSERTs into the seeded
schema, then `_load_player_pool_impl()` is called and the resulting DataFrame is asserted
to contain every ytd_* column required for downstream weighted AVG/OBP/ERA/WHIP math
in `pages/1_My_Team.py` (and any other consumer that does the rate-stat aggregation
without an extra round-trip to `load_season_stats`).
"""

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.database as db_mod
from src.database import _load_player_pool_impl, init_db

# Required columns for weighted AVG/OBP/ERA/WHIP aggregation
REQUIRED_YTD_HITTING = [
    "ytd_r",
    "ytd_hr",
    "ytd_rbi",
    "ytd_sb",
    "ytd_avg",
    "ytd_obp",
    "ytd_ab",
    "ytd_h",
    "ytd_bb",
    "ytd_hbp",
    "ytd_sf",
]

REQUIRED_YTD_PITCHING = [
    "ytd_w",
    "ytd_l",
    "ytd_sv",
    "ytd_k",
    "ytd_era",
    "ytd_whip",
    "ytd_ip",
    "ytd_er",
    "ytd_bb_allowed",
    "ytd_h_allowed",
]


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


def _seed_hitter(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO players (name, team, positions, is_hitter) VALUES (?, ?, ?, ?)",
        ("Hitter McTest", "NYY", "OF", 1),
    )
    pid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    # Realistic mid-season hitting line — ensures every ytd_* hitting column has a
    # non-zero value the test can assert against.
    conn.execute(
        """
        INSERT INTO season_stats
            (player_id, season, pa, ab, h, r, hr, rbi, sb, avg, obp,
             bb, hbp, sf, games_played)
        VALUES (?, 2026, 470, 400, 110, 80, 25, 75, 10, 0.275, 0.355,
                50, 5, 3, 120)
        """,
        (pid,),
    )
    # Need at least one projection so the ROS / blended path returns rows
    conn.execute(
        "INSERT INTO projections (player_id, system, pa, ab, h, hr, sb) VALUES (?, 'blended', 600, 540, 150, 28, 12)",
        (pid,),
    )
    conn.commit()
    conn.close()
    return pid


def _seed_pitcher(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO players (name, team, positions, is_hitter) VALUES (?, ?, ?, ?)",
        ("Pitcher McTest", "NYY", "SP", 0),
    )
    pid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    # Realistic pitching line covering every ytd_* pitching column the spec requires.
    conn.execute(
        """
        INSERT INTO season_stats
            (player_id, season, w, l, sv, k, ip, er, era, whip,
             bb_allowed, h_allowed, games_played)
        VALUES (?, 2026, 12, 8, 0, 180, 175.5, 60, 3.08, 1.18,
                50, 130, 28)
        """,
        (pid,),
    )
    conn.execute(
        "INSERT INTO projections (player_id, system, ip, w, k) VALUES (?, 'blended', 200.0, 14, 200)",
        (pid,),
    )
    conn.commit()
    conn.close()
    return pid


def test_pool_has_all_required_ytd_hitting_columns(temp_db):
    _seed_hitter(temp_db)
    pool = _load_player_pool_impl()
    missing = [c for c in REQUIRED_YTD_HITTING if c not in pool.columns]
    assert missing == [], f"Pool missing ytd hitting columns: {missing}"


def test_pool_has_all_required_ytd_pitching_columns(temp_db):
    _seed_pitcher(temp_db)
    pool = _load_player_pool_impl()
    missing = [c for c in REQUIRED_YTD_PITCHING if c not in pool.columns]
    assert missing == [], f"Pool missing ytd pitching columns: {missing}"


def test_pool_ytd_hitting_values_propagate_correctly(temp_db):
    pid = _seed_hitter(temp_db)
    pool = _load_player_pool_impl()
    hitter = pool[pool["player_id"] == pid].iloc[0]
    assert float(hitter.get("ytd_r", 0)) == pytest.approx(80), "ytd_r should be 80 from season_stats"
    assert float(hitter.get("ytd_ab", 0)) == pytest.approx(400), "ytd_ab should be 400 from season_stats"
    assert float(hitter.get("ytd_h", 0)) == pytest.approx(110), "ytd_h should be 110 from season_stats"
    assert float(hitter.get("ytd_bb", 0)) == pytest.approx(50), "ytd_bb should be 50 from season_stats"
    assert float(hitter.get("ytd_hbp", 0)) == pytest.approx(5), "ytd_hbp should be 5 from season_stats"
    assert float(hitter.get("ytd_sf", 0)) == pytest.approx(3), "ytd_sf should be 3 from season_stats"
    assert float(hitter.get("ytd_obp", 0)) == pytest.approx(0.355, abs=1e-3), (
        "ytd_obp should be 0.355 from season_stats"
    )


def test_pool_ytd_pitching_values_propagate_correctly(temp_db):
    pid = _seed_pitcher(temp_db)
    pool = _load_player_pool_impl()
    pitcher = pool[pool["player_id"] == pid].iloc[0]
    assert float(pitcher.get("ytd_w", 0)) == pytest.approx(12), "ytd_w should be 12 from season_stats"
    assert float(pitcher.get("ytd_l", 0)) == pytest.approx(8), "ytd_l should be 8 from season_stats"
    assert float(pitcher.get("ytd_ip", 0)) == pytest.approx(175.5), "ytd_ip should be 175.5 from season_stats"
    assert float(pitcher.get("ytd_er", 0)) == pytest.approx(60), "ytd_er should be 60 from season_stats"
    assert float(pitcher.get("ytd_bb_allowed", 0)) == pytest.approx(50), "ytd_bb_allowed should be 50 from season_stats"
    assert float(pitcher.get("ytd_h_allowed", 0)) == pytest.approx(130), "ytd_h_allowed should be 130 from season_stats"

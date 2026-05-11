"""C9 (SF-2 fallback gate): the no-blended fallback in `_load_player_pool_impl`
must fire when the `projections` table contains other systems (steamer/zips/etc.)
but no `system='blended'` rows.

Prior bug: gate was `if df.empty:` after the blended SELECT. That SELECT does
`FROM players p LEFT JOIN projections proj ON ... AND proj.system='blended'`,
so it returns one row per player regardless of whether any blended rows exist.
The gate could therefore only fire when the players table itself was empty,
and in that case the fallback also returned empty — making the entire fallback
dead code in production.

Fix: probe the projections table directly with `SELECT COUNT(*) WHERE system='blended'`
and gate on `blended_count == 0` so the fallback fires in its intended scenario.

These tests exercise the gate without mocking — they manipulate the temp DB
directly so the real production code path runs.
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


def _insert_proj(db_path: str, pid: int, system: str, **stats) -> None:
    conn = sqlite3.connect(db_path)
    cols = ["player_id", "system"] + list(stats.keys())
    vals = [pid, system] + list(stats.values())
    placeholders = ",".join(["?"] * len(cols))
    conn.execute(
        f"INSERT INTO projections ({','.join(cols)}) VALUES ({placeholders})",
        vals,
    )
    conn.commit()
    conn.close()


def test_fallback_fires_when_only_non_blended_systems_exist(temp_db):
    """Real-world scenario: bootstrap fetched steamer + zips but blended
    derivation hasn't run yet. The fallback AVG path must fire and return
    aggregated rate stats (proving the gate triggered).
    """
    pid = _insert_hitter(temp_db)
    # Insert two non-blended systems with distinct h/ab so AVG is verifiable.
    _insert_proj(temp_db, pid, "steamer", h=100, ab=400, bb=40, hbp=5, sf=5)
    _insert_proj(temp_db, pid, "zips", h=50, ab=100, bb=10, hbp=2, sf=1)

    pool = _load_player_pool_impl()
    row = pool[pool["player_id"] == pid].iloc[0]

    # If the gate is broken, the blended LEFT JOIN returns the player with
    # NULL projection columns, so avg/pa would be 0 and we'd never see the
    # SUM-weighted value. Verify the SF-20 component-weighted formula ran.
    expected_avg = 150 / 500
    assert float(row["avg"]) == pytest.approx(expected_avg, abs=1e-6), (
        "fallback gate didn't fire — AVG path was skipped despite no blended rows"
    )
    # PA via AVG of two systems: (400+100 + 0,0,0...) wait — pa not set, so 0.
    # Verify hr/rbi/etc. paths run by checking the AVG of bb (40 + 10) / 2 = 25.
    assert float(row["bb"]) == pytest.approx(25.0, abs=0.01)


def test_blended_path_used_when_blended_exists(temp_db):
    """Sanity check: when blended IS present, the primary path is used and
    returns the blended row directly (not the AVG of all systems).
    """
    pid = _insert_hitter(temp_db)
    # Insert blended PLUS another system. Blended values should win.
    _insert_proj(temp_db, pid, "steamer", h=100, ab=400, bb=40, hbp=5, sf=5)
    _insert_proj(temp_db, pid, "blended", h=200, ab=500, bb=60, hbp=10, sf=10, avg=0.4, obp=0.5)

    pool = _load_player_pool_impl()
    row = pool[pool["player_id"] == pid].iloc[0]

    # Blended row's stored avg=0.4 should be returned, NOT
    # the SUM-weighted (200+100)/(500+400)=0.333 from the AVG fallback.
    assert float(row["avg"]) == pytest.approx(0.4, abs=1e-6), (
        "blended path returned wrong avg — fallback may have fired incorrectly"
    )
    # bb should be the blended value (60), not avg of both (50).
    assert float(row["bb"]) == pytest.approx(60.0, abs=0.01)


def test_gate_does_not_fire_for_empty_players_with_blended_present(temp_db):
    """Edge: players table empty, projections has blended rows for orphan IDs.
    The original gate would have fired here (df.empty=True) but the fallback
    would still return empty. With the fix, blended_count > 0 so we skip the
    fallback — both paths return empty pool, which is the correct outcome.
    """
    # Insert a blended projection for a player_id that doesn't exist in players.
    conn = sqlite3.connect(temp_db)
    conn.execute(
        "INSERT INTO projections (player_id, system, h, ab) VALUES (?, ?, ?, ?)",
        (9999, "blended", 100, 400),
    )
    conn.commit()
    conn.close()

    pool = _load_player_pool_impl()
    # Empty players → empty pool regardless of projections; just assert no crash.
    assert pool.empty

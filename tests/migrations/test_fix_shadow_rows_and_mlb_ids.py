"""Test migration 2026-05-11-fix-shadow-rows-and-mlb-ids."""

import importlib.util
import sqlite3
from pathlib import Path

import pytest

MIG_PATH = Path(__file__).resolve().parents[2] / "scripts" / "migrations" / "2026-05-11-fix-shadow-rows-and-mlb-ids.py"


@pytest.fixture
def migration():
    """Load the migration module by file path (filename contains a hyphen)."""
    spec = importlib.util.spec_from_file_location("migrate_shadow", MIG_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def populated_db(tmp_path, monkeypatch):
    """Fixture DB mimicking production bug state.

    Players:
    - pid 1100, "Real Twin Test", BAL, 666197 — real twin for shadow pid 108
    - pid 108, "Real Twin Test", MLB, 600770 — shadow WITH a real twin (repointable)
    - pid 109, "Some Orphan", MLB, 600771 — shadow WITHOUT a twin and NOT in KNOWN_SHADOW_REPAIRS
    - pid 110, "Corbin Burnes", MLB, 600770 — shadow WITHOUT a twin but IS in KNOWN_SHADOW_REPAIRS
      (this exercises the new Phase 0 in-place repair path)
    - pid 4643, "Hunter Greene", CIN, NULL — rostered with NULL mlb_id (Phase 1 backfill)
    - pid 999, "Active Player", NYY, 600000000 — normal player, mlb_id outside shadow range
    """
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE players (
            player_id INTEGER PRIMARY KEY,
            name TEXT,
            team TEXT,
            mlb_id INTEGER,
            is_injured INTEGER DEFAULT 0
        );
        CREATE TABLE league_rosters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_name TEXT,
            team_index INTEGER,
            player_id INTEGER,
            roster_slot TEXT,
            is_user_team INTEGER DEFAULT 0,
            status TEXT,
            selected_position TEXT,
            editorial_team_abbr TEXT,
            is_undroppable INTEGER DEFAULT 0
        );
        INSERT INTO players VALUES (1100, 'Real Twin Test', 'BAL', 666197, 0);
        INSERT INTO players VALUES (4643, 'Hunter Greene', 'CIN', NULL, 0);
        INSERT INTO players VALUES (999, 'Active Player', 'NYY', 600000000, 0);
        INSERT INTO players VALUES (108, 'Real Twin Test', 'MLB', 600770, 0);
        INSERT INTO players VALUES (109, 'Some Orphan', 'MLB', 600771, 0);
        INSERT INTO players VALUES (110, 'Corbin Burnes', 'MLB', 600770, 0);
        INSERT INTO league_rosters (team_name, team_index, player_id, status) VALUES
            ('My Team', 0, 108, 'IL15'),
            ('My Team', 0, 109, 'active'),
            ('My Team', 0, 110, 'IL15'),
            ('My Team', 0, 4643, 'active');
    """)
    conn.commit()
    conn.close()

    import src.database as db_mod

    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    return db_path


def test_dry_run_makes_no_changes(populated_db, migration):
    counts = migration.run_migration(dry_run=True)
    conn = sqlite3.connect(populated_db)
    try:
        # Phase 0 candidate (pid 110) should still be a shadow row in DB
        assert conn.execute("SELECT mlb_id FROM players WHERE player_id = 110").fetchone()[0] == 600770
        # Phase 2 shadow rows still in DB
        assert conn.execute("SELECT COUNT(*) FROM players WHERE player_id = 108").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM players WHERE player_id = 109").fetchone()[0] == 1
        # Phase 1 NULL mlb_id still NULL
        assert conn.execute("SELECT mlb_id FROM players WHERE player_id = 4643").fetchone()[0] is None
        # league_rosters refs still point at the shadow pids
        assert conn.execute("SELECT COUNT(*) FROM league_rosters WHERE player_id = 108").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM league_rosters WHERE player_id = 110").fetchone()[0] == 1
    finally:
        conn.close()
    # Phase 0 finds pid 110 in KNOWN_SHADOW_REPAIRS (Burnes); pids 130/164/187 not in fixture
    assert counts["rostered_shadows_repaired"] == 1
    # Phase 2 explicitly filters out Phase-0-repaired pids (108 + 109 remain; 110 excluded)
    assert counts["shadow_rows_found"] == 2
    assert counts["null_mlb_backfilled"] == 1
    # Phase 2 repoint: pid 108 → pid 1100 (twin); pid 109 has no twin
    assert counts["league_rosters_repointed"] == 1
    # Phase 3 delete: pid 108 deletable (refs repointed); pid 109 has refs and no twin → kept
    assert counts["shadow_rows_deleted"] == 1


def test_commit_repoints_and_deletes(populated_db, migration):
    counts = migration.run_migration(dry_run=False)

    conn = sqlite3.connect(populated_db)
    try:
        # Phase 0: pid 110 (Burnes) was repaired in place — row still exists, mlb_id updated
        assert conn.execute("SELECT COUNT(*) FROM players WHERE player_id = 110").fetchone()[0] == 1
        assert conn.execute("SELECT mlb_id FROM players WHERE player_id = 110").fetchone()[0] == 669203
        # Phase 1: Hunter Greene mlb_id backfilled
        assert conn.execute("SELECT mlb_id FROM players WHERE player_id = 4643").fetchone()[0] == 668881
        # Phase 2 + 3: pid 108 (shadow with twin) deleted, refs repointed to pid 1100
        assert conn.execute("SELECT COUNT(*) FROM players WHERE player_id = 108").fetchone()[0] == 0
        # pid 109 (shadow without twin, not in KNOWN_SHADOW_REPAIRS) still alive (refs intact)
        assert conn.execute("SELECT COUNT(*) FROM players WHERE player_id = 109").fetchone()[0] == 1
        # league_rosters refs
        rows = conn.execute("SELECT player_id FROM league_rosters ORDER BY id").fetchall()
        assert (1100,) in rows, f"Expected league_rosters now references real twin pid=1100; got {rows}"
        assert (108,) not in rows, f"Expected pid=108 reference removed; got {rows}"
        # pid 110 ref preserved (Phase 0 repaired the row in place — no repoint needed)
        assert (110,) in rows, f"Expected pid=110 ref preserved (Phase 0 repaired in place); got {rows}"
    finally:
        conn.close()

    assert counts["rostered_shadows_repaired"] == 1
    assert counts["null_mlb_backfilled"] == 1
    assert counts["league_rosters_repointed"] == 1
    assert counts["shadow_rows_deleted"] == 1


def test_commit_is_idempotent(populated_db, migration):
    migration.run_migration(dry_run=False)
    counts_second = migration.run_migration(dry_run=False)
    assert counts_second["rostered_shadows_repaired"] == 0
    assert counts_second["null_mlb_backfilled"] == 0
    assert counts_second["league_rosters_repointed"] == 0
    assert counts_second["shadow_rows_deleted"] == 0

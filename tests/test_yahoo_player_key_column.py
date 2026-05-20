"""SFH M4 guard (2026-05-20): league_rosters.yahoo_player_key column +
TWP roster-entity distinction.

Background:
  Yahoo treats two-way players (Ohtani) as TWO separate entities — one
  with position=Pitcher (yahoo player_id="10480"-ish) and one with
  position=Batter (different yahoo player_id) — both mapping to the
  same HEATER ``players.player_id`` via match_player_id. Without
  recording which Yahoo entity a roster row points at, downstream
  queries can't tell whether "team X has Ohtani" means the pitcher
  half or the hitter half, and stats aggregation routes the wrong
  half of the player's production.

  This is real today: Ohtani (player_id=2) is rostered on Over the
  Rembow (as P slot) AND on Baty Babies (as Util slot) — two different
  Yahoo entities, same HEATER player_id. The data integrity check
  ``SELECT player_id, COUNT(DISTINCT team_name) ... HAVING COUNT > 1``
  flags this as a duplication, but it's the genuine Yahoo state.

This file pins:
  1. The yahoo_player_key column exists post-init_db().
  2. upsert_league_roster_entry writes the value.
  3. Two roster rows on the SAME team with the SAME player_id but
     different yahoo_player_key still collide on UNIQUE(team_name,
     player_id) — the schema preserves "one slot per player per team"
     even for TWPs. The yahoo_player_key just records WHICH Yahoo
     entity is being rostered.
  4. Two roster rows on DIFFERENT teams with the SAME player_id but
     different yahoo_player_key BOTH persist (the Ohtani case).
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.database as db_mod
from src.database import init_db, upsert_league_roster_entry


@pytest.fixture(autouse=True)
def temp_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    original = db_mod.DB_PATH
    db_mod.DB_PATH = Path(tmp.name)
    init_db()
    # Seed Ohtani-like TWP player.
    conn = sqlite3.connect(tmp.name)
    conn.execute(
        "INSERT INTO players (player_id, mlb_id, name, team, positions, is_hitter) "
        "VALUES (2, 660271, 'Shohei Ohtani', 'LAD', 'DH,SP,TWP', 1)"
    )
    conn.commit()
    conn.close()
    yield tmp.name
    db_mod.DB_PATH = original
    try:
        os.unlink(tmp.name)
    except PermissionError:
        pass


def _query_roster_rows(temp_db, **filters):
    """Return list of dict rows from league_rosters matching the filters."""
    conn = sqlite3.connect(temp_db)
    conn.row_factory = sqlite3.Row
    try:
        where = " AND ".join(f"{k} = ?" for k in filters)
        sql = f"SELECT * FROM league_rosters{' WHERE ' + where if where else ''}"
        return [dict(r) for r in conn.execute(sql, tuple(filters.values()))]
    finally:
        conn.close()


def test_yahoo_player_key_column_exists_after_init(temp_db):
    """SFH M4: the column must exist after init_db so legacy DBs migrate."""
    conn = sqlite3.connect(temp_db)
    try:
        cur = conn.execute("PRAGMA table_info(league_rosters)")
        cols = {row[1] for row in cur.fetchall()}
    finally:
        conn.close()
    assert "yahoo_player_key" in cols, (
        "SFH M4 migration missing — yahoo_player_key column must be added by init_db "
        "via _safe_add_column. Without it, two-way players can't be distinguished."
    )


def test_upsert_writes_yahoo_player_key(temp_db):
    """SFH M4: the column gets populated when caller passes yahoo_player_key."""
    upsert_league_roster_entry(
        team_name="Test Team",
        team_index=0,
        player_id=2,
        roster_slot="SP,P",
        yahoo_player_key="10480",
    )
    rows = _query_roster_rows(temp_db, team_name="Test Team")
    assert len(rows) == 1
    assert rows[0]["yahoo_player_key"] == "10480"


def test_upsert_default_yahoo_player_key_empty_string(temp_db):
    """SFH M4 back-compat: legacy callers that don't pass yahoo_player_key
    still work — the column defaults to empty string (NOT None / NULL),
    matching the _safe_add_column DEFAULT."""
    upsert_league_roster_entry(
        team_name="Legacy Team",
        team_index=0,
        player_id=2,
        roster_slot="DH",
    )
    rows = _query_roster_rows(temp_db, team_name="Legacy Team")
    assert len(rows) == 1
    assert rows[0]["yahoo_player_key"] == ""


def test_twp_on_different_teams_both_persist(temp_db):
    """SFH M4 main scenario: Ohtani-Pitcher (Yahoo key A) on Team X AND
    Ohtani-Batter (Yahoo key B) on Team Y. Both player_id=2, but on
    different teams + different yahoo_player_keys — both rows persist.
    This is the real Yahoo state we observed in the FourzynBurn league
    for 2026 Ohtani."""
    upsert_league_roster_entry(
        team_name="Over the Rembow",
        team_index=5,
        player_id=2,
        roster_slot="SP,P",
        selected_position="P",
        yahoo_player_key="10480",  # Ohtani-Pitcher Yahoo entity
    )
    upsert_league_roster_entry(
        team_name="Baty Babies",
        team_index=9,
        player_id=2,
        roster_slot="Util",
        selected_position="Util",
        yahoo_player_key="10481",  # Ohtani-Batter Yahoo entity
    )

    pitcher_rows = _query_roster_rows(temp_db, team_name="Over the Rembow", player_id=2)
    batter_rows = _query_roster_rows(temp_db, team_name="Baty Babies", player_id=2)
    assert len(pitcher_rows) == 1
    assert len(batter_rows) == 1
    assert pitcher_rows[0]["yahoo_player_key"] == "10480"
    assert batter_rows[0]["yahoo_player_key"] == "10481"
    assert pitcher_rows[0]["selected_position"] == "P"
    assert batter_rows[0]["selected_position"] == "Util"


def test_upsert_same_team_same_player_id_replaces_not_duplicates(temp_db):
    """SFH M4: the UNIQUE(team_name, player_id) constraint is unchanged.
    If the SAME team somehow rosters both Yahoo entities of a TWP, only
    one row survives — the second call updates the first via ON CONFLICT.
    Documenting this behavior so future TWP edge cases aren't surprising.
    (In practice Yahoo's UI won't let a single team roster both halves,
    so this collapses to the most-recently-synced entity.)"""
    upsert_league_roster_entry(
        team_name="Same Team",
        team_index=0,
        player_id=2,
        roster_slot="SP",
        yahoo_player_key="10480",
    )
    upsert_league_roster_entry(
        team_name="Same Team",
        team_index=0,
        player_id=2,
        roster_slot="Util",
        yahoo_player_key="10481",
    )
    rows = _query_roster_rows(temp_db, team_name="Same Team", player_id=2)
    assert len(rows) == 1, "UNIQUE(team_name, player_id) prevents same-team duplicate"
    # Second call wins (ON CONFLICT DO UPDATE).
    assert rows[0]["roster_slot"] == "Util"
    assert rows[0]["yahoo_player_key"] == "10481"


def test_unique_constraint_still_intact(temp_db):
    """SFH M4 regression guard: adding yahoo_player_key did NOT remove the
    UNIQUE(team_name, player_id) constraint. A direct INSERT bypassing
    ON CONFLICT must still error."""
    upsert_league_roster_entry(
        team_name="Constraint Team",
        team_index=0,
        player_id=2,
        yahoo_player_key="10480",
    )
    conn = sqlite3.connect(temp_db)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO league_rosters (team_name, team_index, player_id, yahoo_player_key) VALUES (?, ?, ?, ?)",
                ("Constraint Team", 0, 2, "10481"),
            )
    finally:
        conn.close()

"""DNA-collision guard (2026-05-21 post-PR-#110 follow-up):

Two MLB players sharing a name but with DIFFERENT mlb_ids are DIFFERENT
players. The bootstrap pipeline must NOT collapse them into a single row.

Root cause history:
  - 2026-04-08: original Muncy DNA bug (LAD veteran's stats served as ATH
    rookie's). Fixed by inserting a second players row via
    scripts/migrate_muncy_dna_2026_05_21.py.
  - 2026-05-21 follow-up: migration's row gets DELETEd on every
    bootstrap_all_data(force=True) because:
      1. upsert_player_bulk did name-only SELECT-first existence check
         (src/database.py:2796) — collapsed both Muncys into one row.
      2. deduplicate_players grouped by name only (src/database.py:2877),
         then merged them, deleting the second row.

This guard pins both behaviors:
  - upsert_player_bulk MUST treat rows with different mlb_ids as different
    players (insert new rows, not UPDATE the wrong one).
  - deduplicate_players MUST preserve rows with different non-null mlb_ids
    even when names match.
  - Backward compat: same name + same mlb_id (or both NULL) STILL merges,
    so existing dedup behavior for legitimate duplicates is preserved.
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
from src.database import (
    deduplicate_players,
    get_connection,
    init_db,
    upsert_player_bulk,
)


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
        pass


# ---------------------------------------------------------------------------
# Test 1: upsert_player_bulk preserves same-name-different-mlb_id rows
# (The MLB Stats API legitimately returns both LAD Muncy and ATH Muncy in
# the same fetch — both must end up in the players table.)
# ---------------------------------------------------------------------------


def test_upsert_preserves_distinct_mlb_ids():
    """Feeding upsert_player_bulk both LAD Muncy (mlb_id=571970) and ATH
    Muncy (mlb_id=691777) must produce TWO rows in the players table.
    The legacy name-only SELECT-first collapses them into one — that's
    the bug this guard pins against."""
    upsert_player_bulk(
        [
            {
                "name": "Max Muncy",
                "team": "ATH",
                "positions": "3B",
                "is_hitter": 1,
                "mlb_id": 691777,
            },
            {
                "name": "Max Muncy",
                "team": "LAD",
                "positions": "3B,2B,1B",
                "is_hitter": 1,
                "mlb_id": 571970,
            },
        ]
    )

    conn = get_connection()
    rows = conn.execute("SELECT player_id, team, mlb_id FROM players WHERE name='Max Muncy' ORDER BY mlb_id").fetchall()
    conn.close()

    assert len(rows) == 2, (
        f"upsert_player_bulk collapsed two distinct Max Muncys into {len(rows)} row(s). "
        f"Same-name rows with different mlb_ids must be preserved as separate players."
    )
    teams = {r[1] for r in rows}
    mlb_ids = {r[2] for r in rows}
    assert teams == {"ATH", "LAD"}, f"Expected teams ATH+LAD, got {teams}"
    assert mlb_ids == {571970, 691777}, f"Expected mlb_ids 571970+691777, got {mlb_ids}"


# ---------------------------------------------------------------------------
# Test 2: deduplicate_players does NOT merge different mlb_ids
# ---------------------------------------------------------------------------


def test_dedup_preserves_distinct_mlb_ids():
    """When players contains two rows with the same name but different
    non-null mlb_ids, deduplicate_players must leave both intact. They
    are different MLB players who happen to share a name."""
    conn = get_connection()
    conn.execute(
        "INSERT INTO players (name, team, positions, is_hitter, mlb_id) VALUES (?, ?, ?, ?, ?)",
        ("Max Muncy", "ATH", "3B", 1, 691777),
    )
    conn.execute(
        "INSERT INTO players (name, team, positions, is_hitter, mlb_id) VALUES (?, ?, ?, ?, ?)",
        ("Max Muncy", "LAD", "3B,2B,1B", 1, 571970),
    )
    conn.commit()
    conn.close()

    result = deduplicate_players()

    # Must NOT report a merge — these are different players.
    assert result["players_merged"] == 0, (
        f"deduplicate_players merged two DIFFERENT MLB players sharing a name. "
        f"merged={result['players_merged']} — expected 0."
    )

    conn = get_connection()
    rows = conn.execute("SELECT player_id, team, mlb_id FROM players WHERE name='Max Muncy' ORDER BY mlb_id").fetchall()
    conn.close()

    assert len(rows) == 2, f"Expected 2 Muncy rows after dedup, got {len(rows)}: {rows}"
    teams = {r[1] for r in rows}
    mlb_ids = {r[2] for r in rows}
    assert teams == {"ATH", "LAD"}
    assert mlb_ids == {571970, 691777}


# ---------------------------------------------------------------------------
# Test 3: backward-compat — same name + same mlb_id still merges
# (Two records of the same player from different sources — legitimate dup.)
# ---------------------------------------------------------------------------


def test_dedup_still_merges_same_mlb_id():
    """Same name + same mlb_id is a legitimate duplicate (e.g. one row from
    FanGraphs and another from MLB Stats API, both for the same player).
    Existing dedup behavior must be preserved."""
    conn = get_connection()
    conn.execute(
        "INSERT INTO players (name, team, positions, is_hitter, mlb_id) VALUES (?, ?, ?, ?, ?)",
        ("Aaron Judge", "NYY", "OF", 1, 592450),
    )
    conn.execute(
        "INSERT INTO players (name, team, positions, is_hitter, mlb_id) VALUES (?, ?, ?, ?, ?)",
        ("Aaron Judge", "NYY", "OF,DH", 1, 592450),  # same mlb_id, slightly different positions
    )
    conn.commit()
    conn.close()

    result = deduplicate_players()

    assert result["players_merged"] == 1, (
        f"Expected dedup to merge same-mlb_id duplicate, merged={result['players_merged']}"
    )
    conn = get_connection()
    rows = conn.execute("SELECT player_id, positions FROM players WHERE name='Aaron Judge'").fetchall()
    conn.close()
    assert len(rows) == 1, f"Expected 1 row after dedup, got {len(rows)}"
    assert "DH" in rows[0][1] and "OF" in rows[0][1], f"Positions should be merged: {rows[0][1]}"


# ---------------------------------------------------------------------------
# Test 4: backward-compat — same name + both NULL mlb_id still merges
# (Legacy / seed data without mlb_ids — preserve existing behavior.)
# ---------------------------------------------------------------------------


def test_dedup_still_merges_null_mlb_ids():
    """Legacy rows with NULL mlb_id are unidentifiable beyond name. For
    backward compat with seed data, same name + both NULL still merges."""
    conn = get_connection()
    conn.execute(
        "INSERT INTO players (name, team, positions, is_hitter) VALUES (?, ?, ?, ?)",
        ("Corey Seager", "TEX", "SS", 1),
    )
    conn.execute(
        "INSERT INTO players (name, team, positions, is_hitter) VALUES (?, ?, ?, ?)",
        ("Corey Seager", "MLB", "SS,DH", 1),
    )
    conn.commit()
    conn.close()

    result = deduplicate_players()
    assert result["players_merged"] == 1, f"Expected NULL-mlb_id duplicate to merge, merged={result['players_merged']}"


# ---------------------------------------------------------------------------
# Test 5: backward-compat — name match + one NULL + one real mlb_id MERGES
# (Data enrichment case: a seed row gets enriched with MLB API mlb_id.)
# ---------------------------------------------------------------------------


def test_dedup_merges_null_into_enriched():
    """A legacy row with NULL mlb_id and a newer row with the same name +
    real mlb_id should still merge — this is data enrichment, not a
    DNA collision. The real-mlb_id row is canonical."""
    conn = get_connection()
    conn.execute(
        "INSERT INTO players (name, team, positions, is_hitter) VALUES (?, ?, ?, ?)",
        ("Shohei Ohtani", "LAA", "P,DH", 0),  # legacy, NULL mlb_id
    )
    conn.execute(
        "INSERT INTO players (name, team, positions, is_hitter, mlb_id) VALUES (?, ?, ?, ?, ?)",
        ("Shohei Ohtani", "LAD", "P,DH", 0, 660271),  # enriched
    )
    conn.commit()
    conn.close()

    result = deduplicate_players()
    assert result["players_merged"] == 1, (
        f"Expected NULL+real-mlb_id to merge as enrichment, merged={result['players_merged']}"
    )
    conn = get_connection()
    rows = conn.execute("SELECT mlb_id FROM players WHERE name='Shohei Ohtani'").fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0][0] == 660271, f"Canonical should retain real mlb_id, got {rows[0][0]}"


# ---------------------------------------------------------------------------
# Test 6: full round-trip — upsert both Muncys, dedup, match_player_id
# resolves Yahoo's "Max Muncy LAD" to the LAD row, not ATH.
# ---------------------------------------------------------------------------


def test_round_trip_upsert_dedup_match():
    """End-to-end: MLB API returns both Muncys → upsert preserves both →
    dedup preserves both → match_player_id with Yahoo team='LAD' returns
    the LAD Muncy row."""
    upsert_player_bulk(
        [
            {
                "name": "Max Muncy",
                "team": "ATH",
                "positions": "3B",
                "is_hitter": 1,
                "mlb_id": 691777,
            },
            {
                "name": "Max Muncy",
                "team": "LAD",
                "positions": "3B,2B,1B",
                "is_hitter": 1,
                "mlb_id": 571970,
            },
        ]
    )
    deduplicate_players()

    from src.live_stats import match_player_id

    pid_lad = match_player_id("Max Muncy", "LAD")
    pid_ath = match_player_id("Max Muncy", "ATH")

    conn = get_connection()
    lad_team = conn.execute("SELECT team FROM players WHERE player_id=?", (pid_lad,)).fetchone()[0]
    ath_team = conn.execute("SELECT team FROM players WHERE player_id=?", (pid_ath,)).fetchone()[0]
    conn.close()

    assert lad_team == "LAD", f"Yahoo LAD must resolve to LAD row, got team={lad_team}"
    assert ath_team == "ATH", f"Yahoo ATH must resolve to ATH row, got team={ath_team}"
    assert pid_lad != pid_ath, "LAD and ATH Muncys must have distinct player_ids"


# ---------------------------------------------------------------------------
# Test 7: upsert preserves existing rows when re-running with same mlb_ids
# (Idempotency under multiple bootstrap cycles.)
# ---------------------------------------------------------------------------


def test_upsert_idempotent_across_bootstraps():
    """Running upsert_player_bulk twice with the same payload must not
    duplicate rows or destroy data. Bootstrap is supposed to be safe to
    re-run."""
    payload = [
        {
            "name": "Max Muncy",
            "team": "ATH",
            "positions": "3B",
            "is_hitter": 1,
            "mlb_id": 691777,
        },
        {
            "name": "Max Muncy",
            "team": "LAD",
            "positions": "3B,2B,1B",
            "is_hitter": 1,
            "mlb_id": 571970,
        },
    ]
    upsert_player_bulk(payload)
    upsert_player_bulk(payload)  # second cycle

    conn = get_connection()
    rows = conn.execute("SELECT mlb_id, team FROM players WHERE name='Max Muncy' ORDER BY mlb_id").fetchall()
    conn.close()

    assert len(rows) == 2, f"After 2 upsert cycles, expected 2 rows, got {len(rows)}: {rows}"
    assert {r[0] for r in rows} == {571970, 691777}
    assert {r[1] for r in rows} == {"ATH", "LAD"}

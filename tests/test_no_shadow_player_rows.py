"""Permanent guard against BUG-001 regression: no shadow rows in players table.

A "shadow row" is defined as a player with team='MLB' and mlb_id in the fake
range used by the pre-2026-05-11 mass-bootstrap. If this test fails in CI,
it means new shadow rows were inserted — investigate the inserting code path.
"""

import os
import sqlite3

import pytest

DB_PATH = os.environ.get(
    "HEATER_DB_PATH",
    "C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0/data/draft_tool.db",
)
SHADOW_RANGE = (600000, 601999)


@pytest.mark.skipif(not os.path.isfile(DB_PATH), reason="live DB not present (CI/dev env)")
def test_no_shadow_rows_in_players():
    conn = sqlite3.connect(DB_PATH)
    try:
        shadow = conn.execute(
            "SELECT COUNT(*) FROM players WHERE team='MLB' AND mlb_id BETWEEN ? AND ?",
            SHADOW_RANGE,
        ).fetchone()[0]
    finally:
        conn.close()
    assert shadow == 0, (
        f"BUG-001 regression: {shadow} shadow player rows found with team='MLB' "
        f"and mlb_id in {SHADOW_RANGE}. These will cause live-stats pipelines to "
        "fetch DSL/VSL prospect stats into real-player rows. "
        "Run scripts/migrations/2026-05-11-fix-shadow-rows-and-mlb-ids.py --dry-run."
    )


@pytest.mark.skipif(not os.path.isfile(DB_PATH), reason="live DB not present")
def test_no_rostered_null_mlb_ids():
    conn = sqlite3.connect(DB_PATH)
    try:
        null_count = conn.execute(
            """SELECT COUNT(*) FROM players p
               JOIN league_rosters lr ON p.player_id = lr.player_id
               WHERE p.mlb_id IS NULL"""
        ).fetchone()[0]
        if null_count:
            offenders = conn.execute(
                """SELECT p.player_id, p.name FROM players p
                   JOIN league_rosters lr ON p.player_id = lr.player_id
                   WHERE p.mlb_id IS NULL LIMIT 10"""
            ).fetchall()
        else:
            offenders = []
    finally:
        conn.close()
    assert null_count == 0, (
        f"BUG-002 regression: {null_count} rostered players have NULL mlb_id "
        f"(invisible to live-stats pipelines). Sample: {offenders}. "
        "Add them to KNOWN_NULL_MLB_BACKFILL in scripts/migrations/2026-05-11-fix-shadow-rows-and-mlb-ids.py "
        "and re-run."
    )

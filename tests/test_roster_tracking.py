"""Tests for opponent roster change tracking."""

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.database as db_mod
from src.database import get_roster_changes, init_db, snapshot_league_rosters


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


def test_snapshot_creates_entries():
    conn = sqlite3.connect(str(db_mod.DB_PATH))
    conn.execute(
        "INSERT INTO league_rosters (team_name, team_index, player_id, roster_slot, is_user_team) "
        "VALUES ('Team A', 0, 1, 'OF', 0)"
    )
    conn.commit()
    conn.close()
    count = snapshot_league_rosters()
    assert count >= 1


def test_detects_added_player():
    conn = sqlite3.connect(str(db_mod.DB_PATH))
    conn.execute(
        "INSERT INTO league_rosters (team_name, team_index, player_id, roster_slot, is_user_team) "
        "VALUES ('Team A', 0, 1, 'OF', 0)"
    )
    conn.commit()
    conn.close()
    snapshot_league_rosters()

    # Simulate next day by inserting snapshot with different date
    from datetime import UTC, datetime, timedelta

    yesterday = (datetime.now(UTC) - timedelta(days=1)).strftime("%Y-%m-%d")
    conn = sqlite3.connect(str(db_mod.DB_PATH))
    # Move existing snapshot to yesterday
    conn.execute("UPDATE roster_snapshots SET snapshot_date = ?", (yesterday,))
    # Add player 2 to current roster
    conn.execute(
        "INSERT INTO league_rosters (team_name, team_index, player_id, roster_slot, is_user_team) "
        "VALUES ('Team A', 0, 2, 'SP', 0)"
    )
    conn.commit()
    conn.close()
    snapshot_league_rosters()

    changes = get_roster_changes("Team A", days=7)
    adds = [c for c in changes if c["change_type"] == "add"]
    assert len(adds) >= 1
    assert any(c["player_id"] == 2 for c in adds)


def test_empty_changes_when_no_snapshots():
    changes = get_roster_changes("Team A", days=7)
    assert changes == []

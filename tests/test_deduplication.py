"""Tests for player deduplication logic — ensures duplicate player entries
get merged correctly with all FK references remapped."""

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.database as db_mod
from src.database import deduplicate_players, get_connection, init_db


@pytest.fixture(autouse=True)
def temp_db():
    """Use a temporary database for each test."""
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


def _insert_player(conn, name, team, positions, is_hitter=1):
    """Insert a player directly and return the player_id."""
    conn.execute(
        "INSERT INTO players (name, team, positions, is_hitter) VALUES (?, ?, ?, ?)",
        (name, team, positions, is_hitter),
    )
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def _insert_projection(conn, player_id, system="blended"):
    """Insert a dummy projection for a player."""
    conn.execute(
        "INSERT INTO projections (player_id, system, pa, hr, avg) VALUES (?, ?, 500, 25, 0.280)",
        (player_id, system),
    )


def test_no_duplicates_returns_zero():
    """When no duplicates exist, dedup does nothing."""
    conn = get_connection()
    _insert_player(conn, "Aaron Judge", "NYY", "OF")
    _insert_player(conn, "Shohei Ohtani", "LAD", "DH")
    conn.commit()
    conn.close()

    result = deduplicate_players()
    assert result["duplicates_found"] == 0
    assert result["players_merged"] == 0


def test_basic_dedup_merges_two_entries():
    """Two entries for the same name get merged into one."""
    conn = get_connection()
    id1 = _insert_player(conn, "Corey Seager", "TEX", "SS")
    id2 = _insert_player(conn, "Corey Seager", "MLB", "SS,DH")
    _insert_projection(conn, id1, "blended")
    conn.commit()
    conn.close()

    result = deduplicate_players()
    assert result["duplicates_found"] == 1
    assert result["players_merged"] == 1

    # Verify only one entry remains
    conn = get_connection()
    rows = conn.execute("SELECT player_id, positions FROM players WHERE name = 'Corey Seager'").fetchall()
    conn.close()
    assert len(rows) == 1
    # Canonical should be id1 (has projections)
    assert rows[0][0] == id1
    # Positions should be merged
    assert "DH" in rows[0][1]
    assert "SS" in rows[0][1]


def test_dedup_prefers_entry_with_projections():
    """Canonical entry is the one with projections, not necessarily lowest ID."""
    conn = get_connection()
    id1 = _insert_player(conn, "Devin Williams", "MIL", "RP")  # No projections
    id2 = _insert_player(conn, "Devin Williams", "NYM", "RP")  # Has projections
    _insert_projection(conn, id2, "blended")
    conn.commit()
    conn.close()

    deduplicate_players()

    conn = get_connection()
    rows = conn.execute("SELECT player_id FROM players WHERE name = 'Devin Williams'").fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0][0] == id2  # The one with projections is kept


def test_dedup_remaps_league_rosters():
    """league_rosters FK references get remapped to canonical ID."""
    conn = get_connection()
    id1 = _insert_player(conn, "Marcus Semien", "TEX", "2B")
    id2 = _insert_player(conn, "Marcus Semien", "MLB", "2B")
    _insert_projection(conn, id1, "blended")
    # Roster references the duplicate (id2)
    conn.execute(
        "INSERT INTO league_rosters (team_name, team_index, player_id, is_user_team) VALUES (?, ?, ?, ?)",
        ("Team Hickey", 9, id2, 1),
    )
    conn.commit()
    conn.close()

    deduplicate_players()

    conn = get_connection()
    roster_row = conn.execute("SELECT player_id FROM league_rosters WHERE team_name = 'Team Hickey'").fetchone()
    conn.close()
    assert roster_row[0] == id1  # Remapped to canonical


def test_dedup_remaps_season_stats():
    """season_stats with composite PK (player_id, season) get remapped."""
    conn = get_connection()
    id1 = _insert_player(conn, "Bo Bichette", "TOR", "SS")
    id2 = _insert_player(conn, "Bo Bichette", "MLB", "SS")
    _insert_projection(conn, id1, "blended")
    # Stats on the duplicate
    conn.execute(
        "INSERT INTO season_stats (player_id, season, hr, avg) VALUES (?, 2026, 15, 0.270)",
        (id2,),
    )
    conn.commit()
    conn.close()

    deduplicate_players()

    conn = get_connection()
    stats = conn.execute("SELECT player_id FROM season_stats WHERE season = 2026").fetchall()
    players = conn.execute("SELECT player_id FROM players WHERE name = 'Bo Bichette'").fetchall()
    conn.close()
    assert len(players) == 1
    assert players[0][0] == id1
    assert len(stats) == 1
    assert stats[0][0] == id1  # Remapped


def test_dedup_handles_season_stats_conflict():
    """When both canonical and duplicate have same season, canonical data is kept."""
    conn = get_connection()
    id1 = _insert_player(conn, "Mookie Betts", "LAD", "OF,SS")
    id2 = _insert_player(conn, "Mookie Betts", "MLB", "OF")
    _insert_projection(conn, id1, "blended")
    # Both have 2026 stats
    conn.execute(
        "INSERT INTO season_stats (player_id, season, hr) VALUES (?, 2026, 28)",
        (id1,),
    )
    conn.execute(
        "INSERT INTO season_stats (player_id, season, hr) VALUES (?, 2026, 20)",
        (id2,),
    )
    conn.commit()
    conn.close()

    deduplicate_players()

    conn = get_connection()
    stats = conn.execute("SELECT player_id, hr FROM season_stats WHERE season = 2026").fetchall()
    conn.close()
    # Only one row should remain — the canonical's data (28 HR)
    assert len(stats) == 1
    assert stats[0][0] == id1
    assert stats[0][1] == 28


def test_dedup_remaps_injury_history():
    """injury_history with UNIQUE(player_id, season) gets remapped."""
    conn = get_connection()
    id1 = _insert_player(conn, "Max Scherzer", "TEX", "SP", is_hitter=0)
    id2 = _insert_player(conn, "Max Scherzer", "MLB", "SP", is_hitter=0)
    _insert_projection(conn, id1, "blended")
    conn.execute(
        "INSERT INTO injury_history (player_id, season, games_played, games_available) VALUES (?, 2024, 20, 162)",
        (id2,),
    )
    conn.commit()
    conn.close()

    deduplicate_players()

    conn = get_connection()
    hist = conn.execute("SELECT player_id FROM injury_history WHERE season = 2024").fetchall()
    conn.close()
    assert len(hist) == 1
    assert hist[0][0] == id1  # Remapped to canonical


def test_dedup_merges_mlb_id():
    """Canonical entry inherits mlb_id from duplicate if it doesn't have one."""
    conn = get_connection()
    id1 = _insert_player(conn, "Julio Rodriguez", "SEA", "OF")
    id2 = _insert_player(conn, "Julio Rodriguez", "MLB", "OF")
    _insert_projection(conn, id1, "blended")
    # Duplicate has mlb_id, canonical doesn't
    conn.execute("UPDATE players SET mlb_id = 12345 WHERE player_id = ?", (id2,))
    conn.commit()
    conn.close()

    deduplicate_players()

    conn = get_connection()
    row = conn.execute("SELECT mlb_id FROM players WHERE name = 'Julio Rodriguez'").fetchone()
    conn.close()
    assert row[0] == 12345


def test_dedup_multiple_duplicates():
    """Three entries for same name all merge into one."""
    conn = get_connection()
    id1 = _insert_player(conn, "Trea Turner", "PHI", "SS")
    id2 = _insert_player(conn, "Trea Turner", "MLB", "SS")
    id3 = _insert_player(conn, "Trea Turner", "", "SS,2B")
    _insert_projection(conn, id2, "blended")
    conn.commit()
    conn.close()

    result = deduplicate_players()
    assert result["players_merged"] == 2  # Two duplicates merged

    conn = get_connection()
    rows = conn.execute("SELECT player_id FROM players WHERE name = 'Trea Turner'").fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0][0] == id2  # The one with projections


def test_dedup_case_insensitive():
    """Deduplication is case-insensitive on player names."""
    conn = get_connection()
    id1 = _insert_player(conn, "Ronald Acuna Jr.", "ATL", "OF")
    id2 = _insert_player(conn, "ronald acuna jr.", "MLB", "OF")
    _insert_projection(conn, id1, "blended")
    conn.commit()
    conn.close()

    result = deduplicate_players()
    assert result["duplicates_found"] == 1

    conn = get_connection()
    rows = conn.execute("SELECT COUNT(*) FROM players WHERE LOWER(name) LIKE 'ronald acuna%'").fetchone()
    conn.close()
    assert rows[0] == 1


def test_upsert_player_name_fallback():
    """_upsert_player in database.py falls back to name-only match."""
    conn = get_connection()
    cursor = conn.cursor()
    # Insert player with one team
    cursor.execute(
        "INSERT INTO players (name, team, positions, is_hitter) VALUES (?, ?, ?, ?)",
        ("Freddie Freeman", "LAD", "1B", 1),
    )
    pid1 = cursor.lastrowid
    conn.commit()

    # Now upsert with different team — should find existing entry
    from src.database import _upsert_player

    pid2 = _upsert_player(cursor, "Freddie Freeman", "MLB", "1B,DH", True)
    conn.commit()
    conn.close()

    assert pid2 == pid1  # Same player, no duplicate created


def test_match_player_id_prefers_projections():
    """match_player_id prefers the entry with projections when duplicates exist."""
    conn = get_connection()
    id1 = _insert_player(conn, "Gunnar Henderson", "BAL", "SS,3B")
    id2 = _insert_player(conn, "Gunnar Henderson", "MLB", "SS")
    _insert_projection(conn, id2, "blended")
    conn.commit()
    conn.close()

    from src.live_stats import match_player_id

    result = match_player_id("Gunnar Henderson", "")
    assert result == id2  # The one with projections

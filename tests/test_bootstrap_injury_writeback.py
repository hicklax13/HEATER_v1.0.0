"""Test BUG-003a fix: _bootstrap_injury_writeback joins on player_id not name."""

import sqlite3

import pytest


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE players (
            player_id INTEGER PRIMARY KEY,
            name TEXT,
            is_injured INTEGER DEFAULT 0,
            injury_note TEXT
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
        CREATE TABLE refresh_log (
            source TEXT PRIMARY KEY,
            last_refresh TEXT,
            status TEXT DEFAULT 'unknown',
            rows_written INTEGER,
            rows_expected_min INTEGER,
            message TEXT,
            tier TEXT DEFAULT 'primary'
        );
        INSERT INTO players (player_id, name) VALUES
            (1, 'Player A'),
            (2, 'Player B'),
            (3, 'Player C'),
            (4, 'Player D');
        INSERT INTO league_rosters (team_name, team_index, player_id, status) VALUES
            ('My Team', 0, 1, 'IL10'),
            ('My Team', 0, 2, 'IL15'),
            ('Other Team', 1, 3, 'DTD'),
            ('Other Team', 1, 4, 'active');
    """)
    conn.commit()
    conn.close()

    import src.database as db_mod

    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    return db_path


def test_injury_writeback_flags_il_and_dtd_players(test_db, monkeypatch):
    """Players with IL10/IL15/IL60/DTD status should be flagged is_injured=1.
    'active' status players should remain is_injured=0."""
    monkeypatch.setattr("src.espn_injuries.fetch_espn_injuries", lambda: [])
    monkeypatch.setattr("src.espn_injuries.update_player_injury_flags", lambda x: 0)

    from src.data_bootstrap import BootstrapProgress, _bootstrap_injury_writeback

    result = _bootstrap_injury_writeback(BootstrapProgress())
    assert "Error" not in result, f"Function returned error: {result}"

    conn = sqlite3.connect(test_db)
    try:
        il_flagged = conn.execute("SELECT player_id FROM players WHERE is_injured = 1 ORDER BY player_id").fetchall()
        active = conn.execute("SELECT player_id FROM players WHERE is_injured = 0 ORDER BY player_id").fetchall()
    finally:
        conn.close()

    assert il_flagged == [(1,), (2,), (3,)], f"Expected players 1,2,3 flagged; got {il_flagged}"
    assert active == [(4,)], f"Expected only player 4 active; got {active}"

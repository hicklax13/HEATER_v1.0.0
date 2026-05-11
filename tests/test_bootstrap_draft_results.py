"""Test BUG-003b fix: _bootstrap_draft_results flags undroppable by player_id."""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest


@pytest.fixture
def test_db_and_client(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE players (
            player_id INTEGER PRIMARY KEY,
            name TEXT
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
        CREATE TABLE league_draft_results (
            pick INTEGER PRIMARY KEY,
            round INTEGER,
            team_name TEXT,
            player_id INTEGER,
            player_name TEXT
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
            (1, 'Player A'), (2, 'Player B'), (3, 'Player C'), (4, 'Player D');
        INSERT INTO league_rosters (team_name, team_index, player_id) VALUES
            ('My Team', 0, 1),
            ('My Team', 0, 2),
            ('Other Team', 1, 3),
            ('Other Team', 1, 4);
    """)
    conn.commit()
    conn.close()

    import src.database as db_mod

    monkeypatch.setattr(db_mod, "DB_PATH", db_path)

    # Mock Yahoo client: R1 = Player A (pid 1), R2 = Player C (pid 3), R4 = Player B (pid 2)
    mock_df = pd.DataFrame(
        [
            {"pick": 1, "round": 1, "team_name": "My Team", "player_id": 1, "player_name": "Player A"},
            {"pick": 2, "round": 2, "team_name": "Other Team", "player_id": 3, "player_name": "Player C"},
            {"pick": 25, "round": 4, "team_name": "My Team", "player_id": 2, "player_name": "Player B"},
        ]
    )
    mock_client = MagicMock()
    mock_client.get_draft_results.return_value = mock_df
    return db_path, mock_client


def test_draft_results_flags_undroppable_for_rounds_1_to_3_by_player_id(test_db_and_client):
    """Players drafted in rounds 1-3 should be flagged is_undroppable=1 in league_rosters."""
    test_db, mock_client = test_db_and_client
    from src.data_bootstrap import BootstrapProgress, _bootstrap_draft_results

    result = _bootstrap_draft_results(BootstrapProgress(), yahoo_client=mock_client)
    assert "Error" not in result, f"Function returned error: {result}"

    conn = sqlite3.connect(test_db)
    try:
        undroppable = sorted(
            row[0] for row in conn.execute("SELECT player_id FROM league_rosters WHERE is_undroppable = 1").fetchall()
        )
    finally:
        conn.close()

    # Players 1 (R1) and 3 (R2) should be undroppable; player 2 (R4) should NOT be
    assert undroppable == [1, 3], f"Expected [1, 3]; got {undroppable}"

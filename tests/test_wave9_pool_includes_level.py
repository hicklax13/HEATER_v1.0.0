"""Wave 9 / Task 4: load_player_pool() returns level column."""

import pytest


@pytest.fixture
def temp_db_with_players(tmp_path, monkeypatch):
    import sqlite3

    db_path = tmp_path / "test.db"
    monkeypatch.setattr("src.database.DB_PATH", db_path)
    from src.database import init_db

    init_db()
    conn = sqlite3.connect(db_path)
    try:
        conn.executemany(
            "INSERT INTO players (mlb_id, name, team, positions, is_hitter, level) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (700001, "MLB Player", "NYY", "OF", 1, "MLB"),
                (700002, "AAA Prospect", "SCR", "OF", 1, "AAA"),
                (700003, "AA Prospect", "SOM", "SP", 0, "AA"),
                (700004, "Legacy Player", "BOS", "1B", 1, None),  # legacy, level NULL
            ],
        )
        conn.commit()
    finally:
        conn.close()
    return db_path


def test_player_pool_exposes_level_column(temp_db_with_players):
    """load_player_pool returns a `level` column with values from players.level."""
    from src.database import load_player_pool

    pool = load_player_pool()
    assert "level" in pool.columns, (
        f"Wave 9 regression: load_player_pool() missing 'level' column. Found: {list(pool.columns)}"
    )
    # Spot-check: AAA player has level='AAA'
    aaa = pool[pool["mlb_id"] == 700002]
    if not aaa.empty:
        assert aaa.iloc[0]["level"] == "AAA"

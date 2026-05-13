"""Wave 9 / Task 1: players table has `level` column after init_db."""

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr("src.database.DB_PATH", db_path)
    return db_path


def test_init_db_adds_level_column(temp_db):
    """After init_db(), players.level must exist (NULL for legacy rows)."""
    from src.database import init_db

    init_db()
    conn = sqlite3.connect(temp_db)
    try:
        cols = [row[1] for row in conn.execute("PRAGMA table_info(players)").fetchall()]
    finally:
        conn.close()
    assert "level" in cols, f"Wave 9 regression: players.level column missing after init_db. Found columns: {cols}"


def test_safe_add_column_level_idempotent(temp_db):
    """Calling _safe_add_column twice must not raise."""
    from src.database import _safe_add_column, get_connection, init_db

    init_db()
    conn = get_connection()
    try:
        _safe_add_column(conn, "players", "level", "TEXT")
        _safe_add_column(conn, "players", "level", "TEXT")
    finally:
        conn.close()

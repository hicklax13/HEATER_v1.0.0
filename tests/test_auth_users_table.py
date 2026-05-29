"""The users table must be created additively by init_db()."""

import pytest


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Point src.database at a throwaway SQLite file with the real schema.

    get_connection() reads the module-global DB_PATH at call time, so
    monkeypatching it redirects every connection for the duration of the test.
    """
    db_file = tmp_path / "auth_test.db"
    monkeypatch.setattr("src.database.DB_PATH", db_file)
    from src.database import init_db

    init_db()
    return db_file


def _columns(conn, table):
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def test_users_table_exists(temp_db):
    from src.database import get_connection

    conn = get_connection()
    try:
        row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'").fetchone()
        assert row is not None, "init_db() must create the users table"
    finally:
        conn.close()


def test_users_table_columns(temp_db):
    from src.database import get_connection

    conn = get_connection()
    try:
        cols = _columns(conn, "users")
    finally:
        conn.close()
    expected = {
        "user_id",
        "username",
        "password_hash",
        "display_name",
        "team_name",
        "status",
        "is_admin",
        "created_at",
        "approved_at",
        "approved_by",
        "last_seen_at",
    }
    assert expected.issubset(cols), f"missing columns: {expected - cols}"


def test_username_unique_case_insensitive(temp_db):
    import sqlite3

    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, status, is_admin, created_at) "
            "VALUES ('Sam', 'x', 'pending', 0, '2026-05-28')"
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO users (username, password_hash, status, is_admin, created_at) "
                "VALUES ('sam', 'y', 'pending', 0, '2026-05-28')"
            )
            conn.commit()
    finally:
        conn.close()

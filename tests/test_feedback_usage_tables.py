"""init_db() creates the v2 feedback + usage_events tables (additive, idempotent)."""

import pytest


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db = tmp_path / "tables.db"
    monkeypatch.setattr("src.database.DB_PATH", db)
    from src.database import init_db

    init_db()
    return db


def _table_columns(table: str) -> set[str]:
    from src.database import get_connection

    conn = get_connection()
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {r[1] for r in rows}
    finally:
        conn.close()


def test_feedback_table_created(temp_db):
    cols = _table_columns("feedback")
    assert {
        "id",
        "user_id",
        "page",
        "feature_tag",
        "message",
        "app_version",
        "data_state",
        "status",
        "admin_notes",
        "created_at",
    } <= cols


def test_usage_events_table_created(temp_db):
    cols = _table_columns("usage_events")
    assert {"id", "user_id", "page", "action", "session_id", "created_at"} <= cols


def test_init_db_idempotent_for_new_tables(temp_db, monkeypatch):
    # A second init_db() on the same DB must not raise.
    from src.database import init_db

    init_db()
    assert "status" in _table_columns("feedback")

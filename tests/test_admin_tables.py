"""init_db() creates the v2 admin-dashboard tables (additive, idempotent)."""

import pytest


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db = tmp_path / "admin_tables.db"
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


def test_feature_flags_table(temp_db):
    assert {"key", "enabled", "updated_by", "updated_at"} <= _table_columns("feature_flags")


def test_audit_log_table(temp_db):
    assert {"id", "admin_id", "action", "target", "detail", "created_at"} <= _table_columns("audit_log")


def test_app_settings_table(temp_db):
    assert {"key", "value", "updated_by", "updated_at"} <= _table_columns("app_settings")


def test_sessions_table(temp_db):
    assert {"session_id", "user_id", "login_at", "last_activity_at"} <= _table_columns("sessions")


def test_page_visits_table(temp_db):
    assert {
        "id",
        "session_id",
        "user_id",
        "page",
        "enter_at",
        "exit_at",
        "dwell_seconds",
    } <= _table_columns("page_visits")


def test_init_db_idempotent_for_admin_tables(temp_db):
    # A second init_db() on the same DB must not raise.
    from src.database import init_db

    init_db()
    assert "enabled" in _table_columns("feature_flags")

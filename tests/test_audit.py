"""Admin audit log: write is MULTI_USER-gated; read joins username and is newest-first."""

import pytest


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db = tmp_path / "audit.db"
    monkeypatch.setattr("src.database.DB_PATH", db)
    from src.database import init_db

    init_db()
    return db


@pytest.fixture
def _flag_on(monkeypatch):
    monkeypatch.setenv("MULTI_USER", "1")


def _seed_admin(username: str = "admin_amy") -> int:
    from src.auth import approve_user, create_user, get_user
    from src.database import get_connection

    create_user(username, "pw")
    approve_user(username, team_name="Team " + username, approved_by="test")
    uid = get_user(username)["user_id"]
    conn = get_connection()
    try:
        conn.execute("UPDATE users SET is_admin = 1 WHERE user_id = ?", (uid,))
        conn.commit()
    finally:
        conn.close()
    return uid


def _rows() -> list[dict]:
    from src.database import get_connection

    conn = get_connection()
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM audit_log ORDER BY id").fetchall()]
    finally:
        conn.close()


def test_log_action_writes_row(temp_db, _flag_on):
    from src.audit import log_action

    uid = _seed_admin()
    log_action(uid, "approve_user", target="bob", detail={"team": "Team Bob"})
    rows = _rows()
    assert len(rows) == 1
    assert rows[0]["action"] == "approve_user"
    assert rows[0]["target"] == "bob"
    assert rows[0]["detail"] == '{"team": "Team Bob"}'


def test_log_action_noop_when_flag_off(temp_db, monkeypatch):
    monkeypatch.delenv("MULTI_USER", raising=False)
    from src.audit import log_action

    log_action(1, "approve_user", target="bob")
    assert _rows() == []


def test_log_action_none_detail_stores_null(temp_db, _flag_on):
    from src.audit import log_action

    uid = _seed_admin()
    log_action(uid, "exit_view_as")
    assert _rows()[0]["detail"] is None


def test_list_audit_returns_newest_first(temp_db, _flag_on):
    from src.audit import list_audit, log_action

    uid = _seed_admin()
    log_action(uid, "first")
    log_action(uid, "second")
    actions = [r["action"] for r in list_audit()]
    assert actions[:2] == ["second", "first"]
    assert list_audit()[0]["admin_username"] == "admin_amy"


def test_list_audit_filters_by_action(temp_db, _flag_on):
    from src.audit import list_audit, log_action

    uid = _seed_admin()
    log_action(uid, "toggle_flag", target="page:1_My_Team")
    log_action(uid, "export_csv", target="usage")
    only = list_audit(action="export_csv")
    assert len(only) == 1
    assert only[0]["target"] == "usage"

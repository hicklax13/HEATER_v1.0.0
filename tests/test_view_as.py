"""View-as-user: admin swaps identity, restores on exit; non-admins blocked; flag-gated."""

import pytest


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db = tmp_path / "viewas.db"
    monkeypatch.setattr("src.database.DB_PATH", db)
    from src.database import init_db

    init_db()
    return db


@pytest.fixture
def state(monkeypatch):
    import src.auth as auth

    s: dict = {}
    monkeypatch.setattr(auth, "_session_state", lambda: s)
    return s


@pytest.fixture
def _flag_on(monkeypatch):
    monkeypatch.setenv("MULTI_USER", "1")


def _seed(username: str, is_admin: int = 0) -> dict:
    from src.auth import approve_user, create_user, get_user
    from src.database import get_connection

    create_user(username, "pw")
    approve_user(username, team_name="Team " + username, approved_by="test")
    if is_admin:
        conn = get_connection()
        try:
            conn.execute("UPDATE users SET is_admin = 1 WHERE username = ?", (username,))
            conn.commit()
        finally:
            conn.close()
    return get_user(username)


def _audit_actions() -> list[str]:
    from src.database import get_connection

    conn = get_connection()
    try:
        return [r["action"] for r in conn.execute("SELECT action FROM audit_log ORDER BY id").fetchall()]
    finally:
        conn.close()


def test_enter_view_as_swaps_identity(temp_db, state, _flag_on):
    from src.auth import current_user, enter_view_as

    admin = _seed("admin_amy", is_admin=1)
    _seed("bob")
    state["auth_user"] = admin
    enter_view_as("bob", admin_id=admin["user_id"])
    assert current_user()["username"] == "bob"


def test_exit_view_as_restores_admin(temp_db, state, _flag_on):
    from src.auth import current_user, enter_view_as, exit_view_as

    admin = _seed("admin_amy", is_admin=1)
    _seed("bob")
    state["auth_user"] = admin
    enter_view_as("bob", admin_id=admin["user_id"])
    exit_view_as()
    assert current_user()["username"] == "admin_amy"


def test_non_admin_cannot_view_as(temp_db, state, _flag_on):
    from src.auth import current_user, enter_view_as

    carol = _seed("carol")
    _seed("bob")
    state["auth_user"] = carol
    enter_view_as("bob", admin_id=carol["user_id"])
    assert current_user()["username"] == "carol"


def test_view_as_writes_audit(temp_db, state, _flag_on):
    from src.auth import enter_view_as, exit_view_as

    admin = _seed("admin_amy", is_admin=1)
    _seed("bob")
    state["auth_user"] = admin
    enter_view_as("bob", admin_id=admin["user_id"])
    exit_view_as()
    assert _audit_actions() == ["view_as", "exit_view_as"]


def test_view_as_noop_when_flag_off(temp_db, state, monkeypatch):
    monkeypatch.delenv("MULTI_USER", raising=False)
    from src.auth import current_user, enter_view_as

    admin = _seed("admin_amy", is_admin=1)
    _seed("bob")
    state["auth_user"] = admin
    enter_view_as("bob", admin_id=admin["user_id"])
    assert current_user()["username"] == "admin_amy"

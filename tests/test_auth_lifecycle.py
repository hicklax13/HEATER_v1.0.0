"""Account lifecycle: register (pending) → get → (Task 6 adds approve/revoke)."""

import pytest


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_file = tmp_path / "auth_lifecycle.db"
    monkeypatch.setattr("src.database.DB_PATH", db_file)
    from src.database import init_db

    init_db()
    return db_file


def test_create_user_starts_pending(temp_db):
    from src.auth import create_user

    user = create_user("alice", "pw-alice", display_name="Alice A.")
    assert user["username"] == "alice"
    assert user["status"] == "pending"
    assert user["is_admin"] == 0
    assert user["team_name"] is None
    assert user["display_name"] == "Alice A."


def test_get_user_roundtrip(temp_db):
    from src.auth import create_user, get_user

    create_user("bob", "pw-bob")
    fetched = get_user("bob")
    assert fetched is not None
    assert fetched["username"] == "bob"


def test_get_user_is_case_insensitive(temp_db):
    from src.auth import create_user, get_user

    create_user("Carol", "pw")
    assert get_user("carol") is not None
    assert get_user("CAROL") is not None


def test_get_unknown_user_returns_none(temp_db):
    from src.auth import get_user

    assert get_user("nobody") is None


def test_duplicate_username_raises(temp_db):
    from src.auth import create_user

    create_user("dave", "pw1")
    with pytest.raises(ValueError):
        create_user("dave", "pw2")


def test_duplicate_username_case_insensitive_raises(temp_db):
    from src.auth import create_user

    create_user("Eve", "pw1")
    with pytest.raises(ValueError):
        create_user("eve", "pw2")


def test_stored_password_is_hashed(temp_db):
    from src.auth import create_user, verify_password

    create_user("frank", "frank-secret")
    from src.auth import get_user

    row = get_user("frank")
    assert row["password_hash"] != "frank-secret"
    assert verify_password("frank-secret", row["password_hash"]) is True


# ── classify_login (pure; no DB) ─────────────────────────────────────


def _make_user(status="active", password="pw"):
    from src.auth import hash_password

    return {
        "username": "u",
        "password_hash": hash_password(password),
        "status": status,
        "is_admin": 0,
        "team_name": "Team Hickey",
    }


def test_classify_login_no_such_user():
    from src.auth import classify_login

    assert classify_login(None, "anything") == "bad_credentials"


def test_classify_login_wrong_password():
    from src.auth import classify_login

    assert classify_login(_make_user(password="right"), "wrong") == "bad_credentials"


def test_classify_login_pending():
    from src.auth import classify_login

    assert classify_login(_make_user(status="pending"), "pw") == "pending"


def test_classify_login_revoked():
    from src.auth import classify_login

    assert classify_login(_make_user(status="revoked"), "pw") == "revoked"


def test_classify_login_active_ok():
    from src.auth import classify_login

    assert classify_login(_make_user(status="active"), "pw") == "ok"


def test_classify_login_checks_password_before_status():
    # A pending user with the WRONG password is bad_credentials, not "pending"
    # (don't leak account-existence/status to someone who can't authenticate).
    from src.auth import classify_login

    assert classify_login(_make_user(status="pending", password="right"), "wrong") == "bad_credentials"


# ── Admin lifecycle (needs temp_db) ──────────────────────────────────


def test_approve_user_activates_and_assigns_team(temp_db):
    from src.auth import approve_user, create_user, get_user

    create_user("grace", "pw")
    approve_user("grace", team_name="Team Hickey", approved_by="admin")
    u = get_user("grace")
    assert u["status"] == "active"
    assert u["team_name"] == "Team Hickey"
    assert u["approved_by"] == "admin"
    assert u["approved_at"] is not None


def test_revoke_user(temp_db):
    from src.auth import approve_user, create_user, get_user, revoke_user

    create_user("heidi", "pw")
    approve_user("heidi", team_name="Team A")
    revoke_user("heidi")
    assert get_user("heidi")["status"] == "revoked"


def test_set_user_team_reassigns(temp_db):
    from src.auth import approve_user, create_user, get_user, set_user_team

    create_user("ivan", "pw")
    approve_user("ivan", team_name="Team A")
    set_user_team("ivan", "Team B")
    assert get_user("ivan")["team_name"] == "Team B"


def test_list_users_filters_by_status(temp_db):
    from src.auth import approve_user, create_user, list_users

    create_user("p1", "pw")
    create_user("p2", "pw")
    create_user("a1", "pw")
    approve_user("a1", team_name="Team A")

    pending = {u["username"] for u in list_users(status="pending")}
    active = {u["username"] for u in list_users(status="active")}
    everyone = {u["username"] for u in list_users()}

    assert pending == {"p1", "p2"}
    assert active == {"a1"}
    assert {"p1", "p2", "a1"}.issubset(everyone)


def test_approve_unknown_user_raises(temp_db):
    from src.auth import approve_user

    with pytest.raises(ValueError):
        approve_user("ghost", team_name="Team A")


# ── Bootstrap admin from env ─────────────────────────────────────────


def test_ensure_bootstrap_admin_creates_active_admin(temp_db, monkeypatch):
    from src.auth import ensure_bootstrap_admin, get_user

    monkeypatch.setenv("ADMIN_USERNAME", "connor")
    monkeypatch.setenv("ADMIN_PASSWORD", "admin-pw")
    monkeypatch.setenv("ADMIN_TEAM_NAME", "Team Hickey")

    ensure_bootstrap_admin()
    u = get_user("connor")
    assert u is not None
    assert u["is_admin"] == 1
    assert u["status"] == "active"
    assert u["team_name"] == "Team Hickey"


def test_ensure_bootstrap_admin_is_idempotent(temp_db, monkeypatch):
    from src.auth import ensure_bootstrap_admin, get_user, verify_password

    monkeypatch.setenv("ADMIN_USERNAME", "connor")
    monkeypatch.setenv("ADMIN_PASSWORD", "first-pw")
    ensure_bootstrap_admin()
    # Second call with a DIFFERENT password must NOT reset the existing admin.
    monkeypatch.setenv("ADMIN_PASSWORD", "second-pw")
    ensure_bootstrap_admin()
    u = get_user("connor")
    assert verify_password("first-pw", u["password_hash"]) is True
    assert verify_password("second-pw", u["password_hash"]) is False


def test_ensure_bootstrap_admin_noop_without_env(temp_db, monkeypatch):
    from src.auth import ensure_bootstrap_admin, list_users

    monkeypatch.delenv("ADMIN_USERNAME", raising=False)
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    ensure_bootstrap_admin()
    assert list_users() == []

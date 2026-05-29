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

"""BR-1: persistent (cookie/token-backed) sessions.

A page refresh / bookmark / deep-link starts a fresh st.session_state (it is
per-websocket), which previously dropped the logged-in member back to the login
wall. BR-1 mints an opaque, server-stored, revocable token at login, sets it in
a browser cookie, and re-hydrates ``auth_user`` from that token on app load when
the session is empty — so refresh no longer forces a re-login.

Security model (the cookie cannot be HttpOnly because Streamlit can only set
cookies from JS): the token is opaque (secrets.token_urlsafe), stored
server-side, validated server-side on every restore, expires (~30d), and is
revoked on logout. It is NEVER identity-bearing on its own.

HARD INVARIANT: when MULTI_USER is off, NONE of this runs — no cookie read, no
cookie write, no token table touched. v1 stays byte-for-byte.
"""

from __future__ import annotations

import pytest

# ── DB fixture (throwaway SQLite, schema initialized) ────────────────


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr("src.database.DB_PATH", tmp_path / "persistent_sessions.db")
    from src.database import init_db

    init_db()
    return tmp_path


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv("MULTI_USER", "1")


@pytest.fixture
def flag_off(monkeypatch):
    monkeypatch.delenv("MULTI_USER", raising=False)


def _make_active_user(username="member1", team="Team One", is_admin=0):
    """Insert an active user directly and return its row dict."""
    from src.auth import get_user
    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, display_name, team_name, "
            "status, is_admin, created_at) VALUES (?, 'x', ?, ?, 'active', ?, '2026-01-01')",
            (username, username, team, is_admin),
        )
        conn.commit()
    finally:
        conn.close()
    return get_user(username)


# ── Schema ───────────────────────────────────────────────────────────


def test_auth_tokens_table_created_by_init_db(temp_db):
    """init_db() creates the additive auth_tokens table (idempotent)."""
    from src.database import get_connection

    conn = get_connection()
    try:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(auth_tokens)").fetchall()}
    finally:
        conn.close()
    # token is the opaque secret; the rest are the lifecycle fields.
    assert {"token", "user_id", "created_at", "expires_at", "revoked"} <= cols


# ── Token lifecycle (server-side, flag-independent core) ──────────────


def test_issue_session_token_is_opaque_and_persisted(temp_db):
    from src.auth import issue_session_token
    from src.database import get_connection

    user = _make_active_user()
    token = issue_session_token(user["user_id"], days=30)

    # Opaque + long enough to resist guessing.
    assert isinstance(token, str)
    assert len(token) >= 32
    # Persisted with the right user + not revoked.
    conn = get_connection()
    try:
        row = conn.execute("SELECT user_id, revoked FROM auth_tokens WHERE token = ?", (token,)).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert row["user_id"] == user["user_id"]
    assert row["revoked"] == 0


def test_issue_session_token_unique_each_call(temp_db):
    from src.auth import issue_session_token

    user = _make_active_user()
    t1 = issue_session_token(user["user_id"])
    t2 = issue_session_token(user["user_id"])
    assert t1 != t2


def test_validate_session_token_returns_active_user(temp_db):
    from src.auth import issue_session_token, validate_session_token

    user = _make_active_user(username="alice", team="Alice Squad")
    token = issue_session_token(user["user_id"])

    got = validate_session_token(token)
    assert got is not None
    assert got["username"] == "alice"
    assert got["team_name"] == "Alice Squad"


def test_validate_rejects_unknown_token(temp_db):
    from src.auth import validate_session_token

    _make_active_user()
    assert validate_session_token("not-a-real-token") is None


def test_validate_rejects_none_and_empty(temp_db):
    from src.auth import validate_session_token

    assert validate_session_token(None) is None
    assert validate_session_token("") is None


def test_validate_rejects_expired_token(temp_db):
    from src.auth import issue_session_token, validate_session_token

    user = _make_active_user()
    # Negative lifetime → already expired at issue time.
    token = issue_session_token(user["user_id"], days=-1)
    assert validate_session_token(token) is None


def test_validate_rejects_revoked_token(temp_db):
    from src.auth import (
        issue_session_token,
        revoke_session_token,
        validate_session_token,
    )

    user = _make_active_user()
    token = issue_session_token(user["user_id"])
    assert validate_session_token(token) is not None  # sanity
    revoke_session_token(token)
    assert validate_session_token(token) is None


def test_validate_rejects_token_for_non_active_user(temp_db):
    """A token survives revoke_user; validation must re-check status=active."""
    from src.auth import issue_session_token, revoke_user, validate_session_token

    user = _make_active_user(username="bob")
    token = issue_session_token(user["user_id"])
    revoke_user("bob")  # admin revokes the account, not the token
    assert validate_session_token(token) is None


def test_revoke_session_token_unknown_is_noop(temp_db):
    from src.auth import revoke_session_token

    # Must not raise on an unknown token.
    revoke_session_token("ghost")


# ── require_auth re-hydration (flag ON) ──────────────────────────────


class _StopCalled(Exception):
    pass


@pytest.fixture
def _streamlit_stub(monkeypatch):
    """Stub st.stop() and the login renderer so require_auth is testable."""
    import src.auth as auth

    rendered = {"login": False}

    def _stop(*a, **k):
        raise _StopCalled()

    monkeypatch.setattr("src.auth.st.stop", _stop, raising=False)
    monkeypatch.setattr(auth, "_render_login_and_register", lambda: rendered.__setitem__("login", True))
    return rendered


def test_require_auth_rehydrates_from_valid_cookie_without_login(temp_db, flag_on, _streamlit_stub, monkeypatch):
    """Empty session + valid token cookie => auth_user restored, NO login form."""
    import src.auth as auth

    user = _make_active_user(username="carol", team="Carol FC")
    token = auth.issue_session_token(user["user_id"])

    # Fresh per-websocket session: no auth_user yet.
    session = {}
    monkeypatch.setattr(auth, "_session_state", lambda: session)
    # Cookie layer returns our valid token.
    monkeypatch.setattr(auth, "_read_auth_cookie", lambda: token)
    # Skip the DB-schema/admin re-bootstrap inside require_auth (already done).
    monkeypatch.setattr(auth, "_ensure_session_bootstrap", lambda: None)

    auth.require_auth()  # must NOT raise _StopCalled

    assert _streamlit_stub["login"] is False
    assert session.get("auth_user", {}).get("username") == "carol"


def test_require_auth_shows_login_when_no_session_and_no_cookie(temp_db, flag_on, _streamlit_stub, monkeypatch):
    import src.auth as auth

    _make_active_user()
    session = {}
    monkeypatch.setattr(auth, "_session_state", lambda: session)
    monkeypatch.setattr(auth, "_read_auth_cookie", lambda: None)
    monkeypatch.setattr(auth, "_ensure_session_bootstrap", lambda: None)

    with pytest.raises(_StopCalled):
        auth.require_auth()
    assert _streamlit_stub["login"] is True


def test_require_auth_shows_login_when_cookie_token_expired(temp_db, flag_on, _streamlit_stub, monkeypatch):
    import src.auth as auth

    user = _make_active_user()
    expired = auth.issue_session_token(user["user_id"], days=-1)
    session = {}
    monkeypatch.setattr(auth, "_session_state", lambda: session)
    monkeypatch.setattr(auth, "_read_auth_cookie", lambda: expired)
    monkeypatch.setattr(auth, "_ensure_session_bootstrap", lambda: None)

    with pytest.raises(_StopCalled):
        auth.require_auth()
    assert _streamlit_stub["login"] is True
    assert "auth_user" not in session


def test_require_auth_shows_login_when_cookie_token_revoked(temp_db, flag_on, _streamlit_stub, monkeypatch):
    import src.auth as auth

    user = _make_active_user()
    token = auth.issue_session_token(user["user_id"])
    auth.revoke_session_token(token)
    session = {}
    monkeypatch.setattr(auth, "_session_state", lambda: session)
    monkeypatch.setattr(auth, "_read_auth_cookie", lambda: token)
    monkeypatch.setattr(auth, "_ensure_session_bootstrap", lambda: None)

    with pytest.raises(_StopCalled):
        auth.require_auth()
    assert _streamlit_stub["login"] is True


# ── logout revokes the token + clears the cookie ─────────────────────


def test_logout_revokes_token_and_clears_cookie(temp_db, flag_on, monkeypatch):
    import src.auth as auth

    user = _make_active_user()
    token = auth.issue_session_token(user["user_id"])
    session = {"auth_user": user, auth._SESSION_TOKEN_KEY: token}
    monkeypatch.setattr(auth, "_session_state", lambda: session)

    cleared = {"called": False}
    monkeypatch.setattr(auth, "_clear_auth_cookie", lambda: cleared.__setitem__("called", True))

    auth.logout()

    assert "auth_user" not in session
    assert auth.validate_session_token(token) is None  # token revoked server-side
    assert cleared["called"] is True


# ── HARD INVARIANT: flag OFF is byte-for-byte (no cookie/token at all) ─


def test_flag_off_require_auth_is_noop_no_cookie_no_token(temp_db, flag_off, monkeypatch):
    """MULTI_USER off => require_auth touches no cookie layer and no token table."""
    import src.auth as auth

    touched = {"read": False, "write": False, "validate": False}
    monkeypatch.setattr(auth, "_read_auth_cookie", lambda: touched.__setitem__("read", True) or "x")
    monkeypatch.setattr(auth, "_write_auth_cookie", lambda *a, **k: touched.__setitem__("write", True))
    monkeypatch.setattr(auth, "validate_session_token", lambda *a, **k: touched.__setitem__("validate", True))

    assert auth.require_auth() is None
    assert touched == {"read": False, "write": False, "validate": False}


def test_flag_off_logout_does_not_touch_cookie(temp_db, flag_off, monkeypatch):
    """v1 logout just clears session_state — no cookie clear, no token revoke."""
    import src.auth as auth

    session = {"auth_user": {"username": "x"}}
    monkeypatch.setattr(auth, "_session_state", lambda: session)
    cleared = {"called": False}
    monkeypatch.setattr(auth, "_clear_auth_cookie", lambda: cleared.__setitem__("called", True))

    auth.logout()
    assert "auth_user" not in session
    assert cleared["called"] is False

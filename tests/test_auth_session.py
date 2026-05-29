"""Session identity helpers, tested via a monkeypatched _session_state()."""

import pytest


@pytest.fixture
def fake_session(monkeypatch):
    state: dict = {}
    monkeypatch.setattr("src.auth._session_state", lambda: state)
    return state


def test_current_user_none_when_empty(fake_session):
    from src.auth import current_user

    assert current_user() is None


def test_set_and_get_session_user(fake_session):
    from src.auth import _set_session_user, current_user

    _set_session_user({"username": "alice", "team_name": "Team Hickey", "is_admin": 0})
    assert current_user()["username"] == "alice"


def test_logout_clears_user(fake_session):
    from src.auth import _set_session_user, current_user, logout

    _set_session_user({"username": "alice"})
    logout()
    assert current_user() is None


# ── require_auth / require_admin ─────────────────────────────────────


class _Stop(Exception):
    """Sentinel standing in for st.stop()."""


@pytest.fixture
def stub_st(monkeypatch):
    """Stub the st.* calls require_auth/require_admin touch."""

    def _stop(*a, **k):
        raise _Stop()

    monkeypatch.setattr("src.auth.st.stop", _stop, raising=False)
    monkeypatch.setattr("src.auth.st.error", lambda *a, **k: None, raising=False)


def test_require_auth_noop_when_disabled(monkeypatch):
    # The load-bearing back-compat guarantee: flag off → never touches session/DB/st.
    from src.auth import require_auth

    monkeypatch.setattr("src.auth.multi_user_enabled", lambda: False)
    assert require_auth() is None


def test_require_auth_stops_when_no_user(monkeypatch, fake_session, stub_st):
    import src.auth as auth

    monkeypatch.setattr(auth, "multi_user_enabled", lambda: True)
    monkeypatch.setattr(auth, "_ensure_session_bootstrap", lambda: None)
    rendered = {"called": False}
    monkeypatch.setattr(auth, "_render_login_and_register", lambda: rendered.__setitem__("called", True))

    with pytest.raises(_Stop):
        auth.require_auth()
    assert rendered["called"] is True


def test_require_auth_ok_for_active_user(monkeypatch, fake_session):
    import src.auth as auth

    monkeypatch.setattr(auth, "multi_user_enabled", lambda: True)
    monkeypatch.setattr(auth, "_ensure_session_bootstrap", lambda: None)
    auth._set_session_user({"username": "alice", "status": "active"})
    # DB re-validation returns a fresh active row.
    monkeypatch.setattr(
        auth, "get_user", lambda u: {"username": "alice", "status": "active", "team_name": "Team Hickey", "is_admin": 0}
    )
    assert auth.require_auth() is None
    # session copy refreshed with the fresh DB row (team_name now present)
    assert auth.current_user()["team_name"] == "Team Hickey"


def test_require_auth_logs_out_revoked_user(monkeypatch, fake_session, stub_st):
    import src.auth as auth

    monkeypatch.setattr(auth, "multi_user_enabled", lambda: True)
    monkeypatch.setattr(auth, "_ensure_session_bootstrap", lambda: None)
    monkeypatch.setattr(auth, "_render_login_and_register", lambda: None)
    auth._set_session_user({"username": "bob", "status": "active"})
    # Admin revoked bob since login.
    monkeypatch.setattr(auth, "get_user", lambda u: {"username": "bob", "status": "revoked"})

    with pytest.raises(_Stop):
        auth.require_auth()
    assert auth.current_user() is None  # logged out


def test_require_admin_blocks_non_admin(monkeypatch, fake_session, stub_st):
    import src.auth as auth

    monkeypatch.setattr(auth, "require_auth", lambda: None)
    auth._set_session_user({"username": "alice", "status": "active", "is_admin": 0})
    with pytest.raises(_Stop):
        auth.require_admin()


def test_require_admin_allows_admin(monkeypatch, fake_session):
    import src.auth as auth

    monkeypatch.setattr(auth, "require_auth", lambda: None)
    auth._set_session_user({"username": "connor", "status": "active", "is_admin": 1})
    assert auth.require_admin() is None

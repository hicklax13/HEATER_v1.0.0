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

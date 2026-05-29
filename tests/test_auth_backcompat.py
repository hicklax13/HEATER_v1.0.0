"""MULTI_USER off ⇒ exact v1 behavior (the load-bearing rollout guarantee)."""

import pytest


@pytest.fixture(autouse=True)
def _flag_off(monkeypatch):
    monkeypatch.delenv("MULTI_USER", raising=False)


def test_require_auth_is_noop_when_off():
    from src.auth import require_auth

    # Must not touch the DB, session, or st.* — just return.
    assert require_auth() is None


def test_require_admin_is_noop_path_when_off(monkeypatch):
    # With the flag off require_auth() returns immediately; require_admin then
    # inspects current_user(). Simulate "no session" → it should hard-stop only
    # via st.stop, which we capture. (Admin pages are still admin-gated even in
    # single-user mode — there's simply no logged-in user, so access is denied.)
    import src.auth as auth

    class _Stop(Exception):
        pass

    monkeypatch.setattr("src.auth.st.stop", lambda *a, **k: (_ for _ in ()).throw(_Stop()), raising=False)
    monkeypatch.setattr("src.auth.st.error", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(auth, "current_user", lambda: None)
    with pytest.raises(_Stop):
        auth.require_admin()


def test_user_team_name_uses_league_teams_when_off(monkeypatch):
    import src.yahoo_data_service as yds_mod

    captured = {"sql": None}

    class _Cur:
        def execute(self, sql, *a):
            captured["sql"] = sql
            return self

        def fetchone(self):
            return ("Team Hickey",)

    class _Conn:
        def cursor(self):
            return _Cur()

        def close(self):
            pass

    monkeypatch.setattr("src.database.get_connection", lambda: _Conn())
    svc = yds_mod.YahooDataService.__new__(yds_mod.YahooDataService)
    assert svc._get_user_team_name() == "Team Hickey"
    assert "is_user_team = 1" in captured["sql"]

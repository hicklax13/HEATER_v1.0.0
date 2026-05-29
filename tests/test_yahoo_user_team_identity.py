"""_get_user_team_name reflects session identity in multi-user mode."""


def _service():
    """A bare YahooDataService instance without running __init__ side effects."""
    from src.yahoo_data_service import YahooDataService

    return YahooDataService.__new__(YahooDataService)


def test_multiuser_returns_session_team(monkeypatch):
    import src.yahoo_data_service as yds_mod

    monkeypatch.setattr(yds_mod, "multi_user_enabled", lambda: True, raising=False)
    monkeypatch.setattr(
        yds_mod, "current_user", lambda: {"username": "alice", "team_name": "Team Alice"}, raising=False
    )
    svc = _service()
    assert svc._get_user_team_name() == "Team Alice"


def test_multiuser_no_session_falls_back_to_db(monkeypatch):
    import src.yahoo_data_service as yds_mod

    monkeypatch.setattr(yds_mod, "multi_user_enabled", lambda: True, raising=False)
    monkeypatch.setattr(yds_mod, "current_user", lambda: None, raising=False)
    # current_user None → fall through to the legacy league_teams query.
    called = {"db": False}

    def _fake_conn():
        called["db"] = True
        raise RuntimeError("stop here — we only assert the DB path was taken")

    monkeypatch.setattr("src.database.get_connection", _fake_conn)
    svc = _service()
    assert svc._get_user_team_name() is None
    assert called["db"] is True


def test_single_user_mode_uses_db(monkeypatch):
    import src.yahoo_data_service as yds_mod

    monkeypatch.setattr(yds_mod, "multi_user_enabled", lambda: False, raising=False)
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
    svc = _service()
    assert svc._get_user_team_name() == "Team Hickey"
    assert "is_user_team = 1" in captured["sql"]

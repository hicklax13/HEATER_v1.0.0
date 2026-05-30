"""Per-page feature flags: absence = enabled; non-admins blocked on disabled pages."""

import pytest


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db = tmp_path / "flags.db"
    monkeypatch.setattr("src.database.DB_PATH", db)
    from src.database import init_db

    init_db()
    return db


@pytest.fixture
def _flag_on(monkeypatch):
    monkeypatch.setenv("MULTI_USER", "1")


def _audit_actions() -> list[str]:
    from src.database import get_connection

    conn = get_connection()
    try:
        return [r["action"] for r in conn.execute("SELECT action FROM audit_log ORDER BY id").fetchall()]
    finally:
        conn.close()


def test_absent_flag_is_enabled(temp_db, _flag_on):
    from src.feature_flags import is_page_enabled

    assert is_page_enabled("page:1_My_Team") is True


def test_set_disabled_then_enabled(temp_db, _flag_on):
    from src.feature_flags import is_page_enabled, set_page_flag

    set_page_flag("page:17_Leaders", False, admin_id=1)
    assert is_page_enabled("page:17_Leaders") is False
    set_page_flag("page:17_Leaders", True, admin_id=1)
    assert is_page_enabled("page:17_Leaders") is True


def test_list_page_flags(temp_db, _flag_on):
    from src.feature_flags import list_page_flags, set_page_flag

    set_page_flag("page:11_Trade_Analyzer", False, admin_id=1)
    flags = list_page_flags()
    assert flags["page:11_Trade_Analyzer"] is False


def test_toggle_writes_audit(temp_db, _flag_on):
    from src.feature_flags import set_page_flag

    set_page_flag("page:14_Free_Agents", False, admin_id=3)
    assert _audit_actions() == ["toggle_flag"]


def test_is_page_enabled_true_when_flag_off(temp_db, monkeypatch):
    monkeypatch.delenv("MULTI_USER", raising=False)
    from src.feature_flags import is_page_enabled

    assert is_page_enabled("page:anything") is True


def test_require_page_enabled_admin_bypass(temp_db, _flag_on, monkeypatch):
    from src import feature_flags
    from src.feature_flags import require_page_enabled, set_page_flag

    set_page_flag("page:1_My_Team", False, admin_id=1)
    monkeypatch.setattr("src.auth.current_user", lambda: {"is_admin": 1})
    # Admin on a disabled page must NOT stop.
    require_page_enabled("page:1_My_Team")


def test_require_page_enabled_noop_when_enabled(temp_db, _flag_on, monkeypatch):
    from src.feature_flags import require_page_enabled

    monkeypatch.setattr("src.auth.current_user", lambda: {"is_admin": 0})
    require_page_enabled("page:1_My_Team")  # enabled by absence → no stop


def test_require_page_enabled_stops_disabled_for_non_admin(temp_db, _flag_on, monkeypatch):
    import sys
    import types

    from src.feature_flags import require_page_enabled, set_page_flag

    set_page_flag("page:1_My_Team", False, admin_id=1)
    monkeypatch.setattr("src.auth.current_user", lambda: {"is_admin": 0})

    calls = {"error": None}

    def _stop():
        raise RuntimeError("stopped")

    fake_st = types.SimpleNamespace(error=lambda m: calls.update(error=m), stop=_stop)
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)

    with pytest.raises(RuntimeError, match="stopped"):
        require_page_enabled("page:1_My_Team")
    assert "disabled" in calls["error"].lower()

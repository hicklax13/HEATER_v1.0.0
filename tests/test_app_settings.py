"""app_settings: typed broadcast/maintenance toggles; setters flag-gated + one audit row each."""

import pytest


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db = tmp_path / "settings.db"
    monkeypatch.setattr("src.database.DB_PATH", db)
    from src.database import init_db

    init_db()
    return db


@pytest.fixture
def _flag_on(monkeypatch):
    monkeypatch.setenv("MULTI_USER", "1")


def _audit_rows() -> list[dict]:
    from src.database import get_connection

    conn = get_connection()
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM audit_log ORDER BY id").fetchall()]
    finally:
        conn.close()


def test_default_disabled(temp_db):
    from src.app_settings import get_broadcast, get_maintenance

    assert get_broadcast() == {"enabled": False, "message": ""}
    assert get_maintenance() == {"enabled": False, "message": ""}


def test_set_broadcast_roundtrips(temp_db, _flag_on):
    from src.app_settings import get_broadcast, set_broadcast

    set_broadcast(True, "Heads up: trade deadline Friday", admin_id=1)
    assert get_broadcast() == {"enabled": True, "message": "Heads up: trade deadline Friday"}


def test_set_broadcast_writes_one_audit_row(temp_db, _flag_on):
    from src.app_settings import set_broadcast

    set_broadcast(True, "hi", admin_id=7)
    rows = _audit_rows()
    assert len(rows) == 1
    assert rows[0]["action"] == "set_broadcast"
    assert rows[0]["admin_id"] == 7


def test_maintenance_roundtrips(temp_db, _flag_on):
    from src.app_settings import get_maintenance, set_maintenance

    set_maintenance(True, "Back at 5pm ET", admin_id=1)
    assert get_maintenance() == {"enabled": True, "message": "Back at 5pm ET"}


def test_maintenance_audit_action(temp_db, _flag_on):
    from src.app_settings import set_maintenance

    set_maintenance(True, "down", admin_id=2)
    assert _audit_rows()[0]["action"] == "toggle_maintenance"


def test_setters_noop_when_flag_off(temp_db, monkeypatch):
    monkeypatch.delenv("MULTI_USER", raising=False)
    from src.app_settings import get_broadcast, set_broadcast

    set_broadcast(True, "should not persist", admin_id=1)
    assert get_broadcast() == {"enabled": False, "message": ""}
    assert _audit_rows() == []


def test_corrupt_json_falls_back_to_default(temp_db):
    from src.app_settings import _put_setting, get_broadcast

    _put_setting("broadcast", "not-json", admin_id=1)
    assert get_broadcast() == {"enabled": False, "message": ""}

"""TDD for the data_bootstrap mlb_id auto-enrichment phase (_enrich_mlb_ids).

Network is never hit — `resolve_mlb_id` is monkeypatched. The phase backfills
null/0 mlb_id for FA/roster-relevant players via the shared, Muncy-DNA-safe
resolver, with the same write guards the operator script uses (never overwrite a
non-null id; never duplicate an id onto two rows; bounded; idempotent).
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr("src.database.DB_PATH", db_path)
    from src.db.engine import reset_engine_cache

    reset_engine_cache()
    from src.database import init_db

    init_db()
    yield db_path
    reset_engine_cache()


def _player(conn, player_id, name, team, mlb_id=None, positions="OF", is_hitter=1):
    conn.execute(
        "INSERT INTO players (player_id, name, team, mlb_id, positions, is_hitter, is_injured) VALUES (?,?,?,?,?,?,0)",
        (player_id, name, team, mlb_id, positions, is_hitter),
    )


def _roster(conn, player_id, team_name="Team Hickey", team_index=0, is_user=0):
    conn.execute(
        "INSERT INTO league_rosters (team_name, team_index, player_id, is_user_team) VALUES (?,?,?,?)",
        (team_name, team_index, player_id, is_user),
    )


def _own(conn, player_id, pct, date="2026-06-21"):
    conn.execute(
        "INSERT INTO ownership_trends (player_id, date, percent_owned) VALUES (?,?,?)",
        (player_id, date, pct),
    )


def _progress():
    p = MagicMock()
    p.phase = ""
    p.detail = ""
    return p


def _mlb_id(player_id):
    from src.database import get_connection

    conn = get_connection()
    try:
        row = conn.execute("SELECT mlb_id FROM players WHERE player_id = ?", (player_id,)).fetchone()
    finally:
        conn.close()
    return row[0] if row else None


def _refresh_log(source="mlb_ids"):
    from src.database import get_connection

    conn = get_connection()
    try:
        row = conn.execute("SELECT status, message FROM refresh_log WHERE source = ?", (source,)).fetchone()
    finally:
        conn.close()
    return row  # (status, message) or None


def test_enriches_rostered_null_id_player(temp_db, monkeypatch):
    from src.database import get_connection

    conn = get_connection()
    try:
        _player(conn, 1, "River Ryan", "LAD", mlb_id=None)
        _roster(conn, 1)
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setattr("src.player_id_resolver.resolve_mlb_id", lambda n, t: (689981, "match"))
    from src.data_bootstrap import _enrich_mlb_ids

    _enrich_mlb_ids(_progress())

    assert _mlb_id(1) == 689981
    status, _msg = _refresh_log()
    assert status == "success"


def test_enriches_owned_null_id_player(temp_db, monkeypatch):
    from src.database import get_connection

    conn = get_connection()
    try:
        _player(conn, 2, "Emmanuel Clase", "CLE", mlb_id=None, is_hitter=0, positions="RP")
        _own(conn, 2, 5.0)
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setattr("src.player_id_resolver.resolve_mlb_id", lambda n, t: (661403, "match"))
    from src.data_bootstrap import _enrich_mlb_ids

    _enrich_mlb_ids(_progress())

    assert _mlb_id(2) == 661403


def test_skips_irrelevant_null_id_player(temp_db, monkeypatch):
    """A null-id player that is neither rostered nor owned is out of scope."""
    from src.database import get_connection

    conn = get_connection()
    try:
        _player(conn, 3, "Some Minor Leaguer", "AAA", mlb_id=None)  # not rostered, not owned
        conn.commit()
    finally:
        conn.close()

    called = []

    def tracking_resolver(name, team):
        called.append(name)
        return (333, "match")

    monkeypatch.setattr("src.player_id_resolver.resolve_mlb_id", tracking_resolver)
    from src.data_bootstrap import _enrich_mlb_ids

    _enrich_mlb_ids(_progress())

    assert _mlb_id(3) is None
    assert "Some Minor Leaguer" not in called  # resolver never even consulted


def test_never_overwrites_nonnull_id(temp_db, monkeypatch):
    from src.database import get_connection

    conn = get_connection()
    try:
        _player(conn, 4, "Already Resolved", "NYY", mlb_id=999)
        _roster(conn, 4)
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setattr("src.player_id_resolver.resolve_mlb_id", lambda n, t: (444, "match"))
    from src.data_bootstrap import _enrich_mlb_ids

    _enrich_mlb_ids(_progress())

    assert _mlb_id(4) == 999  # untouched


def test_collision_guard_skips_taken_id(temp_db, monkeypatch):
    """Resolving to an mlb_id already owned by another row must be refused."""
    from src.database import get_connection

    conn = get_connection()
    try:
        _player(conn, 5, "Existing Owner", "BOS", mlb_id=500)  # already has 500
        _player(conn, 6, "Collider", "TB", mlb_id=None)
        _roster(conn, 6)
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setattr("src.player_id_resolver.resolve_mlb_id", lambda n, t: (500, "match"))
    from src.data_bootstrap import _enrich_mlb_ids

    _enrich_mlb_ids(_progress())

    assert _mlb_id(6) is None  # collision -> skipped
    assert _mlb_id(5) == 500  # untouched


def test_idempotent_no_targets_no_resolver_calls(temp_db, monkeypatch):
    from src.database import get_connection

    conn = get_connection()
    try:
        _player(conn, 7, "All Set", "SF", mlb_id=700)
        _roster(conn, 7)
        conn.commit()
    finally:
        conn.close()

    resolver = MagicMock(return_value=(1, "x"))
    monkeypatch.setattr("src.player_id_resolver.resolve_mlb_id", resolver)
    from src.data_bootstrap import _enrich_mlb_ids

    _enrich_mlb_ids(_progress())

    resolver.assert_not_called()
    status, _msg = _refresh_log()
    assert status in ("success", "no_data")


def test_resolver_none_leaves_player_null(temp_db, monkeypatch):
    from src.database import get_connection

    conn = get_connection()
    try:
        _player(conn, 8, "Unresolvable Prospect", "PIT", mlb_id=None)
        _roster(conn, 8)
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setattr("src.player_id_resolver.resolve_mlb_id", lambda n, t: (None, "no match"))
    from src.data_bootstrap import _enrich_mlb_ids

    result = _enrich_mlb_ids(_progress())

    assert _mlb_id(8) is None
    assert "error" not in result.lower()


def test_error_path_writes_refresh_log_error(temp_db, monkeypatch):
    from src.database import get_connection

    conn = get_connection()
    try:
        _player(conn, 9, "Boom Player", "ATL", mlb_id=None)
        _roster(conn, 9)
        conn.commit()
    finally:
        conn.close()

    def boom(name, team):
        raise RuntimeError("simulated resolver bug")

    monkeypatch.setattr("src.player_id_resolver.resolve_mlb_id", boom)
    from src.data_bootstrap import _enrich_mlb_ids

    result = _enrich_mlb_ids(_progress())  # must NOT raise

    assert "error" in result.lower()
    status, _msg = _refresh_log()
    assert status == "error"


def test_caps_resolutions_and_logs(temp_db, monkeypatch):
    from src.database import get_connection

    conn = get_connection()
    try:
        for pid in (10, 11, 12):
            _player(conn, pid, f"Player {pid}", "LAD", mlb_id=None)
            _roster(conn, pid)
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setattr("src.data_bootstrap._MLB_ID_ENRICH_CAP", 2)
    monkeypatch.setattr("src.player_id_resolver.resolve_mlb_id", lambda n, t: (1000 + int(n.split()[-1]), "m"))
    from src.data_bootstrap import _enrich_mlb_ids

    _enrich_mlb_ids(_progress())

    resolved = sum(1 for pid in (10, 11, 12) if _mlb_id(pid) is not None)
    assert resolved == 2  # capped
    _status, msg = _refresh_log()
    assert "cap" in msg.lower()


_OUTAGE = (None, "lookup_player ERRORED (API outage?) - not a confirmed absence")


def test_total_outage_reports_error_not_success(temp_db, monkeypatch):
    """All lookups erroring (statsapi down) must NOT look like a healthy no-op."""
    from src.database import get_connection

    conn = get_connection()
    try:
        _player(conn, 13, "Outage One", "LAD", mlb_id=None)
        _player(conn, 14, "Outage Two", "NYY", mlb_id=None)
        _roster(conn, 13)
        _roster(conn, 14)
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setattr("src.player_id_resolver.resolve_mlb_id", lambda n, t: _OUTAGE)
    from src.data_bootstrap import _enrich_mlb_ids

    _enrich_mlb_ids(_progress())

    status, _msg = _refresh_log()
    assert status == "error"
    assert _mlb_id(13) is None and _mlb_id(14) is None


def test_partial_outage_stays_success_but_surfaces_error_count(temp_db, monkeypatch):
    from src.database import get_connection

    conn = get_connection()
    try:
        _player(conn, 15, "Resolves Fine", "LAD", mlb_id=None)
        _player(conn, 16, "Errors Out", "NYY", mlb_id=None)
        _roster(conn, 15)
        _roster(conn, 16)
        conn.commit()
    finally:
        conn.close()

    def resolver(name, team):
        return (151515, "match") if name == "Resolves Fine" else _OUTAGE

    monkeypatch.setattr("src.player_id_resolver.resolve_mlb_id", resolver)
    from src.data_bootstrap import _enrich_mlb_ids

    _enrich_mlb_ids(_progress())

    status, msg = _refresh_log()
    assert status == "success"  # real work happened
    assert "lookup error" in msg.lower()  # but the outage subset is visible
    assert _mlb_id(15) == 151515 and _mlb_id(16) is None


def test_phase_is_dispatched_in_bootstrap():
    """Structural guard: the phase must be wired into bootstrap_all_data."""
    from src.data_bootstrap import bootstrap_all_data

    src = inspect.getsource(bootstrap_all_data)
    assert "_enrich_mlb_ids" in src, "_enrich_mlb_ids is not dispatched in bootstrap_all_data"

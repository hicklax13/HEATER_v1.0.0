"""Tests for cross-team write guard + traceback logging on roster write endpoints.

All tests are DB-free; they monkeypatch clerk_configured, _owner_team, and _client
so nothing touches the live Yahoo layer or the SQLite DB.

Guard design:
 - Clerk ON + caller_team != owner  → ok=False, no Yahoo call made
 - Clerk ON + caller_team is None   → ok=False, no Yahoo call made
 - Clerk ON + caller_team == owner  → proceeds to write → ok=True
 - Clerk OFF (dormant)              → guard skipped, write proceeds (ok=True)
 - Exception in write path          → ok=False + WARNING with exc_info logged
"""

from __future__ import annotations

import logging

import pytest

from api.contracts.roster_write import AddDropRequest, LineupAssignment, LineupSetRequest, MutationResult
from api.services.roster_write_service import RosterWriteService

_OWNER_TEAM = "Team Hickey"
_OTHER_TEAM = "Other Team"

# Minimal valid requests -------------------------------------------------------


def _lineup_req() -> LineupSetRequest:
    return LineupSetRequest(
        team_name=_OWNER_TEAM,
        date="2026-06-22",
        assignments=[LineupAssignment(yahoo_player_key="469.p.11111", slot="SS")],
    )


def _add_drop_req() -> AddDropRequest:
    return AddDropRequest(add_player_key="469.p.22222", drop_player_key="469.p.33333")


# Fake client that records and optionally errors --------------------------------


class _FakeClient:
    """Records whether write methods were called; raises if instructed."""

    def __init__(self, raise_exc: Exception | None = None):
        self.set_lineup_calls: list = []
        self.add_drop_calls: list = []
        self._raise = raise_exc

    def set_lineup(self, assignments, date):
        if self._raise:
            raise self._raise
        self.set_lineup_calls.append((assignments, date))
        return {"ok": True, "applied": len(assignments), "status": None}

    def add_drop(self, add_key, drop_key):
        if self._raise:
            raise self._raise
        self.add_drop_calls.append((add_key, drop_key))
        return {"ok": True, "status": None}


# Helper: build a patched service ----------------------------------------------


def _service(
    monkeypatch,
    *,
    clerk_on: bool,
    owner_team: str | None,
    fake_client: _FakeClient | None = None,
) -> RosterWriteService:
    svc = RosterWriteService()
    monkeypatch.setattr(
        "api.services.roster_write_service.clerk_configured",
        lambda: clerk_on,
    )
    monkeypatch.setattr(
        svc,
        "_owner_team",
        lambda: owner_team,
    )
    if fake_client is not None:
        monkeypatch.setattr(svc, "_client", lambda: fake_client)
    return svc


# ===========================================================================
# Case 1: Clerk ON + wrong caller team → refused, Yahoo NOT called
# ===========================================================================


def test_set_lineup_wrong_caller_refused(monkeypatch):
    client = _FakeClient()
    svc = _service(monkeypatch, clerk_on=True, owner_team=_OWNER_TEAM, fake_client=client)

    result = svc.set_lineup(_lineup_req(), caller_team=_OTHER_TEAM)

    assert result.ok is False
    assert result.error is not None and "authorized" in result.error.lower()
    assert len(client.set_lineup_calls) == 0, "Yahoo client must NOT be called on a refused write"


def test_add_drop_wrong_caller_refused(monkeypatch):
    client = _FakeClient()
    svc = _service(monkeypatch, clerk_on=True, owner_team=_OWNER_TEAM, fake_client=client)

    result = svc.add_drop(_add_drop_req(), caller_team=_OTHER_TEAM)

    assert result.ok is False
    assert result.error is not None and "authorized" in result.error.lower()
    assert len(client.add_drop_calls) == 0, "Yahoo client must NOT be called on a refused write"


# ===========================================================================
# Case 2: Clerk ON + caller_team is None (unassigned) → refused
# ===========================================================================


def test_set_lineup_no_caller_team_refused(monkeypatch):
    client = _FakeClient()
    svc = _service(monkeypatch, clerk_on=True, owner_team=_OWNER_TEAM, fake_client=client)

    result = svc.set_lineup(_lineup_req(), caller_team=None)

    assert result.ok is False
    assert len(client.set_lineup_calls) == 0


def test_add_drop_no_caller_team_refused(monkeypatch):
    client = _FakeClient()
    svc = _service(monkeypatch, clerk_on=True, owner_team=_OWNER_TEAM, fake_client=client)

    result = svc.add_drop(_add_drop_req(), caller_team=None)

    assert result.ok is False
    assert len(client.add_drop_calls) == 0


# ===========================================================================
# Case 3: Clerk ON + caller_team == owner → write proceeds → ok=True
# ===========================================================================


def test_set_lineup_authorized_proceeds(monkeypatch):
    client = _FakeClient()
    svc = _service(monkeypatch, clerk_on=True, owner_team=_OWNER_TEAM, fake_client=client)

    result = svc.set_lineup(_lineup_req(), caller_team=_OWNER_TEAM)

    assert result.ok is True
    assert len(client.set_lineup_calls) == 1


def test_add_drop_authorized_proceeds(monkeypatch):
    client = _FakeClient()
    svc = _service(monkeypatch, clerk_on=True, owner_team=_OWNER_TEAM, fake_client=client)

    result = svc.add_drop(_add_drop_req(), caller_team=_OWNER_TEAM)

    assert result.ok is True
    assert len(client.add_drop_calls) == 1


# ===========================================================================
# Case 4: Clerk OFF (dormant) → guard skipped, write proceeds regardless of caller_team
# ===========================================================================


def test_set_lineup_clerk_off_guard_skipped(monkeypatch):
    """With Clerk off the guard must be a complete no-op — today's behavior."""
    client = _FakeClient()
    svc = _service(monkeypatch, clerk_on=False, owner_team=_OWNER_TEAM, fake_client=client)

    # Even a different caller_team must NOT be refused when Clerk is off
    result = svc.set_lineup(_lineup_req(), caller_team=_OTHER_TEAM)

    assert result.ok is True
    assert len(client.set_lineup_calls) == 1


def test_add_drop_clerk_off_guard_skipped(monkeypatch):
    client = _FakeClient()
    svc = _service(monkeypatch, clerk_on=False, owner_team=_OWNER_TEAM, fake_client=client)

    result = svc.add_drop(_add_drop_req(), caller_team=_OTHER_TEAM)

    assert result.ok is True
    assert len(client.add_drop_calls) == 1


def test_set_lineup_clerk_off_no_caller_team(monkeypatch):
    """caller_team=None with Clerk off: guard skipped, still proceeds."""
    client = _FakeClient()
    svc = _service(monkeypatch, clerk_on=False, owner_team=_OWNER_TEAM, fake_client=client)

    result = svc.set_lineup(_lineup_req(), caller_team=None)

    assert result.ok is True
    assert len(client.set_lineup_calls) == 1


# ===========================================================================
# Case 5: Exception in write path → ok=False + WARNING with traceback logged
# ===========================================================================


def test_set_lineup_exception_logs_traceback(monkeypatch, caplog):
    exc = RuntimeError("Yahoo API exploded")
    client = _FakeClient(raise_exc=exc)
    svc = _service(monkeypatch, clerk_on=False, owner_team=_OWNER_TEAM, fake_client=client)

    with caplog.at_level(logging.WARNING, logger="api.services.roster_write_service"):
        result = svc.set_lineup(_lineup_req(), caller_team=None)

    assert result.ok is False
    # Must log at WARNING level with exc_info so the traceback is captured
    warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warning_records) >= 1
    # exc_info=True means exc_info is truthy on the log record
    assert any(r.exc_info is not None for r in warning_records), (
        "Exception path must log with exc_info=True so operators see the traceback"
    )


def test_add_drop_exception_logs_traceback(monkeypatch, caplog):
    exc = RuntimeError("Yahoo API exploded")
    client = _FakeClient(raise_exc=exc)
    svc = _service(monkeypatch, clerk_on=False, owner_team=_OWNER_TEAM, fake_client=client)

    with caplog.at_level(logging.WARNING, logger="api.services.roster_write_service"):
        result = svc.add_drop(_add_drop_req(), caller_team=None)

    assert result.ok is False
    warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warning_records) >= 1
    assert any(r.exc_info is not None for r in warning_records), (
        "Exception path must log with exc_info=True so operators see the traceback"
    )


# ===========================================================================
# Refusal warning is logged (Clerk ON, wrong team)
# ===========================================================================


def test_set_lineup_refused_logs_warning(monkeypatch, caplog):
    client = _FakeClient()
    svc = _service(monkeypatch, clerk_on=True, owner_team=_OWNER_TEAM, fake_client=client)

    with caplog.at_level(logging.WARNING, logger="api.services.roster_write_service"):
        svc.set_lineup(_lineup_req(), caller_team=_OTHER_TEAM)

    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("refused" in m.lower() or "set_lineup" in m.lower() for m in warning_messages)


def test_add_drop_refused_logs_warning(monkeypatch, caplog):
    client = _FakeClient()
    svc = _service(monkeypatch, clerk_on=True, owner_team=_OWNER_TEAM, fake_client=client)

    with caplog.at_level(logging.WARNING, logger="api.services.roster_write_service"):
        svc.add_drop(_add_drop_req(), caller_team=_OTHER_TEAM)

    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("refused" in m.lower() or "add_drop" in m.lower() for m in warning_messages)


# ===========================================================================
# Default caller_team=None is accepted by the method signature (no TypeError)
# ===========================================================================


def test_set_lineup_default_caller_team_clerk_off(monkeypatch):
    """Old call site: service.set_lineup(req) with no caller_team arg.
    Must work as today when Clerk is off (dormant path)."""
    client = _FakeClient()
    svc = _service(monkeypatch, clerk_on=False, owner_team=_OWNER_TEAM, fake_client=client)

    # Call WITHOUT the caller_team kwarg — simulates the old call site
    result = svc.set_lineup(_lineup_req())

    assert result.ok is True


def test_add_drop_default_caller_team_clerk_off(monkeypatch):
    client = _FakeClient()
    svc = _service(monkeypatch, clerk_on=False, owner_team=_OWNER_TEAM, fake_client=client)

    result = svc.add_drop(_add_drop_req())

    assert result.ok is True

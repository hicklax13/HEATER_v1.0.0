"""#9 (2026-06-07): scheduler boot auto-start + cross-process single-writer gate.

`start.sh` launches a DEDICATED scheduler process at container boot so data warms
right after a deploy without waiting for the first browser session (Streamlit only
runs main() — which starts the scheduler — on a real session/websocket connect).

To keep the single-writer invariant (single replica + in-process scheduler is the
sole SQLite writer), boot-managed mode uses two env flags:

- ``HEATER_SCHEDULER_BOOT=1``      — set by start.sh for the whole container; tells
  Streamlit session processes they are READ-ONLY (the boot process owns refresh).
- ``HEATER_SCHEDULER_IS_OWNER=1``  — set ONLY on the dedicated boot process; it is
  the sole writer.

When BOOT is unset (v1/local, pytest), behavior is unchanged: start_background_refresh
starts the thread (so the Plan-4 idempotency guard + v1 byte-for-byte both hold).
"""

import os
import threading

import pytest

import src.scheduler as scheduler


def _quiesce_refresh_threads():
    """Stop the scheduler AND join any leaked 'heater-refresh' thread.

    These tests assert on the PROCESS-GLOBAL thread list, so a refresh thread
    leaked by another test file co-located on the same xdist worker (e.g. an
    AppTest that runs app.main() under MULTI_USER, which starts the scheduler
    and never stops it) would otherwise make
    ``test_reader_process_does_not_start_thread`` see a phantom thread. Wake
    every heater-refresh thread via the stop event and join it, for a
    deterministic baseline regardless of cross-file execution order under
    ``-n auto --dist loadfile`` (2026-06-16: surfaced when a new test file
    shifted the loadfile distribution; the scheduler itself is unaffected and
    passes this file in isolation)."""
    scheduler.stop_background_refresh()
    scheduler._scheduler_running = False
    scheduler._stop_event.set()
    for t in list(threading.enumerate()):
        if t.name == "heater-refresh":
            t.join(timeout=2)
    scheduler._stop_event.clear()


@pytest.fixture(autouse=True)
def _clean_scheduler(monkeypatch):
    # No-op the whole refresh cycle so the started thread never touches the
    # DB/Yahoo/network — this test is purely about thread lifecycle + the gate.
    monkeypatch.setattr(scheduler, "_refresh_once", lambda: None)
    _quiesce_refresh_threads()  # clean baseline before each test (incl. leaks)
    yield
    _quiesce_refresh_threads()
    # The boot-process entrypoint sets these directly (not via monkeypatch); pop
    # so ownership never leaks into another test.
    for var in ("HEATER_SCHEDULER_BOOT", "HEATER_SCHEDULER_IS_OWNER"):
        os.environ.pop(var, None)


def _refresh_threads():
    return [t for t in threading.enumerate() if t.name == "heater-refresh"]


def test_reader_process_does_not_start_thread(monkeypatch):
    """Boot-managed mode, non-owner (a Streamlit session) → NO refresh thread.
    The dedicated boot process is the sole writer; sessions stay read-only."""
    monkeypatch.setenv("HEATER_SCHEDULER_BOOT", "1")
    monkeypatch.delenv("HEATER_SCHEDULER_IS_OWNER", raising=False)
    scheduler.start_background_refresh()
    assert _refresh_threads() == []
    assert scheduler.is_running() is False


def test_owner_process_starts_thread(monkeypatch):
    """The dedicated boot process (owner) DOES start the refresh thread."""
    monkeypatch.setenv("HEATER_SCHEDULER_BOOT", "1")
    monkeypatch.setenv("HEATER_SCHEDULER_IS_OWNER", "1")
    scheduler.start_background_refresh()
    assert len(_refresh_threads()) == 1
    assert scheduler.is_running() is True


def test_unmanaged_mode_starts_thread(monkeypatch):
    """No boot-management (v1/local/pytest — HEATER_SCHEDULER_BOOT unset): the
    current behavior is preserved — start_background_refresh starts the thread."""
    monkeypatch.delenv("HEATER_SCHEDULER_BOOT", raising=False)
    monkeypatch.delenv("HEATER_SCHEDULER_IS_OWNER", raising=False)
    scheduler.start_background_refresh()
    assert len(_refresh_threads()) == 1
    assert scheduler.is_running() is True


def test_boot_process_noop_when_multi_user_off(monkeypatch):
    """The dedicated boot process is INERT under MULTI_USER off (v1/local): no
    writer thread, returns immediately — v1 stays byte-for-byte."""
    monkeypatch.setattr("src.auth.multi_user_enabled", lambda: False, raising=False)
    scheduler._run_as_boot_process()
    assert _refresh_threads() == []


def test_boot_process_starts_writer_when_multi_user_on(monkeypatch):
    """Under MULTI_USER on, the boot process marks itself the owner and starts the
    sole writer thread — no browser session required (the point of #9)."""
    monkeypatch.setattr("src.auth.multi_user_enabled", lambda: True, raising=False)
    monkeypatch.setattr("src.database.init_db", lambda *a, **k: None, raising=False)
    scheduler._run_as_boot_process()
    assert os.environ.get("HEATER_SCHEDULER_IS_OWNER") == "1"
    assert len(_refresh_threads()) == 1

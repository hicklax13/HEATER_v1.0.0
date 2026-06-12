"""Plan 4 regression guard: the background scheduler (the SOLE SQLite writer
under MULTI_USER) must reconnect Yahoo from the saved token each cycle and pass
the client into bootstrap_all_data — otherwise the Yahoo sync phase self-skips
and league_rosters/standings never land, so user sessions read an empty SQLite
fallback ("No league data loaded").

This complements test_plan4_scheduler_wiring.py, which only proves the thread
STARTS (its fake bootstrap is a no-op) and never proves it actually logs in.
"""

import pytest

import src.scheduler as scheduler


@pytest.fixture(autouse=True)
def _reset_scheduler_state(monkeypatch):
    """Keep module globals from leaking between tests / into other suites."""
    scheduler._scheduler_running = False
    scheduler._stop_event.clear()
    # _refresh_once drains the AI forced_refresh_queue first (a SHARED table other
    # suites write to). These tests assert the reconnect+staleness bootstrap flow
    # only, so neutralize the drain step — otherwise a sibling test's leftover
    # pending row makes the drain fire an extra bootstrap and flakes calls==1.
    monkeypatch.setattr("src.ai.refresh_queue.drain_queue", lambda: 0)
    yield
    scheduler._scheduler_running = False
    scheduler._stop_event.clear()


def _arm_loop():
    """Put the loop in a runnable state (as start_background_refresh would)."""
    scheduler._scheduler_running = True
    scheduler._stop_event.clear()


def test_refresh_loop_reconnects_and_passes_client(monkeypatch):
    """One cycle: reconnect via try_reconnect_yahoo() and hand the client to
    bootstrap_all_data(force=False)."""
    sentinel_client = object()
    captured = {}

    monkeypatch.setattr("src.yahoo_api.try_reconnect_yahoo", lambda: sentinel_client)

    def fake_bootstrap(*args, **kwargs):
        captured["yahoo_client"] = kwargs.get("yahoo_client")
        captured["force"] = kwargs.get("force")
        captured["calls"] = captured.get("calls", 0) + 1
        # Exit after exactly one iteration (set() makes _stop_event.wait return
        # True immediately, so we never hit the 300s sleep).
        scheduler._scheduler_running = False
        scheduler._stop_event.set()
        return {}

    monkeypatch.setattr("src.data_bootstrap.bootstrap_all_data", fake_bootstrap)

    _arm_loop()
    scheduler._refresh_loop()

    assert captured["calls"] == 1
    assert captured["yahoo_client"] is sentinel_client
    assert captured["force"] is False


def test_refresh_loop_survives_reconnect_exception(monkeypatch):
    """If reconnect raises, the loop must NOT crash — it degrades to
    yahoo_client=None (the Yahoo phase then self-skips) and still bootstraps."""
    captured = {}

    def boom():
        raise RuntimeError("yahoo down")

    monkeypatch.setattr("src.yahoo_api.try_reconnect_yahoo", boom)

    def fake_bootstrap(*args, **kwargs):
        captured["yahoo_client"] = kwargs.get("yahoo_client")
        captured["calls"] = captured.get("calls", 0) + 1
        scheduler._scheduler_running = False
        scheduler._stop_event.set()
        return {}

    monkeypatch.setattr("src.data_bootstrap.bootstrap_all_data", fake_bootstrap)

    _arm_loop()
    scheduler._refresh_loop()  # must return cleanly, not raise

    assert captured["calls"] == 1
    assert captured["yahoo_client"] is None

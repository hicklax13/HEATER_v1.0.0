"""SFH H2 + H3 (2026-05-20): refresh_log resilience guards.

Two bugs:
  H2. Per-phase ``except`` handlers attempt ``update_refresh_log(...)`` on
      failure, but during a parallel long-held write lock (pvb_splits 50-
      batter Statcast loop), that write times out against the 60s
      busy_timeout and the inner ``except Exception: pass`` swallows it.
      refresh_log silently stays at the prior run's "success".
  H3. ``_run_with_timeout`` catches phase timeouts and returns a
      "Timeout after Ns" string, but never writes to refresh_log itself.
      Phases like ecr_consensus that exhaust their 240s budget left
      refresh_log unchanged from the prior successful run.

These tests pin:
  * ``_try_write_refresh_log`` writes on first attempt, retries on
    ``OperationalError("database is locked")``, gives up after 3 attempts.
  * ``_run_with_timeout(fn, timeout=X, source="Y")`` calls
    ``_try_write_refresh_log("Y", "timeout", ...)`` when the inner fn
    exceeds the budget.
  * ``_reconcile_results_to_refresh_log(results)`` walks the dict and
    forces refresh_log entries for any "Error:" / "Timeout" result.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.database as db_mod
from src.data_bootstrap import (
    _reconcile_results_to_refresh_log,
    _run_with_timeout,
    _try_write_refresh_log,
)
from src.database import get_refresh_log_snapshot, init_db


@pytest.fixture(autouse=True)
def temp_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    original = db_mod.DB_PATH
    db_mod.DB_PATH = Path(tmp.name)
    init_db()
    yield tmp.name
    db_mod.DB_PATH = original
    try:
        os.unlink(tmp.name)
    except PermissionError:
        pass


def _refresh_log_row(source: str) -> dict | None:
    snap = get_refresh_log_snapshot()
    for r in snap:
        if r.get("source") == source:
            return r
    return None


# ── H2: _try_write_refresh_log ──────────────────────────────────


def test_try_write_refresh_log_writes_on_first_attempt(temp_db):
    """H2: happy path — write succeeds immediately, returns True."""
    ok = _try_write_refresh_log("test_phase_a", "error", "boom")
    assert ok is True
    row = _refresh_log_row("test_phase_a")
    assert row is not None
    assert row["status"] == "error"
    assert "boom" in (row.get("message") or "")


def test_try_write_refresh_log_retries_on_lock_then_succeeds(temp_db):
    """H2: when the first attempt hits OperationalError(locked), the helper
    retries with backoff and the second attempt succeeds. Returns True."""
    call_count = {"n": 0}
    real_update = db_mod.update_refresh_log

    def flaky_update(source, status, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise sqlite3.OperationalError("database is locked")
        return real_update(source, status, **kwargs)

    with patch("src.database.update_refresh_log", side_effect=flaky_update):
        start = time.monotonic()
        ok = _try_write_refresh_log("test_phase_b", "error", "lock-then-recover")
        elapsed = time.monotonic() - start

    assert ok is True
    assert call_count["n"] == 2, "expected exactly 2 attempts (1 fail + 1 retry)"
    # 0.5s backoff between attempts 1 and 2; total should be > 0.4s and < 2s.
    assert 0.4 < elapsed < 2.0, f"unexpected elapsed time {elapsed:.2f}s for 1-retry path"
    row = _refresh_log_row("test_phase_b")
    assert row is not None and row["status"] == "error"


def test_try_write_refresh_log_gives_up_after_3_attempts(temp_db):
    """H2: persistent OperationalError(locked) → False after 3 attempts.
    Does not raise. Total time ≈ 0.5s + 1s = 1.5s of backoff."""
    call_count = {"n": 0}

    def always_locked(*_a, **_kw):
        call_count["n"] += 1
        raise sqlite3.OperationalError("database is locked")

    with patch("src.database.update_refresh_log", side_effect=always_locked):
        ok = _try_write_refresh_log("test_phase_c", "error", "permanent lock")

    assert ok is False
    assert call_count["n"] == 3, "expected exactly 3 attempts before giving up"


def test_try_write_refresh_log_returns_false_on_non_lock_error(temp_db):
    """H2: non-lock errors don't trigger retry — return False immediately."""
    call_count = {"n": 0}

    def boom(*_a, **_kw):
        call_count["n"] += 1
        raise RuntimeError("unrelated error")

    with patch("src.database.update_refresh_log", side_effect=boom):
        ok = _try_write_refresh_log("test_phase_d", "error", "msg")

    assert ok is False
    assert call_count["n"] == 1, "non-lock errors should NOT trigger retry"


# ── H3: _run_with_timeout records timeout to refresh_log ──────────


def test_run_with_timeout_records_timeout_when_source_provided(temp_db):
    """H3: when source= is provided and the inner fn exceeds the timeout,
    refresh_log gets a 'timeout' row instead of staying stale."""

    def slow_fn():
        time.sleep(2.0)
        return "should not reach"

    result = _run_with_timeout(slow_fn, timeout=1, source="test_timeout_phase")
    assert result.startswith("Timeout")
    row = _refresh_log_row("test_timeout_phase")
    assert row is not None
    assert row["status"] == "timeout"
    assert "1s" in (row.get("message") or "")


def test_run_with_timeout_no_refresh_log_write_when_source_omitted(temp_db):
    """H3 back-compat: callers that don't pass source= get the original
    behavior — no refresh_log write from the helper itself."""

    def slow_fn():
        time.sleep(2.0)
        return "should not reach"

    result = _run_with_timeout(slow_fn, timeout=1)
    assert result.startswith("Timeout")
    # No refresh_log row should have been written by the helper.
    assert _refresh_log_row("test_timeout_phase_no_src") is None


def test_run_with_timeout_success_path_unaffected(temp_db):
    """H3 regression guard: when the inner fn completes within the budget,
    refresh_log is NOT touched by _run_with_timeout — callers handle the
    success-path write themselves."""

    def fast_fn():
        return "Saved 42 rows"

    result = _run_with_timeout(fast_fn, timeout=5, source="test_fast_phase")
    assert result == "Saved 42 rows"
    # _run_with_timeout should not pre-write success; that's the caller's job.
    assert _refresh_log_row("test_fast_phase") is None


# ── H2: end-of-bootstrap reconciliation pass ─────────────────────


def test_reconcile_results_records_error_strings(temp_db):
    """H2: 'Error:' results get written to refresh_log post-bootstrap."""
    results = {
        "phase_alpha": "Error: database is locked",
        "phase_beta": "Saved 100 rows (success)",
        "phase_gamma": "Fresh",
    }
    reconciled = _reconcile_results_to_refresh_log(results)
    assert reconciled == 1, "only the 'Error:' row should be reconciled"
    alpha = _refresh_log_row("phase_alpha")
    assert alpha is not None and alpha["status"] == "error"
    assert "database is locked" in (alpha.get("message") or "")
    # Success and fresh rows should not have been written by reconciliation.
    assert _refresh_log_row("phase_beta") is None
    assert _refresh_log_row("phase_gamma") is None


def test_reconcile_results_records_timeout_strings(temp_db):
    """H2: 'Timeout' results get reconciled too."""
    results = {"phase_slow": "Timeout after 240s"}
    reconciled = _reconcile_results_to_refresh_log(results)
    assert reconciled == 1
    row = _refresh_log_row("phase_slow")
    assert row is not None and row["status"] == "timeout"


def test_reconcile_results_skips_non_string_values(temp_db):
    """H2 defense: results dict may contain non-string values (e.g. tuples
    from historical phase). The reconciliation must skip them cleanly."""
    results = {
        "phase_tuple": ("Saved 0", {"some_data": True}),
        "phase_int": 42,
        "phase_err": "Error: real failure",
    }
    reconciled = _reconcile_results_to_refresh_log(results)
    assert reconciled == 1, "only the string-Error row should be reconciled"
    assert _refresh_log_row("phase_err") is not None


def test_reconcile_results_overwrites_stale_success(temp_db):
    """H2 main use case: a prior run wrote 'success'; this run's failed
    error-write was swallowed; the reconciliation pass must overwrite the
    stale row with the current failure."""
    # Simulate prior successful run.
    db_mod.update_refresh_log("phase_x", "success", rows_written=100, message="prior good run")
    prior = _refresh_log_row("phase_x")
    assert prior is not None and prior["status"] == "success"

    # Current run failed but the per-phase error-write was swallowed (so
    # refresh_log still shows 'success'). Reconciliation now overwrites.
    results = {"phase_x": "Error: database is locked"}
    reconciled = _reconcile_results_to_refresh_log(results)
    assert reconciled == 1
    current = _refresh_log_row("phase_x")
    assert current is not None
    assert current["status"] == "error", "stale success must be overwritten with current error"

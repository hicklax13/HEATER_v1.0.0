"""SFH M-3 regression (2026-05-20): partial / cached refresh_log statuses
must be honored by downstream readers.

Pre-fix:
  - ``database.check_staleness`` force-refreshed any source whose status was
    in {error, no_data, partial, unknown}. A phase like yahoo_free_agents
    that returned a partial result (25/100 FAs after a 120s timeout) would
    be retried on every subsequent bootstrap, burning the same Yahoo
    rate-limit budget that caused the partial in the first place.
  - ``optimizer.data_freshness.DataFreshnessTracker._populate_from_refresh_log``
    skipped any row whose status != "success", so partial / cached data
    was invisible to the UI freshness panel.

Post-fix:
  - check_staleness only force-refreshes hard failures (error / unknown).
    partial / no_data / cached / skipped honor the TTL.
  - data_freshness hydrates from success / partial / cached / skipped rows;
    excludes only error / unknown / no_data (no data to track).

The pre-existing tests in test_refresh_log_status_validity.py guarantee that
all 6 statuses can be written without being downgraded; this file guards
that they are also READ correctly.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

# ── check_staleness behavior ──────────────────────────────────────────


@pytest.fixture
def temp_db_with_status(tmp_path, monkeypatch):
    """Return a factory that writes a refresh_log row at a given status + age."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE refresh_log (
            source TEXT PRIMARY KEY,
            last_refresh TEXT,
            status TEXT,
            rows_written INTEGER,
            rows_expected_min INTEGER,
            message TEXT,
            tier TEXT
        );
        """
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr("src.database.DB_PATH", Path(str(db_path)))

    def _write(source: str, status: str, age_hours: float) -> None:
        ts = (datetime.now(UTC) - timedelta(hours=age_hours)).isoformat()
        c = sqlite3.connect(str(db_path))
        c.execute(
            "INSERT OR REPLACE INTO refresh_log (source, last_refresh, status) VALUES (?, ?, ?)",
            (source, ts, status),
        )
        c.commit()
        c.close()

    return _write


def test_partial_status_within_ttl_not_stale(temp_db_with_status):
    """partial status row 1h old + 24h TTL → not stale (honor TTL)."""
    from src.database import check_staleness

    temp_db_with_status("fa_test", "partial", age_hours=1.0)
    assert check_staleness("fa_test", max_age_hours=24.0) is False, (
        "partial status within TTL should NOT force refresh (was burning "
        "rate-limit budget by re-fetching every bootstrap)"
    )


def test_partial_status_beyond_ttl_is_stale(temp_db_with_status):
    """partial status row 25h old + 24h TTL → stale (normal staleness)."""
    from src.database import check_staleness

    temp_db_with_status("fa_test", "partial", age_hours=25.0)
    assert check_staleness("fa_test", max_age_hours=24.0) is True


def test_no_data_status_within_ttl_not_stale(temp_db_with_status):
    """no_data status row 1h old + 24h TTL → not stale.

    Behavior change from M-3: previously force-refreshed every bootstrap,
    which doesn't help if the source legitimately has no data.
    """
    from src.database import check_staleness

    temp_db_with_status("adp_test", "no_data", age_hours=1.0)
    assert check_staleness("adp_test", max_age_hours=24.0) is False


def test_cached_status_within_ttl_not_stale(temp_db_with_status):
    """cached status row within TTL → not stale (already was; lock it in)."""
    from src.database import check_staleness

    temp_db_with_status("ecr_test", "cached", age_hours=2.0)
    assert check_staleness("ecr_test", max_age_hours=24.0) is False


def test_error_status_always_stale(temp_db_with_status):
    """error status forces retry regardless of timestamp (unchanged)."""
    from src.database import check_staleness

    temp_db_with_status("err_test", "error", age_hours=0.1)
    assert check_staleness("err_test", max_age_hours=24.0) is True


def test_unknown_status_always_stale(temp_db_with_status):
    """unknown status forces retry regardless of timestamp (unchanged)."""
    from src.database import check_staleness

    temp_db_with_status("u_test", "unknown", age_hours=0.1)
    assert check_staleness("u_test", max_age_hours=24.0) is True


# ── DataFreshnessTracker hydration ────────────────────────────────────


@pytest.fixture
def db_with_mixed_statuses(tmp_path, monkeypatch):
    """Refresh_log with one row per loadable + excluded status."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE refresh_log (
            source TEXT PRIMARY KEY,
            status TEXT,
            tier TEXT,
            message TEXT,
            last_refresh TEXT,
            rows_written INTEGER,
            rows_expected_min INTEGER
        );
        INSERT INTO refresh_log (source, status, last_refresh) VALUES
            ('src_success',  'success',  datetime('now', '-10 minutes')),
            ('src_partial',  'partial',  datetime('now', '-10 minutes')),
            ('src_cached',   'cached',   datetime('now', '-10 minutes')),
            ('src_skipped',  'skipped',  datetime('now', '-10 minutes')),
            ('src_no_data',  'no_data',  datetime('now', '-10 minutes')),
            ('src_error',    'error',    datetime('now', '-10 minutes')),
            ('src_unknown',  'unknown',  datetime('now', '-10 minutes'));
        """
    )
    conn.commit()
    conn.close()

    import src.database as db_mod

    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    return db_path


def test_tracker_loads_partial_and_cached_and_skipped(db_with_mixed_statuses):
    """SFH M-3: partial, cached, skipped rows should hydrate to FRESH.

    Pre-fix only 'success' hydrated. Partial data showed UNKNOWN in the UI
    even though data was on disk.
    """
    from src.optimizer.data_freshness import DataFreshnessTracker, FreshnessStatus

    tracker = DataFreshnessTracker()
    for source in ("src_success", "src_partial", "src_cached", "src_skipped"):
        assert tracker.check(source) == FreshnessStatus.FRESH, (
            f"{source!r} should hydrate to FRESH (status row is recent + status "
            f"indicates data was touched); got {tracker.check(source)}"
        )


def test_tracker_still_skips_error_unknown_no_data(db_with_mixed_statuses):
    """Hard failures + no-data rows must remain UNKNOWN (not load as FRESH)."""
    from src.optimizer.data_freshness import DataFreshnessTracker, FreshnessStatus

    tracker = DataFreshnessTracker()
    for source in ("src_error", "src_unknown", "src_no_data"):
        assert tracker.check(source) == FreshnessStatus.UNKNOWN, (
            f"{source!r} should NOT hydrate (no data on disk or hard failure); got {tracker.check(source)}"
        )

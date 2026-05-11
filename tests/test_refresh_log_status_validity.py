"""SF: All status strings used by bootstrap phases must be accepted by the
refresh_log validator without being downgraded to 'unknown'.

The bootstrap code in src/data_bootstrap.py writes several status strings
to refresh_log via update_refresh_log() — including "cached" (e.g. ECR
consensus when cache is fresh) and "skipped". If the validator's allowlist
doesn't include them, they are silently downgraded to "unknown" and the
UI Data Freshness panel can no longer distinguish cached-from-fresh from
genuine no-data.
"""

import sqlite3
from pathlib import Path

import pytest

from src.database import get_refresh_status, update_refresh_log

# Statuses we have observed the bootstrap actually writing.
# - success / partial / no_data / error: standard outcomes
# - cached: written by ecr_consensus when the cache is reused (data_bootstrap.py:936)
# - skipped: written by pvb_splits / depth_charts when a phase short-circuits
EXPECTED_VALID = ["success", "partial", "cached", "skipped", "no_data", "error"]


@pytest.mark.parametrize("status", EXPECTED_VALID)
def test_status_round_trips(tmp_path, monkeypatch, status):
    """Each observed-in-production status must round-trip through the DB."""
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

    # Patch the module-level DB_PATH so get_connection() uses our tmp DB.
    # DB_PATH is a pathlib.Path in production; keep it a Path here so
    # `DB_PATH.parent.mkdir(...)` inside get_connection() still works.
    monkeypatch.setattr("src.database.DB_PATH", Path(str(db_path)))

    update_refresh_log(f"src_{status}", status, message=f"test of {status}")

    info = get_refresh_status(f"src_{status}")
    assert info is not None, f"refresh_log row not found for status={status}"
    assert info["status"] == status, f"status was downgraded: wrote {status!r}, got back {info['status']!r}"

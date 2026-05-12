"""INFRA-F4 fix: DataFreshnessTracker hydrates from refresh_log on init."""

import sqlite3

import pytest


@pytest.fixture
def test_db_with_refresh_log(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.executescript("""
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
            ('projections', 'success', datetime('now', '-2 hours')),
            ('players', 'success', datetime('now', '-30 minutes')),
            ('game_day', 'success', datetime('now', '-15 minutes')),
            ('historical_stats', 'no_data', datetime('now', '-1 minute'));
    """)
    conn.commit()
    conn.close()

    import src.database as db_mod

    # NOTE: get_connection() calls DB_PATH.parent.mkdir(...) so this must
    # remain a pathlib.Path, not a str — passing a str raises AttributeError
    # and the tracker silently falls back to UNKNOWN.
    monkeypatch.setattr(db_mod, "DB_PATH", db_path)
    return db_path


def test_tracker_hydrates_from_refresh_log_on_init(test_db_with_refresh_log):
    """A fresh DataFreshnessTracker() should report FRESH (not UNKNOWN) for
    sources that have recent successful entries in refresh_log."""
    from src.optimizer.data_freshness import DataFreshnessTracker, FreshnessStatus

    tracker = DataFreshnessTracker()
    # players refreshed 30 min ago, within any TTL → FRESH
    assert tracker.check("players") == FreshnessStatus.FRESH, (
        "INFRA-F4 regression: DataFreshnessTracker did not hydrate "
        "from refresh_log on init. tracker.check('players') returned "
        f"{tracker.check('players')} instead of FRESH."
    )


def test_tracker_skips_non_success_entries(test_db_with_refresh_log):
    """no_data / error / partial rows should not produce FRESH status — only
    'success' (or its variants that imply data is present) hydrate."""
    from src.optimizer.data_freshness import DataFreshnessTracker, FreshnessStatus

    tracker = DataFreshnessTracker()
    # historical_stats was 'no_data' → should remain UNKNOWN, not FRESH
    assert tracker.check("historical_stats") == FreshnessStatus.UNKNOWN

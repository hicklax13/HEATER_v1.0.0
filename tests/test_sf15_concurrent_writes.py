"""SF-15: statcast_archive concurrent writes must not raise UNIQUE constraint violations.

Bug: _bootstrap_batting_stats and _bootstrap_sprint_speed both write to
statcast_archive in the same Phase 22-24 parallel pool. The existing
SELECT-then-INSERT pattern is racy: two threads can each pass the SELECT
1 check, then both INSERTs collide on UNIQUE(player_id, season).

Fix: Use UPSERT (INSERT ... ON CONFLICT(player_id, season) DO UPDATE)
in both INSERT statements.
"""

import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch

import pytest


@pytest.fixture
def temp_db(tmp_path):
    db_path = tmp_path / "test.db"
    with patch("src.database.DB_PATH", db_path):
        from src.database import init_db, upsert_player_bulk

        init_db()
        upsert_player_bulk([{"name": "Mike Trout", "team": "LAA", "positions": "OF", "is_hitter": True}])
        yield db_path


def _read_insert_sql(file_path: str, anchor: str) -> str:
    """Read the INSERT INTO statcast_archive line near the given anchor for a fix verification."""
    text = open(file_path, encoding="utf-8").read()
    pos = text.index(anchor)
    snippet = text[pos : pos + 1000]
    return snippet


class TestSF15ConcurrentWrites:
    def test_batting_stats_insert_uses_upsert(self):
        """The INSERT in _bootstrap_batting_stats must use ON CONFLICT or INSERT OR REPLACE."""
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / "src" / "data_bootstrap.py"
        text = path.read_text(encoding="utf-8")
        idx = text.find("def _bootstrap_batting_stats")
        end = text.find("\ndef ", idx + 1)
        block = text[idx:end]
        assert "INSERT INTO statcast_archive" in block or "INSERT OR REPLACE INTO statcast_archive" in block, (
            "_bootstrap_batting_stats must still write to statcast_archive"
        )
        assert "ON CONFLICT" in block or "INSERT OR REPLACE" in block, (
            "_bootstrap_batting_stats INSERT into statcast_archive must use ON CONFLICT or INSERT OR REPLACE "
            "to avoid the UNIQUE(player_id, season) race with _bootstrap_sprint_speed"
        )

    def test_sprint_speed_insert_uses_upsert(self):
        """The INSERT in _bootstrap_sprint_speed must use ON CONFLICT or INSERT OR REPLACE."""
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / "src" / "data_bootstrap.py"
        text = path.read_text(encoding="utf-8")
        idx = text.find("def _bootstrap_sprint_speed")
        end = text.find("\ndef ", idx + 1)
        block = text[idx:end]
        assert "INSERT INTO statcast_archive" in block or "INSERT OR REPLACE INTO statcast_archive" in block
        assert "ON CONFLICT" in block or "INSERT OR REPLACE" in block, (
            "_bootstrap_sprint_speed INSERT into statcast_archive must use ON CONFLICT or INSERT OR REPLACE "
            "to avoid the UNIQUE(player_id, season) race with _bootstrap_batting_stats"
        )

    def test_concurrent_writes_no_integrity_error(self, temp_db):
        """Two threads writing to statcast_archive for the same (player_id, season) must not raise IntegrityError."""
        from src.database import get_connection

        # Get the player_id for Mike Trout
        conn = get_connection()
        try:
            row = conn.execute("SELECT player_id FROM players WHERE name = 'Mike Trout'").fetchone()
            assert row is not None
            pid = row[0]
        finally:
            conn.close()

        season = 2026

        def _write_a():
            conn = get_connection()
            try:
                conn.execute(
                    "INSERT INTO statcast_archive (player_id, season, babip) VALUES (?, ?, ?) "
                    "ON CONFLICT(player_id, season) DO UPDATE SET babip = excluded.babip",
                    (pid, season, 0.301),
                )
                conn.commit()
            finally:
                conn.close()
            return "a-ok"

        def _write_b():
            conn = get_connection()
            try:
                conn.execute(
                    "INSERT INTO statcast_archive (player_id, season, sprint_speed) VALUES (?, ?, ?) "
                    "ON CONFLICT(player_id, season) DO UPDATE SET sprint_speed = excluded.sprint_speed",
                    (pid, season, 28.7),
                )
                conn.commit()
            finally:
                conn.close()
            return "b-ok"

        with ThreadPoolExecutor(max_workers=2) as ex:
            futs = [ex.submit(_write_a), ex.submit(_write_b)]
            for f in as_completed(futs):
                # IntegrityError or any exception fails the test
                try:
                    f.result()
                except sqlite3.IntegrityError as exc:
                    pytest.fail(f"UNIQUE constraint violation on concurrent writes: {exc}")

        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT babip, sprint_speed FROM statcast_archive WHERE player_id = ? AND season = ?",
                (pid, season),
            ).fetchone()
        finally:
            conn.close()
        assert row is not None
        assert row[0] is not None
        assert row[1] is not None

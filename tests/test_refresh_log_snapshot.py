"""E4: get_refresh_log_snapshot returns full refresh_log state."""

import sqlite3
from pathlib import Path

import pytest

from src.database import get_refresh_log_snapshot, update_refresh_log


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE refresh_log (
            source TEXT PRIMARY KEY, last_refresh TEXT, status TEXT,
            rows_written INTEGER, rows_expected_min INTEGER,
            message TEXT, tier TEXT
        );
        """
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr("src.database.DB_PATH", Path(str(db_path)))
    return db_path


def test_empty_snapshot_returns_empty_list(temp_db):
    assert get_refresh_log_snapshot() == []


def test_snapshot_returns_all_sources(temp_db):
    update_refresh_log("phase_a", "success", message="hello", tier="primary")
    update_refresh_log("phase_b", "cached", message="all hit cache", tier="emergency")
    update_refresh_log("phase_c", "no_data", message="nothing fetched")
    snap = get_refresh_log_snapshot()
    # Scope assertions to THIS test's sources. Under the non-sharded `-n auto`
    # Coverage Floor run, a sibling test's update_refresh_log (e.g. the
    # data_bootstrap "news_intelligence" write, src/data_bootstrap.py) can leak
    # into the shared snapshot via test-ordering races on the module-global
    # DB_PATH. Asserting an exact GLOBAL count (`len(snap) == 3`) was therefore
    # flaky (seen: `assert 4 == 3`). Validate presence + correctness + relative
    # sort order of our own rows instead — immune to leaked sources, same intent.
    mine = [r for r in snap if r["source"] in {"phase_a", "phase_b", "phase_c"}]
    assert len(mine) == 3
    sources = {r["source"] for r in mine}
    assert sources == {"phase_a", "phase_b", "phase_c"}
    # Sorted by source name (among our rows)
    assert [r["source"] for r in mine] == ["phase_a", "phase_b", "phase_c"]
    # Tier propagates
    by_src = {r["source"]: r for r in mine}
    assert by_src["phase_a"]["tier"] == "primary"
    assert by_src["phase_b"]["tier"] == "emergency"


def test_snapshot_handles_missing_columns_gracefully(tmp_path, monkeypatch):
    """On a very old DB (pre-T3 migration) some columns may not exist."""
    db_path = tmp_path / "old.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("CREATE TABLE refresh_log (source TEXT PRIMARY KEY, last_refresh TEXT, status TEXT)")
    conn.commit()
    conn.close()
    monkeypatch.setattr("src.database.DB_PATH", Path(str(db_path)))
    # Should NOT raise even though tier/message/rows_* columns missing
    snap = get_refresh_log_snapshot()
    assert isinstance(snap, list)  # gracefully degrades

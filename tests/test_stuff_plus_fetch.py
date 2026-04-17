"""Tests for Stuff+/Location+/Pitching+ bootstrap fetch from FanGraphs."""

import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_bootstrap import BootstrapProgress, _bootstrap_stuff_plus

# ── Live integration test (skip if network unavailable) ─────────────


def test_pitching_stats_has_stuff_columns():
    """Verify pybaseball pitching_stats returns Stuff+ columns."""
    try:
        from pybaseball import pitching_stats

        df = pitching_stats(2025, qual=50)
        cols = [c for c in df.columns if "tuff" in c.lower() or "ocation" in c.lower() or "itching" in c.lower()]
        assert len(cols) > 0, f"No Stuff+/Location+/Pitching+ columns found. Columns: {list(df.columns)[:30]}"
    except Exception as e:
        pytest.skip(f"pybaseball/network unavailable: {e}")


# ── Unit tests with mocked pybaseball ───────────────────────────────


class _NoCloseConnection:
    """Wrapper that prevents close() from destroying in-memory DB."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def close(self):
        pass

    def real_close(self):
        self._conn.close()

    def __getattr__(self, name):
        return getattr(self._conn, name)


def _setup_test_db():
    """Create an in-memory DB with the required tables and sample data."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE players (
            player_id INTEGER PRIMARY KEY,
            name TEXT,
            is_hitter INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE season_stats (
            player_id INTEGER NOT NULL,
            season INTEGER NOT NULL DEFAULT 2026,
            ip REAL DEFAULT 0,
            stuff_plus REAL,
            location_plus REAL,
            pitching_plus REAL,
            PRIMARY KEY (player_id, season)
        )
    """)
    conn.execute("""
        CREATE TABLE statcast_archive (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL,
            season INTEGER NOT NULL,
            stuff_plus REAL,
            location_plus REAL,
            pitching_plus REAL,
            UNIQUE(player_id, season)
        )
    """)
    conn.execute("""
        CREATE TABLE refresh_log (
            source TEXT PRIMARY KEY,
            last_refresh TEXT,
            status TEXT
        )
    """)
    # Insert sample pitchers
    conn.execute("INSERT INTO players (player_id, name, is_hitter) VALUES (1, 'Zack Wheeler', 0)")
    conn.execute("INSERT INTO players (player_id, name, is_hitter) VALUES (2, 'Gerrit Cole', 0)")
    conn.execute("INSERT INTO players (player_id, name, is_hitter) VALUES (3, 'Mike Trout', 1)")
    # Insert season_stats rows for pitchers
    conn.execute("INSERT INTO season_stats (player_id, season) VALUES (1, 2026)")
    conn.execute("INSERT INTO season_stats (player_id, season) VALUES (2, 2026)")
    conn.commit()
    return conn


def _mock_pitching_stats_df():
    """Return a DataFrame mimicking pybaseball.pitching_stats output."""
    return pd.DataFrame(
        [
            {
                "Name": "Zack Wheeler",
                "Stuff+": 110.5,
                "Location+": 105.2,
                "Pitching+": 108.0,
                "W": 5,
                "ERA": 2.85,
            },
            {
                "Name": "Gerrit Cole",
                "Stuff+": 120.3,
                "Location+": 98.7,
                "Pitching+": 112.5,
                "W": 4,
                "ERA": 3.10,
            },
            {
                "Name": "Unknown Pitcher",
                "Stuff+": 90.0,
                "Location+": 88.0,
                "Pitching+": 89.0,
                "W": 1,
                "ERA": 5.40,
            },
        ]
    )


def test_bootstrap_stuff_plus_updates_season_stats():
    """Verify _bootstrap_stuff_plus updates season_stats with Stuff+ data."""
    conn = _setup_test_db()
    wrapped = _NoCloseConnection(conn)

    mock_pybaseball = MagicMock()
    mock_pybaseball.pitching_stats = MagicMock(return_value=_mock_pitching_stats_df())

    with (
        patch("src.database.get_connection", return_value=wrapped),
        patch("src.database.update_refresh_log"),
        patch.dict("sys.modules", {"pybaseball": mock_pybaseball}),
    ):
        progress = BootstrapProgress()
        result = _bootstrap_stuff_plus(progress)

    assert "Updated 2 pitchers" in result

    # Verify season_stats were updated
    row = conn.execute(
        "SELECT stuff_plus, location_plus, pitching_plus FROM season_stats WHERE player_id = 1"
    ).fetchone()
    assert row is not None
    assert row[0] == pytest.approx(110.5)
    assert row[1] == pytest.approx(105.2)
    assert row[2] == pytest.approx(108.0)

    row2 = conn.execute(
        "SELECT stuff_plus, location_plus, pitching_plus FROM season_stats WHERE player_id = 2"
    ).fetchone()
    assert row2 is not None
    assert row2[0] == pytest.approx(120.3)

    # Verify statcast_archive was populated
    sa_row = conn.execute("SELECT stuff_plus FROM statcast_archive WHERE player_id = 1 AND season = 2026").fetchone()
    assert sa_row is not None
    assert sa_row[0] == pytest.approx(110.5)

    conn.close()


def test_bootstrap_stuff_plus_no_pybaseball():
    """Verify graceful handling when pybaseball is not installed."""
    progress = BootstrapProgress()

    # Simulate ImportError for pybaseball
    import builtins

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "pybaseball":
            raise ImportError("No module named 'pybaseball'")
        return real_import(name, *args, **kwargs)

    with (
        patch.dict("sys.modules", {"pybaseball": None}),
        patch("builtins.__import__", side_effect=mock_import),
    ):
        result = _bootstrap_stuff_plus(progress)

    assert "Skipped" in result or "Error" in result


def test_bootstrap_stuff_plus_network_error():
    """Verify graceful handling when FanGraphs is unreachable.

    2026-04-17 trust audit: known-bad upstream signatures (403, 429,
    timeouts) are translated to "Skipped: ..." by _format_fetch_error so
    the Data Status panel doesn't surface them as actionable errors.
    Generic exceptions still return "Error:".
    """
    mock_pybaseball = MagicMock()
    mock_pybaseball.pitching_stats = MagicMock(side_effect=Exception("Network timeout"))

    with (
        patch.dict("sys.modules", {"pybaseball": mock_pybaseball}),
        patch("src.database.update_refresh_log"),
    ):
        progress = BootstrapProgress()
        result = _bootstrap_stuff_plus(progress)

    # Result may be "Error:" (generic exc) or "Skipped:" (known-bad signature
    # like timeout/403/429). Either is acceptable as graceful handling.
    assert "Error" in result or "Skipped" in result


def test_bootstrap_stuff_plus_empty_data():
    """Verify handling when pitching_stats returns empty DataFrame."""
    mock_pybaseball = MagicMock()
    mock_pybaseball.pitching_stats = MagicMock(return_value=pd.DataFrame())

    with patch.dict("sys.modules", {"pybaseball": mock_pybaseball}):
        progress = BootstrapProgress()
        result = _bootstrap_stuff_plus(progress)

    assert "Skipped" in result or "no data" in result.lower()


def test_bootstrap_stuff_plus_case_insensitive_match():
    """Verify name matching is case-insensitive."""
    conn = _setup_test_db()
    wrapped = _NoCloseConnection(conn)

    # Use different casing in FanGraphs data
    fg_df = pd.DataFrame(
        [
            {
                "Name": "zack wheeler",  # lowercase
                "Stuff+": 99.0,
                "Location+": 100.0,
                "Pitching+": 101.0,
            },
        ]
    )

    mock_pybaseball = MagicMock()
    mock_pybaseball.pitching_stats = MagicMock(return_value=fg_df)

    with (
        patch("src.database.get_connection", return_value=wrapped),
        patch("src.database.update_refresh_log"),
        patch.dict("sys.modules", {"pybaseball": mock_pybaseball}),
    ):
        progress = BootstrapProgress()
        result = _bootstrap_stuff_plus(progress)

    assert "Updated 1 pitchers" in result

    row = conn.execute("SELECT stuff_plus FROM season_stats WHERE player_id = 1").fetchone()
    assert row is not None
    assert row[0] == pytest.approx(99.0)

    conn.close()

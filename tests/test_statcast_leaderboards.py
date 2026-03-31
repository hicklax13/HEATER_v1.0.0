"""Tests for Statcast leaderboard functions in src/live_stats.py."""

import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.live_stats import fetch_statcast_leaderboards, save_statcast_leaderboards_to_db


class _NoCloseConnection:
    """Wrapper around sqlite3.Connection that makes close() a no-op.

    sqlite3.Connection.close is read-only, so we cannot monkey-patch it.
    This wrapper delegates everything except close() to the real connection.
    """

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def close(self):
        pass  # no-op — prevents the function's finally block from destroying the DB

    def real_close(self):
        self._conn.close()

    def __getattr__(self, name):
        return getattr(self._conn, name)


# ── Sample data fixtures ─────────────────────────────────────────────


def _mock_batter_df():
    return pd.DataFrame(
        [
            {
                "player_id": 545361,
                "player_name": "Mike Trout",
                "est_ba": 0.280,
                "est_slg": 0.520,
                "est_woba": 0.380,
                "brl_percent": 15.0,
                "avg_hit_speed": 92.5,
                "hard_hit_percent": 45.0,
            },
            {
                "player_id": 660271,
                "player_name": "Shohei Ohtani",
                "est_ba": 0.295,
                "est_slg": 0.550,
                "est_woba": 0.400,
                "brl_percent": 18.0,
                "avg_hit_speed": 93.1,
                "hard_hit_percent": 48.0,
            },
        ]
    )


def _mock_pitcher_df():
    return pd.DataFrame(
        [
            {
                "player_id": 543037,
                "player_name": "Gerrit Cole",
                "est_ba": 0.220,
                "est_slg": 0.340,
                "est_woba": 0.290,
                "brl_percent": 6.0,
                "avg_hit_speed": 87.0,
                "hard_hit_percent": 30.0,
            }
        ]
    )


# ── fetch_statcast_leaderboards tests ────────────────────────────────


def test_fetch_returns_dataframe_with_mocked_pybaseball():
    """Verify fetch produces correct DataFrame when pybaseball returns data."""
    mock_batter_fn = MagicMock(return_value=_mock_batter_df())
    mock_pitcher_fn = MagicMock(return_value=_mock_pitcher_df())

    mock_pybaseball = MagicMock()
    mock_pybaseball.statcast_batter_expected_stats = mock_batter_fn
    mock_pybaseball.statcast_pitcher_expected_stats = mock_pitcher_fn

    with patch.dict("sys.modules", {"pybaseball": mock_pybaseball}):
        df = fetch_statcast_leaderboards(season=2026, min_pa=25)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3  # 2 batters + 1 pitcher
    assert set(df.columns) == {
        "mlb_id",
        "player_name",
        "is_hitter",
        "xba",
        "xslg",
        "xwoba",
        "barrel_pct",
        "exit_velocity_avg",
        "hard_hit_pct",
    }
    # Check batter rows
    batters = df[df["is_hitter"]]
    assert len(batters) == 2
    trout = batters[batters["mlb_id"] == 545361].iloc[0]
    assert trout["player_name"] == "Mike Trout"
    assert trout["xba"] == pytest.approx(0.280)
    assert trout["exit_velocity_avg"] == pytest.approx(92.5)

    # Check pitcher row
    pitchers = df[~df["is_hitter"]]
    assert len(pitchers) == 1
    cole = pitchers.iloc[0]
    assert cole["mlb_id"] == 543037
    assert cole["xwoba"] == pytest.approx(0.290)


def test_fetch_returns_empty_when_pybaseball_unavailable():
    """Verify graceful degradation when pybaseball cannot be imported."""
    # Remove pybaseball from sys.modules so the lazy import inside the
    # function raises ImportError.
    saved = sys.modules.pop("pybaseball", None)
    try:
        with patch.dict("sys.modules", {"pybaseball": None}):
            df = fetch_statcast_leaderboards()
        assert isinstance(df, pd.DataFrame)
        assert df.empty
    finally:
        if saved is not None:
            sys.modules["pybaseball"] = saved


def test_fetch_handles_empty_batter_data():
    """Verify function handles None/empty batter data gracefully."""
    mock_pybaseball = MagicMock()
    mock_pybaseball.statcast_batter_expected_stats = MagicMock(return_value=pd.DataFrame())
    mock_pybaseball.statcast_pitcher_expected_stats = MagicMock(return_value=_mock_pitcher_df())

    with patch.dict("sys.modules", {"pybaseball": mock_pybaseball}):
        df = fetch_statcast_leaderboards(season=2026, min_pa=25)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1  # Only pitcher
    assert not df.iloc[0]["is_hitter"]


def test_fetch_handles_none_return():
    """Verify function handles None returns from pybaseball."""
    mock_pybaseball = MagicMock()
    mock_pybaseball.statcast_batter_expected_stats = MagicMock(return_value=None)
    mock_pybaseball.statcast_pitcher_expected_stats = MagicMock(return_value=None)

    with patch.dict("sys.modules", {"pybaseball": mock_pybaseball}):
        df = fetch_statcast_leaderboards(season=2026, min_pa=25)

    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_fetch_handles_batter_exception():
    """Verify function continues when batter fetch raises an exception."""
    mock_pybaseball = MagicMock()
    mock_pybaseball.statcast_batter_expected_stats = MagicMock(side_effect=Exception("Network error"))
    mock_pybaseball.statcast_pitcher_expected_stats = MagicMock(return_value=_mock_pitcher_df())

    with patch.dict("sys.modules", {"pybaseball": mock_pybaseball}):
        df = fetch_statcast_leaderboards(season=2026, min_pa=25)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1  # Only pitcher survived


def test_fetch_handles_pitcher_exception():
    """Verify function continues when pitcher fetch raises an exception."""
    mock_pybaseball = MagicMock()
    mock_pybaseball.statcast_batter_expected_stats = MagicMock(return_value=_mock_batter_df())
    mock_pybaseball.statcast_pitcher_expected_stats = MagicMock(side_effect=Exception("Timeout"))

    with patch.dict("sys.modules", {"pybaseball": mock_pybaseball}):
        df = fetch_statcast_leaderboards(season=2026, min_pa=25)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2  # Only batters survived


def test_fetch_handles_missing_columns():
    """Verify function handles missing columns with default 0 values."""
    sparse_df = pd.DataFrame([{"player_id": 999, "player_name": "Test Player"}])

    mock_pybaseball = MagicMock()
    mock_pybaseball.statcast_batter_expected_stats = MagicMock(return_value=sparse_df)
    mock_pybaseball.statcast_pitcher_expected_stats = MagicMock(return_value=pd.DataFrame())

    with patch.dict("sys.modules", {"pybaseball": mock_pybaseball}):
        df = fetch_statcast_leaderboards(season=2026, min_pa=1)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["xba"] == 0.0
    assert row["xslg"] == 0.0
    assert row["barrel_pct"] == 0.0


def test_fetch_passes_season_and_min_pa():
    """Verify season and min_pa are forwarded to pybaseball functions."""
    mock_pybaseball = MagicMock()
    mock_batter_fn = MagicMock(return_value=pd.DataFrame())
    mock_pitcher_fn = MagicMock(return_value=pd.DataFrame())
    mock_pybaseball.statcast_batter_expected_stats = mock_batter_fn
    mock_pybaseball.statcast_pitcher_expected_stats = mock_pitcher_fn

    with patch.dict("sys.modules", {"pybaseball": mock_pybaseball}):
        fetch_statcast_leaderboards(season=2025, min_pa=50)

    mock_batter_fn.assert_called_once_with(2025, minPA=50)
    mock_pitcher_fn.assert_called_once_with(2025, minPA=50)


# ── save_statcast_leaderboards_to_db tests ───────────────────────────


def test_save_returns_zero_for_empty_df():
    """Verify save returns 0 when given empty DataFrame."""
    assert save_statcast_leaderboards_to_db(pd.DataFrame()) == 0


def test_save_returns_zero_for_none():
    """Verify save returns 0 when given None."""
    assert save_statcast_leaderboards_to_db(None) == 0


def test_save_stores_rows_with_matching_player_ids():
    """Verify save resolves mlb_id to player_id and stores rows."""
    leaderboard_df = pd.DataFrame(
        [
            {
                "mlb_id": 545361,
                "player_name": "Mike Trout",
                "is_hitter": True,
                "xba": 0.280,
                "xslg": 0.520,
                "xwoba": 0.380,
                "barrel_pct": 15.0,
                "exit_velocity_avg": 92.5,
                "hard_hit_pct": 45.0,
            }
        ]
    )

    # Create an in-memory database with the required schema.
    # Wrap in _NoCloseConnection so the function's finally block
    # does not destroy the in-memory DB before we verify stored data.
    raw_conn = sqlite3.connect(":memory:")
    raw_conn.execute(
        """CREATE TABLE players (
            player_id INTEGER PRIMARY KEY,
            name TEXT,
            mlb_id INTEGER
        )"""
    )
    raw_conn.execute(
        """CREATE TABLE statcast_archive (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL,
            season INTEGER NOT NULL,
            ev_mean REAL, ev_p90 REAL, barrel_pct REAL, hard_hit_pct REAL,
            xba REAL, xslg REAL, xwoba REAL, whiff_pct REAL, chase_rate REAL,
            sprint_speed REAL, ff_avg_speed REAL, ff_spin_rate REAL,
            k_pct REAL, bb_pct REAL, gb_pct REAL,
            stuff_plus REAL, location_plus REAL, pitching_plus REAL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(player_id, season)
        )"""
    )
    raw_conn.execute("INSERT INTO players (player_id, name, mlb_id) VALUES (1, 'Mike Trout', 545361)")
    raw_conn.commit()

    wrapper = _NoCloseConnection(raw_conn)

    with patch("src.live_stats.get_connection", return_value=wrapper):
        count = save_statcast_leaderboards_to_db(leaderboard_df, season=2026)

    assert count == 1

    # Verify the stored data
    row = raw_conn.execute(
        "SELECT xba, xslg, xwoba, barrel_pct, ev_mean, hard_hit_pct FROM statcast_archive WHERE player_id = 1"
    ).fetchone()
    assert row is not None
    assert row[0] == pytest.approx(0.280)  # xba
    assert row[1] == pytest.approx(0.520)  # xslg
    assert row[2] == pytest.approx(0.380)  # xwoba
    assert row[3] == pytest.approx(15.0)  # barrel_pct
    assert row[4] == pytest.approx(92.5)  # ev_mean
    assert row[5] == pytest.approx(45.0)  # hard_hit_pct

    raw_conn.close()


def test_save_skips_unmatched_mlb_ids():
    """Verify save skips rows where mlb_id has no matching player_id."""
    leaderboard_df = pd.DataFrame(
        [
            {
                "mlb_id": 999999,
                "player_name": "Unknown Player",
                "is_hitter": True,
                "xba": 0.250,
                "xslg": 0.400,
                "xwoba": 0.330,
                "barrel_pct": 10.0,
                "exit_velocity_avg": 89.0,
                "hard_hit_pct": 35.0,
            }
        ]
    )

    raw_conn = sqlite3.connect(":memory:")
    raw_conn.execute(
        """CREATE TABLE players (
            player_id INTEGER PRIMARY KEY,
            name TEXT,
            mlb_id INTEGER
        )"""
    )
    raw_conn.execute(
        """CREATE TABLE statcast_archive (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL,
            season INTEGER NOT NULL,
            ev_mean REAL, barrel_pct REAL, hard_hit_pct REAL,
            xba REAL, xslg REAL, xwoba REAL,
            UNIQUE(player_id, season)
        )"""
    )
    raw_conn.commit()

    wrapper = _NoCloseConnection(raw_conn)

    with patch("src.live_stats.get_connection", return_value=wrapper):
        count = save_statcast_leaderboards_to_db(leaderboard_df, season=2026)

    assert count == 0
    raw_conn.close()

"""SF-13: _bootstrap_pvb_splits should report honest refresh_log status.

Audit finding (2026-04-18):
    The function unconditionally writes ``update_refresh_log("pvb_splits", "success")``
    even when 0 new matchups were fetched and all results were cache hits. UI shows
    green-success when nothing was actually fetched.

Required behavior:
    - updated > 0           → status="success", message describes new + cached
    - updated == 0, skipped > 0  → status="cached", message says all N already cached
    - updated == 0, skipped == 0 → status="no_data"
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@pytest.fixture
def temp_db(tmp_path):
    db_path = tmp_path / "test.db"
    with patch("src.database.DB_PATH", db_path):
        from src.database import init_db

        init_db()
        yield db_path


def _seed_minimal(temp_db, n_hitters=2, n_pitchers=2):
    """Seed players, league_rosters, opp_pitcher_stats so pvb function reaches the loop."""
    from src.database import get_connection

    conn = get_connection()
    try:
        for i in range(n_hitters):
            conn.execute(
                """INSERT INTO players (player_id, name, mlb_id, is_hitter, team, positions)
                   VALUES (?, ?, ?, 1, 'TST', 'OF')""",
                (1000 + i, f"Hitter {i}", 600000 + i),
            )
            conn.execute(
                """INSERT INTO league_rosters (team_name, team_index, player_id, roster_slot)
                   VALUES ('Test Team', 1, ?, 'OF')""",
                (1000 + i,),
            )
        for j in range(n_pitchers):
            conn.execute(
                """INSERT INTO opp_pitcher_stats (pitcher_id, name, season, fetched_at)
                   VALUES (?, ?, 2026, '2026-05-10T00:00:00')""",
                (700000 + j, f"Pitcher {j}"),
            )
        conn.commit()
    finally:
        conn.close()


class TestPvbSplitsAllCached:
    """When every (batter, pitcher) pair already has a row, status must be 'cached'."""

    def test_all_cached_writes_cached_status(self, temp_db):
        from src.database import get_connection

        with patch("src.database.DB_PATH", temp_db):
            _seed_minimal(temp_db, n_hitters=2, n_pitchers=2)

            conn = get_connection()
            try:
                for h in range(2):
                    for p in range(2):
                        conn.execute(
                            """INSERT INTO pvb_splits
                               (batter_id, pitcher_id, pa, avg, obp, slg, hr, k, bb, woba, fetched_at)
                               VALUES (?, ?, 30, 0.300, 0.350, 0.500, 1, 5, 2, 0.350, '2026-05-10T00:00:00')""",
                            (1000 + h, 700000 + p),
                        )
                conn.commit()
            finally:
                conn.close()

            mock_log = MagicMock()
            with (
                patch("src.data_bootstrap.update_refresh_log", mock_log, create=True),
                patch("src.database.update_refresh_log", mock_log),
            ):
                from src.data_bootstrap import BootstrapProgress, _bootstrap_pvb_splits

                progress = BootstrapProgress()
                _bootstrap_pvb_splits(progress)

            calls = [c for c in mock_log.call_args_list if c.args and c.args[0] == "pvb_splits"]
            assert len(calls) >= 1, f"No refresh_log call for pvb_splits; got: {mock_log.call_args_list}"
            final_call = calls[-1]
            status_arg = final_call.args[1] if len(final_call.args) >= 2 else final_call.kwargs.get("status")
            assert status_arg == "cached", (
                f"Expected status='cached' when all pairs already in cache, got '{status_arg}'"
            )


class TestPvbSplitsNoData:
    """When there are no rostered hitters AND nothing cached → no_data."""

    def test_no_rostered_hitters_returns_skipped(self, temp_db):
        with patch("src.database.DB_PATH", temp_db):
            mock_log = MagicMock()
            with (
                patch("src.data_bootstrap.update_refresh_log", mock_log, create=True),
                patch("src.database.update_refresh_log", mock_log),
            ):
                from src.data_bootstrap import BootstrapProgress, _bootstrap_pvb_splits

                progress = BootstrapProgress()
                result = _bootstrap_pvb_splits(progress)

            assert "Skipped" in result or "no rostered" in result.lower()


class TestPvbSplitsFreshFetch:
    """When pybaseball returns real data → status='success'."""

    def test_real_fetch_writes_success(self, temp_db, monkeypatch):
        from src.database import get_connection

        with patch("src.database.DB_PATH", temp_db):
            _seed_minimal(temp_db, n_hitters=1, n_pitchers=1)

            fake_df = pd.DataFrame(
                {
                    "pitcher": [700000, 700000, 700000, 700000, 700000],
                    "events": ["single", "strikeout", "walk", "double", "home_run"],
                    "estimated_woba_using_speedangle": [0.4, 0.0, 0.6, 0.5, 0.9],
                }
            )

            with patch("pybaseball.statcast_batter", return_value=fake_df):
                mock_log = MagicMock()
                with (
                    patch("src.data_bootstrap.update_refresh_log", mock_log, create=True),
                    patch("src.database.update_refresh_log", mock_log),
                ):
                    from src.data_bootstrap import BootstrapProgress, _bootstrap_pvb_splits

                    progress = BootstrapProgress()
                    _bootstrap_pvb_splits(progress)

                calls = [c for c in mock_log.call_args_list if c.args and c.args[0] == "pvb_splits"]
                assert len(calls) >= 1, "No refresh_log call for pvb_splits"
                final_call = calls[-1]
                status_arg = final_call.args[1] if len(final_call.args) >= 2 else final_call.kwargs.get("status")
                assert status_arg == "success", f"Expected 'success' on a fresh fetch, got '{status_arg}'"

            conn = get_connection()
            try:
                row = conn.execute("SELECT COUNT(*) FROM pvb_splits").fetchone()
                assert row[0] >= 1, "Expected at least one pvb_splits row after fresh fetch"
            finally:
                conn.close()

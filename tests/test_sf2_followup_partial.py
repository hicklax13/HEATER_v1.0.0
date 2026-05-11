"""SF-2 follow-up: _bootstrap_live_stats must surface 'partial' when save rate is low.

Bug: live DB shows season_stats='success' with 'saved 1073/7478 rows' (14% saved).
The static `EXPECTED_MIN = 500` floor lets the call write 'success' even when the
vast majority of rows were dropped during name/team matching.

Fix: compute threshold dynamically from input size (e.g. 80% of len(df)).
"""

from unittest.mock import patch

import pandas as pd
import pytest


@pytest.fixture
def temp_db(tmp_path):
    db_path = tmp_path / "test.db"
    with patch("src.database.DB_PATH", db_path):
        from src.database import init_db

        init_db()
        yield db_path


def _build_input_df(n_rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_name": [f"Player {i}" for i in range(n_rows)],
            "team": ["NYY"] * n_rows,
            "is_hitter": [True] * n_rows,
            "pa": [10] * n_rows,
            "ab": [9] * n_rows,
            "h": [3] * n_rows,
        }
    )


class TestSF2FollowupPartial:
    def test_low_save_rate_writes_partial(self, temp_db):
        """7478 fetched, 1073 saved (14%) → status must be 'partial', not 'success'."""
        with patch("src.database.DB_PATH", temp_db):
            with (
                patch("src.live_stats.fetch_season_stats", return_value=_build_input_df(7478)),
                patch("src.live_stats.save_season_stats_to_db", return_value=1073),
            ):
                from src.data_bootstrap import BootstrapProgress, _bootstrap_live_stats
                from src.database import get_refresh_status

                _bootstrap_live_stats(BootstrapProgress())
                status = get_refresh_status("season_stats")
                assert status is not None
                assert status["status"] == "partial", (
                    f"Expected 'partial' for 1073/7478 (14%) saves, got '{status['status']}'"
                )

    def test_high_save_rate_writes_success(self, temp_db):
        """7478 fetched, 7000 saved (94%) → still 'success'."""
        with patch("src.database.DB_PATH", temp_db):
            with (
                patch("src.live_stats.fetch_season_stats", return_value=_build_input_df(7478)),
                patch("src.live_stats.save_season_stats_to_db", return_value=7000),
            ):
                from src.data_bootstrap import BootstrapProgress, _bootstrap_live_stats
                from src.database import get_refresh_status

                _bootstrap_live_stats(BootstrapProgress())
                status = get_refresh_status("season_stats")
                assert status is not None
                assert status["status"] == "success"

    def test_min_floor_still_enforced_for_small_inputs(self, temp_db):
        """Tiny input (e.g. 100 rows) — threshold floor of 500 still applies."""
        with patch("src.database.DB_PATH", temp_db):
            with (
                patch("src.live_stats.fetch_season_stats", return_value=_build_input_df(100)),
                patch("src.live_stats.save_season_stats_to_db", return_value=100),
            ):
                from src.data_bootstrap import BootstrapProgress, _bootstrap_live_stats
                from src.database import get_refresh_status

                _bootstrap_live_stats(BootstrapProgress())
                status = get_refresh_status("season_stats")
                assert status is not None
                assert status["status"] == "partial"

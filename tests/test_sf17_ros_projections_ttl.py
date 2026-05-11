"""SF-17: ROS projections must use a fixed 4h TTL and run AFTER team_strength.

Bug: Phase 19 (inline ROS block) currently uses _live_stats_ttl_hours()
(0.25-1.0h dynamic). PyMC MCMC sampling is expensive — refreshing it
every 15 min during the game window is wasteful. Also, update_ros_projections
depends on team_strength data, but the block runs BEFORE team_strength (P21).

Fix:
- Replace dynamic TTL with fixed 4.0h.
- Move ROS block to after team_strength.
"""

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def temp_db(tmp_path):
    db_path = tmp_path / "test.db"
    with patch("src.database.DB_PATH", db_path):
        from src.database import init_db

        init_db()
        yield db_path


_MOCKED_PHASES = (
    "_bootstrap_players",
    "_bootstrap_park_factors",
    "_bootstrap_projections",
    "_bootstrap_live_stats",
    "_bootstrap_historical",
    "_bootstrap_injury_data",
    "_bootstrap_yahoo",
    "_bootstrap_extended_roster",
    "_bootstrap_adp_sources",
    "_bootstrap_depth_charts",
    "_bootstrap_contracts",
    "_bootstrap_news",
    "_bootstrap_prospects",
    "_bootstrap_news_intel",
    "_bootstrap_ecr_consensus",
    "_bootstrap_game_day",
    "_bootstrap_team_strength",
    "_bootstrap_stuff_plus",
    "_bootstrap_batting_stats",
    "_bootstrap_sprint_speed",
    "_bootstrap_bat_speed",
    "_bootstrap_forty_man",
    "_bootstrap_umpire_tendencies",
    "_bootstrap_catcher_framing",
    "_bootstrap_pvb_splits",
    "_bootstrap_game_logs",
    "_bootstrap_injury_writeback",
    "_bootstrap_draft_results",
)


def _mock_all_phases_recording(stack: ExitStack, call_log: list) -> dict:
    """Mock every phase, recording call order in `call_log`."""
    mocks = {}
    for name in _MOCKED_PHASES:
        m = MagicMock(side_effect=lambda *args, _n=name, **kw: (call_log.append(_n), "ok")[1])
        stack.enter_context(patch(f"src.data_bootstrap.{name}", m))
        mocks[name] = m
    return mocks


class TestSF17RosProjectionsTtl:
    def test_uses_fixed_4h_ttl(self, temp_db):
        """ROS projections staleness check must call check_staleness('ros_projections', 4.0)."""
        with patch("src.database.DB_PATH", temp_db):
            calls: list[tuple] = []

            def _record(source, hours):
                calls.append((source, hours))
                return False  # treat as fresh — skip block

            with ExitStack() as stack:
                _mock_all_phases_recording(stack, [])
                stack.enter_context(patch("src.database.check_staleness", side_effect=_record))

                from src.data_bootstrap import bootstrap_all_data

                bootstrap_all_data(force=False)

            ros_calls = [c for c in calls if c[0] == "ros_projections"]
            assert len(ros_calls) == 1
            assert ros_calls[0][1] == 4.0, (
                f"ROS projections TTL must be 4.0h (was {ros_calls[0][1]}). "
                "PyMC MCMC is too expensive for the dynamic 0.25-1.0h window."
            )


class TestSF17RosProjectionsOrdering:
    def test_ros_runs_after_team_strength(self, temp_db):
        """Bayesian ROS update needs team strength populated first."""
        with patch("src.database.DB_PATH", temp_db):
            call_log: list[str] = []
            ros_call_order: list[int] = []

            def _ros_update():
                ros_call_order.append(len(call_log))
                return 0

            with ExitStack() as stack:
                _mock_all_phases_recording(stack, call_log)
                stack.enter_context(patch("src.database.check_staleness", return_value=True))
                stack.enter_context(patch("src.bayesian.update_ros_projections", side_effect=_ros_update))

                from src.data_bootstrap import bootstrap_all_data

                bootstrap_all_data(force=True)

            assert "_bootstrap_team_strength" in call_log, "team_strength must run during bootstrap"
            ts_index = call_log.index("_bootstrap_team_strength")
            assert ros_call_order, "ROS update must be invoked"
            ros_position = ros_call_order[0]
            assert ros_position > ts_index, (
                f"ROS update ran at call index {ros_position}, but team_strength ran at {ts_index}. "
                "ROS must run AFTER team_strength so its inputs are populated."
            )

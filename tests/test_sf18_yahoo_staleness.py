"""SF-18: Phases 17 (yahoo_transactions) and 18 (yahoo_free_agents) need staleness gates.

Bug: Both Yahoo phases run on every bootstrap regardless of recency. Phase 18
also has no `update_refresh_log` entry at all so its freshness is invisible.

Fix:
- Gate Phase 17 with `check_staleness("yahoo_transactions", 0.25)` (15 min).
- Gate Phase 18 with `check_staleness("yahoo_free_agents", 1.0)` (1 hour).
- Add `update_refresh_log("yahoo_free_agents", "success")` at end of Phase 18.
"""

from contextlib import ExitStack
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


def _mock_all_phases(stack: ExitStack) -> None:
    for name in _MOCKED_PHASES:
        stack.enter_context(patch(f"src.data_bootstrap.{name}", return_value="ok"))


def _make_yahoo_client_with_data():
    yahoo = MagicMock()
    yahoo.get_league_transactions.return_value = pd.DataFrame(
        [{"player_name": "Mike Trout", "type": "add", "team_from": "", "team_to": "Team A", "timestamp": "1234"}]
    )
    yahoo.get_free_agents.return_value = pd.DataFrame(
        [
            {
                "player_key": "k1",
                "player_name": "Joe Smith",
                "positions": "OF",
                "team": "NYY",
                "percent_owned": 12,
            }
        ]
    )
    return yahoo


class TestSF18YahooTransactionsGate:
    def test_skipped_when_fresh(self, temp_db):
        """Yahoo transactions endpoint is NOT called when staleness check returns False."""
        with patch("src.database.DB_PATH", temp_db):
            yahoo = _make_yahoo_client_with_data()
            with ExitStack() as stack:
                _mock_all_phases(stack)
                stack.enter_context(patch("src.database.check_staleness", return_value=False))

                from src.data_bootstrap import bootstrap_all_data

                bootstrap_all_data(yahoo_client=yahoo, force=False)
                yahoo.get_league_transactions.assert_not_called()

    def test_called_when_stale(self, temp_db):
        """Yahoo transactions IS called when staleness check returns True."""
        with patch("src.database.DB_PATH", temp_db):
            yahoo = _make_yahoo_client_with_data()
            with ExitStack() as stack:
                _mock_all_phases(stack)
                stack.enter_context(patch("src.database.check_staleness", return_value=True))
                stack.enter_context(patch("src.live_stats.match_player_id", return_value=1))

                from src.data_bootstrap import bootstrap_all_data

                bootstrap_all_data(yahoo_client=yahoo, force=False)
                yahoo.get_league_transactions.assert_called()


class TestSF18YahooFreeAgentsGate:
    def test_skipped_when_fresh(self, temp_db):
        """Yahoo free agents endpoint is NOT called when staleness check returns False."""
        with patch("src.database.DB_PATH", temp_db):
            yahoo = _make_yahoo_client_with_data()
            with ExitStack() as stack:
                _mock_all_phases(stack)
                stack.enter_context(patch("src.database.check_staleness", return_value=False))

                from src.data_bootstrap import bootstrap_all_data

                bootstrap_all_data(yahoo_client=yahoo, force=False)
                yahoo.get_free_agents.assert_not_called()

    def test_writes_refresh_log_on_success(self, temp_db):
        """After successful Phase 18 fetch, refresh_log has 'yahoo_free_agents' entry."""
        with patch("src.database.DB_PATH", temp_db):
            yahoo = _make_yahoo_client_with_data()
            with ExitStack() as stack:
                _mock_all_phases(stack)
                stack.enter_context(patch("src.database.check_staleness", return_value=True))
                stack.enter_context(patch("src.live_stats.match_player_id", return_value=1))

                from src.data_bootstrap import bootstrap_all_data
                from src.database import get_refresh_status

                bootstrap_all_data(yahoo_client=yahoo, force=False)
                status = get_refresh_status("yahoo_free_agents")
                assert status is not None, "Phase 18 must write refresh_log entry for yahoo_free_agents"
                # Wave 7 INFRA-F6: row-count gate downgrades to "partial" when
                # < expected_min rows; the mock fixture writes few FAs, so
                # accept either "success" or "partial" (both mean data exists).
                assert status["status"] in ("success", "partial"), (
                    f"Expected success or partial, got {status['status']!r}"
                )

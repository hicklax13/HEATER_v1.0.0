"""PR18: Pin that ctx.player_pool respects the level_filter parameter
so minor leaguers don't pollute MLB-only FA recommendations."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.optimizer.shared_data_layer import build_optimizer_context
from src.valuation import LeagueConfig


@pytest.fixture
def fake_pool_with_minors():
    """Pool with 5 MLB players (level='MLB') + 5 minor leaguers (level='AAA')."""
    rows = []
    for i in range(5):
        rows.append(
            {
                "player_id": i + 1,
                "name": f"MLB Player {i}",
                "level": "MLB",
                "positions": "OF",
                "is_hitter": 1,
                "r": 50,
                "hr": 10,
                "rbi": 40,
                "sb": 5,
                "ab": 300,
                "h": 80,
                "bb": 30,
                "hbp": 2,
                "sf": 2,
                "avg": 0.267,
                "obp": 0.340,
            }
        )
    for i in range(5):
        rows.append(
            {
                "player_id": i + 100,
                "name": f"Minor Player {i}",
                "level": "AAA",
                "positions": "OF",
                "is_hitter": 1,
                "r": 0,
                "hr": 0,
                "rbi": 0,
                "sb": 0,
                "ab": 0,
                "h": 0,
                "bb": 0,
                "hbp": 0,
                "sf": 0,
                "avg": 0,
                "obp": 0,
            }
        )
    return pd.DataFrame(rows)


def test_default_level_filter_is_mlb_only(fake_pool_with_minors):
    """Default level_filter='MLB only' must remove minor leaguers from
    ctx.player_pool. The FA page's default UI selection is MLB only,
    so the engine's default should match."""
    config = LeagueConfig()
    yds = MagicMock()
    yds.get_rosters.return_value = pd.DataFrame()
    yds.get_free_agents.return_value = pd.DataFrame()
    yds.get_matchup.return_value = None

    with patch("src.optimizer.shared_data_layer.load_player_pool", return_value=fake_pool_with_minors):
        ctx = build_optimizer_context(scope="rest_of_season", yds=yds, config=config)

    # All MLB players present, no minors
    assert len(ctx.player_pool) == 5
    assert (ctx.player_pool["level"] == "MLB").all()


def test_level_filter_all_keeps_minors(fake_pool_with_minors):
    """When level_filter='All', minor leaguers are kept (e.g., for
    prospect lookups on the Player Databank page)."""
    config = LeagueConfig()
    yds = MagicMock()
    yds.get_rosters.return_value = pd.DataFrame()
    yds.get_free_agents.return_value = pd.DataFrame()
    yds.get_matchup.return_value = None

    with patch("src.optimizer.shared_data_layer.load_player_pool", return_value=fake_pool_with_minors):
        ctx = build_optimizer_context(scope="rest_of_season", yds=yds, config=config, level_filter="All")

    assert len(ctx.player_pool) == 10

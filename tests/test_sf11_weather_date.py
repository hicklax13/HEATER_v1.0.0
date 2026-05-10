"""SF-11: Weather lookup must use ET-anchored game date, not UTC."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimizer.daily_optimizer import build_daily_dcv_table
from src.valuation import LeagueConfig


def test_weather_lookup_uses_get_target_game_date():
    cfg = LeagueConfig()
    roster = pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "X",
                "positions": "OF",
                "team": "NYY",
                "is_hitter": 1,
                "fp_proj": 5.0,
                "hr": 0,
                "rbi": 0,
                "r": 0,
                "sb": 0,
                "avg": 0,
                "obp": 0,
                "ip": 0,
                "k": 0,
                "w": 0,
                "l": 0,
                "sv": 0,
                "era": 0,
                "whip": 0,
            }
        ]
    )
    with (
        patch(
            "src.optimizer.daily_optimizer.get_target_game_date",
            return_value="2099-01-01",
        ) as gtgd,
        patch(
            "src.database.load_game_day_weather",
            return_value=pd.DataFrame(),
        ) as lgdw,
    ):
        build_daily_dcv_table(
            roster=roster,
            matchup=None,
            schedule_today=None,
            park_factors={},
            config=cfg,
        )
    assert gtgd.called, "get_target_game_date must be invoked"
    assert lgdw.called, "load_game_day_weather must be invoked"
    call_args = lgdw.call_args
    call_date = call_args[0][0] if call_args[0] else call_args.kwargs.get("date")
    assert call_date == "2099-01-01", f"Weather got {call_date}, expected 2099-01-01"

"""Track Record guards: replay scoring vs actual box-score lines.

Phase 5 of the Pitcher Streaming Analyzer plan. ``replay_stream_date`` must
(a) carry ``proxy_caveat=True`` — HEATER stores no point-in-time projections,
so replay scores use current data and the UI must disclose it; (b) aggregate
actuals with the weighted rate idiom; (c) degrade to empty results without
raising when a date has no games or statsapi is unavailable. The page must
actually render the caveat (text reference check).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.optimizer.stream_analyzer import replay_stream_date
from src.valuation import LeagueConfig

_PAGE = Path(__file__).resolve().parents[1] / "pages" / "4_Pitcher_Streaming.py"


def _pool():
    return pd.DataFrame(
        [
            {
                "player_id": 1,
                "mlb_id": 660271,
                "player_name": "Streamer A",
                "team": "SEA",
                "positions": "SP",
                "is_hitter": 0,
                "throws": "R",
                "era": 3.40,
                "whip": 1.10,
                "k": 170.0,
                "w": 11.0,
                "ip": 168.0,
                "percent_owned": 14.0,
            },
            {
                "player_id": 2,
                "mlb_id": 543037,
                "player_name": "Streamer B",
                "team": "BOS",
                "positions": "SP",
                "is_hitter": 0,
                "throws": "L",
                "era": 4.20,
                "whip": 1.30,
                "k": 140.0,
                "w": 8.0,
                "ip": 150.0,
                "percent_owned": 6.0,
            },
        ]
    )


def _ctx():
    return SimpleNamespace(
        player_pool=_pool(),
        user_roster_ids=[],
        league_rostered_ids=set(),
        team_strength={},
        park_factors={},
        weather={},
        two_start_pitchers=[],
        recent_form={},
        todays_schedule=[],
        config=LeagueConfig(),
    )


def _past_date() -> str:
    return (datetime.now(UTC) - timedelta(days=3)).strftime("%Y-%m-%d")


def _schedule(date_str):
    return [
        {
            "game_date": date_str,
            "home_name": "Seattle Mariners",
            "away_name": "Chicago White Sox",
            "home_probable_pitcher": "Streamer A",
            "away_probable_pitcher": "",
            "status": "Final",
        },
        {
            "game_date": date_str,
            "home_name": "Boston Red Sox",
            "away_name": "New York Yankees",
            "home_probable_pitcher": "Streamer B",
            "away_probable_pitcher": "",
            "status": "Final",
        },
    ]


def _actuals(date_str):
    return {
        660271: pd.DataFrame(
            [
                {
                    "date": date_str,
                    "opponent": "CWS",
                    "is_home": True,
                    "venue": "",
                    "ip": 7.0,
                    "k": 9.0,
                    "er": 1.0,
                    "bb": 1.0,
                    "h": 4.0,
                    "w": 1.0,
                    "l": 0.0,
                }
            ]
        ),
        543037: pd.DataFrame(
            [
                {
                    "date": date_str,
                    "opponent": "NYY",
                    "is_home": True,
                    "venue": "",
                    "ip": 3.0,
                    "k": 2.0,
                    "er": 6.0,
                    "bb": 4.0,
                    "h": 8.0,
                    "w": 0.0,
                    "l": 1.0,
                }
            ]
        ),
    }


def test_replay_carries_proxy_caveat():
    date = _past_date()
    result = replay_stream_date(_ctx(), date, schedule=_schedule(date), actuals=_actuals(date))
    assert result["proxy_caveat"] is True


def test_replay_matches_actual_lines():
    date = _past_date()
    result = replay_stream_date(_ctx(), date, schedule=_schedule(date), actuals=_actuals(date))
    actuals = result["actuals"].set_index("player_name")
    assert actuals.loc["Streamer A", "actual_k"] == 9.0
    assert actuals.loc["Streamer B", "actual_er"] == 6.0
    # Weighted aggregate across the picked starts: ERA = ER*9/IP summed.
    summary = result["summary"]
    assert summary["era"] == pytest.approx(7 * 9 / 10.0, abs=0.01)
    assert summary["whip"] == pytest.approx((5 + 12) / 10.0, abs=0.01)
    assert summary["qs_rate"] == pytest.approx(0.5)


def test_replay_no_games_is_empty_not_raise():
    date = _past_date()
    result = replay_stream_date(_ctx(), date, schedule=[], actuals={})
    assert result["board_then"].empty
    assert result["actuals"].empty
    assert result["summary"] == {}
    assert result["proxy_caveat"] is True


def test_page_renders_proxy_caveat():
    src = _PAGE.read_text(encoding="utf-8")
    assert "proxy_caveat" in src, (
        "the Track Record tab must check/render the replay proxy caveat — replay accuracy must never be oversold"
    )

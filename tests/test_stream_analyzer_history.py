"""Matchup Microscope engine guards: pitcher history + lineup exposure.

Phase 3 of the Pitcher Streaming Analyzer plan.
``get_pitcher_vs_team_history`` must aggregate game logs with the weighted
rate-stat idiom (ERA = ER*9/IP, WHIP = (BB+H)/IP — never averaged), filter to
an opponent when asked, and degrade to an empty frame (no raise) when statsapi
is unavailable — the conftest network guard makes the failure path
deterministic. ``compute_lineup_exposure`` must regress PvB samples toward the
batter's generic wOBA (60-PA stabilization) and return None when there is no
overlap to reason about.
"""

from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

from src.optimizer.stream_analyzer import (
    aggregate_pitcher_history,
    compute_lineup_exposure,
    get_pitcher_vs_team_history,
)

_FAKE_LOG = {
    "stats": [
        {
            "date": "2026-06-01",
            "isHome": True,
            "opponent": {"name": "Chicago White Sox"},
            "stats": {
                "inningsPitched": "6.1",
                "strikeOuts": 8,
                "earnedRuns": 2,
                "baseOnBalls": 1,
                "hits": 4,
                "wins": 1,
                "losses": 0,
            },
        },
        {
            "date": "2026-05-26",
            "isHome": False,
            "opponent": {"name": "Houston Astros"},
            "stats": {
                "inningsPitched": "4.2",
                "strikeOuts": 3,
                "earnedRuns": 5,
                "baseOnBalls": 3,
                "hits": 9,
                "wins": 0,
                "losses": 1,
            },
        },
        {
            "date": "2026-05-20",
            "isHome": True,
            "opponent": {"name": "Chicago White Sox"},
            "stats": {
                "inningsPitched": "7.0",
                "strikeOuts": 9,
                "earnedRuns": 1,
                "baseOnBalls": 0,
                "hits": 3,
                "wins": 1,
                "losses": 0,
            },
        },
    ]
}


@pytest.fixture
def fake_statsapi(monkeypatch):
    mod = types.ModuleType("statsapi")
    mod.player_stat_data = lambda *a, **k: _FAKE_LOG
    monkeypatch.setitem(sys.modules, "statsapi", mod)
    return mod


def test_history_returns_per_start_rows(fake_statsapi):
    df = get_pitcher_vs_team_history(660271)
    assert len(df) == 3
    assert {"date", "opponent", "is_home", "ip", "k", "er", "bb", "h", "w", "l"} <= set(df.columns)
    # IP parsed via the canonical outs converter: "6.1" -> 6.333..., "4.2" -> 4.666...
    by_date = df.set_index("date")
    assert by_date.loc["2026-06-01", "ip"] == pytest.approx(6 + 1 / 3)
    assert by_date.loc["2026-05-26", "ip"] == pytest.approx(4 + 2 / 3)


def test_history_opponent_filter(fake_statsapi):
    df = get_pitcher_vs_team_history(660271, opp_team="CWS")
    assert len(df) == 2
    assert set(df["opponent"]) == {"CWS"}


def test_history_last_n(fake_statsapi):
    df = get_pitcher_vs_team_history(660271, last_n=1)
    assert len(df) == 1
    assert df.iloc[0]["date"] == "2026-06-01"  # newest first


def test_history_statsapi_failure_returns_empty(monkeypatch):
    mod = types.ModuleType("statsapi")

    def _boom(*a, **k):
        raise RuntimeError("network down")

    mod.player_stat_data = _boom
    monkeypatch.setitem(sys.modules, "statsapi", mod)
    df = get_pitcher_vs_team_history(660271)
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_aggregate_uses_weighted_rates(fake_statsapi):
    df = get_pitcher_vs_team_history(660271)
    agg = aggregate_pitcher_history(df)
    total_ip = 6 + 1 / 3 + 4 + 2 / 3 + 7.0
    assert agg["ip"] == pytest.approx(total_ip, abs=0.01)
    assert agg["era"] == pytest.approx(8 * 9 / total_ip, abs=0.01), (
        "ERA must be ER*9/IP over summed components, never an average of game ERAs"
    )
    assert agg["whip"] == pytest.approx((4 + 16) / total_ip, abs=0.01)
    assert agg["k"] == 20
    assert agg["games"] == 3


def test_aggregate_empty():
    assert aggregate_pitcher_history(pd.DataFrame()) == {}


# ── compute_lineup_exposure ──────────────────────────────────────────────────


def _mini_pool():
    return pd.DataFrame(
        [
            {"player_id": 11, "player_name": "Batter One", "xwoba": 0.380},
            {"player_id": 12, "player_name": "Batter Two", "xwoba": 0.300},
            {"player_id": 13, "player_name": "Batter Three", "xwoba": float("nan")},
        ]
    )


def test_lineup_exposure_regresses_pvb():
    # Batter One owns this pitcher over a real sample; Batter Two has none.
    pvb = {
        (11, 99): {"pa": 60, "woba": 0.500},
    }
    exposure = compute_lineup_exposure(99, [11, 12], _mini_pool(), pvb_data=pvb)
    assert exposure is not None
    # Batter One at full 60-PA weight = 0.500; Batter Two falls back to
    # generic 0.300; mean = 0.400 → exposure vs league average is positive.
    assert exposure > 0.0


def test_lineup_exposure_small_sample_shrinks():
    big = compute_lineup_exposure(
        99, [11], _mini_pool(), pvb_data={(11, 99): {"pa": 60, "woba": 0.600}}
    )
    small = compute_lineup_exposure(
        99, [11], _mini_pool(), pvb_data={(11, 99): {"pa": 6, "woba": 0.600}}
    )
    assert big is not None and small is not None
    assert big > small, "a 6-PA PvB sample must shrink toward the generic wOBA"


def test_lineup_exposure_none_without_batters():
    assert compute_lineup_exposure(99, [], _mini_pool(), pvb_data={}) is None

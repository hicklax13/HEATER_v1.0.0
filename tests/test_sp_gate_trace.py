"""SF gate trace: confirm the pure-SP probable-starter gate at
daily_optimizer.py fires whether positions is 'SP' (Yahoo-merged) or
'P' (raw MLB players.positions)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimizer.daily_optimizer import build_daily_dcv_table
from src.valuation import LeagueConfig


@pytest.fixture(autouse=True)
def _isolate_roster_statuses(monkeypatch):
    monkeypatch.setattr(
        "src.trade_intelligence._load_roster_statuses",
        lambda: {},
    )


@pytest.fixture
def cfg():
    return LeagueConfig()


def _make_roster(positions: str, name: str = "Test Pitcher", team: str = "NYY") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": name,
                "positions": positions,
                "team": team,
                "is_hitter": 0,
                "bats": "R",
                "throws": "R",
                "status": "active",
                "fp_proj": 5.0,
                "ip": 5.0,
                "k": 5.0,
                "w": 0.5,
                "sv": 0.0,
                "era": 4.0,
                "whip": 1.3,
                "hr": 0.0,
                "rbi": 0.0,
                "r": 0.0,
                "sb": 0.0,
                "avg": 0.0,
                "obp": 0.0,
                "l": 0.5,
            }
        ]
    )


def _make_schedule(home_probable: str = "Other Pitcher") -> list[dict]:
    return [
        {
            "home_name": "NEW YORK YANKEES",
            "away_name": "BOSTON RED SOX",
            "home_short": "NYY",
            "away_short": "BOS",
            "home_probable_pitcher": home_probable,
            "away_probable_pitcher": "Some Other",
        }
    ]


def test_pure_sp_pos_marker_gates_when_not_probable(cfg):
    roster = _make_roster("SP")
    schedule_today = _make_schedule()
    df = build_daily_dcv_table(
        roster=roster,
        matchup=None,
        schedule_today=schedule_today,
        park_factors={},
        config=cfg,
    )
    row = df[df["player_id"] == 1].iloc[0]
    assert row["volume_factor"] == 0.0, "Pure SP not in probables must zero out"
    assert row["reason"] == "OFF_DAY", f"Expected OFF_DAY, got {row['reason']}"


def test_pure_p_pos_marker_also_gates_when_not_probable(cfg):
    roster = _make_roster("P")
    schedule_today = _make_schedule()
    df = build_daily_dcv_table(
        roster=roster,
        matchup=None,
        schedule_today=schedule_today,
        park_factors={},
        config=cfg,
    )
    row = df[df["player_id"] == 1].iloc[0]
    assert row["volume_factor"] == 0.0, (
        "positions='P' (no SP/RP qualifier) on team with named probable: player is provably not pitching, must zero out"
    )


def test_sp_rp_hybrid_not_zeroed_when_not_probable(cfg):
    roster = _make_roster("SP,RP")
    schedule_today = _make_schedule()
    df = build_daily_dcv_table(
        roster=roster,
        matchup=None,
        schedule_today=schedule_today,
        park_factors={},
        config=cfg,
    )
    row = df[df["player_id"] == 1].iloc[0]
    assert row["volume_factor"] > 0.0, "SP/RP hybrid retains 0.9 baseline"


def test_sp_probable_today_keeps_volume(cfg):
    roster = _make_roster("SP", name="Gerrit Cole")
    schedule_today = _make_schedule(home_probable="Gerrit Cole")
    df = build_daily_dcv_table(
        roster=roster,
        matchup=None,
        schedule_today=schedule_today,
        park_factors={},
        config=cfg,
    )
    row = df[df["player_id"] == 1].iloc[0]
    assert row["volume_factor"] > 0.0, "Named probable must keep volume"

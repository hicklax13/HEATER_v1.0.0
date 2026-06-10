"""Week Planner guards: add-budget ceiling, IP pacing from ip_tracker, dedup.

Phase 4 of the Pitcher Streaming Analyzer plan. The planner must never plan
more adds than remain in the weekly budget, must report IP pacing against the
canonical ip_tracker constants (never literals), and must list each pitcher at
most once even when he has multiple scheduled starts in the window.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pandas as pd

from src.ip_tracker import MIN_IP, WEEKLY_TARGET
from src.optimizer.stream_analyzer import build_week_plan
from src.valuation import LeagueConfig


def _pool():
    rows = []
    for pid, name, team in [
        (1, "Streamer A", "SEA"),
        (2, "Streamer B", "BOS"),
        (3, "Streamer C", "NYM"),
    ]:
        rows.append(
            {
                "player_id": pid,
                "player_name": name,
                "team": team,
                "positions": "SP",
                "is_hitter": 0,
                "throws": "R",
                "era": 3.30,
                "whip": 1.08,
                "k": 180.0,
                "w": 12.0,
                "ip": 175.0,
                "percent_owned": 10.0,
            }
        )
    return pd.DataFrame(rows)


def _ctx(**overrides):
    base = {
        "player_pool": _pool(),
        "user_roster_ids": [],
        "league_rostered_ids": set(),
        "team_strength": {},
        "park_factors": {},
        "weather": {},
        "two_start_pitchers": [],
        "recent_form": {},
        "todays_schedule": [],
        "config": LeagueConfig(),
        "adds_remaining_this_week": 10,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _date(offset: int) -> str:
    return (datetime.now(UTC) + timedelta(days=offset)).strftime("%Y-%m-%d")


def _game(date_str, home, away, probable):
    return {
        "game_date": date_str,
        "home_name": home,
        "away_name": away,
        "home_probable_pitcher": probable,
        "away_probable_pitcher": "",
        "status": "Scheduled",
    }


def _schedule():
    return [
        _game(_date(1), "Seattle Mariners", "Chicago White Sox", "Streamer A"),
        _game(_date(2), "Boston Red Sox", "New York Yankees", "Streamer B"),
        _game(_date(3), "New York Mets", "Miami Marlins", "Streamer C"),
    ]


def test_plan_respects_add_budget():
    plan = build_week_plan(_ctx(adds_remaining_this_week=2), schedule=_schedule())
    assert len(plan["plan"]) <= 2
    assert plan["summary"]["max_adds"] == 2


def test_plan_max_adds_override():
    plan = build_week_plan(_ctx(), schedule=_schedule(), max_adds=1)
    assert len(plan["plan"]) == 1


def test_summary_uses_ip_tracker_constants():
    plan = build_week_plan(_ctx(), schedule=_schedule())
    assert plan["summary"]["ip_floor"] == MIN_IP
    assert plan["summary"]["ip_target"] == WEEKLY_TARGET


def test_under_floor_flag():
    low = build_week_plan(_ctx(), schedule=_schedule(), max_adds=1, base_weekly_ip=5.0)
    high = build_week_plan(_ctx(), schedule=_schedule(), base_weekly_ip=50.0)
    none = build_week_plan(_ctx(), schedule=_schedule())
    assert low["summary"]["under_floor"] is True
    assert high["summary"]["under_floor"] is False
    assert none["summary"]["under_floor"] is None


def test_pitcher_with_two_window_starts_planned_once():
    sched = _schedule() + [
        _game(_date(5), "Seattle Mariners", "Miami Marlins", "Streamer A"),
    ]
    plan = build_week_plan(_ctx(), schedule=sched)
    names = [e["player_name"] for e in plan["plan"]]
    assert names.count("Streamer A") == 1


def test_rostered_pitchers_never_planned():
    plan = build_week_plan(_ctx(league_rostered_ids={1, 2, 3}), schedule=_schedule())
    assert plan["plan"] == []


def test_empty_schedule_yields_empty_plan():
    plan = build_week_plan(_ctx(), schedule=[])
    assert plan["plan"] == []
    assert plan["summary"]["ip_added"] == 0.0

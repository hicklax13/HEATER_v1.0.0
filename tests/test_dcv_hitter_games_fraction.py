"""Guard: DCV-A1-012 — hitter daily fraction uses ~145 games, not 162.

The last open LOW finding from the 2026-05-14 DCV-engine audit. A regular
position player appears in ~145 of 162 team games, so spreading a hitter's
season counting projection across the full 162 under-weighted every hitter's
daily DCV by ~10% (145/162). The fix divides hitters by
``_HITTER_GAMES_PER_SEASON`` (145) in both the rate-stat daily fraction and
the counting-stat per-game path, while pitchers keep ``_FULL_SEASON_GAMES``
(162) — their per-appearance frequency is carried by the role daily fraction.
"""

from __future__ import annotations

import pandas as pd

import src.optimizer.daily_optimizer as daily_optimizer
from src.optimizer.daily_optimizer import (
    _FULL_SEASON_GAMES,
    _HITTER_GAMES_PER_SEASON,
    build_daily_dcv_table,
)


def test_hitter_games_constant_value() -> None:
    assert _HITTER_GAMES_PER_SEASON == 145.0
    # Hitters and pitchers must use different season divisors.
    assert _HITTER_GAMES_PER_SEASON != _FULL_SEASON_GAMES


def _one_hitter_roster() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "Reg Bat",
                "positions": "OF",
                "team": "NYY",
                "is_hitter": 1,
                "r": 90,
                "hr": 36,
                "rbi": 100,
                "sb": 12,
                "avg": 0.280,
                "obp": 0.360,
                "ab": 580,
                "h": 162,
                "bb": 55,
                "hbp": 5,
                "sf": 5,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "era": 0.0,
                "whip": 0.0,
                "status": "active",
            }
        ]
    )


def _schedule_team_playing() -> list[dict]:
    return [{"home_team": "NYY", "away_team": "BOS"}]


def test_hitter_counting_dcv_uses_145_divisor(monkeypatch) -> None:
    """Counting-stat DCV must scale by 162/145 vs the old full-season divisor."""
    roster = _one_hitter_roster()
    schedule = _schedule_team_playing()

    dcv_145 = build_daily_dcv_table(roster, None, schedule, {})
    hr_145 = float(dcv_145.iloc[0]["dcv_hr"])

    # Monkeypatch the constant back to 162 → simulates the pre-fix behavior.
    monkeypatch.setattr(daily_optimizer, "_HITTER_GAMES_PER_SEASON", 162.0)
    dcv_162 = build_daily_dcv_table(roster, None, schedule, {})
    hr_162 = float(dcv_162.iloc[0]["dcv_hr"])

    assert hr_145 > 0
    assert hr_162 > 0
    # 145 divisor yields a larger daily DCV than 162, by ~162/145 = 1.117.
    assert hr_145 > hr_162
    assert hr_145 / hr_162 == __import__("pytest").approx(162.0 / 145.0, rel=1e-3)


def test_pitcher_counting_dcv_unaffected_by_hitter_constant(monkeypatch) -> None:
    """Changing the hitter divisor must NOT alter a pitcher's counting DCV."""
    roster = pd.DataFrame(
        [
            {
                "player_id": 2,
                "name": "Closer Guy",
                "positions": "RP",
                "team": "NYY",
                "is_hitter": 0,
                "r": 0,
                "hr": 0,
                "rbi": 0,
                "sb": 0,
                "avg": 0.0,
                "obp": 0.0,
                "w": 5,
                "l": 3,
                "sv": 30,
                "k": 90,
                "ip": 65,
                "er": 22,
                "bb_allowed": 20,
                "h_allowed": 50,
                "era": 3.05,
                "whip": 1.08,
                "status": "active",
            }
        ]
    )
    schedule = _schedule_team_playing()

    dcv_a = build_daily_dcv_table(roster, None, schedule, {})
    k_a = float(dcv_a.iloc[0]["dcv_k"])

    monkeypatch.setattr(daily_optimizer, "_HITTER_GAMES_PER_SEASON", 100.0)
    dcv_b = build_daily_dcv_table(roster, None, schedule, {})
    k_b = float(dcv_b.iloc[0]["dcv_k"])

    assert k_a == k_b

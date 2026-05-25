"""Guard: Kalman true-talent variance is wired into the weekly matrix.

Item 5 (2026-05-25): the design-Q3 follow-up shipped the kalman_var
plumbing in _category_win_prob but no caller built the variances. This
turns the plumbing into a real caller: build_team_kalman_variances derives
per-team per-category true-talent variance from YTD sample sizes, and
compute_weekly_matrix(enable_kalman=True) threads it in. A roster resting on
little YTD evidence carries more true-talent uncertainty, widening the
per-week win-probability distribution toward 0.5.
"""

from __future__ import annotations

import pandas as pd

from src.engine.output.weekly_matrix import (
    build_team_kalman_variances,
    compute_weekly_matrix,
)
from src.valuation import LeagueConfig

CONFIG = LeagueConfig()
CATS = list(CONFIG.all_categories)


def _hitter(pid: int, ytd_pa: float) -> dict:
    return {
        "player_id": pid,
        "name": f"H{pid}",
        "is_hitter": 1,
        "positions": "OF",
        "r": 80,
        "hr": 22,
        "rbi": 80,
        "sb": 12,
        "h": 140,
        "ab": 520,
        "bb": 55,
        "hbp": 5,
        "sf": 5,
        "pa": 585,
        "avg": 0.27,
        "obp": 0.34,
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "ip": 0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
        "era": 0,
        "whip": 0,
        "ytd_pa": ytd_pa,
        "ytd_ip": 0,
    }


def test_builder_returns_all_cats() -> None:
    pool = pd.DataFrame([_hitter(1, 400), _hitter(2, 400)])
    var = build_team_kalman_variances([1, 2], pool, CATS, CONFIG.season_weeks, CONFIG)
    assert set(var.keys()) == set(CATS)
    assert all(v >= 0.0 for v in var.values())


def test_low_sample_roster_has_more_variance() -> None:
    """Rookie roster (low YTD PA) carries more true-talent variance than a
    fully-sampled veteran roster with the same projections."""
    veterans = pd.DataFrame([_hitter(1, 600), _hitter(2, 600)])
    rookies = pd.DataFrame([_hitter(3, 5), _hitter(4, 5)])
    vet_var = build_team_kalman_variances([1, 2], veterans, CATS, CONFIG.season_weeks, CONFIG)
    rook_var = build_team_kalman_variances([3, 4], rookies, CATS, CONFIG.season_weeks, CONFIG)
    # HR is a counting hitting cat with a non-zero mean → variance should be
    # strictly larger for the under-sampled roster.
    assert rook_var["HR"] > vet_var["HR"]
    assert rook_var["R"] > vet_var["R"]


def test_empty_roster_returns_zeros() -> None:
    pool = pd.DataFrame([_hitter(1, 400)])
    var = build_team_kalman_variances([999], pool, CATS, CONFIG.season_weeks, CONFIG)
    assert all(v == 0.0 for v in var.values())


def test_enable_kalman_shrinks_win_probs_toward_half() -> None:
    """Adding true-talent variance pulls a favorable matchup toward 0.5."""
    me = [_hitter(1, 5), _hitter(2, 5), _hitter(3, 5)]  # under-sampled
    opp = [_hitter(4, 5), _hitter(5, 5), _hitter(6, 5)]
    # Make "me" clearly stronger in HR.
    for r in me:
        r["hr"] = 35
    pool = pd.DataFrame(me + opp)
    rosters = {"Me": [1, 2, 3], "Opp": [4, 5, 6]}
    schedule = {10: "Opp"}

    base = compute_weekly_matrix(rosters["Me"], pool, schedule, rosters, config=CONFIG, enable_kalman=False)
    kal = compute_weekly_matrix(rosters["Me"], pool, schedule, rosters, config=CONFIG, enable_kalman=True)

    base_hr = float(base["matrix"].loc[10, "HR"])
    kal_hr = float(kal["matrix"].loc[10, "HR"])
    # Favorable HR edge (>0.5) should move closer to 0.5 once Kalman
    # uncertainty is added for these under-sampled rosters.
    assert base_hr > 0.5
    assert kal_hr < base_hr
    assert "kalman" in kal["method"]

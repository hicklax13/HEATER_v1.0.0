"""Regression guard for #S1365 (2026-06-16): the playoff sim must score
OPPONENTS with the same majority-win-probability method (the copula path) as
the user — not a separate inline normal-approximation.

Root cause: ``simulate_playoff_outcomes`` scored the user's weekly win prob via
``_prob_majority_cat_wins`` (copula) while scoring opponents via an inline
independent normal-approximation in both opponent paths
(``_team_average_weekly_win_prob`` / Path B and
``_compute_per_week_per_team_p_win`` / Path A). The normal-approx is
over-confident: it SATURATES a moderately-strong team's weekly win prob toward
~1.0, collapsing its win-count variance and locking it into a top seed. With
several saturated rivals the user is mathematically excluded → exactly 0.0%
playoff odds even for a competitive team.

The fix routes BOTH opponent paths through ``_prob_majority_cat_wins`` with the
same ``correlate_categories`` flag the user uses, so every team is scored with
one method. These tests lock that routing at the function boundary: each
opponent path must reproduce the copula value exactly. They fail on the pre-fix
code (which returned the normal-approx value) and pass after.

(The de-saturation *behaviour* of the copula path itself is covered by
``test_playoff_championship_sim.test_prob_majority_cat_wins_*``; here we only
guard that opponents actually go through it.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.engine.output.playoff_sim import (
    _compute_per_week_per_team_p_win,
    _prob_majority_cat_wins,
    _team_average_weekly_win_prob,
)
from src.engine.output.weekly_matrix import _category_win_prob, _per_week_means
from src.valuation import LeagueConfig

CFG = LeagueConfig()
CATS = list(CFG.all_categories)
INV = set(CFG.inverse_stats)
SW = int(CFG.season_weeks)
CV = {c: 0.15 for c in CATS}
SEED = 7


def _player(pid: int, scale: float, is_hitter: bool) -> dict:
    if is_hitter:
        return {
            "player_id": pid,
            "name": f"P{pid}",
            "player_name": f"P{pid}",
            "is_hitter": 1,
            "positions": "OF",
            "status": "active",
            "r": 80 * scale,
            "hr": 24 * scale,
            "rbi": 82 * scale,
            "sb": 12 * scale,
            "h": 150 * scale,
            "ab": 540,
            "bb": 60,
            "hbp": 5,
            "sf": 5,
            "pa": 610,
            "avg": 0.250 * scale,
            "obp": 0.320 * scale,
            "w": 0,
            "l": 0,
            "sv": 0,
            "k": 0,
            "ip": 0,
            "ytd_ip": 0,
            "er": 0,
            "bb_allowed": 0,
            "h_allowed": 0,
            "era": 0,
            "whip": 0,
        }
    return {
        "player_id": pid,
        "name": f"P{pid}",
        "player_name": f"P{pid}",
        "is_hitter": 0,
        "positions": "SP",
        "status": "active",
        "r": 0,
        "hr": 0,
        "rbi": 0,
        "sb": 0,
        "h": 0,
        "ab": 0,
        "bb": 0,
        "hbp": 0,
        "sf": 0,
        "pa": 0,
        "avg": 0,
        "obp": 0,
        "w": 13 * scale,
        "l": 8 / scale,
        "sv": 0,
        "k": 190 * scale,
        "ip": 180,
        "ytd_ip": 0,
        "er": 75 / scale,
        "bb_allowed": 50,
        "h_allowed": 150,
        "era": 3.8 / scale,
        "whip": 1.20 / scale,
    }


def _team(pid: int, scale: float):
    rows, ids = [], []
    for k in range(4):  # 2 hitters, 2 pitchers → all 12 cats differentiate
        rows.append(_player(pid, scale, is_hitter=(k % 2 == 0)))
        ids.append(pid)
        pid += 1
    return rows, ids


def _build(specs: list[tuple[str, float]]) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    rows: list[dict] = []
    rosters: dict[str, list[int]] = {}
    pid = 1
    for name, scale in specs:
        trows, tids = _team(pid, scale)
        rows.extend(trows)
        rosters[name] = tids
        pid += len(tids)
    return pd.DataFrame(rows), rosters


def test_team_average_win_prob_routes_through_copula() -> None:
    """Path B opponent scoring (`_team_average_weekly_win_prob`) must reproduce
    the copula majority value, not the inline normal-approximation."""
    pool, rosters = _build([("STRONG", 1.20), ("MID", 1.00), ("WEAK", 0.82)])

    out = _team_average_weekly_win_prob(rosters, pool, CFG, CV, correlate_categories=True, seed=SEED)

    team_means = {t: _per_week_means(ids, pool, CATS, SW) for t, ids in rosters.items()}
    league_avg = {c: float(np.mean([team_means[t].get(c, 0.0) for t in team_means])) for c in CATS}
    for team in ("STRONG", "MID", "WEAK"):
        p_per_cat = np.array(
            [
                _category_win_prob(
                    mu_h=team_means[team].get(c, 0.0), mu_o=league_avg.get(c, 0.0), cv=0.15, inverse=c in INV
                )
                for c in CATS
            ]
        )
        expected_copula = float(_prob_majority_cat_wins(p_per_cat[None, :], correlate_categories=True, seed=SEED)[0])
        assert abs(out[team] - expected_copula) < 1e-9, (
            f"{team}: opponent average win-prob must be the copula majority value "
            "(same method as the user), not the inline normal-approximation (#S1365)"
        )


def test_per_week_per_team_matrix_routes_through_copula() -> None:
    """Path A opponent scoring (`_compute_per_week_per_team_p_win`) must reproduce
    the copula majority value, and keep P(A)+P(B)==1."""
    pool, rosters = _build([("STRONG", 1.20), ("WEAK", 0.85)])
    team_names = sorted(rosters.keys())  # ["STRONG", "WEAK"]
    schedule = {1: [("STRONG", "WEAK")]}

    matrix = _compute_per_week_per_team_p_win(
        full_league_schedule=schedule,
        all_team_rosters=rosters,
        player_pool=pool,
        config=CFG,
        variance_cv=CV,
        weeks_to_include=[1],
        team_names=team_names,
        correlate_categories=True,
        seed=SEED,
    )
    p_strong = float(matrix[team_names.index("STRONG"), 0])
    p_weak = float(matrix[team_names.index("WEAK"), 0])

    means_s = _per_week_means(rosters["STRONG"], pool, CATS, SW)
    means_w = _per_week_means(rosters["WEAK"], pool, CATS, SW)
    p_per_cat = np.array(
        [
            _category_win_prob(mu_h=means_s.get(c, 0.0), mu_o=means_w.get(c, 0.0), cv=0.15, inverse=c in INV)
            for c in CATS
        ]
    )
    expected_copula = float(_prob_majority_cat_wins(p_per_cat[None, :], correlate_categories=True, seed=SEED)[0])

    assert abs(p_strong - expected_copula) < 1e-9, (
        "Path A per-team-per-week matrix must use the copula majority method, "
        "not the inline normal-approximation (#S1365)"
    )
    assert abs(p_strong + p_weak - 1.0) < 1e-9, "P(A wins) + P(B wins) must equal 1"

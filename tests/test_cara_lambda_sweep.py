"""Guard: CARA risk-aversion λ-sensitivity sweep.

Item 6 (2026-05-25): per report Sections H.10 / C.7, the risk-adjusted
utility should be reported across a λ sweep {0.05, 0.15, 0.30} so the user
can see how the recommendation shifts with their risk stance, not just at
the single calibrated λ=0.15.
"""

from __future__ import annotations

import pandas as pd

from src.engine.output.playoff_sim import _LAMBDA_SWEEP, simulate_trade_playoff_delta
from src.valuation import LeagueConfig

CONFIG = LeagueConfig()


def _mini_league(n_teams: int = 12):
    rows = []
    rosters: dict[str, list[int]] = {}
    wins: dict[str, int] = {}
    pid = 1
    for i in range(n_teams):
        team = f"Team_{i}"
        rosters[team] = []
        wins[team] = 4 + i // 3
        scale = 0.70 + 0.05 * i
        for _ in range(3):
            rows.append(
                {
                    "player_id": pid,
                    "name": f"P{pid}",
                    "is_hitter": 1 if pid % 2 == 0 else 0,
                    "positions": "OF" if pid % 2 == 0 else "SP",
                    "r": 80 * scale,
                    "hr": 22 * scale,
                    "rbi": 80 * scale,
                    "sb": 12 * scale,
                    "h": 130 * scale,
                    "ab": 500,
                    "bb": 55,
                    "hbp": 5,
                    "sf": 5,
                    "pa": 570,
                    "avg": 0.26 * scale,
                    "obp": 0.33 * scale,
                    "w": 12 * scale,
                    "l": 8 / scale,
                    "sv": 0,
                    "k": 180 * scale,
                    "ip": 180,
                    "er": 70 / scale,
                    "bb_allowed": 50,
                    "h_allowed": 160,
                    "era": 3.5 / scale,
                    "whip": 1.17 / scale,
                }
            )
            rosters[team].append(pid)
            pid += 1
    return pd.DataFrame(rows), rosters, wins


def test_sweep_present_with_expected_keys() -> None:
    pool, rosters, wins = _mini_league()
    sched = {w: t for w, t in zip(range(15, 26), [t for t in rosters if t != "Team_8"], strict=False)}
    res = simulate_trade_playoff_delta(
        before_roster_ids=rosters["Team_8"],
        after_roster_ids=rosters["Team_8"][:-1] + [1],
        user_team_name="Team_8",
        all_team_rosters=rosters,
        user_schedule=sched,
        current_wins=wins,
        player_pool=pool,
        config=CONFIG,
        n_sims=2000,
    )
    assert "cara_utility_sweep" in res
    sweep = res["cara_utility_sweep"]
    assert set(sweep.keys()) == set(_LAMBDA_SWEEP)
    # Central λ matches the standalone cara_utility (both from same e/var).
    assert abs(sweep[0.15] - res["cara_utility"]) < 1e-6


def test_higher_lambda_not_greater_when_variance_positive() -> None:
    """CARA = E - (λ/2)Var. With Var ≥ 0, utility is non-increasing in λ."""
    pool, rosters, wins = _mini_league()
    sched = {w: t for w, t in zip(range(15, 26), [t for t in rosters if t != "Team_8"], strict=False)}
    res = simulate_trade_playoff_delta(
        before_roster_ids=rosters["Team_8"],
        after_roster_ids=rosters["Team_8"][:-1] + [1],
        user_team_name="Team_8",
        all_team_rosters=rosters,
        user_schedule=sched,
        current_wins=wins,
        player_pool=pool,
        config=CONFIG,
        n_sims=2000,
    )
    sweep = res["cara_utility_sweep"]
    assert sweep[0.05] >= sweep[0.15] >= sweep[0.30] - 1e-9

"""Diagnostic (2026-06-16, #S1365): quantify the user-vs-opponent scoring
asymmetry in playoff_sim and test whether it can drive ~0% playoff odds for a
good-roster / low-record team (the live Hickey symptom).

Part 1: directly measure normal-approx vs copula majority prob for the SAME
        per-cat probabilities, across team-strength levels (isolates METHOD).
Part 2: build a realistic 12-team league where the USER has a strong roster
        but a LOW current-win record, run the full sim, and print the
        per-team projected picture + playoff_prob (mirrors the live diag).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.engine.output.playoff_sim import _prob_majority_cat_wins, simulate_playoff_outcomes
from src.engine.output.weekly_matrix import _category_win_prob, _per_week_means
from src.valuation import LeagueConfig

CFG = LeagueConfig()
CATS = list(CFG.all_categories)
INV = set(CFG.inverse_stats)
SW = int(CFG.season_weeks)


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


def _team_rows(start_pid: int, scale: float) -> tuple[list[dict], list[int]]:
    rows, ids = [], []
    for k in range(4):  # 2 hitters, 2 pitchers
        rows.append(_player(start_pid, scale, is_hitter=(k % 2 == 0)))
        ids.append(start_pid)
        start_pid += 1
    return rows, ids


def _round_robin(team_names: list[str], n_weeks: int) -> dict[int, list[tuple[str, str]]]:
    n = len(team_names)
    fixed = team_names[0]
    others = team_names[1:]
    sched: dict[int, list[tuple[str, str]]] = {}
    for w_idx in range(n_weeks):
        rot = w_idx % (n - 1)
        rotated = others[rot:] + others[:rot]
        m = [(fixed, rotated[0])]
        for i in range(1, (n - 1) // 2 + 1):
            m.append((rotated[i], rotated[-i]))
        sched[w_idx + 1] = m
    return sched


# ── Part 1: method gap (normal-approx vs copula) for the same per-cat probs ──
print("=" * 72)
print("PART 1 — method gap: P(win majority) normal-approx vs copula, same probs")
print("=" * 72)
# Build a reference 'league average' roster (scale 1.0) and probe rosters.
avg_rows, avg_ids = _team_rows(1000, 1.0)
avg_pool = pd.DataFrame(avg_rows)
avg_means = _per_week_means(avg_ids, avg_pool, CATS, SW)
print(f"{'roster scale':>12} | {'p_normal':>9} | {'p_copula':>9} | {'gap (N-C)':>9}")
for scale in (0.80, 0.90, 1.00, 1.10, 1.20, 1.30):
    rows, ids = _team_rows(2000 + int(scale * 100) * 10, scale)
    pool = pd.DataFrame(rows)
    means = _per_week_means(ids, pool, CATS, SW)
    p_per_cat = np.array(
        [
            _category_win_prob(mu_h=means.get(c, 0.0), mu_o=avg_means.get(c, 0.0), cv=0.15, inverse=c in INV)
            for c in CATS
        ]
    )
    # normal-approx (current opponent method)
    mean_cats = p_per_cat.sum()
    var_cats = (p_per_cat * (1 - p_per_cat)).sum()
    from scipy.stats import norm

    z = (len(CATS) / 2.0 - mean_cats) / float(np.sqrt(max(var_cats, 1e-12)))
    p_normal = float(1.0 - norm.cdf(z))
    # copula (current user method)
    p_copula = float(_prob_majority_cat_wins(p_per_cat[None, :], correlate_categories=True, seed=7)[0])
    print(f"{scale:>12.2f} | {p_normal:>9.3f} | {p_copula:>9.3f} | {p_normal - p_copula:>+9.3f}")


# ── Part 2: realistic league, strong-roster / low-record user ───────────────
print()
print("=" * 72)
print("PART 2 — realistic league: USER has a STRONG roster but LOW current wins")
print("=" * 72)
# Strengths spread across the league; current wins roughly track strength
# EXCEPT the user, who is strong (scale 1.25) but stuck at 4 current wins.
team_specs = {
    "T00_USER": (1.10, 6),  # solid, on the bubble for the 4th seed
    "T01": (1.22, 9),
    "T02": (1.18, 8),
    "T03": (1.15, 8),  # clear top 3
    "T04": (1.12, 7),
    "T05": (1.11, 6),
    "T06": (1.09, 6),  # cluster w/ user
    "T07": (1.00, 5),
    "T08": (0.95, 4),
    "T09": (0.88, 3),
    "T10": (0.83, 3),
    "T11": (0.80, 2),
}
rows: list[dict] = []
rosters: dict[str, list[int]] = {}
current_wins: dict[str, int] = {}
pid = 1
for name, (scale, cw) in team_specs.items():
    trows, tids = _team_rows(pid, scale)
    rows.extend(trows)
    rosters[name] = tids
    current_wins[name] = cw
    pid += len(tids)
pool = pd.DataFrame(rows)

team_names = list(team_specs.keys())
n_weeks = 13
full_sched = _round_robin(team_names, n_weeks)
user_team = "T00_USER"
user_sched: dict[int, str] = {}
for wk, ms in full_sched.items():
    for a, b in ms:
        if a == user_team:
            user_sched[wk] = b
        elif b == user_team:
            user_sched[wk] = a

res = simulate_playoff_outcomes(
    user_roster_ids=rosters[user_team],
    user_team_name=user_team,
    all_team_rosters=rosters,
    user_schedule=user_sched,
    current_wins=current_wins,
    player_pool=pool,
    config=CFG,
    n_sims=8000,
    seed=7,
    full_league_schedule=full_sched,
)
print(
    f"playoff_prob = {res['playoff_prob'] * 100:.2f}%   "
    f"E[additional wins] = {res['mean_regular_season_wins']:.1f}   "
    f"weeks_remaining = {res['n_weeks_remaining']}"
)
print(f"{'team':>10} | {'now':>3} | {'+proj':>6} | {'=final':>6} | {'wk_p':>5}")
pf = res["team_proj_final"]
pa = res["team_proj_additional"]
wp = res["team_weekly_p"]
cw = res["current_wins_used"]
for rank, (t, fin) in enumerate(sorted(pf.items(), key=lambda kv: kv[1], reverse=True), 1):
    star = " <= USER" if t == user_team else ""
    print(f"{t:>10} | {cw.get(t, 0):>3} | {pa.get(t, 0):>+6.1f} | {fin:>6.1f} | {wp.get(t, 0):>5.2f}{star}")
    if rank == 4:
        print(f"{'':>10} |  -- playoff cutline (top 4) --")

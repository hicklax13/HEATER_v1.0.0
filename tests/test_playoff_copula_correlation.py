"""TE-E2: copula-correlate the playoff-sim weekly category outcomes.

`_prob_majority_cat_wins` aggregated 12 per-category win probabilities into
P(majority) via a Poisson-binomial NORMAL approximation that assumed the 12
categories are INDEPENDENT within a week. In reality HR/RBI/R move together
and ERA/WHIP/K move together (the engine already owns DEFAULT_CORRELATION in
copula.py — the same matrix LO-E3 + the Matchup Planner use). Independence
understates weekly-cat-win variance → over-confident playoff/champ odds.

These tests verify:
  1. The correlated path yields a WIDER weekly-cat-win distribution (pulled
     toward 0.5, less over-confident) than the independent approx for a
     boom/bust (uniformly-favored, correlated) roster.
  2. The independent normal-approx fast fallback is preserved + reachable.
  3. simulate_playoff_outcomes accepts correlate_categories (default ON),
     keeps champ_prob <= playoff_prob, probabilities in [0, 1].
  4. Identical rosters → zero delta (paired-MC discipline) under the
     correlated path.
  5. Seed determinism under the correlated path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.engine.output.playoff_sim import (
    _prob_majority_cat_wins,
    simulate_playoff_outcomes,
    simulate_trade_playoff_delta,
)

# ── 1. Correlated path widens the weekly-cat-win distribution ──────────


def test_correlated_majority_prob_less_overconfident() -> None:
    """For a uniformly-favored roster, correlation pulls P(majority) toward 0.5.

    Every one of the 12 categories is favored at p=0.65. Under independence
    the win-count variance is small, so P(win >= 7 of 12) is high. Positive
    within-cluster correlation (HR/RBI/R, ERA/WHIP/K) inflates the win-count
    variance, so the correlated estimate must be LOWER (closer to 0.5) — less
    over-confident.
    """
    p = np.full((1, 12), 0.65)

    indep = _prob_majority_cat_wins(p, correlate_categories=False)[0]
    corr = _prob_majority_cat_wins(p, correlate_categories=True, seed=42)[0]

    # Both still favor the user (>0.5), but the correlated estimate is less
    # extreme (variance widened by positive intra-cluster correlation).
    assert indep > 0.5
    assert corr > 0.5
    assert corr < indep, (
        f"Correlated P(majority) ({corr:.3f}) must be less over-confident than "
        f"the independent approx ({indep:.3f}) for a uniformly-favored roster"
    )


def test_correlated_majority_symmetric_for_underdog() -> None:
    """For a uniformly-DISfavored roster, correlation pulls P(majority) UP toward 0.5.

    Mirror of the above: every cat at p=0.35. Independence → low P(majority);
    correlation widens variance → estimate closer to 0.5 (higher).
    """
    p = np.full((1, 12), 0.35)

    indep = _prob_majority_cat_wins(p, correlate_categories=False)[0]
    corr = _prob_majority_cat_wins(p, correlate_categories=True, seed=42)[0]

    assert indep < 0.5
    assert corr < 0.5
    assert corr > indep, (
        f"Correlated P(majority) ({corr:.3f}) must be pulled UP toward 0.5 vs "
        f"the independent approx ({indep:.3f}) for a uniform underdog"
    )


def test_correlated_majority_50_50_stays_50_50() -> None:
    """All cats at exactly 0.5 → P(majority) ~0.5 under both paths."""
    p = np.full((1, 12), 0.5)
    corr = _prob_majority_cat_wins(p, correlate_categories=True, seed=7)[0]
    assert 0.42 < corr < 0.58


def test_prob_majority_default_is_independent_backcompat() -> None:
    """Default call (no flag) preserves the legacy independent normal-approx.

    The existing simulate paths and guard tests call _prob_majority_cat_wins
    positionally with no flag — default must equal correlate_categories=False
    byte-for-byte so nothing silently shifts.
    """
    p = np.array([[0.7, 0.6, 0.55, 0.4, 0.65, 0.5, 0.45, 0.5, 0.3, 0.6, 0.55, 0.5]])
    default = _prob_majority_cat_wins(p)
    explicit_indep = _prob_majority_cat_wins(p, correlate_categories=False)
    assert np.allclose(default, explicit_indep)


# ── Mini-league fixture (mirrors test_playoff_championship_sim) ────────


def _build_mini_league(n_teams: int = 12):
    rows = []
    rosters: dict[str, list[int]] = {}
    current_wins: dict[str, int] = {}
    pid = 1
    for team_idx in range(n_teams):
        team_name = f"Team_{team_idx}"
        rosters[team_name] = []
        current_wins[team_name] = 4 + team_idx // 3
        scale = 0.70 + 0.05 * team_idx
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
                    "avg": 0.260 * scale,
                    "obp": 0.330 * scale,
                    "w": 12 * scale,
                    "l": 8 / scale,
                    "sv": 0,
                    "k": 180 * scale,
                    "ip": 180,
                    "er": 70 / scale,
                    "bb_allowed": 50,
                    "h_allowed": 160,
                    "era": 3.50 / scale,
                    "whip": 1.17 / scale,
                }
            )
            rosters[team_name].append(pid)
            pid += 1
    return pd.DataFrame(rows), rosters, current_wins


def _build_schedule(opponents: list[str], start_week: int, end_week: int) -> dict[int, str]:
    weeks = list(range(start_week, end_week + 1))
    return {w: opponents[(w - start_week) % len(opponents)] for w in weeks}


# ── 3. simulate_playoff_outcomes correlate_categories flag ────────────


def test_simulate_accepts_correlate_flag_and_invariants_hold() -> None:
    pool, rosters, wins = _build_mini_league()
    schedule = _build_schedule([t for t in rosters if t != "Team_8"], 10, 26)
    result = simulate_playoff_outcomes(
        user_roster_ids=rosters["Team_8"],
        user_team_name="Team_8",
        all_team_rosters=rosters,
        user_schedule=schedule,
        current_wins=wins,
        player_pool=pool,
        n_sims=2000,
        seed=123,
        correlate_categories=True,
    )
    assert 0.0 <= result["playoff_prob"] <= 1.0
    assert 0.0 <= result["champ_prob"] <= 1.0
    assert result["champ_prob"] <= result["playoff_prob"] + 1e-9
    # Method tag should advertise the correlated path.
    assert "corr" in result["method"]


def test_simulate_default_uses_correlated_path() -> None:
    """Default (no flag) should adopt the better correlated path per TE-E2."""
    pool, rosters, wins = _build_mini_league()
    schedule = _build_schedule([t for t in rosters if t != "Team_5"], 10, 26)
    result = simulate_playoff_outcomes(
        user_roster_ids=rosters["Team_5"],
        user_team_name="Team_5",
        all_team_rosters=rosters,
        user_schedule=schedule,
        current_wins=wins,
        player_pool=pool,
        n_sims=1500,
        seed=99,
    )
    assert "corr" in result["method"]


# ── 4. Paired-MC discipline + 5. determinism ──────────────────────────


def test_correlated_identical_rosters_zero_delta() -> None:
    """Before == After → exact-zero delta under the correlated path."""
    pool, rosters, wins = _build_mini_league()
    schedule = _build_schedule([t for t in rosters if t != "Team_5"], 10, 26)
    result = simulate_trade_playoff_delta(
        before_roster_ids=rosters["Team_5"],
        after_roster_ids=rosters["Team_5"],
        user_team_name="Team_5",
        all_team_rosters=rosters,
        user_schedule=schedule,
        current_wins=wins,
        player_pool=pool,
        n_sims=2000,
        correlate_categories=True,
    )
    assert result["delta_playoff_prob"] == 0.0
    assert result["delta_champ_prob"] == 0.0


def test_correlated_playoff_sim_deterministic() -> None:
    """Same seed → identical playoff/champ probs under the correlated path."""
    pool, rosters, wins = _build_mini_league()
    schedule = _build_schedule([t for t in rosters if t != "Team_7"], 10, 26)
    kwargs = dict(
        user_roster_ids=rosters["Team_7"],
        user_team_name="Team_7",
        all_team_rosters=rosters,
        user_schedule=schedule,
        current_wins=wins,
        player_pool=pool,
        n_sims=2000,
        seed=2024,
        correlate_categories=True,
    )
    r1 = simulate_playoff_outcomes(**kwargs)
    r2 = simulate_playoff_outcomes(**kwargs)
    assert r1["playoff_prob"] == r2["playoff_prob"]
    assert r1["champ_prob"] == r2["champ_prob"]

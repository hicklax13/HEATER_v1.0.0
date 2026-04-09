# src/standings_projection.py
"""Projected season standings via copula-based H2H MC simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.validation.constant_optimizer import load_constants
from src.valuation import LeagueConfig

_CONSTANTS = load_constants()
_LC = LeagueConfig()

# Weekly stat standard deviations (tau) per category
WEEKLY_TAU: dict[str, float] = {
    "R": 6.0,
    "HR": 2.5,
    "RBI": 6.5,
    "SB": 1.5,
    "AVG": 0.020,
    "OBP": 0.025,
    "W": 1.0,
    "L": 1.0,
    "SV": 1.5,
    "K": 8.0,
    "ERA": _CONSTANTS.get("standings_tau_era"),
    "WHIP": 0.15,
}

INVERSE_CATS: set[str] = set(_LC.inverse_stats)

ALL_CATEGORIES: list[str] = list(_LC.all_categories)


def compute_category_win_probability(
    team_a_total: float,
    team_b_total: float,
    tau_a: float,
    tau_b: float,
    is_inverse: bool = False,
) -> float:
    """P(A wins category) using Normal CDF.

    For inverse stats (ERA, WHIP, L): lower is better, so we flip.
    """
    combined_sigma = max(1e-9, (tau_a**2 + tau_b**2) ** 0.5)
    if is_inverse:
        z = (team_b_total - team_a_total) / combined_sigma
    else:
        z = (team_a_total - team_b_total) / combined_sigma
    return float(norm.cdf(z))


def generate_round_robin_schedule(team_names: list[str], n_weeks: int = 22) -> list[list[tuple[str, str]]]:
    """Generate a round-robin schedule for N teams over n_weeks.

    Each week has N/2 matchups. Repeats pairings cyclically.
    Returns list of weeks, each week is list of (team_a, team_b) tuples.
    """
    n = len(team_names)
    if n < 2:
        return []

    # Circle method for guaranteed N/2 matchups per week
    teams = list(team_names)
    if n % 2 == 1:
        teams.append(None)  # BYE team for odd count
    n_circle = len(teams)

    # Generate one full round-robin cycle
    rounds = []
    fixed = teams[0]
    rotating = teams[1:]
    for _ in range(n_circle - 1):
        round_pairs = []
        current = [fixed] + rotating
        for j in range(n_circle // 2):
            a = current[j]
            b = current[n_circle - 1 - j]
            if a is not None and b is not None:
                round_pairs.append((a, b))
        rounds.append(round_pairs)
        rotating = [rotating[-1]] + rotating[:-1]

    # Repeat cyclically for n_weeks
    schedule = []
    for week_idx in range(n_weeks):
        schedule.append(rounds[week_idx % len(rounds)])

    return schedule


def simulate_season(
    team_totals: dict[str, dict[str, float]],
    categories: list[str] | None = None,
    schedule: list[list[tuple[str, str]]] | None = None,
    n_sims: int = 1000,
    seed: int = 42,
    weekly_tau: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Simulate H2H season via MC.

    Args:
        team_totals: {team_name: {category: projected_weekly_total}}
        categories: List of category names (default: ALL_CATEGORIES)
        schedule: Round-robin schedule, or auto-generated
        n_sims: Number of season simulations
        seed: Random seed
        weekly_tau: Per-category weekly standard deviations

    Returns:
        DataFrame: team_name, mean_wins, mean_losses, mean_ties,
                   win_p5, win_p95, playoff_pct
    """
    cats = categories or ALL_CATEGORIES
    tau = weekly_tau or WEEKLY_TAU
    team_names = sorted(team_totals.keys())
    n_teams = len(team_names)

    if n_teams < 2:
        return pd.DataFrame()

    if schedule is None:
        schedule = generate_round_robin_schedule(team_names)

    rng = np.random.default_rng(seed)

    # Pre-compute: per-team, per-category win probabilities for each matchup
    all_wins = np.zeros((n_sims, n_teams))
    all_losses = np.zeros((n_sims, n_teams))
    all_ties = np.zeros((n_sims, n_teams))

    team_idx = {name: i for i, name in enumerate(team_names)}

    for sim in range(n_sims):
        for week_matchups in schedule:
            for team_a, team_b in week_matchups:
                ia = team_idx[team_a]
                ib = team_idx[team_b]

                cats_won_a = 0
                cats_won_b = 0

                for cat in cats:
                    mu_a = team_totals[team_a].get(cat, 0.0)
                    mu_b = team_totals[team_b].get(cat, 0.0)
                    sigma = tau.get(cat, 1.0)

                    # Sample weekly outcome
                    val_a = rng.normal(mu_a, sigma)
                    val_b = rng.normal(mu_b, sigma)

                    is_inv = cat in INVERSE_CATS
                    if is_inv:
                        if val_a < val_b:
                            cats_won_a += 1
                        elif val_b < val_a:
                            cats_won_b += 1
                    else:
                        if val_a > val_b:
                            cats_won_a += 1
                        elif val_b > val_a:
                            cats_won_b += 1

                if cats_won_a > cats_won_b:
                    all_wins[sim, ia] += 1
                    all_losses[sim, ib] += 1
                elif cats_won_b > cats_won_a:
                    all_wins[sim, ib] += 1
                    all_losses[sim, ia] += 1
                else:
                    all_ties[sim, ia] += 1
                    all_ties[sim, ib] += 1

    # Aggregate results
    results = []
    for name in team_names:
        i = team_idx[name]
        wins = all_wins[:, i]
        losses = all_losses[:, i]
        ties = all_ties[:, i]

        # Playoff = top 4 (out of N teams)
        playoff_count = 0
        for sim in range(n_sims):
            # Rank by wins (desc), then ties (desc)
            sim_records = [(all_wins[sim, j], all_ties[sim, j], j) for j in range(n_teams)]
            sim_records.sort(key=lambda x: (x[0], x[1]), reverse=True)
            ranks = {rec[2]: rank + 1 for rank, rec in enumerate(sim_records)}
            if ranks[i] <= min(4, n_teams):
                playoff_count += 1

        results.append(
            {
                "team_name": name,
                "mean_wins": round(float(wins.mean()), 1),
                "mean_losses": round(float(losses.mean()), 1),
                "mean_ties": round(float(ties.mean()), 1),
                "win_p5": round(float(np.percentile(wins, 5)), 1),
                "win_p95": round(float(np.percentile(wins, 95)), 1),
                "playoff_pct": round(playoff_count / n_sims * 100, 1),
            }
        )

    df = pd.DataFrame(results)
    return df.sort_values("mean_wins", ascending=False).reset_index(drop=True)

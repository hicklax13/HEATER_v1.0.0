"""Standings engine — shared computation for League Standings + Matchup Planner.

Pure functions (no Streamlit dependency). All Yahoo/DB I/O happens in callers.

This module provides:
- Schedule helpers: parse scoreboard matchups, find user opponents
- Category result parsing: W/L/T determination per category
- (Future) Category win probabilities, MC season simulation, magic numbers
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# ── Schedule helpers ────────────────────────────────────────────────


def parse_scoreboard_matchups(
    matchups: list[dict],
) -> list[tuple[str, str]]:
    """Extract (team_a, team_b) pairs from a scoreboard response.

    Args:
        matchups: List of dicts with ``team_a`` and ``team_b`` keys.

    Returns:
        List of (team_a_name, team_b_name) tuples.
    """
    result: list[tuple[str, str]] = []
    for m in matchups:
        team_a = str(m.get("team_a", ""))
        team_b = str(m.get("team_b", ""))
        if team_a and team_b:
            result.append((team_a, team_b))
    return result


def find_user_opponent(
    schedule: dict[int, list[tuple[str, str]]],
    week: int,
    user_team_name: str,
) -> str | None:
    """Find the user's opponent for a specific week.

    Args:
        schedule: ``{week: [(team_a, team_b), ...]}``
        week: Week number to look up.
        user_team_name: User's team name.

    Returns:
        Opponent team name, or ``None`` if not found.
    """
    matchups = schedule.get(week, [])
    for team_a, team_b in matchups:
        if team_a == user_team_name:
            return team_b
        if team_b == user_team_name:
            return team_a
    return None


def parse_week_category_results(
    categories: list[dict],
) -> list[dict]:
    """Parse category-level W/L/T results from a completed matchup.

    Args:
        categories: List of dicts with keys ``name``, ``user_val``,
            ``opp_val``, ``is_inverse``.

    Returns:
        List of dicts with keys ``name``, ``user_val``, ``opp_val``,
        ``result`` (one of ``"W"``, ``"L"``, ``"T"``).
    """
    results: list[dict] = []
    for cat in categories:
        name = cat["name"]
        user_val = float(cat["user_val"])
        opp_val = float(cat["opp_val"])
        is_inverse = bool(cat.get("is_inverse", False))

        if user_val == opp_val:
            result = "T"
        elif is_inverse:
            result = "W" if user_val < opp_val else "L"
        else:
            result = "W" if user_val > opp_val else "L"

        results.append(
            {
                "name": name,
                "user_val": user_val,
                "opp_val": opp_val,
                "result": result,
            }
        )
    return results


# ── Category win probability engine ──────────────────────────────────

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.standings_projection import INVERSE_CATS, WEEKLY_TAU
from src.valuation import LeagueConfig

# ── Category correlation matrix ─────────────────────────────────────

CATEGORY_CORRELATIONS: dict[tuple[str, str], float] = {
    ("HR", "R"): 0.72,   ("HR", "RBI"): 0.68,   ("R", "RBI"): 0.65,
    ("AVG", "OBP"): 0.85, ("ERA", "WHIP"): 0.78,
    ("W", "K"): 0.45,     ("SB", "AVG"): -0.15,
    ("W", "L"): -0.30,    ("SV", "W"): -0.10,
}

ALL_CATEGORIES: list[str] = [
    "R", "HR", "RBI", "SB", "AVG", "OBP",
    "W", "L", "SV", "K", "ERA", "WHIP",
]


def _build_correlation_matrix(categories: list[str]) -> np.ndarray:
    """Build an NxN correlation matrix from pairwise correlations."""
    n = len(categories)
    corr = np.eye(n)
    cat_idx = {c: i for i, c in enumerate(categories)}
    for (a, b), rho in CATEGORY_CORRELATIONS.items():
        if a in cat_idx and b in cat_idx:
            i, j = cat_idx[a], cat_idx[b]
            corr[i, j] = rho
            corr[j, i] = rho
    # Ensure positive semi-definite via nearest PSD
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 1e-6)
    corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    np.fill_diagonal(corr, 1.0)
    return corr


def _estimate_team_weekly_stats(
    roster_ids: list[int],
    player_pool: pd.DataFrame,
    config: LeagueConfig,
    weeks_remaining: int = 16,
) -> dict[str, float]:
    """Estimate a team's expected weekly production per category.

    Returns: {category: weekly_mean}
    """
    roster = player_pool[player_pool["player_id"].isin(roster_ids)].copy()
    hitters = roster[roster["is_hitter"] == 1]
    pitchers = roster[roster["is_hitter"] == 0]
    weeks = max(weeks_remaining, 1)

    stats: dict[str, float] = {}

    # Counting stats — sum and scale to weekly
    for cat in ("r", "hr", "rbi", "sb"):
        stats[cat.upper()] = float(hitters[cat].sum()) / weeks if not hitters.empty else 0.0
    for cat in ("w", "l", "sv", "k"):
        stats[cat.upper()] = float(pitchers[cat].sum()) / weeks if not pitchers.empty else 0.0

    # Rate stats — weighted aggregation
    if not hitters.empty:
        total_ab = float(hitters["ab"].sum())
        total_h = float(hitters["h"].sum())
        total_pa = float(hitters["pa"].sum())
        total_bb = float(hitters["bb"].sum())
        total_hbp = float(hitters.get("hbp", pd.Series([0])).sum())
        total_sf = float(hitters.get("sf", pd.Series([0])).sum())
        stats["AVG"] = total_h / total_ab if total_ab > 0 else 0.250
        denom_obp = total_ab + total_bb + total_hbp + total_sf
        stats["OBP"] = (total_h + total_bb + total_hbp) / denom_obp if denom_obp > 0 else 0.320
    else:
        stats["AVG"] = 0.250
        stats["OBP"] = 0.320

    if not pitchers.empty:
        total_ip = float(pitchers["ip"].sum())
        total_er = float(pitchers["er"].sum())
        total_bb_p = float(pitchers["bb_allowed"].sum())
        total_h_p = float(pitchers["h_allowed"].sum())
        stats["ERA"] = (total_er * 9 / total_ip) if total_ip > 0 else 4.50
        stats["WHIP"] = (total_bb_p + total_h_p) / total_ip if total_ip > 0 else 1.30
    else:
        stats["ERA"] = 4.50
        stats["WHIP"] = 1.30

    return stats


def compute_category_win_probabilities(
    user_roster_ids: list[int],
    opp_roster_ids: list[int],
    player_pool: pd.DataFrame,
    config: LeagueConfig,
    weeks_played: int = 0,
    weeks_remaining: int = 16,
    n_sims: int = 10000,
    seed: int = 42,
) -> dict:
    """Compute per-category P(user wins) for a head-to-head matchup.

    Uses Bayesian-updated Normal margins with Gaussian copula for
    correlated overall matchup probability.

    Returns: {
        overall_win_pct, overall_tie_pct, overall_loss_pct,
        projected_score: {W, L, T},
        categories: [{name, user_proj, opp_proj, win_pct, confidence, is_inverse}]
    }
    """
    categories = ALL_CATEGORIES
    inverse = INVERSE_CATS

    user_stats = _estimate_team_weekly_stats(user_roster_ids, player_pool, config, weeks_remaining)
    opp_stats = _estimate_team_weekly_stats(opp_roster_ids, player_pool, config, weeks_remaining)

    # Per-category independent probabilities
    cat_results = []
    marginal_probs = []

    for cat in categories:
        mu_user = user_stats.get(cat, 0.0)
        mu_opp = opp_stats.get(cat, 0.0)
        is_inv = cat in inverse

        # Variance: base tau, with Bayesian shrinkage if weeks played > 0
        tau = WEEKLY_TAU.get(cat, 1.0)
        if tau is None:
            tau = 1.0  # ERA tau can be None from dynamic loading
        base_var = float(tau) ** 2

        # Bayesian shrinkage: more weeks played -> tighter variance
        if weeks_played >= 4:
            shrinkage = 1.0 / (1.0 + weeks_played / 10.0)
            var_cat = base_var * shrinkage
        else:
            var_cat = base_var

        sigma_diff = np.sqrt(2 * var_cat)  # Both teams have same variance

        if sigma_diff < 1e-10:
            p_win = 0.5
        elif is_inv:
            p_win = float(norm.cdf((mu_opp - mu_user) / sigma_diff))
        else:
            p_win = float(norm.cdf((mu_user - mu_opp) / sigma_diff))

        p_win = max(0.01, min(0.99, p_win))

        confidence = "high" if weeks_played >= 8 else "medium" if weeks_played >= 4 else "low"

        cat_results.append({
            "name": cat,
            "user_proj": round(mu_user, 3),
            "opp_proj": round(mu_opp, 3),
            "win_pct": round(p_win, 4),
            "confidence": confidence,
            "is_inverse": is_inv,
        })
        marginal_probs.append(p_win)

    # Correlated overall matchup probability via Gaussian copula
    corr_matrix = _build_correlation_matrix(categories)
    rng = np.random.default_rng(seed)

    try:
        L = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        L = np.eye(len(categories))

    z = rng.standard_normal((n_sims, len(categories)))
    correlated_z = z @ L.T

    # Convert to win/loss using marginal probabilities
    # P(win cat i) = marginal_probs[i]
    # Win if Phi(z_i) < marginal_prob[i]
    uniform = norm.cdf(correlated_z)
    wins_matrix = uniform < np.array(marginal_probs)  # (n_sims, n_cats)

    cats_won = wins_matrix.sum(axis=1)  # per sim
    n_cats = len(categories)
    half = n_cats / 2.0

    overall_win = float(np.mean(cats_won > half))
    overall_tie = float(np.mean(cats_won == half))
    overall_loss = float(np.mean(cats_won < half))

    # Projected score (expected category W-L-T)
    expected_w = float(np.mean(cats_won))
    expected_l = float(n_cats - expected_w)

    return {
        "overall_win_pct": round(overall_win, 4),
        "overall_tie_pct": round(overall_tie, 4),
        "overall_loss_pct": round(overall_loss, 4),
        "projected_score": {
            "W": round(expected_w, 1),
            "L": round(expected_l, 1),
            "T": round(n_cats - expected_w - expected_l, 1) if expected_w + expected_l < n_cats else 0.0,
        },
        "categories": cat_results,
    }


# ── Enhanced MC Season Simulation ──────────────────────────────────


def simulate_season_enhanced(
    current_standings: dict[str, dict[str, int]],
    team_weekly_totals: dict[str, dict[str, float]],
    full_schedule: dict[int, list[tuple[str, str]]],
    current_week: int = 1,
    n_sims: int = 1000,
    seed: int = 42,
    momentum_data: dict[str, float] | None = None,
    playoff_spots: int = 4,
) -> dict:
    """Schedule-aware MC season simulation.

    Args:
        current_standings: {team: {W, L, T}} -- current H2H record.
        team_weekly_totals: {team: {cat: weekly_mean}} -- projected weekly averages.
        full_schedule: {week: [(team_a, team_b), ...]} -- actual matchups.
        current_week: First week to simulate (prior weeks use actual results).
        n_sims: Monte Carlo iterations.
        seed: RNG seed.
        momentum_data: {team: float} -- momentum multiplier [0.5, 2.0].
        playoff_spots: Number of playoff spots (default 4).

    Returns: {
        projected_records: {team: {W, L, T, win_pct}},
        playoff_probability: {team: float},
        confidence_intervals: {team: {p5_wins, p95_wins}},
        strength_of_schedule: {team: float},
    }
    """
    teams = list(current_standings.keys())
    n_teams = len(teams)
    categories = ALL_CATEGORIES
    n_cats = len(categories)
    half_cats = n_cats / 2.0

    # Build correlation matrix and Cholesky
    corr_matrix = _build_correlation_matrix(categories)
    try:
        L = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        L = np.eye(n_cats)

    rng = np.random.default_rng(seed)

    # Get tau values
    tau_vals = []
    for cat in categories:
        t = WEEKLY_TAU.get(cat, 1.0)
        tau_vals.append(float(t) if t is not None else 1.0)
    tau_arr = np.array(tau_vals)

    # Build team means matrix (n_teams x n_cats)
    team_means = np.zeros((n_teams, n_cats))
    for ti, team in enumerate(teams):
        totals = team_weekly_totals.get(team, {})
        for ci, cat in enumerate(categories):
            val = totals.get(cat, 0.0)
            # Apply momentum adjustment if available
            if momentum_data and team in momentum_data:
                mom = momentum_data[team]
                # Momentum adjusts mean by +/-5% max
                adjustment = 1.0 + 0.05 * (mom - 1.0)
                val *= adjustment
            team_means[ti, ci] = val

    # Determine remaining weeks
    all_weeks = sorted(full_schedule.keys())
    remaining_weeks = [w for w in all_weeks if w >= current_week]
    n_remaining = len(remaining_weeks)

    # Inverse category mask
    inverse_mask = np.array([cat in INVERSE_CATS for cat in categories])

    # Simulation
    win_counts = np.zeros((n_sims, n_teams))
    loss_counts = np.zeros((n_sims, n_teams))
    tie_counts = np.zeros((n_sims, n_teams))

    for sim in range(n_sims):
        # Start with current records
        sim_wins = np.array([current_standings[t]["W"] for t in teams], dtype=float)
        sim_losses = np.array([current_standings[t]["L"] for t in teams], dtype=float)
        sim_ties = np.array([current_standings[t]["T"] for t in teams], dtype=float)

        for week in remaining_weeks:
            matchups = full_schedule.get(week, [])
            for team_a, team_b in matchups:
                if team_a not in teams or team_b not in teams:
                    continue
                ti_a = teams.index(team_a)
                ti_b = teams.index(team_b)

                # Generate correlated noise for both teams
                z_a = rng.standard_normal(n_cats)
                z_b = rng.standard_normal(n_cats)
                noise_a = L @ z_a * tau_arr
                noise_b = L @ z_b * tau_arr

                vals_a = team_means[ti_a] + noise_a
                vals_b = team_means[ti_b] + noise_b

                # Count category wins
                a_wins_cat = np.where(
                    inverse_mask,
                    vals_a < vals_b,  # inverse: lower is better
                    vals_a > vals_b,  # normal: higher is better
                )
                cats_won_a = int(a_wins_cat.sum())
                cats_won_b = int((~a_wins_cat.astype(bool) & (vals_a != vals_b)).sum())

                if cats_won_a > half_cats:
                    sim_wins[ti_a] += 1
                    sim_losses[ti_b] += 1
                elif cats_won_b > half_cats:
                    sim_wins[ti_b] += 1
                    sim_losses[ti_a] += 1
                else:
                    sim_ties[ti_a] += 1
                    sim_ties[ti_b] += 1

        win_counts[sim] = sim_wins
        loss_counts[sim] = sim_losses
        tie_counts[sim] = sim_ties

    # Aggregate results
    mean_wins = win_counts.mean(axis=0)
    mean_losses = loss_counts.mean(axis=0)
    mean_ties = tie_counts.mean(axis=0)
    p5_wins = np.percentile(win_counts, 5, axis=0)
    p95_wins = np.percentile(win_counts, 95, axis=0)

    # Playoff probability: top `playoff_spots` by wins
    playoff_counts = np.zeros(n_teams)
    for sim in range(n_sims):
        ranked = np.argsort(-win_counts[sim])
        for spot in range(min(playoff_spots, n_teams)):
            playoff_counts[ranked[spot]] += 1
    playoff_probs = playoff_counts / n_sims

    # Strength of schedule: avg opponent quality
    total_games_per_team = {
        t: current_standings[t]["W"] + current_standings[t]["L"] + current_standings[t]["T"]
        for t in teams
    }
    sos: dict[str, float] = {}
    for ti, team in enumerate(teams):
        opp_quality = []
        total_possible = total_games_per_team[team] + n_remaining
        for week in remaining_weeks:
            for ta, tb in full_schedule.get(week, []):
                if ta == team and tb in teams:
                    oi = teams.index(tb)
                    opp_quality.append(mean_wins[oi] / max(total_possible, 1))
                elif tb == team and ta in teams:
                    oi = teams.index(ta)
                    opp_quality.append(mean_wins[oi] / max(total_possible, 1))
        sos[team] = round(float(np.mean(opp_quality)) if opp_quality else 0.5, 3)

    projected_records: dict[str, dict] = {}
    confidence_intervals: dict[str, dict] = {}
    playoff_probability: dict[str, float] = {}

    for ti, team in enumerate(teams):
        total_w = round(float(mean_wins[ti]), 1)
        total_l = round(float(mean_losses[ti]), 1)
        total_t = round(float(mean_ties[ti]), 1)
        total_matchups = total_w + total_l + total_t
        projected_records[team] = {
            "W": total_w,
            "L": total_l,
            "T": total_t,
            "win_pct": round(total_w / max(total_matchups, 1), 3),
        }
        confidence_intervals[team] = {
            "p5_wins": round(float(p5_wins[ti]), 1),
            "p95_wins": round(float(p95_wins[ti]), 1),
        }
        playoff_probability[team] = round(float(playoff_probs[ti]), 4)

    return {
        "projected_records": projected_records,
        "playoff_probability": playoff_probability,
        "confidence_intervals": confidence_intervals,
        "strength_of_schedule": sos,
    }


# ── Magic Numbers ──────────────────────────────────────────────────


def compute_magic_numbers(
    current_wins: dict[str, int | float],
    remaining_matchups: int,
    playoff_spots: int = 4,
) -> dict[str, int | None]:
    """Compute magic number (wins to clinch playoff spot) per team.

    Classic formula adapted for H2H: magic = remaining + 1 - (my_wins - Nth_place_wins)
    where N = playoff_spots + 1 (the first team outside playoffs).

    Returns: {team: magic_number} where 0 = clinched, None = eliminated.
    """
    teams_sorted = sorted(current_wins.keys(), key=lambda t: -current_wins[t])
    results: dict[str, int | None] = {}

    for team in teams_sorted:
        my_wins = current_wins[team]

        # The team we need to stay ahead of: (playoff_spots + 1)th place
        # or the best non-playoff team
        chase_teams = [t for t in teams_sorted if t != team]
        # Sort others by wins descending
        chase_teams.sort(key=lambda t: -current_wins[t])

        if len(chase_teams) < playoff_spots:
            results[team] = 0  # Auto-clinch (fewer teams than spots)
            continue

        # Nth place team's max possible wins
        nth_team = chase_teams[playoff_spots - 1]  # (playoff_spots)th OTHER team
        nth_wins = current_wins[nth_team]
        nth_max = nth_wins + remaining_matchups

        # Magic number: wins I need so that even if Nth team wins everything,
        # I'm still ahead
        magic = max(0, nth_max - my_wins + 1)

        if magic > remaining_matchups:
            results[team] = None  # Eliminated -- can't get enough wins
        elif magic <= 0:
            results[team] = 0  # Clinched
        else:
            results[team] = int(magic)

    return results


# ── Team Strength Profiles ─────────────────────────────────────────


def compute_team_strength_profiles(
    team_weekly_totals: dict[str, dict[str, float]],
    full_schedule: dict[int, list[tuple[str, str]]] | None = None,
    current_week: int = 1,
    momentum_data: dict[str, float] | None = None,
) -> list[dict]:
    """Compute 5-factor team strength profiles using existing power_rankings.py.

    Wires up all 5 factors:
    - roster_quality (40%): Z-score of team's avg category rank
    - category_balance (25%): Fraction of cats beating median
    - schedule_strength (15%): Avg opponent roster_quality for remaining weeks
    - injury_exposure (10%): Placeholder (requires live health data from caller)
    - momentum (10%): From momentum_data if provided
    """
    from src.power_rankings import bootstrap_confidence_interval, compute_power_rankings

    teams = list(team_weekly_totals.keys())
    n_teams = len(teams)
    categories = ALL_CATEGORIES

    # Compute z-scores per category per team
    cat_values: dict[str, list[float]] = {cat: [] for cat in categories}
    for team in teams:
        for cat in categories:
            cat_values[cat].append(team_weekly_totals[team].get(cat, 0.0))

    # Roster quality: normalized z-score of category ranks
    roster_quality: dict[str, float] = {}
    cat_balance: dict[str, float] = {}
    for ti, team in enumerate(teams):
        ranks = []
        above_median = 0
        for cat in categories:
            vals = cat_values[cat]
            team_val = vals[ti]
            if cat in INVERSE_CATS:
                rank = sum(1 for v in vals if v < team_val) + 1
                median_val = sorted(vals)[n_teams // 2]
                if team_val <= median_val:
                    above_median += 1
            else:
                rank = sum(1 for v in vals if v > team_val) + 1
                median_val = sorted(vals)[n_teams // 2]
                if team_val >= median_val:
                    above_median += 1
            ranks.append(rank)

        avg_rank = sum(ranks) / len(ranks)
        # Normalize: best possible = 1.0, worst = 0.0
        rq = max(0.0, min(1.0, 1.0 - (avg_rank - 1) / max(n_teams - 1, 1)))
        roster_quality[team] = rq
        cat_balance[team] = max(0.1, above_median / len(categories))

    # Schedule strength: avg opponent roster_quality for remaining weeks
    sos: dict[str, float] = {}
    remaining_weeks = (
        [w for w in sorted(full_schedule.keys()) if w >= current_week]
        if full_schedule
        else []
    )
    for team in teams:
        opp_qualities: list[float] = []
        for week in remaining_weeks:
            for ta, tb in full_schedule.get(week, []):
                if ta == team and tb in roster_quality:
                    opp_qualities.append(roster_quality[tb])
                elif tb == team and ta in roster_quality:
                    opp_qualities.append(roster_quality[ta])
        sos[team] = float(np.mean(opp_qualities)) if opp_qualities else 0.5

    # Build power ranking input
    team_data = []
    for team in teams:
        entry = {
            "team_name": team,
            "roster_quality": roster_quality[team],
            "category_balance": cat_balance[team],
            "schedule_strength": sos[team],
            "injury_exposure": None,  # caller can provide via health data
            "momentum": momentum_data.get(team) if momentum_data else None,
        }
        team_data.append(entry)

    pr_df = compute_power_rankings(team_data)

    # Add bootstrap CIs
    results = []
    for _, row in pr_df.iterrows():
        p5, p95 = bootstrap_confidence_interval(row["power_rating"])
        results.append({
            "team_name": row["team_name"],
            "power_rating": row["power_rating"],
            "roster_quality": row["roster_quality"],
            "category_balance": row["category_balance"],
            "schedule_strength": row.get("schedule_strength"),
            "injury_exposure": row.get("injury_exposure"),
            "momentum": row.get("momentum"),
            "rank": row["rank"],
            "ci_low": p5,
            "ci_high": p95,
        })
    return results

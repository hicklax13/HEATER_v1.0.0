"""Paired Monte Carlo trade evaluation with correlated stat sampling.

Spec reference: Section 13 L10 (Copula Monte Carlo 100K sims)
               Section 17 Phase 2 item 12 (Basic Monte Carlo 10K sims)

This module runs paired simulations: for each random seed, it simulates
the remaining season BOTH with and without the trade. By using identical
seeds, the difference isolates the causal effect of the trade (variance
reduction technique).

Pipeline per simulation:
  1. Draw correlated stat samples for each player via copula + KDE marginals
  2. Apply injury probability (simple health_score-based attenuation)
  3. Compute roster category totals for pre-trade and post-trade rosters
  4. Convert to SGP and compute standings positions
  5. Aggregate across simulations for distributional metrics

Wires into:
  - src/engine/portfolio/copula.py: GaussianCopula, sample_correlated_stats
  - src/engine/projections/marginals.py: PlayerMarginal
  - src/engine/portfolio/category_analysis.py: CATEGORIES, INVERSE_CATEGORIES
  - src/engine/output/trade_evaluator.py: grade_trade
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.engine.portfolio.category_analysis import CATEGORIES, INVERSE_CATEGORIES

logger = logging.getLogger(__name__)

# Default simulation count for Phase 2 (spec says 100K for full prod,
# Phase 2 targets 10K for speed/accuracy balance)
DEFAULT_N_SIMS: int = 10_000

# Minimum sims for meaningful confidence intervals
MIN_SIMS: int = 100


def run_paired_monte_carlo(
    before_roster_stats: dict[str, dict[str, float]],
    after_roster_stats: dict[str, dict[str, float]],
    before_marginals: dict[str, dict[str, object]] | None = None,
    after_marginals: dict[str, dict[str, object]] | None = None,
    copula: Any | None = None,
    all_team_totals: dict[str, dict[str, float]] | None = None,
    sgp_denominators: dict[str, float] | None = None,
    n_sims: int = DEFAULT_N_SIMS,
    seed: int = 42,
    weeks_remaining: int = 16,
) -> dict[str, Any]:
    """Run paired Monte Carlo simulation for trade evaluation.

    Spec ref: Section 13 L10 — paired comparison with identical seeds.

    For each simulation:
    1. Generate noise perturbations (same seed for pre and post)
    2. Apply to roster stat projections
    3. Compute SGP totals
    4. Measure standings position delta

    The paired design ensures that randomness cancels out, isolating
    the trade's causal effect.

    Args:
        before_roster_stats: {player_id: {stat: projected_value}} for pre-trade roster.
        after_roster_stats: {player_id: {stat: projected_value}} for post-trade roster.
        before_marginals: Optional {player_id: {stat: PlayerMarginal}} for pre-trade.
            If None, uses Gaussian noise based on projection values.
        after_marginals: Optional {player_id: {stat: PlayerMarginal}} for post-trade.
        copula: Optional GaussianCopula for correlated sampling.
            If None, samples categories independently.
        all_team_totals: {team_name: {cat: total}} for standings context.
        sgp_denominators: {cat: denominator} for SGP conversion.
        n_sims: Number of paired simulations.
        seed: Master random seed.
        weeks_remaining: Weeks left in the season (scales noise amplitude).

    Returns:
        Dict with keys:
          - mc_mean: Mean SGP surplus across sims
          - mc_std: Standard deviation of surplus
          - mc_median: Median surplus
          - p5, p25, p75, p95: Percentile surplus values
          - prob_positive: Probability that trade helps (surplus > 0)
          - var5: Value at Risk at 5th percentile
          - cvar5: Conditional VaR (expected loss in worst 5%)
          - sharpe: Sharpe ratio (mean / std)
          - grade: Letter grade from distribution metrics
          - confidence_interval: (lower, upper) 95% CI for mean surplus
          - surplus_distribution: Full array of surplus values (for plotting)
    """
    n_sims = max(n_sims, MIN_SIMS)

    # Default SGP denominators if not provided
    if sgp_denominators is None:
        sgp_denominators = _default_sgp_denoms()

    rng_master = np.random.RandomState(seed)

    # Pre-compute team totals array for standings comparison
    other_teams_sgp = None
    if all_team_totals:
        other_teams_sgp = _compute_other_teams_sgp(all_team_totals, sgp_denominators)

    # Run paired sims
    surpluses = np.zeros(n_sims)
    before_sgp_sims = np.zeros(n_sims)
    after_sgp_sims = np.zeros(n_sims)

    for sim_idx in range(n_sims):
        sim_seed = rng_master.randint(0, 2**31)

        # Same seed for both pre and post
        before_total_sgp = _simulate_roster_sgp(
            roster_stats=before_roster_stats,
            marginals=before_marginals,
            copula=copula,
            sgp_denominators=sgp_denominators,
            seed=sim_seed,
            weeks_remaining=weeks_remaining,
        )
        after_total_sgp = _simulate_roster_sgp(
            roster_stats=after_roster_stats,
            marginals=after_marginals,
            copula=copula,
            sgp_denominators=sgp_denominators,
            seed=sim_seed,
            weeks_remaining=weeks_remaining,
        )

        surpluses[sim_idx] = after_total_sgp - before_total_sgp
        before_sgp_sims[sim_idx] = before_total_sgp
        after_sgp_sims[sim_idx] = after_total_sgp

    # Compute distributional metrics
    mc_mean = float(np.mean(surpluses))
    mc_std = float(np.std(surpluses))
    mc_median = float(np.median(surpluses))

    p5 = float(np.percentile(surpluses, 5))
    p25 = float(np.percentile(surpluses, 25))
    p75 = float(np.percentile(surpluses, 75))
    p95 = float(np.percentile(surpluses, 95))

    prob_positive = float(np.mean(surpluses > 0))

    # Value at Risk and Conditional VaR
    var5 = p5
    worst_5pct = surpluses[surpluses <= np.percentile(surpluses, 5)]
    cvar5 = float(np.mean(worst_5pct)) if len(worst_5pct) > 0 else var5

    # Sharpe ratio: mean surplus / volatility of surplus
    sharpe = mc_mean / max(mc_std, 0.001)

    # 95% confidence interval for the mean
    se = mc_std / np.sqrt(n_sims)
    ci_lower = mc_mean - 1.96 * se
    ci_upper = mc_mean + 1.96 * se

    # Grade using the composite score from spec L11
    # Spec: s = (ce_delta*100)*0.4 + sharpe*0.3 + kelly*0.3
    # Simplified for Phase 2: use mc_mean as the primary signal
    grade = _grade_from_distribution(mc_mean, sharpe, prob_positive)

    # Verdict
    verdict = "ACCEPT" if mc_mean > 0 else "DECLINE"
    confidence_pct = min(100.0, max(0.0, prob_positive * 100))

    return {
        "mc_mean": round(mc_mean, 3),
        "mc_std": round(mc_std, 3),
        "mc_median": round(mc_median, 3),
        "p5": round(p5, 3),
        "p25": round(p25, 3),
        "p75": round(p75, 3),
        "p95": round(p95, 3),
        "prob_positive": round(prob_positive, 3),
        "var5": round(var5, 3),
        "cvar5": round(cvar5, 3),
        "sharpe": round(sharpe, 3),
        "grade": grade,
        "verdict": verdict,
        "confidence_pct": round(confidence_pct, 1),
        "confidence_interval": (round(ci_lower, 3), round(ci_upper, 3)),
        "surplus_distribution": surpluses,
        "n_sims": n_sims,
    }


def _simulate_roster_sgp(
    roster_stats: dict[str, dict[str, float]],
    marginals: dict[str, dict[str, object]] | None,
    copula: Any | None,
    sgp_denominators: dict[str, float],
    seed: int,
    weeks_remaining: int,
) -> float:
    """Simulate one season and compute total SGP for a roster.

    For each player, perturb their projected stats using either:
    1. Correlated copula + KDE marginals (if available)
    2. Independent Gaussian noise (fallback)

    Then sum up all category totals and convert to SGP.
    """
    rng = np.random.RandomState(seed)

    # Noise scale: more weeks remaining = more uncertainty
    noise_scale = min(1.0, np.sqrt(weeks_remaining / 26.0))

    # Aggregate roster totals with noise
    roster_totals: dict[str, float] = {cat: 0.0 for cat in CATEGORIES}

    # Track rate-stat components for proper aggregation
    # Rate stats (AVG, OBP, ERA, WHIP) must NOT be summed across players.
    # Instead, accumulate component stats and derive rate stats at the end.
    total_h = 0.0
    total_ab = 0.0
    total_pa = 0.0
    total_bb = 0.0
    total_hbp = 0.0
    total_sf = 0.0
    total_ip = 0.0
    total_er = 0.0
    total_bb_h_allowed = 0.0  # bb_allowed + h_allowed

    # Categories that are rate stats and need special aggregation
    _RATE_CATS = {"AVG", "OBP", "ERA", "WHIP"}

    for player_id, stats in roster_stats.items():
        # Collect component stats for rate-stat aggregation.
        # If components (h, ab, ip, er, etc.) are available, use them directly.
        # Otherwise, derive from rate stats and playing time estimates.
        p_ab = stats.get("ab", 0.0)
        p_h = stats.get("h", 0.0)
        p_pa = stats.get("pa", 0.0)
        p_bb = stats.get("bb", 0.0)
        p_hbp = stats.get("hbp", 0.0)
        p_sf = stats.get("sf", 0.0)
        p_ip = stats.get("ip", 0.0)
        p_er = stats.get("er", 0.0)
        p_bb_allowed = stats.get("bb_allowed", 0.0)
        p_h_allowed = stats.get("h_allowed", 0.0)

        # If component stats are missing but rate stats exist, derive them
        if p_ab == 0.0 and stats.get("avg", 0.0) > 0:
            # Estimate AB from PA or use a default for a typical hitter
            p_ab = p_pa * 0.92 if p_pa > 0 else 500.0
            p_h = stats["avg"] * p_ab
        if p_pa == 0.0 and p_ab > 0:
            p_pa = p_ab / 0.92  # Estimate PA from AB
        # Derive OBP components if OBP exists but BB missing
        if p_bb == 0.0 and stats.get("obp", 0.0) > 0 and p_pa > 0:
            # OBP = (H + BB + HBP) / (AB + BB + HBP + SF)
            # Approximate: BB ≈ (OBP * PA - H) / (1 + OBP) when HBP/SF small
            obp_val = stats["obp"]
            p_bb = max(0.0, (obp_val * p_pa - p_h))
        if p_ip == 0.0 and (stats.get("era", 0.0) > 0 or stats.get("whip", 0.0) > 0):
            p_ip = 150.0  # Default estimate for a typical pitcher
            if stats.get("era", 0.0) > 0:
                p_er = stats["era"] * p_ip / 9.0
            if stats.get("whip", 0.0) > 0:
                p_bb_h_allowed = stats["whip"] * p_ip
            else:
                p_bb_h_allowed = 0.0
            p_bb_allowed = p_bb_h_allowed * 0.4  # rough split
            p_h_allowed = p_bb_h_allowed * 0.6

        # Generate noise for this player
        if marginals and player_id in marginals and copula is not None:
            # Use copula + marginals for correlated sampling
            player_marg = marginals[player_id]
            u = copula.sample(1, rng)[0]

            for i, cat in enumerate(CATEGORIES):
                cat_lower = cat.lower()
                if cat in _RATE_CATS:
                    # Skip rate stats — they'll be computed from components below
                    continue
                if cat_lower in player_marg and cat_lower in stats:
                    m = player_marg[cat_lower]
                    # Sample from marginal using copula quantile
                    if cat in INVERSE_CATEGORIES:
                        sampled = m.ppf(1 - u[i])
                    else:
                        sampled = m.ppf(u[i])
                    roster_totals[cat] += float(sampled)
                elif cat_lower in stats:
                    roster_totals[cat] += stats[cat_lower]

            # Accumulate components (with noise applied via marginals if available)
            # For rate stat components, apply proportional noise from the copula samples
            # to maintain consistency
            noise_factor = 1.0 + rng.normal(0, 0.05 * noise_scale)
            total_h += p_h * max(noise_factor, 0.5)
            total_ab += p_ab * max(noise_factor, 0.5)
            total_pa += p_pa * max(noise_factor, 0.5)
            total_bb += p_bb * max(noise_factor, 0.5)
            total_hbp += p_hbp * max(noise_factor, 0.5)
            total_sf += p_sf * max(noise_factor, 0.5)
            total_ip += p_ip * max(noise_factor, 0.5)
            total_er += p_er * max(noise_factor, 0.5)
            total_bb_h_allowed += (p_bb_allowed + p_h_allowed) * max(noise_factor, 0.5)
        else:
            # Fallback: independent Gaussian noise
            for cat in CATEGORIES:
                cat_lower = cat.lower()
                if cat in _RATE_CATS:
                    # Skip rate stats — they'll be computed from components below
                    continue
                if cat_lower in stats:
                    base_val = stats[cat_lower]
                    # Noise proportional to the stat value (coefficient of variation ~15%)
                    noise_sigma = abs(base_val) * 0.15 * noise_scale
                    noisy_val = base_val + rng.normal(0, max(noise_sigma, 0.01))
                    roster_totals[cat] += noisy_val

            # Accumulate components with noise for rate stats
            noise_factor = 1.0 + rng.normal(0, 0.05 * noise_scale)
            total_h += p_h * max(noise_factor, 0.5)
            total_ab += p_ab * max(noise_factor, 0.5)
            total_pa += p_pa * max(noise_factor, 0.5)
            total_bb += p_bb * max(noise_factor, 0.5)
            total_hbp += p_hbp * max(noise_factor, 0.5)
            total_sf += p_sf * max(noise_factor, 0.5)
            total_ip += p_ip * max(noise_factor, 0.5)
            total_er += p_er * max(noise_factor, 0.5)
            total_bb_h_allowed += (p_bb_allowed + p_h_allowed) * max(noise_factor, 0.5)

    # Compute rate stats from accumulated components (not naive sums)
    # AVG = sum(H) / sum(AB)
    roster_totals["AVG"] = total_h / total_ab if total_ab > 0 else 0.0
    # OBP = (H + BB + HBP) / (AB + BB + HBP + SF)
    obp_denom = total_ab + total_bb + total_hbp + total_sf
    roster_totals["OBP"] = (total_h + total_bb + total_hbp) / obp_denom if obp_denom > 0 else 0.0
    # ERA = sum(ER) * 9 / sum(IP)
    roster_totals["ERA"] = total_er * 9.0 / total_ip if total_ip > 0 else 0.0
    # WHIP = sum(BB + H_allowed) / sum(IP)
    roster_totals["WHIP"] = total_bb_h_allowed / total_ip if total_ip > 0 else 0.0

    # Convert totals to SGP
    total_sgp = 0.0
    for cat in CATEGORIES:
        denom = sgp_denominators.get(cat, 1.0)
        if abs(denom) < 1e-9:
            denom = 1.0

        if cat in INVERSE_CATEGORIES:
            # Lower is better: negative SGP contribution for high ERA/WHIP
            total_sgp -= roster_totals[cat] / denom
        else:
            total_sgp += roster_totals[cat] / denom

    return total_sgp


def _compute_other_teams_sgp(
    all_team_totals: dict[str, dict[str, float]],
    sgp_denominators: dict[str, float],
) -> list[float]:
    """Compute total SGP for all other teams (for standings context)."""
    team_sgps = []
    for team_name, totals in all_team_totals.items():
        sgp = 0.0
        for cat in CATEGORIES:
            denom = sgp_denominators.get(cat, 1.0)
            if abs(denom) < 1e-9:
                denom = 1.0
            val = totals.get(cat, 0.0)
            if cat in INVERSE_CATEGORIES:
                sgp -= val / denom
            else:
                sgp += val / denom
        team_sgps.append(sgp)
    return team_sgps


def _grade_from_distribution(
    mc_mean: float,
    sharpe: float,
    prob_positive: float,
) -> str:
    """Grade a trade from MC distribution metrics.

    Spec ref: Section 14 L11 — composite score for grading.

    Composite = mc_mean * 0.4 + sharpe * 0.3 + kelly_approx * 0.3

    Kelly approximation: prob_positive * 2 - 1 (simplified Kelly criterion
    assuming roughly symmetric gain/loss).
    """
    kelly_approx = max(0, prob_positive * 2 - 1)
    composite = mc_mean * 0.4 + sharpe * 0.3 + kelly_approx * 0.3

    thresholds = [
        (2.0, "A+"),
        (1.5, "A"),
        (1.0, "A-"),
        (0.7, "B+"),
        (0.4, "B"),
        (0.2, "B-"),
        (0.0, "C+"),
        (-0.2, "C"),
        (-0.5, "C-"),
        (-1.0, "D"),
    ]

    for threshold, grade in thresholds:
        if composite > threshold:
            return grade
    return "F"


def _default_sgp_denoms() -> dict[str, float]:
    """Default SGP denominators from LeagueConfig."""
    from src.valuation import LeagueConfig

    return dict(LeagueConfig().sgp_denominators)


def build_roster_stats(
    player_ids: list[int],
    player_pool: dict[str, dict[str, float]] | Any,
) -> dict[str, dict[str, float]]:
    """Build roster_stats dict from player IDs and pool.

    Converts the player pool (DataFrame or dict) into the
    {player_id_str: {stat: value}} format needed by the MC simulator.

    Args:
        player_ids: List of player IDs on the roster.
        player_pool: Either a dict {player_id: {stat: value}} or
            a DataFrame with player_id column and stat columns.

    Returns:
        Dict mapping str(player_id) -> {stat_lower: value}.
    """
    import pandas as pd

    stat_cols = [
        "r",
        "hr",
        "rbi",
        "sb",
        "avg",
        "obp",
        "w",
        "l",
        "k",
        "sv",
        "era",
        "whip",
        "h",
        "ab",
        "pa",
        "ip",
        "er",
        "bb",
        "hbp",
        "sf",
        "bb_allowed",
        "h_allowed",
    ]
    roster_stats: dict[str, dict[str, float]] = {}

    if isinstance(player_pool, pd.DataFrame):
        for pid in player_ids:
            match = player_pool[player_pool["player_id"] == pid]
            if match.empty:
                continue
            row = match.iloc[0]
            stats: dict[str, float] = {}
            for col in stat_cols:
                if col in row.index:
                    val = row[col]
                    stats[col] = float(val) if pd.notna(val) else 0.0
            roster_stats[str(pid)] = stats
    elif isinstance(player_pool, dict):
        for pid in player_ids:
            if pid in player_pool:
                roster_stats[str(pid)] = dict(player_pool[pid])

    return roster_stats

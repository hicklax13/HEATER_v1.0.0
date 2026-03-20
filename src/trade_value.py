"""Trade Value Chart: Universal player valuation for H2H Categories leagues.

Computes a single 0-100 trade value per player using SGP surplus with
G-Score H2H variance adjustment (arXiv:2307.02188). Provides both
universal (league-neutral) and contextual (team-need-adjusted) values.

Core formula:
    sgp_surplus = total_sgp - replacement_sgp[position]
    g_score = sgp_surplus / sqrt(1 + kappa * tau²/sigma²)
    trade_value = 100 * (g_score / max_g_score) ^ 0.7

Tiers: Elite (90-100), Star (75-89), Solid (55-74), Flex (35-54), Replacement (0-34)
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.valuation import (
    LeagueConfig,
    SGPCalculator,
    compute_replacement_levels,
    compute_vorp,
)

# Minimum weeks_remaining to avoid degenerate scaling
_MIN_WEEKS_REMAINING = 1

# ── Weekly Variance Defaults (tau² per category) ──────────────────────
# Empirical week-to-week standard deviation of a roster's category total
# in a 12-team H2H league. Higher tau = more weekly randomness = lower
# H2H value per G-Score theory. Sources: FanGraphs, arXiv:2307.02188.

WEEKLY_TAU: dict[str, float] = {
    "R": 8.5,
    "HR": 3.2,
    "RBI": 8.0,
    "SB": 2.8,
    "AVG": 0.015,
    "OBP": 0.014,
    "W": 1.6,
    "L": 1.5,
    "SV": 2.4,
    "K": 12.0,
    "ERA": 0.75,
    "WHIP": 0.06,
}

# ── Tier Definitions ──────────────────────────────────────────────────

TIERS = [
    (90, "Elite"),
    (75, "Star"),
    (55, "Solid Starter"),
    (35, "Flex"),
    (0, "Replacement"),
]

TIER_COLORS = {
    "Elite": "#ffd60a",
    "Star": "#457b9d",
    "Solid Starter": "#2d6a4f",
    "Flex": "#666666",
    "Replacement": "#cc3333",
}

LEAGUE_BUDGET = 3120.0  # 12 teams × $260


def assign_tier(trade_value: float) -> str:
    """Map a 0-100 trade value to a named tier."""
    for threshold, name in TIERS:
        if trade_value >= threshold:
            return name
    return "Replacement"


# ── G-Score H2H Adjustment ────────────────────────────────────────────


def compute_g_score_adjustment(
    per_cat_sgp: dict[str, float],
    pool_sigma: dict[str, float],
    config: LeagueConfig | None = None,
) -> float:
    """Apply G-Score variance correction for H2H category leagues.

    In H2H, volatile categories (SB, SV) are less valuable than stable
    ones because weekly randomness means you can win them by luck even
    with lower investment. The G-Score (Rosenof 2023, arXiv:2307.02188)
    corrects for this by penalizing high-tau categories.

    Args:
        per_cat_sgp: Player's SGP per category.
        pool_sigma: Cross-player standard deviation of SGP per category.
        config: League configuration.

    Returns:
        G-Score adjusted total SGP (float).
    """
    if config is None:
        config = LeagueConfig()

    n_teams = config.num_teams
    kappa = (2 * n_teams) / (2 * n_teams - 1)
    total = 0.0

    for cat in config.all_categories:
        cat_sgp = per_cat_sgp.get(cat, 0.0)
        sigma = pool_sigma.get(cat, 1.0)
        tau = WEEKLY_TAU.get(cat, 0.0)

        if abs(sigma) < 1e-9:
            sigma = 1.0

        # Convert tau to SGP units: tau_sgp = tau / sgp_denominator
        denom = config.sgp_denominators.get(cat, 1.0)
        if abs(denom) < 1e-9:
            denom = 1.0
        tau_sgp = tau / denom

        # G-Score denominator adjustment for this category
        # G = sgp / sqrt(sigma² + kappa * tau²) * sigma
        # Simplifies to: weight = sigma / sqrt(sigma² + kappa * tau²)
        denom_adj = math.sqrt(sigma**2 + kappa * tau_sgp**2)
        if abs(denom_adj) < 1e-9:
            denom_adj = 1.0

        weight = sigma / denom_adj
        total += cat_sgp * weight

    return total


# ── Main Trade Value Computation ──────────────────────────────────────


def compute_trade_values(
    player_pool: pd.DataFrame,
    config: LeagueConfig | None = None,
    standings: pd.DataFrame | None = None,
    weeks_remaining: int = 16,
) -> pd.DataFrame:
    """Compute trade values for all players in the pool.

    Returns a DataFrame with columns:
        player_id, name, team, positions, is_hitter,
        total_sgp, vorp, g_score, trade_value (0-100),
        dollar_value, tier, rank, pos_rank

    Args:
        player_pool: Full player pool DataFrame.
        config: League configuration.
        standings: Optional league standings for live SGP denominators.
        weeks_remaining: Weeks left in season (scales time-decayed value).

    Returns:
        DataFrame sorted by trade_value descending.
    """
    if config is None:
        config = LeagueConfig()

    # Clamp weeks_remaining to avoid degenerate scaling
    weeks_remaining = max(weeks_remaining, _MIN_WEEKS_REMAINING)

    if player_pool.empty:
        return pd.DataFrame()

    pool = player_pool.copy()

    # Ensure name column exists
    if "player_name" in pool.columns and "name" not in pool.columns:
        pool["name"] = pool["player_name"]
    elif "name" not in pool.columns:
        pool["name"] = "Unknown"

    # Update SGP denominators from standings if available
    # Use a local copy to avoid mutating the shared config object
    denoms = dict(config.sgp_denominators)
    if standings is not None and not standings.empty:
        try:
            from src.engine.portfolio.valuation import compute_sgp_from_standings

            live_denoms = compute_sgp_from_standings(standings, config)
            denoms = {**denoms, **live_denoms}
        except ImportError:
            pass

    # Build a local config copy with merged denominators
    local_config = LeagueConfig()
    local_config.__dict__.update(config.__dict__)
    local_config.sgp_denominators = denoms

    sgp_calc = SGPCalculator(local_config)
    replacement_levels = compute_replacement_levels(pool, local_config, sgp_calc)

    # ── Step 1: Compute per-player SGP and VORP ───────────────────────

    per_cat_sgp_list = []
    total_sgp_list = []
    vorp_list = []

    for _, player in pool.iterrows():
        cat_sgp = sgp_calc.player_sgp(player)
        per_cat_sgp_list.append(cat_sgp)
        total_sgp_list.append(sum(cat_sgp.values()))
        vorp_list.append(compute_vorp(player, sgp_calc, replacement_levels))

    pool["total_sgp"] = total_sgp_list
    pool["vorp"] = vorp_list

    # ── Step 2: Compute pool-level sigma per category ─────────────────

    pool_sigma: dict[str, float] = {}
    for cat in local_config.all_categories:
        cat_sgp_vals = [d.get(cat, 0.0) for d in per_cat_sgp_list]
        std = np.std(cat_sgp_vals) if len(cat_sgp_vals) > 1 else 1.0
        pool_sigma[cat] = max(std, 1e-6)

    # ── Step 3: Apply G-Score adjustment ──────────────────────────────

    g_scores = []
    for cat_sgp in per_cat_sgp_list:
        g = compute_g_score_adjustment(cat_sgp, pool_sigma, local_config)
        g_scores.append(g)

    pool["g_score"] = g_scores

    # ── Step 4: Compute SGP surplus (G-Score adjusted) ────────────────
    # Replacement levels are in raw SGP units; g_scores are variance-adjusted.
    # Compute the pool-wide ratio to convert replacement levels to G-Score scale.
    raw_sum = sum(total_sgp_list)
    gscore_sum = sum(g_scores)
    if abs(raw_sum) > 1e-9:
        gscore_ratio = gscore_sum / raw_sum
    else:
        gscore_ratio = 1.0

    surplus_list = []
    for idx, row in pool.iterrows():
        positions = [p.strip() for p in str(row.get("positions", "Util")).split(",")]
        valid = [p for p in positions if p in replacement_levels]
        if valid:
            best_repl = max(replacement_levels.get(p, 0) for p in valid)
        else:
            best_repl = 0
        # Scale replacement level to G-Score units for apples-to-apples comparison
        surplus = row["g_score"] - best_repl * gscore_ratio
        surplus_list.append(surplus)

    pool["sgp_surplus"] = surplus_list

    # ── Step 5: Scale to 0-100 trade value ────────────────────────────

    max_surplus = pool["sgp_surplus"].max()
    if max_surplus <= 0:
        max_surplus = 1.0

    # Non-linear scaling: exponent 0.7 compresses top, expands mid-range
    pool["trade_value"] = pool["sgp_surplus"].apply(lambda s: round(100.0 * (max(s, 0) / max_surplus) ** 0.7, 1))

    # ── Step 6: Time decay adjustment ─────────────────────────────────
    # Scale values by remaining season fraction (full season = 26 weeks)
    # Applied BEFORE tier assignment so tiers reflect actual trade value.
    total_weeks = 26.0
    time_factor = min(weeks_remaining / total_weeks, 1.0)
    pool["trade_value"] = (pool["trade_value"] * time_factor).round(1)

    # ── Step 7: Compute dollar values ─────────────────────────────────

    positive_surplus = pool.loc[pool["sgp_surplus"] > 0, "sgp_surplus"]
    total_positive = positive_surplus.sum()
    if total_positive > 0:
        pool["dollar_value"] = pool["sgp_surplus"].apply(
            lambda s: round(max(s, 0) / total_positive * LEAGUE_BUDGET + 1.0, 1) if s > 0 else 1.0
        )
    else:
        pool["dollar_value"] = 1.0
    pool["dollar_value"] = (pool["dollar_value"] * time_factor).round(1)

    # ── Step 8: Assign tiers and ranks ────────────────────────────────

    pool["tier"] = pool["trade_value"].apply(assign_tier)

    pool = pool.sort_values("trade_value", ascending=False).reset_index(drop=True)
    pool["rank"] = range(1, len(pool) + 1)

    # Position-specific ranks
    pos_ranks = []
    pos_counters: dict[str, int] = {}
    for _, row in pool.iterrows():
        positions = [p.strip() for p in str(row.get("positions", "Util")).split(",")]
        primary = positions[0] if positions else "Util"
        pos_counters[primary] = pos_counters.get(primary, 0) + 1
        pos_ranks.append(pos_counters[primary])
    pool["pos_rank"] = pos_ranks

    # Select output columns
    output_cols = [
        "player_id",
        "name",
        "team",
        "positions",
        "is_hitter",
        "total_sgp",
        "vorp",
        "g_score",
        "sgp_surplus",
        "trade_value",
        "dollar_value",
        "tier",
        "rank",
        "pos_rank",
    ]
    available = [c for c in output_cols if c in pool.columns]
    return pool[available]


def compute_contextual_values(
    trade_values: pd.DataFrame,
    user_totals: dict[str, float],
    all_team_totals: dict[str, dict[str, float]],
    user_team_name: str,
    config: LeagueConfig | None = None,
    player_pool: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add contextual overlay: adjust trade values by YOUR team's category needs.

    Players who fill your weak categories get boosted; players redundant to
    your strengths get discounted.

    Args:
        trade_values: Output from compute_trade_values().
        user_totals: Your team's current category totals.
        all_team_totals: All teams' totals (for marginal SGP computation).
        user_team_name: Your team's name/id.
        config: League configuration.
        player_pool: Original player pool with stat columns (r, hr, rbi, etc.).
            When provided, stat columns are merged onto trade_values so that
            per-category SGP can be computed correctly.  Without this the
            trade_values DataFrame lacks stat columns and every SGP is zero.

    Returns:
        DataFrame with added 'contextual_value' and 'contextual_tier' columns.
    """
    if config is None:
        config = LeagueConfig()

    from src.engine.portfolio.category_analysis import (
        category_gap_analysis,
        compute_category_weights_from_analysis,
    )

    result = trade_values.copy()

    if result.empty:
        result["contextual_value"] = pd.Series(dtype=float)
        result["contextual_tier"] = pd.Series(dtype=str)
        return result

    if not user_totals or not all_team_totals:
        result["contextual_value"] = result["trade_value"]
        result["contextual_tier"] = result["tier"]
        return result

    # Merge stat columns from player_pool so player_sgp() can read them
    if player_pool is not None and "player_id" in result.columns:
        stat_cols = list(config.STAT_MAP.values())
        # Also include component columns needed for rate-stat SGP
        extra_cols = ["ab", "h", "bb", "hbp", "sf", "ip", "er", "bb_allowed", "h_allowed", "pa"]
        merge_cols = ["player_id"] + [
            c for c in stat_cols + extra_cols if c in player_pool.columns and c not in result.columns
        ]
        if len(merge_cols) > 1:
            result = result.merge(
                player_pool[merge_cols].drop_duplicates(subset="player_id"),
                on="player_id",
                how="left",
            )

    # Get category weights from gap analysis
    analysis = category_gap_analysis(user_totals, all_team_totals, user_team_name)
    weights = compute_category_weights_from_analysis(analysis)

    sgp_calc = SGPCalculator(config)
    contextual_values = []

    for _, row in result.iterrows():
        # Compute weighted SGP using category need weights
        cat_sgp = sgp_calc.player_sgp(row)
        weighted = sum(cat_sgp.get(c, 0) * weights.get(c, 1.0) for c in config.all_categories)
        unweighted = sum(cat_sgp.values())

        if abs(unweighted) > 1e-6:
            multiplier = weighted / unweighted
        else:
            multiplier = 1.0

        # Blend: 60% universal, 40% contextual
        ctx_val = row["trade_value"] * (0.6 + 0.4 * multiplier)
        contextual_values.append(round(max(ctx_val, 0), 1))

    result["contextual_value"] = contextual_values
    result["contextual_tier"] = result["contextual_value"].apply(assign_tier)

    return result


def filter_by_position(trade_values: pd.DataFrame, position: str) -> pd.DataFrame:
    """Filter trade values to players eligible at a specific position."""
    if position == "All":
        return trade_values

    if trade_values.empty or "positions" not in trade_values.columns:
        return trade_values

    mask = trade_values["positions"].apply(lambda x: position in [p.strip() for p in str(x or "").split(",")])
    filtered = trade_values[mask].copy()
    filtered["pos_rank"] = range(1, len(filtered) + 1)
    return filtered

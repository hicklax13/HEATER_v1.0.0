"""Z-score + SGP valuation engine for trade analysis.

Spec reference: Section 17 Phase 1 items 3 (Z-score + SGP valuation)
               Section 10 L7A (marginal category elasticity)

Wires into existing:
  - src/valuation.py: LeagueConfig, SGPCalculator, compute_replacement_levels, compute_vorp
  - src/database.py: load_player_pool, load_league_standings, get_connection
  - src/in_season.py: _roster_category_totals

Computes:
  - Per-player z-scores across all 12 fantasy categories
  - Category-level SGP from league standings
  - VORP relative to the free agent pool
  - Composite valuation score combining z-score, SGP, and positional scarcity
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.database import load_league_standings, load_player_pool
from src.valuation import (
    LeagueConfig,
    SGPCalculator,
    compute_replacement_levels,
    compute_vorp,
)

if TYPE_CHECKING:
    pass

# Stat column mapping: category name -> DataFrame column name
STAT_MAP: dict[str, str] = {
    "R": "r",
    "HR": "hr",
    "RBI": "rbi",
    "SB": "sb",
    "AVG": "avg",
    "OBP": "obp",
    "W": "w",
    "L": "l",
    "SV": "sv",
    "K": "k",
    "ERA": "era",
    "WHIP": "whip",
}

CATEGORIES: list[str] = ["R", "HR", "RBI", "SB", "AVG", "OBP", "W", "L", "SV", "K", "ERA", "WHIP"]
INVERSE_CATEGORIES: set[str] = {"L", "ERA", "WHIP"}


def compute_player_zscores(
    player_pool: pd.DataFrame,
    config: LeagueConfig | None = None,
) -> pd.DataFrame:
    """Compute per-player z-scores across all 12 fantasy categories.

    Spec ref: Section 17 item 3 — Z-score normalization across draftable pool.

    Z-scores are computed within the correct peer group:
      - Hitting categories: z-scored against hitters only
      - Pitching categories: z-scored against pitchers only
    For inverse stats (ERA, WHIP), the sign is flipped so higher z = better.

    Args:
        player_pool: Full player pool DataFrame with stat columns.
        config: League configuration. Defaults to standard 12-team H2H 12-cat.

    Returns:
        DataFrame with original columns plus z_{cat} columns for each category
        and a 'z_composite' column (sum of all z-scores).
    """
    if config is None:
        config = LeagueConfig()

    pool = player_pool.copy()

    z_cols: list[str] = []
    for cat in config.all_categories:
        col = STAT_MAP.get(cat, cat.lower())
        z_col = f"z_{cat}"
        z_cols.append(z_col)

        if col not in pool.columns:
            pool[z_col] = 0.0
            continue

        # Filter to correct peer group for z-score computation
        if cat in config.hitting_categories:
            peer_mask = pool["is_hitter"] == 1
        else:
            peer_mask = pool["is_hitter"] == 0
            # Exclude 0-IP pitchers for rate stats to avoid inflated z-scores
            if cat in config.inverse_stats and "ip" in pool.columns:
                peer_mask = peer_mask & (pool["ip"].fillna(0) > 0)

        peer_vals = pool.loc[peer_mask, col].dropna()
        mean = peer_vals.mean() if len(peer_vals) > 0 else 0.0
        std = peer_vals.std() if len(peer_vals) > 1 else 1.0
        if std == 0 or pd.isna(std):
            std = 1.0

        raw_z = (pool[col].fillna(0) - mean) / std

        # Flip sign for inverse stats so higher z = better
        if cat in config.inverse_stats:
            pool[z_col] = -raw_z
        else:
            pool[z_col] = raw_z

        # Zero out z-scores for players outside the peer group
        # (e.g., don't give pitchers hitting z-scores)
        if cat in config.hitting_categories:
            pool.loc[pool["is_hitter"] == 0, z_col] = 0.0
        else:
            pool.loc[pool["is_hitter"] == 1, z_col] = 0.0

    pool["z_composite"] = pool[z_cols].sum(axis=1)
    return pool


def compute_sgp_from_standings(
    standings: pd.DataFrame,
    config: LeagueConfig | None = None,
) -> dict[str, float]:
    """Compute SGP denominators from actual league standings data.

    Spec ref: Section 10 L7A — standings-based marginal SGP.

    Uses the gap between adjacent teams in standings to determine how many
    raw stat points equal one standings point in each category.

    Args:
        standings: League standings DataFrame with columns:
                   team_name, category, total, rank.
        config: League configuration.

    Returns:
        Dict mapping category name to SGP denominator (raw stat per standings point).
    """
    if config is None:
        config = LeagueConfig()

    sgp_denoms: dict[str, float] = {}

    # Handle truly empty DataFrame (no columns at all)
    if standings.empty or "category" not in standings.columns:
        return dict(config.sgp_denominators)

    for cat in config.all_categories:
        cat_rows = standings[standings["category"] == cat]
        if cat_rows.empty or len(cat_rows) < 3:
            # Fall back to default denominators
            sgp_denoms[cat] = config.sgp_denominators.get(cat, 1.0)
            continue

        totals = cat_rows["total"].sort_values(ascending=True).values
        # Compute gaps between adjacent teams
        gaps = np.diff(totals)
        # Remove zero gaps and compute median gap
        nonzero_gaps = gaps[gaps > 0]
        if len(nonzero_gaps) == 0:
            sgp_denoms[cat] = config.sgp_denominators.get(cat, 1.0)
        else:
            sgp_denoms[cat] = float(np.median(nonzero_gaps))

    return sgp_denoms


def compute_player_vorp(
    player_pool: pd.DataFrame,
    config: LeagueConfig | None = None,
) -> pd.DataFrame:
    """Add VORP (Value Over Replacement Player) to each player.

    Spec ref: Section 17 item 3 — VORP using FA pool.

    Wires into existing compute_replacement_levels() and compute_vorp()
    from src/valuation.py.

    Args:
        player_pool: Full player pool DataFrame.
        config: League configuration.

    Returns:
        DataFrame with added 'vorp' column.
    """
    if config is None:
        config = LeagueConfig()

    sgp_calc = SGPCalculator(config)
    replacement_levels = compute_replacement_levels(player_pool, config, sgp_calc)

    pool = player_pool.copy()
    pool["total_sgp"] = pool.apply(sgp_calc.total_sgp, axis=1)
    pool["vorp"] = pool.apply(lambda p: compute_vorp(p, sgp_calc, replacement_levels), axis=1)
    return pool


def build_valuation_context(
    config: LeagueConfig | None = None,
) -> dict:
    """Build the full valuation context from database state.

    Loads player pool, standings, computes z-scores, SGP denominators, and VORP.
    Returns a dict with all computed data for use by other engine modules.

    Args:
        config: League configuration.

    Returns:
        Dict with keys: player_pool, standings, sgp_denominators,
        sgp_calc, replacement_levels.
    """
    if config is None:
        config = LeagueConfig()

    pool = load_player_pool()
    if pool.empty:
        return {
            "player_pool": pool,
            "standings": pd.DataFrame(),
            "sgp_denominators": config.sgp_denominators,
            "sgp_calc": SGPCalculator(config),
            "replacement_levels": {},
        }

    pool = pool.rename(columns={"name": "player_name"})

    # Compute z-scores
    pool = compute_player_zscores(pool, config)

    # Load standings and compute live SGP denominators
    standings = load_league_standings()
    denoms = dict(config.sgp_denominators)
    if not standings.empty:
        live_denoms = compute_sgp_from_standings(standings, config)
        # Merge into local copy — do NOT mutate shared config
        denoms = {**denoms, **live_denoms}

    # Build a local config copy with the merged denominators for SGP calc
    local_config = LeagueConfig()
    local_config.__dict__.update(config.__dict__)
    local_config.sgp_denominators = denoms

    sgp_calc = SGPCalculator(local_config)
    replacement_levels = compute_replacement_levels(pool, local_config, sgp_calc)

    # Add VORP
    pool["total_sgp"] = pool.apply(sgp_calc.total_sgp, axis=1)
    pool["vorp"] = pool.apply(lambda p: compute_vorp(p, sgp_calc, replacement_levels), axis=1)

    return {
        "player_pool": pool,
        "standings": standings,
        "sgp_denominators": denoms,
        "sgp_calc": sgp_calc,
        "replacement_levels": replacement_levels,
    }

"""Stochastic lineup optimization via scenario generation.

Provides scenario-based tools for robust lineup optimization:
  1. Scenario generation — sample correlated stat lines for each player
  2. Mean-variance adjustments — risk-penalized player values for LP
  3. CVaR constraints — tail-risk protection via Rockafellar-Uryasev
  4. Post-optimization analytics — per-scenario lineup value computation

These functions produce data structures consumed by the PuLP-based
LineupOptimizer in src/lineup_optimizer.py. This module does NOT
depend on PuLP itself — it builds arrays and dicts that the LP
solver integrates as objective coefficients and constraints.

Uses GaussianCopula from src/engine/portfolio/copula.py when available
for correlated sampling; falls back to independent Normal draws.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)

# ── Category definitions ─────────────────────────────────────────────

ALL_CATS: list[str] = ["r", "hr", "rbi", "sb", "avg", "w", "sv", "k", "era", "whip"]
INVERSE_CATS: set[str] = {"era", "whip"}

# Category index lookup for fast access
_CAT_IDX: dict[str, int] = {cat: i for i, cat in enumerate(ALL_CATS)}

# ── Default coefficients of variation ────────────────────────────────
# Used when per-player projection variance is unavailable.
# Counting stats use CV (std / mean), rate stats use absolute std.

DEFAULT_CV: dict[str, float] = {
    "r": 0.15,
    "hr": 0.20,
    "rbi": 0.18,
    "sb": 0.30,
    "avg": 0.07,
    "w": 0.25,
    "sv": 0.35,
    "k": 0.15,
    "era": 0.15,
    "whip": 0.10,
}

# Rate stats use absolute standard deviation instead of CV
_RATE_STATS: set[str] = {"avg", "era", "whip"}
_RATE_STD: dict[str, float] = {
    "avg": 0.020,
    "era": 0.50,
    "whip": 0.05,
}

# ── Simplified correlation structure ─────────────────────────────────
# Pairwise correlations for the 10 fantasy categories.
# Symmetric: (a, b) and (b, a) map to the same value.

DEFAULT_CORRELATIONS: dict[tuple[str, str], float] = {
    ("hr", "rbi"): 0.85,
    ("hr", "r"): 0.70,
    ("r", "rbi"): 0.75,
    ("sb", "hr"): -0.15,
    ("sb", "avg"): 0.20,
    ("era", "whip"): 0.90,
    ("k", "era"): -0.65,
    ("k", "whip"): -0.50,
    ("w", "k"): 0.40,
    ("w", "era"): -0.45,
}

# ── Copula import (optional) ─────────────────────────────────────────

try:
    from src.engine.portfolio.copula import GaussianCopula

    _COPULA_AVAILABLE = True
except ImportError:
    _COPULA_AVAILABLE = False


# ── Internal helpers ─────────────────────────────────────────────────


def _build_correlation_matrix() -> np.ndarray:
    """Build 10x10 correlation matrix from DEFAULT_CORRELATIONS."""
    n = len(ALL_CATS)
    corr = np.eye(n, dtype=float)

    for (cat_a, cat_b), rho in DEFAULT_CORRELATIONS.items():
        if cat_a not in _CAT_IDX or cat_b not in _CAT_IDX:
            continue
        i, j = _CAT_IDX[cat_a], _CAT_IDX[cat_b]
        corr[i, j] = rho
        corr[j, i] = rho

    return corr


def _player_stat_std(projected_value: float, cat: str) -> float:
    """Compute standard deviation for a single stat projection.

    For rate stats (avg, era, whip), uses fixed absolute std.
    For counting stats, uses coefficient of variation * projected value.

    Args:
        projected_value: The player's projected stat value.
        cat: Category name (lowercase).

    Returns:
        Standard deviation estimate (always >= 1e-6).
    """
    if cat in _RATE_STATS:
        return max(_RATE_STD.get(cat, 0.01), 1e-6)

    cv = DEFAULT_CV.get(cat, 0.15)
    return max(abs(projected_value) * cv, 1e-6)


def _extract_player_stats(roster: dict | list | np.ndarray) -> np.ndarray:
    """Extract (n_players, 10) stat matrix from a roster.

    Accepts:
      - list of dicts with category keys
      - numpy array of shape (n_players, 10)
      - pandas DataFrame with category columns

    Returns:
        Array of shape (n_players, 10).
    """
    if isinstance(roster, np.ndarray):
        if roster.ndim == 2 and roster.shape[1] == len(ALL_CATS):
            return roster.astype(float)
        msg = f"Expected (n_players, {len(ALL_CATS)}), got {roster.shape}"
        raise ValueError(msg)

    # Try pandas DataFrame
    try:
        import pandas as pd

        if isinstance(roster, pd.DataFrame):
            stats = np.zeros((len(roster), len(ALL_CATS)), dtype=float)
            for j, cat in enumerate(ALL_CATS):
                if cat in roster.columns:
                    stats[:, j] = pd.to_numeric(roster[cat], errors="coerce").fillna(0).values
            return stats
    except ImportError:
        pass

    # List of dicts
    if isinstance(roster, list):
        n = len(roster)
        stats = np.zeros((n, len(ALL_CATS)), dtype=float)
        for i, player in enumerate(roster):
            for j, cat in enumerate(ALL_CATS):
                val = player.get(cat, 0.0) if isinstance(player, dict) else 0.0
                stats[i, j] = float(val) if val is not None else 0.0
        return stats

    msg = f"Unsupported roster type: {type(roster)}"
    raise TypeError(msg)


# ── Public API ───────────────────────────────────────────────────────


def generate_stat_scenarios(
    roster: list[dict] | np.ndarray,
    n_scenarios: int = 200,
    seed: int = 42,
) -> np.ndarray:
    """Generate correlated stat scenarios for a roster.

    For each scenario, samples a correlated stat line for every player
    using either GaussianCopula (if available) or independent Normal draws
    with a Cholesky-based correlation approach.

    Args:
        roster: Player projections. List of dicts with stat keys, or
            numpy array of shape (n_players, 10).
        n_scenarios: Number of scenarios to generate.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (n_scenarios, n_players, 10) with sampled stats.
        Categories in order: r, hr, rbi, sb, avg, w, sv, k, era, whip.
    """
    stats = _extract_player_stats(roster)
    n_players = stats.shape[0]
    n_cats = len(ALL_CATS)

    # Handle empty roster
    if n_players == 0:
        return np.zeros((n_scenarios, 0, n_cats), dtype=float)

    rng = np.random.RandomState(seed)
    scenarios = np.zeros((n_scenarios, n_players, n_cats), dtype=float)

    # Build per-player std matrix
    std_matrix = np.zeros_like(stats)
    for i in range(n_players):
        for j, cat in enumerate(ALL_CATS):
            std_matrix[i, j] = _player_stat_std(stats[i, j], cat)

    # Try copula-based correlated sampling
    if _COPULA_AVAILABLE:
        try:
            copula = GaussianCopula()  # uses 10x10 DEFAULT_CORRELATION
            for i in range(n_players):
                # Draw correlated uniforms: (n_scenarios, 10)
                u = copula.sample(n_scenarios, rng=rng)
                # Convert to stat values via inverse CDF
                for j, cat in enumerate(ALL_CATS):
                    mu = stats[i, j]
                    sigma = std_matrix[i, j]
                    scenarios[:, i, j] = norm.ppf(u[:, j], loc=mu, scale=sigma)
            logger.debug(
                "Generated %d scenarios with GaussianCopula for %d players",
                n_scenarios,
                n_players,
            )
            return scenarios
        except Exception:
            logger.warning("Copula sampling failed, falling back to Cholesky")

    # Fallback: Cholesky-based correlated Normal sampling
    corr = _build_correlation_matrix()

    # Ensure positive definiteness
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 1e-10)
    corr_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(corr_pd))
    corr_pd = corr_pd / np.outer(d, d)

    cholesky = np.linalg.cholesky(corr_pd)

    for i in range(n_players):
        z = rng.standard_normal((n_scenarios, n_cats))
        correlated = z @ cholesky.T
        for j in range(n_cats):
            scenarios[:, i, j] = stats[i, j] + correlated[:, j] * std_matrix[i, j]

    logger.debug(
        "Generated %d scenarios with Cholesky fallback for %d players",
        n_scenarios,
        n_players,
    )
    return scenarios


def mean_variance_adjustments(
    roster: list[dict] | np.ndarray,
    lambda_risk: float = 0.15,
) -> dict[int, float]:
    """Compute mean-variance risk adjustments for each player.

    For binary decision variables x_i in {0, 1}, x_i^2 = x_i, so the
    quadratic penalty lambda * sigma_i^2 * x_i remains linear in x_i.
    This lets us integrate risk aversion into the PuLP LP objective
    without needing quadratic programming.

    The adjustment is: -lambda * total_variance_i
    Negative values indicate a risk penalty; near-zero means low risk.

    Args:
        roster: Player projections. List of dicts or (n_players, 10) array.
        lambda_risk: Risk aversion parameter. 0 = risk-neutral, higher =
            more conservative. Default 0.15.

    Returns:
        Dict mapping player index (0-based) to risk adjustment (float).
        Values are typically negative (penalty) or zero.
    """
    stats = _extract_player_stats(roster)
    n_players = stats.shape[0]
    adjustments: dict[int, float] = {}

    if lambda_risk == 0.0:
        for i in range(n_players):
            adjustments[i] = 0.0
        return adjustments

    for i in range(n_players):
        total_var = estimate_player_variance(
            {cat: stats[i, j] for j, cat in enumerate(ALL_CATS)},
            is_hitter=_is_hitter_from_stats(stats[i]),
        )
        adjustments[i] = -lambda_risk * total_var

    return adjustments


def build_cvar_constraints(
    scenarios: np.ndarray,
    player_indices: list[int],
    category_weights: dict[str, float],
    alpha: float = 0.05,
) -> dict:
    """Build CVaR auxiliary data for PuLP integration.

    Uses the Rockafellar-Uryasev (2000) linearization of CVaR:

        maximize  eta - (1 / (N * alpha)) * sum_s z_s
        s.t.      z_s >= 0                     for all s
                  z_s >= eta - value_s          for all s

    where value_s is the weighted stat total in scenario s.

    This function does NOT create PuLP variables — it returns the data
    structures the LP solver needs to build those constraints.

    Args:
        scenarios: Array of shape (n_scenarios, n_players, 10).
        player_indices: Indices of players eligible for lineup slots.
        category_weights: Dict mapping category name to weight.
            Inverse categories (era, whip) should have positive weights;
            sign flipping is handled internally.
        alpha: CVaR tail probability (default 0.05 = worst 5%).

    Returns:
        Dict with keys:
            n_scenarios: int
            scenario_values: list of dicts {player_idx: scenario_value}
            alpha: float
            eta_name: str (suggested PuLP variable name)
    """
    n_scenarios = scenarios.shape[0]
    n_cats = len(ALL_CATS)

    # Build weight vector (sign-flip inverse cats)
    weight_vec = np.zeros(n_cats, dtype=float)
    for j, cat in enumerate(ALL_CATS):
        w = category_weights.get(cat, 0.0)
        if cat in INVERSE_CATS:
            weight_vec[j] = -w  # lower ERA/WHIP is better
        else:
            weight_vec[j] = w

    # Compute per-player, per-scenario weighted values
    scenario_values: list[dict[int, float]] = []
    for s in range(n_scenarios):
        sv: dict[int, float] = {}
        for idx in player_indices:
            if idx < scenarios.shape[1]:
                player_stats = scenarios[s, idx, :]
                sv[idx] = float(np.dot(player_stats, weight_vec))
        scenario_values.append(sv)

    return {
        "n_scenarios": n_scenarios,
        "scenario_values": scenario_values,
        "alpha": alpha,
        "eta_name": "cvar_eta",
    }


def estimate_player_variance(
    player_stats: dict[str, float],
    is_hitter: bool = True,
) -> float:
    """Estimate total projection variance for a player.

    Sums the variance contribution of each relevant category.
    Hitters are scored on hitting cats; pitchers on pitching cats.
    Players with stats in both are scored on all categories.

    Args:
        player_stats: Dict mapping category name to projected value.
        is_hitter: Whether the player is primarily a hitter.

    Returns:
        Total variance estimate (sum of per-category variances).
    """
    hitting_cats = {"r", "hr", "rbi", "sb", "avg"}
    pitching_cats = {"w", "sv", "k", "era", "whip"}
    relevant = hitting_cats if is_hitter else pitching_cats

    total_var = 0.0
    for cat in relevant:
        val = player_stats.get(cat, 0.0)
        if val is None:
            val = 0.0
        std = _player_stat_std(float(val), cat)
        total_var += std * std

    return total_var


def compute_scenario_lineup_values(
    scenarios: np.ndarray,
    assignments: dict[int, float],
    category_weights: dict[str, float],
    scale_factors: dict[str, float] | None = None,
) -> np.ndarray:
    """Compute per-scenario lineup values for a given assignment.

    Evaluates the total weighted stat output of a lineup across all
    generated scenarios. Used post-optimization to report VaR and
    CVaR metrics.

    Args:
        scenarios: Array of shape (n_scenarios, n_players, 10).
        assignments: Dict mapping player index to assignment weight
            (typically 1.0 for starters, 0.0 for bench).
        category_weights: Dict mapping category name to weight.
        scale_factors: Optional per-category scaling (e.g., SGP denominators).

    Returns:
        Array of shape (n_scenarios,) with lineup value per scenario.
    """
    n_scenarios = scenarios.shape[0]
    n_cats = len(ALL_CATS)

    if n_scenarios == 0 or scenarios.shape[1] == 0:
        return np.zeros(n_scenarios, dtype=float)

    # Build weight vector
    weight_vec = np.zeros(n_cats, dtype=float)
    for j, cat in enumerate(ALL_CATS):
        w = category_weights.get(cat, 0.0)
        sf = 1.0
        if scale_factors:
            sf = scale_factors.get(cat, 1.0)
            if sf == 0.0:
                sf = 1.0
        if cat in INVERSE_CATS:
            weight_vec[j] = -w / sf
        else:
            weight_vec[j] = w / sf

    values = np.zeros(n_scenarios, dtype=float)
    for s in range(n_scenarios):
        total = 0.0
        for player_idx, assignment_weight in assignments.items():
            if player_idx < scenarios.shape[1] and assignment_weight > 0:
                player_stats = scenarios[s, player_idx, :]
                total += assignment_weight * np.dot(player_stats, weight_vec)
        values[s] = total

    return values


# ── Internal helpers (continued) ─────────────────────────────────────


def _is_hitter_from_stats(stat_row: np.ndarray) -> bool:
    """Infer whether a player is a hitter from their stat line.

    Heuristic: if IP-related stats (w, sv, k, era, whip) are all zero
    or near-zero, the player is a hitter.
    """
    # Pitching cat indices: w=5, sv=6, k=7, era=8, whip=9
    pitching_sum = abs(stat_row[5]) + abs(stat_row[6]) + abs(stat_row[7])
    return pitching_sum < 0.01

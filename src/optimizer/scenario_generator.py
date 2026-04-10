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
import pandas as pd
from scipy.stats import norm

from src.validation.constant_optimizer import load_constants

logger = logging.getLogger(__name__)

_CONSTANTS = load_constants()

# ── Category definitions ─────────────────────────────────────────────

ALL_CATS: list[str] = ["r", "hr", "rbi", "sb", "avg", "obp", "w", "l", "sv", "k", "era", "whip"]
INVERSE_CATS: set[str] = {"l", "era", "whip"}

# Category index lookup for fast access
_CAT_IDX: dict[str, int] = {cat: i for i, cat in enumerate(ALL_CATS)}

# ── Default coefficients of variation ────────────────────────────────
# Used when per-player projection variance is unavailable.
# Counting stats use CV (std / mean), rate stats use absolute std.
#
# Empirical: 2022-2024 MLB data (N=624 hitter-seasons, 393 pitcher-seasons)
# Cross-sectional CVs scaled to ~50% for projection uncertainty
# (cross-sectional includes talent variation; projection error is smaller).

DEFAULT_CV: dict[str, float] = {
    "r": 0.18,  # Empirical cross-sectional: 0.29; projection CV ~0.18
    "hr": 0.25,  # Empirical cross-sectional: 0.51; projection CV ~0.25
    "rbi": 0.20,  # Empirical cross-sectional: 0.32; projection CV ~0.20
    "sb": 0.45,  # Empirical cross-sectional: 1.10; projection CV ~0.45
    "avg": 0.07,  # Rate stat — see _RATE_STD
    "obp": 0.06,  # Rate stat — see _RATE_STD
    "w": 0.30,  # Empirical cross-sectional: 0.41; projection CV ~0.30
    "l": 0.28,  # Empirical cross-sectional: 0.35; projection CV ~0.28
    "sv": 0.50,  # Empirical cross-sectional: 5.78; high role-dependence, capped at 0.50
    "k": 0.20,  # Empirical cross-sectional: 0.29; projection CV ~0.20
    "era": 0.15,  # Rate stat — see _RATE_STD
    "whip": 0.10,  # Rate stat — see _RATE_STD
}

# Rate stats use absolute standard deviation instead of CV
# Empirical: 2022-2024 MLB data (N=624 hitter-seasons, 393 pitcher-seasons)
_RATE_STATS: set[str] = {"avg", "obp", "era", "whip"}
_RATE_STD: dict[str, float] = {
    "avg": 0.022,  # Empirical cross-sectional std: 0.028; projection ~0.022
    "obp": 0.025,  # Empirical cross-sectional std: 0.031; projection ~0.025
    "era": 0.70,  # Empirical cross-sectional std: 0.90; projection ~0.70
    "whip": 0.10,  # Empirical cross-sectional std: 0.17; projection ~0.10
}

# ── Simplified correlation structure ─────────────────────────────────
# Pairwise correlations for the 12 fantasy categories.
# Symmetric: (a, b) and (b, a) map to the same value.
# Empirical: 2022-2024 MLB data (N=624 hitter-seasons, 393 pitcher-seasons)

DEFAULT_CORRELATIONS: dict[tuple[str, str], float] = {
    ("hr", "rbi"): _CONSTANTS.get("copula_hr_rbi_corr"),  # Empirical: 0.84
    ("hr", "r"): 0.66,  # Empirical: 0.66 (was 0.70)
    ("r", "rbi"): 0.71,  # Empirical: 0.71 (was 0.75)
    ("sb", "hr"): 0.00,  # Empirical: 0.01 (was -0.15); speed-power tradeoff gone
    ("sb", "avg"): 0.15,  # Empirical: 0.15 (was 0.20)
    ("avg", "obp"): 0.72,  # Empirical: 0.72 (was 0.90); OBP depends on BB rate too
    ("obp", "r"): 0.56,  # Empirical: 0.56 (was 0.55)
    ("obp", "rbi"): 0.41,  # Empirical: 0.41 (was 0.40)
    ("era", "whip"): 0.81,  # Empirical: 0.81 (was 0.90)
    ("k", "era"): -0.46,  # Empirical: -0.46 (was -0.65); K suppresses ERA less
    ("k", "whip"): -0.51,  # Empirical: -0.51 (was -0.50)
    ("w", "k"): 0.66,  # Empirical: 0.66 (was 0.40); good pitchers win more
    ("w", "era"): -0.55,  # Empirical: -0.55 (was -0.45)
    ("w", "l"): -0.16,  # Empirical: -0.16 (was -0.30); W-L weakly anti-correlated
    ("l", "era"): 0.43,  # Empirical: 0.43 (was 0.50)
    ("l", "whip"): 0.36,  # Empirical: 0.36 (was 0.40)
}

# ── Copula import (optional) ─────────────────────────────────────────

try:
    from src.engine.portfolio.copula import GaussianCopula

    _COPULA_AVAILABLE = True
except ImportError:
    _COPULA_AVAILABLE = False


# ── Internal helpers ─────────────────────────────────────────────────


def _build_correlation_matrix() -> np.ndarray:
    """Build 12x12 correlation matrix from DEFAULT_CORRELATIONS."""
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
      - numpy array of shape (n_players, n_cats)
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
            numpy array of shape (n_players, n_cats).
        n_scenarios: Number of scenarios to generate.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (n_scenarios, n_players, n_cats) with sampled stats.
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

            # The copula module uses a different category ordering than this
            # module.  Build a mapping from copula column index to our index.
            from src.engine.portfolio.copula import CATEGORIES as _COPULA_CATS

            _copula_to_local = []
            for copula_cat in _COPULA_CATS:
                _copula_to_local.append(_CAT_IDX[copula_cat.lower()])

            for i in range(n_players):
                # Draw correlated uniforms: (n_scenarios, 10)
                u = copula.sample(n_scenarios, rng=rng)
                # Convert to stat values via inverse CDF, remapping columns
                for copula_j in range(len(_COPULA_CATS)):
                    local_j = _copula_to_local[copula_j]
                    cat = ALL_CATS[local_j]
                    mu = stats[i, local_j]
                    sigma = std_matrix[i, local_j]
                    scenarios[:, i, local_j] = norm.ppf(u[:, copula_j], loc=mu, scale=sigma)
            # Clip non-negative counting stats to >= 0 (same as Cholesky path)
            _NON_NEG_COPULA: set[str] = {"r", "hr", "rbi", "sb", "w", "l", "sv", "k"}
            for j, cat in enumerate(ALL_CATS):
                if cat in _NON_NEG_COPULA:
                    scenarios[:, :, j] = np.maximum(0, scenarios[:, :, j])

            logger.debug(
                "Generated %d scenarios with GaussianCopula for %d players",
                n_scenarios,
                n_players,
            )
            return scenarios
        except Exception as exc:
            logger.warning("Copula sampling failed: %s, falling back to Cholesky", exc, exc_info=True)

    # Fallback: Cholesky-based correlated Normal sampling
    corr = _build_correlation_matrix()

    # Ensure positive definiteness
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 1e-10)
    corr_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(corr_pd))
    corr_pd = corr_pd / np.outer(d, d)

    try:
        cholesky = np.linalg.cholesky(corr_pd)
    except np.linalg.LinAlgError:
        cholesky = np.eye(n_cats)

    # Non-negative counting stats that must be clipped to >= 0
    _NON_NEGATIVE_STATS: set[str] = {"r", "hr", "rbi", "sb", "w", "l", "sv", "k"}

    for i in range(n_players):
        z = rng.standard_normal((n_scenarios, n_cats))
        correlated = z @ cholesky.T
        for j in range(n_cats):
            scenarios[:, i, j] = stats[i, j] + correlated[:, j] * std_matrix[i, j]
        # Clip non-negative counting stats to >= 0
        for j, cat in enumerate(ALL_CATS):
            if cat in _NON_NEGATIVE_STATS:
                scenarios[:, i, j] = np.maximum(0, scenarios[:, i, j])

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
        scenarios: Array of shape (n_scenarios, n_players, n_cats).
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
    hitting_cats = {"r", "hr", "rbi", "sb", "avg", "obp"}
    pitching_cats = {"w", "l", "sv", "k", "era", "whip"}
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
        scenarios: Array of shape (n_scenarios, n_players, n_cats).
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


def compute_empirical_correlations(
    df: pd.DataFrame,
    min_sample: int = 20,
) -> dict[tuple[str, str], float]:
    """Compute pairwise Pearson correlations from observed player stats.

    Takes a DataFrame of player statistics and returns a dictionary
    mapping stat-pair tuples to their empirical correlation coefficients.
    Only numeric columns are included. Pairs where both columns are the
    same are excluded (diagonal of correlation matrix).

    Args:
        df: DataFrame of player stats. Each row is a player, each column
            is a stat category.
        min_sample: Minimum number of rows required to compute correlations.
            Returns empty dict if ``len(df) < min_sample``.

    Returns:
        Dict mapping ``(stat_a, stat_b)`` tuples to Pearson correlation
        coefficients. Only the upper triangle is returned (no duplicates).
    """
    if len(df) < min_sample:
        return {}

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 2:
        return {}

    corr_matrix = df[numeric_cols].corr()

    result: dict[tuple[str, str], float] = {}
    for i, col_a in enumerate(numeric_cols):
        for col_b in numeric_cols[i + 1 :]:
            val = corr_matrix.loc[col_a, col_b]
            if pd.notna(val):
                result[(col_a, col_b)] = float(val)

    return result


def compute_empirical_cvs(
    player_stats: pd.DataFrame,
    stat_cols: list[str] | None = None,
    min_sample: int = 20,
) -> dict[str, float]:
    """Compute empirical coefficients of variation from player stat data.

    For counting stats, computes CV = std(stat) / mean(stat).
    For rate stats (avg, obp, era, whip), computes absolute std instead,
    since CV is not meaningful when means are far from zero.

    Args:
        player_stats: DataFrame of player stats. Each row is a player-season,
            each column is a stat category.
        stat_cols: List of column names to compute CVs for. If ``None``,
            uses all numeric columns.
        min_sample: Minimum number of rows required. Returns empty dict
            if ``len(player_stats) < min_sample``.

    Returns:
        Dict mapping stat name to CV (counting) or absolute std (rate).
    """
    if len(player_stats) < min_sample:
        return {}

    if stat_cols is None:
        stat_cols = player_stats.select_dtypes(include="number").columns.tolist()

    result: dict[str, float] = {}
    for col in stat_cols:
        if col not in player_stats.columns:
            continue
        vals = pd.to_numeric(player_stats[col], errors="coerce").dropna()
        if len(vals) < min_sample:
            continue

        if col in _RATE_STATS:
            # Rate stats: absolute standard deviation
            result[col] = float(vals.std())
        else:
            # Counting stats: coefficient of variation
            mean_val = vals.mean()
            if abs(mean_val) > 1e-6:
                result[col] = float(vals.std() / mean_val)

    return result


def _is_hitter_from_stats(stat_row: np.ndarray) -> bool:
    """Infer whether a player is a hitter from their stat line.

    Heuristic: if IP-related stats (w, sv, k, era, whip) are all zero
    or near-zero, the player is a hitter.
    """
    # Pitching cat indices (from ALL_CATS): w=6, l=7, sv=8, k=9, era=10, whip=11
    w_idx = _CAT_IDX.get("w", 6)
    sv_idx = _CAT_IDX.get("sv", 8)
    k_idx = _CAT_IDX.get("k", 9)
    pitching_sum = abs(stat_row[w_idx]) + abs(stat_row[sv_idx]) + abs(stat_row[k_idx])
    return pitching_sum < 0.01

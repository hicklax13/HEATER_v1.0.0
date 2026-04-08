"""Weighted projection stacking via ridge regression.

Computes per-system weights by regressing each projection system's forecasts
against prior-year actuals. Ridge regularization (L2) ensures stable,
non-negative weights even when systems are highly correlated.

Usage (future wiring into create_blended_projections):
    weights = compute_stacking_weights(systems, actuals, stat="hr", alpha=1.0)
    # -> {"steamer": 0.35, "zips": 0.28, "depthcharts": 0.22, "marcel": 0.15}
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Minimum matched players required for regression; below this we fall back
# to uniform 1/n weights.
MIN_PLAYERS_FOR_REGRESSION = 10


def compute_stacking_weights(
    systems: dict[str, pd.DataFrame],
    actuals: pd.DataFrame,
    stat: str = "hr",
    alpha: float = 1.0,
) -> dict[str, float]:
    """Compute per-system weights via ridge regression.

    Each system's projection for *stat* is a feature column. The target vector
    is the actual values from *actuals*. Ridge regression (closed-form) yields
    weights that are then clipped to non-negative and normalized to sum to 1.

    Parameters
    ----------
    systems : dict[str, DataFrame]
        Mapping of system name -> DataFrame. Each DataFrame must contain
        ``player_id`` and the column named *stat*.
    actuals : DataFrame
        Must contain ``player_id`` and the column named *stat*.
    stat : str
        The statistic to regress on (e.g. ``"hr"``, ``"avg"``, ``"era"``).
    alpha : float
        Ridge regularization strength. Higher values shrink weights toward
        uniform. Default 1.0.

    Returns
    -------
    dict[str, float]
        ``{system_name: weight}`` summing to 1.0.
    """
    system_names = sorted(systems.keys())
    n_systems = len(system_names)

    if n_systems == 0:
        return {}

    uniform: dict[str, float] = {s: 1.0 / n_systems for s in system_names}

    # ---- Validate that actuals has the required column ----
    if stat not in actuals.columns or "player_id" not in actuals.columns:
        logger.warning(
            "Actuals missing '%s' or 'player_id' column; returning uniform weights.",
            stat,
        )
        return uniform

    # ---- Find common player_ids across ALL systems and actuals ----
    common_ids: set[Any] = set(actuals["player_id"])
    valid_systems: list[str] = []
    for name in system_names:
        df = systems[name]
        if stat not in df.columns or "player_id" not in df.columns:
            logger.info("System '%s' missing stat '%s'; skipping.", name, stat)
            continue
        common_ids &= set(df["player_id"])
        valid_systems.append(name)

    if len(valid_systems) == 0:
        return uniform

    common_ids_list = sorted(common_ids)

    if len(common_ids_list) < MIN_PLAYERS_FOR_REGRESSION:
        logger.info(
            "Only %d matched players (need %d); returning uniform weights.",
            len(common_ids_list),
            MIN_PLAYERS_FOR_REGRESSION,
        )
        return {s: 1.0 / len(valid_systems) for s in valid_systems}

    # ---- Build feature matrix X (n_players x n_valid_systems) and target y ----
    actuals_indexed = actuals.set_index("player_id")
    y = np.array(
        [float(actuals_indexed.loc[pid, stat]) for pid in common_ids_list],
        dtype=np.float64,
    )

    X = np.zeros((len(common_ids_list), len(valid_systems)), dtype=np.float64)
    for j, name in enumerate(valid_systems):
        df_indexed = systems[name].set_index("player_id")
        for i, pid in enumerate(common_ids_list):
            X[i, j] = float(df_indexed.loc[pid, stat])

    # ---- Guard: skip if target or any system has zero variance ----
    if np.std(y) < 1e-12:
        logger.info("Target stat '%s' has zero variance; returning uniform.", stat)
        return {s: 1.0 / len(valid_systems) for s in valid_systems}

    # Remove systems with zero variance (constant predictions)
    variances = np.std(X, axis=0)
    keep_mask = variances > 1e-12
    if not np.any(keep_mask):
        return {s: 1.0 / len(valid_systems) for s in valid_systems}

    kept_systems = [valid_systems[j] for j in range(len(valid_systems)) if keep_mask[j]]
    X_kept = X[:, keep_mask]

    # ---- Ridge regression closed form: w = (X'X + alpha*I)^{-1} X'y ----
    XtX = X_kept.T @ X_kept
    Xty = X_kept.T @ y
    n_kept = XtX.shape[0]

    try:
        w = np.linalg.solve(XtX + alpha * np.eye(n_kept), Xty)
    except np.linalg.LinAlgError:
        logger.warning("Ridge solve failed; returning uniform weights.")
        return {s: 1.0 / len(valid_systems) for s in valid_systems}

    # ---- Clip to non-negative and normalize ----
    w = np.clip(w, 0.0, None)
    total = w.sum()
    if total < 1e-12:
        # All weights clipped to zero — fall back to uniform among kept systems
        return {s: 1.0 / len(kept_systems) for s in kept_systems}

    w = w / total

    return {name: float(w[i]) for i, name in enumerate(kept_systems)}


def compute_all_stat_weights(
    systems: dict[str, pd.DataFrame],
    actuals: pd.DataFrame,
    stats: list[str] | None = None,
    alpha: float = 1.0,
) -> dict[str, dict[str, float]]:
    """Compute stacking weights for multiple stats.

    Parameters
    ----------
    systems : dict[str, DataFrame]
        Mapping of system name -> DataFrame with ``player_id`` + stat columns.
    actuals : DataFrame
        Must contain ``player_id`` + stat columns.
    stats : list[str] | None
        Stats to compute weights for. If None, uses the default fantasy
        categories: ``["r", "hr", "rbi", "sb", "avg", "obp", "w", "l",
        "sv", "k", "era", "whip"]``.
    alpha : float
        Ridge regularization strength.

    Returns
    -------
    dict[str, dict[str, float]]
        ``{stat: {system_name: weight}}``.
    """
    if stats is None:
        stats = ["r", "hr", "rbi", "sb", "avg", "obp", "w", "l", "sv", "k", "era", "whip"]

    return {stat: compute_stacking_weights(systems, actuals, stat=stat, alpha=alpha) for stat in stats}

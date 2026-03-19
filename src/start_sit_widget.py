# src/start_sit_widget.py
"""Quick 2-4 player 'Who Should I Start?' compare with density overlap."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm


def compute_fantasy_points_distribution(
    player: pd.Series,
    sgp_denominators: dict[str, float] | None = None,
) -> tuple[float, float]:
    """Estimate mean and std of weekly fantasy point contribution (SGP units).

    Uses projected stats + per-stat volatility to build a Normal approximation.

    Returns:
        (mean_sgp, std_sgp)
    """
    sgp = sgp_denominators or {
        "R": 18.0,
        "HR": 8.0,
        "RBI": 22.0,
        "SB": 5.0,
        "AVG": 0.004,
        "OBP": 0.005,
        "W": 2.5,
        "SV": 6.0,
        "K": 28.0,
        "ERA": 0.27,
        "WHIP": 0.02,
        "L": 2.0,
    }
    # Per-stat weekly volatility (fraction of season projection)
    vol_fracs = {
        "r": 0.35,
        "hr": 0.45,
        "rbi": 0.35,
        "sb": 0.50,
        "w": 0.50,
        "sv": 0.45,
        "k": 0.30,
    }

    is_hitter = bool(player.get("is_hitter", True))
    mean_sgp = 0.0
    var_sgp = 0.0

    if is_hitter:
        stat_map = {"R": "r", "HR": "hr", "RBI": "rbi", "SB": "sb"}
        for cat, col in stat_map.items():
            val = float(player.get(col, 0) or 0)
            denom = sgp.get(cat, 1.0)
            if denom > 0:
                weekly = val / 22.0  # ~22 weeks in season
                mean_sgp += weekly / denom
                vol = vol_fracs.get(col, 0.35) * weekly
                var_sgp += (vol / denom) ** 2
    else:
        stat_map = {"W": "w", "SV": "sv", "K": "k"}
        for cat, col in stat_map.items():
            val = float(player.get(col, 0) or 0)
            denom = sgp.get(cat, 1.0)
            if denom > 0:
                weekly = val / 22.0
                mean_sgp += weekly / denom
                vol = vol_fracs.get(col, 0.40) * weekly
                var_sgp += (vol / denom) ** 2

    std_sgp = max(0.01, math.sqrt(var_sgp))
    return round(mean_sgp, 4), round(std_sgp, 4)


def compute_overlap_probability(mu1: float, sigma1: float, mu2: float, sigma2: float) -> float:
    """Compute overlap coefficient between two Normal distributions.

    OVL = 2 * Phi(-|mu1 - mu2| / sqrt(sigma1^2 + sigma2^2))

    Returns value in [0, 1] where 1.0 = identical distributions.
    """
    if sigma1 <= 0 or sigma2 <= 0:
        return 0.0
    diff = abs(mu1 - mu2)
    combined_sigma = math.sqrt(sigma1**2 + sigma2**2)
    if combined_sigma <= 0:
        return 0.0
    return round(float(2.0 * norm.cdf(-diff / combined_sigma)), 4)


def generate_density_data(mu: float, sigma: float, n_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Generate x, y arrays for plotting a Normal density curve.

    Returns:
        (x_values, y_values) arrays of length n_points
    """
    x_min = mu - 4 * sigma
    x_max = mu + 4 * sigma
    x = np.linspace(x_min, x_max, n_points)
    y = norm.pdf(x, mu, sigma)
    return x, y


def quick_start_sit(
    player_ids: list[int],
    player_pool: pd.DataFrame,
    sgp_denominators: dict[str, float] | None = None,
) -> dict:
    """Compare 2-4 players for start/sit decision.

    Returns:
        {
            "recommendation": int (player_id of best start),
            "players": [
                {"player_id": int, "name": str, "mean_sgp": float,
                 "std_sgp": float, "density_x": array, "density_y": array}
            ],
            "overlaps": {(pid1, pid2): float, ...}  # pairwise overlap coefficients
        }
    """
    if len(player_ids) < 2:
        return {"recommendation": player_ids[0] if player_ids else 0, "players": [], "overlaps": {}}

    results = []
    for pid in player_ids:
        match = player_pool[player_pool["player_id"] == pid]
        if match.empty:
            continue
        row = match.iloc[0]
        mean_sgp, std_sgp = compute_fantasy_points_distribution(row, sgp_denominators)
        x, y = generate_density_data(mean_sgp, std_sgp)
        results.append(
            {
                "player_id": pid,
                "name": str(row.get("name", f"Player {pid}")),
                "mean_sgp": mean_sgp,
                "std_sgp": std_sgp,
                "density_x": x,
                "density_y": y,
            }
        )

    # Pairwise overlaps
    overlaps: dict[tuple[int, int], float] = {}
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a = results[i]
            b = results[j]
            ovl = compute_overlap_probability(a["mean_sgp"], a["std_sgp"], b["mean_sgp"], b["std_sgp"])
            overlaps[(a["player_id"], b["player_id"])] = ovl

    # Recommendation: highest mean SGP
    best = max(results, key=lambda r: r["mean_sgp"]) if results else None
    recommendation = best["player_id"] if best else 0

    return {
        "recommendation": recommendation,
        "players": results,
        "overlaps": overlaps,
    }

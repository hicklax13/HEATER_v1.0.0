"""Exponential signal decay weighting for recency.

Spec reference: Section 4 L1A (Signal Processing — Exponential Decay)
               Section 17 Phase 3 item 14

Different signal types have different half-lives because the underlying
processes change at different rates:
  - Exit velocity (35 days) — mechanical changes show up fast
  - Spin rate (46 days) — pitch adjustments are moderately sticky
  - Plate discipline (58 days) — approach changes evolve slowly
  - Traditional rates (87 days) — AVG/OBP need large samples to stabilize
  - Sprint speed (139 days) — speed changes slowly (often injury-related)
  - Injury history (231 days) — historical health patterns are very persistent

Usage:
  Given a time series of observations, each observation gets a weight
  based on how recent it is. More recent observations dominate.

Wires into:
  - src/engine/signals/statcast.py: weight Statcast observations by recency
  - src/engine/signals/kalman.py: feed weighted data for true talent estimation
"""

from __future__ import annotations

from datetime import date

import numpy as np

# Decay parameters (lambda) by signal type
# Half-life = ln(2) / lambda ≈ 0.693 / lambda
# Spec ref: Section 4 L1A DECAY_LAMBDAS
DECAY_LAMBDAS: dict[str, float] = {
    "statcast_ev": 0.020,  # half-life ~35 days
    "statcast_spin": 0.015,  # half-life ~46 days
    "plate_discipline": 0.012,  # half-life ~58 days
    "traditional_rate": 0.008,  # half-life ~87 days
    "sprint_speed": 0.005,  # half-life ~139 days
    "injury_history": 0.003,  # half-life ~231 days
    "season_counting": 0.000,  # no decay (cumulative stats like R, HR, RBI)
}

# Map feature names to their decay category
FEATURE_DECAY_MAP: dict[str, str] = {
    # Batted ball → statcast_ev
    "ev_mean": "statcast_ev",
    "ev_p90": "statcast_ev",
    "ev_std": "statcast_ev",
    "barrel_pct": "statcast_ev",
    "hard_hit_pct": "statcast_ev",
    "la_mean": "statcast_ev",
    "la_sweet_spot_pct": "statcast_ev",
    "xba": "statcast_ev",
    "xslg": "statcast_ev",
    "xwoba": "statcast_ev",
    "xiso": "statcast_ev",
    # Pitching stuff → statcast_spin
    "ff_avg_speed": "statcast_spin",
    "ff_spin_rate": "statcast_spin",
    "extension": "statcast_spin",
    # Plate discipline
    "o_swing_pct": "plate_discipline",
    "z_swing_pct": "plate_discipline",
    "o_contact_pct": "plate_discipline",
    "z_contact_pct": "plate_discipline",
    "swstr_pct": "plate_discipline",
    "csw_pct": "plate_discipline",
    "whiff_pct": "plate_discipline",
    "k_pct": "plate_discipline",
    "bb_pct": "plate_discipline",
    # Traditional rates
    "avg": "traditional_rate",
    "era": "traditional_rate",
    "whip": "traditional_rate",
    # Speed
    "sprint_speed": "sprint_speed",
    "sprint_speed_delta": "sprint_speed",
    # Counting stats (no decay)
    "r": "season_counting",
    "hr": "season_counting",
    "rbi": "season_counting",
    "sb": "season_counting",
    "w": "season_counting",
    "k": "season_counting",
    "sv": "season_counting",
}


def decay_weight(
    observation_date: date,
    today: date | None = None,
    lambda_param: float = 0.020,
) -> float:
    """Compute exponential decay weight for a single observation.

    Spec ref: Section 4 L1A — decay_weight function.

    w(t) = exp(-lambda * days_ago)

    Args:
        observation_date: When the observation was recorded.
        today: Reference date (defaults to today).
        lambda_param: Decay rate. Higher = faster decay.

    Returns:
        Weight in (0, 1]. 1.0 for today, decaying toward 0.
    """
    if today is None:
        today = date.today()

    days_ago = max(0, (today - observation_date).days)
    return float(np.exp(-lambda_param * days_ago))


def get_feature_lambda(feature_name: str) -> float:
    """Look up the decay rate for a feature.

    Args:
        feature_name: Feature name (e.g., "ev_mean", "xwoba").

    Returns:
        Lambda value for exponential decay.
    """
    category = FEATURE_DECAY_MAP.get(feature_name, "traditional_rate")
    return DECAY_LAMBDAS.get(category, 0.008)


def apply_decay_weights(
    observations: list[dict],
    feature_name: str,
    today: date | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply recency-weighted decay to a time series of observations.

    Args:
        observations: List of {date: date_val, value: float_val} dicts,
            sorted chronologically.
        feature_name: Feature name (determines decay rate).
        today: Reference date.

    Returns:
        Tuple of (values, weights) arrays, same length as observations.
    """
    if not observations:
        return np.array([]), np.array([])

    lam = get_feature_lambda(feature_name)

    values = np.array([obs["value"] for obs in observations], dtype=float)
    weights = np.array(
        [decay_weight(obs["date"], today, lam) for obs in observations],
        dtype=float,
    )

    return values, weights


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute decay-weighted mean.

    Args:
        values: Observation values.
        weights: Corresponding weights.

    Returns:
        Weighted mean, or 0.0 if no valid weights.
    """
    w_sum = weights.sum()
    if w_sum < 1e-10:
        return 0.0
    return float(np.sum(values * weights) / w_sum)


def weighted_variance(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute decay-weighted variance.

    Args:
        values: Observation values.
        weights: Corresponding weights.

    Returns:
        Weighted variance.
    """
    w_sum = weights.sum()
    if w_sum < 1e-10:
        return 0.0

    w_mean = weighted_mean(values, weights)
    return float(np.sum(weights * (values - w_mean) ** 2) / w_sum)


def half_life_days(lambda_param: float) -> float:
    """Convert lambda to half-life in days.

    Args:
        lambda_param: Decay rate.

    Returns:
        Number of days until weight drops to 0.5.
    """
    if lambda_param <= 0:
        return float("inf")
    return float(np.log(2) / lambda_param)

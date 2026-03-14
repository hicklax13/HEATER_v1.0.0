"""Kalman filter for true talent estimation.

Spec reference: Section 4 L1D (Kalman Filter for True Talent)
               Section 17 Phase 3 item 15

The Kalman filter treats player stats as a state-space model:

  State equation:    True_talent(t) = True_talent(t-1) + process_noise
  Measurement eq:    Observed(t) = True_talent(t) + observation_noise

This separates "true skill" from "noise" (BABIP luck, small samples).
The filter updates its estimate as each new observation arrives, with
a Kalman gain that determines how much to trust the new data vs the
prior estimate.

Key property: Observation variance depends on sample size. A .350 AVG
over 20 AB has high observation variance (the true talent might be
anywhere from .250 to .450). A .350 AVG over 500 AB has low variance
(true talent is likely .330-.370).

Wires into:
  - src/engine/signals/statcast.py: feed rolling feature observations
  - src/engine/signals/decay.py: recency-weighted prior initialization
  - src/engine/projections/bayesian_blend.py: blend Kalman output with BMA
"""

from __future__ import annotations

import numpy as np

# Base variance parameters by stat type
# These define how noisy an observation is given the sample size.
# Spec ref: Section 4 L1D — observation_variance function
OBSERVATION_VARIANCE_BASE: dict[str, callable] = {
    "ba": lambda n: 0.25 / max(n, 1),  # binomial variance of batting average
    "hr_rate": lambda n: 0.03 / max(n, 1),
    "sb_rate": lambda n: 0.05 / max(n, 1),
    "era": lambda n: 15.0 / max(n, 1),  # run variance
    "whip": lambda n: 0.5 / max(n, 1),
    "k_rate": lambda n: 0.20 / max(n, 1),
    "xwoba": lambda n: 0.15 / max(n, 1),
    "ev_mean": lambda n: 25.0 / max(n, 1),  # exit velocity variance
    "barrel_pct": lambda n: 0.10 / max(n, 1),
    "whiff_pct": lambda n: 0.15 / max(n, 1),
    "csw_pct": lambda n: 0.10 / max(n, 1),
    "sprint_speed": lambda n: 0.5 / max(n, 1),
}

# Process variance: how much true talent can change per time step.
# Lower values → more stable estimates (good for established players).
# Higher values → more responsive to changes (good for breakouts).
PROCESS_VARIANCE_DEFAULTS: dict[str, float] = {
    "ba": 0.0001,
    "hr_rate": 0.0002,
    "sb_rate": 0.0003,
    "era": 0.01,
    "whip": 0.001,
    "k_rate": 0.0005,
    "xwoba": 0.0002,
    "ev_mean": 0.5,
    "barrel_pct": 0.001,
    "whiff_pct": 0.001,
    "csw_pct": 0.0005,
    "sprint_speed": 0.01,
}


def kalman_true_talent(
    observations: np.ndarray,
    obs_variance: np.ndarray,
    process_variance: float,
    prior_mean: float,
    prior_variance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Kalman filter to estimate true talent from noisy observations.

    Spec ref: Section 4 L1D — kalman_true_talent function.

    State-space model:
      True_talent(t) = True_talent(t-1) + N(0, process_variance)
      Observed(t) = True_talent(t) + N(0, obs_variance(t))

    At each step:
      1. Predict: variance grows by process_variance
      2. Update: Kalman gain K = pred_var / (pred_var + obs_var)
      3. New estimate = old_estimate + K * (observation - old_estimate)
      4. New variance = (1 - K) * pred_var

    When observation variance is high (small sample), K is small → trust prior.
    When observation variance is low (large sample), K is large → trust data.

    Args:
        observations: Array of n observed values (e.g., rolling 14-day AVG).
        obs_variance: Array of n observation variances (depends on sample size).
        process_variance: How much true talent can change between observations.
        prior_mean: Initial estimate of true talent (e.g., preseason projection).
        prior_variance: Initial uncertainty in the prior.

    Returns:
        Tuple of (filtered_means, filtered_variances), each array of length n.
    """
    n = len(observations)
    if n == 0:
        return np.array([]), np.array([])

    filtered_mean = np.zeros(n)
    filtered_var = np.zeros(n)

    pred_mean = prior_mean
    pred_var = prior_variance

    for t in range(n):
        # Predict step: variance grows
        pred_var += process_variance

        # Update step: Kalman gain
        K = pred_var / (pred_var + obs_variance[t])

        # Filtered state
        filtered_mean[t] = pred_mean + K * (observations[t] - pred_mean)
        filtered_var[t] = (1 - K) * pred_var

        # Set up next prediction
        pred_mean = filtered_mean[t]
        pred_var = filtered_var[t]

    return filtered_mean, filtered_var


def observation_variance(stat_type: str, sample_size: float) -> float:
    """Compute observation variance for a stat given sample size.

    Spec ref: Section 4 L1D — observation_variance function.

    Larger samples → smaller variance → Kalman filter trusts data more.

    Args:
        stat_type: Type of stat (e.g., "ba", "era", "xwoba").
        sample_size: Number of PA, IP, or pitches in the observation.

    Returns:
        Observation variance (float).
    """
    var_fn = OBSERVATION_VARIANCE_BASE.get(stat_type, lambda n: 1.0 / max(n, 1))
    return var_fn(sample_size)


def get_process_variance(stat_type: str) -> float:
    """Get default process variance for a stat type.

    Args:
        stat_type: Type of stat.

    Returns:
        Process variance for the Kalman filter.
    """
    return PROCESS_VARIANCE_DEFAULTS.get(stat_type, 0.001)


def run_kalman_for_feature(
    rolling_values: list[dict],
    stat_type: str,
    prior_mean: float | None = None,
    prior_variance: float | None = None,
) -> dict:
    """Convenience: run Kalman filter on a rolling feature time series.

    Args:
        rolling_values: List of {value: float, sample_size: float} dicts,
            sorted chronologically.
        stat_type: Stat type for variance calibration.
        prior_mean: Initial estimate. Defaults to first observation.
        prior_variance: Initial uncertainty. Defaults to large value.

    Returns:
        Dict with:
          - filtered_mean: Final filtered estimate (best guess of true talent)
          - filtered_var: Final uncertainty
          - filtered_means: Full trajectory
          - filtered_vars: Full variance trajectory
          - kalman_gain_final: Last Kalman gain (how responsive the filter is)
    """
    if not rolling_values:
        return {
            "filtered_mean": prior_mean or 0.0,
            "filtered_var": prior_variance or 1.0,
            "filtered_means": np.array([]),
            "filtered_vars": np.array([]),
            "kalman_gain_final": 0.0,
        }

    obs = np.array([v["value"] for v in rolling_values])
    obs_var = np.array([observation_variance(stat_type, v.get("sample_size", 50)) for v in rolling_values])

    if prior_mean is None:
        prior_mean = obs[0]
    if prior_variance is None:
        # Start with high uncertainty — let the data speak
        prior_variance = obs_var[0] * 10

    proc_var = get_process_variance(stat_type)

    f_mean, f_var = kalman_true_talent(
        observations=obs,
        obs_variance=obs_var,
        process_variance=proc_var,
        prior_mean=prior_mean,
        prior_variance=prior_variance,
    )

    # Compute final Kalman gain
    final_pred_var = f_var[-1] + proc_var if len(f_var) > 0 else prior_variance
    final_obs_var = obs_var[-1] if len(obs_var) > 0 else 1.0
    final_K = final_pred_var / (final_pred_var + final_obs_var)

    return {
        "filtered_mean": float(f_mean[-1]) if len(f_mean) > 0 else prior_mean,
        "filtered_var": float(f_var[-1]) if len(f_var) > 0 else prior_variance,
        "filtered_means": f_mean,
        "filtered_vars": f_var,
        "kalman_gain_final": float(final_K),
    }

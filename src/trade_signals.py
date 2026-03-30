"""Trade Signals — Kalman + regime trend adjustment for trade valuations.

Produces a -15 to +15 trend adjustment score that modifies the Trade
Readiness composite. Uses the Kalman filter for true-talent estimation
and simple slope detection for trend direction.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

MAX_ADJUSTMENT = 15.0
MIN_OBSERVATIONS = 3


def compute_trend_adjustment(
    observations: np.ndarray,
    prior_mean: float,
    process_variance: float = 0.001,
) -> float:
    """Compute a trend adjustment score from recent observations.

    Uses a lightweight Kalman filter to estimate true talent, then
    compares the slope of the filtered signal to detect trends.

    Args:
        observations: Array of recent performance values (e.g., rolling AVG).
        prior_mean: Expected baseline (e.g., projected AVG).
        process_variance: How much true talent can shift per period.

    Returns:
        Float in [-15, +15]. Positive = trending up, negative = trending down.
        Returns 0.0 if insufficient data.
    """
    if len(observations) < MIN_OBSERVATIONS:
        return 0.0

    try:
        from src.engine.signals.kalman import kalman_true_talent

        obs_var = np.full(len(observations), 0.01)
        filtered_means, _ = kalman_true_talent(
            observations=observations,
            obs_variance=obs_var,
            process_variance=process_variance,
            prior_mean=prior_mean,
            prior_variance=process_variance * 10,
        )

        n = len(filtered_means)
        x = np.arange(n, dtype=float)
        slope = np.polyfit(x, filtered_means, 1)[0]
        adjustment = slope * 1000
        return float(np.clip(adjustment, -MAX_ADJUSTMENT, MAX_ADJUSTMENT))

    except ImportError:
        logger.debug("Kalman filter not available, returning zero adjustment")
        return 0.0
    except Exception:
        logger.debug("Trend adjustment failed", exc_info=True)
        return 0.0

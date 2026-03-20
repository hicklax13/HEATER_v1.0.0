"""Regime detection via BOCPD and Hidden Markov Model.

Spec reference: Section 5 L2 (Regime Detection)
               Section 17 Phase 3 item 16

Two complementary approaches for detecting when a player's underlying
performance has fundamentally changed:

1. **BOCPD (Bayesian Online Changepoint Detection)**
   - Adams & MacKay (2007) algorithm
   - Detects discrete changepoints: mechanical changes, injury returns,
     role changes, pitch mix adjustments
   - When changepoint probability > threshold, only post-changepoint
     data should feed projections
   - Binary signal: "something changed" or "steady state"

2. **HMM (Hidden Markov Model)**
   - 4 states: Elite, Above-average, Below-average, Replacement
   - Softer regime detection — gives probability distribution over states
   - Observation matrix: [xwOBA_14d, ev_p90_14d, barrel_pct_14d]
   - State probabilities feed into projection blending as mixture weights

Wires into:
  - src/engine/signals/statcast.py: rolling feature observations
  - src/engine/signals/kalman.py: reset Kalman prior on changepoint
  - src/engine/projections/bayesian_blend.py: regime-conditional projections
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)

# HMM availability flag
try:
    from hmmlearn import hmm

    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False


# ── BOCPD ─────────────────────────────────────────────────────────────


class BOCPD:
    """Bayesian Online Changepoint Detection.

    Spec ref: Section 5 L2A — Adams & MacKay (2007).

    Maintains a posterior distribution over run lengths (how long since
    the last changepoint). At each new observation, computes:
    1. P(new changepoint) — integrates out all run lengths
    2. P(continuation at each run length) — extends existing runs
    3. Updates sufficient statistics for the conjugate model

    Uses Normal-Inverse-Gamma conjugate model for Gaussian data.

    Attributes:
        hazard: Prior probability of a changepoint at each time step.
            Default 1/200 = expect a change every ~200 observations.
        run_length_probs: Posterior over run lengths.
    """

    def __init__(
        self,
        hazard_lambda: int = 200,
        mu0: float = 0.0,
        kappa0: float = 1.0,
        alpha0: float = 1.0,
        beta0: float = 1.0,
    ) -> None:
        """Initialize BOCPD.

        Args:
            hazard_lambda: Expected run length (1/hazard = P(changepoint)).
            mu0: Prior mean for Normal model.
            kappa0: Prior precision count for mean.
            alpha0: Prior shape for variance.
            beta0: Prior scale for variance.
        """
        self.hazard = 1.0 / max(hazard_lambda, 1)
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0 = beta0

        # Posterior state
        self.run_length_probs = np.array([1.0])
        self.suff_stats: list[tuple[float, float, float, float]] = [(mu0, kappa0, alpha0, beta0)]

    def update(self, x: float) -> tuple[float, np.ndarray]:
        """Process one new observation.

        Spec ref: Section 5 L2A — BOCPD.update method.

        Args:
            x: New observation value (e.g., 14-day xwOBA).

        Returns:
            Tuple of:
              - changepoint_prob: P(changepoint just occurred)
              - run_length_probs: Full posterior over all run lengths
        """
        n = len(self.run_length_probs)

        # Compute predictive probability under each run length
        pred_probs = np.zeros(n)
        for r in range(n):
            mu, kappa, alpha, beta = self.suff_stats[r]
            # Student-t predictive (approximated as Normal for simplicity)
            pred_var = beta * (kappa + 1) / (alpha * kappa)
            pred_var = max(pred_var, 1e-10)
            pred_probs[r] = norm.pdf(x, loc=mu, scale=np.sqrt(pred_var))

        # Growth probabilities (no changepoint)
        growth = self.run_length_probs * pred_probs * (1 - self.hazard)

        # Changepoint probability (sum of all run lengths transitioning to r=0)
        cp = float(np.sum(self.run_length_probs * pred_probs * self.hazard))

        # New posterior
        new_probs = np.append(cp, growth)
        total = new_probs.sum()
        if total > 0:
            new_probs /= total
        else:
            new_probs = np.ones(len(new_probs)) / len(new_probs)
        self.run_length_probs = new_probs

        # Update sufficient statistics for each run length
        new_stats: list[tuple[float, float, float, float]] = [(self.mu0, self.kappa0, self.alpha0, self.beta0)]
        for r in range(n):
            mu, kappa, alpha, beta = self.suff_stats[r]
            kn = kappa + 1
            mn = (kappa * mu + x) / kn
            an = alpha + 0.5
            bn = beta + kappa * (x - mu) ** 2 / (2 * kn)
            new_stats.append((mn, kn, an, bn))
        self.suff_stats = new_stats

        return float(new_probs[0]), new_probs

    def reset(self) -> None:
        """Reset to initial state."""
        self.run_length_probs = np.array([1.0])
        self.suff_stats = [(self.mu0, self.kappa0, self.alpha0, self.beta0)]


def detect_changepoints(
    time_series: np.ndarray,
    hazard_lambda: int = 200,
    threshold: float = 0.7,
) -> dict[str, Any]:
    """Run BOCPD on a full time series and detect changepoints.

    Detection uses the run length distribution, not just the raw cp_prob.
    A changepoint is detected when the mode run length drops below
    `threshold` fraction of the current time step, indicating the model
    believes recent data comes from a new distribution.

    Note: cp_prob (P(r_t=0)) is bounded by 1/hazard_lambda. The real
    signal is the run length distribution resetting — the mode drops
    from a high run length to near zero.

    Args:
        time_series: Array of sequential observations (e.g., rolling xwOBA).
        hazard_lambda: Expected run length between changepoints.
        threshold: Detection sensitivity. Lower = more sensitive.
            When the mode run length drops to < threshold * t, a
            changepoint is flagged.

    Returns:
        Dict with:
          - changepoint_indices: List of indices where changepoints detected
          - changepoint_probs: Array of P(changepoint) at each time step
          - last_changepoint: Index of most recent changepoint (or None)
          - current_regime_length: How many observations since last changepoint
    """
    if len(time_series) < 2:
        return {
            "changepoint_indices": [],
            "changepoint_probs": np.array([]),
            "last_changepoint": None,
            "current_regime_length": len(time_series),
        }

    # Initialize BOCPD with data-driven prior
    mu0 = float(np.mean(time_series[: min(10, len(time_series))]))
    detector = BOCPD(
        hazard_lambda=hazard_lambda,
        mu0=mu0,
        kappa0=1.0,
        alpha0=1.0,
        beta0=float(np.var(time_series[: min(10, len(time_series))])) + 0.01,
    )

    cp_probs = np.zeros(len(time_series))
    mode_run_lengths = np.zeros(len(time_series), dtype=int)
    changepoints: list[int] = []
    prev_mode_rl = 0

    for t, x in enumerate(time_series):
        cp_prob, rl_probs = detector.update(float(x))
        cp_probs[t] = cp_prob
        mode_rl = int(np.argmax(rl_probs))
        mode_run_lengths[t] = mode_rl

        # Detect changepoint: mode run length drops sharply
        # (from tracking a long run to a short one)
        # threshold scales detection sensitivity — lower threshold = more sensitive
        if t > 5 and prev_mode_rl > threshold * t and mode_rl <= 2:
            changepoints.append(t)

        prev_mode_rl = mode_rl

    last_cp = changepoints[-1] if changepoints else None
    regime_length = len(time_series) - last_cp if last_cp is not None else len(time_series)

    return {
        "changepoint_indices": changepoints,
        "changepoint_probs": cp_probs,
        "last_changepoint": last_cp,
        "current_regime_length": regime_length,
    }


# ── Hidden Markov Model ──────────────────────────────────────────────


# State labels for the 4-state HMM
REGIME_STATES: list[str] = ["Elite", "Above-avg", "Below-avg", "Replacement"]

# Default state probabilities when HMM is unavailable
DEFAULT_STATE_PROBS: np.ndarray = np.array([0.1, 0.4, 0.4, 0.1])


def fit_player_hmm(
    obs_matrix: np.ndarray,
    n_states: int = 4,
) -> tuple[Any | None, np.ndarray]:
    """Fit a Gaussian HMM to player observation matrix.

    Spec ref: Section 5 L2B — Hidden Markov Model (4 States).

    Observations: (n_timepoints, n_features) matrix.
    Typical features: [xwOBA_14d, ev_p90_14d, barrel_pct_14d]

    States represent performance regimes:
      0 = Elite (top-tier production)
      1 = Above-average
      2 = Below-average
      3 = Replacement (struggling / injured return)

    Args:
        obs_matrix: Array of shape (n_timepoints, n_features).
        n_states: Number of hidden states.

    Returns:
        Tuple of (model, current_state_probs):
          - model: Fitted GaussianHMM (or None if unavailable)
          - current_state_probs: Array of shape (n_states,) with
            probability of being in each state at the latest time point
    """
    if not HMM_AVAILABLE:
        logger.info("hmmlearn not installed — using default state probs")
        return None, DEFAULT_STATE_PROBS[:n_states]

    if len(obs_matrix) < 10:
        # Not enough data for HMM
        return None, DEFAULT_STATE_PROBS[:n_states]

    try:
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=200,
            random_state=42,
        )
        # Set initial state probabilities
        model.startprob_ = np.array([0.1, 0.5, 0.3, 0.1])[:n_states]
        model.startprob_ /= model.startprob_.sum()

        model.fit(obs_matrix)

        # Get current state probabilities (from the last observation)
        state_probs = model.predict_proba(obs_matrix)
        current_probs = state_probs[-1]

        return model, current_probs
    except Exception as exc:
        logger.warning("HMM fitting failed: %s — using defaults", exc)
        return None, DEFAULT_STATE_PROBS[:n_states]


def regime_conditional_projection(
    current_probs: np.ndarray,
    projections_by_state: list[dict[str, float]],
) -> dict[str, float]:
    """Compute mixture projection weighted by current state probabilities.

    Spec ref: Section 5 L2B — regime_conditional_projection.

    If a player is [0.6 Elite, 0.3 Above, 0.1 Below, 0.0 Repl]:
      projection = 0.6 * elite_proj + 0.3 * above_proj + 0.1 * below_proj

    Args:
        current_probs: State probability vector of length n_states.
        projections_by_state: List of n_states projection dicts.
            projections_by_state[i] = {stat: value} for state i.

    Returns:
        Blended projection dict {stat: weighted_value}.
    """
    if not projections_by_state:
        return {}

    # Collect all stat keys
    all_stats: set[str] = set()
    for proj in projections_by_state:
        all_stats.update(proj.keys())

    result: dict[str, float] = {}
    for stat in all_stats:
        result[stat] = sum(
            float(current_probs[s]) * projections_by_state[s].get(stat, 0.0)
            for s in range(min(len(current_probs), len(projections_by_state)))
        )

    return result


def classify_regime_simple(
    recent_xwoba: float,
    season_xwoba: float,
    league_avg_xwoba: float = 0.315,
) -> tuple[str, np.ndarray]:
    """Simple rule-based regime classification (fallback when no HMM data).

    Uses comparison of recent vs season xwOBA to estimate regime.

    Args:
        recent_xwoba: Rolling 14-day xwOBA.
        season_xwoba: Season-long xwOBA.
        league_avg_xwoba: League average xwOBA.

    Returns:
        Tuple of (regime_label, state_probabilities).
    """
    delta = recent_xwoba - season_xwoba
    level = recent_xwoba - league_avg_xwoba

    # Classify based on level and trend
    if level > 0.060:  # Well above average
        probs = np.array([0.6, 0.3, 0.08, 0.02])
        label = "Elite"
    elif level > 0.020:
        probs = np.array([0.2, 0.5, 0.25, 0.05])
        label = "Above-avg"
    elif level > -0.030:
        probs = np.array([0.05, 0.25, 0.5, 0.2])
        label = "Below-avg"
    else:
        probs = np.array([0.02, 0.08, 0.3, 0.6])
        label = "Replacement"

    # Shift probabilities based on trend (recent vs season)
    # Accumulate boundary mass to preserve total probability
    if delta > 0.030:  # Trending up
        shifted = np.zeros_like(probs)
        shifted[:-1] = probs[1:]  # shift mass toward better (lower) indices
        shifted[0] += probs[0]  # accumulate boundary mass at best state
        probs = shifted / shifted.sum()
    elif delta < -0.030:  # Trending down
        shifted = np.zeros_like(probs)
        shifted[1:] = probs[:-1]  # shift mass toward worse (higher) indices
        shifted[-1] += probs[-1]  # accumulate boundary mass at worst state
        probs = shifted / shifted.sum()

    return label, probs

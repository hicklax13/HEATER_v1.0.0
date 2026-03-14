"""Injury stochastic process for Monte Carlo trade simulation.

Spec reference: Section 8 L5 (Injury and Availability Engine)
               Section 17 Phase 4 item 18

Extends the deterministic health scores from src/injury_model.py with
stochastic modeling:
  1. Weibull-distributed injury durations by body part
  2. Frailty multipliers from health history
  3. Season availability sampling for MC integration

The key insight: a player with 0.70 health score isn't just "30% less
available." Their DISTRIBUTION of outcomes is different — they have a
fat left tail of injury scenarios. A healthy player's distribution is
tight around full availability; a fragile player has high variance with
significant downside mass.

Wires into:
  - src/injury_model.py: health scores, age-risk curves
  - src/engine/monte_carlo/trade_simulator.py: availability in MC sims
  - src/engine/output/trade_evaluator.py: injury risk context
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import weibull_min

from src.injury_model import (
    age_risk_adjustment,
)

logger = logging.getLogger(__name__)


# ── Injury Duration Parameters ──────────────────────────────────────

# Weibull distribution parameters by injury type
# shape (c): controls tail heaviness (higher = less variable)
# scale: typical duration in days
# loc: minimum duration (shift parameter)
# Source: Spec Section 8 L5B
INJURY_DURATION: dict[str, dict[str, float]] = {
    "hamstring": {"shape": 1.8, "scale": 18, "loc": 10},
    "oblique": {"shape": 2.0, "scale": 25, "loc": 15},
    "ucl": {"shape": 3.0, "scale": 365, "loc": 300},
    "shoulder": {"shape": 1.5, "scale": 45, "loc": 15},
    "back": {"shape": 1.3, "scale": 30, "loc": 10},
    "other": {"shape": 1.5, "scale": 20, "loc": 10},
}

# Base probability of any injury per 30-day period (MLB average)
BASE_INJURY_PROB_30D: float = 0.08

# Season length in days (for availability fraction calculation)
SEASON_DAYS: int = 183  # ~April 1 to September 30


# ── Injury Duration Sampling ───────────────────────────────────────


def sample_injury_duration(
    injury_type: str = "other",
    frailty: float = 1.0,
    rng: np.random.RandomState | None = None,
) -> int:
    """Sample an injury duration from a Weibull distribution.

    Spec ref: Section 8 L5B — injury duration (Semi-Markov).

    The Weibull distribution captures the realistic pattern that:
    - Most injuries cluster around a typical duration
    - Some injuries linger much longer (right tail)
    - There's a hard minimum (the loc parameter)

    Args:
        injury_type: Type of injury (hamstring, oblique, ucl, etc.).
        frailty: Player-specific multiplier for duration (>1 = longer).
            Derived from health score via frailty_from_health_score().
        rng: Random state for reproducibility in MC sims.

    Returns:
        Injury duration in days (minimum 10).
    """
    params = INJURY_DURATION.get(injury_type.lower(), INJURY_DURATION["other"])

    if rng is not None:
        # Use rng for reproducible sampling
        u = rng.uniform(0, 1)
        duration = weibull_min.ppf(u, c=params["shape"], scale=params["scale"] * frailty, loc=params["loc"])
    else:
        duration = weibull_min.rvs(c=params["shape"], scale=params["scale"] * frailty, loc=params["loc"])

    return max(10, int(duration))


def frailty_from_health_score(health_score: float) -> float:
    """Convert a 0-1 health score to a Weibull frailty multiplier.

    A player with perfect health (1.0) gets frailty=1.0 (normal
    injury durations). Injury-prone players (low health scores)
    get frailty>1.0 (longer expected injury durations).

    Formula: frailty = 1.0 / max(health_score, 0.5)
    Clamped so frailty never exceeds 2.0 (health_score floor at 0.5).

    Args:
        health_score: Player health score in [0, 1].

    Returns:
        Frailty multiplier (1.0 = normal, 2.0 = max fragility).
    """
    return 1.0 / max(health_score, 0.5)


# ── Injury Probability ─────────────────────────────────────────────


def estimate_injury_probability(
    health_score: float,
    age: int | None = None,
    is_pitcher: bool = False,
    horizon_days: int = 30,
) -> float:
    """Estimate probability of injury within a time horizon.

    Uses health score + age risk as a heuristic proxy for Cox PH
    (no lifelines dependency required). The formula:

        p_injury = (1 - health_score) * age_risk_factor * (horizon / 162)

    where age_risk_factor amplifies probability for older players.

    Args:
        health_score: Player health score in [0, 1].
        age: Player age (None = no age adjustment).
        is_pitcher: Whether the player is a pitcher.
        horizon_days: Number of days to estimate probability over.

    Returns:
        Probability of injury in [0, 1].
    """
    # Base injury probability from health score
    base_prob = 1.0 - max(health_score, 0.0)

    # Age risk amplification
    age_mult = 1.0
    if age is not None:
        # age_risk_adjustment returns a 0.5-1.0 multiplier where lower = more risk
        # Invert it: 1.0/age_adj gives a >1.0 amplifier for older players
        age_adj = age_risk_adjustment(age, is_pitcher)
        age_mult = 1.0 / max(age_adj, 0.5)

    # Scale by time horizon (longer horizon = higher cumulative probability)
    horizon_scale = min(horizon_days / 162.0, 1.0)

    return np.clip(base_prob * age_mult * horizon_scale, 0.0, 0.95)


# ── Season Availability Sampling ───────────────────────────────────


def sample_season_availability(
    health_score: float,
    age: int | None = None,
    is_pitcher: bool = False,
    weeks_remaining: int = 16,
    rng: np.random.RandomState | None = None,
) -> float:
    """Sample a player's fraction of remaining season available.

    Monte Carlo-friendly function that combines injury probability,
    duration sampling, and games-missed calculation. Used in paired
    MC simulations to model availability uncertainty.

    The sampling process:
    1. Compute injury probability for remaining season
    2. Flip a coin: does an injury occur?
    3. If yes, sample duration from Weibull
    4. Compute fraction of remaining games missed

    Args:
        health_score: Player health score in [0, 1].
        age: Player age (None = no age adjustment).
        is_pitcher: Whether the player is a pitcher.
        weeks_remaining: Weeks left in the season.
        rng: Random state for reproducibility.

    Returns:
        Fraction of remaining season available, in [0, 1].
    """
    if rng is None:
        rng = np.random.RandomState()

    days_remaining = weeks_remaining * 7

    # Step 1: Compute injury probability over remaining season
    p_injury = estimate_injury_probability(
        health_score=health_score,
        age=age,
        is_pitcher=is_pitcher,
        horizon_days=days_remaining,
    )

    # Step 2: Does an injury occur?
    if rng.uniform() > p_injury:
        return 1.0  # No injury — fully available

    # Step 3: Sample injury duration
    frailty = frailty_from_health_score(health_score)
    duration = sample_injury_duration(
        injury_type="other",  # Generic injury type
        frailty=frailty,
        rng=rng,
    )

    # Step 4: Fraction of remaining days missed
    days_missed = min(duration, days_remaining)
    availability = max(0.0, 1.0 - days_missed / max(days_remaining, 1))

    return availability


def sample_availability_batch(
    health_scores: np.ndarray,
    ages: np.ndarray | None = None,
    is_pitcher_flags: np.ndarray | None = None,
    weeks_remaining: int = 16,
    n_sims: int = 1,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """Batch-sample availability for multiple players across multiple sims.

    Vectorized for performance in MC simulations.

    Args:
        health_scores: Array of health scores, shape (n_players,).
        ages: Array of ages (None = no age adjustment).
        is_pitcher_flags: Boolean array (None = all batters).
        weeks_remaining: Weeks left in season.
        n_sims: Number of simulations.
        rng: Random state for reproducibility.

    Returns:
        Array of shape (n_sims, n_players) with availability fractions.
    """
    if rng is None:
        rng = np.random.RandomState()

    n_players = len(health_scores)
    result = np.ones((n_sims, n_players))

    for sim in range(n_sims):
        for p in range(n_players):
            hs = health_scores[p]
            age = int(ages[p]) if ages is not None else None
            is_p = bool(is_pitcher_flags[p]) if is_pitcher_flags is not None else False

            result[sim, p] = sample_season_availability(
                health_score=hs,
                age=age,
                is_pitcher=is_p,
                weeks_remaining=weeks_remaining,
                rng=rng,
            )

    return result

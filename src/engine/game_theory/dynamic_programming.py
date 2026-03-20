"""Dynamic programming rollout for future trade option value.

Spec reference: Section 12 L9 (Dynamic Programming / Real Options)
               Section 17 Phase 5 item 23

Approximates the value of future trade options via Monte Carlo rollout.
The Bellman equation:

    V(trade) = immediate_surplus + γ × E[max(0, future_trade_surplus)]

The key insight: accepting a marginally negative trade NOW might open up
better trade opportunities LATER (positional flexibility, roster balance).
Conversely, a good trade today might lock you into a roster that has no
further improvement path. This module estimates that option value.

The discount factor γ depends on playoff probability:
  - Contending (>70%): γ=0.98 (future matters a lot)
  - Bubble (30-70%): γ=0.95 (moderate discounting)
  - Rebuilding (<30%): γ=0.85 (focus on now)

Wires into:
  - src/engine/output/trade_evaluator.py: option value overlay
  - src/engine/portfolio/category_analysis.py: for evaluating future trades
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Discount factors by competitive tier
GAMMA_CONTENDING: float = 0.98  # Playoff prob > 70%
GAMMA_BUBBLE: float = 0.95  # Playoff prob 30-70%
GAMMA_REBUILDING: float = 0.85  # Playoff prob < 30%

# Rollout parameters
DEFAULT_N_LOOKAHEAD: int = 2  # Weeks to look ahead
DEFAULT_N_ROLLOUT_SIMS: int = 200  # MC rollout sims (lighter than full MC)

# Plausible trade parameters
PLAUSIBLE_TRADE_PROB: float = 0.30  # Probability of a plausible trade per week
PLAUSIBLE_SURPLUS_MEAN: float = 0.3  # Mean surplus of plausible trades (SGP)
PLAUSIBLE_SURPLUS_STD: float = 0.8  # Std dev — some trades are much better/worse


def get_gamma(playoff_probability: float) -> float:
    """Get discount factor based on playoff probability.

    Spec ref: Section 12 L9 — discount factor selection.

    Contending teams value future flexibility highly (γ near 1.0).
    Rebuilding teams discount the future more aggressively.

    Args:
        playoff_probability: Estimated probability of making playoffs [0, 1].

    Returns:
        Discount factor γ in [0.85, 0.98].
    """
    if playoff_probability > 0.70:
        return GAMMA_CONTENDING
    if playoff_probability > 0.30:
        return GAMMA_BUBBLE
    return GAMMA_REBUILDING


def estimate_playoff_probability(
    standings_rank: int,
    num_teams: int = 12,
    weeks_remaining: int = 16,
) -> float:
    """Estimate playoff probability from standings position.

    Simple heuristic: top N/2 teams make playoffs. Probability
    decays linearly from rank 1 (near-certain) to rank N (near-zero),
    with more uncertainty when more weeks remain.

    Args:
        standings_rank: Current standings position (1 = first).
        num_teams: Number of teams in league.
        weeks_remaining: Weeks remaining in season.

    Returns:
        Estimated playoff probability in [0, 1].
    """
    playoff_spots = num_teams // 2  # Top half makes playoffs
    base_prob = max(0.0, 1.0 - (standings_rank - 1) / max(num_teams - 1, 1))

    # Boost teams at or above playoff cutoff, penalize those below
    if standings_rank <= playoff_spots:
        base_prob = min(1.0, base_prob + 0.1)
    else:
        base_prob = max(0.0, base_prob - 0.1)

    # More weeks remaining = more uncertainty = regress toward 0.5
    certainty = 1.0 - (weeks_remaining / 26.0)  # 26-week season
    certainty = max(0.0, min(1.0, certainty))

    return base_prob * certainty + 0.5 * (1.0 - certainty)


def sample_plausible_trade_surplus(
    rng: np.random.RandomState,
    roster_balance: float = 0.0,
) -> float | None:
    """Sample a plausible future trade surplus.

    Simulates whether a plausible trade arises and, if so, its surplus.
    Better-balanced rosters have slightly better future trade options
    (more flexibility to package players).

    Args:
        rng: Random state for reproducibility.
        roster_balance: Score in [-1, 1] indicating roster balance.
            Positive = well-balanced, negative = lopsided.

    Returns:
        Trade surplus in SGP if a trade occurs, None if no trade this week.
    """
    # Adjust trade probability by roster balance
    adjusted_prob = PLAUSIBLE_TRADE_PROB * (1.0 + 0.1 * roster_balance)
    adjusted_prob = max(0.05, min(0.50, adjusted_prob))

    if rng.uniform() > adjusted_prob:
        return None  # No trade opportunity this week

    # Sample surplus from a Normal distribution
    surplus = rng.normal(PLAUSIBLE_SURPLUS_MEAN, PLAUSIBLE_SURPLUS_STD)

    # Only positive surplus trades would be accepted
    return max(0.0, surplus)


def bellman_rollout(
    immediate_surplus: float,
    weeks_remaining: int = 16,
    playoff_probability: float = 0.50,
    roster_balance_before: float = 0.0,
    roster_balance_after: float = 0.0,
    n_lookahead: int = DEFAULT_N_LOOKAHEAD,
    n_sims: int = DEFAULT_N_ROLLOUT_SIMS,
    seed: int = 42,
) -> dict[str, float]:
    """Approximate Bellman value of a trade including future option value.

    Spec ref: Section 12 L9 — Bellman rollout.

    Computes:
        V = immediate_surplus + γ × (E[future_after] - E[future_before])

    The difference in expected future trade value captures the
    "option value" of the trade: does this trade open up or close off
    future improvements?

    Args:
        immediate_surplus: Phase 1 deterministic surplus (SGP).
        weeks_remaining: Weeks left in season.
        playoff_probability: Team's estimated playoff probability.
        roster_balance_before: Pre-trade roster balance score.
        roster_balance_after: Post-trade roster balance score.
        n_lookahead: Weeks to simulate ahead.
        n_sims: Number of rollout simulations.
        seed: Random seed.

    Returns:
        Dict with:
          - immediate: the raw surplus
          - future_before: expected future value without trade
          - future_after: expected future value with trade
          - option_value: future_after - future_before (can be negative)
          - gamma: discount factor used
          - total_value: immediate + discounted option value
    """
    gamma = get_gamma(playoff_probability)
    rng = np.random.RandomState(seed)

    # Simulate future trade opportunities for BOTH scenarios
    future_before_values: list[float] = []
    future_after_values: list[float] = []

    for _ in range(n_sims):
        # Generate a per-sim seed from the master RNG, then use
        # identical seeds for both scenarios (paired draws) so the
        # comparison isolates the causal effect of roster balance
        # rather than including random noise.
        sim_seed = int(rng.randint(0, 2**31))

        # Scenario 1: Don't make this trade (keep current roster)
        rng_before = np.random.RandomState(sim_seed)
        cum_before = 0.0
        for week in range(min(n_lookahead, weeks_remaining)):
            surplus = sample_plausible_trade_surplus(rng_before, roster_balance_before)
            if surplus is not None:
                cum_before += surplus * (gamma ** (week + 1))
        future_before_values.append(cum_before)

        # Scenario 2: Make this trade (post-trade roster)
        rng_after = np.random.RandomState(sim_seed)
        cum_after = 0.0
        for week in range(min(n_lookahead, weeks_remaining)):
            surplus = sample_plausible_trade_surplus(rng_after, roster_balance_after)
            if surplus is not None:
                cum_after += surplus * (gamma ** (week + 1))
        future_after_values.append(cum_after)

    future_before = float(np.mean(future_before_values))
    future_after = float(np.mean(future_after_values))
    option_value = future_after - future_before

    total_value = immediate_surplus + gamma * option_value

    return {
        "immediate": round(immediate_surplus, 3),
        "future_before": round(future_before, 3),
        "future_after": round(future_after, 3),
        "option_value": round(option_value, 3),
        "gamma": gamma,
        "total_value": round(total_value, 3),
    }


def compute_roster_balance(
    roster_category_ranks: dict[str, int],
    num_teams: int = 12,
) -> float:
    """Score how balanced a roster is across categories.

    A balanced roster (all ranks near median) has more trade flexibility
    than a lopsided one (dominant in some, terrible in others).

    Scoring: average z-score of ranks, inverted so positive = balanced.

    Args:
        roster_category_ranks: Dict of {category: rank} (1=best, 12=worst).
        num_teams: Total teams in league.

    Returns:
        Balance score in [-1, 1]. Positive = balanced, negative = lopsided.
    """
    if not roster_category_ranks:
        return 0.0

    median_rank = (num_teams + 1) / 2.0
    deviations = [abs(rank - median_rank) / median_rank for rank in roster_category_ranks.values()]

    # Lower average deviation = more balanced
    avg_deviation = sum(deviations) / len(deviations)

    # Invert and normalize to [-1, 1]
    # deviation of 0 → score of 1.0 (perfectly balanced)
    # deviation of 1 → score of -1.0 (maximally lopsided)
    return round(max(-1.0, min(1.0, 1.0 - 2.0 * avg_deviation)), 3)

"""Adverse selection discount for trade evaluation.

Spec reference: Section 11 L8B (Game Theory Layer)
               Section 17 Phase 5 item 22

Implements Bayesian updating of the probability that an offered player
has a hidden flaw, calibrated from the offering manager's trade history.

The key insight: when someone offers you a player, that's informative.
If they know something you don't (injury brewing, role change, platoon
incoming), the offered player is likely overvalued. This is the classic
"lemons problem" from Akerlof 1970.

The model:
  P(flaw|offered) = P(offered|flaw) × P(flaw) / P(offered)

Where:
  P(flaw) = base rate of players having hidden flaws (~15%)
  P(offered|flaw) = probability a flawed player gets offered (60%)
  P(offered|ok) = probability a non-flawed player gets offered (20%)

Calibration: When the offering manager has trade history, we estimate
P(flaw) from their track record of trades that underperformed projections.

Wires into:
  - src/engine/output/trade_evaluator.py: adverse selection discount on received players
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Default prior probability of a hidden flaw in offered players
DEFAULT_P_FLAW: float = 0.15

# Likelihood of being offered given flaw status
P_OFFERED_GIVEN_FLAW: float = 0.60  # Flawed players are more likely to be offered
P_OFFERED_GIVEN_OK: float = 0.20  # Non-flawed players are sometimes offered too

# Performance threshold: "underperformed" if actual < projected * this factor
UNDERPERFORMANCE_THRESHOLD: float = 0.80

# Maximum discount factor (minimum multiplier — don't reduce value below 75%)
MAX_DISCOUNT: float = 0.25

# Minimum trade history to override prior
MIN_HISTORY_FOR_CALIBRATION: int = 3


def adverse_selection_discount(
    offering_manager_history: list[dict] | None = None,
    p_flaw_prior: float = DEFAULT_P_FLAW,
) -> float:
    """Compute the adverse selection discount multiplier.

    Spec ref: Section 11 L8B — adverse selection discount.

    Uses Bayes' theorem to compute P(flaw | offered), then converts
    to a multiplicative discount factor. A discount of 0.95 means
    "reduce received player value by 5%."

    The offering manager's track record calibrates the prior:
    - Manager with clean history → P(flaw) drops → small discount
    - Manager who frequently dumps underperformers → P(flaw) rises → big discount

    Args:
        offering_manager_history: List of past trade outcomes from this manager.
            Each dict should have 'actual' and 'projected' keys (season stat totals
            or SGP values). None = use default prior.
        p_flaw_prior: Base probability of flaw (overridden by history when available).

    Returns:
        Multiplicative discount factor in (0.75, 1.0]. Values below 1.0
        indicate suspicion of adverse selection.
    """
    p_flaw = p_flaw_prior

    # Calibrate from manager history if available
    if offering_manager_history is not None and len(offering_manager_history) >= MIN_HISTORY_FOR_CALIBRATION:
        underperformers = sum(1 for trade in offering_manager_history if _trade_underperformed(trade))
        p_flaw = underperformers / len(offering_manager_history)

    # Bayes' theorem: P(flaw|offered)
    p_offered = P_OFFERED_GIVEN_FLAW * p_flaw + P_OFFERED_GIVEN_OK * (1.0 - p_flaw)

    if p_offered <= 0:
        return 1.0

    p_flaw_given_offered = (P_OFFERED_GIVEN_FLAW * p_flaw) / p_offered

    # Convert to discount: reduce value by p_flaw_given_offered * MAX_DISCOUNT
    discount = 1.0 - (p_flaw_given_offered * MAX_DISCOUNT)

    return round(max(discount, 1.0 - MAX_DISCOUNT), 4)


def _trade_underperformed(trade: dict) -> bool:
    """Check if a trade outcome underperformed projections.

    Args:
        trade: Dict with 'actual' and 'projected' keys.

    Returns:
        True if actual performance was below threshold of projected.
    """
    actual = trade.get("actual", 0)
    projected = trade.get("projected", 0)

    if projected <= 0:
        return False

    return actual < projected * UNDERPERFORMANCE_THRESHOLD


def compute_discount_for_trade(
    receiving_player_count: int = 1,
    offering_manager_history: list[dict] | None = None,
) -> dict[str, float]:
    """Compute adverse selection analysis for a trade.

    Returns a breakdown of the adverse selection risk.

    Args:
        receiving_player_count: Number of players being received.
        offering_manager_history: Manager's trade history.

    Returns:
        Dict with:
          - discount_factor: multiplicative factor (0.75-1.0)
          - p_flaw: estimated probability of hidden flaw
          - p_flaw_given_offered: posterior probability after Bayesian update
          - sgp_adjustment: per-player SGP adjustment (negative)
          - total_sgp_adjustment: total across all received players
          - risk_level: "low", "medium", or "high"
    """
    p_flaw = DEFAULT_P_FLAW

    if offering_manager_history is not None and len(offering_manager_history) >= MIN_HISTORY_FOR_CALIBRATION:
        underperformers = sum(1 for trade in offering_manager_history if _trade_underperformed(trade))
        p_flaw = underperformers / len(offering_manager_history)

    # Compute posterior
    p_offered = P_OFFERED_GIVEN_FLAW * p_flaw + P_OFFERED_GIVEN_OK * (1.0 - p_flaw)
    p_flaw_given_offered = (P_OFFERED_GIVEN_FLAW * p_flaw) / p_offered if p_offered > 0 else 0.0

    discount = adverse_selection_discount(
        offering_manager_history=offering_manager_history,
        p_flaw_prior=DEFAULT_P_FLAW,
    )

    # Per-player SGP adjustment (how much to penalize each received player)
    # Typical player is ~2-3 SGP, so discount of 0.95 = ~0.1-0.15 SGP reduction
    per_player_adj = -(1.0 - discount) * 2.5  # Assume ~2.5 SGP average player
    total_adj = per_player_adj * receiving_player_count

    # Risk level classification
    if p_flaw_given_offered > 0.40:
        risk_level = "high"
    elif p_flaw_given_offered > 0.25:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "discount_factor": discount,
        "p_flaw": round(p_flaw, 4),
        "p_flaw_given_offered": round(p_flaw_given_offered, 4),
        "sgp_adjustment": round(per_player_adj, 3),
        "total_sgp_adjustment": round(total_adj, 3),
        "risk_level": risk_level,
    }

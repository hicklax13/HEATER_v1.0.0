"""Simulation scaling configuration.

Spec reference: Section 17 Phase 6 item 25

Provides adaptive simulation count based on trade complexity and
available compute time. Scales from 10K (quick preview) to 100K
(full production) based on:
  - Number of players in trade (more players = more variance = need more sims)
  - Available time budget (Streamlit needs to respond within seconds)
  - Convergence feedback (increase sims if ESS is low)

The key insight: not all trades need 100K sims. A simple 1-for-1 trade
between two similar players converges quickly at 10K. A complex 3-for-2
trade with a mix of hitters/pitchers needs more sims to resolve the
distributional tails.

Wires into:
  - src/engine/output/trade_evaluator.py: adaptive n_sims selection
  - src/engine/monte_carlo/trade_simulator.py: runs with selected n_sims
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Simulation count tiers
SIM_QUICK: int = 1_000  # Quick preview (sub-second)
SIM_STANDARD: int = 10_000  # Standard evaluation (1-2 seconds)
SIM_PRODUCTION: int = 50_000  # High-confidence (3-5 seconds)
SIM_FULL: int = 100_000  # Full production (5-10 seconds)

# Complexity scaling factors
BASE_SIMS: int = 10_000
SIMS_PER_EXTRA_PLAYER: int = 5_000  # Each extra player adds variance
MAX_SIMS: int = 100_000

# Time budget defaults (seconds)
TIME_BUDGET_INTERACTIVE: float = 3.0  # Streamlit responsive
TIME_BUDGET_BATCH: float = 30.0  # Background processing
ESTIMATED_SIMS_PER_SECOND: int = 15_000  # Rough estimate on typical hardware


def compute_adaptive_n_sims(
    n_giving: int = 1,
    n_receiving: int = 1,
    mode: str = "standard",
    time_budget_s: float | None = None,
) -> int:
    """Compute adaptive simulation count based on trade complexity.

    More complex trades (more players, mixed hitter/pitcher) need
    more simulations to capture the variance structure.

    Args:
        n_giving: Number of players being given.
        n_receiving: Number of players being received.
        mode: Simulation mode:
          - "quick": fast preview (1K sims)
          - "standard": normal evaluation (10K base)
          - "production": high confidence (50K base)
          - "full": maximum precision (100K)
        time_budget_s: Maximum time in seconds. If provided, caps
            n_sims to fit within this budget.

    Returns:
        Recommended number of simulations.
    """
    if mode == "quick":
        return SIM_QUICK
    if mode == "full":
        return SIM_FULL

    base = SIM_PRODUCTION if mode == "production" else SIM_STANDARD

    # Scale by trade complexity
    total_players = n_giving + n_receiving
    extra_players = max(0, total_players - 2)  # 1-for-1 is baseline
    complexity_sims = base + (extra_players * SIMS_PER_EXTRA_PLAYER)

    n_sims = min(complexity_sims, MAX_SIMS)

    # Apply time budget cap
    if time_budget_s is not None:
        max_by_time = int(time_budget_s * ESTIMATED_SIMS_PER_SECOND)
        n_sims = min(n_sims, max_by_time)

    return max(SIM_QUICK, n_sims)


def get_sim_mode(interactive: bool = True) -> str:
    """Determine simulation mode based on execution context.

    Args:
        interactive: True if running in interactive Streamlit context.

    Returns:
        Mode string: "standard" for interactive, "production" for batch.
    """
    return "standard" if interactive else "production"


def estimate_runtime_seconds(n_sims: int) -> float:
    """Estimate how long a simulation will take.

    Args:
        n_sims: Number of simulations.

    Returns:
        Estimated runtime in seconds.
    """
    return n_sims / ESTIMATED_SIMS_PER_SECOND


def sim_config_summary(
    n_giving: int = 1,
    n_receiving: int = 1,
    mode: str = "standard",
    time_budget_s: float | None = None,
) -> dict[str, int | float | str]:
    """Get a full simulation configuration summary.

    Args:
        n_giving: Players being given.
        n_receiving: Players being received.
        mode: Simulation mode.
        time_budget_s: Time budget.

    Returns:
        Dict with:
          - n_sims: recommended simulation count
          - mode: selected mode
          - estimated_runtime_s: estimated time
          - complexity: trade complexity score
          - capped_by_time: whether time budget capped the count
    """
    n_sims = compute_adaptive_n_sims(
        n_giving=n_giving,
        n_receiving=n_receiving,
        mode=mode,
        time_budget_s=time_budget_s,
    )

    total_players = n_giving + n_receiving
    complexity = total_players - 2 + (1 if total_players > 3 else 0)

    # Check if time budget was the binding constraint
    uncapped_sims = compute_adaptive_n_sims(
        n_giving=n_giving,
        n_receiving=n_receiving,
        mode=mode,
        time_budget_s=None,
    )
    capped = n_sims < uncapped_sims

    return {
        "n_sims": n_sims,
        "mode": mode,
        "estimated_runtime_s": round(estimate_runtime_seconds(n_sims), 2),
        "complexity": complexity,
        "capped_by_time": capped,
    }

"""Historical backtesting validator for the Lineup Optimizer.

Compares optimizer predictions against actual MLB outcomes using
standard accuracy metrics: RMSE, rank correlation, bust rate,
and lineup quality grading.
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)


def compute_projection_rmse(
    projected: dict[str, float],
    actual: dict[str, float],
) -> float:
    """Compute RMSE between projected and actual category totals.

    Args:
        projected: Dict of category -> projected value.
        actual: Dict of category -> actual value.

    Returns:
        Root Mean Squared Error across all shared categories.
    """
    shared = set(projected) & set(actual)
    if not shared:
        return float("inf")
    squared_errors = [(projected[c] - actual[c]) ** 2 for c in shared]
    return math.sqrt(sum(squared_errors) / len(squared_errors))


def compute_rank_correlation(
    projected_order: list[int],
    actual_order: list[int],
) -> float:
    """Compute Spearman rank correlation between two player rankings.

    Args:
        projected_order: Player IDs ranked by projected value (best first).
        actual_order: Player IDs ranked by actual value (best first).

    Returns:
        Spearman rho in [-1, 1].  1.0 = perfect agreement.
    """
    n = len(projected_order)
    if n < 2:
        return 0.0

    actual_rank = {pid: i for i, pid in enumerate(actual_order)}
    d_squared = 0.0
    matched = 0
    for i, pid in enumerate(projected_order):
        if pid in actual_rank:
            d_squared += (i - actual_rank[pid]) ** 2
            matched += 1

    if matched < 2:
        return 0.0

    rho = 1 - (6 * d_squared) / (matched * (matched**2 - 1))
    return max(-1.0, min(1.0, rho))


def compute_bust_rate(
    projected: dict[int, float],
    actual: dict[int, float],
    threshold: float = 0.50,
) -> float:
    """Compute bust rate: fraction of players whose actual < threshold * projected.

    A "bust" is a player who delivers less than the threshold fraction
    of their projected value (e.g., < 50% of projected HR).

    Args:
        projected: Dict of player_id -> projected value.
        actual: Dict of player_id -> actual value.
        threshold: Fraction below which a player is a bust.

    Returns:
        Bust rate in [0.0, 1.0].
    """
    shared = set(projected) & set(actual)
    if not shared:
        return 0.0

    eligible = 0
    busts = 0
    for pid in shared:
        if projected[pid] <= 0:
            continue
        eligible += 1
        if actual[pid] < threshold * projected[pid]:
            busts += 1

    if eligible == 0:
        return 0.0
    return busts / eligible


def grade_lineup_quality(
    optimizer_value: float,
    optimal_value: float,
    threshold_a: float = 0.90,
    threshold_b: float = 0.80,
) -> str:
    """Grade optimizer lineup quality against actual-optimal.

    Compares total fantasy value of optimizer's recommended lineup
    against the best possible lineup (computed with hindsight).

    Args:
        optimizer_value: Total value from optimizer's recommended lineup.
        optimal_value: Total value from hindsight-optimal lineup.
        threshold_a: Fraction of optimal needed for 'A' grade.
        threshold_b: Fraction of optimal needed for 'B' grade.

    Returns:
        Letter grade: 'A', 'B', or 'C'.
    """
    if optimal_value <= 0:
        return "A"

    ratio = optimizer_value / optimal_value
    if ratio >= threshold_a:
        return "A"
    if ratio >= threshold_b:
        return "B"
    return "C"

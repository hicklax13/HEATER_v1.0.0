"""Sigmoid K-value calibration for category urgency scoring.

Grid-searches over COUNTING_STAT_K and RATE_STAT_K to find the values
that best identify actionable H2H categories.  Uses synthetic matchup
scenarios so no network calls or database access are required.

The calibration loop patches the module-level constants in
``category_urgency`` via ``unittest.mock.patch``, computes urgency
weights for each scenario, and scores how well the urgency correctly
identifies categories the team is losing (and should target).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from unittest.mock import patch

logger = logging.getLogger(__name__)


# ── Data Classes ──────────────────────────────────────────────────────


@dataclass
class CalibrationResult:
    """Result of a sigmoid k-value calibration run."""

    counting_k: float
    rate_k: float
    avg_win_rate: float
    avg_rank_correlation: float
    avg_rmse: float
    n_weeks_tested: int


# ── H2H Simulation ───────────────────────────────────────────────────


def simulate_h2h_week(
    my_totals: dict[str, float],
    opp_totals: dict[str, float],
) -> tuple[int, int, int]:
    """Simulate one H2H matchup week across 12 categories.

    For normal stats (R, HR, RBI, SB, AVG, OBP, W, SV, K):
        I win if my value > opponent's value.
    For inverse stats (L, ERA, WHIP):
        I win if my value < opponent's value (lower is better).

    Args:
        my_totals: My team's projected category totals.
        opp_totals: Opponent's projected category totals.

    Returns:
        Tuple of (win_count, loss_count, tie_count).
    """
    # Lazy import to avoid circular dependency
    from src.valuation import LeagueConfig

    config = LeagueConfig()
    inverse_stats = config.inverse_stats  # {L, ERA, WHIP}

    wins = 0
    losses = 0
    ties = 0

    for cat in config.all_categories:
        my_val = float(my_totals.get(cat, my_totals.get(cat.lower(), 0)) or 0)
        opp_val = float(opp_totals.get(cat, opp_totals.get(cat.lower(), 0)) or 0)

        if cat in inverse_stats:
            # Lower is better: I win when my value < opponent's
            if my_val < opp_val:
                wins += 1
            elif my_val > opp_val:
                losses += 1
            else:
                ties += 1
        else:
            # Higher is better: I win when my value > opponent's
            if my_val > opp_val:
                wins += 1
            elif my_val < opp_val:
                losses += 1
            else:
                ties += 1

    return wins, losses, ties


# ── Urgency Scoring ──────────────────────────────────────────────────


def score_urgency_weights(
    urgency: dict[str, float],
    my_totals: dict[str, float],
    opp_totals: dict[str, float],
) -> float:
    """Score how well urgency weights identify actionable categories.

    A good urgency function assigns high urgency (> 0.6) to categories
    the team is losing and low urgency (< 0.4) to categories the team
    is winning comfortably.  Categories near a tie should have urgency
    close to 0.5.

    Args:
        urgency: Dict of category -> urgency weight (0.0 to 1.0).
        my_totals: My team's category totals.
        opp_totals: Opponent's category totals.

    Returns:
        Score from 0.0 to 1.0 (higher = better urgency calibration).
    """
    from src.valuation import LeagueConfig

    config = LeagueConfig()
    inverse_stats = config.inverse_stats

    if not urgency:
        return 0.0

    correct = 0
    total = 0

    for cat in config.all_categories:
        if cat not in urgency:
            continue

        my_val = float(my_totals.get(cat, my_totals.get(cat.lower(), 0)) or 0)
        opp_val = float(opp_totals.get(cat, opp_totals.get(cat.lower(), 0)) or 0)
        u = urgency[cat]

        # Determine if I'm winning, losing, or tied in this category
        if cat in inverse_stats:
            gap = my_val - opp_val  # Positive = I'm losing
        else:
            gap = opp_val - my_val  # Positive = I'm losing

        total += 1

        # Score: high urgency when losing = correct; low urgency when winning = correct
        if gap > 0 and u > 0.6:
            correct += 1  # Correctly flagged losing category as urgent
        elif gap < 0 and u < 0.4:
            correct += 1  # Correctly flagged winning category as non-urgent
        elif abs(gap) < 1e-9 and 0.4 <= u <= 0.6:
            correct += 1  # Correctly identified tied category as neutral

    return correct / total if total > 0 else 0.0


# ── Synthetic Scenarios ──────────────────────────────────────────────


def _build_scenarios(n_scenarios: int = 5) -> list[tuple[dict[str, float], dict[str, float], str]]:
    """Build synthetic H2H matchup scenarios for calibration.

    Each scenario is a (my_totals, opp_totals, label) triple.
    Scenarios cover the range of typical H2H outcomes without
    requiring any network calls.

    Args:
        n_scenarios: Number of scenarios (3 for quick, 5 for full).

    Returns:
        List of (my_totals, opp_totals, label) tuples.
    """
    scenarios: list[tuple[dict[str, float], dict[str, float], str]] = []

    # Scenario 1: Blowout win -- winning all counting stats, competitive rates
    my_1 = {
        "R": 45,
        "HR": 12,
        "RBI": 40,
        "SB": 8,
        "AVG": 0.275,
        "OBP": 0.345,
        "W": 6,
        "L": 2,
        "SV": 4,
        "K": 55,
        "ERA": 2.80,
        "WHIP": 1.05,
    }
    opp_1 = {
        "R": 28,
        "HR": 5,
        "RBI": 22,
        "SB": 3,
        "AVG": 0.240,
        "OBP": 0.305,
        "W": 3,
        "L": 5,
        "SV": 1,
        "K": 35,
        "ERA": 4.50,
        "WHIP": 1.38,
    }
    scenarios.append((my_1, opp_1, "blowout_win"))

    # Scenario 2: Close win -- slight edges in most categories
    my_2 = {
        "R": 35,
        "HR": 8,
        "RBI": 32,
        "SB": 5,
        "AVG": 0.268,
        "OBP": 0.338,
        "W": 5,
        "L": 3,
        "SV": 3,
        "K": 48,
        "ERA": 3.20,
        "WHIP": 1.12,
    }
    opp_2 = {
        "R": 32,
        "HR": 7,
        "RBI": 30,
        "SB": 4,
        "AVG": 0.262,
        "OBP": 0.332,
        "W": 4,
        "L": 4,
        "SV": 2,
        "K": 45,
        "ERA": 3.50,
        "WHIP": 1.18,
    }
    scenarios.append((my_2, opp_2, "close_win"))

    # Scenario 3: Tied -- nearly identical totals
    my_3 = {
        "R": 33,
        "HR": 7,
        "RBI": 30,
        "SB": 4,
        "AVG": 0.265,
        "OBP": 0.335,
        "W": 4,
        "L": 3,
        "SV": 2,
        "K": 42,
        "ERA": 3.40,
        "WHIP": 1.15,
    }
    opp_3 = {
        "R": 33,
        "HR": 7,
        "RBI": 30,
        "SB": 4,
        "AVG": 0.265,
        "OBP": 0.335,
        "W": 4,
        "L": 3,
        "SV": 2,
        "K": 42,
        "ERA": 3.40,
        "WHIP": 1.15,
    }
    scenarios.append((my_3, opp_3, "tied"))

    if n_scenarios <= 3:
        return scenarios

    # Scenario 4: Close loss -- opponent has slight edges
    my_4 = {
        "R": 30,
        "HR": 6,
        "RBI": 28,
        "SB": 3,
        "AVG": 0.255,
        "OBP": 0.322,
        "W": 3,
        "L": 4,
        "SV": 1,
        "K": 40,
        "ERA": 3.80,
        "WHIP": 1.22,
    }
    opp_4 = {
        "R": 34,
        "HR": 9,
        "RBI": 35,
        "SB": 6,
        "AVG": 0.272,
        "OBP": 0.345,
        "W": 5,
        "L": 3,
        "SV": 3,
        "K": 50,
        "ERA": 3.10,
        "WHIP": 1.08,
    }
    scenarios.append((my_4, opp_4, "close_loss"))

    # Scenario 5: Blowout loss -- losing most categories badly
    my_5 = {
        "R": 22,
        "HR": 3,
        "RBI": 18,
        "SB": 1,
        "AVG": 0.232,
        "OBP": 0.298,
        "W": 2,
        "L": 5,
        "SV": 0,
        "K": 30,
        "ERA": 4.80,
        "WHIP": 1.42,
    }
    opp_5 = {
        "R": 48,
        "HR": 14,
        "RBI": 45,
        "SB": 9,
        "AVG": 0.285,
        "OBP": 0.360,
        "W": 7,
        "L": 2,
        "SV": 5,
        "K": 60,
        "ERA": 2.50,
        "WHIP": 0.98,
    }
    scenarios.append((my_5, opp_5, "blowout_loss"))

    return scenarios


# ── Grid Search Calibration ──────────────────────────────────────────


def calibrate_sigmoid_k(
    *,
    counting_k_grid: list[float] | None = None,
    rate_k_grid: list[float] | None = None,
    n_scenarios: int = 5,
) -> CalibrationResult:
    """Grid-search over (counting_k, rate_k) to maximize urgency quality.

    For each (counting_k, rate_k) pair, patches the module-level constants,
    computes urgency for each synthetic scenario, and scores correctness.

    Args:
        counting_k_grid: Values to test for COUNTING_STAT_K.
            Defaults to [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0].
        rate_k_grid: Values to test for RATE_STAT_K.
            Defaults to [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0].
        n_scenarios: Number of synthetic scenarios (3 for quick, 5 for full).

    Returns:
        CalibrationResult with the best-performing k-value pair.
    """
    from src.optimizer.category_urgency import compute_category_urgency

    if counting_k_grid is None:
        counting_k_grid = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    if rate_k_grid is None:
        rate_k_grid = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

    scenarios = _build_scenarios(n_scenarios)

    best_score = -1.0
    best_counting_k = 2.0
    best_rate_k = 3.0
    best_win_rate = 0.0
    best_rmse = float("inf")

    for ck in counting_k_grid:
        for rk in rate_k_grid:
            total_urgency_score = 0.0
            total_win_rate = 0.0
            total_rmse = 0.0

            with (
                patch("src.optimizer.category_urgency.COUNTING_STAT_K", ck),
                patch("src.optimizer.category_urgency.RATE_STAT_K", rk),
            ):
                for my_totals, opp_totals, _label in scenarios:
                    # Compute urgency with patched k-values
                    urgency = compute_category_urgency(my_totals, opp_totals)

                    # Score urgency quality
                    u_score = score_urgency_weights(urgency, my_totals, opp_totals)
                    total_urgency_score += u_score

                    # Simulate H2H outcome
                    wins, losses, ties = simulate_h2h_week(my_totals, opp_totals)
                    total_cats = wins + losses + ties
                    win_rate = wins / total_cats if total_cats > 0 else 0.0
                    total_win_rate += win_rate

                    # RMSE: measure how far urgency is from ideal (1.0 for
                    # losing cats, 0.0 for winning cats)
                    rmse_sum = 0.0
                    rmse_n = 0
                    for cat, u in urgency.items():
                        cat_upper = cat.upper() if cat.upper() in my_totals else cat
                        my_v = my_totals.get(cat_upper, my_totals.get(cat, 0))
                        opp_v = opp_totals.get(cat_upper, opp_totals.get(cat, 0))

                        from src.valuation import LeagueConfig

                        cfg = LeagueConfig()
                        if cat_upper in cfg.inverse_stats:
                            losing = float(my_v or 0) > float(opp_v or 0)
                        else:
                            losing = float(opp_v or 0) > float(my_v or 0)

                        ideal = 1.0 if losing else 0.0
                        if abs(float(my_v or 0) - float(opp_v or 0)) < 1e-9:
                            ideal = 0.5

                        rmse_sum += (u - ideal) ** 2
                        rmse_n += 1

                    if rmse_n > 0:
                        total_rmse += math.sqrt(rmse_sum / rmse_n)

            n = len(scenarios)
            avg_score = total_urgency_score / n if n > 0 else 0.0
            avg_win = total_win_rate / n if n > 0 else 0.0
            avg_rmse = total_rmse / n if n > 0 else 0.0

            # Primary ranking: urgency score (higher = better);
            # secondary: lower RMSE
            if avg_score > best_score or (avg_score == best_score and avg_rmse < best_rmse):
                best_score = avg_score
                best_counting_k = ck
                best_rate_k = rk
                best_win_rate = avg_win
                best_rmse = avg_rmse

    logger.info(
        "Calibration complete: counting_k=%.2f, rate_k=%.2f, score=%.3f",
        best_counting_k,
        best_rate_k,
        best_score,
    )

    return CalibrationResult(
        counting_k=best_counting_k,
        rate_k=best_rate_k,
        avg_win_rate=best_win_rate,
        avg_rank_correlation=best_score,  # urgency-correctness score
        avg_rmse=best_rmse,
        n_weeks_tested=len(scenarios),
    )


# ── Human-Readable Recommendation ────────────────────────────────────


def recommend_k_values(
    *,
    counting_k_grid: list[float] | None = None,
    rate_k_grid: list[float] | None = None,
    n_scenarios: int = 5,
) -> str:
    """Run calibration and return a human-readable recommendation.

    Compares the best-found k-values against the current defaults
    (COUNTING_STAT_K=2.0, RATE_STAT_K=3.0) and reports the percentage
    change.

    Args:
        counting_k_grid: Counting-stat k-values to test.
        rate_k_grid: Rate-stat k-values to test.
        n_scenarios: Number of synthetic scenarios.

    Returns:
        Multi-line summary string.
    """
    from src.optimizer.category_urgency import COUNTING_STAT_K, RATE_STAT_K

    result = calibrate_sigmoid_k(
        counting_k_grid=counting_k_grid,
        rate_k_grid=rate_k_grid,
        n_scenarios=n_scenarios,
    )

    current_ck = COUNTING_STAT_K
    current_rk = RATE_STAT_K

    ck_change = ((result.counting_k - current_ck) / current_ck) * 100
    rk_change = ((result.rate_k - current_rk) / current_rk) * 100

    lines = [
        "=" * 60,
        "  SIGMOID K-VALUE CALIBRATION RESULTS",
        "=" * 60,
        "",
        f"  Scenarios tested:     {result.n_weeks_tested}",
        f"  Urgency score:        {result.avg_rank_correlation:.3f}",
        f"  Urgency RMSE:         {result.avg_rmse:.3f}",
        f"  Avg H2H win rate:     {result.avg_win_rate:.3f}",
        "",
        "  COUNTING_STAT_K:",
        f"    Current:  {current_ck:.2f}",
        f"    Best:     {result.counting_k:.2f}  ({ck_change:+.1f}%)",
        "",
        "  RATE_STAT_K:",
        f"    Current:  {current_rk:.2f}",
        f"    Best:     {result.rate_k:.2f}  ({rk_change:+.1f}%)",
        "",
    ]

    if abs(ck_change) < 1e-9 and abs(rk_change) < 1e-9:
        lines.append("  Recommendation: Current defaults are optimal.")
    else:
        lines.append("  Recommendation: Update constants in category_urgency.py:")
        if abs(ck_change) >= 1e-9:
            lines.append(f"    COUNTING_STAT_K = {result.counting_k:.1f}")
        if abs(rk_change) >= 1e-9:
            lines.append(f"    RATE_STAT_K = {result.rate_k:.1f}")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)

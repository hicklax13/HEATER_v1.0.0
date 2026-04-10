"""Calibrate sigmoid K-values for the category urgency module.

Grid-searches over COUNTING_STAT_K and RATE_STAT_K to find values
that best identify actionable H2H categories.  Uses synthetic matchup
data -- no network calls required.

Usage:
    python scripts/calibrate_sigmoid.py               # Full 9x8 grid, 5 scenarios
    python scripts/calibrate_sigmoid.py --quick        # Quick 5x5 grid, 3 scenarios
    python scripts/calibrate_sigmoid.py --verbose      # Detailed logging
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.optimizer.sigmoid_calibrator import (  # noqa: E402
    calibrate_sigmoid_k,
    recommend_k_values,
)

# ── Grid definitions ─────────────────────────────────────────────────

QUICK_COUNTING_K = [1.0, 2.0, 3.0, 4.0, 5.0]
QUICK_RATE_K = [1.5, 2.5, 3.0, 4.0, 5.0]

FULL_COUNTING_K = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
FULL_RATE_K = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate sigmoid K-values for category urgency",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 5x5 grid (25 combos), 3 scenarios",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.quick:
        counting_grid = QUICK_COUNTING_K
        rate_grid = QUICK_RATE_K
        n_scenarios = 3
        n_combos = len(counting_grid) * len(rate_grid)
        print(f"Quick mode: {n_combos} combinations, {n_scenarios} scenarios")
    else:
        counting_grid = FULL_COUNTING_K
        rate_grid = FULL_RATE_K
        n_scenarios = 5
        n_combos = len(counting_grid) * len(rate_grid)
        print(f"Full mode: {n_combos} combinations, {n_scenarios} scenarios")

    print()

    # Run calibration
    result = calibrate_sigmoid_k(
        counting_k_grid=counting_grid,
        rate_k_grid=rate_grid,
        n_scenarios=n_scenarios,
    )

    # Print grid results table
    print("-" * 60)
    print("  BEST RESULT")
    print("-" * 60)
    print(f"  counting_k:       {result.counting_k:.2f}")
    print(f"  rate_k:           {result.rate_k:.2f}")
    print(f"  urgency score:    {result.avg_rank_correlation:.3f}")
    print(f"  urgency RMSE:     {result.avg_rmse:.3f}")
    print(f"  avg win rate:     {result.avg_win_rate:.3f}")
    print(f"  scenarios tested: {result.n_weeks_tested}")
    print()

    # Print full recommendation
    recommendation = recommend_k_values(
        counting_k_grid=counting_grid,
        rate_k_grid=rate_grid,
        n_scenarios=n_scenarios,
    )
    print(recommendation)


if __name__ == "__main__":
    main()

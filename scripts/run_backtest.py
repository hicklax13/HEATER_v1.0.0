"""Run historical backtest of the optimizer across 10 weeks of 2025 data.

Replays past MLB weeks through the optimizer, comparing recommended
lineup performance to actual player stats. Grades accuracy with RMSE,
Spearman rank correlation, bust rate, and lineup quality.

Usage:
    python scripts/run_backtest.py              # All 10 weeks
    python scripts/run_backtest.py --weeks 3    # First 3 weeks only
    python scripts/run_backtest.py --quick      # Quick mode (fewer players)
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.optimizer.backtest_runner import (  # noqa: E402
    BACKTEST_PLAYER_IDS,
    format_report,
    run_backtest,
)

# ── 10 weeks from the 2025 MLB season (Monday-Sunday) ────────────────

WEEKS: list[tuple[date, date]] = [
    (date(2025, 4, 7), date(2025, 4, 13)),
    (date(2025, 4, 14), date(2025, 4, 20)),
    (date(2025, 4, 21), date(2025, 4, 27)),
    (date(2025, 4, 28), date(2025, 5, 4)),
    (date(2025, 5, 5), date(2025, 5, 11)),
    (date(2025, 5, 12), date(2025, 5, 18)),
    (date(2025, 5, 19), date(2025, 5, 25)),
    (date(2025, 5, 26), date(2025, 6, 1)),
    (date(2025, 6, 2), date(2025, 6, 8)),
    (date(2025, 6, 9), date(2025, 6, 15)),
]


def _build_synthetic_roster() -> pd.DataFrame:
    """Build a synthetic roster from BACKTEST_PLAYER_IDS with season projections.

    Uses rough 2025-caliber projections for each player so the backtest
    has projected stats to compare against actuals. In production, this
    would load projections from the database.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(2025)
    rows = []

    hitter_ids = list(BACKTEST_PLAYER_IDS.keys())[:10]
    pitcher_ids = list(BACKTEST_PLAYER_IDS.keys())[10:]

    for pid in hitter_ids:
        rows.append(
            {
                "player_id": pid,
                "mlb_id": pid,
                "name": BACKTEST_PLAYER_IDS[pid],
                "positions": "OF",
                "is_hitter": 1,
                "r": int(rng.integers(70, 115)),
                "hr": int(rng.integers(20, 45)),
                "rbi": int(rng.integers(60, 120)),
                "sb": int(rng.integers(5, 35)),
                "avg": round(float(rng.uniform(0.260, 0.310)), 3),
                "obp": round(float(rng.uniform(0.330, 0.400)), 3),
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "era": 0.0,
                "whip": 0.0,
                "ip": 0.0,
            }
        )

    for pid in pitcher_ids:
        rows.append(
            {
                "player_id": pid,
                "mlb_id": pid,
                "name": BACKTEST_PLAYER_IDS[pid],
                "positions": "SP",
                "is_hitter": 0,
                "r": 0,
                "hr": 0,
                "rbi": 0,
                "sb": 0,
                "avg": 0.0,
                "obp": 0.0,
                "w": int(rng.integers(10, 18)),
                "l": int(rng.integers(5, 12)),
                "sv": 0,
                "k": int(rng.integers(160, 250)),
                "era": round(float(rng.uniform(2.50, 4.00)), 2),
                "whip": round(float(rng.uniform(0.95, 1.25)), 2),
                "ip": round(float(rng.uniform(170, 220)), 1),
            }
        )

    return pd.DataFrame(rows)


def _try_load_db_roster() -> pd.DataFrame | None:
    """Attempt to load roster from the SQLite database.

    Returns None if the database is empty or unavailable.
    """
    try:
        from src.database import load_player_pool

        pool = load_player_pool()
        if pool is not None and not pool.empty and len(pool) >= 20:
            return pool
    except Exception:
        pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run optimizer backtest")
    parser.add_argument(
        "--weeks",
        type=int,
        default=len(WEEKS),
        help=f"Number of weeks to test (max {len(WEEKS)})",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: first 10 player IDs only",
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

    # Select weeks
    n_weeks = min(args.weeks, len(WEEKS))
    selected_weeks = WEEKS[:n_weeks]

    # Load or build roster
    print("Loading roster projections...")
    roster = _try_load_db_roster()
    if roster is None:
        print("  Database not available -- using synthetic projections")
        roster = _build_synthetic_roster()
    else:
        print(f"  Loaded {len(roster)} players from database")

    # Select player IDs
    player_ids = list(BACKTEST_PLAYER_IDS.keys())
    if args.quick:
        player_ids = player_ids[:10]
        print(f"  Quick mode: testing {len(player_ids)} players")

    print(f"\nRunning backtest across {n_weeks} weeks...")
    print(f"  Players: {len(player_ids)}")
    print(f"  Period: {selected_weeks[0][0]} to {selected_weeks[-1][1]}")
    print()

    report = run_backtest(selected_weeks, roster, player_ids)

    print(format_report(report))


if __name__ == "__main__":
    main()

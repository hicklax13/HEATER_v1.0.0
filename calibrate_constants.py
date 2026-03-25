"""Calibrate HEATER's magic numbers from historical Yahoo Fantasy data.

Usage:
    # View current constants and calibration status
    python calibrate_constants.py --report

    # Generate defaults file (no Yahoo required)
    python calibrate_constants.py --save-defaults

    # Full calibration (requires Yahoo client connected)
    python calibrate_constants.py --calibrate --season 2025
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def cmd_report() -> None:
    """Print the current constants report."""
    from src.validation.constant_optimizer import ConstantRegistry

    registry = ConstantRegistry()
    print(registry.report())


def cmd_save_defaults() -> None:
    """Save current default values to calibrated_constants.json."""
    from src.validation.constant_optimizer import ALL_CONSTANTS, ConstantSet

    cs = ConstantSet()
    for c in ALL_CONSTANTS:
        cs.values[c.name] = c.default_value
        cs.metadata[c.name] = {
            "calibrated": False,
            "source": "default",
            "previous_value": c.current_value,
        }
    cs.save()
    print(f"Saved {len(cs.values)} default constants to data/calibrated_constants.json")


def cmd_calibrate(season: int) -> None:
    """Run calibration against historical Yahoo data."""
    from src.validation.calibration_data import fetch_calibration_data
    from src.validation.constant_optimizer import ConstantRegistry

    # Try to load Yahoo client from session
    yahoo_client = None
    try:
        from src.yahoo_api import YahooFantasyClient

        client = YahooFantasyClient()
        if client.is_connected():
            yahoo_client = client
    except Exception:
        pass

    if yahoo_client is None:
        logger.error("No Yahoo client available. Connect your Yahoo league first, then re-run calibration.")
        sys.exit(1)

    logger.info("Fetching calibration data for season %d...", season)
    dataset = fetch_calibration_data(yahoo_client, season=season)
    if dataset is None:
        logger.error("Could not fetch calibration data for season %d", season)
        sys.exit(1)

    logger.info(
        "Dataset: %d draft picks, %d trades, %d matchups",
        len(dataset.draft_picks),
        len(dataset.trades),
        len(dataset.weekly_matchups),
    )

    registry = ConstantRegistry()
    result = registry.calibrate_all([dataset])
    registry.save()

    print()
    print(registry.report())
    print("\nCalibrated constants saved to data/calibrated_constants.json")
    print(f"Constants with specific calibrators: {len(result.values)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate HEATER magic numbers from historical data")
    parser.add_argument("--report", action="store_true", help="Print constants report")
    parser.add_argument(
        "--save-defaults",
        action="store_true",
        help="Save default values to JSON",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run full calibration (requires Yahoo)",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=2025,
        help="Season to calibrate from (default: 2025)",
    )

    args = parser.parse_args()

    if args.report:
        cmd_report()
    elif args.save_defaults:
        cmd_save_defaults()
    elif args.calibrate:
        cmd_calibrate(args.season)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

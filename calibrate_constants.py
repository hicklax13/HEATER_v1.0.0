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
    import json
    import os

    from src.validation.calibration_data import fetch_calibration_data
    from src.validation.constant_optimizer import ConstantRegistry

    # Try to load Yahoo client. (BUG-016 fix: previously called
    # YahooFantasyClient() with no args which raises TypeError on the
    # required league_id positional arg, and `.is_connected()` doesn't
    # exist on the client — the bare except swallowed both errors
    # leaving the user with an opaque "No Yahoo client available".)
    yahoo_client = None
    try:
        from src.yahoo_api import _AUTH_DIR, YahooFantasyClient

        league_id = os.environ.get("YAHOO_LEAGUE_ID", "").strip()
        if not league_id:
            logger.error(
                "YAHOO_LEAGUE_ID env var not set. Connect your Yahoo league and re-run with YAHOO_LEAGUE_ID exported."
            )
            sys.exit(1)

        token_file = _AUTH_DIR / "yahoo_token.json"
        if not token_file.exists():
            logger.error(
                "Yahoo token file not found at %s. Authenticate via the app "
                "(streamlit run app.py) first to seed credentials.",
                token_file,
            )
            sys.exit(1)

        try:
            token_data = json.loads(token_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("Could not read Yahoo token file %s: %s", token_file, exc)
            sys.exit(1)

        consumer_key = token_data.get("consumer_key", os.environ.get("YAHOO_CLIENT_ID", ""))
        consumer_secret = token_data.get("consumer_secret", os.environ.get("YAHOO_CLIENT_SECRET", ""))
        if not consumer_key or not consumer_secret:
            logger.error(
                "Yahoo token file missing consumer_key/secret and env vars "
                "YAHOO_CLIENT_ID/YAHOO_CLIENT_SECRET are unset."
            )
            sys.exit(1)

        client = YahooFantasyClient(league_id=league_id)
        if client.authenticate(consumer_key, consumer_secret, token_data=token_data):
            if client.is_authenticated:
                yahoo_client = client
    except SystemExit:
        raise
    except Exception as exc:
        logger.error("Could not initialize YahooFantasyClient: %s", exc, exc_info=True)

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

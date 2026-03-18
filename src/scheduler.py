"""Background data refresh scheduler.

Runs staleness-based refresh checks on a daemon thread.
Safe to call in Streamlit — thread dies with the main process.

Usage:
    from src.scheduler import start_background_refresh
    start_background_refresh()  # idempotent — safe to call multiple times
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)

_scheduler_running = False
_scheduler_thread: threading.Thread | None = None
_CHECK_INTERVAL_SECONDS = 300  # Check every 5 minutes


def start_background_refresh():
    """Start background refresh thread (idempotent).

    The thread runs bootstrap_all_data(force=False) periodically,
    which internally checks staleness thresholds before fetching.
    """
    global _scheduler_running, _scheduler_thread
    if _scheduler_running:
        return
    _scheduler_running = True
    _scheduler_thread = threading.Thread(target=_refresh_loop, daemon=True, name="heater-refresh")
    _scheduler_thread.start()
    logger.info("Background refresh scheduler started (interval=%ds)", _CHECK_INTERVAL_SECONDS)


def stop_background_refresh():
    """Stop the background refresh thread."""
    global _scheduler_running
    _scheduler_running = False
    logger.info("Background refresh scheduler stopped")


def is_running() -> bool:
    """Check if the scheduler is currently running."""
    return _scheduler_running


def _refresh_loop():
    """Main scheduler loop — runs bootstrap with staleness checks."""
    while _scheduler_running:
        try:
            from src.data_bootstrap import bootstrap_all_data

            results = bootstrap_all_data(force=False)
            refreshed = [k for k, v in results.items() if v != "Fresh"]
            if refreshed:
                logger.info("Background refresh updated: %s", refreshed)
        except Exception as e:
            logger.warning("Background refresh error: %s", e)
        time.sleep(_CHECK_INTERVAL_SECONDS)

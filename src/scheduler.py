"""Background data refresh scheduler.

Runs staleness-based refresh checks on a daemon thread.
Safe to call in Streamlit — thread dies with the main process.

Usage:
    from src.scheduler import start_background_refresh
    start_background_refresh()  # idempotent — safe to call multiple times
"""

import logging
import threading

logger = logging.getLogger(__name__)

_scheduler_running = False
_scheduler_thread: threading.Thread | None = None
_scheduler_lock = threading.Lock()
_stop_event = threading.Event()
_CHECK_INTERVAL_SECONDS = 300  # Check every 5 minutes
_last_yahoo_ok: bool | None = None  # for degrade/recover transition logging


def start_background_refresh():
    """Start background refresh thread (idempotent).

    The thread runs bootstrap_all_data(force=False) periodically,
    which internally checks staleness thresholds before fetching.
    """
    global _scheduler_running, _scheduler_thread
    with _scheduler_lock:
        if _scheduler_running:
            return
        _stop_event.clear()
        _scheduler_running = True
        _scheduler_thread = threading.Thread(target=_refresh_loop, daemon=True, name="heater-refresh")
        _scheduler_thread.start()
    logger.info("Background refresh scheduler started (interval=%ds)", _CHECK_INTERVAL_SECONDS)


def stop_background_refresh():
    """Stop the background refresh thread."""
    global _scheduler_running, _scheduler_thread
    with _scheduler_lock:
        _scheduler_running = False
        _stop_event.set()
        thread = _scheduler_thread
        _scheduler_thread = None
    if thread is not None:
        thread.join(timeout=5)
    logger.info("Background refresh scheduler stopped")


def is_running() -> bool:
    """Check if the scheduler is currently running."""
    return _scheduler_running


def _refresh_once() -> None:
    """One scheduler cycle: pull the relayed token, reconnect, bootstrap. Extracted
    from the loop so it is unit-testable."""
    global _last_yahoo_ok
    from src.data_bootstrap import bootstrap_all_data

    # Relay: refresh the on-disk token from the gist BEFORE reconnecting, so yfpy
    # sees a valid token and never calls Yahoo's (datacenter-blocked) refresh.
    try:
        from src.token_relay import pull_relayed_token

        pull_relayed_token()
    except Exception as exc:
        logger.warning("Scheduler: relay token pull failed: %s", exc)

    yahoo_client = None
    try:
        from src.yahoo_api import try_reconnect_yahoo

        yahoo_client = try_reconnect_yahoo()
    except Exception as exc:
        logger.warning("Scheduler Yahoo reconnect failed: %s", exc)

    # Degrade/recover transition — log ONCE, not every cycle (no error storm).
    if yahoo_client is None and _last_yahoo_ok is not False:
        logger.warning("Yahoo sync degraded — relayed token stale? Is the mini-PC relay running?")
        _last_yahoo_ok = False
    elif yahoo_client is not None and _last_yahoo_ok is not True:
        if _last_yahoo_ok is False:
            logger.info("Yahoo sync recovered.")
        _last_yahoo_ok = True

    results = bootstrap_all_data(yahoo_client=yahoo_client, force=False)
    if yahoo_client is not None:
        try:
            yahoo_client.persist_current_token()
        except Exception:
            logger.warning("Scheduler: persisting refreshed Yahoo token failed.", exc_info=True)
    refreshed = [k for k, v in results.items() if v != "Fresh"]
    if refreshed:
        logger.info("Background refresh updated: %s", refreshed)


def _refresh_loop():
    """Main scheduler loop — see _refresh_once for one cycle's work."""
    while _scheduler_running:
        try:
            _refresh_once()
        except Exception as e:
            logger.warning("Background refresh error: %s", e)
        # Interruptible sleep — stop_event.set() wakes us immediately
        if _stop_event.wait(timeout=_CHECK_INTERVAL_SECONDS):
            break

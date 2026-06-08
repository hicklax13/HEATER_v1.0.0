"""Background data refresh scheduler.

Runs staleness-based refresh checks on a daemon thread.
Safe to call in Streamlit — thread dies with the main process.

Usage:
    from src.scheduler import start_background_refresh
    start_background_refresh()  # idempotent — safe to call multiple times
"""

import logging
import os
import threading

logger = logging.getLogger(__name__)

_scheduler_running = False
_scheduler_thread: threading.Thread | None = None
_scheduler_lock = threading.Lock()
_stop_event = threading.Event()
_CHECK_INTERVAL_SECONDS = 300  # Check every 5 minutes
_last_yahoo_ok: bool | None = None  # for degrade/recover transition logging


def _is_boot_managed_reader() -> bool:
    """True for a read-only process under boot-managed mode.

    When ``start.sh`` launches a DEDICATED scheduler process at container boot
    (``HEATER_SCHEDULER_BOOT=1``), that process marks itself the owner
    (``HEATER_SCHEDULER_IS_OWNER=1``) and is the SOLE SQLite writer. Every other
    process (each Streamlit session) must stay read-only and NOT start a second
    refresh thread — this preserves the single-replica / single-writer invariant.

    Unset ``HEATER_SCHEDULER_BOOT`` (v1, local, pytest) ⇒ not boot-managed ⇒ the
    caller starts the thread as before (byte-for-byte v1 behavior).
    """
    return os.environ.get("HEATER_SCHEDULER_BOOT") == "1" and os.environ.get("HEATER_SCHEDULER_IS_OWNER") != "1"


def start_background_refresh():
    """Start background refresh thread (idempotent).

    The thread runs bootstrap_all_data(force=False) periodically,
    which internally checks staleness thresholds before fetching.

    Under boot-managed mode (see ``_is_boot_managed_reader``) a non-owner caller
    no-ops: the dedicated boot process owns refresh, so session processes stay
    read-only and there is never more than one SQLite writer.
    """
    global _scheduler_running, _scheduler_thread
    if _is_boot_managed_reader():
        return
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


def _run_as_boot_process() -> None:
    """Set up the DEDICATED boot scheduler process (``python -m src.scheduler``).

    Started by ``start.sh`` at container boot so data refreshes WITHOUT waiting
    for the first browser session (Streamlit runs main() — which starts the
    scheduler — only on a real session). This process marks itself the sole
    writer; Streamlit session processes stay read-only (see ``_is_boot_managed_reader``).

    Inert under MULTI_USER off (v1/local) so flag-off stays byte-for-byte. Does
    NOT block — the blocking join lives in ``__main__`` so this stays unit-testable.
    """
    os.environ["HEATER_SCHEDULER_IS_OWNER"] = "1"
    from src.auth import multi_user_enabled

    if not multi_user_enabled():
        logger.info("Scheduler boot process: MULTI_USER off — nothing to do.")
        return
    try:
        from src.database import init_db

        init_db()  # idempotent; ensure tables exist on a fresh volume
    except Exception:
        logger.warning("Scheduler boot process: init_db failed.", exc_info=True)
    # A flushed print (not logger) so this ALWAYS reaches Railway/Docker stdout:
    # the refresh thread re-touches logging handlers during bootstrap, a race that
    # can swallow this process's early log lines. This greppable banner lets the
    # operator confirm the boot scheduler is alive (the #9 observability goal).
    print("[heater] dedicated boot scheduler started — warming data (MULTI_USER on)", flush=True)
    start_background_refresh()


if __name__ == "__main__":  # pragma: no cover - exercised by the Docker boot test
    from src.logging_setup import configure_src_logging

    configure_src_logging()  # send src.* logs to stdout (Railway captures them)
    _run_as_boot_process()
    # Block on the daemon refresh thread so this process (and the thread) stay
    # alive for the container's lifetime. If MULTI_USER was off, there is no
    # thread and we exit immediately (start.sh only launches us when MULTI_USER=1).
    _thread = _scheduler_thread
    if _thread is not None:
        _thread.join()

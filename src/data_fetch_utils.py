"""Shared data-fetch utilities: browser headers, 3-tier fallback chains.

Every external data source should use ``fetch_with_fallback()`` to implement
the Tier 1 (primary) -> Tier 2 (fallback) -> Tier 3 (emergency) chain.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Browser-like headers for FanGraphs / pybaseball calls.
# FanGraphs returns 403 to non-browser User-Agents on leaders-legacy.aspx.
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.fangraphs.com/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _is_empty(result: Any) -> bool:
    """Return True if *result* is None, empty list/dict, or empty DataFrame."""
    if result is None:
        return True
    if isinstance(result, pd.DataFrame):
        return result.empty
    if hasattr(result, "__len__"):
        return len(result) == 0
    return False


def fetch_with_fallback(
    source_name: str,
    primary_fn: Callable[[], Any],
    fallback_fn: Callable[[], Any] | None = None,
    emergency_fn: Callable[[], Any] | None = None,
) -> tuple[Any, str]:
    """Execute a 3-tier fallback chain.

    Returns ``(data, tier_used)`` where *tier_used* is one of
    ``"primary"``, ``"fallback"``, ``"emergency"``, or ``"failed"``.
    """
    tiers: list[tuple[str, Callable[[], Any] | None]] = [
        ("primary", primary_fn),
        ("fallback", fallback_fn),
        ("emergency", emergency_fn),
    ]
    for tier_name, fn in tiers:
        if fn is None:
            continue
        try:
            result = fn()
            if not _is_empty(result):
                if tier_name == "emergency":
                    logger.warning("Using emergency fallback for %s", source_name)
                return result, tier_name
        except Exception as exc:
            logger.warning("Tier %s failed for %s: %s", tier_name, source_name, exc)
    return None, "failed"


@contextlib.contextmanager
def patch_pybaseball_session():
    """Context manager that patches pybaseball's requests session headers.

    Usage::

        with patch_pybaseball_session():
            df = pitching_stats(2026)
    """
    try:
        import pybaseball as _pb

        session = getattr(_pb, "session", None)
        if session is None:
            # pybaseball >= 2.3 uses a module-level ``session`` object
            try:
                from pybaseball import cache as _cache

                session = getattr(_cache, "session", None)
            except ImportError:
                session = None

        if session is not None:
            old_headers = dict(session.headers)
            session.headers.update(_BROWSER_HEADERS)
            try:
                yield session
            finally:
                session.headers.clear()
                session.headers.update(old_headers)
        else:
            logger.debug("Could not locate pybaseball session object; skipping header patch")
            yield None
    except ImportError:
        yield None


@contextlib.contextmanager
def patch_requests_browser_headers(host_filter: str | None = None):
    """Monkey-patch ``requests.get`` and ``requests.Session.get`` to inject
    browser headers for the duration of the ``with`` block.

    Used by the SF-6 (Option C) FanGraphs scraper attempt: pybaseball calls
    ``requests.get`` directly (no session), so we need to intercept the call
    and merge browser headers into whatever the caller passed.

    Args:
        host_filter: If provided, only inject headers when the request URL
            contains this substring (e.g. ``"fangraphs.com"``). Other hosts
            pass through untouched. Pass ``None`` to inject on every request
            (use sparingly -- can interfere with other callers).

    Headers merge with caller-supplied headers (caller's win on collision).

    Usage::

        with patch_requests_browser_headers(host_filter="fangraphs.com"):
            df = pybaseball.pitching_stats(2026)
    """
    import requests

    original_get = requests.get
    original_session_get = requests.Session.get

    def _should_inject(url: object) -> bool:
        if host_filter is None:
            return True
        try:
            return host_filter in str(url)
        except Exception:
            return False

    def _merge_headers(kwargs: dict) -> dict:
        existing = kwargs.get("headers") or {}
        merged = dict(_BROWSER_HEADERS)
        merged.update(existing)  # caller-supplied headers win
        kwargs["headers"] = merged
        return kwargs

    def patched_get(url, **kwargs):
        if _should_inject(url):
            kwargs = _merge_headers(kwargs)
        return original_get(url, **kwargs)

    def patched_session_get(self, url, **kwargs):
        if _should_inject(url):
            kwargs = _merge_headers(kwargs)
        return original_session_get(self, url, **kwargs)

    requests.get = patched_get
    requests.Session.get = patched_session_get
    try:
        yield
    finally:
        requests.get = original_get
        requests.Session.get = original_session_get


def fetch_fangraphs_with_browser_headers(fetch_fn: Callable[[], Any]) -> Any:
    """Run a FanGraphs-bound fetcher with browser headers injected into
    ``requests.get`` calls. Returns the fetcher's result or raises its
    exception.

    Convenience wrapper for the SF-6 (Option C) Tier 1 attempt -- pairs with
    a Tier 2 ``fallback_fn`` that doesn't try the browser-headers trick (or
    surfaces the documented limitation message).
    """
    with patch_requests_browser_headers(host_filter="fangraphs.com"):
        return fetch_fn()

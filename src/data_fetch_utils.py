"""Shared data-fetch utilities: browser headers, 3-tier fallback chains.

Every external data source should use ``fetch_with_fallback()`` to implement
the Tier 1 (primary) -> Tier 2 (fallback) -> Tier 3 (emergency) chain.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
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

# Browser headers for Baseball Savant (requires a separate Referer).
# Savant returns 200 with these headers on endpoints that 403 raw `requests`.
SAVANT_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Referer": "https://baseballsavant.mlb.com/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch_savant_leaderboard_json(
    url: str,
    *,
    var_name: str = "data",
    timeout: int = 15,
) -> list[dict[str, Any]] | None:
    """Fetch a Baseball Savant leaderboard page and extract its embedded JSON.

    Most Savant leaderboards (e.g. catcher-framing) ship the table data as
    a JS literal ``const data = [{...}, {...}]`` in the HTML. This helper
    sends browser headers (Savant 403s raw `requests`), locates the literal,
    and parses it with a brace-balanced scan + ``json.loads``.

    Returns the parsed list of dicts, or ``None`` if the request fails or
    no parseable array is found.
    """
    try:
        import requests
    except ImportError:
        logger.warning("requests not installed; cannot fetch Savant URL")
        return None

    try:
        resp = requests.get(url, headers=SAVANT_BROWSER_HEADERS, timeout=timeout)
    except Exception as exc:
        logger.warning("Savant fetch failed for %s: %s", url, exc)
        return None

    if resp.status_code != 200:
        logger.warning("Savant returned %d for %s", resp.status_code, url)
        return None

    text = resp.text
    pattern = rf"(?:const|let|var)\s+{re.escape(var_name)}\s*=\s*(\[)"
    match = re.search(pattern, text)
    if not match:
        logger.warning("Could not locate `%s = [...]` literal in Savant HTML", var_name)
        return None

    start = match.start(1)
    depth = 0
    end = None
    # Brace-balanced scan: count [ and ] until depth returns to 0.
    for i in range(start, min(len(text), start + 2_000_000)):
        c = text[i]
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end is None:
        logger.warning("Could not find balanced bracket end for %s array", var_name)
        return None

    raw = text[start:end]
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Savant JSON parse failed: %s", exc)
        return None


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

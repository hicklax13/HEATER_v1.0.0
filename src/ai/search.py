"""Web search + deep-research tools for the AI chat (Phase 2).

- ``web_search``: free DuckDuckGo search (DDGS, no API key) -> [{title, url, snippet}].
- ``deep_research``: multi-source gather — search + fetch the top pages + condense —
  returning sources for the CALLING model to synthesize with citations (no nested
  LLM call here; the model that invoked the tool writes the cited answer).

Both work with any provider (DeepSeek/Anthropic/etc.) because the search runs
app-side, not via a provider-native tool. Network calls are isolated behind
``_ddgs_text`` / ``_fetch_url`` so tests mock them (honoring the conftest network
guard). Neither function raises — they return an ``error`` string instead.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_DEFAULT_RESULTS = 5
_MAX_FETCH = 3
_FETCH_CHARS = 1500
_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 HEATER-AI/1.0"


def _ddgs_text(query: str, max_results: int) -> list[dict]:
    """Raw DuckDuckGo text search (isolated for mocking)."""
    from ddgs import DDGS

    with DDGS() as d:
        return list(d.text(query, max_results=max_results))


def _is_public_http_url(url: str) -> bool:
    """True only for http(s) URLs whose host resolves to a GLOBAL (public) IP.

    Blocks SSRF to loopback/private/link-local/metadata endpoints (e.g.
    http://localhost, http://169.254.169.254) that a malicious search result
    could smuggle in for deep_research to fetch.
    """
    import ipaddress
    import socket
    from urllib.parse import urlparse

    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            return False
        ip = ipaddress.ip_address(socket.gethostbyname(parsed.hostname))
        return ip.is_global
    except Exception:
        return False


def _fetch_url(url: str) -> str:
    """Fetch a URL and return its visible text (isolated for mocking)."""
    import requests
    from bs4 import BeautifulSoup

    if not _is_public_http_url(url):
        raise ValueError("refusing to fetch non-public URL")
    resp = requests.get(url, timeout=10, headers={"User-Agent": _UA})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())


def web_search(query: str, max_results: int = _DEFAULT_RESULTS) -> dict:
    """Return {"results": [{title, url, snippet}], "error": str|None}. Never raises."""
    try:
        raw = _ddgs_text(query, max_results)
    except Exception as exc:  # network/library failure -> graceful error
        logger.warning("web_search failed: %s", exc)
        return {"results": [], "error": f"search failed: {type(exc).__name__}"}
    results = []
    for r in (raw or [])[:max_results]:
        results.append(
            {
                "title": r.get("title", ""),
                "url": r.get("href") or r.get("url", ""),
                "snippet": r.get("body") or r.get("snippet", ""),
            }
        )
    return {"results": results, "error": None}


def deep_research(query: str, max_results: int = _DEFAULT_RESULTS, max_fetch: int = _MAX_FETCH) -> dict:
    """Search + fetch the top pages -> condensed sources for the model to synthesize.

    Returns {"sources": [{title, url, content}], "error": str|None}. The top
    ``max_fetch`` results are fetched in full (truncated to _FETCH_CHARS); the rest
    are included snippet-only. A failed fetch falls back to that result's snippet.
    """
    search = web_search(query, max_results=max_results)
    if search["error"]:
        return {"sources": [], "error": search["error"]}
    sources = []
    for r in search["results"][:max_fetch]:
        url = r["url"]
        if not url:
            continue
        try:
            content = _fetch_url(url)[:_FETCH_CHARS]
        except Exception as exc:
            logger.info("deep_research fetch fell back to snippet for %s: %s", url, type(exc).__name__)
            content = r["snippet"]
        sources.append({"title": r["title"], "url": url, "content": content})
    for r in search["results"][max_fetch:]:
        sources.append({"title": r["title"], "url": r["url"], "content": r["snippet"]})
    return {"sources": sources, "error": None}

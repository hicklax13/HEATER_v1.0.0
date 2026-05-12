"""SF-6 Option C: browser-headers scraper for FanGraphs Stuff+ / batting_stats.

CLAUDE.md SF-6: FanGraphs blocks the pybaseball UA on leaders-legacy.aspx
with HTTP 403. This test suite verifies that the bootstrap phases attempt
the fetch with browser headers injected via ``patch_requests_browser_headers``
(Option C) and:

1. When the browser-headers fetch succeeds (mocked), data is persisted and
   ``refresh_log`` records ``status='success'`` + ``tier='primary'``.
2. When the browser-headers fetch still 403s (real-world FanGraphs case),
   the phase falls through to a clean SF-6 documentation message that
   tells users the optimizer falls back to FIP/xFIP K-boost proxy.

The probe at the start of this PR confirmed FanGraphs returns 403 even
with full Chrome UA + Sec-Fetch-* headers. Option C is documented as
"attempted, still blocked" in production. Option B (FIP/xFIP proxy) is
the active value-add. This test exists to ensure the attempt is wired
in and the telemetry is honest when the attempt fails.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data_fetch_utils import (
    fetch_fangraphs_with_browser_headers,
    patch_requests_browser_headers,
)

# ---------------------------------------------------------------------------
# Header injection: patch_requests_browser_headers context manager
# ---------------------------------------------------------------------------


def test_patch_injects_browser_headers_for_matching_host():
    """When host_filter matches the URL, browser headers are merged in."""
    captured = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs.get("headers")
        resp = MagicMock()
        resp.status_code = 200
        resp.content = b"<html></html>"
        return resp

    with patch("requests.get", side_effect=fake_get):
        with patch_requests_browser_headers(host_filter="fangraphs.com"):
            import requests

            requests.get("https://www.fangraphs.com/leaders-legacy.aspx")

    headers = captured.get("headers") or {}
    # Must contain a browser-like User-Agent (not the bare pybaseball/python-requests)
    ua = headers.get("User-Agent", "")
    assert "Mozilla" in ua, f"Expected browser UA, got {ua!r}"
    assert "Chrome" in ua, f"Expected Chrome UA, got {ua!r}"
    # Referer must hint we came from FanGraphs (Cloudflare-style bot check)
    assert headers.get("Referer", "").startswith("https://www.fangraphs.com")


def test_patch_does_not_inject_for_non_matching_host():
    """When host_filter doesn't match, headers pass through untouched."""
    captured = {}

    def fake_get(url, **kwargs):
        captured["headers"] = kwargs.get("headers")
        resp = MagicMock()
        resp.status_code = 200
        resp.content = b"ok"
        return resp

    # Caller passes no headers; with host_filter="fangraphs.com",
    # an mlb.com URL should NOT get browser headers injected.
    with patch("requests.get", side_effect=fake_get):
        with patch_requests_browser_headers(host_filter="fangraphs.com"):
            import requests

            requests.get("https://statsapi.mlb.com/api/v1/people")

    headers = captured.get("headers") or {}
    # Either None or empty dict — the patch should NOT have added a UA.
    assert "Mozilla" not in headers.get("User-Agent", "")


def test_patch_restores_requests_get_on_exit():
    """Original requests.get is restored after the context manager exits."""
    import requests

    original_get = requests.get
    with patch_requests_browser_headers(host_filter="fangraphs.com"):
        # Inside the block, requests.get is wrapped
        assert requests.get is not original_get
    # After exit, restored
    assert requests.get is original_get


def test_patch_restores_even_on_exception():
    """Even if the inner block raises, requests.get is restored."""
    import requests

    original_get = requests.get
    try:
        with patch_requests_browser_headers(host_filter="fangraphs.com"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert requests.get is original_get


def test_caller_supplied_headers_win_on_collision():
    """If caller already passes a User-Agent, the caller's value wins."""
    captured = {}

    def fake_get(url, **kwargs):
        captured["headers"] = kwargs.get("headers")
        resp = MagicMock()
        resp.status_code = 200
        return resp

    with patch("requests.get", side_effect=fake_get):
        with patch_requests_browser_headers(host_filter="fangraphs.com"):
            import requests

            requests.get(
                "https://www.fangraphs.com/leaders-legacy.aspx",
                headers={"User-Agent": "MyCustomAgent/1.0"},
            )

    headers = captured.get("headers") or {}
    # Caller's UA wins on collision
    assert headers.get("User-Agent") == "MyCustomAgent/1.0"
    # But Referer (which caller didn't supply) should still be injected
    assert "fangraphs.com" in headers.get("Referer", "")


# ---------------------------------------------------------------------------
# fetch_fangraphs_with_browser_headers: convenience wrapper
# ---------------------------------------------------------------------------


def test_fetch_wrapper_returns_inner_result():
    """The wrapper passes the inner fetcher's return value through unchanged."""
    expected = pd.DataFrame({"Name": ["Foo"], "Stuff+": [120]})

    def inner_fetcher():
        return expected

    result = fetch_fangraphs_with_browser_headers(inner_fetcher)
    assert result is expected


def test_fetch_wrapper_propagates_exceptions():
    """If the inner fetcher raises (e.g. HTTP 403), the wrapper re-raises."""

    def inner_fetcher():
        raise RuntimeError("Error accessing 'leaders-legacy.aspx'. Received status code 403")

    with pytest.raises(RuntimeError, match="403"):
        fetch_fangraphs_with_browser_headers(inner_fetcher)


# ---------------------------------------------------------------------------
# Bootstrap phase: success path (browser-headers fetch returns data)
# ---------------------------------------------------------------------------


def test_bootstrap_stuff_plus_records_primary_tier_on_success(tmp_path, monkeypatch):
    """When the mocked FanGraphs fetch returns data, refresh_log gets
    status='success' + tier='primary' so the UI shows the browser-headers
    attempt actually unblocked the data."""
    from src.data_bootstrap import _bootstrap_stuff_plus
    from src.database import get_connection

    # Use a temp DB so we don't touch production
    db_path = tmp_path / "test.db"
    monkeypatch.setattr("src.database.DB_PATH", db_path)

    # Initialize the schema
    from src.database import init_db

    init_db()

    # Insert a fake pitcher so the name lookup matches
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO players (name, is_hitter, positions) VALUES (?, 0, 'P')",
            ("Test Pitcher",),
        )
        pid = conn.execute("SELECT player_id FROM players WHERE name = ?", ("Test Pitcher",)).fetchone()[0]
        # Stub a season_stats row so the UPDATE has somewhere to land.
        conn.execute(
            "INSERT INTO season_stats (player_id, season) VALUES (?, ?)",
            (pid, pd.Timestamp.now().year),
        )
        conn.commit()
    finally:
        conn.close()

    # Mock pybaseball.pitching_stats to return Stuff+ data for our test pitcher
    fake_df = pd.DataFrame(
        {
            "Name": ["Test Pitcher"],
            "Stuff+": [115.0],
            "Location+": [98.0],
            "Pitching+": [105.0],
        }
    )
    with patch("pybaseball.pitching_stats", return_value=fake_df):
        progress = MagicMock()
        progress.phase = ""
        progress.detail = ""
        result = _bootstrap_stuff_plus(progress)

    # The phase should have written success
    assert "Updated" in result, f"Expected success message, got: {result}"

    # refresh_log should record tier=primary
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT status, tier, message FROM refresh_log WHERE source = ?",
            ("stuff_plus",),
        ).fetchone()
        assert row is not None, "refresh_log row missing"
        status, tier, message = row[0], row[1], row[2]
        # Wave 7 INFRA-F6: row-count gate downgrades to "partial" when
        # < expected_min rows; the mock fixture writes 1 pitcher, so
        # accept either "success" or "partial" (both mean data exists).
        assert status in ("success", "partial"), f"Expected success or partial, got {status!r}"
        assert tier == "primary", f"Expected tier='primary', got {tier!r}"
        assert "browser-headers" in (message or ""), f"Expected browser-headers in message, got: {message!r}"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Bootstrap phase: 403 fall-through still produces SF-6 documentation
# ---------------------------------------------------------------------------


def test_bootstrap_stuff_plus_403_fall_through_documents_sf6(tmp_path, monkeypatch):
    """When the browser-headers fetch still returns 403 (real-world case),
    the fall-through message must cite SF-6 + tell users about the FIP/xFIP
    proxy fallback."""
    from src.data_bootstrap import _bootstrap_stuff_plus
    from src.database import get_connection, init_db

    db_path = tmp_path / "test.db"
    monkeypatch.setattr("src.database.DB_PATH", db_path)
    init_db()

    # Mock pybaseball.pitching_stats to raise the 403 we see in production
    def raise_403(*args, **kwargs):
        raise RuntimeError("Error accessing 'https://www.fangraphs.com/leaders-legacy.aspx'. Received status code 403")

    with patch("pybaseball.pitching_stats", side_effect=raise_403):
        progress = MagicMock()
        progress.phase = ""
        progress.detail = ""
        result = _bootstrap_stuff_plus(progress)

    # Result string must mention SF-6 so users know it's documented behaviour
    assert "SF-6" in result, f"Expected SF-6 mention in result, got: {result!r}"
    # And/or hint at the FIP/xFIP proxy fallback so users know it isn't broken
    assert "FIP" in result or "proxy" in result.lower(), f"Expected FIP/proxy hint in result, got: {result!r}"

    # refresh_log message should also cite SF-6 + the proxy fallback
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT status, message FROM refresh_log WHERE source = ?",
            ("stuff_plus",),
        ).fetchone()
        assert row is not None, "refresh_log row missing"
        status, message = row[0], row[1]
        assert status == "skipped", f"Expected status='skipped' for 403, got {status!r}"
        assert "SF-6" in (message or ""), f"Expected SF-6 in message, got: {message!r}"
        assert "FIP" in (message or "") or "proxy" in (message or "").lower(), (
            f"Expected FIP/proxy hint in message, got: {message!r}"
        )
    finally:
        conn.close()


def test_bootstrap_batting_stats_403_fall_through_documents_sf6(tmp_path, monkeypatch):
    """Same 403 fall-through contract for the batting_stats phase."""
    from src.data_bootstrap import _bootstrap_batting_stats
    from src.database import get_connection, init_db

    db_path = tmp_path / "test.db"
    monkeypatch.setattr("src.database.DB_PATH", db_path)
    init_db()

    def raise_403(*args, **kwargs):
        raise RuntimeError("Error accessing 'https://www.fangraphs.com/leaders-legacy.aspx'. Received status code 403")

    with patch("pybaseball.batting_stats", side_effect=raise_403):
        progress = MagicMock()
        progress.phase = ""
        progress.detail = ""
        result = _bootstrap_batting_stats(progress)

    assert "SF-6" in result, f"Expected SF-6 mention in result, got: {result!r}"

    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT status, message FROM refresh_log WHERE source = ?",
            ("batting_stats",),
        ).fetchone()
        assert row is not None
        status, message = row[0], row[1]
        assert status == "skipped"
        assert "SF-6" in (message or "")
    finally:
        conn.close()

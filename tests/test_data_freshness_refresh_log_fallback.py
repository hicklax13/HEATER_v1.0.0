"""YDS get_data_freshness must consult refresh_log when the session-state cache
is cold, AND label by age-vs-TTL (2026-06-03 update).

Background:
  After a bootstrap (which writes SQLite + refresh_log but NOT the YDS
  session-state cache), a page that doesn't call every yds.get_*() method left
  the Data Freshness widget showing "Stale (will refresh)" for sources whose
  data was actually fresh in SQLite. The refresh_log fallback fixes that.

  2026-06-03: the fallback previously hard-coded "Cached (...)" for ANY age, so
  data the scheduler refreshes every 5 min still read "Cached". Now
  _refresh_log_freshness_label ages the timestamp against the source's TTL via
  _age_freshness_label: within TTL -> "Live", within 2x -> "Cached", beyond ->
  "Stale". The h/m/d suffix formatting is preserved.

This file pins:
  1. _refresh_log_freshness_label -> TTL-aware Live/Cached/Stale label.
  2. None for missing source, missing timestamp, or non-data statuses.
  3. get_data_freshness uses the fallback (now Live when fresh) when the
     session cache is cold but refresh_log has a recent success row.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.yahoo_data_service import YahooDataService

_TTL_30M = 1800.0  # rosters/standings default window


def _row(source: str, status: str, age_minutes: float = 1.0) -> dict:
    """Build a refresh_log row dict with an ISO timestamp `age_minutes` old."""
    ts = datetime.now(UTC) - timedelta(minutes=age_minutes)
    return {
        "source": source,
        "status": status,
        "last_refresh": ts.isoformat(),
    }


def test_refresh_log_freshness_label_within_ttl_is_live():
    """Within the TTL window → 'Live (Xm ago)' (the data is genuinely fresh)."""
    rl = {"yahoo_standings": _row("yahoo_standings", "success", age_minutes=15)}
    label = YahooDataService._refresh_log_freshness_label(rl, "yahoo_standings", _TTL_30M)
    assert label == "Live (15m ago)"


def test_refresh_log_freshness_label_under_one_minute():
    """<1 minute → 'Live (just now)'."""
    rl = {"yahoo_data": _row("yahoo_data", "success", age_minutes=0.3)}
    label = YahooDataService._refresh_log_freshness_label(rl, "yahoo_data", _TTL_30M)
    assert label == "Live (just now)"


def test_refresh_log_freshness_label_hours_past_ttl_is_stale():
    """Hours old, well past a 30m window → 'Stale (Xh ago)' (h formatting kept)."""
    rl = {"yahoo_transactions": _row("yahoo_transactions", "success", age_minutes=180)}
    label = YahooDataService._refresh_log_freshness_label(rl, "yahoo_transactions", _TTL_30M)
    assert label == "Stale (3h ago)"


def test_refresh_log_freshness_label_days_past_ttl_is_stale():
    """Days old → 'Stale (Xd ago)' (d formatting kept)."""
    rl = {"yahoo_free_agents": _row("yahoo_free_agents", "success", age_minutes=60 * 50)}
    label = YahooDataService._refresh_log_freshness_label(rl, "yahoo_free_agents", _TTL_30M)
    assert label == "Stale (2d ago)"


def test_refresh_log_freshness_label_just_past_ttl_is_cached():
    """Between TTL and 2x TTL → 'Cached (Xm ago)' (aging but still usable)."""
    rl = {"yahoo_standings": _row("yahoo_standings", "success", age_minutes=45)}
    label = YahooDataService._refresh_log_freshness_label(rl, "yahoo_standings", _TTL_30M)
    assert label == "Cached (45m ago)"


def test_refresh_log_freshness_label_partial_status_accepted():
    """'partial' is a successful write status — accept it."""
    rl = {"yahoo_free_agents": _row("yahoo_free_agents", "partial", age_minutes=5)}
    label = YahooDataService._refresh_log_freshness_label(rl, "yahoo_free_agents", _TTL_30M)
    assert label == "Live (5m ago)"


def test_refresh_log_freshness_label_cached_status_accepted():
    """'cached' (e.g. ECR cache hit, no fresh fetch) — accept it."""
    rl = {"yahoo_data": _row("yahoo_data", "cached", age_minutes=10)}
    label = YahooDataService._refresh_log_freshness_label(rl, "yahoo_data", _TTL_30M)
    assert label == "Live (10m ago)"


def test_refresh_log_freshness_label_skipped_status_accepted():
    """'skipped' (e.g. FA fetch hit timeout but cached data is used) — still
    indicates DATA exists in SQLite, so it's a usable freshness signal."""
    rl = {"yahoo_free_agents": _row("yahoo_free_agents", "skipped", age_minutes=3)}
    label = YahooDataService._refresh_log_freshness_label(rl, "yahoo_free_agents", _TTL_30M)
    assert label == "Live (3m ago)"


@pytest.mark.parametrize("bad_status", ["error", "unknown", "no_data", "timeout"])
def test_refresh_log_freshness_label_non_data_status_returns_none(bad_status):
    """Statuses that mean 'no data was written' should NOT surface as fresh —
    return None so the caller falls back to 'Stale' / 'Offline' label."""
    rl = {"yahoo_data": _row("yahoo_data", bad_status, age_minutes=5)}
    label = YahooDataService._refresh_log_freshness_label(rl, "yahoo_data", _TTL_30M)
    assert label is None


def test_refresh_log_freshness_label_missing_source():
    """Source not in dict → None."""
    label = YahooDataService._refresh_log_freshness_label({}, "yahoo_data", _TTL_30M)
    assert label is None


def test_refresh_log_freshness_label_none_source():
    """Source=None → None."""
    rl = {"yahoo_data": _row("yahoo_data", "success", age_minutes=5)}
    label = YahooDataService._refresh_log_freshness_label(rl, None, _TTL_30M)
    assert label is None


def test_refresh_log_freshness_label_invalid_timestamp():
    """Malformed last_refresh → None (gracefully)."""
    rl = {"yahoo_data": {"source": "yahoo_data", "status": "success", "last_refresh": "garbage"}}
    label = YahooDataService._refresh_log_freshness_label(rl, "yahoo_data", _TTL_30M)
    assert label is None


def test_get_data_freshness_uses_refresh_log_when_session_cache_cold(monkeypatch):
    """End-to-end: session cache empty, but refresh_log has recent success rows
    → tracked sources show 'Live (8m ago)' (8 min is inside every window),
    instead of the misleading 'Stale (will refresh)'."""
    fake_snapshot = [
        _row("yahoo_data", "success", age_minutes=8),
        _row("yahoo_standings", "success", age_minutes=8),
        _row("yahoo_free_agents", "partial", age_minutes=8),
        _row("yahoo_transactions", "success", age_minutes=8),
    ]
    monkeypatch.setattr(
        "src.database.get_refresh_log_snapshot",
        MagicMock(return_value=fake_snapshot),
    )

    yds = YahooDataService.__new__(YahooDataService)
    yds._client = MagicMock()  # connected
    yds._client.is_connected = lambda: True

    with patch("src.yahoo_data_service._get_state_store", return_value={}):
        freshness = yds.get_data_freshness()

    # Tracked sources fall back to refresh_log, aged vs TTL → "Live (8m ago)".
    assert freshness["rosters"] == "Live (8m ago)"
    assert freshness["standings"] == "Live (8m ago)"
    assert freshness["free_agents"] == "Live (8m ago)"
    assert freshness["transactions"] == "Live (8m ago)"
    # Sources missing from this snapshot keep the legacy "Stale" fallback
    # (no refresh_log row -> None -> is_connected() True -> stale).
    assert freshness["matchup"] == "Stale (will refresh)"
    assert freshness["settings"] == "Stale (will refresh)"
    assert freshness["schedule"] == "Stale (will refresh)"


def test_get_data_freshness_session_cache_wins_over_refresh_log(monkeypatch):
    """Regression guard: when session cache has fresh data, the refresh_log
    fallback is NOT consulted — session is authoritative for the live tier."""
    from src.yahoo_data_service import _CacheEntry

    monkeypatch.setattr(
        "src.database.get_refresh_log_snapshot",
        MagicMock(return_value=[_row("yahoo_data", "success", age_minutes=8)]),
    )

    yds = YahooDataService.__new__(YahooDataService)
    yds._client = MagicMock()
    yds._client.is_connected = lambda: True

    # Pre-populate a hot session cache entry (~0 seconds old).
    fake_store = {f"{yds._PREFIX}rosters": _CacheEntry(value="x", ttl=300)}
    with patch("src.yahoo_data_service._get_state_store", return_value=fake_store):
        freshness = yds.get_data_freshness()

    # Hot session cache → "Live (just now)".
    assert freshness["rosters"] == "Live (just now)"


def test_get_data_freshness_offline_falls_back_to_offline_label(monkeypatch):
    """When refresh_log is empty AND client is offline → 'Offline (DB)'."""
    monkeypatch.setattr(
        "src.database.get_refresh_log_snapshot",
        MagicMock(return_value=[]),
    )

    yds = YahooDataService.__new__(YahooDataService)
    yds._client = None  # offline

    with patch("src.yahoo_data_service._get_state_store", return_value={}):
        freshness = yds.get_data_freshness()

    for key in ("rosters", "standings", "matchup", "free_agents", "transactions"):
        assert freshness[key] == "Offline (DB)"

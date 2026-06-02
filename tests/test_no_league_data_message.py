"""F3 (2026-06-02 silent-failure sweep): when a personalized page gets empty
league data, the message must reflect WHY.

Previously all four personalized pages (My Team, Lineup, Trade Analyzer, Free
Agents) showed one generic string — "No league data loaded. Connect your Yahoo
league..." — which implies the league was never connected even when the real
cause is a transient Yahoo timeout (Yahoo slow → SQLite empty). That masked a
recoverable "try Refresh" situation behind a "you never connected" message.

`YahooDataService.data_unavailable_reason()` classifies the cause from cache
stats; `ui_shared.no_league_data_message(reason)` renders the matching text.
"""

from __future__ import annotations

from pathlib import Path

from src.ui_shared import no_league_data_message
from src.yahoo_data_service import CacheStats, YahooDataService

_REPO = Path(__file__).resolve().parent.parent
_PERSONALIZED_PAGES = [
    "pages/1_My_Team.py",
    "pages/2_Line-up_Optimizer.py",
    "pages/11_Trade_Analyzer.py",
    "pages/14_Free_Agents.py",
]


def _yds_with_error(key: str, err: str | None) -> YahooDataService:
    """Build a YahooDataService shell with only its cache stats populated —
    data_unavailable_reason() reads nothing else, so we skip __init__."""
    yds = object.__new__(YahooDataService)
    yds._stats = CacheStats()
    if err is not None:
        yds._stats.record_error(key, err)
    return yds


def test_reason_timeout():
    yds = _yds_with_error("rosters", "TimeoutError: fetch hung >15s")
    assert yds.data_unavailable_reason() == "timeout"


def test_reason_generic_error():
    yds = _yds_with_error("rosters", "ConnectionError: refused")
    assert yds.data_unavailable_reason() == "error"


def test_reason_clean_is_empty():
    yds = _yds_with_error("rosters", None)
    assert yds.data_unavailable_reason() == ""


def test_message_timeout_is_distinct_and_actionable():
    msg = no_league_data_message("timeout").lower()
    assert "timed out" in msg
    # Must point the member at the recovery action, not imply "never connected".
    assert "refresh" in msg
    assert "connect your yahoo league" not in msg


def test_message_error_is_temporarily_unavailable():
    msg = no_league_data_message("error").lower()
    assert "temporarily unavailable" in msg
    assert "refresh" in msg


def test_message_default_unchanged_intent():
    msg = no_league_data_message("").lower()
    # The genuinely-not-loaded case still guides reconnect / warming up.
    assert "no league data" in msg or "warming up" in msg


def test_no_placeholder_feedback_url_in_app():
    """F-UI-1 (2026-06-02): the beta banner must not ship a dead
    forms.gle/PLACEHOLDER 'Share feedback' link — members would click into
    nowhere. The in-app feedback popover is the live channel."""
    src = (_REPO / "app.py").read_text(encoding="utf-8")
    assert "forms.gle/PLACEHOLDER" not in src, (
        "app.py still contains the placeholder feedback URL — set a real Google Form URL or drop the banner link."
    )


def test_personalized_pages_use_the_helper():
    """Structural guard: the 4 personalized pages render their empty-state via
    no_league_data_message() (so the adaptive message can't silently regress to
    the old generic string)."""
    missing = []
    for rel in _PERSONALIZED_PAGES:
        src = (_REPO / rel).read_text(encoding="utf-8")
        if "no_league_data_message" not in src:
            missing.append(rel)
    assert not missing, f"pages not using no_league_data_message(): {missing}"

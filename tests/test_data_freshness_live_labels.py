"""Data Freshness must read "Live" when data is genuinely within its refresh
window, not "Cached" for everything that came from refresh_log.

2026-06-03 goal: the freshness card showed Rosters/Standings/Free Agents/
Transactions as "Cached" even though the scheduler refreshes them every 5 min —
well inside their 15-30 min windows. Root cause: `_refresh_log_freshness_label`
hard-coded "Cached (...)" for any age. New `_age_freshness_label(age, ttl)`
labels by age-vs-TTL: within TTL = Live, within 2x = Cached, beyond = Stale.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.yahoo_data_service import _age_freshness_label

_HALF_HOUR = 30 * 60  # a roster/standings TTL


def test_just_now_is_live():
    assert _age_freshness_label(0, _HALF_HOUR) == "Live (just now)"


def test_within_ttl_reads_live():
    # Scheduler refreshes every 5 min; 5 min is well inside a 30 min window.
    assert _age_freshness_label(5 * 60, _HALF_HOUR).startswith("Live")
    assert _age_freshness_label(29 * 60, _HALF_HOUR).startswith("Live")


def test_past_ttl_reads_cached():
    # Older than the window but still usable → Cached, not Live.
    assert _age_freshness_label(40 * 60, _HALF_HOUR).startswith("Cached")


def test_far_past_ttl_reads_stale():
    assert _age_freshness_label(3 * 60 * 60, _HALF_HOUR).startswith("Stale")


def test_short_ttl_matchup_window():
    # Matchup TTL is short (5 min). 2 min old → Live; 20 min old → Stale.
    five_min = 5 * 60
    assert _age_freshness_label(2 * 60, five_min).startswith("Live")
    assert _age_freshness_label(20 * 60, five_min).startswith("Stale")


def test_get_data_freshness_reports_live_from_fresh_refresh_log():
    """End-to-end: a just-written refresh_log row → the card reads 'Live', not
    'Cached'. This is the member-facing path (no session client, scheduler wrote
    the data) the 2026-06-03 goal targets."""
    from src.database import update_refresh_log
    from src.yahoo_data_service import YahooDataService, _get_state_store

    _get_state_store().clear()  # force the refresh_log path, not session cache
    update_refresh_log("yahoo_data", "success")  # 'rosters' source, timestamp = now

    yds = YahooDataService(yahoo_client=None)
    label = yds.get_data_freshness()["rosters"]
    assert label.startswith("Live"), f"expected Live, got {label!r}"

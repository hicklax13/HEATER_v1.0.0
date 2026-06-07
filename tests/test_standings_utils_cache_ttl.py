"""MS-C3 fix: get_all_team_totals module cache has a TTL so a long-lived
Railway replica doesn't serve stale standings forever.

Before the fix, _cached_team_totals short-circuited indefinitely once
populated (returning before the TTL'd YahooDataService call), and nothing in
production called clear_cache(). On the always-on single replica that meant
standings could go stale for the life of the process.

The fix stamps the cache with a wall-clock time and treats it stale after a
short TTL (default ~5 min). Within the TTL window the existing same-object
contract is preserved; after expiry the next call recomputes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

import src.standings_utils as su
from src.standings_utils import clear_cache, get_all_team_totals


def _make_pool() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "Player A",
                "is_hitter": 1,
                "r": 80,
                "hr": 25,
                "rbi": 70,
                "sb": 10,
                "avg": 0.280,
                "obp": 0.350,
                "ab": 500,
                "h": 140,
                "bb": 50,
                "hbp": 5,
                "sf": 3,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "ip": 0,
                "er": 0,
                "era": 0,
                "whip": 0,
                "bb_allowed": 0,
                "h_allowed": 0,
            },
        ]
    )


def test_cache_has_ttl_constant():
    """A positive TTL must be defined so the module cache can go stale."""
    assert hasattr(su, "_TEAM_TOTALS_TTL_SECS")
    assert su._TEAM_TOTALS_TTL_SECS > 0


def test_cache_returns_same_object_within_ttl(monkeypatch):
    """Within the TTL window, back-to-back calls still return the cached
    object (preserves the existing same-object contract)."""
    clear_cache()
    # Freeze time so both calls land inside the TTL window.
    monkeypatch.setattr(su.time, "time", lambda: 1000.0)
    pool = _make_pool()
    rosters = {"Team A": [1]}
    first = get_all_team_totals(rosters, pool)
    second = get_all_team_totals(rosters, pool)
    assert first is second
    clear_cache()


def test_cache_refreshes_after_ttl_expiry(monkeypatch):
    """After the TTL elapses, the next call recomputes (a fresh object),
    proving a long-lived replica picks up scheduler-written updates."""
    clear_cache()
    now = {"t": 1000.0}
    monkeypatch.setattr(su.time, "time", lambda: now["t"])
    pool = _make_pool()
    rosters = {"Team A": [1]}

    first = get_all_team_totals(rosters, pool)
    # Advance the clock past the TTL.
    now["t"] = 1000.0 + su._TEAM_TOTALS_TTL_SECS + 1.0
    second = get_all_team_totals(rosters, pool)

    assert first is not second, (
        "MS-C3: cache must recompute after the TTL expires, not serve the stale object forever on a long-lived replica"
    )
    assert first == second  # same inputs → same values
    clear_cache()


def test_clear_cache_resets_timestamp(monkeypatch):
    """clear_cache() must reset the stamp so a subsequent populate is fresh."""
    clear_cache()
    monkeypatch.setattr(su.time, "time", lambda: 5000.0)
    pool = _make_pool()
    rosters = {"Team A": [1]}
    get_all_team_totals(rosters, pool)
    clear_cache()
    assert su._cached_team_totals is None
    clear_cache()

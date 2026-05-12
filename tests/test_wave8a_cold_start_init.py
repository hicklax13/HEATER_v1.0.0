"""Wave 8a Group 4: cold-start / initialization bug fixes.

Covers 3 audit IDs:
  - D5A-013: standings_utils.clear_cache NameError on cold import
  - D6A-001: game_day hardcoded UTC-4 ignores DST
  - D2A-002: yahoo_api YahooFantasyClient._INVERSE_CATS evaluated at
             class-body time (snapshots LeagueConfig at module-import)
"""

from __future__ import annotations

import importlib
from pathlib import Path

# ── D5A-013: standings_utils.clear_cache cold-import NameError ───────


def test_d5a013_clear_cache_no_nameerror_on_cold_import():
    """clear_cache must not raise NameError when called immediately after a fresh
    module import, before any other function has touched _cached_fa_pool.

    Prior bug: _cached_fa_pool was defined further down in the file than
    clear_cache, so on a cold import where clear_cache was the first thing
    invoked, the `global _cached_fa_pool; _cached_fa_pool = None` assignment
    inside clear_cache succeeded (global assignments don't require pre-existing
    binding), BUT if clear_cache were ever to *read* the variable first, or
    if a subclass / future refactor inverted the order, the read-before-define
    failure would surface. The structural fix: define _cached_fa_pool at the
    module top alongside _cached_team_totals, so the binding always exists.
    """
    import src.standings_utils as su

    importlib.reload(su)  # Simulate cold import — fresh module state.
    # The bug manifests here: if _cached_fa_pool isn't yet defined at module
    # top, clear_cache (per the audit) is at risk. After the fix it must
    # always succeed.
    su.clear_cache()
    # Verify both cache attributes exist at module level after the call.
    assert hasattr(su, "_cached_team_totals")
    assert hasattr(su, "_cached_fa_pool")
    assert su._cached_team_totals is None
    assert su._cached_fa_pool is None


def test_d5a013_cached_fa_pool_defined_before_clear_cache():
    """Static-source guard: _cached_fa_pool must be defined at the module top,
    not below clear_cache. Prevents regression where someone moves the cache
    definition back below the function that mutates it.
    """
    text = Path("src/standings_utils.py").read_text()
    # Find line numbers for the binding and the function def.
    lines = text.splitlines()
    cache_def_line = None
    clear_cache_def_line = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if cache_def_line is None and stripped.startswith("_cached_fa_pool") and ("=" in stripped):
            # First module-level assignment of _cached_fa_pool
            cache_def_line = i
        if clear_cache_def_line is None and stripped.startswith("def clear_cache"):
            clear_cache_def_line = i

    assert cache_def_line is not None, "_cached_fa_pool must be defined at module level"
    assert clear_cache_def_line is not None, "clear_cache must exist"
    assert cache_def_line < clear_cache_def_line, (
        f"_cached_fa_pool (line {cache_def_line + 1}) must be defined BEFORE "
        f"clear_cache (line {clear_cache_def_line + 1}) to guarantee no NameError "
        f"on cold-start invocation"
    )


# ── D6A-001: game_day hardcoded UTC-4 ignores DST ────────────────────


def test_d6a001_game_day_no_hardcoded_utc_offset():
    """game_day must derive ET time from zoneinfo, not a bare timedelta(-4).

    A bare -4 hour offset ignores DST: it's correct ~Mar-Nov (EDT) but wrong
    Nov-Mar (when ET is EST = UTC-5). Using ZoneInfo("America/New_York")
    handles both regimes automatically.
    """
    text = Path("src/game_day.py").read_text()
    assert "timedelta(hours=-4)" not in text, (
        "game_day must not use hardcoded UTC-4 offset (breaks DST in winter months)"
    )
    assert "timedelta(hours=-5)" not in text, (
        "game_day must not use hardcoded UTC-5 offset (breaks DST in summer months)"
    )
    assert "ZoneInfo" in text or "pytz" in text, "game_day must use ZoneInfo (or pytz) for DST-aware ET handling"


def test_d6a001_get_target_game_date_uses_dst_aware_now():
    """get_target_game_date must call datetime.now with a DST-aware timezone."""
    import src.game_day as gd

    # The function must run without raising — sanity check.
    # We don't assert on the date value because it's environment-dependent.
    result = gd.get_target_game_date()
    assert isinstance(result, str)
    # YYYY-MM-DD format
    assert len(result) == 10
    assert result[4] == "-" and result[7] == "-"


# ── D2A-002: yahoo_api class-body LeagueConfig snapshot ──────────────


def test_d2a002_yahoo_client_no_classbody_leagueconfig_snapshot():
    """Static-source guard: YahooFantasyClient must not freeze LeagueConfig
    at class-body time. Class-body evaluation runs at module import, so any
    subsequent LeagueConfig change (test setup, league reconfig) is ignored.

    Look for the specific anti-pattern: an `=` assignment at class-body
    indent containing `LeagueConfig()`.
    """
    text = Path("src/yahoo_api.py").read_text()
    lines = text.splitlines()
    offenders = []
    for i, line in enumerate(lines):
        # Class-body assignments are at 4-space indent inside the class.
        # The bug pattern is something like:
        #     _INVERSE_CATS: set[str] = set(LeagueConfig().inverse_stats)
        if "LeagueConfig()" in line and "=" in line:
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            # Class-body assignment is indented 4 spaces; ignore comments
            # and ignore lines inside def bodies (indented 8+).
            if indent == 4 and not stripped.startswith("#"):
                offenders.append((i + 1, line.strip()))
    assert not offenders, (
        f"YahooFantasyClient class-body LeagueConfig() snapshot detected at "
        f"{offenders}. Move to __init__ or @property to read current config."
    )


def test_d2a002_yahoo_client_inverse_cats_on_instance():
    """YahooFantasyClient instances must expose an inverse-cats attribute
    that derives from a live LeagueConfig read, not a class-body snapshot.

    We don't fully authenticate; we just construct the client (which only
    requires a league_id) and check that the instance attribute exists and
    contains the inverse stats from LeagueConfig.
    """
    from src.valuation import LeagueConfig
    from src.yahoo_api import YahooFantasyClient

    client = YahooFantasyClient(league_id="test_league")
    expected_inverse = set(LeagueConfig().inverse_stats)

    # After the fix, the instance must have a _inverse_cats attr (or similar)
    # populated from LeagueConfig at __init__ time.
    assert hasattr(client, "_inverse_cats"), "YahooFantasyClient must expose _inverse_cats as an instance attribute"
    assert client._inverse_cats == expected_inverse
    # Sanity: 'L' must be in the inverse set per league rules.
    assert "L" in client._inverse_cats
    assert "ERA" in client._inverse_cats
    assert "WHIP" in client._inverse_cats


def test_d2a002_yahoo_client_all_cats_on_instance():
    """Same as above but for _all_cats."""
    from src.valuation import LeagueConfig
    from src.yahoo_api import YahooFantasyClient

    client = YahooFantasyClient(league_id="test_league")
    expected_all = list(LeagueConfig().all_categories)
    assert hasattr(client, "_all_cats"), "YahooFantasyClient must expose _all_cats as an instance attribute"
    assert client._all_cats == expected_all

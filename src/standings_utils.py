"""Shared standings utilities with session caching.

Eliminates redundant computations of roster category totals
across Trade Finder, League Standings, and My Team pages.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

_cached_team_totals: dict[str, dict[str, float]] | None = None


def get_all_team_totals(
    league_rosters: dict[str, list[int]],
    player_pool: pd.DataFrame,
    force_refresh: bool = False,
) -> dict[str, dict[str, float]]:
    """Get category totals for all teams. Cached in module scope."""
    global _cached_team_totals
    if _cached_team_totals is not None and not force_refresh:
        return _cached_team_totals

    from src.in_season import _roster_category_totals

    result = {}
    for team_name, pids in league_rosters.items():
        result[team_name] = _roster_category_totals(pids, player_pool)
    _cached_team_totals = result
    return result


def clear_cache():
    """Clear cached totals (call when rosters change)."""
    global _cached_team_totals
    _cached_team_totals = None

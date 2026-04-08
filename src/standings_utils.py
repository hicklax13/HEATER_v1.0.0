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


_cached_fa_pool: pd.DataFrame | None = None


def get_fa_pool(
    player_pool: pd.DataFrame,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """V5: Single access point for free agent pool across all pages.

    Priority: Yahoo FA data → enriched pool filter → DB query.
    Cached in module scope to avoid redundant computation.

    Returns:
        DataFrame of free agent players (not on any roster).
    """
    global _cached_fa_pool
    if _cached_fa_pool is not None and not force_refresh:
        return _cached_fa_pool

    fa_pool = pd.DataFrame()

    # Try Yahoo first
    try:
        import streamlit as st

        yds = st.session_state.get("_yahoo_data_service")
        if yds is not None and yds.is_connected():
            fa_pool = yds.get_free_agents()
    except Exception:
        pass

    # Fallback: filter enriched pool to exclude rostered players
    if fa_pool.empty:
        try:
            from src.database import get_all_rostered_player_ids

            rostered = get_all_rostered_player_ids()
            if rostered:
                fa_pool = player_pool[~player_pool["player_id"].isin(rostered)].copy()
            else:
                fa_pool = player_pool.copy()
        except Exception:
            fa_pool = player_pool.copy()

    _cached_fa_pool = fa_pool
    return fa_pool

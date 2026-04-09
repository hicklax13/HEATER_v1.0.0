"""Shared standings utilities with session caching.

Eliminates redundant computations of roster category totals
across Trade Finder, League Standings, and My Team pages.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# ── Match-record columns to exclude from standings (not stat categories) ──
_MATCH_RECORD_CATS = frozenset({"WINS", "LOSSES", "TIES", "PERCENTAGE", "POINTS_FOR", "POINTS_AGAINST", "STREAK"})


_cached_team_totals: dict[str, dict[str, float]] | None = None


def get_all_team_totals(
    league_rosters: dict[str, list[int]] | None = None,
    player_pool: pd.DataFrame | None = None,
    force_refresh: bool = False,
) -> dict[str, dict[str, float]]:
    """Get category totals for all teams with standings-first strategy.

    Priority order:
    1. Yahoo standings (actual season-to-date stats) — most accurate mid-season
    2. Projection-based computation from league_rosters + player_pool — fallback

    Cached in module scope to avoid redundant computation across pages.

    Args:
        league_rosters: {team_name: [player_ids]} — used for projection fallback.
            Can be None if Yahoo standings are expected to be available.
        player_pool: Player pool DataFrame — used for projection fallback.
            Can be None if Yahoo standings are expected to be available.
        force_refresh: If True, bypass cache and recompute.

    Returns:
        Dict mapping team_name -> {category: total_value}.
    """
    global _cached_team_totals
    if _cached_team_totals is not None and not force_refresh:
        return _cached_team_totals

    result: dict[str, dict[str, float]] = {}

    # ── Tier 1: Yahoo standings (actual season stats) ──────────────────
    try:
        import streamlit as st

        yds = st.session_state.get("_yahoo_data_service")
        if yds is not None and yds.is_connected():
            standings_df = yds.get_standings()
            if standings_df is not None and not standings_df.empty:
                if "category" in standings_df.columns and "team_name" in standings_df.columns:
                    # Long format: one row per team-category pair
                    standings_df["total"] = pd.to_numeric(standings_df["total"], errors="coerce").fillna(0)
                    for _, srow in standings_df.iterrows():
                        tname = str(srow.get("team_name", "")).strip()
                        cat = str(srow.get("category", "")).strip()
                        if not tname or not cat or cat.upper() in _MATCH_RECORD_CATS:
                            continue
                        val = float(srow.get("total", 0) or 0)
                        result.setdefault(tname, {})[cat] = val
                elif "team_name" in standings_df.columns:
                    # Wide format fallback
                    from src.valuation import LeagueConfig

                    config = LeagueConfig()
                    for _, row in standings_df.iterrows():
                        team = str(row.get("team_name", "")).strip()
                        if team:
                            totals = {}
                            for cat in config.all_categories:
                                totals[cat] = float(pd.to_numeric(row.get(cat, 0), errors="coerce") or 0)
                            result[team] = totals

                if result:
                    logger.debug("Team totals loaded from Yahoo standings (%d teams)", len(result))
                    _cached_team_totals = result
                    return result
    except Exception:
        logger.debug("Yahoo standings unavailable for team totals", exc_info=True)

    # ── Tier 2: Projection-based fallback ──────────────────────────────
    if league_rosters and player_pool is not None and not player_pool.empty:
        try:
            from src.in_season import _roster_category_totals

            for team_name, pids in league_rosters.items():
                totals = _roster_category_totals(pids, player_pool)
                if totals:
                    result[team_name] = totals

            if result:
                logger.debug("Team totals computed from projections (%d teams)", len(result))
        except Exception:
            logger.warning("Projection-based team totals failed", exc_info=True)

    _cached_team_totals = result if result else None
    return result


def clear_cache():
    """Clear cached totals and FA pool (call when rosters change)."""
    global _cached_team_totals, _cached_fa_pool
    _cached_team_totals = None
    _cached_fa_pool = None


_cached_fa_pool: pd.DataFrame | None = None


def get_fa_pool(
    player_pool: pd.DataFrame,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Single access point for free agent pool across all pages.

    Always returns a player_pool-enriched DataFrame (with projection columns).
    When Yahoo is connected, fetches FA list from Yahoo then cross-references
    with player_pool to get full projection data. Falls back to filtering
    player_pool by rostered IDs.

    Args:
        player_pool: Enriched player pool DataFrame with projections.
        force_refresh: If True, bypass cache and recompute.

    Returns:
        DataFrame of free agent players with projection columns.
    """
    global _cached_fa_pool
    if _cached_fa_pool is not None and not force_refresh:
        return _cached_fa_pool

    fa_pool = pd.DataFrame()

    # ── Tier 1: Yahoo FA list → cross-reference with player_pool ──────
    try:
        import streamlit as st

        yds = st.session_state.get("_yahoo_data_service")
        if yds is not None and yds.is_connected():
            yahoo_fa = yds.get_free_agents()
            if yahoo_fa is not None and not yahoo_fa.empty and not player_pool.empty:
                # Cross-reference Yahoo FAs with player_pool to get projections
                from src.live_stats import match_player_id

                yahoo_pids: set[int] = set()
                for _, row in yahoo_fa.iterrows():
                    pid = match_player_id(
                        row.get("player_name", row.get("name", "")),
                        row.get("team", ""),
                    )
                    if pid is not None:
                        yahoo_pids.add(pid)
                if yahoo_pids:
                    fa_pool = player_pool[player_pool["player_id"].isin(yahoo_pids)].copy()
                    logger.debug("FA pool: %d players from Yahoo cross-ref", len(fa_pool))
    except Exception:
        logger.warning("Yahoo FA fetch failed in get_fa_pool — falling back to pool filter")

    # ── Tier 2: Filter player_pool by rostered IDs ────────────────────
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

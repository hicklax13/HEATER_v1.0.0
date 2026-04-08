"""Matchup Context Service — session-scoped singleton for dynamic matchup data.

Wraps existing game_day, opponent_intel, category_urgency, and
matchup_adjustments modules with appropriate TTL caching. Pages call
this service instead of duplicating matchup data loading.

Usage::

    from src.matchup_context import get_matchup_context

    ctx = get_matchup_context()
    opponent = ctx.get_opponent_context()
    form = ctx.get_player_form(545361)
    urgency = ctx.get_category_urgency()
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ── TTL Constants (seconds) ──────────────────────────────────────────

TTL_OPPONENT = 1800  # 30 min — opponent profile is stable within a week
TTL_TEAM_STRENGTH = 86400  # 24 hr — wRC+/FIP changes slowly
TTL_PLAYER_FORM = 7200  # 2 hr — L7/L14/L30 form
TTL_WEATHER = 7200  # 2 hr — weather forecasts
TTL_MATCHUP_ADJ = 7200  # 2 hr — park/platoon/weather adjustments
TTL_URGENCY = 300  # 5 min — live matchup score (matches YDS matchup TTL)
TTL_ALL_STRENGTHS = 86400  # 24 hr — batch team strengths


# ── Cache Entry ──────────────────────────────────────────────────────


class _CacheEntry:
    """TTL-based cache entry matching the YDS pattern."""

    __slots__ = ("value", "created_at", "ttl")

    def __init__(self, value: Any, ttl: int):
        self.value = value
        self.created_at = time.monotonic()
        self.ttl = ttl

    @property
    def is_stale(self) -> bool:
        return (time.monotonic() - self.created_at) > self.ttl


# ── Matchup Context Service ─────────────────────────────────────────


class MatchupContextService:
    """Session-scoped singleton for dynamic matchup and game-day context.

    Provides weekly opponent intelligence, team strength data,
    weather/park adjustments, and recent player form — all with
    appropriate TTLs and in-memory caching.
    """

    def __init__(self):
        self._cache: dict[str, _CacheEntry] = {}

    def _get_cached(self, key: str) -> Any | None:
        """Return cached value if present and fresh, else None."""
        entry = self._cache.get(key)
        if entry is not None and not entry.is_stale:
            return entry.value
        return None

    def _set_cached(self, key: str, value: Any, ttl: int) -> None:
        """Store value in cache with TTL."""
        self._cache[key] = _CacheEntry(value, ttl)

    # ── Opponent Context ─────────────────────────────────────────────

    def get_opponent_context(self, week: int | None = None) -> dict:
        """Get opponent profile + category needs for this/specified week.

        Returns:
            Dict with keys: name, tier, threat, strengths, weaknesses,
            needs (per-category gap analysis), week.
        """
        cache_key = f"opponent_{week or 'current'}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        result: dict[str, Any] = {}
        try:
            from src.opponent_intel import get_current_opponent, get_opponent_for_week
            from src.yahoo_data_service import get_yahoo_data_service

            yds = get_yahoo_data_service()
            if week is not None:
                result = get_opponent_for_week(week, yds=yds)
            else:
                result = get_current_opponent(yds=yds)
        except Exception as e:
            logger.debug("get_opponent_context failed: %s", e)
            result = {"name": "Unknown", "tier": "Unknown", "threat": "Unknown"}

        # Merge opponent category needs when standings are available
        try:
            from src.opponent_trade_analysis import compute_opponent_needs
            from src.yahoo_data_service import get_yahoo_data_service

            yds = get_yahoo_data_service()
            standings_df = yds.get_standings()
            if not standings_df.empty and result.get("name"):
                from src.valuation import LeagueConfig

                config = LeagueConfig()
                # Build all_team_totals from standings
                all_team_totals: dict[str, dict[str, float]] = {}
                if "category" in standings_df.columns:
                    standings_df["total"] = pd.to_numeric(standings_df["total"], errors="coerce").fillna(0)
                    wide = standings_df.pivot_table(
                        index="team_name", columns="category", values="total", aggfunc="first"
                    ).reset_index()
                    for _, row in wide.iterrows():
                        team = row.get("team_name", "")
                        if team:
                            totals = {}
                            for cat in config.all_categories:
                                totals[cat] = float(pd.to_numeric(row.get(cat, 0), errors="coerce") or 0)
                            all_team_totals[team] = totals

                if all_team_totals and result["name"] in all_team_totals:
                    needs = compute_opponent_needs(result["name"], all_team_totals)
                    result["needs"] = needs
        except Exception as e:
            logger.debug("opponent needs merge failed: %s", e)

        self._set_cached(cache_key, result, TTL_OPPONENT)
        return result

    # ── Team Strength ────────────────────────────────────────────────

    def get_team_strength(self, team_abbr: str) -> dict:
        """Get team batting/pitching strength metrics.

        Returns:
            Dict with wrc_plus, fip, team_ops, team_era, team_whip, k_pct, bb_pct.
        """
        cache_key = f"strength_{team_abbr}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            from src.game_day import get_team_strength

            result = get_team_strength(team_abbr)
        except Exception as e:
            logger.debug("get_team_strength(%s) failed: %s", team_abbr, e)
            result = {
                "wrc_plus": 100.0,
                "fip": 4.00,
                "team_ops": 0.720,
                "team_era": 4.00,
                "team_whip": 1.25,
                "k_pct": 22.0,
                "bb_pct": 8.0,
            }

        self._set_cached(cache_key, result, TTL_TEAM_STRENGTH)
        return result

    def get_all_team_strengths(self) -> dict[str, dict]:
        """Get strength metrics for all 30 MLB teams.

        Returns:
            Dict mapping team_abbr to strength dict.
        """
        cache_key = "all_strengths"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        teams = [
            "ARI",
            "ATL",
            "BAL",
            "BOS",
            "CHC",
            "CHW",
            "CIN",
            "CLE",
            "COL",
            "DET",
            "HOU",
            "KC",
            "LAA",
            "LAD",
            "MIA",
            "MIL",
            "MIN",
            "NYM",
            "NYY",
            "OAK",
            "PHI",
            "PIT",
            "SD",
            "SF",
            "SEA",
            "STL",
            "TB",
            "TEX",
            "TOR",
            "WSH",
        ]
        result = {}
        for team in teams:
            result[team] = self.get_team_strength(team)

        self._set_cached(cache_key, result, TTL_ALL_STRENGTHS)
        return result

    # ── Player Recent Form ───────────────────────────────────────────

    def get_player_form(self, mlb_id: int) -> dict:
        """Get L7/L14/L30 recent form for a player.

        Returns:
            Dict with last_7, last_14, last_30 sub-dicts and trend
            ("hot", "cold", or "neutral").
        """
        cache_key = f"form_{mlb_id}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        result: dict[str, Any] = {"trend": "neutral"}
        try:
            from src.game_day import get_player_recent_form_cached

            form_data = get_player_recent_form_cached(mlb_id)
            if form_data and isinstance(form_data, dict):
                result = form_data
                # Classify trend from L7 vs L30
                l7 = form_data.get("last_7", {})
                l30 = form_data.get("last_30", {})
                l7_avg = l7.get("avg", 0) if isinstance(l7, dict) else 0
                l30_avg = l30.get("avg", 0) if isinstance(l30, dict) else 0
                if l7_avg > 0 and l30_avg > 0:
                    if l7_avg > l30_avg * 1.15:
                        result["trend"] = "hot"
                    elif l7_avg < l30_avg * 0.85:
                        result["trend"] = "cold"
                    else:
                        result["trend"] = "neutral"
        except Exception as e:
            logger.debug("get_player_form(%s) failed: %s", mlb_id, e)

        self._set_cached(cache_key, result, TTL_PLAYER_FORM)
        return result

    # ── Weather ──────────────────────────────────────────────────────

    def get_weather(self, team_abbr: str, game_date: str | None = None) -> dict:
        """Get weather data for a stadium.

        Returns:
            Dict with temp_f, wind_mph, wind_dir, precip_pct, humidity_pct.
            Empty dict if no data available.
        """
        cache_key = f"weather_{team_abbr}_{game_date or 'today'}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        result: dict = {}
        try:
            from src.database import get_connection

            conn = get_connection()
            try:
                query = (
                    "SELECT temp_f, wind_mph, wind_dir, precip_pct, humidity_pct "
                    "FROM game_day_weather WHERE venue_team = ?"
                )
                params: list = [team_abbr]
                if game_date:
                    query += " AND game_date = ?"
                    params.append(game_date)
                query += " ORDER BY game_date DESC LIMIT 1"
                df = pd.read_sql_query(query, conn, params=params)
                if not df.empty:
                    result = df.iloc[0].to_dict()
            finally:
                conn.close()
        except Exception as e:
            logger.debug("get_weather(%s) failed: %s", team_abbr, e)

        self._set_cached(cache_key, result, TTL_WEATHER)
        return result

    # ── Matchup Adjustments ──────────────────────────────────────────

    def get_matchup_adjustments(
        self,
        roster: pd.DataFrame,
        week_schedule: list[dict] | None = None,
    ) -> pd.DataFrame:
        """Get park/platoon/weather-adjusted projections for a roster.

        Returns:
            DataFrame with matchup_adjusted column added.
        """
        # No caching for this — roster input varies per call
        try:
            from src.optimizer.matchup_adjustments import compute_weekly_matchup_adjustments

            park_factors = {}
            try:
                from src.data_bootstrap import PARK_FACTORS

                park_factors = PARK_FACTORS or {}
            except ImportError:
                pass

            if week_schedule is None:
                try:
                    from src.optimizer.matchup_adjustments import get_weekly_schedule

                    week_schedule = get_weekly_schedule()
                except Exception:
                    week_schedule = []

            return compute_weekly_matchup_adjustments(
                roster=roster,
                week_schedule=week_schedule,
                park_factors=park_factors,
            )
        except Exception as e:
            logger.debug("get_matchup_adjustments failed: %s", e)
            return roster

    # ── Category Urgency ─────────────────────────────────────────────

    def get_category_urgency(self) -> dict:
        """Get per-category urgency weights from live matchup.

        Returns:
            Dict with urgency (per-cat float), rate_modes (per-cat str),
            summary (winning/losing/tied lists).
        """
        cache_key = "urgency"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        result: dict = {"urgency": {}, "rate_modes": {}, "summary": {}}
        try:
            from src.optimizer.category_urgency import compute_urgency_weights
            from src.yahoo_data_service import get_yahoo_data_service

            yds = get_yahoo_data_service()
            matchup = yds.get_matchup()
            if matchup:
                result = compute_urgency_weights(matchup)
        except Exception as e:
            logger.debug("get_category_urgency failed: %s", e)

        self._set_cached(cache_key, result, TTL_URGENCY)
        return result

    # ── Unified Category Weights ─────────────────────────────────────

    def get_category_weights(
        self,
        mode: str = "blended",
        alpha: float = 0.6,
    ) -> dict[str, float]:
        """Unified category weights across all subsystems.

        Provides 3 modes so every page uses the same weight source:

        - ``"matchup"``: H2H urgency-based (high weight for categories
          you're losing in the current matchup).
        - ``"standings"``: Gap-analysis-based (high weight for categories
          where you can gain standings positions).
        - ``"blended"`` (default): ``alpha * matchup + (1 - alpha) * standings``,
          normalized so the mean weight equals 1.0.

        Args:
            mode: One of ``"matchup"``, ``"standings"``, or ``"blended"``.
            alpha: Blend ratio (only used for ``"blended"``). 0.6 means
                60 % matchup, 40 % standings.

        Returns:
            Dict mapping each of the 12 scoring categories to a
            non-negative float weight. Mean is ~1.0 for blended mode.
        """
        cache_key = f"cat_weights_{mode}_{alpha}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        from src.valuation import LeagueConfig

        config = LeagueConfig()
        all_cats = config.all_categories
        equal = {cat: 1.0 for cat in all_cats}

        # ── Matchup weights ──────────────────────────────────────────
        matchup_weights: dict[str, float] | None = None
        try:
            urgency_data = self.get_category_urgency()
            urgency_map = urgency_data.get("urgency", {})
            if urgency_map:
                matchup_weights = {
                    cat: 0.5 + float(urgency_map.get(cat, 0.5))
                    for cat in all_cats
                }
        except Exception as exc:
            logger.debug("matchup weights failed: %s", exc)

        # ── Standings weights ────────────────────────────────────────
        standings_weights: dict[str, float] | None = None
        try:
            from src.yahoo_data_service import get_yahoo_data_service

            yds = get_yahoo_data_service()
            standings_df = yds.get_standings()
            if standings_df is not None and not standings_df.empty:
                # Build all_team_totals dict from standings
                all_team_totals: dict[str, dict[str, float]] = {}
                user_team_name: str | None = None
                if "category" in standings_df.columns:
                    standings_df["total"] = pd.to_numeric(
                        standings_df["total"], errors="coerce"
                    ).fillna(0)
                    wide = standings_df.pivot_table(
                        index="team_name",
                        columns="category",
                        values="total",
                        aggfunc="first",
                    ).reset_index()
                    for _, row in wide.iterrows():
                        team = str(row.get("team_name", ""))
                        if team:
                            totals = {}
                            for cat in all_cats:
                                totals[cat] = float(
                                    pd.to_numeric(row.get(cat, 0), errors="coerce") or 0
                                )
                            all_team_totals[team] = totals

                # Determine user team name
                try:
                    user_team_name = yds._user_team_name  # type: ignore[attr-defined]
                except AttributeError:
                    pass
                if not user_team_name:
                    # Fallback: try session_state
                    try:
                        import streamlit as st

                        user_team_name = st.session_state.get("team_name")
                    except Exception:
                        pass

                if user_team_name and all_team_totals:
                    from src.trade_intelligence import (
                        get_category_weights as _ti_get_weights,
                    )

                    standings_weights = _ti_get_weights(
                        user_team_name, all_team_totals, config
                    )
        except Exception as exc:
            logger.debug("standings weights failed: %s", exc)

        # ── Assemble result ──────────────────────────────────────────
        if mode == "matchup":
            result = matchup_weights if matchup_weights else equal
        elif mode == "standings":
            result = standings_weights if standings_weights else equal
        else:
            # blended (default)
            if matchup_weights and standings_weights:
                raw: dict[str, float] = {}
                for cat in all_cats:
                    mw = matchup_weights.get(cat, 1.0)
                    sw = standings_weights.get(cat, 1.0)
                    raw[cat] = alpha * mw + (1.0 - alpha) * sw
                # Normalize to mean = 1.0
                mean_val = sum(raw.values()) / max(len(raw), 1)
                if mean_val > 1e-9:
                    result = {c: v / mean_val for c, v in raw.items()}
                else:
                    result = equal
            elif matchup_weights:
                result = matchup_weights
            elif standings_weights:
                result = standings_weights
            else:
                result = equal

        # Ensure non-negative
        result = {c: max(0.0, v) for c, v in result.items()}

        self._set_cached(cache_key, result, TTL_URGENCY)
        return result

    # ── Data Freshness ───────────────────────────────────────────────

    def get_data_freshness(self) -> dict[str, str]:
        """Human-readable age labels for all cached data."""
        freshness: dict[str, str] = {}
        for key, entry in self._cache.items():
            age_s = time.monotonic() - entry.created_at
            if age_s < 60:
                label = f"Live ({int(age_s)}s ago)"
            elif age_s < 3600:
                label = f"{int(age_s / 60)}m ago"
            else:
                label = f"{age_s / 3600:.1f}h ago"
            freshness[key] = label
        return freshness


# ── Singleton Accessor ───────────────────────────────────────────────


def get_matchup_context() -> MatchupContextService:
    """Get or create the session-scoped MatchupContextService."""
    try:
        import streamlit as st

        key = "_matchup_context_service"
        if key not in st.session_state:
            st.session_state[key] = MatchupContextService()
        return st.session_state[key]
    except (ImportError, Exception):
        # Non-Streamlit context (tests, CLI) — return fresh instance
        return MatchupContextService()

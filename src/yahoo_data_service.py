"""Yahoo Data Service — live-first data layer with three-tier caching.

Provides a unified interface for all Yahoo Fantasy league data. Every page
calls this service instead of querying SQLite directly for league context.

Cache tiers:
    1. st.session_state (survives Streamlit reruns, ~0ms latency)
    2. Yahoo Fantasy API (0.5s per call, live source of truth)
    3. SQLite fallback (survives across sessions, used when Yahoo offline)

Write-through: every successful Yahoo fetch writes to SQLite so that
un-migrated pages automatically get fresher data.

Usage::

    from src.yahoo_data_service import get_yahoo_data_service

    yds = get_yahoo_data_service()
    rosters = yds.get_rosters()
    standings = yds.get_standings()
    matchup = yds.get_matchup()
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TTL configuration (seconds)
# ---------------------------------------------------------------------------


@dataclass
class TTLConfig:
    """Centralized TTL settings for each data category."""

    rosters: int = 1800  # 30 min — trades/waivers process hourly
    standings: int = 1800  # 30 min — scores update during games
    matchup: int = 300  # 5 min — live game tracking
    free_agents: int = 3600  # 1 hour — pool changes slowly, expensive call
    transactions: int = 900  # 15 min — trade deadline awareness
    settings: int = 86400  # 24 hours — static during season
    schedule: int = 86400  # 24 hours — set at season start
    draft_results: int = 86400  # 24 hours — post-draft, static


DEFAULT_TTL = TTLConfig()


# ---------------------------------------------------------------------------
# Cache entry (follows CacheEntry pattern from engine/production/cache.py)
# ---------------------------------------------------------------------------


class _CacheEntry:
    """Single cache entry with value and staleness tracking."""

    __slots__ = ("value", "created_at", "ttl", "hits")

    def __init__(self, value: Any, ttl: int):
        self.value = value
        self.created_at = time.monotonic()
        self.ttl = ttl
        self.hits = 0

    @property
    def is_stale(self) -> bool:
        return (time.monotonic() - self.created_at) > self.ttl

    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self.created_at


# ---------------------------------------------------------------------------
# Cache statistics
# ---------------------------------------------------------------------------


@dataclass
class CacheStats:
    """Observability data for cache performance."""

    hits: dict[str, int] = field(default_factory=dict)
    misses: dict[str, int] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    def record_hit(self, key: str) -> None:
        self.hits[key] = self.hits.get(key, 0) + 1

    def record_miss(self, key: str) -> None:
        self.misses[key] = self.misses.get(key, 0) + 1

    def record_error(self, key: str, error: str) -> None:
        self.errors[key] = error

    def summary(self) -> dict:
        total_hits = sum(self.hits.values())
        total_misses = sum(self.misses.values())
        total = total_hits + total_misses
        return {
            "total_hits": total_hits,
            "total_misses": total_misses,
            "hit_rate": round(total_hits / total * 100, 1) if total > 0 else 0.0,
            "per_key_hits": dict(self.hits),
            "per_key_misses": dict(self.misses),
            "errors": dict(self.errors),
        }


# ---------------------------------------------------------------------------
# Session state adapter (works outside Streamlit for testing)
# ---------------------------------------------------------------------------


def _get_state_store() -> dict:
    """Return session_state or a fallback dict for non-Streamlit contexts."""
    try:
        import streamlit as st

        return st.session_state
    except (ImportError, RuntimeError):
        # Running in tests or outside Streamlit — use module-level dict
        return _fallback_store


_fallback_store: dict = {}


# ---------------------------------------------------------------------------
# YahooDataService
# ---------------------------------------------------------------------------


class YahooDataService:
    """Unified live-first data layer for all Yahoo Fantasy data.

    Wraps YahooFantasyClient with multi-tier caching:
    session_state -> Yahoo API -> SQLite fallback.
    """

    _PREFIX = "_yds_"  # session_state key prefix to avoid collisions

    def __init__(
        self,
        yahoo_client: Any | None = None,
        ttl_config: TTLConfig | None = None,
    ):
        self._client = yahoo_client
        self._ttl = ttl_config or DEFAULT_TTL
        self._stats = CacheStats()

    # ------------------------------------------------------------------
    # Core three-tier fetch
    # ------------------------------------------------------------------

    def _get_cached(
        self,
        key: str,
        ttl: int,
        fetch_fn: Callable[[], Any],
        db_fallback_fn: Callable[[], Any],
        sync_fn: Callable[[Any], None] | None = None,
        force: bool = False,
    ) -> Any:
        """Three-tier fetch: session_state -> Yahoo API -> SQLite.

        Args:
            key: Cache key (auto-prefixed with ``_yds_``).
            ttl: Time-to-live in seconds.
            fetch_fn: Callable that fetches from Yahoo API.
            db_fallback_fn: Callable that reads from SQLite.
            sync_fn: Optional callable to write Yahoo data to SQLite.
            force: If True, skip cache and re-fetch from Yahoo.

        Returns:
            The data (DataFrame, dict, etc.).
        """
        store = _get_state_store()
        cache_key = f"{self._PREFIX}{key}"

        # Tier 1: session_state cache
        if not force:
            entry = store.get(cache_key)
            if isinstance(entry, _CacheEntry) and not entry.is_stale:
                entry.hits += 1
                self._stats.record_hit(key)
                return entry.value

        # Tier 2: Yahoo API (live)
        if self.is_connected():
            try:
                data = fetch_fn()
                # Validate we got real data
                if self._is_valid_data(data):
                    # Cache in session_state
                    store[cache_key] = _CacheEntry(value=data, ttl=ttl)
                    self._stats.record_miss(key)

                    # Write-through to SQLite
                    if sync_fn is not None:
                        try:
                            sync_fn(data)
                        except Exception:
                            logger.exception("Write-through to DB failed for %s", key)

                    return data
                else:
                    logger.debug("Yahoo returned empty data for %s, falling back to DB", key)
            except Exception as exc:
                self._stats.record_error(key, str(exc))
                logger.warning("Yahoo API error for %s: %s — falling back to DB", key, exc)

        # Tier 3: SQLite fallback
        self._stats.record_miss(key)
        try:
            data = db_fallback_fn()
            # Cache DB result in session_state too (with shorter TTL)
            # so repeated page reruns don't hit the DB every time
            if self._is_valid_data(data):
                store[cache_key] = _CacheEntry(value=data, ttl=min(ttl, 300))
            return data
        except Exception as exc:
            logger.exception("DB fallback also failed for %s", key)
            self._stats.record_error(key, f"DB fallback: {exc}")
            # Return empty data matching expected type
            return pd.DataFrame()

    @staticmethod
    def _is_valid_data(data: Any) -> bool:
        """Check if fetched data is non-empty."""
        if data is None:
            return False
        if isinstance(data, pd.DataFrame):
            return not data.empty
        if isinstance(data, dict):
            return len(data) > 0
        if isinstance(data, list):
            return len(data) > 0
        return True

    # ------------------------------------------------------------------
    # Page-facing accessors
    # ------------------------------------------------------------------

    def get_rosters(self, force_refresh: bool = False) -> pd.DataFrame:
        """All 12 teams' rosters. Same schema as ``load_league_rosters()``."""
        from src.database import load_league_rosters

        return self._get_cached(
            key="rosters",
            ttl=self._ttl.rosters,
            fetch_fn=self._fetch_and_sync_rosters,
            db_fallback_fn=load_league_rosters,
            force=force_refresh,
        )

    def get_team_roster(
        self,
        team_name: str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Single team's roster with player info and stats.

        Filters the full rosters DataFrame rather than making a separate
        API call, to stay within rate limits.
        """
        from src.league_manager import get_team_roster

        # Ensure rosters are fresh first
        rosters = self.get_rosters(force_refresh=force_refresh)
        if rosters.empty:
            return pd.DataFrame()

        # Use league_manager's join logic for enriched data
        return get_team_roster(team_name)

    def get_standings(self, force_refresh: bool = False) -> pd.DataFrame:
        """Live league standings. Same schema as ``load_league_standings()``."""
        from src.database import load_league_standings

        return self._get_cached(
            key="standings",
            ttl=self._ttl.standings,
            fetch_fn=self._fetch_and_sync_standings,
            db_fallback_fn=load_league_standings,
            force=force_refresh,
        )

    def get_matchup(self, force_refresh: bool = False) -> dict | None:
        """Current week's matchup scorecard for the user's team."""
        return self._get_cached(
            key="matchup",
            ttl=self._ttl.matchup,
            fetch_fn=self._fetch_matchup,
            db_fallback_fn=lambda: None,
            force=force_refresh,
        )

    def get_free_agents(
        self,
        max_players: int = 500,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Yahoo free agent list with ownership %."""
        return self._get_cached(
            key="free_agents",
            ttl=self._ttl.free_agents,
            fetch_fn=lambda: self._fetch_free_agents(max_players),
            db_fallback_fn=self._db_fallback_free_agents,
            force=force_refresh,
        )

    def get_transactions(self, force_refresh: bool = False) -> pd.DataFrame:
        """Recent league transactions (adds/drops/trades)."""
        return self._get_cached(
            key="transactions",
            ttl=self._ttl.transactions,
            fetch_fn=self._fetch_transactions,
            db_fallback_fn=lambda: pd.DataFrame(),
            force=force_refresh,
        )

    def get_settings(self, force_refresh: bool = False) -> dict:
        """League settings (scoring categories, roster positions, etc.)."""
        return self._get_cached(
            key="settings",
            ttl=self._ttl.settings,
            fetch_fn=self._fetch_settings,
            db_fallback_fn=lambda: {},
            force=force_refresh,
        )

    def get_schedule(self, force_refresh: bool = False) -> dict:
        """League schedule: ``{week_number: opponent_name}``."""
        from src.database import load_league_schedule

        return self._get_cached(
            key="schedule",
            ttl=self._ttl.schedule,
            fetch_fn=self._fetch_and_sync_schedule,
            db_fallback_fn=load_league_schedule,
            force=force_refresh,
        )

    def get_full_league_schedule(
        self,
        force_refresh: bool = False,
        total_weeks: int = 24,
    ) -> dict[int, list[tuple[str, str]]]:
        """Fetch all matchups for all weeks (all 12 teams).

        Returns:
            ``{week: [(team_a, team_b), ...]}`` with 6 matchups per week.
            Cached in session_state with 24h TTL. Stored in
            ``league_schedule_full`` table.
        """
        try:
            from src.database import load_league_schedule_full, upsert_league_schedule_full
        except ImportError:
            logger.warning("league_schedule_full DB functions not available")
            return {}

        store = _get_state_store()
        cache_key = f"{self._PREFIX}full_league_schedule"
        ts_key = f"{self._PREFIX}full_league_schedule_ts"

        # Check session cache (24h TTL)
        if not force_refresh:
            entry = store.get(cache_key)
            if isinstance(entry, _CacheEntry) and not entry.is_stale:
                entry.hits += 1
                self._stats.record_hit("full_league_schedule")
                return entry.value

        # Check DB cache
        db_schedule = load_league_schedule_full()
        if db_schedule and not force_refresh:
            store[cache_key] = _CacheEntry(value=db_schedule, ttl=86400)
            self._stats.record_miss("full_league_schedule")
            return db_schedule

        # Fetch from Yahoo API
        if not self._client or not self.is_connected():
            logger.warning("Yahoo not connected -- returning DB/empty schedule")
            self._stats.record_miss("full_league_schedule")
            return db_schedule or {}

        schedule: dict[int, list[tuple[str, str]]] = {}
        try:
            for week in range(1, total_weeks + 1):
                try:
                    scoreboard = self._client._query.get_league_scoreboard_by_week(
                        chosen_week=week,
                    )
                    if not scoreboard or not hasattr(scoreboard, "matchups"):
                        continue
                    week_matchups: list[tuple[str, str]] = []
                    for matchup in scoreboard.matchups:
                        teams = getattr(matchup, "teams", None)
                        if not teams or len(teams) < 2:
                            continue
                        team_a_name = ""
                        team_b_name = ""
                        for i, team_entry in enumerate(teams):
                            team_obj = (
                                team_entry.get("team")
                                if isinstance(team_entry, dict)
                                else getattr(team_entry, "team", team_entry)
                            )
                            name = getattr(team_obj, "name", None)
                            if isinstance(name, bytes):
                                name = name.decode("utf-8", errors="replace")
                            name = str(name) if name else f"Team_{i}"
                            if i == 0:
                                team_a_name = name
                            else:
                                team_b_name = name
                        if team_a_name and team_b_name:
                            week_matchups.append((team_a_name, team_b_name))
                            upsert_league_schedule_full(week, team_a_name, team_b_name)
                    schedule[week] = week_matchups
                    time.sleep(0.5)  # Rate limit between week fetches
                except Exception as exc:
                    logger.warning("Failed to fetch week %d schedule: %s", week, exc)
                    continue
        except Exception as exc:
            logger.error("Full schedule fetch failed: %s", exc)

        # Merge with DB for any missing weeks
        if not schedule:
            schedule = db_schedule or {}
        else:
            for week, matchups in (db_schedule or {}).items():
                if week not in schedule:
                    schedule[week] = matchups

        store[cache_key] = _CacheEntry(value=schedule, ttl=86400)
        self._stats.record_miss("full_league_schedule")
        return schedule

    def get_draft_results(self, force_refresh: bool = False) -> pd.DataFrame:
        """Draft results for ADP and opponent modeling."""
        return self._get_cached(
            key="draft_results",
            ttl=self._ttl.draft_results,
            fetch_fn=self._fetch_draft_results,
            db_fallback_fn=lambda: pd.DataFrame(),
            force=force_refresh,
        )

    def get_opponent_profile(self, team_name: str) -> dict:
        """Build opponent profile from live standings + roster data.

        Derives tier, threat level, strengths, and weaknesses from
        the current standings. Falls back to hardcoded AVIS profiles
        when live data is unavailable.
        """
        standings = self.get_standings()
        if standings.empty:
            return self._fallback_opponent_profile(team_name)

        return self._build_live_profile(team_name, standings)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def is_connected(self) -> bool:
        """Whether Yahoo client is authenticated and available."""
        if self._client is None:
            return False
        try:
            return self._client.is_authenticated
        except Exception:
            return False

    def force_refresh_all(self) -> dict[str, str]:
        """Invalidate all caches and re-fetch everything from Yahoo.

        Returns:
            Dict of ``{data_type: status_message}`` for each fetch.
        """
        results: dict[str, str] = {}

        for key, method in [
            ("rosters", self.get_rosters),
            ("standings", self.get_standings),
            ("matchup", self.get_matchup),
            ("free_agents", self.get_free_agents),
            ("transactions", self.get_transactions),
            ("settings", self.get_settings),
            ("schedule", self.get_schedule),
        ]:
            try:
                data = method(force_refresh=True)
                if self._is_valid_data(data):
                    results[key] = "Refreshed"
                elif key == "matchup" and data is None:
                    results[key] = "No active matchup"
                else:
                    results[key] = "Empty (no data available)"
            except Exception as exc:
                results[key] = f"Error: {exc}"

        return results

    def invalidate(self, data_type: str) -> None:
        """Invalidate a single data type's cache.

        Args:
            data_type: One of ``rosters``, ``standings``, ``matchup``,
                ``free_agents``, ``transactions``, ``settings``, ``schedule``.
        """
        store = _get_state_store()
        cache_key = f"{self._PREFIX}{data_type}"
        store.pop(cache_key, None)

    def get_data_freshness(self) -> dict[str, str]:
        """Return human-readable freshness labels per data type.

        Returns:
            Dict like ``{"rosters": "Live (2m ago)", "standings": "Offline (DB)"}``
        """
        store = _get_state_store()
        freshness: dict[str, str] = {}

        for key in [
            "rosters",
            "standings",
            "matchup",
            "free_agents",
            "transactions",
            "settings",
            "schedule",
        ]:
            cache_key = f"{self._PREFIX}{key}"
            entry = store.get(cache_key)
            if isinstance(entry, _CacheEntry) and not entry.is_stale:
                age_min = int(entry.age_seconds / 60)
                if age_min < 1:
                    freshness[key] = "Live (just now)"
                else:
                    freshness[key] = f"Cached ({age_min}m ago)"
            elif self.is_connected():
                freshness[key] = "Stale (will refresh)"
            else:
                freshness[key] = "Offline (DB)"

        return freshness

    def get_cache_stats(self) -> dict:
        """Return observability data: hits, misses, ages, errors."""
        return self._stats.summary()

    # ------------------------------------------------------------------
    # Internal fetch + sync methods
    # ------------------------------------------------------------------

    def _fetch_and_sync_rosters(self) -> pd.DataFrame:
        """Fetch rosters from Yahoo and write-through to SQLite.

        Uses ``sync_to_db()`` which does ``clear_league_rosters()`` + re-insert.
        Checks ``refresh_log`` to avoid racing with the bootstrap pipeline —
        if data was synced within the last 5 minutes, skip the expensive sync
        and just return the DB data.
        """
        from src.database import check_staleness, load_league_rosters, update_refresh_log

        # Guard: skip sync if bootstrap or another YDS call just ran
        if not check_staleness("yahoo_data", max_age_hours=5 / 60):
            return load_league_rosters()

        self._client.sync_to_db()

        update_refresh_log("yahoo_data", "success")

        # Take a roster snapshot for change tracking
        try:
            from src.database import snapshot_league_rosters

            snapshot_league_rosters()
        except Exception:
            pass  # Non-fatal

        return load_league_rosters()

    def _fetch_and_sync_standings(self) -> pd.DataFrame:
        """Fetch standings from Yahoo, write-through to SQLite, return long-format.

        Yahoo returns wide-format (columns: team_name, rank, r, hr, ...).
        Pages expect long-format (columns: team_name, category, total, rank).
        We write-through to SQLite then re-read in long-format — same pattern
        as ``_fetch_and_sync_rosters``. This ensures all code paths get the
        same schema regardless of whether data came from cache or DB.
        """
        from src.database import load_league_standings, update_refresh_log, upsert_league_standing

        standings_df = self._client.get_league_standings()
        if standings_df.empty:
            return standings_df

        # Write-through to SQLite (wide -> long conversion)
        # Exclude non-category columns: H2H record fields and metadata
        meta_cols = {
            "team_name",
            "team_key",
            "rank",
            "wins",
            "losses",
            "ties",
            "percentage",
            "points_for",
            "points_against",
            "streak",
        }
        cat_cols = [c for c in standings_df.columns if c.lower() not in meta_cols]
        for _, row in standings_df.iterrows():
            team_name = row.get("team_name", "")
            rank = int(row.get("rank", 0))
            for cat in cat_cols:
                upsert_league_standing(
                    team_name=team_name,
                    category=cat.upper(),
                    total=float(row.get(cat, 0)),
                    rank=rank,
                )

        # ── Also capture W-L-T records ──────────────────────────
        try:
            from src.database import upsert_league_record
        except ImportError:
            upsert_league_record = None  # Task 1 not yet merged

        if upsert_league_record is not None:
            meta_col_map = {
                "wins": "wins",
                "losses": "losses",
                "ties": "ties",
                "percentage": "win_pct",
                "points_for": "points_for",
                "points_against": "points_against",
                "streak": "streak",
                "rank": "rank",
            }
            for _, row in standings_df.iterrows():
                team_name = str(row.get("team_name", ""))
                if not team_name:
                    continue
                record_kwargs: dict = {}
                for src_col, dst_key in meta_col_map.items():
                    val = row.get(src_col)
                    if val is not None:
                        if dst_key in ("wins", "losses", "ties", "rank"):
                            try:
                                record_kwargs[dst_key] = int(float(val))
                            except (ValueError, TypeError):
                                record_kwargs[dst_key] = 0
                        elif dst_key == "win_pct":
                            try:
                                record_kwargs[dst_key] = float(val)
                            except (ValueError, TypeError):
                                record_kwargs[dst_key] = 0.0
                        else:
                            record_kwargs[dst_key] = str(val) if val else ""
                if record_kwargs:
                    try:
                        upsert_league_record(team_name, **record_kwargs)
                    except Exception:
                        logger.debug(
                            "Failed to upsert league record for %s",
                            team_name,
                            exc_info=True,
                        )

        update_refresh_log("yahoo_standings", "success")
        # Return long-format from DB so all consumers get consistent schema
        return load_league_standings()

    def _fetch_matchup(self) -> dict | None:
        """Fetch current matchup from Yahoo."""
        return self._client.get_current_matchup()

    def _fetch_free_agents(self, max_players: int) -> pd.DataFrame:
        """Fetch free agents from Yahoo with pagination."""
        return self._client.get_all_free_agents(max_players=max_players)

    def _fetch_transactions(self) -> pd.DataFrame:
        """Fetch recent transactions from Yahoo."""
        return self._client.get_league_transactions()

    def _fetch_settings(self) -> dict:
        """Fetch league settings from Yahoo."""
        return self._client.get_league_settings()

    def _fetch_draft_results(self) -> pd.DataFrame:
        """Fetch draft results from Yahoo and write-through to SQLite."""
        df = self._client.get_draft_results()
        if not df.empty:
            try:
                from src.database import save_league_draft_picks

                save_league_draft_picks(df)
                logger.info("Stored %d league draft picks", len(df))
            except Exception:
                logger.debug("Draft picks write-through failed", exc_info=True)
        return df

    def _fetch_and_sync_schedule(self) -> dict:
        """Fetch schedule from Yahoo matchup data and sync to SQLite.

        Yahoo doesn't have a direct "schedule" endpoint. We derive
        the schedule from the scoreboard by querying each week's matchups.
        This is expensive (1 call per week), so the 24-hour TTL is critical.

        Falls back to the AVIS hardcoded schedule if Yahoo call fails.
        """
        from src.database import upsert_league_schedule

        # Try to get user team name from league_teams table
        user_team_name = self._get_user_team_name()
        if not user_team_name:
            return self._fallback_schedule()

        schedule: dict[int, str] = {}

        # The matchup data from get_current_matchup() gives us one week.
        # For the full schedule, we'd need to query each week's scoreboard.
        # For now, try the current matchup to get this week's opponent,
        # and fall back to the stored schedule for other weeks.
        matchup = self._fetch_matchup()
        if matchup and "opponent" in matchup:
            week = matchup.get("week", 0)
            if week:
                schedule[week] = matchup["opponent"]
                upsert_league_schedule(week, user_team_name, matchup["opponent"])

        # Fill remaining weeks from the stored schedule or fallback
        from src.database import load_league_schedule

        stored = load_league_schedule()
        for week_num, opponent in stored.items():
            if week_num not in schedule:
                schedule[week_num] = opponent

        # If we still have gaps, use AVIS fallback
        fallback = self._fallback_schedule()
        for week_num, opponent in fallback.items():
            if week_num not in schedule:
                schedule[week_num] = opponent

        return schedule

    def _db_fallback_free_agents(self) -> pd.DataFrame:
        """DB fallback for free agents — filter player pool by rostered IDs."""
        try:
            from src.database import load_player_pool
            from src.league_manager import get_free_agents

            pool = load_player_pool()
            if pool.empty:
                return pd.DataFrame()
            return get_free_agents(pool)
        except Exception:
            logger.exception("DB fallback for free agents failed")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Opponent profile helpers
    # ------------------------------------------------------------------

    def _build_live_profile(self, team_name: str, standings: pd.DataFrame) -> dict:
        """Build opponent profile from live standings data."""
        team_standings = standings[standings["team_name"] == team_name]
        if team_standings.empty:
            return self._fallback_opponent_profile(team_name)

        # Compute rank-based tier
        # Average rank across all categories
        ranks = team_standings["rank"].dropna()
        avg_rank = ranks.mean() if not ranks.empty else 6.0

        if avg_rank <= 3:
            tier, threat = 1, "High"
        elif avg_rank <= 6:
            tier, threat = 2, "Medium"
        elif avg_rank <= 9:
            tier, threat = 3, "Low"
        else:
            tier, threat = 4, "Minimal"

        # Identify strengths (top 4) and weaknesses (bottom 4)
        cat_ranks = {}
        for _, row in team_standings.iterrows():
            cat = row.get("category", "")
            rank = row.get("rank")
            if cat and rank is not None:
                cat_ranks[cat] = int(rank)

        strengths = [c for c, r in sorted(cat_ranks.items(), key=lambda x: x[1]) if r <= 4]
        weaknesses = [c for c, r in sorted(cat_ranks.items(), key=lambda x: -x[1]) if r >= 9]

        # Try to get manager name from league_teams
        manager = ""
        try:
            from src.database import get_connection

            conn = get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT manager_name FROM league_teams WHERE team_name = ?",
                    (team_name,),
                )
                result = cursor.fetchone()
                if result:
                    manager = result[0] or ""
            finally:
                conn.close()
        except Exception:
            pass

        return {
            "tier": tier,
            "threat": threat,
            "manager": manager,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "notes": f"Live profile — avg category rank: {avg_rank:.1f}",
        }

    @staticmethod
    def _fallback_opponent_profile(team_name: str) -> dict:
        """Fall back to AVIS hardcoded profiles."""
        try:
            from src.opponent_intel import OPPONENT_PROFILES

            return OPPONENT_PROFILES.get(
                team_name,
                {
                    "tier": 3,
                    "threat": "Unknown",
                    "manager": "",
                    "strengths": [],
                    "weaknesses": [],
                    "notes": "No profile available",
                },
            )
        except ImportError:
            return {
                "tier": 3,
                "threat": "Unknown",
                "manager": "",
                "strengths": [],
                "weaknesses": [],
                "notes": "No profile available",
            }

    @staticmethod
    def _fallback_schedule() -> dict[int, str]:
        """Fall back to AVIS hardcoded schedule."""
        try:
            from src.opponent_intel import TEAM_HICKEY_SCHEDULE

            return dict(TEAM_HICKEY_SCHEDULE)
        except ImportError:
            return {}

    def _get_user_team_name(self) -> str | None:
        """Get the authenticated user's team name from the DB."""
        try:
            from src.database import get_connection

            conn = get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT team_name FROM league_teams WHERE is_user_team = 1")
                result = cursor.fetchone()
                return result[0] if result else None
            finally:
                conn.close()
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------


def get_yahoo_data_service() -> YahooDataService:
    """Get or create the session-scoped YahooDataService singleton.

    Stores the service in ``st.session_state["_yahoo_data_service"]``.
    If ``yahoo_client`` exists in session_state, wires it in automatically.

    Returns:
        The singleton YahooDataService instance.
    """
    store = _get_state_store()
    key = "_yahoo_data_service"

    if key not in store:
        # Try to find a Yahoo client in session_state
        client = store.get("yahoo_client")
        store[key] = YahooDataService(yahoo_client=client)
    else:
        # Rewire client if it appeared since last call (e.g., user connected Yahoo)
        svc = store[key]
        if svc._client is None:
            client = store.get("yahoo_client")
            if client is not None:
                svc._client = client

    return store[key]

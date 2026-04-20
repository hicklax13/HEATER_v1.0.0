"""Zero-interaction data bootstrap pipeline.

Fetches all MLB player data from free APIs on app startup.
Uses staleness-based smart refresh to avoid unnecessary API calls.
"""

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from src.analytics_context import AnalyticsContext, DataQuality

logger = logging.getLogger(__name__)

# Maximum seconds any single bootstrap phase is allowed to run.
# Raised from 90s → 180s to accommodate slower networks and
# phases that legitimately take 2-3 min (ECR scraping, ROS Bayesian).
# Per-phase overrides below for known-slow phases.
_PHASE_TIMEOUT_SECONDS = 180
_TIMEOUT_GAME_LOGS = 300  # iterates per-player stat_data calls
_TIMEOUT_ROS_PROJECTIONS = 300  # PyMC MCMC sampling can be slow
_TIMEOUT_ECR_CONSENSUS = 240  # multi-source scraping + ranking

# Module-level context from the last bootstrap run.
# Pages import this to show data freshness badges.
_LAST_BOOTSTRAP_CTX: AnalyticsContext | None = None


def _live_stats_ttl_hours(default_hours: float = 1.0) -> float:
    """Return the live_stats TTL in hours, tightened during the MLB game window.

    Between 7 PM and 1 AM US Eastern (when most MLB games are in-progress or
    just-finished), use a 15-minute TTL so season_stats re-fetches quickly
    after a game ends. Outside that window the standard 1-hour TTL applies.
    This only matters for within-session refreshes — force=True on launch
    bypasses staleness entirely.
    """
    try:
        from datetime import datetime as _dt

        try:
            from zoneinfo import ZoneInfo as _ZI

            now_et = _dt.now(_ZI("America/New_York"))
        except Exception:
            # Fallback: UTC-4 approximation (EDT); not perfect across DST
            # transitions but better than silently disabling the window.
            from datetime import timedelta, timezone

            now_et = _dt.now(timezone(timedelta(hours=-4)))
        hour = now_et.hour
        if hour >= 19 or hour < 1:
            return 0.25  # 15 minutes during active game window
    except Exception:
        pass
    return default_hours


def _format_fetch_error(exc: Exception, source: str = "FanGraphs") -> str:
    """Translate known HTTP failure signatures to cleaner status messages.

    FanGraphs started returning HTTP 403 to non-browser requests on the legacy
    leaderboard URL (leaders-legacy.aspx) in 2025. When that happens we don't
    want the Data Status panel to show a noisy stack trace — we want a clear
    "known limitation, data unavailable" message so it's obvious these aren't
    actionable bugs for the user. 429 follows the same pattern for rate-limits.
    """
    msg = str(exc) if exc else ""
    if "403" in msg or "leaders-legacy" in msg:
        return f"Skipped: {source} endpoint unavailable (HTTP 403)"
    if "429" in msg:
        return f"Skipped: {source} rate-limited (HTTP 429)"
    if "timeout" in msg.lower() or "timed out" in msg.lower():
        return f"Skipped: {source} request timed out"
    return f"Error: {exc}"


def _classify_fetch_error(exc: Exception) -> str:
    """Return the refresh_log ``status`` string for an exception.

    "skipped" for known-unavailable upstreams (403/429/timeout) so the DB
    agrees with what ``_format_fetch_error`` surfaces in the UI. "error"
    for anything else. Fixes the 2026-04-17 mismatch where the UI showed
    "Skipped" but refresh_log said "error".
    """
    msg = str(exc) if exc else ""
    if (
        "403" in msg
        or "leaders-legacy" in msg
        or "429" in msg
        or "timeout" in msg.lower()
        or "timed out" in msg.lower()
    ):
        return "skipped"
    return "error"


def _run_with_timeout(fn: Callable, timeout: int = _PHASE_TIMEOUT_SECONDS) -> str:
    """Run *fn* in a thread and return its result, or a timeout message.

    This prevents any single bootstrap phase from hanging the entire app.
    The worker thread is daemonized so it won't block process exit.
    """
    result_box: list[str] = []
    error_box: list[Exception] = []

    def _worker():
        try:
            result_box.append(fn())
        except Exception as exc:
            error_box.append(exc)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        logger.warning("Bootstrap phase timed out after %ds", timeout)
        return f"Timeout after {timeout}s"
    if error_box:
        raise error_box[0]
    return result_box[0] if result_box else "No result"


def get_bootstrap_context() -> AnalyticsContext | None:
    """Return the AnalyticsContext from the most recent bootstrap run."""
    return _LAST_BOOTSTRAP_CTX


def _stamp_from_result(ctx: AnalyticsContext, source: str, result: str) -> None:
    """Stamp data source quality from a bootstrap result string."""
    r = result.lower()
    if r == "fresh":
        ctx.stamp_data(source, DataQuality.LIVE, notes="Within staleness threshold")
    elif r.startswith("saved") or "refreshed" in r or r.startswith("merged"):
        ctx.stamp_data(source, DataQuality.LIVE, notes=result)
    elif r.startswith("error"):
        ctx.stamp_data(source, DataQuality.MISSING, notes=result)
    elif r.startswith("no "):
        ctx.stamp_data(source, DataQuality.MISSING, notes=result)
    elif r.startswith("skipped"):
        ctx.stamp_data(source, DataQuality.STALE, notes=result)
    else:
        ctx.stamp_data(source, DataQuality.STALE, notes=result)


@dataclass
class StalenessConfig:
    """Max age (hours) before each data source is refreshed."""

    players_hours: float = 168  # 7 days
    live_stats_hours: float = 1  # 1 hour (overridden to 0.25h during game window — see _live_stats_ttl_hours())
    projections_hours: float = 24  # 24 hours
    historical_hours: float = 720  # 30 days
    park_factors_hours: float = 720  # 30 days
    yahoo_hours: float = 0.5  # 30 minutes
    prospects_hours: float = 168  # 7 days
    news_hours: float = 1  # 1 hour
    ecr_consensus_hours: float = 24  # 24 hours
    game_day_hours: float = 2  # 2 hours
    team_strength_hours: float = 24  # 24 hours
    stuff_plus_hours: float = 24  # 24 hours
    batting_stats_hours: float = 24  # 24 hours
    sprint_speed_hours: float = 168  # 7 days (doesn't change often)
    umpire_hours: float = 24  # 24 hours (daily assignments)
    catcher_framing_hours: float = 168  # 7 days (stable metric)
    pvb_splits_hours: float = 168  # 7 days (stable with >=60 PA)


@dataclass
class BootstrapProgress:
    """Progress state for splash screen callback."""

    phase: str = ""
    detail: str = ""
    pct: float = 0.0


# Emergency park factors — FanGraphs 5yr regressed "Basic" (updated 2026-04-18).
# Used ONLY when both live sources (pybaseball + MLB API) fail.
# Scale: 1.000 = neutral. >1.0 = hitter-friendly, <1.0 = pitcher-friendly.
# Source: fangraphs.com/guts.aspx?type=pf
_PARK_FACTORS_EMERGENCY_2026: dict[str, float] = {
    "ARI": 1.007,
    "ATL": 1.001,
    "BAL": 0.986,
    "BOS": 1.042,
    "CHC": 0.979,
    "CWS": 1.003,
    "CIN": 1.046,
    "CLE": 0.989,
    "COL": 1.134,
    "DET": 1.003,
    "HOU": 0.995,
    "KC": 1.031,
    "LAA": 1.012,
    "LAD": 0.991,
    "MIA": 1.010,
    "MIL": 0.989,
    "MIN": 1.008,
    "NYM": 0.963,
    "NYY": 0.989,
    "ATH": 1.029,
    "PHI": 1.013,
    "PIT": 1.015,
    "SD": 0.959,
    "SF": 0.973,
    "SEA": 0.935,
    "STL": 0.975,
    "TB": 1.009,
    "TEX": 0.987,
    "TOR": 0.995,
    "WSH": 0.996,
}

# Backwards-compatible alias — many modules import PARK_FACTORS from this module.
# Always points to the current emergency dict.
PARK_FACTORS: dict[str, float] = _PARK_FACTORS_EMERGENCY_2026

# ── Lazy imports ─────────────────────────────────────────────────────
# These are imported inside functions to avoid circular import issues
# and to keep module-level imports minimal.


def _bootstrap_players(progress: BootstrapProgress) -> str:
    """Fetch all active MLB players and upsert to DB."""
    from src.database import (
        update_refresh_log,
        update_refresh_log_auto,
        upsert_player_bulk,
    )
    from src.live_stats import fetch_all_mlb_players

    progress.phase = "Players"
    progress.detail = "Fetching MLB roster..."
    try:
        df = fetch_all_mlb_players()
        if df.empty:
            update_refresh_log(
                "players",
                "no_data",
                rows_written=0,
                message="fetch_all_mlb_players returned empty",
            )
            return "No players returned from API"
        players = df.to_dict("records")
        count = upsert_player_bulk(players)
        status = update_refresh_log_auto(
            "players",
            count,
            expected_min=500,
            message=f"{count} players upserted from {len(df)} API rows",
        )
        return f"Saved {count} players ({status})"
    except Exception as e:
        update_refresh_log("players", "error", message=str(e)[:200])
        return f"Error: {e}"


def _bootstrap_projections(progress: BootstrapProgress) -> str:
    """Fetch projections from FanGraphs."""
    from src.database import update_refresh_log

    progress.phase = "Projections"
    progress.detail = "Fetching FanGraphs projections..."
    try:
        from src.data_pipeline import refresh_if_stale

        success = refresh_if_stale(force=True)
        if success:
            update_refresh_log("projections", "success")
            return "Projections refreshed"
        return "No projection data returned"
    except Exception as e:
        update_refresh_log("projections", "error", message=str(e)[:200])
        return f"Error: {e}"


def _bootstrap_live_stats(progress: BootstrapProgress) -> str:
    """Fetch current season stats.

    2026-04-17 audit: now validates actual saved row count and reports
    status='partial' / 'no_data' when below the expected floor instead of
    writing silent 'success' on a 0-row save.
    """
    from src.database import update_refresh_log, update_refresh_log_auto
    from src.live_stats import fetch_season_stats, save_season_stats_to_db

    progress.phase = "Live Stats"
    current_year = datetime.now(UTC).year
    progress.detail = f"Fetching {current_year} season stats..."
    # Expected floor: 30 teams × ~40 (40-man roster) = ~1200. Set conservatively.
    EXPECTED_MIN = 500
    try:
        df = fetch_season_stats(season=current_year)
        if df.empty:
            update_refresh_log(
                "season_stats",
                "no_data",
                rows_written=0,
                expected_min=EXPECTED_MIN,
                message="fetch_season_stats returned empty DataFrame",
            )
            return "No live stats available yet"
        count = save_season_stats_to_db(df)
        status = update_refresh_log_auto(
            "season_stats",
            count,
            expected_min=EXPECTED_MIN,
            message=f"saved {count}/{len(df)} rows",
        )
        return f"Saved {count} player stats ({status})"
    except Exception as e:
        update_refresh_log("season_stats", "error", message=str(e)[:200])
        return f"Error: {e}"


def _bootstrap_historical(progress: BootstrapProgress) -> tuple[str, dict | None]:
    """Fetch 3 years of historical stats.

    Returns:
        Tuple of (result_string, historical_data_dict_or_None).
    """
    from src.database import update_refresh_log
    from src.live_stats import fetch_historical_stats, save_season_stats_to_db

    current_year = datetime.now(UTC).year
    seasons = [current_year - 1]  # Only previous year (2025)
    progress.phase = "Historical"
    progress.detail = f"Fetching {seasons[0]} stats..."
    try:
        historical = fetch_historical_stats(seasons=seasons)
        total = 0
        for year, df in historical.items():
            count = save_season_stats_to_db(df, season=year)
            total += count
        if total > 0:
            update_refresh_log("historical_stats", "success")
        return (f"Saved {total} historical records across {len(historical)} seasons", historical)
    except Exception as e:
        update_refresh_log("historical_stats", "error", message=str(e)[:200])
        return (f"Error: {e}", None)


def _bootstrap_injury_data(progress: BootstrapProgress, historical: dict | None = None) -> str:
    """Extract injury history from historical stats and save."""
    from src.database import update_refresh_log, upsert_injury_history_bulk
    from src.live_stats import fetch_historical_stats, fetch_injury_data_bulk, match_player_id

    current_year = datetime.now(UTC).year
    seasons = [current_year - 1]  # Only previous year (2025)
    progress.phase = "Injury Data"
    progress.detail = "Processing injury history..."
    try:
        if historical is None:
            historical = fetch_historical_stats(seasons=seasons)
        raw_records = fetch_injury_data_bulk(historical)

        db_records = []
        for r in raw_records:
            pid = match_player_id(r["player_name"], r.get("team", ""))
            if pid is not None:
                db_records.append(
                    {
                        "player_id": pid,
                        "season": r["season"],
                        "games_played": r["games_played"],
                        "games_available": r["games_available"],
                    }
                )

        if db_records:
            count = upsert_injury_history_bulk(db_records)
            update_refresh_log("injury_data", "success")
            return f"Saved {count} injury records"
        return "No injury records matched"
    except Exception as e:
        update_refresh_log("injury_data", "error", message=str(e)[:200])
        return f"Error: {e}"


def _bootstrap_park_factors(progress: BootstrapProgress) -> str:
    """Phase 2: Fetch park factors via live sources with emergency fallback."""
    from src.data_fetch_utils import fetch_with_fallback, patch_pybaseball_session
    from src.database import update_refresh_log_auto, upsert_park_factors

    progress.phase = "Park Factors"
    progress.detail = "Fetching live park factors..."

    def _tier1_pybaseball():
        """Tier 1: pybaseball with browser headers."""
        with patch_pybaseball_session():
            from pybaseball import team_batting

            bat = team_batting(datetime.now(UTC).year)
            if bat is None or bat.empty:
                return None
            return bat

    def _tier3_emergency():
        """Tier 3: Hardcoded 2026 FanGraphs 5yr values."""
        return _PARK_FACTORS_EMERGENCY_2026

    try:
        data, tier = fetch_with_fallback(
            "park_factors",
            primary_fn=_tier1_pybaseball,
            fallback_fn=None,
            emergency_fn=_tier3_emergency,
        )

        # For now, always use the emergency dict to build the factors list.
        # When Tier 1 returns a DataFrame, we can extract park factors from it.
        # This ensures the pipeline works immediately while Tier 1 matures.
        source_dict = _PARK_FACTORS_EMERGENCY_2026
        if isinstance(data, dict):
            source_dict = data

        factors = [
            {
                "team_code": t,
                "factor_hitting": pf,
                "factor_pitching": 1.0 + (pf - 1.0) * 0.85,
            }
            for t, pf in source_dict.items()
        ]
        count = upsert_park_factors(factors)
        update_refresh_log_auto(
            "park_factors",
            count,
            expected_min=28,
            message=f"Saved {count} park factors",
            tier=tier,
        )
        return f"Saved {count} park factors (tier={tier})"
    except Exception as e:
        from src.database import update_refresh_log

        update_refresh_log("park_factors", "error", message=str(e)[:200])
        return f"Error: {e}"


def _store_yahoo_adp(adp_records: list[dict]) -> int:
    """Store Yahoo ADP records in the adp table via fuzzy name matching.

    Args:
        adp_records: List of dicts with keys ``name`` and ``yahoo_adp``.

    Returns:
        Number of rows upserted.
    """
    if not adp_records:
        return 0

    from src.database import get_connection

    conn = get_connection()
    try:
        cursor = conn.cursor()
        count = 0
        for rec in adp_records:
            name = str(rec.get("name", "")).strip()
            adp_val = float(rec.get("yahoo_adp", 0))
            if not name or adp_val <= 0:
                continue

            # Exact name match
            cursor.execute("SELECT player_id FROM players WHERE name = ?", (name,))
            result = cursor.fetchone()

            # Fuzzy fallback: first + last name LIKE match
            if result is None:
                parts = name.split()
                if len(parts) >= 2:
                    cursor.execute(
                        "SELECT player_id FROM players WHERE name LIKE ? AND name LIKE ?",
                        (f"%{parts[0]}%", f"%{parts[-1]}%"),
                    )
                    matches = cursor.fetchall()
                    if len(matches) == 1:
                        result = matches[0]
                    elif len(matches) > 1:
                        logger.debug(
                            "Yahoo ADP: fuzzy match for '%s' returned %d players, skipping",
                            name,
                            len(matches),
                        )

            if result is None:
                logger.debug("Yahoo ADP: no player_id found for '%s'", name)
                continue

            player_id = result[0]
            cursor.execute(
                """INSERT INTO adp (player_id, yahoo_adp, adp) VALUES (?, ?, ?)
                   ON CONFLICT(player_id) DO UPDATE SET yahoo_adp = ?, adp = min(adp, ?)""",
                (player_id, adp_val, adp_val, adp_val, adp_val),
            )
            count += 1

        conn.commit()
        return count
    finally:
        conn.close()


def _bootstrap_yahoo(progress: BootstrapProgress, yahoo_client=None) -> str:
    """Sync Yahoo league data if client is available."""
    from src.database import update_refresh_log

    progress.phase = "Yahoo Sync"
    if yahoo_client is None:
        return "Skipped (no Yahoo connection)"
    progress.detail = "Syncing Yahoo league data..."
    try:
        yahoo_client.sync_to_db()

        # Fetch and store Yahoo ADP from draft results
        progress.detail = "Fetching Yahoo ADP..."
        try:
            adp_records = yahoo_client.fetch_yahoo_adp()
            if adp_records:
                adp_count = _store_yahoo_adp(adp_records)
                logger.info("Stored %d Yahoo ADP records", adp_count)
        except Exception as adp_exc:
            # ADP may not be available pre-draft — non-fatal
            logger.warning("Yahoo ADP fetch failed (non-fatal): %s", adp_exc)

        update_refresh_log("yahoo_data", "success")
        return "Yahoo league data synced"
    except Exception as e:
        update_refresh_log("yahoo_data", "error", message=str(e)[:200])
        return f"Error: {e}"


def _bootstrap_extended_roster(progress: BootstrapProgress) -> str:
    """Extended roster (40-man + spring training).

    2026-04-17 reordered to run BEFORE live_stats so that players.mlb_id is
    populated when fetch_season_stats' save path matches rows by mlb_id.
    The previous order (live_stats → extended_roster) meant that the mlb_id
    column was mostly NULL at season_stats save time, forcing the fuzzy
    name-match fallback that corrupted hitter rows with pitcher stats
    (the Bellinger-with-7.2-IP class of bug).
    """
    progress.phase = "Extended Roster"
    progress.detail = "Fetching 40-man + spring training rosters..."
    try:
        from src.database import (
            update_refresh_log,
            update_refresh_log_auto,
            upsert_player_bulk,
        )
        from src.live_stats import fetch_extended_roster

        df = fetch_extended_roster()
        if df.empty:
            update_refresh_log(
                "extended_roster",
                "no_data",
                rows_written=0,
                message="fetch_extended_roster returned empty",
            )
            return "Extended roster: no data"
        count = upsert_player_bulk(df.to_dict("records"))
        status = update_refresh_log_auto(
            "extended_roster",
            count,
            expected_min=500,
            message=f"{count} players upserted from {len(df)} API rows",
        )
        return f"Extended roster: {count} players ({status})"
    except Exception as e:
        logger.warning("Extended roster bootstrap failed: %s", e)
        try:
            from src.database import update_refresh_log

            update_refresh_log("extended_roster", "error", message=str(e)[:200])
        except Exception:
            pass
        return f"Extended roster: error ({e})"


def _store_external_adp(df, name_col: str, adp_col: str, source: str) -> int:
    """Resolve player names and persist ADP values to the adp table.

    Args:
        df: DataFrame with at least *name_col* and *adp_col* columns.
        name_col: Column containing player names.
        adp_col: Column containing numeric ADP values.
        source: Label used for the ``fantasypros_adp`` column
            (``"fantasypros"`` or ``"nfbc"``).

    Returns:
        Number of rows upserted.
    """
    if df.empty:
        return 0

    from src.database import get_connection

    conn = get_connection()
    try:
        cursor = conn.cursor()
        count = 0
        for _, row in df.iterrows():
            name = str(row.get(name_col, "")).strip()
            try:
                adp_val = float(row.get(adp_col, 0))
            except (ValueError, TypeError):
                continue
            if not name or adp_val <= 0:
                continue

            # Exact name match
            cursor.execute("SELECT player_id FROM players WHERE name = ?", (name,))
            result = cursor.fetchone()

            # Fuzzy fallback: first + last name LIKE match
            if result is None:
                parts = name.split()
                if len(parts) >= 2:
                    cursor.execute(
                        "SELECT player_id FROM players WHERE name LIKE ? AND name LIKE ?",
                        (f"%{parts[0]}%", f"%{parts[-1]}%"),
                    )
                    matches = cursor.fetchall()
                    if len(matches) == 1:
                        result = matches[0]

            if result is None:
                continue

            player_id = result[0]
            col_name = "nfbc_adp" if source == "nfbc" else "fantasypros_adp"
            cursor.execute(
                f"""INSERT INTO adp (player_id, {col_name}, adp) VALUES (?, ?, ?)
                   ON CONFLICT(player_id) DO UPDATE SET {col_name} = ?, adp = min(adp, ?)""",
                (player_id, adp_val, adp_val, adp_val, adp_val),
            )
            count += 1

        conn.commit()
        return count
    finally:
        conn.close()


def _bootstrap_adp_sources(progress: BootstrapProgress) -> str:
    """Phase 9: Multi-source ADP (FantasyPros ECR + NFBC).

    FantasyPros ECR and NFBC leaderboards are JS-rendered pages that return
    empty HTML to non-browser requests. In-season (draft already complete)
    these sources are not critical; surface the skip clearly rather than as
    an error.
    """
    progress.phase = "ADP Sources"
    progress.detail = "Fetching FantasyPros + NFBC ADP..."
    try:
        from src.adp_sources import fetch_fantasypros_ecr, fetch_nfbc_adp
        from src.database import update_refresh_log

        results = []
        try:
            ecr = fetch_fantasypros_ecr()
            if not ecr.empty:
                stored = _store_external_adp(ecr, "player_name", "ecr_rank", "fantasypros")
                results.append(f"FantasyPros: {len(ecr)} fetched, {stored} stored")
        except Exception:
            logger.debug("FantasyPros ECR fetch failed", exc_info=True)
        try:
            nfbc = fetch_nfbc_adp()
            if not nfbc.empty:
                stored = _store_external_adp(nfbc, "player_name", "nfbc_adp", "nfbc")
                results.append(f"NFBC: {len(nfbc)} fetched, {stored} stored")
        except Exception:
            logger.debug("NFBC ADP fetch failed", exc_info=True)
        if results:
            update_refresh_log("adp_sources", "success")
            return f"ADP sources: {', '.join(results)}"
        update_refresh_log("adp_sources", "no_data")
        _month = datetime.now(UTC).month
        if 3 <= _month <= 10:
            return "Skipped: in-season (ADP less relevant post-draft)"
        return "Skipped: ADP sources returned no data (JS-gated pages)"
    except Exception as e:
        logger.warning("ADP sources bootstrap failed: %s", e)
        return _format_fetch_error(e, "ADP sources")


def _bootstrap_contracts(progress: BootstrapProgress) -> str:
    """Phase 10: Contract year data from BB-Ref.

    Persists contract_year=1 on matching players in the DB.
    """
    progress.phase = "Contract Data"
    progress.detail = "Fetching free agent list..."
    try:
        from src.contract_data import fetch_contract_year_players
        from src.database import update_refresh_log

        names = fetch_contract_year_players()
        if names:
            _persist_contract_years(names)
        update_refresh_log("contracts", "success")
        return f"Contracts: {len(names)} players in contract year"
    except Exception as e:
        logger.warning("Contract data bootstrap failed: %s", e)
        return f"Contracts: error ({e})"


def _bootstrap_news(progress: BootstrapProgress) -> str:
    """Phase 11: Recent MLB transactions/news.

    Also computes and persists per-player news_sentiment to the DB.
    """
    progress.phase = "News"
    progress.detail = "Fetching recent transactions..."
    try:
        from src.database import update_refresh_log
        from src.news_fetcher import fetch_recent_transactions

        items = fetch_recent_transactions(days_back=7)
        if items:
            _persist_news_sentiment(items)
        update_refresh_log("news", "success")
        return f"News: {len(items)} transactions"
    except Exception as e:
        logger.warning("News bootstrap failed: %s", e)
        return f"News: error ({e})"


# ── Computed field persistence helpers ────────────────────────────────


def _persist_contract_years(contract_names: set[str]) -> int:
    """Set contract_year=1 for players matching the contract-year name set.

    Resets all players to 0 first, then marks matching names as 1.

    Args:
        contract_names: Set of lowercased player names in their contract year.

    Returns:
        Number of players marked as contract year.
    """
    from src.database import get_connection

    conn = get_connection()
    try:
        # Reset all to 0
        conn.execute("UPDATE players SET contract_year = 0")
        updated = 0
        for name in contract_names:
            cursor = conn.execute(
                "UPDATE players SET contract_year = 1 WHERE LOWER(name) = ?",
                (name.lower(),),
            )
            updated += cursor.rowcount
        conn.commit()
        logger.info("Persisted contract_year=1 for %d players", updated)
        return updated
    except Exception:
        logger.exception("Failed to persist contract years")
        return 0
    finally:
        conn.close()


def _persist_news_sentiment(transactions: list[dict]) -> int:
    """Compute per-player news sentiment and persist to players table.

    Uses the existing news_sentiment module for scoring and
    aggregate_player_news for name→player_id resolution.

    Args:
        transactions: List of transaction dicts from fetch_recent_transactions().

    Returns:
        Number of players updated with sentiment scores.
    """
    from src.database import get_connection
    from src.news_fetcher import aggregate_player_news
    from src.news_sentiment import compute_news_sentiment

    conn = get_connection()
    try:
        # Build name->id mapping from DB
        rows = conn.execute("SELECT player_id, name FROM players").fetchall()
        name_to_id = {row[1]: row[0] for row in rows if row[1]}

        # Aggregate transactions by player_id
        player_news = aggregate_player_news(transactions, name_to_id)
        if not player_news:
            return 0

        updated = 0
        for pid, descriptions in player_news.items():
            sentiment = compute_news_sentiment(descriptions)
            conn.execute(
                "UPDATE players SET news_sentiment = ? WHERE player_id = ?",
                (sentiment, pid),
            )
            updated += 1
        conn.commit()
        logger.info("Persisted news_sentiment for %d players", updated)
        return updated
    except Exception:
        logger.exception("Failed to persist news sentiment")
        return 0
    finally:
        conn.close()


def _persist_depth_chart_roles(depth_data: dict) -> int:
    """Persist depth_chart_role and lineup_slot to players table.

    Args:
        depth_data: Output from fetch_depth_charts() — mapping of
            team code to {lineup, rotation, bullpen}.

    Returns:
        Number of players updated.
    """
    from src.database import get_connection
    from src.depth_charts import get_player_lineup_slot, get_player_role

    conn = get_connection()
    try:
        rows = conn.execute("SELECT player_id, name FROM players").fetchall()
        updated = 0
        for player_id, name in rows:
            if not name:
                continue
            role = get_player_role(name, depth_data)
            slot = get_player_lineup_slot(name, depth_data)
            if role != "bench" or slot is not None:
                conn.execute(
                    "UPDATE players SET depth_chart_role = ?, lineup_slot = ? WHERE player_id = ?",
                    (role, slot, player_id),
                )
                updated += 1
        conn.commit()
        logger.info("Persisted depth_chart_role for %d players", updated)
        return updated
    except Exception:
        logger.exception("Failed to persist depth chart roles")
        return 0
    finally:
        conn.close()


def _bootstrap_depth_charts(progress: BootstrapProgress) -> str:
    """Fetch depth charts and persist roles/lineup slots to DB.

    Depth chart source (Roster Resource / FanGraphs) returns empty when the
    endpoint is unavailable or JS-gated. Surface that as a Skip rather than
    a bare "no data" so the Data Status panel is self-explanatory.
    """
    progress.phase = "Depth Charts"
    progress.detail = "Fetching depth charts..."
    try:
        from src.database import update_refresh_log
        from src.depth_charts import fetch_depth_charts

        depth_data = fetch_depth_charts()
        if depth_data:
            count = _persist_depth_chart_roles(depth_data)
            update_refresh_log("depth_charts", "success")
            return f"Depth charts: {len(depth_data)} teams, {count} roles persisted"
        update_refresh_log("depth_charts", "no_data")
        return "Skipped: depth chart endpoint returned no data"
    except Exception as e:
        logger.warning("Depth chart bootstrap failed: %s", e)
        return _format_fetch_error(e, "Depth charts")


# ── FP Edge Feature Phases ────────────────────────────────────────────


def _bootstrap_prospects(progress: BootstrapProgress) -> str:
    """Phase 13: Prospect rankings from FanGraphs + MiLB stats."""
    progress.phase = "Prospects"
    progress.detail = "Refreshing prospect rankings..."
    try:
        from src.database import update_refresh_log
        from src.prospect_engine import refresh_prospect_rankings

        df = refresh_prospect_rankings(force=True)
        update_refresh_log("prospect_rankings", "success")
        return f"Prospects: {len(df)} ranked"
    except Exception as e:
        logger.warning("Prospect bootstrap failed: %s", e)
        return f"Prospects: error ({e})"


def _bootstrap_news_intel(progress: BootstrapProgress, yahoo_client=None) -> str:
    """Phase 14: Multi-source news intelligence."""
    progress.phase = "News Intelligence"
    progress.detail = "Fetching news from ESPN, RotoWire, MLB API..."
    try:
        from src.database import update_refresh_log
        from src.player_news import refresh_all_news

        count = refresh_all_news(yahoo_client=yahoo_client, force=True)
        update_refresh_log("news_intelligence", "success")
        return f"News: {count} items from multi-source"
    except Exception as e:
        logger.warning("News intelligence bootstrap failed: %s", e)
        return f"News intel: error ({e})"


def _bootstrap_ecr_consensus(progress: BootstrapProgress) -> str:
    """Phase 15: ECR consensus from multi-platform ranking sources."""
    progress.phase = "ECR Consensus"
    progress.detail = "Building multi-platform ranking consensus..."
    try:
        from src.database import update_refresh_log
        from src.ecr import refresh_ecr_consensus

        df = refresh_ecr_consensus(force=True)
        is_cached = getattr(df, "attrs", {}).get("ecr_is_cached", False)
        if is_cached:
            logger.warning("ECR consensus returned cached data — all sources failed")
            update_refresh_log("ecr_consensus", "cached")
            return f"ECR Consensus: {len(df)} players (cached — sources unavailable)"
        update_refresh_log("ecr_consensus", "success")
        return f"ECR Consensus: {len(df)} players ranked"
    except Exception as e:
        logger.warning("ECR consensus bootstrap failed: %s", e)
        return f"ECR consensus: error ({e})"


def _bootstrap_game_day(progress: BootstrapProgress) -> str:
    """Phase 20: Fetch game-day intelligence (weather, lineups, opposing pitchers)."""
    progress.phase = "Game Day Intel"
    progress.detail = "Fetching today's weather, lineups, opposing pitchers..."
    try:
        from src.game_day import fetch_game_day_intelligence

        result = fetch_game_day_intelligence()
        from src.database import update_refresh_log

        update_refresh_log("game_day", "success")
        games = result.get("games_count", 0)
        pitchers = result.get("pitcher_count", 0)
        return f"Saved game-day intel for {games} games, {pitchers} pitchers"
    except Exception as exc:
        logger.exception("Game-day bootstrap failed: %s", exc)
        from src.database import update_refresh_log

        update_refresh_log("game_day", "error")
        return f"Error: {exc}"


def _bootstrap_team_strength(progress: BootstrapProgress) -> str:
    """Phase 21: Fetch team-level batting and pitching strength metrics."""
    progress.phase = "Team Strength"
    progress.detail = "Fetching team batting/pitching metrics..."
    try:
        from src.database import get_connection, update_refresh_log, update_refresh_log_auto
        from src.game_day import fetch_team_strength

        # Purge stale rows before fetch
        conn = get_connection()
        try:
            conn.execute("DELETE FROM team_strength WHERE fetched_at < datetime('now', '-3 days')")
            conn.commit()
        finally:
            conn.close()

        df = fetch_team_strength(datetime.now(UTC).year)
        count = len(df) if df is not None and not df.empty else 0
        status = update_refresh_log_auto(
            "team_strength",
            count,
            expected_min=28,
            message=f"Saved {count} teams",
        )
        return f"Saved team strength for {count} teams ({status})"
    except Exception as exc:
        logger.exception("Team strength bootstrap failed: %s", exc)
        from src.database import update_refresh_log

        update_refresh_log("team_strength", "error", message=str(exc)[:200])
        return f"Error: {exc}"


def _bootstrap_stuff_plus(progress: BootstrapProgress) -> str:
    """Phase 22: Fetch Stuff+/Location+/Pitching+ from FanGraphs via pybaseball."""
    progress.phase = "Stuff+ Metrics"
    progress.detail = "Fetching Stuff+/Location+/Pitching+ from FanGraphs..."
    try:
        from pybaseball import pitching_stats
    except ImportError:
        logger.warning("pybaseball not installed — skipping Stuff+ fetch")
        return "Skipped: pybaseball not installed"

    try:
        import pandas as pd

        from src.database import get_connection, update_refresh_log

        year = datetime.now(UTC).year
        logger.info("Fetching FanGraphs pitching stats for %d (qual=0)...", year)
        fg_df = pitching_stats(year, qual=0)

        if fg_df is None or fg_df.empty:
            logger.warning("pybaseball pitching_stats returned empty data")
            return "Skipped: no data returned"

        # Identify the Stuff+/Location+/Pitching+ and gmLI columns
        # FanGraphs uses "Stuff+" or "stuff_plus" depending on pybaseball version
        col_map = {}
        for col in fg_df.columns:
            cl = col.lower().replace(" ", "").replace("_", "")
            if cl in ("stuff+", "stuffplus"):
                col_map[col] = "stuff_plus"
            elif cl in ("location+", "locationplus"):
                col_map[col] = "location_plus"
            elif cl in ("pitching+", "pitchingplus"):
                col_map[col] = "pitching_plus"
            # T5: gmLI (game-log leverage index) for closer monitor
            elif cl in ("gmli", "gmli", "leverageindex", "gmleverageindex"):
                col_map[col] = "gmli"

        if not col_map:
            logger.warning(
                "No Stuff+/Location+/Pitching+/gmLI columns found in FanGraphs data. Columns: %s",
                list(fg_df.columns)[:30],
            )
            return "Skipped: target columns not in FanGraphs data"

        logger.info("Found FanGraphs columns: %s", col_map)

        # Rename to DB column names
        fg_df = fg_df.rename(columns=col_map)

        # Build a name→player_id lookup from the players table
        conn = get_connection()
        try:
            players_df = pd.read_sql(
                "SELECT player_id, name FROM players WHERE is_hitter = 0",
                conn,
            )
            # Case-insensitive name lookup
            name_to_id = {}
            for _, row in players_df.iterrows():
                if row["name"]:
                    name_to_id[str(row["name"]).strip().lower()] = int(row["player_id"])

            updated = 0
            found_cols = [c for c in ("stuff_plus", "location_plus", "pitching_plus", "gmli") if c in fg_df.columns]

            # FanGraphs "Name" column contains the pitcher name
            name_col = "Name" if "Name" in fg_df.columns else None
            if name_col is None:
                # Try lowercase
                for c in fg_df.columns:
                    if c.lower() == "name":
                        name_col = c
                        break

            if name_col is None:
                logger.warning("No 'Name' column in FanGraphs data")
                return "Skipped: no Name column in FanGraphs data"

            for _, row in fg_df.iterrows():
                fg_name = str(row[name_col]).strip().lower()
                pid = name_to_id.get(fg_name)
                if pid is None:
                    continue

                # Build SET clause for available columns
                set_parts = []
                values = []
                for col in found_cols:
                    val = row.get(col)
                    if pd.notna(val):
                        set_parts.append(f"{col} = ?")
                        values.append(float(val))

                if not set_parts:
                    continue

                # Update season_stats
                values_ss = values + [pid, year]
                conn.execute(
                    f"UPDATE season_stats SET {', '.join(set_parts)} WHERE player_id = ? AND season = ?",
                    values_ss,
                )

                # Update statcast_archive (upsert: insert if missing)
                existing = conn.execute(
                    "SELECT 1 FROM statcast_archive WHERE player_id = ? AND season = ?",
                    (pid, year),
                ).fetchone()
                if existing:
                    values_sa = values + [pid, year]
                    conn.execute(
                        f"UPDATE statcast_archive SET {', '.join(set_parts)} WHERE player_id = ? AND season = ?",
                        values_sa,
                    )
                else:
                    insert_cols = ["player_id", "season"] + found_cols
                    placeholders = ", ".join(["?"] * len(insert_cols))
                    insert_vals = [pid, year] + [
                        float(row.get(c)) if pd.notna(row.get(c)) else None for c in found_cols
                    ]
                    conn.execute(
                        f"INSERT INTO statcast_archive ({', '.join(insert_cols)}) VALUES ({placeholders})",
                        insert_vals,
                    )

                updated += 1

            conn.commit()
        finally:
            conn.close()

        update_refresh_log("stuff_plus", "success")
        logger.info("Stuff+ metrics: updated %d pitchers from %d FanGraphs rows", updated, len(fg_df))
        return f"Updated {updated} pitchers with Stuff+/Location+/Pitching+"

    except Exception as exc:
        logger.exception("Stuff+ bootstrap failed: %s", exc)
        try:
            from src.database import update_refresh_log

            update_refresh_log(
                "stuff_plus",
                _classify_fetch_error(exc),
                message=_format_fetch_error(exc, "FanGraphs Stuff+"),
            )
        except Exception:
            pass
        return _format_fetch_error(exc, "FanGraphs Stuff+")


def _bootstrap_batting_stats(progress: BootstrapProgress) -> str:
    """Phase 23: Fetch advanced batting stats (BABIP, ISO, K%, BB%, etc.) from FanGraphs."""
    progress.phase = "Batting Stats"
    progress.detail = "Fetching BABIP/ISO/K%/BB% from FanGraphs..."
    try:
        from pybaseball import batting_stats
    except ImportError:
        logger.warning("pybaseball not installed — skipping batting stats fetch")
        return "Skipped: pybaseball not installed"

    try:
        import pandas as pd

        from src.database import get_connection, update_refresh_log

        year = datetime.now(UTC).year
        logger.info("Fetching FanGraphs batting stats for %d (qual=0)...", year)
        fg_df = batting_stats(year, qual=0)

        if fg_df is None or fg_df.empty:
            logger.warning("pybaseball batting_stats returned empty data")
            return "Skipped: no data returned"

        # Map FanGraphs column names to DB column names
        col_map = {}
        for col in fg_df.columns:
            cl = col.lower().replace(" ", "").replace("_", "")
            if cl == "babip":
                col_map[col] = "babip"
            elif cl == "iso":
                col_map[col] = "iso"
            elif cl in ("k%", "kpct", "k_pct"):
                col_map[col] = "hitter_k_pct"
            elif cl in ("bb%", "bbpct", "bb_pct"):
                col_map[col] = "hitter_bb_pct"
            elif cl in ("ld%", "ldpct", "ld_pct"):
                col_map[col] = "ld_pct"
            elif cl in ("fb%", "fbpct", "fb_pct"):
                col_map[col] = "hitter_fb_pct"
            elif cl in ("gb%", "gbpct", "gb_pct"):
                col_map[col] = "hitter_gb_pct"

        if not col_map:
            logger.warning(
                "No batting advanced stat columns found. Columns: %s",
                list(fg_df.columns)[:30],
            )
            return "Skipped: batting stat columns not in FanGraphs data"

        logger.info("Found FanGraphs batting columns: %s", col_map)
        fg_df = fg_df.rename(columns=col_map)

        # Convert percentage strings (e.g., "25.3 %") to floats
        pct_cols = ["hitter_k_pct", "hitter_bb_pct", "ld_pct", "hitter_fb_pct", "hitter_gb_pct"]
        for pcol in pct_cols:
            if pcol in fg_df.columns:
                fg_df[pcol] = (
                    fg_df[pcol].astype(str).str.replace("%", "", regex=False).str.replace(" ", "", regex=False)
                )
                fg_df[pcol] = pd.to_numeric(fg_df[pcol], errors="coerce")

        # Name→player_id lookup (hitters only)
        conn = get_connection()
        try:
            players_df = pd.read_sql(
                "SELECT player_id, name FROM players WHERE is_hitter = 1",
                conn,
            )
            name_to_id = {}
            for _, row in players_df.iterrows():
                if row["name"]:
                    name_to_id[str(row["name"]).strip().lower()] = int(row["player_id"])

            # Find the Name column
            name_col = None
            for c in fg_df.columns:
                if c.lower() == "name":
                    name_col = c
                    break
            if name_col is None:
                logger.warning("No 'Name' column in FanGraphs batting data")
                return "Skipped: no Name column"

            target_cols = [c for c in col_map.values() if c in fg_df.columns]
            updated = 0

            for _, row in fg_df.iterrows():
                fg_name = str(row[name_col]).strip().lower()
                pid = name_to_id.get(fg_name)
                if pid is None:
                    continue

                set_parts = []
                values = []
                for col in target_cols:
                    val = row.get(col)
                    if pd.notna(val):
                        set_parts.append(f"{col} = ?")
                        values.append(float(val))

                if not set_parts:
                    continue

                # Upsert statcast_archive
                existing = conn.execute(
                    "SELECT 1 FROM statcast_archive WHERE player_id = ? AND season = ?",
                    (pid, year),
                ).fetchone()
                if existing:
                    conn.execute(
                        f"UPDATE statcast_archive SET {', '.join(set_parts)} WHERE player_id = ? AND season = ?",
                        values + [pid, year],
                    )
                else:
                    insert_cols = ["player_id", "season"] + target_cols
                    placeholders = ", ".join(["?"] * len(insert_cols))
                    insert_vals = [pid, year] + [
                        float(row.get(c)) if pd.notna(row.get(c)) else None for c in target_cols
                    ]
                    conn.execute(
                        f"INSERT INTO statcast_archive ({', '.join(insert_cols)}) VALUES ({placeholders})",
                        insert_vals,
                    )
                updated += 1

            conn.commit()
        finally:
            conn.close()

        update_refresh_log("batting_stats", "success")
        logger.info("Batting stats: updated %d hitters from %d FanGraphs rows", updated, len(fg_df))
        return f"Updated {updated} hitters with BABIP/ISO/K%%/BB%%"

    except Exception as exc:
        logger.exception("Batting stats bootstrap failed: %s", exc)
        try:
            from src.database import update_refresh_log

            update_refresh_log(
                "batting_stats",
                _classify_fetch_error(exc),
                message=_format_fetch_error(exc, "FanGraphs Batting Stats"),
            )
        except Exception:
            pass
        return _format_fetch_error(exc, "FanGraphs Batting Stats")


def _bootstrap_sprint_speed(progress: BootstrapProgress) -> str:
    """Phase 24: Fetch Statcast sprint speed data."""
    progress.phase = "Sprint Speed"
    progress.detail = "Fetching sprint speed from Statcast..."
    try:
        from pybaseball import statcast_sprint_speed
    except ImportError:
        logger.warning("pybaseball not installed — skipping sprint speed fetch")
        return "Skipped: pybaseball not installed"

    try:
        import pandas as pd

        from src.database import get_connection, update_refresh_log

        year = datetime.now(UTC).year
        logger.info("Fetching Statcast sprint speed for %d...", year)
        ss_df = statcast_sprint_speed(year, min_opp=5)

        if ss_df is None or ss_df.empty:
            logger.warning("Sprint speed returned empty data")
            return "Skipped: no data returned"

        # Find the sprint speed and player name columns
        speed_col = None
        name_col = None
        for c in ss_df.columns:
            cl = c.lower().replace(" ", "").replace("_", "")
            if cl in ("hpsprint", "sprintspeed", "sprint_speed"):
                speed_col = c
            elif cl in ("last_name,first_name", "player_name", "name"):
                name_col = c
        # Fallback: common column names from pybaseball
        if speed_col is None and "hp_to_1b" in ss_df.columns:
            speed_col = "hp_to_1b"
        if speed_col is None:
            for c in ss_df.columns:
                if "sprint" in c.lower() or "speed" in c.lower():
                    speed_col = c
                    break
        if name_col is None:
            for c in ss_df.columns:
                if "name" in c.lower():
                    name_col = c
                    break

        if speed_col is None or name_col is None:
            logger.warning("Sprint speed columns not found. Columns: %s", list(ss_df.columns)[:20])
            return "Skipped: sprint speed columns not found"

        logger.info("Sprint speed columns: name=%s, speed=%s", name_col, speed_col)

        # Name→player_id lookup (hitters)
        conn = get_connection()
        try:
            players_df = pd.read_sql(
                "SELECT player_id, name FROM players WHERE is_hitter = 1",
                conn,
            )
            name_to_id = {}
            for _, row in players_df.iterrows():
                if row["name"]:
                    name_to_id[str(row["name"]).strip().lower()] = int(row["player_id"])

            updated = 0
            for _, row in ss_df.iterrows():
                raw_name = str(row[name_col]).strip()
                # Statcast uses "Last, First" format — convert to "First Last"
                if "," in raw_name:
                    parts = raw_name.split(",", 1)
                    raw_name = f"{parts[1].strip()} {parts[0].strip()}"
                pid = name_to_id.get(raw_name.lower())
                if pid is None:
                    continue

                speed = row.get(speed_col)
                if pd.isna(speed):
                    continue
                speed = float(speed)

                existing = conn.execute(
                    "SELECT 1 FROM statcast_archive WHERE player_id = ? AND season = ?",
                    (pid, year),
                ).fetchone()
                if existing:
                    conn.execute(
                        "UPDATE statcast_archive SET sprint_speed = ? WHERE player_id = ? AND season = ?",
                        (speed, pid, year),
                    )
                else:
                    conn.execute(
                        "INSERT INTO statcast_archive (player_id, season, sprint_speed) VALUES (?, ?, ?)",
                        (pid, year, speed),
                    )
                updated += 1

            conn.commit()
        finally:
            conn.close()

        update_refresh_log("sprint_speed", "success")
        logger.info("Sprint speed: updated %d players from %d Statcast rows", updated, len(ss_df))
        return f"Updated {updated} players with sprint speed"

    except Exception as exc:
        logger.exception("Sprint speed bootstrap failed: %s", exc)
        try:
            from src.database import update_refresh_log

            update_refresh_log("sprint_speed", "error")
        except Exception:
            pass
        return f"Error: {exc}"


def _bootstrap_umpire_tendencies(progress: BootstrapProgress) -> str:
    """T7: Fetch umpire assignments and build per-umpire tendency table.

    Uses MLB Stats API schedule to find today's umpire assignments,
    then computes per-umpire K%/BB%/run environment from historical game data.
    """
    progress.phase = "Umpire Data"
    progress.detail = "Fetching umpire assignments..."

    try:
        import statsapi as _statsapi
    except ImportError:
        return "Skipped: statsapi not installed"

    try:
        from src.database import get_connection, update_refresh_log

        year = datetime.now(UTC).year

        # Fetch all games for the season to build umpire tendency profiles
        logger.info("Fetching MLB schedule for %d to build umpire profiles...", year)
        schedule = _statsapi.schedule(
            start_date=f"{year}-03-20",
            end_date=datetime.now(UTC).strftime("%Y-%m-%d"),
        )

        if not schedule:
            # 2026-04-17 FIX: previously silently returned with no refresh_log
            # write, so the phase appeared "never ran" in Data Status.
            update_refresh_log(
                "umpire_tendencies",
                "no_data",
                rows_written=0,
                message="MLB schedule fetch returned empty",
            )
            return "Skipped: no schedule data"

        # Aggregate per-umpire stats from game data
        umpire_stats: dict[str, dict] = {}  # name -> {games, total_k, total_bb, total_runs, total_pa}
        for game in schedule:
            game_pk = game.get("game_id")
            if not game_pk:
                continue
            # Only completed games
            status = game.get("status", "")
            if "Final" not in status and "Completed" not in status:
                continue

            try:
                boxscore = _statsapi.boxscore_data(game_pk)
            except Exception:
                continue

            if not boxscore:
                continue

            # Extract home plate umpire from game info
            game_info = boxscore.get("gameBoxInfo", [])
            hp_umpire = None
            for info_item in game_info:
                label = str(info_item.get("label", ""))
                value = str(info_item.get("value", ""))
                if "HP" in label or "Home Plate" in label:
                    hp_umpire = value.strip()
                    break

            if not hp_umpire:
                continue

            # Extract game scoring/strikeout data from team stats
            away_stats = boxscore.get("awayBatting", {})
            home_stats = boxscore.get("homeBatting", {})

            total_k = 0
            total_bb = 0
            total_runs = 0
            total_pa = 0

            for team_stats in [away_stats, home_stats]:
                if isinstance(team_stats, dict):
                    # Team totals row
                    totals = team_stats.get("teamStats", {})
                    if isinstance(totals, dict):
                        batting = totals.get("batting", {})
                        total_k += int(batting.get("strikeOuts", 0))
                        total_bb += int(batting.get("baseOnBalls", 0))
                        total_runs += int(batting.get("runs", 0))
                        total_pa += int(batting.get("plateAppearances", batting.get("atBats", 0)))

            if total_pa < 30:
                continue

            if hp_umpire not in umpire_stats:
                umpire_stats[hp_umpire] = {
                    "games": 0,
                    "total_k": 0,
                    "total_bb": 0,
                    "total_runs": 0,
                    "total_pa": 0,
                }

            ump = umpire_stats[hp_umpire]
            ump["games"] += 1
            ump["total_k"] += total_k
            ump["total_bb"] += total_bb
            ump["total_runs"] += total_runs
            ump["total_pa"] += total_pa

        if not umpire_stats:
            # 2026-04-17 FIX: previously silently returned with no log write.
            update_refresh_log(
                "umpire_tendencies",
                "no_data",
                rows_written=0,
                message="no umpire names extracted from completed-game boxscores",
            )
            return "Skipped: no umpire data extracted"

        # Compute league averages for delta calculation
        league_k = sum(u["total_k"] for u in umpire_stats.values())
        league_bb = sum(u["total_bb"] for u in umpire_stats.values())
        league_runs = sum(u["total_runs"] for u in umpire_stats.values())
        league_pa = sum(u["total_pa"] for u in umpire_stats.values())
        league_games = sum(u["games"] for u in umpire_stats.values())

        avg_k_pct = league_k / max(1, league_pa)
        avg_bb_pct = league_bb / max(1, league_pa)
        avg_rpg = league_runs / max(1, league_games)

        # Write to DB
        conn = get_connection()
        try:
            now = datetime.now(UTC).isoformat()
            updated = 0
            for name, stats in umpire_stats.items():
                if stats["games"] < 3:
                    continue  # Need min sample size
                k_pct = stats["total_k"] / max(1, stats["total_pa"])
                bb_pct = stats["total_bb"] / max(1, stats["total_pa"])
                rpg = stats["total_runs"] / max(1, stats["games"])

                conn.execute(
                    """INSERT OR REPLACE INTO umpire_tendencies
                       (umpire_name, games_umped, k_pct, bb_pct, runs_per_game,
                        k_pct_delta, bb_pct_delta, run_env_delta, season, fetched_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        name,
                        stats["games"],
                        round(k_pct, 4),
                        round(bb_pct, 4),
                        round(rpg, 2),
                        round(k_pct - avg_k_pct, 4),
                        round(bb_pct - avg_bb_pct, 4),
                        round(rpg - avg_rpg, 2),
                        year,
                        now,
                    ),
                )
                updated += 1
            conn.commit()
        finally:
            conn.close()

        try:
            from src.database import update_refresh_log_auto

            status = update_refresh_log_auto(
                "umpire_tendencies",
                updated,
                expected_min=10,
                message=f"{updated} umpires from {league_games} games",
            )
        except ImportError:
            update_refresh_log("umpire_tendencies", "success")
            status = "success"
        logger.info("T7: Umpire tendencies — %d umpires from %d games", updated, league_games)
        return f"Saved {updated} umpire profiles from {league_games} games ({status})"

    except Exception as exc:
        logger.exception("T7 umpire tendencies failed: %s", exc)
        try:
            from src.database import update_refresh_log

            update_refresh_log("umpire_tendencies", "error", message=str(exc)[:200])
        except Exception:
            pass
        return f"Error: {exc}"


def _bootstrap_catcher_framing(progress: BootstrapProgress) -> str:
    """T8: Fetch catcher framing runs and pop time from pybaseball.

    Uses Baseball Savant catcher framing leaderboards via pybaseball.
    """
    progress.phase = "Catcher Framing"
    progress.detail = "Fetching catcher framing + pop time..."

    try:
        import pybaseball  # noqa: F401
    except ImportError:
        return "Skipped: pybaseball not installed"

    try:
        import pandas as pd

        from src.database import get_connection, update_refresh_log

        year = datetime.now(UTC).year

        # 2026-04-17 FIX: swapped primary path to Baseball Savant's
        # statcast_catcher_framing (pybaseball), which hits a different
        # endpoint than the 403-blocked FanGraphs leaders-legacy.aspx.
        # The former FanGraphs primary (batting_stats(pos="c")) is now the
        # last fallback and is expected to 403 — it's kept only in case
        # FanGraphs restores the endpoint.
        framing_data = None
        try:
            from pybaseball import statcast_catcher_framing

            sv_df = statcast_catcher_framing(year)
            if sv_df is not None and not sv_df.empty:
                framing_data = sv_df
                logger.info(
                    "T8: Got %d catchers from Baseball Savant statcast_catcher_framing",
                    len(sv_df),
                )
        except Exception as e:
            logger.warning("T8: Savant catcher framing failed: %s", e)

        if framing_data is None:
            try:
                from pybaseball import batting_stats

                fg_df = batting_stats(year, qual=0, pos="c")
                if fg_df is not None and not fg_df.empty:
                    framing_data = fg_df
                    logger.info("T8: Got %d catchers from FanGraphs batting_stats", len(fg_df))
            except Exception as e:
                logger.warning("T8: FanGraphs catcher stats failed: %s", e)

        if framing_data is None:
            # Fallback: use statsapi catcher stats
            try:
                import statsapi as _statsapi

                # Get catchers from our DB and fetch their fielding stats
                conn_temp = get_connection()
                try:
                    catchers = pd.read_sql(
                        "SELECT player_id, name, mlb_id FROM players WHERE positions LIKE '%C%' AND is_hitter = 1",
                        conn_temp,
                    )
                finally:
                    conn_temp.close()

                if catchers.empty:
                    update_refresh_log(
                        "catcher_framing",
                        "no_data",
                        rows_written=0,
                        message="no catchers in players table",
                    )
                    return "Skipped: no catchers in DB"

                rows = []
                for _, c in catchers.iterrows():
                    mlb_id = c.get("mlb_id")
                    if pd.isna(mlb_id) or mlb_id is None:
                        continue
                    try:
                        mlb_id = int(mlb_id)
                        stats = _statsapi.player_stat_data(mlb_id, group="fielding", type="season")
                        if stats and "stats" in stats:
                            for stat_group in stats["stats"]:
                                for split in stat_group.get("stats", []):
                                    if split.get("position", {}).get("abbreviation") == "C":
                                        rows.append(
                                            {
                                                "player_id": int(c["player_id"]),
                                                "games": int(split.get("gamesPlayed", 0)),
                                                "cs_pct": float(split.get("caughtStealingPct", 0)) / 100.0
                                                if split.get("caughtStealingPct")
                                                else 0.0,
                                                "pop_time": 0.0,  # Not available from statsapi
                                                "framing_runs": 0.0,
                                            }
                                        )
                    except Exception:
                        continue

                framing_data = pd.DataFrame(rows) if rows else None
                if framing_data is not None:
                    logger.info("T8: Got %d catchers from statsapi fielding", len(framing_data))
            except Exception as e:
                logger.warning("T8: statsapi catcher fallback failed: %s", e)

        if framing_data is None or framing_data.empty:
            update_refresh_log(
                "catcher_framing",
                "no_data",
                rows_written=0,
                message="all framing sources returned empty (Savant + FanGraphs + statsapi)",
            )
            return "Skipped: no catcher data available"

        # Build name→player_id mapping
        conn = get_connection()
        try:
            players_df = pd.read_sql(
                "SELECT player_id, name FROM players WHERE positions LIKE '%C%' AND is_hitter = 1",
                conn,
            )
            name_to_id = {}
            for _, row in players_df.iterrows():
                if row["name"]:
                    name_to_id[str(row["name"]).strip().lower()] = int(row["player_id"])

            now = datetime.now(UTC).isoformat()
            updated = 0

            for _, row in framing_data.iterrows():
                # Try to find player_id from name or direct ID
                pid = row.get("player_id")
                if pid is None or pd.isna(pid):
                    name_val = row.get("Name", row.get("name", ""))
                    if name_val:
                        pid = name_to_id.get(str(name_val).strip().lower())
                if pid is None:
                    continue
                pid = int(pid)

                framing_runs = float(row.get("framing_runs", row.get("FRM", row.get("Framing", 0.0))) or 0.0)
                pop_time_val = float(row.get("pop_time", row.get("Pop Time", 0.0)) or 0.0)
                cs_pct_val = float(row.get("cs_pct", row.get("CS%", row.get("CaughtStealing%", 0.0))) or 0.0)
                games_val = int(row.get("games", row.get("G", row.get("gamesPlayed", 0))) or 0)

                framing_rpg = framing_runs / max(1, games_val)

                conn.execute(
                    """INSERT OR REPLACE INTO catcher_framing
                       (player_id, season, framing_runs, framing_runs_per_game,
                        pop_time, cs_pct, games_caught, fetched_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        pid,
                        year,
                        round(framing_runs, 2),
                        round(framing_rpg, 4),
                        round(pop_time_val, 3),
                        round(cs_pct_val, 4),
                        games_val,
                        now,
                    ),
                )
                updated += 1

            conn.commit()
        finally:
            conn.close()

        try:
            from src.database import update_refresh_log_auto

            status = update_refresh_log_auto(
                "catcher_framing",
                updated,
                expected_min=10,
                message=f"{updated} catchers from {len(framing_data)} rows",
            )
        except ImportError:
            update_refresh_log("catcher_framing", "success")
            status = "success"
        logger.info("T8: Catcher framing — %d catchers updated", updated)
        return f"Saved {updated} catcher framing profiles ({status})"

    except Exception as exc:
        logger.exception("T8 catcher framing failed: %s", exc)
        try:
            from src.database import update_refresh_log

            update_refresh_log(
                "catcher_framing",
                _classify_fetch_error(exc),
                message=str(exc)[:200],
            )
        except Exception:
            pass
        return f"Error: {exc}"


def _bootstrap_pvb_splits(progress: BootstrapProgress) -> str:
    """T12: Fetch pitcher-vs-batter splits for rostered players.

    Uses pybaseball.statcast_batter for specific batter-pitcher matchups.
    Only fetches for rostered hitters vs upcoming opposing pitchers.
    PvB stats stabilize at ~60 PA — cache aggressively.
    """
    progress.phase = "PvB Splits"
    progress.detail = "Fetching pitcher-batter matchup history..."

    try:
        from pybaseball import statcast_batter as _statcast_batter
    except ImportError:
        return "Skipped: pybaseball not installed"

    try:
        import pandas as pd

        from src.database import get_connection, update_refresh_log

        year = datetime.now(UTC).year

        # Get rostered hitter MLB IDs and upcoming pitcher MLB IDs
        conn = get_connection()
        try:
            rostered_hitters = pd.read_sql(
                """SELECT DISTINCT p.player_id, p.mlb_id, p.name
                   FROM players p
                   JOIN league_rosters lr ON p.player_id = lr.player_id
                   WHERE p.is_hitter = 1 AND p.mlb_id IS NOT NULL""",
                conn,
            )
            # Get opposing pitcher IDs from opp_pitcher_stats
            opp_pitchers = pd.read_sql(
                "SELECT pitcher_id, name FROM opp_pitcher_stats WHERE season = ?",
                conn,
                params=(year,),
            )
        finally:
            conn.close()

        if rostered_hitters.empty:
            return "Skipped: no rostered hitters with MLB IDs"

        if opp_pitchers.empty:
            return "Skipped: no opposing pitcher data"

        # Limit to avoid API hammering (top 50 hitters x top 30 pitchers = 1500 lookups max)
        max_hitters = min(50, len(rostered_hitters))
        max_pitchers = min(30, len(opp_pitchers))
        hitter_sample = rostered_hitters.head(max_hitters)
        pitcher_ids = opp_pitchers["pitcher_id"].head(max_pitchers).tolist()

        conn = get_connection()
        try:
            now = datetime.now(UTC).isoformat()
            updated = 0
            skipped = 0

            for idx, (_, hitter) in enumerate(hitter_sample.iterrows()):
                batter_mlb_id = int(hitter["mlb_id"])
                batter_pid = int(hitter["player_id"])

                # Check which pitchers still need data for this batter
                uncached_pitcher_ids = []
                for pid in pitcher_ids:
                    pid = int(pid)
                    existing = conn.execute(
                        """SELECT fetched_at FROM pvb_splits
                           WHERE batter_id = ? AND pitcher_id = ?""",
                        (batter_pid, pid),
                    ).fetchone()
                    if existing:
                        skipped += 1
                    else:
                        uncached_pitcher_ids.append(pid)

                if not uncached_pitcher_ids:
                    continue  # All pitchers cached for this batter

                # Fetch this batter's Statcast data ONCE (not per-pitcher!)
                progress.detail = f"PvB splits: batter {idx + 1}/{max_hitters}..."
                try:
                    all_batter_data = _statcast_batter(
                        f"{year - 3}-01-01",
                        f"{year}-12-31",
                        batter_mlb_id,
                    )
                    if all_batter_data is None or all_batter_data.empty:
                        continue
                except Exception:
                    continue

                # Now filter for each uncached pitcher from the single fetch
                for pitcher_mlb_id in uncached_pitcher_ids:
                    pvb = all_batter_data[all_batter_data["pitcher"] == pitcher_mlb_id]

                    if pvb is None or pvb.empty:
                        continue

                    pa_count = len(pvb)
                    if pa_count < 3:
                        continue  # Too few PA to be meaningful

                    # Aggregate PvB stats
                    events = pvb["events"].dropna()
                    hits = events.isin(["single", "double", "triple", "home_run"]).sum()
                    hrs = events.isin(["home_run"]).sum()
                    walks = events.isin(["walk"]).sum()
                    strikeouts = events.isin(["strikeout", "strikeout_double_play"]).sum()
                    ab = (
                        pa_count
                        - walks
                        - events.isin(["hit_by_pitch", "sac_fly", "sac_bunt", "sac_fly_double_play"]).sum()
                    )

                    avg_val = hits / max(1, ab)
                    obp_val = (hits + walks) / max(1, pa_count)
                    slg_val = 0.0  # Simplified — would need total bases
                    woba_est = (
                        pvb["estimated_woba_using_speedangle"].mean()
                        if "estimated_woba_using_speedangle" in pvb.columns
                        else 0.0
                    )
                    if pd.isna(woba_est):
                        woba_est = 0.0

                    conn.execute(
                        """INSERT OR REPLACE INTO pvb_splits
                           (batter_id, pitcher_id, pa, avg, obp, slg, hr, k, bb, woba, fetched_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            batter_pid,
                            pitcher_mlb_id,
                            pa_count,
                            round(avg_val, 3),
                            round(obp_val, 3),
                            round(slg_val, 3),
                            int(hrs),
                            int(strikeouts),
                            int(walks),
                            round(woba_est, 3),
                            now,
                        ),
                    )
                    updated += 1

            conn.commit()
        finally:
            conn.close()

        update_refresh_log("pvb_splits", "success")
        logger.info("T12: PvB splits — %d matchups saved, %d skipped (cached)", updated, skipped)
        return f"Saved {updated} PvB matchups ({skipped} cached)"

    except Exception as exc:
        logger.exception("T12 PvB splits failed: %s", exc)
        try:
            from src.database import update_refresh_log

            update_refresh_log("pvb_splits", "error")
        except Exception:
            pass
        return f"Error: {exc}"


def _bootstrap_dynamic_park_factors(progress: BootstrapProgress) -> str:
    """T4: Refresh park factors mid-season from pybaseball team stats."""
    progress.phase = "Park Factors"
    progress.detail = "Refreshing park factors from pybaseball..."
    try:
        from pybaseball import team_batting
    except ImportError:
        return "Skipped: pybaseball not installed"

    try:
        from src.database import get_connection, update_refresh_log

        year = datetime.now(UTC).year
        logger.info("Fetching team batting stats for %d park factor refresh...", year)
        tb = team_batting(year)
        if tb is None or tb.empty:
            return "Skipped: no team batting data"

        # Park factor derivation: compare home/away OPS splits
        # FanGraphs team_batting includes team abbreviation
        conn = get_connection()
        try:
            updated = 0
            for _, row in tb.iterrows():
                team = str(row.get("Team", row.get("Tm", ""))).strip()
                if not team:
                    continue
                # Use OPS+ as park factor proxy (100 = neutral)
                ops_plus = float(row.get("OPS+", row.get("wRC+", 100)))
                if ops_plus <= 0:
                    continue
                pf = ops_plus / 100.0
                # Clamp to reasonable range
                pf = max(0.80, min(1.40, pf))
                conn.execute(
                    "UPDATE park_factors SET factor_hitting = ? WHERE team_code = ?",
                    (round(pf, 3), team),
                )
                updated += 1
            conn.commit()
        finally:
            conn.close()

        update_refresh_log("park_factors_dynamic", "success")
        return f"Updated {updated} park factors from {year} team stats"

    except Exception as exc:
        logger.warning("Dynamic park factor refresh failed (non-fatal): %s", exc)
        return _format_fetch_error(exc, "FanGraphs Park Factors")


def _bootstrap_bat_speed(progress: BootstrapProgress) -> str:
    """T9: Fetch bat speed data from Baseball Savant."""
    progress.phase = "Bat Speed"
    progress.detail = "Fetching bat speed from Statcast..."
    try:
        from pybaseball import statcast_batter_bat_tracking
    except ImportError:
        return "Skipped: pybaseball bat tracking not available"

    try:
        import pandas as pd

        from src.database import get_connection, update_refresh_log

        year = datetime.now(UTC).year
        logger.info("Fetching Statcast bat speed for %d...", year)
        bt = statcast_batter_bat_tracking(year)

        if bt is None or bt.empty:
            return "Skipped: no bat speed data"

        # Find bat speed column
        speed_col = None
        name_col = None
        for c in bt.columns:
            cl = c.lower().replace(" ", "").replace("_", "")
            if cl in ("avgbatspeed", "batspeed", "bat_speed"):
                speed_col = c
            elif "name" in c.lower():
                name_col = c
        if speed_col is None or name_col is None:
            logger.warning("Bat speed columns not found. Columns: %s", list(bt.columns)[:20])
            return "Skipped: bat speed columns not found"

        conn = get_connection()
        try:
            players_df = pd.read_sql("SELECT player_id, name FROM players WHERE is_hitter = 1", conn)
            name_to_id = {}
            for _, row in players_df.iterrows():
                if row["name"]:
                    name_to_id[str(row["name"]).strip().lower()] = int(row["player_id"])

            updated = 0
            for _, row in bt.iterrows():
                raw_name = str(row[name_col]).strip()
                if "," in raw_name:
                    parts = raw_name.split(",", 1)
                    raw_name = f"{parts[1].strip()} {parts[0].strip()}"
                pid = name_to_id.get(raw_name.lower())
                if pid is None:
                    continue
                speed = row.get(speed_col)
                if pd.isna(speed):
                    continue

                existing = conn.execute(
                    "SELECT 1 FROM statcast_archive WHERE player_id = ? AND season = ?",
                    (pid, year),
                ).fetchone()
                if existing:
                    conn.execute(
                        "UPDATE statcast_archive SET bat_speed = ? WHERE player_id = ? AND season = ?",
                        (float(speed), pid, year),
                    )
                else:
                    conn.execute(
                        "INSERT INTO statcast_archive (player_id, season, bat_speed) VALUES (?, ?, ?)",
                        (pid, year, float(speed)),
                    )
                updated += 1
            conn.commit()
        finally:
            conn.close()

        update_refresh_log("bat_speed", "success")
        return f"Updated {updated} players with bat speed"

    except Exception as exc:
        logger.warning("Bat speed fetch failed (non-fatal): %s", exc)
        return f"Error: {exc}"


def _bootstrap_forty_man(progress: BootstrapProgress) -> str:
    """T10: Fetch 40-man roster status from MLB Stats API."""
    progress.phase = "40-Man Rosters"
    progress.detail = "Fetching 40-man roster status..."
    try:
        import statsapi
    except ImportError:
        return "Skipped: statsapi not installed"

    try:
        from src.database import get_connection, update_refresh_log

        conn = get_connection()
        try:
            # Get all team IDs
            teams = statsapi.get("teams", {"sportId": 1}, request_kwargs={"timeout": 30}).get("teams", [])
            updated = 0
            for team in teams:
                tid = team.get("id")
                if not tid:
                    continue
                try:
                    roster_data = statsapi.get(
                        "team_roster", {"teamId": tid, "rosterType": "40Man"}, request_kwargs={"timeout": 30}
                    )
                    for entry in roster_data.get("roster", []):
                        person = entry.get("person", {})
                        mlb_id = person.get("id")
                        if not mlb_id:
                            continue
                        # Mark as on_40_man in players table
                        conn.execute(
                            "UPDATE players SET roster_type = '40man' WHERE mlb_id = ?",
                            (mlb_id,),
                        )
                        updated += 1
                except Exception:
                    continue
            conn.commit()
        finally:
            conn.close()

        update_refresh_log("forty_man", "success")
        return f"Updated {updated} 40-man roster entries"

    except Exception as exc:
        logger.warning("40-man roster fetch failed (non-fatal): %s", exc)
        return f"Error: {exc}"


def _bootstrap_game_logs(progress: BootstrapProgress, force: bool = False) -> str:
    """Phase 31: Fetch per-game logs from MLB Stats API for Player Databank.

    Fetches the current season (2026) game logs and, when historical data is
    absent, also back-fills 2025 and 2024.  All work is delegated to
    ``fetch_game_logs_from_api`` which handles INSERT OR REPLACE internally.

    2026-04-17 audit: uses update_refresh_log_auto so a 0-row fetch now
    writes 'no_data' instead of the silent 'success' that hid the
    statsapi.player_stat_data() wrapper bug for weeks.
    """
    progress.phase = "Game Logs"
    progress.detail = "Fetching per-game stats from MLB Stats API..."
    # Expected floor: ~315 rostered players × 10+ games-so-far by mid-April.
    # Conservative floor of 200 rows accepts early-season sparsity while
    # still surfacing totally-empty fetches as 'partial' / 'no_data'.
    EXPECTED_MIN = 200
    try:
        from src.database import (
            get_connection,
            update_refresh_log,
            update_refresh_log_auto,
        )
        from src.player_databank import fetch_game_logs_from_api

        current_year = datetime.now(UTC).year

        # Current season
        progress.detail = f"Fetching {current_year} game logs..."
        count_current = fetch_game_logs_from_api(season=current_year, force=force)

        # Back-fill historical seasons if the game_logs table is sparse.
        # Trigger when fewer than 100 rows exist for a season (was: strictly 0)
        # so a partially-failed prior run doesn't permanently short-circuit
        # the backfill path.
        hist_counts: dict[int, int] = {}
        for hist_year in (current_year - 1, current_year - 2):
            conn = get_connection()
            try:
                row = conn.execute("SELECT COUNT(*) FROM game_logs WHERE season = ?", (hist_year,)).fetchone()
                existing = int(row[0]) if row else 0
            except Exception:
                existing = 0
            finally:
                conn.close()

            if existing < 100 or force:
                progress.detail = f"Back-filling {hist_year} game logs..."
                hist_counts[hist_year] = fetch_game_logs_from_api(season=hist_year, force=force)

        total_current_plus_hist = count_current + sum(hist_counts.values())
        status = update_refresh_log_auto(
            "game_logs",
            total_current_plus_hist,
            expected_min=EXPECTED_MIN,
            message=(f"{current_year}={count_current}, " + ", ".join(f"{y}={c}" for y, c in hist_counts.items())),
        )

        parts = [f"{current_year}: {count_current} rows"]
        for yr, cnt in hist_counts.items():
            parts.append(f"{yr}: {cnt} rows")
        return f"Game logs ({status}) — {', '.join(parts)}"

    except Exception as exc:
        logger.warning("Game logs bootstrap failed (non-fatal): %s", exc)
        try:
            from src.database import update_refresh_log

            update_refresh_log("game_logs", "error", message=str(exc)[:200])
        except Exception:
            pass
        return f"Game logs: error ({exc})"


# ── Master Orchestrator ──────────────────────────────────────────────


def bootstrap_all_data(
    yahoo_client=None,
    on_progress: Callable[[BootstrapProgress], None] | None = None,
    force: bool = False,
    staleness: StalenessConfig | None = None,
) -> dict[str, str]:
    """Master orchestrator: refresh all data sources based on staleness.

    Args:
        yahoo_client: Optional YahooFantasyClient instance
        on_progress: Callback for splash screen updates
        force: If True, refresh everything regardless of staleness
        staleness: Custom staleness thresholds (uses defaults if None)

    Returns:
        Dict of {source: result_message}
    """
    from src.database import check_staleness, get_connection

    if staleness is None:
        staleness = StalenessConfig()
    progress = BootstrapProgress()
    results = {}

    def _notify(pct: float):
        progress.pct = pct
        if on_progress:
            on_progress(progress)

    # Phase 1: Players (must come first — other phases need player_ids)
    _notify(0.0)
    if force or check_staleness("players", staleness.players_hours):
        try:
            results["players"] = _run_with_timeout(
                lambda: _bootstrap_players(progress),
                timeout=_PHASE_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            logger.warning("Players bootstrap timed out or failed: %s", exc)
            results["players"] = f"Error: {exc}"
    else:
        results["players"] = "Fresh"

    # Phases 2+3: Park factors + Projections (parallel — both independent after players)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    _notify(0.15)
    pf_stale = force or check_staleness("park_factors", staleness.park_factors_hours)
    proj_stale = force or check_staleness("projections", staleness.projections_hours)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}
        if pf_stale:
            futures[executor.submit(_bootstrap_park_factors, progress)] = "park_factors"
        else:
            results["park_factors"] = "Fresh"
        if proj_stale:
            futures[executor.submit(_bootstrap_projections, progress)] = "projections"
        else:
            results["projections"] = "Fresh"
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as exc:
                logger.exception("Bootstrap %s failed: %s", key, exc)
                results[key] = f"Error: {exc}"

    # Phase 3b (2026-04-17): Extended roster (40-man + spring training)
    # MUST run before live_stats so players.mlb_id is populated when
    # save_season_stats_to_db matches rows via mlb_id. Previously this ran
    # at Phase 8 (post-live_stats), which forced fuzzy name-match fallbacks
    # that corrupted hitter rows with pitcher stats.
    _notify(0.40)
    if force or check_staleness("extended_roster", staleness.players_hours):
        try:
            results["extended_roster"] = _run_with_timeout(
                lambda: _bootstrap_extended_roster(progress),
                timeout=_PHASE_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            logger.warning("Extended roster bootstrap timed out or failed: %s", exc)
            results["extended_roster"] = f"Error: {exc}"
    else:
        results["extended_roster"] = "Fresh"

    # Phases 4+5: Live stats + Historical (parallel — both independent)
    _notify(0.45)
    live_stale = force or check_staleness("season_stats", _live_stats_ttl_hours(staleness.live_stats_hours))
    hist_stale = force or check_staleness("historical_stats", staleness.historical_hours)
    historical_data = None

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}
        if live_stale:
            futures[executor.submit(_bootstrap_live_stats, progress)] = "live_stats"
        else:
            results["live_stats"] = "Fresh"
        if hist_stale:
            futures[executor.submit(_bootstrap_historical, progress)] = "historical"
        else:
            results["historical"] = "Fresh"
        for future in as_completed(futures):
            key = futures[future]
            try:
                result_val = future.result()
                if key == "historical" and isinstance(result_val, tuple):
                    results[key] = result_val[0]
                    historical_data = result_val[1]
                else:
                    results[key] = result_val
            except Exception as exc:
                logger.exception("Bootstrap %s failed: %s", key, exc)
                results[key] = f"Error: {exc}"

    # Phase 6: Injury data (derived from historical)
    _notify(0.80)
    if force or check_staleness("injury_data", staleness.historical_hours):
        results["injury_data"] = _bootstrap_injury_data(progress, historical=historical_data)
    else:
        results["injury_data"] = "Fresh"

    # Phase 7: Yahoo league data (optional)
    _notify(0.90)
    if force or check_staleness("yahoo_data", staleness.yahoo_hours):
        try:
            results["yahoo"] = _run_with_timeout(
                lambda: _bootstrap_yahoo(progress, yahoo_client),
                timeout=_PHASE_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            logger.warning("Yahoo bootstrap timed out or failed: %s", exc)
            results["yahoo"] = f"Error: {exc}"
    else:
        results["yahoo"] = "Fresh"

    # Phase 8: Extended roster — moved to Phase 3b (pre-live_stats) on
    # 2026-04-17. Leave the slot here as a no-op so the progress index
    # stays stable for any downstream tooling that counts phases.
    _notify(0.91)
    if "extended_roster" not in results:
        # Safety net: if the earlier call somehow didn't run, fall back.
        results["extended_roster"] = _bootstrap_extended_roster(progress)

    # Phase 9: Multi-source ADP
    _notify(0.92)
    if force or check_staleness("adp_sources", 24):
        results["adp_sources"] = _bootstrap_adp_sources(progress)
    else:
        results["adp_sources"] = "Fresh"

    # Phase 9b: Depth charts (roles + lineup slots)
    _notify(0.93)
    if force or check_staleness("depth_charts", 168):
        results["depth_charts"] = _bootstrap_depth_charts(progress)
    else:
        results["depth_charts"] = "Fresh"

    # Phase 10: Contract year data
    _notify(0.94)
    if force or check_staleness("contracts", 720):
        results["contracts"] = _bootstrap_contracts(progress)
    else:
        results["contracts"] = "Fresh"

    # Phase 11: News/transactions
    _notify(0.95)
    if force or check_staleness("news", staleness.news_hours if staleness else 1):
        results["news"] = _bootstrap_news(progress)
    else:
        results["news"] = "Fresh"

    # Phase 12: Deduplicate players (fix ID mismatches from different data sources)
    _notify(0.96)
    progress.phase = "Deduplication"
    progress.detail = "Merging duplicate player entries..."
    if on_progress:
        on_progress(progress)
    try:
        from src.database import deduplicate_players

        dedup_result = deduplicate_players()
        merged = dedup_result.get("players_merged", 0)
        results["deduplication"] = f"Merged {merged} duplicates" if merged > 0 else "No duplicates"
        logger.info("Deduplication: %s", dedup_result)
    except Exception as exc:
        logger.warning("Deduplication failed (non-fatal): %s", exc)
        results["deduplication"] = f"Skipped: {exc}"

    # Phase 13: Prospect rankings
    _notify(0.97)
    if force or check_staleness("prospect_rankings", staleness.prospects_hours):
        results["prospects"] = _bootstrap_prospects(progress)
    else:
        results["prospects"] = "Fresh"

    # Phase 14: News intelligence (multi-source)
    _notify(0.98)
    if force or check_staleness("news_intelligence", staleness.news_hours):
        try:
            results["news_intelligence"] = _run_with_timeout(
                lambda: _bootstrap_news_intel(progress, yahoo_client),
                timeout=_PHASE_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            logger.warning("News intelligence timed out or failed: %s", exc)
            results["news_intelligence"] = f"Error: {exc}"
    else:
        results["news_intelligence"] = "Fresh"

    # Phase 15: ECR consensus (depends on Phase 3 projections + Phase 9 ADP)
    _notify(0.99)
    if force or check_staleness("ecr_consensus", staleness.ecr_consensus_hours):
        try:
            results["ecr_consensus"] = _run_with_timeout(
                lambda: _bootstrap_ecr_consensus(progress),
                timeout=_TIMEOUT_ECR_CONSENSUS,
            )
        except Exception as exc:
            logger.warning("ECR consensus timed out or failed: %s", exc)
            results["ecr_consensus"] = f"Error: {exc}"
    else:
        results["ecr_consensus"] = "Fresh"

    # Phase 16: Populate player_id_map from cross-references (BUG-005 fix)
    _notify(0.995)
    progress.phase = "Player ID Map"
    progress.detail = "Building cross-platform ID mappings..."
    if on_progress:
        on_progress(progress)
    try:
        conn = get_connection()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO player_id_map (player_id, mlb_id, name)
                SELECT player_id, mlb_id, name FROM players
                WHERE mlb_id IS NOT NULL AND mlb_id != 0
            """)
            conn.execute("""
                UPDATE player_id_map SET fg_id = (
                    SELECT p.fangraphs_id FROM players p
                    WHERE p.player_id = player_id_map.player_id
                    AND p.fangraphs_id IS NOT NULL
                )
            """)
            conn.commit()
            id_count = conn.execute("SELECT COUNT(*) FROM player_id_map").fetchone()[0]
            results["player_id_map"] = f"Populated {id_count} ID mappings"
            logger.info("Player ID map: %d entries", id_count)
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("Player ID map population failed: %s", exc)
        results["player_id_map"] = f"Error: {exc}"

    # Phase 17: Yahoo transactions sync (BUG-017 fix)
    if yahoo_client is not None:
        try:
            txn_df = yahoo_client.get_league_transactions()
            if not txn_df.empty:
                from src.database import update_refresh_log
                from src.live_stats import match_player_id

                txn_stored = 0
                conn = get_connection()
                try:
                    for _, row in txn_df.iterrows():
                        pid = match_player_id(row.get("player_name", ""), "")
                        if pid is None:
                            continue
                        conn.execute(
                            "INSERT OR IGNORE INTO transactions "
                            "(player_id, type, team_from, team_to, timestamp) "
                            "VALUES (?, ?, ?, ?, ?)",
                            (
                                pid,
                                row.get("type", ""),
                                row.get("team_from", ""),
                                row.get("team_to", ""),
                                row.get("timestamp", ""),
                            ),
                        )
                        txn_stored += 1
                    conn.commit()
                finally:
                    conn.close()
                results["transactions"] = f"Stored {txn_stored} transactions"
                update_refresh_log("yahoo_transactions", "success")
            else:
                results["transactions"] = "No transactions"
        except Exception as exc:
            logger.warning("Transaction sync failed: %s", exc)
            results["transactions"] = f"Error: {exc}"

    # Phase 18: Yahoo free agents (BUG-019 fix)
    if yahoo_client is not None:
        try:
            progress.phase = "Yahoo Free Agents"
            progress.detail = "Fetching league free agents..."
            if on_progress:
                on_progress(progress)
            fa_df = yahoo_client.get_free_agents(count=200)
            if not fa_df.empty:
                from src.database import upsert_player_bulk
                from src.live_stats import match_player_id

                new_players = 0
                for _, row in fa_df.iterrows():
                    pname = row.get("player_name", "")
                    if not pname:
                        continue
                    existing = match_player_id(pname, "")
                    if existing is None:
                        upsert_player_bulk(
                            [
                                {
                                    "name": pname,
                                    "team": "",
                                    "positions": row.get("positions", "Util"),
                                    "is_hitter": 1 if row.get("positions", "") not in ("P", "SP", "RP") else 0,
                                }
                            ]
                        )
                        new_players += 1
                # Also populate the yahoo_free_agents table for ownership tracking
                from src.database import get_connection as _get_conn

                _fa_conn = _get_conn()
                try:
                    from datetime import UTC, datetime

                    _now = datetime.now(UTC).isoformat()
                    for _, row in fa_df.iterrows():
                        _fa_conn.execute(
                            """INSERT OR REPLACE INTO yahoo_free_agents
                               (player_key, player_name, positions, team, percent_owned, fetched_at)
                               VALUES (?, ?, ?, ?, ?, ?)""",
                            (
                                row.get("player_key", ""),
                                row.get("player_name", ""),
                                row.get("positions", ""),
                                row.get("team", ""),
                                float(row.get("percent_owned", 0) or 0),
                                _now,
                            ),
                        )
                    _fa_conn.commit()
                finally:
                    _fa_conn.close()
                results["yahoo_free_agents"] = (
                    f"Checked {len(fa_df)} FAs, added {new_players} new, stored {len(fa_df)} to yahoo_free_agents"
                )
            else:
                results["yahoo_free_agents"] = "No FA data from Yahoo"
        except Exception as exc:
            logger.warning("Yahoo FA fetch failed: %s", exc)
            results["yahoo_free_agents"] = f"Error: {exc}"

    # Phase 19: ROS Bayesian projections (depends on live stats + projections)
    _notify(0.965)
    if force or check_staleness("ros_projections", _live_stats_ttl_hours(staleness.live_stats_hours)):
        progress.phase = "ROS Projections"
        progress.detail = "Updating Bayesian rest-of-season projections..."
        if on_progress:
            on_progress(progress)
        try:
            from src.bayesian import update_ros_projections

            def _ros_update():
                count = update_ros_projections()
                return f"Updated {count} ROS projections"

            results["ros_projections"] = _run_with_timeout(_ros_update, timeout=_TIMEOUT_ROS_PROJECTIONS)
            logger.info("ROS Bayesian projections: %s", results["ros_projections"])
        except Exception as exc:
            logger.warning("ROS projection update failed: %s", exc)
            results["ros_projections"] = f"Error: {exc}"
    else:
        results["ros_projections"] = "Fresh"

    # Phase 20: Game-day intelligence (SOLO — no longer parallel with team_strength)
    _notify(0.96)
    gd_stale = force or check_staleness("game_day", staleness.game_day_hours)
    if gd_stale:
        try:
            results["game_day"] = _bootstrap_game_day(progress)
        except Exception as exc:
            logger.exception("Bootstrap game_day failed: %s", exc)
            results["game_day"] = f"Error: {exc}"
    else:
        results["game_day"] = "Fresh"

    # Phase 21: Team strength (SOLO — after game_day to avoid double fetch + SQLite lock)
    _notify(0.97)
    ts_stale = force or check_staleness("team_strength", staleness.team_strength_hours)
    if ts_stale:
        try:
            results["team_strength"] = _bootstrap_team_strength(progress)
        except Exception as exc:
            logger.exception("Bootstrap team_strength failed: %s", exc)
            results["team_strength"] = f"Error: {exc}"
    else:
        results["team_strength"] = "Fresh"

    # Phase 22-24: Stuff+, Batting stats, Sprint speed (parallel — all independent)
    _notify(0.98)
    sp_stale = force or check_staleness("stuff_plus", staleness.stuff_plus_hours)
    bs_stale = force or check_staleness("batting_stats", staleness.batting_stats_hours)
    ss_stale = force or check_staleness("sprint_speed", staleness.sprint_speed_hours)

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {}
        if sp_stale:
            futures[executor.submit(_bootstrap_stuff_plus, progress)] = "stuff_plus"
        else:
            results["stuff_plus"] = "Fresh"
        if bs_stale:
            futures[executor.submit(_bootstrap_batting_stats, progress)] = "batting_stats"
        else:
            results["batting_stats"] = "Fresh"
        if ss_stale:
            futures[executor.submit(_bootstrap_sprint_speed, progress)] = "sprint_speed"
        else:
            results["sprint_speed"] = "Fresh"
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as exc:
                logger.exception("Bootstrap %s failed: %s", key, exc)
                results[key] = f"Error: {exc}"

    # Phase 25-27: T4 park factors, T9 bat speed, T10 40-man (parallel, non-critical)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {}
        if force or check_staleness("park_factors_dynamic", 168):
            futures[executor.submit(_bootstrap_dynamic_park_factors, progress)] = "park_factors_dynamic"
        if force or check_staleness("bat_speed", 168):
            futures[executor.submit(_bootstrap_bat_speed, progress)] = "bat_speed"
        if force or check_staleness("forty_man", 168):
            futures[executor.submit(_bootstrap_forty_man, progress)] = "forty_man"
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as exc:
                logger.warning("Bootstrap %s failed (non-critical): %s", key, exc)
                results[key] = f"Error: {exc}"

    # Phase 28-30: T7 umpire, T8 catcher framing, T12 PvB splits (parallel, non-critical)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {}
        if force or check_staleness("umpire_tendencies", staleness.umpire_hours):
            futures[executor.submit(_bootstrap_umpire_tendencies, progress)] = "umpire_tendencies"
        else:
            results["umpire_tendencies"] = "Fresh"
        if force or check_staleness("catcher_framing", staleness.catcher_framing_hours):
            futures[executor.submit(_bootstrap_catcher_framing, progress)] = "catcher_framing"
        else:
            results["catcher_framing"] = "Fresh"
        if force or check_staleness("pvb_splits", staleness.pvb_splits_hours):
            futures[executor.submit(_bootstrap_pvb_splits, progress)] = "pvb_splits"
        else:
            results["pvb_splits"] = "Fresh"
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as exc:
                logger.warning("Bootstrap %s failed (non-critical): %s", key, exc)
                results[key] = f"Error: {exc}"

    # Phase 31: Game logs for Player Databank (1-hour staleness, non-critical)
    if force or check_staleness("game_logs", 1):
        try:
            results["game_logs"] = _run_with_timeout(
                lambda: _bootstrap_game_logs(progress, force=force),
                timeout=_TIMEOUT_GAME_LOGS,
            )
        except Exception as exc:
            logger.warning("Game logs timed out or failed: %s", exc)
            results["game_logs"] = f"Error: {exc}"
    else:
        results["game_logs"] = "Fresh"

    # Post-bootstrap validation (BUG-010 / data quality logging)
    try:
        conn = get_connection()
        try:
            proj_count = conn.execute("SELECT COUNT(*) FROM projections WHERE system='blended'").fetchone()[0]
            adp_count = conn.execute("SELECT COUNT(*) FROM adp WHERE adp < 999").fetchone()[0]
            team_count = conn.execute("SELECT COUNT(*) FROM players WHERE team != '' AND team IS NOT NULL").fetchone()[
                0
            ]
            player_count = conn.execute("SELECT COUNT(*) FROM players").fetchone()[0]
            logger.info(
                "Bootstrap validation: %d players, %d with teams, %d projections, %d ADP",
                player_count,
                team_count,
                proj_count,
                adp_count,
            )
        finally:
            conn.close()
    except Exception:
        pass

    _notify(1.0)
    progress.phase = "Complete"
    progress.detail = "All data loaded!"
    logger.info("Bootstrap results: %s", results)

    # Stamp all results into AnalyticsContext for data quality badges
    global _LAST_BOOTSTRAP_CTX  # noqa: PLW0603
    ctx = AnalyticsContext(pipeline="data_bootstrap")
    for source, result_msg in results.items():
        _stamp_from_result(ctx, source, result_msg)
    _LAST_BOOTSTRAP_CTX = ctx

    return results

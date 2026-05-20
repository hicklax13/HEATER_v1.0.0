"""Zero-interaction data bootstrap pipeline.

Fetches all MLB player data from free APIs on app startup.
Uses staleness-based smart refresh to avoid unnecessary API calls.
"""

import json
import logging
import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.analytics_context import AnalyticsContext, DataQuality

logger = logging.getLogger(__name__)

# Persistent log file for post-mortem analysis (SF-14).
# 2026-05-20 SFH L: do NOT attach when running under pytest. The handler
# is on the "src" logger which catches every src.* module, so unit tests
# that exercise fallback paths (e.g. test_wave8b_silent_failures with
# patched DB connections raising RuntimeError("DB out")) used to write
# their mock noise into data/logs/bootstrap.log, making post-mortem
# analysis confusing — fake "DB out" errors looked like real production
# failures. HEATER_DISABLE_FILE_LOG=1 gives an explicit override.
_UNDER_PYTEST = "pytest" in sys.modules or os.environ.get("HEATER_DISABLE_FILE_LOG") == "1"

if not _UNDER_PYTEST:
    _LOG_DIR = Path("data/logs")
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    _file_handler = RotatingFileHandler(_LOG_DIR / "bootstrap.log", maxBytes=5_000_000, backupCount=3)
    _file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger("src").addHandler(_file_handler)

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
    except Exception as exc:
        logger.warning(
            "data_bootstrap._live_stats_ttl_hours: ET clock probe failed; "
            "falling back to default_hours=%.2f (no game-window acceleration): %s",
            default_hours,
            exc,
            exc_info=True,
        )
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


def _try_write_refresh_log(source: str, status: str, message: str = "") -> bool:
    """Best-effort refresh_log write with retry-on-lock backoff.

    SFH H2 (2026-05-20): the per-phase ``except`` handlers attempt to write
    refresh_log via ``update_refresh_log(...)`` — but during a parallel
    long-held write lock (e.g. pvb_splits 50-batter Statcast loop), that
    write hits the 60s busy_timeout and the inner ``except Exception: pass``
    swallows it. refresh_log silently stays at the prior run's "success"
    and DataFreshnessTracker shows stale-but-healthy.

    This helper retries 3× with 0.5s/1s backoff, then logs a warning if
    still failing. Returns True/False so callers can chain fallback writes.
    """
    import sqlite3
    import time

    for attempt in range(3):
        try:
            from src.database import update_refresh_log

            update_refresh_log(source, status, message=(message[:200] if message else ""))
            return True
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc).lower() and attempt < 2:
                time.sleep(0.5 * (2**attempt))  # 0.5s, 1s
                continue
            logger.warning(
                "refresh_log write failed for %s/%s after %d attempts: %s",
                source,
                status,
                attempt + 1,
                exc,
            )
            return False
        except Exception as exc:
            logger.warning("refresh_log write error for %s/%s: %s", source, status, exc)
            return False
    return False


def _reconcile_results_to_refresh_log(results: dict) -> int:
    """SFH H2 (2026-05-20): post-bootstrap reconciliation pass.

    Walks the results dict and forces refresh_log entries for any phase
    whose return string indicates failure ("Error:" or "Timeout"). The
    per-phase error handlers attempt this directly, but can themselves
    fail during a long parallel write (umpire+catcher_framing vs
    pvb_splits) — leaving refresh_log stale at the prior run's "success".
    Running this pass at the END of bootstrap (after parallel groups
    complete and write locks release) ensures DataFreshnessTracker sees
    the true status.

    Returns the number of refresh_log rows reconciled.
    """
    reconciled = 0
    for source, result in results.items():
        if not isinstance(result, str):
            continue
        if result.startswith("Error:"):
            if _try_write_refresh_log(source, "error", result):
                reconciled += 1
        elif result.startswith("Timeout"):
            if _try_write_refresh_log(source, "timeout", result):
                reconciled += 1
    if reconciled > 0:
        logger.info("Reconciled %d failed-phase rows to refresh_log post-bootstrap", reconciled)
    return reconciled


def _run_with_timeout(fn: Callable, timeout: int = _PHASE_TIMEOUT_SECONDS, source: str | None = None) -> str:
    """Run *fn* in a thread and return its result, or a timeout message.

    This prevents any single bootstrap phase from hanging the entire app.
    The worker thread is daemonized so it won't block process exit.

    SFH H3 (2026-05-20): when ``source`` is provided and the phase times
    out, this records refresh_log(source, "timeout") so the operator
    dashboard reflects the real state. Without ``source``, behavior is
    unchanged (caller is responsible for recording).
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
        logger.warning("Bootstrap phase timed out after %ds (source=%s)", timeout, source or "unknown")
        if source:
            _try_write_refresh_log(source, "timeout", f"phase timed out after {timeout}s")
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
    elif r.startswith("error") or r.startswith("no "):
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
    adp_sources_hours: float = 24  # 24 hours
    depth_charts_hours: float = 168  # 7 days
    contracts_hours: float = 720  # 30 days
    bat_speed_hours: float = 168  # 7 days
    forty_man_hours: float = 168  # 7 days
    game_logs_hours: float = 1  # 1 hour


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


def _bootstrap_minor_league_rosters(progress: BootstrapProgress) -> str:
    """Wave 9 INFRA-F5: fetch AAA/AA rosters and upsert with level column.

    Adds ~900 minor-league players (top 30 per affiliate x ~60 affiliates,
    de-duped on mlb_id) to the player universe. These players lack Yahoo
    ownership data — consumers (UI filters) must handle NULL percent_owned.
    """
    from src.database import (
        update_refresh_log,
        update_refresh_log_auto,
        upsert_player_bulk,
    )

    # Lazy-import inside the function body so test patches of
    # src.live_stats.fetch_minor_league_players take effect.
    from src.live_stats import fetch_minor_league_players

    progress.phase = "Minor League Rosters"
    progress.detail = "Fetching AAA + AA rosters..."
    try:
        df = fetch_minor_league_players(season=2026, levels=("AAA", "AA"), top_n_per_team=30)
        if df.empty:
            update_refresh_log(
                "minor_league_rosters",
                "no_data",
                rows_written=0,
                message="fetch_minor_league_players returned empty",
            )
            return "Skipped: minor-league API returned no data"
        players = df.to_dict("records")
        count = upsert_player_bulk(players)
        status = update_refresh_log_auto(
            "minor_league_rosters",
            count,
            expected_min=500,  # ~900 expected; 500 floor allows for partial AAA-only success
            message=f"{count} minor leaguers upserted from {len(df)} API rows",
        )
        return f"Saved {count} minor leaguers ({status})"
    except Exception as e:
        update_refresh_log("minor_league_rosters", "error", message=str(e)[:200])
        return f"Error: {e}"


def _bootstrap_projections(progress: BootstrapProgress) -> str:
    """Fetch projections from FanGraphs."""
    from src.database import get_connection, update_refresh_log, update_refresh_log_auto

    progress.phase = "Projections"
    progress.detail = "Fetching FanGraphs projections..."
    try:
        from src.data_pipeline import refresh_if_stale

        success = refresh_if_stale(force=True)
        if success:
            # INFRA-F6: verify row count after refresh — a True return with 0
            # rows would otherwise silently log 'success'.
            conn = get_connection()
            try:
                row = conn.execute("SELECT COUNT(*) FROM projections").fetchone()
                count = int(row[0]) if row else 0
            finally:
                conn.close()
            status = update_refresh_log_auto(
                "projections",
                count,
                expected_min=1000,
                message=f"{count} projection rows in db",
            )
            return f"Projections refreshed ({count} rows, {status})"
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
    EXPECTED_MIN_FLOOR = 500
    try:
        df = fetch_season_stats(season=current_year)
        if df.empty:
            update_refresh_log(
                "season_stats",
                "no_data",
                rows_written=0,
                expected_min=EXPECTED_MIN_FLOOR,
                message="fetch_season_stats returned empty DataFrame",
            )
            return "No live stats available yet"
        count = save_season_stats_to_db(df)
        expected_min = max(EXPECTED_MIN_FLOOR, int(len(df) * 0.80))
        status = update_refresh_log_auto(
            "season_stats",
            count,
            expected_min=expected_min,
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
    from src.database import update_refresh_log, update_refresh_log_auto
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
            # INFRA-F6: row-count gate (was bare "success"); any history is fine.
            update_refresh_log_auto(
                "historical_stats",
                total,
                expected_min=1,
                message=f"{total} historical records across {len(historical)} seasons",
            )
        return (f"Saved {total} historical records across {len(historical)} seasons", historical)
    except Exception as e:
        update_refresh_log("historical_stats", "error", message=str(e)[:200])
        return (f"Error: {e}", None)


def _bootstrap_injury_data(progress: BootstrapProgress, historical: dict | None = None) -> str:
    """Extract injury history from historical stats and save."""
    from src.database import update_refresh_log, update_refresh_log_auto, upsert_injury_history_bulk
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
            # INFRA-F6: row-count gate (was bare "success").
            update_refresh_log_auto(
                "injury_data",
                count,
                expected_min=1,
                message=f"{count} injury records upserted from {len(raw_records)} raw rows",
            )
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

        if isinstance(data, dict) and len(data) > 0:
            source_dict = data
        else:
            source_dict = _PARK_FACTORS_EMERGENCY_2026
            tier = "emergency"

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
            cursor.execute("SELECT player_id FROM players WHERE name = ? COLLATE NOCASE", (name,))
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
    from src.database import get_connection, update_refresh_log, update_refresh_log_auto

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

        # INFRA-F6: gate "success" on actual roster row count to catch silent
        # 0-row syncs (was bare "success").
        conn = get_connection()
        try:
            row = conn.execute("SELECT COUNT(*) FROM league_rosters").fetchone()
            roster_count = int(row[0]) if row else 0
        finally:
            conn.close()
        status = update_refresh_log_auto(
            "yahoo_data",
            roster_count,
            expected_min=1,
            message=f"{roster_count} roster rows in league_rosters",
        )
        return f"Yahoo league data synced ({roster_count} rosters, {status})"
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
            cursor.execute("SELECT player_id FROM players WHERE name = ? COLLATE NOCASE", (name,))
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
        from src.database import update_refresh_log, update_refresh_log_auto

        results = []
        total_stored = 0
        try:
            ecr = fetch_fantasypros_ecr()
            if not ecr.empty:
                stored = _store_external_adp(ecr, "player_name", "ecr_rank", "fantasypros")
                total_stored += stored
                results.append(f"FantasyPros: {len(ecr)} fetched, {stored} stored")
        except Exception:
            logger.debug("FantasyPros ECR fetch failed", exc_info=True)
        try:
            nfbc = fetch_nfbc_adp()
            if not nfbc.empty:
                stored = _store_external_adp(nfbc, "player_name", "nfbc_adp", "nfbc")
                total_stored += stored
                results.append(f"NFBC: {len(nfbc)} fetched, {stored} stored")
        except Exception:
            logger.debug("NFBC ADP fetch failed", exc_info=True)
        if results:
            # INFRA-F6: row-count gate (was bare "success").
            update_refresh_log_auto(
                "adp_sources",
                total_stored,
                expected_min=100,
                message=f"{total_stored} ADP rows stored from {len(results)} sources",
            )
            return f"ADP sources: {', '.join(results)}"
        update_refresh_log("adp_sources", "no_data")
        _month = datetime.now(UTC).month
        if 3 <= _month <= 10:
            return "Skipped: in-season (ADP less relevant post-draft)"
        return "Skipped: ADP sources returned no data (JS-gated pages)"
    except Exception as e:
        # SFH D: surface error in refresh_log so operator sees the failure.
        logger.warning("ADP sources bootstrap failed: %s", e)
        from src.database import update_refresh_log

        update_refresh_log("adp_sources", "error", message=str(e)[:200])
        return _format_fetch_error(e, "ADP sources")


def _bootstrap_contracts(progress: BootstrapProgress) -> str:
    """Phase 10: Contract year data from BB-Ref.

    Persists contract_year=1 on matching players in the DB.
    """
    progress.phase = "Contract Data"
    progress.detail = "Fetching free agent list..."
    try:
        from src.contract_data import fetch_contract_year_players
        from src.database import update_refresh_log_auto

        names = fetch_contract_year_players()
        matched = _persist_contract_years(names) if names else 0
        update_refresh_log_auto(
            "contracts",
            matched,
            expected_min=1,
            message=f"{matched} players flagged contract_year=1 (from {len(names)} fetched)",
        )
        return f"Contracts: {matched} players flagged (from {len(names)} fetched)"
    except Exception as e:
        # SFH D: surface error in refresh_log so operator sees the failure.
        logger.warning("Contract data bootstrap failed: %s", e)
        from src.database import update_refresh_log

        update_refresh_log("contracts", "error", message=str(e)[:200])
        return f"Contracts: error ({e})"


def _bootstrap_news(progress: BootstrapProgress) -> str:
    """Phase 11: Recent MLB transactions/news.

    Also computes and persists per-player news_sentiment to the DB.
    """
    progress.phase = "News"
    progress.detail = "Fetching recent transactions..."
    try:
        from src.database import update_refresh_log_auto
        from src.news_fetcher import fetch_recent_transactions

        items = fetch_recent_transactions(days_back=7)
        if items:
            _persist_news_sentiment(items)
        # INFRA-F6: row-count gate (was bare "success"); any news is fine.
        update_refresh_log_auto(
            "news",
            len(items),
            expected_min=1,
            message=f"{len(items)} recent transactions",
        )
        return f"News: {len(items)} transactions"
    except Exception as e:
        # SFH D: surface error in refresh_log so operator sees the failure.
        logger.warning("News bootstrap failed: %s", e)
        from src.database import update_refresh_log

        update_refresh_log("news", "error", message=str(e)[:200])
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

    Two-tier fetch (SF-5 fix, 2026-05-10):

    1. **Primary** — Roster Resource scrape (``fetch_depth_charts``).
       Often returns empty: the endpoint is JS-gated or 403's non-browser UAs.
    2. **Fallback** — MLB Stats API ``team_roster`` hydration
       (``fetch_depth_charts_via_statsapi``). Classifies pitchers by
       ``gamesStarted`` / ``saves`` and treats all position players as lineup
       starters. Less precise than Roster Resource (no batting order, no SU/MR
       distinction), but sufficient to populate ``depth_chart_role`` and
       ``lineup_slot`` so the closer monitor + lineup protection are not
       silently degraded.

    The ``tier`` column on ``refresh_log`` records which source actually
    succeeded so the Data Status panel can show "fallback" honestly.
    """
    progress.phase = "Depth Charts"
    progress.detail = "Fetching depth charts..."
    try:
        from src.database import update_refresh_log, update_refresh_log_auto
        from src.depth_charts import fetch_depth_charts, fetch_depth_charts_via_statsapi

        # Tier 1: Roster Resource scrape (primary, more accurate)
        depth_data = fetch_depth_charts()
        tier = "primary"
        if not depth_data:
            # Tier 2: MLB Stats API fallback (rotation/closer detection)
            progress.detail = "Roster Resource empty — falling back to MLB Stats API..."
            depth_data = fetch_depth_charts_via_statsapi()
            tier = "fallback" if depth_data else None

        if depth_data:
            count = _persist_depth_chart_roles(depth_data)
            # INFRA-F6: row-count gate (was bare "success", tier=tier).
            update_refresh_log_auto(
                "depth_charts",
                count,
                expected_min=100,
                message=f"{len(depth_data)} teams, {count} roles persisted",
                tier=tier,
            )
            return f"Depth charts: {len(depth_data)} teams, {count} roles persisted ({tier})"
        update_refresh_log("depth_charts", "no_data")
        return "Skipped: depth chart endpoints returned no data (Roster Resource + MLB Stats API both empty)"
    except Exception as e:
        # SFH D: surface error in refresh_log so operator sees the failure.
        logger.warning("Depth chart bootstrap failed: %s", e)
        from src.database import update_refresh_log

        update_refresh_log("depth_charts", "error", message=str(e)[:200])
        return _format_fetch_error(e, "Depth charts")


def _enrich_pitcher_positions(progress: BootstrapProgress) -> str:
    """Phase 9c: Add SP/RP qualifier to pitcher rows that only have 'P'.

    Walks every player row whose ``positions`` contains a 'P' token but
    no 'SP' or 'RP', then resolves the qualifier using
    :func:`resolve_pitcher_positions` against ``depth_chart_role`` (just
    persisted by :func:`_bootstrap_depth_charts`) and aggregated
    ``season_stats`` (saves, ip, games_played).

    Idempotent — pre-enriched rows are left alone.
    """
    progress.phase = "Pitcher Positions"
    progress.detail = "Enriching pitcher SP/RP qualifiers..."

    from src.database import get_connection, update_refresh_log_auto
    from src.depth_charts import resolve_pitcher_positions

    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT p.player_id,
                   p.positions,
                   p.depth_chart_role,
                   COALESCE(SUM(ss.sv), 0)            AS total_sv,
                   COALESCE(SUM(ss.ip), 0.0)          AS total_ip,
                   COALESCE(SUM(ss.games_played), 0)  AS total_gp
            FROM players p
            LEFT JOIN season_stats ss ON p.player_id = ss.player_id
            WHERE p.positions IS NOT NULL
              AND p.positions LIKE '%P%'
            GROUP BY p.player_id
            """
        ).fetchall()

        updated = 0
        for player_id, positions, role, sv, ip, gp in rows:
            new_pos = resolve_pitcher_positions(
                positions,
                depth_chart_role=role,
                saves=int(sv) if sv is not None else None,
                innings_pitched=float(ip) if ip is not None else None,
                games_played=int(gp) if gp is not None else None,
            )
            if new_pos and new_pos != positions:
                conn.execute(
                    "UPDATE players SET positions = ? WHERE player_id = ?",
                    (new_pos, player_id),
                )
                updated += 1
        conn.commit()
        # INFRA-F6: row-count gate. Most refreshes won't change anything
        # (pitchers stay enriched once set), so a low floor is correct.
        update_refresh_log_auto(
            "pitcher_positions",
            updated,
            expected_min=0,
            message=f"{updated} pitchers enriched with SP/RP",
        )
        logger.info("Enriched SP/RP qualifier on %d pitcher rows", updated)
        return f"Pitcher positions: {updated} rows enriched"
    except Exception as e:
        # 2026-05-20 SFH MED-1 follow-up: same DNA as PR #65 D-fix — write
        # refresh_log on the error path so the next bootstrap retries and
        # the Data Status panel doesn't show stale-but-healthy. The PR #65
        # structural guard filtered on `_bootstrap_*` prefix and missed
        # this function which uses `_enrich_*` — guard widened in this PR.
        logger.exception("Pitcher position enrichment failed")
        from src.database import update_refresh_log

        update_refresh_log("pitcher_positions", "error", message=str(e)[:200])
        return f"Pitcher positions: error ({e})"
    finally:
        conn.close()


# ── FP Edge Feature Phases ────────────────────────────────────────────


def _bootstrap_prospects(progress: BootstrapProgress) -> str:
    """Phase 13: Prospect rankings from FanGraphs + MiLB stats."""
    progress.phase = "Prospects"
    progress.detail = "Refreshing prospect rankings..."
    try:
        from src.database import update_refresh_log_auto
        from src.prospect_engine import refresh_prospect_rankings

        df = refresh_prospect_rankings(force=True)
        prospect_count = len(df) if df is not None else 0
        # INFRA-F6: row-count gate. Floor of 20 matches the documented Tier 3
        # emergency fallback (prospect_engine._STATIC_PROSPECTS, 20 entries);
        # a higher floor would spuriously downgrade that path to "partial".
        update_refresh_log_auto(
            "prospect_rankings",
            prospect_count,
            expected_min=20,
            message=f"{prospect_count} prospects ranked",
        )
        return f"Prospects: {prospect_count} ranked"
    except Exception as e:
        # SFH D: surface error in refresh_log so operator sees the failure.
        logger.warning("Prospect bootstrap failed: %s", e)
        from src.database import update_refresh_log

        update_refresh_log("prospect_rankings", "error", message=str(e)[:200])
        return f"Prospects: error ({e})"


def _bootstrap_news_intel(progress: BootstrapProgress, yahoo_client=None) -> str:
    """Phase 14: Multi-source news intelligence."""
    progress.phase = "News Intelligence"
    progress.detail = "Fetching news from ESPN, RotoWire, MLB API..."
    try:
        from src.database import update_refresh_log_auto
        from src.player_news import refresh_all_news

        count = refresh_all_news(yahoo_client=yahoo_client, force=True)
        # INFRA-F6: row-count gate (was bare "success"); any item is fine.
        update_refresh_log_auto(
            "news_intelligence",
            count,
            expected_min=1,
            message=f"{count} news items from multi-source",
        )
        return f"News: {count} items from multi-source"
    except Exception as e:
        # SFH D: surface error in refresh_log so operator sees the failure.
        logger.warning("News intelligence bootstrap failed: %s", e)
        from src.database import update_refresh_log

        update_refresh_log("news_intelligence", "error", message=str(e)[:200])
        return f"News intel: error ({e})"


def _bootstrap_ecr_consensus(progress: BootstrapProgress) -> str:
    """Phase 15: ECR consensus from multi-platform ranking sources."""
    progress.phase = "ECR Consensus"
    progress.detail = "Building multi-platform ranking consensus..."
    try:
        from src.database import update_refresh_log, update_refresh_log_auto
        from src.ecr import refresh_ecr_consensus

        df = refresh_ecr_consensus(force=True)
        is_cached = getattr(df, "attrs", {}).get("ecr_is_cached", False)
        if is_cached:
            logger.warning("ECR consensus returned cached data — all sources failed")
            update_refresh_log("ecr_consensus", "cached")
            return f"ECR Consensus: {len(df)} players (cached — sources unavailable)"
        # INFRA-F6: row-count gate (was bare "success").
        update_refresh_log_auto(
            "ecr_consensus",
            len(df),
            expected_min=100,
            message=f"{len(df)} players ranked",
        )
        return f"ECR Consensus: {len(df)} players ranked"
    except Exception as e:
        # 2026-05-20 SFH D: write refresh_log on the error path so the
        # next bootstrap can see the failure and retry (was silently
        # frozen at last successful timestamp — phase looked healthy).
        logger.warning("ECR consensus bootstrap failed: %s", e)
        from src.database import update_refresh_log

        update_refresh_log("ecr_consensus", "error", message=str(e)[:200])
        return f"ECR consensus: error ({e})"


def _bootstrap_game_day(progress: BootstrapProgress) -> str:
    """Phase 20: Fetch game-day intelligence (weather, lineups, opposing pitchers)."""
    progress.phase = "Game Day Intel"
    progress.detail = "Fetching today's weather, lineups, opposing pitchers..."
    try:
        from src.game_day import fetch_game_day_intelligence

        result = fetch_game_day_intelligence()
        from src.database import update_refresh_log_auto

        games = result.get("games_count", 0)
        pitchers = result.get("pitcher_count", 0)
        # INFRA-F6: row-count gate (was bare "success"); any game-day intel is fine.
        update_refresh_log_auto(
            "game_day",
            games + pitchers,
            expected_min=1,
            message=f"{games} games, {pitchers} pitchers",
        )
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
    """Phase 22: Fetch Stuff+/Location+/Pitching+ from FanGraphs via pybaseball.

    SF-6 Option C: attempt with browser headers injected into requests.get
    (Tier 1) before giving up. FanGraphs' leaders-legacy.aspx is gated by
    a Cloudflare-style bot check that rejects the bare pybaseball UA, but
    the browser-headers attempt costs only one extra HTTP round-trip and
    leaves a clean telemetry trail when it fails.
    """
    progress.phase = "Stuff+ Metrics"
    progress.detail = "Fetching Stuff+/Location+/Pitching+ from FanGraphs..."
    try:
        from pybaseball import pitching_stats
    except ImportError:
        logger.warning("pybaseball not installed — skipping Stuff+ fetch")
        return "Skipped: pybaseball not installed"

    try:
        import pandas as pd

        from src.data_fetch_utils import fetch_fangraphs_with_browser_headers
        from src.database import get_connection, update_refresh_log, update_refresh_log_auto

        year = datetime.now(UTC).year
        logger.info(
            "Fetching FanGraphs pitching stats for %d (qual=0) with browser headers...",
            year,
        )
        # SF-6 Option C: try browser-headers fetch first.
        try:
            fg_df = fetch_fangraphs_with_browser_headers(lambda: pitching_stats(year, qual=0))
        except Exception as inner_exc:
            # Re-raise so the outer except chain handles 403 logging cleanly.
            raise inner_exc

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
            elif cl in ("gmli", "leverageindex", "gmleverageindex"):
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

                present_cols = [c for c in found_cols if pd.notna(row.get(c))]
                if present_cols:
                    insert_cols = ["player_id", "season"] + present_cols
                    placeholders = ", ".join(["?"] * len(insert_cols))
                    insert_vals = [pid, year] + [float(row.get(c)) for c in present_cols]
                    update_clause = ", ".join(f"{c} = excluded.{c}" for c in present_cols)
                    conn.execute(
                        f"INSERT INTO statcast_archive ({', '.join(insert_cols)}) VALUES ({placeholders}) "
                        f"ON CONFLICT(player_id, season) DO UPDATE SET {update_clause}",
                        insert_vals,
                    )

                updated += 1

            conn.commit()
        finally:
            conn.close()

        # INFRA-F6: row-count gate (was bare "success", tier="primary").
        update_refresh_log_auto(
            "stuff_plus",
            updated,
            expected_min=50,
            message=f"browser-headers fetch ok — {updated}/{len(fg_df)} pitchers",
            tier="primary",
        )
        logger.info(
            "Stuff+ metrics: primary tier ok — updated %d pitchers from %d FanGraphs rows",
            updated,
            len(fg_df),
        )
        return f"Updated {updated} pitchers with Stuff+/Location+/Pitching+"

    except Exception as exc:
        logger.exception("Stuff+ bootstrap failed: %s", exc)
        try:
            from src.database import update_refresh_log

            # SF-6: 403 means FanGraphs rejected even browser headers (Cloudflare-style
            # bot block). Optimizer K-boost falls back to FIP/xFIP proxy via Wave 4-J
            # Option B (_stuff_plus_k_multiplier). Honest message so users see the
            # data is unavailable rather than a stack trace.
            base_msg = _format_fetch_error(exc, "FanGraphs Stuff+")
            if "403" in str(exc):
                base_msg += " — known limitation, see CLAUDE.md SF-6 (browser-headers also blocked); optimizer falls back to FIP/xFIP K-boost proxy"
            update_refresh_log(
                "stuff_plus",
                _classify_fetch_error(exc),
                message=base_msg,
            )
        except Exception as log_exc:
            logger.warning(
                "data_bootstrap._bootstrap_stuff_plus: refresh_log update failed during "
                "error-handling path; bootstrap status panel will be missing this phase: %s",
                log_exc,
                exc_info=True,
            )
        base = _format_fetch_error(exc, "FanGraphs Stuff+")
        if base.startswith("Skipped:") or "403" in str(exc):
            return (
                "Skipped: FanGraphs Stuff+ unavailable (HTTP 403 — known limitation, "
                "see CLAUDE.md SF-6). Optimizer K-boost falls back to FIP/xFIP proxy "
                "(neutral 1.0× when both unavailable)."
            )
        return base


def _bootstrap_batting_stats(progress: BootstrapProgress) -> str:
    """Phase 23: Fetch advanced batting stats (BABIP, ISO, K%, BB%, etc.) from FanGraphs.

    SF-6 Option C: same browser-headers Tier 1 attempt as Stuff+. When 403 still
    blocks, optimizer falls back to neutral defaults for BABIP/ISO/K%/BB%.
    """
    progress.phase = "Batting Stats"
    progress.detail = "Fetching BABIP/ISO/K%/BB% from FanGraphs..."
    try:
        from pybaseball import batting_stats
    except ImportError:
        logger.warning("pybaseball not installed — skipping batting stats fetch")
        return "Skipped: pybaseball not installed"

    try:
        import pandas as pd

        from src.data_fetch_utils import fetch_fangraphs_with_browser_headers
        from src.database import get_connection, update_refresh_log, update_refresh_log_auto

        year = datetime.now(UTC).year
        logger.info(
            "Fetching FanGraphs batting stats for %d (qual=0) with browser headers...",
            year,
        )
        # SF-6 Option C: try browser-headers fetch first.
        try:
            fg_df = fetch_fangraphs_with_browser_headers(lambda: batting_stats(year, qual=0))
        except Exception as inner_exc:
            raise inner_exc

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

                present_cols = [c for c in target_cols if pd.notna(row.get(c))]
                if not present_cols:
                    continue

                insert_cols = ["player_id", "season"] + present_cols
                placeholders = ", ".join(["?"] * len(insert_cols))
                insert_vals = [pid, year] + [float(row.get(c)) for c in present_cols]
                update_clause = ", ".join(f"{c} = excluded.{c}" for c in present_cols)
                conn.execute(
                    f"INSERT INTO statcast_archive ({', '.join(insert_cols)}) VALUES ({placeholders}) "
                    f"ON CONFLICT(player_id, season) DO UPDATE SET {update_clause}",
                    insert_vals,
                )
                updated += 1

            conn.commit()
        finally:
            conn.close()

        # INFRA-F6: row-count gate (was bare "success", tier="primary").
        update_refresh_log_auto(
            "batting_stats",
            updated,
            expected_min=100,
            message=f"browser-headers fetch ok — {updated}/{len(fg_df)} hitters",
            tier="primary",
        )
        logger.info(
            "Batting stats: primary tier ok — updated %d hitters from %d FanGraphs rows",
            updated,
            len(fg_df),
        )
        return f"Updated {updated} hitters with BABIP/ISO/K%%/BB%%"

    except Exception as exc:
        logger.exception("Batting stats bootstrap failed: %s", exc)
        try:
            from src.database import update_refresh_log

            # SF-6: same browser-headers Tier 1 attempt as Stuff+. When 403 still
            # blocks, optimizer keeps using neutral defaults for BABIP/ISO/K%/BB%.
            base_msg = _format_fetch_error(exc, "FanGraphs Batting Stats")
            if "403" in str(exc):
                base_msg += " — known limitation, see CLAUDE.md SF-6 (browser-headers also blocked); optimizer falls back to default BABIP/ISO/K%/BB%"
            update_refresh_log(
                "batting_stats",
                _classify_fetch_error(exc),
                message=base_msg,
            )
        except Exception as log_exc:
            logger.warning(
                "data_bootstrap._bootstrap_batting_stats: refresh_log update failed during "
                "error-handling path; bootstrap status panel will be missing this phase: %s",
                log_exc,
                exc_info=True,
            )
        base = _format_fetch_error(exc, "FanGraphs Batting Stats")
        if base.startswith("Skipped:") or "403" in str(exc):
            return (
                "Skipped: FanGraphs Batting Stats unavailable (HTTP 403 — known limitation, "
                "see CLAUDE.md SF-6). Optimizer uses neutral defaults for BABIP/ISO/K%/BB%."
            )
        return base


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

        from src.database import get_connection, update_refresh_log, update_refresh_log_auto

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

                conn.execute(
                    "INSERT INTO statcast_archive (player_id, season, sprint_speed) VALUES (?, ?, ?) "
                    "ON CONFLICT(player_id, season) DO UPDATE SET sprint_speed = excluded.sprint_speed",
                    (pid, year, speed),
                )
                updated += 1

            conn.commit()
        finally:
            conn.close()

        # INFRA-F6: row-count gate (was bare "success").
        update_refresh_log_auto(
            "sprint_speed",
            updated,
            expected_min=100,
            message=f"{updated} players from {len(ss_df)} Statcast rows",
        )
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


def _load_umpire_tendencies_seed():
    """Tier 3: load 2024 seed data from data/seed/umpire_tendencies_2024.json.

    Returns a list of dicts in the bootstrap's INSERT shape (name, games,
    k_pct, bb_pct, rpg + deltas), or ``None`` if the file is missing/malformed.
    """
    seed_path = Path("data/seed/umpire_tendencies_2024.json")
    if not seed_path.exists():
        return None
    try:
        payload = json.loads(seed_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("T7: failed to parse seed file %s: %s", seed_path, exc)
        return None

    league_k = float(payload.get("league_avg_k_pct", 0.226) or 0.226)
    league_bb = float(payload.get("league_avg_bb_pct", 0.084) or 0.084)
    league_rpg = float(payload.get("league_avg_runs_per_game", 4.39) or 4.39)

    umpires = payload.get("umpires") or []
    rows = []
    for u in umpires:
        # Reconstruct absolute values from the diff stored in the seed.
        k_pct_diff = float(u.get("k_pct_diff", 0.0) or 0.0)
        bb_pct_diff = float(u.get("bb_pct_diff", 0.0) or 0.0)
        rpg_diff = float(u.get("runs_per_game_diff", 0.0) or 0.0)
        rows.append(
            {
                "name": u.get("name", ""),
                "games": int(u.get("games", 0) or 0),
                "k_pct": league_k + k_pct_diff,
                "bb_pct": league_bb + bb_pct_diff,
                "rpg": league_rpg + rpg_diff,
                "k_pct_delta": k_pct_diff,
                "bb_pct_delta": bb_pct_diff,
                "run_env_delta": rpg_diff,
            }
        )
    return rows or None


def _bootstrap_umpire_tendencies(progress: BootstrapProgress) -> str:
    """T7: Fetch umpire assignments and build per-umpire tendency table.

    3-tier waterfall (SF-7):
      Tier 1: MLB Stats API schedule → boxscore_data → HP umpire extraction (existing)
      Tier 2: Savant umpire leaderboard scrape — NOT VIABLE (Savant 404s; documented)
      Tier 3: shipped 2024 seed file at data/seed/umpire_tendencies_2024.json (NEW)
    """
    progress.phase = "Umpire Data"
    progress.detail = "Fetching umpire assignments..."

    # INFRA-F3: cap Tier 1 schedule iteration to 60s wall-clock so we fall
    # through to Tier 3 seed in bounded time. Without this, ~2000 boxscore_data
    # calls mid-season exceed the orchestrator's 240s per-phase timeout and
    # the shipped seed never serves.
    _UMPIRE_TIER1_TIMEOUT_S = 60.0

    try:
        from src.database import get_connection, update_refresh_log

        year = datetime.now(UTC).year
        tier_used = None
        umpire_stats: dict[str, dict] = {}
        league_games = 0

        # ── Tier 1: existing MLB Stats API schedule + boxscore extraction ──
        try:
            import statsapi as _statsapi

            logger.info("T7 [primary]: Fetching MLB schedule for %d to build umpire profiles...", year)
            schedule = _statsapi.schedule(
                start_date=f"{year}-03-20",
                end_date=datetime.now(UTC).strftime("%Y-%m-%d"),
            )

            tier1_start = time.monotonic()
            tier1_timed_out = False
            games_processed = 0

            if schedule:
                for game in schedule:
                    # INFRA-F3: enforce wall-clock budget on Tier 1.
                    elapsed = time.monotonic() - tier1_start
                    if elapsed > _UMPIRE_TIER1_TIMEOUT_S:
                        logger.warning(
                            "T7 [primary]: Tier 1 schedule iteration exceeded %.0fs budget "
                            "after processing %d games (%d umpires collected); breaking out "
                            "to let Tier 3 seed serve if needed. (INFRA-F3 fix.)",
                            _UMPIRE_TIER1_TIMEOUT_S,
                            games_processed,
                            len(umpire_stats),
                        )
                        tier1_timed_out = True
                        break

                    game_pk = game.get("game_id")
                    if not game_pk:
                        continue
                    status = game.get("status", "")
                    if "Final" not in status and "Completed" not in status:
                        continue

                    try:
                        boxscore = _statsapi.boxscore_data(game_pk)
                    except Exception:
                        continue
                    games_processed += 1

                    if not boxscore:
                        continue

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

                    away_stats = boxscore.get("awayBatting", {})
                    home_stats = boxscore.get("homeBatting", {})

                    total_k = 0
                    total_bb = 0
                    total_runs = 0
                    total_pa = 0

                    for team_stats in [away_stats, home_stats]:
                        if isinstance(team_stats, dict):
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

                if umpire_stats:
                    tier_used = "primary"
                    league_games = sum(u["games"] for u in umpire_stats.values())
                    logger.info(
                        "T7 [primary]: extracted %d umpires from %d games%s",
                        len(umpire_stats),
                        league_games,
                        " (timed out; partial result)" if tier1_timed_out else "",
                    )
        except ImportError:
            logger.warning("T7 [primary]: statsapi not installed; skipping Tier 1")
        except Exception as exc:
            logger.warning("T7 [primary]: extraction failed: %s", exc)

        # ── Tier 2: Savant umpire scrape (NOT VIABLE — Savant 404s on /umpire) ──
        # Documented as known limitation; we skip cleanly to Tier 3.
        # If Savant ever publishes an umpire leaderboard, wire it here.

        # ── Tier 3: shipped 2024 seed file (NEW) ───────────────────────────
        seed_used = False
        if not umpire_stats:
            seed = _load_umpire_tendencies_seed()
            if seed:
                tier_used = "emergency"
                seed_used = True
                logger.warning(
                    "T7 [emergency]: All live sources failed — using 2024 seed file (%d umpires)",
                    len(seed),
                )
                # Convert seed rows into the same `umpire_stats` shape so the
                # downstream INSERT loop is shared.
                # The seed already has computed deltas + absolute values, so
                # we mark a separate code path below.

        if not umpire_stats and not seed_used:
            update_refresh_log(
                "umpire_tendencies",
                "no_data",
                rows_written=0,
                message="all sources failed: schedule empty + no seed file",
            )
            return (
                "Skipped: no umpire data (boxscore HP-umpire extraction failed). "
                "Optimizer uses neutral 1.0× umpire multiplier — see CLAUDE.md SF-7."
            )

        conn = get_connection()
        try:
            now = datetime.now(UTC).isoformat()
            updated = 0

            if seed_used:
                # Seed path: write directly with stored deltas; season tagged 2024.
                seed_rows = _load_umpire_tendencies_seed() or []
                for row in seed_rows:
                    if not row.get("name"):
                        continue
                    conn.execute(
                        """INSERT OR REPLACE INTO umpire_tendencies
                           (umpire_name, games_umped, k_pct, bb_pct, runs_per_game,
                            k_pct_delta, bb_pct_delta, run_env_delta, season, fetched_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            row["name"],
                            row["games"],
                            round(row["k_pct"], 4),
                            round(row["bb_pct"], 4),
                            round(row["rpg"], 2),
                            round(row["k_pct_delta"], 4),
                            round(row["bb_pct_delta"], 4),
                            round(row["run_env_delta"], 2),
                            2024,
                            now,
                        ),
                    )
                    updated += 1
            else:
                # Live path: compute league averages + deltas and write.
                league_k = sum(u["total_k"] for u in umpire_stats.values())
                league_bb = sum(u["total_bb"] for u in umpire_stats.values())
                league_runs = sum(u["total_runs"] for u in umpire_stats.values())
                league_pa = sum(u["total_pa"] for u in umpire_stats.values())

                avg_k_pct = league_k / max(1, league_pa)
                avg_bb_pct = league_bb / max(1, league_pa)
                avg_rpg = league_runs / max(1, league_games)

                for name, stats in umpire_stats.items():
                    if stats["games"] < 3:
                        continue
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

        from src.database import update_refresh_log_auto

        tier_label = tier_used or "primary"
        msg = (
            f"{updated} umpires from 2024 seed [SEED]"
            if seed_used
            else f"{updated} umpires from {league_games} games (tier={tier_label})"
        )
        # INFRA-F6: ImportError fallback that called update_refresh_log("...", "success", tier=...)
        # has been removed — update_refresh_log_auto is in src/database.py alongside
        # update_refresh_log and is always importable.
        status = update_refresh_log_auto(
            "umpire_tendencies",
            updated,
            expected_min=1,
            message=msg,
            tier=tier_label,
        )
        logger.info("T7: Umpire tendencies — %d umpires updated (tier=%s)", updated, tier_used)
        suffix = "from 2024 seed" if seed_used else f"from {league_games} games"
        return f"Saved {updated} umpire profiles {suffix} via tier={tier_used} ({status})"

    except Exception as exc:
        logger.exception("T7 umpire tendencies failed: %s", exc)
        try:
            from src.database import update_refresh_log

            update_refresh_log("umpire_tendencies", "error", message=str(exc)[:200])
        except Exception:
            pass
        return f"Error: {exc}"


def _fetch_catcher_framing_savant_scrape(year: int):
    """Tier 2: scrape Savant's catcher-framing leaderboard via embedded JSON.

    Uses browser headers to bypass Savant's 403-on-raw-requests behavior, then
    extracts ``const data = [...]`` from the HTML. Returns a list of dicts in
    a normalized shape (``framing_runs``, ``pop_time``, etc.) so the caller
    can reuse the same row-mapping loop as the pybaseball path.
    """
    try:
        from src.data_fetch_utils import fetch_savant_leaderboard_json
    except ImportError:
        return None

    url = f"https://baseballsavant.mlb.com/leaderboard/catcher-framing?year={year}&team=&min=300&type=Pitcher&sort=4,1"
    raw = fetch_savant_leaderboard_json(url)
    if not raw:
        return None

    rows = []
    for r in raw:
        # Savant aggregates by (catcher, team) — we want the per-team rows
        # but skip the 'zMLB' multi-team aggregate to avoid double-counting.
        team = str(r.get("team_name", "")).strip()
        if team == "zMLB":
            continue
        name_lf = r.get("f2_name_display_first_last", "")
        if not name_lf:
            continue
        # Convert "Bailey, Patrick" → "Patrick Bailey" to match player table.
        if "," in name_lf:
            last, first = [p.strip() for p in name_lf.split(",", 1)]
            name = f"{first} {last}"
        else:
            name = name_lf
        rv_tot = float(r.get("rv_tot", 0.0) or 0.0)
        pitches = int(r.get("pitches", 0) or 0)
        # Savant exposes run value (rv_tot); framing_runs ≈ rv_tot here.
        # Approximate games as pitches // 90 (typical 90 catcher pitches/game).
        games = max(1, pitches // 90)
        rows.append(
            {
                "name": name,
                "framing_runs": rv_tot,
                "games": games,
                "pop_time": 0.0,
                "cs_pct": 0.0,
            }
        )
    return rows or None


def _load_catcher_framing_seed():
    """Tier 3: load 2024 seed data from data/seed/catcher_framing_2024.json.

    Returns a list of dicts in the same normalized shape as the Tier 2 scraper
    so the caller can apply one INSERT loop. ``None`` if the seed file is
    missing or malformed.
    """
    seed_path = Path("data/seed/catcher_framing_2024.json")
    if not seed_path.exists():
        return None
    try:
        payload = json.loads(seed_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("T8: failed to parse seed file %s: %s", seed_path, exc)
        return None

    catchers = payload.get("catchers") or []
    rows = []
    for c in catchers:
        rows.append(
            {
                "name": c.get("player_name", ""),
                "framing_runs": float(c.get("framing_runs", 0.0) or 0.0),
                "games": int(c.get("games", 0) or 0),
                "pop_time": float(c.get("pop_time", 0.0) or 0.0),
                "cs_pct": float(c.get("cs_pct", 0.0) or 0.0),
            }
        )
    return rows or None


def _bootstrap_catcher_framing(progress: BootstrapProgress) -> str:
    """T8: Fetch catcher framing runs and pop time.

    3-tier waterfall (SF-7):
      Tier 1: pybaseball Savant + FanGraphs + statsapi (existing primary path)
      Tier 2: direct Savant scrape with browser headers (NEW)
      Tier 3: shipped 2024 seed file at data/seed/catcher_framing_2024.json (NEW)

    refresh_log records which tier succeeded via update_refresh_log_auto(tier=...).
    """
    progress.phase = "Catcher Framing"
    progress.detail = "Fetching catcher framing + pop time..."

    try:
        import pandas as pd

        from src.database import get_connection, update_refresh_log

        year = datetime.now(UTC).year

        # ── Tier 1: existing pybaseball path ──────────────────────────────
        framing_data = None
        tier_used = None
        try:
            import pybaseball  # noqa: F401

            try:
                from pybaseball import statcast_catcher_framing

                sv_df = statcast_catcher_framing(year)
                if sv_df is not None and not sv_df.empty:
                    framing_data = sv_df
                    tier_used = "primary"
                    logger.info(
                        "T8 [primary]: Got %d catchers from pybaseball statcast_catcher_framing",
                        len(sv_df),
                    )
            except Exception as e:
                logger.warning("T8 [primary]: pybaseball Savant failed: %s", e)

            if framing_data is None:
                try:
                    from pybaseball import batting_stats

                    # SFH M2 (2026-05-20): pybaseball 2.2.7's batting_stats no
                    # longer accepts pos=. The kwarg is passed through to
                    # FangraphsDataTable.fetch() which raises
                    # `TypeError: unexpected keyword argument 'pos'` — silently
                    # dropping this entire fallback path. Drop the kwarg and
                    # filter by position post-hoc instead.
                    fg_df = batting_stats(year, qual=0)
                    if fg_df is not None and not fg_df.empty:
                        # FanGraphs returns position column as "Pos" or
                        # "Positions" depending on pybaseball version.
                        pos_col = next((c for c in ("Pos", "Positions") if c in fg_df.columns), None)
                        if pos_col is not None:
                            fg_df = fg_df[fg_df[pos_col].astype(str).str.contains("C", na=False)]
                        if not fg_df.empty:
                            framing_data = fg_df
                            tier_used = "primary"
                            logger.info(
                                "T8 [primary]: Got %d catchers from FanGraphs batting_stats",
                                len(fg_df),
                            )
                except Exception as e:
                    logger.warning("T8 [primary]: FanGraphs catcher stats failed: %s", e)
        except ImportError:
            logger.warning("T8 [primary]: pybaseball not installed; skipping Tier 1")

        if framing_data is None:
            # statsapi-fielding sub-fallback (still Tier 1 conceptually)
            try:
                import statsapi as _statsapi

                conn_temp = get_connection()
                try:
                    catchers = pd.read_sql(
                        "SELECT player_id, name, mlb_id FROM players WHERE positions LIKE '%C%' AND is_hitter = 1",
                        conn_temp,
                    )
                finally:
                    conn_temp.close()

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
                                                "pop_time": 0.0,
                                                "framing_runs": 0.0,
                                            }
                                        )
                    except Exception:
                        continue

                if rows:
                    framing_data = pd.DataFrame(rows)
                    tier_used = "primary"
                    logger.info("T8 [primary]: Got %d catchers from statsapi fielding", len(rows))
            except Exception as e:
                logger.warning("T8 [primary]: statsapi catcher fallback failed: %s", e)

        # ── Tier 2: direct Savant scrape with browser headers (NEW) ───────
        if framing_data is None or (hasattr(framing_data, "empty") and framing_data.empty):
            try:
                logger.info("T8 [fallback]: Attempting direct Savant scrape with browser headers")
                scraped = _fetch_catcher_framing_savant_scrape(year)
                if scraped:
                    framing_data = pd.DataFrame(scraped)
                    tier_used = "fallback"
                    logger.info(
                        "T8 [fallback]: Got %d catchers from Savant browser-header scrape",
                        len(scraped),
                    )
            except Exception as e:
                logger.warning("T8 [fallback]: Savant scrape failed: %s", e)

        # ── Tier 3: shipped 2024 seed file (NEW) ──────────────────────────
        seed_used = False
        if framing_data is None or (hasattr(framing_data, "empty") and framing_data.empty):
            seed = _load_catcher_framing_seed()
            if seed:
                framing_data = pd.DataFrame(seed)
                tier_used = "emergency"
                seed_used = True
                logger.warning(
                    "T8 [emergency]: All live sources failed — using 2024 seed file (%d catchers)",
                    len(seed),
                )

        if framing_data is None or framing_data.empty:
            update_refresh_log(
                "catcher_framing",
                "no_data",
                rows_written=0,
                message="all framing sources returned empty (primary + Savant scrape + seed)",
            )
            return (
                "Skipped: all framing sources failed (Savant + FanGraphs + statsapi). "
                "Optimizer uses neutral 1.0× framing multiplier — see CLAUDE.md SF-7."
            )

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
                        # 2024 seed values stored under year=2024 to flag origin;
                        # live data uses current year.
                        2024 if seed_used else year,
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

        from src.database import update_refresh_log_auto

        tier_label = tier_used or "primary"
        msg_suffix = " [SEED]" if seed_used else ""
        # INFRA-F6: ImportError fallback that called update_refresh_log("...", "success", tier=...)
        # has been removed — update_refresh_log_auto is in src/database.py alongside
        # update_refresh_log and is always importable.
        status = update_refresh_log_auto(
            "catcher_framing",
            updated,
            expected_min=1,
            message=f"{updated} catchers from {len(framing_data)} rows (tier={tier_label}){msg_suffix}",
            tier=tier_label,
        )
        logger.info(
            "T8: Catcher framing — %d catchers updated (tier=%s)",
            updated,
            tier_used,
        )
        return f"Saved {updated} catcher framing profiles via tier={tier_used} ({status})"

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

        from src.database import get_connection, update_refresh_log, update_refresh_log_auto

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

                # SFH M1 (2026-05-20): commit per-batter so the write lock
                # releases between batters. Without this, the entire 50-batter
                # loop ran inside one transaction — holding the lock for the
                # duration of every Statcast fetch (1-3s each, network-bound)
                # and blowing through the 60s busy_timeout on parallel writers
                # (umpire_tendencies + catcher_framing). PR #69 bumping
                # busy_timeout 30s→60s was a band-aid; this is the root cause.
                conn.commit()

        finally:
            conn.close()

        if updated > 0:
            # INFRA-F6: row-count gate via update_refresh_log_auto (was bare "success").
            update_refresh_log_auto(
                "pvb_splits",
                updated,
                expected_min=1,
                message=f"{updated} new, {skipped} cached",
            )
        elif skipped > 0:
            update_refresh_log(
                "pvb_splits",
                "cached",
                rows_written=0,
                message=f"all {skipped} matchups already cached",
            )
        else:
            update_refresh_log("pvb_splits", "no_data", rows_written=0)
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

        from src.database import get_connection, update_refresh_log_auto

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

                conn.execute(
                    "INSERT INTO statcast_archive (player_id, season, bat_speed) VALUES (?, ?, ?) "
                    "ON CONFLICT(player_id, season) DO UPDATE SET bat_speed = excluded.bat_speed",
                    (pid, year, float(speed)),
                )
                updated += 1
            conn.commit()
        finally:
            conn.close()

        # INFRA-F6: row-count gate (was bare "success").
        update_refresh_log_auto(
            "bat_speed",
            updated,
            expected_min=100,
            message=f"{updated} players with bat speed",
        )
        return f"Updated {updated} players with bat speed"

    except Exception as exc:
        # SFH D: surface error in refresh_log so operator sees the failure.
        logger.warning("Bat speed fetch failed (non-fatal): %s", exc)
        from src.database import update_refresh_log

        update_refresh_log("bat_speed", "error", message=str(exc)[:200])
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
        from src.database import get_connection, update_refresh_log_auto

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

        # INFRA-F6: row-count gate (was bare "success"); ~30 teams × ~40 players.
        update_refresh_log_auto(
            "forty_man",
            updated,
            expected_min=800,
            message=f"{updated} 40-man roster entries",
        )
        return f"Updated {updated} 40-man roster entries"

    except Exception as exc:
        # SFH D: surface error in refresh_log so operator sees the failure.
        logger.warning("40-man roster fetch failed (non-fatal): %s", exc)
        from src.database import update_refresh_log

        update_refresh_log("forty_man", "error", message=str(exc)[:200])
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


def _bootstrap_injury_writeback(progress: BootstrapProgress) -> str:
    """Phase 33: Consolidate Yahoo + ESPN injuries into players.is_injured."""
    progress.phase = "Injury Writeback"
    progress.detail = "Updating player injury flags..."
    yahoo_skipped = False
    try:
        from src.database import get_connection, update_refresh_log_auto

        conn = get_connection()
        try:
            # 2026-05-17 broken-pipeline fix: if league_rosters is empty (Yahoo
            # disconnected on cold start), the old code would clear every
            # is_injured flag and never re-set them — losing days of IL data.
            # Skip the reset if the source-of-truth table is empty.
            lr_count = conn.execute(
                "SELECT COUNT(*) FROM league_rosters WHERE status IN ('IL10','IL15','IL60','DTD')"
            ).fetchone()[0]
            if lr_count == 0:
                # 2026-05-20 SFH B: mark the Yahoo-half as skipped so the
                # message reflects what actually happened. ESPN section
                # below still runs and provides the injured_count basis.
                logger.warning(
                    "Injury writeback: league_rosters has 0 IL-status rows; "
                    "skipping Yahoo reset to preserve existing flags"
                )
                conn.commit()
                yahoo_skipped = True
            else:
                # Step 1: Reset all injury flags
                conn.execute("UPDATE players SET is_injured = 0, injury_note = NULL")

                # Step 2: Set is_injured from Yahoo roster data (authoritative for league)
                conn.execute(
                    """UPDATE players SET is_injured = 1, injury_note = lr.status
                       FROM league_rosters lr
                       WHERE players.player_id = lr.player_id
                         AND lr.status IN ('IL10', 'IL15', 'IL60', 'DTD')"""
                )

                conn.commit()
        finally:
            conn.close()

        # Step 3: Set is_injured from ESPN injuries
        from src.espn_injuries import fetch_espn_injuries, update_player_injury_flags

        espn_injuries = fetch_espn_injuries()
        espn_count = update_player_injury_flags(espn_injuries)

        # Count total injured
        conn = get_connection()
        try:
            injured_count = conn.execute("SELECT COUNT(*) FROM players WHERE is_injured = 1").fetchone()[0]
        finally:
            conn.close()

        # 2026-05-20 SFH B: surface Yahoo-skipped state in the message so the
        # operator can tell "Yahoo had no IL rows" (preserved-flags path) from
        # "full reset + Yahoo+ESPN ran".
        # 2026-05-20 SFH B follow-up (LOW-3): when Yahoo is skipped,
        # injured_count mixes fresh ESPN flags with stale Yahoo flags from
        # a prior bootstrap. Split the count so the operator can tell
        # what's fresh from what's preserved.
        if yahoo_skipped:
            stale_yahoo_count = max(injured_count - espn_count, 0)
            message = (
                f"{injured_count} flagged ({espn_count} fresh ESPN + "
                f"{stale_yahoo_count} stale yahoo) [yahoo skipped: rosters empty]"
            )
            yahoo_note = " [yahoo skipped: rosters empty]"
        else:
            message = f"{injured_count} players flagged injured (ESPN: {espn_count})"
            yahoo_note = ""
        update_refresh_log_auto(
            "injury_writeback",
            injured_count,
            expected_min=10,
            message=message,
        )
        return f"Flagged {injured_count} players as injured{yahoo_note}"
    except Exception as exc:
        logger.exception("Injury writeback failed: %s", exc)
        from src.database import update_refresh_log

        update_refresh_log("injury_writeback", "error", message=str(exc)[:200])
        return f"Error: {exc}"


def _bootstrap_draft_results(progress: BootstrapProgress, yahoo_client=None) -> str:
    """Phase 32: Fetch draft results from Yahoo, flag rounds 1-3 as undroppable."""
    progress.phase = "Draft Results"
    progress.detail = "Fetching draft picks + undroppable flags..."
    if yahoo_client is None:
        return "Skipped (no Yahoo client)"
    try:
        from src.database import get_connection, save_league_draft_picks, update_refresh_log, update_refresh_log_auto

        df = yahoo_client.get_draft_results()
        if df.empty:
            update_refresh_log_auto("draft_results", 0, expected_min=200)
            return "No draft results available"

        # Save all draft picks (reuse existing db helper)
        saved = save_league_draft_picks(df)

        # Flag rounds 1-3 as undroppable in league_rosters (by player_id, not name)
        undroppable_pick_rows = df[df["round"] <= 3]
        # Prefer existing player_id from draft DF; fall back to name resolution
        undroppable_player_ids: list[int] = []
        conn = get_connection()
        try:
            conn.execute("UPDATE league_rosters SET is_undroppable = 0")
            for _, pick_row in undroppable_pick_rows.iterrows():
                pid_raw = pick_row.get("player_id")
                if pid_raw is not None and not (isinstance(pid_raw, float) and pid_raw != pid_raw):  # NaN check
                    try:
                        pid = int(pid_raw)
                    except (TypeError, ValueError):
                        pid = None
                else:
                    pid = None
                if pid is None:
                    # Resolve by name as last resort
                    name = pick_row.get("player_name", "")
                    row = conn.execute(
                        "SELECT player_id FROM players WHERE name = ? COLLATE NOCASE LIMIT 1", (name,)
                    ).fetchone()
                    pid = row[0] if row else None
                if pid is None:
                    logger.warning(
                        "Could not resolve player_id for draft pick: %s",
                        pick_row.to_dict(),
                    )
                    continue
                undroppable_player_ids.append(pid)
                conn.execute(
                    "UPDATE league_rosters SET is_undroppable = 1 WHERE player_id = ?",
                    (pid,),
                )
            conn.commit()
        finally:
            conn.close()

        update_refresh_log_auto(
            "draft_results",
            saved,
            expected_min=200,
            message=f"{saved} picks, {len(undroppable_player_ids)} undroppable",
        )
        return f"Saved {saved} picks, {len(undroppable_player_ids)} undroppable"
    except Exception as exc:
        logger.exception("Draft results fetch failed: %s", exc)
        from src.database import update_refresh_log

        update_refresh_log("draft_results", "error", message=str(exc)[:200])
        return f"Error: {exc}"


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
                source="players",
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
                source="extended_roster",
            )
        except Exception as exc:
            logger.warning("Extended roster bootstrap timed out or failed: %s", exc)
            results["extended_roster"] = f"Error: {exc}"
    else:
        results["extended_roster"] = "Fresh"

    # Phase 3c (Wave 9 INFRA-F5): Minor league rosters (AAA + AA top-30 per team)
    # Expands the player universe by ~900 rows. Runs after extended_roster so
    # MLB players are already canonicalized before we add minor-league rows.
    # 7-day staleness — minor-league rosters don't churn within a week.
    _notify(0.42)
    if force or check_staleness("minor_league_rosters", 168.0):
        try:
            results["minor_league_rosters"] = _run_with_timeout(
                lambda: _bootstrap_minor_league_rosters(progress),
                timeout=_PHASE_TIMEOUT_SECONDS,
                source="minor_league_rosters",
            )
        except Exception as exc:
            logger.warning("Minor league rosters bootstrap timed out or failed: %s", exc)
            results["minor_league_rosters"] = f"Error: {exc}"
    else:
        results["minor_league_rosters"] = "Fresh"

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
                source="yahoo_data",
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
    if force or check_staleness("adp_sources", staleness.adp_sources_hours):
        results["adp_sources"] = _bootstrap_adp_sources(progress)
    else:
        results["adp_sources"] = "Fresh"

    # Phase 9b: Depth charts (roles + lineup slots)
    _notify(0.93)
    if force or check_staleness("depth_charts", staleness.depth_charts_hours):
        results["depth_charts"] = _bootstrap_depth_charts(progress)
    else:
        results["depth_charts"] = "Fresh"

    # Phase 9c (Wave 10 / SF-84): Enrich generic 'P' pitcher rows with
    # SP/RP qualifiers using the depth_chart_role we just persisted plus
    # season_stats fallback. Always runs (cheap; idempotent) so freshly
    # imported players from earlier phases also get enriched.
    _notify(0.935)
    results["pitcher_positions"] = _enrich_pitcher_positions(progress)

    # Phase 10: Contract year data
    _notify(0.94)
    if force or check_staleness("contracts", staleness.contracts_hours):
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
                source="news_intelligence",
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
                source="ecr_consensus",
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
    # SF-18: gated by check_staleness("yahoo_transactions", 0.25h)
    if yahoo_client is not None and (force or check_staleness("yahoo_transactions", 0.25)):
        try:
            txn_df = yahoo_client.get_league_transactions()
            if not txn_df.empty:
                from src.database import update_refresh_log_auto
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
                # INFRA-F6: row-count gate (was bare "success"). Quiet periods may
                # have 0 txns matched by player_id — let the auto-status downgrade
                # those to "no_data" so the operator sees the empty write.
                update_refresh_log_auto(
                    "yahoo_transactions",
                    txn_stored,
                    expected_min=1,
                    message=f"{txn_stored} transactions stored from {len(txn_df)} fetched",
                )
            else:
                results["transactions"] = "No transactions"
        except Exception as exc:
            logger.warning("Transaction sync failed: %s", exc)
            results["transactions"] = f"Error: {exc}"
    elif yahoo_client is not None:
        results["transactions"] = "Fresh"

    # Phase 18: Yahoo free agents (BUG-019 fix)
    # SF-18: gated by check_staleness("yahoo_free_agents", 1.0h) and writes refresh_log on success
    if yahoo_client is not None and (force or check_staleness("yahoo_free_agents", 1.0)):
        try:
            progress.phase = "Yahoo Free Agents"
            progress.detail = "Fetching league free agents..."
            if on_progress:
                on_progress(progress)
            # 2026-05-20: wrap the FA fetch in a timeout to prevent indefinite
            # hangs when Yahoo aggressively rate-limits (429s with exponential
            # backoff × 500-FA pagination can exceed 10 minutes). On timeout,
            # fall back to whatever's in the yahoo_free_agents SQLite table from
            # a prior successful run — page 14 Free Agents will still load.
            import pandas as _pd

            _fa_holder: dict[str, _pd.DataFrame] = {}

            def _fetch_fa():
                # SFH L6 (2026-05-20): use the paginated method. Yahoo's API
                # caps single get_free_agents() calls at 25 results per page
                # regardless of the `count` parameter — so the previous
                # `count=200` returned only 25 FAs (every bootstrap stored
                # 25 to yahoo_free_agents instead of the intended ~500).
                # get_all_free_agents iterates start=0,25,50,... until the
                # API returns an empty page or max_players is reached.
                _fa_holder["df"] = yahoo_client.get_all_free_agents(max_players=500)
                return "ok"

            _result = _run_with_timeout(_fetch_fa, timeout=120, source="yahoo_free_agents")
            if _result.startswith("Timeout"):
                # 2026-05-20 SFH H-1: when the fetch times out, mark
                # refresh_log "skipped" and STOP — do NOT fall through to
                # the FA-storage block below, because fa_df would be empty
                # and the empty-branch would clobber "skipped" → "no_data",
                # losing the rate-limit signal operators need.
                logger.warning(
                    "Yahoo FA fetch hit 120s timeout — Yahoo likely rate-limiting. "
                    "Using cached yahoo_free_agents SQLite data from prior run."
                )
                # SFH H2 (2026-05-20): results string must NOT start with
                # "Timeout" — otherwise the end-of-bootstrap reconciliation
                # would overwrite our deliberate "skipped" status with
                # "timeout" (treating the cached-data fallback as a hard
                # failure). Use "Skipped:" prefix to signal "we handled it".
                results["yahoo_free_agents"] = "Skipped: Yahoo FA fetch hit 120s timeout — using cached SQLite data"
                from src.database import update_refresh_log

                update_refresh_log("yahoo_free_agents", "skipped")
            else:
                fa_df = _fa_holder.get("df", _pd.DataFrame())
                if not fa_df.empty:
                    from src.database import update_refresh_log_auto, upsert_player_bulk
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
                    from src.database import get_connection as _get_conn

                    _fa_conn = _get_conn()
                    try:
                        from datetime import UTC, datetime

                        _now = datetime.now(UTC).isoformat()
                        _today = datetime.now(UTC).strftime("%Y-%m-%d")
                        _ownership_writes = 0
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
                            # 2026-05-17 broken-pipeline fix: previously only rostered
                            # players wrote to ownership_trends, so FAs always read
                            # percent_owned=NULL in the player pool. Mirror the rostered
                            # write-through here so pool's percent_owned column populates
                            # for FAs too.
                            _fa_pname = row.get("player_name", "")
                            if _fa_pname:
                                _fa_pid = match_player_id(_fa_pname, "")
                                if _fa_pid is not None:
                                    _fa_conn.execute(
                                        "INSERT OR REPLACE INTO ownership_trends "
                                        "(player_id, date, percent_owned) VALUES (?, ?, ?)",
                                        (_fa_pid, _today, float(row.get("percent_owned", 0) or 0)),
                                    )
                                    _ownership_writes += 1
                        _fa_conn.commit()
                    finally:
                        _fa_conn.close()
                    results["yahoo_free_agents"] = (
                        f"Checked {len(fa_df)} FAs, added {new_players} new, stored {len(fa_df)} to yahoo_free_agents"
                    )
                    # INFRA-F6: row-count gate (was bare "success").
                    update_refresh_log_auto(
                        "yahoo_free_agents",
                        len(fa_df),
                        expected_min=100,
                        message=f"{len(fa_df)} FAs stored, {new_players} new players added",
                    )
                else:
                    # Fetch succeeded but Yahoo returned an empty list — distinct
                    # from the timeout path above (different remediation: API quirk
                    # vs rate-limit). Keep status="no_data" for this real signal.
                    results["yahoo_free_agents"] = "No FA data from Yahoo"
                    from src.database import update_refresh_log

                    update_refresh_log("yahoo_free_agents", "no_data")
        except Exception as exc:
            logger.warning("Yahoo FA fetch failed: %s", exc)
            results["yahoo_free_agents"] = f"Error: {exc}"
    elif yahoo_client is not None:
        results["yahoo_free_agents"] = "Fresh"

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

    # Phase 19: ROS Bayesian projections — moved here (post team_strength) per SF-17.
    # Bayesian update consumes team_strength; running before P21 risks stale inputs.
    # TTL fixed at 4h (was dynamic 0.25-1.0h) — PyMC MCMC is too expensive for the
    # short refresh window.
    _notify(0.975)
    if force or check_staleness("ros_projections", 4.0):
        progress.phase = "ROS Projections"
        progress.detail = "Updating Bayesian rest-of-season projections..."
        if on_progress:
            on_progress(progress)
        try:
            from src.bayesian import update_ros_projections

            def _ros_update():
                count = update_ros_projections()
                return f"Updated {count} ROS projections"

            results["ros_projections"] = _run_with_timeout(
                _ros_update, timeout=_TIMEOUT_ROS_PROJECTIONS, source="ros_projections"
            )
            logger.info("ROS Bayesian projections: %s", results["ros_projections"])
        except Exception as exc:
            logger.warning("ROS projection update failed: %s", exc)
            results["ros_projections"] = f"Error: {exc}"
    else:
        results["ros_projections"] = "Fresh"

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

    # Phase 25-26: T9 bat speed, T10 40-man (parallel, non-critical)
    # NOTE: BUG-008 — removed _bootstrap_dynamic_park_factors phase. It used
    # team OPS+/wRC+ (park-ADJUSTED metrics) as a park factor proxy, silently
    # overwriting correct Tier 1 / emergency park factors every 7 days. The
    # _bootstrap_park_factors phase already populates park_factors correctly.
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {}
        if force or check_staleness("bat_speed", staleness.bat_speed_hours):
            futures[executor.submit(_bootstrap_bat_speed, progress)] = "bat_speed"
        if force or check_staleness("forty_man", staleness.forty_man_hours):
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

    # Phase 31: Game logs for Player Databank (non-critical)
    if force or check_staleness("game_logs", staleness.game_logs_hours):
        try:
            results["game_logs"] = _run_with_timeout(
                lambda: _bootstrap_game_logs(progress, force=force),
                timeout=_TIMEOUT_GAME_LOGS,
                source="game_logs",
            )
        except Exception as exc:
            logger.warning("Game logs timed out or failed: %s", exc)
            results["game_logs"] = f"Error: {exc}"
    else:
        results["game_logs"] = "Fresh"

    # Phase 32: Draft results + undroppable flags (Yahoo-dependent, non-critical)
    _notify(0.99)
    try:
        results["draft_results"] = _bootstrap_draft_results(progress, yahoo_client)
    except Exception as exc:
        logger.warning("Draft results bootstrap failed: %s", exc)
        results["draft_results"] = f"Error: {exc}"

    # Phase 33: Injury writeback (consolidates Yahoo + ESPN → players.is_injured)
    _notify(0.995)
    try:
        results["injury_writeback"] = _bootstrap_injury_writeback(progress)
    except Exception as exc:
        logger.exception("Injury writeback failed: %s", exc)
        results["injury_writeback"] = f"Error: {exc}"

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

    # SFH H2 (2026-05-20): post-bootstrap reconciliation pass. Catches phase
    # failures whose per-phase error-write was itself blocked by a parallel
    # DB lock (umpire+catcher_framing vs pvb_splits) or by a timeout that
    # the orchestrator caught in results but never wrote to refresh_log.
    # Running here — AFTER all phases complete and write locks release —
    # ensures DataFreshnessTracker reflects the true status of this run.
    _reconcile_results_to_refresh_log(results)

    # Stamp all results into AnalyticsContext for data quality badges
    global _LAST_BOOTSTRAP_CTX  # noqa: PLW0603
    ctx = AnalyticsContext(pipeline="data_bootstrap")
    for source, result_msg in results.items():
        _stamp_from_result(ctx, source, result_msg)
    _LAST_BOOTSTRAP_CTX = ctx

    # Persist results to disk for post-mortem analysis (SF-14)
    try:
        results_path = _LOG_DIR / "bootstrap_results.json"
        results_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    except Exception:
        logger.debug("Failed to persist bootstrap_results.json", exc_info=True)

    return results

"""Zero-interaction data bootstrap pipeline.

Fetches all MLB player data from free APIs on app startup.
Uses staleness-based smart refresh to avoid unnecessary API calls.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from src.analytics_context import AnalyticsContext, DataQuality

logger = logging.getLogger(__name__)

# Module-level context from the last bootstrap run.
# Pages import this to show data freshness badges.
_LAST_BOOTSTRAP_CTX: AnalyticsContext | None = None


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
    live_stats_hours: float = 1  # 1 hour
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


# FanGraphs 2024 park factors (hitting) — update annually with new FG data.
# Values > 1.0 = hitter-friendly, < 1.0 = pitcher-friendly.
# OAK updated for 2025+ Sacramento Sutter Health Park (hot/dry, MiLB data suggests slightly hitter-friendly).
PARK_FACTORS: dict[str, float] = {
    "ARI": 1.06,
    "ATL": 1.01,
    "BAL": 1.03,
    "BOS": 1.04,
    "CHC": 1.02,
    "CWS": 1.01,
    "CIN": 1.08,
    "CLE": 0.97,
    "COL": 1.38,
    "DET": 0.96,
    "HOU": 1.00,
    "KC": 0.98,
    "LAA": 0.97,
    "LAD": 0.98,
    "MIA": 0.88,
    "MIL": 1.02,
    "MIN": 1.03,
    "NYM": 0.95,
    "NYY": 1.05,
    "OAK": 1.02,
    "PHI": 1.03,
    "PIT": 0.94,
    "SD": 0.93,
    "SF": 0.93,
    "SEA": 0.95,
    "STL": 0.98,
    "TB": 0.96,
    "TEX": 1.05,
    "TOR": 1.03,
    "WSH": 1.00,
}

# ── Lazy imports ─────────────────────────────────────────────────────
# These are imported inside functions to avoid circular import issues
# and to keep module-level imports minimal.


def _bootstrap_players(progress: BootstrapProgress) -> str:
    """Fetch all active MLB players and upsert to DB."""
    from src.database import update_refresh_log, upsert_player_bulk
    from src.live_stats import fetch_all_mlb_players

    progress.phase = "Players"
    progress.detail = "Fetching MLB roster..."
    try:
        df = fetch_all_mlb_players()
        if df.empty:
            return "No players returned from API"
        players = df.to_dict("records")
        count = upsert_player_bulk(players)
        update_refresh_log("players", "success")
        return f"Saved {count} players"
    except Exception as e:
        update_refresh_log("players", f"error: {e}")
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
        update_refresh_log("projections", f"error: {e}")
        return f"Error: {e}"


def _bootstrap_live_stats(progress: BootstrapProgress) -> str:
    """Fetch current season stats."""
    from src.database import update_refresh_log
    from src.live_stats import fetch_season_stats, save_season_stats_to_db

    progress.phase = "Live Stats"
    current_year = datetime.now(UTC).year
    progress.detail = f"Fetching {current_year} season stats..."
    try:
        df = fetch_season_stats(season=current_year)
        if not df.empty:
            count = save_season_stats_to_db(df)
            update_refresh_log("season_stats", "success")
            return f"Saved {count} player stats"
        return "No live stats available yet"
    except Exception as e:
        update_refresh_log("season_stats", f"error: {e}")
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
        update_refresh_log("historical_stats", f"error: {e}")
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
        update_refresh_log("injury_data", f"error: {e}")
        return f"Error: {e}"


def _bootstrap_park_factors(progress: BootstrapProgress) -> str:
    """Save hardcoded park factors to DB."""
    from src.database import update_refresh_log, upsert_park_factors

    progress.phase = "Park Factors"
    progress.detail = "Loading park factors..."
    try:
        # Pitching PF is typically ~85% of the hitting PF deviation from 1.0
        factors = [
            {
                "team_code": t,
                "factor_hitting": pf,
                "factor_pitching": 1.0 + (pf - 1.0) * 0.85,
            }
            for t, pf in PARK_FACTORS.items()
        ]
        count = upsert_park_factors(factors)
        update_refresh_log("park_factors", "success")
        return f"Saved {count} park factors"
    except Exception as e:
        update_refresh_log("park_factors", f"error: {e}")
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
        update_refresh_log("yahoo_data", f"error: {e}")
        return f"Error: {e}"


def _bootstrap_extended_roster(progress: BootstrapProgress) -> str:
    """Phase 8: Extended roster (40-man + spring training)."""
    progress.phase = "Extended Roster"
    progress.detail = "Fetching 40-man + spring training rosters..."
    try:
        from src.database import update_refresh_log, upsert_player_bulk
        from src.live_stats import fetch_extended_roster

        df = fetch_extended_roster()
        if df.empty:
            return "Extended roster: no data"
        upsert_player_bulk(df.to_dict("records"))
        update_refresh_log("extended_roster", "success")
        return f"Extended roster: {len(df)} players"
    except Exception as e:
        logger.warning("Extended roster bootstrap failed: %s", e)
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
    """Phase 9: Multi-source ADP (FantasyPros ECR + NFBC)."""
    progress.phase = "ADP Sources"
    progress.detail = "Fetching FantasyPros + NFBC ADP..."
    try:
        from src.adp_sources import fetch_fantasypros_ecr, fetch_nfbc_adp
        from src.database import update_refresh_log

        results = []
        ecr = fetch_fantasypros_ecr()
        if not ecr.empty:
            stored = _store_external_adp(ecr, "player_name", "ecr_rank", "fantasypros")
            results.append(f"FantasyPros: {len(ecr)} fetched, {stored} stored")
        nfbc = fetch_nfbc_adp()
        if not nfbc.empty:
            stored = _store_external_adp(nfbc, "player_name", "nfbc_adp", "nfbc")
            results.append(f"NFBC: {len(nfbc)} fetched, {stored} stored")
        if results:
            update_refresh_log("adp_sources", "success")
        else:
            update_refresh_log("adp_sources", "no_data")
        return f"ADP sources: {', '.join(results)}" if results else "ADP sources: no data"
    except Exception as e:
        logger.warning("ADP sources bootstrap failed: %s", e)
        return f"ADP sources: error ({e})"


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
    """Fetch depth charts and persist roles/lineup slots to DB."""
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
        return "Depth charts: no data"
    except Exception as e:
        logger.warning("Depth chart bootstrap failed: %s", e)
        return f"Depth charts: error ({e})"


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
    progress.detail = "Fetching team batting/pitching metrics from FanGraphs..."
    try:
        from src.game_day import fetch_team_strength

        df = fetch_team_strength(datetime.now(UTC).year)
        from src.database import update_refresh_log

        update_refresh_log("team_strength", "success")
        return f"Saved team strength for {len(df)} teams"
    except Exception as exc:
        logger.exception("Team strength bootstrap failed: %s", exc)
        from src.database import update_refresh_log

        update_refresh_log("team_strength", "error")
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

            update_refresh_log("stuff_plus", "error")
        except Exception:
            pass
        return f"Error: {exc}"


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

            update_refresh_log("batting_stats", "error")
        except Exception:
            pass
        return f"Error: {exc}"


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

        update_refresh_log("umpire_tendencies", "success")
        logger.info("T7: Umpire tendencies — %d umpires from %d games", updated, league_games)
        return f"Saved {updated} umpire profiles from {league_games} games"

    except Exception as exc:
        logger.exception("T7 umpire tendencies failed: %s", exc)
        try:
            from src.database import update_refresh_log

            update_refresh_log("umpire_tendencies", "error")
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

        # pybaseball doesn't have a direct catcher framing function,
        # so we use statcast_catcher_framing or fall back to FanGraphs data.
        # Try FanGraphs catching stats first (includes framing runs).
        framing_data = None
        try:
            from pybaseball import batting_stats

            # FanGraphs batting_stats for catchers include framing runs in advanced stats
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

        update_refresh_log("catcher_framing", "success")
        logger.info("T8: Catcher framing — %d catchers updated", updated)
        return f"Saved {updated} catcher framing profiles"

    except Exception as exc:
        logger.exception("T8 catcher framing failed: %s", exc)
        try:
            from src.database import update_refresh_log

            update_refresh_log("catcher_framing", "error")
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

            for _, hitter in hitter_sample.iterrows():
                batter_mlb_id = int(hitter["mlb_id"])
                batter_pid = int(hitter["player_id"])

                for pitcher_mlb_id in pitcher_ids:
                    pitcher_mlb_id = int(pitcher_mlb_id)

                    # Check if we already have recent data
                    existing = conn.execute(
                        """SELECT fetched_at FROM pvb_splits
                           WHERE batter_id = ? AND pitcher_id = ?""",
                        (batter_pid, pitcher_mlb_id),
                    ).fetchone()
                    if existing:
                        skipped += 1
                        continue

                    try:
                        pvb = _statcast_batter(
                            f"{year - 3}-01-01",
                            f"{year}-12-31",
                            batter_mlb_id,
                            pitcher_mlb_id,
                        )
                    except Exception:
                        continue

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
            teams = statsapi.get("teams", {"sportId": 1}).get("teams", [])
            updated = 0
            for team in teams:
                tid = team.get("id")
                if not tid:
                    continue
                try:
                    roster_data = statsapi.get("team_roster", {"teamId": tid, "rosterType": "40Man"})
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
        results["players"] = _bootstrap_players(progress)
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

    # Phases 4+5: Live stats + Historical (parallel — both independent)
    _notify(0.45)
    live_stale = force or check_staleness("season_stats", staleness.live_stats_hours)
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
        results["yahoo"] = _bootstrap_yahoo(progress, yahoo_client)
    else:
        results["yahoo"] = "Fresh"

    # Phase 8: Extended roster (40-man + spring training)
    _notify(0.91)
    if force or check_staleness("extended_roster", staleness.players_hours):
        results["extended_roster"] = _bootstrap_extended_roster(progress)
    else:
        results["extended_roster"] = "Fresh"

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
        results["news_intelligence"] = _bootstrap_news_intel(progress, yahoo_client)
    else:
        results["news_intelligence"] = "Fresh"

    # Phase 15: ECR consensus (depends on Phase 3 projections + Phase 9 ADP)
    _notify(0.99)
    if force or check_staleness("ecr_consensus", staleness.ecr_consensus_hours):
        results["ecr_consensus"] = _bootstrap_ecr_consensus(progress)
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
    if force or check_staleness("ros_projections", staleness.live_stats_hours):
        progress.phase = "ROS Projections"
        progress.detail = "Updating Bayesian rest-of-season projections..."
        if on_progress:
            on_progress(progress)
        try:
            from src.bayesian import update_ros_projections

            ros_count = update_ros_projections()
            results["ros_projections"] = f"Updated {ros_count} ROS projections"
            logger.info("ROS Bayesian projections: %d updated", ros_count)
        except Exception as exc:
            logger.warning("ROS projection update failed: %s", exc)
            results["ros_projections"] = f"Error: {exc}"
    else:
        results["ros_projections"] = "Fresh"

    # Phase 20+21: Game-day intelligence + Team strength (parallel)
    _notify(0.97)
    gd_stale = force or check_staleness("game_day", staleness.game_day_hours)
    ts_stale = force or check_staleness("team_strength", staleness.team_strength_hours)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}
        if gd_stale:
            futures[executor.submit(_bootstrap_game_day, progress)] = "game_day"
        else:
            results["game_day"] = "Fresh"
        if ts_stale:
            futures[executor.submit(_bootstrap_team_strength, progress)] = "team_strength"
        else:
            results["team_strength"] = "Fresh"
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as exc:
                logger.exception("Bootstrap %s failed: %s", key, exc)
                results[key] = f"Error: {exc}"

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

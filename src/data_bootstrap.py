"""Zero-interaction data bootstrap pipeline.

Fetches all MLB player data from free APIs on app startup.
Uses staleness-based smart refresh to avoid unnecessary API calls.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


@dataclass
class StalenessConfig:
    """Max age (hours) before each data source is refreshed."""

    players_hours: float = 168  # 7 days
    live_stats_hours: float = 1  # 1 hour
    projections_hours: float = 168  # 7 days
    historical_hours: float = 720  # 30 days
    park_factors_hours: float = 720  # 30 days
    yahoo_hours: float = 6  # 6 hours
    prospects_hours: float = 168  # 7 days
    news_hours: float = 1  # 1 hour
    ecr_consensus_hours: float = 24  # 24 hours


@dataclass
class BootstrapProgress:
    """Progress state for splash screen callback."""

    phase: str = ""
    detail: str = ""
    pct: float = 0.0


# FanGraphs 2024 park factors (publicly known, updated annually)
# Values > 1.0 = hitter-friendly, < 1.0 = pitcher-friendly
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
    "OAK": 0.96,
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

    progress.phase = "Historical"
    progress.detail = "Fetching 2023-2025 stats..."
    try:
        historical = fetch_historical_stats(seasons=[2023, 2024, 2025])
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

    progress.phase = "Injury Data"
    progress.detail = "Processing injury history..."
    try:
        if historical is None:
            historical = fetch_historical_stats(seasons=[2023, 2024, 2025])
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
        factors = [{"team_code": t, "factor_hitting": pf, "factor_pitching": pf} for t, pf in PARK_FACTORS.items()]
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
            results.append(f"FantasyPros: {len(ecr)}")
        nfbc = fetch_nfbc_adp()
        if not nfbc.empty:
            results.append(f"NFBC: {len(nfbc)}")
        update_refresh_log("adp_sources", "success")
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
    from src.database import check_staleness

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

    # Phase 2: Park factors (independent, fast)
    _notify(0.15)
    if force or check_staleness("park_factors", staleness.park_factors_hours):
        results["park_factors"] = _bootstrap_park_factors(progress)
    else:
        results["park_factors"] = "Fresh"

    # Phase 3: Projections (FanGraphs)
    _notify(0.30)
    if force or check_staleness("projections", staleness.projections_hours):
        results["projections"] = _bootstrap_projections(progress)
    else:
        results["projections"] = "Fresh"

    # Phase 4: Live stats (current season)
    _notify(0.50)
    if force or check_staleness("season_stats", staleness.live_stats_hours):
        results["live_stats"] = _bootstrap_live_stats(progress)
    else:
        results["live_stats"] = "Fresh"

    # Phase 5: Historical stats (3 years)
    _notify(0.65)
    historical_data = None
    if force or check_staleness("historical_stats", staleness.historical_hours):
        hist_result, historical_data = _bootstrap_historical(progress)
        results["historical"] = hist_result
    else:
        results["historical"] = "Fresh"

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
    _notify(0.82)
    if force or check_staleness("extended_roster", staleness.players_hours):
        results["extended_roster"] = _bootstrap_extended_roster(progress)
    else:
        results["extended_roster"] = "Fresh"

    # Phase 9: Multi-source ADP
    _notify(0.85)
    if force or check_staleness("adp_sources", 24):
        results["adp_sources"] = _bootstrap_adp_sources(progress)
    else:
        results["adp_sources"] = "Fresh"

    # Phase 9b: Depth charts (roles + lineup slots)
    _notify(0.87)
    if force or check_staleness("depth_charts", 168):
        results["depth_charts"] = _bootstrap_depth_charts(progress)
    else:
        results["depth_charts"] = "Fresh"

    # Phase 10: Contract year data
    _notify(0.88)
    if force or check_staleness("contracts", 720):
        results["contracts"] = _bootstrap_contracts(progress)
    else:
        results["contracts"] = "Fresh"

    # Phase 11: News/transactions
    _notify(0.91)
    if force or check_staleness("news", 6):
        results["news"] = _bootstrap_news(progress)
    else:
        results["news"] = "Fresh"

    # Phase 12: Deduplicate players (fix ID mismatches from different data sources)
    _notify(0.95)
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
    _notify(0.96)
    if force or check_staleness("prospect_rankings", staleness.prospects_hours):
        results["prospects"] = _bootstrap_prospects(progress)
    else:
        results["prospects"] = "Fresh"

    # Phase 14: News intelligence (multi-source)
    _notify(0.97)
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

    _notify(1.0)
    progress.phase = "Complete"
    progress.detail = "All data loaded!"
    logger.info("Bootstrap results: %s", results)
    return results

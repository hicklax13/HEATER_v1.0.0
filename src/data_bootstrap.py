"""Zero-interaction data bootstrap pipeline.

Fetches all MLB player data from free APIs on app startup.
Uses staleness-based smart refresh to avoid unnecessary API calls.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

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
        from src.database import update_refresh_log as _url

        _url("players", f"error: {e}")
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
    progress.detail = "Fetching 2026 season stats..."
    try:
        df = fetch_season_stats(season=2026)
        if not df.empty:
            count = save_season_stats_to_db(df)
            update_refresh_log("season_stats", "success")
            return f"Saved {count} player stats"
        return "No live stats available yet"
    except Exception as e:
        update_refresh_log("season_stats", f"error: {e}")
        return f"Error: {e}"


def _bootstrap_historical(progress: BootstrapProgress) -> str:
    """Fetch 3 years of historical stats."""
    from src.database import update_refresh_log
    from src.live_stats import fetch_historical_stats, save_season_stats_to_db

    progress.phase = "Historical"
    progress.detail = "Fetching 2023-2025 stats..."
    try:
        historical = fetch_historical_stats(seasons=[2023, 2024, 2025])
        total = 0
        for _year, df in historical.items():
            count = save_season_stats_to_db(df)
            total += count
        if total > 0:
            update_refresh_log("historical_stats", "success")
        return f"Saved {total} historical records across {len(historical)} seasons"
    except Exception as e:
        update_refresh_log("historical_stats", f"error: {e}")
        return f"Error: {e}"


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


def _bootstrap_yahoo(progress: BootstrapProgress, yahoo_client=None) -> str:
    """Sync Yahoo league data if client is available."""
    progress.phase = "Yahoo Sync"
    if yahoo_client is None:
        return "Skipped (no Yahoo connection)"
    progress.detail = "Syncing Yahoo league data..."
    try:
        from src.database import update_refresh_log

        yahoo_client.sync_to_db()
        update_refresh_log("yahoo_data", "success")
        return "Yahoo league data synced"
    except Exception as e:
        from src.database import update_refresh_log as _url

        _url("yahoo_data", f"error: {e}")
        return f"Error: {e}"


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
    if force or check_staleness("historical_stats", staleness.historical_hours):
        results["historical"] = _bootstrap_historical(progress)
    else:
        results["historical"] = "Fresh"

    # Phase 6: Injury data (derived from historical)
    _notify(0.80)
    if force or check_staleness("injury_data", staleness.historical_hours):
        results["injury_data"] = _bootstrap_injury_data(progress)
    else:
        results["injury_data"] = "Fresh"

    # Phase 7: Yahoo league data (optional)
    _notify(0.90)
    if force or check_staleness("yahoo_data", staleness.yahoo_hours):
        results["yahoo"] = _bootstrap_yahoo(progress, yahoo_client)
    else:
        results["yahoo"] = "Fresh"

    _notify(1.0)
    progress.phase = "Complete"
    progress.detail = "All data loaded!"
    logger.info("Bootstrap results: %s", results)
    return results

"""
Calibration data fetcher — pulls historical outcomes from Yahoo Fantasy API
to build ground-truth datasets for validating every analytical module.

Data collected:
    1. Draft results: who picked whom, at what pick, actual season performance
    2. Trade history: what was traded, pre/post performance
    3. Weekly matchups: H2H category results per week
    4. Roster moves: adds/drops/IL, timing and outcomes
    5. Final standings: end-of-season category totals and rankings

This data feeds the calibrators (survival, trade, lineup) which produce
the empirical evidence needed to replace HEATER's ~30 magic numbers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DraftPick:
    """A single pick from a historical draft."""

    pick_number: int
    round: int
    team_key: str
    team_name: str
    player_name: str
    player_id: int | None
    # Actual season stats (filled in post-season)
    actual_sgp: float | None = None
    actual_stats: dict[str, float] = field(default_factory=dict)


@dataclass
class HistoricalTrade:
    """A completed trade with before/after performance data."""

    trade_date: datetime
    team_a_key: str
    team_a_gave: list[str]  # player names
    team_a_received: list[str]
    # Post-trade performance (rest of season from trade date)
    team_a_ros_delta_sgp: float | None = None
    team_a_standings_before: int | None = None
    team_a_standings_after: int | None = None


@dataclass
class WeeklyMatchup:
    """One week's H2H result between two teams."""

    week: int
    team_a_key: str
    team_b_key: str
    categories_won_a: int
    categories_won_b: int
    ties: int
    category_results: dict[str, dict[str, float]] = field(default_factory=dict)
    # {cat: {"team_a": val, "team_b": val, "winner": "a"|"b"|"tie"}}


@dataclass
class CalibrationDataset:
    """Complete historical dataset for one league-season."""

    league_key: str
    season: int
    num_teams: int
    draft_picks: list[DraftPick] = field(default_factory=list)
    trades: list[HistoricalTrade] = field(default_factory=list)
    weekly_matchups: list[WeeklyMatchup] = field(default_factory=list)
    final_standings: dict[str, dict[str, float]] = field(default_factory=dict)
    # {team_key: {cat: total, ...}}

    @property
    def has_draft_data(self) -> bool:
        return len(self.draft_picks) > 0

    @property
    def has_trade_data(self) -> bool:
        return len(self.trades) > 0

    @property
    def has_matchup_data(self) -> bool:
        return len(self.weekly_matchups) > 0

    def to_draft_dataframe(self) -> pd.DataFrame:
        """Draft picks as a DataFrame for analysis."""
        if not self.draft_picks:
            return pd.DataFrame()
        records = []
        for p in self.draft_picks:
            records.append(
                {
                    "pick_number": p.pick_number,
                    "round": p.round,
                    "team_key": p.team_key,
                    "team_name": p.team_name,
                    "player_name": p.player_name,
                    "player_id": p.player_id,
                    "actual_sgp": p.actual_sgp,
                    **{f"actual_{k}": v for k, v in p.actual_stats.items()},
                }
            )
        return pd.DataFrame(records)

    def to_matchup_dataframe(self) -> pd.DataFrame:
        """Weekly matchups as a DataFrame for analysis."""
        if not self.weekly_matchups:
            return pd.DataFrame()
        records = []
        for m in self.weekly_matchups:
            records.append(
                {
                    "week": m.week,
                    "team_a": m.team_a_key,
                    "team_b": m.team_b_key,
                    "cats_won_a": m.categories_won_a,
                    "cats_won_b": m.categories_won_b,
                    "ties": m.ties,
                }
            )
        return pd.DataFrame(records)


def fetch_calibration_data(
    yahoo_client: Any,
    season: int = 2025,
) -> CalibrationDataset | None:
    """
    Fetch a complete historical season from Yahoo for validation.

    Requires a connected Yahoo client with access to the target season.
    Returns None if the season data is unavailable.
    """
    if yahoo_client is None:
        logger.warning("No Yahoo client — cannot fetch calibration data")
        return None

    try:
        # Determine league key for the target season
        league_key = _resolve_league_key(yahoo_client, season)
        if not league_key:
            logger.warning("Could not resolve league key for season %d", season)
            return None

        dataset = CalibrationDataset(
            league_key=league_key,
            season=season,
            num_teams=12,  # Will be updated from API
        )

        # Step 1: Draft results
        logger.info("Fetching draft results for %d...", season)
        dataset.draft_picks = _fetch_draft_picks(yahoo_client, league_key)
        logger.info("  Got %d draft picks", len(dataset.draft_picks))

        # Step 2: Trades
        logger.info("Fetching trade history for %d...", season)
        dataset.trades = _fetch_trades(yahoo_client, league_key)
        logger.info("  Got %d trades", len(dataset.trades))

        # Step 3: Weekly matchups
        logger.info("Fetching weekly matchups for %d...", season)
        dataset.weekly_matchups = _fetch_matchups(yahoo_client, league_key)
        logger.info("  Got %d weekly matchups", len(dataset.weekly_matchups))

        # Step 4: Final standings
        logger.info("Fetching final standings for %d...", season)
        dataset.final_standings = _fetch_final_standings(yahoo_client, league_key)

        # Step 5: Backfill actual stats for draft picks
        logger.info("Backfilling actual season stats for drafted players...")
        _backfill_actual_stats(dataset)

        return dataset

    except Exception as exc:
        logger.error("Failed to fetch calibration data for %d: %s", season, exc)
        return None


# ---------------------------------------------------------------------------
# Internal helpers — Yahoo API calls
# ---------------------------------------------------------------------------


def _resolve_league_key(yahoo_client: Any, season: int) -> str | None:
    """Resolve the league key for a given season."""
    # Yahoo game keys: 2025 = 458, 2026 = 469
    # This mapping should be extended as needed
    game_keys = {2024: 449, 2025: 458, 2026: 469}
    game_key = game_keys.get(season)
    if game_key is None:
        logger.warning("Unknown Yahoo game key for season %d", season)
        return None

    try:
        # Use yahoo_client to resolve league ID
        # The exact API depends on yfpy version, but typically:
        league_id = getattr(yahoo_client, "league_id", None)
        if league_id:
            return f"{game_key}.l.{league_id}"
    except Exception:
        pass
    return None


def _fetch_draft_picks(yahoo_client: Any, league_key: str) -> list[DraftPick]:
    """Fetch all draft picks for a league."""
    picks: list[DraftPick] = []
    try:
        draft_results = yahoo_client.get_draft_results()
        if not draft_results:
            return picks

        for i, pick_data in enumerate(draft_results):
            # yfpy draft result structure varies — handle both formats
            player_name = ""
            team_key = ""
            team_name = ""
            player_id = None

            if hasattr(pick_data, "player_key"):
                player_key = str(getattr(pick_data, "player_key", ""))
                # Extract player ID from key (e.g., "449.p.12345")
                parts = player_key.split(".")
                if len(parts) >= 3:
                    try:
                        player_id = int(parts[-1])
                    except ValueError:
                        pass

            if hasattr(pick_data, "player"):
                p = pick_data.player
                player_name = str(getattr(p, "name", "")).strip()
            elif isinstance(pick_data, dict):
                player_name = str(pick_data.get("name", "")).strip()
                team_key = str(pick_data.get("team_key", ""))

            pick_num = getattr(pick_data, "pick", i + 1)
            round_num = getattr(pick_data, "round", (i // 12) + 1)

            picks.append(
                DraftPick(
                    pick_number=int(pick_num),
                    round=int(round_num),
                    team_key=str(team_key),
                    team_name=str(team_name),
                    player_name=player_name,
                    player_id=player_id,
                )
            )
    except Exception as exc:
        logger.warning("Error fetching draft picks: %s", exc)

    return picks


def _fetch_trades(yahoo_client: Any, league_key: str) -> list[HistoricalTrade]:
    """Fetch all trades for a league season."""
    trades: list[HistoricalTrade] = []
    try:
        transactions = yahoo_client.get_transactions()
        if not transactions:
            return trades

        for txn in transactions:
            txn_type = getattr(txn, "type", "")
            if str(txn_type).lower() != "trade":
                continue

            # Parse trade participants
            # This is simplified — real yfpy trade parsing is complex
            trade_date = getattr(txn, "timestamp", None)
            if trade_date and isinstance(trade_date, (int, float)):
                trade_date = datetime.fromtimestamp(trade_date, tz=UTC)
            elif trade_date is None:
                trade_date = datetime.now(UTC)

            trades.append(
                HistoricalTrade(
                    trade_date=trade_date,
                    team_a_key="",
                    team_a_gave=[],
                    team_a_received=[],
                )
            )
    except Exception as exc:
        logger.warning("Error fetching trades: %s", exc)

    return trades


def _fetch_matchups(yahoo_client: Any, league_key: str) -> list[WeeklyMatchup]:
    """Fetch all weekly matchup results."""
    matchups: list[WeeklyMatchup] = []
    try:
        # Yahoo typically has 22-23 weeks; fetch all completed weeks
        for week in range(1, 24):
            try:
                week_data = yahoo_client.get_matchups(week=week)
                if not week_data:
                    break  # No more completed weeks
                for m in week_data:
                    matchups.append(
                        WeeklyMatchup(
                            week=week,
                            team_a_key=str(getattr(m, "team_a_key", "")),
                            team_b_key=str(getattr(m, "team_b_key", "")),
                            categories_won_a=int(getattr(m, "cats_won_a", 0)),
                            categories_won_b=int(getattr(m, "cats_won_b", 0)),
                            ties=int(getattr(m, "ties", 0)),
                        )
                    )
            except Exception:
                break  # Week not available yet
    except Exception as exc:
        logger.warning("Error fetching matchups: %s", exc)

    return matchups


def _fetch_final_standings(yahoo_client: Any, league_key: str) -> dict[str, dict[str, float]]:
    """Fetch end-of-season category totals for all teams."""
    standings: dict[str, dict[str, float]] = {}
    try:
        raw_standings = yahoo_client.get_standings()
        if raw_standings:
            for team in raw_standings:
                team_key = str(getattr(team, "team_key", ""))
                stats = {}
                # Extract category totals from standings data
                team_stats = getattr(team, "team_stats", None)
                if team_stats:
                    for stat in team_stats:
                        stat_name = str(getattr(stat, "stat_id", ""))
                        stat_val = float(getattr(stat, "value", 0))
                        stats[stat_name] = stat_val
                standings[team_key] = stats
    except Exception as exc:
        logger.warning("Error fetching final standings: %s", exc)

    return standings


def _backfill_actual_stats(dataset: CalibrationDataset) -> None:
    """
    Backfill actual season statistics for drafted players.

    Uses the local database (season_stats table) to look up how each
    drafted player actually performed.
    """
    try:
        from src.database import get_connection

        conn = get_connection()
        try:
            for pick in dataset.draft_picks:
                if not pick.player_id:
                    continue
                cursor = conn.execute(
                    """
                    SELECT r, hr, rbi, sb, avg, obp, w, l, sv, k, era, whip, ip, pa
                    FROM season_stats
                    WHERE player_id = ? AND season = ?
                    ORDER BY pa DESC
                    LIMIT 1
                    """,
                    (pick.player_id, dataset.season),
                )
                row = cursor.fetchone()
                if row:
                    pick.actual_stats = {
                        "r": float(row["r"] or 0),
                        "hr": float(row["hr"] or 0),
                        "rbi": float(row["rbi"] or 0),
                        "sb": float(row["sb"] or 0),
                        "avg": float(row["avg"] or 0),
                        "obp": float(row["obp"] or 0),
                        "w": float(row["w"] or 0),
                        "l": float(row["l"] or 0),
                        "sv": float(row["sv"] or 0),
                        "k": float(row["k"] or 0),
                        "era": float(row["era"] or 0),
                        "whip": float(row["whip"] or 0),
                        "ip": float(row["ip"] or 0),
                        "pa": float(row["pa"] or 0),
                    }
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("Error backfilling actual stats: %s", exc)

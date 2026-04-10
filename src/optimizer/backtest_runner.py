"""Historical backtest runner for the Lineup Optimizer.

Replays past MLB weeks through the optimizer pipeline, comparing
recommended lineups against actual player performance to measure
prediction accuracy and lineup decision quality.

Uses MLB Stats API (statsapi) to fetch game logs for historical
date ranges, aggregates into weekly totals, and scores accuracy
with RMSE, Spearman rank correlation, bust rate, and lineup grades.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from src.optimizer.backtest_validator import (
    compute_bust_rate,
    compute_projection_rmse,
    compute_rank_correlation,
    grade_lineup_quality,
)

logger = logging.getLogger(__name__)


# ── Data Classes ──────────────────────────────────────────────────────


@dataclass
class WeekResult:
    """Results from backtesting one week."""

    week_start: date
    week_end: date
    projection_rmse: float
    rank_correlation: float
    bust_rate: float
    lineup_grade: str
    n_players: int
    category_rmse: dict[str, float] = field(default_factory=dict)


@dataclass
class BacktestReport:
    """Aggregated results across all backtested weeks."""

    weeks: list[WeekResult]
    avg_rmse: float
    avg_rank_correlation: float
    avg_bust_rate: float
    grade_distribution: dict[str, int]


# ── Constants ─────────────────────────────────────────────────────────

HITTING_CATEGORIES: list[str] = ["r", "hr", "rbi", "sb", "avg", "obp"]
PITCHING_CATEGORIES: list[str] = ["w", "l", "sv", "k", "era", "whip"]
ALL_CATEGORIES: list[str] = HITTING_CATEGORIES + PITCHING_CATEGORIES

# Well-known MLB player IDs for backtesting when no roster is available.
# These represent a mix of top hitters and pitchers from recent seasons.
BACKTEST_PLAYER_IDS: dict[int, str] = {
    # Hitters
    592450: "Aaron Judge",
    660271: "Shohei Ohtani",
    665742: "Juan Soto",
    605141: "Mookie Betts",
    641355: "Corey Seager",
    608324: "Kyle Tucker",
    656941: "Bobby Witt Jr.",
    672284: "Gunnar Henderson",
    668939: "Julio Rodriguez",
    681297: "Corbin Carroll",
    # Pitchers
    477132: "Cole Ragans",
    571945: "Gerrit Cole",
    675911: "Spencer Strider",
    592662: "Zack Wheeler",
    663993: "Tarik Skubal",
    656302: "Logan Webb",
    543037: "Corbin Burnes",
    669373: "Paul Skenes",
    621111: "Dylan Cease",
    657006: "Framber Valdez",
}


# ── Fetching Actual Stats ─────────────────────────────────────────────


def _parse_ip(ip_str: str) -> float:
    """Parse MLB inningsPitched string to float.

    MLB format uses '6.1' to mean 6 and 1/3 innings, '6.2' for 6 and 2/3.
    """
    try:
        parts = str(ip_str).split(".")
        full = int(parts[0])
        thirds = int(parts[1]) if len(parts) > 1 else 0
        return full + thirds / 3.0
    except (ValueError, IndexError):
        return 0.0


def _aggregate_hitting_games(games: list[dict]) -> dict[str, float]:
    """Aggregate per-game hitting stat dicts into weekly totals.

    Computes rate stats from components: AVG = H/AB, OBP = (H+BB+HBP)/(AB+BB+HBP+SF).
    """
    if not games:
        return {}

    h = sum(g.get("hits", 0) for g in games)
    ab = sum(g.get("atBats", 0) for g in games)
    hr = sum(g.get("homeRuns", 0) for g in games)
    rbi = sum(g.get("rbi", 0) for g in games)
    sb = sum(g.get("stolenBases", 0) for g in games)
    r = sum(g.get("runs", 0) for g in games)
    bb = sum(g.get("baseOnBalls", 0) for g in games)
    hbp = sum(g.get("hitByPitch", 0) for g in games)
    sf = sum(g.get("sacFlies", 0) for g in games)

    avg = h / ab if ab > 0 else 0.0
    obp_denom = ab + bb + hbp + sf
    obp = (h + bb + hbp) / obp_denom if obp_denom > 0 else 0.0

    return {
        "r": float(r),
        "hr": float(hr),
        "rbi": float(rbi),
        "sb": float(sb),
        "avg": round(avg, 3),
        "obp": round(obp, 3),
        "h": float(h),
        "ab": float(ab),
        "bb": float(bb),
        "hbp": float(hbp),
        "sf": float(sf),
        "games": float(len(games)),
    }


def _aggregate_pitching_games(games: list[dict]) -> dict[str, float]:
    """Aggregate per-game pitching stat dicts into weekly totals.

    Computes rate stats from components: ERA = ER*9/IP, WHIP = (BB+H)/IP.
    """
    if not games:
        return {}

    total_ip = 0.0
    for g in games:
        ip_str = g.get("inningsPitched", "0")
        total_ip += _parse_ip(ip_str)

    k = sum(g.get("strikeOuts", 0) for g in games)
    w = sum(g.get("wins", 0) for g in games)
    l_val = sum(g.get("losses", 0) for g in games)
    sv = sum(g.get("saves", 0) for g in games)
    er = sum(g.get("earnedRuns", 0) for g in games)
    bb_allowed = sum(g.get("baseOnBalls", 0) for g in games)
    h_allowed = sum(g.get("hits", 0) for g in games)

    era = (er * 9 / total_ip) if total_ip > 0 else 0.0
    whip = (bb_allowed + h_allowed) / total_ip if total_ip > 0 else 0.0

    return {
        "w": float(w),
        "l": float(l_val),
        "sv": float(sv),
        "k": float(k),
        "era": round(era, 2),
        "whip": round(whip, 2),
        "ip": round(total_ip, 1),
        "er": float(er),
        "bb_allowed": float(bb_allowed),
        "h_allowed": float(h_allowed),
        "games": float(len(games)),
    }


def _fetch_player_game_logs(
    player_id: int,
    player_name: str,
    week_start: date,
    week_end: date,
) -> dict[str, float] | None:
    """Fetch game logs for a single player within a date range.

    Tries hitting first, then pitching. Filters game log entries
    to only include games within the specified week.

    Returns:
        Aggregated stat dict, or None if no data found.
    """
    try:
        import statsapi
    except ImportError:
        return None

    start_str = str(week_start)
    end_str = str(week_end)

    # Try hitting game log
    try:
        result = statsapi.player_stat_data(
            player_id,
            group="hitting",
            type="gameLog",
            sportId=1,
        )
        all_games = result.get("stats", [])
        # Filter to the target week by date
        week_games = []
        for entry in all_games:
            game_date = entry.get("date", "")
            if game_date and start_str <= game_date <= end_str:
                week_games.append(entry.get("stats", entry))

        if week_games:
            stats = _aggregate_hitting_games(week_games)
            stats["player_id"] = float(player_id)
            stats["player_name"] = player_name
            stats["is_hitter"] = 1.0
            return stats
    except Exception:
        logger.debug("No hitting gameLog for %s (ID %d)", player_name, player_id)

    time.sleep(0.3)

    # Try pitching game log
    try:
        result = statsapi.player_stat_data(
            player_id,
            group="pitching",
            type="gameLog",
            sportId=1,
        )
        all_games = result.get("stats", [])
        week_games = []
        for entry in all_games:
            game_date = entry.get("date", "")
            if game_date and start_str <= game_date <= end_str:
                week_games.append(entry.get("stats", entry))

        if week_games:
            stats = _aggregate_pitching_games(week_games)
            stats["player_id"] = float(player_id)
            stats["player_name"] = player_name
            stats["is_hitter"] = 0.0
            return stats
    except Exception:
        logger.debug("No pitching gameLog for %s (ID %d)", player_name, player_id)

    return None


def fetch_weekly_actuals(
    week_start: date,
    week_end: date,
    player_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Fetch actual MLB stats for a specific week.

    Uses MLB Stats API (statsapi) to get game logs for each player
    in the date range, then aggregates into weekly totals.

    Args:
        week_start: Monday of the week.
        week_end: Sunday of the week.
        player_ids: Optional list of MLB player IDs to fetch.
            If None, uses BACKTEST_PLAYER_IDS.

    Returns:
        DataFrame with columns: player_id, player_name, is_hitter,
        r, hr, rbi, sb, avg, obp, w, l, sv, k, era, whip, ip,
        h, ab, games.  Empty DataFrame if API unavailable.
    """
    try:
        import statsapi  # noqa: F401
    except ImportError:
        logger.warning("statsapi not available; returning empty actuals")
        return pd.DataFrame()

    if player_ids is None:
        player_ids_map = BACKTEST_PLAYER_IDS
    else:
        # Build name map from the constant or use generic names
        player_ids_map = {pid: BACKTEST_PLAYER_IDS.get(pid, f"Player {pid}") for pid in player_ids}

    rows: list[dict] = []
    for pid, name in player_ids_map.items():
        stats = _fetch_player_game_logs(pid, name, week_start, week_end)
        if stats is not None:
            rows.append(stats)
        time.sleep(0.3)  # Rate limit courtesy

    if not rows:
        logger.warning(
            "No actuals fetched for week %s to %s",
            week_start,
            week_end,
        )
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Ensure all expected columns exist with defaults
    for col in ALL_CATEGORIES + ["ip", "h", "ab", "games", "player_id", "player_name"]:
        if col not in df.columns:
            df[col] = 0.0

    return df


# ── Projection Simulation ─────────────────────────────────────────────


def _build_projected_stats(
    roster: pd.DataFrame,
    actual_ids: list[int],
) -> pd.DataFrame:
    """Extract projected weekly stats from a roster DataFrame.

    Scales season-long projections to weekly estimates (divide counting
    stats by 26 weeks). Rate stats are kept as-is.

    Args:
        roster: Roster with projected stat columns.
        actual_ids: Player IDs that have actual data (for filtering).

    Returns:
        DataFrame with projected stats, filtered to players with actuals.
    """
    WEEKS_IN_SEASON = 26

    proj = roster.copy()

    # Filter to players that have actual data
    id_col = "mlb_id" if "mlb_id" in proj.columns else "player_id"
    if id_col in proj.columns:
        proj = proj[proj[id_col].isin(actual_ids)]

    if proj.empty:
        return proj

    # Scale counting stats to weekly
    counting_stats = ["r", "hr", "rbi", "sb", "w", "l", "sv", "k"]
    for stat in counting_stats:
        if stat in proj.columns:
            proj[stat] = pd.to_numeric(proj[stat], errors="coerce").fillna(0) / WEEKS_IN_SEASON

    return proj


# ── Single Week Backtest ──────────────────────────────────────────────


def backtest_week(
    week_start: date,
    week_end: date,
    roster: pd.DataFrame,
    player_ids: list[int] | None = None,
) -> WeekResult:
    """Run a single-week backtest.

    1. Fetch actual stats for the week from MLB Stats API.
    2. Scale roster projections to weekly estimates.
    3. Compare projected vs actual using scoring functions.
    4. Grade lineup quality.

    Args:
        week_start: Monday of the week.
        week_end: Sunday of the week.
        roster: Roster DataFrame with season-long projected stats.
        player_ids: Optional player IDs to fetch actuals for.

    Returns:
        WeekResult with accuracy metrics.
    """
    logger.info("Backtesting week: %s to %s", week_start, week_end)

    # Fetch actuals
    actuals = fetch_weekly_actuals(week_start, week_end, player_ids)

    if actuals.empty:
        logger.warning("No actuals for week %s -- returning default result", week_start)
        return WeekResult(
            week_start=week_start,
            week_end=week_end,
            projection_rmse=float("inf"),
            rank_correlation=0.0,
            bust_rate=0.0,
            lineup_grade="C",
            n_players=0,
        )

    actual_ids = actuals["player_id"].astype(int).tolist()

    # Build projected weekly stats
    projected = _build_projected_stats(roster, actual_ids)

    if projected.empty:
        logger.warning(
            "No projected players matched actuals for week %s",
            week_start,
        )
        return WeekResult(
            week_start=week_start,
            week_end=week_end,
            projection_rmse=float("inf"),
            rank_correlation=0.0,
            bust_rate=0.0,
            lineup_grade="C",
            n_players=0,
        )

    # Compute projection RMSE per category
    category_rmse: dict[str, float] = {}
    for cat in ALL_CATEGORIES:
        if cat not in projected.columns or cat not in actuals.columns:
            continue
        proj_vals = pd.to_numeric(projected[cat], errors="coerce").fillna(0)
        act_vals = pd.to_numeric(actuals[cat], errors="coerce").fillna(0)

        # Match on length (use min of both)
        n = min(len(proj_vals), len(act_vals))
        if n == 0:
            continue

        proj_dict = {f"cat_{i}": float(proj_vals.iloc[i]) for i in range(n)}
        act_dict = {f"cat_{i}": float(act_vals.iloc[i]) for i in range(n)}
        category_rmse[cat] = compute_projection_rmse(proj_dict, act_dict)

    # Overall RMSE across categories
    proj_totals: dict[str, float] = {}
    act_totals: dict[str, float] = {}
    for cat in ALL_CATEGORIES:
        if cat in projected.columns:
            proj_totals[cat] = float(pd.to_numeric(projected[cat], errors="coerce").fillna(0).sum())
        if cat in actuals.columns:
            act_totals[cat] = float(pd.to_numeric(actuals[cat], errors="coerce").fillna(0).sum())

    overall_rmse = compute_projection_rmse(proj_totals, act_totals)

    # Rank correlation on HR (representative counting stat)
    rank_stat = "hr"
    if rank_stat in projected.columns and rank_stat in actuals.columns:
        id_col = "mlb_id" if "mlb_id" in projected.columns else "player_id"
        proj_ranked = projected[[id_col, rank_stat]].sort_values(rank_stat, ascending=False)
        act_ranked = actuals[["player_id", rank_stat]].sort_values(rank_stat, ascending=False)
        proj_order = proj_ranked[id_col].astype(int).tolist()
        act_order = act_ranked["player_id"].astype(int).tolist()
        rank_corr = compute_rank_correlation(proj_order, act_order)
    else:
        rank_corr = 0.0

    # Bust rate on HR
    if rank_stat in projected.columns and rank_stat in actuals.columns:
        id_col = "mlb_id" if "mlb_id" in projected.columns else "player_id"
        proj_bust = dict(
            zip(
                projected[id_col].astype(int),
                pd.to_numeric(projected[rank_stat], errors="coerce").fillna(0),
                strict=False,
            )
        )
        act_bust = dict(
            zip(
                actuals["player_id"].astype(int),
                pd.to_numeric(actuals[rank_stat], errors="coerce").fillna(0),
                strict=False,
            )
        )
        bust_rate = compute_bust_rate(proj_bust, act_bust)
    else:
        bust_rate = 0.0

    # Lineup grade: compare total projected value vs total actual value
    proj_total_value = sum(proj_totals.get(c, 0) for c in HITTING_CATEGORIES if c not in ("avg", "obp"))
    act_total_value = sum(act_totals.get(c, 0) for c in HITTING_CATEGORIES if c not in ("avg", "obp"))
    lineup_grade = grade_lineup_quality(proj_total_value, max(act_total_value, 0.01))

    n_players = min(len(projected), len(actuals))

    return WeekResult(
        week_start=week_start,
        week_end=week_end,
        projection_rmse=overall_rmse,
        rank_correlation=rank_corr,
        bust_rate=bust_rate,
        lineup_grade=lineup_grade,
        n_players=n_players,
        category_rmse=category_rmse,
    )


# ── Multi-Week Runner ─────────────────────────────────────────────────


def run_backtest(
    weeks: list[tuple[date, date]],
    roster: pd.DataFrame,
    player_ids: list[int] | None = None,
) -> BacktestReport:
    """Run backtest across multiple historical weeks.

    For each week:
    1. Fetch actual stats from MLB Stats API.
    2. Scale roster projections to weekly estimates.
    3. Compare projected vs actual using scoring functions.
    4. Grade lineup quality.

    Args:
        weeks: List of (start_date, end_date) tuples for each week.
        roster: Roster DataFrame with season-long projected stats.
        player_ids: Optional player IDs to limit the backtest.

    Returns:
        BacktestReport with per-week and aggregate results.
    """
    results: list[WeekResult] = []

    for week_start, week_end in weeks:
        result = backtest_week(week_start, week_end, roster, player_ids)
        results.append(result)
        logger.info(
            "Week %s: RMSE=%.3f, rho=%.3f, bust=%.1f%%, grade=%s (%d players)",
            result.week_start,
            result.projection_rmse,
            result.rank_correlation,
            result.bust_rate * 100,
            result.lineup_grade,
            result.n_players,
        )

    # Aggregate
    valid_results = [r for r in results if r.n_players > 0]

    if valid_results:
        avg_rmse = float(np.mean([r.projection_rmse for r in valid_results if np.isfinite(r.projection_rmse)]) or 0.0)
        avg_rank_corr = float(np.mean([r.rank_correlation for r in valid_results]))
        avg_bust = float(np.mean([r.bust_rate for r in valid_results]))
    else:
        avg_rmse = float("inf")
        avg_rank_corr = 0.0
        avg_bust = 0.0

    grade_dist: dict[str, int] = {"A": 0, "B": 0, "C": 0}
    for r in results:
        grade_dist[r.lineup_grade] = grade_dist.get(r.lineup_grade, 0) + 1

    return BacktestReport(
        weeks=results,
        avg_rmse=avg_rmse,
        avg_rank_correlation=avg_rank_corr,
        avg_bust_rate=avg_bust,
        grade_distribution=grade_dist,
    )


# ── Report Formatting ─────────────────────────────────────────────────


def format_report(report: BacktestReport) -> str:
    """Format a BacktestReport as a human-readable summary string."""
    lines = [
        "=" * 60,
        "  LINEUP OPTIMIZER BACKTEST REPORT",
        "=" * 60,
        "",
        f"  Weeks tested:          {len(report.weeks)}",
        f"  Avg projection RMSE:   {report.avg_rmse:.3f}",
        f"  Avg rank correlation:  {report.avg_rank_correlation:+.3f}",
        f"  Avg bust rate:         {report.avg_bust_rate:.1%}",
        "",
        "  Grade distribution:",
    ]

    for grade in ("A", "B", "C"):
        count = report.grade_distribution.get(grade, 0)
        bar = "#" * count
        lines.append(f"    {grade}: {count:2d}  {bar}")

    lines.append("")
    lines.append("-" * 60)
    lines.append("  PER-WEEK DETAIL")
    lines.append("-" * 60)

    for wr in report.weeks:
        rmse_str = f"{wr.projection_rmse:.3f}" if np.isfinite(wr.projection_rmse) else "N/A"
        lines.append(
            f"  {wr.week_start} - {wr.week_end}  |  "
            f"RMSE={rmse_str}  rho={wr.rank_correlation:+.3f}  "
            f"bust={wr.bust_rate:.0%}  grade={wr.lineup_grade}  "
            f"n={wr.n_players}"
        )
        if wr.category_rmse:
            cat_strs = [f"{c}={v:.2f}" for c, v in sorted(wr.category_rmse.items())]
            lines.append(f"    Category RMSE: {', '.join(cat_strs)}")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)

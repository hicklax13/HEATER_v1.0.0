"""Injury risk modeling for fantasy baseball player valuation.

Provides health scores, age-based risk adjustments, workload flags,
and projection adjustments based on injury history. Integrates with
the MLB Stats API for historical games-played data.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

try:
    import statsapi
except ImportError:
    statsapi = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

LEAGUE_AVG_HEALTH: float = 0.85
"""Default health score when season data is missing."""

POSITION_PLAYER_AGE_THRESHOLD: int = 30
"""Age after which position players accrue additional injury risk."""

PITCHER_AGE_THRESHOLD: int = 28
"""Age after which pitchers accrue additional injury risk."""

POSITION_PLAYER_RISK_PER_YEAR: float = 0.02
"""Extra injury risk per year past threshold for position players."""

PITCHER_RISK_PER_YEAR: float = 0.03
"""Extra injury risk per year past threshold for pitchers."""

AGE_RISK_FLOOR: float = 0.5
"""Minimum age-risk multiplier (no one drops below 50%)."""

DEFAULT_WORKLOAD_THRESHOLD: float = 40.0
"""IP increase over previous season that triggers a workload flag."""

GAMES_AVAILABLE_HITTER: int = 162
"""Full-season games available for a position player."""

GAMES_AVAILABLE_PITCHER: int = 162
"""Full-season games available for a pitcher (used as default denominator)."""

HISTORY_SEASONS: int = 3
"""Number of past seasons used for health-score calculation."""

# Counting-stat columns that scale linearly with playing time.
COUNTING_STATS: list[str] = ["r", "hr", "rbi", "sb", "w", "sv", "k"]

# Playing-time columns that also scale down.
PLAYING_TIME_COLS: list[str] = ["pa", "ab", "h", "ip"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_health_score(
    games_played_3yr: list[int],
    games_available_3yr: list[int],
) -> float:
    """Return a 0.0-1.0 health score averaged over up to 3 seasons.

    Parameters
    ----------
    games_played_3yr:
        Games played in each of the last 1-3 seasons.
    games_available_3yr:
        Games available (e.g. 162) in each corresponding season.

    Missing seasons (fewer than 3 entries) are filled with the
    league-average health score.  Invalid entries (zero or negative
    ``games_available``, negative ``games_played``) are treated as
    missing.
    """
    if games_played_3yr is None:
        games_played_3yr = []
    if games_available_3yr is None:
        games_available_3yr = []

    ratios: list[float] = []

    for i in range(HISTORY_SEASONS):
        if i < len(games_played_3yr) and i < len(games_available_3yr):
            gp = games_played_3yr[i]
            ga = games_available_3yr[i]

            # Guard against nonsensical values.
            if ga is None or gp is None or ga <= 0 or gp < 0:
                ratios.append(LEAGUE_AVG_HEALTH)
                continue

            ratio = min(gp / ga, 1.0)  # cap at 1.0
            ratios.append(ratio)
        else:
            ratios.append(LEAGUE_AVG_HEALTH)

    return float(np.mean(ratios))


def age_risk_adjustment(age: int, is_pitcher: bool) -> float:
    """Return a multiplier (0.5-1.0) reflecting age-based injury risk.

    Position players lose 2% per year past 30; pitchers lose 3% per
    year past 28.  The multiplier never drops below ``AGE_RISK_FLOOR``.
    """
    if age is None:
        return 1.0

    if is_pitcher:
        threshold = PITCHER_AGE_THRESHOLD
        risk_per_year = PITCHER_RISK_PER_YEAR
    else:
        threshold = POSITION_PLAYER_AGE_THRESHOLD
        risk_per_year = POSITION_PLAYER_RISK_PER_YEAR

    if age <= threshold:
        return 1.0

    years_over = age - threshold
    multiplier = 1.0 - years_over * risk_per_year
    return max(multiplier, AGE_RISK_FLOOR)


def workload_flag(
    ip_current: float,
    ip_previous: float,
    threshold: float = DEFAULT_WORKLOAD_THRESHOLD,
) -> bool:
    """Return ``True`` if innings pitched jumped by more than *threshold*.

    Handles ``None`` and negative values gracefully — returns ``False``
    when either input is unusable.
    """
    if ip_current is None or ip_previous is None:
        return False
    if ip_current < 0 or ip_previous < 0:
        return False

    return (ip_current - ip_previous) > threshold


def apply_injury_adjustment(
    projections: pd.DataFrame,
    health_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Scale projections by each player's combined injury factor.

    Parameters
    ----------
    projections:
        Must contain ``player_id`` plus stat columns (r, hr, rbi, sb,
        w, sv, k, pa, ab, h, ip, avg, era, whip).  Columns ``er``,
        ``bb_allowed``, and ``h_allowed`` are used for rate-stat
        recalculation when present.
    health_scores:
        Must contain ``player_id`` and ``health_score``.  An optional
        ``age_risk_mult`` column provides the age multiplier; defaults
        to 1.0 when absent.

    Returns
    -------
    pd.DataFrame
        A *copy* of ``projections`` with adjusted values.
    """
    if projections is None or projections.empty:
        return projections.copy() if projections is not None else pd.DataFrame()
    if health_scores is None or health_scores.empty:
        return projections.copy()

    adj = projections.copy()

    # Ensure the age multiplier column exists.
    hs = health_scores.copy()
    if "age_risk_mult" not in hs.columns:
        hs["age_risk_mult"] = 1.0

    # Merge on player_id (left join keeps all projection rows).
    adj = adj.merge(
        hs[["player_id", "health_score", "age_risk_mult"]],
        on="player_id",
        how="left",
    )

    # Players with no health data keep their projections unchanged.
    adj["health_score"] = adj["health_score"].fillna(1.0)
    adj["age_risk_mult"] = adj["age_risk_mult"].fillna(1.0)
    adj["_combined_factor"] = adj["health_score"] * adj["age_risk_mult"]

    # Scale counting stats.
    for col in COUNTING_STATS:
        if col in adj.columns:
            adj[col] = adj[col] * adj["_combined_factor"]

    # Scale playing-time columns.
    for col in PLAYING_TIME_COLS:
        if col in adj.columns:
            adj[col] = adj[col] * adj["_combined_factor"]

    # Scale rate-stat component columns so ERA/WHIP recalculation is correct.
    for col in ["er", "bb_allowed", "h_allowed"]:
        if col in adj.columns:
            adj[col] = adj[col] * adj["_combined_factor"]

    # Recalculate rate stats from adjusted components.
    if "h" in adj.columns and "ab" in adj.columns:
        adj["avg"] = np.where(adj["ab"] > 0, adj["h"] / adj["ab"], 0.0)

    if "er" in adj.columns and "ip" in adj.columns:
        adj["era"] = np.where(adj["ip"] > 0, adj["er"] * 9 / adj["ip"], 0.0)

    if "ip" in adj.columns:
        bb_col = "bb_allowed" if "bb_allowed" in adj.columns else None
        h_col = "h_allowed" if "h_allowed" in adj.columns else None
        if bb_col and h_col:
            adj["whip"] = np.where(
                adj["ip"] > 0,
                (adj[bb_col] + adj[h_col]) / adj["ip"],
                0.0,
            )

    # Drop helper columns.
    adj.drop(
        columns=["health_score", "age_risk_mult", "_combined_factor"],
        inplace=True,
        errors="ignore",
    )

    return adj


def load_injury_history_from_api(
    player_ids: list[int],
) -> pd.DataFrame:
    """Fetch games-played history from the MLB Stats API.

    Returns a DataFrame with columns ``player_id``, ``season``,
    ``games_played``, and ``games_available`` for the last
    ``HISTORY_SEASONS`` seasons.

    Falls back to an empty DataFrame (with correct columns) on any
    network or API error.
    """
    columns = ["player_id", "season", "games_played", "games_available"]

    if statsapi is None:
        logger.warning("MLB-StatsAPI not installed, cannot fetch injury history")
        return pd.DataFrame(columns=columns)

    if not player_ids:
        return pd.DataFrame(columns=columns)

    import datetime

    current_year = datetime.datetime.now(datetime.UTC).year
    seasons = list(range(current_year - HISTORY_SEASONS, current_year))

    rows: list[dict] = []

    for pid in player_ids:
        try:
            stats = statsapi.player_stat_data(
                pid,
                group="hitting,pitching",
                type="season",
                sportId=1,
            )
        except Exception:  # noqa: BLE001
            # Network failure, missing player, etc. — skip this player.
            continue

        for season in seasons:
            games_played = 0
            for stat_group in stats.get("stats", []):
                for split in stat_group.get("splits", []):
                    if split.get("season") == str(season):
                        games_played = int(split.get("stat", {}).get("gamesPlayed", 0))
                        break

            rows.append(
                {
                    "player_id": pid,
                    "season": season,
                    "games_played": games_played,
                    "games_available": GAMES_AVAILABLE_HITTER,
                }
            )

    if not rows:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rows, columns=columns)


def get_injury_badge(health_score: float) -> tuple[str, str]:
    """Return an ``(icon_html, label)`` tuple for UI display.

    Returns a small inline CSS dot (colored circle) instead of emoji
    for consistent cross-platform rendering.

    Thresholds:
    * >= 0.90 : green / Low Risk
    * >= 0.75 : amber / Moderate Risk
    * <  0.75 : red / High Risk
    """
    _dot = (
        '<span style="display:inline-block;width:10px;height:10px;'
        "border-radius:50%;vertical-align:middle;margin-right:4px;"
        'background:{color};"></span>'
    )
    if health_score is None:
        return (_dot.format(color="#fb923c"), "Unknown")
    if health_score >= 0.9:
        return (_dot.format(color="#84cc16"), "Low Risk")
    if health_score >= 0.75:
        return (_dot.format(color="#fb923c"), "Moderate Risk")
    return (_dot.format(color="#f43f5e"), "High Risk")

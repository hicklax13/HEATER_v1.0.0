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

# B2: Position-specific age thresholds and risk rates.
# Catchers age faster (squatting wears knees/back), DHs age slowest.
POSITION_AGE_PROFILES: dict[str, dict] = {
    "C": {"threshold": 28, "risk_per_year": 0.03},
    "SP": {"threshold": 28, "risk_per_year": 0.03},
    "RP": {"threshold": 29, "risk_per_year": 0.025},
    "SS": {"threshold": 30, "risk_per_year": 0.02},
    "2B": {"threshold": 30, "risk_per_year": 0.02},
    "CF": {"threshold": 30, "risk_per_year": 0.02},
    "3B": {"threshold": 31, "risk_per_year": 0.02},
    "OF": {"threshold": 31, "risk_per_year": 0.02},
    "1B": {"threshold": 32, "risk_per_year": 0.015},
    "DH": {"threshold": 34, "risk_per_year": 0.01},
}
DEFAULT_AGE_PROFILE: dict = {"threshold": 30, "risk_per_year": 0.02}

AGE_RISK_FLOOR: float = 0.5
"""Minimum age-risk multiplier (no one drops below 50%)."""

PITCHER_AGE_THRESHOLD: int = 28
"""Age at which pitchers begin showing elevated injury risk."""

POSITION_PLAYER_AGE_THRESHOLD: int = 30
"""Age at which position players begin showing elevated injury risk."""

# B3: Injury-type severity floors (minimum health score while recovering).
# TJ surgery is career-altering; hamstrings recur; concussions are unpredictable.
INJURY_TYPE_FLOORS: dict[str, dict] = {
    "tommy john": {"floor": 0.40, "duration_years": 2},
    "tj": {"floor": 0.40, "duration_years": 2},
    "ucl": {"floor": 0.40, "duration_years": 2},
    "hamstring": {"floor": 0.70, "duration_years": 1},
    "oblique": {"floor": 0.65, "duration_years": 1},
    "concussion": {"floor": 0.60, "duration_years": 1},
    "shoulder": {"floor": 0.55, "duration_years": 2},
    "elbow": {"floor": 0.55, "duration_years": 1},
    "back": {"floor": 0.60, "duration_years": 1},
    "knee": {"floor": 0.60, "duration_years": 1},
    "ankle": {"floor": 0.70, "duration_years": 1},
    "wrist": {"floor": 0.70, "duration_years": 1},
}

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
    il_stints_3yr: list[int] | None = None,
    il_days_3yr: list[int] | None = None,
) -> float:
    """Return a 0.0-1.0 health score averaged over up to 3 seasons.

    Parameters
    ----------
    games_played_3yr:
        Games played in each of the last 1-3 seasons.
    games_available_3yr:
        Games available (e.g. 162) in each corresponding season.
    il_stints_3yr:
        Number of IL stints per season (optional). When provided, the
        score is based on IL time rather than raw GP/GA ratio — this
        correctly handles pitchers (who appear in ~30-70 games normally)
        and young players (who may have been in the minors).
    il_days_3yr:
        Total IL days per season (optional). Used with il_stints for a
        more accurate health assessment.

    Missing seasons (fewer than 3 entries) are filled with the
    league-average health score.  Invalid entries (zero or negative
    ``games_available``, negative ``games_played``) are treated as
    missing.
    """
    if games_played_3yr is None:
        games_played_3yr = []
    if games_available_3yr is None:
        games_available_3yr = []

    # If IL data is available, use it for a more accurate score.
    # IL-based scoring: each season scores 1.0 (healthy) minus a penalty
    # based on IL days. 60+ IL days in a season = 0.0 for that season.
    if il_stints_3yr is not None and il_days_3yr is not None:
        ratios: list[float] = []
        for i in range(HISTORY_SEASONS):
            if i < len(il_days_3yr) and i < len(il_stints_3yr):
                days = il_days_3yr[i] if il_days_3yr[i] is not None else 0
                stints = il_stints_3yr[i] if il_stints_3yr[i] is not None else 0
                if stints == 0 and days == 0:
                    ratios.append(1.0)  # Fully healthy season
                else:
                    # Penalty: IL days / 162 (full season), capped at 1.0
                    penalty = min(days / 162.0, 1.0)
                    ratios.append(max(1.0 - penalty, 0.0))
            else:
                ratios.append(LEAGUE_AVG_HEALTH)
        return float(np.mean(ratios))

    # Fallback: GP/GA ratio (less accurate for pitchers)
    ratios = []
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


def age_risk_adjustment(age: int, is_pitcher: bool, position: str | None = None) -> float:
    """B2: Position-specific age-risk multiplier (0.5-1.0).

    Uses graduated thresholds: catchers age at 28 (0.03/yr), DHs at 34
    (0.01/yr). Falls back to pitcher/hitter defaults when position unknown.
    """
    if age is None:
        return 1.0

    # B2: Use position-specific profile when available
    if is_pitcher:
        profile = POSITION_AGE_PROFILES.get("SP", DEFAULT_AGE_PROFILE)
    else:
        profile = DEFAULT_AGE_PROFILE
    if position:
        primary = position.split(",")[0].strip().upper()
        if primary in POSITION_AGE_PROFILES:
            profile = POSITION_AGE_PROFILES[primary]

    threshold = profile["threshold"]
    risk_per_year = profile["risk_per_year"]

    if age <= threshold:
        return 1.0

    years_over = age - threshold
    multiplier = 1.0 - years_over * risk_per_year
    return max(multiplier, AGE_RISK_FLOOR)


def injury_type_adjustment(injury_note: str | None) -> float:
    """B3: Return a health-score floor based on injury type.

    TJ surgery = 0.40 floor (career-altering). Hamstring = 0.70 (recurrent).
    Returns 1.0 when no injury type matched (no adjustment).
    """
    if not injury_note:
        return 1.0

    note_lower = injury_note.lower()
    best_floor = 1.0
    for keyword, profile in INJURY_TYPE_FLOORS.items():
        if keyword in note_lower:
            if profile["floor"] < best_floor:
                best_floor = profile["floor"]
    return best_floor


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
        columns=["age_risk_mult", "_combined_factor"],
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

    Thresholds (calibrated for fantasy relevance):
    * >= 0.85 : green / Low Risk (played 85%+ of games over 3 years)
    * >= 0.65 : amber / Moderate Risk (missed some time but mostly available)
    * >= 0.40 : orange / Elevated Risk (significant missed time)
    * <  0.40 : red / High Risk (severe injury history or very limited track record)
    """
    _dot = (
        '<span style="display:inline-block;width:10px;height:10px;'
        "border-radius:50%;vertical-align:middle;margin-right:4px;"
        'background:{color};"></span>'
    )
    if health_score is None:
        return (_dot.format(color="#fb923c"), "Unknown")
    if health_score >= 0.85:
        return (_dot.format(color="#84cc16"), "Low Risk")
    if health_score >= 0.65:
        return (_dot.format(color="#fb923c"), "Moderate Risk")
    if health_score >= 0.40:
        return (_dot.format(color="#ff9f1c"), "Elevated Risk")
    return (_dot.format(color="#f43f5e"), "High Risk")

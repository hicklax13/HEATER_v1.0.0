"""Weekly H2H matchup strategy engine.

The missing brain: every recommendation flows through this module's
understanding of "what categories can I win THIS WEEK?"
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone

from src.valuation import LeagueConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ET = timezone(timedelta(hours=-4))

# MLB 2026 season start date (used for week calculation fallback)
_SEASON_START = datetime(2026, 3, 25, tzinfo=_ET)

# Typical weekly production for a competitive 12-team roster.
# Used to assess whether a counting-stat gap is closeable within a week.
WEEKLY_COUNTING_RATES: dict[str, float] = {
    "R": 35.0,
    "HR": 9.0,
    "RBI": 34.0,
    "SB": 5.0,
    "W": 3.0,
    "L": 3.0,
    "SV": 2.7,
    "K": 50.0,
}

# Rate stat standard deviations (typical weekly spread across 12 teams).
# If gap is less than this, the category is still in play.
RATE_STAT_STD: dict[str, float] = {
    "AVG": 0.025,
    "OBP": 0.025,
    "ERA": 0.80,
    "WHIP": 0.12,
}

# Thresholds for closeability assessment.
# Counting: gap < CLOSEABLE_FRACTION * weekly rate -> closeable.
CLOSEABLE_FRACTION = 0.30
# Rate: gap < RATE_CLOSEABLE_STDS standard deviations -> closeable.
RATE_CLOSEABLE_STDS = 0.5

# Thresholds for "punt" assessment (gap too large).
PUNT_COUNTING_FRACTION = 0.70
PUNT_RATE_STDS = 1.5

# Per-game production estimates for FA search criteria.
PER_GAME_RATES: dict[str, float] = {
    "HR": 0.15,
    "R": 0.55,
    "RBI": 0.55,
    "SB": 0.10,
    "W": 0.20,
    "SV": 0.35,
    "K": 6.5,
}


# ---------------------------------------------------------------------------
# 1. compute_weekly_matchup_state
# ---------------------------------------------------------------------------


def compute_weekly_matchup_state(yds=None) -> dict:
    """Build a full picture of the current week's H2H matchup.

    Args:
        yds: YahooDataService instance. If None or not connected, returns
            sensible defaults so callers never crash.

    Returns:
        Dict with keys: week, score, record, standings_rank, categories,
        winnable_cats, protect_cats, punt_cats, desperation_level.
    """
    config = LeagueConfig()
    matchup = _safe_get_matchup(yds)
    standings = _safe_get_standings(yds)

    # -- Parse matchup data -----------------------------------------------
    week = matchup.get("week", 0) if matchup else 0
    wins = matchup.get("wins", 0) if matchup else 0
    losses = matchup.get("losses", 0) if matchup else 0
    ties = matchup.get("ties", 0) if matchup else 0
    score = (wins, losses, ties)

    # -- Parse category-level data ----------------------------------------
    categories = _parse_categories(matchup, config)

    # -- Classify categories into winnable / protect / punt ---------------
    winnable_cats = []
    protect_cats = []
    punt_cats = []
    for cat_info in categories:
        name = cat_info["name"]
        status = cat_info["status"]
        gap = cat_info["gap"]
        is_rate = cat_info["is_rate"]

        closeable = _is_closeable(name, gap, is_rate)
        puntable = _is_puntable(name, gap, is_rate)

        if status == "winning":
            protect_cats.append(name)
        elif status == "losing" and closeable:
            winnable_cats.append(name)
        elif status == "losing" and puntable:
            punt_cats.append(name)
        elif status == "losing":
            # Not clearly closeable or puntable -- still winnable
            winnable_cats.append(name)
        elif status == "tied":
            winnable_cats.append(name)

    # -- Season record and standings rank ---------------------------------
    record, standings_rank = _extract_record_and_rank(standings, matchup)

    # -- Desperation level ------------------------------------------------
    desperation = compute_desperation_level(record, score, standings_rank)

    return {
        "week": week,
        "score": score,
        "record": record,
        "standings_rank": standings_rank,
        "categories": categories,
        "winnable_cats": winnable_cats,
        "protect_cats": protect_cats,
        "punt_cats": punt_cats,
        "desperation_level": desperation,
    }


# ---------------------------------------------------------------------------
# 2. compute_desperation_level
# ---------------------------------------------------------------------------


def compute_desperation_level(
    record: tuple[int, int, int],
    current_score: tuple[int, int, int],
    standings_rank: int,
    total_teams: int = 12,
) -> float:
    """Compute how desperate the team situation is (0.0 = comfortable, 1.0 = dire).

    Factors:
        40% -- season record (more losses = more desperate)
        30% -- current weekly score (losing more categories = more desperate)
        30% -- standings position (lower in standings = more desperate)

    Args:
        record: (season_W, season_L, season_T).
        current_score: (week_wins, week_losses, week_ties).
        standings_rank: 1-based rank in league standings.
        total_teams: Number of teams in the league.

    Returns:
        Float clamped to [0.0, 1.0].
    """
    season_w, season_l, _season_t = record
    total_decisions = season_w + season_l
    if total_decisions > 0:
        record_factor = season_l / total_decisions
    else:
        record_factor = 0.5  # no games played yet, neutral

    week_w, week_l, _week_t = current_score
    total_cats = week_w + week_l  # ties are neutral
    if total_cats > 0:
        score_factor = 1.0 - (week_w / max(1, week_w + week_l))
    else:
        score_factor = 0.5  # no data yet, neutral

    rank_factor = standings_rank / max(1, total_teams)

    desperation = record_factor * 0.4 + score_factor * 0.3 + rank_factor * 0.3
    return max(0.0, min(1.0, desperation))


# ---------------------------------------------------------------------------
# 3. get_dynamic_weeks_remaining
# ---------------------------------------------------------------------------


def get_dynamic_weeks_remaining(
    current_week: int | None = None,
    total_weeks: int = 24,
) -> int:
    """Return the number of weeks remaining in the fantasy season.

    Args:
        current_week: If provided, used directly. Otherwise detected from
            date math (MLB 2026 season start = March 25).
        total_weeks: Total weeks in the fantasy season.

    Returns:
        Integer weeks remaining (minimum 0).
    """
    if current_week is not None:
        return max(0, total_weeks - current_week)

    now = datetime.now(_ET)
    elapsed_days = (now - _SEASON_START).days
    if elapsed_days < 0:
        return total_weeks  # season hasn't started

    estimated_week = max(1, (elapsed_days // 7) + 1)
    return max(0, total_weeks - estimated_week)


# ---------------------------------------------------------------------------
# 4. get_optimal_alpha
# ---------------------------------------------------------------------------


def get_optimal_alpha(
    desperation_level: float,
    weeks_remaining: int,
    total_weeks: int = 24,
) -> float:
    """Compute the H2H-vs-season-long alpha blend for the optimizer.

    In H2H leagues every week is a separate contest. The alpha returned is
    the weight toward weekly (H2H) optimization:
        1.0 = pure weekly focus
        0.0 = pure season-long focus (wrong for H2H)

    Rules:
        - alpha never drops below 0.4 in an H2H league
        - late season and high desperation push toward 0.95
        - early season and low desperation settle around 0.5

    Args:
        desperation_level: 0.0 (comfortable) to 1.0 (dire).
        weeks_remaining: Weeks left in the season.
        total_weeks: Total weeks in the season.

    Returns:
        Float in [0.4, 0.95].
    """
    # Season progress: 0.0 = start, 1.0 = end
    progress = 1.0 - (weeks_remaining / max(1, total_weeks))

    # Base alpha ramps from 0.5 early to 0.75 late season
    base = 0.5 + 0.25 * progress

    # Desperation adds up to 0.20 additional alpha
    desp_boost = desperation_level * 0.20

    alpha = base + desp_boost
    return max(0.4, min(0.95, alpha))


# ---------------------------------------------------------------------------
# 5. get_category_targets
# ---------------------------------------------------------------------------


def get_category_targets(matchup_state: dict) -> dict:
    """Translate the matchup state into strategic category priorities.

    Args:
        matchup_state: Output of ``compute_weekly_matchup_state()``.

    Returns:
        Dict with keys ``must_win``, ``protect``, ``concede``, ``swing``.
        Each value is a list of dicts with: name, gap, priority_score,
        suggested_action.
    """
    categories = matchup_state.get("categories", [])
    winnable = set(matchup_state.get("winnable_cats", []))
    protect = set(matchup_state.get("protect_cats", []))
    punt = set(matchup_state.get("punt_cats", []))

    must_win: list[dict] = []
    protect_list: list[dict] = []
    concede_list: list[dict] = []
    swing_list: list[dict] = []

    for cat_info in categories:
        name = cat_info["name"]
        gap = cat_info["gap"]
        status = cat_info["status"]
        is_rate = cat_info["is_rate"]
        is_inverse = cat_info["is_inverse"]

        entry = {
            "name": name,
            "gap": gap,
            "priority_score": 0.0,
            "suggested_action": "",
        }

        if name in punt:
            entry["priority_score"] = 0.0
            entry["suggested_action"] = f"Concede {name} -- gap too large to close"
            concede_list.append(entry)

        elif name in protect:
            # Priority to protect: higher when lead is slim
            if is_rate:
                std = RATE_STAT_STD.get(name, 0.5)
                slim = abs(gap) < std * 0.5
            else:
                weekly = WEEKLY_COUNTING_RATES.get(name, 10.0)
                slim = abs(gap) < weekly * 0.10
            entry["priority_score"] = 0.8 if slim else 0.5
            if is_rate and not is_inverse:
                entry["suggested_action"] = f"Protect {name} lead -- avoid low-{name} streamers"
            elif is_rate and is_inverse:
                entry["suggested_action"] = f"Protect {name} lead -- limit risky pitcher starts"
            else:
                entry["suggested_action"] = f"Protect {name} lead ({gap:+.1f} gap)"
            protect_list.append(entry)

        elif status == "tied":
            entry["priority_score"] = 0.7
            entry["suggested_action"] = f"{name} is tied -- target with lineup/adds"
            swing_list.append(entry)

        elif name in winnable:
            # Losing but closeable -- high priority
            closeable_score = _closeable_priority(name, gap, is_rate)
            entry["priority_score"] = closeable_score
            if is_rate:
                entry["suggested_action"] = f"Close {name} gap ({gap:+.3f}) -- adjust lineup composition"
            else:
                entry["suggested_action"] = f"Close {name} gap ({gap:+.0f}) -- stream/add for {name}"
            must_win.append(entry)

        else:
            # Fallback: losing, not classified
            entry["priority_score"] = 0.3
            entry["suggested_action"] = f"{name} is tough but not impossible"
            must_win.append(entry)

    # Sort each bucket by priority descending
    must_win.sort(key=lambda x: x["priority_score"], reverse=True)
    swing_list.sort(key=lambda x: x["priority_score"], reverse=True)
    protect_list.sort(key=lambda x: x["priority_score"], reverse=True)

    return {
        "must_win": must_win,
        "protect": protect_list,
        "concede": concede_list,
        "swing": swing_list,
    }


# ---------------------------------------------------------------------------
# 6. get_adjusted_trade_params
# ---------------------------------------------------------------------------


def get_adjusted_trade_params(desperation_level: float) -> dict:
    """Return trade finder parameters adjusted by desperation.

    As desperation increases, the team becomes more willing to sell high on
    elite players, target weak categories harder, and accept more lopsided
    proposals.

    Args:
        desperation_level: 0.0 (comfortable) to 1.0 (dire).

    Returns:
        Dict with keys: elite_return_floor, max_weight_ratio,
        loss_aversion, max_opp_loss.
    """
    d = max(0.0, min(1.0, desperation_level))
    return {
        "elite_return_floor": _lerp(0.75, 0.50, d),
        "max_weight_ratio": _lerp(1.5, 2.5, d),
        "loss_aversion": _lerp(1.8, 1.3, d),
        "max_opp_loss": _lerp(-0.5, -1.5, d),
    }


# ---------------------------------------------------------------------------
# 7. get_matchup_aware_fa_criteria
# ---------------------------------------------------------------------------


def get_matchup_aware_fa_criteria(
    matchup_state: dict,
    days_remaining: int = 5,
) -> list[dict]:
    """Generate FA search criteria based on current matchup gaps.

    Returns a prioritized list of what to look for on the waiver wire.

    Args:
        matchup_state: Output of ``compute_weekly_matchup_state()``.
        days_remaining: Days left in the matchup week.

    Returns:
        List of dicts with: category, stat_threshold, priority, reason.
    """
    config = LeagueConfig()
    criteria: list[dict] = []
    categories = {c["name"]: c for c in matchup_state.get("categories", [])}
    winnable = set(matchup_state.get("winnable_cats", []))

    for cat_name in winnable:
        cat_info = categories.get(cat_name)
        if cat_info is None:
            continue

        gap = cat_info["gap"]
        is_rate = cat_info["is_rate"]
        is_inverse = cat_info["is_inverse"]

        if is_rate:
            # Rate stats: suggest composition changes, not raw thresholds
            if is_inverse:
                reason = (
                    f"Losing {cat_name} ({cat_info['user_val']:.3f} vs "
                    f"{cat_info['opp_val']:.3f}) -- pick up low-{cat_name} "
                    f"pitcher to bring down team {cat_name}"
                )
                criteria.append(
                    {
                        "category": cat_name,
                        "stat_threshold": f"projected_{cat_name.lower()} <= league_avg",
                        "priority": 0.7,
                        "reason": reason,
                    }
                )
            else:
                reason = (
                    f"Losing {cat_name} ({cat_info['user_val']:.3f} vs "
                    f"{cat_info['opp_val']:.3f}) -- pick up high-{cat_name} "
                    f"hitter for remaining {days_remaining} days"
                )
                criteria.append(
                    {
                        "category": cat_name,
                        "stat_threshold": f"projected_{cat_name.lower()} >= league_avg",
                        "priority": 0.7,
                        "reason": reason,
                    }
                )
        else:
            # Counting stats: compute how much production is needed
            abs_gap = abs(gap)
            per_game = PER_GAME_RATES.get(cat_name, 0.1)
            needed_per_day = abs_gap / max(1, days_remaining) if abs_gap > 0 else 0
            threshold = needed_per_day / max(0.01, per_game)

            priority = _closeable_priority(cat_name, gap, is_rate=False)
            user_val = cat_info.get("user_val", 0)
            opp_val = cat_info.get("opp_val", 0)

            reason = (
                f"Losing {cat_name} {int(user_val)}-{int(opp_val)}, need {int(abs_gap)} more over {days_remaining} days"
            )
            stat_col = cat_name.lower()
            criteria.append(
                {
                    "category": cat_name,
                    "stat_threshold": (f"projected_{stat_col}_per_game >= {per_game:.2f}"),
                    "priority": priority,
                    "reason": reason,
                }
            )

    # Sort by priority descending
    criteria.sort(key=lambda x: x["priority"], reverse=True)
    return criteria


# ---------------------------------------------------------------------------
# 8. generate_weekly_action_plan
# ---------------------------------------------------------------------------


def generate_weekly_action_plan(
    yds=None,
    player_pool=None,
) -> dict:
    """Master function: full weekly action plan for H2H matchup optimization.

    Calls every other function in this module and assembles a coherent
    strategy for the week.

    Args:
        yds: YahooDataService instance (optional).
        player_pool: DataFrame from ``load_player_pool()`` (optional,
            loaded from DB if not provided).

    Returns:
        Dict with: matchup_state, desperation_level, alpha,
        category_targets, trade_params, fa_criteria, lineup_priorities,
        summary.
    """
    matchup_state = compute_weekly_matchup_state(yds)
    desperation = matchup_state["desperation_level"]

    week = matchup_state.get("week", 0)
    weeks_remaining = get_dynamic_weeks_remaining(current_week=week if week > 0 else None)

    alpha = get_optimal_alpha(desperation, weeks_remaining)
    category_targets = get_category_targets(matchup_state)
    trade_params = get_adjusted_trade_params(desperation)
    fa_criteria = get_matchup_aware_fa_criteria(matchup_state)

    # -- Lineup priorities ------------------------------------------------
    lineup_priorities = _build_lineup_priorities(matchup_state, category_targets)

    # -- Human-readable summary -------------------------------------------
    summary = _build_summary(matchup_state, alpha, category_targets, desperation)

    return {
        "matchup_state": matchup_state,
        "desperation_level": desperation,
        "alpha": alpha,
        "weeks_remaining": weeks_remaining,
        "category_targets": category_targets,
        "trade_params": trade_params,
        "fa_criteria": fa_criteria,
        "lineup_priorities": lineup_priorities,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_get_matchup(yds) -> dict | None:
    """Get matchup data, returning None on any error."""
    if yds is None:
        return None
    try:
        return yds.get_matchup()
    except Exception:
        logger.debug("Failed to get matchup from YDS", exc_info=True)
        return None


def _safe_get_standings(yds) -> dict | None:
    """Get standings DataFrame, returning empty on any error."""
    if yds is None:
        return None
    try:
        return yds.get_standings()
    except Exception:
        logger.debug("Failed to get standings from YDS", exc_info=True)
        return None


def _parse_categories(matchup: dict | None, config: LeagueConfig) -> list[dict]:
    """Parse Yahoo matchup categories into a uniform list of dicts.

    Each dict: name, user_val, opp_val, gap, status, is_inverse, is_rate.
    """
    if not matchup:
        # Return defaults -- all neutral
        result = []
        for cat in config.all_categories:
            result.append(
                {
                    "name": cat,
                    "user_val": 0.0,
                    "opp_val": 0.0,
                    "gap": 0.0,
                    "status": "tied",
                    "is_inverse": cat in config.inverse_stats,
                    "is_rate": cat in config.rate_stats,
                }
            )
        return result

    raw_cats = matchup.get("categories", [])
    inverse = config.inverse_stats
    rate = config.rate_stats
    parsed: list[dict] = []

    for cat_dict in raw_cats:
        name = str(cat_dict.get("cat", "")).upper()
        if not name:
            continue

        try:
            user_val = float(cat_dict.get("you", 0)) if cat_dict.get("you") not in ("-", "", None) else 0.0
        except (ValueError, TypeError):
            user_val = 0.0
        try:
            opp_val = float(cat_dict.get("opp", 0)) if cat_dict.get("opp") not in ("-", "", None) else 0.0
        except (ValueError, TypeError):
            opp_val = 0.0

        is_inverse = name in inverse
        is_rate = name in rate

        # Gap: positive = user is losing (needs improvement)
        if is_inverse:
            # Lower is better: gap = user - opp (positive means user is worse)
            gap = user_val - opp_val
        else:
            # Higher is better: gap = opp - user (positive means user is behind)
            gap = opp_val - user_val

        result_str = str(cat_dict.get("result", "")).upper()
        if result_str == "WIN":
            status = "winning"
        elif result_str == "LOSS":
            status = "losing"
        else:
            status = "tied"

        parsed.append(
            {
                "name": name,
                "user_val": user_val,
                "opp_val": opp_val,
                "gap": gap,
                "status": status,
                "is_inverse": is_inverse,
                "is_rate": is_rate,
            }
        )

    return parsed


def _extract_record_and_rank(standings, matchup) -> tuple[tuple[int, int, int], int]:
    """Extract season W-L-T record and standings rank.

    Tries standings DataFrame first, then matchup metadata. Falls back
    to neutral defaults.
    """
    import pandas as pd

    record = (0, 0, 0)
    rank = 6  # middle of 12

    # Try matchup metadata (user_name lets us find in standings)
    user_name = matchup.get("user_name", "") if matchup else ""

    if standings is not None and isinstance(standings, pd.DataFrame) and not standings.empty:
        # Standings may have wins/losses/ties columns or W-L-T meta
        # The standings DataFrame is long-format: team_name, category, total, rank.
        # But the raw Yahoo standings_df may also include record columns.
        # Try to get the rank from the first matching row.
        if user_name:
            team_rows = standings[standings["team_name"].astype(str).str.contains(user_name[:10], case=False, na=False)]
            if not team_rows.empty:
                first = team_rows.iloc[0]
                rank = int(first.get("rank", 6))

        # W-L-T are typically in league_records table or standings meta.
        # Try to load from DB as fallback.
        try:
            from src.database import get_connection

            conn = get_connection()
            try:
                cursor = conn.execute(
                    "SELECT wins, losses, ties, rank FROM league_records WHERE team_name LIKE ?",
                    (f"%{user_name[:10]}%",),
                )
                row = cursor.fetchone()
                if row:
                    record = (int(row[0] or 0), int(row[1] or 0), int(row[2] or 0))
                    rank = int(row[3] or rank)
            finally:
                conn.close()
        except Exception:
            logger.debug("Could not load league_records", exc_info=True)

    return record, rank


def _is_closeable(cat_name: str, gap: float, is_rate: bool) -> bool:
    """Determine if a category gap is closeable within a week."""
    if gap <= 0:
        return True  # already winning or tied
    if is_rate:
        std = RATE_STAT_STD.get(cat_name, 0.5)
        return gap < std * RATE_CLOSEABLE_STDS
    else:
        weekly = WEEKLY_COUNTING_RATES.get(cat_name, 10.0)
        return gap < weekly * CLOSEABLE_FRACTION


def _is_puntable(cat_name: str, gap: float, is_rate: bool) -> bool:
    """Determine if a category gap is too large to close."""
    if gap <= 0:
        return False  # not losing
    if is_rate:
        std = RATE_STAT_STD.get(cat_name, 0.5)
        return gap > std * PUNT_RATE_STDS
    else:
        weekly = WEEKLY_COUNTING_RATES.get(cat_name, 10.0)
        return gap > weekly * PUNT_COUNTING_FRACTION


def _closeable_priority(cat_name: str, gap: float, is_rate: bool) -> float:
    """Score 0-1 for how urgently a closeable gap should be targeted.

    Closer gaps get higher priority (easier to close = better ROI).
    """
    if gap <= 0:
        return 0.5  # tied, moderate priority
    if is_rate:
        std = RATE_STAT_STD.get(cat_name, 0.5)
        if std < 1e-9:
            return 0.5
        # Sigmoid: small gap -> high priority, large gap -> low priority
        ratio = gap / std
        return max(0.1, min(0.95, 1.0 / (1.0 + math.exp(2.0 * (ratio - 0.5)))))
    else:
        weekly = WEEKLY_COUNTING_RATES.get(cat_name, 10.0)
        if weekly < 1e-9:
            return 0.5
        ratio = gap / weekly
        return max(0.1, min(0.95, 1.0 / (1.0 + math.exp(4.0 * (ratio - 0.15)))))


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation from a to b by factor t in [0, 1]."""
    return a + (b - a) * t


def _build_lineup_priorities(
    matchup_state: dict,
    category_targets: dict,
) -> list[str]:
    """Build human-readable lineup priority strings."""
    priorities: list[str] = []
    config = LeagueConfig()

    # Must-win categories -> positive actions
    for entry in category_targets.get("must_win", []):
        name = entry["name"]
        if name in ("HR", "R", "RBI"):
            priorities.append(f"Start power hitters -- need {name}")
        elif name == "SB":
            priorities.append("Start speed guys -- need SB")
        elif name in ("W", "K"):
            priorities.append(f"Start high-volume pitchers -- need {name}")
        elif name == "SV":
            priorities.append("Start all closers -- need SV")
        elif name == "AVG":
            priorities.append("Start high-AVG hitters -- bench low-AVG sluggers")
        elif name == "OBP":
            priorities.append("Start patient hitters -- need OBP")

    # Protect categories -> defensive actions
    for entry in category_targets.get("protect", []):
        name = entry["name"]
        if name == "ERA":
            priorities.append("Consider benching risky SP starts to protect ERA lead")
        elif name == "WHIP":
            priorities.append("Limit pitcher volume to protect WHIP lead")
        elif name == "AVG":
            priorities.append("Avoid low-AVG streamers to protect AVG lead")
        elif name == "L":
            priorities.append("Winning L (fewer losses) -- bench shaky starters")

    # Concede -> resource reallocation
    concede_names = [e["name"] for e in category_targets.get("concede", [])]
    if concede_names:
        cats_str = ", ".join(concede_names)
        priorities.append(f"Conceding {cats_str} -- redirect resources elsewhere")

    # Cap at 6 priorities to keep it actionable
    return priorities[:6]


def _build_summary(
    matchup_state: dict,
    alpha: float,
    category_targets: dict,
    desperation: float,
) -> list[str]:
    """Build 3-5 bullet summary of the weekly strategy."""
    summary: list[str] = []
    score = matchup_state.get("score", (0, 0, 0))
    week = matchup_state.get("week", 0)

    # Headline
    w, l, t = score
    if w > l:
        summary.append(f"Week {week}: Leading {w}-{l}-{t} -- protect and extend")
    elif l > w:
        summary.append(f"Week {week}: Trailing {w}-{l}-{t} -- aggressive mode")
    elif w == 0 and l == 0:
        summary.append(f"Week {week}: No scores yet -- set optimal lineup early")
    else:
        summary.append(f"Week {week}: Tied {w}-{l}-{t} -- every category matters")

    # Desperation context
    if desperation >= 0.7:
        summary.append(f"Desperation level {desperation:.0%} -- consider sell-high trades and aggressive streaming")
    elif desperation >= 0.4:
        summary.append(f"Moderate pressure (desperation {desperation:.0%}) -- targeted moves recommended")
    else:
        summary.append(f"Comfortable position (desperation {desperation:.0%}) -- play the long game")

    # Must-win categories
    must_win = category_targets.get("must_win", [])
    if must_win:
        names = [e["name"] for e in must_win[:3]]
        summary.append(f"Priority targets: {', '.join(names)}")

    # Protect categories
    protect = category_targets.get("protect", [])
    if protect:
        names = [e["name"] for e in protect[:3]]
        summary.append(f"Protect leads in: {', '.join(names)}")

    # Alpha recommendation
    summary.append(f"Recommended optimizer alpha: {alpha:.2f} (H2H weekly focus)")

    return summary[:5]

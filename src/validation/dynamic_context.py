"""
Dynamic Context — computes real-world values that are currently hardcoded.

Replaces:
    weeks_remaining=16       → compute_weeks_remaining()
    schedule_strength=0.5    → compute_schedule_strength()
    injury_exposure=0.1      → compute_injury_exposure()
    momentum=1.0             → compute_momentum()

These are the "Root Cause 4" fixes: real-world context that should be
computed from actual league state, not hardcoded defaults.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime

logger = logging.getLogger(__name__)

# MLB season boundaries (approximate, adjust per season)
MLB_SEASON_START: dict[int, date] = {
    2025: date(2025, 3, 27),
    2026: date(2026, 3, 26),
}
MLB_SEASON_END: dict[int, date] = {
    2025: date(2025, 9, 28),
    2026: date(2026, 9, 27),
}

# Fantasy playoff weeks (typically last 3-4 weeks of regular season)
FANTASY_REGULAR_SEASON_WEEKS = 22  # Typical Yahoo H2H


def compute_weeks_remaining(
    as_of: date | None = None,
    season: int = 2026,
    total_weeks: int = FANTASY_REGULAR_SEASON_WEEKS,
) -> int:
    """
    Compute weeks remaining in the fantasy season.

    Replaces every hardcoded `weeks_remaining=16` in the codebase.

    Returns:
        Integer weeks remaining (capped at 1 minimum, total_weeks maximum)
    """
    if as_of is None:
        as_of = datetime.now(UTC).date()

    season_start = MLB_SEASON_START.get(season, date(season, 3, 27))
    season_end = MLB_SEASON_END.get(season, date(season, 9, 28))

    if as_of < season_start:
        return total_weeks  # Pre-season: full season ahead

    if as_of > season_end:
        return 1  # Post-season: minimum

    # Compute current week number
    days_elapsed = (as_of - season_start).days
    current_week = days_elapsed // 7 + 1
    remaining = max(1, total_weeks - current_week + 1)

    return min(remaining, total_weeks)


def compute_schedule_strength(
    team_key: str,
    standings: dict[str, dict[str, float]] | None = None,
    remaining_opponents: list[str] | None = None,
) -> float:
    """
    Compute schedule strength based on remaining opponents' quality.

    Replaces the hardcoded `schedule_strength=0.5` in pages/8_Standings.py.

    Returns:
        0.0 (easiest remaining schedule) to 1.0 (hardest)
    """
    if not standings or not remaining_opponents:
        logger.debug("No standings/schedule data — cannot compute schedule strength")
        return None  # Return None instead of fake 0.5

    # Compute average opponent quality (win percentage or composite rank)
    opponent_qualities = []
    for opp_key in remaining_opponents:
        opp_stats = standings.get(opp_key, {})
        # Use win percentage as quality proxy
        wins = opp_stats.get("wins", 0)
        losses = opp_stats.get("losses", 0)
        total = wins + losses
        if total > 0:
            opponent_qualities.append(wins / total)

    if not opponent_qualities:
        return None

    avg_quality = sum(opponent_qualities) / len(opponent_qualities)
    # Normalize to 0-1 (0.5 = average, 1.0 = all top opponents)
    return round(min(1.0, max(0.0, avg_quality)), 3)


def compute_injury_exposure(
    roster_player_ids: list[int],
    injury_data: dict[int, float] | None = None,
    current_il: set[int] | None = None,
) -> float | None:
    """
    Compute injury exposure for a roster.

    Replaces the hardcoded `injury_exposure=0.1` in pages/8_Standings.py.

    Returns:
        0.0 (very healthy roster) to 1.0 (high injury risk), or None if no data
    """
    if not roster_player_ids:
        return None

    if not injury_data and not current_il:
        logger.debug("No injury data — cannot compute exposure")
        return None  # Return None instead of fake 0.1

    # Factor 1: What fraction of roster is currently on IL?
    il_fraction = 0.0
    if current_il:
        il_count = len(current_il.intersection(set(roster_player_ids)))
        il_fraction = il_count / len(roster_player_ids)

    # Factor 2: Average health risk across roster
    avg_risk = 0.0
    if injury_data:
        risks = []
        for pid in roster_player_ids:
            health_score = injury_data.get(pid, 0.85)  # default = low risk
            risks.append(1.0 - health_score)  # Invert: high health = low risk
        if risks:
            avg_risk = sum(risks) / len(risks)

    # Blend: 60% current IL status, 40% historical risk
    exposure = 0.6 * il_fraction + 0.4 * avg_risk
    return round(min(1.0, max(0.0, exposure)), 3)


def compute_momentum(
    recent_matchup_results: list[tuple[int, int]] | None = None,
    n_weeks: int = 3,
) -> float | None:
    """
    Compute momentum from recent matchup results.

    Replaces the hardcoded `momentum=1.0` in pages/8_Standings.py.

    Args:
        recent_matchup_results: List of (cats_won, cats_lost) for last N weeks
        n_weeks: How many recent weeks to consider

    Returns:
        0.0 (cold streak) to 2.0 (hot streak), centered at 1.0, or None if no data
    """
    if not recent_matchup_results:
        logger.debug("No recent matchup data — cannot compute momentum")
        return None  # Return None instead of fake 1.0

    # Use last n_weeks
    recent = recent_matchup_results[-n_weeks:]
    if not recent:
        return None

    total_won = sum(w for w, _ in recent)
    total_lost = sum(l for _, l in recent)
    total_cats = total_won + total_lost

    if total_cats == 0:
        return 1.0  # No data = neutral

    win_rate = total_won / total_cats
    # Map to 0-2 scale (0.5 win rate = 1.0 momentum)
    momentum = win_rate * 2.0
    return round(min(2.0, max(0.0, momentum)), 3)

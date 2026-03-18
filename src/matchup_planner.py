"""Weekly Matchup Planner for the HEATER fantasy baseball app.

Rates hitter and pitcher matchups for an upcoming week using a layered
multiplicative model with percentile-based color grading. Each player
receives a 1-10 rating and a 5-level color tier (smash / favorable /
neutral / unfavorable / avoid) based on their percentile rank among all
players of the same type.

Reuses schedule, park factor, and platoon utilities from
``src/optimizer/matchup_adjustments`` and park factor data from
``src/data_bootstrap``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.data_bootstrap import PARK_FACTORS
from src.optimizer.matchup_adjustments import (
    _build_team_schedule,
    get_weekly_schedule,
    platoon_adjustment,
)
from src.valuation import LeagueConfig

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

# Average number of games per team per week (baseline for scaling)
_BASELINE_GAMES_PER_WEEK: float = 6.5

# Home-field advantage multiplier for pitchers
_PITCHER_HOME_ADVANTAGE: float = 1.05

# Home-field advantage multiplier for hitters (smaller effect)
_HITTER_HOME_ADVANTAGE: float = 1.02

# Default base wOBA when player stats are missing
_DEFAULT_BASE_WOBA: float = 0.320

# Default xFIP when pitcher stats are missing
_DEFAULT_XFIP: float = 4.20

# Default opponent wRC+ when team batting stats are missing
_DEFAULT_OPP_WRC_PLUS: float = 100.0

# Floor for opponent wRC+ to avoid division issues
_MIN_OPP_WRC_PLUS: float = 50.0

# Tier thresholds (percentile boundaries)
_TIER_SMASH: float = 80.0
_TIER_FAVORABLE: float = 60.0
_TIER_NEUTRAL: float = 40.0
_TIER_UNFAVORABLE: float = 20.0


# ── Color Tier ───────────────────────────────────────────────────────


def color_tier(percentile_rank: float) -> str:
    """Map a percentile rank to a 5-level color tier.

    Args:
        percentile_rank: Value in [0, 100] representing where this
            player's raw matchup score falls among peers.

    Returns:
        One of: ``'smash'``, ``'favorable'``, ``'neutral'``,
        ``'unfavorable'``, ``'avoid'``.
    """
    if percentile_rank >= _TIER_SMASH:
        return "smash"
    if percentile_rank >= _TIER_FAVORABLE:
        return "favorable"
    if percentile_rank >= _TIER_NEUTRAL:
        return "neutral"
    if percentile_rank >= _TIER_UNFAVORABLE:
        return "unfavorable"
    return "avoid"


# ── Single-Game Ratings ──────────────────────────────────────────────


def compute_hitter_game_rating(
    player_stats: dict[str, Any],
    opposing_pitcher_stats: dict[str, Any] | None,
    park_factor: float,
    is_home: bool,
    batter_hand: str | None = None,
    pitcher_hand: str | None = None,
) -> dict[str, Any]:
    """Compute a single-game hitter matchup rating.

    Uses a multiplicative model:
        raw_score = base_wOBA * park_factor * platoon_adj * home_away

    The raw score is an intermediate value; the final 1-10 display
    rating is assigned later by ``compute_all_ratings_with_percentiles``
    across the full set of hitters.

    Args:
        player_stats: Dict with at least ``woba`` (or ``obp`` as proxy).
        opposing_pitcher_stats: Dict with pitcher quality info (unused
            in the raw formula but reserved for future enhancement).
        park_factor: Multiplicative park factor (e.g. 1.38 for Coors).
        is_home: Whether the player is the home team.
        batter_hand: ``'L'`` or ``'R'`` for platoon calculation.
        pitcher_hand: ``'L'`` or ``'R'`` for platoon calculation.

    Returns:
        Dict with ``raw_score`` (float) and component breakdown.
    """
    base_woba = float(player_stats.get("woba", 0) or player_stats.get("obp", 0) or _DEFAULT_BASE_WOBA)
    if base_woba <= 0:
        base_woba = _DEFAULT_BASE_WOBA

    pf = float(park_factor) if park_factor else 1.0

    # Platoon adjustment
    platoon_adj = 1.0
    if batter_hand and pitcher_hand:
        platoon_adj = platoon_adjustment(batter_hand, pitcher_hand)

    # Home advantage
    home_away = _HITTER_HOME_ADVANTAGE if is_home else 1.0

    raw_score = base_woba * pf * platoon_adj * home_away

    return {
        "raw_score": raw_score,
        "base_woba": base_woba,
        "park_factor": pf,
        "platoon_adj": platoon_adj,
        "home_away": home_away,
    }


def compute_pitcher_game_rating(
    pitcher_stats: dict[str, Any],
    opponent_team_stats: dict[str, Any] | None,
    park_factor: float,
    is_home: bool,
) -> dict[str, Any]:
    """Compute a single-game pitcher matchup rating.

    Uses a multiplicative model:
        raw_score = (10 - xFIP) * (100 / max(opp_wRC+, 50))
                    * inverse_park * starts * home_away

    Higher raw scores mean better pitcher matchups. The ``inverse_park``
    factor is ``2.0 - park_factor`` so that hitter-friendly parks
    penalise pitchers.

    Args:
        pitcher_stats: Dict with ``xfip`` (or ``era`` as fallback)
            and optionally ``starts`` (int, number of starts this week).
        opponent_team_stats: Dict with ``wrc_plus`` for opponent lineup.
        park_factor: Hitting park factor (will be inverted for pitcher).
        is_home: Whether the pitcher is on the home team.

    Returns:
        Dict with ``raw_score`` (float) and component breakdown.
    """
    xfip = float(pitcher_stats.get("xfip", 0) or pitcher_stats.get("era", 0) or _DEFAULT_XFIP)
    if xfip <= 0:
        xfip = _DEFAULT_XFIP

    opp_wrc_plus = _DEFAULT_OPP_WRC_PLUS
    if opponent_team_stats:
        opp_wrc_plus = float(opponent_team_stats.get("wrc_plus", _DEFAULT_OPP_WRC_PLUS) or _DEFAULT_OPP_WRC_PLUS)
    opp_wrc_plus = max(opp_wrc_plus, _MIN_OPP_WRC_PLUS)

    pf = float(park_factor) if park_factor else 1.0
    inverse_park = 2.0 - pf

    starts = int(pitcher_stats.get("starts", 1) or 1)
    if starts < 1:
        starts = 1

    home_away = _PITCHER_HOME_ADVANTAGE if is_home else 1.0

    raw_score = (10.0 - xfip) * (100.0 / opp_wrc_plus) * inverse_park * starts * home_away

    return {
        "raw_score": raw_score,
        "xfip": xfip,
        "opp_wrc_plus": opp_wrc_plus,
        "inverse_park": inverse_park,
        "starts": starts,
        "home_away": home_away,
    }


# ── Percentile Ranking ───────────────────────────────────────────────


def compute_all_ratings_with_percentiles(
    ratings_list: list[float],
) -> list[dict[str, Any]]:
    """Convert raw scores to percentile-based 1-10 ratings and tiers.

    Each raw score is ranked among all scores in ``ratings_list``.
    The percentile rank determines the display rating and color tier.

    Args:
        ratings_list: List of raw matchup scores (higher is better).

    Returns:
        List of dicts (same order as input), each containing:
          - ``percentile_rank``: float in [0, 100]
          - ``rating``: float in [1, 10]
          - ``tier``: str (smash/favorable/neutral/unfavorable/avoid)
    """
    if not ratings_list:
        return []

    arr = np.array(ratings_list, dtype=float)
    n = len(arr)

    results: list[dict[str, Any]] = []
    for val in arr:
        # Percentile rank: fraction of values strictly less than val
        if n == 1:
            pct = 50.0
        else:
            pct = float(np.sum(arr < val)) / (n - 1) * 100.0
        pct = max(0.0, min(100.0, pct))
        rating = 1.0 + 9.0 * (pct / 100.0)
        tier = color_tier(pct)
        results.append(
            {
                "percentile_rank": round(pct, 2),
                "rating": round(rating, 2),
                "tier": tier,
            }
        )
    return results


# ── Weekly Matchup Ratings ───────────────────────────────────────────


def compute_weekly_matchup_ratings(
    roster: pd.DataFrame,
    weekly_schedule: list[dict[str, Any]] | None = None,
    park_factors: dict[str, float] | None = None,
    team_batting_stats: dict[str, dict[str, Any]] | None = None,
    config: LeagueConfig | None = None,
) -> pd.DataFrame:
    """Compute matchup ratings for every player on the roster for the week.

    For each rostered player, iterates over the team's games this week,
    computes per-game raw scores, averages them (with games-count scaling
    for hitters), then uses percentile ranking to assign 1-10 ratings
    and color tiers.

    Args:
        roster: DataFrame with columns: ``player_id``, ``name``,
            ``team``, ``positions``, ``is_hitter``, and stat columns
            (``woba``/``obp`` for hitters, ``xfip``/``era`` for pitchers).
        weekly_schedule: Output of ``get_weekly_schedule()``.
            If None, the function will attempt to fetch it live.
        park_factors: Team abbreviation -> park factor dict.
            Defaults to ``PARK_FACTORS`` from ``data_bootstrap``.
        team_batting_stats: Dict mapping team abbreviation to a dict
            with ``wrc_plus`` (for pitcher matchup rating). Optional.
        config: League configuration. Defaults to ``LeagueConfig()``.

    Returns:
        DataFrame with columns: ``player_id``, ``name``, ``positions``,
        ``is_hitter``, ``games`` (list of per-game dicts),
        ``weekly_matchup_rating`` (1-10), ``matchup_tier`` (str),
        ``games_count`` (int), ``projected_stats_adjusted`` (dict).
        Returns empty DataFrame if roster is empty or no schedule data.
    """
    if roster is None or roster.empty:
        return _empty_result()

    if park_factors is None:
        park_factors = PARK_FACTORS

    if config is None:
        config = LeagueConfig()

    # Fetch schedule if not provided
    if weekly_schedule is None:
        weekly_schedule = get_weekly_schedule(days_ahead=7)

    if not weekly_schedule:
        return _empty_result()

    # Build per-team schedule lookup
    team_schedule = _build_team_schedule(weekly_schedule)

    # Collect per-player data
    hitter_rows: list[dict[str, Any]] = []
    pitcher_rows: list[dict[str, Any]] = []

    for _, row in roster.iterrows():
        player_id = row.get("player_id", 0)
        name = str(row.get("name", row.get("player_name", "")))
        team = str(row.get("team", "")).upper().strip()
        positions = row.get("positions", "")
        is_hitter = bool(row.get("is_hitter", True))
        games = team_schedule.get(team, [])
        n_games = len(games)

        if n_games == 0:
            entry = {
                "player_id": player_id,
                "name": name,
                "positions": positions,
                "is_hitter": is_hitter,
                "games": [],
                "games_count": 0,
                "raw_score": 0.0,
                "projected_stats_adjusted": {},
            }
            if is_hitter:
                hitter_rows.append(entry)
            else:
                pitcher_rows.append(entry)
            continue

        if is_hitter:
            player_stats = _extract_hitter_stats(row)
            game_details: list[dict[str, Any]] = []
            raw_scores: list[float] = []

            for game in games:
                pf = park_factors.get(game.get("park_team", ""), 1.0)
                is_home = bool(game.get("is_home", False))
                batter_hand = str(row.get("bats", "")) or None
                pitcher_hand = game.get("opposing_pitcher_hand") or None

                result = compute_hitter_game_rating(
                    player_stats=player_stats,
                    opposing_pitcher_stats=None,
                    park_factor=pf,
                    is_home=is_home,
                    batter_hand=batter_hand,
                    pitcher_hand=pitcher_hand,
                )
                result["game_date"] = game.get("game_date", "")
                result["opponent"] = game.get("opponent", "")
                game_details.append(result)
                raw_scores.append(result["raw_score"])

            # Weekly raw score: average per-game score * games_scaling
            avg_raw = float(np.mean(raw_scores)) if raw_scores else 0.0
            games_scaling = n_games / _BASELINE_GAMES_PER_WEEK
            weekly_raw = avg_raw * games_scaling

            hitter_rows.append(
                {
                    "player_id": player_id,
                    "name": name,
                    "positions": positions,
                    "is_hitter": True,
                    "games": game_details,
                    "games_count": n_games,
                    "raw_score": weekly_raw,
                    "projected_stats_adjusted": _projected_hitter_adjusted(player_stats, games, park_factors),
                }
            )
        else:
            pitcher_stats = _extract_pitcher_stats(row)
            game_details = []
            raw_scores = []

            for game in games:
                pf = park_factors.get(game.get("park_team", ""), 1.0)
                is_home = bool(game.get("is_home", False))
                opp_team = game.get("opponent", "")
                opp_stats = (team_batting_stats or {}).get(opp_team)

                result = compute_pitcher_game_rating(
                    pitcher_stats=pitcher_stats,
                    opponent_team_stats=opp_stats,
                    park_factor=pf,
                    is_home=is_home,
                )
                result["game_date"] = game.get("game_date", "")
                result["opponent"] = opp_team
                game_details.append(result)
                raw_scores.append(result["raw_score"])

            # Pitcher weekly raw: sum across starts (more starts = higher value)
            weekly_raw = float(np.sum(raw_scores)) if raw_scores else 0.0

            pitcher_rows.append(
                {
                    "player_id": player_id,
                    "name": name,
                    "positions": positions,
                    "is_hitter": False,
                    "games": game_details,
                    "games_count": n_games,
                    "raw_score": weekly_raw,
                    "projected_stats_adjusted": {},
                }
            )

    # Compute percentile ratings separately for hitters and pitchers
    _apply_percentile_ratings(hitter_rows)
    _apply_percentile_ratings(pitcher_rows)

    all_rows = hitter_rows + pitcher_rows
    if not all_rows:
        return _empty_result()

    df = pd.DataFrame(all_rows)
    # Keep only the output columns
    output_cols = [
        "player_id",
        "name",
        "positions",
        "is_hitter",
        "games",
        "weekly_matchup_rating",
        "matchup_tier",
        "games_count",
        "projected_stats_adjusted",
    ]
    for col in output_cols:
        if col not in df.columns:
            df[col] = None
    return df[output_cols].reset_index(drop=True)


# ── Helpers ──────────────────────────────────────────────────────────


def _empty_result() -> pd.DataFrame:
    """Return an empty DataFrame with the expected output columns."""
    return pd.DataFrame(
        columns=[
            "player_id",
            "name",
            "positions",
            "is_hitter",
            "games",
            "weekly_matchup_rating",
            "matchup_tier",
            "games_count",
            "projected_stats_adjusted",
        ]
    )


def _extract_hitter_stats(row: pd.Series) -> dict[str, Any]:
    """Extract hitter stat fields from a roster row."""
    return {
        "woba": row.get("woba", 0) or 0,
        "obp": row.get("obp", 0) or 0,
        "avg": row.get("avg", 0) or 0,
        "hr": row.get("hr", 0) or 0,
        "r": row.get("r", 0) or 0,
        "rbi": row.get("rbi", 0) or 0,
        "sb": row.get("sb", 0) or 0,
    }


def _extract_pitcher_stats(row: pd.Series) -> dict[str, Any]:
    """Extract pitcher stat fields from a roster row."""
    return {
        "xfip": row.get("xfip", 0) or 0,
        "era": row.get("era", 0) or 0,
        "whip": row.get("whip", 0) or 0,
        "k": row.get("k", 0) or 0,
        "w": row.get("w", 0) or 0,
        "sv": row.get("sv", 0) or 0,
        "starts": row.get("starts", 1) or 1,
    }


def _apply_percentile_ratings(rows: list[dict[str, Any]]) -> None:
    """Apply percentile-based ratings and tiers to a list of player rows in place.

    Players with zero games get rating 1.0 and tier 'avoid'.
    """
    raw_scores = [r["raw_score"] for r in rows if r["games_count"] > 0]
    rated = compute_all_ratings_with_percentiles(raw_scores)

    rated_idx = 0
    for row in rows:
        if row["games_count"] == 0:
            row["weekly_matchup_rating"] = 1.0
            row["matchup_tier"] = "avoid"
        else:
            info = rated[rated_idx]
            row["weekly_matchup_rating"] = info["rating"]
            row["matchup_tier"] = info["tier"]
            rated_idx += 1


def _projected_hitter_adjusted(
    player_stats: dict[str, Any],
    games: list[dict[str, Any]],
    park_factors: dict[str, float],
) -> dict[str, float]:
    """Compute park-adjusted projected stat totals for a hitter over the week.

    Scales counting stats by the average park factor across games.

    Args:
        player_stats: Base stat projections for the hitter.
        games: List of game dicts (with ``park_team``).
        park_factors: Team abbreviation -> park factor.

    Returns:
        Dict of stat name -> adjusted value.
    """
    if not games:
        return {}

    avg_pf = float(np.mean([park_factors.get(g.get("park_team", ""), 1.0) for g in games]))

    adjusted: dict[str, float] = {}
    for stat in ("hr", "r", "rbi", "sb"):
        base_val = float(player_stats.get(stat, 0) or 0)
        adjusted[stat] = round(base_val * avg_pf, 3)

    return adjusted

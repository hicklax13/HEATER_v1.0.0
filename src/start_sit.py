"""Start/Sit Advisor: 3-layer decision model for roster slot decisions.

Helps users decide which player to start when choosing between 2-4 players
competing for the same roster slot. Uses H2H category weights, weekly
matchup adjustments, and risk-adjusted scoring.

Core algorithm (3 layers):
  Layer 1: start_score = sum(h2h_weight[cat] * weekly_proj[cat] * matchup_factor[cat])
  Layer 2: risk_score = alpha * start_score + (1-alpha) * risk_component
  Layer 3: category_impact[cat] = weekly_proj[cat] / sgp_denom[cat] * h2h_weight[cat]

Wires into:
  - src/optimizer/h2h_engine.py: H2H category weights (Normal PDF)
  - src/engine/context/matchup.py: Log5 matchup adjustment factor
  - src/optimizer/matchup_adjustments.py: platoon, park factor, schedule
  - src/valuation.py: LeagueConfig, SGPCalculator
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.valuation import LeagueConfig, SGPCalculator

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

# Risk alpha by matchup state
_ALPHA_MAP: dict[str, float] = {
    "winning": 0.8,
    "close": 0.5,
    "losing": 0.2,
}

# Home field advantage multiplier
_HOME_ADVANTAGE: float = 1.0
_AWAY_DISCOUNT: float = 0.97

# Number of games in a typical MLB week
_DEFAULT_GAMES_PER_WEEK: int = 6

# Minimum players for comparison
_MIN_PLAYERS: int = 2
_MAX_PLAYERS: int = 4


# ── Matchup State Classification ────────────────────────────────────


def classify_matchup_state(
    my_weekly_totals: dict[str, float] | None,
    opp_weekly_totals: dict[str, float] | None,
    config: LeagueConfig | None = None,
) -> str:
    """Classify the current H2H matchup as winning, close, or losing.

    Counts expected category wins based on which side is ahead in each
    category. For inverse categories (ERA, WHIP, L), lower is better.

    Args:
        my_weekly_totals: Dict of my team's projected weekly totals per category.
        opp_weekly_totals: Dict of opponent's projected weekly totals per category.
        config: League configuration. Uses default if None.

    Returns:
        One of 'winning', 'close', or 'losing'.
        Returns 'close' if either totals dict is None or empty.
    """
    if not my_weekly_totals or not opp_weekly_totals:
        return "close"

    if config is None:
        config = LeagueConfig()

    inverse = {c.lower() for c in config.inverse_stats}
    wins = 0
    losses = 0
    total = 0

    for cat in config.all_categories:
        cat_lower = cat.lower()
        my_val = my_weekly_totals.get(cat, my_weekly_totals.get(cat_lower))
        opp_val = opp_weekly_totals.get(cat, opp_weekly_totals.get(cat_lower))

        if my_val is None or opp_val is None:
            continue

        total += 1
        my_val = float(my_val)
        opp_val = float(opp_val)

        if cat_lower in inverse or cat in config.inverse_stats:
            # Lower is better for inverse stats
            if my_val < opp_val:
                wins += 1
            elif my_val > opp_val:
                losses += 1
        else:
            if my_val > opp_val:
                wins += 1
            elif my_val < opp_val:
                losses += 1

    if total == 0:
        return "close"

    win_pct = wins / total
    if win_pct >= 0.58:
        return "winning"
    elif win_pct <= 0.42:
        return "losing"
    return "close"


# ── Risk-Adjusted Score ──────────────────────────────────────────────


def risk_adjusted_score(
    expected_value: float,
    p10_value: float,
    p90_value: float,
    matchup_state: str,
) -> float:
    """Compute risk-adjusted score based on matchup context.

    When winning, prefer the safe floor (alpha=0.8).
    When losing, chase the ceiling (alpha=0.2).
    When close, balance both (alpha=0.5).

    Args:
        expected_value: Expected (median) score.
        p10_value: 10th percentile (floor) score.
        p90_value: 90th percentile (ceiling) score.
        matchup_state: One of 'winning', 'close', or 'losing'.

    Returns:
        Risk-adjusted score as a float.
    """
    alpha = _ALPHA_MAP.get(matchup_state, 0.5)

    if matchup_state == "winning":
        # Protect the lead: weight toward floor
        risk_component = p10_value
    elif matchup_state == "losing":
        # Need upside: weight toward ceiling
        risk_component = p90_value
    else:
        # Balanced: average of floor and ceiling
        risk_component = (p10_value + p90_value) / 2.0

    return alpha * expected_value + (1.0 - alpha) * risk_component


# ── Weekly Projection ────────────────────────────────────────────────


def compute_weekly_projection(
    player: pd.Series,
    weekly_schedule: list[dict[str, Any]] | None = None,
    park_factors: dict[str, float] | None = None,
) -> dict[str, float]:
    """Adjust ROS per-game projection for this week's specific matchups.

    Scales season-rate projections to a weekly total based on the number
    of games this week. Applies park factor adjustments per game when
    schedule and park factor data are available.

    Args:
        player: Player Series with projection columns (r, hr, rbi, sb, etc.)
            and metadata (team, is_hitter, bats).
        weekly_schedule: List of game dicts from get_weekly_schedule().
            Each dict has game_date, home_name, away_name, etc.
        park_factors: Dict mapping team abbreviation to park factor.

    Returns:
        Dict of projected stats for the week. Keys are lowercase stat names.
        Falls back to ROS per-game rates scaled to default games if no
        schedule is available.
    """
    is_hitter = bool(player.get("is_hitter", 1))

    if is_hitter:
        stat_cols = ["r", "hr", "rbi", "sb", "avg", "obp"]
        counting_cols = ["r", "hr", "rbi", "sb"]
        # Also need component stats for rate stat calculation
        component_cols = ["ab", "h", "bb", "hbp", "sf", "pa"]
    else:
        stat_cols = ["w", "l", "sv", "k", "era", "whip"]
        counting_cols = ["w", "l", "sv", "k"]
        component_cols = ["ip", "er", "bb_allowed", "h_allowed"]

    # Get season totals from projections
    season_totals = {}
    for col in stat_cols + component_cols:
        season_totals[col] = float(player.get(col, 0) or 0)

    # Estimate games in the season for per-game rate
    if is_hitter:
        pa = season_totals.get("pa", 0) or season_totals.get("ab", 0) * 1.1
        # ~4.3 PA per game
        season_games = max(pa / 4.3, 1) if pa > 0 else 140
    else:
        ip = season_totals.get("ip", 0)
        # SP: ~6 IP/start, ~32 starts; RP: ~1 IP/app, ~65 apps
        season_games = max(ip / 5.5, 1) if ip > 0 else 30

    # Determine number of games this week
    team = str(player.get("team", "")).upper().strip()
    games_this_week = _count_team_games(team, weekly_schedule)

    if games_this_week == 0:
        games_this_week = _DEFAULT_GAMES_PER_WEEK if is_hitter else 1

    # Per-game rates
    per_game = {}
    for col in counting_cols + component_cols:
        per_game[col] = season_totals[col] / season_games if season_games > 0 else 0

    # Scale counting stats to this week's games
    weekly = {}
    for col in counting_cols:
        weekly[col] = per_game[col] * games_this_week

    # Apply park factor adjustments if available
    if park_factors and weekly_schedule:
        pf_avg = _average_park_factor(team, weekly_schedule, park_factors, is_hitter)
        for col in counting_cols:
            if is_hitter:
                weekly[col] *= pf_avg

    # Compute rate stats from components
    if is_hitter:
        weekly_ab = per_game.get("ab", 0) * games_this_week
        weekly_h = per_game.get("h", 0) * games_this_week
        weekly_bb = per_game.get("bb", 0) * games_this_week
        weekly_hbp = per_game.get("hbp", 0) * games_this_week
        weekly_sf = per_game.get("sf", 0) * games_this_week

        if weekly_ab > 0:
            weekly["avg"] = weekly_h / weekly_ab
        else:
            weekly["avg"] = season_totals.get("avg", 0.250)

        obp_denom = weekly_ab + weekly_bb + weekly_hbp + weekly_sf
        if obp_denom > 0:
            weekly["obp"] = (weekly_h + weekly_bb + weekly_hbp) / obp_denom
        else:
            weekly["obp"] = season_totals.get("obp", 0.320)
    else:
        weekly_ip = per_game.get("ip", 0) * games_this_week
        weekly_er = per_game.get("er", 0) * games_this_week
        weekly_bb_a = per_game.get("bb_allowed", 0) * games_this_week
        weekly_h_a = per_game.get("h_allowed", 0) * games_this_week

        if weekly_ip > 0:
            weekly["era"] = weekly_er * 9 / weekly_ip
            weekly["whip"] = (weekly_bb_a + weekly_h_a) / weekly_ip
        else:
            weekly["era"] = season_totals.get("era", 4.00)
            weekly["whip"] = season_totals.get("whip", 1.25)

    return weekly


# ── Main Recommendation ──────────────────────────────────────────────


def start_sit_recommendation(
    player_ids: list[int],
    player_pool: pd.DataFrame,
    config: LeagueConfig | None = None,
    weekly_schedule: list[dict[str, Any]] | None = None,
    park_factors: dict[str, float] | None = None,
    my_weekly_totals: dict[str, float] | None = None,
    opp_weekly_totals: dict[str, float] | None = None,
    standings: pd.DataFrame | None = None,
    team_name: str | None = None,
) -> dict[str, Any]:
    """Compare 2-4 players competing for the same roster slot.

    Uses a 3-layer decision model:
      Layer 1: H2H-weighted weekly projection score
      Layer 2: Risk adjustment based on matchup state
      Layer 3: Per-category SGP impact analysis

    Args:
        player_ids: List of 2-4 player IDs to compare.
        player_pool: Full player pool DataFrame.
        config: League configuration. Uses default if None.
        weekly_schedule: Weekly schedule from get_weekly_schedule().
        park_factors: Dict mapping team abbreviation to park factor.
        my_weekly_totals: My team's projected weekly category totals.
        opp_weekly_totals: Opponent's projected weekly category totals.
        standings: League standings DataFrame (for SGP context).
        team_name: My team name (for standings lookups).

    Returns:
        Dict with:
          - recommendation: player_id of the recommended start
          - confidence: float in [0, 1]
          - confidence_label: 'Clear Start', 'Lean Start', or 'Toss-up'
          - players: list of per-player detail dicts
    """
    if config is None:
        config = LeagueConfig()

    # Validate player count
    if len(player_ids) < _MIN_PLAYERS:
        if len(player_ids) == 1:
            # Single player: trivially recommend them
            p = player_pool[player_pool["player_id"] == player_ids[0]]
            name = ""
            if not p.empty:
                name = str(p.iloc[0].get("name", p.iloc[0].get("player_name", "")))
            return {
                "recommendation": player_ids[0],
                "confidence": 1.0,
                "confidence_label": "Clear Start",
                "players": [
                    {
                        "player_id": player_ids[0],
                        "name": name,
                        "start_score": 1.0,
                        "matchup_factors": {},
                        "floor": 1.0,
                        "ceiling": 1.0,
                        "category_impact": {},
                        "reasoning": ["Only candidate for this slot"],
                    }
                ],
            }
        return {
            "recommendation": None,
            "confidence": 0.0,
            "confidence_label": "Toss-up",
            "players": [],
        }

    if len(player_ids) > _MAX_PLAYERS:
        player_ids = player_ids[:_MAX_PLAYERS]

    sgp_calc = SGPCalculator(config)

    # Compute H2H category weights
    h2h_weights = _get_h2h_weights(my_weekly_totals, opp_weekly_totals, config)

    # Classify matchup state for risk adjustment
    matchup_state = classify_matchup_state(my_weekly_totals, opp_weekly_totals, config)

    player_results = []

    for pid in player_ids:
        if "player_id" not in player_pool.columns:
            continue
        p_rows = player_pool[player_pool["player_id"] == pid]
        if p_rows.empty:
            continue

        player = p_rows.iloc[0]
        name = str(player.get("name", player.get("player_name", f"Player {pid}")))
        is_hitter = bool(player.get("is_hitter", 1))

        # Layer 1: Weekly projection with matchup factors
        weekly_proj = compute_weekly_projection(player, weekly_schedule, park_factors)

        # Compute per-game matchup factors
        matchup_factors = _compute_matchup_factors(player, weekly_schedule, park_factors, config)

        # Start score: sum of H2H-weighted weekly projections
        start_score = _compute_start_score(weekly_proj, h2h_weights, matchup_factors, config, is_hitter)

        # Floor/ceiling estimates (P10/P90 approximation)
        floor_score = start_score * 0.75
        ceiling_score = start_score * 1.30

        # Layer 2: Risk adjustment
        adjusted_score = risk_adjusted_score(start_score, floor_score, ceiling_score, matchup_state)

        # Layer 3: Category impact (SGP contribution)
        category_impact = _compute_category_impact(weekly_proj, h2h_weights, config, is_hitter)

        # Generate reasoning
        reasoning = _generate_reasoning(
            player, matchup_factors, category_impact, weekly_proj, park_factors, config, is_hitter
        )

        player_results.append(
            {
                "player_id": pid,
                "name": name,
                "start_score": round(adjusted_score, 4),
                "matchup_factors": matchup_factors,
                "floor": round(floor_score, 4),
                "ceiling": round(ceiling_score, 4),
                "category_impact": {k: round(v, 4) for k, v in category_impact.items()},
                "reasoning": reasoning,
                "_raw_score": adjusted_score,  # For sorting
            }
        )

    if not player_results:
        return {
            "recommendation": None,
            "confidence": 0.0,
            "confidence_label": "Toss-up",
            "players": [],
        }

    # Sort by adjusted score descending
    player_results.sort(key=lambda x: x["_raw_score"], reverse=True)

    # Compute confidence from gap between top 2 players
    best_score = player_results[0]["_raw_score"]
    second_score = player_results[1]["_raw_score"] if len(player_results) > 1 else 0.0

    score_sum = abs(best_score) + abs(second_score)
    if score_sum > 0:
        confidence = abs(best_score - second_score) / score_sum
    else:
        confidence = 0.0

    confidence = min(confidence, 1.0)

    if confidence > 0.30:
        confidence_label = "Clear Start"
    elif confidence > 0.15:
        confidence_label = "Lean Start"
    else:
        confidence_label = "Toss-up"

    # Remove internal sorting key
    for pr in player_results:
        pr.pop("_raw_score", None)

    return {
        "recommendation": player_results[0]["player_id"],
        "confidence": round(confidence, 4),
        "confidence_label": confidence_label,
        "players": player_results,
    }


# ── Internal Helpers ─────────────────────────────────────────────────


def _get_h2h_weights(
    my_totals: dict[str, float] | None,
    opp_totals: dict[str, float] | None,
    config: LeagueConfig,
) -> dict[str, float]:
    """Get H2H category weights, falling back to uniform weights.

    Args:
        my_totals: My team's projected totals.
        opp_totals: Opponent's projected totals.
        config: League configuration.

    Returns:
        Dict mapping category name (lowercase) to weight.
    """
    if my_totals and opp_totals:
        try:
            from src.optimizer.h2h_engine import compute_h2h_category_weights

            weights = compute_h2h_category_weights(my_totals, opp_totals)
            if weights:
                return weights
        except Exception:
            logger.debug("Failed to compute H2H weights, using uniform", exc_info=True)

    # Uniform fallback
    return {cat.lower(): 1.0 for cat in config.all_categories}


def _count_team_games(
    team: str,
    weekly_schedule: list[dict[str, Any]] | None,
) -> int:
    """Count number of games for a team in the weekly schedule.

    Args:
        team: 3-letter team abbreviation.
        weekly_schedule: List of game dicts.

    Returns:
        Number of games found, 0 if schedule unavailable.
    """
    if not weekly_schedule or not team:
        return 0

    # Build reverse lookup for full team names
    try:
        from src.optimizer.matchup_adjustments import _MLB_TEAM_ABBREVS

        abbrev_to_full = {v: k for k, v in _MLB_TEAM_ABBREVS.items()}
    except ImportError:
        abbrev_to_full = {}

    full_name = abbrev_to_full.get(team, team)

    count = 0
    for game in weekly_schedule:
        home = game.get("home_name", "")
        away = game.get("away_name", "")
        if team in (home, away) or full_name in (home, away):
            count += 1
    return count


def _average_park_factor(
    team: str,
    weekly_schedule: list[dict[str, Any]],
    park_factors: dict[str, float],
    is_hitter: bool,
) -> float:
    """Compute average park factor across a week of games.

    Args:
        team: Player's team abbreviation.
        weekly_schedule: List of game dicts.
        park_factors: Dict of park factors by team abbreviation.
        is_hitter: Whether the player is a hitter.

    Returns:
        Average park factor for the week. 1.0 if no data.
    """
    try:
        from src.optimizer.matchup_adjustments import (
            _MLB_TEAM_ABBREVS,
        )

        abbrev_to_full = {v: k for k, v in _MLB_TEAM_ABBREVS.items()}
    except ImportError:
        return 1.0

    full_name = abbrev_to_full.get(team, team)
    factors = []

    for game in weekly_schedule:
        home = game.get("home_name", "")
        away = game.get("away_name", "")

        if team in (home, away) or full_name in (home, away):
            # Determine the venue (home team's park)
            if home in (team, full_name):
                venue_team = team
            else:
                # Player is away; venue is the other team's park
                # Convert full name to abbreviation
                for abbr, full in abbrev_to_full.items():
                    if full == home:
                        venue_team = abbr
                        break
                else:
                    venue_team = home

            pf = park_factors.get(venue_team, 1.0)
            if not is_hitter:
                pf = 1.0  # Neutral for pitchers
            factors.append(pf)

    if not factors:
        return 1.0
    return float(np.mean(factors))


def _compute_matchup_factors(
    player: pd.Series,
    weekly_schedule: list[dict[str, Any]] | None,
    park_factors: dict[str, float] | None,
    config: LeagueConfig,
) -> dict[str, float]:
    """Compute per-factor matchup adjustments for a player.

    Each factor is a multiplicative modifier:
      matchup_factor = log5_factor * park_factor * platoon_adj * home_away

    Args:
        player: Player Series.
        weekly_schedule: Weekly schedule data.
        park_factors: Park factor dict.
        config: League configuration.

    Returns:
        Dict with keys: log5, park, platoon, home_away, combined.
    """
    team = str(player.get("team", "")).upper().strip()
    is_hitter = bool(player.get("is_hitter", 1))

    factors = {
        "log5": 1.0,
        "park": 1.0,
        "platoon": 1.0,
        "home_away": 1.0,
        "combined": 1.0,
    }

    if not weekly_schedule:
        return factors

    # Park factor average
    if park_factors:
        factors["park"] = _average_park_factor(team, weekly_schedule, park_factors, is_hitter)

    # Platoon adjustment (average across games)
    batter_hand = str(player.get("bats", "R")).upper().strip()
    if is_hitter and batter_hand in ("L", "R"):
        try:
            from src.optimizer.matchup_adjustments import platoon_adjustment

            platoon_adjustments = []
            for game in weekly_schedule:
                home = game.get("home_name", "")
                away = game.get("away_name", "")
                if team in (home, away) or any(team == a for a in (home, away)):
                    # Get probable pitcher hand (simplified: assume unknown -> R)
                    pitcher_hand = "R"  # Default assumption
                    platoon_adjustments.append(platoon_adjustment(batter_hand, pitcher_hand))

            if platoon_adjustments:
                factors["platoon"] = float(np.mean(platoon_adjustments))
        except ImportError:
            pass

    # Home/away proportion
    if team:
        try:
            from src.optimizer.matchup_adjustments import _MLB_TEAM_ABBREVS

            abbrev_to_full = {v: k for k, v in _MLB_TEAM_ABBREVS.items()}
        except ImportError:
            abbrev_to_full = {}

        full_name = abbrev_to_full.get(team, team)
        home_games = 0
        total_games = 0
        for game in weekly_schedule:
            home = game.get("home_name", "")
            away = game.get("away_name", "")
            if team in (home, away) or full_name in (home, away):
                total_games += 1
                if home in (team, full_name):
                    home_games += 1

        if total_games > 0:
            home_pct = home_games / total_games
            factors["home_away"] = home_pct * _HOME_ADVANTAGE + (1.0 - home_pct) * _AWAY_DISCOUNT

    factors["combined"] = factors["log5"] * factors["park"] * factors["platoon"] * factors["home_away"]

    return factors


def _compute_start_score(
    weekly_proj: dict[str, float],
    h2h_weights: dict[str, float],
    matchup_factors: dict[str, float],
    config: LeagueConfig,
    is_hitter: bool,
) -> float:
    """Compute Layer 1 start score from weighted projections.

    start_score = sum(h2h_weight[cat] * weekly_proj[cat] * combined_matchup_factor)

    For inverse categories (ERA, WHIP, L), the contribution is negative
    (lower is better, so a high ERA hurts).

    Args:
        weekly_proj: Weekly stat projections.
        h2h_weights: H2H category weights.
        matchup_factors: Matchup adjustment factors.
        config: League configuration.
        is_hitter: Whether the player is a hitter.

    Returns:
        Weighted start score as float.
    """
    combined_factor = matchup_factors.get("combined", 1.0)
    inverse = {c.lower() for c in config.inverse_stats}

    score = 0.0
    cats = config.hitting_categories if is_hitter else config.pitching_categories

    for cat in cats:
        cat_lower = cat.lower()
        proj_val = weekly_proj.get(cat_lower, 0.0)
        weight = h2h_weights.get(cat_lower, 1.0)
        denom = config.sgp_denominators.get(cat, 1.0)

        if abs(denom) < 1e-9:
            denom = 1.0

        # Normalize to SGP-scale so categories are comparable
        sgp_contribution = proj_val / denom

        if cat_lower in inverse:
            # Lower is better: negative contribution
            score -= sgp_contribution * weight * combined_factor
        else:
            score += sgp_contribution * weight * combined_factor

    return score


def _compute_category_impact(
    weekly_proj: dict[str, float],
    h2h_weights: dict[str, float],
    config: LeagueConfig,
    is_hitter: bool,
) -> dict[str, float]:
    """Compute per-category SGP impact (Layer 3).

    category_impact[cat] = weekly_proj[cat] / sgp_denom[cat] * h2h_weight[cat]

    Args:
        weekly_proj: Weekly stat projections.
        h2h_weights: H2H category weights.
        config: League configuration.
        is_hitter: Whether the player is a hitter.

    Returns:
        Dict mapping category name to SGP impact delta.
    """
    inverse = {c.lower() for c in config.inverse_stats}
    impact = {}

    cats = config.hitting_categories if is_hitter else config.pitching_categories

    for cat in cats:
        cat_lower = cat.lower()
        proj_val = weekly_proj.get(cat_lower, 0.0)
        weight = h2h_weights.get(cat_lower, 1.0)
        denom = config.sgp_denominators.get(cat, 1.0)

        if abs(denom) < 1e-9:
            denom = 1.0

        sgp_impact = proj_val / denom * weight
        if cat_lower in inverse:
            sgp_impact = -sgp_impact

        impact[cat] = sgp_impact

    return impact


def _generate_reasoning(
    player: pd.Series,
    matchup_factors: dict[str, float],
    category_impact: dict[str, float],
    weekly_proj: dict[str, float],
    park_factors: dict[str, float] | None,
    config: LeagueConfig,
    is_hitter: bool,
) -> list[str]:
    """Generate human-readable reasoning for the recommendation.

    Surfaces top 2-3 factors driving the recommendation, including
    matchup quality, park factor advantage, platoon edge, and
    category need.

    Args:
        player: Player Series.
        matchup_factors: Per-factor matchup adjustments.
        category_impact: Per-category SGP impact.
        weekly_proj: Weekly projections.
        park_factors: Park factor dict.
        config: League configuration.
        is_hitter: Whether player is a hitter.

    Returns:
        List of 1-3 reasoning strings.
    """
    reasons = []
    team = str(player.get("team", "")).upper().strip()

    # Park factor reasoning
    pf = matchup_factors.get("park", 1.0)
    if pf > 1.05:
        reasons.append(f"Favorable park factors this week ({pf:.2f}x)")
    elif pf < 0.95:
        reasons.append(f"Unfavorable park factors this week ({pf:.2f}x)")

    # Platoon reasoning
    platoon = matchup_factors.get("platoon", 1.0)
    if platoon > 1.02:
        reasons.append("Platoon advantage against probable pitchers")
    elif platoon < 0.98:
        reasons.append("Platoon disadvantage against probable pitchers")

    # Home/away reasoning
    ha = matchup_factors.get("home_away", 1.0)
    if ha > 0.99:
        reasons.append("Majority home games this week")
    elif ha < 0.97:
        reasons.append("Majority away games this week")

    # Top category impact
    if category_impact:
        sorted_cats = sorted(category_impact.items(), key=lambda x: abs(x[1]), reverse=True)
        top_cat, top_val = sorted_cats[0]
        if abs(top_val) > 0.01:
            direction = "strong" if top_val > 0 else "negative"
            reasons.append(f"Projected {direction} {top_cat} impact ({top_val:+.3f} Standings Gained Points)")

    # Volume reasoning for hitters
    if is_hitter:
        games = _count_team_games(team, None)
        # Can't determine without schedule, so skip if no data

    # Ensure at least one reason
    if not reasons:
        reasons.append("Comparable projected value to alternatives")

    return reasons[:3]

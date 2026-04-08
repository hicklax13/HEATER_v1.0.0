"""Weekly matchup adjustments for the lineup optimizer.

Adjusts player projections based on game-specific context:
  - Platoon splits (LHB vs RHP advantage, RHB vs LHP advantage)
  - Park factors (Coors inflates, Miami deflates)
  - Weather (HR adjustment per Nathan's physics-of-baseball)
  - Opposing pitcher quality via Log5 matchup model

Each function degrades gracefully when data is unavailable
and returns neutral (1.0) adjustment factors by default.

Wires into:
  - src/engine/context/matchup.py: matchup_adjustment_factor, park_adjust
  - src/data_bootstrap.py: PARK_FACTORS (30-team dict)
  - statsapi: MLB schedule data (imported inside function body)
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from src.validation.constant_optimizer import load_constants

logger = logging.getLogger(__name__)

_CONSTANTS = load_constants()

# ── Constants ────────────────────────────────────────────────────────

# Map full MLB team names (from statsapi) to abbreviations used in player tables
_MLB_TEAM_ABBREVS: dict[str, str] = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}

# Reverse mapping: abbreviation -> full name (for lookups)
_ABBREV_TO_FULL: dict[str, str] = {v: k for k, v in _MLB_TEAM_ABBREVS.items()}

# The Book regression PA for platoon split regression-to-mean.
# LHB platoon splits stabilize faster (1000 PA) than RHB (2200 PA).
PLATOON_REGRESSION_PA: dict[str, int] = {"LHB": 1000, "RHB": 2200}

# Default league-average platoon advantages (wOBA-based).
# LHB vs RHP: +8.6% wOBA advantage. RHB vs LHP: +6.1% wOBA advantage.
# Source: The Book — Playing the Percentages in Baseball (Tango, Lichtman, Dolphin)
DEFAULT_PLATOON_ADVANTAGE: dict[str, float] = {"LHB": 0.086, "RHB": 0.061}

# Hitter counting stat columns that get matchup adjustments
_HITTER_COUNTING_STATS: list[str] = ["r", "hr", "rbi", "sb"]

# Pitcher counting stat columns that get matchup adjustments
_PITCHER_COUNTING_STATS: list[str] = ["w", "sv", "k"]

# Reference temperature (Fahrenheit) for Nathan's HR physics model
_REFERENCE_TEMP_F: float = 72.0

# HR increase per degree F above reference (Nathan 2012, calibratable)
_HR_TEMP_COEFFICIENT: float = _CONSTANTS.get("hr_temp_coefficient")


# ── Schedule Fetching ────────────────────────────────────────────────


def get_weekly_schedule(days_ahead: int = 7) -> list[dict[str, Any]]:
    """Fetch MLB schedule for the next N days.

    Uses ``statsapi.schedule()`` to retrieve upcoming games.
    Each entry includes game date, team names, and probable pitchers.

    Args:
        days_ahead: Number of days to look ahead (default 7).

    Returns:
        List of dicts, each containing:
          - game_date: str (YYYY-MM-DD)
          - home_name: str (full team name)
          - away_name: str (full team name)
          - home_probable_pitcher: str or ""
          - away_probable_pitcher: str or ""

        Returns empty list if statsapi is unavailable or API fails.
    """
    try:
        import statsapi
    except ImportError:
        logger.debug("statsapi not installed; returning empty schedule")
        return []

    try:
        today = datetime.now(UTC)
        end = today + timedelta(days=days_ahead)
        raw = statsapi.schedule(
            start_date=today.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
        )
    except Exception:
        logger.warning("Failed to fetch MLB schedule", exc_info=True)
        return []

    results: list[dict[str, Any]] = []
    for game in raw:
        results.append(
            {
                "game_date": game.get("game_date", ""),
                "home_name": game.get("home_name", ""),
                "away_name": game.get("away_name", ""),
                "home_probable_pitcher": game.get("home_probable_pitcher", ""),
                "away_probable_pitcher": game.get("away_probable_pitcher", ""),
            }
        )
    return results


# ── Platoon Adjustment ───────────────────────────────────────────────


def platoon_adjustment(
    batter_hand: str,
    pitcher_hand: str,
    batter_split_avg: float | None = None,
    batter_overall_avg: float | None = None,
    sample_pa: int = 0,
) -> float:
    """Compute a multiplicative platoon adjustment factor.

    Uses The Book's regression formula to blend individual split data
    with league-average platoon splits. When no individual data is
    available, returns the league-average platoon advantage.

    The Book regression formula:
        regressed = (sample_pa * split_rate + regression_pa * overall)
                    / (sample_pa + regression_pa)

    Args:
        batter_hand: "L" or "R" for left/right-handed batter.
        pitcher_hand: "L" or "R" for left/right-handed pitcher.
        batter_split_avg: Batter's batting average in this split
            (e.g., LHB vs RHP). None if unavailable.
        batter_overall_avg: Batter's overall batting average.
            None if unavailable.
        sample_pa: Number of PA in this specific platoon split.

    Returns:
        Multiplicative adjustment factor:
          - > 1.0 when batter has platoon advantage (opposite hand)
          - < 1.0 when batter faces same-handed pitcher
          - 1.0 when hands are unknown or data is ambiguous
    """
    bh = batter_hand.upper().strip() if batter_hand else ""
    ph = pitcher_hand.upper().strip() if pitcher_hand else ""

    # Unknown handedness -> neutral
    if bh not in ("L", "R") or ph not in ("L", "R"):
        return 1.0

    # Same hand: batter is at a disadvantage (pitcher's advantage)
    # Opposite hand: batter has the platoon advantage
    has_advantage = bh != ph

    # Determine which platoon category applies
    if bh == "L":
        default_advantage = DEFAULT_PLATOON_ADVANTAGE["LHB"]
        regression_pa = PLATOON_REGRESSION_PA["LHB"]
    else:
        default_advantage = DEFAULT_PLATOON_ADVANTAGE["RHB"]
        regression_pa = PLATOON_REGRESSION_PA["RHB"]

    # If no individual split data, use league-average platoon effect
    if batter_split_avg is None or batter_overall_avg is None or sample_pa <= 0:
        if has_advantage:
            return 1.0 + default_advantage
        else:
            return 1.0 - default_advantage

    # The Book regression: blend individual split with overall rate
    # Weight individual data by sample size, overall by regression PA
    regressed = (sample_pa * batter_split_avg + regression_pa * batter_overall_avg) / (sample_pa + regression_pa)

    # Adjustment is regressed split relative to overall
    if batter_overall_avg > 0:
        factor = regressed / batter_overall_avg
    else:
        factor = 1.0

    return float(factor)


# ── Bayesian Platoon Adjustment ──────────────────────────────────────

# League-average platoon advantages (from The Book)
_LHB_VS_RHP_ADVANTAGE = 0.086  # +8.6% wOBA
_RHB_VS_LHP_ADVANTAGE = 0.061  # +6.1% wOBA
_LHB_STABILIZATION_PA = 1000
_RHB_STABILIZATION_PA = 2200


def bayesian_platoon_adjustment(
    batter_hand: str,
    pitcher_hand: str,
    individual_split_avg: float | None = None,
    individual_overall_avg: float | None = None,
    sample_pa: int = 0,
) -> float:
    """Bayesian-blended platoon adjustment.

    Blends individual split data with league-average splits based on sample size.
    At 200 PA of individual LHB vs LHP data: 80% league average, 20% individual.
    At 1000 PA: 50/50.

    Args:
        batter_hand: "L" or "R" for left/right-handed batter.
        pitcher_hand: "L" or "R" for left/right-handed pitcher.
        individual_split_avg: Batter's batting average in this split.
            None if unavailable.
        individual_overall_avg: Batter's overall batting average.
            None if unavailable.
        sample_pa: Number of PA in this specific platoon split.

    Returns:
        Multiplicative adjustment factor clamped to [0.80, 1.20].
    """
    bh = batter_hand.upper().strip() if batter_hand else ""
    ph = pitcher_hand.upper().strip() if pitcher_hand else ""

    # Unknown handedness -> neutral
    if bh not in ("L", "R") or ph not in ("L", "R"):
        return 1.0

    # Determine league-average advantage
    has_advantage = bh != ph

    if bh == "L":
        league_advantage = _LHB_VS_RHP_ADVANTAGE if has_advantage else -_LHB_VS_RHP_ADVANTAGE
        stab_pa = _LHB_STABILIZATION_PA
    else:
        league_advantage = _RHB_VS_LHP_ADVANTAGE if has_advantage else -_RHB_VS_LHP_ADVANTAGE
        stab_pa = _RHB_STABILIZATION_PA

    # Bayesian blend
    if (
        individual_split_avg is not None
        and individual_overall_avg is not None
        and sample_pa > 0
        and individual_overall_avg > 0
    ):
        individual_advantage = (individual_split_avg - individual_overall_avg) / individual_overall_avg
        blend_weight = min(1.0, sample_pa / stab_pa)
        blended_advantage = blend_weight * individual_advantage + (1 - blend_weight) * league_advantage
    else:
        blended_advantage = league_advantage

    return max(0.80, min(1.20, 1.0 + blended_advantage))


# ── Calibrated Pitcher Quality Multiplier ────────────────────────────


def calibrated_pitcher_quality_mult(
    opp_era: float,
    league_avg_era: float = 4.20,
    league_std_era: float = 0.80,
) -> float:
    """Calibrated opposing pitcher quality multiplier.

    Research: facing ace vs replacement is approximately plus or minus 15%
    counting stat adjustment. Uses z-score of pitcher ERA vs league average,
    scaled to plus or minus 15% max.

    Args:
        opp_era: Opposing pitcher's ERA.
        league_avg_era: League average ERA (default 4.20).
        league_std_era: League ERA standard deviation (default 0.80).

    Returns:
        Multiplier centered at 1.0 (1.15 for weak pitcher, 0.85 for ace).
        Clamped to [0.85, 1.15].
    """
    z = (opp_era - league_avg_era) / max(league_std_era, 0.01)
    # Clamp z-score to [-2, +2] -> multiplier to [0.85, 1.15]
    z_clamped = max(-2.0, min(2.0, z))
    return 1.0 + z_clamped * 0.075  # +/-0.15 at z=+/-2


# ── Park Factor Adjustment ───────────────────────────────────────────


def park_factor_adjustment(
    player_team: str,
    opponent_team: str,
    park_factors: dict[str, float],
    is_hitter: bool = True,
) -> float:
    """Get the park factor for a matchup based on the game's ballpark.

    Uses the home team's park factor. The convention is that the home
    team determines the park.

    In practice, for a week of games a player might be home and away,
    but this function handles a single game context. The caller
    iterates over a week's schedule.

    Args:
        player_team: 3-letter abbreviation of the player's team.
        opponent_team: 3-letter abbreviation of the opponent team.
        park_factors: Dict mapping team abbreviation -> park factor.
        is_hitter: True for hitters (use hitting PF), False for pitchers.

    Returns:
        Multiplicative park factor (1.0 = neutral).
        Returns 1.0 if either team is not found in park_factors.
    """
    # Determine home team: the game venue is the home team's park.
    # Without explicit home/away info, we need schedule context.
    # This function receives both teams and looks up the park.
    # The caller should pass (player_team, opponent_team) using schedule
    # data to determine which is home.
    #
    # For simplicity, use opponent_team as the venue when the player is
    # away, or player_team when home. The caller should resolve this.
    # Here we just return the park factor for the provided venue.

    pt = player_team.upper().strip() if player_team else ""
    ot = opponent_team.upper().strip() if opponent_team else ""

    # Try home team's park factor, then opponent's, then neutral
    # The caller typically passes the home team as opponent_team
    # when the player is visiting.
    pf = park_factors.get(ot, park_factors.get(pt, 1.0))

    if not is_hitter:
        # C5: Reciprocal park factor for pitchers — Coors (1.38) → 0.72 for pitchers
        # (hitter-friendly parks hurt pitcher stats). More accurate than linear dampening.
        return 1.0 / pf if pf > 0 else 1.0

    return float(pf)


# ── Weather Adjustment ───────────────────────────────────────────────


def weather_hr_adjustment(temp_f: float = 72.0) -> float:
    """Compute HR adjustment based on game temperature.

    Based on Alan Nathan's physics-of-baseball research:
    fly ball distance increases roughly linearly with temperature,
    producing about +0.9% more home runs per degree F above 72.

    Only applies a positive bonus for hot weather. Cold weather
    (below 72F) does not reduce HRs below baseline in this model
    to avoid double-penalizing cold-weather parks that already have
    low park factors.

    Args:
        temp_f: Game temperature in Fahrenheit (default 72 = neutral).

    Returns:
        Multiplicative adjustment factor for HR only.
        Always >= 1.0.
    """
    return 1.0 + _HR_TEMP_COEFFICIENT * max(0.0, temp_f - _REFERENCE_TEMP_F)


# ── Rain Adjustment ─────────────────────────────────────────────────


def weather_rain_adjustment(precip_pct: float = 0.0) -> dict[str, float]:
    """Adjust K and BB projections based on rain probability.

    Research (AMS): rain >40% increases walks by ~9.6% and decreases
    strikeouts by ~10.1%.  Linear interpolation from 0% effect at
    precip=0 to full effect at precip>=40%.

    Args:
        precip_pct: Precipitation probability as a percentage (0-100).

    Returns:
        Dict with k_mult (strikeout multiplier) and bb_mult (walk multiplier).
        Both are 1.0 when no rain.
    """
    if precip_pct <= 0:
        return {"k_mult": 1.0, "bb_mult": 1.0}
    rain_factor = min(1.0, precip_pct / 40.0)
    return {
        "k_mult": 1.0 - 0.101 * rain_factor,  # Up to -10.1% K
        "bb_mult": 1.0 + 0.096 * rain_factor,  # Up to +9.6% BB
    }


# ── Wind HR Adjustment ──────────────────────────────────────────────


def is_wind_blowing_out(
    wind_direction_deg: float,
    outfield_bearing_deg: float,
    tolerance_deg: float = 60.0,
) -> bool:
    """E6: Determine if wind blows toward outfield based on direction and stadium orientation.

    Wind blows "out" when it flows in the same direction as the outfield bearing
    (within tolerance degrees). Wind "in" flows opposite.

    Args:
        wind_direction_deg: Meteorological wind direction (degrees from North, 0-360).
            This is the direction wind comes FROM.
        outfield_bearing_deg: Direction CF faces (degrees from North, from T11).
        tolerance_deg: Angular tolerance for "out" classification (default 60°).

    Returns:
        True if wind is blowing out (toward outfield).
    """
    # Meteorological convention: wind_direction = where wind comes FROM.
    # Wind blows TOWARD: (wind_direction + 180) % 360.
    wind_toward = (wind_direction_deg + 180) % 360
    # Angular difference between wind target and outfield bearing
    diff = abs(wind_toward - outfield_bearing_deg)
    if diff > 180:
        diff = 360 - diff
    return diff <= tolerance_deg


def weather_wind_hr_adjustment(wind_mph: float = 0.0, wind_out: bool = False) -> float:
    """Adjust HR projection for wind speed and direction.

    Research: wind blowing out >10 mph increases HR rate by 15-20%.
    Wind blowing in reduces HR rate by 10-15%.

    Args:
        wind_mph: Wind speed in miles per hour.
        wind_out: True if wind is blowing out (toward outfield),
            False if blowing in or crosswind.

    Returns:
        HR multiplier (1.0 = neutral).
    """
    if wind_mph < 5:
        return 1.0
    if wind_out:
        return 1.0 + min(0.20, (wind_mph - 5) * 0.04)  # +4% per mph above 5, cap 20%
    else:
        return max(0.85, 1.0 - (wind_mph - 5) * 0.03)  # -3% per mph above 5, floor 85%


# ── Weekly Matchup Adjustment Pipeline ───────────────────────────────


def _resolve_team_abbrev(full_name: str) -> str:
    """Convert a full MLB team name to its abbreviation.

    Args:
        full_name: Full team name (e.g., "Colorado Rockies").

    Returns:
        3-letter abbreviation (e.g., "COL") or empty string.
    """
    return _MLB_TEAM_ABBREVS.get(full_name, "")


def _load_weather_by_date(game_dates: set[str]) -> dict[str, dict[str, Any]]:
    """Load weather data from DB for a set of game dates.

    Returns a dict keyed by (game_date, venue_team) -> weather dict.
    """
    weather_lookup: dict[str, dict[str, Any]] = {}
    try:
        from src.database import load_game_day_weather

        for gd in game_dates:
            df = load_game_day_weather(gd)
            if df.empty:
                continue
            for _, row in df.iterrows():
                key = f"{gd}:{row.get('venue_team', '')}"
                weather_lookup[key] = {
                    "temp_f": row.get("temp_f"),
                    "wind_mph": row.get("wind_mph"),
                    "wind_dir": row.get("wind_dir"),
                    "precip_pct": row.get("precip_pct"),
                    "humidity_pct": row.get("humidity_pct"),
                }
    except Exception:
        logger.debug("Could not load weather data from DB", exc_info=True)
    return weather_lookup


def _build_team_schedule(
    week_schedule: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group schedule entries by team abbreviation.

    For each team, stores a list of games with the opponent, venue,
    probable pitchers, and weather data (temp_f, wind_mph) when available.

    Args:
        week_schedule: List of game dicts from ``get_weekly_schedule()``.

    Returns:
        Dict mapping team abbreviation to list of game info dicts.
    """
    team_games: dict[str, list[dict[str, Any]]] = {}

    # Collect all game dates for batch weather lookup
    game_dates: set[str] = set()
    for game in week_schedule:
        gd = game.get("game_date", "")
        if gd:
            game_dates.add(gd)

    # Load weather from DB (populated by game_day.fetch_game_day_weather)
    weather_lookup = _load_weather_by_date(game_dates) if game_dates else {}

    for game in week_schedule:
        home_full = game.get("home_name", "")
        away_full = game.get("away_name", "")
        home_abbrev = _resolve_team_abbrev(home_full)
        away_abbrev = _resolve_team_abbrev(away_full)
        game_date = game.get("game_date", "")
        home_pitcher = game.get("home_probable_pitcher", "")
        away_pitcher = game.get("away_probable_pitcher", "")

        # Look up weather for the venue (home team's park)
        weather_key = f"{game_date}:{home_abbrev}"
        weather = weather_lookup.get(weather_key, {})

        # Build shared weather info dict for both teams in this game
        game_weather = {
            "temp_f": weather.get("temp_f"),
            "wind_mph": weather.get("wind_mph"),
            "wind_dir": weather.get("wind_dir"),
            "precip_pct": weather.get("precip_pct"),
        }

        if home_abbrev:
            team_games.setdefault(home_abbrev, []).append(
                {
                    "game_date": game_date,
                    "opponent": away_abbrev,
                    "is_home": True,
                    "park_team": home_abbrev,
                    "opposing_pitcher": away_pitcher,
                    **game_weather,
                }
            )

        if away_abbrev:
            team_games.setdefault(away_abbrev, []).append(
                {
                    "game_date": game_date,
                    "opponent": home_abbrev,
                    "is_home": False,
                    "park_team": home_abbrev,
                    "opposing_pitcher": home_pitcher,
                    **game_weather,
                }
            )

    return team_games


def compute_weekly_matchup_adjustments(
    roster: pd.DataFrame,
    week_schedule: list[dict[str, Any]],
    park_factors: dict[str, float],
    enable_platoon: bool = True,
    enable_park: bool = True,
    enable_weather: bool = True,
    enable_opposing_pitcher: bool = True,
) -> pd.DataFrame:
    """Apply weekly matchup adjustments to a roster's projections.

    For each player on the roster, looks up their team's games this
    week and applies multiplicative adjustments to counting stats.

    Adjustments applied:
      - Park factors (per-game venue)
      - Platoon splits (if handedness data available)
      - Weather (HR only, if temp data available)
      - Opposing pitcher quality (via matchup_adjustment_factor)

    The final adjustment for each stat is the product of the average
    per-game adjustment across the week's games.

    Args:
        roster: DataFrame with columns: name/player_name, team,
            is_hitter, and stat columns (r, hr, rbi, sb, w, sv, k).
        week_schedule: Schedule data from ``get_weekly_schedule()``.
        park_factors: Team abbreviation -> park factor dict.
        enable_platoon: Whether to apply platoon adjustments.
        enable_park: Whether to apply park factor adjustments.
        enable_weather: Whether to apply weather HR adjustments.
        enable_opposing_pitcher: Whether to apply opposing pitcher adjustments.

    Returns:
        Copy of roster with adjusted stats and a new boolean column
        ``matchup_adjusted`` indicating which rows were modified.
    """
    if roster.empty or not week_schedule:
        result = roster.copy()
        result["matchup_adjusted"] = False
        return result

    # Build per-team schedule lookup
    team_schedule = _build_team_schedule(week_schedule)

    result = roster.copy()
    result["matchup_adjusted"] = False

    for idx, row in result.iterrows():
        team = str(row.get("team", "")).upper().strip()
        is_hitter = bool(row.get("is_hitter", True))
        games = team_schedule.get(team, [])

        if not games:
            continue

        # Accumulate per-game adjustment factors
        n_games = len(games)
        stat_adjustments: dict[str, float] = {}
        counting_stats = _HITTER_COUNTING_STATS if is_hitter else _PITCHER_COUNTING_STATS

        for stat in counting_stats:
            total_factor = 0.0
            for game in games:
                factor = 1.0

                # Park factor
                if enable_park:
                    park_team = game.get("park_team", "")
                    pf = park_factors.get(park_team, 1.0)
                    if is_hitter:
                        factor *= pf
                    # Pitchers: park factor is neutral for counting stats

                # Weather adjustments (when data is present)
                if enable_weather:
                    temp = game.get("temp_f")
                    precip = game.get("precip_pct")
                    wind_mph_val = game.get("wind_mph")
                    wind_dir_val = game.get("wind_dir")

                    # Temperature HR adjustment (hitters only)
                    if stat == "hr" and is_hitter and temp is not None:
                        factor *= weather_hr_adjustment(float(temp))

                    # E6: Wind HR adjustment using stadium outfield bearing
                    if stat == "hr" and is_hitter and wind_mph_val is not None:
                        wind_blowing_out = False
                        if isinstance(wind_dir_val, (int, float)):
                            # Numeric wind direction (degrees) — use T11 bearings
                            try:
                                from src.game_day import DOME_TEAMS, OUTFIELD_BEARING

                                _park = game.get("park_team", "")
                                if _park not in DOME_TEAMS and _park in OUTFIELD_BEARING:
                                    wind_blowing_out = is_wind_blowing_out(
                                        float(wind_dir_val), OUTFIELD_BEARING[_park]
                                    )
                            except Exception:
                                pass
                        elif isinstance(wind_dir_val, str) and "out" in wind_dir_val.lower():
                            # Fallback: string-based "blowing out" from API
                            wind_blowing_out = True
                        factor *= weather_wind_hr_adjustment(
                            float(wind_mph_val), wind_out=wind_blowing_out
                        )

                    # Rain K/BB adjustment (pitchers: K stat)
                    if stat == "k" and not is_hitter and precip is not None:
                        rain_adj = weather_rain_adjustment(float(precip))
                        factor *= rain_adj["k_mult"]

                # Platoon adjustment (when handedness data is present)
                if enable_platoon and is_hitter:
                    batter_hand = str(row.get("bats", "")).strip()
                    pitcher_hand = game.get("opposing_pitcher_hand", "")
                    if batter_hand and pitcher_hand:
                        factor *= platoon_adjustment(batter_hand, pitcher_hand)

                factor = float(factor)
                total_factor += factor

            # Average adjustment across games
            avg_factor = total_factor / n_games if n_games > 0 else 1.0
            stat_adjustments[stat] = avg_factor

        # Apply adjustments to counting stats
        any_adjusted = False
        for stat, adj in stat_adjustments.items():
            if stat in result.columns and abs(adj - 1.0) > 1e-9:
                current_val = pd.to_numeric(result.at[idx, stat], errors="coerce")
                if pd.notna(current_val):
                    result.at[idx, stat] = current_val * adj
                    any_adjusted = True

        if any_adjusted:
            result.at[idx, "matchup_adjusted"] = True

    return result

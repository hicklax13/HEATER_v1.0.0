"""Daily Category Value (DCV) -- per-player, per-category, per-day optimization.

Computes how much each player is expected to contribute to each scoring
category TODAY, accounting for projections, matchup, availability, and
H2H urgency. Feeds into the LP solver for optimal lineup assignment.

Part of the Lineup Optimizer V2 (pipeline stages 10-12).
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Stabilization points for Bayesian blend (from FanGraphs research)
STABILIZATION_POINTS: dict[str, float] = {
    "r": 460,
    "hr": 170,
    "rbi": 300,
    "sb": 200,
    "avg": 910,
    "obp": 460,
    "w": 200,
    "l": 200,
    "sv": 100,
    "k": 70,
    "era": 630,
    "whip": 540,
}

# Top N players by ROS projection that get stud floor protection
STUD_FLOOR_TOP_N = 8


# ---------------------------------------------------------------------------
# Function 1: Bayesian blended projection
# ---------------------------------------------------------------------------


def compute_blended_projection(
    preseason_rate: float,
    observed_numerator: float,
    observed_denominator: float,
    stat_key: str,
) -> float:
    """Bayesian blend: (preseason * stab + observed) / (stab + denom).

    Combines a preseason prior with observed season data, weighted by the
    stat-specific stabilization point. Early in the season the prior
    dominates; as denominator grows the blend shifts toward observed.

    Args:
        preseason_rate: Preseason projection rate for the stat.
        observed_numerator: Observed stat total so far this season.
        observed_denominator: Denominator (PA for hitters, IP for pitchers).
        stat_key: Lowercase stat key (e.g. "hr", "era").

    Returns:
        Blended projection rate.
    """
    stab = STABILIZATION_POINTS.get(stat_key, 200)
    if stab <= 0:
        stab = 200
    prior = preseason_rate * stab
    total = stab + observed_denominator
    if total <= 0:
        return preseason_rate
    return (prior + observed_numerator) / total


# ---------------------------------------------------------------------------
# Function 2: Health factor
# ---------------------------------------------------------------------------


def compute_health_factor(status: str) -> float:
    """Return health factor. IL/DTD/NA = 0.0 (excluded). Active = 1.0.

    All injured or inactive statuses result in full exclusion (0.0).
    Only truly active players contribute to DCV.

    Args:
        status: Player roster status string (e.g. "active", "IL15", "DTD").

    Returns:
        1.0 for active players, 0.0 for all others.
    """
    if not status:
        return 1.0
    s = str(status).lower().strip()
    # ALL injured/inactive statuses = EXCLUDED
    if s in (
        "il10",
        "il15",
        "il60",
        "dl",
        "dtd",
        "day-to-day",
        "na",
        "not active",
        "minors",
        "out",
        "suspended",
    ):
        return 0.0
    return 1.0


# ---------------------------------------------------------------------------
# Function 3: Volume factor
# ---------------------------------------------------------------------------


def compute_volume_factor(
    team_playing_today: bool,
    in_confirmed_lineup: bool | None,
    is_doubleheader: bool = False,
) -> float:
    """Volume factor based on game availability.

    Maps the player's game-day situation to a production multiplier:
    off days yield zero, confirmed starters yield full value, and
    doubleheaders double the output.

    Args:
        team_playing_today: Whether the player's team has a game today.
        in_confirmed_lineup: True = confirmed starter, False = benched,
            None = lineup not yet posted.
        is_doubleheader: Whether the team has a doubleheader today.

    Returns:
        0.0 = off day (zero production)
        0.3 = team plays but player benched
        0.9 = team plays, lineup not yet posted
        1.0 = confirmed in starting lineup
        2.0 = confirmed in doubleheader starting lineup
    """
    if not team_playing_today:
        return 0.0
    if in_confirmed_lineup is None:
        # Lineup not posted yet
        return 1.8 if is_doubleheader else 0.9
    if not in_confirmed_lineup:
        return 0.3  # Bench/pinch-hit only
    if is_doubleheader:
        return 2.0
    return 1.0


# ---------------------------------------------------------------------------
# Function 4: Matchup multiplier
# ---------------------------------------------------------------------------


def compute_matchup_multiplier(
    is_hitter: bool,
    batter_hand: str,
    pitcher_hand: str,
    player_team: str,
    opponent_team: str,
    park_factors: dict,
    pitcher_xfip: float | None = None,
    temp_f: float | None = None,
) -> float:
    """Compute combined matchup multiplier for counting stat adjustment.

    Combines platoon advantage, park factor, opposing pitcher quality, and
    game-time weather into a single multiplicative factor.  For rate stats,
    this multiplier should be applied to COMPONENTS (H, AB, ER, IP), not to
    the rate itself.

    Args:
        is_hitter: True for position players, False for pitchers.
        batter_hand: "L" or "R" for batter handedness.
        pitcher_hand: "L" or "R" for opposing pitcher handedness.
        player_team: 3-letter team abbreviation of the player.
        opponent_team: 3-letter team abbreviation of the opponent.
        park_factors: Dict mapping team abbreviation to park factor.
        pitcher_xfip: Opposing pitcher's xFIP (None if unavailable).
        temp_f: Game-time temperature in Fahrenheit (None if unavailable).

    Returns:
        Multiplicative adjustment clamped to [0.3, 3.0].
    """
    mult = 1.0

    # Platoon adjustment
    try:
        from src.optimizer.matchup_adjustments import platoon_adjustment

        plat = platoon_adjustment(batter_hand, pitcher_hand, None, None, 0)
        # platoon_adjustment returns a multiplicative factor
        if plat and abs(plat) > 0:
            mult *= max(0.8, min(1.2, plat))
    except (ImportError, Exception):
        pass

    # Park factor
    try:
        from src.optimizer.matchup_adjustments import park_factor_adjustment

        pf = park_factor_adjustment(player_team, opponent_team, park_factors, is_hitter)
        if pf and pf > 0:
            mult *= pf
    except (ImportError, Exception):
        pass

    # Opposing pitcher quality (xFIP-based)
    if pitcher_xfip is not None and is_hitter:
        # League avg xFIP ~4.20. Better pitcher = lower multiplier for hitter
        quality = max(0.5, min(2.0, 2.0 - pitcher_xfip / 4.20))
        # Invert for hitters: good pitcher hurts hitter value
        mult *= 1.0 / max(0.5, quality)

    # Weather adjustment — hot temps boost HR/power production
    if temp_f is not None and is_hitter:
        try:
            from src.optimizer.matchup_adjustments import weather_hr_adjustment

            weather_mult = weather_hr_adjustment(temp_f)
            if weather_mult and weather_mult > 0:
                mult *= weather_mult
        except (ImportError, Exception):
            pass

    return max(0.3, min(3.0, mult))  # Clamp to reasonable range


# ---------------------------------------------------------------------------
# Function 5: Stud floor protection
# ---------------------------------------------------------------------------


def apply_stud_floor(
    dcv_table: pd.DataFrame,
    roster: pd.DataFrame,
    config=None,
) -> pd.DataFrame:
    """Prevent benching elite players by setting a DCV floor.

    Players in the top STUD_FLOOR_TOP_N by total ROS projection get
    a minimum DCV that ensures they are never benched (except on off days
    or when excluded by health factor = 0).

    Args:
        dcv_table: DataFrame with columns total_dcv, player_id,
            volume_factor, etc.
        roster: Player pool DataFrame with projection columns.
        config: LeagueConfig instance. Uses default if None.

    Returns:
        Updated dcv_table with stud floor applied where needed.
    """
    if config is None:
        from src.valuation import LeagueConfig

        config = LeagueConfig()

    # Compute total SGP per player from ROS projections
    total_sgp: dict[int, float] = {}
    for _, row in roster.iterrows():
        pid = row.get("player_id")
        sgp = 0.0
        for cat in config.all_categories:
            val = float(row.get(cat.lower(), 0) or 0)
            denom = config.sgp_denominators.get(cat, 1.0)
            if abs(denom) > 1e-9:
                if cat in config.inverse_stats:
                    sgp -= val / denom
                else:
                    sgp += val / denom
        total_sgp[pid] = sgp

    # Find the threshold for top N (use fraction of roster when roster is small)
    sorted_sgps = sorted(total_sgp.values(), reverse=True)
    stud_count = min(STUD_FLOOR_TOP_N, max(1, len(sorted_sgps) // 3))
    if len(sorted_sgps) >= stud_count:
        threshold = sorted_sgps[stud_count - 1]
    else:
        threshold = sorted_sgps[-1] if sorted_sgps else 0

    # Apply floor: stud players get minimum DCV that keeps them starting
    # Only applies when volume_factor > 0 (playing today)
    if "total_dcv" in dcv_table.columns and "player_id" in dcv_table.columns:
        if "volume_factor" in dcv_table.columns:
            active_dcv = dcv_table.loc[dcv_table["volume_factor"] > 0, "total_dcv"]
        else:
            active_dcv = dcv_table.loc[dcv_table["total_dcv"] > 0, "total_dcv"]
        median_dcv = active_dcv.median() if not active_dcv.empty else 1.0
        if median_dcv <= 0:
            median_dcv = 1.0
        for idx, row in dcv_table.iterrows():
            pid = row.get("player_id")
            vol = row.get("volume_factor", 0)
            if pid in total_sgp and total_sgp[pid] >= threshold and vol > 0:
                current = row.get("total_dcv", 0)
                floor_val = median_dcv * 1.5  # Studs get 1.5x median as floor
                if current < floor_val:
                    dcv_table.at[idx, "total_dcv"] = floor_val
                    dcv_table.at[idx, "stud_floor_applied"] = True

    return dcv_table


# ---------------------------------------------------------------------------
# Function 6: Master DCV table builder
# ---------------------------------------------------------------------------


def build_daily_dcv_table(
    roster: pd.DataFrame,
    matchup: dict | None,
    schedule_today: list[dict] | None,
    park_factors: dict | None,
    config=None,
    urgency_weights: dict | None = None,
    confirmed_lineups: dict[str, list] | None = None,
    recent_form: dict[int, dict] | None = None,
    rate_modes: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build the Daily Category Value table for all roster players.

    This is the master function for Stage 10 of the optimizer pipeline.
    It combines blended projections, matchup multipliers, health status,
    volume (game schedule), and H2H urgency into a single DCV per player
    per category, then applies stud floor protection.

    Args:
        roster: Player pool DataFrame with projection columns.
        matchup: Yahoo matchup dict from yds.get_matchup().
        schedule_today: List of game dicts from statsapi for today.
        park_factors: Dict of team -> park factor.
        config: LeagueConfig.
        urgency_weights: Output of compute_urgency_weights(). When provided,
            each dcv_{cat} is multiplied by urgency_weights["urgency"][cat]
            AFTER initial DCV computation, then total_dcv is recomputed.
            When None, the function computes urgency internally (existing behavior).
        confirmed_lineups: Dict mapping team abbreviation to list of player
            names confirmed in today's starting lineup. Used for volume factor
            lookup. When None, all lineups treated as not-yet-posted (0.9 default).
        recent_form: Dict mapping player_id to form data dict. Expected
            structure: {player_id: {"l14": {"avg": .., "obp": .., "era": ..,
            "whip": .., "games": ..}}}. When provided, blends L14 form at 25%
            weight with preseason projections (clamped +/-20%). When None, no
            recent form adjustment is applied.
        rate_modes: Dict mapping rate stat category name to mode string
            (e.g., {"ERA": "abandon", "WHIP": "abandon"}). When a rate stat
            is in "abandon" mode, pitcher DCV for that category is zeroed out
            so the optimizer focuses on flippable categories (W, K, SV).
            When None, all categories contribute normally (existing behavior).

    Returns:
        DataFrame with columns: player_id, name, positions,
        volume_factor, health_factor, matchup_mult,
        dcv_{category} for each scoring category, total_dcv,
        stud_floor_applied.
    """
    if config is None:
        from src.valuation import LeagueConfig

        config = LeagueConfig()

    if park_factors is None:
        park_factors = {}

    # Get roster statuses for health factor
    try:
        from src.trade_intelligence import _load_roster_statuses

        statuses = _load_roster_statuses()
    except Exception:
        statuses = {}

    # Determine which teams play today
    # Schedule may use full names ("CHICAGO CUBS") or abbreviations ("CHC")
    # Normalize both to abbreviations for matching against roster team column
    _FULL_TO_ABBR: dict[str, str] = {
        "ATHLETICS": "ATH",
        "ATLANTA BRAVES": "ATL",
        "BALTIMORE ORIOLES": "BAL",
        "BOSTON RED SOX": "BOS",
        "CHICAGO CUBS": "CHC",
        "CHICAGO WHITE SOX": "CWS",
        "CINCINNATI REDS": "CIN",
        "CLEVELAND GUARDIANS": "CLE",
        "COLORADO ROCKIES": "COL",
        "DETROIT TIGERS": "DET",
        "HOUSTON ASTROS": "HOU",
        "KANSAS CITY ROYALS": "KC",
        "LOS ANGELES ANGELS": "LAA",
        "LOS ANGELES DODGERS": "LAD",
        "MIAMI MARLINS": "MIA",
        "MILWAUKEE BREWERS": "MIL",
        "MINNESOTA TWINS": "MIN",
        "NEW YORK METS": "NYM",
        "NEW YORK YANKEES": "NYY",
        "OAKLAND ATHLETICS": "ATH",
        "PHILADELPHIA PHILLIES": "PHI",
        "PITTSBURGH PIRATES": "PIT",
        "SAN DIEGO PADRES": "SD",
        "SAN FRANCISCO GIANTS": "SF",
        "SEATTLE MARINERS": "SEA",
        "ST. LOUIS CARDINALS": "STL",
        "TAMPA BAY RAYS": "TB",
        "TEXAS RANGERS": "TEX",
        "TORONTO BLUE JAYS": "TOR",
        "WASHINGTON NATIONALS": "WSH",
        "ARIZONA DIAMONDBACKS": "ARI",
    }
    teams_playing: set[str] = set()
    if schedule_today:
        for game in schedule_today:
            for key in ("away_name", "away_team", "home_name", "home_team"):
                raw = str(game.get(key, "")).upper().strip()
                if raw:
                    # Try mapping full name to abbreviation
                    abbr = _FULL_TO_ABBR.get(raw, raw)
                    teams_playing.add(abbr)
                    # Also add the raw value in case it's already an abbreviation
                    teams_playing.add(raw)

    # Get urgency weights from matchup (use caller-provided if available)
    if urgency_weights is not None:
        urgency = urgency_weights.get("urgency", {})
        _external_urgency = True
    else:
        _external_urgency = False
        try:
            from src.optimizer.category_urgency import compute_urgency_weights as _cuw

            urgency_result = _cuw(matchup, config)
            urgency = urgency_result.get("urgency", {})
        except Exception:
            logger.error("Category urgency computation failed — using equal weights", exc_info=True)
            urgency = {cat: 0.5 for cat in config.all_categories}

    # Load weather data for today -- build team -> temp_f lookup
    # Both home and away teams at a given venue experience the same weather
    weather_by_team: dict[str, float] = {}
    try:
        from datetime import UTC as _utc
        from datetime import datetime as _dt

        from src.database import load_game_day_weather

        today_date = _dt.now(_utc).strftime("%Y-%m-%d")
        weather_df = load_game_day_weather(today_date)
        if not weather_df.empty:
            # Build venue_team -> temp_f mapping, then map both home and
            # away teams at that venue to the same temperature
            for _, wrow in weather_df.iterrows():
                venue = str(wrow.get("venue_team", "")).upper()
                temp = wrow.get("temp_f")
                if venue and temp is not None:
                    weather_by_team[venue] = float(temp)
            # Also map away teams to the home venue's weather
            if schedule_today:
                for game in schedule_today:
                    home_raw = str(game.get("home_name", "")).upper().strip()
                    away_raw = str(game.get("away_name", "")).upper().strip()
                    home_abbr = _FULL_TO_ABBR.get(home_raw, home_raw)
                    away_abbr = _FULL_TO_ABBR.get(away_raw, away_raw)
                    if home_abbr in weather_by_team and away_abbr:
                        weather_by_team[away_abbr] = weather_by_team[home_abbr]
    except Exception:
        logger.debug("Could not load weather for DCV table", exc_info=True)

    # Build set of today's probable starting pitchers from schedule
    probable_starters: set[str] = set()
    if schedule_today:
        for game in schedule_today:
            for pp_key in ("home_probable_pitcher", "away_probable_pitcher"):
                pp_name = str(game.get(pp_key, "") or "").strip()
                if pp_name and pp_name.upper() not in ("TBD", ""):
                    probable_starters.add(pp_name)

    rows: list[dict] = []
    for _, player in roster.iterrows():
        pid = player.get("player_id")
        name = str(player.get("name", player.get("player_name", "")))
        positions = str(player.get("positions", ""))
        team = str(player.get("team", "")).upper()
        is_hitter = bool(player.get("is_hitter", 1))

        # Health factor
        status = statuses.get(pid, str(player.get("status", "active")))
        health = compute_health_factor(status)

        # Volume factor
        team_plays = team in teams_playing if teams_playing else True
        in_lineup = None  # default: lineup not posted
        lineup_slot = 0  # 0 = unknown batting order slot
        if confirmed_lineups is not None:
            is_hitter = bool(int(player.get("is_hitter", 1)))
            if is_hitter and team in confirmed_lineups:
                # Batting lineups only meaningful for hitters
                team_lineup = confirmed_lineups[team]
                in_lineup = name in team_lineup
                if in_lineup:
                    # Batting order slot is 1-based index in the lineup list
                    try:
                        lineup_slot = team_lineup.index(name) + 1
                    except ValueError:
                        lineup_slot = 0
        # Pitcher probable-starter check (independent of confirmed_lineups —
        # probable pitcher data comes from schedule_today, not batting lineups)
        if not is_hitter and probable_starters:
            pos_upper = positions.upper()
            if "SP" in pos_upper:
                in_lineup = name in probable_starters
            # Pure RP keeps in_lineup=None → volume=0.9 (any reliever can enter)
        volume = compute_volume_factor(team_plays, in_lineup)

        # Skip entirely if excluded
        if health == 0.0 or volume == 0.0:
            row_data: dict = {
                "player_id": pid,
                "name": name,
                "positions": positions,
                "team": team,
                "is_hitter": is_hitter,
                "health_factor": health,
                "volume_factor": volume,
                "matchup_mult": 0.0,
                "total_dcv": 0.0,
                "stud_floor_applied": False,
                "status": status,
            }
            for cat in config.all_categories:
                row_data[f"dcv_{cat.lower()}"] = 0.0
            rows.append(row_data)
            continue

        # Matchup multiplier — uses handedness, park factors, weather
        player_temp = weather_by_team.get(team)
        _batter_hand = str(player.get("bats", "") or "")
        _pitcher_hand = ""
        # Look up opposing pitcher handedness from schedule if available
        if is_hitter and schedule_today:
            for _sg in schedule_today:
                _home = str(_sg.get("home_name", ""))
                _away = str(_sg.get("away_name", ""))
                if team in (_home, _away, str(_sg.get("home_short", "")), str(_sg.get("away_short", ""))):
                    _pp_key = (
                        "away_probable_pitcher"
                        if team == _home or team == str(_sg.get("home_short", ""))
                        else "home_probable_pitcher"
                    )
                    # Probable pitcher throws data not in schedule — use roster pool
                    break
        matchup_mult = compute_matchup_multiplier(
            is_hitter=is_hitter,
            batter_hand=_batter_hand,
            pitcher_hand=_pitcher_hand,
            player_team=team,
            opponent_team="",
            park_factors=park_factors,
            temp_f=player_temp,
        )

        # Recent form blending: adjust projections with L14 data
        form_adjustments: dict[str, float] = {}
        if recent_form is not None:
            form = recent_form.get(pid, {}).get("l14", {})
            form_games = int(form.get("games", 0) or 0)
            if form_games >= 7:
                # Dynamic form weight: scales from 0.10 (7 games) to 0.25 (14+ games)
                _form_weight = min(0.25, 0.10 + (form_games - 7) * 0.02)
                _base_weight = 1.0 - _form_weight
                if is_hitter:
                    # Rate stats: blend directly
                    for stat_key in ("avg", "obp"):
                        form_val = form.get(stat_key)
                        if form_val is not None:
                            orig = float(player.get(stat_key, 0) or 0)
                            if orig > 0:
                                blended = _base_weight * orig + _form_weight * float(form_val)
                                lo = orig * 0.80
                                hi = orig * 1.20
                                form_adjustments[stat_key] = max(lo, min(hi, blended))
                    # Counting stats: use rate-ratio from L14 per-game vs projected per-game
                    for stat_key in ("r", "hr", "rbi", "sb"):
                        form_val = form.get(stat_key)
                        if form_val is not None and form_games > 0:
                            orig = float(player.get(stat_key, 0) or 0)
                            if orig > 0:
                                # L14 per-game rate vs projected per-162 per-game rate
                                form_per_game = float(form_val) / form_games
                                proj_per_game = orig / 162.0
                                if proj_per_game > 0:
                                    ratio = form_per_game / proj_per_game
                                    adj = _base_weight * 1.0 + _form_weight * ratio
                                    adj = max(0.80, min(1.20, adj))
                                    form_adjustments[stat_key] = orig * adj
                else:
                    # Pitcher rate stats: blend directly
                    for stat_key in ("era", "whip"):
                        form_val = form.get(stat_key)
                        if form_val is not None:
                            orig = float(player.get(stat_key, 0) or 0)
                            if orig > 0:
                                blended = _base_weight * orig + _form_weight * float(form_val)
                                lo = orig * 0.80
                                hi = orig * 1.20
                                form_adjustments[stat_key] = max(lo, min(hi, blended))
                    # Pitcher counting stats: K, W, SV
                    for stat_key in ("k", "w", "sv"):
                        form_val = form.get(stat_key)
                        if form_val is not None and form_games > 0:
                            orig = float(player.get(stat_key, 0) or 0)
                            if orig > 0:
                                form_per_game = float(form_val) / form_games
                                proj_per_game = orig / 162.0
                                if proj_per_game > 0:
                                    ratio = form_per_game / proj_per_game
                                    adj = _base_weight * 1.0 + _form_weight * ratio
                                    adj = max(0.80, min(1.20, adj))
                                    form_adjustments[stat_key] = orig * adj

        # Compute DCV per category
        total_dcv = 0.0
        row_data = {
            "player_id": pid,
            "name": name,
            "positions": positions,
            "team": team,
            "is_hitter": is_hitter,
            "health_factor": health,
            "volume_factor": volume,
            "matchup_mult": round(matchup_mult, 3),
            "stud_floor_applied": False,
            "status": status,
        }

        # Batting order PA multiplier for confirmed hitters
        pa_mult = 1.0
        if lineup_slot >= 1 and is_hitter:
            try:
                from src.contextual_factors import batting_order_pa_multiplier

                pa_mult = batting_order_pa_multiplier(lineup_slot)
            except Exception:
                pa_mult = 1.0

        for cat in config.all_categories:
            col = cat.lower()
            proj_val = form_adjustments.get(col, float(player.get(col, 0) or 0))

            # Per-game rate: divide counting stats by ~162 games.
            # Rate stats (AVG, OBP, ERA, WHIP) are already per-game rates.
            if cat in config.rate_stats:
                daily_proj = proj_val
            else:
                daily_proj = proj_val / 162.0

            # Zero out abandoned rate stats for pitchers so DCV focuses
            # on flippable categories (W, K, SV) instead of unrecoverable ones.
            if rate_modes and not is_hitter and rate_modes.get(cat) == "abandon":
                daily_proj = 0.0

            # Apply batting order PA adjustment to counting stats only
            if pa_mult != 1.0 and cat in config.counting_stats:
                daily_proj *= pa_mult

            # Apply factors
            dcv = daily_proj * matchup_mult * health * volume

            # Weight by urgency (skip if external urgency -- applied post-hoc)
            if not _external_urgency:
                cat_urgency = urgency.get(cat, 0.5)
                dcv = dcv * cat_urgency

            # SGP normalization
            weighted_dcv = dcv
            denom = config.sgp_denominators.get(cat, 1.0)
            if abs(denom) > 1e-9:
                if cat in config.inverse_stats:
                    sgp_dcv = -weighted_dcv / denom
                else:
                    sgp_dcv = weighted_dcv / denom
            else:
                sgp_dcv = 0.0

            # T3-5a: Stuff+ boost for pitcher K DCV
            if col == "k" and not is_hitter:
                try:
                    _stuff = float(player.get("stuff_plus", 0) or 0)
                    if _stuff > 120:
                        sgp_dcv *= 1.10
                    elif _stuff > 110:
                        sgp_dcv *= 1.05
                except (TypeError, ValueError):
                    pass

            # T3-5b: Sprint speed boost for hitter SB DCV
            if col == "sb" and is_hitter:
                try:
                    _speed = float(player.get("sprint_speed", 0) or 0)
                    if _speed > 29.0:
                        sgp_dcv *= 1.10
                    elif _speed > 28.0:
                        sgp_dcv *= 1.05
                except (TypeError, ValueError):
                    pass

            row_data[f"dcv_{col}"] = round(sgp_dcv, 4)
            total_dcv += sgp_dcv

        row_data["total_dcv"] = round(total_dcv, 4)
        rows.append(row_data)

    dcv_df = pd.DataFrame(rows) if rows else pd.DataFrame()

    # Post-hoc urgency weighting when caller provides urgency_weights
    if _external_urgency and not dcv_df.empty:
        for cat in config.all_categories:
            col = f"dcv_{cat.lower()}"
            if col in dcv_df.columns:
                cat_urg = urgency.get(cat, 1.0)
                dcv_df[col] = dcv_df[col] * cat_urg
        # Recompute total_dcv as sum of weighted dcv columns
        dcv_cols = [f"dcv_{c.lower()}" for c in config.all_categories if f"dcv_{c.lower()}" in dcv_df.columns]
        if dcv_cols:
            dcv_df["total_dcv"] = dcv_df[dcv_cols].sum(axis=1).round(4)

    # Apply stud floor
    if not dcv_df.empty:
        dcv_df = apply_stud_floor(dcv_df, roster, config)

    # Sort by total DCV descending
    if not dcv_df.empty and "total_dcv" in dcv_df.columns:
        dcv_df = dcv_df.sort_values("total_dcv", ascending=False).reset_index(drop=True)

    return dcv_df


# ---------------------------------------------------------------------------
# IP minimum override
# ---------------------------------------------------------------------------


def check_ip_override(
    dcv_table: pd.DataFrame,
    weekly_ip_projected: float,
    ip_minimum: float = 20.0,
) -> pd.DataFrame:
    """Force-start a pitcher if weekly IP is below minimum threshold.

    If projected IP is below the minimum, find the best available pitcher
    and boost their DCV to ensure they start. This enforces the
    requirement for minimum weekly innings pitched.

    Args:
        dcv_table: DCV DataFrame from build_daily_dcv_table().
        weekly_ip_projected: Current projected IP for the week.
        ip_minimum: Minimum IP threshold (default 20.0).

    Returns:
        Updated dcv_table with pitcher boost applied if needed.
    """
    if weekly_ip_projected >= ip_minimum:
        return dcv_table

    if dcv_table.empty:
        return dcv_table

    # Find pitchers with volume > 0 (playing today)
    pitchers = dcv_table[
        (~dcv_table["is_hitter"]) & (dcv_table["volume_factor"] > 0) & (dcv_table["health_factor"] > 0)
    ]

    if pitchers.empty:
        return dcv_table

    # Boost the best pitcher's DCV to ensure they start
    best_pitcher_idx = pitchers["total_dcv"].idxmax()
    current_dcv = dcv_table.at[best_pitcher_idx, "total_dcv"]
    max_dcv = dcv_table["total_dcv"].max()
    dcv_table.at[best_pitcher_idx, "total_dcv"] = max(current_dcv, max_dcv * 1.5)

    logger.info(
        "IP override: boosted pitcher at index %d (%.1f IP projected, need %.1f)",
        best_pitcher_idx,
        weekly_ip_projected,
        ip_minimum,
    )

    return dcv_table


# ---------------------------------------------------------------------------
# T3-6: Per-pitcher IP pace constraint awareness
# ---------------------------------------------------------------------------


def apply_ip_pace_scaling(
    dcv_table: pd.DataFrame,
    weekly_ip_projected: float,
    weekly_ip_target: float = 55.0,
) -> pd.DataFrame:
    """Scale pitcher counting-stat DCV by remaining IP budget fraction.

    When a team has already used most of its weekly IP budget, pitcher
    counting stats (K/W/SV) should be scaled down to reflect reduced
    remaining innings. This prevents over-valuing pitchers late in the
    week when IP is tight.

    Args:
        dcv_table: DCV DataFrame from build_daily_dcv_table().
        weekly_ip_projected: Total projected IP already used/committed.
        weekly_ip_target: Season-wide weekly IP target (default 55.0).

    Returns:
        Updated dcv_table with pitcher DCV scaled by IP budget.
    """
    if dcv_table.empty:
        return dcv_table

    # Compute IP budget usage fraction
    ip_used_fraction = weekly_ip_projected / max(weekly_ip_target, 1.0)
    if ip_used_fraction < 0.80:
        # Plenty of budget remaining — no scaling needed
        return dcv_table

    if ip_used_fraction >= 0.95:
        # Near or over budget — heavily scale down pitcher counting stats
        scale = 0.3
    else:
        # 80-95% used — linear interpolation from 1.0 to 0.6
        scale = 1.0 - (ip_used_fraction - 0.80) * (0.4 / 0.15)
        scale = max(0.3, min(1.0, scale))

    # Apply scale to pitcher counting-stat DCV columns only (K, W, SV)
    if "is_hitter" not in dcv_table.columns:
        return dcv_table

    counting_cols = ["dcv_k", "dcv_w", "dcv_sv"]
    pitcher_mask = ~dcv_table["is_hitter"]
    for col in counting_cols:
        if col in dcv_table.columns:
            dcv_table.loc[pitcher_mask, col] = dcv_table.loc[pitcher_mask, col] * scale

    # Recompute total_dcv for affected pitchers
    all_dcv_cols = [c for c in dcv_table.columns if c.startswith("dcv_")]
    if all_dcv_cols:
        dcv_table.loc[pitcher_mask, "total_dcv"] = dcv_table.loc[pitcher_mask, all_dcv_cols].sum(axis=1).round(4)

    logger.info(
        "IP pace scaling: %.1f/%.1f IP used (%.0f%%), pitcher counting DCV scaled by %.2f",
        weekly_ip_projected,
        weekly_ip_target,
        ip_used_fraction * 100,
        scale,
    )

    return dcv_table

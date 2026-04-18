"""Daily Category Value (DCV) -- per-player, per-category, per-day optimization.

Computes how much each player is expected to contribute to each scoring
category TODAY, accounting for projections, matchup, availability, and
H2H urgency. Feeds into the LP solver for optimal lineup assignment.

Part of the Lineup Optimizer V2 (pipeline stages 10-12).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from datetime import UTC

import pandas as pd

logger = logging.getLogger(__name__)


def _normalize_pitcher_name(name: str) -> str:
    """Normalize a player name for robust matching across data sources.

    MLB Stats API probable-pitcher strings and roster names can differ by
    accents, punctuation, suffixes ("Jr.", "III"), and casing. Two calls that
    should match ("Chris Sale" vs "Chris Sale "; "José Ramírez" vs "Jose
    Ramirez") must normalize to the same key. Used by the SP probable-today
    gate to avoid silently missing legitimate probable starters.
    """
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", str(name))
    s = s.encode("ascii", "ignore").decode("ascii").lower().strip()
    for suffix in (" jr.", " jr", " sr.", " sr", " iii", " iv", " ii"):
        if s.endswith(suffix):
            s = s[: -len(suffix)].strip()
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return " ".join(s.split())


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
    opponent_offense_wrc_plus: float | None = None,
) -> float:
    """Compute combined matchup multiplier for counting stat adjustment.

    Combines platoon advantage, park factor, opposing pitcher/offense quality,
    and game-time weather into a single multiplicative factor. For pitchers,
    the opposing offense's wRC+ (relative to league-average 100) reduces or
    boosts expected production. For rate stats, this multiplier should be
    applied to COMPONENTS (H, AB, ER, IP), not to the rate itself.

    Args:
        is_hitter: True for position players, False for pitchers.
        batter_hand: "L" or "R" for batter handedness.
        pitcher_hand: "L" or "R" for opposing pitcher handedness.
        player_team: 3-letter team abbreviation of the player.
        opponent_team: 3-letter team abbreviation of the opponent.
        park_factors: Dict mapping team abbreviation to park factor.
        pitcher_xfip: Opposing pitcher's xFIP (None if unavailable, hitters only).
        temp_f: Game-time temperature in Fahrenheit (None if unavailable).
        opponent_offense_wrc_plus: Opposing team's wRC+ (~100 = league avg,
            higher = stronger offense → worse matchup for a pitcher). Only
            applied when is_hitter=False.

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

    # Opposing pitcher quality (xFIP-based) — hitters only
    if pitcher_xfip is not None and is_hitter:
        # League avg xFIP ~4.20. Better pitcher = lower multiplier for hitter
        quality = max(0.5, min(2.0, 2.0 - pitcher_xfip / 4.20))
        # Invert for hitters: good pitcher hurts hitter value
        mult *= 1.0 / max(0.5, quality)

    # Opposing team offense quality (wRC+-based) — pitchers only.
    # League-average wRC+ = 100. A pitcher facing a 120 wRC+ offense (NYY)
    # gets a dampened multiplier; facing a 80 wRC+ offense (bottom-third)
    # gets a boost. Scale: ~1.0 per 40 wRC+ points, clamped to [0.80, 1.20]
    # so no single matchup can more than 20% swing a pitcher's value.
    if opponent_offense_wrc_plus is not None and not is_hitter:
        try:
            _wrcp = float(opponent_offense_wrc_plus)
            # Inverse: 120 wRC+ → (100-120)/40 = -0.5 → ~0.95 multiplier
            # 80 wRC+ → (100-80)/40 = 0.5 → ~1.05 multiplier
            _off_mult = 1.0 + (100.0 - _wrcp) / 80.0
            mult *= max(0.80, min(1.20, _off_mult))
        except (TypeError, ValueError):
            pass

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

    # Apply floor: stud players get minimum DCV that keeps them starting.
    # Only applies when both volume_factor > 0 (team playing today) AND
    # health_factor > 0 (not IL/DTD/NA). Injured players must stay at 0 so
    # the LP correctly routes them to the IL slot.
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
            health = row.get("health_factor", 1.0)
            if pid in total_sgp and total_sgp[pid] >= threshold and vol > 0 and health > 0:
                current = row.get("total_dcv", 0)
                # Scale floor by matchup_mult so studs with different matchups
                # (park, platoon, opposing pitcher, weather) don't collapse to
                # identical DCVs. The floor exists to correct 1/162 daily-fraction
                # scaling artifacts, not to override legitimate matchup signal.
                try:
                    mm = float(row.get("matchup_mult", 1.0) or 1.0)
                except (TypeError, ValueError):
                    mm = 1.0
                floor_val = median_dcv * 1.5 * mm
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
    team_strength: dict[str, dict] | None = None,
    _retry_attempted: bool = False,
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
    # Equivalence classes: different data sources use different abbreviations
    # for the same team. Yahoo, MLB Stats API, and FanGraphs disagree.
    _TEAM_EQUIVALENCES = {
        "WSH": {"WSH", "WSN", "WAS"},
        "SF": {"SF", "SFG"},
        "SD": {"SD", "SDP"},
        "TB": {"TB", "TBR"},
        "KC": {"KC", "KCR"},
        "CWS": {"CWS", "CHW"},
        "ATH": {"ATH", "OAK"},
    }

    def _expand_equivalences(abbr: str) -> set[str]:
        for canon, variants in _TEAM_EQUIVALENCES.items():
            if abbr in variants or abbr == canon:
                return variants | {canon}
        return {abbr}

    teams_playing: set[str] = set()
    # Teams whose game has already started or finished today — players on
    # these teams are "locked" by Yahoo (you can't swap them anymore), so
    # their forward-looking DCV should be 0. This prevents the mid-day
    # optimizer view from suggesting actions on games already in progress.
    locked_teams: set[str] = set()
    _now_utc = None
    try:
        from datetime import datetime as _dt

        _now_utc = _dt.now(UTC)
    except Exception:
        _now_utc = None
    if schedule_today:
        for game in schedule_today:
            # Determine if this game is locked (started or finished)
            _game_locked = False
            _status_str = str(game.get("status", "")).lower()
            if any(s in _status_str for s in ("in progress", "final", "game over", "completed")):
                _game_locked = True
            elif _now_utc is not None:
                _ts = game.get("game_datetime") or game.get("game_date")
                if _ts:
                    try:
                        from datetime import datetime as _dt2

                        _game_time = _dt2.fromisoformat(str(_ts).replace("Z", "+00:00"))
                        if _game_time <= _now_utc:
                            _game_locked = True
                    except (ValueError, TypeError):
                        pass
            for key in ("away_name", "away_team", "home_name", "home_team"):
                raw = str(game.get(key, "")).upper().strip()
                if raw:
                    # Try mapping full name to abbreviation
                    abbr = _FULL_TO_ABBR.get(raw, raw)
                    teams_playing.add(abbr)
                    # Also add the raw value in case it's already an abbreviation
                    teams_playing.add(raw)
                    # Expand via equivalence map (WSN/WSH/WAS all play if one does)
                    teams_playing.update(_expand_equivalences(abbr))
                    if _game_locked:
                        locked_teams.add(abbr)
                        locked_teams.add(raw)
                        locked_teams.update(_expand_equivalences(abbr))

    # Get urgency weights from matchup (use caller-provided if available)
    if urgency_weights is not None:
        urgency = urgency_weights.get("urgency", {})
        _external_urgency = True
        # Guard: if ALL urgency values are 0 (or missing), post-hoc
        # multiplication would zero out every DCV score. Fall back to
        # internal equal-weight urgency instead.
        if urgency and all(abs(v) < 1e-9 for v in urgency.values()):
            logger.warning(
                "All external urgency weights are zero — falling back to "
                "internal equal-weight urgency (0.5) to prevent all-zero DCV"
            )
            urgency = {cat: 0.5 for cat in config.all_categories}
            _external_urgency = False
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
        # Pitcher probable-starter gating (independent of confirmed_lineups)
        # Pure SP not in today's probables → volume_factor forced to 0.0 (they
        # will NOT pitch today). SP/RP hybrids remain uncertain (0.9) because
        # they can still come out of the bullpen. Pure RPs stay at 0.9 baseline.
        # Name comparison uses normalization to handle accents/suffixes.
        _pitcher_volume_override: float | None = None
        if not is_hitter and team_plays:
            pos_upper = positions.upper()
            _has_sp = "SP" in pos_upper
            _has_rp = "RP" in pos_upper
            if _has_sp and probable_starters:
                _norm_name = _normalize_pitcher_name(name)
                _norm_probable = {_normalize_pitcher_name(p) for p in probable_starters}
                _is_probable_today = _norm_name in _norm_probable
                if _is_probable_today:
                    in_lineup = True
                elif not _has_rp:
                    # Pure SP confirmed NOT pitching today — zero out
                    in_lineup = False
                    _pitcher_volume_override = 0.0
                # SP/RP hybrid not probable → can still relieve, leave as None
            # probable_starters empty (API miss / all TBD) → keep default None
        volume = compute_volume_factor(team_plays, in_lineup)
        if _pitcher_volume_override is not None:
            volume = _pitcher_volume_override
        # Deduct already-played game contributions: if this player's team
        # has a game in progress or final today, Yahoo has locked the slot
        # and the player can no longer be swapped in/out. Forward-looking
        # DCV must be 0 — the optimizer shouldn't "recommend" someone whose
        # game is already started/done as if action is still possible.
        if team in locked_teams:
            volume = 0.0

        # Skip entirely if excluded
        if health == 0.0 or volume == 0.0:
            row_data: dict = {
                "player_id": pid,
                "name": name,
                "positions": positions,
                "team": team,
                "is_hitter": is_hitter,
                "game_locked": team in locked_teams,
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

        # Matchup multiplier — resolve opponent team, home/away, opposing
        # pitcher handedness + xFIP, and pass VENUE (home team code) to
        # park_factor_adjustment. Previously passed opponent_team="" which
        # silently fell back to player's home park for every player
        # (Moniak at COL always got 1.38 regardless of home/away).
        player_temp = weather_by_team.get(team)
        _batter_hand = str(player.get("bats", "") or "")
        _pitcher_hand = ""
        _opp_pitcher_name = ""
        _venue_team = ""  # home team code — whose park factor to use
        _opp_team = ""  # literal opponent team code — for opposing offense wRC+
        if schedule_today and team:
            _team_variants = _expand_equivalences(team)
            for _sg in schedule_today:
                _home_raw = str(_sg.get("home_name", "")).upper().strip()
                _away_raw = str(_sg.get("away_name", "")).upper().strip()
                _home_short = str(_sg.get("home_short", "")).upper().strip()
                _away_short = str(_sg.get("away_short", "")).upper().strip()
                _home_ab = _FULL_TO_ABBR.get(_home_raw, _home_raw) or _home_short
                _away_ab = _FULL_TO_ABBR.get(_away_raw, _away_raw) or _away_short
                _home_variants = _expand_equivalences(_home_ab) if _home_ab else set()
                _away_variants = _expand_equivalences(_away_ab) if _away_ab else set()
                if _team_variants & _home_variants or team == _home_short:
                    _venue_team = _home_ab  # player is home → venue is own park
                    _opp_team = _away_ab
                    _opp_pitcher_name = str(_sg.get("away_probable_pitcher", "") or "")
                    break
                if _team_variants & _away_variants or team == _away_short:
                    _venue_team = _home_ab  # player is away → venue is opponent park
                    _opp_team = _home_ab
                    _opp_pitcher_name = str(_sg.get("home_probable_pitcher", "") or "")
                    break

        _opp_pitcher_xfip: float | None = None
        if is_hitter and _opp_pitcher_name:
            _norm_opp = _normalize_pitcher_name(_opp_pitcher_name)
            _name_col = "player_name" if "player_name" in roster.columns else "name"
            if _name_col in roster.columns:
                _opp_match = roster[roster[_name_col].apply(lambda n: _normalize_pitcher_name(str(n)) == _norm_opp)]
                if not _opp_match.empty:
                    _opp_row = _opp_match.iloc[0]
                    _pitcher_hand = str(_opp_row.get("throws", "") or "")
                    for _xcol in ("xfip", "fip", "era"):
                        _xval = _opp_row.get(_xcol, None)
                        try:
                            if _xval is not None and not pd.isna(_xval):
                                _opp_pitcher_xfip = float(_xval)
                                break
                        except (TypeError, ValueError):
                            continue

        # Resolve opposing-team offensive strength (wRC+) for PITCHER matchup.
        # Without this a SP facing the Yankees gets the same multiplier as one
        # facing the Marlins. 100 = league-avg, higher = tougher for pitcher.
        _opp_wrc_plus: float | None = None
        if not is_hitter and team_strength and _opp_team:
            try:
                _ts_entry = None
                if _opp_team in team_strength:
                    _ts_entry = team_strength[_opp_team]
                else:
                    # Try equivalence classes (CWS/CHW, WSH/WSN/WAS, etc.)
                    for _variant in _expand_equivalences(_opp_team):
                        if _variant in team_strength:
                            _ts_entry = team_strength[_variant]
                            break
                if _ts_entry:
                    _wrcp_raw = _ts_entry.get("wrc_plus") or _ts_entry.get("wrcPlus")
                    if _wrcp_raw is not None and not pd.isna(_wrcp_raw):
                        _opp_wrc_plus = float(_wrcp_raw)
            except (TypeError, ValueError):
                _opp_wrc_plus = None

        matchup_mult = compute_matchup_multiplier(
            is_hitter=is_hitter,
            batter_hand=_batter_hand,
            pitcher_hand=_pitcher_hand,
            player_team=team,
            opponent_team=_venue_team,  # VENUE (home team) not literal opponent
            park_factors=park_factors or {},
            pitcher_xfip=_opp_pitcher_xfip,
            temp_f=player_temp,
            opponent_offense_wrc_plus=_opp_wrc_plus,
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
            "game_locked": False,
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

        # Volume-weighted SGP for rate stats.
        # Rate stats don't add: team_AVG != sum(player_AVGs). A player's
        # real contribution to team AVG is (player_hits - player_AB *
        # team_AVG) / team_AB. We use a replacement-level baseline so that
        # an average MLB starter scores ~0 and stars/scrubs deviate from
        # there. This is the only mathematically correct way to do daily
        # rate-stat DCV; raw-rate-divided-by-SGP-denom (the prior
        # implementation) gave structurally negative pitcher DCV.
        #
        # Replacement levels (12-team H2H mixed league, deep replacement):
        _REPL_AVG = 0.240
        _REPL_OBP = 0.305
        _REPL_ERA = 4.50
        _REPL_WHIP = 1.35
        # Raw-unit SGP denominators: how many raw units shift one standings
        # point. Derived from team-total denominators × team-volume:
        #   AVG: 0.004 × 5500 AB ≈ 22 hits
        #   OBP: 0.005 × 6100 PA ≈ 30 on-base events
        #   ERA: 0.20 × 1400 IP / 9 ≈ 31 ER
        #   WHIP: 0.020 × 1400 IP ≈ 28 walks+hits
        _RAW_SGP_DENOM = {"AVG": 22.0, "OBP": 30.0, "ERA": 31.0, "WHIP": 28.0}
        # Daily volume fraction (today's contribution / annual contribution).
        # Hitters: ~145 games out of 162 → 1/162. SP: ~30 starts → 1/30.
        # RP: ~50 appearances → 1/50.
        _hitter_daily_frac = 1.0 / 162.0
        _is_starter_pitcher = "SP" in str(player.get("positions", "")).upper()
        _pitcher_daily_frac = 1.0 / 30.0 if _is_starter_pitcher else 1.0 / 50.0

        for cat in config.all_categories:
            col = cat.lower()
            proj_val = form_adjustments.get(col, float(player.get(col, 0) or 0))

            # Per-game rate: divide counting stats by ~162 games.
            # Rate stats use volume-weighted SGP (computed below).
            if cat in config.rate_stats:
                # Compute volume-weighted SGP contribution directly,
                # bypassing the per-game daily_proj path.
                daily_proj = 0.0  # placeholder; sgp_dcv computed below
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
            if cat in config.rate_stats:
                # Volume-weighted SGP for rate stats. Compute the player's
                # annual SGP contribution from raw stat components, then
                # scale by today's volume fraction.
                _abandon = bool(rate_modes and not is_hitter and rate_modes.get(cat) == "abandon")
                if _abandon:
                    sgp_dcv = 0.0
                else:
                    annual_sgp = 0.0
                    if cat == "AVG" and is_hitter:
                        ab = float(player.get("ab", 0) or 0)
                        h = float(player.get("h", 0) or 0)
                        if ab > 0:
                            annual_sgp = (h - ab * _REPL_AVG) / _RAW_SGP_DENOM["AVG"]
                    elif cat == "OBP" and is_hitter:
                        ab = float(player.get("ab", 0) or 0)
                        h = float(player.get("h", 0) or 0)
                        bb = float(player.get("bb", 0) or 0)
                        hbp = float(player.get("hbp", 0) or 0)
                        sf = float(player.get("sf", 0) or 0)
                        denom_pa = ab + bb + hbp + sf
                        if denom_pa > 0:
                            annual_sgp = ((h + bb + hbp) - denom_pa * _REPL_OBP) / _RAW_SGP_DENOM["OBP"]
                    elif cat == "ERA" and not is_hitter:
                        ip = float(player.get("ip", 0) or 0)
                        er = float(player.get("er", 0) or 0)
                        if ip > 0:
                            repl_er = _REPL_ERA * ip / 9.0
                            annual_sgp = (repl_er - er) / _RAW_SGP_DENOM["ERA"]
                    elif cat == "WHIP" and not is_hitter:
                        ip = float(player.get("ip", 0) or 0)
                        bb_a = float(player.get("bb_allowed", 0) or 0)
                        h_a = float(player.get("h_allowed", 0) or 0)
                        if ip > 0:
                            repl_wh = _REPL_WHIP * ip
                            annual_sgp = (repl_wh - (bb_a + h_a)) / _RAW_SGP_DENOM["WHIP"]
                    # Apply daily fraction, matchup, health, volume, urgency
                    daily_frac = _hitter_daily_frac if is_hitter else _pitcher_daily_frac
                    sgp_dcv = annual_sgp * daily_frac * matchup_mult * health * volume
                    if not _external_urgency:
                        sgp_dcv *= urgency.get(cat, 0.5)
            elif abs(denom) > 1e-9:
                if cat in config.inverse_stats:
                    # Counting inverse stats (L): more is bad, so negate.
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

            row_data[f"dcv_{col}"] = round(sgp_dcv, 6)
            total_dcv += sgp_dcv

        # DCV scale: rate-stat math produces tiny per-category SGP values
        # (~0.001 per cat per day). Multiply by 1000 to express in
        # "milli-SGP per day" (mSGP) so display values are readable
        # integers/single decimals comparable across players. The LP
        # is scale-invariant, so this only affects presentation.
        for cat in config.all_categories:
            _k = f"dcv_{cat.lower()}"
            if _k in row_data:
                row_data[_k] = round(row_data[_k] * 1000.0, 4)
        row_data["total_dcv"] = round(total_dcv * 1000.0, 4)
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

    # Sanity check: if all active players have DCV=0 but the roster has
    # real stats, something went wrong in urgency/rate-mode weighting.
    # Retry without external urgency so results degrade to equal-weight
    # rather than all-zeros (which causes random START/BENCH decisions).
    if not _retry_attempted and not dcv_df.empty and "total_dcv" in dcv_df.columns:
        _active = dcv_df.loc[dcv_df.get("health_factor", pd.Series(1.0, index=dcv_df.index)) > 0]
        # Threshold: per-player per-day DCV is in mSGP units (~0.1-1.0 typical).
        # Sum across active players should be > 1.0 mSGP if anything is happening.
        if not _active.empty and _active["total_dcv"].abs().sum() < 1.0:
            # Check if the roster actually has stat data
            _stat_cols = [c for c in ["r", "hr", "rbi", "avg", "w", "k", "era"] if c in roster.columns]
            _has_stats = roster[_stat_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum().sum() > 0
            if _has_stats:
                logger.warning(
                    "All DCV scores are near-zero despite non-zero roster stats — retrying without external urgency weights"
                )
                return build_daily_dcv_table(
                    roster=roster,
                    matchup=matchup,
                    schedule_today=schedule_today,
                    park_factors=park_factors,
                    config=config,
                    urgency_weights=None,  # Fall back to internal equal urgency
                    confirmed_lineups=confirmed_lineups,
                    recent_form=recent_form,
                    rate_modes=None,  # Also clear rate_modes in case abandon zeroed it
                    _retry_attempted=True,  # Bound recursion to one retry
                )

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

"""Daily Category Value (DCV) -- per-player, per-category, per-day optimization.

Computes how much each player is expected to contribute to each scoring
category TODAY, accounting for projections, matchup, availability, and
H2H urgency. Feeds into the LP solver for optimal lineup assignment.

Part of the Lineup Optimizer V2 (pipeline stages 10-12).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC
from typing import Any

import pandas as pd

from src.game_day import LOCKED_GAME_STATUSES as _LOCKED_STATUSES
from src.game_day import get_target_game_date
from src.optimizer.constants_registry import CONSTANTS_REGISTRY as _CR
from src.valuation import canonicalize_team, normalize_player_name, team_name_to_abbr

logger = logging.getLogger(__name__)

# Bounds for the recent-form (L14) adjustment multiplier on per-stat
# projections. Per spec, a player's L14-vs-projected ratio is clamped to
# [_FORM_CLIP_LO, _FORM_CLIP_HI] to prevent a hot/cold streak from
# swinging a single-game DCV by more than ±20%.
_FORM_CLIP_LO: float = 0.80
_FORM_CLIP_HI: float = 1.20

# Bounds for the matchup multiplier from opposing-offense wRC+ (pitchers
# only). League-average wRC+ = 100; divisor is 80 so the multiplier moves
# by 0.5 per 40 wRC+ points (Wave 11B DCV-A1-016: comment previously said
# "1.0 per 40" which mismatched the formula). Capped at ±20% so no single
# matchup dominates a pitcher's value.
_OFFENSE_MULT_LO: float = 0.80
_OFFENSE_MULT_HI: float = 1.20

# Bounds for the platoon multiplier from batter-vs-pitcher handedness.
# Clamped tighter than other matchup factors because platoon splits are
# already conservative (~7.5% L-vs-R wOBA edge).
_PLATOON_MULT_LO: float = 0.80
_PLATOON_MULT_HI: float = 1.20

# Full-season game count used for per-game rate computation.
_FULL_SEASON_GAMES: float = 162.0

# Hitter games-played per season (DCV-A1-012 fix): a regular position
# player appears in ~145 of 162 team games (4-5 scheduled rest days plus
# minor day-to-day absences), so a hitter's season counting projection is
# spread across ~145 games, not the full 162. Dividing by 162 under-weighted
# every hitter's daily DCV by ~10% (145/162). Pitchers keep _FULL_SEASON_GAMES
# in the counting path — their appearance frequency is handled separately by
# the per-role daily fraction (1/30 SP, 1/40 swing, 1/50 RP). Source: FanGraphs
# games-played distributions for qualified hitters, 2022-2024 (~143-147 median).
_HITTER_GAMES_PER_SEASON: float = 145.0

# Replacement-level baselines for rate-stat marginal-contribution SGP.
# Used by build_daily_dcv_table's rate-stat path AND by apply_stud_floor
# for stud ranking. A player's contribution is computed as
# ``(component - opportunity × replacement) / raw_sgp_denom`` so that an
# average MLB starter scores ~0 and stars/scrubs deviate from there.
#
# Per DCV-A1-001 fix (Wave 11A): values now sourced from CONSTANTS_REGISTRY
# (imported as _CR at the top of the module) so sensitivity_analysis can
# perturb them. Loaded at module import; the canonical entry's value field
# is the single source of truth.
_REPL_AVG: float = _CR["repl_avg"].value
_REPL_OBP: float = _CR["repl_obp"].value
_REPL_ERA: float = _CR["repl_era"].value
_REPL_WHIP: float = _CR["repl_whip"].value

_RAW_SGP_DENOM: dict[str, float] = {
    "AVG": _CR["raw_sgp_denom_avg"].value,
    "OBP": _CR["raw_sgp_denom_obp"].value,
    "ERA": _CR["raw_sgp_denom_era"].value,
    "WHIP": _CR["raw_sgp_denom_whip"].value,
}
# Note: `_CR` is intentionally kept as a module-level alias to
# CONSTANTS_REGISTRY (itself a module-level registry, not a per-import
# singleton). Wave 11B uses it for r_stabilization_pa, league_avg_xfip,
# and default_team_weekly_ip lookups at runtime.


@dataclass
class DailyDCVContext:
    """Optional context bundle for :func:`build_daily_dcv_table`.

    Wave 8c (audit D3D-001/D3D-006): the function took 11 keyword args of
    optional context (urgency_weights, confirmed_lineups, recent_form,
    rate_modes, team_strength, etc.). Callers had to pass all 5-6 by
    name; CLAUDE.md flagged "DCV pipeline drops confirmed_lineups /
    recent_form / team_strength before DCV" (BUG-011) as a direct
    consequence — the pipeline forwarded only a subset.

    Bundling lets callers pass a single ``ctx=DailyDCVContext(...)``
    argument; the legacy kwargs continue to work for backwards compat.

    Example:
        ctx = DailyDCVContext(
            urgency_weights=ur,
            confirmed_lineups=lineups,
            recent_form=form,
            team_strength=strength,
        )
        df = build_daily_dcv_table(roster, matchup, sched, parks, ctx=ctx)
    """

    urgency_weights: dict | None = None
    confirmed_lineups: dict[str, list] | None = None
    recent_form: dict[int, dict] | None = None
    rate_modes: dict[str, str] | None = None
    team_strength: dict[str, dict] | None = None

    def merge_into_kwargs(self, **explicit: Any) -> dict[str, Any]:
        """Merge dataclass fields into kwargs, with explicit kwargs winning."""
        merged = {
            "urgency_weights": self.urgency_weights,
            "confirmed_lineups": self.confirmed_lineups,
            "recent_form": self.recent_form,
            "rate_modes": self.rate_modes,
            "team_strength": self.team_strength,
        }
        for k, v in explicit.items():
            if v is not None:
                merged[k] = v
        return merged


def _stuff_plus_k_multiplier(
    stuff_plus: object,
    fip: object = None,
    xfip: object = None,
) -> float:
    """SF-6 helper: multiplicative K-boost from a pitcher's Stuff+ score.

    Wave 4-J Option A baseline: when Stuff+ is unavailable (FanGraphs 403,
    NaN, 0, negative, or non-numeric) the helper returns ``1.0`` -- the
    K-boost path is a provable no-op.

    SF-6 Option B extension: when Stuff+ is missing but the pitcher has
    a valid FIP or xFIP (loaded from the season_stats / ros_projections
    pool), derive a proxy multiplier from FIP. Lower FIP -> better stuff:

        proxy = clip(1.0 + (4.0 - fip) / 10.0, 0.85, 1.15)

    Examples (clipped to [0.85, 1.15]):
        FIP 3.0 -> 1.10x
        FIP 4.0 -> 1.00x (league average)
        FIP 5.0 -> 0.90x
        FIP 1.0 -> 1.15x (clipped)
        FIP 7.0 -> 0.85x (clipped)

    xFIP is preferred over FIP when both are present (xFIP normalises
    home-run luck so it's a cleaner stuff proxy). FIP is the fallback.

    Stuff+ ramp (preserves the previous T3-5a tiers, wins over FIP proxy
    when both are available):
        stuff_plus > 120  -> 1.10x
        stuff_plus > 110  -> 1.05x
        otherwise         -> 1.00x  (or FIP proxy if Stuff+ is invalid)
    """
    try:
        val = float(stuff_plus) if stuff_plus is not None else 0.0
    except (TypeError, ValueError):
        val = 0.0
    stuff_valid = (val == val) and (val > 0.0)
    if stuff_valid:
        if val > 120:
            return 1.10
        if val > 110:
            return 1.05
        return 1.00

    for candidate in (xfip, fip):
        if candidate is None:
            continue
        try:
            f = float(candidate)
        except (TypeError, ValueError):
            continue
        if f != f or f <= 0.0:
            continue
        proxy = 1.0 + (4.0 - f) / 10.0
        return max(0.85, min(1.15, proxy))

    return 1.0


# 2026-05-19 Section 3 D2: backward-compat alias for tests + closer_monitor
# transition. New code MUST import normalize_player_name from src.valuation;
# this alias allows existing call sites (test_optimizer_audit_fixes.py, others)
# to continue working without mass-editing imports across the test suite.
_normalize_pitcher_name = normalize_player_name


# Stabilization points for Bayesian blend (from FanGraphs research).
# Wave 11B DCV-A1-007: r=460 was a copy-paste from OBP; per FanGraphs runs
# stabilize much faster (~250 PA). Read from CONSTANTS_REGISTRY so
# sigmoid_calibrator / sensitivity_analysis can perturb it.
#
# OQ-4 (2026-05-15): W, L, and SV are NOT canonical rate-stabilization
# points. Sabermetric research (Carleton; FanGraphs Sample Size library)
# calibrates stabilization only for rate stats with well-defined per-
# opportunity denominators (BF, AB, BIP, FB). W/L/SV are discrete-outcome
# counting stats — wins are dominated by team_win_pct, not pitcher rate
# stability; saves are role-driven (closer_monitor.job_security is the
# right signal); losses inherit the same problem as wins. The values
# below (w=200, l=200, sv=100) are non-canonical and ``_BLENDABLE_KEYS``
# below excludes them so ``compute_blended_projection`` is a pure
# pass-through for these three keys. Full refactor (derive W from
# IP_proj × winpct via team_strength, drive SV from closer_monitor) is
# tracked as a future-wave item; this minimal change prevents the
# undefined-behavior blend the audit flagged.
STABILIZATION_POINTS: dict[str, float] = {
    "r": int(_CR["r_stabilization_pa"].value),
    "hr": 170,
    "rbi": 300,
    "sb": 200,
    "avg": 910,
    "obp": 460,
    # OQ-4 non-canonical entries — included only so callers that iterate
    # STABILIZATION_POINTS keys don't KeyError, but the blend skips them
    # via ``_NON_BLENDABLE_KEYS`` below.
    "w": 200,
    "l": 200,
    "sv": 100,
    "k": 70,
    "era": 630,
    "whip": 540,
}

# OQ-4 (2026-05-15): these three keys are NOT valid rate-stabilization
# stats — they're discrete-outcome counting stats. compute_blended_projection
# treats them as pass-throughs (returns preseason_rate unchanged) so the
# math is never undefined. Future-wave work: derive W/L/SV from underlying
# rates (IP_proj × winpct via team_strength; closer_monitor for SV).
_NON_BLENDABLE_KEYS: frozenset[str] = frozenset({"w", "l", "sv"})

# Top N players by ROS projection that get stud floor protection.
# Wave 11B DCV-A1-014: 8 ≈ top ~28% of a 28-slot Yahoo roster (~23 active +
# 5 BN). Heuristic ceiling for "obvious starts no matter what the matchup
# says" — keeps elite players from being benched by single-day DCV noise.
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

    OQ-4 (2026-05-15): for ``stat_key in _NON_BLENDABLE_KEYS`` (currently
    ``{"w", "l", "sv"}``) the function is a pure pass-through — those
    stats are discrete-outcome counting events, not rates with calibrated
    stabilization points, so a Bayesian blend on them produces undefined
    behavior (no canonical denominator). The preseason_rate is returned
    unchanged.

    Args:
        preseason_rate: Preseason projection rate for the stat.
        observed_numerator: Observed stat total so far this season.
        observed_denominator: Denominator (PA for hitters, IP for pitchers).
        stat_key: Lowercase stat key (e.g. "hr", "era").

    Returns:
        Blended projection rate (or preseason_rate unchanged for
        non-blendable stats).
    """
    if stat_key in _NON_BLENDABLE_KEYS:
        return preseason_rate
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
        "susp",
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
            mult *= max(_PLATOON_MULT_LO, min(_PLATOON_MULT_HI, plat))
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

    # Opposing pitcher quality (xFIP-based) — hitters only.
    # Wave 11B DCV-A1-009: league-avg xFIP from CONSTANTS_REGISTRY so the
    # baseline updates yearly via that registry entry; previously hardcoded 4.20.
    # 2026-05-17 Section 2 L6 fix: pitcher_xfip=0 (missing-data default) passed
    # the `is not None` check and set quality=2.0 (elite), halving hitter DCV.
    # Require xfip > 0 to skip the adjustment when xFIP is missing/zero.
    if pitcher_xfip is not None and pitcher_xfip > 0 and is_hitter:
        _xfip_baseline = _CR["league_avg_xfip"].value
        quality = max(0.5, min(2.0, 2.0 - pitcher_xfip / _xfip_baseline))
        # Invert for hitters: good pitcher hurts hitter value
        mult *= 1.0 / max(0.5, quality)

    # Opposing team offense quality (wRC+-based) — pitchers only.
    # League-average wRC+ = 100. A pitcher facing a 120 wRC+ offense (NYY)
    # gets a dampened multiplier; facing a 80 wRC+ offense (bottom-third)
    # gets a boost. Divisor is 80 so the multiplier moves 0.5 per 40 wRC+
    # (Wave 11B DCV-A1-016: comment previously said "~1.0 per 40 wRC+" which
    # mismatched the formula). Clamped to [0.80, 1.20] so no single matchup
    # can more than 20% swing a pitcher's value.
    if opponent_offense_wrc_plus is not None and not is_hitter:
        try:
            _wrcp = float(opponent_offense_wrc_plus)
            # Inverse: 120 wRC+ → (100-120)/40 = -0.5 → ~0.95 multiplier
            # 80 wRC+ → (100-80)/40 = 0.5 → ~1.05 multiplier
            _off_mult = 1.0 + (100.0 - _wrcp) / 80.0
            mult *= max(_OFFENSE_MULT_LO, min(_OFFENSE_MULT_HI, _off_mult))
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

    # Compute total SGP per player from ROS projections.
    #
    # Counting stats use ``val / denom`` (the standing-point contribution).
    # Rate stats MUST use marginal contribution from raw components
    # (h, ab, bb, hbp, sf, ip, er, bb_allowed, h_allowed) and the
    # replacement-level baselines (_REPL_AVG/OBP/ERA/WHIP). The previous
    # implementation summed ``raw_rate / denom`` for rate stats too, which
    # produced contributions ~45× larger than counting stats and biased
    # the stud ranking toward high-AVG hitters (DCV-A1-002 fix).
    total_sgp: dict[int, float] = {}
    for _, row in roster.iterrows():
        pid = row.get("player_id")
        sgp = 0.0
        is_hitter = bool(row.get("is_hitter", 1))

        for cat in config.all_categories:
            if cat in config.rate_stats:
                # Rate stats: marginal contribution from raw components.
                # Mirrors build_daily_dcv_table's annual_sgp formula.
                if cat == "AVG" and is_hitter:
                    ab = float(row.get("ab", 0) or 0)
                    h = float(row.get("h", 0) or 0)
                    if ab > 0:
                        sgp += (h - ab * _REPL_AVG) / _RAW_SGP_DENOM["AVG"]
                elif cat == "OBP" and is_hitter:
                    ab = float(row.get("ab", 0) or 0)
                    h = float(row.get("h", 0) or 0)
                    bb = float(row.get("bb", 0) or 0)
                    hbp = float(row.get("hbp", 0) or 0)
                    sf = float(row.get("sf", 0) or 0)
                    denom_pa = ab + bb + hbp + sf
                    if denom_pa > 0:
                        sgp += ((h + bb + hbp) - denom_pa * _REPL_OBP) / _RAW_SGP_DENOM["OBP"]
                elif cat == "ERA" and not is_hitter:
                    ip = float(row.get("ip", 0) or 0)
                    er = float(row.get("er", 0) or 0)
                    if ip > 0:
                        repl_er = _REPL_ERA * ip / 9.0
                        sgp += (repl_er - er) / _RAW_SGP_DENOM["ERA"]
                elif cat == "WHIP" and not is_hitter:
                    ip = float(row.get("ip", 0) or 0)
                    bb_a = float(row.get("bb_allowed", 0) or 0)
                    h_a = float(row.get("h_allowed", 0) or 0)
                    if ip > 0:
                        repl_wh = _REPL_WHIP * ip
                        sgp += (repl_wh - (bb_a + h_a)) / _RAW_SGP_DENOM["WHIP"]
            else:
                # Counting stats: standard val/denom standing-point contribution.
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
                # Wave 11B DCV-A1-018: 1.5× median is heuristic — meant to
                # lift a stud from "marginal" to "comfortably above median"
                # while still tracking matchup quality via `mm`. No formal
                # calibration; chosen by inspection on 2026-04-17 rosters.
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
    *,
    ctx: DailyDCVContext | None = None,
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
    # Merge ctx dataclass into kwargs (explicit kwargs win).
    if ctx is not None:
        if urgency_weights is None:
            urgency_weights = ctx.urgency_weights
        if confirmed_lineups is None:
            confirmed_lineups = ctx.confirmed_lineups
        if recent_form is None:
            recent_form = ctx.recent_form
        if rate_modes is None:
            rate_modes = ctx.rate_modes
        if team_strength is None:
            team_strength = ctx.team_strength

    if config is None:
        from src.valuation import LeagueConfig

        config = LeagueConfig()

    # SF-25: Single SGPCalculator instance reused across all players + cats.
    # Replaces the inline weighted_dcv/denom + manual sign-flip for counting
    # stats at the per-category branch below; keeps math centralized in
    # SGPCalculator.totals_sgp (the V1-V6 unified entry point).
    from src.valuation import SGPCalculator

    sgp_calc = SGPCalculator(config)

    if park_factors is None:
        park_factors = {}

    # Get roster statuses for health factor
    try:
        from src.trade_intelligence import _load_roster_statuses

        statuses = _load_roster_statuses()
    except Exception:
        statuses = {}

    # Determine which teams play today. Schedule may use full names
    # ("CHICAGO CUBS") or abbreviations ("CHC"); normalize both to
    # abbreviations via the canonical `team_name_to_abbr` helper.
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
            # Wave 11B DCV-A4-001: consume LOCKED_GAME_STATUSES from
            # src.game_day (centralized) instead of an inline tuple. We
            # use substring containment to remain tolerant of statsapi
            # status drift (e.g. "Completed Early — Rain"). The
            # constant in game_day uses normalized lowercase forms.
            if any(s in _status_str for s in _LOCKED_STATUSES):
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
                    # team_name_to_abbr returns the raw input on miss, so 3-letter
                    # codes pass through unchanged; canonicalize_team then resolves
                    # legacy variants (OAK→ATH, WSN→WSH, etc.).
                    abbr = team_name_to_abbr(raw)
                    canon = canonicalize_team(abbr)
                    teams_playing.add(canon)
                    if _game_locked:
                        locked_teams.add(canon)

    # Doubleheader detection (OQ-2 resolution, 2026-05-15): when a team
    # has two scheduled game entries today, both games contribute their
    # stats to Yahoo H2H — so a confirmed starter gets ~2× the daily
    # opportunity. Count (canonical-home, canonical-away) pairs from
    # schedule_today; >1 entry for the same pair = doubleheader for both
    # teams. We carry the set of doubleheader teams through to the
    # volume-factor call below.
    doubleheader_teams: set[str] = set()
    if schedule_today:
        _team_game_counts: dict[str, int] = {}
        for game in schedule_today:
            _h_raw = str(game.get("home_name", "") or game.get("home_team", "")).upper().strip()
            _a_raw = str(game.get("away_name", "") or game.get("away_team", "")).upper().strip()
            _h_canon = canonicalize_team(team_name_to_abbr(_h_raw)) if _h_raw else ""
            _a_canon = canonicalize_team(team_name_to_abbr(_a_raw)) if _a_raw else ""
            for tc in (_h_canon, _a_canon):
                if tc:
                    _team_game_counts[tc] = _team_game_counts.get(tc, 0) + 1
        doubleheader_teams = {tc for tc, n in _team_game_counts.items() if n >= 2}

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
        from src.database import load_game_day_weather

        today_date = get_target_game_date()
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
                    home_abbr = team_name_to_abbr(game.get("home_name", ""))
                    away_abbr = team_name_to_abbr(game.get("away_name", ""))
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
        team_canon = canonicalize_team(team)
        is_hitter = bool(player.get("is_hitter", 1))

        # Health factor
        status = statuses.get(pid, str(player.get("status", "active")))
        health = compute_health_factor(status)

        # Volume factor
        team_plays = team_canon in teams_playing if teams_playing else True
        in_lineup = None  # default: lineup not posted
        lineup_slot = 0  # 0 = unknown batting order slot
        if confirmed_lineups is not None:
            is_hitter = bool(int(player.get("is_hitter", 1)))
            # Try canonical code first (handles OAK→ATH, WSN→WSH, CHW→CWS,
            # AZ→ARI drift), fall back to raw team string for legacy
            # callers that key by whatever was in the players.team column.
            _lineup_key: str | None = None
            if is_hitter:
                if team_canon in confirmed_lineups:
                    _lineup_key = team_canon
                elif team in confirmed_lineups:
                    _lineup_key = team
            if _lineup_key is not None:
                # Batting lineups only meaningful for hitters
                team_lineup = confirmed_lineups[_lineup_key]
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
            _pos_tokens = {p.strip() for p in pos_upper.split(",")}
            _has_sp = "SP" in _pos_tokens
            _has_rp = "RP" in _pos_tokens
            _has_p_only = _pos_tokens == {"P"}
            if (_has_sp or _has_p_only) and probable_starters:
                _norm_name = normalize_player_name(name)
                _norm_probable = {normalize_player_name(p) for p in probable_starters}
                _is_probable_today = _norm_name in _norm_probable
                if _is_probable_today:
                    in_lineup = True
                elif not _has_rp:
                    in_lineup = False
                    _pitcher_volume_override = 0.0
        # OQ-2 (2026-05-15): doubleheader teams get the 2.0× boost for
        # confirmed starters / 1.8× for not-yet-posted lineups. Only hitters
        # benefit — a single SP doesn't pitch both legs, so the pitcher
        # gate above already keeps `_pitcher_volume_override` authoritative.
        _is_dh = is_hitter and team_canon in doubleheader_teams
        volume = compute_volume_factor(team_plays, in_lineup, is_doubleheader=_is_dh)
        if _pitcher_volume_override is not None:
            volume = _pitcher_volume_override
        # Deduct already-played game contributions: if this player's team
        # has a game in progress or final today, Yahoo has locked the slot
        # and the player can no longer be swapped in/out. Forward-looking
        # DCV must be 0 — the optimizer shouldn't "recommend" someone whose
        # game is already started/done as if action is still possible.
        if team_canon in locked_teams:
            volume = 0.0

        # Skip entirely if excluded
        if health == 0.0 or volume == 0.0:
            _is_locked = team_canon in locked_teams
            _excl_reason = "LOCKED" if _is_locked else ("IL" if health == 0.0 else "OFF_DAY")
            row_data: dict = {
                "player_id": pid,
                "name": name,
                "positions": positions,
                "team": team,
                "is_hitter": is_hitter,
                "game_locked": _is_locked,
                "health_factor": health,
                "volume_factor": volume,
                "matchup_mult": None,
                "total_dcv": 0.0,
                "stud_floor_applied": False,
                "status": status,
                "reason": _excl_reason,
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
        # Try canonical team code first (handles OAK→ATH etc. drift) so a
        # player whose DB row still says "OAK" still gets ATH's weather.
        player_temp = weather_by_team.get(team_canon) or weather_by_team.get(team)
        _batter_hand = str(player.get("bats", "") or "")
        _pitcher_hand = ""
        _opp_pitcher_name = ""
        _venue_team = ""  # home team code — whose park factor to use
        _opp_team = ""  # literal opponent team code — for opposing offense wRC+
        if schedule_today and team:
            for _sg in schedule_today:
                _home_short = str(_sg.get("home_short", "")).upper().strip()
                _away_short = str(_sg.get("away_short", "")).upper().strip()
                _home_ab = team_name_to_abbr(_sg.get("home_name", "")) or _home_short
                _away_ab = team_name_to_abbr(_sg.get("away_name", "")) or _away_short
                _home_canon = canonicalize_team(_home_ab) if _home_ab else ""
                _away_canon = canonicalize_team(_away_ab) if _away_ab else ""
                _home_short_canon = canonicalize_team(_home_short) if _home_short else ""
                _away_short_canon = canonicalize_team(_away_short) if _away_short else ""
                if team_canon == _home_canon or team_canon == _home_short_canon:
                    _venue_team = _home_ab  # player is home → venue is own park
                    _opp_team = _away_ab
                    _opp_pitcher_name = str(_sg.get("away_probable_pitcher", "") or "")
                    break
                if team_canon == _away_canon or team_canon == _away_short_canon:
                    _venue_team = _home_ab  # player is away → venue is opponent park
                    _opp_team = _home_ab
                    _opp_pitcher_name = str(_sg.get("home_probable_pitcher", "") or "")
                    break

        _opp_pitcher_xfip: float | None = None
        if is_hitter and _opp_pitcher_name:
            _norm_opp = normalize_player_name(_opp_pitcher_name)
            _name_col = "player_name" if "player_name" in roster.columns else "name"
            if _name_col in roster.columns:
                _opp_match = roster[roster[_name_col].apply(lambda n: normalize_player_name(str(n)) == _norm_opp)]
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
                _opp_canon = canonicalize_team(_opp_team)
                if _opp_team in team_strength:
                    _ts_entry = team_strength[_opp_team]
                elif _opp_canon in team_strength:
                    # Try canonical form (resolves CWS/CHW, WSH/WSN/WAS, AZ/ARI, etc.)
                    _ts_entry = team_strength[_opp_canon]
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
                                lo = orig * _FORM_CLIP_LO
                                hi = orig * _FORM_CLIP_HI
                                form_adjustments[stat_key] = max(lo, min(hi, blended))
                    # Counting stats: use rate-ratio from L14 per-game vs projected per-game
                    for stat_key in ("r", "hr", "rbi", "sb"):
                        form_val = form.get(stat_key)
                        if form_val is not None and form_games > 0:
                            orig = float(player.get(stat_key, 0) or 0)
                            if orig > 0:
                                # L14 per-game rate vs projected per-162 per-game rate
                                form_per_game = float(form_val) / form_games
                                proj_per_game = orig / _FULL_SEASON_GAMES
                                if proj_per_game > 0:
                                    ratio = form_per_game / proj_per_game
                                    adj = _base_weight * 1.0 + _form_weight * ratio
                                    adj = max(_FORM_CLIP_LO, min(_FORM_CLIP_HI, adj))
                                    form_adjustments[stat_key] = orig * adj
                else:
                    # Pitcher rate stats: blend directly
                    for stat_key in ("era", "whip"):
                        form_val = form.get(stat_key)
                        if form_val is not None:
                            orig = float(player.get(stat_key, 0) or 0)
                            if orig > 0:
                                blended = _base_weight * orig + _form_weight * float(form_val)
                                lo = orig * _FORM_CLIP_LO
                                hi = orig * _FORM_CLIP_HI
                                form_adjustments[stat_key] = max(lo, min(hi, blended))
                    # Pitcher counting stats: K, W, SV
                    for stat_key in ("k", "w", "sv"):
                        form_val = form.get(stat_key)
                        if form_val is not None and form_games > 0:
                            orig = float(player.get(stat_key, 0) or 0)
                            if orig > 0:
                                form_per_game = float(form_val) / form_games
                                proj_per_game = orig / _FULL_SEASON_GAMES
                                if proj_per_game > 0:
                                    ratio = form_per_game / proj_per_game
                                    adj = _base_weight * 1.0 + _form_weight * ratio
                                    adj = max(_FORM_CLIP_LO, min(_FORM_CLIP_HI, adj))
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
        # Replacement levels + raw-unit SGP denominators are module-level
        # constants (_REPL_*, _RAW_SGP_DENOM) so apply_stud_floor can share
        # the same calibration. See module-level definitions above.
        # Daily volume fraction (today's contribution / annual contribution).
        # Hitters: ~145 games played → 1/145 (DCV-A1-012). SP: ~30 starts → 1/30.
        # RP: ~50 appearances → 1/50.
        _hitter_daily_frac = 1.0 / _HITTER_GAMES_PER_SEASON
        # Token-set match (mirrors the SP gate above): a starter must have
        # "SP" as a literal token in `positions` (e.g. "SP" / "SP,RP"); a
        # substring check would also fire on a hypothetical "RSP" or any
        # future code that contains "SP" as a substring. Reuse the same
        # tokenisation used at line ~764 so the two paths can't diverge.
        # Wave 11B DCV-A1-006: SP/RP hybrids (e.g. swingmen) get a blended
        # 1/40 daily fraction — they make ~5-10 starts (~1/30 frac) and
        # ~20-30 relief appearances (~1/50 frac) per season. Pure-SP uses
        # 1/30; pure-RP / pure-P uses 1/50.
        _pitcher_pos_tokens = {p.strip() for p in str(player.get("positions", "")).upper().split(",")}
        _has_sp_token = "SP" in _pitcher_pos_tokens
        _has_rp_token = "RP" in _pitcher_pos_tokens
        if _has_sp_token and _has_rp_token:
            _pitcher_daily_frac = 1.0 / 40.0
        elif _has_sp_token:
            _pitcher_daily_frac = 1.0 / 30.0
        else:
            _pitcher_daily_frac = 1.0 / 50.0

        for cat in config.all_categories:
            col = cat.lower()
            proj_val = form_adjustments.get(col, float(player.get(col, 0) or 0))

            # Per-game rate: spread the season counting projection across the
            # player's games. Hitters play ~145 (DCV-A1-012); pitchers keep the
            # full-season denominator (their per-appearance frequency is carried
            # by the daily fraction / volume factor, not this divisor).
            # Rate stats use volume-weighted SGP (computed below).
            if cat in config.rate_stats:
                # Compute volume-weighted SGP contribution directly,
                # bypassing the per-game daily_proj path.
                daily_proj = 0.0  # placeholder; sgp_dcv computed below
            else:
                _games_divisor = _HITTER_GAMES_PER_SEASON if is_hitter else _FULL_SEASON_GAMES
                daily_proj = proj_val / _games_divisor

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
            config.sgp_denominators.get(cat, 1.0)
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
            else:
                # SF-25: counting-stat SGP via canonical SGPCalculator.
                # Equivalent to (weighted_dcv / denom) * sign — sign is -1
                # for inverse cats (L), +1 otherwise; pathological
                # zero-denom cats contribute 0 (matches prior else branch).
                sgp_dcv = sgp_calc.totals_sgp({cat: weighted_dcv})

            # T3-5a: Stuff+ boost for pitcher K DCV.
            # SF-6: helper guarantees neutral 1.0x when stuff_plus is
            # missing/0/NaN (FanGraphs 403 leaves the column NULL). FIP/xFIP
            # serve as a proxy when Stuff+ is unavailable.
            if col == "k" and not is_hitter:
                sgp_dcv *= _stuff_plus_k_multiplier(
                    player.get("stuff_plus"),
                    fip=player.get("fip"),
                    xfip=player.get("xfip"),
                )

            # T3-5b: Sprint speed boost for hitter SB DCV.
            # Wave 11B DCV-A1-013: thresholds 28.0/29.0 ft/s reference
            # Statcast league-avg sprint speed ~27.0; 28 = "fast," 29 = "elite."
            # Multipliers 1.05/1.10 calibrated to give elite burners a modest
            # SB-only edge without over-weighting raw speed.
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
                    team_strength=team_strength,  # Preserve opp-offense wRC+ on retry
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
    ip_minimum: float | None = None,
    config: object | None = None,
) -> pd.DataFrame:
    """Force-start a pitcher if weekly IP is below the league IP minimum.

    FourzynBurn (OQ-3 resolution, 2026-05-15): the league enforces a 20 IP
    floor combining SPs + RPs per matchup week — teams below the floor
    incur forfeit-style penalties on pitching categories. When the
    projected IP for the week falls below this floor, this routine boosts
    the best available pitcher's DCV so the LP starts them even on a
    bad-matchup day, recovering toward the floor.

    Args:
        dcv_table: DCV DataFrame from build_daily_dcv_table().
        weekly_ip_projected: Current projected IP for the week.
        ip_minimum: Minimum IP threshold. When None, reads
            ``LeagueConfig.weekly_ip_minimum`` from the supplied ``config``;
            falls back to ``20.0`` if no config is provided.
        config: Optional LeagueConfig — for tests / future leagues that
            change the IP minimum without re-deploying.

    Returns:
        Updated dcv_table with pitcher boost applied if needed.
    """
    if ip_minimum is None:
        if config is not None:
            ip_minimum = float(getattr(config, "weekly_ip_minimum", 20.0))
        else:
            from src.valuation import LeagueConfig

            ip_minimum = float(LeagueConfig().weekly_ip_minimum)
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
    weekly_ip_target: float | None = None,
) -> pd.DataFrame:
    """Scale pitcher counting-stat DCV by remaining IP budget fraction.

    When a team has already used most of its weekly IP budget, pitcher
    counting stats (K/W/SV/L) should be scaled down to reflect reduced
    remaining innings. This prevents over-valuing pitchers late in the
    week when IP is tight.

    Args:
        dcv_table: DCV DataFrame from build_daily_dcv_table().
        weekly_ip_projected: Total projected IP already used/committed.
        weekly_ip_target: Season-wide weekly IP target. When None
            (Wave 11B DCV-A2-003), reads `default_team_weekly_ip` from
            CONSTANTS_REGISTRY so the value updates as that entry is
            recalibrated.

    Returns:
        Updated dcv_table with pitcher DCV scaled by IP budget.
    """
    if dcv_table.empty:
        return dcv_table

    if weekly_ip_target is None:
        weekly_ip_target = float(_CR["default_team_weekly_ip"].value)

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

    # Apply scale to pitcher counting-stat DCV columns (K, W, SV, L).
    # Wave 11B DCV-A1-011: L is an inverse counting stat — leaving it
    # unscaled while shrinking K/W/SV over-penalised pitchers near the
    # IP cap (a pitcher with no remaining opportunity also can't take
    # a Loss, so dcv_l should fade proportionally).
    if "is_hitter" not in dcv_table.columns:
        return dcv_table

    counting_cols = ["dcv_k", "dcv_w", "dcv_sv", "dcv_l"]
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

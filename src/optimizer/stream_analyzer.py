"""Pitcher Streaming Analyzer engine.

Composition layer for the Pitcher Streaming page
(design: docs/superpowers/specs/2026-06-09-pitcher-streaming-analyzer-design.md).

Composes the existing canonical pieces — ``streaming.py`` (marginal SGP,
Bayesian stream score), ``two_start.py`` (pitcher-vs-opponent matchup score),
``matchup_adjustments.py`` (park, PvB), ``game_day.py`` (schedule, locked
statuses, team strength) — into a single 0-100 Stream Score per scheduled
start, plus the board/deep-dive/replay data structures the page renders.

Engine purity rules (guarded by tests/test_stream_analyzer_*.py):
- No streamlit imports.
- All tunable weights/thresholds read from CONSTANTS_REGISTRY at call time.
- The weekly add budget derives from league_rules.WEEKLY_TRANSACTION_LIMIT,
  never streaming.py's legacy alias.
- SGP flows through compute_streaming_value / SGPCalculator paths only —
  inverse stats (ERA/WHIP/L) must never get inline sign math here.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from src.game_day import _NEUTRAL_DEFAULTS as _TEAM_NEUTRAL
from src.game_day import (
    DOME_TEAMS,
    FINAL_GAME_STATUSES,
    LOCKED_GAME_STATUSES,
    OUTFIELD_BEARING,
)
from src.optimizer.constants_registry import CONSTANTS_REGISTRY as _CR
from src.optimizer.matchup_adjustments import (
    is_wind_blowing_out,
    weather_wind_hr_adjustment,
)
from src.optimizer.streaming import (
    _WHIP_SAFETY_CEILING,
    compute_bayesian_stream_score,
    compute_streaming_value,
)
from src.two_start import _confidence_tier, compute_pitcher_matchup_score
from src.valuation import normalize_player_name, team_name_to_abbr

logger = logging.getLogger(__name__)

# Sigmoid steepness mapping the weighted component blend ([-1, 1]) onto the
# 0-100 display scale. Pure display shaping (neutral blend ⇒ 50), not a
# tunable model parameter — the tunables are the registry weights.
_SCORE_SIGMOID_K: float = 3.0

# L14 IP below which the recent-form signal is noise (mirrors the FA engine's
# l14_ip volume gate, PR #110 / P5b).
_FORM_MIN_L14_IP: float = 5.0

# Form deltas are clipped to ±20% before normalization — same magnitude as
# streaming.py's _FORM_MULT_LO/HI band and the DCV form clip.
_FORM_DELTA_CLIP: float = 0.20

# The six Stream Score components, in registry order. Weights are read from
# CONSTANTS_REGISTRY at call time (never cached at import) so calibration
# tooling that patches the registry takes effect without a restart.
_SCORE_WEIGHT_KEYS: tuple[str, ...] = (
    "stream_score_w_matchup",
    "stream_score_w_sgp",
    "stream_score_w_form",
    "stream_score_w_lineup",
    "stream_score_w_env",
    "stream_score_w_winprob",
)


def _score_weights() -> dict[str, float]:
    """Return component-name → weight, read from the registry at call time."""
    return {key.removeprefix("stream_score_w_"): float(_CR[key].value) for key in _SCORE_WEIGHT_KEYS}


def get_opponent_offense_context(
    team_abbr: str,
    vs_throws: str | None,
    team_strength: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    """Opposing-offense snapshot for one scheduled start.

    Prefers a vs-handedness split (``wrc_plus_vs_lhp`` / ``wrc_plus_vs_rhp``
    keys, present when the optional team-splits fetch ran) and falls back to
    the overall team line; ``split_source`` tells the UI which one it got so
    the fallback can be footnoted.

    Args:
        team_abbr: Opposing team code (canonical MLB Stats API form).
        vs_throws: The streaming pitcher's throwing hand ("L"/"R"), or None.
        team_strength: ``ctx.team_strength``-shaped mapping
            (team → {"wrc_plus", "k_pct", "bb_pct", ...}; k_pct in percent).

    Returns:
        {"wrc_plus", "k_pct", "bb_pct", "iso", "l14_wrc_plus", "split_source"}
        — ``iso`` / ``l14_wrc_plus`` are None when the source lacks them;
        missing teams get the neutral league-average line.
    """
    entry: dict[str, Any] = {}
    if team_strength:
        entry = dict(team_strength.get(team_abbr) or {})

    hand = (vs_throws or "").upper()
    split_suffix = {"L": "vs_lhp", "R": "vs_rhp"}.get(hand)

    split_source = "overall"
    wrc_plus = entry.get("wrc_plus")
    k_pct = entry.get("k_pct")
    if split_suffix:
        split_wrc = entry.get(f"wrc_plus_{split_suffix}")
        if split_wrc is not None:
            wrc_plus = split_wrc
            k_pct = entry.get(f"k_pct_{split_suffix}", k_pct)
            split_source = "vs_hand"

    if wrc_plus is None:
        wrc_plus = _TEAM_NEUTRAL["wrc_plus"]
    if k_pct is None:
        k_pct = _TEAM_NEUTRAL["k_pct"]

    return {
        "wrc_plus": float(wrc_plus),
        "k_pct": float(k_pct),
        "bb_pct": float(entry.get("bb_pct", _TEAM_NEUTRAL["bb_pct"])),
        "iso": entry.get("iso"),
        "l14_wrc_plus": entry.get("l14_wrc_plus"),
        "split_source": split_source,
    }


def compute_hitter_matchup_score(
    opp_sp_stats: Any,
    team_offense: Any,
    park_factor: float = 1.0,
    hitters_home: bool = True,
) -> float:
    """Calibrated inverse of the pitcher matchup, for the batting team's side.

    Mirrors ``score_stream_candidate``'s matchup block (same input extraction) so the
    result is the EXACT complement of the Probable grid's ``matchup_score``: a starter
    who is a great stream against this offense is, by construction, a tough matchup for
    these bats.

    Args:
        opp_sp_stats: The opposing starter's pool row (dict/Series): reads
            ``k_bb_pct``/``xfip``/``csw_pct``/``era``/``whip`` (missing -> engine defaults).
        team_offense: The batting team's offense vs the SP's hand: ``wrc_plus`` and
            ``k_pct`` (in PERCENT, as ``get_opponent_offense_context`` emits).
        park_factor: Venue park factor (the batting team's park when they are home).
        hitters_home: True if the batting team is at home.

    Returns:
        0-100, higher = better hitting matchup. NaN-safe, never raises.
    """
    pitcher_stats = {
        key: val
        for key in ("k_bb_pct", "xfip", "csw_pct", "era", "whip")
        if (val := _get_num(opp_sp_stats, key)) is not None
    }
    opp_stats = {
        "wrc_plus": _get_num(team_offense, "wrc_plus", _TEAM_NEUTRAL["wrc_plus"]),
        "k_pct": _get_num(team_offense, "k_pct", _TEAM_NEUTRAL["k_pct"]) / 100.0,
    }
    pitcher_score = compute_pitcher_matchup_score(
        pitcher_stats,
        opponent_team_stats=opp_stats,
        park_factor=park_factor,
        is_home=not hitters_home,
    )
    return round(max(0.0, min(100.0, (10.0 - pitcher_score) * 10.0)), 1)


# ── Component helpers ────────────────────────────────────────────────


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    # NaN → neutral 0.0, NOT the bound: ``min(hi, nan)`` returns ``nan`` (NaN
    # comparisons are False), which would pin a data-missing component to the MAX.
    if value != value or math.isinf(value):
        return 0.0
    return max(lo, min(hi, value))


def _get_num(obj: Any, key: str, default: float | None = None) -> float | None:
    """Extract a finite float from a dict-like row, else *default*."""
    try:
        val = obj.get(key) if hasattr(obj, "get") else obj[key]
    except (KeyError, TypeError, IndexError):
        return default
    try:
        out = float(val)
    except (TypeError, ValueError):
        return default
    if math.isnan(out):
        return default
    return out


def _lower_keys(mapping: dict[str, float] | None) -> dict[str, float] | None:
    """streaming.py expects lowercase stat keys; LeagueConfig uses uppercase."""
    if not mapping:
        return None
    return {str(k).lower(): float(v) for k, v in mapping.items()}


def _ip_per_start(pitcher_row: Any) -> float | None:
    """Best-effort projected IP per start; None when underivable."""
    explicit = _get_num(pitcher_row, "ip_per_start")
    if explicit is not None and explicit > 0:
        return explicit
    ip = _get_num(pitcher_row, "ip")
    gs = _get_num(pitcher_row, "gs")
    if ip is not None and gs is not None and gs > 0:
        return ip / gs
    return None


def _form_component(pitcher_row: Any, recent_form: dict | None) -> float:
    """L14-vs-baseline form in [-1, 1]; 0 when data is missing or thin."""
    if not recent_form:
        return 0.0
    l14 = recent_form.get("l14") or {}
    l14_ip = _get_num(l14, "ip", 0.0) or 0.0
    if l14_ip < _FORM_MIN_L14_IP:
        return 0.0

    base_era = _get_num(pitcher_row, "era")
    l14_era = _get_num(l14, "era")
    era_term = 0.0
    if base_era and base_era > 0 and l14_era is not None:
        # Lower L14 ERA than baseline = hot.
        era_term = _clamp((base_era - l14_era) / base_era, -_FORM_DELTA_CLIP, _FORM_DELTA_CLIP)

    base_ip = _get_num(pitcher_row, "ip")
    base_k = _get_num(pitcher_row, "k")
    l14_k = _get_num(l14, "k")
    k_term = 0.0
    if base_ip and base_ip > 0 and base_k is not None and l14_k is not None and l14_ip > 0:
        base_k9 = base_k * 9.0 / base_ip
        l14_k9 = l14_k * 9.0 / l14_ip
        if base_k9 > 0:
            k_term = _clamp((l14_k9 - base_k9) / base_k9, -_FORM_DELTA_CLIP, _FORM_DELTA_CLIP)

    return _clamp((0.5 * era_term + 0.5 * k_term) / _FORM_DELTA_CLIP)


def _env_component(park_factor: float, weather: dict | None, venue: str) -> tuple[float, bool]:
    """Park × weather environment in [-1, 1] (pitcher perspective).

    Returns (component, wind_out) — wind_out feeds the WIND_OUT risk flag.
    Domes ignore weather entirely (park-only).
    """
    park_term = 0.0
    if park_factor and park_factor > 0:
        park_term = _clamp(1.0 / park_factor, 0.5, 2.0) - 1.0

    wind_term = 0.0
    wind_out = False
    if weather and venue and venue not in DOME_TEAMS:
        wind_mph = _get_num(weather, "wind_mph", 0.0) or 0.0
        wind_dir = _get_num(weather, "wind_dir")
        bearing = OUTFIELD_BEARING.get(venue)
        if wind_dir is not None and bearing is not None:
            wind_out = is_wind_blowing_out(wind_dir, bearing)
        hr_mult = weather_wind_hr_adjustment(wind_mph, wind_out=wind_out)
        # hr_mult > 1 (wind out) hurts the pitcher.
        wind_term = 1.0 - hr_mult

    return _clamp(2.0 * (park_term + wind_term)), wind_out


def _risk_flags(
    pitcher_row: Any,
    start_info: dict,
    opp_context: dict,
    wind_out: bool,
) -> list[str]:
    flags: list[str] = []

    whip = _get_num(pitcher_row, "whip")
    if whip is not None and whip > _WHIP_SAFETY_CEILING:
        flags.append("HIGH_WHIP")

    ip_ps = _ip_per_start(pitcher_row)
    if ip_ps is not None and ip_ps < float(_CR["stream_risk_short_leash_ip"].value):
        flags.append("SHORT_LEASH")

    if float(opp_context.get("wrc_plus", 0.0)) >= float(_CR["stream_risk_elite_offense_wrc"].value):
        flags.append("ELITE_OFFENSE")

    park = float(start_info.get("park_factor") or 1.0)
    if park >= float(_CR["stream_risk_hitter_park"].value):
        flags.append("HITTER_PARK")

    weather = start_info.get("weather") or {}
    wind_mph = _get_num(weather, "wind_mph", 0.0) or 0.0
    if wind_out and wind_mph >= float(_CR["stream_risk_wind_out_mph"].value):
        flags.append("WIND_OUT")

    if str(start_info.get("confidence", "")).upper() == "LOW":
        flags.append("LOW_CONFIDENCE")

    return flags


# ── Scoring ──────────────────────────────────────────────────────────


def score_stream_candidate(
    pitcher_row: Any,
    start_info: dict[str, Any],
    opp_context: dict[str, Any],
    config: Any = None,
    category_weights: dict[str, float] | None = None,
    recent_form: dict | None = None,
    lineup_exposure: float | None = None,
) -> dict[str, Any]:
    """Score one (pitcher, scheduled start) as a streaming candidate.

    Composes the canonical engines into a 0-100 Stream Score:
    ``compute_pitcher_matchup_score`` (matchup), ``compute_streaming_value``
    (marginal SGP — owns the inverse-stat signs), ``compute_bayesian_stream_score``
    (expected line + win prob), park/weather helpers (environment), L14 form,
    and regressed PvB lineup exposure when supplied.

    Args:
        pitcher_row: Pool-row dict/Series (era/whip/k/w/ip [+ k_bb_pct,
            xfip, csw_pct, fip, throws, ip_per_start, gs] preferred).
        start_info: Per-start context: opponent, is_home, venue,
            park_factor, weather, confidence, num_starts, game_date.
        opp_context: Output of :func:`get_opponent_offense_context`.
        config: LeagueConfig (sgp_denominators); None falls back to
            streaming.py's registry-fed defaults.
        category_weights: Optional per-category multipliers (any key case).
        recent_form: ``{"l14": {...}}`` from game_day recent form.
        lineup_exposure: Regressed opposing-lineup wOBA delta vs league
            average (positive = dangerous lineup). None ⇒ neutral.

    Returns:
        {"stream_score", "components", "risk_flags", "expected_line",
         "net_sgp", "matchup_score", "win_probability"}
    """
    weights = _score_weights()
    is_home = bool(start_info.get("is_home", False))
    num_starts = int(start_info.get("num_starts", 1) or 1)
    park_factor = float(start_info.get("park_factor") or 1.0)
    sgp_denoms = _lower_keys(getattr(config, "sgp_denominators", None))
    cat_weights = _lower_keys(category_weights)

    # 1. Matchup (two_start.py 0-10 scale → [-1, 1]).
    pitcher_stats = {
        key: val
        for key in ("k_bb_pct", "xfip", "csw_pct", "era", "whip")
        if (val := _get_num(pitcher_row, key)) is not None
    }
    # team_strength carries k_pct in percent; the matchup scorer expects a frac.
    opp_stats = {
        "wrc_plus": float(opp_context.get("wrc_plus", _TEAM_NEUTRAL["wrc_plus"])),
        "k_pct": float(opp_context.get("k_pct", _TEAM_NEUTRAL["k_pct"])) / 100.0,
    }
    matchup_score = compute_pitcher_matchup_score(
        pitcher_stats,
        opponent_team_stats=opp_stats,
        park_factor=park_factor,
        is_home=is_home,
    )
    comp_matchup = _clamp((matchup_score - 5.0) / 5.0)

    # 2. Marginal SGP (canonical path — inverse-stat signs live in streaming.py).
    sv = compute_streaming_value(
        pitcher_row,
        weekly_games=num_starts,
        team_park_factor=park_factor,
        category_weights=cat_weights,
        sgp_denominators=sgp_denoms,
    )
    net_sgp = float(sv["net_value"])
    comp_sgp = _clamp(net_sgp)

    # 3. Recent form.
    comp_form = _form_component(pitcher_row, recent_form)

    # 4. Opposing-lineup / PvB exposure (day-of only; neutral when absent).
    if lineup_exposure is None:
        comp_lineup = 0.0
    else:
        # ±0.050 wOBA exposure spans the full component range; dangerous
        # lineups (positive exposure) push the component negative.
        comp_lineup = _clamp(-float(lineup_exposure) / 0.050)

    # 5. Environment (park × wind; dome ⇒ park-only).
    venue = str(start_info.get("venue") or "")
    comp_env, wind_out = _env_component(park_factor, start_info.get("weather"), venue)

    # 6. Win probability + expected line.
    era = _get_num(pitcher_row, "era", _TEAM_NEUTRAL["team_era"]) or _TEAM_NEUTRAL["team_era"]
    ip = _get_num(pitcher_row, "ip", 0.0) or 0.0
    k = _get_num(pitcher_row, "k", 0.0) or 0.0
    k9 = (k * 9.0 / ip) if ip > 0 else 7.5
    fip = _get_num(pitcher_row, "fip") or _get_num(pitcher_row, "xfip") or era
    bayes = compute_bayesian_stream_score(
        pitcher_era=era,
        pitcher_k9=k9,
        pitcher_fip=fip,
        opp_k_pct=float(opp_context.get("k_pct", _TEAM_NEUTRAL["k_pct"])) / 100.0,
        is_home=is_home,
        sgp_denominators=sgp_denoms,
    )
    win_prob = float(bayes["win_probability"])
    comp_winprob = _clamp((win_prob - 0.5) * 2.0)

    components = {
        "matchup": round(comp_matchup, 4),
        "sgp": round(comp_sgp, 4),
        "form": round(comp_form, 4),
        "lineup": round(comp_lineup, 4),
        "env": round(comp_env, 4),
        "winprob": round(comp_winprob, 4),
    }
    blended = sum(weights[name] * components[name] for name in components)
    stream_score = 100.0 / (1.0 + math.exp(-_SCORE_SIGMOID_K * blended))

    return {
        "stream_score": round(stream_score, 1),
        "components": components,
        "risk_flags": _risk_flags(pitcher_row, start_info, opp_context, wind_out),
        "expected_line": {
            "ip": float(bayes["expected_ip"]),
            "k": float(bayes["expected_k"]),
            "er": float(bayes["expected_er"]),
            "win_prob": win_prob,
        },
        "net_sgp": net_sgp,
        "matchup_score": float(matchup_score),
        "win_probability": win_prob,
    }


# ── Stream board ─────────────────────────────────────────────────────


def _fetch_schedule_for_date(target_date: str) -> list[dict[str, Any]]:
    """Fetch one day's schedule (with probables + status) via statsapi."""
    try:
        import statsapi

        return statsapi.schedule(start_date=target_date, end_date=target_date)
    except Exception:
        logger.warning("stream_analyzer: schedule fetch failed for %s", target_date, exc_info=True)
        return []


def _ensure_opponent_strength(
    team_strength: dict[str, dict[str, Any]],
    schedule: list[dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Fill team-strength gaps for every team appearing in *schedule*.

    Uses game_day.get_team_strength (session-cached); fetch failures leave
    the team absent so downstream falls back to the documented neutral line.
    Never mutates the caller's dict.
    """
    if not schedule:
        return team_strength
    abbrs: set[str] = set()
    for game in schedule:
        for side in ("home_name", "away_name"):
            abbr = team_name_to_abbr(str(game.get(side, "")), default="")
            if abbr:
                abbrs.add(abbr)
    missing = [a for a in abbrs if a not in team_strength]
    if not missing:
        return team_strength
    try:
        from src.game_day import get_team_strength
    except Exception:
        return team_strength
    for abbr in missing:
        try:
            entry = get_team_strength(abbr)
            if entry:
                team_strength[abbr] = entry
        except Exception:
            continue
    return team_strength


def _pool_name_index(pool: pd.DataFrame) -> dict[str, int]:
    """normalized player name → positional index into *pool*."""
    name_col = "player_name" if "player_name" in pool.columns else "name"
    index: dict[str, int] = {}
    for pos, raw in enumerate(pool[name_col].astype(str)):
        key = normalize_player_name(raw)
        if key and key not in index:
            index[key] = pos
    return index


def build_stream_board(
    ctx: Any,
    target_date: str,
    schedule: list[dict[str, Any]] | None = None,
    include_rostered: bool = False,
) -> pd.DataFrame:
    """One scored row per streamable (pitcher, start) on *target_date*.

    Rows whose game has started/finished (today only) are retained but marked
    non-actionable (status LOCKED/FINAL) — transparency over hiding. Pitchers
    rostered by other league teams are never streamable; the user's own SPs
    appear only when *include_rostered* is set (rostered=True).

    Args:
        ctx: OptimizerDataContext (duck-typed: player_pool,
            league_rostered_ids, user_roster_ids, team_strength,
            park_factors, weather, two_start_pitchers, recent_form, config,
            todays_schedule, category_weights).
        target_date: ISO date (YYYY-MM-DD), today through ~+7.
        schedule: Schedule rows for the date (statsapi.schedule shape).
            None ⇒ ctx.todays_schedule when target_date is today, else a
            live statsapi fetch (empty list on failure).
    """
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    if schedule is None:
        ctx_sched = getattr(ctx, "todays_schedule", None)
        if target_date == today and ctx_sched:
            schedule = ctx_sched
        else:
            schedule = _fetch_schedule_for_date(target_date)

    pool = getattr(ctx, "player_pool", None)
    if pool is None or len(pool) == 0 or not schedule:
        return pd.DataFrame()

    name_index = _pool_name_index(pool)
    rostered_ids = {int(i) for i in (getattr(ctx, "league_rostered_ids", None) or set())}
    user_ids = {int(i) for i in (getattr(ctx, "user_roster_ids", None) or [])}
    # ctx.team_strength only covers teams ON THE USER'S ROSTER
    # (shared_data_layer._load_team_strength) — stream opponents are
    # league-wide, so enrich missing teams here or every off-roster opponent
    # silently scores as a neutral 100 wRC+ / 22 K% (2026-06-10 live finding).
    team_strength = _ensure_opponent_strength(dict(getattr(ctx, "team_strength", None) or {}), schedule)
    park_factors = getattr(ctx, "park_factors", None) or {}
    weather_map = getattr(ctx, "weather", None) or {}
    two_start_ids = {int(i) for i in (getattr(ctx, "two_start_pitchers", None) or [])}
    recent_form_map = getattr(ctx, "recent_form", None) or {}
    config = getattr(ctx, "config", None)
    category_weights = getattr(ctx, "category_weights", None)

    rows: list[dict[str, Any]] = []
    census = {"games": 0, "probables": 0, "matched": 0, "rostered_excluded": 0}
    for game in schedule:
        if str(game.get("game_date", "")) != target_date:
            continue
        census["games"] += 1
        status_raw = str(game.get("status", "")).lower()
        locked = target_date == today and status_raw in LOCKED_GAME_STATUSES
        if locked:
            status = "FINAL" if status_raw in FINAL_GAME_STATUSES else "LOCKED"
        else:
            status = "PROBABLE"
        home_abbr = team_name_to_abbr(str(game.get("home_name", "")))
        away_abbr = team_name_to_abbr(str(game.get("away_name", "")))

        for side, pitcher_team, opp_team in (
            ("home", home_abbr, away_abbr),
            ("away", away_abbr, home_abbr),
        ):
            probable = str(game.get(f"{side}_probable_pitcher", "") or "").strip()
            if not probable:
                continue
            census["probables"] += 1
            pos = name_index.get(normalize_player_name(probable))
            if pos is None:
                continue
            census["matched"] += 1
            row = pool.iloc[pos]
            pid = _get_num(row, "player_id")
            pid = int(pid) if pid is not None else None

            rostered = pid is not None and pid in rostered_ids
            if rostered and not (include_rostered and pid in user_ids):
                census["rostered_excluded"] += 1
                continue

            throws = str(row.get("throws") or "") or None
            opp_context = get_opponent_offense_context(opp_team, throws, team_strength)
            num_starts = 2 if (pid is not None and pid in two_start_ids) else 1
            start_info = {
                "game_date": target_date,
                "opponent": opp_team,
                "is_home": side == "home",
                "venue": home_abbr,
                "park_factor": float(park_factors.get(home_abbr) or 1.0),
                "weather": weather_map.get(home_abbr) or {},
                "confidence": _confidence_tier(target_date),
                "num_starts": num_starts,
            }
            scored = score_stream_candidate(
                row,
                start_info,
                opp_context,
                config,
                category_weights=category_weights,
                recent_form=recent_form_map.get(pid) if pid is not None else None,
            )
            mlb_id = _get_num(row, "mlb_id")
            rows.append(
                {
                    "player_id": pid,
                    "mlb_id": int(mlb_id) if mlb_id is not None else None,
                    "player_name": str(row.get("player_name", row.get("name", probable))),
                    "team": pitcher_team,
                    "throws": throws or "",
                    "opponent": opp_team,
                    "is_home": side == "home",
                    "game_datetime": str(game.get("game_datetime", "") or ""),
                    "venue": home_abbr,
                    "park_factor": start_info["park_factor"],
                    "opp_wrc_plus": opp_context["wrc_plus"],
                    "opp_k_pct": opp_context["k_pct"],
                    "split_source": opp_context["split_source"],
                    "status": status,
                    "actionable": not locked,
                    "confidence": start_info["confidence"],
                    "num_starts": num_starts,
                    "stream_score": scored["stream_score"],
                    "net_sgp": scored["net_sgp"],
                    "matchup_score": scored["matchup_score"],
                    "expected_ip": scored["expected_line"]["ip"],
                    "expected_k": scored["expected_line"]["k"],
                    "expected_er": scored["expected_line"]["er"],
                    "win_probability": scored["win_probability"],
                    "whip": _get_num(row, "whip"),
                    "components": scored["components"],
                    "risk_flags": scored["risk_flags"],
                    "percent_owned": _get_num(row, "percent_owned"),
                    "rostered": rostered,
                }
            )

    board = pd.DataFrame(rows)
    if board.empty:
        board.attrs["census"] = census
        return board
    board = board.sort_values("stream_score", ascending=False).reset_index(drop=True)
    board.attrs["census"] = census
    return board


# ── Matchup Microscope: pitcher history + lineup exposure ────────────


def get_pitcher_vs_team_history(
    mlb_id: int,
    opp_team: str | None = None,
    venue: str | None = None,
    last_n: int = 10,
) -> pd.DataFrame:
    """Per-start game-log rows for a pitcher, newest first.

    Pulls the statsapi pitching gameLog (the backtest_runner idiom), parses
    IP via the canonical outs converter, and optionally filters to starts
    against *opp_team* and/or at *venue* (the home team's park: the pitcher's
    own team when home, the opponent when away). Empty DataFrame — never a
    raise — when statsapi is unavailable or the fetch fails.

    Columns: date, opponent, is_home, venue, ip, k, er, bb, h, w, l.
    """
    from src.live_stats import _ip_outs_to_decimal

    try:
        import statsapi

        result = statsapi.player_stat_data(int(mlb_id), group="pitching", type="gameLog", sportId=1)
    except Exception:
        logger.warning("stream_analyzer: pitching gameLog fetch failed for mlb_id=%s", mlb_id, exc_info=True)
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for entry in result.get("stats", []) or []:
        stats = entry.get("stats", entry) or {}
        opp_raw = entry.get("opponent") or {}
        opp_name = opp_raw.get("name", "") if isinstance(opp_raw, dict) else str(opp_raw)
        opp_abbr = team_name_to_abbr(opp_name) if opp_name else ""
        is_home = bool(entry.get("isHome", False))
        rows.append(
            {
                "date": str(entry.get("date", "")),
                "opponent": opp_abbr,
                "is_home": is_home,
                "venue": "" if is_home else opp_abbr,  # own park resolved by caller when home
                "ip": _ip_outs_to_decimal(stats.get("inningsPitched", "0")),
                "k": float(stats.get("strikeOuts", 0) or 0),
                "er": float(stats.get("earnedRuns", 0) or 0),
                "bb": float(stats.get("baseOnBalls", 0) or 0),
                "h": float(stats.get("hits", 0) or 0),
                "w": float(stats.get("wins", 0) or 0),
                "l": float(stats.get("losses", 0) or 0),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("date", ascending=False).reset_index(drop=True)
    if opp_team:
        df = df[df["opponent"] == opp_team].reset_index(drop=True)
    if venue:
        df = df[df["venue"] == venue].reset_index(drop=True)
    return df.head(last_n).reset_index(drop=True)


def aggregate_pitcher_history(history: pd.DataFrame) -> dict[str, float]:
    """Weighted aggregate of per-start rows: ERA = ER*9/IP, WHIP = (BB+H)/IP.

    Rate stats are computed from summed components, never averaged per-game
    (the house rate-stat aggregation rule).
    """
    if history is None or history.empty:
        return {}
    total_ip = float(history["ip"].sum())
    er = float(history["er"].sum())
    bb = float(history["bb"].sum())
    h = float(history["h"].sum())
    return {
        "games": int(len(history)),
        "ip": round(total_ip, 2),
        "k": float(history["k"].sum()),
        "w": float(history["w"].sum()),
        "l": float(history["l"].sum()),
        "er": er,
        "era": round(er * 9.0 / total_ip, 2) if total_ip > 0 else 0.0,
        "whip": round((bb + h) / total_ip, 2) if total_ip > 0 else 0.0,
    }


def compute_lineup_exposure(
    pitcher_id: int,
    batter_ids: list[int],
    pool: pd.DataFrame,
    pvb_data: dict[tuple[int, int], dict] | None = None,
) -> float | None:
    """Regressed opposing-lineup wOBA delta vs league average.

    For each opposing batter: start from his generic wOBA (pool ``xwoba``,
    league average when missing) and shrink any PvB-vs-this-pitcher sample
    toward it with the canonical 60-PA stabilization
    (``pvb_matchup_adjustment``). Positive return = the lineup profiles as
    dangerous for this pitcher; None = nothing to reason about (feeds the
    neutral lineup component).
    """
    if not batter_ids:
        return None
    from src.optimizer.matchup_adjustments import get_pvb_matchup_data, pvb_matchup_adjustment

    if pvb_data is None:
        try:
            pvb_data = get_pvb_matchup_data(batter_ids=list(batter_ids), pitcher_ids=[int(pitcher_id)])
        except Exception:
            logger.warning("stream_analyzer: PvB fetch failed", exc_info=True)
            pvb_data = {}

    league_woba = float(_CR["league_avg_woba"].value)
    generic_by_id: dict[int, float] = {}
    if pool is not None and len(pool) and "player_id" in pool.columns:
        for _, prow in pool.iterrows():
            pid = _get_num(prow, "player_id")
            if pid is None:
                continue
            xwoba = _get_num(prow, "xwoba")
            generic_by_id[int(pid)] = xwoba if xwoba is not None else league_woba

    wobas: list[float] = []
    for bid in batter_ids:
        generic = generic_by_id.get(int(bid), league_woba)
        sample = (pvb_data or {}).get((int(bid), int(pitcher_id)))
        if sample:
            wobas.append(
                pvb_matchup_adjustment(
                    batter_generic_woba=generic,
                    pvb_woba=float(sample.get("woba", generic)),
                    pvb_pa=int(sample.get("pa", 0) or 0),
                )
            )
        else:
            wobas.append(generic)

    if not wobas:
        return None
    return float(sum(wobas) / len(wobas) - league_woba)


# ── Week Planner ─────────────────────────────────────────────────────


def build_week_plan(
    ctx: Any,
    schedule: list[dict[str, Any]] | None = None,
    max_adds: int | None = None,
    base_weekly_ip: float | None = None,
    days_ahead: int = 7,
) -> dict[str, Any]:
    """Greedy streaming sequence for the remaining week, under the add budget.

    Scores every actionable FA start from today through *days_ahead* days
    (via :func:`build_stream_board`), then hands the candidates to the
    canonical greedy selector ``optimal_streaming_schedule`` — which dedups
    by pitcher and stops at the budget. The plan is advisory: FCFS waivers
    mean nothing can be reserved, so callers should re-plan on each render.

    Args:
        ctx: OptimizerDataContext (duck-typed; see build_stream_board).
        schedule: Multi-day schedule rows (statsapi.schedule shape). None ⇒
            one live fetch covering the window (empty on failure).
        max_adds: Budget override. Defaults to ctx.adds_remaining_this_week,
            then the canonical league limit.
        base_weekly_ip: Already-projected weekly IP from the current roster.
            When provided, the summary's ``under_floor`` flag compares
            base + planned IP against the Yahoo forfeit floor; None ⇒ flag
            is None (unknown baseline — never guessed).
        days_ahead: Window size in days (default one matchup week).

    Returns:
        {"plan": [entry, ...], "summary": {...}} — entries are
        optimal_streaming_schedule outputs enriched with game_date,
        stream_score, expected line, and num_starts.
    """
    from src.league_rules import WEEKLY_TRANSACTION_LIMIT
    from src.optimizer.streaming import optimal_streaming_schedule

    today = datetime.now(UTC)
    dates = [(today + timedelta(days=off)).strftime("%Y-%m-%d") for off in range(days_ahead)]

    if schedule is None:
        try:
            import statsapi

            schedule = statsapi.schedule(start_date=dates[0], end_date=dates[-1])
        except Exception:
            logger.warning("stream_analyzer: week schedule fetch failed", exc_info=True)
            schedule = []

    budget = max_adds
    if budget is None:
        budget = getattr(ctx, "adds_remaining_this_week", None)
    if budget is None:
        budget = WEEKLY_TRANSACTION_LIMIT
    budget = max(0, int(budget))

    candidates: list[dict[str, Any]] = []
    for date_str in dates:
        day_games = [g for g in schedule if str(g.get("game_date", "")) == date_str]
        if not day_games:
            continue
        board = build_stream_board(ctx, date_str, schedule=day_games)
        if board.empty:
            continue
        for _, row in board.iterrows():
            if not row["actionable"] or row["rostered"]:
                continue
            candidates.append(
                {
                    "player_name": row["player_name"],
                    "player_id": row["player_id"],
                    "team": row["team"],
                    "opponent": row["opponent"],
                    "game_date": date_str,
                    "net_value": float(row["net_sgp"]),
                    "stream_score": float(row["stream_score"]),
                    "expected_ip": float(row["expected_ip"]),
                    "expected_k": float(row["expected_k"]),
                    "num_starts": int(row["num_starts"]),
                    "confidence": row["confidence"],
                    "risk_flags": list(row["risk_flags"]),
                }
            )

    plan = optimal_streaming_schedule(candidates, max_adds=budget)
    plan.sort(key=lambda e: e.get("game_date", ""))

    from src.ip_tracker import MIN_IP, WEEKLY_TARGET

    ip_added = sum(e["expected_ip"] * e["num_starts"] for e in plan)
    k_added = sum(e["expected_k"] * e["num_starts"] for e in plan)
    under_floor: bool | None = None
    if base_weekly_ip is not None:
        under_floor = (float(base_weekly_ip) + ip_added) < MIN_IP

    return {
        "plan": plan,
        "summary": {
            "max_adds": budget,
            "n_planned": len(plan),
            "ip_added": round(ip_added, 1),
            "k_added": round(k_added, 1),
            "net_sgp_total": round(sum(e["net_value"] for e in plan), 2),
            "ip_target": WEEKLY_TARGET,
            "ip_floor": MIN_IP,
            "base_weekly_ip": base_weekly_ip,
            "under_floor": under_floor,
        },
    }


# ── Track Record: replay a past date ─────────────────────────────────


def replay_stream_date(
    ctx: Any,
    target_date: str,
    top_n: int = 5,
    schedule: list[dict[str, Any]] | None = None,
    actuals: dict[int, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Score a PAST date's board and compare against the actual lines.

    HEATER stores no point-in-time projections, so the replay board is scored
    with CURRENT data — the matchup facts (opponent, park, schedule) are
    historically exact, the form/projection inputs are a proxy. The result
    carries ``proxy_caveat=True`` and consumers MUST surface it.

    Args:
        ctx: OptimizerDataContext (duck-typed; see build_stream_board).
        target_date: Past ISO date to replay.
        top_n: How many top board picks to grade.
        schedule: Schedule rows for the date (None ⇒ statsapi fetch).
        actuals: mlb_id → game-log frame override (tests / pre-fetched);
            None ⇒ per-pitcher statsapi game-log fetches.

    Returns:
        {"board_then": top-N scored rows, "actuals": per-pick actual lines,
         "summary": weighted aggregate {era, whip, k, ip, qs_rate, games},
         "proxy_caveat": True}
    """
    board = build_stream_board(ctx, target_date, schedule=schedule)
    empty = {
        "board_then": board,
        "actuals": pd.DataFrame(),
        "summary": {},
        "proxy_caveat": True,
    }
    if board.empty:
        return empty

    top = board.head(top_n).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for _, pick in top.iterrows():
        mlb_id = pick.get("mlb_id")
        if mlb_id is None or (isinstance(mlb_id, float) and math.isnan(mlb_id)):
            continue
        mlb_id = int(mlb_id)
        if actuals is not None:
            history = actuals.get(mlb_id, pd.DataFrame())
        else:
            history = get_pitcher_vs_team_history(mlb_id, last_n=40)
        if history is None or history.empty or "date" not in history.columns:
            continue
        day = history[history["date"] == target_date]
        if day.empty:
            continue
        line = day.iloc[0]
        ip = float(line["ip"])
        er = float(line["er"])
        rows.append(
            {
                "player_name": pick["player_name"],
                "team": pick["team"],
                "opponent": pick["opponent"],
                "stream_score_then": float(pick["stream_score"]),
                "expected_k": float(pick["expected_k"]),
                "actual_ip": ip,
                "actual_k": float(line["k"]),
                "actual_er": er,
                "actual_bb": float(line["bb"]),
                "actual_h": float(line["h"]),
                "actual_w": float(line["w"]),
                "quality_start": bool(ip >= 6.0 and er <= 3.0),
            }
        )

    actuals_df = pd.DataFrame(rows)
    if actuals_df.empty:
        return {**empty, "board_then": top}

    total_ip = float(actuals_df["actual_ip"].sum())
    total_er = float(actuals_df["actual_er"].sum())
    total_bb_h = float(actuals_df["actual_bb"].sum() + actuals_df["actual_h"].sum())
    summary = {
        "games": int(len(actuals_df)),
        "ip": round(total_ip, 1),
        "k": float(actuals_df["actual_k"].sum()),
        "era": round(total_er * 9.0 / total_ip, 2) if total_ip > 0 else 0.0,
        "whip": round(total_bb_h / total_ip, 2) if total_ip > 0 else 0.0,
        "qs_rate": round(float(actuals_df["quality_start"].mean()), 2),
        "k_delta_vs_expected": round(float((actuals_df["actual_k"] - actuals_df["expected_k"]).mean()), 2),
    }
    return {
        "board_then": top,
        "actuals": actuals_df,
        "summary": summary,
        "proxy_caveat": True,
    }


# ── Matchup impact: with-vs-without the streamed start ───────────────


def compute_matchup_impact(
    my_totals: dict[str, float],
    opp_totals: dict[str, float],
    expected_line: dict[str, float],
    pitcher_whip: float,
    config: Any = None,
    num_starts: int = 1,
    team_ip: float | None = None,
    n_sims: int = 500,
) -> dict[str, Any] | None:
    """Project the LIVE matchup with vs without one streamed start.

    Runs the canonical ``estimate_h2h_win_probability`` (the same engine the
    Lineup Optimizer uses) on the current matchup totals, then again with the
    candidate's expected line folded in: K and W add directly (W at the
    start's win probability; L at ``(1 - win_prob) x stream_loss_decision_share``),
    ERA/WHIP recombine from components over ``team_ip + ip_added`` — never
    averaged. Both arms share one RNG seed (paired-MC discipline), so a no-op
    line yields exact-zero deltas.

    Args:
        my_totals / opp_totals: Live matchup category totals (lowercase keys,
            the ``ctx.my_totals`` shape). Empty ⇒ returns None — never guessed.
        expected_line: ``{"ip", "k", "er", "win_prob"}`` per start (the
            ``score_stream_candidate`` expected line).
        pitcher_whip: Candidate's projected WHIP (baserunner rate for the
            team-WHIP recombination).
        config: LeagueConfig (inverse/rate cat awareness lives downstream).
        num_starts: Starts in the window (two-start weeks scale the line).
        team_ip: My team's weekly IP so far/projected. None ⇒ the registry
            ``stream_ip_target`` (conservative full-week dilution).
        n_sims: Copula draws for the overall win-prob (paired seed).

    Returns:
        {"per_cat": {cat: {"before", "after", "delta"}},
         "expected_wins_before/after/delta",
         "overall_win_prob_before/after/delta"} or None.
    """
    if not my_totals or not opp_totals:
        return None

    import numpy as np

    from src.optimizer.h2h_engine import estimate_h2h_win_probability

    n = max(int(num_starts), 0)
    ip_added = float(expected_line.get("ip", 0.0) or 0.0) * n
    k_added = float(expected_line.get("k", 0.0) or 0.0) * n
    er_added = float(expected_line.get("er", 0.0) or 0.0) * n
    win_prob = float(expected_line.get("win_prob", 0.0) or 0.0)
    loss_share = float(_CR["stream_loss_decision_share"].value)

    if team_ip is None or team_ip <= 0:
        team_ip = float(_CR["stream_ip_target"].value)

    after = {str(c).lower(): float(v) for c, v in my_totals.items()}
    if "k" in after:
        after["k"] += k_added
    # No innings ⇒ no start happened ⇒ no decision can be charged.
    if ip_added > 0:
        if "w" in after:
            after["w"] += win_prob * n
        if "l" in after:
            after["l"] += (1.0 - win_prob) * loss_share * n
    if "era" in after and ip_added > 0:
        er_current = after["era"] * team_ip / 9.0
        after["era"] = (er_current + er_added) * 9.0 / (team_ip + ip_added)
    if "whip" in after and ip_added > 0:
        bb_h_current = after["whip"] * team_ip
        bb_h_added = max(float(pitcher_whip or 0.0), 0.0) * ip_added
        after["whip"] = (bb_h_current + bb_h_added) / (team_ip + ip_added)

    before_totals = {str(c).lower(): float(v) for c, v in my_totals.items()}
    opp = {str(c).lower(): float(v) for c, v in opp_totals.items()}

    # Paired arms: same seed so MC jitter cancels in the delta.
    seed = 20260610
    res_before = estimate_h2h_win_probability(before_totals, opp, n_sims=n_sims, rng=np.random.RandomState(seed))
    res_after = estimate_h2h_win_probability(after, opp, n_sims=n_sims, rng=np.random.RandomState(seed))

    per_cat: dict[str, dict[str, float]] = {}
    for cat, p_before in res_before["per_category"].items():
        p_after = res_after["per_category"].get(cat, p_before)
        per_cat[cat] = {
            "before": round(float(p_before), 4),
            "after": round(float(p_after), 4),
            "delta": float(p_after) - float(p_before),
        }

    return {
        "per_cat": per_cat,
        "expected_wins_before": float(res_before["expected_wins"]),
        "expected_wins_after": float(res_after["expected_wins"]),
        "expected_wins_delta": float(res_after["expected_wins"]) - float(res_before["expected_wins"]),
        "overall_win_prob_before": float(res_before["overall_win_prob"]),
        "overall_win_prob_after": float(res_after["overall_win_prob"]),
        "overall_win_prob_delta": float(res_after["overall_win_prob"]) - float(res_before["overall_win_prob"]),
    }

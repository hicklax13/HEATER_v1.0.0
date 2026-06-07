"""Post-optimization FA add/drop recommendation engine.

After the Line-up Optimizer runs (LP or DCV), this module evaluates whether
any free agent swaps would improve the roster given the current matchup state.
Enforces league rules: weekly add budget, closer minimum, IL stash
protection, and category-impact analysis.

Usage:
    ctx = build_optimizer_context("rest_of_week", yds, config)
    moves = recommend_fa_moves(ctx, max_moves=3)
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.optimizer.constants_registry import CONSTANTS_REGISTRY
from src.optimizer.shared_data_layer import OptimizerDataContext

# FA-engine overhaul P3.5 PR15 (2026-05-20): re-export the scarcity helpers
# from src.valuation under the old underscore-prefixed names so existing
# imports (`from src.optimizer.fa_recommender import _positional_scarcity_factor,
# _POSITIONAL_SCARCITY_MAX_BOOST`) — including the structural-invariant
# test in tests/test_fa_recommender_positional_scarcity.py — keep working
# without churn. The boost constant is unused inside this module but
# re-exported as a deliberate public alias.
from src.valuation import (
    POSITIONAL_SCARCITY_MAX_BOOST as _POSITIONAL_SCARCITY_MAX_BOOST,  # noqa: F401
)
from src.valuation import TEAM_NAME_TO_ABBR as _FULL_TO_ABBR
from src.valuation import _num as _num_safe
from src.valuation import (
    compute_positional_scarcity_factor as _positional_scarcity_factor,
)
from src.waiver_wire import (
    compute_drop_cost,
    compute_net_swap_value,
    compute_sustainability_score,
)

logger = logging.getLogger(__name__)


def _is_hitter_safe(val) -> bool:
    """Convert an is_hitter field to bool, defaulting to True (hitter) for NaN/None.

    pandas Series.get(key, default) returns NaN when the key exists but holds
    NaN — not the default.  Plain int(NaN) raises ValueError.  This helper
    centralises the guard so every bool(int(*.get("is_hitter",...))) call-site
    is protected without repetitive try/except noise.
    """
    if val is None:
        return True
    try:
        if pd.isna(val):
            return True
    except (TypeError, ValueError):
        pass
    try:
        return bool(int(val))
    except (TypeError, ValueError):
        return True


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# FA-engine overhaul P3 PR9 (2026-05-21): magic-constant centralization.
# Knobs that materially affect FA recommendations (thresholds, ratios,
# multipliers, IP goals) now read from CONSTANTS_REGISTRY at import time
# so calibration tools (sensitivity_analysis, sigmoid_calibrator) treat
# them as first-class inputs. Module-level names are preserved as aliases
# to keep the legacy call sites untouched.

_MIN_CLOSERS = 2
_CLOSER_SV_THRESHOLD = 5
_MAX_DROP_CANDIDATES = 5
_MAX_FA_CANDIDATES = 10
_CATEGORY_WORSEN_THRESHOLD = CONSTANTS_REGISTRY["category_worsen_threshold"].value
_MAX_WORSENED_CATEGORIES = 3
_HITTER_SLOTS = 10  # C/1B/2B/3B/SS/3OF/2Util = 10 starting hitter slots
_CROSS_TYPE_SGP_MIN = CONSTANTS_REGISTRY["cross_type_sgp_min"].value
_OWNERSHIP_BOOST_DELTA = CONSTANTS_REGISTRY["ownership_boost_delta"].value
_OWNERSHIP_BOOST_MULT = CONSTANTS_REGISTRY["ownership_boost_mult"].value
_FLOOR_PENALTY_MULT = CONSTANTS_REGISTRY["floor_penalty_mult"].value
_FLOOR_PA_MIN = CONSTANTS_REGISTRY["floor_pa_min"].value

# FA-engine overhaul P3.9 PR21 (2026-05-20): playing-time gate calibration.
# A player with 0 GP halfway into the season is almost always an IL stash
# phantom or inactive — their ROS projection (often built off preseason
# expectations of a full season) inflates their FA score above real
# performers. Apply a multiplicative penalty based on YTD playing time
# relative to season progress. Calibration locked by user 2026-05-20.
_PT_GATE_GRACE_DAYS = CONSTANTS_REGISTRY["pt_gate_grace_days"].value
_PT_RATIO_FULL_CREDIT = CONSTANTS_REGISTRY["pt_ratio_full_credit"].value
_PT_RATIO_MILD = CONSTANTS_REGISTRY["pt_ratio_mild"].value
# Below pt_ratio_mild of expected but > 0: 0.60x
_PT_MULT_ZERO_VOLUME = CONSTANTS_REGISTRY["pt_mult_zero_volume"].value
_PT_MULT_LOW = CONSTANTS_REGISTRY["pt_mult_low"].value
_PT_MULT_MILD = CONSTANTS_REGISTRY["pt_mult_mild"].value
_PT_HITTER_GP_PER_DAY = CONSTANTS_REGISTRY["pt_hitter_gp_per_day"].value
_PT_PITCHER_IP_PER_DAY = CONSTANTS_REGISTRY["pt_pitcher_ip_per_day"].value
_PT_TOTAL_SEASON_WEEKS = 26.0  # FourzynBurn season length (matches WEEKS_IN_SEASON; structural-invariant'd elsewhere)
_FLOOR_IP_MIN = CONSTANTS_REGISTRY["floor_ip_min"].value
_IL_EXCLUDE_STATUSES = {"il", "il10", "il15", "il60", "dtd", "day-to-day", "na", "out", "suspended"}

# Streaming knobs (for scope="today" only).
_STREAM_WIN_PROB_MIN = CONSTANTS_REGISTRY["stream_win_prob_min"].value
_STREAM_NET_SGP_MIN = CONSTANTS_REGISTRY["stream_net_sgp_min"].value
_STREAM_HURT_THRESHOLD = CONSTANTS_REGISTRY["stream_hurt_threshold"].value
_STREAM_CROSS_SIDE_RATIO = 0.5  # cross-swap if cross-worst < this * same-worst
_STREAM_DROP_TODAY_BONUS = 0.15  # protect rostered players with today game
_STREAM_MAX_PER_SIDE = 3  # cap per-side recommendations
_STREAM_IP_MIN = CONSTANTS_REGISTRY["stream_ip_min"].value
_STREAM_IP_TARGET = CONSTANTS_REGISTRY["stream_ip_target"].value
_STREAM_NET_SGP_RELAXED = CONSTANTS_REGISTRY["stream_net_sgp_relaxed"].value
_STREAM_IP_RELAX_RATIO = 0.75  # below this fraction of target, relax pitcher SGP bar

# FA-engine overhaul P3.5 PR16 (2026-05-20): roster-construction guard.
# FourzynBurn league roster construction has 10 starting hitters
# (1C/1·1B/1·2B/1·3B/1SS/3OF/2Util), 8 starting pitchers (2SP/2RP/4P),
# 5 BN, 4 IL. Position caps = starters + 1 (one backup beyond starter
# per position). Util/P are flex slots that overlap many positions, so
# they are intentionally excluded from the per-position caps (overlap
# would double-count). Drop floors = starting-lineup minimum counts —
# below these the user cannot field a full starting roster.
_POSITION_CAPS: dict[str, int] = {
    "C": 2,
    "1B": 2,
    "2B": 2,
    "3B": 2,
    "SS": 2,
    "OF": 4,
    "SP": 3,
    "RP": 3,
}
_MIN_ACTIVE_HITTERS = CONSTANTS_REGISTRY["min_active_hitters"].value
_MIN_ACTIVE_PITCHERS = CONSTANTS_REGISTRY["min_active_pitchers"].value

# FA P5f (2026-05-20): punt-category awareness. When a category has a
# very low matchup win probability OR is explicitly tagged as a punt in
# ctx.h2h_strategy, its weight in FA scoring drops to _PUNT_WEIGHT so
# the engine doesn't reward FAs whose value is concentrated in a
# category we're conceding. Near-zero (0.05) instead of exactly zero so
# downstream divide-by-zero paths in marginal_sgp stay safe.
_PUNT_WEIGHT = 0.05
_PUNT_WIN_PROB_THRESHOLD = 0.10


def _playing_time_multiplier(fa_data: pd.Series, ctx: OptimizerDataContext) -> float:
    """De-weight FAs whose YTD playing time is low vs season progress.

    Motivation (FA-engine overhaul P3.9 PR21, 2026-05-20): live diagnostic
    surfaced Jordan Westburg (0 YTD GP, post-wrist-surgery IL stash) ranked
    as top FA pickup because his ROS projection (~57 R / 16 HR for rest of
    season) was built off preseason expectations that ignore his IL status.
    Brandon Marsh (46 GP, .325 AVG — actually producing) was outranked by
    multiple 0-GP phantoms. This gate corrects the asymmetry.

    Calibration locked by user 2026-05-20:
      * Grace period: first ``_PT_GATE_GRACE_DAYS`` (30) of season inactive
      * 0 GP / 0 IP exactly: 0.30x multiplier
      * 1-29% of expected playing time: 0.60x (heavy)
      * 30-59% of expected: 0.85x (mild)
      * >=60% of expected: 1.0x (no penalty)

    Hitter signal: ``ytd_gp`` vs ``days_elapsed × 0.85`` (typical starter pace).
    Pitcher signal: ``ytd_ip`` vs ``days_elapsed × 1.0`` (blended SP+RP).

    Season progress derived from ``ctx.weeks_remaining`` against the
    FourzynBurn 26-week season length. A missing or non-numeric playing-
    time field is treated as 0 (matches industry: no data = IL/inactive).
    """
    weeks_remaining = float(getattr(ctx, "weeks_remaining", 16) or 16)
    days_elapsed = max(1.0, (_PT_TOTAL_SEASON_WEEKS - weeks_remaining) * 7.0)

    if days_elapsed < _PT_GATE_GRACE_DAYS:
        return 1.0  # First 30 days: no penalty (call-ups have no season yet)

    # NOTE: must not use `or 1` fallback here — pitchers have is_hitter=0
    # which is falsy in Python; `0 or 1 == 1` would mis-classify pitchers
    # as hitters. Default only when the key is missing.
    is_hitter_raw = fa_data.get("is_hitter", 1)
    if is_hitter_raw is None or (isinstance(is_hitter_raw, float) and pd.isna(is_hitter_raw)):
        is_hitter_raw = 1
    is_hitter = bool(int(is_hitter_raw))
    if is_hitter:
        try:
            ytd_volume = float(fa_data.get("ytd_gp", 0) or 0)
        except (TypeError, ValueError):
            ytd_volume = 0.0
        expected_volume = days_elapsed * _PT_HITTER_GP_PER_DAY
    else:
        try:
            ytd_volume = float(fa_data.get("ytd_ip", 0) or 0)
        except (TypeError, ValueError):
            ytd_volume = 0.0
        expected_volume = days_elapsed * _PT_PITCHER_IP_PER_DAY

    if ytd_volume <= 0:
        return _PT_MULT_ZERO_VOLUME

    ratio = ytd_volume / max(1.0, expected_volume)
    if ratio >= _PT_RATIO_FULL_CREDIT:
        return 1.0
    if ratio >= _PT_RATIO_MILD:
        return _PT_MULT_MILD
    return _PT_MULT_LOW


def _scale_ros_by_playing_time(fa_data: pd.Series, ctx: OptimizerDataContext) -> pd.Series:
    """Scale a player's ROS projection by their actual playing-time ratio.

    Motivation (FA P5c, 2026-05-20): a player with 0 YTD GP at Day 50 has
    the same ROS projection as a healthy starter — that's wrong because
    their projection assumes regular playing time they aren't getting (IL
    stash, demotion, platoon split, etc.). The PR21 ``_playing_time_multiplier``
    de-weights the composite score, but the ROS COLUMNS themselves still
    feed into ``_blend_fa_row`` and ``marginal_sgp`` at full preseason
    magnitude, so a Westburg-class 0-GP phantom keeps ROS=57R/16HR going
    into the blend. This function discounts those columns at the source.

    Calibration matches the existing playing-time gate philosophy:
      * Grace period: first ``_PT_GATE_GRACE_DAYS`` (30) days — no scaling
        (call-ups have no season yet).
      * Hitter signal: ``ytd_gp`` vs ``days_elapsed * _PT_HITTER_GP_PER_DAY``
        (typical starter pace).
      * Pitcher signal: ``ytd_ip`` vs ``days_elapsed * _PT_PITCHER_IP_PER_DAY``
        (blended SP+RP).
      * If ratio >= ``_PT_RATIO_FULL_CREDIT`` (0.6), return unchanged.
      * Below the threshold, multiply counting columns by ``max(0.2, ratio)``
        so an IL stash with 0 GP still gets a 20% floor (matches PR21's
        ``_PT_MULT_LOW`` magnitude philosophy — phantoms aren't worth zero,
        but they aren't worth full projection either).

    Returns a copy of ``fa_data`` with counting columns scaled. Rate
    stat columns (AVG/OBP/ERA/WHIP) are left untouched — they regenerate
    downstream from numerator/denominator counting columns.
    """
    weeks_remaining = float(getattr(ctx, "weeks_remaining", 16) or 16)
    days_elapsed = max(1.0, (_PT_TOTAL_SEASON_WEEKS - weeks_remaining) * 7.0)
    if days_elapsed < _PT_GATE_GRACE_DAYS:
        return fa_data  # grace period: no scaling

    # NOTE: same is_hitter handling as _playing_time_multiplier — pitchers
    # have is_hitter=0 which is falsy; default only when key is missing.
    is_hitter_raw = fa_data.get("is_hitter", 1)
    if is_hitter_raw is None or (isinstance(is_hitter_raw, float) and pd.isna(is_hitter_raw)):
        is_hitter_raw = 1
    try:
        is_hitter = bool(int(is_hitter_raw))
    except (TypeError, ValueError):
        is_hitter = True

    if is_hitter:
        try:
            ytd_volume = float(fa_data.get("ytd_gp", 0) or 0)
        except (TypeError, ValueError):
            ytd_volume = 0.0
        expected_volume = days_elapsed * _PT_HITTER_GP_PER_DAY
    else:
        try:
            ytd_volume = float(fa_data.get("ytd_ip", 0) or 0)
        except (TypeError, ValueError):
            ytd_volume = 0.0
        expected_volume = days_elapsed * _PT_PITCHER_IP_PER_DAY

    ratio = ytd_volume / max(1.0, expected_volume)
    if ratio >= _PT_RATIO_FULL_CREDIT:
        return fa_data  # full credit

    scale = max(0.2, ratio)

    # Cast to object dtype to avoid pandas FutureWarning when assigning
    # fractional results into int64 counting columns.
    scaled = fa_data.astype(object).copy()
    counting_cols = (
        "r",
        "hr",
        "rbi",
        "sb",
        "ab",
        "h",
        "bb",
        "hbp",
        "sf",
        "w",
        "l",
        "sv",
        "k",
        "ip",
        "er",
        "bb_allowed",
        "h_allowed",
    )
    for col in counting_cols:
        if col not in scaled.index:
            continue
        try:
            v = float(scaled[col] or 0)
            scaled[col] = v * scale
        except (TypeError, ValueError):
            pass
    return scaled


def _player_is_il(ctx: OptimizerDataContext, pid: int) -> bool:
    """Return True if the player is on IL/DTD/NA/suspended per roster or pool.

    Used to enforce slot-matching in swaps: IL drops can only pair with IL
    adds (both go through the 4-slot IL track), and healthy drops can only
    pair with healthy adds (both go through the 23-slot active track).
    Cross-track swaps would either overfill the active roster or leave
    an unusable IL slot sitting empty.
    """
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return False
    for source in (ctx.roster, ctx.player_pool):
        if source is None or source.empty or "player_id" not in source.columns:
            continue
        match = source[source["player_id"] == pid_int]
        if match.empty:
            continue
        status = str(match.iloc[0].get("status", "") or "").strip().lower()
        if status in _IL_EXCLUDE_STATUSES:
            return True
    return False


# ---------------------------------------------------------------------------
# Roster-construction guard (FA-engine overhaul P3.5 PR16)
# ---------------------------------------------------------------------------


def _parse_position_tokens(positions: str) -> set[str]:
    """Split a positions string (e.g. "2B,SS" or "2B/SS,Util") into a set
    of upper-case tokens. Used by the roster-construction guard to test
    membership against ``_POSITION_CAPS``."""
    if not positions:
        return set()
    return {p.strip().upper() for p in str(positions).replace("/", ",").split(",") if p.strip()}


def _count_eligible_at_position(
    roster_ids: list[int],
    pool: pd.DataFrame,
    position: str,
) -> int:
    """Return how many rostered players are eligible at ``position``.

    Counts ALL rostered players, INCLUDING IL — by design. Per the
    user spec: 2 catchers (Raleigh IL + Dingler) count as 2 toward
    the C cap, so a 3rd catcher add is blocked even though one is on
    the IL. Multi-position players count toward each of their positions.
    """
    if pool is None or pool.empty or "player_id" not in pool.columns:
        return 0
    target = position.strip().upper()
    if not target:
        return 0
    count = 0
    for pid in roster_ids:
        match = pool[pool["player_id"] == pid]
        if match.empty:
            continue
        tokens = _parse_position_tokens(str(match.iloc[0].get("positions", "")))
        if target in tokens:
            count += 1
    return count


def _count_active_by_side(
    roster_ids: list[int],
    pool: pd.DataFrame,
    is_hitter: bool,
) -> int:
    """Return how many ACTIVE (non-IL) rostered players match the side.

    Per the user spec: IL players do NOT count toward the active-roster
    floor — they cannot start. BN players DO count (they're flex healthy
    reserves, eligible to start). Status check matches ``_player_is_il``
    semantics so IL/DTD/NA/suspended/out all drop out.
    """
    if pool is None or pool.empty or "player_id" not in pool.columns:
        return 0
    count = 0
    for pid in roster_ids:
        match = pool[pool["player_id"] == pid]
        if match.empty:
            continue
        row = match.iloc[0]
        if _is_hitter_safe(row.get("is_hitter")) != is_hitter:
            continue
        status = str(row.get("status", "") or "").strip().lower()
        if status in _IL_EXCLUDE_STATUSES:
            continue
        count += 1
    return count


def _passes_roster_construction_guard(
    fa_data: pd.Series,
    drop_data: pd.Series,
    ctx: OptimizerDataContext,
) -> tuple[bool, str]:
    """Return ``(True, "")`` if the (add FA, drop) swap leaves a viable
    roster, otherwise ``(False, reason)`` so the caller can skip + log.

    Three checks against the POST-swap roster (drop removed, FA added):

    1. **Position cap**: for each of the FA's eligible positions in
       ``_POSITION_CAPS``, count rostered eligibility (incl. IL). If
       ALL of the FA's positions are at or above their cap, block.
       Multi-position FAs (e.g. 2B/SS) only block when EVERY eligible
       position is at cap — a 2B/SS can fill 2B even when SS is full.
       Util / P / "BN" positions are NOT in ``_POSITION_CAPS`` and are
       skipped entirely (flex slots overlap many positions).
    2. **Active hitter floor**: post-swap active hitter count must
       stay >= ``_MIN_ACTIVE_HITTERS`` (10 = FourzynBurn starting
       lineup count).
    3. **Active pitcher floor**: post-swap active pitcher count must
       stay >= ``_MIN_ACTIVE_PITCHERS`` (8 = starting lineup count).

    IL exclusion uses ``_IL_EXCLUDE_STATUSES`` semantics via
    ``_count_active_by_side``.
    """
    pool = ctx.player_pool
    if pool is None or pool.empty or "player_id" not in pool.columns:
        return True, ""

    try:
        drop_id = int(drop_data.get("player_id"))
        fa_id = int(fa_data.get("player_id"))
    except (TypeError, ValueError):
        return True, ""

    # Build POST-swap roster id list (drop removed, FA added).
    post_swap_ids = [int(pid) for pid in ctx.user_roster_ids if int(pid) != drop_id]
    if fa_id not in post_swap_ids:
        post_swap_ids.append(fa_id)

    # ── Check 1: position cap (FA must have at least one position at-
    # or-below cap; if ALL eligible positions are STRICTLY ABOVE cap → block).
    #
    # FA-engine overhaul P3.8 PR20 (2026-05-20): boundary semantics fix.
    # The pre-PR20 check `cnt < cap` (block at-or-above cap) was off-by-
    # one. Per the design, cap = "Yahoo starting slots + 1" represents the
    # MAX ALLOWED depth — having exactly `cap` players at a position is
    # healthy roster construction (starter + 1 backup). The block should
    # only fire when post-swap count STRICTLY EXCEEDS the cap, otherwise
    # legitimate same-position upgrade swaps get blocked (e.g. dropping
    # Muncy [2B-eligible] for Stott [2B] leaves post-swap 2B count = 2 =
    # cap, which is fine — the pre-fix engine wrongly blocked this).
    fa_positions = _parse_position_tokens(str(fa_data.get("positions", "")))
    capped_positions = [p for p in fa_positions if p in _POSITION_CAPS]
    if capped_positions:
        position_results: list[tuple[str, int, int]] = []
        any_at_or_below_cap = False
        for pos in capped_positions:
            cnt = _count_eligible_at_position(post_swap_ids, pool, pos)
            cap = _POSITION_CAPS[pos]
            position_results.append((pos, cnt, cap))
            if cnt <= cap:
                any_at_or_below_cap = True
                break
        if not any_at_or_below_cap:
            worst = position_results[0]
            return False, f"position-cap: {worst[0]}={worst[1]}>{worst[2]}"

    # ── Check 2: active hitter floor.
    n_hitters = _count_active_by_side(post_swap_ids, pool, is_hitter=True)
    if n_hitters < _MIN_ACTIVE_HITTERS:
        return False, f"below {_MIN_ACTIVE_HITTERS} active hitters (post-swap={n_hitters})"

    # ── Check 3: active pitcher floor.
    n_pitchers = _count_active_by_side(post_swap_ids, pool, is_hitter=False)
    if n_pitchers < _MIN_ACTIVE_PITCHERS:
        return False, f"below {_MIN_ACTIVE_PITCHERS} active pitchers (post-swap={n_pitchers})"

    return True, ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def recommend_fa_moves(
    ctx: OptimizerDataContext,
    max_moves: int = 3,
) -> list[dict[str, Any]]:
    """Recommend FA add/drop swaps that improve the roster.

    Parameters
    ----------
    ctx : OptimizerDataContext
        Fully-built optimizer data context from the shared data layer.
    max_moves : int
        Maximum number of moves to recommend.

    Returns
    -------
    list[dict]
        Recommended swaps sorted by net SGP delta descending.
        Each dict contains add/drop info, category impact, reasoning, etc.
    """
    # ── Weekly add budget check ───────────────────────────────────────
    if ctx.adds_remaining_this_week <= 0:
        logger.info("No adds remaining this week — skipping FA recommendations")
        return []

    if not ctx.user_roster_ids or ctx.player_pool.empty:
        return []

    effective_max = min(max_moves, ctx.adds_remaining_this_week)

    # FA-engine overhaul P3.5 PR15 (2026-05-20): compute replacement levels
    # ONCE up here and pass to BOTH scorers so add-side and drop-side
    # scarcity use the same dict (previously only the add-side computed
    # it inside `_score_fa_candidates`, so `compute_drop_cost` was
    # scarcity-blind — backup catchers looked free to drop). Defensive
    # try/except: a failure leaves replacement_levels=={} and both scorers
    # fall through to neutral 1.0× scarcity (pre-PR15 behavior).
    replacement_levels: dict[str, float] = {}
    try:
        from src.valuation import SGPCalculator, compute_replacement_levels

        _sgp_calc_for_repl = SGPCalculator(ctx.config)
        replacement_levels = compute_replacement_levels(ctx.player_pool, ctx.config, _sgp_calc_for_repl)
    except Exception:
        logger.debug("Could not compute replacement_levels — positional scarcity disabled this call")

    # ── Step 1: Score drop candidates ────────────────────────────────
    drop_candidates = _score_drop_candidates(ctx, replacement_levels)
    if not drop_candidates:
        logger.info("No viable drop candidates found")
        return []

    # ── Step 2: Score FA candidates ──────────────────────────────────
    fa_candidates = _score_fa_candidates(ctx, replacement_levels)
    if not fa_candidates:
        logger.info("No viable FA candidates found")
        return []

    # ── Step 3: Evaluate all (drop, add) pairs ───────────────────────
    swap_results = _evaluate_swaps(ctx, drop_candidates, fa_candidates)

    # ── Step 4: Deduplicate and limit ────────────────────────────────
    final = _deduplicate_and_limit(swap_results, effective_max)

    return final


# ---------------------------------------------------------------------------
# Drop candidate scoring
# ---------------------------------------------------------------------------


def _score_drop_candidates(
    ctx: OptimizerDataContext,
    replacement_levels: dict[str, float] | None = None,
) -> list[dict]:
    """Score rostered players as drop candidates.

    Hard filters:
    - Never drop IL stash players.
    - Never drop a closer if it would reduce closer count below minimum.

    FA-engine overhaul P3.5 PR15 (2026-05-20): ``replacement_levels`` is
    passed through to ``compute_drop_cost`` so dropping a scarce-position
    player (catcher, SS) costs MORE than dropping a deep-position player
    (OF, 1B) with equivalent raw SGP. Default ``None`` preserves the
    pre-PR15 backward-compat path (no scarcity multiplier on cost).
    """
    candidates: list[dict] = []

    for pid in ctx.user_roster_ids:
        # Hard filter: IL stash protection
        if pid in ctx.il_stash_ids:
            continue

        # Hard filter: closer minimum
        if _is_closer(pid, ctx) and ctx.closer_count <= _MIN_CLOSERS:
            continue

        cost = compute_drop_cost(
            pid,
            ctx.user_roster_ids,
            ctx.player_pool,
            ctx.config,
            replacement_levels=replacement_levels,
        )

        # Look up player info
        match = ctx.player_pool[ctx.player_pool["player_id"] == pid]
        if match.empty:
            continue
        row = match.iloc[0]

        candidates.append(
            {
                "player_id": pid,
                "name": str(row.get("name", row.get("player_name", "?"))),
                "positions": str(row.get("positions", "")),
                "is_hitter": _is_hitter_safe(row.get("is_hitter")),
                "drop_cost": cost,
                "is_il": _player_is_il(ctx, pid),
            }
        )

    # Sort by cost ascending (cheapest to drop first), take top N
    candidates.sort(key=lambda x: x["drop_cost"])
    return candidates[:_MAX_DROP_CANDIDATES]


# ---------------------------------------------------------------------------
# FA candidate scoring
# ---------------------------------------------------------------------------


# FA-engine overhaul P3.5 PR15 (2026-05-20): _positional_scarcity_factor
# and _POSITIONAL_SCARCITY_MAX_BOOST were hoisted to src.valuation so
# src.waiver_wire can also import them (waiver_wire is imported BY
# fa_recommender, so it can't import from fa_recommender — putting the
# function in valuation breaks the import cycle). The aliases at the top
# of this module preserve existing import paths so external callers and
# tests don't need to change.


def _score_fa_candidates(
    ctx: OptimizerDataContext,
    replacement_levels: dict[str, float] | None = None,
) -> list[dict]:
    """Score free agents by composite value.

    Filters out unhealthy players (health_score < 0.65 or IL/DTD/NA status).
    Scores by: base value × positional_scarcity_factor, urgency boost,
    ownership trend, sustainability, and floor preference.

    FA-engine overhaul P3.5 PR15 (2026-05-20): ``replacement_levels`` is
    now passed in from ``recommend_fa_moves`` (was previously computed
    locally inside this function). The hoist enables the SAME dict to
    feed the drop-side ``compute_drop_cost`` for symmetric scarcity
    treatment. If ``None``, falls back to a fresh local computation so
    standalone callers / tests still work; on any failure the dict stays
    empty and scarcity is disabled (1.0× neutral fallback).

    FA P5f (2026-05-20): punt-category awareness. When a category is
    explicitly punted in ``ctx.h2h_strategy`` or its ``win_prob`` is below
    ``_PUNT_WIN_PROB_THRESHOLD``, its ``ctx.category_weights`` entry is
    temporarily zeroed (to ``_PUNT_WEIGHT``) for the duration of this
    scoring call. This prevents the engine from rewarding FAs whose value
    is concentrated in a category we're conceding (e.g. a Trea Turner-class
    SB specialist looking strong to a user punting SB). Restored on exit
    via try/finally so the broader context is not mutated.
    """
    if ctx.free_agents.empty:
        return []

    # FA-engine overhaul P1 PR3 / P3.5 PR15: replacement levels now passed
    # in by caller (recommend_fa_moves) so both add-side and drop-side
    # scarcity see the same dict. For standalone callers that pass None,
    # compute locally as a defensive fallback.
    if replacement_levels is None:
        _replacement_levels: dict[str, float] = {}
        try:
            from src.valuation import SGPCalculator, compute_replacement_levels

            _sgp_calc_for_repl = SGPCalculator(ctx.config)
            _replacement_levels = compute_replacement_levels(ctx.player_pool, ctx.config, _sgp_calc_for_repl)
        except Exception:
            logger.debug("Could not compute replacement_levels — positional scarcity factor disabled this call")
    else:
        _replacement_levels = replacement_levels

    # FA P5f (2026-05-20): punt-category detection. Build punt set from
    # explicit h2h_strategy tags AND any category with win_prob below the
    # threshold. Apply the override to ctx.category_weights for the scoring
    # loop; restore on exit.
    #
    # FA-C3 (2026-06-07): the AUTHORITATIVE key is 'punt_cats'. ctx.h2h_strategy
    # is populated solely by shared_data_layer._load_h2h_strategy from
    # weekly_h2h_strategy.compute_weekly_matchup_state, which emits 'punt_cats'
    # (the same key shared_data_layer._build_unified_category_weights reads).
    # The bare 'punt' key is kept as a defensive fallback for any future/external
    # producer, but no current producer emits it.
    _punt_cats: set[str] = set()
    _strategy = getattr(ctx, "h2h_strategy", {}) or {}
    for _key in ("punt_cats", "punt"):
        for c in _strategy.get(_key, []) or []:
            if c:
                _punt_cats.add(str(c).upper())
    _per_cat = (ctx.urgency_weights or {}).get("per_cat", {}) or {}
    for cat, info in _per_cat.items():
        try:
            wp = float((info or {}).get("win_prob", 0.5))
        except (TypeError, ValueError):
            wp = 0.5
        if wp < _PUNT_WIN_PROB_THRESHOLD:
            _punt_cats.add(str(cat).upper())

    _original_weights: dict[str, float] | None = None
    if _punt_cats and ctx.category_weights:
        _original_weights = dict(ctx.category_weights)
        _modified_weights = dict(ctx.category_weights)
        for cat in _punt_cats:
            # category_weights may be keyed upper or lower depending on caller.
            if cat in _modified_weights:
                _modified_weights[cat] = _PUNT_WEIGHT
            if cat.lower() in _modified_weights:
                _modified_weights[cat.lower()] = _PUNT_WEIGHT
        ctx.category_weights = _modified_weights
        # Invalidate cached SGPCalculator state derived from category_weights
        # so the per-category weights are re-read on the next _compute_base_value.
        # (_fa_roster_totals is independent of weights, so we only clear if
        # the underlying calc memoizes weight-dependent state. Currently
        # SGPCalculator.marginal_sgp accepts category_weights per-call, so
        # the cached calc instance is safe to keep — but invalidate defensively
        # to avoid future drift.)

    candidates: list[dict] = []

    # Build set of ALL rostered player IDs (user + opponents) for exclusion.
    # ctx.user_roster_ids covers the user's team; ctx.league_rostered_ids
    # (populated by build_optimizer_context from league_rosters table)
    # covers all other teams. Tests pass an empty set to opt out.
    _excluded_ids: set[int] = set(int(pid) for pid in ctx.user_roster_ids)
    _excluded_ids.update(int(pid) for pid in ctx.league_rostered_ids)

    # FA P5f (2026-05-20): try/finally wrapping the scoring loop so the
    # caller-side category_weights override is always restored even if
    # an exception propagates through marginal_sgp / sustainability.
    try:
        for _, fa_row in ctx.free_agents.iterrows():
            fa_id = fa_row.get("player_id")
            if fa_id is None or pd.isna(fa_id):
                continue
            fa_id = int(fa_id)

            # Roster guard: skip players already on any team
            if fa_id in _excluded_ids:
                continue

            # Identify IL/injured FAs — don't exclude them, just flag. IL FAs
            # are valid pairs for IL drops (upgrading an IL stash), but must
            # not be matched with healthy drops. The matching is enforced
            # downstream in _evaluate_swaps.
            status = str(fa_row.get("status", "")).lower().strip()
            pool_match = ctx.player_pool[ctx.player_pool["player_id"] == fa_id]
            if not pool_match.empty:
                pool_status = str(pool_match.iloc[0].get("status", "")).lower().strip()
                if pool_status and pool_status != "active":
                    status = pool_status
                fa_data = pool_match.iloc[0]
            else:
                fa_data = fa_row
            health = ctx.health_scores.get(fa_id, 1.0)
            fa_is_il = status in _IL_EXCLUDE_STATUSES or health < 0.65

            # Base value × positional scarcity (FA-engine overhaul P1 PR3).
            # Top-2 catcher with raw SGP X is more valuable to the team than
            # the 25th-best OF with raw SGP X — the catcher pool is shallow.
            base_value = _compute_base_value(fa_data, ctx)
            _scarcity_mult = _positional_scarcity_factor(str(fa_data.get("positions", "")), _replacement_levels)
            base_value *= _scarcity_mult

            # FA-engine overhaul P2 PR6 (2026-05-20): urgency is now applied
            # MULTIPLICATIVELY inside _compute_base_value via per-category weights
            # (ctx.category_weights) rather than additively here. The old
            # `composite = base_value * ... + urgency_boost` mixed scales — the
            # additive boost summed ctx.category_weights across all relevant
            # categories (4-6 cats × ~0.5-1.8 per cat = boost 2-12), which
            # dwarfed base_value for marginal FAs (often 0.5-2 SGP) and rewarded
            # FAs touching MANY categories over ones with concentrated value.
            # Keeping the additive term turned at zero preserves backward
            # compatibility of the variable name in the composite line below;
            # the multiplicative weighting in _compute_base_value is the
            # authoritative urgency signal now. _compute_urgency_boost stays
            # callable for legacy paths but is no longer added to composite.
            urgency_boost = 0.0

            # Ownership trend boost
            ownership_mult = 1.0
            trend = ctx.ownership_trends.get(fa_id, {})
            delta_7d = trend.get("delta_7d", 0.0)
            if delta_7d > _OWNERSHIP_BOOST_DELTA:
                ownership_mult = _OWNERSHIP_BOOST_MULT

            # Sustainability
            sustainability = compute_sustainability_score(fa_data)

            # Floor preference penalty
            floor_mult = 1.0
            is_hitter = _is_hitter_safe(fa_data.get("is_hitter"))
            if is_hitter:
                pa = float(fa_data.get("pa", 0) or 0)
                if pa < _FLOOR_PA_MIN:
                    floor_mult = _FLOOR_PENALTY_MULT
            else:
                ip = float(fa_data.get("ip", 0) or 0)
                if ip < _FLOOR_IP_MIN:
                    floor_mult = _FLOOR_PENALTY_MULT

            # FA-engine overhaul P3.9 PR21 (2026-05-20): playing-time gate.
            # De-weight FAs whose YTD playing time is well below season-progress
            # expectations (IL stash phantoms with inflated preseason ROS but
            # near-zero actual games). See `_playing_time_multiplier` docstring
            # for full calibration rationale.
            pt_mult = _playing_time_multiplier(fa_data, ctx)

            # PR10 Part A (2026-05-20): regression flag adjustment.
            # The pool's regression_flag column (BUY_LOW / SELL_HIGH / empty) was
            # being loaded but never consumed. Wire it as a final ranking nudge:
            # BUY_LOW = 1.05x (engine slightly favors regression-favorable players);
            # SELL_HIGH = 0.95x (slight discount). Industry consensus: regression
            # signals from xwOBA-wOBA gap and similar metrics are 5-10% accurate
            # over a single matchup, so a 5% multiplier is the right order of
            # magnitude — neither dominates nor disappears.
            _reg_flag = str(fa_data.get("regression_flag", "") or "").strip().upper()

            composite = base_value * sustainability * ownership_mult * floor_mult * pt_mult + urgency_boost

            # Sign-aware regression nudge: BUY_LOW always moves composite toward
            # positive (better ranking), SELL_HIGH always moves toward negative.
            # Naïve ×1.05 inverts direction when composite < 0 — use reciprocal.
            if _reg_flag == "BUY_LOW":
                composite *= 1.05 if composite >= 0 else (1.0 / 1.05)
            elif _reg_flag == "SELL_HIGH":
                composite *= 0.95 if composite >= 0 else (1.0 / 0.95)

            # T3-4: ECR stddev consensus adjustment
            try:
                _ecr_stddev = float(fa_data.get("ecr_rank_stddev", 0) or 0)
                if _ecr_stddev > 20:
                    composite *= 0.95  # Polarizing pick — small discount
                elif 0 < _ecr_stddev < 5:
                    composite *= 1.02  # Consensus pick — small premium
            except (TypeError, ValueError):
                pass

            # Ownership trend label
            pct_owned = trend.get("pct_owned", 0.0)
            if delta_7d > _OWNERSHIP_BOOST_DELTA:
                trend_label = f"Rising ({pct_owned:.0f}%, +{delta_7d:.1f}%)"
            elif delta_7d < -_OWNERSHIP_BOOST_DELTA:
                trend_label = f"Falling ({pct_owned:.0f}%, {delta_7d:.1f}%)"
            else:
                trend_label = f"Stable ({pct_owned:.0f}%)"

            candidates.append(
                {
                    "player_id": fa_id,
                    "name": str(fa_data.get("name", fa_data.get("player_name", "?"))),
                    "positions": str(fa_data.get("positions", "")),
                    "is_hitter": is_hitter,
                    "composite_score": composite,
                    "sustainability": round(sustainability, 3),
                    "ownership_trend": trend_label,
                    "ownership_delta_7d": delta_7d,
                    "is_il": fa_is_il,
                }
            )

        # Sort by composite descending, take top N
        candidates.sort(key=lambda x: x["composite_score"], reverse=True)
        return candidates[:_MAX_FA_CANDIDATES]
    finally:
        # FA P5f: restore original category_weights even on exception path.
        if _original_weights is not None:
            ctx.category_weights = _original_weights


# ---------------------------------------------------------------------------
# Swap evaluation
# ---------------------------------------------------------------------------


def _evaluate_swaps(
    ctx: OptimizerDataContext,
    drop_candidates: list[dict],
    fa_candidates: list[dict],
) -> list[dict]:
    """Evaluate all (drop, add) pairs and filter by league rules."""
    results: list[dict] = []
    losing_cats = _get_losing_categories(ctx)
    tied_cats = _get_tied_categories(ctx)
    target_cats = set(losing_cats) | set(tied_cats)

    for fa in fa_candidates:
        fa_id = fa["player_id"]
        fa_is_hitter = fa["is_hitter"]

        for drop in drop_candidates:
            drop_id = drop["player_id"]
            drop_is_hitter = drop["is_hitter"]

            # IL/active matching: in Yahoo, IL and active slots are separate
            # pools (4 IL + 23 active). A 1-for-1 swap can only move within
            # one pool — dropping IL frees an IL slot, dropping active frees
            # an active slot. Cross-pool swaps would either overfill the
            # active roster or leave an empty IL slot. IL-only FAs are still
            # valid targets when paired with an IL drop (upgrade stash).
            if bool(drop.get("is_il", False)) != bool(fa.get("is_il", False)):
                continue

            # Same-type check (cross-type only when surplus + improves losing/tied cat)
            if fa_is_hitter != drop_is_hitter:
                if not _allow_cross_type(ctx, fa, drop, target_cats):
                    continue

            # Compute net swap value
            swap = compute_net_swap_value(fa_id, drop_id, ctx.user_roster_ids, ctx.player_pool, ctx.config)

            net_sgp = swap["net_sgp"]
            cat_deltas = swap["category_deltas"]

            # Skip non-positive swaps
            if net_sgp <= 0:
                continue

            # FA-engine overhaul P3.5 PR16 (2026-05-20): roster-construction
            # guard. Without this the engine happily recommends a 3rd C when
            # the user has 2 (Raleigh IL + Dingler), or drops the 2nd SP
            # when only 8 healthy starters remain. Logged at DEBUG to avoid
            # noisy WARNINGs on every blocked swap.
            passes_construction, _block_reason = _passes_roster_construction_guard(fa, drop, ctx)
            if not passes_construction:
                logger.debug(
                    "Roster-construction guard blocked %s → %s: %s",
                    drop.get("name", drop.get("player_id")),
                    fa.get("name", fa.get("player_id")),
                    _block_reason,
                )
                continue

            # Category worsening check (informational — log but don't block)
            sum(1 for v in cat_deltas.values() if v < _CATEGORY_WORSEN_THRESHOLD)
            # Note: previously auto-rejected when worsened >= 3; now uses net SGP only

            # Build reasoning
            reasoning = _build_reasoning(fa, drop, swap, ctx)

            # Urgency categories: which losing/tied cats does this swap help?
            urgency_cats = [cat for cat in target_cats if cat_deltas.get(cat, 0) > 0.01]

            # News warning
            news_warning = ctx.news_flags.get(fa_id)

            results.append(
                {
                    "add_id": fa_id,
                    "add_name": fa["name"],
                    "add_positions": fa["positions"],
                    "drop_id": drop_id,
                    "drop_name": drop["name"],
                    "drop_positions": drop["positions"],
                    "net_sgp_delta": round(net_sgp, 4),
                    "category_impact": {k: round(v, 4) for k, v in cat_deltas.items()},
                    "reasoning": reasoning,
                    "urgency_categories": urgency_cats,
                    "news_warning": news_warning,
                    "ownership_trend": fa["ownership_trend"],
                    "sustainability": fa["sustainability"],
                }
            )

    # Sort by net SGP descending
    results.sort(key=lambda x: x["net_sgp_delta"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Deduplication and limiting
# ---------------------------------------------------------------------------


def _deduplicate_and_limit(results: list[dict], max_moves: int) -> list[dict]:
    """Deduplicate so each FA and each drop candidate is used at most once."""
    used_adds: set[int] = set()
    used_drops: set[int] = set()
    final: list[dict] = []

    for swap in results:
        if len(final) >= max_moves:
            break
        add_id = swap["add_id"]
        drop_id = swap["drop_id"]
        if add_id in used_adds or drop_id in used_drops:
            continue
        used_adds.add(add_id)
        used_drops.add(drop_id)
        final.append(swap)

    return final


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_closer(player_id: int, ctx: OptimizerDataContext) -> bool:
    """Check if a player is a closer (is_closer flag or projected SV >= 5)."""
    if ctx.player_pool.empty:
        return False
    match = ctx.player_pool[ctx.player_pool["player_id"] == player_id]
    if match.empty:
        return False
    row = match.iloc[0]
    if row.get("is_closer", False):
        return True
    sv = float(row.get("sv", 0) or 0)
    return sv >= _CLOSER_SV_THRESHOLD


# FA-engine overhaul P2 PR4 (2026-05-20) + FA P5b PR (2026-05-20): canonical
# published blend weights (Smart Fantasy Baseball, The Athletic, FanGraphs
# methodology).
#   0.70 × ROS projection  (anchor — projection systems still beat YTD)
#   0.20 × YTD pace        (in-season actual)
#   0.10 × last-14-day     (recent form, wired from l14_* row columns)
# When L14 data is absent or below the per-side volume gate (hitter l14_pa
# >= 20, pitcher l14_ip >= 5), the weights renormalize so the remaining
# components still sum to 1.0 (preserves relative ROS/YTD ratio). PR21 P5b
# replaced the previous "L14 NOT YET WIRED" placeholder with the active
# blend below.
_BLEND_WEIGHT_ROS = 0.70
_BLEND_WEIGHT_YTD = 0.20
_BLEND_WEIGHT_L14 = 0.10
_BLEND_YTD_MIN_GAMES = 30  # below this, YTD sample too noisy — skip blend
_BLEND_L14_MIN_PA = 20  # hitter L14 volume gate
_BLEND_L14_MIN_IP = 5.0  # pitcher L14 volume gate
_BLEND_L14_DEFAULT_GAMES = 14  # used when l14_games column is absent

# Counting stats that should be blended (hitting + pitching). Rate stats
# (AVG/OBP/ERA/WHIP) are derived from sum totals downstream — once the
# numerator + denominator columns blend, the rates blend implicitly.
_BLENDABLE_COUNTING_COLS = (
    "r",
    "hr",
    "rbi",
    "sb",
    "ab",
    "h",
    "bb",
    "hbp",
    "sf",
    "w",
    "l",
    "sv",
    "k",
    "ip",
    "er",
    "bb_allowed",
    "h_allowed",
)


def _blend_fa_row(fa_data: pd.Series, l14_form: dict | None = None) -> pd.Series:
    """Blend a player's stat row from ROS projection + YTD pace + L14 recent form.

    Industry consensus (Marcel-style blending, Smart Fantasy Baseball's
    'how much do current-season stats matter' research): trust the
    projection system as the anchor, layer in YTD pace as a sample-size-
    gated correction, mix in L14 for short-term form. Canonical weights
    are 0.70 ROS / 0.20 YTD / 0.10 L14.

    L14 wiring (FA P5b, 2026-05-20): L14 is read from ``l14_*`` row columns
    when present (l14_hr, l14_r, l14_k, l14_ip, l14_pa, etc.), OR from the
    ``l14_form`` recent-form dict (FA-C2, 2026-06-07) — the live FA path, since
    the FA pool carries no l14_* columns. The dict wins when present. Per-side
    volume gate: hitters need ``l14_pa >= _BLEND_L14_MIN_PA`` (20),
    pitchers need ``l14_ip >= _BLEND_L14_MIN_IP`` (5). Below the gate
    L14 is skipped and the remaining weights renormalize so the blend
    still sums to 1.0.

    Returns a copy of ``fa_data`` with counting-stat columns replaced
    by the blended values. Rate stat columns (AVG/OBP/ERA/WHIP) are
    NOT directly blended — they regenerate downstream from the blended
    h/ab/er/ip/etc. Sample-size gating: below ``_BLEND_YTD_MIN_GAMES``
    YTD games, no blend is applied (pure ROS — small samples are too
    noisy to mix in).
    """
    ytd_gp = float(fa_data.get("ytd_gp", 0) or 0)
    if ytd_gp < _BLEND_YTD_MIN_GAMES:
        return fa_data  # insufficient YTD data — use pure ROS

    # Per-game basis lets us compare ROS projection (rate-of-season-
    # remaining) against YTD (rate-of-season-played).
    games_remaining = max(1.0, 162.0 - ytd_gp)

    # ── L14 volume gate (FA P5b, 2026-05-20) ────────────────────────────
    # Determine whether L14 has enough volume to contribute. Use is_hitter
    # to pick the right gate: hitters → l14_pa, pitchers → l14_ip.
    is_hitter_raw = fa_data.get("is_hitter", 1)
    if is_hitter_raw is None or (isinstance(is_hitter_raw, float) and pd.isna(is_hitter_raw)):
        is_hitter_raw = 1
    try:
        is_hitter = bool(int(is_hitter_raw))
    except (TypeError, ValueError):
        is_hitter = True

    # FA-C2 (2026-06-07): unify the L14 signal from two sources — row l14_*
    # columns (back-compat / tests) and the recent-form dict (l14_form, from
    # ctx.recent_form / get_player_recent_form_cached, the live FA path). The
    # dict (keyed by bare stat name: pa/ip/games/hr/r/rbi/sb/k/...) wins when present.
    l14_lookup: dict = {}
    for _c in fa_data.index:
        if isinstance(_c, str) and _c.startswith("l14_"):
            l14_lookup[_c[4:]] = fa_data[_c]
    if l14_form:
        for _k, _v in l14_form.items():
            if _v is not None:
                l14_lookup[_k] = _v

    if is_hitter:
        l14_volume = _num_safe(l14_lookup.get("pa", 0))
        l14_active = l14_volume >= _BLEND_L14_MIN_PA
    else:
        l14_volume = _num_safe(l14_lookup.get("ip", 0))
        l14_active = l14_volume >= _BLEND_L14_MIN_IP

    # L14 sample size in games (for per-game projection). Use the L14 games
    # count if surfaced, else default to 14 (the canonical window length).
    l14_games = _num_safe(l14_lookup.get("games", 0))
    if l14_games <= 0:
        l14_games = float(_BLEND_L14_DEFAULT_GAMES)

    # Renormalize weights based on which components are active. ROS+YTD
    # are always active here (we passed the ytd_gp gate); L14 is gated.
    if l14_active:
        total = _BLEND_WEIGHT_ROS + _BLEND_WEIGHT_YTD + _BLEND_WEIGHT_L14
        ros_weight = _BLEND_WEIGHT_ROS / total
        ytd_weight = _BLEND_WEIGHT_YTD / total
        l14_weight = _BLEND_WEIGHT_L14 / total
    else:
        total = _BLEND_WEIGHT_ROS + _BLEND_WEIGHT_YTD
        ros_weight = _BLEND_WEIGHT_ROS / total
        ytd_weight = _BLEND_WEIGHT_YTD / total
        l14_weight = 0.0

    # Cast blended series to float-friendly dtype to avoid pandas FutureWarning
    # when assigning fractional values back into int64-typed columns.
    blended = fa_data.astype(object).copy()
    for col in _BLENDABLE_COUNTING_COLS:
        if col not in fa_data.index:
            continue
        ros_val = _num_safe(fa_data.get(col, 0))
        ytd_col = f"ytd_{col}"
        if ytd_col not in fa_data.index:
            continue
        ytd_val = _num_safe(fa_data.get(ytd_col, 0))

        # Both rates normalized to per-game so they can be combined.
        ros_per_game = ros_val / games_remaining if games_remaining > 0 else 0.0
        ytd_per_game = ytd_val / ytd_gp if ytd_gp > 0 else 0.0

        # L14 contribution — only when l14_active AND this specific l14_*
        # column is present on the row. A missing per-stat l14 column
        # means we have no signal for that stat; redistribute its weight
        # to ROS+YTD proportionally so the per-row blend still sums to 1.
        if l14_active and col in l14_lookup:
            l14_val = _num_safe(l14_lookup.get(col, 0))
            l14_per_game = l14_val / l14_games if l14_games > 0 else 0.0
            blended_per_game = ros_weight * ros_per_game + ytd_weight * ytd_per_game + l14_weight * l14_per_game
        else:
            # No L14 for this column — renormalize ROS/YTD only for this stat.
            local_total = _BLEND_WEIGHT_ROS + _BLEND_WEIGHT_YTD
            local_ros = _BLEND_WEIGHT_ROS / local_total
            local_ytd = _BLEND_WEIGHT_YTD / local_total
            blended_per_game = local_ros * ros_per_game + local_ytd * ytd_per_game

        # Project blended per-game pace back to ROS-equivalent total so
        # downstream marginal_sgp math sees the same shape it expects.
        blended[col] = blended_per_game * games_remaining

    return blended


def _resolve_fa_l14(fa_data: pd.Series, ctx: OptimizerDataContext) -> dict | None:
    """Resolve a free agent's last-14-day form dict for the blend (FA-C2).

    The FA pool carries no ``l14_*`` columns, so the 0.10 L14 blend term was dead
    for free agents. Source it from recent form instead: prefer a pre-loaded
    ``ctx.recent_form[pid]['l14']`` entry; else lazily fetch via
    ``get_player_recent_form_cached`` (the same 2h-cached source the optimizer's
    projections use) — bounded by the scored-candidate set and deduped by the
    cache. Returns the l14 dict (keys games/pa/ip/hr/r/rbi/sb/k/era/whip/...) or
    ``None`` (→ the blend renormalizes to ROS+YTD, prior behavior).
    """
    rf = getattr(ctx, "recent_form", None)
    pid = fa_data.get("player_id")
    if rf and pid is not None and not (isinstance(pid, float) and pd.isna(pid)):
        try:
            entry = rf.get(int(pid))
        except (TypeError, ValueError):
            entry = None
        if isinstance(entry, dict):
            l14 = entry.get("l14")
            if isinstance(l14, dict) and l14:
                return l14
    mlb = fa_data.get("mlb_id")
    if mlb is not None and not (isinstance(mlb, float) and pd.isna(mlb)):
        try:
            from src.game_day import get_player_recent_form_cached

            form = get_player_recent_form_cached(int(mlb))
            if isinstance(form, dict):
                l14 = form.get("l14")
                if isinstance(l14, dict) and l14:
                    return l14
        except Exception:
            return None
    return None


def _compute_base_value(fa_data: pd.Series, ctx: OptimizerDataContext) -> float:
    """Compute base value for an FA candidate.

    FA-engine overhaul P3.7 PR19 (2026-05-20): delegates to the canonical
    SGPCalculator.marginal_sgp instead of reinventing per-category SGP math.
    The pre-PR19 inline formula ``value += (stat/denom) * weight`` summed
    inverse stats (ERA/WHIP/L) with the WRONG SIGN — high ERA was treated as
    a positive contribution, so unknown pitchers with default ERA=9.0/WHIP=2.0
    scored base=+38.8, outranking real MLB hitters at base=+18.3. marginal_sgp
    handles inverse-stat signs and rate-stat volume-weighting correctly.

    FA-engine overhaul P2 PR4 (2026-05-20): inputs blended (ROS + YTD with
    canonical published weights) before scoring. Sample-size gated so early-
    season small samples don't dominate.
    """
    from src.in_season import _roster_category_totals
    from src.valuation import SGPCalculator

    # FA P5c (2026-05-20): scale ROS by playing-time ratio BEFORE blending.
    # An IL stash phantom (0 YTD GP) still has the same ROS projection as a
    # healthy starter — the projection assumes regular playing time the
    # player isn't getting. Discounting at the source means downstream
    # marginal_sgp sees the deflated ROS columns. The PR21 multiplier on
    # the composite score still applies downstream (stacked penalty).
    fa_data = _scale_ros_by_playing_time(fa_data, ctx)

    l14_form = _resolve_fa_l14(fa_data, ctx)
    blended = _blend_fa_row(fa_data, l14_form=l14_form)

    # Cache roster_totals + sgp_calc on ctx so marginal_sgp's rate-stat
    # math sees a stable per-call roster context (it depends on team
    # totals for AVG/OBP/ERA/WHIP volume-weighting).
    if not hasattr(ctx, "_fa_roster_totals"):
        try:
            ctx._fa_roster_totals = _roster_category_totals(ctx.user_roster_ids, ctx.player_pool)
        except Exception:
            logger.debug("Could not compute roster_totals for FA scoring", exc_info=True)
            ctx._fa_roster_totals = {}
    if not hasattr(ctx, "_fa_sgp_calc"):
        ctx._fa_sgp_calc = SGPCalculator(ctx.config)

    try:
        sgp_dict = ctx._fa_sgp_calc.marginal_sgp(blended, ctx._fa_roster_totals, ctx.category_weights)
        value = sum(sgp_dict.values())
    except Exception:
        logger.debug("marginal_sgp failed for FA scoring — falling back to 0", exc_info=True)
        value = 0.0

    # Fallback to precomputed marginal_value if marginal_sgp produced no
    # signal AND the rank_free_agents column is available (preserves
    # existing test contracts that rely on marginal_value as a base).
    if abs(value) < 1e-9 and "marginal_value" in fa_data.index and pd.notna(fa_data.get("marginal_value")):
        return float(fa_data["marginal_value"])

    return value


def _compute_urgency_boost(fa_data: pd.Series, ctx: OptimizerDataContext) -> float:
    """Sum category weights for categories this FA contributes to."""
    boost = 0.0
    is_hitter = _is_hitter_safe(fa_data.get("is_hitter"))
    stat_map = ctx.config.STAT_MAP

    if is_hitter:
        relevant_cats = ctx.config.hitting_categories
    else:
        relevant_cats = ctx.config.pitching_categories

    for cat in relevant_cats:
        col = stat_map.get(cat, cat.lower())
        if col in fa_data.index:
            stat_val = float(fa_data.get(col, 0) or 0)
            if abs(stat_val) > 0.001:
                weight = ctx.category_weights.get(cat, ctx.category_weights.get(cat.lower(), 1.0))
                boost += weight

    return boost


def _get_losing_categories(ctx: OptimizerDataContext) -> list[str]:
    """Get list of categories the user is currently losing."""
    summary = ctx.urgency_weights.get("summary", {})
    return list(summary.get("losing", []))


def _get_tied_categories(ctx: OptimizerDataContext) -> list[str]:
    """Get list of categories the user is currently tied in."""
    summary = ctx.urgency_weights.get("summary", {})
    return list(summary.get("tied", []))


def _allow_cross_type(
    ctx: OptimizerDataContext,
    fa: dict,
    drop: dict,
    target_cats: set[str],
) -> bool:
    """Allow cross-type swap only when user has positional surplus AND
    the swap improves a losing/tied category by >= 0.3 SGP.
    """
    # Check positional surplus
    hitter_count = 0
    for pid in ctx.user_roster_ids:
        match = ctx.player_pool[ctx.player_pool["player_id"] == pid]
        if not match.empty and _is_hitter_safe(match.iloc[0].get("is_hitter")):
            hitter_count += 1

    fa_is_hitter = fa["is_hitter"]

    # If adding a hitter and dropping a pitcher, need pitcher surplus (hitters < slots)
    # If adding a pitcher and dropping a hitter, need hitter surplus
    if fa_is_hitter and hitter_count >= _HITTER_SLOTS:
        # Already at or above hitter capacity, no room for another hitter
        return False
    if not fa_is_hitter and hitter_count <= _HITTER_SLOTS:
        # No hitter surplus — can't drop a hitter for a pitcher
        return False

    # Check if swap improves a target category by >= 0.3 SGP
    swap = compute_net_swap_value(
        fa["player_id"],
        drop["player_id"],
        ctx.user_roster_ids,
        ctx.player_pool,
        ctx.config,
    )
    cat_deltas = swap["category_deltas"]
    for cat in target_cats:
        delta = cat_deltas.get(cat, 0)
        if delta >= _CROSS_TYPE_SGP_MIN:
            return True

    return False


def _build_reasoning(
    fa: dict,
    drop: dict,
    swap: dict,
    ctx: OptimizerDataContext,
) -> list[str]:
    """Build human-readable reasoning list for the recommendation."""
    reasons: list[str] = []
    cat_deltas = swap["category_deltas"]

    # PR13 (FA P3.10): track which category deltas are surfaced explicitly
    # in the reasoning so we can reconcile the remainder against net_sgp.
    mentioned_cats: set[str] = set()

    # Best category improvement
    if cat_deltas:
        best_cat = max(cat_deltas, key=lambda c: cat_deltas[c])
        best_val = cat_deltas[best_cat]
        if best_val > 0:
            reasons.append(f"{fa['name']} adds +{best_val:.2f} SGP in {best_cat}")
            mentioned_cats.add(best_cat)

    # Worst category cost
    if cat_deltas:
        worst_cat = min(cat_deltas, key=lambda c: cat_deltas[c])
        worst_val = cat_deltas[worst_cat]
        if worst_val < _CATEGORY_WORSEN_THRESHOLD:
            reasons.append(f"Costs {worst_val:.2f} SGP in {worst_cat}")
            mentioned_cats.add(worst_cat)

    # Urgency mention
    losing = _get_losing_categories(ctx)
    helped_losing = [c for c in losing if cat_deltas.get(c, 0) > 0.01]
    if helped_losing:
        reasons.append(f"Helps in losing categories: {', '.join(helped_losing)}")

    # Sustainability
    sust = fa["sustainability"]
    if sust < 0.4:
        reasons.append("Caution: current stats may not be sustainable")
    elif sust > 0.7:
        reasons.append("Strong underlying metrics support continued production")

    # Ownership trend
    if fa.get("ownership_delta_7d", 0) > _OWNERSHIP_BOOST_DELTA:
        reasons.append(f"Ownership trending up: {fa['ownership_trend']}")

    # News flag
    news = ctx.news_flags.get(fa["player_id"])
    if news:
        reasons.append(f"Recent news: {news}")

    # PR13 (FA P3.10): reconciliation. The best/worst-cat lines above only
    # surface 1-2 categories' SGP contribution, but the net_sgp summary
    # below sums ALL categories. When the residual (everything not yet
    # mentioned) is materially non-zero, append an explicit "Other
    # categories: +X.XX SGP" line so the user can see the reasoning
    # adds up to the net.
    other_total = sum(v for c, v in cat_deltas.items() if c not in mentioned_cats)
    if abs(other_total) > 0.10:
        sign = "+" if other_total >= 0 else ""
        reasons.append(f"Other categories: {sign}{other_total:.2f} SGP")

    # Net SGP summary
    reasons.append(f"Net team improvement: +{swap['net_sgp']:.2f} SGP")

    return reasons


# ---------------------------------------------------------------------------
# Daily streaming recommendations (scope="today" only)
# ---------------------------------------------------------------------------


def _compute_ros_sgp(row: pd.Series, config) -> float:
    """Approximate ROS SGP from a player's projection row."""
    sgp = 0.0
    for cat in config.all_categories:
        val = float(row.get(cat.lower(), 0) or 0)
        denom = config.sgp_denominators.get(cat, 1.0)
        if abs(denom) < 1e-9:
            continue
        if cat in config.inverse_stats:
            sgp -= val / denom
        else:
            sgp += val / denom
    return sgp


# Full team-name → 3-letter abbreviation is shared across modules; imported
# from src.valuation above as `_FULL_TO_ABBR`. The original alias is kept
# so the rest of this file's call sites stay readable.
#
# Different data sources disagree on 3-letter codes. Expand each set so a
# match on any variant satisfies the membership check.
_TEAM_EQUIVALENCES: dict[str, set[str]] = {
    "WSH": {"WSH", "WSN", "WAS"},
    "SF": {"SF", "SFG"},
    "SD": {"SD", "SDP"},
    "TB": {"TB", "TBR"},
    "KC": {"KC", "KCR"},
    "CWS": {"CWS", "CHW"},
    "ATH": {"ATH", "OAK"},
}


def _expand_team_equivalences(abbr: str) -> set[str]:
    for canon, variants in _TEAM_EQUIVALENCES.items():
        if abbr in variants or abbr == canon:
            return variants | {canon}
    return {abbr}


def _normalize_team(raw: str) -> set[str]:
    """Return the set of codes that represent the same MLB team as `raw`.
    Raw can be a full name ("Boston Red Sox") or any 3-letter variant."""
    if not raw:
        return set()
    up = str(raw).upper().strip()
    abbr = _FULL_TO_ABBR.get(up, up)
    return _expand_team_equivalences(abbr) | {abbr, up}


def _get_teams_playing_today(ctx: OptimizerDataContext) -> set[str]:
    """All team codes (with equivalence variants) whose team has a game today."""
    teams: set[str] = set()
    for game in ctx.todays_schedule or []:
        for key in ("home_name", "away_name", "home_team", "away_team"):
            raw = game.get(key)
            if raw:
                teams |= _normalize_team(raw)
    return teams


def _get_probable_starter_ids_today(ctx: OptimizerDataContext) -> set[int]:
    """Extract player IDs of SPs scheduled to start today.

    statsapi returns ``home_probable_pitcher`` / ``away_probable_pitcher``
    as name strings (e.g. "Brandon Pfaadt"), not IDs. We resolve each name
    against the player_pool to find the corresponding player_id.
    """
    names_lower: set[str] = set()
    for game in ctx.todays_schedule or []:
        for side in ("home_probable_pitcher", "away_probable_pitcher"):
            raw = game.get(side)
            if not raw:
                continue
            if isinstance(raw, dict):
                name = raw.get("fullName") or raw.get("name") or ""
            else:
                name = str(raw)
            name = name.strip()
            if name and name.upper() not in ("TBD", "TBA", "UNKNOWN", ""):
                names_lower.add(name.lower())
    if not names_lower or ctx.player_pool is None or ctx.player_pool.empty:
        return set()

    name_col = "name" if "name" in ctx.player_pool.columns else "player_name"
    if name_col not in ctx.player_pool.columns:
        return set()
    pool_names = ctx.player_pool[name_col].astype(str).str.strip().str.lower()
    matches = ctx.player_pool[pool_names.isin(names_lower)]
    ids: set[int] = set()
    for _, row in matches.iterrows():
        pid = row.get("player_id")
        if pid is None:
            continue
        try:
            ids.add(int(pid))
        except (TypeError, ValueError):
            pass
    return ids


def _stream_drop_score(pid: int, ctx: OptimizerDataContext, teams_playing_today: set[str]) -> float:
    """Combined drop score: lower == better drop candidate.

    drop_score = remaining_week_sgp + (today_bonus if team plays today)

    Remaining-week SGP dominates; the today bonus softly protects players
    whose team is playing today from being dropped on a live game day.
    """
    match = ctx.player_pool[ctx.player_pool["player_id"] == pid]
    if match.empty:
        return float("inf")  # can't evaluate → not a drop candidate
    row = match.iloc[0]
    ros_sgp = _compute_ros_sgp(row, ctx.config)
    team_raw = str(row.get("team", "") or "").upper()
    team_variants = _normalize_team(team_raw)
    # remaining_games_this_week keys are full names (per shared_data_layer);
    # look up by any equivalent variant or full name.
    remaining_games = 3
    for key in (team_raw, *team_variants):
        if key in ctx.remaining_games_this_week:
            remaining_games = int(ctx.remaining_games_this_week[key] or 3)
            break
    # Scale ROS SGP to remaining-week contribution; 162-game season.
    weekly_sgp = ros_sgp * max(0, remaining_games) / 162.0
    today_bonus = _STREAM_DROP_TODAY_BONUS if team_variants & teams_playing_today else 0.0
    return weekly_sgp + today_bonus


def _parse_positions(positions: str) -> set[str]:
    """Split a positions string ("2B,OF,Util" or "2B/OF/Util") into a set."""
    if not positions:
        return set()
    return {p.strip().upper() for p in str(positions).replace("/", ",").split(",") if p.strip()}


def _worst_rostered(
    ctx: OptimizerDataContext,
    is_hitter: bool,
    teams_playing_today: set[str],
    fa_positions: str | None = None,
) -> tuple[int | None, float | None]:
    """Find the worst (lowest stream_drop_score) non-IL rostered player on
    the given side. When ``fa_positions`` is provided, only consider roster
    players whose eligibility overlaps with the FA's positions — this is the
    "slot-aware" drop candidate selection that avoids e.g. a 3B streamer
    targeting an OF-only player as the drop.
    """
    worst_pid: int | None = None
    worst_score: float | None = None
    fa_pos_set = _parse_positions(fa_positions) if fa_positions else set()
    for pid in ctx.user_roster_ids:
        match = ctx.player_pool[ctx.player_pool["player_id"] == pid]
        if match.empty:
            continue
        row = match.iloc[0]
        if _is_hitter_safe(row.get("is_hitter")) != is_hitter:
            continue
        if _player_is_il(ctx, pid):
            continue
        if pid in ctx.il_stash_ids:
            continue
        # Slot-aware filter: only consider roster players whose positions
        # overlap with the FA's. Util overlaps any hitter, P overlaps any
        # pitcher, so those rarely filter anything — by design.
        if fa_pos_set:
            roster_pos = _parse_positions(str(row.get("positions", "")))
            if not (fa_pos_set & roster_pos):
                continue
        score = _stream_drop_score(pid, ctx, teams_playing_today)
        if worst_score is None or score < worst_score:
            worst_score = score
            worst_pid = pid
    return worst_pid, worst_score


def _passes_ip_minimum(ctx: OptimizerDataContext, add_id: int, drop_id: int) -> bool:
    """Check if the post-swap weekly IP projection stays above forfeit minimum."""
    try:
        from src.ip_tracker import compute_weekly_ip_projection, get_days_remaining_in_week

        new_ids = [p for p in ctx.user_roster_ids if p != drop_id] + [add_id]
        pitchers = []
        for pid in new_ids:
            m = ctx.player_pool[ctx.player_pool["player_id"] == pid]
            if m.empty:
                continue
            r = m.iloc[0]
            if bool(int(r.get("is_hitter", 1))):
                continue
            pitchers.append(
                {
                    "player_id": pid,
                    "ip": float(r.get("ip", 0) or 0),
                    "positions": str(r.get("positions", "")),
                    "status": str(r.get("status", "active")),
                    "is_starter": "SP" in str(r.get("positions", "")).upper(),
                }
            )
        ip_result = compute_weekly_ip_projection(pitchers, get_days_remaining_in_week())
        return ip_result.get("projected_ip", 0) >= _STREAM_IP_MIN
    except Exception:
        logger.debug("IP check failed; not blocking", exc_info=True)
        return True


def recommend_streaming_moves(
    ctx: OptimizerDataContext,
    max_per_side: int = _STREAM_MAX_PER_SIDE,
) -> dict[str, list[dict[str, Any]]]:
    """Daily streaming recommendations (scope="today" only).

    Returns a dict ``{"pitchers": [...], "batters": [...]}`` each sorted
    best-to-worst and capped at ``max_per_side``.

    Rules:
    - Only fires when ``ctx.scope == "today"``.
    - Target categories: those with per-category win probability
      >= 27.55% (categories still realistically in play this week).
    - FA pitcher candidates must be probable starters today.
    - FA batter candidates must be on a team with a game today.
    - Drop target = worst rostered player on the streamer's side, or on
      the opposite side if the cross-side worst is < 50% of same-side
      worst (cross-swap is pitcher-stream only).
    - Net SGP gain must be >= 0.70.
    - Hurts guard blocks any swap that hurts a protected cat by more
      than 0.10 SGP. Protected cats = in-play (>=27.55% win prob) PLUS
      any currently-losing or tied cats regardless of win prob.
    - Must help at least one currently-losing or tied cat (gap-close
      requirement). Pure lead-extension swaps don't justify the FA-loss
      risk of dropping a rostered player.
    - Post-swap weekly IP must stay >= 20 (pitcher streams only).
    - IL FAs are ineligible for streaming (use regular FA engine for
      IL→IL stash upgrades).
    """
    diagnostics: dict[str, Any] = {
        "scope": ctx.scope,
        "in_play_cats": [],
        "protected_cats": [],
        "n_probable_sps": 0,
        "n_teams_playing_today": 0,
        "n_fa_considered": 0,
        "n_fa_filtered_no_game": 0,
        "n_fa_filtered_net_sgp": 0,
        "n_fa_filtered_hurts": 0,
        "n_fa_filtered_no_gap_close": 0,
        "n_fa_filtered_ip": 0,
        "n_fa_filtered_il": 0,
        "note": "",
    }
    empty: dict[str, Any] = {"pitchers": [], "batters": [], "diagnostics": diagnostics}
    if ctx.scope != "today":
        diagnostics["note"] = f"Scope is '{ctx.scope}' (streaming only activates on Today)."
        return empty

    # Per-category win probability
    try:
        from src.optimizer.h2h_engine import estimate_h2h_win_probability

        wp_result = estimate_h2h_win_probability(ctx.my_totals, ctx.opp_totals)
    except Exception:
        logger.debug("Win-prob computation failed", exc_info=True)
        diagnostics["note"] = "Win-probability computation failed — no matchup totals available."
        return empty
    wp = {str(k).lower(): float(v) for k, v in wp_result.get("per_category", {}).items()}
    target_cats = {c for c, p in wp.items() if p >= _STREAM_WIN_PROB_MIN}
    diagnostics["in_play_cats"] = sorted(target_cats)
    # Hurts guard protects a superset: target (in-play) cats PLUS any cat
    # the user is currently losing or tied in, regardless of win prob.
    # Rationale: a cat with <27.55% win prob this week is usually lost,
    # but we still refuse to make it worse — a -0.10+ SGP hit to an
    # already-losing cat risks blowing it open and hurts standings-gained
    # math over the season.
    losing_cats = {str(c).lower() for c in _get_losing_categories(ctx)}
    tied_cats = {str(c).lower() for c in _get_tied_categories(ctx)}
    protected_cats = target_cats | losing_cats | tied_cats
    diagnostics["protected_cats"] = sorted(protected_cats)
    if not target_cats:
        diagnostics["note"] = (
            f"No categories with win probability >= {_STREAM_WIN_PROB_MIN:.0%}. Streaming only fires for in-play cats."
        )
        return empty

    probable_sp_ids = _get_probable_starter_ids_today(ctx)
    teams_playing = _get_teams_playing_today(ctx)
    diagnostics["n_probable_sps"] = len(probable_sp_ids)
    # Deduplicate to canonical 3-letter abbreviations before counting so the
    # diagnostic reads sensibly (30 teams max, not 60+ inflated by full-name
    # + abbreviation + equivalence duplicates). Canonical set = all-caps
    # alphabetic tokens of length ≤ 4 (filters out "CHICAGO CUBS", "WSN", etc.
    # in favor of "CHC", "WSH").
    _canonical_teams = {t for t in teams_playing if t and t.isalpha() and 2 <= len(t) <= 4 and t == t.upper()}
    diagnostics["n_teams_playing_today"] = len(_canonical_teams) if _canonical_teams else len(teams_playing)

    # Dynamic SGP threshold for pitcher streams: if weekly IP is below 75%
    # of the target (54 IP), relax the +0.70 SGP bar to +0.40 so more
    # marginal-but-helpful pitcher pickups surface. Otherwise the engine can
    # be over-conservative and miss easy IP-adding streams when the user
    # has an IP deficit (e.g., 38.5/54 = 71% with zero streams today).
    _ip_projected = 0.0
    try:
        from src.ip_tracker import compute_weekly_ip_projection, get_days_remaining_in_week

        _pitchers_for_ip = []
        for pid in ctx.user_roster_ids:
            m = ctx.player_pool[ctx.player_pool["player_id"] == pid]
            if m.empty:
                continue
            r = m.iloc[0]
            if bool(int(r.get("is_hitter", 1))):
                continue
            _pitchers_for_ip.append(
                {
                    "player_id": pid,
                    "ip": float(r.get("ip", 0) or 0),
                    "positions": str(r.get("positions", "")),
                    "status": str(r.get("status", "active")),
                    "is_starter": "SP" in str(r.get("positions", "")).upper(),
                }
            )
        _ip_res = compute_weekly_ip_projection(_pitchers_for_ip, get_days_remaining_in_week())
        _ip_projected = float(_ip_res.get("projected_ip", 0) or 0)
    except Exception:
        logger.debug("IP projection for dynamic SGP threshold failed", exc_info=True)
        _ip_projected = 0.0
    _pitcher_sgp_threshold = (
        _STREAM_NET_SGP_RELAXED
        if _STREAM_IP_TARGET > 0 and _ip_projected / _STREAM_IP_TARGET < _STREAM_IP_RELAX_RATIO
        else _STREAM_NET_SGP_MIN
    )
    diagnostics["ip_projected_pre_swap"] = round(_ip_projected, 1)
    diagnostics["pitcher_sgp_threshold"] = _pitcher_sgp_threshold

    worst_pitcher_id, worst_pitcher_score = _worst_rostered(ctx, is_hitter=False, teams_playing_today=teams_playing)
    worst_batter_id, worst_batter_score = _worst_rostered(ctx, is_hitter=True, teams_playing_today=teams_playing)

    def _pick_drop(fa_is_hitter: bool, fa_positions: str | None = None) -> int | None:
        # Slot-aware same-side drop: prefer a rostered player whose
        # eligibility overlaps the FA's positions so e.g. a 3B streamer
        # doesn't target an OF-only player. Previously used a single
        # "globally worst" player per side, which caused every batter
        # stream to target the same drop candidate regardless of position.
        same_id: int | None
        same_score: float | None
        if fa_positions:
            same_id, same_score = _worst_rostered(
                ctx,
                is_hitter=fa_is_hitter,
                teams_playing_today=teams_playing,
                fa_positions=fa_positions,
            )
        else:
            same_id = worst_batter_id if fa_is_hitter else worst_pitcher_id
            same_score = worst_batter_score if fa_is_hitter else worst_pitcher_score
        # Fallback: if no slot-compatible roster player exists (unusual),
        # fall back to the global worst on this side so a valid move still
        # surfaces rather than being silently dropped.
        if same_id is None:
            same_id = worst_batter_id if fa_is_hitter else worst_pitcher_id
            same_score = worst_batter_score if fa_is_hitter else worst_pitcher_score
        # Cross-swap policy: ONLY for pitcher streaming (drop worst batter
        # instead of worst pitcher when the batter is much worse). Batter
        # streams never drop a pitcher.
        if not fa_is_hitter:
            cross_id = worst_batter_id
            cross_score = worst_batter_score
            if same_score is not None and cross_score is not None:
                if cross_score < same_score * _STREAM_CROSS_SIDE_RATIO:
                    return cross_id
        return same_id

    pitcher_streamers: list[dict[str, Any]] = []
    batter_streamers: list[dict[str, Any]] = []

    if ctx.free_agents.empty:
        diagnostics["note"] = "No free agents in context."
        return empty

    excluded_ids = set(int(p) for p in ctx.user_roster_ids)
    excluded_ids.update(int(p) for p in ctx.league_rostered_ids)

    for _, fa_row in ctx.free_agents.iterrows():
        fa_id_raw = fa_row.get("player_id")
        if fa_id_raw is None:
            continue
        try:
            fa_id = int(fa_id_raw)
        except (TypeError, ValueError):
            continue

        if fa_id in excluded_ids:
            continue
        if _player_is_il(ctx, fa_id):
            diagnostics["n_fa_filtered_il"] += 1
            continue

        diagnostics["n_fa_considered"] += 1
        fa_is_hitter = bool(int(fa_row.get("is_hitter", 1)))

        # Game-today filter
        if not fa_is_hitter:
            if fa_id not in probable_sp_ids:
                diagnostics["n_fa_filtered_no_game"] += 1
                continue
        else:
            team_raw = str(fa_row.get("team", "") or "").upper()
            team_variants = _normalize_team(team_raw)
            if not team_variants or not (team_variants & teams_playing):
                diagnostics["n_fa_filtered_no_game"] += 1
                continue

        drop_id = _pick_drop(fa_is_hitter, fa_positions=str(fa_row.get("positions", "") or ""))
        if drop_id is None:
            continue

        swap = compute_net_swap_value(fa_id, drop_id, ctx.user_roster_ids, ctx.player_pool, ctx.config)
        net_sgp = float(swap.get("net_sgp", 0))
        cat_deltas = {str(k).lower(): float(v) for k, v in swap.get("category_deltas", {}).items()}

        # Pitcher streams get the dynamic threshold (relaxed when IP deficit);
        # batter streams keep the stricter +0.70 SGP bar. This prevents batter
        # noise when IP is low while still opening up pitcher pickups that
        # would help close the IP gap.
        _applicable_min = _pitcher_sgp_threshold if not fa_is_hitter else _STREAM_NET_SGP_MIN
        if net_sgp < _applicable_min:
            diagnostics["n_fa_filtered_net_sgp"] += 1
            continue

        # Hurts guard: protected cats are target (in-play) cats PLUS any
        # currently-losing or tied cats regardless of win prob. Blocks
        # the swap if any protected cat would drop by more than 0.10 SGP.
        if any(cat_deltas.get(c, 0) < _STREAM_HURT_THRESHOLD for c in protected_cats):
            diagnostics["n_fa_filtered_hurts"] += 1
            continue

        # Must help at least one LOSING or TIED cat (gap-closing or tie-
        # breaking, not lead-extending). Swaps that only pad comfortable
        # leads don't justify the FA-loss risk of dropping a rostered
        # player. Falls back to in-play target cats only if no losing/tied
        # cats exist (user winning everything) — in that rare case, any
        # in-play help is fine since there's nothing to close.
        gap_close_cats = losing_cats | tied_cats
        if gap_close_cats:
            helpful_targets = [c for c in gap_close_cats if cat_deltas.get(c, 0) > 0.01]
            filter_reason = "n_fa_filtered_no_gap_close"
        else:
            helpful_targets = [c for c in target_cats if cat_deltas.get(c, 0) > 0.01]
            filter_reason = "n_fa_filtered_hurts"
        if not helpful_targets:
            diagnostics[filter_reason] = diagnostics.get(filter_reason, 0) + 1
            continue

        # IP minimum (pitcher streams only)
        if not fa_is_hitter and not _passes_ip_minimum(ctx, fa_id, drop_id):
            diagnostics["n_fa_filtered_ip"] += 1
            continue

        # Build display payload
        helps = {c: round(v, 2) for c, v in cat_deltas.items() if v > 0.01}
        hurts = {c: round(v, 2) for c, v in cat_deltas.items() if v < -0.01}

        drop_match = ctx.player_pool[ctx.player_pool["player_id"] == drop_id]
        drop_name = "?"
        if not drop_match.empty:
            _dr = drop_match.iloc[0]
            drop_name = str(_dr.get("name", _dr.get("player_name", "?")))

        streamer = {
            "add_id": fa_id,
            "add_name": str(fa_row.get("name", fa_row.get("player_name", "?"))),
            "add_positions": str(fa_row.get("positions", "")),
            "add_team": str(fa_row.get("team", "")),
            "drop_id": drop_id,
            "drop_name": drop_name,
            "is_hitter": fa_is_hitter,
            "helps": helps,
            "hurts": hurts,
            "net_sgp": round(net_sgp, 2),
            "target_cats_helped": sorted(helpful_targets),
        }
        if fa_is_hitter:
            batter_streamers.append(streamer)
        else:
            pitcher_streamers.append(streamer)

    pitcher_streamers.sort(key=lambda x: x["net_sgp"], reverse=True)
    batter_streamers.sort(key=lambda x: x["net_sgp"], reverse=True)

    if not pitcher_streamers and not batter_streamers and not diagnostics["note"]:
        diagnostics["note"] = (
            f"Considered {diagnostics['n_fa_considered']} FAs. Filtered out: "
            f"{diagnostics['n_fa_filtered_no_game']} not playing today, "
            f"{diagnostics['n_fa_filtered_net_sgp']} below +0.70 net SGP, "
            f"{diagnostics['n_fa_filtered_hurts']} hurt a protected cat, "
            f"{diagnostics['n_fa_filtered_no_gap_close']} didn't help any losing/tied cat, "
            f"{diagnostics['n_fa_filtered_ip']} failed IP minimum check."
        )

    return {
        "pitchers": pitcher_streamers[:max_per_side],
        "batters": batter_streamers[:max_per_side],
        "diagnostics": diagnostics,
    }

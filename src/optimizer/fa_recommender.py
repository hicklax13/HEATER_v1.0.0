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

from src.optimizer.shared_data_layer import OptimizerDataContext
from src.waiver_wire import (
    compute_drop_cost,
    compute_net_swap_value,
    compute_sustainability_score,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_CLOSERS = 2
_CLOSER_SV_THRESHOLD = 5
_MAX_DROP_CANDIDATES = 5
_MAX_FA_CANDIDATES = 10
_CATEGORY_WORSEN_THRESHOLD = -0.1
_MAX_WORSENED_CATEGORIES = 3
_HITTER_SLOTS = 10  # C/1B/2B/3B/SS/3OF/2Util = 10 starting hitter slots
_CROSS_TYPE_SGP_MIN = 0.3
_OWNERSHIP_BOOST_DELTA = 5.0
_OWNERSHIP_BOOST_MULT = 1.10
_FLOOR_PENALTY_MULT = 0.85
_FLOOR_PA_MIN = 50
_FLOOR_IP_MIN = 20
_IL_EXCLUDE_STATUSES = {"il", "il10", "il15", "il60", "dtd", "day-to-day", "na", "out", "suspended"}

# Streaming knobs (for scope="today" only).
_STREAM_WIN_PROB_MIN = 0.38  # categories with >=38% win prob are in-play
_STREAM_NET_SGP_MIN = 0.70  # minimum net SGP gain to surface a streamer
_STREAM_HURT_THRESHOLD = -0.10  # max allowed hurt on any in-play category
_STREAM_CROSS_SIDE_RATIO = 0.5  # cross-swap if cross-worst < this * same-worst
_STREAM_DROP_TODAY_BONUS = 0.15  # protect rostered players with today game
_STREAM_MAX_PER_SIDE = 3  # cap per-side recommendations
_STREAM_IP_MIN = 20  # weekly IP minimum (Yahoo forfeit line)


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

    # ── Step 1: Score drop candidates ────────────────────────────────
    drop_candidates = _score_drop_candidates(ctx)
    if not drop_candidates:
        logger.info("No viable drop candidates found")
        return []

    # ── Step 2: Score FA candidates ──────────────────────────────────
    fa_candidates = _score_fa_candidates(ctx)
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


def _score_drop_candidates(ctx: OptimizerDataContext) -> list[dict]:
    """Score rostered players as drop candidates.

    Hard filters:
    - Never drop IL stash players.
    - Never drop a closer if it would reduce closer count below minimum.
    """
    candidates: list[dict] = []

    for pid in ctx.user_roster_ids:
        # Hard filter: IL stash protection
        if pid in ctx.il_stash_ids:
            continue

        # Hard filter: closer minimum
        if _is_closer(pid, ctx) and ctx.closer_count <= _MIN_CLOSERS:
            continue

        cost = compute_drop_cost(pid, ctx.user_roster_ids, ctx.player_pool, ctx.config)

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
                "is_hitter": bool(int(row.get("is_hitter", 1))),
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


def _score_fa_candidates(ctx: OptimizerDataContext) -> list[dict]:
    """Score free agents by composite value.

    Filters out unhealthy players (health_score < 0.65 or IL/DTD/NA status).
    Scores by: base value, urgency boost, ownership trend, sustainability,
    and floor preference.
    """
    if ctx.free_agents.empty:
        return []

    candidates: list[dict] = []

    # Build set of ALL rostered player IDs (user + opponents) for exclusion.
    # ctx.user_roster_ids covers the user's team; ctx.league_rostered_ids
    # (populated by build_optimizer_context from league_rosters table)
    # covers all other teams. Tests pass an empty set to opt out.
    _excluded_ids: set[int] = set(int(pid) for pid in ctx.user_roster_ids)
    _excluded_ids.update(int(pid) for pid in ctx.league_rostered_ids)

    for _, fa_row in ctx.free_agents.iterrows():
        fa_id = fa_row.get("player_id")
        if fa_id is None:
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

        # Base value
        base_value = _compute_base_value(fa_data, ctx)

        # Urgency boost: sum category weights for categories this FA contributes to
        urgency_boost = _compute_urgency_boost(fa_data, ctx)

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
        is_hitter = bool(int(fa_data.get("is_hitter", 1)))
        if is_hitter:
            pa = float(fa_data.get("pa", 0) or 0)
            if pa < _FLOOR_PA_MIN:
                floor_mult = _FLOOR_PENALTY_MULT
        else:
            ip = float(fa_data.get("ip", 0) or 0)
            if ip < _FLOOR_IP_MIN:
                floor_mult = _FLOOR_PENALTY_MULT

        composite = base_value * sustainability * ownership_mult * floor_mult + urgency_boost

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

            # Category worsening check (informational — log but don't block)
            worsened = sum(1 for v in cat_deltas.values() if v < _CATEGORY_WORSEN_THRESHOLD)
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


def _compute_base_value(fa_data: pd.Series, ctx: OptimizerDataContext) -> float:
    """Compute base value for an FA candidate.

    Uses marginal_value column if present, otherwise sums counting stat
    projections weighted by category weights.
    """
    if "marginal_value" in fa_data.index and pd.notna(fa_data.get("marginal_value")):
        return float(fa_data["marginal_value"])

    # Fallback: weighted sum of counting stats
    value = 0.0
    stat_map = ctx.config.STAT_MAP
    for cat_display, col in stat_map.items():
        if col in fa_data.index:
            stat_val = float(fa_data.get(col, 0) or 0)
            weight = ctx.category_weights.get(cat_display, ctx.category_weights.get(cat_display.lower(), 1.0))
            denom = ctx.config.sgp_denominators.get(cat_display, 1.0)
            if abs(denom) < 1e-9:
                denom = 1.0
            value += (stat_val / denom) * weight
    return value


def _compute_urgency_boost(fa_data: pd.Series, ctx: OptimizerDataContext) -> float:
    """Sum category weights for categories this FA contributes to."""
    boost = 0.0
    is_hitter = bool(int(fa_data.get("is_hitter", 1)))
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
        if not match.empty and bool(int(match.iloc[0].get("is_hitter", 1))):
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

    # Best category improvement
    if cat_deltas:
        best_cat = max(cat_deltas, key=lambda c: cat_deltas[c])
        best_val = cat_deltas[best_cat]
        if best_val > 0:
            reasons.append(f"{fa['name']} adds +{best_val:.2f} SGP in {best_cat}")

    # Worst category cost
    if cat_deltas:
        worst_cat = min(cat_deltas, key=lambda c: cat_deltas[c])
        worst_val = cat_deltas[worst_cat]
        if worst_val < _CATEGORY_WORSEN_THRESHOLD:
            reasons.append(f"Costs {worst_val:.2f} SGP in {worst_cat}")

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


# Full team-name → 3-letter abbreviation. MLB Stats API (statsapi) returns
# schedules with full names (e.g. "Boston Red Sox"), but Yahoo/FanGraphs
# rows store 3-letter codes (e.g. "BOS"). We normalize both sides.
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


def _worst_rostered(
    ctx: OptimizerDataContext,
    is_hitter: bool,
    teams_playing_today: set[str],
) -> tuple[int | None, float | None]:
    """Find the worst (lowest stream_drop_score) non-IL rostered player on
    the given side. Returns (pid, score) or (None, None) if no candidate."""
    worst_pid: int | None = None
    worst_score: float | None = None
    for pid in ctx.user_roster_ids:
        match = ctx.player_pool[ctx.player_pool["player_id"] == pid]
        if match.empty:
            continue
        row = match.iloc[0]
        if bool(int(row.get("is_hitter", 1))) != is_hitter:
            continue
        if _player_is_il(ctx, pid):
            continue
        # Skip protected IL stashes too (Bieber, Strider, players
        # returning within 2 weeks) — handled in il_stash_ids.
        if pid in ctx.il_stash_ids:
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
    - Target categories: those with per-category win probability >= 0.38
      (categories that are still realistically in play this week).
    - FA pitcher candidates must be probable starters today.
    - FA batter candidates must be on a team with a game today.
    - Drop target = worst rostered player on the streamer's side, or on
      the opposite side if the cross-side worst is < 50% of same-side worst.
    - Net SGP gain must be >= 0.70.
    - Any in-play category hurt by more than -0.10 SGP blocks the swap.
    - Post-swap weekly IP must stay >= 20 (pitcher streams only).
    - IL FAs are ineligible for streaming (use regular FA engine for
      IL→IL stash upgrades).
    """
    diagnostics: dict[str, Any] = {
        "scope": ctx.scope,
        "in_play_cats": [],
        "n_probable_sps": 0,
        "n_teams_playing_today": 0,
        "n_fa_considered": 0,
        "n_fa_filtered_no_game": 0,
        "n_fa_filtered_net_sgp": 0,
        "n_fa_filtered_hurts": 0,
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
    if not target_cats:
        diagnostics["note"] = "No categories with win probability >= 38%. Streaming only fires for in-play cats."
        return empty

    probable_sp_ids = _get_probable_starter_ids_today(ctx)
    teams_playing = _get_teams_playing_today(ctx)
    diagnostics["n_probable_sps"] = len(probable_sp_ids)
    diagnostics["n_teams_playing_today"] = len(teams_playing)

    worst_pitcher_id, worst_pitcher_score = _worst_rostered(ctx, is_hitter=False, teams_playing_today=teams_playing)
    worst_batter_id, worst_batter_score = _worst_rostered(ctx, is_hitter=True, teams_playing_today=teams_playing)

    def _pick_drop(fa_is_hitter: bool) -> int | None:
        # Same-side default: drop worst-valued player on the streamer's side.
        same_id = worst_batter_id if fa_is_hitter else worst_pitcher_id
        same_score = worst_batter_score if fa_is_hitter else worst_pitcher_score
        # Cross-swap policy: ONLY for pitcher streaming (drop worst batter
        # instead of worst pitcher when the batter is much worse). Batter
        # streams never drop a pitcher — doing so would silently reduce
        # the pitcher roster and hurt pitching categories without warning,
        # even for cats that fell below the 0.38 in-play threshold.
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

        drop_id = _pick_drop(fa_is_hitter)
        if drop_id is None:
            continue

        swap = compute_net_swap_value(fa_id, drop_id, ctx.user_roster_ids, ctx.player_pool, ctx.config)
        net_sgp = float(swap.get("net_sgp", 0))
        cat_deltas = {str(k).lower(): float(v) for k, v in swap.get("category_deltas", {}).items()}

        if net_sgp < _STREAM_NET_SGP_MIN:
            diagnostics["n_fa_filtered_net_sgp"] += 1
            continue

        # Hurts guard on in-play cats
        if any(cat_deltas.get(c, 0) < _STREAM_HURT_THRESHOLD for c in target_cats):
            diagnostics["n_fa_filtered_hurts"] += 1
            continue

        # Must help at least one in-play target cat
        helpful_targets = [c for c in target_cats if cat_deltas.get(c, 0) > 0.01]
        if not helpful_targets:
            diagnostics["n_fa_filtered_hurts"] += 1  # no in-play help
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
            f"{diagnostics['n_fa_filtered_hurts']} hurt an in-play cat or didn't help one, "
            f"{diagnostics['n_fa_filtered_ip']} failed IP minimum check."
        )

    return {
        "pitchers": pitcher_streamers[:max_per_side],
        "batters": batter_streamers[:max_per_side],
        "diagnostics": diagnostics,
    }

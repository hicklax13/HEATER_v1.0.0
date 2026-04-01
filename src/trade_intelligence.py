"""Trade Intelligence — integration layer for enhanced trade valuations.

Wires injury model, Bayesian ROS confidence, category gap analysis,
free agent gating, and positional scarcity into the trade finder
pipeline. No new algorithms — this module connects existing infrastructure.

Used by:
    - ``src/trade_finder.py``: enhanced ``find_trade_opportunities()``
    - ``pages/10_Trade_Finder.py``: Trade Readiness tab
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.valuation import LeagueConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Closer scarcity: only ~30 true closers in MLB across 30 teams
SV_SCARCITY_MULT = 1.3
# Scarce position premium (C, SS, 2B have thin talent pools)
SCARCE_POS_MULT = 1.15
SCARCE_POSITIONS = {"C", "SS", "2B"}

# IL/DTD production multipliers (fraction of remaining season retained)
# Based on 22 weeks remaining as of early April
STATUS_MULTIPLIERS: dict[str, float] = {
    "il10": 0.91,  # ~2 weeks lost
    "il15": 0.84,  # ~3.5 weeks lost
    "il60": 0.55,  # ~10 weeks lost
    "dtd": 0.95,  # ~0.5 weeks expected
    "dl": 0.84,  # generic DL = assume IL15
    "out": 0.70,  # "out" with no timeline = pessimistic
    "na": 0.0,  # minors/not active = excluded
}

# FA gate: flag trade if FA value >= this fraction of trade target value
FA_GATE_THRESHOLD = 0.70

# Category correlation clusters (from deep research -- FanGraphs data)
# HR-RBI correlation r=0.86, HR-R r=0.74, R-RBI r=0.70 -- triple-counting the same skill
# SB is nearly independent of all other hitting categories (mean r=0.12)
# AVG-OBP correlation r=0.70 -- overlapping plate discipline
# ERA-WHIP correlation r=0.84 -- same underlying pitch quality

POWER_CLUSTER = {"HR", "R", "RBI"}
CONTACT_CLUSTER = {"AVG", "OBP"}
PITCHING_RATE_CLUSTER = {"ERA", "WHIP"}

# Discount = reduce SGP credit because the categories move together
# Premium = increase SGP credit because the category is independent
POWER_CLUSTER_DISCOUNT = 0.83  # 17% discount
SB_INDEPENDENCE_PREMIUM = 1.08  # 8% premium
CONTACT_CLUSTER_DISCOUNT = 0.90  # 10% discount
PITCHING_RATE_DISCOUNT = 0.88  # 12% discount


def apply_correlation_adjustments(
    sgp_dict: dict[str, float],
    config: Any = None,
) -> dict[str, float]:
    """Apply correlation discount/premium to per-category SGP values.

    Categories in correlated clusters get discounted because gaining
    one category in the cluster tends to gain the others too (double-counting).
    Independent categories (SB) get a premium because their value is additive.

    Args:
        sgp_dict: Per-category SGP values {cat: sgp_value}.
        config: League configuration (unused currently, reserved for future).

    Returns:
        Adjusted SGP dict with correlation modifiers applied.
    """
    adjusted = dict(sgp_dict)

    for cat in POWER_CLUSTER:
        if cat in adjusted:
            adjusted[cat] *= POWER_CLUSTER_DISCOUNT

    if "SB" in adjusted:
        adjusted["SB"] *= SB_INDEPENDENCE_PREMIUM

    for cat in CONTACT_CLUSTER:
        if cat in adjusted:
            adjusted[cat] *= CONTACT_CLUSTER_DISCOUNT

    for cat in PITCHING_RATE_CLUSTER:
        if cat in adjusted:
            adjusted[cat] *= PITCHING_RATE_DISCOUNT

    return adjusted


def compute_schedule_urgency(
    weeks_ahead: int = 3,
    yds=None,
) -> float:
    """Compute urgency multiplier based on upcoming schedule difficulty.

    Facing Tier 1 opponents increases urgency; Tier 3-4 decreases it.

    Args:
        weeks_ahead: Number of weeks to look ahead (default 3).
        yds: Optional YahooDataService for live schedule data.

    Returns:
        float multiplier in [0.85, 1.25].
    """
    from src.opponent_intel import get_opponent_for_week, get_week_number

    current_week = get_week_number()
    tier_scores = {1: 2.0, 2: 1.0, 3: 0.0, 4: -0.5}
    total_difficulty = 0.0
    count = 0

    for w in range(current_week + 1, current_week + 1 + weeks_ahead):
        if w > 24:
            break
        opp = get_opponent_for_week(w, yds=yds)
        if opp:
            tier = opp.get("tier", 3)
            total_difficulty += tier_scores.get(tier, 0.0)
            count += 1

    if count == 0:
        return 1.0  # No schedule data -- neutral

    avg_difficulty = total_difficulty / count
    # Map [-0.5, 2.0] range to [0.85, 1.25] multiplier
    urgency = 0.85 + (avg_difficulty + 0.5) * (0.40 / 2.5)
    return max(0.85, min(1.25, urgency))


def compute_dynamic_fa_threshold(avg_pa: float) -> float:
    """Compute FA gate threshold that adapts to season progress.

    Early season (< 50 PA): threshold ~ 0.85 (less aggressive)
    Mid season (200 PA): threshold ~ 0.70 (standard)
    Late season (500+ PA): threshold ~ 0.60 (more aggressive)

    At zero PA the threshold is 0.85 (maximum conservatism).
    """
    import math

    if avg_pa <= 0:
        return 0.85

    decay = 0.25 / (1 + math.exp(-0.015 * (avg_pa - 200)))
    return max(0.55, 0.85 - decay)


# Average stabilization point for projection confidence
_AVG_HITTER_STAB = 300  # PA — rough average across batting stats
_AVG_PITCHER_STAB = 100  # IP — rough average across pitching stats


# ---------------------------------------------------------------------------
# Health-adjusted projections
# ---------------------------------------------------------------------------


def get_health_adjusted_pool(
    player_pool: pd.DataFrame,
    config: LeagueConfig | None = None,
) -> pd.DataFrame:
    """Apply injury health scores and IL status to player projections.

    Loads 3-year injury history, computes per-player health scores, and
    reduces counting-stat projections proportionally. Players with Yahoo
    status "NA" (minors/not active) are excluded entirely.

    Args:
        player_pool: DataFrame from ``load_player_pool()``.
        config: League config (uses default if None).

    Returns:
        Adjusted player pool with ``health_score`` and ``status`` columns.
    """
    if player_pool.empty:
        return player_pool

    pool = player_pool.copy()

    # --- Load injury history for health scores ---
    health_scores = _load_health_scores()

    # --- Load roster statuses (IL/DTD/NA) ---
    roster_statuses = _load_roster_statuses()

    # --- Apply health scores ---
    pool["health_score"] = pool["player_id"].map(health_scores).fillna(0.85)
    pool["_orig_health"] = pool["health_score"].copy()
    pool["status"] = pool["player_id"].map(roster_statuses).fillna("active")

    # --- Adjust health scores based on current IL/DTD status ---
    # Players on IL with no injury history get a lower health score than the default
    for idx, row in pool.iterrows():
        status = str(row.get("status", "active")).lower().strip()
        if status in ("il10", "il15", "dl") and row.get("health_score", 0.85) >= 0.80:
            pool.at[idx, "health_score"] = 0.65  # Moderate risk — currently injured
        elif status in ("il60", "out") and row.get("health_score", 0.85) >= 0.60:
            pool.at[idx, "health_score"] = 0.40  # Elevated risk — long-term injury
        elif status == "dtd" and row.get("health_score", 0.85) >= 0.80:
            pool.at[idx, "health_score"] = 0.75  # Slightly elevated — day-to-day

    # --- Exclude NA/minors players ---
    na_mask = pool["status"].str.lower().isin(["na", "not active", "minors"])
    if na_mask.any():
        logger.info("Excluding %d NA/minors players from trade pool", na_mask.sum())
        pool = pool[~na_mask].copy()

    # --- Apply IL/DTD multipliers to counting stats ---
    counting_cols = [
        "r",
        "hr",
        "rbi",
        "sb",
        "w",
        "l",
        "sv",
        "k",
        "pa",
        "ab",
        "h",
        "ip",
        "er",
        "bb_allowed",
        "h_allowed",
        "bb",
        "hbp",
        "sf",
    ]
    # Cast counting columns to float to avoid FutureWarning from pandas
    for col in counting_cols:
        if col in pool.columns:
            pool[col] = pd.to_numeric(pool[col], errors="coerce").fillna(0).astype(float)

    for _, row in pool.iterrows():
        status = str(row.get("status", "active")).lower().strip()
        mult = STATUS_MULTIPLIERS.get(status, 1.0)
        if mult < 1.0:
            idx = row.name
            health = row.get("_orig_health", 0.85)
            # Combined multiplier: IL status * historical health
            combined = mult * min(health / 0.85, 1.0)
            for col in counting_cols:
                if col in pool.columns:
                    pool.at[idx, col] = pool.at[idx, col] * combined

    # --- Recalculate rate stats from adjusted components ---
    ab = pd.to_numeric(pool.get("ab", 0), errors="coerce").fillna(0)
    h = pd.to_numeric(pool.get("h", 0), errors="coerce").fillna(0)
    ip = pd.to_numeric(pool.get("ip", 0), errors="coerce").fillna(0)
    er = pd.to_numeric(pool.get("er", 0), errors="coerce").fillna(0)
    bb_allowed = pd.to_numeric(pool.get("bb_allowed", 0), errors="coerce").fillna(0)
    h_allowed = pd.to_numeric(pool.get("h_allowed", 0), errors="coerce").fillna(0)

    pool["avg"] = np.where(ab > 0, h / ab, 0.0)
    pool["era"] = np.where(ip > 0, er * 9 / ip, 0.0)
    pool["whip"] = np.where(ip > 0, (bb_allowed + h_allowed) / ip, 0.0)

    bb_col = (
        pd.to_numeric(pool["bb"], errors="coerce").fillna(0) if "bb" in pool.columns else pd.Series(0, index=pool.index)
    )
    hbp_col = (
        pd.to_numeric(pool["hbp"], errors="coerce").fillna(0)
        if "hbp" in pool.columns
        else pd.Series(0, index=pool.index)
    )
    sf_col = (
        pd.to_numeric(pool["sf"], errors="coerce").fillna(0) if "sf" in pool.columns else pd.Series(0, index=pool.index)
    )
    obp_denom = ab + bb_col + hbp_col + sf_col
    pool["obp"] = np.where(obp_denom > 0, (h + bb_col + hbp_col) / obp_denom, 0.0)

    pool.drop(columns=["_orig_health"], inplace=True, errors="ignore")

    return pool


def _load_health_scores() -> dict[int, float]:
    """Load per-player health scores from injury_history table."""
    try:
        from src.database import get_connection
        from src.injury_model import compute_health_score

        conn = get_connection()
        try:
            df = pd.read_sql_query(
                "SELECT player_id, games_played, games_available FROM injury_history ORDER BY player_id, season DESC",
                conn,
            )
        finally:
            conn.close()

        scores: dict[int, float] = {}
        if not df.empty:
            for pid, group in df.groupby("player_id"):
                gp = group["games_played"].tolist()[:3]
                ga = group["games_available"].tolist()[:3]
                scores[int(pid)] = compute_health_score(gp, ga)
        return scores
    except Exception:
        logger.debug("Could not load health scores, using defaults", exc_info=True)
        return {}


def _load_roster_statuses() -> dict[int, str]:
    """Load per-player roster status from league_rosters table."""
    try:
        from src.database import get_connection

        conn = get_connection()
        try:
            df = pd.read_sql_query(
                "SELECT player_id, status FROM league_rosters WHERE status IS NOT NULL",
                conn,
            )
        finally:
            conn.close()

        if df.empty:
            return {}
        return dict(zip(df["player_id"].astype(int), df["status"].astype(str)))
    except Exception:
        logger.debug("Could not load roster statuses", exc_info=True)
        return {}


# ---------------------------------------------------------------------------
# Category-weighted SGP
# ---------------------------------------------------------------------------


def get_category_weights(
    user_team_name: str,
    all_team_totals: dict[str, dict[str, float]],
    config: LeagueConfig | None = None,
    weeks_remaining: int = 16,
) -> dict[str, float]:
    """Compute marginal category weights from standings gap analysis.

    Returns weights where categories you can gain positions in are
    weighted higher, and punted categories get zero weight.
    """
    if config is None:
        config = LeagueConfig()

    user_totals = all_team_totals.get(user_team_name, {})
    if not user_totals or len(all_team_totals) < 2:
        # Can't compute weights without standings context
        return {cat: 1.0 for cat in config.all_categories}

    try:
        from src.engine.portfolio.category_analysis import (
            category_gap_analysis,
            compute_category_weights_from_analysis,
        )

        analysis = category_gap_analysis(
            your_totals=user_totals,
            all_team_totals=all_team_totals,
            your_team_id=user_team_name,
            weeks_remaining=weeks_remaining,
        )
        return compute_category_weights_from_analysis(analysis)
    except Exception:
        logger.debug("Category weight computation failed, using equal weights", exc_info=True)
        return {cat: 1.0 for cat in config.all_categories}


# ---------------------------------------------------------------------------
# Free agent gating
# ---------------------------------------------------------------------------


def compute_fa_comparisons(
    opponent_player_ids: list[int],
    user_roster_ids: list[int],
    fa_pool: pd.DataFrame,
    player_pool: pd.DataFrame,
    config: LeagueConfig | None = None,
) -> dict[int, dict]:
    """Pre-compute FA alternatives for a batch of opponent players.

    For each opponent player, finds the best available free agent at the
    same position and compares their marginal value.

    Returns:
        Dict mapping ``player_id`` -> ``{"has_alternative": bool,
        "fa_name": str, "fa_value": float, "target_value": float,
        "fa_pct": float}``.
    """
    if config is None:
        config = LeagueConfig()

    results: dict[int, dict] = {}

    if fa_pool.empty or player_pool.empty:
        return results

    # Dynamic threshold based on average PA in the pool
    avg_pool_pa = float(player_pool["pa"].mean()) if "pa" in player_pool.columns else 0
    dynamic_threshold = compute_dynamic_fa_threshold(avg_pool_pa)

    # Pre-compute FA marginal values by position
    fa_by_pos: dict[str, list[tuple[str, float]]] = {}
    for _, fa in fa_pool.iterrows():
        positions = str(fa.get("positions", "")).split(",")
        sgp = _quick_player_sgp(fa, config)
        name = str(fa.get("player_name", fa.get("name", "")))
        for pos in positions:
            pos = pos.strip()
            if pos:
                fa_by_pos.setdefault(pos, []).append((name, sgp))

    # Sort each position's FAs by value (best first)
    for pos in fa_by_pos:
        fa_by_pos[pos].sort(key=lambda x: x[1], reverse=True)

    # Compare each opponent player to best FA at their position
    for pid in opponent_player_ids:
        p = player_pool[player_pool["player_id"] == pid]
        if p.empty:
            continue
        p = p.iloc[0]
        positions = str(p.get("positions", "")).split(",")
        target_sgp = _quick_player_sgp(p, config)

        best_fa_name = ""
        best_fa_value = 0.0
        for pos in positions:
            pos = pos.strip()
            candidates = fa_by_pos.get(pos, [])
            if candidates:
                name, val = candidates[0]
                if val > best_fa_value:
                    best_fa_name = name
                    best_fa_value = val

        if target_sgp > 0.01:
            fa_pct = best_fa_value / target_sgp
        elif target_sgp < -0.01:
            # Pitcher: both negative. More negative = better.
            # ratio of absolute values gives correct comparison.
            fa_pct = best_fa_value / target_sgp  # neg/neg = positive
        else:
            fa_pct = 0.0
        results[pid] = {
            "has_alternative": fa_pct >= dynamic_threshold,
            "fa_name": best_fa_name,
            "fa_value": round(best_fa_value, 2),
            "target_value": round(target_sgp, 2),
            "fa_pct": round(fa_pct, 2),
        }

    return results


def _quick_player_sgp(player_row: Any, config: LeagueConfig) -> float:
    """Compute a quick total SGP for a single player row."""
    total = 0.0
    for cat in config.hitting_categories + config.pitching_categories:
        col = cat.lower()
        val = float(player_row.get(col, 0) or 0)
        denom = config.sgp_denominators.get(cat, 1.0)
        if denom > 0:
            sgp = val / denom
            if cat in config.inverse_stats:
                sgp = -sgp
            total += sgp
    return total


# ---------------------------------------------------------------------------
# Scarcity premium
# ---------------------------------------------------------------------------


def apply_scarcity_flags(player_pool: pd.DataFrame) -> pd.DataFrame:
    """Add scarcity premium flags to the player pool.

    Players with projected SV >= 5 get a closer flag.
    Players at C/SS/2B get a scarce position flag.

    Returns:
        Pool with ``is_closer`` and ``scarcity_mult`` columns added.
    """
    pool = player_pool.copy()
    sv = pd.to_numeric(pool.get("sv", 0), errors="coerce").fillna(0)
    pool["is_closer"] = sv >= 5

    def _scarcity_mult(row):
        if row.get("is_closer", False):
            return SV_SCARCITY_MULT
        positions = set(p.strip() for p in str(row.get("positions", "")).split(","))
        if positions & SCARCE_POSITIONS:
            return SCARCE_POS_MULT
        return 1.0

    pool["scarcity_mult"] = pool.apply(_scarcity_mult, axis=1)
    return pool


# ---------------------------------------------------------------------------
# Trade Readiness composite score
# ---------------------------------------------------------------------------


def compute_trade_readiness(
    player_id: int,
    user_roster_ids: list[int],
    user_totals: dict[str, float],
    all_team_totals: dict[str, dict[str, float]],
    user_team_name: str,
    fa_pool: pd.DataFrame,
    player_pool: pd.DataFrame,
    config: LeagueConfig | None = None,
) -> dict:
    """Compute a 0-100 Trade Readiness composite score for a player.

    Components (weights):
        - category_fit (40%): SGP contribution to user's weak categories
        - projection_quality (25%): Bayesian confidence from sample size
        - health (15%): 3-year injury history health score
        - scarcity (10%): Closer/scarce position premium
        - fa_advantage (10%): How much better than best available FA

    Returns:
        Dict with ``score`` (0-100) and all sub-component values.
    """
    if config is None:
        config = LeagueConfig()

    p = player_pool[player_pool["player_id"] == player_id]
    if p.empty:
        return {"score": 0, "category_fit": 0, "projection_quality": 0, "health": 0, "scarcity": 0, "fa_advantage": 0}
    p = p.iloc[0]

    # --- 1. Category fit (40%) ---
    cat_weights = get_category_weights(user_team_name, all_team_totals, config)
    weighted_sgp = 0.0
    for cat in config.all_categories:
        col = cat.lower()
        val = float(p.get(col, 0) or 0)
        denom = config.sgp_denominators.get(cat, 1.0)
        w = cat_weights.get(cat, 1.0)
        if denom > 0:
            sgp = val / denom
            if cat in config.inverse_stats:
                sgp = -sgp
            weighted_sgp += sgp * w

    # Normalize to 0-100 (typical range: -5 to +15 weighted SGP)
    cat_fit = max(0, min(100, (weighted_sgp + 5) * (100 / 20)))

    # --- 2. Projection quality (25%) ---
    is_hitter = bool(p.get("is_hitter", 1))
    if is_hitter:
        pa = float(p.get("pa", 0) or 0)
        proj_quality = min(pa / _AVG_HITTER_STAB, 1.0) * 100
    else:
        ip = float(p.get("ip", 0) or 0)
        proj_quality = min(ip / _AVG_PITCHER_STAB, 1.0) * 100

    # --- 3. Health score (15%) ---
    health = float(p.get("health_score", 0.85)) * 100

    # --- 4. Scarcity premium (10%) ---
    scarcity_mult = float(p.get("scarcity_mult", 1.0))
    scarcity = min(100, (scarcity_mult - 0.9) * (100 / 0.5))  # 1.0→20, 1.3→80

    # --- 5. FA advantage (10%) ---
    fa_comp = compute_fa_comparisons([player_id], user_roster_ids, fa_pool, player_pool, config).get(player_id, {})
    target_val = fa_comp.get("target_value", 1.0)
    fa_val = fa_comp.get("fa_value", 0.0)
    if target_val > 0:
        fa_adv = max(0, (target_val - fa_val) / target_val) * 100
    else:
        fa_adv = 0.0

    # --- Composite ---
    score = 0.40 * cat_fit + 0.25 * proj_quality + 0.15 * health + 0.10 * scarcity + 0.10 * fa_adv

    return {
        "score": round(max(0, min(100, score)), 1),
        "category_fit": round(cat_fit, 1),
        "projection_quality": round(proj_quality, 1),
        "health": round(health, 1),
        "scarcity": round(scarcity, 1),
        "fa_advantage": round(fa_adv, 1),
        "fa_best": fa_comp.get("fa_name", ""),
    }


def compute_trade_readiness_batch(
    player_ids: list[int],
    user_roster_ids: list[int],
    user_totals: dict[str, float],
    all_team_totals: dict[str, dict[str, float]],
    user_team_name: str,
    fa_pool: pd.DataFrame,
    player_pool: pd.DataFrame,
    config: LeagueConfig | None = None,
    max_players: int = 100,
) -> pd.DataFrame:
    """Compute Trade Readiness scores for a batch of players.

    Pre-computes shared context (category weights, FA comparisons) once
    to avoid redundant work across 100+ players.

    Returns:
        DataFrame with columns: player_id, name, positions, score,
        category_fit, projection_quality, health, scarcity, fa_advantage, fa_best.
    """
    if config is None:
        config = LeagueConfig()

    # Limit to top players by raw SGP to avoid computing 300+ scores
    candidates = player_pool[player_pool["player_id"].isin(player_ids)].copy()
    if candidates.empty:
        return pd.DataFrame()

    # Deduplicate — keep first occurrence of each player_id
    candidates = candidates.drop_duplicates(subset=["player_id"], keep="first")

    candidates["_raw_sgp"] = candidates.apply(lambda r: _quick_player_sgp(r, config), axis=1)
    if "is_hitter" in candidates.columns:
        hitter_mask = candidates["is_hitter"].astype(bool)
    else:
        # Fallback: classify by position string
        hitter_mask = ~candidates.get("positions", pd.Series("", index=candidates.index)).str.contains(
            "SP|RP|P$", na=False, regex=True
        )
    n_half = max(max_players // 2, 1)
    top_hitters = candidates[hitter_mask].nlargest(n_half, "_raw_sgp")
    top_pitchers = candidates[~hitter_mask].nlargest(n_half, "_raw_sgp")
    candidates = pd.concat([top_hitters, top_pitchers])

    # Pre-compute shared context
    cat_weights = get_category_weights(user_team_name, all_team_totals, config)
    fa_comps = compute_fa_comparisons(candidates["player_id"].tolist(), user_roster_ids, fa_pool, player_pool, config)

    rows = []
    for _, p in candidates.iterrows():
        pid = int(p["player_id"])

        # Category fit
        weighted_sgp = 0.0
        for cat in config.all_categories:
            col = cat.lower()
            val = float(p.get(col, 0) or 0)
            denom = config.sgp_denominators.get(cat, 1.0)
            w = cat_weights.get(cat, 1.0)
            if denom > 0:
                sgp = val / denom
                if cat in config.inverse_stats:
                    sgp = -sgp
                weighted_sgp += sgp * w
        cat_fit = max(0, min(100, (weighted_sgp + 5) * (100 / 20)))

        # Projection quality
        is_hitter = bool(p.get("is_hitter", 1))
        if is_hitter:
            pa = float(p.get("pa", 0) or 0)
            proj_quality = min(pa / _AVG_HITTER_STAB, 1.0) * 100
        else:
            ip = float(p.get("ip", 0) or 0)
            proj_quality = min(ip / _AVG_PITCHER_STAB, 1.0) * 100

        # Health
        health = float(p.get("health_score", 0.85)) * 100

        # Scarcity
        scarcity_mult = float(p.get("scarcity_mult", 1.0))
        scarcity = min(100, (scarcity_mult - 0.9) * (100 / 0.5))

        # FA advantage
        fa_comp = fa_comps.get(pid, {})
        target_val = fa_comp.get("target_value", 1.0)
        fa_val = fa_comp.get("fa_value", 0.0)
        fa_adv = max(0, (target_val - fa_val) / target_val) * 100 if target_val > 0 else 0.0

        score = 0.40 * cat_fit + 0.25 * proj_quality + 0.15 * health + 0.10 * scarcity + 0.10 * fa_adv

        rows.append(
            {
                "player_id": pid,
                "name": str(p.get("name", p.get("player_name", ""))),
                "positions": str(p.get("positions", "")),
                "team": str(p.get("team", "")),
                "score": round(max(0, min(100, score)), 1),
                "category_fit": round(cat_fit, 1),
                "projection_quality": round(proj_quality, 1),
                "health": round(health, 1),
                "scarcity": round(scarcity, 1),
                "fa_advantage": round(fa_adv, 1),
                "fa_best": fa_comp.get("fa_name", ""),
                "is_closer": bool(p.get("is_closer", False)),
                "status": str(p.get("status", "active")),
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("score", ascending=False).reset_index(drop=True)
    return result

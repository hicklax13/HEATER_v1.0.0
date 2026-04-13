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
SB_INDEPENDENCE_PREMIUM = 1.15  # H4: 15% premium (R²=0.0002 with RBI — nearly independent)
CONTACT_CLUSTER_DISCOUNT = 0.90  # 10% discount
PITCHING_RATE_DISCOUNT = 0.88  # 12% discount


def apply_time_decay(
    player_sgp_by_cat: dict[str, float],
    weeks_remaining: int,
    total_weeks: int = 24,
    config: Any = None,
) -> dict[str, float]:
    """Apply differential time decay to per-category SGP.

    Counting stats: scale by weeks_remaining / total_weeks (linear decay)
    Rate stats: stay at 1.0 but apply confidence penalty below 8 weeks

    Args:
        player_sgp_by_cat: Per-category SGP values {cat: sgp_value}.
        weeks_remaining: Weeks left in the fantasy season.
        total_weeks: Total weeks in the fantasy season.
        config: League configuration for identifying rate stats.

    Returns:
        Adjusted {cat: sgp} dict with time decay applied.
    """
    if config is None:
        config = LeagueConfig()

    time_fraction = weeks_remaining / max(total_weeks, 1)
    rate_confidence = min(1.0, weeks_remaining / 8.0)

    adjusted = {}
    for cat, sgp in player_sgp_by_cat.items():
        if cat in config.rate_stats:
            # Rate stats: constant value, confidence penalty late season
            adjusted[cat] = sgp * rate_confidence
        else:
            # Counting stats: linear decay
            adjusted[cat] = sgp * time_fraction
    return adjusted


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


def _get_prior_season_pa_ip(player_id: int, is_hitter: bool) -> float:
    """Query total PA (hitters) or IP (pitchers) from prior seasons (2024-2025).

    Used by T3-3 to enrich projection confidence with career sample size.
    Returns 0.0 on any error so callers degrade gracefully.
    """
    try:
        from src.database import get_connection

        conn = get_connection()
        try:
            cursor = conn.execute(
                "SELECT COALESCE(SUM(pa), 0) AS total_pa, COALESCE(SUM(ip), 0) AS total_ip "
                "FROM season_stats WHERE player_id = ? AND season IN (2024, 2025)",
                (player_id,),
            )
            row = cursor.fetchone()
            if row:
                return float(row[0]) if is_hitter else float(row[1])
        finally:
            conn.close()
    except Exception:
        pass
    return 0.0


def _batch_prior_season_pa_ip(player_ids: list[int]) -> dict[tuple[int, bool], float]:
    """Batch-query prior season PA/IP for multiple players.

    Returns dict mapping (player_id, is_hitter_bool) -> total PA or IP.
    """
    result: dict[tuple[int, bool], float] = {}
    if not player_ids:
        return result
    try:
        from src.database import get_connection

        conn = get_connection()
        try:
            placeholders = ",".join("?" for _ in player_ids)
            cursor = conn.execute(
                f"SELECT player_id, COALESCE(SUM(pa), 0), COALESCE(SUM(ip), 0) "
                f"FROM season_stats WHERE player_id IN ({placeholders}) "
                f"AND season IN (2024, 2025) GROUP BY player_id",
                player_ids,
            )
            for row in cursor.fetchall():
                pid = int(row[0])
                result[(pid, True)] = float(row[1])  # PA for hitters
                result[(pid, False)] = float(row[2])  # IP for pitchers
        finally:
            conn.close()
    except Exception:
        pass
    return result


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

    # --- Load health scores and statuses ---
    # If the enriched player pool already has these columns (from _enrich_pool()
    # in database.py), skip the expensive DB queries. The health_score and status
    # columns are already set with display-safe values.
    if "health_score" in pool.columns and "status" in pool.columns:
        pool["_orig_health"] = pool["health_score"].copy()
    else:
        health_scores = _load_health_scores()
        roster_statuses = _load_roster_statuses()
        pool["health_score"] = pool["player_id"].map(health_scores).fillna(0.85)
        pool["_orig_health"] = pool["health_score"].copy()
        pool["status"] = pool["player_id"].map(roster_statuses).fillna("active")

        # Adjust health scores based on current IL/DTD status
        for idx, row in pool.iterrows():
            status = str(row.get("status", "active")).lower().strip()
            if status in ("il10", "il15", "dl") and row.get("health_score", 0.85) >= 0.80:
                pool.at[idx, "health_score"] = 0.65
            elif status in ("il60", "out") and row.get("health_score", 0.85) >= 0.60:
                pool.at[idx, "health_score"] = 0.40
            elif status == "dtd" and row.get("health_score", 0.85) >= 0.80:
                pool.at[idx, "health_score"] = 0.75

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
                "SELECT player_id, games_played, games_available FROM injury_history WHERE season >= 2025 ORDER BY player_id, season DESC",
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

    If the enriched player pool already has these columns (from
    ``_enrich_pool()`` in ``database.py``), returns immediately.

    Returns:
        Pool with ``is_closer`` and ``scarcity_mult`` columns added.
    """
    # Short-circuit if pool already enriched
    if "is_closer" in player_pool.columns and "scarcity_mult" in player_pool.columns:
        return player_pool

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
        # T3-3: Enrich with career PA from prior seasons for confidence
        prior_pa = _get_prior_season_pa_ip(player_id, is_hitter=True)
        effective_pa = pa + 0.3 * prior_pa
        proj_quality = min(effective_pa / _AVG_HITTER_STAB, 1.0) * 100
    else:
        ip = float(p.get("ip", 0) or 0)
        prior_ip = _get_prior_season_pa_ip(player_id, is_hitter=False)
        effective_ip = ip + 0.3 * prior_ip
        proj_quality = min(effective_ip / _AVG_PITCHER_STAB, 1.0) * 100

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

    # T3-3: Batch-load prior season PA/IP for projection confidence
    prior_pa_ip_cache = _batch_prior_season_pa_ip(candidates["player_id"].tolist())

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

        # Projection quality (T3-3: enriched with career PA/IP)
        is_hitter = bool(p.get("is_hitter", 1))
        if is_hitter:
            pa = float(p.get("pa", 0) or 0)
            prior_pa = prior_pa_ip_cache.get((pid, True), 0.0)
            effective_pa = pa + 0.3 * prior_pa
            proj_quality = min(effective_pa / _AVG_HITTER_STAB, 1.0) * 100
        else:
            ip = float(p.get("ip", 0) or 0)
            prior_ip = prior_pa_ip_cache.get((pid, False), 0.0)
            effective_ip = ip + 0.3 * prior_ip
            proj_quality = min(effective_ip / _AVG_PITCHER_STAB, 1.0) * 100

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


# ---------------------------------------------------------------------------
# Category need scores
# ---------------------------------------------------------------------------


def compute_category_need_scores(
    gap_analysis: dict[str, dict[str, Any]],
    config: LeagueConfig | None = None,
) -> dict[str, float]:
    """Convert category gap analysis into 0-1 need scores.

    Higher score = you need improvement in this category more.
    Used to weight trade evaluations toward addressing weaknesses.

    Need tiers (based on rank):
        Rank 10-12 + punt: 0.0 (punted, don't invest)
        Rank 8-9: 1.0 (critical weakness)
        Rank 5-7: 0.6 (competitive gap)
        Rank 3-4: 0.3 (slight edge, protect)
        Rank 1-2: 0.1 (dominant, can afford to trade from)

    Args:
        gap_analysis: Output of ``category_gap_analysis()`` — dict mapping
            category name to ``{"rank": int, "is_punt": bool,
            "marginal_value": float, "gap_to_next": float,
            "gainable_positions": int}``.
        config: League configuration (uses default if None).

    Returns:
        Dict mapping category -> need score (0.0-1.0).
    """
    config = config or LeagueConfig()

    # Default: 0.5 for all categories when no analysis available
    if not gap_analysis:
        return {cat: 0.5 for cat in config.all_categories}

    scores: dict[str, float] = {}
    for cat in config.all_categories:
        info = gap_analysis.get(cat)
        if info is None:
            scores[cat] = 0.5
            continue

        rank = info.get("rank", 6)
        is_punt = info.get("is_punt", False)

        if is_punt:
            scores[cat] = 0.0
        elif rank >= 10:
            # Rank 10-12 but not officially punt — still very weak
            scores[cat] = 0.0
        elif rank >= 8:
            scores[cat] = 1.0
        elif rank >= 5:
            scores[cat] = 0.6
        elif rank >= 3:
            scores[cat] = 0.3
        else:
            scores[cat] = 0.1

    return scores


# ---------------------------------------------------------------------------
# Need-efficiency trade scoring
# ---------------------------------------------------------------------------


def score_trade_by_need_efficiency(
    category_impact: dict[str, float],
    category_needs: dict[str, float],
    config: LeagueConfig | None = None,
) -> dict[str, Any]:
    """Score a trade by how efficiently it boosts weak categories at the expense of strong ones.

    For each category in *category_impact*:

    * If the impact is **positive** (gaining SGP), it is weighted by the
      category need — gaining SGP in a category you desperately need is
      worth more than gaining in one where you are already dominant.
    * If the impact is **negative** (losing SGP), the cost is weighted by
      how *expendable* that category is. Losing from strength is cheaper
      than losing from weakness.

    Args:
        category_impact: Per-category SGP change from the trade
            (positive = gaining, negative = losing).
        category_needs: Per-category need scores (0-1) from
            ``compute_category_need_scores()``.
        config: League configuration (uses default if None).

    Returns:
        Dict with keys:
            efficiency_ratio: float (higher = smarter trade, >1.0 means
                gaining more need-weighted value than losing)
            need_weighted_gain: float (sum of SGP gains weighted by need)
            affordability_weighted_cost: float (sum of SGP losses weighted
                by how much it hurts to lose them)
            boosted_cats: list[str] (categories improved)
            costly_cats: list[str] (categories worsened)
    """
    config = config or LeagueConfig()

    need_weighted_gain = 0.0
    affordability_weighted_cost = 0.0
    boosted_cats: list[str] = []
    costly_cats: list[str] = []

    for cat, impact in category_impact.items():
        need = category_needs.get(cat, 0.5)

        if impact > 0.05:
            # Gaining — weight by how much we need it
            need_weighted_gain += impact * need
            boosted_cats.append(cat)
        elif impact < -0.05:
            # Losing — affordability = 1.0 - need (high need = low affordability)
            affordability = 1.0 - need
            # Cost = |impact| * (1 - affordability) = |impact| * need
            affordability_weighted_cost += abs(impact) * (1.0 - affordability)
            costly_cats.append(cat)

    efficiency_ratio = need_weighted_gain / max(affordability_weighted_cost, 0.01)
    # Cap at 5x to prevent unrealistic ratios from dominating rankings
    efficiency_ratio = min(efficiency_ratio, 5.0)

    return {
        "efficiency_ratio": round(efficiency_ratio, 3),
        "need_weighted_gain": round(need_weighted_gain, 3),
        "affordability_weighted_cost": round(affordability_weighted_cost, 3),
        "boosted_cats": boosted_cats,
        "costly_cats": costly_cats,
    }


# ---------------------------------------------------------------------------
# Targeted trade proposal generator
# ---------------------------------------------------------------------------


def generate_targeted_proposals(
    target_player_id: int,
    user_roster_ids: list[int],
    player_pool: pd.DataFrame,
    config: LeagueConfig | None = None,
    all_team_totals: dict[str, dict[str, float]] | None = None,
    user_team_name: str | None = None,
    opponent_team_name: str | None = None,
    weeks_remaining: int = 22,
) -> dict[str, Any]:
    """Generate lowball and fair value trade proposals for a target player.

    Builds two proposals targeting different acceptance probability ranges:

    * **Lowball** (~30% acceptance) — cheapest offer that is not insulting.
      Give SGP between 55-75% of target SGP.
    * **Fair value** (~60% acceptance) — competitive offer the opponent
      should seriously consider. Give SGP between 85-115% of target SGP.

    For each proposal the function runs the trade evaluator, computes
    acceptance probability, ADP fairness, and need-efficiency.

    Args:
        target_player_id: ``player_id`` of the player you want to acquire.
        user_roster_ids: List of ``player_id`` on your roster.
        player_pool: Full player pool DataFrame.
        config: League configuration (uses default if None).
        all_team_totals: All teams' category totals for gap analysis.
        user_team_name: Your team name in the standings.
        opponent_team_name: The team that owns the target player.
        weeks_remaining: Weeks left in the fantasy season.

    Returns:
        Dict with keys:
            target: dict (player info — name, positions, sgp, stats, ecr)
            lowball: dict | None (proposal targeting ~30% acceptance)
            fair_value: dict | None (proposal targeting ~60% acceptance)

        Each proposal contains:
            giving_ids, giving_names, grade, surplus_sgp, category_impact,
            acceptance_probability, adp_fairness, ecr_fairness, efficiency,
            historical_stats, ecr_ranks
    """
    config = config or LeagueConfig()
    sgp_calc = _import_sgp_calculator(config)

    # --- Load IL stash names to exclude from give candidates ---
    il_stash_names = _load_il_stash_names()

    # --- Target player info ---
    target_rows = player_pool[player_pool["player_id"] == target_player_id]
    if target_rows.empty:
        logger.warning("Target player_id %d not found in player pool", target_player_id)
        return {"target": {}, "lowball": None, "fair_value": None}

    target = target_rows.iloc[0]
    target_sgp = sgp_calc.total_sgp(target) if sgp_calc else _quick_player_sgp(target, config)
    target_name = str(target.get("name", target.get("player_name", "Unknown")))
    target_positions = str(target.get("positions", ""))

    # --- User category needs ---
    gap_analysis = _get_user_gap_analysis(user_team_name, all_team_totals, weeks_remaining)
    category_needs = compute_category_need_scores(gap_analysis, config)

    # --- Opponent needs (best-effort) ---
    opp_needs_analysis: dict[str, dict] = {}
    if opponent_team_name and all_team_totals:
        try:
            from src.opponent_trade_analysis import compute_opponent_needs

            opp_needs_analysis = compute_opponent_needs(opponent_team_name, all_team_totals, weeks_remaining)
        except Exception:
            logger.debug("Could not compute opponent needs", exc_info=True)

    # --- Build give candidate list from user roster ---
    candidates = _build_give_candidates(user_roster_ids, player_pool, config, sgp_calc, il_stash_names)
    if not candidates:
        logger.info("No tradeable candidates on user roster")
        return {
            "target": _target_info(target, target_sgp, target_positions),
            "lowball": None,
            "fair_value": None,
        }

    # Sort by SGP ascending (cheapest first)
    candidates.sort(key=lambda c: c["sgp"])

    # --- Generate proposals ---
    lowball = _find_proposal(
        candidates,
        target_sgp,
        target_player_id,
        user_roster_ids,
        player_pool,
        config,
        category_needs,
        opp_needs_analysis,
        opponent_team_name,
        weeks_remaining,
        min_frac=0.55,
        max_frac=0.75,
        label="lowball",
        user_team_name=user_team_name,
    )

    fair_value = _find_proposal(
        candidates,
        target_sgp,
        target_player_id,
        user_roster_ids,
        player_pool,
        config,
        category_needs,
        opp_needs_analysis,
        opponent_team_name,
        weeks_remaining,
        min_frac=0.85,
        max_frac=1.15,
        label="fair_value",
        user_team_name=user_team_name,
    )

    # --- Load historical stats and ECR for all involved players ---
    involved_ids = [target_player_id]
    if lowball:
        involved_ids.extend(lowball.get("giving_ids", []))
    if fair_value:
        involved_ids.extend(fair_value.get("giving_ids", []))
    involved_ids = list(set(involved_ids))

    historical = _load_historical_stats(involved_ids)
    ecr_ranks = _load_ecr_ranks(involved_ids)

    # Attach stats and ECR to proposals
    target_info = _target_info(target, target_sgp, target_positions)
    target_info["historical_stats"] = historical.get(target_player_id, {})
    target_info["ecr_rank"] = ecr_ranks.get(target_player_id)

    for proposal in (lowball, fair_value):
        if proposal is not None:
            proposal["historical_stats"] = {pid: historical.get(pid, {}) for pid in proposal.get("giving_ids", [])}
            proposal["ecr_ranks"] = {pid: ecr_ranks.get(pid) for pid in proposal.get("giving_ids", [])}

    return {
        "target": target_info,
        "lowball": lowball,
        "fair_value": fair_value,
    }


# ---------------------------------------------------------------------------
# Private helpers for generate_targeted_proposals
# ---------------------------------------------------------------------------


def _import_sgp_calculator(config: LeagueConfig):
    """Import and instantiate SGPCalculator, returning None on failure."""
    try:
        from src.valuation import SGPCalculator

        return SGPCalculator(config)
    except Exception:
        logger.debug("Could not import SGPCalculator", exc_info=True)
        return None


def _load_il_stash_names() -> set[str]:
    """Load IL stash player names from alerts module."""
    try:
        from src.alerts import IL_STASH_NAMES

        return set(IL_STASH_NAMES)
    except Exception:
        return set()


def _get_user_gap_analysis(
    user_team_name: str | None,
    all_team_totals: dict[str, dict[str, float]] | None,
    weeks_remaining: int,
) -> dict[str, dict]:
    """Run category gap analysis for the user, returning empty on failure."""
    if not user_team_name or not all_team_totals:
        return {}
    user_totals = all_team_totals.get(user_team_name, {})
    if not user_totals:
        return {}
    try:
        from src.engine.portfolio.category_analysis import category_gap_analysis

        return category_gap_analysis(
            your_totals=user_totals,
            all_team_totals=all_team_totals,
            your_team_id=user_team_name,
            weeks_remaining=weeks_remaining,
        )
    except Exception:
        logger.debug("Could not compute user gap analysis", exc_info=True)
        return {}


def _target_info(target_row: Any, target_sgp: float, positions: str) -> dict:
    """Build a summary dict for the target player."""
    return {
        "player_id": int(target_row.get("player_id", 0)),
        "name": str(target_row.get("name", target_row.get("player_name", "Unknown"))),
        "positions": positions,
        "team": str(target_row.get("team", "")),
        "sgp": round(target_sgp, 2),
    }


def _build_give_candidates(
    user_roster_ids: list[int],
    player_pool: pd.DataFrame,
    config: LeagueConfig,
    sgp_calc,
    il_stash_names: set[str],
) -> list[dict]:
    """Build sorted list of tradeable roster players with SGP values."""
    candidates: list[dict] = []
    for pid in user_roster_ids:
        rows = player_pool[player_pool["player_id"] == pid]
        if rows.empty:
            continue
        row = rows.iloc[0]
        name = str(row.get("name", row.get("player_name", "")))

        # Exclude IL stash players
        if name in il_stash_names:
            continue

        # Exclude NA/minors and IL players (injured players can't be traded realistically)
        status = str(row.get("status", "active")).lower().strip()
        if status in ("na", "not active", "minors", "il", "il10", "il15", "il60", "dl", "out"):
            continue

        # Also check selected_position for IL (Yahoo roster slot)
        sel_pos = str(row.get("selected_position", "")).upper().strip()
        if sel_pos == "IL":
            continue

        sgp = sgp_calc.total_sgp(row) if sgp_calc else _quick_player_sgp(row, config)
        candidates.append(
            {
                "player_id": pid,
                "name": name,
                "sgp": sgp,
                "positions": str(row.get("positions", "")),
            }
        )
    return candidates


def _find_proposal(
    candidates: list[dict],
    target_sgp: float,
    target_player_id: int,
    user_roster_ids: list[int],
    player_pool: pd.DataFrame,
    config: LeagueConfig,
    category_needs: dict[str, float],
    opp_needs_analysis: dict[str, dict],
    opponent_team_name: str | None,
    weeks_remaining: int,
    min_frac: float,
    max_frac: float,
    label: str,
    user_team_name: str | None = None,
) -> dict | None:
    """Find a 1-for-1 or 2-for-1 proposal within the given SGP fraction range."""
    min_sgp = target_sgp * min_frac
    max_sgp = target_sgp * max_frac

    # --- Try 1-for-1 ---
    for c in candidates:
        if min_sgp <= c["sgp"] <= max_sgp:
            return _evaluate_proposal(
                giving=[c],
                target_player_id=target_player_id,
                user_roster_ids=user_roster_ids,
                player_pool=player_pool,
                config=config,
                category_needs=category_needs,
                opp_needs_analysis=opp_needs_analysis,
                opponent_team_name=opponent_team_name,
                weeks_remaining=weeks_remaining,
                label=label,
                user_team_name=user_team_name,
            )

    # --- Try 2-for-1: find cheapest pair that sums into range ---
    for i, c1 in enumerate(candidates):
        for c2 in candidates[i + 1 :]:
            total_sgp = c1["sgp"] + c2["sgp"]
            if min_sgp <= total_sgp <= max_sgp:
                return _evaluate_proposal(
                    giving=[c1, c2],
                    target_player_id=target_player_id,
                    user_roster_ids=user_roster_ids,
                    player_pool=player_pool,
                    config=config,
                    category_needs=category_needs,
                    opp_needs_analysis=opp_needs_analysis,
                    opponent_team_name=opponent_team_name,
                    weeks_remaining=weeks_remaining,
                    label=label,
                    user_team_name=user_team_name,
                )

    return None


def _evaluate_proposal(
    giving: list[dict],
    target_player_id: int,
    user_roster_ids: list[int],
    player_pool: pd.DataFrame,
    config: LeagueConfig,
    category_needs: dict[str, float],
    opp_needs_analysis: dict[str, dict],
    opponent_team_name: str | None,
    weeks_remaining: int,
    label: str,
    user_team_name: str | None = None,
) -> dict:
    """Evaluate a single proposal: run trade evaluator, acceptance, efficiency."""
    giving_ids = [c["player_id"] for c in giving]
    giving_names = [c["name"] for c in giving]
    receiving_ids = [target_player_id]

    # --- Run trade evaluator (Phase 1 only for speed) ---
    eval_result: dict[str, Any] = {}
    try:
        from src.engine.output.trade_evaluator import evaluate_trade

        eval_result = evaluate_trade(
            giving_ids=giving_ids,
            receiving_ids=receiving_ids,
            user_roster_ids=user_roster_ids,
            player_pool=player_pool,
            config=config,
            user_team_name=user_team_name,
            weeks_remaining=weeks_remaining,
            enable_mc=False,
            enable_context=False,
            enable_game_theory=False,
        )
    except Exception:
        logger.debug("Trade evaluator failed for %s proposal", label, exc_info=True)

    grade = eval_result.get("grade", "?")
    surplus_sgp = eval_result.get("surplus_sgp", 0.0)
    category_impact = eval_result.get("category_impact", {})

    # --- Fallback: raw per-player SGP delta when LP produces all zeros ---
    # The LP optimizer can produce all-zero impacts when both traded players
    # are bench-quality and don't affect the optimal starting lineup.
    # In that case, compute a direct player-vs-player SGP comparison.
    all_zero = all(abs(v) < 0.001 for v in category_impact.values()) if category_impact else True
    if all_zero:
        try:
            sgp_calc = _import_sgp_calculator(config)
            if sgp_calc is not None:
                # Sum SGP of all give players
                give_sgps: dict[str, float] = {cat: 0.0 for cat in config.all_categories}
                for gid in giving_ids:
                    g_rows = player_pool[player_pool["player_id"] == gid]
                    if not g_rows.empty:
                        g_player_sgp = sgp_calc.player_sgp(g_rows.iloc[0])
                        for cat, val in g_player_sgp.items():
                            give_sgps[cat] = give_sgps.get(cat, 0.0) + val

                # SGP of received player
                r_rows = player_pool[player_pool["player_id"] == target_player_id]
                if not r_rows.empty:
                    recv_sgps = sgp_calc.player_sgp(r_rows.iloc[0])
                    category_impact = {
                        cat: round(recv_sgps.get(cat, 0.0) - give_sgps.get(cat, 0.0), 3)
                        for cat in config.all_categories
                    }
                    surplus_sgp = sum(category_impact.values())
        except Exception:
            logger.debug("Raw SGP fallback failed", exc_info=True)

    # --- Acceptance probability ---
    acceptance = 0.0
    adp_fair = 0.5
    try:
        from src.trade_finder import compute_adp_fairness, estimate_acceptance_probability

        # ADP fairness: use first give player vs target
        adp_fair = compute_adp_fairness(giving_ids[0], target_player_id, player_pool)

        # Opponent need match
        opp_need_match = 0.5
        if opp_needs_analysis:
            opp_need_scores = compute_category_need_scores(opp_needs_analysis, config)
            # How well do our give players address opponent needs?
            matched = sum(
                1 for cat, need in opp_need_scores.items() if need >= 0.6 and category_impact.get(cat, 0) < -0.05
            )
            total_needs = sum(1 for v in opp_need_scores.values() if v >= 0.6)
            opp_need_match = matched / max(total_needs, 1)

        # Opponent trade willingness
        opp_willingness = 0.5
        if opponent_team_name:
            try:
                from src.opponent_trade_analysis import get_opponent_archetype

                archetype = get_opponent_archetype(opponent_team_name)
                opp_willingness = archetype.get("trade_willingness", 0.5)
            except Exception:
                pass

        acceptance = estimate_acceptance_probability(
            user_gain_sgp=surplus_sgp,
            opponent_gain_sgp=-surplus_sgp,
            need_match_score=opp_need_match,
            adp_fairness=adp_fair,
            opponent_need_match=opp_need_match,
            opponent_trade_willingness=opp_willingness,
        )
    except Exception:
        logger.debug("Acceptance probability computation failed", exc_info=True)

    # --- ECR fairness (if available) ---
    ecr_fair = 0.5
    try:
        ecr_ranks = _load_ecr_ranks(giving_ids + [target_player_id])
        give_ecr = ecr_ranks.get(giving_ids[0])
        recv_ecr = ecr_ranks.get(target_player_id)
        if give_ecr and recv_ecr and give_ecr > 0 and recv_ecr > 0:
            import math

            ecr_fair = math.sqrt(min(give_ecr, recv_ecr) / max(give_ecr, recv_ecr))
    except Exception:
        pass

    # --- Need efficiency ---
    efficiency = score_trade_by_need_efficiency(category_impact, category_needs, config)

    return {
        "giving_ids": giving_ids,
        "giving_names": giving_names,
        "grade": grade,
        "surplus_sgp": round(surplus_sgp, 2) if isinstance(surplus_sgp, (int, float)) else 0.0,
        "category_impact": category_impact,
        "acceptance_probability": round(acceptance, 3),
        "adp_fairness": round(adp_fair, 3),
        "ecr_fairness": round(ecr_fair, 3),
        "efficiency": efficiency,
    }


def _load_historical_stats(player_ids: list[int]) -> dict[int, dict]:
    """Load 2025 + 2026 season stats for a list of players from SQLite."""
    if not player_ids:
        return {}
    try:
        from src.database import get_connection

        conn = get_connection()
        try:
            placeholders = ",".join("?" for _ in player_ids)
            df = pd.read_sql_query(
                f"SELECT player_id, season, pa, ab, h, r, hr, rbi, sb, avg, "
                f"ip, w, sv, k, era, whip "
                f"FROM season_stats WHERE player_id IN ({placeholders}) "
                f"AND season IN (2025, 2026) ORDER BY player_id, season DESC",
                conn,
                params=player_ids,
            )
        finally:
            conn.close()

        result: dict[int, dict] = {}
        if df.empty:
            return result

        for pid, group in df.groupby("player_id"):
            pid_int = int(pid)
            seasons: dict[int, dict] = {}
            for _, row in group.iterrows():
                season = int(row["season"])
                seasons[season] = {
                    col: (float(row[col]) if pd.notna(row[col]) else 0.0)
                    for col in ["pa", "ab", "h", "r", "hr", "rbi", "sb", "avg", "ip", "w", "sv", "k", "era", "whip"]
                }
            result[pid_int] = seasons
        return result
    except Exception:
        logger.debug("Could not load historical stats", exc_info=True)
        return {}


def _load_ecr_ranks(player_ids: list[int]) -> dict[int, int | None]:
    """Load consensus ECR rank for a list of players from SQLite."""
    if not player_ids:
        return {}
    try:
        from src.database import get_connection

        conn = get_connection()
        try:
            placeholders = ",".join("?" for _ in player_ids)
            df = pd.read_sql_query(
                f"SELECT player_id, consensus_rank FROM ecr_consensus WHERE player_id IN ({placeholders})",
                conn,
                params=player_ids,
            )
        finally:
            conn.close()

        if df.empty:
            return {pid: None for pid in player_ids}

        ranks: dict[int, int | None] = {}
        for _, row in df.iterrows():
            pid = int(row["player_id"])
            rank = row["consensus_rank"]
            ranks[pid] = int(rank) if pd.notna(rank) else None

        # Fill missing
        for pid in player_ids:
            if pid not in ranks:
                ranks[pid] = None
        return ranks
    except Exception:
        logger.debug("Could not load ECR ranks", exc_info=True)
        return {pid: None for pid in player_ids}


# ---------------------------------------------------------------------------
# Auto-scan trade recommendations by category need
# ---------------------------------------------------------------------------


def recommend_trades_by_need(
    user_roster_ids: list[int],
    player_pool: pd.DataFrame,
    config: LeagueConfig | None = None,
    all_team_totals: dict[str, dict[str, float]] | None = None,
    user_team_name: str | None = None,
    league_rosters: pd.DataFrame | None = None,
    weeks_remaining: int = 22,
    max_results: int = 20,
) -> list[dict[str, Any]]:
    """Auto-scan all opponents and produce ranked trade recommendations.

    Prioritizes trades that boost weak categories the most for the least
    cost in strong categories (category need efficiency).

    Composite score: 25% efficiency + 30% acceptance + 15% ADP + 15% ECR + 15% YTD

    Args:
        user_roster_ids: List of player_id on the user's roster.
        player_pool: Full player pool DataFrame.
        config: League configuration (uses default if None).
        all_team_totals: All teams' category totals for gap analysis.
        user_team_name: User's team name in standings.
        league_rosters: DataFrame with columns ``team_name``, ``player_id``
            (and optionally ``status``, ``name``).
        weeks_remaining: Weeks left in the fantasy season.
        max_results: Maximum trades to return.

    Returns:
        List of dicts sorted by composite score descending, each containing:
        giving_id, giving_name, giving_positions, receiving_id,
        receiving_name, receiving_positions, opponent_team, user_sgp_gain,
        category_impact, need_efficiency, acceptance_probability,
        adp_fairness, ecr_fairness, ytd_modifier, composite_score,
        grade_estimate, boosted_cats, costly_cats, give_ecr_rank,
        recv_ecr_rank.
    """
    config = config or LeagueConfig()
    sgp_calc = _import_sgp_calculator(config)
    il_stash_names = _load_il_stash_names()

    # ── 1. User category needs ──────────────────────────────────────────
    gap_analysis = _get_user_gap_analysis(user_team_name, all_team_totals, weeks_remaining)
    category_needs = compute_category_need_scores(gap_analysis, config)

    # ── 2. Build user give-candidates (reuse existing helper) ───────────
    user_candidates = _build_give_candidates(
        user_roster_ids,
        player_pool,
        config,
        sgp_calc,
        il_stash_names,
    )
    if not user_candidates:
        logger.info("recommend_trades_by_need: no tradeable user roster players")
        return []

    user_id_set = set(user_roster_ids)

    # ── 3. Build opponent rosters: {team_name: [player_ids]} ────────────
    opponent_rosters: dict[str, list[int]] = {}
    if league_rosters is not None and not league_rosters.empty:
        for team_name, grp in league_rosters.groupby("team_name"):
            team_str = str(team_name)
            # Skip user's own team
            if user_team_name and team_str == user_team_name:
                continue
            pids = [int(pid) for pid in grp["player_id"].dropna().unique() if int(pid) not in user_id_set]
            if pids:
                opponent_rosters[team_str] = pids

    if not opponent_rosters:
        logger.info("recommend_trades_by_need: no opponent rosters available")
        return []

    # ── 4. Batch-load ECR ranks and YTD stats for ALL relevant players ──
    all_candidate_ids = [c["player_id"] for c in user_candidates]
    all_opp_ids: list[int] = []
    for pids in opponent_rosters.values():
        all_opp_ids.extend(pids)
    all_player_ids = list(set(all_candidate_ids + all_opp_ids))

    ecr_ranks = _load_ecr_ranks(all_player_ids)
    ytd_stats = _batch_load_ytd_stats(all_player_ids)

    # ── 5. Precompute per-category SGP for each user candidate ──────────
    user_cat_sgps: dict[int, dict[str, float]] = {}
    for c in user_candidates:
        user_cat_sgps[c["player_id"]] = _player_category_sgps(
            c["player_id"],
            player_pool,
            config,
            sgp_calc,
        )

    # ── 6. Opponent archetype cache (best-effort) ───────────────────────
    opp_willingness_cache: dict[str, float] = {}
    try:
        from src.opponent_trade_analysis import get_opponent_archetype

        for team_name in opponent_rosters:
            try:
                arch = get_opponent_archetype(team_name)
                opp_willingness_cache[team_name] = arch.get("trade_willingness", 0.5)
            except Exception:
                opp_willingness_cache[team_name] = 0.5
    except ImportError:
        pass

    # ── 7. Scan all opponent players ────────────────────────────────────
    raw_results: list[dict[str, Any]] = []

    for team_name, opp_pids in opponent_rosters.items():
        opp_willingness = opp_willingness_cache.get(team_name, 0.5)

        for recv_id in opp_pids:
            recv_rows = player_pool[player_pool["player_id"] == recv_id]
            if recv_rows.empty:
                continue
            recv = recv_rows.iloc[0]

            # Skip NA/injured/minors
            recv_status = str(recv.get("status", "active")).lower().strip()
            if recv_status in ("na", "not active", "minors"):
                continue
            recv_name = str(recv.get("name", recv.get("player_name", "Unknown")))
            if recv_name in il_stash_names:
                continue

            recv_sgp = sgp_calc.total_sgp(recv) if sgp_calc else _quick_player_sgp(recv, config)
            recv_positions = str(recv.get("positions", ""))
            recv_cat_sgps = _player_category_sgps(recv_id, player_pool, config, sgp_calc)

            # ── Find best 1-for-1 give candidate ────────────────────────
            best_give: dict[str, Any] | None = None
            best_efficiency: float = -999.0

            for c in user_candidates:
                give_sgp = c["sgp"]
                # Quick SGP ratio filter: skip wildly unfair pairs
                if recv_sgp > 0 and give_sgp / max(recv_sgp, 0.01) < 0.30:
                    continue
                if recv_sgp > 0 and give_sgp / max(recv_sgp, 0.01) > 3.0:
                    continue

                # Quick category impact: recv_cat - give_cat per category
                cat_impact: dict[str, float] = {}
                for cat in config.all_categories:
                    recv_val = recv_cat_sgps.get(cat, 0.0)
                    give_val = user_cat_sgps.get(c["player_id"], {}).get(cat, 0.0)
                    cat_impact[cat] = recv_val - give_val

                eff = score_trade_by_need_efficiency(cat_impact, category_needs, config)
                eff_ratio = eff.get("efficiency_ratio", 0.0)

                if eff_ratio > best_efficiency:
                    best_efficiency = eff_ratio
                    best_give = {
                        "candidate": c,
                        "cat_impact": cat_impact,
                        "efficiency": eff,
                        "user_sgp_gain": recv_sgp - give_sgp,
                    }

            if best_give is None or best_efficiency <= 0:
                continue

            give_c = best_give["candidate"]
            cat_impact = best_give["cat_impact"]
            eff = best_give["efficiency"]
            user_sgp_gain = best_give["user_sgp_gain"]

            # ── Acceptance probability ──────────────────────────────────
            try:
                from src.trade_finder import (
                    compute_adp_fairness,
                    estimate_acceptance_probability,
                )

                adp_fair = compute_adp_fairness(give_c["player_id"], recv_id, player_pool)

                p_accept = estimate_acceptance_probability(
                    user_gain_sgp=user_sgp_gain,
                    opponent_gain_sgp=-user_sgp_gain,
                    need_match_score=0.5,
                    adp_fairness=adp_fair,
                    opponent_need_match=0.5,
                    opponent_trade_willingness=opp_willingness,
                )
            except Exception:
                adp_fair = 0.5
                p_accept = 0.3

            # ── ECR fairness ────────────────────────────────────────────
            ecr_fair = 0.5
            give_ecr = ecr_ranks.get(give_c["player_id"])
            recv_ecr = ecr_ranks.get(recv_id)
            if give_ecr and recv_ecr and give_ecr > 0 and recv_ecr > 0:
                import math

                ecr_fair = math.sqrt(min(give_ecr, recv_ecr) / max(give_ecr, recv_ecr))

            # ── YTD modifier ────────────────────────────────────────────
            ytd_mod = 1.0
            recv_ytd = ytd_stats.get(recv_id, {})
            if recv_ytd.get("pa", 0) >= 10:
                proj_avg = float(recv.get("avg", 0.260) or 0.260)
                ytd_avg = recv_ytd.get("avg", proj_avg)
                if proj_avg > 0:
                    ytd_mod = max(0.90, min(1.10, ytd_avg / proj_avg))

            # ── Acceptance floor: skip trades below 15% acceptance ──────
            if p_accept < 0.15:
                continue

            # ── Composite score ─────────────────────────────────────────
            # Acceptance is heaviest factor — a brilliant trade that gets
            # rejected is worthless.
            norm_eff = min(eff.get("efficiency_ratio", 0.0) / 5.0, 1.0)
            # Normalize ytd_mod from [0.90, 1.10] to [0.0, 1.0] so it differentiates
            ytd_score = (ytd_mod - 0.90) / 0.20
            composite = 0.25 * norm_eff + 0.30 * p_accept + 0.15 * adp_fair + 0.15 * ecr_fair + 0.15 * ytd_score

            # ── Grade estimate (quick heuristic from SGP delta) ─────────
            grade_estimate = _estimate_grade(user_sgp_gain)

            raw_results.append(
                {
                    "giving_id": give_c["player_id"],
                    "giving_name": give_c["name"],
                    "giving_positions": give_c["positions"],
                    "receiving_id": recv_id,
                    "receiving_name": recv_name,
                    "receiving_positions": recv_positions,
                    "opponent_team": team_name,
                    "user_sgp_gain": round(user_sgp_gain, 2),
                    "category_impact": {k: round(v, 3) for k, v in cat_impact.items()},
                    "need_efficiency": round(eff.get("efficiency_ratio", 0.0), 3),
                    "acceptance_probability": round(p_accept, 3),
                    "adp_fairness": round(adp_fair, 3),
                    "ecr_fairness": round(ecr_fair, 3),
                    "ytd_modifier": round(ytd_mod, 3),
                    "composite_score": round(composite, 4),
                    "grade_estimate": grade_estimate,
                    "boosted_cats": eff.get("boosted_cats", []),
                    "costly_cats": eff.get("costly_cats", []),
                    "give_ecr_rank": give_ecr,
                    "recv_ecr_rank": recv_ecr,
                }
            )

    # ── 8. Sort by composite, take top 50 for full evaluation ───────────
    raw_results.sort(key=lambda x: x["composite_score"], reverse=True)
    top_candidates = raw_results[: min(50, len(raw_results))]

    # ── 9. Run full evaluate_trade() on top candidates ─────────────────
    # This replaces the quick SGP estimate with LP-constrained category
    # impacts, real grades, and real surplus SGP.
    try:
        from src.engine.output.trade_evaluator import evaluate_trade as _eval_trade

        for rec in top_candidates:
            try:
                eval_result = _eval_trade(
                    giving_ids=[rec["giving_id"]],
                    receiving_ids=[rec["receiving_id"]],
                    user_roster_ids=user_roster_ids,
                    player_pool=player_pool,
                    config=config,
                    user_team_name=user_team_name,
                    weeks_remaining=weeks_remaining,
                    enable_mc=False,
                    enable_context=False,
                    enable_game_theory=False,
                )
                # Override quick estimates with real engine results
                real_impact = eval_result.get("category_impact", {})
                # If LP produced all zeros, keep the quick estimate
                if real_impact and not all(abs(v) < 0.001 for v in real_impact.values()):
                    rec["category_impact"] = {k: round(v, 3) for k, v in real_impact.items()}
                    rec["user_sgp_gain"] = round(eval_result.get("surplus_sgp", rec["user_sgp_gain"]), 2)
                    # Recompute efficiency with real impacts
                    real_eff = score_trade_by_need_efficiency(real_impact, category_needs, config)
                    rec["need_efficiency"] = round(real_eff.get("efficiency_ratio", 0.0), 3)
                    rec["boosted_cats"] = real_eff.get("boosted_cats", [])
                    rec["costly_cats"] = real_eff.get("costly_cats", [])
                rec["grade_estimate"] = eval_result.get("grade", rec["grade_estimate"])
            except Exception:
                pass  # Keep quick estimate if engine fails
    except ImportError:
        logger.debug("evaluate_trade not available for full evaluation pass")

    # ── 10. Re-sort after full evaluation and return top results ───────
    # Recompute composite with real data
    for rec in top_candidates:
        norm_eff = min(rec.get("need_efficiency", 0.0) / 5.0, 1.0)
        # Normalize ytd_modifier from [0.90, 1.10] to [0.0, 1.0]
        ytd_score = (rec.get("ytd_modifier", 1.0) - 0.90) / 0.20
        rec["composite_score"] = round(
            0.25 * norm_eff
            + 0.30 * rec.get("acceptance_probability", 0.3)
            + 0.15 * rec.get("adp_fairness", 0.5)
            + 0.15 * rec.get("ecr_fairness", 0.5)
            + 0.15 * ytd_score,
            4,
        )
    top_candidates.sort(key=lambda x: x["composite_score"], reverse=True)
    return top_candidates[:max_results]


# ---------------------------------------------------------------------------
# Private helpers for recommend_trades_by_need
# ---------------------------------------------------------------------------


def _player_category_sgps(
    player_id: int,
    player_pool: pd.DataFrame,
    config: LeagueConfig,
    sgp_calc: Any,
) -> dict[str, float]:
    """Compute per-category SGP for a single player (lightweight).

    Uses ``SGPCalculator.player_sgp()`` when available, which correctly
    handles rate stats (AVG, OBP, ERA, WHIP) via volume weighting.
    Falls back to naive counting-stat division only when the calculator
    is unavailable.
    """
    rows = player_pool[player_pool["player_id"] == player_id]
    if rows.empty:
        return {}
    row = rows.iloc[0]

    # Prefer SGPCalculator — it handles rate stats correctly
    if sgp_calc is not None:
        try:
            return sgp_calc.player_sgp(row)
        except Exception:
            pass  # Fall through to manual computation

    # Fallback: manual computation (counting stats only are reliable)
    result: dict[str, float] = {}
    for cat in config.all_categories:
        col = cat.lower()
        val = float(row.get(col, 0) or 0)
        denom = config.sgp_denominators.get(cat, 1.0)
        if cat in config.rate_stats:
            # Rate stats need volume weighting — skip in fallback
            result[cat] = 0.0
        elif denom > 0:
            sgp = val / denom
            if cat in config.inverse_stats:
                sgp = -sgp
            result[cat] = sgp
        else:
            result[cat] = 0.0
    return result


def _batch_load_ytd_stats(player_ids: list[int]) -> dict[int, dict]:
    """Batch-load 2026 YTD stats for a list of players from SQLite."""
    if not player_ids:
        return {}
    try:
        from src.database import get_connection

        conn = get_connection()
        try:
            placeholders = ",".join("?" for _ in player_ids)
            df = pd.read_sql_query(
                f"SELECT player_id, pa, avg, hr, rbi, sb, era, whip "
                f"FROM season_stats WHERE season = 2026 AND pa > 0 "
                f"AND player_id IN ({placeholders})",
                conn,
                params=player_ids,
            )
        finally:
            conn.close()

        result: dict[int, dict] = {}
        if df.empty:
            return result
        for _, row in df.iterrows():
            pid = int(row["player_id"])
            result[pid] = {
                "pa": int(row.get("pa", 0) or 0),
                "avg": float(row.get("avg", 0) or 0),
                "hr": int(row.get("hr", 0) or 0),
                "rbi": int(row.get("rbi", 0) or 0),
                "sb": int(row.get("sb", 0) or 0),
                "era": float(row.get("era", 0) or 0),
                "whip": float(row.get("whip", 0) or 0),
            }
        return result
    except Exception:
        logger.debug("Could not batch-load YTD stats", exc_info=True)
        return {}


def _estimate_grade(sgp_gain: float) -> str:
    """Quick letter grade from SGP delta (no full trade evaluation)."""
    if sgp_gain >= 3.0:
        return "A+"
    if sgp_gain >= 2.0:
        return "A"
    if sgp_gain >= 1.0:
        return "B+"
    if sgp_gain >= 0.5:
        return "B"
    if sgp_gain >= 0.0:
        return "C+"
    if sgp_gain >= -0.5:
        return "C"
    if sgp_gain >= -1.0:
        return "D"
    return "F"

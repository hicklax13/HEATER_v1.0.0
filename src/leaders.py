"""Category leaders, points leaders, and breakout detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

INVERSE_CATS = {"ERA", "WHIP", "L"}


def compute_category_leaders(
    season_stats_df: pd.DataFrame,
    categories: list[str] | None = None,
    min_pa: int = 50,
    min_ip: float = 20.0,
    top_n: int = 20,
) -> dict[str, pd.DataFrame]:
    """Compute leaders per category. Ascending sort for ERA/WHIP/L."""
    cats = categories or ["R", "HR", "RBI", "SB", "AVG", "OBP", "W", "SV", "K", "ERA", "WHIP"]
    stat_map = {
        "R": "r",
        "HR": "hr",
        "RBI": "rbi",
        "SB": "sb",
        "AVG": "avg",
        "OBP": "obp",
        "W": "w",
        "SV": "sv",
        "K": "k",
        "ERA": "era",
        "WHIP": "whip",
        "L": "l",
    }
    leaders = {}
    for cat in cats:
        col = stat_map.get(cat, cat.lower())
        if col not in season_stats_df.columns:
            continue
        df = season_stats_df.copy()
        # Filter by minimum playing time
        is_pitch = cat in {"W", "SV", "K", "ERA", "WHIP", "L"}
        if is_pitch and "ip" in df.columns:
            df = df[pd.to_numeric(df["ip"], errors="coerce").fillna(0) >= min_ip]
        elif not is_pitch and "pa" in df.columns:
            df = df[pd.to_numeric(df["pa"], errors="coerce").fillna(0) >= min_pa]
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        ascending = cat in INVERSE_CATS
        df = df.sort_values(col, ascending=ascending).head(top_n)
        leaders[cat] = df
    return leaders


def compute_points_leaders(
    season_stats_df: pd.DataFrame,
    hitting_weights: dict[str, float],
    pitching_weights: dict[str, float],
    top_n: int = 20,
) -> pd.DataFrame:
    """Top N players by fantasy points."""
    try:
        from src.points_league import compute_fantasy_points

        result = compute_fantasy_points(season_stats_df, hitting_weights, pitching_weights)
        return result.sort_values("fantasy_points", ascending=False).head(top_n).reset_index(drop=True)
    except ImportError:
        return pd.DataFrame()


def compute_category_value_leaders(
    season_stats_df: pd.DataFrame,
    min_pa: int = 50,
    min_ip: float = 20.0,
    top_n: int = 20,
) -> pd.DataFrame:
    """Rank players by z-score composite across H2H Categories scoring.

    For each of the 12 scoring categories (R, HR, RBI, SB, AVG, OBP,
    W, L, SV, K, ERA, WHIP), compute each player's z-score relative
    to the eligible population.  Inverse stats (L, ERA, WHIP) are
    sign-flipped so lower raw values produce higher z-scores.

    Hitters are scored on hitting cats only; pitchers on pitching only.
    The resulting ``category_value`` is the sum of z-scores — a single
    number that captures overall category impact in an H2H league.

    Returns a DataFrame sorted by ``category_value`` descending with
    the top *top_n* players.
    """
    if season_stats_df.empty:
        return pd.DataFrame()

    hit_cats = {"r": False, "hr": False, "rbi": False, "sb": False, "avg": False, "obp": False}
    pit_cats = {"w": False, "l": True, "sv": False, "k": False, "era": True, "whip": True}

    df = season_stats_df.copy()

    # Coerce all stat columns to numeric
    for col in list(hit_cats) + list(pit_cats) + ["pa", "ip"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Split hitters and pitchers
    hitters = df.copy()
    pitchers = df.copy()
    if "is_hitter" in df.columns:
        hitters = df[df["is_hitter"].astype(int) == 1].copy()
        pitchers = df[df["is_hitter"].astype(int) == 0].copy()
    if "pa" in hitters.columns:
        hitters = hitters[hitters["pa"] >= min_pa]
    if "ip" in pitchers.columns:
        pitchers = pitchers[pitchers["ip"] >= min_ip]

    def _zscore_sum(sub_df: pd.DataFrame, cat_map: dict[str, bool]) -> pd.Series:
        total = pd.Series(0.0, index=sub_df.index)
        for col, is_inverse in cat_map.items():
            if col not in sub_df.columns:
                continue
            vals = sub_df[col].astype(float)
            std = vals.std()
            if std < 1e-9:
                continue
            z = (vals - vals.mean()) / std
            if is_inverse:
                z = -z  # Lower ERA/WHIP/L = higher z
            total = total + z
        return total

    # Compute z-scores
    if not hitters.empty:
        hitters["category_value"] = _zscore_sum(hitters, hit_cats).round(2)
    if not pitchers.empty:
        pitchers["category_value"] = _zscore_sum(pitchers, pit_cats).round(2)

    combined = pd.concat([hitters, pitchers], ignore_index=True)
    if combined.empty:
        return pd.DataFrame()

    combined = combined.sort_values("category_value", ascending=False).head(top_n)

    display_cols = ["player_id", "name", "team", "positions", "category_value", "mlb_id"]
    available = [c for c in display_cols if c in combined.columns]
    return combined[available].reset_index(drop=True)


def filter_leaders_by_position(leaders_df: pd.DataFrame, position: str, pos_col: str = "positions") -> pd.DataFrame:
    """Filter leaders to a specific position."""
    if pos_col not in leaders_df.columns:
        return leaders_df
    return leaders_df[leaders_df[pos_col].str.contains(position, case=False, na=False)]


def detect_breakouts(
    season_stats_df: pd.DataFrame,
    preseason_df: pd.DataFrame,
    stats: list[str] | None = None,
    threshold: float = 1.5,
) -> pd.DataFrame:
    """Detect players significantly outperforming projections.
    z = (observed - projected) / max(std, 0.01), breakout if z > threshold.
    """
    inverse_stats = {"era", "whip"}
    check_stats = stats or ["hr", "rbi", "sb", "k", "avg"]
    results = []
    name_col = "name" if "name" in season_stats_df.columns else "player_name"
    pre_name = "name" if "name" in preseason_df.columns else "player_name"
    pre_lookup = {}
    for _, row in preseason_df.iterrows():
        pre_lookup[str(row.get(pre_name, "")).lower()] = row
    for _, actual in season_stats_df.iterrows():
        pname = str(actual.get(name_col, "")).lower()
        if pname not in pre_lookup:
            continue
        projected = pre_lookup[pname]
        max_z = 0.0
        best_stat = ""
        for stat in check_stats:
            obs = float(actual.get(stat, 0) or 0)
            proj = float(projected.get(stat, 0) or 0)
            # Use 15% of projected as rough std, with stat-aware minimum
            min_std = 0.5 if stat in ("hr", "sb", "w", "sv") else 0.01 if stat in ("avg", "obp", "era", "whip") else 1.0
            std = max(min_std, abs(proj) * 0.15)
            # For inverse stats (ERA, WHIP), lower observed is better
            if stat in inverse_stats:
                z = (proj - obs) / std
            else:
                z = (obs - proj) / std
            if z > max_z:
                max_z = z
                best_stat = stat
        if max_z > threshold:
            results.append(
                {
                    "name": actual.get(name_col, ""),
                    "breakout_stat": best_stat,
                    "z_score": round(max_z, 2),
                }
            )
    return (
        pd.DataFrame(results).sort_values("z_score", ascending=False).reset_index(drop=True)
        if results
        else pd.DataFrame(columns=["name", "breakout_stat", "z_score"])
    )


# ---------------------------------------------------------------------------
# Statcast Breakout Score
# ---------------------------------------------------------------------------

# Columns that indicate enriched Statcast data is available
_HITTER_STATCAST_COLS = ("barrel_pct", "xwoba", "hard_hit_pct")
_PITCHER_STATCAST_COLS = ("stuff_plus", "pitching_plus")


def _percentile_rank(series: pd.Series, value: float) -> float:
    """Return 0-100 percentile rank of *value* within *series*.

    Uses the "weak" method: percentage of values strictly less than *value*.
    NaN values are excluded from the comparison pool.
    """
    clean = series.dropna()
    if clean.empty:
        return 50.0
    return float((clean < value).sum()) / len(clean) * 100.0


def _safe_float(val, default: float = 0.0) -> float:
    """Coerce *val* to float, returning *default* on failure."""
    try:
        v = float(val)
        return default if np.isnan(v) else v
    except (TypeError, ValueError):
        return default


def compute_breakout_score(player_row, pool: pd.DataFrame | None = None, config=None) -> float:
    """Composite Statcast breakout score (0-100).

    Hitters: barrel_rate percentile (30%) + xwOBA-wOBA gap (25%) +
             hard_hit_pct percentile (20%) + HR rate vs projection (15%) +
             K% improvement (10%)

    Pitchers: Stuff+ percentile (30%) + K-BB% (25%) +
              SIERA-ERA gap (20%) + SwStr% percentile (15%) +
              Pitching+ percentile (10%)

    Parameters
    ----------
    player_row : dict-like
        A single player row (Series or dict) with stat columns.
    pool : DataFrame, optional
        Full player pool used to compute percentile ranks.  When *None*,
        returns a neutral fallback score of 50.
    config : optional
        League config (currently unused, reserved for future weighting).

    Returns
    -------
    float
        Score clamped to [0, 100].  >70 = "Breakout Candidate".
    """
    if pool is None or pool.empty:
        return 50.0

    raw_hitter = player_row.get("is_hitter")
    is_hitter = bool(int(raw_hitter)) if raw_hitter is not None else True

    if is_hitter:
        return _hitter_breakout_score(player_row, pool)
    return _pitcher_breakout_score(player_row, pool)


def _hitter_breakout_score(row, pool: pd.DataFrame) -> float:
    """Breakout score for a hitter (0-100)."""
    hitters = pool.copy()
    if "is_hitter" in hitters.columns:
        hitters = hitters[hitters["is_hitter"].astype(int) == 1]
    if hitters.empty:
        return 50.0

    has_statcast = all(c in hitters.columns for c in _HITTER_STATCAST_COLS)

    if not has_statcast:
        return _fallback_hitter_score(row, hitters)

    # --- Component 1: Barrel rate percentile (30%) ---
    barrel = _safe_float(row.get("barrel_pct"))
    barrel_pctile = _percentile_rank(pd.to_numeric(hitters["barrel_pct"], errors="coerce"), barrel)

    # --- Component 2: xwOBA - wOBA gap (25%) ---
    xwoba = _safe_float(row.get("xwoba"))
    # Use actual wOBA if available; fall back to OBP as rough proxy
    woba_col = "woba" if "woba" in hitters.columns else "obp"
    woba = _safe_float(row.get(woba_col))
    gap = xwoba - woba  # positive = unlucky, breakout signal
    # Compute gap for all hitters to percentile-rank it
    hitters_xwoba = pd.to_numeric(hitters.get("xwoba"), errors="coerce").fillna(0)
    hitters_woba = pd.to_numeric(hitters.get(woba_col), errors="coerce").fillna(0)
    all_gaps = hitters_xwoba - hitters_woba
    gap_pctile = _percentile_rank(all_gaps, gap)

    # --- Component 3: Hard hit percentage percentile (20%) ---
    hard_hit = _safe_float(row.get("hard_hit_pct"))
    hard_hit_pctile = _percentile_rank(pd.to_numeric(hitters["hard_hit_pct"], errors="coerce"), hard_hit)

    # --- Component 4: HR rate vs projection (15%) ---
    hr_actual = _safe_float(row.get("hr"))
    # Try projected HR from "proj_hr" or fall back to "hr" column average
    hr_proj = _safe_float(row.get("proj_hr"))
    if hr_proj <= 0:
        hr_proj = max(1.0, float(pd.to_numeric(hitters.get("hr"), errors="coerce").mean()))
    hr_ratio = min(2.0, hr_actual / max(0.1, hr_proj))  # cap at 2x
    hr_score = hr_ratio / 2.0 * 100.0  # scale to 0-100

    # --- Component 5: K% improvement (10%) ---
    # Lower K% is better for hitters; we reward below-average K%
    if "k_pct" in hitters.columns:
        k_pct = _safe_float(row.get("k_pct"))
        # Invert: lower K% -> higher score
        k_pctile = 100.0 - _percentile_rank(pd.to_numeric(hitters["k_pct"], errors="coerce"), k_pct)
    else:
        k_pctile = 50.0

    score = barrel_pctile * 0.30 + gap_pctile * 0.25 + hard_hit_pctile * 0.20 + hr_score * 0.15 + k_pctile * 0.10
    return float(np.clip(score, 0.0, 100.0))


def _pitcher_breakout_score(row, pool: pd.DataFrame) -> float:
    """Breakout score for a pitcher (0-100)."""
    pitchers = pool.copy()
    if "is_hitter" in pitchers.columns:
        pitchers = pitchers[pitchers["is_hitter"].astype(int) == 0]
    if pitchers.empty:
        return 50.0

    has_statcast = any(c in pitchers.columns for c in _PITCHER_STATCAST_COLS)

    if not has_statcast:
        return _fallback_pitcher_score(row, pitchers)

    # --- Component 1: Stuff+ percentile (30%) ---
    if "stuff_plus" in pitchers.columns:
        stuff = _safe_float(row.get("stuff_plus"))
        stuff_pctile = _percentile_rank(pd.to_numeric(pitchers["stuff_plus"], errors="coerce"), stuff)
    else:
        stuff_pctile = 50.0

    # --- Component 2: K-BB% (25%) ---
    k_rate = _safe_float(row.get("k_pct", row.get("k9")))
    bb_rate = _safe_float(row.get("bb_pct", row.get("bb9")))
    k_bb = k_rate - bb_rate
    # Percentile across pool
    pit_k = pd.to_numeric(pitchers.get("k_pct", pitchers.get("k9", pd.Series(dtype=float))), errors="coerce").fillna(0)
    pit_bb = pd.to_numeric(pitchers.get("bb_pct", pitchers.get("bb9", pd.Series(dtype=float))), errors="coerce").fillna(
        0
    )
    all_k_bb = pit_k - pit_bb
    k_bb_pctile = _percentile_rank(all_k_bb, k_bb)

    # --- Component 3: SIERA-ERA gap (20%) ---
    # Positive gap (SIERA < ERA) means pitcher is better than ERA shows
    era = _safe_float(row.get("era"))
    siera = _safe_float(row.get("siera"))
    if siera > 0 and era > 0:
        era_gap = era - siera  # positive = pitcher better than ERA
        pit_era = pd.to_numeric(pitchers.get("era"), errors="coerce").fillna(0)
        pit_siera = pd.to_numeric(pitchers.get("siera"), errors="coerce").fillna(0)
        all_era_gaps = pit_era - pit_siera
        siera_pctile = _percentile_rank(all_era_gaps, era_gap)
    else:
        siera_pctile = 50.0

    # --- Component 4: SwStr% percentile (15%) ---
    if "swstr_pct" in pitchers.columns:
        swstr = _safe_float(row.get("swstr_pct"))
        swstr_pctile = _percentile_rank(pd.to_numeric(pitchers["swstr_pct"], errors="coerce"), swstr)
    else:
        swstr_pctile = 50.0

    # --- Component 5: Pitching+ percentile (10%) ---
    if "pitching_plus" in pitchers.columns:
        pplus = _safe_float(row.get("pitching_plus"))
        pplus_pctile = _percentile_rank(pd.to_numeric(pitchers["pitching_plus"], errors="coerce"), pplus)
    else:
        pplus_pctile = 50.0

    score = stuff_pctile * 0.30 + k_bb_pctile * 0.25 + siera_pctile * 0.20 + swstr_pctile * 0.15 + pplus_pctile * 0.10
    return float(np.clip(score, 0.0, 100.0))


def _fallback_hitter_score(row, hitters: pd.DataFrame) -> float:
    """Z-score fallback when Statcast columns are missing (hitters)."""
    score_parts = []
    for stat, weight, inverse in [
        ("hr", 0.30, False),
        ("rbi", 0.25, False),
        ("sb", 0.20, False),
        ("avg", 0.15, False),
        ("obp", 0.10, False),
    ]:
        if stat not in hitters.columns:
            score_parts.append(50.0 * weight)
            continue
        vals = pd.to_numeric(hitters[stat], errors="coerce")
        player_val = _safe_float(row.get(stat))
        pctile = _percentile_rank(vals, player_val)
        if inverse:
            pctile = 100.0 - pctile
        score_parts.append(pctile * weight)
    return float(np.clip(sum(score_parts), 0.0, 100.0))


def _fallback_pitcher_score(row, pitchers: pd.DataFrame) -> float:
    """Z-score fallback when Statcast columns are missing (pitchers)."""
    score_parts = []
    for stat, weight, inverse in [
        ("k", 0.30, False),
        ("era", 0.25, True),
        ("whip", 0.20, True),
        ("w", 0.15, False),
        ("sv", 0.10, False),
    ]:
        if stat not in pitchers.columns:
            score_parts.append(50.0 * weight)
            continue
        vals = pd.to_numeric(pitchers[stat], errors="coerce")
        player_val = _safe_float(row.get(stat))
        pctile = _percentile_rank(vals, player_val)
        if inverse:
            pctile = 100.0 - pctile
        score_parts.append(pctile * weight)
    return float(np.clip(sum(score_parts), 0.0, 100.0))


def compute_breakout_scores_batch(pool: pd.DataFrame, config=None) -> pd.DataFrame:
    """Add ``breakout_score`` column to player pool.

    Returns a **copy** of *pool* with the new column.  Each player's score
    is computed relative to the full pool using :func:`compute_breakout_score`.
    """
    if pool.empty:
        result = pool.copy()
        result["breakout_score"] = pd.Series(dtype=float)
        return result

    result = pool.copy()
    scores = []
    for _, row in result.iterrows():
        scores.append(compute_breakout_score(row, pool=pool, config=config))
    result["breakout_score"] = scores
    return result


def compute_projection_skew(player_pool: pd.DataFrame) -> pd.DataFrame:
    """O4: Detect projection system agreement/disagreement.

    When 5+ of 7 systems project above the consensus (blended) average,
    that's positive skew — the player is likely undervalued. Especially
    valuable for mid-round pitchers (+50-100% ROI per FanGraphs research).

    Returns pool copy with ``projection_skew`` column:
    - "Positive" when 5+ systems above consensus
    - "Negative" when 5+ systems below consensus
    - "" when balanced or insufficient data

    Also adds ``skew_ratio`` (float): fraction of systems above consensus (0-1).
    """
    result = player_pool.copy()
    result["projection_skew"] = ""
    result["skew_ratio"] = 0.5

    try:
        from src.database import get_connection

        conn = get_connection()
        try:
            # Load per-system projections for all players
            systems_df = pd.read_sql_query(
                """
                SELECT player_id, system,
                    COALESCE(hr, 0) + COALESCE(rbi, 0) + COALESCE(r, 0) + COALESCE(sb, 0) AS counting_sum,
                    COALESCE(avg, 0) AS avg_proj
                FROM projections
                WHERE system != 'blended'
                """,
                conn,
            )
        finally:
            conn.close()

        if systems_df.empty:
            return result

        # Get per-player blended (consensus) counting sum
        blended = systems_df.groupby("player_id")["counting_sum"].mean().to_dict()

        # Count systems above/below consensus per player
        skew_data = {}
        for pid, grp in systems_df.groupby("player_id"):
            consensus = blended.get(pid, 0)
            if consensus <= 0:
                continue
            n_systems = len(grp)
            if n_systems < 3:
                continue
            above = (grp["counting_sum"] > consensus).sum()
            ratio = above / n_systems
            skew_data[pid] = ratio

        result["skew_ratio"] = result["player_id"].map(skew_data).fillna(0.5)
        # 5/7 = 0.714 → positive skew. 2/7 = 0.286 → negative skew.
        result.loc[result["skew_ratio"] >= 0.70, "projection_skew"] = "Positive"
        result.loc[result["skew_ratio"] <= 0.30, "projection_skew"] = "Negative"

    except Exception:
        pass

    return result

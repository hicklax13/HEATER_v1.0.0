"""Player Trend Tracker: Hot/cold detection using Kalman filtering and regime analysis.

Computes player trends by comparing actual season stats against pre-season
projections. Uses rate-stat deltas for hitters (AVG, HR rate, RBI rate, SB rate)
and pitchers (ERA, WHIP, K rate) to classify players as HOT, COLD, or NEUTRAL.

Optional Kalman filter integration provides true-talent-filtered estimates for
breakout/decline detection. Sell-high candidate identification combines hot
trends with low sustainability scores (BABIP regression signals).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.valuation import LeagueConfig

# ── Optional imports ─────────────────────────────────────────────────

try:
    from src.engine.signals.kalman import run_kalman_for_feature

    _HAS_KALMAN = True
except ImportError:
    _HAS_KALMAN = False

try:
    from src.engine.signals.regime import classify_regime_simple  # noqa: F401

    _HAS_REGIME = True
except ImportError:
    _HAS_REGIME = False

try:
    from src.waiver_wire import compute_sustainability_score

    _HAS_SUSTAINABILITY = True
except ImportError:
    _HAS_SUSTAINABILITY = False


# ── Constants ────────────────────────────────────────────────────────

# Trend classification thresholds
HOT_THRESHOLD = 0.10
COLD_THRESHOLD = -0.10

# Kalman signal thresholds
BREAKOUT_THRESHOLD = 0.15
DECLINE_THRESHOLD = -0.15

# Sell-high: hot trend + sustainability below this value
SELL_HIGH_SUSTAINABILITY_CAP = 0.45

# Minimum plate appearances / innings pitched for trend reliability
MIN_HITTER_PA = 30
MIN_PITCHER_IP = 10.0

# Rate stat floor to avoid division by near-zero
RATE_FLOOR = 0.001

# Hitter key stats for trend computation
HITTER_RATE_STATS = ("avg", "hr_rate", "rbi_rate", "sb_rate")

# Pitcher key stats for trend computation (ERA/WHIP are inverse)
PITCHER_RATE_STATS = ("era", "whip", "k_rate")

# Inverse stats: lower actual is better, so delta sign flips
INVERSE_RATE_STATS = {"era", "whip"}


# ── Core Functions ───────────────────────────────────────────────────


def classify_trend(trend_delta: float) -> str:
    """Classify a trend delta into HOT, COLD, or NEUTRAL.

    Args:
        trend_delta: Average rate-stat delta (actual vs projected).

    Returns:
        "HOT" if trend_delta > 0.10, "COLD" if < -0.10, else "NEUTRAL".
    """
    if trend_delta > HOT_THRESHOLD:
        return "HOT"
    elif trend_delta < COLD_THRESHOLD:
        return "COLD"
    return "NEUTRAL"


def _compute_rate_delta(actual: float, projected: float, inverse: bool = False) -> float:
    """Compute normalized delta between actual and projected rate stat.

    Args:
        actual: Observed rate stat value.
        projected: Pre-season projected rate stat value.
        inverse: If True, flip sign (lower actual = better = positive delta).

    Returns:
        Normalized delta: (actual - projected) / max(|projected|, RATE_FLOOR).
        For inverse stats, sign is flipped so positive = improvement.
    """
    denom = max(abs(projected), RATE_FLOOR)
    raw_delta = (actual - projected) / denom
    if inverse:
        raw_delta = -raw_delta
    return raw_delta


def _hitter_rate_deltas(actual_row: pd.Series, proj_row: pd.Series) -> dict[str, float]:
    """Compute rate-stat deltas for a hitter.

    Compares actual season stats to projected stats for AVG, HR rate,
    RBI rate, and SB rate.

    Args:
        actual_row: Season stats row with pa, avg, hr, rbi, sb.
        proj_row: Projection row with pa, avg, hr, rbi, sb.

    Returns:
        Dict of {stat_name: delta} for each hitter rate stat.
    """
    pa_actual = float(actual_row.get("pa", 0) or 0)
    pa_proj = float(proj_row.get("pa", 0) or 0)

    deltas: dict[str, float] = {}

    # AVG: direct rate comparison
    avg_actual = float(actual_row.get("avg", 0) or 0)
    avg_proj = float(proj_row.get("avg", 0) or 0)
    deltas["avg"] = _compute_rate_delta(avg_actual, avg_proj)

    # HR rate: hr / pa
    hr_actual = float(actual_row.get("hr", 0) or 0)
    hr_proj = float(proj_row.get("hr", 0) or 0)
    hr_rate_actual = hr_actual / max(pa_actual, 1)
    hr_rate_proj = hr_proj / max(pa_proj, 1)
    deltas["hr_rate"] = _compute_rate_delta(hr_rate_actual, hr_rate_proj)

    # RBI rate: rbi / pa
    rbi_actual = float(actual_row.get("rbi", 0) or 0)
    rbi_proj = float(proj_row.get("rbi", 0) or 0)
    rbi_rate_actual = rbi_actual / max(pa_actual, 1)
    rbi_rate_proj = rbi_proj / max(pa_proj, 1)
    deltas["rbi_rate"] = _compute_rate_delta(rbi_rate_actual, rbi_rate_proj)

    # SB rate: sb / pa
    sb_actual = float(actual_row.get("sb", 0) or 0)
    sb_proj = float(proj_row.get("sb", 0) or 0)
    sb_rate_actual = sb_actual / max(pa_actual, 1)
    sb_rate_proj = sb_proj / max(pa_proj, 1)
    deltas["sb_rate"] = _compute_rate_delta(sb_rate_actual, sb_rate_proj)

    return deltas


def _pitcher_rate_deltas(actual_row: pd.Series, proj_row: pd.Series) -> dict[str, float]:
    """Compute rate-stat deltas for a pitcher.

    Compares actual season stats to projected stats for ERA, WHIP,
    and K rate. ERA and WHIP are inverse (lower = better).

    Args:
        actual_row: Season stats row with ip, era, whip, k.
        proj_row: Projection row with ip, era, whip, k.

    Returns:
        Dict of {stat_name: delta} for each pitcher rate stat.
    """
    ip_actual = float(actual_row.get("ip", 0) or 0)
    ip_proj = float(proj_row.get("ip", 0) or 0)

    deltas: dict[str, float] = {}

    # ERA (inverse: lower actual = positive delta)
    era_actual = float(actual_row.get("era", 0) or 0)
    era_proj = float(proj_row.get("era", 0) or 0)
    deltas["era"] = _compute_rate_delta(era_actual, era_proj, inverse=True)

    # WHIP (inverse: lower actual = positive delta)
    whip_actual = float(actual_row.get("whip", 0) or 0)
    whip_proj = float(proj_row.get("whip", 0) or 0)
    deltas["whip"] = _compute_rate_delta(whip_actual, whip_proj, inverse=True)

    # K rate: k / ip
    k_actual = float(actual_row.get("k", 0) or 0)
    k_proj = float(proj_row.get("k", 0) or 0)
    k_rate_actual = k_actual / max(ip_actual, 1)
    k_rate_proj = k_proj / max(ip_proj, 1)
    deltas["k_rate"] = _compute_rate_delta(k_rate_actual, k_rate_proj)

    return deltas


def compute_player_trends(
    player_pool: pd.DataFrame,
    season_stats: pd.DataFrame,
    config: LeagueConfig | None = None,
) -> pd.DataFrame:
    """Compute trend deltas for all players with both projections and season stats.

    Merges player pool (projections) with season stats, computes per-player
    rate-stat deltas, and classifies each player as HOT/COLD/NEUTRAL.

    Args:
        player_pool: DataFrame from load_player_pool() with projection columns.
        season_stats: DataFrame from load_season_stats() with actual stats.
        config: Optional LeagueConfig (unused currently, reserved for future).

    Returns:
        DataFrame: player_pool rows merged with trend_delta, trend_label,
        and per-stat delta columns. Only includes players present in both
        pool and season_stats with sufficient sample size.
    """
    if player_pool.empty or season_stats.empty:
        result = player_pool.copy()
        result["trend_delta"] = 0.0
        result["trend_label"] = "NEUTRAL"
        return result

    # Identify name column in pool
    name_col = "player_name" if "player_name" in player_pool.columns else "name"

    # Build lookup from season_stats by player_id
    stats_by_id: dict[int, pd.Series] = {}
    if "player_id" in season_stats.columns:
        for _, row in season_stats.iterrows():
            pid = int(row["player_id"])
            stats_by_id[pid] = row

    trend_deltas: list[float] = []
    trend_labels: list[str] = []
    stat_detail_rows: list[dict] = []

    for _, proj_row in player_pool.iterrows():
        pid = int(proj_row.get("player_id", 0))
        is_hitter = int(proj_row.get("is_hitter", 1) or 1)
        actual_row = stats_by_id.get(pid)

        if actual_row is None:
            trend_deltas.append(0.0)
            trend_labels.append("NEUTRAL")
            stat_detail_rows.append({})
            continue

        # Check minimum sample size
        if is_hitter:
            pa = float(actual_row.get("pa", 0) or 0)
            if pa < MIN_HITTER_PA:
                trend_deltas.append(0.0)
                trend_labels.append("NEUTRAL")
                stat_detail_rows.append({})
                continue
            deltas = _hitter_rate_deltas(actual_row, proj_row)
        else:
            ip = float(actual_row.get("ip", 0) or 0)
            if ip < MIN_PITCHER_IP:
                trend_deltas.append(0.0)
                trend_labels.append("NEUTRAL")
                stat_detail_rows.append({})
                continue
            deltas = _pitcher_rate_deltas(actual_row, proj_row)

        # Average delta across key stats
        if deltas:
            avg_delta = float(np.mean(list(deltas.values())))
        else:
            avg_delta = 0.0

        trend_deltas.append(avg_delta)
        trend_labels.append(classify_trend(avg_delta))
        stat_detail_rows.append(deltas)

    result = player_pool.copy()
    result["trend_delta"] = trend_deltas
    result["trend_label"] = trend_labels

    return result


def apply_kalman_trends(
    player_stats_series: list[dict],
    stat_name: str,
    projected_value: float,
) -> dict:
    """Apply Kalman filter to a player's stat time series for trend detection.

    Uses the Kalman filter to separate true talent from noise, then compares
    the filtered estimate to the pre-season projection.

    Args:
        player_stats_series: List of {value: float, sample_size: float} dicts
            sorted chronologically (e.g., weekly rolling averages).
        stat_name: Stat type key for Kalman variance calibration
            (e.g., "ba", "era", "k_rate").
        projected_value: Pre-season projection for this stat.

    Returns:
        Dict with:
          - filtered_mean: Kalman-filtered true talent estimate.
          - filtered_var: Remaining uncertainty.
          - kalman_delta: (filtered_mean - projected) / max(|projected|, 0.001).
          - kalman_signal: "BREAKOUT" if delta > 0.15, "DECLINING" if < -0.15,
            else "STABLE".
    """
    if not _HAS_KALMAN or not player_stats_series:
        return {
            "filtered_mean": projected_value,
            "filtered_var": 0.0,
            "kalman_delta": 0.0,
            "kalman_signal": "STABLE",
        }

    kalman_result = run_kalman_for_feature(
        rolling_values=player_stats_series,
        stat_type=stat_name,
        prior_mean=projected_value,
    )

    filtered_mean = kalman_result.get("filtered_mean", projected_value)
    filtered_var = kalman_result.get("filtered_var", 0.0)

    # Compute delta: how far has true talent shifted from projection?
    denom = max(abs(projected_value), RATE_FLOOR)
    kalman_delta = (filtered_mean - projected_value) / denom

    # Classify signal
    if kalman_delta > BREAKOUT_THRESHOLD:
        signal = "BREAKOUT"
    elif kalman_delta < DECLINE_THRESHOLD:
        signal = "DECLINING"
    else:
        signal = "STABLE"

    return {
        "filtered_mean": filtered_mean,
        "filtered_var": filtered_var,
        "kalman_delta": kalman_delta,
        "kalman_signal": signal,
    }


def detect_sell_high_candidates(
    player_pool: pd.DataFrame,
    season_stats: pd.DataFrame,
    config: LeagueConfig | None = None,
) -> pd.DataFrame:
    """Identify sell-high candidates: hot players with low sustainability.

    Players trending HOT whose underlying metrics suggest regression (high
    BABIP, unsustainable peripherals). These are players to trade away at
    peak perceived value.

    Args:
        player_pool: DataFrame from load_player_pool().
        season_stats: DataFrame from load_season_stats().
        config: Optional LeagueConfig.

    Returns:
        DataFrame subset of hot players with sustainability_score below
        SELL_HIGH_SUSTAINABILITY_CAP. Includes trend_delta, trend_label,
        and sustainability_score columns.
    """
    trended = compute_player_trends(player_pool, season_stats, config)

    # Filter to HOT players only
    hot_mask = trended["trend_label"] == "HOT"
    hot_players = trended[hot_mask].copy()

    if hot_players.empty:
        hot_players["sustainability_score"] = pd.Series(dtype=float)
        return hot_players

    # Compute sustainability for each hot player
    # Need season stats merged for BABIP calculation
    stats_by_id: dict[int, pd.Series] = {}
    if not season_stats.empty and "player_id" in season_stats.columns:
        for _, row in season_stats.iterrows():
            stats_by_id[int(row["player_id"])] = row

    sustainability_scores: list[float] = []
    for _, row in hot_players.iterrows():
        pid = int(row.get("player_id", 0))
        actual = stats_by_id.get(pid)

        if actual is not None and _HAS_SUSTAINABILITY:
            score = compute_sustainability_score(actual)
        else:
            # Default: assume moderately sustainable if we cannot compute
            score = 0.6

        sustainability_scores.append(score)

    hot_players["sustainability_score"] = sustainability_scores

    # Filter to low-sustainability hot players (sell-high targets)
    sell_high = hot_players[hot_players["sustainability_score"] < SELL_HIGH_SUSTAINABILITY_CAP].copy()

    # Sort by trend_delta descending (hottest first)
    sell_high = sell_high.sort_values("trend_delta", ascending=False)

    return sell_high

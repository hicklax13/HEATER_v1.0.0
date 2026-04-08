"""Playing time prediction model for in-season projection updates.

Blends preseason projected PA/IP with observed in-season usage rates
using a Bayesian-style weighted average. The blend shifts from projection-heavy
early in the season to observed-heavy as sample size grows.

This is the #1 source of projection error: static preseason PA/IP never
gets updated. This module fixes that.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Blend thresholds (observed PA accumulated so far)
EARLY_SEASON_PA_THRESHOLD: int = 50
MID_SEASON_PA_THRESHOLD: int = 200

# Blend weights: (projection_weight, observed_weight)
EARLY_SEASON_BLEND: tuple[float, float] = (0.80, 0.20)
MID_SEASON_BLEND: tuple[float, float] = (0.50, 0.50)
LATE_SEASON_BLEND: tuple[float, float] = (0.20, 0.80)

# Adjustments
BENCH_PLAYING_TIME_FACTOR: float = 0.40
PLATOON_REDUCTION: float = 0.35  # 35% reduction
RELIEVER_APPEARANCES_PER_WEEK: float = 2.5
RELIEVER_IP_PER_APPEARANCE: float = 1.0

# Full-season baseline for computing observed rate
FULL_SEASON_GAMES: int = 162


def _compute_blend_weights(observed_pa: float) -> tuple[float, float]:
    """Return (projection_weight, observed_weight) based on sample size.

    Linearly interpolates between thresholds for smooth transitions.
    """
    if observed_pa < EARLY_SEASON_PA_THRESHOLD:
        # Interpolate from (1.0, 0.0) at 0 PA to EARLY blend at threshold
        t = observed_pa / EARLY_SEASON_PA_THRESHOLD
        proj_w = 1.0 - t * (1.0 - EARLY_SEASON_BLEND[0])
        obs_w = t * EARLY_SEASON_BLEND[1]
        return (proj_w, obs_w)
    elif observed_pa < MID_SEASON_PA_THRESHOLD:
        # Interpolate from EARLY blend to MID blend
        t = (observed_pa - EARLY_SEASON_PA_THRESHOLD) / (MID_SEASON_PA_THRESHOLD - EARLY_SEASON_PA_THRESHOLD)
        proj_w = EARLY_SEASON_BLEND[0] + t * (MID_SEASON_BLEND[0] - EARLY_SEASON_BLEND[0])
        obs_w = EARLY_SEASON_BLEND[1] + t * (MID_SEASON_BLEND[1] - EARLY_SEASON_BLEND[1])
        return (proj_w, obs_w)
    else:
        # Interpolate from MID blend toward LATE blend
        # Cap at 400 PA to avoid extreme weighting
        extra = min(observed_pa - MID_SEASON_PA_THRESHOLD, 200)
        t = extra / 200.0
        proj_w = MID_SEASON_BLEND[0] + t * (LATE_SEASON_BLEND[0] - MID_SEASON_BLEND[0])
        obs_w = MID_SEASON_BLEND[1] + t * (LATE_SEASON_BLEND[1] - MID_SEASON_BLEND[1])
        return (proj_w, obs_w)


def predict_remaining_pa(
    projected_pa: float,
    recent_pa_per_game: float,
    games_remaining: int,
    health_score: float = 0.85,
    depth_chart_starter: bool = True,
    is_platoon: bool = False,
    observed_pa: float = 0.0,
) -> float:
    """Predict remaining plate appearances for the rest of season.

    Uses a Bayesian blend of preseason projection and observed PA rate:
    - Early season (< 50 PA observed): weight projection ~80%, observed ~20%
    - Mid season (50-200 PA): weight ~50/50
    - Late season (> 200 PA): weight projection ~20%, observed ~80%

    Args:
        projected_pa: Full-season preseason projection (e.g., 600).
        recent_pa_per_game: Observed PA per team game so far this season.
        games_remaining: Number of team games left in the season.
        health_score: 0-1 multiplier from injury model (1.0 = fully healthy).
        depth_chart_starter: True if everyday starter, False if bench player.
        is_platoon: True if platoon player (only plays vs opposite hand).
        observed_pa: PA accumulated so far (used to determine blend weights).

    Returns:
        Predicted remaining PA (float, >= 0).
    """
    if games_remaining <= 0:
        return 0.0

    # Projected remaining PA rate (from preseason projection)
    # Estimate games already played from observed PA and rate
    if recent_pa_per_game > 0 and observed_pa > 0:
        games_played_est = observed_pa / recent_pa_per_game
        total_games = games_played_est + games_remaining
        if total_games > 0:
            proj_remaining = projected_pa * (games_remaining / total_games)
        else:
            proj_remaining = 0.0
    else:
        # No observed data yet: use projection prorated to remaining games
        proj_remaining = projected_pa * (games_remaining / FULL_SEASON_GAMES)

    # Observed rate projection
    obs_remaining = recent_pa_per_game * games_remaining

    # Blend
    proj_w, obs_w = _compute_blend_weights(observed_pa)
    total_w = proj_w + obs_w
    if total_w > 0:
        blended = (proj_w * proj_remaining + obs_w * obs_remaining) / total_w
    else:
        blended = proj_remaining

    # Apply adjustments
    if not depth_chart_starter:
        blended *= BENCH_PLAYING_TIME_FACTOR

    if is_platoon:
        blended *= 1.0 - PLATOON_REDUCTION

    blended *= max(0.0, min(1.0, health_score))

    return max(0.0, blended)


def predict_remaining_ip(
    projected_ip: float,
    recent_ip_per_start: float,
    starts_remaining: int,
    health_score: float = 0.85,
    is_starter: bool = True,
    observed_ip: float = 0.0,
) -> float:
    """Predict remaining innings pitched for the rest of season.

    For starters: IP = starts_remaining * blended IP/start rate.
    For relievers: IP = weeks_remaining * appearances_per_week * IP/appearance.

    Args:
        projected_ip: Full-season preseason projection (e.g., 180.0).
        recent_ip_per_start: Observed IP per start (starters) or IP per
            appearance (relievers) so far this season.
        starts_remaining: Estimated remaining starts (starters) or
            remaining appearances (relievers).
        health_score: 0-1 multiplier from injury model.
        is_starter: True for starting pitchers, False for relievers.
        observed_ip: IP accumulated so far (for blend weight calculation).

    Returns:
        Predicted remaining IP (float, >= 0).
    """
    if starts_remaining <= 0:
        return 0.0

    # Convert observed IP to equivalent "PA" scale for blend weights
    # Rough heuristic: 1 IP ~ 4 batters faced ~ 4 PA equivalent
    equivalent_pa = observed_ip * 4.0
    proj_w, obs_w = _compute_blend_weights(equivalent_pa)

    if is_starter:
        # Projected IP per start from preseason
        # Assume ~32 starts for a full season starter
        proj_ip_per_start = projected_ip / 32.0 if projected_ip > 0 else 5.5
        obs_ip_per_start = recent_ip_per_start if recent_ip_per_start > 0 else proj_ip_per_start

        total_w = proj_w + obs_w
        if total_w > 0:
            blended_per_start = (proj_w * proj_ip_per_start + obs_w * obs_ip_per_start) / total_w
        else:
            blended_per_start = proj_ip_per_start

        result = blended_per_start * starts_remaining
    else:
        # Reliever: use IP per appearance
        proj_ip_per_app = projected_ip / 65.0 if projected_ip > 0 else RELIEVER_IP_PER_APPEARANCE
        obs_ip_per_app = recent_ip_per_start if recent_ip_per_start > 0 else proj_ip_per_app

        total_w = proj_w + obs_w
        if total_w > 0:
            blended_per_app = (proj_w * proj_ip_per_app + obs_w * obs_ip_per_app) / total_w
        else:
            blended_per_app = proj_ip_per_app

        result = blended_per_app * starts_remaining

    result *= max(0.0, min(1.0, health_score))

    return max(0.0, result)


def predict_remaining_pa_batch(
    pool: pd.DataFrame,
    weeks_remaining: int = 16,
    games_per_week: float = 6.5,
) -> pd.DataFrame:
    """Add 'predicted_remaining_pa' and 'predicted_remaining_ip' columns.

    For hitters: uses projected PA, observed PA rate, health score.
    For pitchers: uses projected IP, observed IP rate, health score.

    Args:
        pool: Player pool DataFrame. Expected columns:
            - pa (projected full-season PA)
            - ip (projected full-season IP)
            - is_hitter (bool)
            - positions (str)
            - health_score (float, optional)
            - ytd_pa (float, optional) - PA accumulated this season
            - ytd_ip (float, optional) - IP accumulated this season
            - games_played (int, optional) - team games played
        weeks_remaining: Weeks left in the fantasy season.
        games_per_week: Average MLB games per team per week.

    Returns:
        Copy of pool with 'predicted_remaining_pa' and
        'predicted_remaining_ip' columns added.
    """
    result = pool.copy()
    games_remaining = int(weeks_remaining * games_per_week)

    remaining_pa = []
    remaining_ip = []

    for _, row in result.iterrows():
        health = float(row.get("health_score", 0.85) or 0.85)
        is_hitter = bool(row.get("is_hitter", True))
        positions = str(row.get("positions", ""))

        # Determine depth chart status
        # Utility/DH-only players with low PA are likely bench
        is_starter = True  # default assumption

        # Platoon detection: crude heuristic based on positions
        is_platoon = False

        if is_hitter:
            projected_pa = float(row.get("pa", 0) or 0)
            ytd_pa = float(row.get("ytd_pa", 0) or 0)
            gp = float(row.get("games_played", 0) or 0)

            # Compute observed rate
            if gp > 0:
                recent_rate = ytd_pa / gp
            else:
                recent_rate = 0.0

            # Bench detection: projected PA < 300 suggests part-time
            if projected_pa < 300 and projected_pa > 0:
                is_starter = False

            pred = predict_remaining_pa(
                projected_pa=projected_pa,
                recent_pa_per_game=recent_rate,
                games_remaining=games_remaining,
                health_score=health,
                depth_chart_starter=is_starter,
                is_platoon=is_platoon,
                observed_pa=ytd_pa,
            )
            remaining_pa.append(pred)
            remaining_ip.append(0.0)
        else:
            projected_ip = float(row.get("ip", 0) or 0)
            ytd_ip = float(row.get("ytd_ip", 0) or 0)

            # Starter vs reliever
            is_sp = "SP" in positions
            if is_sp:
                # Estimate starts remaining
                starts_per_week = 1.0  # once per 5 days ~ 1 per week
                starts_remaining = int(weeks_remaining * starts_per_week)
                # Observed IP per start
                # Rough: assume 1 start per week so far
                weeks_elapsed = max(1, 26 - weeks_remaining)
                starts_so_far = max(1, weeks_elapsed) if ytd_ip > 0 else 0
                ip_per_start = ytd_ip / starts_so_far if starts_so_far > 0 else 0.0
            else:
                # Reliever
                starts_remaining = int(weeks_remaining * RELIEVER_APPEARANCES_PER_WEEK)
                weeks_elapsed = max(1, 26 - weeks_remaining)
                apps_so_far = int(weeks_elapsed * RELIEVER_APPEARANCES_PER_WEEK) if ytd_ip > 0 else 0
                ip_per_start = ytd_ip / apps_so_far if apps_so_far > 0 else 0.0

            pred = predict_remaining_ip(
                projected_ip=projected_ip,
                recent_ip_per_start=ip_per_start,
                starts_remaining=starts_remaining,
                health_score=health,
                is_starter=is_sp,
                observed_ip=ytd_ip,
            )
            remaining_pa.append(0.0)
            remaining_ip.append(pred)

    result["predicted_remaining_pa"] = remaining_pa
    result["predicted_remaining_ip"] = remaining_ip

    return result

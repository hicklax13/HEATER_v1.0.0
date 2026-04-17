"""IP Tracker — Weekly innings pitched monitoring.

Tracks two thresholds:
- 20 IP minimum: forfeit threshold (Yahoo H2H rule). Below this you
  forfeit all pitching categories for the week.
- ~54 IP weekly target: derived from 1,400 IP season target spread
  across 26 weeks. This is the "competitive" pace, not the minimum.
"""

import logging
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

# Forfeit threshold (Yahoo H2H rule)
MIN_IP = 20.0
# Competitive weekly target = season_target / weeks
SEASON_IP_TARGET = 1400.0
SEASON_WEEKS = 26.0
WEEKLY_TARGET = SEASON_IP_TARGET / SEASON_WEEKS  # ~53.85


def compute_weekly_ip_projection(roster_pitchers: list[dict], days_remaining: int = 7) -> dict:
    """Project total IP for the current week based on rostered pitchers.

    Args:
        roster_pitchers: List of dicts with keys: name, positions, ip (projected season IP),
            is_starter (bool), games_this_week (int, 0 if unknown).
        days_remaining: Days left in the fantasy week (7 = Monday, 1 = Sunday).

    Returns:
        Dict with:
            projected_ip: weekly IP projection from current rostered pitchers
            ip_min: forfeit threshold (20.0)
            ip_target: competitive weekly target (~54.0)
            ip_needed: kept for backward compat = ip_min
            ip_pace: % of competitive target reached
            ip_pace_vs_min: % of forfeit threshold reached
            status: 'safe' / 'warning' / 'danger'
            message: human-readable status
            streaming_needed: bool
    """
    total_projected = 0.0
    for p in roster_pitchers:
        ip_season = float(p.get("ip", 0) or 0)
        if ip_season <= 0:
            continue

        # Skip IL stash players (0 actual IP, not pitching)
        status = str(p.get("status", "active")).strip().lower()
        if status in ("il10", "il15", "il60", "il", "na", "dl"):
            continue

        is_starter = bool(p.get("is_starter", False))
        positions = str(p.get("positions", "")).upper()
        if not is_starter:
            is_starter = "SP" in positions

        if is_starter:
            # SP: ~6 IP per start, starts every 5 days
            ip_per_start = min(ip_season / 30.0, 7.0)  # Cap at 7 IP per start
            expected_starts = max(1.0, days_remaining / 5.0)
            projected_contribution = ip_per_start * expected_starts
        else:
            # RP: ~1 IP per appearance, appears ~50-60% of game days
            ip_per_app = min(ip_season / 60.0, 2.0)  # Cap at 2 IP per appearance
            expected_appearances = days_remaining * 0.55
            projected_contribution = ip_per_app * expected_appearances

        total_projected += projected_contribution

    # Status thresholds use the forfeit minimum (20 IP) — that's the
    # red line that loses categories. The weekly target is the goal,
    # but missing it isn't catastrophic.
    status = "safe"
    streaming_needed = False
    pct_of_target = total_projected / WEEKLY_TARGET * 100.0 if WEEKLY_TARGET > 0 else 0
    pct_of_min = total_projected / MIN_IP * 100.0 if MIN_IP > 0 else 0

    # Single-source display value. Callers formatting projected_ip with :.1f
    # would otherwise double-round (round(x, 2) then :.1f) and disagree with
    # the narrative message at boundary values (e.g., 39.55 → header 39.5,
    # message 39.6). Use ip_display_str everywhere the UI needs a 1-dp number.
    _ip_disp = f"{total_projected:.1f}"

    if total_projected < MIN_IP * 0.5:
        status = "danger"
        message = f"DANGER: Only {_ip_disp} IP projected. Need {MIN_IP:.0f} minimum. Stream SP immediately."
        streaming_needed = True
    elif total_projected < MIN_IP:
        status = "warning"
        message = f"WARNING: {_ip_disp} IP projected, need {MIN_IP:.0f} minimum to avoid forfeit."
        streaming_needed = True
    elif total_projected < WEEKLY_TARGET * 0.85:
        status = "warning"
        message = (
            f"{_ip_disp} IP projected — above {MIN_IP:.0f} minimum but below "
            f"{WEEKLY_TARGET:.0f} weekly target ({pct_of_target:.0f}%). Consider streaming."
        )
    elif total_projected < WEEKLY_TARGET:
        status = "safe"
        message = f"{_ip_disp} IP projected ({pct_of_target:.0f}% of {WEEKLY_TARGET:.0f} weekly target)."
    else:
        status = "safe"
        message = f"{_ip_disp} IP projected ({pct_of_target:.0f}% of {WEEKLY_TARGET:.0f} weekly target)."

    return {
        "projected_ip": round(total_projected, 2),
        "projected_ip_display": _ip_disp,
        "ip_min": MIN_IP,
        "ip_target": round(WEEKLY_TARGET, 1),
        "ip_needed": MIN_IP,  # backward compat for callers
        "ip_pace": round(pct_of_target, 0),  # % of competitive target
        "ip_pace_vs_min": round(pct_of_min, 0),
        "status": status,
        "message": message,
        "streaming_needed": streaming_needed,
        "days_remaining": days_remaining,
    }


def get_days_remaining_in_week() -> int:
    """Compute days remaining in the current fantasy week (Mon-Sun)."""
    now = datetime.now(UTC)
    # Fantasy weeks run Monday (0) through Sunday (6)
    day_of_week = now.weekday()  # 0=Mon, 6=Sun
    return 7 - day_of_week

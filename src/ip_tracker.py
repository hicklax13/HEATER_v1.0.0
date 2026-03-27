"""IP Tracker — Weekly innings pitched monitoring for 20 IP minimum.

AVIS Rule #1: Never forfeit pitching categories. Hitting 20 IP is non-negotiable.
"""

import logging
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


def compute_weekly_ip_projection(roster_pitchers: list[dict], days_remaining: int = 7) -> dict:
    """Project total IP for the current week based on rostered pitchers.

    Args:
        roster_pitchers: List of dicts with keys: name, positions, ip (projected season IP),
            is_starter (bool), games_this_week (int, 0 if unknown).
        days_remaining: Days left in the fantasy week (7 = Monday, 1 = Sunday).

    Returns:
        Dict with: projected_ip, ip_needed (20.0), ip_pace, status ('safe'/'warning'/'danger'),
        message, streaming_needed (bool).
    """
    MIN_IP = 20.0

    total_projected = 0.0
    for p in roster_pitchers:
        ip_season = float(p.get("ip", 0) or 0)
        if ip_season <= 0:
            continue

        # Estimate weekly IP from season projection
        # ~26 weeks in MLB season, so weekly IP ≈ season IP / 26
        weekly_ip = ip_season / 26.0

        # Adjust by days remaining in the week
        daily_ip = weekly_ip / 7.0
        projected_contribution = daily_ip * days_remaining
        total_projected += projected_contribution

    status = "safe"
    message = f"On pace for {total_projected:.2f} IP this week."
    streaming_needed = False

    if total_projected < MIN_IP * 0.5:
        status = "danger"
        message = f"DANGER: Only {total_projected:.2f} IP projected. Need {MIN_IP:.0f}. Stream SP immediately."
        streaming_needed = True
    elif total_projected < MIN_IP:
        status = "warning"
        message = f"WARNING: {total_projected:.2f} IP projected, need {MIN_IP:.0f}. Consider streaming a SP."
        streaming_needed = True
    elif total_projected < MIN_IP * 1.2:
        status = "safe"
        message = f"Projected {total_projected:.2f} IP (just above minimum). Monitor closely."

    return {
        "projected_ip": round(total_projected, 2),
        "ip_needed": MIN_IP,
        "ip_pace": round(total_projected / MIN_IP * 100, 0) if MIN_IP > 0 else 100,
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

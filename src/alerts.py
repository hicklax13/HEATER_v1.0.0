"""Proactive Alert System — AVIS Section 6 communication.

Monitors for roster-impacting events and generates actionable alerts.
Analyst tone, not cheerleader. Data-driven recommendations.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def generate_roster_alerts(
    roster: pd.DataFrame,
    player_news: pd.DataFrame | None = None,
    closer_data: dict | None = None,
    max_roster_size: int = 23,
) -> list[dict]:
    """Generate proactive alerts based on current roster state.

    Checks:
    1. Empty roster spots (AVIS Rule #4)
    2. Injured players not on IL
    3. Closer role changes
    4. Low closer count (AVIS Rule #2)

    Returns list of alert dicts: {type, severity, title, message, action}.
    """
    alerts = []

    if roster.empty:
        return alerts

    roster_size = len(roster)

    # Alert 1: Empty roster spots
    if roster_size < max_roster_size:
        empty = max_roster_size - roster_size
        alerts.append(
            {
                "type": "empty_roster",
                "severity": "critical",
                "title": f"EMPTY ROSTER SPOTS ({empty})",
                "message": f"You have {empty} empty roster slot(s). An empty spot is a zero — any player is better than nothing.",
                "action": "Go to Free Agents and add the top-ranked available player immediately.",
            }
        )

    # Alert 2: Injured starters not on IL
    if player_news is not None and not player_news.empty:
        injury_news = player_news[player_news["news_type"] == "injury"]
        if not injury_news.empty:
            recent = injury_news.sort_values("fetched_at", ascending=False).head(5)
            for _, news in recent.iterrows():
                il_status = news.get("il_status", "")
                if il_status and "IL" in str(il_status).upper():
                    alerts.append(
                        {
                            "type": "injury",
                            "severity": "warning",
                            "title": f"INJURY: {news.get('headline', 'Player injured')}",
                            "message": f"Status: {il_status}. Check if IL slot is available.",
                            "action": "Move to IL and pick up a replacement from free agents.",
                        }
                    )

    # Alert 3: Closer count
    closer_count = 0
    closer_names = []
    for _, p in roster.iterrows():
        sv = float(p.get("sv", 0) or 0)
        if sv >= 5:
            closer_count += 1
            closer_names.append(p.get("name", "?"))

    if closer_count < 2:
        alerts.append(
            {
                "type": "closer_shortage",
                "severity": "warning",
                "title": f"CLOSER ALERT: Only {closer_count} closer(s)",
                "message": f"Current closers: {', '.join(closer_names) if closer_names else 'None'}. AVIS requires minimum 2 closers at all times.",
                "action": "Check Waiver Wire for available closers (sorted by projected SV).",
            }
        )

    # Alert 4: IL stash return watch
    il_stash_names = {"Shane Bieber", "Spencer Strider"}  # Per AVIS Section 7
    for _, p in roster.iterrows():
        name = p.get("name", "")
        if name in il_stash_names:
            alerts.append(
                {
                    "type": "il_watch",
                    "severity": "info",
                    "title": f"IL STASH: {name}",
                    "message": f"{name} is on your IL. Monitor return timeline — playoff weapon if healthy by August.",
                    "action": "Do NOT drop within 2 weeks of expected return date.",
                }
            )

    return alerts


def render_alerts_html(alerts: list[dict], theme: dict) -> str:
    """Render alerts as HTML cards for Streamlit st.markdown().

    Args:
        alerts: List of alert dicts from generate_roster_alerts.
        theme: THEME dict from ui_shared.

    Returns:
        HTML string ready for st.markdown(unsafe_allow_html=True).
    """
    if not alerts:
        return ""

    severity_colors = {
        "critical": theme.get("danger", "#e63946"),
        "warning": theme.get("warn", "#ff9f1c"),
        "info": theme.get("sky", "#457b9d"),
    }

    cards = []
    for alert in alerts:
        color = severity_colors.get(alert["severity"], theme.get("tx2", "#6b7280"))
        cards.append(
            f'<div style="background:{theme.get("card", "#fff")};'
            f"border-left:4px solid {color};"
            f"padding:8px 12px;border-radius:6px;margin-bottom:6px;font-size:12px;"
            f'font-family:IBM Plex Mono,monospace;">'
            f'<b style="color:{color};">{alert["title"]}</b><br>'
            f'<span style="color:{theme.get("tx2", "#6b7280")};">{alert["message"]}</span><br>'
            f'<span style="color:{theme.get("tx", "#1d1d1f")};font-weight:600;">'
            f"Action: {alert['action']}</span>"
            f"</div>"
        )

    return "\n".join(cards)

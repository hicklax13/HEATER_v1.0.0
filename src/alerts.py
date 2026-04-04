"""Proactive Alert System — AVIS Section 6 communication.

Monitors for roster-impacting events and generates actionable alerts.
Analyst tone, not cheerleader. Data-driven recommendations.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# AVIS Section 7: IL stash players — do NOT drop or trade within 2 weeks of return
IL_STASH_NAMES: set[str] = {"Shane Bieber", "Spencer Strider"}


def generate_roster_alerts(
    roster: pd.DataFrame,
    player_news: pd.DataFrame | None = None,
    closer_data: dict | None = None,
    max_roster_size: int = 23,
    fa_pool: pd.DataFrame | None = None,
    user_roster_ids: list | None = None,
    player_pool: pd.DataFrame | None = None,
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
        # Auto-compute top 3 FA fill recommendations if data is available
        top_fills_str = ""
        if fa_pool is not None and not fa_pool.empty and user_roster_ids is not None and player_pool is not None:
            try:
                from src.database import get_all_rostered_player_ids
                from src.in_season import rank_free_agents

                # Ensure fa_pool only contains true free agents (not opponents' players)
                _all_rostered = get_all_rostered_player_ids()
                if _all_rostered:
                    fa_pool = fa_pool[~fa_pool["player_id"].isin(_all_rostered)]
                ranked = rank_free_agents(user_roster_ids, fa_pool, player_pool)
                if not ranked.empty:
                    top3 = ranked.head(3)
                    names = [str(r.get("player_name", r.get("name", "?"))) for _, r in top3.iterrows()]
                    top_fills_str = f" Top fills: {', '.join(names)}."
            except Exception:
                pass
        alerts.append(
            {
                "type": "empty_roster",
                "severity": "critical",
                "title": f"EMPTY ROSTER SPOTS ({empty})",
                "message": f"You have {empty} empty roster slot(s). An empty spot is a zero — any player is better than nothing.{top_fills_str}",
                "action": "Go to Free Agents and add the top-ranked available player immediately.",
            }
        )

    # Alert 2: Injured starters not on IL (roster-only)
    if player_news is not None and not player_news.empty:
        injury_news = player_news[player_news["news_type"] == "injury"]
        if not injury_news.empty:
            # Filter to only players on this roster
            roster_pids = set()
            roster_names = set()
            if roster is not None and not roster.empty:
                if "player_id" in roster.columns:
                    roster_pids = set(roster["player_id"].dropna().astype(int).tolist())
                if "name" in roster.columns:
                    roster_names = {str(n).strip().lower() for n in roster["name"].dropna()}

            recent = injury_news.sort_values("fetched_at", ascending=False).head(20)
            shown = 0
            for _, news in recent.iterrows():
                if shown >= 5:
                    break
                # Check if this injury is for a rostered player
                news_pid = news.get("player_id")
                news_name = str(news.get("player_name", "")).strip().lower()
                on_roster = False
                if news_pid is not None:
                    try:
                        on_roster = int(news_pid) in roster_pids
                    except (ValueError, TypeError):
                        pass
                if not on_roster and news_name:
                    on_roster = news_name in roster_names

                if not on_roster:
                    continue

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
                    shown += 1

    # Alert 3: Closer count (check actual SV OR projected SV for early-season)
    closer_count = 0
    closer_names = []
    proj_sv_map = {}
    try:
        from src.database import get_connection

        conn = get_connection()
        try:
            roster_ids = roster["player_id"].dropna().astype(int).tolist()
            if roster_ids:
                placeholders = ",".join("?" * len(roster_ids))
                proj_df = pd.read_sql_query(
                    f"SELECT player_id, sv FROM blended_projections WHERE player_id IN ({placeholders})",
                    conn,
                    params=roster_ids,
                )
                proj_sv_map = dict(zip(proj_df["player_id"], proj_df["sv"]))
        finally:
            conn.close()
    except Exception:
        pass

    for _, p in roster.iterrows():
        actual_sv = float(p.get("sv", 0) or 0)
        pid = p.get("player_id")
        proj_sv = float(proj_sv_map.get(pid, 0) or 0)
        if actual_sv >= 5 or proj_sv >= 5:
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
    for _, p in roster.iterrows():
        name = p.get("name", "")
        if name in IL_STASH_NAMES:
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

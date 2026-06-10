"""Role-aware navigation for st.navigation(). Used only when MULTI_USER is on."""

from __future__ import annotations

PAGE_REGISTRY = [
    {"key": "1_My_Team", "title": "My Team", "path": "pages/1_My_Team.py"},
    {"key": "2_Line-up_Optimizer", "title": "Lineup Optimizer", "path": "pages/2_Line-up_Optimizer.py"},
    {"key": "3_Closer_Monitor", "title": "Closer Monitor", "path": "pages/3_Closer_Monitor.py"},
    {"key": "4_Pitcher_Streaming", "title": "Pitcher Streaming", "path": "pages/4_Pitcher_Streaming.py"},
    {"key": "5_Matchup_Planner", "title": "Matchup Planner", "path": "pages/5_Matchup_Planner.py"},
    {"key": "6_League_Standings", "title": "League Standings", "path": "pages/6_League_Standings.py"},
    {"key": "10_Punt_Analyzer", "title": "Punt Analyzer", "path": "pages/10_Punt_Analyzer.py"},
    {"key": "11_Trade_Analyzer", "title": "Trade Analyzer", "path": "pages/11_Trade_Analyzer.py"},
    {"key": "12_Trade_Finder", "title": "Trade Finder", "path": "pages/12_Trade_Finder.py"},
    {"key": "14_Free_Agents", "title": "Free Agents", "path": "pages/14_Free_Agents.py"},
    {"key": "16_Player_Compare", "title": "Player Compare", "path": "pages/16_Player_Compare.py"},
    {"key": "17_Leaders", "title": "Leaders", "path": "pages/17_Leaders.py"},
    {"key": "19_Player_Databank", "title": "Player Databank", "path": "pages/19_Player_Databank.py"},
    {"key": "20_Draft_Simulator", "title": "Draft Simulator", "path": "pages/20_Draft_Simulator.py"},
]

_ADMIN_PAGES = [
    {"title": "Admin Console", "path": "pages/_admin_console.py"},
    {"title": "Usage Analytics", "path": "pages/_admin_analytics.py"},
    {"title": "Admin Controls", "path": "pages/_admin_controls.py"},
]


def filter_enabled_pages(keys: list[str], flags: dict[str, bool]) -> list[str]:
    """Pure: keep keys whose flag is truthy or absent (absence = enabled)."""
    return [k for k in keys if flags.get(k, True)]


def build_pages(user: dict, draft_page) -> dict:
    import streamlit as st

    from src.feature_flags import list_page_flags

    raw = list_page_flags()
    flags = {e["key"]: raw.get("page:" + e["key"], True) for e in PAGE_REGISTRY}
    enabled_keys = filter_enabled_pages([e["key"] for e in PAGE_REGISTRY], flags)
    by_key = {e["key"]: e for e in PAGE_REGISTRY}

    home = st.Page(draft_page, title="Draft Tool", default=True)
    season = [st.Page(by_key[k]["path"], title=by_key[k]["title"]) for k in enabled_keys]
    groups = {"Home": [home], "Season": season}

    if user and user.get("is_admin"):
        groups["Admin"] = [st.Page(p["path"], title=p["title"]) for p in _ADMIN_PAGES]

    return groups

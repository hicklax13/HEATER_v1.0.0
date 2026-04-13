"""Player Databank — Yahoo-style player list with live stats.

Full MLB player database with 5-axis filtering, 28 stat views,
custom HTML table with JavaScript sorting, and Excel export.
"""

import logging
from datetime import UTC, datetime

import streamlit as st

from src.database import init_db
from src.player_databank import (
    STAT_VIEW_OPTIONS,
    export_to_excel,
    filter_databank,
    get_data_as_of_label,
    load_databank,
    render_databank_table,
)
from src.ui_shared import T, inject_custom_css, render_page_layout

try:
    from src.yahoo_data_service import get_yahoo_data_service
except ImportError:
    get_yahoo_data_service = None

logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Heater | Player Databank",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)
init_db()
inject_custom_css()

# Page layout
render_page_layout(
    "PLAYER DATABANK",
    banner_teaser="Live stats for every MLB player",
    banner_icon="databank",
)

# Search bar
search = st.text_input(
    "Search player by name",
    placeholder="Search player by name",
    key="db_search",
    label_visibility="collapsed",
)

# Position filter — pill-style buttons via selectbox
# (Custom HTML pills are visual only; selectbox drives the logic)
POSITION_OPTIONS = [
    ("B", "All Batters"),
    ("P", "All Pitchers"),
    ("C", "C"),
    ("1B", "1B"),
    ("2B", "2B"),
    ("3B", "3B"),
    ("SS", "SS"),
    ("OF", "OF"),
    ("Util", "Util"),
    ("SP", "SP"),
    ("RP", "RP"),
]

pos_col, show_col = st.columns([5, 1])
with pos_col:
    position = st.selectbox(
        "Position",
        options=[p[0] for p in POSITION_OPTIONS],
        format_func=lambda x: dict(POSITION_OPTIONS).get(x, x),
        index=0,
        key="db_position",
    )
with show_col:
    show_my_team = st.checkbox("Show my team", key="db_show_my_team")

# Filter row — 4 dropdowns + export column
STATUS_OPTIONS = [
    ("ALL", "All Players"),
    ("A", "All Available Players"),
    ("FA", "Free Agents Only"),
    ("W", "Waivers Only"),
    ("T", "All Taken Players"),
]

MLB_TEAMS = [
    "ALL",
    "ARI",
    "OAK",
    "ATL",
    "BAL",
    "BOS",
    "CHC",
    "CHW",
    "CIN",
    "CLE",
    "COL",
    "DET",
    "HOU",
    "KC",
    "LAA",
    "LAD",
    "MIA",
    "MIL",
    "MIN",
    "NYM",
    "NYY",
    "PHI",
    "PIT",
    "SD",
    "SF",
    "SEA",
    "STL",
    "TB",
    "TEX",
    "TOR",
    "WSH",
]

# Get fantasy teams from Yahoo
fantasy_team_options = [("NONE", "No Team Selected")]
try:
    if get_yahoo_data_service is not None:
        yds = get_yahoo_data_service()
        if yds.is_connected():
            rosters = yds.get_rosters()
            if rosters is not None and not rosters.empty:
                team_col = "team_name" if "team_name" in rosters.columns else None
                if team_col:
                    for tn in sorted(rosters[team_col].dropna().unique()):
                        fantasy_team_options.append((str(tn), str(tn)))
except Exception:
    pass

fcol1, fcol2, fcol3, fcol4, fcol5 = st.columns([2, 2, 2, 3, 1.5])

with fcol1:
    status = st.selectbox(
        "Status",
        options=[s[0] for s in STATUS_OPTIONS],
        format_func=lambda x: dict(STATUS_OPTIONS).get(x, x),
        index=1,
        key="db_status",
    )

with fcol2:
    mlb_team = st.selectbox("MLB Teams", MLB_TEAMS, index=0, key="db_mlb_team")

with fcol3:
    fantasy_team = st.selectbox(
        "Fantasy Teams",
        options=[f[0] for f in fantasy_team_options],
        format_func=lambda x: dict(fantasy_team_options).get(x, x),
        index=0,
        key="db_fantasy_team",
    )

with fcol4:
    stat_view_keys = list(STAT_VIEW_OPTIONS.keys())
    stat_view = st.selectbox(
        "Stats",
        options=stat_view_keys,
        format_func=lambda x: STAT_VIEW_OPTIONS.get(x, x),
        index=stat_view_keys.index("S_S_2026"),
        key="db_stat_view",
    )

# Load and filter data
is_pitcher = position in ("P", "SP", "RP")

with st.spinner("Loading player data..."):
    df = load_databank(stat_view)

if df.empty:
    st.warning("No player data available. Run the data bootstrap first.")
    st.stop()

filtered = filter_databank(
    df,
    position=position,
    status=status,
    mlb_team=mlb_team,
    fantasy_team=fantasy_team,
    search=search,
    show_my_team=show_my_team,
)

# Export button
with fcol5:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    if not filtered.empty:
        today_str = datetime.now(UTC).strftime("%Y-%m-%d")
        view_label = STAT_VIEW_OPTIONS.get(stat_view, "stats")
        safe_label = view_label.replace(" ", "_").replace("(", "").replace(")", "")
        filename = f"HEATER_Player_Databank_{safe_label}_{today_str}.xlsx"
        excel_bytes = export_to_excel(filtered, view_label)
        st.download_button(
            label="Export",
            data=excel_bytes,
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="db_export",
        )

# Player count + data freshness label
as_of_label = get_data_as_of_label(stat_view)
freshness_html = f' &middot; <span style="color:{T["amber"]}">{as_of_label}</span>' if as_of_label else ""
st.markdown(
    f'<div style="color:{T["tx2"]};font-size:13px;margin:4px 0 8px 0;">'
    f"Showing {len(filtered):,} players{freshness_html}</div>",
    unsafe_allow_html=True,
)

# Render table
table_html = render_databank_table(
    filtered,
    stat_view=stat_view,
    is_pitcher=is_pitcher,
)
st.markdown(table_html, unsafe_allow_html=True)

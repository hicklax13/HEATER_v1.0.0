"""Player Databank — Yahoo-style player list with live stats.

Full MLB player database with 5-axis filtering, 28 stat views,
custom HTML table with JavaScript sorting, and Excel export.
Search button defers execution; pagination shows 25 players per page.
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

PAGE_SIZE = 25

# ── Page config ──────────────────────────────────────────────────────────────

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

# ── Filter constants ─────────────────────────────────────────────────────────

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

SORT_OPTIONS = [
    "Pre-Season",
    "Current",
    "GP",
    "% Ros",
    "Adds",
    "Drops",
    "R",
    "HR",
    "RBI",
    "SB",
    "AVG",
    "OBP",
    "IP",
    "W",
    "L",
    "SV",
    "K",
    "ERA",
    "WHIP",
]

SORT_COL_MAP = {
    "Pre-Season": "adp",
    "Current": "consensus_rank",
    "GP": "ytd_gp",
    "% Ros": "percent_owned",
    "Adds": "adds",
    "Drops": "drops",
    "R": "r",
    "HR": "hr",
    "RBI": "rbi",
    "SB": "sb",
    "AVG": "avg",
    "OBP": "obp",
    "IP": "ip",
    "W": "w",
    "L": "l",
    "SV": "sv",
    "K": "k",
    "ERA": "era",
    "WHIP": "whip",
}

# Get fantasy teams from Yahoo (before form so options are ready)
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

# ── Filter form (deferred execution — table renders only on Search) ──────────

with st.form("databank_filters", border=False):
    # Row 1: Search bar
    search = st.text_input(
        "Search player by name",
        placeholder="Search player by name",
        key="db_search",
        label_visibility="collapsed",
    )

    # Row 2: Position + Show My Team
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

    # Row 3: Status, MLB Team, Fantasy Team, Stats, Sort, Order
    fcol1, fcol2, fcol3, fcol4, fcol5, fcol6 = st.columns([2, 2, 2, 3, 2, 1.5])

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

    with fcol5:
        sort_col = st.selectbox(
            "Sort by",
            options=SORT_OPTIONS,
            index=0,
            key="db_sort_col",
        )

    with fcol6:
        sort_dir = st.selectbox(
            "Order",
            options=["Ascending", "Descending"],
            index=0,
            key="db_sort_dir",
        )

    # Row 4: Search button
    search_submitted = st.form_submit_button("Search", type="primary", use_container_width=True)

# ── Search trigger guard ─────────────────────────────────────────────────────

if search_submitted:
    st.session_state["db_page"] = 0  # Reset pagination on new search
    st.session_state["db_search_triggered"] = True

if not st.session_state.get("db_search_triggered", False):
    st.info("Select your filters above, then click **Search** to find players.")
    st.stop()

# ── Load and filter data ─────────────────────────────────────────────────────

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

if filtered.empty:
    st.info("No players match the selected filters.")
    st.stop()

# ── Server-side sort ─────────────────────────────────────────────────────────

sort_column = SORT_COL_MAP.get(sort_col, "adp")
ascending = sort_dir == "Ascending"

# Resolve _calc aliases for rate stats
if sort_column + "_calc" in filtered.columns:
    sort_column = sort_column + "_calc"
# Fall back to ytd_ prefix if base column missing
if sort_column not in filtered.columns and f"ytd_{sort_column}" in filtered.columns:
    sort_column = f"ytd_{sort_column}"

if sort_column in filtered.columns:
    filtered = filtered.sort_values(
        by=sort_column,
        ascending=ascending,
        na_position="last",
    )

# ── Pagination ───────────────────────────────────────────────────────────────

if "db_page" not in st.session_state:
    st.session_state["db_page"] = 0

total_players = len(filtered)
total_pages = max(1, -(-total_players // PAGE_SIZE))  # ceiling division
page = min(st.session_state.get("db_page", 0), total_pages - 1)
st.session_state["db_page"] = page

start_idx = page * PAGE_SIZE
end_idx = min(start_idx + PAGE_SIZE, total_players)

page_df = filtered.iloc[start_idx:end_idx]

# ── Player count + data freshness label ──────────────────────────────────────

as_of_label = get_data_as_of_label(stat_view)
freshness_html = f' &middot; <span style="color:{T["amber"]}">{as_of_label}</span>' if as_of_label else ""
st.markdown(
    f'<div style="color:{T["tx2"]};font-size:13px;margin:4px 0 8px 0;">'
    f"Showing {start_idx + 1}\u2013{end_idx} of {total_players:,} players{freshness_html}</div>",
    unsafe_allow_html=True,
)

# ── Render table ─────────────────────────────────────────────────────────────

table_html = render_databank_table(
    page_df,
    stat_view=stat_view,
    is_pitcher=is_pitcher,
)
st.markdown(table_html, unsafe_allow_html=True)

# ── Pagination controls + Export ─────────────────────────────────────────────

pag_c1, pag_c2, pag_c3, pag_c4 = st.columns([1.5, 2, 1.5, 1.5])

with pag_c1:
    if page > 0:
        if st.button("Previous 25", key="db_prev"):
            st.session_state["db_page"] = page - 1
            st.rerun()

with pag_c2:
    st.markdown(
        f'<div style="text-align:center;color:{T["tx2"]};font-size:13px;padding-top:8px;">'
        f"Page {page + 1} of {total_pages}</div>",
        unsafe_allow_html=True,
    )

with pag_c3:
    if page < total_pages - 1:
        if st.button("Next 25", key="db_next"):
            st.session_state["db_page"] = page + 1
            st.rerun()

with pag_c4:
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

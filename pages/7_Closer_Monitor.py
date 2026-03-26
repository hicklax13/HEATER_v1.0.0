"""Closer Monitor — Track closer depth charts and job security across all 30 teams."""

from __future__ import annotations

import logging

import streamlit as st

from src.closer_monitor import build_closer_grid
from src.database import init_db, load_player_pool
from src.ui_shared import _headshot_img_html, inject_custom_css, page_timer_footer, page_timer_start, render_page_layout

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Heater | Closer Monitor", page_icon="", layout="wide", initial_sidebar_state="collapsed")

init_db()

inject_custom_css()
page_timer_start()

render_page_layout("CLOSER MONITOR", banner_teaser="30-team closer depth chart", banner_icon="closer")

st.info(
    "Closer depth charts are populated from the FanGraphs depth chart data loaded at app launch. "
    "Connect your Yahoo league or run the app bootstrap to refresh depth chart data."
)

# Load player pool for stat lookups
pool = load_player_pool()

# Attempt to load depth chart data from session state (populated by bootstrap)
depth_data: dict = st.session_state.get("closer_depth_data", {})

if not depth_data:
    # Try to build a minimal depth chart from pitcher pool using sv projections
    if not pool.empty:
        pitchers = pool[pool["is_hitter"] == 0].copy()
        if not pitchers.empty and "sv" in pitchers.columns:
            import pandas as pd

            pitchers["sv"] = pd.to_numeric(pitchers["sv"], errors="coerce").fillna(0)
            # Group by team, pick top SV pitcher as closer
            top_sv = pitchers[pitchers["sv"] > 0].sort_values("sv", ascending=False).drop_duplicates("team")
            if not top_sv.empty:
                for _, row in top_sv.iterrows():
                    team = str(row.get("team", ""))
                    if not team:
                        continue
                    sv = float(row.get("sv", 0) or 0)
                    # Confidence heuristic: top SV earners get higher confidence
                    confidence = min(1.0, sv / 35.0)
                    depth_data[team] = {
                        "closer": str(row.get("name", "Unknown")),
                        "setup": [],
                        "closer_confidence": confidence,
                    }

if not depth_data:
    st.warning(
        "No depth chart data available. Launch the app from the main page to bootstrap "
        "data from FanGraphs, or load sample data with: python load_sample_data.py"
    )
else:
    grid = build_closer_grid(depth_data, pool if not pool.empty else None)

    if not grid:
        st.warning("No closer data to display.")
    else:
        st.caption(f"Showing {len(grid)} teams with closer data.")

        # Render in 5-column layout
        cols_per_row = 5
        for row_start in range(0, len(grid), cols_per_row):
            row_items = grid[row_start : row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col_idx, item in enumerate(row_items):
                with cols[col_idx]:
                    security = item["job_security"]
                    color = item["security_color"]
                    pct = int(security * 100)
                    era_str = f"{item['era']:.2f}" if item["era"] else "—"
                    whip_str = f"{item['whip']:.2f}" if item["whip"] else "—"
                    sv_str = f"{int(item['projected_sv'])}" if item["projected_sv"] else "—"
                    setup_str = ", ".join(item["setup_names"]) if item["setup_names"] else "—"
                    headshot = _headshot_img_html(item.get("mlb_id"), size=32)

                    st.markdown(
                        f"""
<div style="
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-left: 4px solid {color};
    border-radius: 8px;
    padding: 10px 10px;
    margin-bottom: 8px;
    font-family: sans-serif;
    min-height: 160px;
">
    <div style="font-size:0.7rem; font-weight:700; color:#888; letter-spacing:0.08em; white-space:nowrap;">
        {item["team"]}
    </div>
    <div style="font-size:0.88rem; font-weight:700; color:#1a1a2e; margin:3px 0 2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; display:flex; align-items:center; gap:5px;">
        {headshot}{item["closer_name"]}
    </div>
    <div style="
        display:inline-block;
        background:{color};
        color:#fff;
        font-size:0.65rem;
        font-weight:700;
        border-radius:4px;
        padding:1px 6px;
        margin-bottom:4px;
        white-space:nowrap;
    ">
        {pct}%
    </div>
    <div style="font-size:0.68rem; color:#555; margin-top:3px; white-space:nowrap;">
        SV: <b>{sv_str}</b>
    </div>
    <div style="font-size:0.68rem; color:#555; white-space:nowrap;">
        ERA: <b>{era_str}</b> &nbsp; WHIP: <b>{whip_str}</b>
    </div>
    <div style="font-size:0.65rem; color:#888; margin-top:3px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
        Setup: {setup_str}
    </div>
</div>
""",
                        unsafe_allow_html=True,
                    )

page_timer_footer("Closer Monitor")

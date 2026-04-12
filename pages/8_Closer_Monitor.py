"""Closer Monitor — Track closer depth charts and job security across all 30 teams."""

from __future__ import annotations

import logging

import streamlit as st

from src.closer_monitor import build_closer_grid
from src.database import get_connection, init_db, load_player_pool
from src.ui_shared import (
    _headshot_img_html,
    format_stat,
    inject_custom_css,
    page_timer_footer,
    page_timer_start,
    render_page_layout,
)

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Heater | Closer Monitor", page_icon="", layout="wide", initial_sidebar_state="collapsed")

init_db()

inject_custom_css()
page_timer_start()

render_page_layout("CLOSER MONITOR", banner_teaser="30-team closer depth chart", banner_icon="closer")


@st.cache_data(ttl=300)
def _load_actual_sv_stats():
    """Load actual 2026 save stats for relievers."""
    import pandas as pd

    conn = get_connection()
    try:
        df = pd.read_sql_query(
            """SELECT p.name, ss.sv, ss.era, ss.whip, ss.games_played
               FROM season_stats ss
               JOIN players p ON ss.player_id = p.player_id
               WHERE ss.season = 2026 AND p.is_hitter = 0 AND ss.sv > 0
               ORDER BY ss.sv DESC""",
            conn,
        )
        if df.empty:
            return {}
        return {
            row["name"]: {
                "sv": int(row["sv"]),
                "era": float(row.get("era", 0) or 0),
                "whip": float(row.get("whip", 0) or 0),
            }
            for _, row in df.iterrows()
        }
    except Exception:
        return {}
    finally:
        conn.close()


st.info(
    "Closer depth charts are populated from the FanGraphs depth chart data loaded at app launch. "
    "Connect your Yahoo league or run the app bootstrap to refresh depth chart data."
)

# Load player pool for stat lookups
pool = load_player_pool()

# Normalize duplicate team abbreviations to canonical 30-team set
_TEAM_NORMALIZE: dict[str, str] = {
    "ATH": "OAK",
    "AZ": "ARI",
    "WSN": "WSH",
    "CHW": "CWS",
    "TBR": "TB",
    "KCR": "KC",
    "SDP": "SD",
    "SFG": "SF",
}


def _normalize_team(abbr: str) -> str:
    return _TEAM_NORMALIZE.get(abbr.upper().strip(), abbr.upper().strip())


# Attempt to load depth chart data from session state (populated by bootstrap)
_raw_depth_data: dict = st.session_state.get("closer_depth_data", {})
# Normalize keys to canonical abbreviations
depth_data: dict = {}
for _k, _v in _raw_depth_data.items():
    _canon = _normalize_team(_k)
    if _canon not in depth_data:
        depth_data[_canon] = _v
    elif _v.get("closer_confidence", 0) > depth_data[_canon].get("closer_confidence", 0):
        depth_data[_canon] = _v

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
                    team = _normalize_team(str(row.get("team", "")))
                    if not team:
                        continue
                    sv = float(row.get("sv", 0) or 0)
                    # Confidence heuristic: top SV earners get higher confidence
                    confidence = min(1.0, sv / 35.0)
                    if team not in depth_data:
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
    actual_sv_stats = _load_actual_sv_stats()

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
                    closer_name = item["closer_name"]
                    actual = actual_sv_stats.get(closer_name, {})
                    actual_sv = actual.get("sv")
                    actual_era = actual.get("era")
                    actual_whip = actual.get("whip")

                    import html as _html

                    era_str = format_stat(item["era"], "ERA") if item["era"] else "—"
                    whip_str = format_stat(item["whip"], "WHIP") if item["whip"] else "—"
                    sv_str = f"{int(item['projected_sv'])}" if item["projected_sv"] else "—"
                    setup_str = _html.escape(", ".join(item["setup_names"]) if item["setup_names"] else "—")
                    closer_name_safe = _html.escape(item["closer_name"])
                    headshot = _headshot_img_html(item.get("mlb_id"), size=32)

                    # Recent form indicator from MatchupContextService
                    form_html = "<!-- no form data -->"
                    try:
                        from src.matchup_context import get_matchup_context

                        _closer_ctx = get_matchup_context()
                        _closer_mlb = item.get("mlb_id")
                        if _closer_mlb is not None:
                            _closer_form = _closer_ctx.get_player_form(int(_closer_mlb))
                            _trend = _closer_form.get("trend", "neutral")
                            if _trend == "hot":
                                form_html = (
                                    '<div style="font-size:0.6rem;color:#2e7d32;font-weight:600;">'
                                    "Recent Form: HOT</div>"
                                )
                            elif _trend == "cold":
                                form_html = (
                                    '<div style="font-size:0.6rem;color:#c62828;font-weight:600;">'
                                    "Recent Form: COLD</div>"
                                )
                    except Exception:
                        pass

                    # gmLI trust indicator (data may come from session_state in future)
                    gmli_html = "<!-- no gmli data -->"
                    _gmli_data = st.session_state.get("closer_gmli_data", {})
                    _player_gmli = _gmli_data.get(closer_name, {})
                    _gmli_val = _player_gmli.get("gmli") if _player_gmli else None
                    _gmli_prev = _player_gmli.get("gmli_prev") if _player_gmli else None
                    if _gmli_val is not None:
                        if _gmli_val >= 1.8:
                            _trust_dot = "#2d6a4f"
                            _trust_label = "High Trust"
                        elif _gmli_val >= 1.0:
                            _trust_dot = "#ff9f1c"
                            _trust_label = "Moderate"
                        else:
                            _trust_dot = "#e63946"
                            _trust_label = "Low Trust"
                        _trend_html = ""
                        if _gmli_prev is not None and (_gmli_prev - _gmli_val) > 0.5:
                            _trend_html = ' <span style="color:#e63946;font-weight:700;">&#8595; Declining</span>'
                        gmli_html = (
                            f'<div style="font-size:0.6rem;margin-top:2px;white-space:nowrap;">'
                            f'<span style="display:inline-block;width:7px;height:7px;'
                            f"border-radius:50%;background:{_trust_dot};"
                            f'vertical-align:middle;margin-right:3px;"></span>'
                            f'<span style="color:{_trust_dot};font-weight:600;">'
                            f"gmLI {_gmli_val:.2f} - {_trust_label}</span>"
                            f"{_trend_html}</div>"
                        )

                    # Build actual stats line if available
                    actual_sv_html = "<!-- no actual stats -->"
                    if actual_sv is not None:
                        actual_era_str = format_stat(actual_era, "ERA") if actual_era else "—"
                        actual_whip_str = format_stat(actual_whip, "WHIP") if actual_whip else "—"
                        actual_sv_html = (
                            f'<div style="font-size:0.65rem;color:#2e7d32;'
                            f'margin-top:2px;white-space:nowrap;font-weight:600;">'
                            f"2026 Actual: {actual_sv} SV | {actual_era_str} ERA | "
                            f"{actual_whip_str} WHIP</div>"
                        )

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
        {headshot}{closer_name_safe}
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
    {actual_sv_html}
    {form_html}
    {gmli_html}
    <div style="font-size:0.65rem; color:#888; margin-top:3px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
        Setup: {setup_str}
    </div>
</div>
""",
                        unsafe_allow_html=True,
                    )

page_timer_footer("Closer Monitor")

"""Leaders — Category leaders, fantasy points leaders, and prospect rankings."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.database import coerce_numeric_df, get_connection, init_db, load_player_pool
from src.ui_shared import (
    THEME,
    get_plotly_layout,
    get_plotly_polar,
    inject_custom_css,
    page_timer_footer,
    page_timer_start,
    render_compact_table,
    render_context_card,
    render_context_columns,
    render_page_layout,
    render_player_select,
)

T = THEME

try:
    from src.leaders import compute_category_leaders, detect_breakouts  # noqa: F401

    LEADERS_AVAILABLE = True
except ImportError:
    LEADERS_AVAILABLE = False

try:
    from src.points_league import compute_fantasy_points, get_scoring_preset  # noqa: F401

    POINTS_AVAILABLE = True
except ImportError:
    POINTS_AVAILABLE = False

try:
    from src.prospect_engine import get_prospect_rankings, refresh_prospect_rankings

    PROSPECTS_AVAILABLE = True
except ImportError:
    PROSPECTS_AVAILABLE = False

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

st.set_page_config(page_title="Heater | Leaders", page_icon="", layout="wide", initial_sidebar_state="collapsed")

init_db()

inject_custom_css()
page_timer_start()

render_page_layout("LEADERS", banner_teaser="Category leaders and breakout detection", banner_icon="leaders")


@st.cache_data(ttl=300)
def _load_stats_df() -> tuple[pd.DataFrame, bool]:
    """Load season stats joined with player metadata.

    Returns (DataFrame, is_projection) where is_projection=True means the
    data came from blended projections because season_stats was empty.
    """
    # Try season_stats for 2026 first
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            """
            SELECT
                s.player_id,
                p.name,
                p.team,
                p.positions,
                p.is_hitter,
                p.mlb_id,
                s.pa, s.ab, s.h,
                s.r, s.hr, s.rbi, s.sb,
                s.avg, s.obp,
                s.ip, s.w, s.l, s.sv, s.k,
                s.era, s.whip
            FROM season_stats s
            JOIN players p ON p.player_id = s.player_id
            WHERE s.season = 2026
            """,
            conn,
        )
    finally:
        conn.close()

    df = coerce_numeric_df(df)

    if not df.empty:
        return df, False

    # Pre-season fallback: use blended projections from player_pool
    proj = load_player_pool()
    if proj.empty:
        return pd.DataFrame(), True

    # player_pool returns 'name' column; alias to match leaders expectations
    keep = [
        "player_id",
        "name",
        "team",
        "positions",
        "is_hitter",
        "mlb_id",
        "pa",
        "ab",
        "h",
        "r",
        "hr",
        "rbi",
        "sb",
        "avg",
        "obp",
        "ip",
        "w",
        "l",
        "sv",
        "k",
        "era",
        "whip",
    ]
    keep = [c for c in keep if c in proj.columns]
    return proj[keep].copy(), True


# -- Prospect tab helper functions -----------------------------------------


def _parse_eta_year(val) -> int | None:
    """Parse ETA string to integer year, returning None on failure."""
    if val is None:
        return None
    try:
        return int(str(val).strip())
    except (ValueError, TypeError):
        return None


def _prospect_is_pitcher(row) -> bool:
    """Determine if prospect is a pitcher based on position."""
    pos = str(row.get("position", "")).upper()
    return "SP" in pos or "RP" in pos or "P" == pos


def _build_scouting_grades(row, is_pitcher: bool) -> list[dict]:
    """Build scouting grade rows from prospect data."""
    grades = []
    if is_pitcher:
        # Pitcher tools: Control present/future
        if row.get("ctrl_present") is not None or row.get("ctrl_future") is not None:
            grades.append(
                {
                    "Tool": "Control",
                    "Present": _grade_val(row.get("ctrl_present")),
                    "Future": _grade_val(row.get("ctrl_future")),
                }
            )
        # Also show hit/game/raw/speed/field if available (two-way players)
        _add_hitter_tools(grades, row)
    else:
        # Hitter tools: Hit, Game Power, Raw Power, Speed, Field
        _add_hitter_tools(grades, row)
        # Also show control if available (two-way players)
        if row.get("ctrl_present") is not None or row.get("ctrl_future") is not None:
            grades.append(
                {
                    "Tool": "Control",
                    "Present": _grade_val(row.get("ctrl_present")),
                    "Future": _grade_val(row.get("ctrl_future")),
                }
            )
    return grades


def _add_hitter_tools(grades: list[dict], row) -> None:
    """Add hitter scouting tools to grades list."""
    tool_map = [
        ("Hit", "hit_present", "hit_future"),
        ("Game Power", "game_present", "game_future"),
        ("Raw Power", "raw_present", "raw_future"),
        ("Speed", "speed", None),
        ("Fielding", "field", None),
    ]
    for label, present_key, future_key in tool_map:
        present = row.get(present_key)
        future = row.get(future_key) if future_key else None
        if present is not None or future is not None:
            entry = {"Tool": label, "Present": _grade_val(present)}
            if future_key is not None:
                entry["Future"] = _grade_val(future)
            else:
                entry["Future"] = "--"
            grades.append(entry)


def _grade_val(val) -> str:
    """Format a scouting grade value for display."""
    if val is None:
        return "--"
    try:
        return str(int(val))
    except (ValueError, TypeError):
        return str(val)


def _render_scouting_radar(name: str, grades: list[dict], is_pitcher: bool) -> None:
    """Render a Plotly radar chart for scouting tool grades."""
    if not HAS_PLOTLY:
        return
    # Build data for radar
    labels = []
    present_vals = []
    future_vals = []
    for g in grades:
        labels.append(g["Tool"])
        try:
            present_vals.append(float(g["Present"]) if g["Present"] != "--" else 0)
        except (ValueError, TypeError):
            present_vals.append(0)
        try:
            future_vals.append(float(g["Future"]) if g["Future"] != "--" else 0)
        except (ValueError, TypeError):
            future_vals.append(0)

    if not labels:
        return

    has_future = any(v > 0 for v in future_vals)

    fig = go.Figure()

    # Present grades trace
    fig.add_trace(
        go.Scatterpolar(
            r=present_vals + [present_vals[0]],
            theta=labels + [labels[0]],
            name="Present",
            line=dict(color=T["sky"]),
            fill="toself",
            fillcolor="rgba(69,123,157,0.15)",
        )
    )

    # Future grades trace (if available)
    if has_future:
        fig.add_trace(
            go.Scatterpolar(
                r=future_vals + [future_vals[0]],
                theta=labels + [labels[0]],
                name="Future",
                line=dict(color=T["hot"]),
                fill="toself",
                fillcolor="rgba(255,109,0,0.15)",
            )
        )

    layout_kwargs = get_plotly_layout(T)
    layout_kwargs["polar"] = get_plotly_polar(T)
    # Set radial axis range for 20-80 scouting scale
    layout_kwargs["polar"]["radialaxis"]["range"] = [20, 80]
    layout_kwargs["polar"]["radialaxis"]["tickvals"] = [20, 30, 40, 50, 60, 70, 80]
    layout_kwargs["legend"] = dict(font=dict(color=T["tx"]))
    layout_kwargs["title"] = dict(
        text=f"{name} Scouting Profile",
        font=dict(size=14, color=T["tx"]),
    )
    layout_kwargs["height"] = 400
    fig.update_layout(**layout_kwargs)
    st.plotly_chart(fig, width="stretch")


# -- 3-zone hybrid layout --------------------------------------------------

ctx, main = render_context_columns()

# Defaults for variables defined conditionally inside ctx
refresh_clicked = False

# -- Context panel (left): tab-sensitive filter controls ------------------

with ctx:
    # Category Leaders controls
    if LEADERS_AVAILABLE:
        render_context_card(
            "Category Filter",
            "<p>Select a statistical category to rank players.</p>",
        )
        category = st.selectbox(
            "Category",
            ["HR", "R", "RBI", "SB", "AVG", "OBP", "W", "SV", "K", "ERA", "WHIP"],
            key="cat_leader",
        )

    # Points Leaders controls
    if POINTS_AVAILABLE:
        render_context_card(
            "Scoring Preset",
            "<p>Select a scoring system for fantasy points calculation.</p>",
        )
        preset_name = st.selectbox("Scoring Preset", ["yahoo", "espn", "cbs"], key="pts_preset")

    # Prospects controls
    if PROSPECTS_AVAILABLE:
        render_context_card(
            "Prospect Filters",
            "<p>Filter prospects by position, organization, and ETA year.</p>",
        )

        # Position filter
        _POSITION_OPTIONS = ["All", "SS", "OF", "SP", "2B", "3B", "1B", "C", "RP"]
        position_filter = st.selectbox(
            "Position",
            _POSITION_OPTIONS,
            key="prospect_pos",
        )

        # Pre-load org options from cached data
        _org_options = ["All"]
        try:
            _preload = get_prospect_rankings(top_n=200)
            if not _preload.empty and "team" in _preload.columns:
                _org_options += sorted(_preload["team"].dropna().unique().tolist())
        except Exception:
            pass

        # Organization filter
        org_filter = st.selectbox(
            "Organization",
            _org_options,
            key="prospect_org",
        )

        # ETA year filter
        eta_filter = st.selectbox(
            "ETA Year",
            ["All", "2025", "2026", "2027", "2028", "2029+"],
            key="prospect_eta",
        )

        # Refresh button
        refresh_clicked = st.button("Refresh Data", key="prospect_refresh")

# -- Main content (right): tabs + data tables -----------------------------

with main:
    tab1, tab2, tab3 = st.tabs(["Category Leaders", "Points Leaders", "Prospects"])

    with tab1:
        if not LEADERS_AVAILABLE:
            st.info("Leaders module not available. Ensure src/leaders.py exists.")
        else:
            st.markdown("Top performers in each statistical category.")

            # Column display mapping for category leaders
            _CAT_DISPLAY = {
                "name": "Player",
                "team": "Team",
                "positions": "Position",
                "r": "Runs",
                "hr": "Home Runs",
                "rbi": "Runs Batted In",
                "sb": "Stolen Bases",
                "avg": "Batting Average",
                "obp": "On-Base Percentage",
                "w": "Wins",
                "sv": "Saves",
                "k": "Strikeouts",
                "era": "Earned Run Average",
                "whip": "Walks + Hits per Inning Pitched",
                "l": "Losses",
                "pa": "Plate Appearances",
                "ip": "Innings Pitched",
            }
            _CAT_COL = {
                "HR": "hr",
                "R": "r",
                "RBI": "rbi",
                "SB": "sb",
                "AVG": "avg",
                "OBP": "obp",
                "W": "w",
                "SV": "sv",
                "K": "k",
                "ERA": "era",
                "WHIP": "whip",
                "L": "l",
            }

            try:
                stats_df, _is_proj = _load_stats_df()
                if stats_df.empty:
                    st.info("No player stats available. Run the data bootstrap to load projections.")
                    st.stop()
                if _is_proj:
                    st.caption("Showing preseason projections — season stats not yet available.")
                total_eligible = len(stats_df)
                leaders = compute_category_leaders(stats_df, categories=[category], top_n=15)
                if category in leaders:
                    ldf = leaders[category].copy()
                    st.caption(f"Showing top {len(ldf)} of {total_eligible:,} eligible players")
                    stat_col = _CAT_COL.get(category, category.lower())
                    show_cols = ["name", "team", "positions", stat_col]
                    # Include mlb_id for headshot rendering (auto-hidden by table)
                    if "mlb_id" in ldf.columns:
                        show_cols.append("mlb_id")
                    show_cols = [c for c in show_cols if c in ldf.columns]
                    ldf = ldf[show_cols].rename(columns=_CAT_DISPLAY)
                    render_compact_table(ldf)

                    # Player card selector
                    if "player_id" in leaders[category].columns and "Player" in ldf.columns:
                        render_player_select(
                            ldf["Player"].tolist(),
                            leaders[category]["player_id"].tolist(),
                            key_suffix="leaders",
                        )
                else:
                    st.info(f"No leaders found for {category}.")
            except Exception as e:
                st.error(f"Failed to compute category leaders: {e}")

    with tab2:
        if not POINTS_AVAILABLE:
            st.info("Points league module not available. Ensure src/points_league.py exists.")
        else:
            st.markdown("Top performers by fantasy points scoring.")

            try:
                hitting_w, pitching_w = get_scoring_preset(preset_name)

                from src.points_league import compute_points_leaders

                stats_df_pts, _is_proj_pts = _load_stats_df()
                if stats_df_pts.empty:
                    st.info("No player stats available. Run the data bootstrap to load projections.")
                    st.stop()
                if _is_proj_pts:
                    st.caption("Showing preseason projections — season stats not yet available.")
                total_pts_eligible = len(stats_df_pts)
                pts_leaders = compute_points_leaders(stats_df_pts, hitting_w, pitching_w, top_n=20)
                if not pts_leaders.empty:
                    pts_df = pts_leaders.copy()
                    st.caption(f"Showing top {len(pts_df)} of {total_pts_eligible:,} eligible players")
                    _pts_show = ["name", "team", "positions", "fantasy_points"]
                    # Include mlb_id for headshot rendering (auto-hidden by table)
                    if "mlb_id" in pts_df.columns:
                        _pts_show.append("mlb_id")
                    _pts_show = [c for c in _pts_show if c in pts_df.columns]
                    _PTS_DISPLAY = {
                        "name": "Player",
                        "team": "Team",
                        "positions": "Position",
                        "fantasy_points": "Fantasy Points",
                    }
                    display_pts = pts_df[_pts_show].rename(columns=_PTS_DISPLAY)
                    render_compact_table(display_pts)

                    # Player card selector
                    if "player_id" in pts_df.columns and "Player" in display_pts.columns:
                        render_player_select(
                            display_pts["Player"].tolist(),
                            pts_df["player_id"].tolist(),
                            key_suffix="leaders_pts",
                        )
                else:
                    st.info("No points leaders computed.")
            except Exception as e:
                st.error(f"Failed to compute points leaders: {e}")

    with tab3:
        if not PROSPECTS_AVAILABLE:
            st.info("Prospect rankings not available. Ensure src/prospect_engine.py exists.")
        else:
            st.markdown("Top MLB prospects with readiness scores and scouting tool grades.")

            try:
                # Refresh if requested
                if refresh_clicked:
                    with st.spinner("Refreshing prospect rankings..."):
                        prospects_df = refresh_prospect_rankings(force=True)
                else:
                    prospects_df = get_prospect_rankings(
                        top_n=100,
                        position=position_filter if position_filter != "All" else None,
                        org=None,  # apply org filter after to populate dropdown
                    )

                if prospects_df.empty:
                    st.info("No prospect data available. Click Refresh Data to fetch rankings.")
                else:
                    # Reload with all filters applied
                    pos_arg = position_filter if position_filter != "All" else None
                    org_arg = org_filter if org_filter != "All" else None
                    prospects_df = get_prospect_rankings(top_n=100, position=pos_arg, org=org_arg)

                    # ETA filter
                    if eta_filter != "All" and "fg_eta" in prospects_df.columns:
                        if eta_filter == "2029+":
                            prospects_df = prospects_df[
                                prospects_df["fg_eta"].apply(
                                    lambda x: _parse_eta_year(x) is not None and _parse_eta_year(x) >= 2029
                                )
                            ]
                        else:
                            target_year = int(eta_filter)
                            prospects_df = prospects_df[
                                prospects_df["fg_eta"].apply(lambda x: _parse_eta_year(x) == target_year)
                            ]
                    elif eta_filter != "All" and "eta" in prospects_df.columns:
                        if eta_filter == "2029+":
                            prospects_df = prospects_df[
                                prospects_df["eta"].apply(
                                    lambda x: _parse_eta_year(x) is not None and _parse_eta_year(x) >= 2029
                                )
                            ]
                        else:
                            target_year = int(eta_filter)
                            prospects_df = prospects_df[
                                prospects_df["eta"].apply(lambda x: _parse_eta_year(x) == target_year)
                            ]

                    if prospects_df.empty:
                        st.info("No prospects match the selected filters.")
                    else:
                        # Build display table
                        display_cols = []
                        col_map = {}
                        if "fg_rank" in prospects_df.columns:
                            display_cols.append("fg_rank")
                            col_map["fg_rank"] = "Rank"
                        elif "rank" in prospects_df.columns:
                            display_cols.append("rank")
                            col_map["rank"] = "Rank"
                        for c, label in [
                            ("name", "Player"),
                            ("team", "Organization"),
                            ("position", "Position"),
                            ("fg_fv", "Future Value"),
                            ("fv", "Future Value"),
                            ("fg_eta", "ETA"),
                            ("eta", "ETA"),
                            ("fg_risk", "Risk"),
                            ("readiness_score", "Readiness Score"),
                        ]:
                            if c in prospects_df.columns and c not in display_cols:
                                # Avoid duplicate ETA/FV columns
                                if label in col_map.values():
                                    continue
                                display_cols.append(c)
                                col_map[c] = label

                        display_df = prospects_df[display_cols].rename(columns=col_map)
                        render_compact_table(display_df, show_avatars=True)

                        # -- Expandable scouting details per prospect -----------------
                        st.markdown("---")
                        st.subheader("Prospect Scouting Details")
                        st.markdown("Select a prospect to view scouting tool grades and radar chart.")

                        # Build name list for selection
                        prospect_names = prospects_df["name"].tolist()
                        selected_name = st.selectbox(
                            "Select Prospect",
                            prospect_names,
                            key="prospect_detail_select",
                        )

                        if selected_name:
                            row = prospects_df[prospects_df["name"] == selected_name].iloc[0]

                            detail_cols = st.columns([1, 1])

                            with detail_cols[0]:
                                # Scouting tool grades table
                                _is_pitcher = _prospect_is_pitcher(row)
                                grades = _build_scouting_grades(row, _is_pitcher)

                                if grades:
                                    st.markdown("**Scouting Tool Grades (20-80 Scale)**")
                                    grade_df = pd.DataFrame(grades)
                                    render_compact_table(grade_df)
                                else:
                                    st.info("No scouting tool grades available for this prospect.")

                                # Additional info
                                info_parts = []
                                if row.get("fg_risk"):
                                    info_parts.append(f"**Risk Level:** {row['fg_risk']}")
                                if row.get("readiness_score") is not None:
                                    score = row["readiness_score"]
                                    info_parts.append(f"**Readiness Score:** {score:.2f} / 100")
                                if row.get("age") is not None:
                                    info_parts.append(f"**Age:** {row['age']}")
                                if row.get("milb_level"):
                                    info_parts.append(f"**Current Level:** {row['milb_level']}")
                                if info_parts:
                                    st.markdown(" | ".join(info_parts))

                                # TL;DR / scouting report
                                if row.get("tldr"):
                                    st.markdown(f"**Summary:** {row['tldr']}")
                                if row.get("scouting_report"):
                                    with st.expander("Full Scouting Report"):
                                        st.markdown(row["scouting_report"])

                            with detail_cols[1]:
                                # Radar chart for scouting tools
                                if HAS_PLOTLY and grades:
                                    _render_scouting_radar(selected_name, grades, _is_pitcher)
                                elif not HAS_PLOTLY:
                                    st.info("Install plotly to view scouting radar charts.")

            except Exception as e:
                st.error(f"Failed to load prospect rankings: {e}")

page_timer_footer("Leaders")

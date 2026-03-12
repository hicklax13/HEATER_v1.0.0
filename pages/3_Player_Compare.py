"""Player Compare — Head-to-head comparison with z-scores and radar chart."""

import pandas as pd
import streamlit as st

from src.database import init_db, load_player_pool
from src.in_season import compare_players
from src.injury_model import compute_health_score, get_injury_badge
from src.ui_shared import (
    ALL_CATEGORIES,
    METRIC_TOOLTIPS,
    get_plotly_layout,
    get_plotly_polar,
    get_theme,
    inject_custom_css,
    render_theme_toggle,
)
from src.valuation import LeagueConfig, add_process_risk, compute_percentile_projections, compute_projection_volatility

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

st.set_page_config(page_title="Player Compare", page_icon="⚔️", layout="wide")

init_db()

inject_custom_css()
render_theme_toggle()

st.title("⚔️ Player Compare")

pool = load_player_pool()
if pool.empty:
    st.warning("No player data loaded.")
    st.stop()

pool = pool.rename(columns={"name": "player_name"})
config = LeagueConfig()

player_names = sorted(pool["player_name"].tolist())

col1, col2 = st.columns(2)
with col1:
    player_a_name = st.selectbox("Player A", options=player_names, index=0, key="pa")
with col2:
    player_b_name = st.selectbox("Player B", options=player_names, index=min(1, len(player_names) - 1), key="pb")

if player_a_name and player_b_name and player_a_name != player_b_name:
    match_a = pool[pool["player_name"] == player_a_name]
    match_b = pool[pool["player_name"] == player_b_name]
    if match_a.empty or match_b.empty:
        st.error("Could not find one or both selected players in the pool.")
        st.stop()
    id_a = match_a.iloc[0]["player_id"]
    id_b = match_b.iloc[0]["player_id"]

    result = compare_players(int(id_a), int(id_b), pool, config)

    # Load health scores from injury_history table
    health_dict = {}
    try:
        from src.database import get_connection

        conn = get_connection()
        injury_df = pd.read_sql_query("SELECT * FROM injury_history", conn)
        conn.close()
        if not injury_df.empty and "player_id" in injury_df.columns:
            for pid, group in injury_df.groupby("player_id"):
                gp = group["games_played"].tolist()
                ga = group["games_available"].tolist()
                health_dict[pid] = compute_health_score(gp, ga)
    except Exception:
        pass

    if "error" in result:
        st.error(result["error"])
    else:
        # Composite scores
        col1, col2 = st.columns(2)
        col1.metric(
            result["player_a"],
            f"{result['composite_a']:+.2f}",
            label_visibility="visible",
            help=METRIC_TOOLTIPS["composite_score"],
        )
        col2.metric(
            result["player_b"],
            f"{result['composite_b']:+.2f}",
            label_visibility="visible",
            help=METRIC_TOOLTIPS["composite_score"],
        )

        # Radar chart
        if HAS_PLOTLY:
            t = get_theme()
            cats = ALL_CATEGORIES
            z_a = [result["z_scores_a"].get(c, 0) for c in cats]
            z_b = [result["z_scores_b"].get(c, 0) for c in cats]

            fig = go.Figure()
            fig.add_trace(
                go.Scatterpolar(
                    r=z_a + [z_a[0]],
                    theta=cats + [cats[0]],
                    name=result["player_a"],
                    line=dict(color=t["amber"]),
                    fill="toself",
                    fillcolor="rgba(245,158,11,0.15)",
                )
            )
            fig.add_trace(
                go.Scatterpolar(
                    r=z_b + [z_b[0]],
                    theta=cats + [cats[0]],
                    name=result["player_b"],
                    line=dict(color=t["teal"]),
                    fill="toself",
                    fillcolor="rgba(6,182,212,0.15)",
                )
            )
            layout_kwargs = get_plotly_layout(t)
            layout_kwargs["polar"] = get_plotly_polar(t)
            layout_kwargs["legend"] = dict(font=dict(color=t["tx"]))
            fig.update_layout(**layout_kwargs)
            st.plotly_chart(fig, width="stretch")

        # Z-score comparison table
        st.subheader("Category Breakdown")
        rows = []
        for cat in ALL_CATEGORIES:
            za = result["z_scores_a"].get(cat, 0)
            zb = result["z_scores_b"].get(cat, 0)
            adv = result["advantages"].get(cat, "TIE")
            rows.append(
                {
                    "Category": cat,
                    f"{result['player_a']} Z": f"{za:+.2f}",
                    f"{result['player_b']} Z": f"{zb:+.2f}",
                    "Advantage": adv,
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption(METRIC_TOOLTIPS["z_score"])

        # Health comparison
        st.subheader("Health & Confidence")
        health_rows = []
        for name_col, pid_col in [(player_a_name, id_a), (player_b_name, id_b)]:
            hs = health_dict.get(pid_col, 0.85)
            icon, label = get_injury_badge(hs)
            health_rows.append({"Player": name_col, "Health": f"{icon} {label}", "Score": f"{hs:.2f}"})

        # Projection confidence: P10-P90 range width per player using key stat cols
        try:
            from src.database import get_connection

            conn = get_connection()
            systems = {}
            for sys_name in ["steamer", "zips", "depthcharts", "blended"]:
                df = pd.read_sql_query(
                    "SELECT * FROM projections WHERE system = ?",
                    conn,
                    params=(sys_name,),
                )
                if not df.empty:
                    systems[sys_name] = df
            conn.close()

            if len(systems) >= 2:
                vol = compute_projection_volatility(systems)
                vol = add_process_risk(vol)
                # pool uses 'player_name' column; pass the pool renamed back to 'name' for base
                base_df = pool.rename(columns={"player_name": "name"})
                pct = compute_percentile_projections(base=base_df, volatility=vol)
                p10_df, p90_df = pct.get(10), pct.get(90)
                stat_cols = ["r", "hr", "rbi", "sb", "w", "sv", "k"]
                if p10_df is not None and p90_df is not None:
                    name_col_key = "name" if "name" in p10_df.columns else "player_name"
                    for row in health_rows:
                        name = row["Player"]
                        p10_row = p10_df[p10_df[name_col_key] == name]
                        p90_row = p90_df[p90_df[name_col_key] == name]
                        if not p10_row.empty and not p90_row.empty:
                            present = [c for c in stat_cols if c in p10_df.columns]
                            if present:
                                p10_sum = p10_row.iloc[0][present].sum()
                                p90_sum = p90_row.iloc[0][present].sum()
                                width = p90_sum - p10_sum
                                row["Confidence"] = f"±{width:.1f}" if width > 0 else "—"
                            else:
                                row["Confidence"] = "—"
                        else:
                            row["Confidence"] = "—"
            else:
                for row in health_rows:
                    row["Confidence"] = "—"
        except Exception:
            for row in health_rows:
                row["Confidence"] = "—"

        st.dataframe(pd.DataFrame(health_rows), hide_index=True, width="stretch")
else:
    if player_a_name == player_b_name:
        st.info("Select two different players to compare.")

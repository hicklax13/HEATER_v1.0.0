"""Player Compare — Head-to-head comparison with z-scores and radar chart."""

import pandas as pd
import streamlit as st

from src.database import init_db, load_player_pool
from src.in_season import compare_players
from src.ui_shared import ALL_CATEGORIES, T
from src.valuation import LeagueConfig

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

st.set_page_config(page_title="Player Compare", page_icon="⚔️", layout="wide")

init_db()

st.markdown(
    f"""<style>
    .stApp {{ background-color: {T["bg"]}; }}
    h1, h2, h3 {{ color: {T["amber"]}; font-family: 'Oswald', sans-serif; }}
    </style>""",
    unsafe_allow_html=True,
)

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

    if "error" in result:
        st.error(result["error"])
    else:
        # Composite scores
        col1, col2 = st.columns(2)
        col1.metric(result["player_a"], f"{result['composite_a']:+.2f}", label_visibility="visible")
        col2.metric(result["player_b"], f"{result['composite_b']:+.2f}", label_visibility="visible")

        # Radar chart
        if HAS_PLOTLY:
            cats = ALL_CATEGORIES
            z_a = [result["z_scores_a"].get(c, 0) for c in cats]
            z_b = [result["z_scores_b"].get(c, 0) for c in cats]

            fig = go.Figure()
            fig.add_trace(
                go.Scatterpolar(
                    r=z_a + [z_a[0]],
                    theta=cats + [cats[0]],
                    name=result["player_a"],
                    line=dict(color=T["amber"]),
                    fill="toself",
                    fillcolor="rgba(245,158,11,0.15)",
                )
            )
            fig.add_trace(
                go.Scatterpolar(
                    r=z_b + [z_b[0]],
                    theta=cats + [cats[0]],
                    name=result["player_b"],
                    line=dict(color=T["teal"]),
                    fill="toself",
                    fillcolor="rgba(6,182,212,0.15)",
                )
            )
            fig.update_layout(
                polar=dict(
                    bgcolor=T["card"],
                    radialaxis=dict(gridcolor=T["card_h"], tickfont=dict(color=T["tx2"])),
                    angularaxis=dict(gridcolor=T["card_h"], tickfont=dict(color=T["tx"])),
                ),
                paper_bgcolor=T["bg"],
                font=dict(color=T["tx"]),
                legend=dict(font=dict(color=T["tx"])),
                margin=dict(l=60, r=60, t=40, b=40),
            )
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
else:
    if player_a_name == player_b_name:
        st.info("Select two different players to compare.")

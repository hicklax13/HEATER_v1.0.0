"""Leaders — Category leaders, fantasy points leaders, and prospect rankings."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.database import init_db
from src.ui_shared import inject_custom_css

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
    from src.ecr import fetch_prospect_rankings, filter_prospects_by_position

    PROSPECTS_AVAILABLE = True
except ImportError:
    PROSPECTS_AVAILABLE = False

st.set_page_config(page_title="Heater | Leaders", page_icon="", layout="wide")

init_db()

inject_custom_css()

st.markdown(
    '<div class="page-title-wrap"><div class="page-title"><span>LEAGUE LEADERS</span></div></div>',
    unsafe_allow_html=True,
)

# Shared demo stat generation
rng = np.random.default_rng(42)
n_players = 50
demo_stats = pd.DataFrame(
    {
        "player_id": range(1, n_players + 1),
        "name": [f"Player {i}" for i in range(1, n_players + 1)],
        "team": [f"TM{i % 12 + 1}" for i in range(n_players)],
        "positions": ["OF" if i % 3 == 0 else "SS" if i % 3 == 1 else "SP" for i in range(n_players)],
        "is_hitter": [i % 3 != 2 for i in range(n_players)],
        "pa": [int(rng.integers(200, 600)) if i % 3 != 2 else 0 for i in range(n_players)],
        "ip": [0.0 if i % 3 != 2 else float(rng.integers(50, 200)) for i in range(n_players)],
        "r": [int(rng.integers(30, 100)) if i % 3 != 2 else 0 for i in range(n_players)],
        "hr": [int(rng.integers(5, 40)) if i % 3 != 2 else 0 for i in range(n_players)],
        "rbi": [int(rng.integers(30, 110)) if i % 3 != 2 else 0 for i in range(n_players)],
        "sb": [int(rng.integers(0, 30)) if i % 3 != 2 else 0 for i in range(n_players)],
        "avg": [round(0.220 + rng.random() * 0.080, 3) if i % 3 != 2 else 0.0 for i in range(n_players)],
        "obp": [round(0.290 + rng.random() * 0.090, 3) if i % 3 != 2 else 0.0 for i in range(n_players)],
        "w": [0 if i % 3 != 2 else int(rng.integers(3, 18)) for i in range(n_players)],
        "sv": [0 if i % 3 != 2 else int(rng.integers(0, 35)) for i in range(n_players)],
        "k": [0 if i % 3 != 2 else int(rng.integers(40, 250)) for i in range(n_players)],
        "era": [0.0 if i % 3 != 2 else round(2.5 + rng.random() * 3.0, 2) for i in range(n_players)],
        "whip": [0.0 if i % 3 != 2 else round(0.90 + rng.random() * 0.50, 2) for i in range(n_players)],
        "l": [0 if i % 3 != 2 else int(rng.integers(2, 14)) for i in range(n_players)],
    }
)

tab1, tab2, tab3 = st.tabs(["Category Leaders", "Points Leaders", "Prospects"])

with tab1:
    if not LEADERS_AVAILABLE:
        st.info("Leaders module not available. Ensure src/leaders.py exists.")
    else:
        st.markdown("Top performers in each statistical category.")

        category = st.selectbox(
            "Category",
            ["HR", "R", "RBI", "SB", "AVG", "OBP", "W", "SV", "K", "ERA", "WHIP"],
            key="cat_leader",
        )

        try:
            leaders = compute_category_leaders(demo_stats, categories=[category], top_n=15)
            if category in leaders:
                st.dataframe(leaders[category], width="stretch", hide_index=True)
            else:
                st.info(f"No leaders found for {category}.")
        except Exception as e:
            st.error(f"Failed to compute category leaders: {e}")

with tab2:
    if not POINTS_AVAILABLE:
        st.info("Points league module not available. Ensure src/points_league.py exists.")
    else:
        st.markdown("Top performers by fantasy points scoring.")

        preset_name = st.selectbox("Scoring Preset", ["yahoo", "espn", "cbs"], key="pts_preset")

        try:
            hitting_w, pitching_w = get_scoring_preset(preset_name)

            from src.points_league import compute_points_leaders

            pts_leaders = compute_points_leaders(demo_stats, hitting_w, pitching_w, top_n=20)
            if pts_leaders:
                pts_df = pd.DataFrame(pts_leaders)
                st.dataframe(pts_df, width="stretch", hide_index=True)
            else:
                st.info("No points leaders computed.")
        except Exception as e:
            st.error(f"Failed to compute points leaders: {e}")

with tab3:
    if not PROSPECTS_AVAILABLE:
        st.info("Prospect rankings not available. Ensure src/ecr.py exists.")
    else:
        st.markdown("Top MLB prospects by consensus ranking.")

        position_filter = st.selectbox(
            "Position Filter",
            ["All", "SP", "OF", "SS", "2B", "3B", "1B", "C"],
            key="prospect_pos",
        )

        try:
            prospects_df = fetch_prospect_rankings()

            if position_filter != "All":
                prospects_df = filter_prospects_by_position(prospects_df, position_filter)

            if prospects_df.empty:
                st.info("No prospects found for this position.")
            else:
                st.dataframe(prospects_df, width="stretch", hide_index=True)
        except Exception as e:
            st.error(f"Failed to fetch prospect rankings: {e}")

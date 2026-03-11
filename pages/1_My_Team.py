"""My Team — Roster overview and category standings."""

import pandas as pd
import streamlit as st

from src.database import init_db, load_league_rosters
from src.league_manager import get_team_roster
from src.live_stats import refresh_all_stats
from src.ui_shared import T

st.set_page_config(page_title="My Team", page_icon="👤", layout="wide")

init_db()

st.markdown(
    f"""<style>
    .stApp {{ background-color: {T["bg"]}; }}
    h1, h2, h3 {{ color: {T["amber"]}; font-family: 'Oswald', sans-serif; }}
    </style>""",
    unsafe_allow_html=True,
)

st.title("👤 My Team")

# Determine user team
rosters = load_league_rosters()
if rosters.empty:
    st.warning("No league data loaded. Go to the main app and import your league rosters via Step 5.")
    st.stop()
else:
    user_teams = rosters[rosters["is_user_team"] == 1]
    if user_teams.empty:
        st.warning("No user team identified in roster data.")
        st.stop()
    else:
        user_team_name = user_teams.iloc[0]["team_name"]
        st.markdown(f"**Team:** {user_team_name}")

        # Refresh button
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🔄 Refresh Stats"):
                with st.spinner("Pulling live stats from MLB..."):
                    result = refresh_all_stats(force=True)
                    for source, status in result.items():
                        st.toast(f"{source}: {status}")
                    st.rerun()

        # Load roster
        roster = get_team_roster(user_team_name)
        if roster.empty:
            st.info("No players on your roster yet.")
        else:
            # Display roster
            st.subheader("Roster")
            display_cols = ["name", "positions", "roster_slot"]
            available_cols = [c for c in display_cols if c in roster.columns]
            st.dataframe(
                roster[available_cols] if available_cols else roster,
                width="stretch",
                hide_index=True,
            )

            # Category totals
            st.subheader("Category Totals")
            hitters = roster[roster["is_hitter"] == 1]
            pitchers = roster[roster["is_hitter"] == 0]

            hit_stats = {}
            if not hitters.empty:
                for cat, col in [("R", "r"), ("HR", "hr"), ("RBI", "rbi"), ("SB", "sb")]:
                    hit_stats[cat] = int(hitters[col].sum()) if col in hitters.columns else 0
                ab = hitters["ab"].sum() if "ab" in hitters.columns else 0
                h = hitters["h"].sum() if "h" in hitters.columns else 0
                hit_stats["AVG"] = f"{h / ab:.3f}" if ab > 0 else ".000"

            pitch_stats = {}
            if not pitchers.empty:
                for cat, col in [("W", "w"), ("SV", "sv"), ("K", "k")]:
                    pitch_stats[cat] = int(pitchers[col].sum()) if col in pitchers.columns else 0
                ip = pitchers["ip"].sum() if "ip" in pitchers.columns else 0
                er = pitchers["er"].sum() if "er" in pitchers.columns else 0
                bb = pitchers["bb_allowed"].sum() if "bb_allowed" in pitchers.columns else 0
                ha = pitchers["h_allowed"].sum() if "h_allowed" in pitchers.columns else 0
                pitch_stats["ERA"] = f"{er * 9 / ip:.2f}" if ip > 0 else "0.00"
                pitch_stats["WHIP"] = f"{(bb + ha) / ip:.3f}" if ip > 0 else "0.000"

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Hitting**")
                if hit_stats:
                    st.dataframe(pd.DataFrame([hit_stats]), hide_index=True)
            with col2:
                st.markdown("**Pitching**")
                if pitch_stats:
                    st.dataframe(pd.DataFrame([pitch_stats]), hide_index=True)

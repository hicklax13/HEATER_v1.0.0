"""Category Tracker — Monitor standings gaps and identify punchable categories."""

import logging
from datetime import UTC, datetime

import pandas as pd
import streamlit as st

from src.database import init_db, load_league_rosters, load_league_standings, load_player_pool
from src.ui_shared import inject_custom_css, render_styled_table
from src.valuation import LeagueConfig

try:
    from src.engine.portfolio.category_analysis import (
        category_gap_analysis,
        compute_category_weights_from_analysis,
    )

    _HAS_CATEGORY_ANALYSIS = True
except ImportError:
    _HAS_CATEGORY_ANALYSIS = False

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Heater | Category Tracker", page_icon="", layout="wide")

init_db()
inject_custom_css()

st.markdown(
    '<div class="page-title-wrap"><div class="page-title"><span>CATEGORY TRACKER</span></div></div>',
    unsafe_allow_html=True,
)

if not _HAS_CATEGORY_ANALYSIS:
    st.error("Category analysis module not available.")
    st.stop()

pool = load_player_pool()
if pool.empty:
    st.warning("No player data loaded.")
    st.stop()

pool = pool.rename(columns={"name": "player_name"})
config = LeagueConfig()

# ── Load Standings ──────────────────────────────────────────────────────────

standings = load_league_standings()
if standings.empty:
    st.warning(
        "No standings data loaded. Connect your Yahoo league in Connect League, "
        "or league data will load automatically on next app launch."
    )
    st.stop()

# Build team totals from long-format standings
all_team_totals: dict[str, dict[str, float]] = {}
if "category" in standings.columns:
    for _, row in standings.iterrows():
        team = str(row.get("team_name", ""))
        cat = str(row.get("category", "")).strip()
        val = float(row.get("total", 0))
        all_team_totals.setdefault(team, {})[cat] = val

if not all_team_totals:
    st.warning("Could not parse standings data.")
    st.stop()

# ── Get User Team ───────────────────────────────────────────────────────────

rosters = load_league_rosters()
user_team_name = None
if not rosters.empty:
    user_teams = rosters[rosters["is_user_team"] == 1]
    if not user_teams.empty:
        user_team_name = str(user_teams.iloc[0]["team_name"])

if user_team_name is None or user_team_name not in all_team_totals:
    # Try to let user pick their team
    team_names = sorted(all_team_totals.keys())
    user_team_name = st.selectbox("Select your team:", team_names)

user_totals = all_team_totals.get(user_team_name, {})

# ── Compute weeks remaining ─────────────────────────────────────────────────

now = datetime.now(UTC)
# MLB season typically ends ~Oct 1. Rough estimate: 26 total weeks, started ~April 1
season_end = datetime(now.year, 10, 1, tzinfo=UTC)
weeks_remaining = max(1, int((season_end - now).days / 7))

# ── Run Category Gap Analysis ───────────────────────────────────────────────

try:
    analysis = category_gap_analysis(
        user_totals,
        all_team_totals,
        user_team_name,
        weeks_remaining=weeks_remaining,
    )
    cat_weights = compute_category_weights_from_analysis(analysis)
except Exception as e:
    logger.exception("Category gap analysis failed")
    st.error(f"Analysis failed: {e}")
    st.stop()

# ── Display Category Grid ──────────────────────────────────────────────────

st.markdown(f"**Team:** {user_team_name} | **Weeks Remaining:** {weeks_remaining}")

rows = []
for cat in config.all_categories:
    info = analysis.get(cat, {})
    rank = info.get("rank", "?")
    total = user_totals.get(cat, 0)
    gap_to_next = info.get("gap_to_next", 0)
    gap_from_behind = info.get("gap_from_behind", 0)
    gainable = info.get("gainable_positions", 0)
    is_punt = info.get("is_punt", False)
    weight = cat_weights.get(cat, 1.0)

    # Classify priority
    if is_punt:
        priority = "PUNT"
    elif weight > 1.2:
        priority = "ATTACK"
    elif weight < 0.3:
        priority = "DEFEND"
    else:
        priority = "HOLD"

    # Format total
    if cat in config.rate_stats:
        total_str = f"{total:.3f}"
        gap_next_str = f"{gap_to_next:+.3f}" if gap_to_next != 0 else "—"
        gap_behind_str = f"{gap_from_behind:+.3f}" if gap_from_behind != 0 else "—"
    else:
        total_str = f"{total:.0f}"
        gap_next_str = f"{gap_to_next:+.0f}" if gap_to_next != 0 else "—"
        gap_behind_str = f"{gap_from_behind:+.0f}" if gap_from_behind != 0 else "—"

    rows.append(
        {
            "Category": cat,
            "Total": total_str,
            "Rank": f"{rank}/12",
            "Gap to Next": gap_next_str,
            "Gap from Behind": gap_behind_str,
            "Gainable Positions": gainable,
            "Priority": priority,
        }
    )

df = pd.DataFrame(rows)
render_styled_table(df)

st.caption(
    "Gap to Next = how much you need to gain one standings position. "
    "Gainable Positions = how many spots you can realistically move up. "
    "PUNT = cannot gain positions AND ranked 10th+. "
    "ATTACK = high marginal value. DEFEND = protect current rank."
)

# ── Punchable Categories ────────────────────────────────────────────────────

st.markdown("### Punchable Categories")
st.caption("Categories where gaining one standings position is within reach.")

punchable = []
for cat in config.all_categories:
    info = analysis.get(cat, {})
    gainable = info.get("gainable_positions", 0)
    gap = info.get("gap_to_next", 0)
    rank = info.get("rank", 12)
    is_punt = info.get("is_punt", False)

    if not is_punt and gainable > 0 and rank > 1:
        punchable.append(
            {
                "Category": cat,
                "Current Rank": f"{rank}/12",
                "Gap to Gain 1 Position": gap,
                "Positions Gainable": gainable,
            }
        )

if punchable:
    punch_df = pd.DataFrame(punchable)
    render_styled_table(punch_df)
else:
    st.info("No punchable categories found — you may be leading all non-punt categories.")

# ── What-If Tool ────────────────────────────────────────────────────────────

st.markdown("### What-If Calculator")
st.caption("Adjust a counting stat to see how your rank changes.")

counting_cats = [c for c in config.all_categories if c not in config.rate_stats]
selected_cat = st.selectbox("Category:", counting_cats, key="whatif_cat")

current_val = user_totals.get(selected_cat, 0)
# Compute ranks at different values
num_teams = len(all_team_totals)
is_inverse = selected_cat in config.inverse_stats

min_adj = -50 if not is_inverse else -5
max_adj = 100 if not is_inverse else 5
step = 1 if not is_inverse else 1

adjustment = st.slider(
    f"Adjust {selected_cat} by:",
    min_value=min_adj,
    max_value=max_adj,
    value=0,
    step=step,
    key="whatif_slider",
)

new_val = current_val + adjustment

# Compute new rank
other_vals = []
for team, tots in all_team_totals.items():
    if team != user_team_name:
        other_vals.append(tots.get(selected_cat, 0))

if is_inverse:
    new_rank = 1 + sum(1 for v in other_vals if v < new_val)
    old_rank = 1 + sum(1 for v in other_vals if v < current_val)
else:
    new_rank = 1 + sum(1 for v in other_vals if v > new_val)
    old_rank = 1 + sum(1 for v in other_vals if v > current_val)

rank_change = old_rank - new_rank  # positive = improved

wi1, wi2, wi3 = st.columns(3)
wi1.metric(f"Current {selected_cat}", f"{current_val:.0f}", help=f"Rank: {old_rank}/12")
wi2.metric(f"Projected {selected_cat}", f"{new_val:.0f}", delta=f"{adjustment:+.0f}")
wi3.metric(
    "Rank Change",
    f"{new_rank}/12",
    delta=f"{rank_change:+d} positions" if rank_change != 0 else "No change",
    delta_color="normal" if rank_change >= 0 else "inverse",
)

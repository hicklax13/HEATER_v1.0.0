"""Playoff Odds — Monte Carlo simulation of remaining H2H season standings."""

import time

import pandas as pd
import streamlit as st

from src.database import coerce_numeric_df, init_db, load_league_rosters, load_league_standings, load_player_pool
from src.ui_shared import T, inject_custom_css, render_styled_table
from src.valuation import LeagueConfig

try:
    from src.playoff_sim import estimate_weeks_remaining, simulate_season

    _HAS_PLAYOFF_SIM = True
except ImportError:
    _HAS_PLAYOFF_SIM = False


def _ordinal_suffix(n: int) -> str:
    """Return ordinal suffix for a number (1st, 2nd, 3rd, etc.)."""
    if 11 <= (n % 100) <= 13:
        return "th"
    last = n % 10
    if last == 1:
        return "st"
    if last == 2:
        return "nd"
    if last == 3:
        return "rd"
    return "th"


st.set_page_config(page_title="Heater | Playoff Odds", page_icon="", layout="wide")

init_db()

inject_custom_css()

st.markdown(
    '<div class="page-title-wrap"><div class="page-title"><span>PLAYOFF ODDS</span></div></div>',
    unsafe_allow_html=True,
)

if not _HAS_PLAYOFF_SIM:
    st.error("Playoff simulation module not available. Check that src/playoff_sim.py exists.")
    st.stop()

# ── Load data ────────────────────────────────────────────────────────

pool = load_player_pool()
if pool.empty:
    st.warning("No player data loaded. Load sample data or import projections first.")
    st.stop()

pool = pool.rename(columns={"name": "player_name"})
config = LeagueConfig()

# Load standings
standings = load_league_standings()
all_team_totals: dict[str, dict[str, float]] = {}
if not standings.empty and "category" in standings.columns:
    for _, srow in standings.iterrows():
        team = str(srow["team_name"])
        cat = str(srow["category"]).strip()
        all_team_totals.setdefault(team, {})[cat] = float(srow.get("total", 0) or 0)

# Load rosters
rosters = load_league_rosters()
if rosters.empty:
    if st.session_state.get("yahoo_connected"):
        st.warning("Yahoo is connected but no roster data found in the database. Try syncing:")
        if st.button("Sync League Data Now", key="sync_league_playoff"):
            client = st.session_state.get("yahoo_client")
            if client:
                progress = st.progress(0, text="Connecting to Yahoo Fantasy...")
                try:
                    progress.progress(30, text="Fetching league standings...")
                    sync_result = client.sync_to_db()
                    progress.progress(100, text="Sync complete!")
                    standings_count = sync_result.get("standings", 0) if sync_result else 0
                    rosters_count = sync_result.get("rosters", 0) if sync_result else 0
                    if rosters_count > 0:
                        st.success(f"Synced {rosters_count} roster entries and {standings_count} standing entries.")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.warning(
                            f"Sync completed but Yahoo returned no roster data "
                            f"(standings: {standings_count}). This may mean the league "
                            f"season hasn't started yet on Yahoo, or rosters haven't been set."
                        )
                except Exception as e:
                    progress.empty()
                    st.error(f"Sync failed: {e}")
            else:
                st.error("Yahoo client not found in session. Return to Connect League and reconnect.")
    else:
        st.warning(
            "No league data loaded. Connect your Yahoo league in Connect League, "
            "or league data will load automatically on next app launch."
        )
    st.stop()

user_teams = rosters[rosters["is_user_team"] == 1]
if user_teams.empty:
    st.warning("No user team identified in roster data.")
    st.stop()

user_team_name = str(user_teams.iloc[0]["team_name"])

# Build league_rosters dict: {team_name: [player_ids]}
rosters = coerce_numeric_df(rosters)
league_rosters_dict: dict[str, list[int]] = {}
for _, row in rosters.iterrows():
    team = str(row["team_name"])
    pid = int(row["player_id"])
    league_rosters_dict.setdefault(team, []).append(pid)

team_names = sorted(league_rosters_dict.keys())
n_teams = len(team_names)

if n_teams < 2:
    st.warning("Need at least 2 teams with roster data to run playoff simulations.")
    st.stop()

# ── Simulation controls ──────────────────────────────────────────────

weeks_default = estimate_weeks_remaining()

ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 2])
with ctrl1:
    weeks_remaining = st.number_input(
        "Weeks remaining",
        min_value=1,
        max_value=26,
        value=weeks_default,
        step=1,
    )
with ctrl2:
    n_sims = st.selectbox(
        "Simulations",
        options=[200, 500, 1000, 2000],
        index=1,
        format_func=lambda x: f"{x:,}",
    )

sim_button = st.button("Simulate Season", type="primary")

# ── Run simulation ───────────────────────────────────────────────────

cache_key = "playoff_sim_results"

if sim_button:
    progress_bar = st.progress(0, text="Running playoff simulations...")

    def _update_progress(frac: float) -> None:
        pct = int(frac * 100)
        progress_bar.progress(frac, text=f"Simulating season... {pct}%")

    start_time = time.time()
    try:
        results = simulate_season(
            all_team_totals=all_team_totals,
            league_rosters=league_rosters_dict,
            player_pool=pool,
            weeks_remaining=int(weeks_remaining),
            n_sims=int(n_sims),
            config=config,
            on_progress=_update_progress,
        )
        elapsed = time.time() - start_time
        progress_bar.progress(1.0, text=f"Complete in {elapsed:.1f}s")
        time.sleep(0.3)
        progress_bar.empty()
        st.session_state[cache_key] = results
    except Exception as e:
        progress_bar.empty()
        st.error(f"Simulation failed: {e}")

# ── Display results ──────────────────────────────────────────────────

results = st.session_state.get(cache_key)

if results is None:
    st.info("Press Simulate Season to run the Monte Carlo playoff odds simulator.")
    st.stop()

if not results:
    st.warning("Simulation returned no results. Check that roster data is available for all teams.")
    st.stop()

# ── Your Playoff Probability ─────────────────────────────────────────

user_result = results.get(user_team_name)
if user_result:
    prob = user_result["playoff_prob"]
    prob_pct = prob * 100.0

    if prob_pct >= 60:
        prob_color = T["green"]
        prob_label = "Strong"
    elif prob_pct >= 40:
        prob_color = T["hot"]
        prob_label = "Moderate"
    else:
        prob_color = T["primary"]
        prob_label = "At Risk"

    st.markdown(
        f'<div style="background:#ffffff;border-radius:14px;padding:24px 32px;margin-bottom:24px;'
        f'box-shadow:0 2px 12px rgba(0,0,0,0.06);display:flex;align-items:center;gap:28px;">'
        f"<div>"
        f'<div style="font-family:Figtree,sans-serif;font-size:14px;font-weight:600;'
        f'color:#666;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">'
        f"Your Playoff Probability</div>"
        f'<div style="font-family:Bebas Neue,sans-serif;font-size:56px;font-weight:700;'
        f'color:{prob_color};line-height:1;">{prob_pct:.1f}%</div>'
        f'<div style="font-family:Figtree,sans-serif;font-size:14px;color:#888;margin-top:4px;">'
        f"{prob_label} -- {user_team_name}</div>"
        f"</div>"
        f'<div style="margin-left:auto;text-align:right;">'
        f'<div style="font-size:13px;color:#888;">Projected Record</div>'
        f'<div style="font-family:Bebas Neue,sans-serif;font-size:28px;color:#1d1d1f;">'
        f"{user_result['avg_wins']:.1f} - {user_result['avg_losses']:.1f}</div>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

# ── Projected Standings Table ─────────────────────────────────────────

st.markdown(
    '<div style="font-family:Figtree,sans-serif;font-size:18px;font-weight:700;'
    'color:#1d1d1f;margin:20px 0 10px 0;">Projected Standings</div>',
    unsafe_allow_html=True,
)

# Build table rows sorted by playoff probability
rows = []
for team in team_names:
    r = results.get(team)
    if r is None:
        continue
    rank_dist = r["rank_distribution"]
    # Most likely finish = rank with highest count
    most_likely_rank = rank_dist.index(max(rank_dist)) + 1
    rows.append(
        {
            "Team": team,
            "Avg Wins": r["avg_wins"],
            "Avg Losses": r["avg_losses"],
            "Playoff %": round(r["playoff_prob"] * 100, 1),
            "Most Likely Finish": most_likely_rank,
        }
    )

standings_df = pd.DataFrame(rows)
standings_df = standings_df.sort_values("Playoff %", ascending=False).reset_index(drop=True)
standings_df.index = standings_df.index + 1  # 1-based ranking
standings_df.index.name = "Rank"


# Highlight user's team row
def _highlight_user(row: pd.Series) -> list[str]:
    if row["Team"] == user_team_name:
        return ["background-color: rgba(230, 92, 0, 0.08); font-weight: 700;"] * len(row)
    return [""] * len(row)


styled = standings_df.style.apply(_highlight_user, axis=1).format(
    {"Playoff %": "{:.1f}%", "Avg Wins": "{:.1f}", "Avg Losses": "{:.1f}"}
)
st.dataframe(styled, use_container_width=True, height=min(460, 38 + 35 * len(standings_df)))

# ── Rank Distribution for User Team ──────────────────────────────────

if user_result:
    st.markdown(
        '<div style="font-family:Figtree,sans-serif;font-size:18px;font-weight:700;'
        'color:#1d1d1f;margin:28px 0 10px 0;">Your Finish Distribution</div>',
        unsafe_allow_html=True,
    )

    rank_dist = user_result["rank_distribution"]
    total_sims = sum(rank_dist) or 1

    # Build a compact table
    dist_rows = []
    for idx, count in enumerate(rank_dist):
        rank = idx + 1
        pct = count / total_sims * 100
        if pct < 0.5:
            continue  # Skip negligible probabilities
        suffix = _ordinal_suffix(rank)
        dist_rows.append(
            {
                "Finish": f"{rank}{suffix}",
                "Probability": f"{pct:.1f}%",
                "Simulations": count,
            }
        )

    if dist_rows:
        dist_df = pd.DataFrame(dist_rows)
        render_styled_table(dist_df)

# ── Category Improvement Suggestions ─────────────────────────────────

if all_team_totals and user_team_name in all_team_totals:
    st.markdown(
        '<div style="font-family:Figtree,sans-serif;font-size:18px;font-weight:700;'
        'color:#1d1d1f;margin:28px 0 10px 0;">What Do I Need?</div>',
        unsafe_allow_html=True,
    )

    user_totals = all_team_totals[user_team_name]
    inverse = config.inverse_stats

    # Rank user in each category
    weak_cats: list[dict] = []
    for cat in config.all_categories:
        user_val = user_totals.get(cat, 0.0)
        all_vals = [t.get(cat, 0.0) for t in all_team_totals.values()]

        if cat in inverse:
            # Lower is better — rank ascending
            rank = sum(1 for v in all_vals if v < user_val) + 1
        else:
            rank = sum(1 for v in all_vals if v > user_val) + 1

        if rank >= 7:  # Bottom half
            weak_cats.append({"Category": cat, "Rank": rank})

    if weak_cats:
        # Sort by worst rank first
        weak_cats.sort(key=lambda x: x["Rank"], reverse=True)
        suggestion_rows = []
        for wc in weak_cats:
            sev = wc["Rank"]
            if sev >= 10:
                priority = "High"
            elif sev >= 8:
                priority = "Medium"
            else:
                priority = "Low"
            suffix = _ordinal_suffix(sev)
            suggestion_rows.append(
                {
                    "Category": wc["Category"],
                    "Current Rank": f"{sev}{suffix}",
                    "Priority": priority,
                }
            )
        suggestion_df = pd.DataFrame(suggestion_rows)
        render_styled_table(suggestion_df)

        st.markdown(
            '<div style="font-family:Figtree,sans-serif;font-size:13px;color:#888;margin-top:8px;">'
            "Target trades and waiver pickups in your weakest categories to maximize playoff odds."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="font-family:Figtree,sans-serif;font-size:14px;color:#2d6a4f;'
            'background:#f0fdf4;border-radius:8px;padding:12px 16px;margin-top:4px;">'
            "Your team ranks in the top half of the league across all categories. "
            "Focus on maintaining depth and streaming pitchers."
            "</div>",
            unsafe_allow_html=True,
        )
elif not all_team_totals:
    st.info("Connect your Yahoo league or load standings data to see category improvement suggestions.")

"""Lineup Optimizer — Enhanced multi-mode optimization with H2H analysis.

Wires the LineupOptimizerPipeline (11 optimizer modules) into a 5-tab UI:
  1. Optimize: mode selector, alpha slider, LP solve, recommendations
  2. H2H Matchup: per-category win probabilities against this week's opponent
  3. Streaming: top pitcher streaming candidates, two-start SP analysis
  4. Category Analysis: non-linear SGP weights, maximin comparison
  5. Roster: current roster with health badges and two-start flags
"""

import time

import pandas as pd
import streamlit as st

from src.database import get_connection, init_db, load_league_rosters, load_league_standings
from src.injury_model import compute_health_score, get_injury_badge
from src.league_manager import get_team_roster
from src.ui_shared import METRIC_TOOLTIPS, PAGE_ICONS, inject_custom_css, render_styled_table

# ── Pipeline import (primary) ─────────────────────────────────────
try:
    from src.optimizer.pipeline import LineupOptimizerPipeline

    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

# ── Fallback: basic LP optimizer ──────────────────────────────────
try:
    from src.lineup_optimizer import LineupOptimizer

    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False

# ── Optional module imports for UI features ───────────────────────
try:
    from src.optimizer.h2h_engine import (
        compute_h2h_category_weights,
        estimate_h2h_win_probability,
    )

    H2H_AVAILABLE = True
except ImportError:
    H2H_AVAILABLE = False

try:
    from src.optimizer.sgp_theory import compute_nonlinear_weights

    SGP_AVAILABLE = True
except ImportError:
    SGP_AVAILABLE = False

try:
    from src.optimizer.matchup_adjustments import get_weekly_schedule

    MATCHUP_AVAILABLE = True
except ImportError:
    MATCHUP_AVAILABLE = False

try:
    from src.data_bootstrap import PARK_FACTORS

    PARK_FACTORS_AVAILABLE = True
except ImportError:
    PARK_FACTORS = {}
    PARK_FACTORS_AVAILABLE = False


# ── Page Config ───────────────────────────────────────────────────

st.set_page_config(page_title="Heater | Lineup Optimizer", page_icon="", layout="wide")

init_db()

inject_custom_css()

st.markdown(
    '<div class="page-title-wrap"><div class="page-title"><span>LINEUP OPTIMIZER</span></div></div>',
    unsafe_allow_html=True,
)


# ── Category display names ────────────────────────────────────────

CAT_DISPLAY_NAMES: dict[str, str] = {
    "r": "Runs",
    "hr": "Home Runs",
    "rbi": "Runs Batted In",
    "sb": "Stolen Bases",
    "avg": "Batting Average",
    "w": "Wins",
    "sv": "Saves",
    "k": "Strikeouts",
    "era": "Earned Run Average",
    "whip": "Walks + Hits per Inning Pitched",
}

ALL_CATS = list(CAT_DISPLAY_NAMES.keys())


# ── Load user team ────────────────────────────────────────────────

rosters = load_league_rosters()
if rosters.empty:
    if st.session_state.get("yahoo_connected"):
        st.warning("Yahoo is connected but no roster data found in the database. Try syncing:")
        if st.button("Sync League Data Now", key="sync_league_lineup"):
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

user_team_name = user_teams.iloc[0]["team_name"]
# Strip emoji for display (Yahoo team names may contain emoji)
_display_team_name = "".join(c for c in user_team_name if ord(c) < 0x10000).strip()
st.markdown(f"**Team:** {_display_team_name}")

roster = get_team_roster(user_team_name)
if roster is None or roster.empty:
    st.info("Your roster is empty. Import roster data first.")
    st.stop()


# ── Normalize columns ─────────────────────────────────────────────

if "name" in roster.columns and "player_name" not in roster.columns:
    roster = roster.rename(columns={"name": "player_name"})

required_cols = ["player_id", "player_name", "positions", "is_hitter"]
stat_cols = ["r", "hr", "rbi", "sb", "avg", "w", "sv", "k", "era", "whip"]
component_cols = ["ip", "h", "ab", "er", "bb_allowed", "h_allowed"]
for col in required_cols + stat_cols + component_cols:
    if col not in roster.columns:
        roster[col] = 0


# ── League config for optimizer ───────────────────────────────────

config = {
    "hitting_categories": ["r", "hr", "rbi", "sb", "avg"],
    "pitching_categories": ["w", "sv", "k", "era", "whip"],
    "roster_slots": {
        "C": 1,
        "1B": 1,
        "2B": 1,
        "3B": 1,
        "SS": 1,
        "OF": 3,
        "Util": 2,
        "SP": 2,
        "RP": 2,
        "P": 4,
    },
}

if not OPTIMIZER_AVAILABLE and not PIPELINE_AVAILABLE:
    st.error("Lineup Optimizer requires PuLP. Install it with: `pip install pulp`")
    st.stop()


# ── Apply health-based stat discount BEFORE optimization ──────────

HEALTH_PENALTY_WEIGHT = 0.15
health_dict: dict[int, float] = {}
try:
    conn = get_connection()
    try:
        injury_df = pd.read_sql_query("SELECT * FROM injury_history", conn)
    finally:
        conn.close()

    if not injury_df.empty:
        counting_cols = ["r", "hr", "rbi", "sb", "w", "sv", "k"]
        # Cast int64 columns to float64 before applying fractional penalties
        for col in counting_cols:
            if col in roster.columns:
                roster[col] = pd.to_numeric(roster[col], errors="coerce").astype(float)
        for idx, row in roster.iterrows():
            pid = row.get("player_id")
            pi = injury_df[injury_df["player_id"] == pid]
            if not pi.empty:
                hs = compute_health_score(pi["games_played"].tolist(), pi["games_available"].tolist())
            else:
                hs = 1.0
            health_dict[pid] = hs
            penalty = HEALTH_PENALTY_WEIGHT * (1.0 - hs)
            if penalty > 0:
                for col in counting_cols:
                    if col in roster.columns:
                        roster.at[idx, col] = float(roster.at[idx, col]) * (1.0 - penalty)
except Exception:
    pass  # Graceful degradation when injury data unavailable


# ── Load standings and schedule ────────────────────────────────────

standings = load_league_standings()

# Build team totals from standings for H2H and SGP.
# league_standings is LONG format: one row per (team_name, category) with a
# ``total`` column, NOT wide format with category names as column headers.
team_totals: dict[str, dict[str, float]] = {}
my_totals: dict[str, float] = {}
if not standings.empty and "category" in standings.columns:
    for _, srow in standings.iterrows():
        tname = srow.get("team_name", "")
        cat = str(srow.get("category", "")).strip()
        raw_total = srow.get("total", 0)
        if not tname or not cat:
            continue
        try:
            val = float(raw_total) if raw_total is not None else 0.0
        except (ValueError, TypeError):
            val = 0.0
        team_totals.setdefault(tname, {})[cat] = val
        if tname == user_team_name:
            my_totals[cat] = val

# Fetch weekly schedule (cached in session state to avoid repeated API calls)
week_schedule = []
if MATCHUP_AVAILABLE:
    if "optimizer_week_schedule" not in st.session_state:
        try:
            st.session_state["optimizer_week_schedule"] = get_weekly_schedule(days_ahead=7)
        except Exception:
            st.session_state["optimizer_week_schedule"] = []
    week_schedule = st.session_state.get("optimizer_week_schedule", [])


# ── Settings controls ──────────────────────────────────────────────

with st.expander("Optimization Settings", expanded=False):
    settings_c1, settings_c2, settings_c3, settings_c4 = st.columns(4)

    with settings_c1:
        mode = st.radio(
            "Optimization Mode",
            options=["quick", "standard", "full"],
            index=1,
            format_func=lambda m: {
                "quick": "Quick (<1s)",
                "standard": "Standard (2-3s)",
                "full": "Full (5-10s)",
            }[m],
            help=(
                "Quick: Enhanced projections + mean-variance only. "
                "Standard: 200 scenarios + CVaR + matchup adjustments. "
                "Full: Stochastic MIP + maximin + streaming + multi-period."
            ),
        )

    with settings_c2:
        alpha = st.slider(
            "H2H vs Roto Balance",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="0.0 = pure Roto optimization, 1.0 = pure H2H weekly focus, 0.5 = balanced.",
        )

    with settings_c3:
        weeks_remaining = st.number_input(
            "Weeks Remaining",
            min_value=1,
            max_value=26,
            value=16,
            help="Number of weeks left in the fantasy season. Affects urgency calculations.",
        )

    with settings_c4:
        risk_aversion = st.slider(
            "Risk Aversion",
            min_value=0.0,
            max_value=0.50,
            value=0.15,
            step=0.05,
            help="Higher = prefer consistent players over boom/bust. 0 = risk neutral.",
        )


# ── Build opponent list for H2H ────────────────────────────────────

opponent_names = [t for t in team_totals if t != user_team_name]
selected_opponent = None
opp_totals: dict[str, float] = {}

if opponent_names:
    # Place opponent selector above tabs for use across multiple tabs
    selected_opponent = st.selectbox(
        "This Week's H2H Opponent",
        options=["(None - Roto Only)"] + sorted(opponent_names),
        index=0,
        help="Select your H2H opponent to enable matchup-specific optimization.",
    )
    if selected_opponent and selected_opponent != "(None - Roto Only)":
        opp_totals = team_totals.get(selected_opponent, {})
    else:
        selected_opponent = None


# ── Tabs ──────────────────────────────────────────────────────────

tab_optimize, tab_h2h, tab_streaming, tab_analysis, tab_roster = st.tabs(
    ["Optimize", "H2H Matchup", "Streaming", "Category Analysis", "Roster"]
)


# ════════════════════════════════════════════════════════════════════
# TAB 1: OPTIMIZE
# ════════════════════════════════════════════════════════════════════

with tab_optimize:
    optimize_clicked = st.button("Optimize Lineup", type="primary")

    if optimize_clicked:
        progress_bar = st.progress(0, text="Initializing optimizer pipeline...")

        if PIPELINE_AVAILABLE:
            # Use the full pipeline
            progress_bar.progress(10, text=f"Running {mode} optimization pipeline...")

            pipeline = LineupOptimizerPipeline(
                roster,
                mode=mode,
                alpha=alpha,
                weeks_remaining=weeks_remaining,
                config=config,
            )

            # Override risk aversion from settings
            pipeline._preset = dict(pipeline._preset)
            pipeline._preset["risk_aversion"] = risk_aversion

            progress_bar.progress(30, text="Computing category weights and projections...")

            result = pipeline.optimize(
                standings=standings if not standings.empty else None,
                team_name=user_team_name,
                h2h_opponent_totals=opp_totals if opp_totals else None,
                my_totals=my_totals if my_totals else None,
                week_schedule=week_schedule if week_schedule else None,
                park_factors=PARK_FACTORS if PARK_FACTORS else None,
            )

            progress_bar.progress(100, text="Optimization complete!")
            time.sleep(0.3)
            progress_bar.empty()

            # Store result in session state for other tabs
            st.session_state["optimizer_result"] = result

        else:
            # Fallback to basic optimizer
            progress_bar.progress(20, text="Building constraint matrix...")
            optimizer = LineupOptimizer(roster, config)
            basic_result = optimizer.optimize_lineup()
            progress_bar.progress(100, text="Optimization complete!")
            time.sleep(0.3)
            progress_bar.empty()

            result = {
                "lineup": basic_result,
                "category_weights": {},
                "h2h_analysis": None,
                "streaming_suggestions": None,
                "risk_metrics": None,
                "maximin_comparison": None,
                "recommendations": [],
                "timing": {"total": 0},
                "mode": "basic",
                "matchup_adjusted": False,
            }
            st.session_state["optimizer_result"] = result

    # ── Display results from session state ────────────────────────
    result = st.session_state.get("optimizer_result")

    if result:
        lineup = result.get("lineup", {})

        if not lineup or not lineup.get("assignments"):
            st.warning("Could not produce a valid lineup. Check your roster data.")
        else:
            # Success banner
            mode_label = result.get("mode", "standard").title()
            timing_total = result.get("timing", {}).get("total", 0)
            matchup_flag = " | Matchup-adjusted" if result.get("matchup_adjusted") else ""
            st.success(f"Optimal lineup found ({mode_label} mode, {timing_total:.2f}s{matchup_flag})")

            # Recommended lineup table
            st.subheader("Recommended Lineup")
            assignments = lineup["assignments"]
            lineup_data = []
            for entry in assignments:
                slot = entry.get("slot", "")
                player = entry.get("player_name", "")
                pid = entry.get("player_id")
                hs = health_dict.get(pid, 1.0) if pid else 1.0
                badge, label = get_injury_badge(hs)
                lineup_data.append(
                    {
                        "Slot": slot,
                        "Player": player,
                        "Health": f"{badge} {label}",
                    }
                )
            render_styled_table(pd.DataFrame(lineup_data))

            # Projected stats
            if lineup.get("projected_stats"):
                st.subheader("Projected Category Totals")
                proj = lineup["projected_stats"]
                # Format for display
                display_proj = {}
                for cat in ALL_CATS:
                    if cat in proj:
                        val = proj[cat]
                        if cat in ("avg",):
                            display_proj[CAT_DISPLAY_NAMES[cat]] = f"{val:.3f}"
                        elif cat in ("era",):
                            display_proj[CAT_DISPLAY_NAMES[cat]] = f"{val:.2f}"
                        elif cat in ("whip",):
                            display_proj[CAT_DISPLAY_NAMES[cat]] = f"{val:.3f}"
                        else:
                            display_proj[CAT_DISPLAY_NAMES[cat]] = f"{val:.0f}"
                render_styled_table(pd.DataFrame([display_proj]))

            # Risk metrics
            risk_metrics = result.get("risk_metrics")
            if risk_metrics:
                st.subheader("Risk Analysis")
                rc1, rc2, rc3, rc4 = st.columns(4)
                with rc1:
                    st.metric("Expected Value", f"{risk_metrics.get('mean', 0):.3f}")
                with rc2:
                    st.metric("Std Deviation", f"{risk_metrics.get('std', 0):.3f}")
                with rc3:
                    st.metric("VaR (5th %ile)", f"{risk_metrics.get('var_5', 0):.3f}")
                with rc4:
                    st.metric("CVaR (Tail Risk)", f"{risk_metrics.get('cvar_5', 0):.3f}")
                st.caption(
                    "Value at Risk (VaR) is the 5th percentile outcome. "
                    "CVaR is the average of the worst 5% of scenarios — your downside floor."
                )

            # Recommendations
            recs = result.get("recommendations", [])
            if recs:
                st.subheader("Recommendations")
                for rec in recs:
                    st.markdown(f"{PAGE_ICONS['trending_up']} {rec}", unsafe_allow_html=True)

            # Bench players
            bench = lineup.get("bench", [])
            if bench:
                st.subheader("Bench")
                # Build name->positions lookup from roster for bench display
                name_col = "player_name" if "player_name" in roster.columns else "name"
                _pos_lookup = dict(zip(roster[name_col].astype(str), roster["positions"].astype(str)))
                bench_data = []
                for entry in bench:
                    if isinstance(entry, dict):
                        pname = entry.get("player_name", "")
                        bench_data.append(
                            {
                                "Player": pname,
                                "Position": entry.get("positions", _pos_lookup.get(pname, "")),
                            }
                        )
                    else:
                        pname = str(entry)
                        bench_data.append({"Player": pname, "Position": _pos_lookup.get(pname, "")})
                if bench_data:
                    render_styled_table(pd.DataFrame(bench_data))

            # Timing breakdown
            timing = result.get("timing", {})
            if len(timing) > 1:
                with st.expander("Performance Breakdown"):
                    timing_data = [
                        {"Stage": k.replace("_", " ").title(), "Time (s)": f"{v:.3f}"} for k, v in timing.items()
                    ]
                    render_styled_table(pd.DataFrame(timing_data))
    else:
        st.info("Click **Optimize Lineup** to generate your optimal start/sit decisions.")


# ════════════════════════════════════════════════════════════════════
# TAB 2: H2H MATCHUP
# ════════════════════════════════════════════════════════════════════

with tab_h2h:
    if not H2H_AVAILABLE:
        st.info("H2H analysis module not available. Ensure scipy is installed.")
    elif not selected_opponent or not opp_totals:
        if not opponent_names:
            st.info(
                "League standings required for H2H analysis. "
                "Connect your Yahoo league in Connect League to import standings data."
            )
        else:
            st.info(
                "Select a H2H opponent above to see per-category win probabilities "
                "and matchup-specific strategy recommendations."
            )
    elif not my_totals:
        st.info("League standings data required for H2H analysis. Sync your Yahoo league to get standings.")
    else:
        _opp_display = "".join(c for c in selected_opponent if ord(c) < 0x10000).strip()
        st.subheader(f"Matchup Analysis: {_display_team_name} vs {_opp_display}")

        # Compute win probabilities
        h2h_result = estimate_h2h_win_probability(my_totals, opp_totals)
        per_cat = h2h_result.get("per_category", {})
        exp_wins = h2h_result.get("expected_wins", 5.0)
        overall_prob = h2h_result.get("overall_win_prob", 0.5)

        # Overall metrics
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.metric("Win Probability", f"{overall_prob:.1%}")
        with mc2:
            st.metric("Expected Category Wins", f"{exp_wins:.1f} / 10")
        with mc3:
            status = "Favored" if exp_wins > 5.5 else "Underdog" if exp_wins < 4.5 else "Toss-up"
            st.metric("Matchup Status", status)

        # Per-category breakdown table
        st.subheader("Category-by-Category Breakdown")
        cat_rows = []
        for cat in ALL_CATS:
            p_win = per_cat.get(cat, 0.5)
            my_val = my_totals.get(cat, 0)
            opp_val = opp_totals.get(cat, 0)

            if cat in ("avg",):
                my_fmt = f"{my_val:.3f}"
                opp_fmt = f"{opp_val:.3f}"
            elif cat in ("era",):
                my_fmt = f"{my_val:.2f}"
                opp_fmt = f"{opp_val:.2f}"
            elif cat in ("whip",):
                my_fmt = f"{my_val:.3f}"
                opp_fmt = f"{opp_val:.3f}"
            else:
                my_fmt = f"{my_val:.0f}"
                opp_fmt = f"{opp_val:.0f}"

            if p_win >= 0.65:
                verdict = "Likely Win"
            elif p_win >= 0.50:
                verdict = "Lean Win"
            elif p_win >= 0.35:
                verdict = "Lean Loss"
            else:
                verdict = "Likely Loss"

            bar_len = int(p_win * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)

            cat_rows.append(
                {
                    "Category": CAT_DISPLAY_NAMES.get(cat, cat.upper()),
                    "You": my_fmt,
                    "Opponent": opp_fmt,
                    "Win %": f"{p_win:.1%}",
                    "Confidence": bar,
                    "Verdict": verdict,
                }
            )

        render_styled_table(pd.DataFrame(cat_rows))

        # H2H category weights (what to focus on)
        st.subheader("Optimal Focus Areas")
        h2h_weights = compute_h2h_category_weights(my_totals, opp_totals)
        if h2h_weights:
            focus_sorted = sorted(h2h_weights.items(), key=lambda x: x[1], reverse=True)
            for cat, weight in focus_sorted[:5]:
                icon = (
                    PAGE_ICONS["fire"]
                    if weight > 1.5
                    else PAGE_ICONS["trending_up"]
                    if weight > 1.0
                    else PAGE_ICONS["minus"]
                )
                display_name = CAT_DISPLAY_NAMES.get(cat, cat.upper())
                bar_len = int(min(weight, 3.0) / 3.0 * 20)
                st.markdown(
                    f"{icon} **{display_name}**: {'█' * bar_len}{'░' * (20 - bar_len)} ({weight:.2f}x weight)",
                    unsafe_allow_html=True,
                )
            st.caption(
                "Categories near the toss-up point get higher weight — "
                "that's where a small improvement has the biggest impact on winning."
            )


# ════════════════════════════════════════════════════════════════════
# TAB 3: STREAMING
# ════════════════════════════════════════════════════════════════════

with tab_streaming:
    st.subheader("Pitcher Streaming Candidates")

    # Check for pipeline streaming results first
    result = st.session_state.get("optimizer_result")
    streaming = result.get("streaming_suggestions") if result else None

    if streaming:
        stream_data = []
        for s in streaming:
            stream_data.append(
                {
                    "Pitcher": s.get("player_name", "Unknown"),
                    "Team": s.get("team", ""),
                    "Net Value (SGP)": f"{s.get('net_value', 0):.3f}",
                    "Counting SGP": f"{s.get('counting_sgp', 0):.3f}",
                    "Rate Impact": f"{s.get('rate_damage', 0):.3f}",
                    "Games": s.get("n_games", 0),
                }
            )
        render_styled_table(pd.DataFrame(stream_data))
        st.caption(
            "Net Value = Counting stat SGP gains minus ERA/WHIP rate damage. "
            "Pick up pitchers with positive net value for streaming starts."
        )
    else:
        st.info(
            "Run optimization in Full mode to see streaming recommendations. Requires free agent data from Yahoo sync."
        )

    # Two-start SP detection
    st.divider()
    st.subheader("Two-Start Starting Pitchers")

    two_start_sps = []
    try:
        from datetime import UTC, datetime, timedelta

        import statsapi

        _MLB_TEAM_ABBREVS: dict[str, str] = {
            "Arizona Diamondbacks": "ARI",
            "Atlanta Braves": "ATL",
            "Baltimore Orioles": "BAL",
            "Boston Red Sox": "BOS",
            "Chicago Cubs": "CHC",
            "Chicago White Sox": "CWS",
            "Cincinnati Reds": "CIN",
            "Cleveland Guardians": "CLE",
            "Colorado Rockies": "COL",
            "Detroit Tigers": "DET",
            "Houston Astros": "HOU",
            "Kansas City Royals": "KC",
            "Los Angeles Angels": "LAA",
            "Los Angeles Dodgers": "LAD",
            "Miami Marlins": "MIA",
            "Milwaukee Brewers": "MIL",
            "Minnesota Twins": "MIN",
            "New York Mets": "NYM",
            "New York Yankees": "NYY",
            "Oakland Athletics": "OAK",
            "Philadelphia Phillies": "PHI",
            "Pittsburgh Pirates": "PIT",
            "San Diego Padres": "SD",
            "San Francisco Giants": "SF",
            "Seattle Mariners": "SEA",
            "St. Louis Cardinals": "STL",
            "Tampa Bay Rays": "TB",
            "Texas Rangers": "TEX",
            "Toronto Blue Jays": "TOR",
            "Washington Nationals": "WSH",
        }

        today = datetime.now(UTC)
        end = today + timedelta(days=7)
        schedule = statsapi.schedule(
            start_date=today.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
        )

        team_game_counts: dict[str, int] = {}
        for game in schedule:
            for team_key in ["home_name", "away_name"]:
                full_name = game.get(team_key, "")
                abbrev = _MLB_TEAM_ABBREVS.get(full_name, "")
                if abbrev:
                    team_game_counts[abbrev] = team_game_counts.get(abbrev, 0) + 1

        TWO_START_THRESHOLD = 7
        for _, row in roster.iterrows():
            if not row.get("is_hitter", True) and "SP" in str(row.get("positions", "")):
                team = str(row.get("team", "")).strip()
                games = team_game_counts.get(team, 0)
                if games >= TWO_START_THRESHOLD:
                    two_start_sps.append(
                        {
                            "Pitcher": row.get("player_name", row.get("name", "")),
                            "Team": team,
                            "Team Games": games,
                        }
                    )

        if two_start_sps:
            st.markdown(
                f"{PAGE_ICONS['trending_up']} **{len(two_start_sps)} potential two-start pitcher(s) this week:**",
                unsafe_allow_html=True,
            )
            render_styled_table(pd.DataFrame(two_start_sps))
            st.caption(
                "Starting pitchers on teams with 7+ games this week likely get two starts. "
                "Two starts double counting stat contributions (K, W) but also double rate-stat exposure (ERA, WHIP)."
            )
        else:
            st.info("No two-start starting pitchers detected for your roster this week.")

    except Exception:
        st.info("MLB schedule data unavailable. Install statsapi for two-start detection.")


# ════════════════════════════════════════════════════════════════════
# TAB 4: CATEGORY ANALYSIS
# ════════════════════════════════════════════════════════════════════

with tab_analysis:
    # Category weights from pipeline result
    result = st.session_state.get("optimizer_result")
    weights = result.get("category_weights", {}) if result else {}

    if not weights and not standings.empty:
        # Compute on the fly if no result yet
        if SGP_AVAILABLE:
            try:
                weights = compute_nonlinear_weights(standings, user_team_name)
            except Exception:
                pass

    if weights:
        st.subheader("Category Weight Distribution")
        st.markdown("**Priority categories** (higher weight = bigger marginal standings impact):")
        weights_sorted = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        for cat, weight in weights_sorted:
            bar_len = int(min(weight, 3.0) / 3.0 * 20)
            icon = (
                PAGE_ICONS["fire"]
                if weight > 1.5
                else PAGE_ICONS["trending_up"]
                if weight > 1.0
                else PAGE_ICONS["minus"]
            )
            display_name = CAT_DISPLAY_NAMES.get(cat, cat.upper())
            st.markdown(
                f"{icon} **{display_name}**: {'█' * bar_len}{'░' * (20 - bar_len)} ({weight:.2f}x)",
                unsafe_allow_html=True,
            )
        st.caption(METRIC_TOOLTIPS.get("cat_targeting", ""))
    elif standings.empty:
        st.info(
            "Import league standings to see category targeting recommendations. "
            "The optimizer will identify where small gains yield the biggest standings jumps."
        )
    else:
        st.info("Category weights will appear after running the optimizer.")

    # Maximin comparison
    maximin = result.get("maximin_comparison") if result else None
    if maximin:
        st.divider()
        st.subheader("Maximin (Balanced) Lineup Comparison")
        z_val = maximin.get("z_value", 0)
        st.metric("Worst-Category Floor (z-score)", f"{z_val:.3f}")

        maximin_assignments = maximin.get("assignments", {})
        if maximin_assignments:
            # maximin_lineup() returns {player_name: 0/1} — show only starters
            maximin_data = [{"Player": name} for name, val in maximin_assignments.items() if val == 1]
            if maximin_data:
                render_styled_table(pd.DataFrame(maximin_data))
        st.caption(
            "The maximin lineup maximizes the WORST category instead of the total. "
            "Compare with your main lineup to see if you're sacrificing too much balance."
        )

    # Standings position for each category
    if not standings.empty and my_totals:
        st.divider()
        st.subheader("Current Standings Position by Category")
        position_rows = []
        for cat in ALL_CATS:
            my_val = my_totals.get(cat, 0)
            # Count how many teams are ahead
            is_inverse = cat in ("era", "whip")
            rank = 1
            for tname, ttotals in team_totals.items():
                if tname == user_team_name:
                    continue
                opp_val = ttotals.get(cat, 0)
                if is_inverse:
                    if opp_val < my_val:  # Lower is better
                        rank += 1
                else:
                    if opp_val > my_val:
                        rank += 1

            if cat in ("avg",):
                val_fmt = f"{my_val:.3f}"
            elif cat in ("era",):
                val_fmt = f"{my_val:.2f}"
            elif cat in ("whip",):
                val_fmt = f"{my_val:.3f}"
            else:
                val_fmt = f"{my_val:.0f}"

            rank_label = f"{rank}" + ("st" if rank == 1 else "nd" if rank == 2 else "rd" if rank == 3 else "th")

            position_rows.append(
                {
                    "Category": CAT_DISPLAY_NAMES.get(cat, cat.upper()),
                    "Your Total": val_fmt,
                    "Rank": rank_label,
                    "Points": str(13 - rank),
                }
            )
        render_styled_table(pd.DataFrame(position_rows))


# ════════════════════════════════════════════════════════════════════
# TAB 5: ROSTER
# ════════════════════════════════════════════════════════════════════

with tab_roster:
    st.subheader("Current Roster")

    # Add health badges
    try:
        health_labels = []
        for _, row in roster.iterrows():
            pid = row.get("player_id")
            hs = health_dict.get(pid, 1.0)
            _, label = get_injury_badge(hs)
            health_labels.append(label)
        roster_display = roster.copy()
        roster_display["Health"] = health_labels
    except Exception:
        roster_display = roster.copy()

    if health_dict:
        st.caption("Health-adjusted projections: counting stats reduced by up to 15% based on injury history risk.")

    # Build display columns — exclude is_hitter (internal) and use display names
    display_cols = ["player_name", "positions", "Health"] + [c for c in stat_cols if c in roster_display.columns]
    display_cols = [c for c in display_cols if c in roster_display.columns]

    roster_show = roster_display[display_cols].copy()

    # Format numeric columns before renaming
    counting = ["r", "hr", "rbi", "sb", "w", "sv", "k"]
    for col in counting:
        if col in roster_show.columns:
            roster_show[col] = pd.to_numeric(roster_show[col], errors="coerce").round(0).astype("Int64")
    for col in ["avg"]:
        if col in roster_show.columns:
            roster_show[col] = roster_show[col].apply(lambda v: f"{float(v):.3f}" if pd.notna(v) and v != 0 else "")
    for col in ["era"]:
        if col in roster_show.columns:
            roster_show[col] = roster_show[col].apply(lambda v: f"{float(v):.2f}" if pd.notna(v) and v != 0 else "")
    for col in ["whip"]:
        if col in roster_show.columns:
            roster_show[col] = roster_show[col].apply(lambda v: f"{float(v):.3f}" if pd.notna(v) and v != 0 else "")

    # Rename all columns to user-friendly display names
    col_rename = {
        "player_name": "Player",
        "positions": "Position",
    }
    col_rename.update({cat: CAT_DISPLAY_NAMES.get(cat, cat.upper()) for cat in stat_cols})
    roster_show = roster_show.rename(columns=col_rename)

    st.dataframe(roster_show, hide_index=True)

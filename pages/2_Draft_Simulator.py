"""Mock Draft — Practice snake draft simulator."""

import time

import numpy as np
import pandas as pd
import streamlit as st

from src.database import init_db, load_player_pool
from src.draft_state import DraftState
from src.simulation import DraftSimulator
from src.ui_shared import T, inject_custom_css, render_styled_table
from src.valuation import (
    LeagueConfig,
    SGPCalculator,
    compute_replacement_levels,
    compute_sgp_denominators,
    value_all_players,
)

try:
    from src.draft_engine import DraftRecommendationEngine

    _HAS_DRAFT_ENGINE = True
except ImportError:
    _HAS_DRAFT_ENGINE = False

st.set_page_config(page_title="Heater | Draft Simulator", page_icon="", layout="wide")

init_db()
inject_custom_css()


# ── Pool / Valuation ────────────────────────────────────────────────────────


def build_mock_pool() -> pd.DataFrame:
    """Load player pool and run full valuation pipeline."""
    progress = st.progress(0, text="Loading player data...")
    raw = load_player_pool()
    if raw is None or raw.empty:
        progress.empty()
        return pd.DataFrame()
    raw["player_name"] = raw["name"]

    progress.progress(20, text="Computing Standings Gained Points denominators...")
    lc = LeagueConfig()
    denoms = compute_sgp_denominators(raw, lc)
    lc.sgp_denominators.update(denoms)
    sgp = SGPCalculator(lc)

    progress.progress(45, text="Computing replacement levels...")
    repl = compute_replacement_levels(raw, lc, sgp)

    progress.progress(70, text="Calculating player valuations...")
    num_rounds = st.session_state.get("mock_num_rounds", 23)
    valued = value_all_players(raw, lc, replacement_levels=repl, num_rounds=num_rounds)
    if "player_name" not in valued.columns:
        valued["player_name"] = valued["name"]

    st.session_state.mock_lc = lc
    st.session_state.mock_sgp = sgp

    progress.progress(100, text="Player pool ready!")
    time.sleep(0.3)
    progress.empty()
    return valued


def get_pool() -> pd.DataFrame:
    if "mock_pool" not in st.session_state or st.session_state.mock_pool is None:
        st.session_state.mock_pool = build_mock_pool()
    return st.session_state.mock_pool


def refresh_pool() -> pd.DataFrame:
    st.session_state.mock_pool = build_mock_pool()
    return st.session_state.mock_pool


# ── Draft Init ──────────────────────────────────────────────────────────────


def init_mock_draft(pool: pd.DataFrame, draft_pos: int) -> None:
    num_teams = st.session_state.get("mock_num_teams", 12)
    num_rounds = st.session_state.get("mock_num_rounds", 23)
    ds = DraftState(num_teams=num_teams, num_rounds=num_rounds, user_team_index=draft_pos - 1)
    st.session_state.mock_ds = ds
    st.session_state.mock_started = True


# ── AI Opponent Logic ───────────────────────────────────────────────────────


def auto_pick_opponents(pool: pd.DataFrame) -> None:
    ds: DraftState = st.session_state.mock_ds
    while not ds.is_user_turn and ds.current_pick < ds.total_picks:
        available = ds.available_players(pool)
        if available.empty:
            break
        candidates = available.nsmallest(min(15, len(available)), "adp")
        size = len(candidates)
        weights = np.arange(size, 0, -1, dtype=float)
        weights /= weights.sum()
        pick_idx = int(np.random.choice(size, p=weights))
        player = candidates.iloc[pick_idx]
        pname = str(player.get("player_name", player.get("name", "Unknown")))
        ds.make_pick(
            player_id=int(player["player_id"]),
            player_name=pname,
            positions=str(player.get("positions", "Util")),
        )


# ── Recommendation Panel ────────────────────────────────────────────────────


def render_recommendations(pool: pd.DataFrame, ds: DraftState, n_sims: int) -> None:
    lc: LeagueConfig = st.session_state.get("mock_lc", LeagueConfig())
    engine_mode = st.session_state.get("mock_engine_mode", "standard")
    use_enhanced = _HAS_DRAFT_ENGINE

    rec_progress = st.progress(0, text="Preparing analysis...")
    try:
        if use_enhanced:
            rec_progress.progress(10, text=f"Running {engine_mode} engine analysis...")
            engine = DraftRecommendationEngine(lc, mode=engine_mode)
            st.session_state.mock_draft_engine = engine
            recs = engine.recommend(pool, ds, top_n=10, n_simulations=n_sims)
        else:
            rec_progress.progress(10, text="Running Monte Carlo simulation...")
            sim = DraftSimulator(lc, sigma=10.0)
            recs = sim.evaluate_candidates(pool, ds, top_n=10, n_simulations=n_sims)
        rec_progress.progress(100, text="Analysis complete!")
    except Exception:
        recs = pd.DataFrame()
    time.sleep(0.3)
    rec_progress.empty()

    # evaluate_candidates returns "name"; alias to "player_name" for rendering
    if not recs.empty and "name" in recs.columns and "player_name" not in recs.columns:
        recs["player_name"] = recs["name"]

    available = ds.available_players(pool)

    if recs.empty or available.empty:
        st.info("No recommendations available.")
        return

    top3 = recs.head(3)
    for rank, (_, row) in enumerate(top3.iterrows(), 1):
        name = row.get("player_name", "Unknown")
        pos = row.get("positions", "?")
        adp = row.get("adp", 999)
        score = float(row.get("combined_score", row.get("pick_score", 0)))
        surv = float(row.get("p_survive", 0))
        urg = float(row.get("urgency", 0))
        border_color = T["amber"] if rank == 1 else T["border"]
        label_color = T["amber"] if rank == 1 else T["tx"]

        # BUY/FAIR/AVOID badge
        bfa = str(row.get("buy_fair_avoid", "fair") or "fair").lower()
        if bfa == "buy":
            bfa_pill = (
                f'<span style="display:inline-block;padding:2px 8px;border-radius:10px;'
                f"background:{T['green']};color:#fff;font-size:10px;font-weight:700;"
                f'letter-spacing:0.5px;margin-left:6px;">BUY</span>'
            )
        elif bfa == "avoid":
            bfa_pill = (
                f'<span style="display:inline-block;padding:2px 8px;border-radius:10px;'
                f"background:{T['primary']};color:#fff;font-size:10px;font-weight:700;"
                f'letter-spacing:0.5px;margin-left:6px;">AVOID</span>'
            )
        else:
            bfa_pill = (
                f'<span style="display:inline-block;padding:2px 8px;border-radius:10px;'
                f"background:{T['sky']};color:#fff;font-size:10px;font-weight:700;"
                f'letter-spacing:0.5px;margin-left:6px;">FAIR</span>'
            )

        # Injury probability indicator
        inj_prob = float(row.get("injury_probability", 0) or 0)
        inj_html = ""
        if inj_prob > 0.01:
            inj_color = T["danger"] if inj_prob > 0.30 else (T["warn"] if inj_prob > 0.15 else T["ok"])
            inj_html = (
                f'<span style="color:{inj_color};font-size:0.78rem;margin-left:8px;">{inj_prob:.0%} injury risk</span>'
            )

        st.markdown(
            f"""
<div style="background:{T["card"]};border:2px solid {border_color};border-radius:12px;
            padding:16px;margin-bottom:10px;">
    <div style="color:{label_color};font-size:1.05rem;font-weight:700;">
        #{rank} {name}{bfa_pill}
    </div>
    <div style="color:{T["tx2"]};font-size:0.82rem;margin-top:2px;">
        {pos} &nbsp;|&nbsp; Average Draft Position: {adp:.0f}{inj_html}
    </div>
    <div style="color:{T["tx"]};font-size:0.88rem;margin-top:8px;">
        Score: {score:.1f} &nbsp;&nbsp; Survival: {surv:.0%} &nbsp;&nbsp; Urgency: {urg:.2f}
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

    # Enhanced metrics for top recommendation
    if not top3.empty:
        top = top3.iloc[0]
        _metrics = []
        cat_m = float(top.get("category_balance_multiplier", 1.0) or 1.0)
        if abs(cat_m - 1.0) > 0.01:
            need_pct = int(min(100, max(0, (cat_m - 0.8) / 0.4 * 100)))
            need_c = T["primary"] if need_pct > 60 else (T["hot"] if need_pct > 30 else T["green"])
            _metrics.append(f'<span style="font-size:11px;color:{need_c};">Need: {cat_m:.2f}x</span>')
        sd = float(top.get("statcast_delta", 0) or 0)
        if sd != 0:
            sd_c = T["green"] if sd > 0 else T["primary"]
            _metrics.append(f'<span style="font-size:11px;color:{sd_c};">Skill: {sd:+.3f}</span>')
        cb = float(top.get("closer_hierarchy_bonus", 0) or 0)
        if cb > 0:
            _metrics.append(f'<span style="font-size:11px;color:{T["green"]};">Closer: +{cb:.1f}</span>')
        sp = float(top.get("streaming_penalty", 0) or 0)
        if sp < 0:
            _metrics.append(f'<span style="font-size:11px;color:{T["primary"]};">Stream: {sp:.1f}</span>')
        if _metrics:
            st.markdown(
                '<div style="display:flex;gap:12px;padding:4px 8px;margin-bottom:8px;">'
                + " ".join(_metrics)
                + "</div>",
                unsafe_allow_html=True,
            )

    # Engine timing
    _eng = st.session_state.get("mock_draft_engine")
    if _eng and hasattr(_eng, "timing") and _eng.timing:
        _t = _eng.timing.get("total", 0)
        _m = st.session_state.get("mock_engine_mode", "standard")
        st.caption(f"Engine: {_t:.1f}s ({_m} mode)")

    st.markdown("---")

    sorted_avail = available.sort_values("pick_score", ascending=False)
    if sorted_avail.empty:
        st.info("No available players.")
        return

    # Search bar
    mock_search = st.text_input(
        "Search players...",
        key="mock_player_search",
        placeholder="Type a player name...",
        label_visibility="collapsed",
    )

    # Filter by search
    if mock_search:
        filtered = sorted_avail[sorted_avail["player_name"].str.contains(mock_search, case=False, na=False)]
    else:
        filtered = sorted_avail

    # Show top 5 as card buttons
    top_picks = filtered.head(5)
    if not top_picks.empty:
        card_cols = st.columns(len(top_picks))
        for ci, (_, prow) in enumerate(top_picks.iterrows()):
            with card_cols[ci]:
                pname = prow.get("player_name", "?")
                ppos = prow.get("positions", "?")
                pscore = prow.get("pick_score", 0)
                st.markdown(
                    f'<div class="player-card">'
                    f'<div class="pc-name">{pname}</div>'
                    f'<div class="pc-pos">{ppos}</div>'
                    f'<div class="pc-score">{pscore:.1f}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if st.button("Draft", key=f"mock_draft_{prow.get('player_id', ci)}", type="primary", width="stretch"):
                    ds.make_pick(
                        player_id=int(prow["player_id"]),
                        player_name=str(prow["player_name"]),
                        positions=str(prow.get("positions", "Util")),
                    )
                    auto_pick_opponents(pool)
                    st.rerun()


# ── User Roster Panel ───────────────────────────────────────────────────────


def render_user_roster(ds: DraftState) -> None:
    st.markdown(
        f'<div style="color:{T["amber"]};font-weight:700;font-size:1rem;margin-bottom:8px;">My Roster</div>',
        unsafe_allow_html=True,
    )
    slots = ds.user_team.slots
    rows = []
    for s in slots:
        rows.append(
            {
                "Slot": s.position,
                "Player": s.player_name or "—",
            }
        )
    df = pd.DataFrame(rows)
    render_styled_table(df, max_height=500)


# ── Recent Picks Feed ───────────────────────────────────────────────────────


def render_recent_picks(ds: DraftState) -> None:
    st.markdown(
        f'<div style="color:{T["amber"]};font-weight:700;font-size:1rem;margin-bottom:8px;">Recent Picks</div>',
        unsafe_allow_html=True,
    )
    log = ds.pick_log[-10:][::-1] if ds.pick_log else []
    if not log:
        st.markdown(f'<div style="color:{T["tx2"]};font-size:0.85rem;">No picks yet.</div>', unsafe_allow_html=True)
        return
    for entry in log:
        is_user = entry["team_index"] == ds.user_team_index
        name_color = T["amber"] if is_user else T["tx"]
        st.markdown(
            f"""
<div style="background:{T["card"]};border:1px solid {T["border"]};border-radius:8px;
            padding:8px 12px;margin-bottom:6px;">
    <span style="color:{T["tx2"]};font-size:0.75rem;">Round {entry["round"]} Pick {entry["pick_in_round"]}</span>
    <span style="color:{T["tx2"]};font-size:0.75rem;"> &nbsp;{entry["team_name"]}</span><br>
    <span style="color:{name_color};font-size:0.9rem;font-weight:600;">{entry["player_name"]}</span>
    <span style="color:{T["tx2"]};font-size:0.78rem;"> {entry["positions"]}</span>
</div>
""",
            unsafe_allow_html=True,
        )


# ── End-of-Draft Summary ────────────────────────────────────────────────────


def render_draft_summary(pool: pd.DataFrame, ds: DraftState) -> None:
    st.markdown(
        f'<div style="color:{T["amber"]};font-size:1.4rem;font-weight:700;margin-bottom:16px;">Draft Complete!</div>',
        unsafe_allow_html=True,
    )
    totals = ds.get_user_roster_totals(pool)

    col_r, col_hr, col_rbi, col_sb, col_avg, col_obp = st.columns(6)
    col_r.metric("Runs", int(totals.get("R", 0)))
    col_hr.metric("Home Runs", int(totals.get("HR", 0)))
    col_rbi.metric("Runs Batted In", int(totals.get("RBI", 0)))
    col_sb.metric("Stolen Bases", int(totals.get("SB", 0)))
    col_avg.metric("Batting Average", f"{totals.get('AVG', 0):.3f}")
    col_obp.metric("On-Base Percentage", f"{totals.get('OBP', 0):.3f}")

    col_w, col_l, col_sv, col_k, col_era, col_whip = st.columns(6)
    col_w.metric("Wins", int(totals.get("W", 0)))
    col_l.metric("Losses", int(totals.get("L", 0)))
    col_sv.metric("Saves", int(totals.get("SV", 0)))
    col_k.metric("Strikeouts", int(totals.get("K", 0)))
    col_era.metric("Earned Run Average", f"{totals.get('ERA', 0):.2f}")
    col_whip.metric("Walks + Hits per Inning Pitched", f"{totals.get('WHIP', 0):.3f}")

    # Estimate grade: sum SGP of user's picks vs other teams
    all_totals = ds.get_all_team_roster_totals(pool)
    if all_totals:
        lc: LeagueConfig = st.session_state.get("mock_lc", LeagueConfig())
        sgp = st.session_state.get("mock_sgp", SGPCalculator(lc))
        user_sgp = 0.0
        for cat in lc.hitting_categories + lc.pitching_categories:
            denom = lc.sgp_denominators.get(cat, 1)
            if denom == 0:
                continue
            val = totals.get(cat, 0)
            if cat in lc.inverse_stats:
                user_sgp -= val / denom
            else:
                user_sgp += val / denom

        team_sgps = []
        for t in all_totals:
            ts = 0.0
            for cat in lc.hitting_categories + lc.pitching_categories:
                denom = lc.sgp_denominators.get(cat, 1)
                if denom == 0:
                    continue
                val = t.get(cat, 0)
                if cat in lc.inverse_stats:
                    ts -= val / denom
                else:
                    ts += val / denom
            team_sgps.append(ts)

        avg_sgp = float(np.mean(team_sgps)) if team_sgps else 0.0
        diff = user_sgp - avg_sgp
        if diff > 8:
            grade, grade_color = "A", "#22c55e"
        elif diff > 4:
            grade, grade_color = "B", "#84cc16"
        elif diff > -2:
            grade, grade_color = "C", T["amber"]
        elif diff > -6:
            grade, grade_color = "D", "#f97316"
        else:
            grade, grade_color = "F", "#ef4444"

        st.markdown(
            f"""
<div style="background:{T["card"]};border:2px solid {grade_color};border-radius:12px;
            padding:24px;text-align:center;margin-top:20px;">
    <div style="color:{grade_color};font-size:3rem;font-weight:900;">{grade}</div>
    <div style="color:{T["tx"]};font-size:1rem;margin-top:4px;">
        Draft Grade &nbsp;|&nbsp; Standings Gained Points: {user_sgp:.1f} &nbsp; (Average: {avg_sgp:.1f})
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("### Final Roster")
    rows = [{"Slot": s.position, "Player": s.player_name or "—"} for s in ds.user_team.slots]
    render_styled_table(pd.DataFrame(rows))

    if st.button("Start New Mock Draft", type="primary"):
        for key in list(st.session_state.keys()):
            if key.startswith("mock_"):
                del st.session_state[key]
        st.rerun()


# ── Tabs: Available / Draft Board / Pick Log ────────────────────────────────


def render_tabs(pool: pd.DataFrame, ds: DraftState) -> None:
    tab_avail, tab_board, tab_log = st.tabs(["Available Players", "Draft Board", "Pick Log"])

    with tab_avail:
        available = ds.available_players(pool)
        if available.empty:
            st.info("No available players.")
        else:
            mock_positions = ["All", "C", "1B", "2B", "3B", "SS", "OF", "SP", "RP"]
            mock_pill_cols = st.columns(len(mock_positions))
            pos_filter = st.session_state.get("mock_pos_filter", "All")
            for pi, mpos in enumerate(mock_positions):
                with mock_pill_cols[pi]:
                    mtype = "primary" if pos_filter == mpos else "secondary"
                    if st.button(mpos, key=f"mock_pill_{mpos}", type=mtype, width="stretch"):
                        st.session_state.mock_pos_filter = mpos
                        st.rerun()

            disp = available.copy()
            if pos_filter != "All":
                disp = disp[disp["positions"].str.contains(pos_filter, na=False)]
            cols = ["player_name", "positions", "team", "adp", "pick_score"]
            cols = [c for c in cols if c in disp.columns]
            disp_sorted = disp[cols].sort_values("pick_score", ascending=False).copy()
            if "adp" in disp_sorted.columns:
                disp_sorted["adp"] = disp_sorted["adp"].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "")
            if "pick_score" in disp_sorted.columns:
                disp_sorted["pick_score"] = disp_sorted["pick_score"].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "")
            disp_sorted = disp_sorted.rename(
                columns={
                    "player_name": "Player",
                    "positions": "Position",
                    "team": "Team",
                    "adp": "ADP",
                    "pick_score": "Pick Score",
                }
            )
            render_styled_table(disp_sorted, max_height=400)

    with tab_board:
        if not ds.pick_log:
            st.info("No picks made yet.")
        else:
            board: dict[int, dict[int, str]] = {}
            for entry in ds.pick_log:
                r = entry["round"]
                ti = entry["team_index"]
                board.setdefault(r, {})[ti] = entry["player_name"]

            team_names = [t.team_name for t in ds.teams]
            max_round = ds.current_round if ds.current_pick < ds.total_picks else ds.num_rounds
            grid_rows = []
            for r in range(1, max_round + 1):
                row = {"Round": r}
                for ti, tname in enumerate(team_names):
                    row[tname] = board.get(r, {}).get(ti, "")
                grid_rows.append(row)

            board_df = pd.DataFrame(grid_rows).set_index("Round")
            render_styled_table(board_df, hide_index=False, max_height=500)

    with tab_log:
        if not ds.pick_log:
            st.info("No picks yet.")
        else:
            log_df = pd.DataFrame(ds.pick_log)[
                ["pick", "round", "pick_in_round", "team_name", "player_name", "positions"]
            ]
            log_df.columns = ["#", "Round", "Pick", "Team", "Player", "Position"]
            render_styled_table(log_df[::-1].reset_index(drop=True), max_height=400)


# ── Main ────────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="page-title-wrap"><div class="page-title"><span>DRAFT SIMULATOR</span></div></div>',
    unsafe_allow_html=True,
)

pool = get_pool()

if pool.empty:
    st.warning(
        "No player data found. Run `python load_sample_data.py` first, or wait for the bootstrap to complete on the main app."
    )
    st.stop()

# ── Pre-Start Settings ──────────────────────────────────────────────────────

if not st.session_state.get("mock_started", False):
    st.markdown(
        f'<div style="color:{T["tx"]};font-size:1rem;margin-bottom:16px;">'
        "Configure your mock draft settings, then click Start.</div>",
        unsafe_allow_html=True,
    )

    st.subheader("Draft Settings")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        num_teams = st.number_input("Number of teams", value=12, min_value=6, max_value=20, key="mock_num_teams_input")
    with col_b:
        num_rounds = st.number_input("Rounds", value=23, min_value=10, max_value=30, key="mock_num_rounds_input")
    with col_c:
        draft_pos = st.number_input(
            "Your draft position", value=1, min_value=1, max_value=num_teams, key="mock_draft_pos_input"
        )

    # Store in mock-prefixed session state
    st.session_state["mock_num_teams"] = num_teams
    st.session_state["mock_num_rounds"] = num_rounds
    st.session_state["mock_draft_pos"] = draft_pos

    # Also write to shared session_state keys so the main draft can read them
    st.session_state["num_teams"] = num_teams
    st.session_state["num_rounds"] = num_rounds
    st.session_state["draft_pos"] = draft_pos

    cfg_col, cfg_col2 = st.columns([1, 1])
    with cfg_col:
        n_sims_choice = st.radio("Simulation Depth", [50, 100, 200], index=0, horizontal=True, key="mock_sims_radio")
        st.session_state.mock_num_sims = n_sims_choice

    with cfg_col2:
        mock_engine_label = st.radio(
            "Engine Mode",
            ["Quick (< 1 second)", "Standard (2-3 seconds)", "Full (5-10 seconds)"],
            index=1,
            help="Quick: base analysis. Standard: Bayesian + injury + Statcast. Full: all contextual factors.",
            key="mock_engine_mode_radio",
        )
        _mock_mode_map = {
            "Quick (< 1 second)": "quick",
            "Standard (2-3 seconds)": "standard",
            "Full (5-10 seconds)": "full",
        }
        st.session_state.mock_engine_mode = _mock_mode_map.get(mock_engine_label, "standard")

    cfg_col_btn, _ = st.columns([1, 2])
    with cfg_col_btn:
        if st.button("Start Mock Draft", type="primary"):
            refresh_pool()
            pool = get_pool()
            init_mock_draft(pool, draft_pos)
            auto_pick_opponents(pool)
            st.rerun()
    st.stop()

# ── Active Draft ────────────────────────────────────────────────────────────

ds: DraftState = st.session_state.mock_ds
n_sims: int = st.session_state.get("mock_num_sims", 50)
draft_pos: int = st.session_state.get("mock_draft_pos", 6)

# Draft complete?
if ds.current_pick >= ds.total_picks:
    render_draft_summary(pool, ds)
    st.stop()

# Header bar
round_num = ds.current_round
pick_num = ds.pick_in_round
picking_idx = ds.picking_team_index()
picking_team = ds.teams[picking_idx].team_name
is_user = ds.is_user_turn

if is_user:
    header_html = (
        f'<div style="background:{T["amber"]};color:{T["ink"]};border-radius:10px;'
        f'padding:12px 20px;font-size:1.1rem;font-weight:700;margin-bottom:12px;">'
        f"Round {round_num} / Pick {pick_num} &nbsp;&mdash;&nbsp; YOUR PICK!</div>"
    )
else:
    header_html = (
        f'<div style="background:{T["card"]};color:{T["tx"]};border:1px solid {T["border"]};'
        f'border-radius:10px;padding:12px 20px;font-size:1.1rem;font-weight:600;margin-bottom:12px;">'
        f"Round {round_num} / Pick {pick_num} &nbsp;&mdash;&nbsp; On the Clock: {picking_team}</div>"
    )
st.markdown(header_html, unsafe_allow_html=True)

# Control buttons
btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
with btn_col1:
    if st.button("Reset Draft"):
        for key in list(st.session_state.keys()):
            if key.startswith("mock_") and key not in (
                "mock_draft_pos",
                "mock_num_sims",
                "mock_num_teams",
                "mock_num_rounds",
            ):
                del st.session_state[key]
        st.rerun()
with btn_col2:
    if st.button("Undo Last Pick") and ds.pick_log:
        ds.undo_last_pick()
        # If we undid back to opponent territory, let AI re-pick up to user turn
        auto_pick_opponents(pool)
        st.rerun()

# Main 3-column layout
left_col, center_col, right_col = st.columns([1, 2, 1])

with left_col:
    render_user_roster(ds)

with center_col:
    if is_user:
        render_recommendations(pool, ds, n_sims)
    else:
        st.markdown(
            f'<div style="color:{T["tx2"]};font-size:0.95rem;padding:20px 0;">'
            f"Waiting for <b>{picking_team}</b> to pick...</div>",
            unsafe_allow_html=True,
        )
        if st.button("Simulate Opponent Pick", key="mock_sim_opp"):
            auto_pick_opponents(pool)
            st.rerun()

with right_col:
    render_recent_picks(ds)

# Tabs below
st.markdown("---")
render_tabs(pool, ds)

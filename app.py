"""Fantasy Baseball Draft Tool — Broadcast Booth Edition.

Live draft assistant for 12-team snake draft with in-season management.
Dark navy + amber accents + sports broadcast typography.
"""

import copy
import logging
import os
import time

# Load .env file for Yahoo credentials (and any other env overrides)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — rely on system env vars

import numpy as np
import pandas as pd
import streamlit as st

from src.data_bootstrap import BootstrapProgress, bootstrap_all_data
from src.database import (
    create_blended_projections,
    init_db,
    load_player_pool,
)
from src.draft_state import DraftState, get_positional_needs
from src.injury_model import (
    PITCHER_AGE_THRESHOLD,
    POSITION_PLAYER_AGE_THRESHOLD,
    apply_injury_adjustment,
    compute_health_score,
    get_injury_badge,
    workload_flag,
)
from src.simulation import DraftSimulator, compute_team_preferences, detect_position_run
from src.ui_shared import (
    METRIC_TOOLTIPS,
    PAGE_ICONS,
    ROSTER_CONFIG,
    T,
    inject_custom_css,
    render_theme_toggle,
    sec,
)
from src.valuation import (
    LeagueConfig,
    SGPCalculator,
    add_process_risk,
    compute_percentile_projections,
    compute_projection_volatility,
    compute_replacement_levels,
    compute_sgp_denominators,
    value_all_players,
)

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from src.yahoo_api import YFPY_AVAILABLE, YahooFantasyClient
except ImportError:
    YFPY_AVAILABLE = False

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Draft Command Center",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── Session Init ─────────────────────────────────────────────────────


def init_session():
    defaults = {
        "page": "setup",
        "setup_step": 1,
        "player_pool": None,
        "draft_state": None,
        "league_config": None,
        "sgp_calc": None,
        "practice_mode": True,
        "practice_draft_state": None,
        "auto_sgp": True,
        "risk_tolerance": 0.5,
        "num_sims": 100,
        "last_lock_in": 0,
        "last_drafted": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Wizard Progress Bar ─────────────────────────────────────────────

WIZARD_STEPS = ["Settings", "Launch"]


def render_wizard_progress(current_step):
    steps_html = ""
    for i, label in enumerate(WIZARD_STEPS, 1):
        if i < current_step:
            cls = "done"
            icon = "&#10003;"
        elif i == current_step:
            cls = "active"
            icon = str(i)
        else:
            cls = "pending"
            icon = str(i)
        steps_html += f'<div class="wizard-step {cls}">{icon} &nbsp;{label}</div>'
    st.markdown(f'<div class="wizard-bar">{steps_html}</div>', unsafe_allow_html=True)


# ── Setup Page ──────────────────────────────────────────────────────


def render_setup_page():
    st.markdown(
        '<div class="page-title">Draft Command Center</div>',
        unsafe_allow_html=True,
    )
    step = st.session_state.setup_step
    render_wizard_progress(step)

    if step == 1:
        render_step_settings()
    elif step == 2:
        render_step_launch()


# ── Step 1: Import ──────────────────────────────────────────────────


def render_splash_screen():
    """Show loading splash while data bootstraps on every app launch."""
    if st.session_state.get("bootstrap_complete"):
        return True  # Already done this session

    placeholder = st.empty()
    with placeholder.container():
        st.markdown(
            f'<div style="text-align:center; padding:4rem 2rem;">'
            f'<h1 style="color:{T["amber"]}; font-family:Oswald;'
            f' letter-spacing:3px;">DRAFT COMMAND CENTER</h1>'
            f'<p style="color:{T["tx2"]};">Loading MLB data...</p>'
            f"</div>",
            unsafe_allow_html=True,
        )
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def on_progress(p: BootstrapProgress):
            progress_bar.progress(min(p.pct, 1.0))
            status_text.text(f"{p.phase}: {p.detail}")

        yahoo_client = st.session_state.get("yahoo_client")
        results = bootstrap_all_data(
            yahoo_client=yahoo_client,
            on_progress=on_progress,
        )

        # Blend projections if we have multi-system data
        try:
            import sqlite3 as _sql

            from src.database import DB_PATH as _dbp

            _tc = _sql.connect(str(_dbp))
            _non_blended = _tc.execute("SELECT COUNT(*) FROM projections WHERE system != 'blended'").fetchone()[0]
            _tc.close()
            if _non_blended > 0:
                create_blended_projections()
        except Exception:
            pass  # Blending failures are non-fatal

        st.session_state["bootstrap_complete"] = True
        st.session_state["bootstrap_results"] = results

    placeholder.empty()
    return True


# ── Step 1: Settings ───────────────────────────────────────────────


def render_step_settings():
    sec("Step 1 — League Settings")

    # ── Yahoo Fantasy Connect (optional) ──────────────────────────────
    if YFPY_AVAILABLE and not st.session_state.get("yahoo_connected"):
        yahoo_key = os.environ.get("YAHOO_CLIENT_ID")
        yahoo_secret = os.environ.get("YAHOO_CLIENT_SECRET")
        yahoo_league_id = os.environ.get("YAHOO_LEAGUE_ID", "").strip()
        if yahoo_key and yahoo_secret and yahoo_league_id:
            with st.expander("Connect Yahoo Fantasy (optional)", expanded=False):
                from src.yahoo_api import build_oauth_url, exchange_code_for_token

                auth_url = build_oauth_url(yahoo_key, redirect_uri="oob")
                st.markdown(
                    f'<a href="{auth_url}" target="_blank" style="'
                    f"display:inline-block;padding:8px 20px;"
                    f"background:{T['amber']};color:{T['ink']};"
                    f"border-radius:8px;font-weight:700;font-family:Oswald,sans-serif;"
                    f'text-decoration:none;font-size:14px;">'
                    f"Authorize with Yahoo</a>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    "Click above to open Yahoo login. After authorizing, "
                    "Yahoo will show a **verification code**. Paste it below."
                )
                with st.form("yahoo_oob_form"):
                    yahoo_code = st.text_input(
                        "Yahoo verification code",
                        placeholder="Paste the code from Yahoo here",
                    )
                    submitted = st.form_submit_button("Connect", type="primary")
                if submitted and yahoo_code:
                    with st.spinner("Exchanging code for token..."):
                        token_data = exchange_code_for_token(yahoo_key, yahoo_secret, yahoo_code)
                    if token_data and token_data.get("access_token"):
                        with st.spinner("Connecting to Yahoo Fantasy..."):
                            try:
                                client = YahooFantasyClient(league_id=yahoo_league_id)
                                if client.authenticate(
                                    yahoo_key,
                                    yahoo_secret,
                                    token_data=token_data,
                                ):
                                    st.session_state.yahoo_client = client
                                    st.session_state.yahoo_connected = True
                                    try:
                                        settings = client.get_league_settings()
                                        if settings:
                                            st.session_state.yahoo_settings = settings
                                        client.sync_to_db()
                                    except Exception:
                                        pass  # Sync failures are non-fatal
                                    st.success("Connected to Yahoo Fantasy!")
                                    st.toast("League data synced!")
                                    st.rerun()
                                else:
                                    st.error("Authentication failed. Check your credentials.")
                            except Exception as e:
                                st.error(f"Connection error: {e}")
                    else:
                        st.error("Invalid or expired code. Click the link above to get a new one.")
    elif st.session_state.get("yahoo_connected"):
        st.markdown(
            f'<div style="background:{T["card"]};border:1px solid {T["ok"]};'
            f'border-radius:12px;padding:12px 16px;margin-bottom:16px;">'
            f'<span style="font-family:Oswald,sans-serif;color:{T["ok"]};'
            f'font-size:14px;">{PAGE_ICONS["check"]} Yahoo Fantasy Connected</span></div>',
            unsafe_allow_html=True,
        )

    # ── Data Status ───────────────────────────────────────────────────
    bootstrap_results = st.session_state.get("bootstrap_results", {})
    if bootstrap_results:
        with st.expander("Data Status", expanded=False):
            for source, result in bootstrap_results.items():
                icon = PAGE_ICONS["check"] if "Error" not in str(result) else PAGE_ICONS["x_mark"]
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;padding:4px 0;">'
                    f'<span style="font-size:14px;">{icon}</span>'
                    f'<span style="color:{T["tx"]};font-size:13px;font-weight:600;">'
                    f"{source.replace('_', ' ').title()}</span>"
                    f'<span style="color:{T["tx2"]};font-size:12px;">— {result}</span></div>',
                    unsafe_allow_html=True,
                )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("**SGP Denominators**")
        auto_sgp = st.toggle("Auto-compute SGP", value=st.session_state.auto_sgp, key="auto_sgp_toggle")
        st.session_state.auto_sgp = auto_sgp

        if not auto_sgp:
            _sgp_help = "Stat increase needed to gain one roto standings point"
            sgp_r = st.number_input("R", value=32.0, step=1.0, key="sgp_r", help=_sgp_help)
            sgp_hr = st.number_input("HR", value=12.0, step=1.0, key="sgp_hr", help=_sgp_help)
            sgp_rbi = st.number_input("RBI", value=30.0, step=1.0, key="sgp_rbi", help=_sgp_help)
            sgp_sb = st.number_input("SB", value=8.0, step=1.0, key="sgp_sb", help=_sgp_help)
            sgp_avg = st.number_input("AVG", value=0.008, step=0.001, format="%.4f", key="sgp_avg", help=_sgp_help)
            sgp_w = st.number_input("W", value=3.0, step=1.0, key="sgp_w", help=_sgp_help)
            sgp_sv = st.number_input("SV", value=7.0, step=1.0, key="sgp_sv", help=_sgp_help)
            sgp_k = st.number_input("K", value=25.0, step=1.0, key="sgp_k", help=_sgp_help)
            sgp_era = st.number_input("ERA", value=0.30, step=0.01, format="%.3f", key="sgp_era", help=_sgp_help)
            sgp_whip = st.number_input("WHIP", value=0.03, step=0.01, format="%.3f", key="sgp_whip", help=_sgp_help)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("**Draft Settings**")
        num_teams = st.number_input("Number of teams", value=12, min_value=6, max_value=20, key="num_teams")
        num_rounds = st.number_input("Rounds", value=23, min_value=10, max_value=30, key="num_rounds")
        draft_pos = st.number_input("Your draft position", value=1, min_value=1, max_value=num_teams, key="draft_pos")

        st.markdown("---")
        st.markdown("**Risk Tolerance**")
        risk = st.slider(
            "Risk appetite",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.risk_tolerance,
            step=0.1,
            format="%.1f",
            key="risk_slider",
            help="0 = Conservative (safe picks), 1 = Aggressive (high-upside gambles)",
        )
        st.session_state.risk_tolerance = risk

        labels = ["Conservative", "Balanced", "Moderate", "Aggressive", "YOLO"]
        idx = min(int(risk * 4), 4)
        st.markdown(f'<span class="badge badge-fair">{labels[idx]}</span>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Nav
    st.markdown("")
    c1, c2, c3 = st.columns([2, 1, 2])
    with c2:
        if st.button("Next →", type="primary", width="stretch"):
            st.session_state.setup_step = 2
            st.rerun()


# ── Step 2: Launch ──────────────────────────────────────────────────


def render_step_launch():
    sec("Step 2 — Ready to Launch")

    # Build league config + player pool
    pool = _build_player_pool()
    if pool is None:
        st.error("No player data found. Data may still be loading — try refreshing.")
        if st.button("← Back to Settings"):
            st.session_state.setup_step = 1
            st.rerun()
        return

    pool_size = len(pool)
    if "is_hitter" in pool.columns:
        hitters = int(pool["is_hitter"].sum())
    else:
        hitters = 0
    pitchers = pool_size - hitters

    # Pre-flight checklist
    checks = [
        ("Player Pool", pool_size > 0, f"{pool_size} players loaded"),
        ("Hitters", hitters > 0, f"{hitters} hitters"),
        ("Pitchers", pitchers > 0, f"{pitchers} pitchers"),
        ("Valuations", "pick_score" in pool.columns, "SGP + VORP computed"),
    ]

    for label, ok, detail in checks:
        icon = PAGE_ICONS["check"] if ok else PAGE_ICONS["x_mark"]
        badge_cls = "badge-value" if ok else "badge-reach"
        st.markdown(
            f'<div class="glass" style="display:flex;align-items:center;gap:12px;padding:12px 16px;">'
            f'<span style="font-size:20px;">{icon}</span>'
            f'<div><span style="font-family:Oswald,sans-serif;font-weight:600;font-size:14px;'
            f'text-transform:uppercase;letter-spacing:1px;color:{T["tx"]};">{label}</span>'
            f'<br><span class="badge {badge_cls}">{detail}</span></div></div>',
            unsafe_allow_html=True,
        )

    all_ok = all(ok for _, ok, _ in checks)

    # Draft settings
    st.markdown("")
    c1, c2 = st.columns(2)
    with c1:
        practice = st.toggle(
            "Practice Mode", value=True, key="launch_practice", help="Auto-picks for opponents so you can test"
        )
    with c2:
        resume = st.toggle("Resume saved draft", value=False, key="launch_resume")

    # Launch button
    st.markdown("")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button("START DRAFT", type="primary", width="stretch", disabled=not all_ok):
            _start_new_draft(pool, practice, resume)

    # Back
    if st.button("← Back", key="s4_back"):
        st.session_state.setup_step = 1
        st.rerun()


def _build_player_pool():
    """Load player pool and run valuation pipeline."""
    try:
        init_db()
        pool = load_player_pool()
        if pool is None or pool.empty:
            return None

        # Build league config
        num_teams = st.session_state.get("num_teams", 12)
        lc = LeagueConfig(num_teams=num_teams)

        if not st.session_state.auto_sgp:
            lc.sgp_r = st.session_state.get("sgp_r", 32.0)
            lc.sgp_hr = st.session_state.get("sgp_hr", 12.0)
            lc.sgp_rbi = st.session_state.get("sgp_rbi", 30.0)
            lc.sgp_sb = st.session_state.get("sgp_sb", 8.0)
            lc.sgp_avg = st.session_state.get("sgp_avg", 0.008)
            lc.sgp_w = st.session_state.get("sgp_w", 3.0)
            lc.sgp_sv = st.session_state.get("sgp_sv", 7.0)
            lc.sgp_k = st.session_state.get("sgp_k", 25.0)
            lc.sgp_era = st.session_state.get("sgp_era", 0.30)
            lc.sgp_whip = st.session_state.get("sgp_whip", 0.03)

        sgp = SGPCalculator(lc)
        if st.session_state.auto_sgp:
            denoms = compute_sgp_denominators(pool, lc)
            for cat, val in denoms.items():
                attr = f"sgp_{cat.lower()}"
                if hasattr(lc, attr):
                    setattr(lc, attr, val)
            sgp = SGPCalculator(lc)  # Rebuild with new denominators

        repl = compute_replacement_levels(pool, lc, sgp)
        pool = value_all_players(pool, lc, replacement_levels=repl)

        # Add player_name alias for UI while keeping 'name' for backend modules
        if "name" in pool.columns and "player_name" not in pool.columns:
            pool["player_name"] = pool["name"]

        st.session_state.player_pool = pool
        st.session_state.league_config = lc
        st.session_state.sgp_calc = sgp
        return pool
    except Exception as e:
        st.error(f"Error building player pool: {e}")
        return None


def _start_new_draft(pool, practice, resume):
    """Initialize draft state and switch to draft page."""
    num_teams = st.session_state.get("num_teams", 12)
    num_rounds = st.session_state.get("num_rounds", 23)
    draft_pos = st.session_state.get("draft_pos", 1)

    if resume:
        try:
            ds = DraftState.load(roster_config=ROSTER_CONFIG)
            st.toast("Resumed saved draft!")
        except FileNotFoundError:
            st.warning("No saved draft found. Starting fresh.")
            ds = DraftState(
                num_teams=num_teams,
                num_rounds=num_rounds,
                user_team_index=draft_pos - 1,
                roster_config=ROSTER_CONFIG,
            )
    else:
        ds = DraftState(
            num_teams=num_teams,
            num_rounds=num_rounds,
            user_team_index=draft_pos - 1,
            roster_config=ROSTER_CONFIG,
        )

    st.session_state.draft_state = ds
    st.session_state.practice_mode = practice
    st.session_state.page = "draft"
    st.rerun()


# ═══════════════════════════════════════════════════════════════════
# DRAFT PAGE
# ═══════════════════════════════════════════════════════════════════


def render_draft_page():
    ds = st.session_state.draft_state
    pool = st.session_state.player_pool
    lc = st.session_state.league_config
    sgp = st.session_state.sgp_calc

    # Practice mode state isolation: use separate ephemeral DraftState
    if ds is not None and st.session_state.practice_mode:
        if st.session_state.practice_draft_state is None:
            # Deep-copy current real state into a practice-only copy
            # This clones ALL attributes including team roster slot data
            st.session_state.practice_draft_state = copy.deepcopy(ds)
        ds = st.session_state.practice_draft_state  # Swap — all reads/writes go to practice state

    if ds is None or pool is None:
        st.error("Draft not initialized. Go back to setup.")
        if st.button("← Setup"):
            st.session_state.page = "setup"
            st.rerun()
        return

    # ── Compute health scores ────────────────────────────────
    if "health_scores" not in st.session_state:
        from src.database import get_connection

        conn = get_connection()
        try:
            injury_df = pd.read_sql_query("SELECT * FROM injury_history", conn)
        except Exception:
            injury_df = pd.DataFrame()
        finally:
            conn.close()

        health_dict = {}
        if not injury_df.empty and "player_id" in injury_df.columns:
            for pid, group in injury_df.groupby("player_id"):
                gp = group["games_played"].tolist()
                ga = group["games_available"].tolist()
                health_dict[pid] = compute_health_score(gp, ga)
        st.session_state.health_scores = health_dict

        # Apply injury adjustment to counting stats so injured players rank lower
        if health_dict:
            health_scores_df = pd.DataFrame([{"player_id": pid, "health_score": hs} for pid, hs in health_dict.items()])
            pool = apply_injury_adjustment(pool, health_scores_df)

    # ── Compute percentile ranges ────────────────────────────────
    if "percentile_data" not in st.session_state:
        from src.database import get_connection

        conn = get_connection()
        try:
            systems = {}
            for system_name in ["steamer", "zips", "depthcharts", "blended"]:
                df = pd.read_sql_query(
                    "SELECT * FROM projections WHERE system = ?",
                    conn,
                    params=(system_name,),
                )
                if not df.empty:
                    systems[system_name] = df
        except Exception:
            systems = {}
        finally:
            conn.close()

        if len(systems) >= 2:
            vol = compute_projection_volatility(systems)
            vol = add_process_risk(vol)
            pct_projs = compute_percentile_projections(base=pool, volatility=vol)
            st.session_state.percentile_data = pct_projs
            st.session_state.has_percentiles = True
        else:
            st.session_state.percentile_data = {}
            st.session_state.has_percentiles = False

    # ── Sidebar ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            f'<div style="font-family:Oswald,sans-serif;font-weight:700;font-size:18px;'
            f'color:{T["amber"]};text-transform:uppercase;letter-spacing:2px;margin-bottom:12px;">'
            f"Controls</div>",
            unsafe_allow_html=True,
        )

        st.session_state.practice_mode = st.toggle(
            "Practice Mode", value=st.session_state.practice_mode, key="draft_practice"
        )

        if st.session_state.practice_mode:
            if st.button("Reset Practice", width="stretch"):
                st.session_state.practice_draft_state = DraftState(
                    num_teams=ds.num_teams,
                    num_rounds=ds.num_rounds,
                    user_team_index=ds.user_team_index,
                    roster_config=ROSTER_CONFIG,
                )
                st.toast("Practice reset!")
                st.rerun()

        if st.button("Undo Last Pick", width="stretch"):
            ds.undo_last_pick()
            st.toast("Pick undone!")
            st.rerun()

        if not st.session_state.get("practice_mode"):
            if st.button("Save Draft", width="stretch"):
                path = ds.save()
                st.toast("Saved!")

        st.markdown("---")
        if st.button("← Back to Setup", width="stretch"):
            st.session_state.page = "setup"
            st.rerun()

    # ── Practice mode banner ──────────────────────────────────────
    if st.session_state.practice_mode:
        st.markdown(
            f'<div style="background:rgba(245,158,11,0.15);border:2px solid {T["warn"]};'
            f'border-radius:8px;padding:10px;text-align:center;margin-bottom:12px;">'
            f'<span style="font-family:Oswald,sans-serif;color:{T["warn"]};">'
            f"PRACTICE MODE — Picks will not be saved</span></div>",
            unsafe_allow_html=True,
        )

    # ── Check draft complete ─────────────────────────────────────
    if ds.current_pick >= ds.total_picks:
        st.markdown(
            f'<div class="hero" style="text-align:center;">'
            f'<div class="p-name" style="color:{T["ok"]};">Draft Complete!</div>'
            f'<div class="p-meta">{ds.total_picks} picks made across {ds.num_rounds} rounds</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
        render_draft_tabs(ds, pool, lc, sgp)
        return

    # ── Practice mode: auto-pick opponents ────────────────────────
    if st.session_state.practice_mode and not ds.is_user_turn:
        _auto_pick_opponent(ds, pool, sgp, lc)
        st.rerun()

    # ── Command Bar ──────────────────────────────────────────────
    render_command_bar(ds)

    # ── Lock-in banner ───────────────────────────────────────────
    if st.session_state.last_drafted and (time.time() - st.session_state.last_lock_in) < 3:
        st.markdown(
            f'<div class="lock-in">{PAGE_ICONS["check"]} {st.session_state.last_drafted} — Locked In</div>',
            unsafe_allow_html=True,
        )

    # ── Run recommendation engine ────────────────────────────────
    available = ds.available_players(pool)
    rec = None
    alts = []
    pos_runs = []

    if ds.is_user_turn and len(available) > 0:
        with st.status("Analyzing best picks...", expanded=False) as status:
            try:
                sim = DraftSimulator(lc)

                # Build volatility array for MC sampling
                perc_kwargs = {}
                if st.session_state.get("has_percentiles", False):
                    vol_array = np.zeros(len(pool))
                    if "total_sgp" in pool.columns:
                        vol_array = np.full(len(pool), pool["total_sgp"].std() * 0.3)
                    perc_kwargs = {
                        "use_percentile_sampling": True,
                        "sgp_volatility": vol_array,
                    }

                candidates = sim.evaluate_candidates(
                    pool,
                    ds,
                    n_simulations=st.session_state.num_sims,
                    **perc_kwargs,
                )

                if candidates is not None and len(candidates) > 0:
                    rec = candidates.iloc[0]
                    alts = candidates.iloc[1:6] if len(candidates) > 1 else pd.DataFrame()

                # Position run detection
                pos_runs = detect_position_run(ds.pick_log)

                status.update(label="Ready", state="complete")
            except Exception as e:
                st.error(f"Engine error: {e}")
                status.update(label="Error", state="error")

    # ── Compute opponent intel ───────────────────────────────────
    threat_alerts = []
    opponent_data = []
    if ds.is_user_turn and rec is not None:
        # Build draft history for team preferences
        pick_log = ds.pick_log
        if len(pick_log) > 0:
            history_rows = []
            for pick in pick_log:
                history_rows.append(
                    {
                        "team_key": str(pick.get("team_index", "")),
                        "positions": pick.get("positions", ""),
                    }
                )
            history_df = pd.DataFrame(history_rows)
            preferences = compute_team_preferences(history_df)
        else:
            preferences = {}
            history_df = None

        # Determine which teams pick before our next turn
        next_user = ds.next_user_pick()
        if next_user and next_user > ds.current_pick:
            picks_between = next_user - ds.current_pick - 1
        else:
            picks_between = ds.num_teams - 1

        hero_pos = rec.get("positions", "").split(",")[0].strip()
        teams_needing_pos = 0

        for offset in range(1, picks_between + 1):
            pick_idx = ds.current_pick + offset
            if pick_idx >= ds.total_picks:
                break
            team_idx = ds.picking_team_index(pick_idx)
            team_name = ds.teams[team_idx].team_name

            # Get positional needs
            state_dict = {"picks": ds.pick_log}
            needs = get_positional_needs(state_dict, team_idx, ROSTER_CONFIG)

            # Check preference bias
            team_key = str(team_idx)
            pref = preferences.get(team_key, {}).get("positional_bias", {})
            bias = pref.get(hero_pos, 0)

            if hero_pos in needs:
                teams_needing_pos += 1

            if bias > 0.6:
                threat_alerts.append(f"{PAGE_ICONS['fire']} {team_name} targets {hero_pos} early ({bias:.0%} bias)")

            opponent_data.append(
                {
                    "Team": team_name,
                    "Needs": ", ".join(sorted(needs.keys())) if needs else "\u2014",
                    "Bias": hero_pos + f" ({bias:.0%})" if bias > 0.3 else "\u2014",
                }
            )

        if teams_needing_pos >= 3:
            threat_alerts.insert(
                0, f"{PAGE_ICONS['warning']} Low survival: {teams_needing_pos} teams need {hero_pos} before you"
            )

        surv = rec.get("p_survive", rec.get("survival_pct", 50))
        if surv > 80 and not threat_alerts:
            threat_alerts.append(f"{PAGE_ICONS['check']} Likely available next round")

    # ── 3-Column Layout ──────────────────────────────────────────
    left, center, right = st.columns([1.2, 3, 1.3])

    with left:
        render_roster_panel(ds, pool)
        render_scarcity_rings(ds, pool)

    with center:
        if ds.is_user_turn:
            if rec is not None:
                render_hero_pick(rec, ds, pool, threat_alerts=threat_alerts)
                if len(alts) > 0:
                    render_alternatives(alts)
            else:
                st.info("No recommendation available.")

            # Position run alerts
            if pos_runs:
                for pos in pos_runs:
                    st.markdown(
                        f'<div class="alert-card">'
                        f'<strong style="color:{T["warn"]};">{PAGE_ICONS["zap"]} {pos} Run!</strong> '
                        f"Position being drafted heavily</div>",
                        unsafe_allow_html=True,
                    )
        else:
            team_idx = ds.picking_team_index()
            team_name = ds.teams[team_idx].team_name
            picks_away = ds.picks_until_user_turn()
            st.markdown(
                f'<div class="glass" style="text-align:center;padding:32px;">'
                f'<div style="font-family:Oswald,sans-serif;font-size:20px;color:{T["tx2"]};">'
                f"{team_name} is on the clock</div>"
                f'<div style="font-family:JetBrains Mono,monospace;font-size:14px;color:{T["tx2"]};'
                f'margin-top:8px;">{picks_away} picks until your turn</div></div>',
                unsafe_allow_html=True,
            )

    with right:
        render_pick_entry(ds, pool, available)
        render_recent_feed(ds)

    # ── Bottom Tabs ──────────────────────────────────────────────
    render_draft_tabs(ds, pool, lc, sgp)


# ── Command Bar ─────────────────────────────────────────────────────


def render_command_bar(ds):
    pct = int((ds.current_pick / ds.total_picks) * 100) if ds.total_picks > 0 else 0

    if ds.is_user_turn:
        center_html = '<div class="your-turn">Your Turn</div>'
    else:
        team_idx = ds.picking_team_index()
        team_name = ds.teams[team_idx].team_name
        picks_away = ds.picks_until_user_turn()
        center_html = f'<div class="waiting">{team_name} picking... {picks_away} to go</div>'

    st.markdown(
        f'<div class="cmd-bar">'
        f'<div class="cmd-left">Round {ds.current_round} &middot; Pick {ds.pick_in_round}</div>'
        f'<div class="cmd-center">{center_html}</div>'
        f'<div class="cmd-right">'
        f"<span>{pct}%</span>"
        f'<div class="prog-track"><div class="prog-fill" style="width:{pct}%;"></div></div>'
        f"<span>Pick {ds.current_pick + 1}/{ds.total_picks}</span>"
        f"</div></div>",
        unsafe_allow_html=True,
    )


# ── Hero Pick Card ──────────────────────────────────────────────────


def render_hero_pick(rec, ds, pool, threat_alerts=None):
    name = rec.get("player_name", rec.get("name", "Unknown"))
    pos = rec.get("positions", rec.get("position", ""))
    score = rec.get("pick_score", rec.get("combined_score", 0))
    reason = rec.get("reason", "Best available by combined score")

    # Survival gauge
    surv = rec.get("survival_pct", rec.get("survival", 50))
    if surv > 70:
        surv_color = T["ok"]
    elif surv > 40:
        surv_color = T["warn"]
    else:
        surv_color = T["danger"]
    surv_deg = int(surv * 3.6)

    # SGP chips
    sgp_cats = [
        ("R", rec.get("sgp_r", 0)),
        ("HR", rec.get("sgp_hr", 0)),
        ("RBI", rec.get("sgp_rbi", 0)),
        ("SB", rec.get("sgp_sb", 0)),
        ("AVG", rec.get("sgp_avg", 0)),
        ("W", rec.get("sgp_w", 0)),
        ("SV", rec.get("sgp_sv", 0)),
        ("K", rec.get("sgp_k", 0)),
        ("ERA", rec.get("sgp_era", 0)),
        ("WHIP", rec.get("sgp_whip", 0)),
    ]
    chips_html = ""
    for cat, val in sgp_cats:
        if val and abs(val) > 0.01:
            cls = "pos" if val > 0 else "neg"
            sign = "+" if val > 0 else ""
            chips_html += f'<span class="sgp-chip {cls}">{cat} {sign}{val:.2f}</span>'

    # Value badge
    adp = rec.get("adp", None)
    pick_num = ds.current_pick + 1
    if adp and adp > 0:
        diff = adp - pick_num
        if diff > 10:
            vb = f'<span class="badge badge-value" title="{METRIC_TOOLTIPS["adp_value"]}">Value</span>'
        elif diff < -10:
            vb = f'<span class="badge badge-reach" title="{METRIC_TOOLTIPS["adp_reach"]}">Reach</span>'
        else:
            vb = f'<span class="badge badge-fair" title="{METRIC_TOOLTIPS["adp_fair"]}">Fair</span>'
    else:
        vb = ""

    # Tier
    tier = int(rec.get("tier", 1))
    tier = max(1, min(tier, 8))

    # Injury & risk badges
    pid = rec.get("player_id", None)
    hs = st.session_state.get("health_scores", {}).get(pid, 0.85)
    badge_icon, badge_label = get_injury_badge(hs)
    injury_html = f'<span style="margin-left:8px;">{badge_icon} {badge_label}</span>'

    # Age flag
    age = rec.get("age", None)
    is_hitter = rec.get("is_hitter", 1)
    age_threshold = POSITION_PLAYER_AGE_THRESHOLD if is_hitter else PITCHER_AGE_THRESHOLD
    age_html = (
        f' <span style="color:{T["warn"]};">{PAGE_ICONS["warning"]} Age {age}</span>'
        if age and age >= age_threshold
        else ""
    )

    # Workload flag (pitchers only — >40 IP increase)
    workload_html = ""
    if not is_hitter:
        ip_current = rec.get("ip", 0)
        ip_prev = rec.get("ip_prev", ip_current)  # Fallback to current if no prev
        if workload_flag(ip_current, ip_prev):
            workload_html = f' <span style="color:{T["danger"]};">{PAGE_ICONS["fire"]} Workload</span>'

    # Percentile range bar
    pct_html = ""
    if st.session_state.get("has_percentiles", False):
        pct_data = st.session_state.get("percentile_data", {})
        p10_df = pct_data.get(10)
        p90_df = pct_data.get(90)
        if p10_df is not None and p90_df is not None and pid is not None:
            p10_row = p10_df[p10_df["player_id"] == pid]
            p90_row = p90_df[p90_df["player_id"] == pid]
            if not p10_row.empty and not p90_row.empty:
                p10_val = p10_row.iloc[0].get("pick_score", p10_row.iloc[0].get("total_sgp", score * 0.7))
                p90_val = p90_row.iloc[0].get("pick_score", p90_row.iloc[0].get("total_sgp", score * 1.3))
                range_width = max(p90_val - p10_val, 0.1)
                fill_pct = int(((score - p10_val) / range_width) * 100)
                fill_pct = max(5, min(95, fill_pct))
                pct_html = (
                    f'<div title="{METRIC_TOOLTIPS["p10_p90"]}" style="margin-top:8px;font-family:JetBrains Mono,monospace;'
                    f'font-size:12px;color:{T["tx2"]};">'
                    f"P10: {p10_val:.1f} "
                    f'<span style="display:inline-block;width:120px;height:8px;'
                    f'background:{T["card_h"]};border-radius:4px;vertical-align:middle;">'
                    f'<span style="display:inline-block;width:{fill_pct}%;height:100%;'
                    f'background:{T["amber"]};border-radius:4px;"></span>'
                    f"</span>"
                    f" P90: {p90_val:.1f}</div>"
                )
    else:
        pct_html = (
            f'<div style="margin-top:4px;font-size:11px;color:{T["tx2"]};font-style:italic;">'
            f"Single projection source — range unavailable</div>"
        )

    # Threat alerts
    alerts_html = ""
    if threat_alerts:
        for alert in threat_alerts[:2]:  # Max 2 lines
            alerts_html += f'<div style="font-size:13px;margin-top:4px;color:{T["tx2"]};">{alert}</div>'

    st.markdown(
        f'<div class="hero">'
        f'<div class="score-badge" title="{METRIC_TOOLTIPS["pick_score"]}">{score:.1f}</div>'
        f'<div style="display:flex;align-items:center;gap:16px;">'
        f'<div class="surv-gauge" title="{METRIC_TOOLTIPS["survival"]}" style="background:conic-gradient({surv_color} {surv_deg}deg, '
        f'{T["card_h"]} {surv_deg}deg);">'
        f'<div style="width:48px;height:48px;border-radius:50%;background:{T["card"]};'
        f'display:flex;align-items:center;justify-content:center;">{surv:.0f}%</div></div>'
        f"<div>"
        f'<div class="p-name">{name}</div>'
        f'<div class="p-meta">{pos} &middot; Tier {tier} {vb}{injury_html}{age_html}{workload_html}</div>'
        f"</div></div>"
        f'<div class="reason">{reason}</div>'
        f'<div class="sgp-row" title="{METRIC_TOOLTIPS["sgp"]}">{chips_html}</div>'
        f"{pct_html}"
        f"{alerts_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Alternative Cards ───────────────────────────────────────────────


def render_alternatives(alts):
    sec("Alternatives")
    cols = st.columns(min(len(alts), 5))
    for i, (_, row) in enumerate(alts.iterrows()):
        if i >= 5:
            break
        with cols[i]:
            name = row.get("player_name", row.get("name", "?"))
            pos = row.get("positions", row.get("position", ""))
            score = row.get("pick_score", row.get("combined_score", 0))
            tier = int(row.get("tier", 1))
            tier = max(1, min(tier, 8))

            # Risk badge
            risk = row.get("risk_score", 0.5)
            if risk < 0.3:
                risk_badge = (
                    f'<span class="badge badge-risk-low" title="{METRIC_TOOLTIPS["health_green"]}">Low Risk</span>'
                )
            elif risk < 0.7:
                risk_badge = (
                    f'<span class="badge badge-risk-med" title="{METRIC_TOOLTIPS["health_yellow"]}">Med Risk</span>'
                )
            else:
                risk_badge = (
                    f'<span class="badge badge-risk-high" title="{METRIC_TOOLTIPS["health_red"]}">High Risk</span>'
                )

            # Compact injury badge
            alt_pid = row.get("player_id", None)
            alt_hs = st.session_state.get("health_scores", {}).get(alt_pid, 0.85)
            alt_icon, _ = get_injury_badge(alt_hs)

            # Compact percentile range
            range_text = ""
            if st.session_state.get("has_percentiles", False):
                pct_data = st.session_state.get("percentile_data", {})
                p10_df = pct_data.get(10)
                p90_df = pct_data.get(90)
                if p10_df is not None and p90_df is not None and alt_pid is not None:
                    p10_r = p10_df[p10_df["player_id"] == alt_pid]
                    p90_r = p90_df[p90_df["player_id"] == alt_pid]
                    if not p10_r.empty and not p90_r.empty:
                        p10_s = p10_r.iloc[0].get("pick_score", score * 0.7)
                        p90_s = p90_r.iloc[0].get("pick_score", score * 1.3)
                        range_text = f'<div title="{METRIC_TOOLTIPS["p10_p90"]}" style="font-size:10px;color:{T["tx2"]};">{p10_s:.1f} — {p90_s:.1f}</div>'

            st.markdown(
                f'<div class="alt tier-{tier}">'
                f'<div class="a-rank">#{i + 2}</div>'
                f'<div class="a-name">{name}</div>'
                f'<div class="a-meta">{pos} {alt_icon} {risk_badge}</div>'
                f'<div class="a-score">{score:.1f}</div>'
                f"{range_text}"
                f"</div>",
                unsafe_allow_html=True,
            )


# ── Pick Entry (Right Panel) ───────────────────────────────────────


def render_pick_entry(ds, pool, available):
    sec("Enter Pick")

    # Search / select player
    player_options = (
        available.sort_values("pick_score", ascending=False) if "pick_score" in available.columns else available
    )
    display_list = [
        f"{row['player_name']} ({row.get('positions', '?')})" for _, row in player_options.head(200).iterrows()
    ]

    selected = st.selectbox(
        "Type to search...",
        options=[""] + display_list,
        key="pick_select",
        label_visibility="collapsed",
    )

    if selected and selected != "":
        # Parse selection
        pname = selected.split(" (")[0]
        match = available[available["player_name"] == pname]
        if not match.empty:
            p = match.iloc[0]
            # Confirmation card
            st.markdown(
                f'<div class="glass" style="border-color:{T["amber"]}44;">'
                f'<div style="font-family:Oswald,sans-serif;font-size:16px;color:{T["tx"]};">'
                f"{p['player_name']}</div>"
                f'<div style="font-size:12px;color:{T["tx2"]};">{p.get("positions", "?")}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

            # Draft button
            btn_label = f"DRAFT {pname.upper()}"
            if ds.is_user_turn:
                if st.button(btn_label, type="primary", width="stretch", key="draft_btn"):
                    _execute_pick(ds, p, pool)
            else:
                # Other team's pick
                team_idx = ds.picking_team_index()
                team_name = ds.teams[team_idx].team_name
                if st.button(f"Record for {team_name}", width="stretch", key="draft_btn"):
                    _execute_pick(ds, p, pool)

    # Quick "Mark as mine" for user picks
    if not ds.is_user_turn:
        st.markdown(
            f'<div style="font-size:11px;color:{T["tx2"]};margin-top:8px;">'
            f"Or record pick for {ds.teams[ds.picking_team_index()].team_name}</div>",
            unsafe_allow_html=True,
        )


def _execute_pick(ds, player_row, pool):
    """Execute a draft pick."""
    ds.make_pick(
        player_id=int(player_row["player_id"]),
        player_name=player_row["player_name"],
        positions=player_row.get("positions", "Util"),
    )
    st.session_state.last_drafted = player_row["player_name"]
    st.session_state.last_lock_in = time.time()
    # Only persist to disk/DB when NOT in practice mode
    if not st.session_state.get("practice_mode"):
        ds.save()
    st.toast(f"Drafted {player_row['player_name']}!")
    st.rerun()


def _auto_pick_opponent(ds, pool, sgp, lc):
    """Auto-pick for opponent in practice mode using ADP + positional need."""
    available = ds.available_players(pool)
    if available.empty:
        return

    team_idx = ds.picking_team_index()
    team = ds.teams[team_idx]
    open_pos = team.open_positions()

    # Prefer ADP-ranked players that fill a positional need
    candidates = available.copy()
    if "adp" in candidates.columns:
        candidates = candidates.sort_values("adp", ascending=True)
    elif "pick_score" in candidates.columns:
        candidates = candidates.sort_values("pick_score", ascending=False)

    best = None
    for _, row in candidates.head(30).iterrows():
        pos_list = [p.strip() for p in str(row.get("positions", "Util")).split(",")]
        for pos in pos_list:
            if pos in open_pos:
                best = row
                break
        if best is not None:
            break

    if best is None:
        if candidates.empty:
            return
        best = candidates.iloc[0]

    ds.make_pick(
        player_id=int(best["player_id"]),
        player_name=best["player_name"],
        positions=best.get("positions", "Util"),
        team_index=team_idx,
    )


# ── Recent Feed ─────────────────────────────────────────────────────


def render_recent_feed(ds):
    sec("Recent Picks")
    recent = ds.pick_log[-5:][::-1] if ds.pick_log else []
    if not recent:
        st.markdown(f'<div style="color:{T["tx2"]};font-size:13px;">No picks yet.</div>', unsafe_allow_html=True)
        return

    for entry in recent:
        is_user = entry["team_index"] == ds.user_team_index
        cls = "user-pick" if is_user else ""
        st.markdown(
            f'<div class="feed-card {cls}">'
            f'<div class="feed-pick-num">#{entry["pick"] + 1} &middot; R{entry["round"]}</div>'
            f'<div class="feed-name">{entry["player_name"]}</div>'
            f'<div class="feed-team">{entry["team_name"]} &middot; {entry["positions"]}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )


# ── Roster Panel ────────────────────────────────────────────────────


def render_roster_panel(ds, pool):
    sec("My Roster")
    team = ds.user_team

    # Group slots by position type
    field_pos = ["C", "1B", "2B", "3B", "SS", "OF"]
    util_pos = ["Util"]
    pitch_pos = ["SP", "RP", "P"]
    bench_pos = ["BN"]

    for group_label, group_positions in [
        ("Field", field_pos),
        ("Utility", util_pos),
        ("Pitching", pitch_pos),
        ("Bench", bench_pos),
    ]:
        slots_in_group = [s for s in team.slots if s.position in group_positions]
        if not slots_in_group:
            continue

        for s in slots_in_group:
            if s.player_id is not None:
                cls = "filled"
                player_html = f'<div class="s-player">{s.player_name}</div>'
            else:
                cls = "empty"
                player_html = f'<div style="color:{T["tx2"]}44;font-size:11px;">Empty</div>'

            st.markdown(
                f'<div class="roster-slot {cls}"><div class="s-label">{s.position}</div>{player_html}</div>',
                unsafe_allow_html=True,
            )


# ── Scarcity Rings ──────────────────────────────────────────────────


def render_scarcity_rings(ds, pool):
    sec("Position Scarcity")
    available = ds.available_players(pool)

    positions = ["C", "1B", "2B", "3B", "SS", "OF", "SP", "RP"]
    cols = st.columns(4)

    for i, pos in enumerate(positions):
        with cols[i % 4]:
            # Count available at position
            if "positions" in available.columns:
                avail_count = len(available[available["positions"].str.contains(pos, na=False)])
            else:
                avail_count = 0

            # How many teams still need this position
            needed = ds.positions_still_needed_league(ROSTER_CONFIG)
            teams_need = needed.get(pos, 0)

            # Ratio: higher = more scarce
            if avail_count > 0:
                ratio = teams_need / avail_count
            else:
                ratio = 10.0  # very scarce

            if ratio > 1.0:
                color = T["danger"]
                critical = "critical"
            elif ratio > 0.5:
                color = T["warn"]
                critical = ""
            else:
                color = T["ok"]
                critical = ""

            pct = min(ratio * 100, 100)
            deg = int(pct * 3.6)

            st.markdown(
                f'<div class="scar-wrap">'
                f'<div class="scar-ring {critical}" style="background:conic-gradient('
                f'{color} {deg}deg, {T["card_h"]} {deg}deg);">'
                f'<div style="width:38px;height:38px;border-radius:50%;background:{T["card"]};'
                f'display:flex;align-items:center;justify-content:center;">{avail_count}</div>'
                f"</div>"
                f'<div class="scar-label">{pos}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

            if critical:
                toast_key = f"scarcity_toast_{pos}_{avail_count}"
                if toast_key not in st.session_state:
                    st.session_state[toast_key] = True
                    st.toast(f"{pos} scarce! Only {avail_count} left")


# ═══════════════════════════════════════════════════════════════════
# DRAFT TABS
# ═══════════════════════════════════════════════════════════════════


def render_draft_tabs(ds, pool, lc, sgp):
    tabs = st.tabs(["Category Balance", "Available Players", "Draft Board", "Draft Log", "Opponent Intel"])

    with tabs[0]:
        render_category_balance(ds, pool)

    with tabs[1]:
        render_available_players(ds, pool)

    with tabs[2]:
        render_draft_board(ds)

    with tabs[3]:
        render_draft_log(ds)

    with tabs[4]:
        render_opponent_intel(ds)


# ── Opponent Intel ──────────────────────────────────────────────────


def render_opponent_intel(ds):
    """Display positional needs, bias, and predicted picks for all opponents."""
    if not ds.pick_log:
        st.info("Opponent data will appear after the first few picks.")
        return

    # Compute team preferences from draft history
    history_rows = [
        {"team_key": str(p.get("team_index", "")), "positions": p.get("positions", "")} for p in ds.pick_log
    ]
    history_df = pd.DataFrame(history_rows)
    preferences = compute_team_preferences(history_df) if not history_df.empty else {}

    # Build data for each opponent
    rows = []
    for team_idx in range(ds.num_teams):
        if team_idx == ds.user_team_index:
            continue
        team_name = ds.teams[team_idx].team_name
        state_dict = {"picks": ds.pick_log}
        needs = get_positional_needs(state_dict, team_idx, ROSTER_CONFIG)
        needs_str = ", ".join(sorted(needs.keys())) if needs else "Full"

        # Historical bias
        team_key = str(team_idx)
        pref = preferences.get(team_key, {}).get("positional_bias", {})
        top_bias = max(pref.items(), key=lambda x: x[1]) if pref else ("\u2014", 0)
        bias_str = f"{top_bias[0]} ({top_bias[1]:.0%})" if top_bias[1] > 0.2 else "\u2014"

        # Predicted next pick: highest-bias position that is also a need
        predicted = "\u2014"
        for pos, frac in sorted(pref.items(), key=lambda x: -x[1]):
            if pos in needs:
                predicted = pos
                break

        # Threat level
        picks_made = sum(1 for p in ds.pick_log if p.get("team_index") == team_idx)
        threat = (
            PAGE_ICONS["alert"]
            if len(needs) <= 3
            else PAGE_ICONS["warning"]
            if len(needs) <= 8
            else PAGE_ICONS["check"]
        )

        rows.append(
            {
                "Threat": threat,
                "Team": team_name,
                "Positions Needed": needs_str,
                "Historical Bias": bias_str,
                "Predicted Next": predicted,
                "Picks Made": picks_made,
            }
        )

    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.info("No opponent data yet.")


# ── Category Balance ────────────────────────────────────────────────


def render_category_balance(ds, pool):
    totals = ds.get_user_roster_totals(pool)

    # Category cards row
    cats = [
        ("R", totals.get("R", 0), ""),
        ("HR", totals.get("HR", 0), ""),
        ("RBI", totals.get("RBI", 0), ""),
        ("SB", totals.get("SB", 0), ""),
        ("AVG", totals.get("AVG", 0), ".3f"),
        ("W", totals.get("W", 0), ""),
        ("SV", totals.get("SV", 0), ""),
        ("K", totals.get("K", 0), ""),
        ("ERA", totals.get("ERA", 0), ".2f"),
        ("WHIP", totals.get("WHIP", 0), ".3f"),
    ]

    cols = st.columns(5)
    for i, (cat, val, fmt) in enumerate(cats):
        with cols[i % 5]:
            display = f"{val:{fmt}}" if fmt else str(int(val))
            st.markdown(
                f'<div class="cat-card"><div class="cat-name">{cat}</div><div class="cat-val">{display}</div></div>',
                unsafe_allow_html=True,
            )

    # Radar chart
    if HAS_PLOTLY and ds.user_team.picks:
        _render_radar_chart(ds, pool)
    elif ds.user_team.picks:
        _render_balance_bars(ds, pool)


def _render_radar_chart(ds, pool):
    """Plotly radar chart: user team vs league average."""
    user_totals = ds.get_user_roster_totals(pool)
    all_totals = ds.get_all_team_roster_totals(pool)

    categories = ["R", "HR", "RBI", "SB", "AVG", "W", "SV", "K", "ERA", "WHIP"]
    invert = {"ERA", "WHIP"}  # lower is better

    user_vals = []
    avg_vals = []
    for cat in categories:
        uv = user_totals.get(cat, 0)
        avgs = [t.get(cat, 0) for t in all_totals]
        league_avg = np.mean(avgs) if avgs else 0
        league_std = np.std(avgs) if avgs else 1
        if league_std == 0:
            league_std = 1

        # Normalize to z-scores
        if cat in invert:
            uz = -(uv - league_avg) / league_std  # invert so lower ERA = better
            az = 0
        else:
            uz = (uv - league_avg) / league_std
            az = 0

        user_vals.append(uz)
        avg_vals.append(az)

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=user_vals + [user_vals[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name="My Team",
            line=dict(color=T["amber"], width=2),
            fillcolor="rgba(245, 158, 11, 0.13)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=avg_vals + [avg_vals[0]],
            theta=categories + [categories[0]],
            name="League Avg",
            line=dict(color=T["tx2"], width=1, dash="dot"),
        )
    )

    fig.update_layout(
        polar=dict(
            bgcolor=T["card"],
            radialaxis=dict(visible=True, color=T["tx2"], gridcolor=T["card_h"]),
            angularaxis=dict(color=T["tx2"], gridcolor=T["card_h"]),
        ),
        paper_bgcolor=T["bg"],
        plot_bgcolor=T["bg"],
        font=dict(family="DM Sans", color=T["tx2"]),
        showlegend=True,
        legend=dict(font=dict(color=T["tx2"])),
        margin=dict(l=40, r=40, t=30, b=30),
        height=350,
    )
    st.plotly_chart(fig, width="stretch")


def _render_balance_bars(ds, pool):
    """Fallback progress bars when plotly unavailable."""
    totals = ds.get_user_roster_totals(pool)
    all_totals = ds.get_all_team_roster_totals(pool)

    categories = ["R", "HR", "RBI", "SB", "AVG", "W", "SV", "K", "ERA", "WHIP"]
    for cat in categories:
        uv = totals.get(cat, 0)
        avgs = [t.get(cat, 0) for t in all_totals]
        max_val = max(avgs) if avgs else 1
        if max_val == 0:
            max_val = 1
        pct = min(uv / max_val, 1.0) if cat not in {"ERA", "WHIP"} else min(1 - (uv / max(max_val, 0.01)), 1.0)
        pct = max(pct, 0)
        st.markdown(f"**{cat}**: {uv:.3f}" if cat in {"AVG", "ERA", "WHIP"} else f"**{cat}**: {int(uv)}")
        st.progress(pct)


# ── Available Players ───────────────────────────────────────────────


def render_available_players(ds, pool):
    available = ds.available_players(pool)

    # Position filter pills
    positions = ["All", "C", "1B", "2B", "3B", "SS", "OF", "SP", "RP"]
    pill_cols = st.columns(len(positions))
    selected_pos = st.session_state.get("pos_filter", "All")

    for i, pos in enumerate(positions):
        with pill_cols[i]:
            btn_type = "primary" if selected_pos == pos else "secondary"
            if st.button(pos, key=f"pill_{pos}", type=btn_type, width="stretch"):
                st.session_state.pos_filter = pos
                st.rerun()

    # Search
    search = st.text_input(
        "Search player...", key="player_search", label_visibility="collapsed", placeholder="Search player name..."
    )

    filtered = available.copy()
    if selected_pos != "All" and "positions" in filtered.columns:
        filtered = filtered[filtered["positions"].str.contains(selected_pos, na=False)]
    if search:
        filtered = filtered[filtered["player_name"].str.contains(search, case=False, na=False)]

    # Sort
    if "pick_score" in filtered.columns:
        filtered = filtered.sort_values("pick_score", ascending=False)

    # Display columns
    display_cols = ["player_name", "positions"]
    optional = ["pick_score", "tier", "adp", "r", "hr", "rbi", "sb", "w", "sv", "k"]
    for c in optional:
        if c in filtered.columns:
            display_cols.append(c)

    display_df = filtered[display_cols].head(50).copy()

    # Rename for display
    rename_map = {"player_name": "Player", "positions": "Pos", "pick_score": "Score", "tier": "Tier", "adp": "ADP"}
    display_df = display_df.rename(columns={k: v for k, v in rename_map.items() if k in display_df.columns})

    col_config = {}
    if "Score" in display_df.columns:
        col_config["Score"] = st.column_config.NumberColumn(format="%.1f")
    if "Tier" in display_df.columns:
        col_config["Tier"] = st.column_config.NumberColumn(format="%d")
    if "ADP" in display_df.columns:
        col_config["ADP"] = st.column_config.NumberColumn(format="%.0f")

    st.dataframe(display_df, width="stretch", hide_index=True, column_config=col_config, height=400)


# ── Draft Board ─────────────────────────────────────────────────────


def render_draft_board(ds):
    # Build HTML table
    headers = ""
    for i in range(ds.num_teams):
        team_name = ds.teams[i].team_name
        cls = 'class="user-col"' if i == ds.user_team_index else ""
        short_name = team_name[:14]
        headers += f"<th {cls}>{short_name}</th>"

    rows = ""
    for r in range(ds.current_round + 2):  # show current + next round
        if r >= ds.num_rounds:
            break
        row_html = f'<td style="font-weight:600;color:{T["amber"]};">R{r + 1}</td>'
        for team_idx in range(ds.num_teams):
            # In snake draft, determine which overall pick this team had in this round
            if r % 2 == 0:
                pos_in_round = team_idx  # forward
            else:
                pos_in_round = ds.num_teams - 1 - team_idx  # reverse
            overall = r * ds.num_teams + pos_in_round

            # Find if this pick was made
            entry = None
            for log in ds.pick_log:
                if log["pick"] == overall:
                    entry = log
                    break

            is_user = team_idx == ds.user_team_index
            is_current = overall == ds.current_pick

            cls_parts = []
            if is_user:
                cls_parts.append("user-col")
            if is_current:
                cls_parts.append("current-pick")
            if entry:
                cls_parts.append("picked")
            cls = f'class="{" ".join(cls_parts)}"' if cls_parts else ""

            if entry:
                name = entry["player_name"]
                short = name[:18] if len(name) > 18 else name
                row_html += f"<td {cls}>{short}</td>"
            else:
                row_html += f'<td {cls} style="color:{T["tx2"]}44;">—</td>'

        rows += f"<tr>{row_html}</tr>"

    st.markdown(
        f'<div style="overflow-x:auto;max-height:500px;overflow-y:auto;">'
        f'<table class="draft-board">'
        f"<thead><tr><th>Rd</th>{headers}</tr></thead>"
        f"<tbody>{rows}</tbody>"
        f"</table></div>",
        unsafe_allow_html=True,
    )


# ── Draft Log ───────────────────────────────────────────────────────


def render_draft_log(ds):
    if not ds.pick_log:
        st.info("No picks recorded yet.")
        return

    # Export buttons
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("Export CSV", key="export_csv"):
            df = pd.DataFrame(ds.pick_log)
            csv = df.to_csv(index=False)
            st.download_button("Download CSV", csv, "draft_log.csv", "text/csv", key="dl_csv")
    with c2:
        if st.button("Export JSON", key="export_json"):
            import json

            j = json.dumps(ds.pick_log, indent=2)
            st.download_button("Download JSON", j, "draft_log.json", "application/json", key="dl_json")

    # Feed-style log (most recent first)
    for entry in reversed(ds.pick_log):
        is_user = entry["team_index"] == ds.user_team_index
        cls = "user-pick" if is_user else ""

        # Round break
        if entry["pick_in_round"] == 1 and entry["pick"] > 0:
            st.markdown(
                f'<div style="text-align:center;padding:4px;font-family:Oswald,sans-serif;'
                f'font-size:11px;color:{T["tx2"]};text-transform:uppercase;letter-spacing:2px;">'
                f"— Round {entry['round']} —</div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div class="feed-card {cls}">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f"<div>"
            f'<div class="feed-pick-num">#{entry["pick"] + 1} &middot; R{entry["round"]}.{entry["pick_in_round"]}</div>'
            f'<div class="feed-name">{entry["player_name"]}</div>'
            f"</div>"
            f'<div style="text-align:right;">'
            f'<div class="feed-team">{entry["team_name"]}</div>'
            f'<div style="font-size:11px;color:{T["tx2"]};">{entry["positions"]}</div>'
            f"</div></div></div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════


def main():
    init_session()
    inject_custom_css()
    render_theme_toggle()

    # Bootstrap all data on every session start (splash screen with progress)
    init_db()
    render_splash_screen()

    if st.session_state.page == "setup":
        render_setup_page()
    elif st.session_state.page == "draft":
        render_draft_page()


if __name__ == "__main__":
    main()

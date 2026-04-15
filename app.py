"""Heater — Fantasy Baseball Draft Tool.

Live draft assistant for 12-team snake draft with in-season management.
Thermal-inspired design with glassmorphic cards and kinetic typography.
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
    build_category_heatmap_html,
    inject_custom_css,
    render_context_card,
    render_page_layout,
    render_player_select,
    render_styled_table,
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
    from src.draft_engine import DraftRecommendationEngine

    HAS_DRAFT_ENGINE = True
except ImportError:
    HAS_DRAFT_ENGINE = False

try:
    from src.yahoo_api import YFPY_AVAILABLE, YahooFantasyClient
except ImportError:
    YFPY_AVAILABLE = False

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Heater",
    page_icon="",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <div style="
        background: linear-gradient(90deg, #e63946 0%, #ff6d00 100%);
        color: #ffffff;
        text-align: center;
        font-size: 14px;
        font-weight: 500;
        padding: 6px 16px;
        margin: 0;
        width: 100%;
        box-sizing: border-box;
    ">
        HEATER Beta &mdash; Thanks for testing! &nbsp;
        <a href="https://forms.gle/PLACEHOLDER" target="_blank" style="color: #ffd60a !important; text-decoration: underline;">Share feedback</a>
    </div>
    """,
    unsafe_allow_html=True,
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
        "engine_mode": "standard",
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
        f'<div class="page-title-wrap"><div class="page-title"><span>{PAGE_ICONS["logo"]} HEATER</span></div></div>',
        unsafe_allow_html=True,
    )
    step = st.session_state.setup_step
    render_wizard_progress(step)

    if step == 1:
        render_step_settings()
    elif step == 2:
        render_step_launch()


# ── Step 1: Import ──────────────────────────────────────────────────


def _try_reconnect_yahoo() -> "YahooFantasyClient | None":
    """Attempt to reconnect to Yahoo Fantasy using a saved token on disk.

    When Streamlit restarts, session state is cleared — but the token file
    persists at ``data/yahoo_token.json``. This function loads it, recreates
    the ``YahooFantasyClient``, and authenticates, so that bootstrap and
    subsequent pages have an active Yahoo connection without re-OAuth.

    Returns:
        An authenticated ``YahooFantasyClient`` or ``None`` on failure.
    """
    import json

    from src.yahoo_api import _AUTH_DIR

    token_file = _AUTH_DIR / "yahoo_token.json"
    if not token_file.exists():
        return None

    try:
        token_data = json.loads(token_file.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("Could not read Yahoo token file.", exc_info=True)
        return None

    consumer_key = token_data.get("consumer_key", os.environ.get("YAHOO_CLIENT_ID", ""))
    consumer_secret = token_data.get("consumer_secret", os.environ.get("YAHOO_CLIENT_SECRET", ""))

    if not consumer_key or not consumer_secret:
        logger.debug("Yahoo token file missing consumer_key/secret.")
        return None

    yahoo_league_id = os.environ.get("YAHOO_LEAGUE_ID", "").strip()
    if not yahoo_league_id:
        logger.debug("YAHOO_LEAGUE_ID env var not set; skipping reconnect.")
        return None

    try:
        client = YahooFantasyClient(league_id=yahoo_league_id)
        if client.authenticate(consumer_key, consumer_secret, token_data=token_data):
            logger.info("Yahoo Fantasy auto-reconnected from saved token.")
            # Persist token data in session for other pages that need it
            st.session_state.yahoo_token_data = token_data
            return client
        logger.warning("Yahoo auto-reconnect: authentication returned False.")
    except Exception:
        logger.debug("Yahoo auto-reconnect failed.", exc_info=True)

    return None


def render_splash_screen():
    """Show loading splash while data bootstraps on every app launch."""
    if st.session_state.get("bootstrap_complete"):
        return True  # Already done this session

    import time as _time

    placeholder = st.empty()
    with placeholder.container():
        st.markdown(
            f'<div style="text-align:center; padding:4rem 2rem;">'
            f"{PAGE_ICONS['logo_lg']}"
            f'<div class="splash-title">HEATER</div>'
            f'<p style="color:{T["tx2"]};font-family:Figtree,sans-serif;">'
            f"Loading MLB data...</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        _boot_start = _time.monotonic()

        def on_progress(p: BootstrapProgress):
            progress_bar.progress(min(p.pct, 1.0))
            elapsed = _time.monotonic() - _boot_start
            # Estimate remaining time from progress rate
            if p.pct > 0.01 and elapsed > 1:
                eta_secs = elapsed * (1.0 - p.pct) / p.pct
                if eta_secs >= 60:
                    eta_str = f"~{int(eta_secs // 60)}m {int(eta_secs % 60)}s remaining"
                else:
                    eta_str = f"~{int(eta_secs)}s remaining"
            elif elapsed < 2:
                eta_str = "Estimating..."
            else:
                eta_str = ""
            status_text.text(f"{p.phase}: {p.detail}  {eta_str}")

        yahoo_client = st.session_state.get("yahoo_client")

        # Auto-reconnect from saved token if no active client
        if yahoo_client is None and YFPY_AVAILABLE:
            yahoo_client = _try_reconnect_yahoo()
            if yahoo_client is not None:
                st.session_state.yahoo_client = yahoo_client
                st.session_state.yahoo_connected = True

        results = bootstrap_all_data(
            yahoo_client=yahoo_client,
            on_progress=on_progress,
        )

        # Blend projections if we have multi-system data
        try:
            import sqlite3 as _sql

            from src.database import DB_PATH as _dbp

            _tc = _sql.connect(str(_dbp))
            try:
                _non_blended = _tc.execute("SELECT COUNT(*) FROM projections WHERE system != 'blended'").fetchone()[0]
            finally:
                _tc.close()
            if _non_blended > 0:
                create_blended_projections()
        except Exception:
            pass  # Blending failures are non-fatal

        # Pre-fetch all YDS data so every page starts with LIVE caches
        if yahoo_client is not None:
            try:
                from src.yahoo_data_service import get_yahoo_data_service

                progress_bar.progress(0.95)
                yds = get_yahoo_data_service()
                _yahoo_items = [
                    ("rosters", yds.get_rosters),
                    ("standings", yds.get_standings),
                    ("matchup", yds.get_matchup),
                    ("free_agents", yds.get_free_agents),
                    ("transactions", yds.get_transactions),
                    ("settings", yds.get_settings),
                    ("schedule", yds.get_schedule),
                ]
                for _yi, (_fetch_name, _fetch_fn) in enumerate(_yahoo_items):
                    _elapsed = _time.monotonic() - _boot_start
                    _yahoo_pct = 0.95 + 0.05 * (_yi / len(_yahoo_items))
                    _eta = _elapsed * (1.0 - _yahoo_pct) / _yahoo_pct if _yahoo_pct > 0 else 0
                    _eta_lbl = f"~{int(_eta)}s remaining" if _eta >= 1 else "Almost done..."
                    status_text.text(f"Yahoo: Fetching {_fetch_name}...  {_eta_lbl}")
                    progress_bar.progress(min(_yahoo_pct, 0.99))
                    try:
                        _fetch_fn()
                    except Exception:
                        logger.debug("YDS pre-fetch failed for %s", _fetch_name)
            except Exception:
                logger.debug("YDS pre-fetch skipped.", exc_info=True)

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
                    f"border-radius:8px;font-weight:700;font-family:Bebas Neue,sans-serif;"
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
                                    st.session_state.yahoo_token_data = token_data
                                    sync_result = None
                                    try:
                                        settings = client.get_league_settings()
                                        if settings:
                                            st.session_state.yahoo_settings = settings
                                        sync_result = client.sync_to_db()
                                        if sync_result:
                                            st.session_state.yahoo_sync_result = sync_result
                                    except Exception as e:
                                        st.warning(f"Yahoo connected but sync encountered an issue: {e}")
                                    standings_n = sync_result.get("standings", 0) if sync_result else 0
                                    rosters_n = sync_result.get("rosters", 0) if sync_result else 0
                                    st.success("Connected to Yahoo Fantasy!")
                                    if rosters_n > 0:
                                        st.toast(f"Synced {rosters_n} roster and {standings_n} standing entries.")
                                    elif standings_n > 0:
                                        st.toast(f"Synced {standings_n} standings but no roster data yet.")
                                    else:
                                        st.toast("Connected, but Yahoo returned no data. Season may not have started.")
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
            f'<span style="font-family:Bebas Neue,sans-serif;color:{T["ok"]};'
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

    # League quick-stats summary cards
    num_teams = st.session_state.get("num_teams", 12)
    num_rounds = st.session_state.get("num_rounds", 23)
    pool_count = len(load_player_pool()) if "bootstrap_complete" in st.session_state else 0
    yahoo_ok = st.session_state.get("yahoo_connected", False)
    qs1, qs2 = st.columns(2)
    with qs1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div style="font-family:Bebas Neue,sans-serif;font-size:13px;letter-spacing:2px;'
            f'color:{T["tx2"]};text-transform:uppercase;">League Format</div>'
            f'<div style="font-family:Figtree,sans-serif;font-size:18px;font-weight:700;'
            f'color:{T["tx"]};margin-top:4px;">{num_teams} Teams</div>'
            f'<div style="font-size:13px;color:{T["tx2"]};">{num_rounds} Rounds &middot; Snake &middot; H2H Categories</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
    with qs2:
        pool_label = f"{pool_count:,}" if pool_count > 0 else "Loading..."
        conn_badge = (
            f'<span style="color:{T["ok"]};font-weight:600;">Yahoo Connected</span>'
            if yahoo_ok
            else f'<span style="color:{T["tx2"]};">Yahoo Not Connected</span>'
        )
        st.markdown(
            f'<div class="metric-card">'
            f'<div style="font-family:Bebas Neue,sans-serif;font-size:13px;letter-spacing:2px;'
            f'color:{T["tx2"]};text-transform:uppercase;">Player Pool</div>'
            f'<div style="font-family:Figtree,sans-serif;font-size:18px;font-weight:700;'
            f'color:{T["tx"]};margin-top:4px;">{pool_label}</div>'
            f'<div style="font-size:13px;">{conn_badge}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Standings Gained Points Denominators**")
        auto_sgp = st.toggle(
            "Auto-compute Standings Gained Points", value=st.session_state.auto_sgp, key="auto_sgp_toggle"
        )
        st.session_state.auto_sgp = auto_sgp

        if not auto_sgp:
            _sgp_help = "Stat increase needed to gain one H2H category win"
            sgp_r = st.number_input("Runs", value=32.0, step=1.0, key="sgp_r", help=_sgp_help)
            sgp_hr = st.number_input("Home Runs", value=12.0, step=1.0, key="sgp_hr", help=_sgp_help)
            sgp_rbi = st.number_input("Runs Batted In", value=30.0, step=1.0, key="sgp_rbi", help=_sgp_help)
            sgp_sb = st.number_input("Stolen Bases", value=8.0, step=1.0, key="sgp_sb", help=_sgp_help)
            sgp_avg = st.number_input(
                "Batting Average", value=0.008, step=0.001, format="%.4f", key="sgp_avg", help=_sgp_help
            )
            sgp_w = st.number_input("Wins", value=3.0, step=1.0, key="sgp_w", help=_sgp_help)
            sgp_sv = st.number_input("Saves", value=7.0, step=1.0, key="sgp_sv", help=_sgp_help)
            sgp_k = st.number_input("Strikeouts", value=25.0, step=1.0, key="sgp_k", help=_sgp_help)
            sgp_era = st.number_input(
                "Earned Run Average", value=0.30, step=0.01, format="%.3f", key="sgp_era", help=_sgp_help
            )
            sgp_whip = st.number_input(
                "Walks + Hits per Inning Pitched", value=0.03, step=0.01, format="%.3f", key="sgp_whip", help=_sgp_help
            )

    with col2:
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

        st.markdown("")
        engine_mode_label = st.radio(
            "Draft Engine Mode",
            ["Quick (< 1 second)", "Standard (2-3 seconds)", "Full (5-10 seconds)"],
            index=1,
            help="Quick: base analysis. Standard: Bayesian + injury + Statcast. Full: all contextual factors.",
            key="engine_mode_selection",
        )
        # Map display label to engine mode key
        _mode_map = {
            "Quick (< 1 second)": "quick",
            "Standard (2-3 seconds)": "standard",
            "Full (5-10 seconds)": "full",
        }
        st.session_state.engine_mode = _mode_map.get(engine_mode_label, "standard")

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
    pool_progress = st.progress(0, text="Building player pool...")
    pool = _build_player_pool(progress=pool_progress)
    pool_progress.empty()
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
        (
            "Valuations",
            "pick_score" in pool.columns,
            "Standings Gained Points + Value Over Replacement Player computed",
        ),
    ]

    for label, ok, detail in checks:
        icon = PAGE_ICONS["check"] if ok else PAGE_ICONS["x_mark"]
        badge_cls = "badge-value" if ok else "badge-reach"
        st.markdown(
            f'<div class="glass" style="display:flex;align-items:center;gap:12px;padding:12px 16px;">'
            f'<span style="font-size:20px;">{icon}</span>'
            f'<div><span style="font-family:Bebas Neue,sans-serif;font-weight:600;font-size:14px;'
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


def _build_player_pool(progress=None):
    """Load player pool and run valuation pipeline."""
    try:
        if progress:
            progress.progress(5, text="Initializing database...")
        init_db()

        if progress:
            progress.progress(10, text="Loading player pool from database...")
        pool = load_player_pool()
        if pool is None or pool.empty:
            return None

        # Build league config
        num_teams = st.session_state.get("num_teams", 12)
        lc = LeagueConfig(num_teams=num_teams)

        if not st.session_state.auto_sgp:
            lc.sgp_denominators["R"] = st.session_state.get("sgp_r", 32.0)
            lc.sgp_denominators["HR"] = st.session_state.get("sgp_hr", 12.0)
            lc.sgp_denominators["RBI"] = st.session_state.get("sgp_rbi", 30.0)
            lc.sgp_denominators["SB"] = st.session_state.get("sgp_sb", 8.0)
            lc.sgp_denominators["AVG"] = st.session_state.get("sgp_avg", 0.008)
            lc.sgp_denominators["W"] = st.session_state.get("sgp_w", 3.0)
            lc.sgp_denominators["SV"] = st.session_state.get("sgp_sv", 7.0)
            lc.sgp_denominators["K"] = st.session_state.get("sgp_k", 25.0)
            lc.sgp_denominators["ERA"] = st.session_state.get("sgp_era", 0.30)
            lc.sgp_denominators["WHIP"] = st.session_state.get("sgp_whip", 0.03)

        if progress:
            progress.progress(30, text="Computing Standings Gained Points denominators...")
        if st.session_state.auto_sgp:
            denoms = compute_sgp_denominators(pool, lc)
            lc.sgp_denominators.update(denoms)
        sgp = SGPCalculator(lc)

        if progress:
            progress.progress(55, text="Computing replacement levels...")
        repl = compute_replacement_levels(pool, lc, sgp)

        if progress:
            progress.progress(75, text="Computing player valuations...")
        pool = value_all_players(pool, lc, replacement_levels=repl)

        # Add player_name alias for UI while keeping 'name' for backend modules
        if "name" in pool.columns and "player_name" not in pool.columns:
            pool["player_name"] = pool["name"]

        st.session_state.player_pool = pool
        st.session_state.league_config = lc
        st.session_state.sgp_calc = sgp

        if progress:
            progress.progress(100, text="Player pool ready!")
            time.sleep(0.3)
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


def _render_draft_controls(ds):
    """Render draft action controls as a context card in the left column.

    Replaces the st.sidebar controls block so all draft actions are visible
    on-screen without requiring the user to open the sidebar.
    """
    mode_label = "Practice" if st.session_state.practice_mode else "Live"
    mode_color = T["warn"] if st.session_state.practice_mode else T["ok"]
    mode_html = (
        f'<span style="display:inline-block;padding:2px 8px;border-radius:8px;'
        f"background:{mode_color}22;color:{mode_color};font-size:11px;font-weight:700;"
        f'letter-spacing:1px;text-transform:uppercase;">{mode_label}</span>'
    )
    render_context_card(
        "Draft Controls",
        f'<div style="margin-bottom:8px;">{mode_html}</div>',
    )

    st.session_state.practice_mode = st.toggle(
        "Practice Mode",
        value=st.session_state.practice_mode,
        key="draft_practice",
        help="Auto-picks for opponents so you can test strategies without saving",
    )

    if st.session_state.practice_mode:
        if st.button("Reset Practice", width="stretch", key="ctrl_reset_practice"):
            st.session_state.practice_draft_state = DraftState(
                num_teams=ds.num_teams,
                num_rounds=ds.num_rounds,
                user_team_index=ds.user_team_index,
                roster_config=ROSTER_CONFIG,
            )
            st.toast("Practice reset!")
            st.rerun()

    if st.button("Undo Last Pick", width="stretch", key="ctrl_undo"):
        ds.undo_last_pick()
        st.toast("Pick undone!")
        st.rerun()

    if not st.session_state.get("practice_mode"):
        if st.button("Save Draft", width="stretch", key="ctrl_save"):
            ds.save()
            st.toast("Saved!")

    # Live Yahoo Draft Sync
    yahoo_client = st.session_state.get("yahoo_client")
    if yahoo_client and not st.session_state.get("practice_mode"):
        st.markdown(
            f'<div style="margin-top:8px;padding-top:8px;border-top:1px solid {T["border"]};"></div>',
            unsafe_allow_html=True,
        )
        if st.button("Sync Yahoo Draft", width="stretch", key="ctrl_sync_draft"):
            try:
                from src.live_draft_sync import LiveDraftSyncer

                if "live_draft_syncer" not in st.session_state:
                    st.session_state.live_draft_syncer = LiveDraftSyncer(
                        yahoo_client=yahoo_client,
                        player_pool=st.session_state.get("player_pool", pd.DataFrame()),
                        num_teams=ds.num_teams,
                        num_rounds=ds.num_rounds,
                        user_team_index=ds.user_team_index,
                    )
                syncer = st.session_state.live_draft_syncer
                # Get draft results via the YahooFantasyClient wrapper
                # Returns a DataFrame with: pick_number, round, team_name,
                # team_key, player_name, player_id
                draft_df = yahoo_client.get_draft_results()
                yahoo_picks = []
                if not draft_df.empty:
                    for _, dp in draft_df.iterrows():
                        team_key = str(dp.get("team_key", ""))
                        # Extract team index from team_key (e.g. "469.l.109662.t.3" -> 3)
                        try:
                            team_idx = int(team_key.split(".")[-1]) - 1  # 0-based
                        except (ValueError, IndexError):
                            team_idx = 0
                        # Resolve player_id to local pool ID
                        pid = syncer.resolve_player(str(dp.get("player_name", "")))
                        yahoo_picks.append(
                            {
                                "team_index": team_idx,
                                "player_id": pid or 0,
                                "player_name": str(dp.get("player_name", "")),
                                "positions": "",
                            }
                        )
                result = syncer.poll_and_sync(yahoo_picks)
                if result.error:
                    st.toast(f"Sync error: {result.error}")
                elif result.new_picks:
                    for pick in result.new_picks:
                        # Apply pick to draft state
                        try:
                            ds.make_pick(pick.player_id, pick.team_index)
                        except Exception:
                            pass  # Player may not be in pool
                    st.toast(f"Synced {len(result.new_picks)} new picks!")
                    st.rerun()
                else:
                    st.toast("No new picks to sync.")
            except Exception as e:
                st.toast(f"Draft sync failed: {e}")


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
            st.session_state.player_pool = pool

    # ── Compute percentile ranges ────────────────────────────────
    if "percentile_data" not in st.session_state:
        from src.database import get_connection

        conn = get_connection()
        try:
            systems = {}
            # Only use independent projection systems — "blended" is a
            # derived average of the others, so including it dampens
            # measured inter-system volatility by ~25-30%.
            for system_name in ["steamer", "zips", "depthcharts"]:
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

    # ── Load consensus ranks ──────────────────────────────────
    if "consensus_loaded" not in st.session_state:
        try:
            from src.ecr import load_ecr_consensus

            consensus_df = load_ecr_consensus()
            if (
                consensus_df is not None
                and not consensus_df.empty
                and "consensus_rank" in consensus_df.columns
                and "player_id" in consensus_df.columns
            ):
                merge_cols = ["player_id", "consensus_rank"]
                if "consensus_avg" in consensus_df.columns:
                    merge_cols.append("consensus_avg")
                if "n_sources" in consensus_df.columns:
                    merge_cols.append("n_sources")
                if "rank_stddev" in consensus_df.columns:
                    merge_cols.append("rank_stddev")
                consensus_subset = consensus_df[merge_cols].drop_duplicates(subset=["player_id"])
                pool = pool.merge(consensus_subset, on="player_id", how="left")
                # Compute delta: positive = HEATER ranks player higher (lower number) than consensus
                heater_rank = pool.get("enhanced_rank", pool.get("rank", pd.Series(dtype=float)))
                if heater_rank is not None and len(heater_rank) > 0:
                    pool["consensus_delta"] = pool["consensus_rank"] - heater_rank
                else:
                    pool["consensus_delta"] = pd.NA
                st.session_state.player_pool = pool
        except Exception:
            pass  # Consensus data unavailable — proceed without it
        st.session_state.consensus_loaded = True

    # ── Sidebar (minimal — Back to Setup only) ───────────────────
    with st.sidebar:
        if st.button("← Back to Setup", width="stretch"):
            st.session_state.page = "setup"
            st.rerun()

    # ── Practice mode banner ──────────────────────────────────────
    if st.session_state.practice_mode:
        st.markdown(
            f'<div class="glass" style="border:2px solid {T["warn"]};'
            f'padding:10px;text-align:center;margin-bottom:12px;">'
            f'<span style="font-family:Bebas Neue,sans-serif;letter-spacing:2px;color:{T["warn"]};">'
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

    # ── Page title + pick-status banner ──────────────────────────
    if ds.is_user_turn:
        _banner_teaser = f"Round {ds.current_round}, Pick {ds.pick_in_round} — Your Turn"
        _banner_icon = "zap"
    else:
        _team_idx = ds.picking_team_index()
        _team_name = ds.teams[_team_idx].team_name
        _picks_away = ds.picks_until_user_turn()
        _banner_teaser = f"Round {ds.current_round}, Pick {ds.pick_in_round} — {_team_name} is on the clock ({_picks_away} picks until your turn)"
        _banner_icon = "warning"
    render_page_layout(
        "Draft",
        banner_teaser=_banner_teaser,
        banner_icon=_banner_icon,
    )

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
        engine_mode = st.session_state.get("engine_mode", "standard")
        use_enhanced = HAS_DRAFT_ENGINE

        rec_progress = st.progress(0, text="Preparing analysis...")
        try:
            if use_enhanced:
                # Enhanced recommendation engine with multi-factor analysis
                rec_progress.progress(10, text=f"Running {engine_mode} engine analysis...")
                engine = DraftRecommendationEngine(lc, mode=engine_mode)
                st.session_state.draft_engine = engine
                candidates = engine.recommend(
                    pool,
                    ds,
                    top_n=10,
                    n_simulations=st.session_state.num_sims,
                    park_factors=st.session_state.get("park_factors"),
                )
            else:
                # Fallback to raw DraftSimulator
                rec_progress.progress(10, text="Running Monte Carlo simulation...")
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

            rec_progress.progress(80, text="Computing survival probabilities and urgency...")

            if candidates is not None and len(candidates) > 0:
                # evaluate_candidates() returns "name", not "player_name" — alias it
                if "name" in candidates.columns and "player_name" not in candidates.columns:
                    candidates["player_name"] = candidates["name"]
                rec = candidates.iloc[0]
                alts = candidates.iloc[1:10] if len(candidates) > 1 else pd.DataFrame()

            # Position run detection
            pos_runs = detect_position_run(ds.pick_log)

            rec_progress.progress(100, text="Analysis complete!")
            time.sleep(0.3)
        except Exception as e:
            st.error(f"Engine error: {e}")
        finally:
            rec_progress.empty()

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

        surv = rec.get("p_survive", 0.5)
        if surv > 0.80 and not threat_alerts:
            threat_alerts.append(f"{PAGE_ICONS['check']} Likely available next round")

    # ── 3-Column Layout ──────────────────────────────────────────
    left, center, right = st.columns([1.2, 3, 1.3])

    with left:
        # Draft controls context panel (replaces sidebar controls)
        _render_draft_controls(ds)

        render_roster_panel(ds, pool)
        render_scarcity_rings(ds, pool)

    with center:
        if ds.is_user_turn:
            if rec is not None:
                render_hero_pick(rec, ds, pool, threat_alerts=threat_alerts)

                # Enhanced metrics bar (from DraftRecommendationEngine)
                _render_enhanced_metrics(rec)

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

            # Engine timing caption
            if HAS_DRAFT_ENGINE and "draft_engine" in st.session_state:
                _eng = st.session_state.draft_engine
                if hasattr(_eng, "timing") and _eng.timing:
                    _total = _eng.timing.get("total", 0)
                    _mode = st.session_state.get("engine_mode", "standard")
                    st.caption(f"Engine: {_total:.1f}s ({_mode} mode)")
        else:
            team_idx = ds.picking_team_index()
            team_name = ds.teams[team_idx].team_name
            picks_away = ds.picks_until_user_turn()
            st.markdown(
                f'<div class="glass" style="text-align:center;padding:32px;">'
                f'<div style="font-family:Bebas Neue,sans-serif;font-size:20px;color:{T["tx2"]};">'
                f"{team_name} is on the clock</div>"
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:14px;color:{T["tx2"]};'
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

    # Survival gauge — p_survive is 0-1, convert to percentage
    surv = rec.get("p_survive", 0.5) * 100
    if surv > 70:
        surv_color = T["ok"]
    elif surv > 40:
        surv_color = T["warn"]
    else:
        surv_color = T["danger"]
    surv_deg = int(surv * 3.6)

    # Standings Gained Points chips
    sgp_cats = [
        ("Runs", rec.get("sgp_r", 0)),
        ("Home Runs", rec.get("sgp_hr", 0)),
        ("Runs Batted In", rec.get("sgp_rbi", 0)),
        ("Stolen Bases", rec.get("sgp_sb", 0)),
        ("Batting Average", rec.get("sgp_avg", 0)),
        ("Wins", rec.get("sgp_w", 0)),
        ("Saves", rec.get("sgp_sv", 0)),
        ("Strikeouts", rec.get("sgp_k", 0)),
        ("Earned Run Average", rec.get("sgp_era", 0)),
        ("Walks + Hits per Inning Pitched", rec.get("sgp_whip", 0)),
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
                    f'<div title="{METRIC_TOOLTIPS["p10_p90"]}" style="margin-top:8px;font-family:IBM Plex Mono,monospace;'
                    f'font-size:12px;color:{T["tx2"]};">'
                    f"10th Percentile: {p10_val:.2f} "
                    f'<span style="display:inline-block;width:120px;height:8px;'
                    f'background:{T["card_h"]};border-radius:4px;vertical-align:middle;">'
                    f'<span style="display:inline-block;width:{fill_pct}%;height:100%;'
                    f'background:{T["amber"]};border-radius:4px;"></span>'
                    f"</span>"
                    f" 90th Percentile: {p90_val:.2f}</div>"
                )
    else:
        pct_html = (
            f'<div style="margin-top:4px;font-size:11px;color:{T["tx2"]};font-style:italic;">'
            f"Single projection source — range unavailable</div>"
        )

    # BUY/FAIR/AVOID badge (from enhanced engine)
    bfa_label = rec.get("buy_fair_avoid", "fair") if hasattr(rec, "get") else "fair"
    bfa_label = str(bfa_label).lower() if bfa_label else "fair"
    if bfa_label == "buy":
        bfa_html = (
            f'<span style="display:inline-block;padding:2px 10px;border-radius:12px;'
            f"background:{T['green']};color:#ffffff;font-size:11px;font-weight:700;"
            f'letter-spacing:1px;text-transform:uppercase;margin-left:8px;">BUY</span>'
        )
    elif bfa_label == "avoid":
        bfa_html = (
            f'<span style="display:inline-block;padding:2px 10px;border-radius:12px;'
            f"background:{T['primary']};color:#ffffff;font-size:11px;font-weight:700;"
            f'letter-spacing:1px;text-transform:uppercase;margin-left:8px;">AVOID</span>'
        )
    else:
        bfa_html = (
            f'<span style="display:inline-block;padding:2px 10px;border-radius:12px;'
            f"background:{T['sky']};color:#ffffff;font-size:11px;font-weight:700;"
            f'letter-spacing:1px;text-transform:uppercase;margin-left:8px;">FAIR</span>'
        )

    # Injury probability indicator (from enhanced engine)
    inj_prob = float(rec.get("injury_probability", 0) or 0)
    inj_prob_html = ""
    if inj_prob > 0.01:
        if inj_prob > 0.30:
            inj_color = T["danger"]
        elif inj_prob > 0.15:
            inj_color = T["warn"]
        else:
            inj_color = T["ok"]
        inj_prob_html = (
            f'<span style="font-size:12px;color:{inj_color};margin-left:8px;">'
            f"{PAGE_ICONS['warning']} {inj_prob:.0%} injury risk</span>"
        )

    # Confidence level badge
    mc_std = float(rec.get("mc_std_sgp", 0) or 0)
    mc_mean = float(rec.get("mc_mean_sgp", 1) or 1)
    if mc_mean > 0 and mc_std / max(mc_mean, 0.01) < 0.15:
        conf_label, conf_color = "HIGH", T["ok"]
    elif mc_mean > 0 and mc_std / max(mc_mean, 0.01) < 0.35:
        conf_label, conf_color = "MEDIUM", T["warn"]
    else:
        conf_label, conf_color = "LOW", T["danger"]
    conf_html = (
        f'<span style="display:inline-block;padding:1px 8px;border-radius:10px;'
        f"border:1px solid {conf_color};color:{conf_color};font-size:10px;"
        f'font-weight:700;letter-spacing:1px;margin-left:8px;">{conf_label}</span>'
    )

    # LAST CHANCE badge — pulsing red when survival < 20%
    last_chance_html = ""
    if surv < 20:
        last_chance_html = '<span class="badge-last-chance">LAST CHANCE</span>'

    # Threat alerts
    alerts_html = ""
    if threat_alerts:
        for alert in threat_alerts[:2]:  # Max 2 lines
            alerts_html += f'<div style="font-size:13px;margin-top:4px;color:{T["tx2"]};">{alert}</div>'

    st.markdown(
        f'<div class="hero">'
        f'<div class="score-badge" title="{METRIC_TOOLTIPS["pick_score"]}">{score:.2f}</div>'
        f'<div style="display:flex;align-items:center;gap:16px;">'
        f'<div class="surv-gauge" title="{METRIC_TOOLTIPS["survival"]}" style="background:conic-gradient({surv_color} {surv_deg}deg, '
        f'{T["card_h"]} {surv_deg}deg);">'
        f'<div style="width:48px;height:48px;border-radius:50%;background:{T["card"]};'
        f'display:flex;align-items:center;justify-content:center;">{surv:.0f}%</div></div>'
        f"<div>"
        f'<div class="p-name">{name}{bfa_html}{conf_html}{last_chance_html}</div>'
        f'<div class="p-meta">{pos} &middot; Tier {tier} {vb}{injury_html}{age_html}{workload_html}{inj_prob_html}</div>'
        f"</div></div>"
        f'<div class="reason">{reason}</div>'
        f'<div class="sgp-row" title="{METRIC_TOOLTIPS["sgp"]}">{chips_html}</div>'
        f"{pct_html}"
        f"{alerts_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Enhanced Metrics Bar ──────────────────────────────────────────


def _render_enhanced_metrics(rec):
    """Show category balance meter + contextual factor highlights below hero card."""
    if rec is None:
        return

    # Collect metrics that have non-default values
    metrics = []
    cat_mult = float(rec.get("category_balance_multiplier", 1.0) or 1.0)
    stat_delta = float(rec.get("statcast_delta", 0) or 0)
    closer_bonus = float(rec.get("closer_hierarchy_bonus", 0) or 0)
    stream_pen = float(rec.get("streaming_penalty", 0) or 0)
    lineup_bonus = float(rec.get("lineup_protection_bonus", 0) or 0)
    flex_bonus = float(rec.get("flex_bonus", 0) or 0)

    # Only show bar if engine produced any of these
    has_data = abs(cat_mult - 1.0) > 0.01 or stat_delta != 0 or closer_bonus > 0 or stream_pen < 0

    if not has_data:
        return

    # Category need meter
    need_pct = int(min(100, max(0, (cat_mult - 0.8) / 0.4 * 100)))
    if need_pct > 60:
        need_color = T["primary"]
        need_label = "High Need"
    elif need_pct > 30:
        need_color = T["hot"]
        need_label = "Moderate"
    else:
        need_color = T["green"]
        need_label = "Low Need"

    # Build metric chips
    chips = []
    if abs(cat_mult - 1.0) > 0.01:
        chips.append(
            f'<div style="flex:1;min-width:120px;">'
            f'<div style="font-size:10px;color:{T["tx2"]};margin-bottom:2px;">Category Need</div>'
            f'<div style="background:{T["border"]};border-radius:4px;height:6px;">'
            f'<div style="background:{need_color};width:{need_pct}%;height:100%;border-radius:4px;"></div>'
            f"</div>"
            f'<div style="font-size:10px;color:{need_color};margin-top:1px;">{need_label} ({cat_mult:.2f}x)</div>'
            f"</div>"
        )
    if stat_delta != 0:
        sd_color = T["green"] if stat_delta > 0 else T["primary"]
        sd_icon = PAGE_ICONS.get("trending_up", "") if stat_delta > 0 else PAGE_ICONS.get("trending_down", "")
        chips.append(
            f'<div style="text-align:center;min-width:80px;">'
            f'<div style="font-size:10px;color:{T["tx2"]};">Skill Delta</div>'
            f'<div style="font-size:14px;font-weight:700;color:{sd_color};">{sd_icon} {stat_delta:+.3f}</div>'
            f"</div>"
        )
    if closer_bonus > 0:
        chips.append(
            f'<div style="text-align:center;min-width:80px;">'
            f'<div style="font-size:10px;color:{T["tx2"]};">Closer Bonus</div>'
            f'<div style="font-size:14px;font-weight:700;color:{T["green"]};">+{closer_bonus:.2f}</div>'
            f"</div>"
        )
    if stream_pen < 0:
        chips.append(
            f'<div style="text-align:center;min-width:80px;">'
            f'<div style="font-size:10px;color:{T["tx2"]};">Stream Penalty</div>'
            f'<div style="font-size:14px;font-weight:700;color:{T["primary"]};">{stream_pen:.2f}</div>'
            f"</div>"
        )
    if lineup_bonus > 0.01:
        chips.append(
            f'<div style="text-align:center;min-width:80px;">'
            f'<div style="font-size:10px;color:{T["tx2"]};">Lineup Bonus</div>'
            f'<div style="font-size:14px;font-weight:700;color:{T["green"]};">+{lineup_bonus:.2f}</div>'
            f"</div>"
        )
    if flex_bonus > 0.01:
        chips.append(
            f'<div style="text-align:center;min-width:80px;">'
            f'<div style="font-size:10px;color:{T["tx2"]};">Flex Bonus</div>'
            f'<div style="font-size:14px;font-weight:700;color:{T["sky"]};">+{flex_bonus:.2f}</div>'
            f"</div>"
        )

    if chips:
        chips_row = "".join(chips)
        st.markdown(
            f'<div style="display:flex;gap:16px;align-items:flex-end;padding:8px 12px;'
            f"margin:4px 0 8px 0;background:{T['card']};border-radius:8px;"
            f'border:1px solid {T["border"]};">{chips_row}</div>',
            unsafe_allow_html=True,
        )


# ── Alternative Cards ───────────────────────────────────────────────


def render_alternatives(alts):
    sec("Alternatives")
    n_show = min(len(alts), 9)
    cols = st.columns(min(n_show, 3))
    for i, (_, row) in enumerate(alts.iterrows()):
        if i >= n_show:
            break
        with cols[i % 3]:
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
                        range_text = f'<div title="{METRIC_TOOLTIPS["p10_p90"]}" style="font-size:10px;color:{T["tx2"]};">{p10_s:.2f} — {p90_s:.2f}</div>'

            # BUY/FAIR/AVOID mini-badge for alternatives
            alt_bfa = str(row.get("buy_fair_avoid", "fair") or "fair").lower()
            if alt_bfa == "buy":
                bfa_pill = (
                    f'<span style="display:inline-block;padding:1px 6px;border-radius:8px;'
                    f"background:{T['green']};color:#fff;font-size:9px;font-weight:700;"
                    f'letter-spacing:0.5px;">BUY</span>'
                )
            elif alt_bfa == "avoid":
                bfa_pill = (
                    f'<span style="display:inline-block;padding:1px 6px;border-radius:8px;'
                    f"background:{T['primary']};color:#fff;font-size:9px;font-weight:700;"
                    f'letter-spacing:0.5px;">AVOID</span>'
                )
            else:
                bfa_pill = ""  # Don't show FAIR pill on alts to reduce clutter

            st.markdown(
                f'<div class="alt tier-{tier}">'
                f'<div class="a-rank">#{i + 2}</div>'
                f'<div class="a-name">{name} {bfa_pill}</div>'
                f'<div class="a-meta">{pos} {alt_icon} {risk_badge}</div>'
                f'<div class="a-score">{score:.2f}</div>'
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
                f'<div style="font-family:Bebas Neue,sans-serif;font-size:16px;color:{T["tx"]};">'
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
            f'<div class="feed-pick-num">#{entry["pick"] + 1} &middot; Round {entry["round"]}</div>'
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

    # Filter to draftable players only (those with projections or ADP)
    # to avoid counting thousands of minor leaguers with no stats
    draftable = available
    if "pick_score" in available.columns:
        draftable = available[available["pick_score"].fillna(0) > 0]
    elif "adp" in available.columns:
        draftable = available[available["adp"].fillna(0) > 0]
    if draftable.empty:
        draftable = available

    positions = ["C", "1B", "2B", "3B", "SS", "OF", "SP", "RP"]
    cols = st.columns(4)

    for i, pos in enumerate(positions):
        with cols[i % 4]:
            # Count available at position
            if "positions" in draftable.columns:
                avail_count = len(draftable[draftable["positions"].str.contains(pos, na=False)])
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
        render_draft_board(ds, pool)

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
        render_styled_table(pd.DataFrame(rows))
    else:
        st.info("No opponent data yet.")


# ── Category Balance ────────────────────────────────────────────────


def render_category_balance(ds, pool):
    totals = ds.get_user_roster_totals(pool)

    # Category cards row
    cats = [
        ("Runs", totals.get("R", 0), ""),
        ("Home Runs", totals.get("HR", 0), ""),
        ("Runs Batted In", totals.get("RBI", 0), ""),
        ("Stolen Bases", totals.get("SB", 0), ""),
        ("Batting Average", totals.get("AVG", 0), ".3f"),
        ("On-Base Percentage", totals.get("OBP", 0), ".3f"),
        ("Wins", totals.get("W", 0), ""),
        ("Losses", totals.get("L", 0), ""),
        ("Saves", totals.get("SV", 0), ""),
        ("Strikeouts", totals.get("K", 0), ""),
        ("Earned Run Average", totals.get("ERA", 0), ".2f"),
        ("Walks + Hits per Inning Pitched", totals.get("WHIP", 0), ".3f"),
    ]

    cols = st.columns(6)
    for i, (cat, val, fmt) in enumerate(cats):
        with cols[i % 6]:
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

    # Category heatmap grid
    if ds.user_team.picks:
        _render_category_heatmap(ds, pool)


def _render_category_heatmap(ds, pool):
    """Render the HTML heatmap grid via Streamlit."""
    user_totals = ds.get_user_roster_totals(pool)
    all_totals = ds.get_all_team_roster_totals(pool)
    html = build_category_heatmap_html(user_totals, all_totals)
    if html:
        st.markdown(html, unsafe_allow_html=True)


def _render_radar_chart(ds, pool):
    """Plotly radar chart: user team vs league average."""
    user_totals = ds.get_user_roster_totals(pool)
    all_totals = ds.get_all_team_roster_totals(pool)

    cat_keys = ["R", "HR", "RBI", "SB", "AVG", "OBP", "W", "L", "SV", "K", "ERA", "WHIP"]
    cat_display = [
        "Runs",
        "Home Runs",
        "Runs Batted In",
        "Stolen Bases",
        "Batting Average",
        "On-Base Percentage",
        "Wins",
        "Losses",
        "Saves",
        "Strikeouts",
        "Earned Run Average",
        "Walks + Hits per Inning Pitched",
    ]
    invert = {"ERA", "WHIP", "L"}  # lower is better

    user_vals = []
    avg_vals = []
    for cat in cat_keys:
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
            theta=cat_display + [cat_display[0]],
            fill="toself",
            name="My Team",
            line=dict(color=T["amber"], width=2),
            fillcolor="rgba(230, 57, 70, 0.13)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=avg_vals + [avg_vals[0]],
            theta=cat_display + [cat_display[0]],
            name="League Average",
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
        font=dict(family="Figtree", color=T["tx2"]),
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

    cat_keys = ["R", "HR", "RBI", "SB", "AVG", "W", "SV", "K", "ERA", "WHIP"]
    cat_names = {
        "R": "Runs",
        "HR": "Home Runs",
        "RBI": "Runs Batted In",
        "SB": "Stolen Bases",
        "AVG": "Batting Average",
        "W": "Wins",
        "SV": "Saves",
        "K": "Strikeouts",
        "ERA": "Earned Run Average",
        "WHIP": "Walks + Hits per Inning Pitched",
    }
    for cat in cat_keys:
        uv = totals.get(cat, 0)
        avgs = [t.get(cat, 0) for t in all_totals]
        max_val = max(avgs) if avgs else 1
        if max_val == 0:
            max_val = 1
        pct = min(uv / max_val, 1.0) if cat not in {"ERA", "WHIP"} else min(1 - (uv / max(max_val, 0.01)), 1.0)
        pct = max(pct, 0)
        display = cat_names[cat]
        st.markdown(f"**{display}**: {uv:.2f}")
        st.progress(pct)


# ── Available Players ───────────────────────────────────────────────


def render_available_players(ds, pool):
    available = ds.available_players(pool)
    has_consensus = "consensus_rank" in available.columns

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

    # Sort and disagreement filter controls
    sort_options = ["HEATER Rank", "Consensus Rank", "Average Draft Position", "Name"]
    if not has_consensus:
        sort_options = ["HEATER Rank", "Average Draft Position", "Name"]

    ctrl_cols = st.columns([2, 2, 4]) if has_consensus else st.columns([2, 6])
    with ctrl_cols[0]:
        sort_by = st.selectbox(
            "Sort by",
            sort_options,
            index=0,
            key="avail_sort_by",
            label_visibility="collapsed",
        )

    if has_consensus:
        with ctrl_cols[1]:
            disagreement_filter = st.radio(
                "Disagreement",
                ["All", "High", "Medium", "Low"],
                index=0,
                key="avail_disagreement",
                horizontal=True,
                label_visibility="collapsed",
            )
    else:
        disagreement_filter = "All"

    # Search
    search = st.text_input(
        "Search player...", key="player_search", label_visibility="collapsed", placeholder="Search player name..."
    )

    filtered = available.copy()
    if selected_pos != "All" and "positions" in filtered.columns:
        filtered = filtered[filtered["positions"].str.contains(selected_pos, na=False)]
    if search:
        filtered = filtered[filtered["player_name"].str.contains(search, case=False, na=False)]

    # Apply disagreement filter
    if has_consensus and disagreement_filter != "All" and "consensus_delta" in filtered.columns:
        abs_delta = filtered["consensus_delta"].abs()
        if disagreement_filter == "High":
            filtered = filtered[abs_delta > 30]
        elif disagreement_filter == "Medium":
            filtered = filtered[(abs_delta > 15) & (abs_delta <= 30)]
        elif disagreement_filter == "Low":
            filtered = filtered[abs_delta <= 15]

    # Sort
    if sort_by == "Consensus Rank" and has_consensus:
        filtered = filtered.sort_values("consensus_rank", ascending=True, na_position="last")
    elif sort_by == "Average Draft Position" and "adp" in filtered.columns:
        filtered = filtered.sort_values("adp", ascending=True, na_position="last")
    elif sort_by == "Name" and "player_name" in filtered.columns:
        filtered = filtered.sort_values("player_name", ascending=True)
    elif "pick_score" in filtered.columns:
        filtered = filtered.sort_values("pick_score", ascending=False)

    # Display columns
    display_cols = ["player_name", "positions"]
    optional = ["pick_score", "tier", "adp"]
    if has_consensus:
        optional.extend(["consensus_rank", "consensus_delta"])
    optional.extend(["r", "hr", "rbi", "sb", "w", "sv", "k"])
    for c in optional:
        if c in filtered.columns:
            display_cols.append(c)

    display_df = filtered[display_cols].head(50).copy()

    # Format consensus delta with color-coded HTML
    if "consensus_delta" in display_df.columns:

        def _format_delta(val):
            if pd.isna(val):
                return ""
            v = int(val)
            if v > 0:
                # Positive delta = consensus ranks lower (HEATER sees more value)
                return f'<span style="color:{T["ok"]};font-weight:600;">+{v}</span>'
            elif v < 0:
                # Negative delta = consensus ranks higher (HEATER sees less value)
                return f'<span style="color:{T["amber"]};font-weight:600;">{v}</span>'
            return f'<span style="color:{T["tx2"]};">0</span>'

        display_df["consensus_delta"] = display_df["consensus_delta"].map(_format_delta)

    # Rename for display
    rename_map = {
        "player_name": "Player",
        "positions": "Position",
        "pick_score": "Score",
        "tier": "Tier",
        "adp": "Average Draft Position",
        "consensus_rank": "Consensus Rank",
        "consensus_delta": "HEATER vs Consensus",
    }
    display_df = display_df.rename(columns={k: v for k, v in rename_map.items() if k in display_df.columns})

    if "Score" in display_df.columns:
        display_df["Score"] = display_df["Score"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    if "Tier" in display_df.columns:
        display_df["Tier"] = display_df["Tier"].map(lambda x: f"{x:.0f}" if pd.notna(x) else "")
    if "Average Draft Position" in display_df.columns:
        display_df["Average Draft Position"] = display_df["Average Draft Position"].map(
            lambda x: f"{x:.0f}" if pd.notna(x) else ""
        )
    if "Consensus Rank" in display_df.columns:
        display_df["Consensus Rank"] = display_df["Consensus Rank"].map(lambda x: f"{x:.0f}" if pd.notna(x) else "")

    render_styled_table(display_df, max_height=400)

    # Player card selector
    if "player_id" in filtered.columns:
        render_player_select(
            display_df["Player"].tolist(),
            filtered.head(50)["player_id"].tolist(),
            key_suffix="draft",
        )


# ── Draft Board ─────────────────────────────────────────────────────


def render_draft_board(ds, pool=None):
    # Build consensus lookup from pool if available
    consensus_lookup = {}
    has_consensus = False
    if pool is not None and "consensus_rank" in pool.columns and "consensus_delta" in pool.columns:
        has_consensus = True
        name_col = "player_name" if "player_name" in pool.columns else "name"
        for _, row in pool.iterrows():
            pname = str(row.get(name_col, ""))
            c_rank = row.get("consensus_rank")
            c_delta = row.get("consensus_delta")
            if pname and pd.notna(c_rank):
                consensus_lookup[pname] = {"rank": int(c_rank), "delta": c_delta}

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
        row_html = f'<td style="font-weight:600;color:{T["amber"]};">Round {r + 1}</td>'
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
                # Append consensus rank badge if available
                consensus_badge = ""
                if has_consensus and name in consensus_lookup:
                    c_info = consensus_lookup[name]
                    c_rank = c_info["rank"]
                    c_delta = c_info["delta"]
                    if pd.notna(c_delta):
                        delta_int = int(c_delta)
                        if delta_int > 0:
                            delta_color = T["ok"]
                        elif delta_int < 0:
                            delta_color = T["amber"]
                        else:
                            delta_color = T["tx2"]
                        consensus_badge = (
                            f'<br><span style="font-size:9px;color:{T["tx2"]};">'
                            f"CR:{c_rank}"
                            f'</span> <span style="font-size:9px;color:{delta_color};font-weight:600;">'
                            f"{('+' if delta_int > 0 else '')}{delta_int}"
                            f"</span>"
                        )
                row_html += f"<td {cls}>{short}{consensus_badge}</td>"
            else:
                row_html += f'<td {cls} style="color:{T["tx2"]}44;">—</td>'

        rows += f"<tr>{row_html}</tr>"

    st.markdown(
        f'<div style="overflow-x:auto;max-height:500px;overflow-y:auto;">'
        f'<table class="draft-board">'
        f"<thead><tr><th>Round</th>{headers}</tr></thead>"
        f"<tbody>{rows}</tbody>"
        f"</table></div>",
        unsafe_allow_html=True,
    )


# ── Draft Log ───────────────────────────────────────────────────────


def render_draft_log(ds):
    if not ds.pick_log:
        st.info("No picks recorded yet.")
        return

    # Export buttons — use st.download_button directly (not nested in st.button,
    # which only returns True for a single render frame)
    import json

    df = pd.DataFrame(ds.pick_log)
    csv = df.to_csv(index=False)
    j = json.dumps(ds.pick_log, indent=2)
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        st.download_button("Export CSV", csv, "draft_log.csv", "text/csv", key="dl_csv")
    with c2:
        st.download_button("Export JSON", j, "draft_log.json", "application/json", key="dl_json")

    # Feed-style log (most recent first)
    for entry in reversed(ds.pick_log):
        is_user = entry["team_index"] == ds.user_team_index
        cls = "user-pick" if is_user else ""

        # Round break
        if entry["pick_in_round"] == 1 and entry["pick"] > 0:
            st.markdown(
                f'<div style="text-align:center;padding:4px;font-family:Bebas Neue,sans-serif;'
                f'font-size:11px;color:{T["tx2"]};text-transform:uppercase;letter-spacing:2px;">'
                f"— Round {entry['round']} —</div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div class="feed-card {cls}">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f"<div>"
            f'<div class="feed-pick-num">#{entry["pick"] + 1} &middot; Round {entry["round"]} Pick {entry["pick_in_round"]}</div>'
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

    # Bootstrap all data on every session start (splash screen with progress)
    init_db()
    render_splash_screen()

    # Force Refresh button in sidebar (only after bootstrap is done)
    if st.session_state.get("bootstrap_complete"):
        with st.sidebar:
            if st.button("Force Refresh Data", key="force_refresh_btn", width="stretch"):
                with st.spinner("Refreshing all data sources..."):
                    yahoo_client = st.session_state.get("yahoo_client")
                    bootstrap_all_data(yahoo_client=yahoo_client, force=True)
                    st.session_state["bootstrap_results"] = None
                st.rerun()

    if st.session_state.page == "setup":
        render_setup_page()
    elif st.session_state.page == "draft":
        render_draft_page()


if __name__ == "__main__":
    main()

"""Fantasy Baseball Draft Tool — Broadcast Booth Edition.

Live draft assistant for Yahoo Sports 12-team snake draft.
Dark navy + amber accents + sports broadcast typography.
"""

import time

import numpy as np
import pandas as pd
import streamlit as st

from src.database import (
    create_blended_projections,
    import_adp_csv,
    import_hitter_csv,
    import_pitcher_csv,
    init_db,
    load_player_pool,
)
from src.draft_state import DraftState
from src.simulation import DraftSimulator, detect_position_run
from src.valuation import (
    LeagueConfig,
    SGPCalculator,
    compute_replacement_levels,
    compute_sgp_denominators,
    value_all_players,
)
from src.yahoo_api import (
    YahooFantasyClient,
    has_credentials,
    load_credentials,
    save_credentials,
    sync_draft_picks,
)

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

st.set_page_config(
    page_title="Draft Command Center",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design Tokens ────────────────────────────────────────────────────

THEME = {
    "bg": "#0a0e1a",
    "card": "#1a1f2e",
    "card_h": "#252b3b",
    "amber": "#f59e0b",
    "amber_l": "#fbbf24",
    "teal": "#06b6d4",
    "ok": "#84cc16",
    "danger": "#f43f5e",
    "warn": "#fb923c",
    "tx": "#f0f0f0",
    "tx2": "#8b95a5",
    "tiers": [
        "#f59e0b",
        "#fbbf24",
        "#84cc16",
        "#06b6d4",
        "#8b5cf6",
        "#f97316",
        "#f43f5e",
        "#6b7280",
    ],
}

ROSTER_CONFIG = {
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
    "BN": 5,
}

T = THEME  # shorthand for f-strings


# ── CSS Injection ────────────────────────────────────────────────────


def inject_custom_css():
    st.markdown(
        f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;500;600;700&family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ── BASE ─────────────────────────────────── */
    .stApp {{
        background: {T["bg"]};
        font-family: 'DM Sans', sans-serif;
        color: {T["tx"]};
    }}
    .block-container {{ padding-top: 1rem; padding-bottom: 1rem; }}

    /* ── SECTION HEADER ───────────────────────── */
    .sec-head {{
        font-family: 'Oswald', sans-serif;
        font-weight: 700;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 2.5px;
        color: {T["tx2"]};
        margin-bottom: 10px;
        padding-bottom: 6px;
        border-bottom: 1px solid {T["card_h"]};
    }}

    /* ── GLASS CARD ───────────────────────────── */
    .glass {{
        background: {T["card"]};
        border: 1px solid {T["card_h"]};
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }}
    .glass:hover {{
        background: {T["card_h"]};
        transform: translateY(-1px);
        transition: all 0.2s ease;
    }}

    /* ── COMMAND BAR ──────────────────────────── */
    .cmd-bar {{
        background: linear-gradient(135deg, {T["card"]} 0%, {T["bg"]} 100%);
        border: 1px solid {T["card_h"]};
        border-radius: 14px;
        padding: 14px 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 16px;
        flex-wrap: wrap;
        gap: 10px;
    }}
    .cmd-left {{
        font-family: 'Oswald', sans-serif;
        font-weight: 600;
        font-size: 18px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: {T["tx"]};
    }}
    .cmd-center {{ display: flex; align-items: center; gap: 12px; }}
    .cmd-right {{
        display: flex; align-items: center; gap: 10px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        color: {T["tx2"]};
    }}

    /* ── YOUR TURN BADGE ─────────────────────── */
    .your-turn {{
        background: {T["amber"]};
        color: {T["bg"]};
        font-family: 'Oswald', sans-serif;
        font-weight: 700;
        font-size: 16px;
        text-transform: uppercase;
        letter-spacing: 2px;
        padding: 8px 20px;
        border-radius: 8px;
        animation: amberGlow 2s ease-in-out infinite;
    }}
    .waiting {{
        background: {T["card_h"]};
        color: {T["tx2"]};
        font-family: 'Oswald', sans-serif;
        font-weight: 600;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        padding: 8px 16px;
        border-radius: 8px;
    }}
    @keyframes amberGlow {{
        0%, 100% {{ box-shadow: 0 0 8px {T["amber"]}66; }}
        50% {{ box-shadow: 0 0 24px {T["amber"]}cc, 0 0 48px {T["amber"]}44; }}
    }}

    /* ── PROGRESS BAR ────────────────────────── */
    .prog-track {{
        width: 120px; height: 6px;
        background: {T["card_h"]};
        border-radius: 3px;
        overflow: hidden;
    }}
    .prog-fill {{
        height: 100%;
        background: linear-gradient(90deg, {T["amber"]}, {T["amber_l"]});
        border-radius: 3px;
        transition: width 0.5s ease;
    }}

    /* ── HERO PICK CARD ──────────────────────── */
    .hero {{
        background: {T["card"]};
        border: 2px solid {T["amber"]};
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        position: relative;
        box-shadow: 0 0 30px {T["amber"]}22, 0 8px 32px rgba(0,0,0,0.4);
    }}
    .hero .p-name {{
        font-family: 'Oswald', sans-serif;
        font-weight: 700;
        font-size: 36px;
        color: {T["tx"]};
        text-transform: uppercase;
        letter-spacing: 1px;
        line-height: 1.1;
    }}
    .hero .p-meta {{
        font-family: 'DM Sans', sans-serif;
        font-size: 14px;
        color: {T["tx2"]};
        margin-top: 4px;
    }}
    .hero .score-badge {{
        position: absolute;
        top: 16px; right: 20px;
        background: {T["amber"]};
        color: {T["bg"]};
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 22px;
        padding: 8px 14px;
        border-radius: 10px;
    }}
    .hero .reason {{
        font-family: 'DM Sans', sans-serif;
        font-size: 15px;
        color: {T["amber_l"]};
        margin-top: 12px;
        font-style: italic;
    }}

    /* ── SURVIVAL GAUGE ──────────────────────── */
    .surv-gauge {{
        width: 64px; height: 64px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 16px;
        color: {T["tx"]};
    }}

    /* ── SGP CHIPS ───────────────────────────── */
    .sgp-row {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }}
    .sgp-chip {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        padding: 4px 10px;
        border-radius: 6px;
        border: 1px solid {T["card_h"]};
        background: {T["bg"]};
        color: {T["tx2"]};
    }}
    .sgp-chip.pos {{ border-color: {T["teal"]}; color: {T["teal"]}; }}
    .sgp-chip.neg {{ border-color: {T["danger"]}; color: {T["danger"]}; }}

    /* ── ALTERNATIVE CARDS ───────────────────── */
    .alt {{
        background: {T["card"]};
        border: 1px solid {T["card_h"]};
        border-left: 4px solid {T["card_h"]};
        border-radius: 10px;
        padding: 12px 14px;
        transition: all 0.2s ease;
        cursor: default;
    }}
    .alt:hover {{
        background: {T["card_h"]};
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }}
    .alt .a-rank {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: {T["tx2"]};
    }}
    .alt .a-name {{
        font-family: 'Oswald', sans-serif;
        font-weight: 600;
        font-size: 16px;
        color: {T["tx"]};
        text-transform: uppercase;
        margin: 2px 0;
    }}
    .alt .a-meta {{
        font-family: 'DM Sans', sans-serif;
        font-size: 12px;
        color: {T["tx2"]};
    }}
    .alt .a-score {{
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        font-size: 14px;
        color: {T["amber"]};
        margin-top: 4px;
    }}

    /* ── TIER BORDERS ────────────────────────── */
    .tier-1 {{ border-left-color: {T["tiers"][0]}; }}
    .tier-2 {{ border-left-color: {T["tiers"][1]}; }}
    .tier-3 {{ border-left-color: {T["tiers"][2]}; }}
    .tier-4 {{ border-left-color: {T["tiers"][3]}; }}
    .tier-5 {{ border-left-color: {T["tiers"][4]}; }}
    .tier-6 {{ border-left-color: {T["tiers"][5]}; }}
    .tier-7 {{ border-left-color: {T["tiers"][6]}; }}
    .tier-8 {{ border-left-color: {T["tiers"][7]}; }}

    /* ── BADGES ───────────────────────────────── */
    .badge {{
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .badge-value {{ background: {T["ok"]}22; color: {T["ok"]}; border: 1px solid {T["ok"]}44; }}
    .badge-reach {{ background: {T["danger"]}22; color: {T["danger"]}; border: 1px solid {T["danger"]}44; }}
    .badge-fair {{ background: {T["card_h"]}; color: {T["tx2"]}; }}
    .badge-risk-low {{ background: {T["ok"]}22; color: {T["ok"]}; }}
    .badge-risk-med {{ background: {T["warn"]}22; color: {T["warn"]}; }}
    .badge-risk-high {{ background: {T["danger"]}22; color: {T["danger"]}; }}

    /* ── ROSTER GRID ─────────────────────────── */
    .roster-grid {{
        display: grid;
        gap: 6px;
    }}
    .roster-slot {{
        background: {T["card"]};
        border: 1px solid {T["card_h"]};
        border-radius: 8px;
        padding: 6px 10px;
        text-align: center;
        min-height: 48px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}
    .roster-slot.filled {{
        border-color: {T["ok"]}66;
        background: {T["card_h"]};
    }}
    .roster-slot.empty {{
        border-style: dashed;
        border-color: {T["tx2"]}44;
    }}
    .roster-slot .s-label {{
        font-family: 'Oswald', sans-serif;
        font-size: 10px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: {T["tx2"]};
    }}
    .roster-slot .s-player {{
        font-family: 'DM Sans', sans-serif;
        font-size: 12px;
        font-weight: 600;
        color: {T["tx"]};
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}

    /* ── SCARCITY RINGS ──────────────────────── */
    .scar-wrap {{
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 4px;
    }}
    .scar-ring {{
        width: 52px; height: 52px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        font-size: 13px;
        color: {T["tx"]};
    }}
    .scar-ring.critical {{
        animation: scarPulse 1.5s ease-in-out infinite;
    }}
    @keyframes scarPulse {{
        0%, 100% {{ transform: scale(1); }}
        50% {{ transform: scale(1.08); }}
    }}
    .scar-label {{
        font-family: 'Oswald', sans-serif;
        font-size: 10px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: {T["tx2"]};
    }}

    /* ── DRAFT BOARD ─────────────────────────── */
    .draft-board {{
        width: 100%;
        border-collapse: collapse;
        font-family: 'DM Sans', sans-serif;
        font-size: 11px;
    }}
    .draft-board th {{
        background: {T["card"]};
        color: {T["tx2"]};
        font-family: 'Oswald', sans-serif;
        font-weight: 600;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        padding: 8px 6px;
        border-bottom: 2px solid {T["card_h"]};
        position: sticky;
        top: 0;
        z-index: 10;
    }}
    .draft-board td {{
        padding: 5px 6px;
        border-bottom: 1px solid {T["card_h"]}44;
        color: {T["tx2"]};
    }}
    .draft-board .user-col {{
        background: {T["amber"]}11;
        border-left: 2px solid {T["amber"]};
        border-right: 2px solid {T["amber"]};
    }}
    .draft-board .user-col th {{
        background: {T["amber"]};
        color: {T["bg"]};
    }}
    .draft-board .current-pick {{
        border: 2px solid {T["amber"]};
        animation: amberGlow 2s ease-in-out infinite;
    }}
    .draft-board .picked {{
        color: {T["tx"]};
        font-weight: 600;
    }}

    /* ── CATEGORY CARDS ──────────────────────── */
    .cat-card {{
        background: {T["card"]};
        border: 1px solid {T["card_h"]};
        border-radius: 8px;
        padding: 10px 12px;
        text-align: center;
    }}
    .cat-name {{
        font-family: 'Oswald', sans-serif;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: {T["tx2"]};
    }}
    .cat-val {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 18px;
        font-weight: 700;
        color: {T["tx"]};
        margin-top: 2px;
    }}

    /* ── ALERTS, FEED, WIZARD ────────────────── */
    .alert-card {{
        background: {T["card"]};
        border-left: 4px solid {T["warn"]};
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 13px;
        color: {T["tx"]};
    }}
    .alert-card.critical {{ border-left-color: {T["danger"]}; }}

    .feed-card {{
        background: {T["card"]};
        border: 1px solid {T["card_h"]};
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 6px;
        font-size: 13px;
    }}
    .feed-card.user-pick {{
        border-left: 3px solid {T["amber"]};
    }}
    .feed-pick-num {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: {T["tx2"]};
    }}
    .feed-name {{
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        color: {T["tx"]};
    }}
    .feed-team {{
        font-size: 11px;
        color: {T["tx2"]};
    }}

    .wizard-bar {{
        display: flex;
        justify-content: center;
        gap: 0;
        margin-bottom: 24px;
    }}
    .wizard-step {{
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 10px 20px;
        font-family: 'Oswald', sans-serif;
        font-weight: 600;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: {T["tx2"]};
        background: {T["card"]};
        border: 1px solid {T["card_h"]};
    }}
    .wizard-step:first-child {{ border-radius: 8px 0 0 8px; }}
    .wizard-step:last-child {{ border-radius: 0 8px 8px 0; }}
    .wizard-step.active {{
        background: {T["amber"]};
        color: {T["bg"]};
        border-color: {T["amber"]};
    }}
    .wizard-step.done {{
        background: {T["ok"]}22;
        color: {T["ok"]};
        border-color: {T["ok"]}44;
    }}

    /* ── LOCK-IN BANNER ──────────────────────── */
    .lock-in {{
        background: linear-gradient(135deg, {T["ok"]}33, {T["ok"]}11);
        border: 1px solid {T["ok"]}66;
        border-radius: 10px;
        padding: 12px 20px;
        text-align: center;
        font-family: 'Oswald', sans-serif;
        font-weight: 600;
        font-size: 16px;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: {T["ok"]};
        animation: lockFade 3s ease-out forwards;
    }}
    @keyframes lockFade {{
        0% {{ opacity: 1; }}
        70% {{ opacity: 1; }}
        100% {{ opacity: 0; }}
    }}

    /* ── STREAMLIT OVERRIDES ─────────────────── */
    .stButton > button {{
        font-family: 'Oswald', sans-serif;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        border-radius: 8px;
        transition: all 0.2s ease;
    }}
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {{
        background: {T["amber"]};
        color: {T["bg"]};
        border: none;
    }}
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {{
        background: {T["amber_l"]};
        box-shadow: 0 4px 16px {T["amber"]}44;
    }}
    .stButton > button[kind="secondary"],
    .stButton > button[data-testid="stBaseButton-secondary"] {{
        background: {T["card"]};
        color: {T["tx"]};
        border: 1px solid {T["card_h"]};
    }}
    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] > div {{
        background: {T["card"]} !important;
        color: {T["tx"]} !important;
        border-color: {T["card_h"]} !important;
        border-radius: 8px !important;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        background: {T["card"]};
        border-radius: 10px;
        padding: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-family: 'Oswald', sans-serif;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 13px;
        border-radius: 8px;
        color: {T["tx2"]};
    }}
    .stTabs [aria-selected="true"] {{
        background: {T["amber"]} !important;
        color: {T["bg"]} !important;
    }}
    div[data-testid="stFileUploader"] {{
        background: {T["card"]};
        border: 2px dashed {T["card_h"]};
        border-radius: 12px;
        padding: 16px;
    }}
    div[data-testid="stFileUploader"]:hover {{
        border-color: {T["amber"]};
    }}
    div[data-testid="stDataFrame"] {{
        border: 1px solid {T["card_h"]};
        border-radius: 10px;
    }}
    .stSidebar {{
        background: {T["card"]} !important;
        border-right: 1px solid {T["card_h"]} !important;
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )


# ── Session Init ─────────────────────────────────────────────────────


def init_session():
    defaults = {
        "page": "setup",
        "setup_step": 1,
        "hitter_data": None,
        "pitcher_data": None,
        "player_pool": None,
        "draft_state": None,
        "league_config": None,
        "sgp_calc": None,
        "practice_mode": True,
        "auto_sgp": True,
        "risk_tolerance": 0.5,
        "num_sims": 100,
        "last_lock_in": 0,
        "last_drafted": "",
        "yahoo_client": None,
        "last_yahoo_sync": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Helper: section header ───────────────────────────────────────────


def sec(title):
    st.markdown(f'<div class="sec-head">{title}</div>', unsafe_allow_html=True)


# ── Wizard Progress Bar ─────────────────────────────────────────────

WIZARD_STEPS = ["Import", "Settings", "Connect", "Launch"]


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
        f'<div style="text-align:center;margin-bottom:8px;">'
        f'<span style="font-family:Oswald,sans-serif;font-weight:700;font-size:32px;'
        f'color:{T["amber"]};text-transform:uppercase;letter-spacing:3px;">'
        f"Draft Command Center</span></div>",
        unsafe_allow_html=True,
    )
    step = st.session_state.setup_step
    render_wizard_progress(step)

    if step == 1:
        render_step_import()
    elif step == 2:
        render_step_league()
    elif step == 3:
        render_step_connect()
    elif step == 4:
        render_step_launch()


# ── Step 1: Import ──────────────────────────────────────────────────


def render_step_import():
    sec("Step 1 — Import Your Data")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f'<div class="glass"><div style="font-family:Oswald,sans-serif;'
            f"font-weight:600;font-size:16px;color:{T['tx']};text-transform:uppercase;"
            f'letter-spacing:1px;margin-bottom:8px;">Hitter Projections</div></div>',
            unsafe_allow_html=True,
        )
        hitter_file = st.file_uploader("Upload hitter CSV", type=["csv"], key="hitter_upload")
        if hitter_file:
            try:
                init_db()
                import_hitter_csv(hitter_file)
                st.session_state.hitter_data = True
                st.toast("Hitters imported!", icon="✅")
            except Exception as e:
                st.error(f"Import error: {e}")

    with col2:
        st.markdown(
            f'<div class="glass"><div style="font-family:Oswald,sans-serif;'
            f"font-weight:600;font-size:16px;color:{T['tx']};text-transform:uppercase;"
            f'letter-spacing:1px;margin-bottom:8px;">Pitcher Projections</div></div>',
            unsafe_allow_html=True,
        )
        pitcher_file = st.file_uploader("Upload pitcher CSV", type=["csv"], key="pitcher_upload")
        if pitcher_file:
            try:
                init_db()
                import_pitcher_csv(pitcher_file)
                st.session_state.pitcher_data = True
                st.toast("Pitchers imported!", icon="✅")
            except Exception as e:
                st.error(f"Import error: {e}")

    # ADP
    st.markdown("---")
    adp_file = st.file_uploader("Upload ADP data (optional)", type=["csv"], key="adp_upload")
    if adp_file:
        try:
            import_adp_csv(adp_file)
            st.toast("ADP imported!", icon="✅")
        except Exception as e:
            st.error(f"ADP import error: {e}")

    # Sample data
    st.markdown("---")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button("Load Sample Data (for testing)", use_container_width=True):
            with st.status("Loading sample data..."):
                try:
                    import importlib

                    import load_sample_data

                    importlib.reload(load_sample_data)
                    load_sample_data.generate_sample_data()
                    # Verify data persisted
                    import sqlite3

                    from src.database import DB_PATH

                    _vconn = sqlite3.connect(str(DB_PATH))
                    _vc = _vconn.cursor()
                    _vc.execute("SELECT COUNT(*) FROM projections")
                    _proj_count = _vc.fetchone()[0]
                    _vc.execute("SELECT COUNT(*) FROM players")
                    _player_count = _vc.fetchone()[0]
                    _vconn.close()
                    if _proj_count > 0:
                        st.session_state.hitter_data = True
                        st.session_state.pitcher_data = True
                except Exception as e:
                    st.error(f"Sample data verification failed: {e}")
            st.rerun()

    # Status chips
    chips = []
    if st.session_state.hitter_data:
        chips.append('<span class="badge badge-value">Hitters ✓</span>')
    if st.session_state.pitcher_data:
        chips.append('<span class="badge badge-value">Pitchers ✓</span>')
    if chips:
        st.markdown(" ".join(chips), unsafe_allow_html=True)

    # Navigation
    st.markdown("")
    c1, c2, c3 = st.columns([2, 1, 2])
    with c2:
        if st.button("Next →", type="primary", use_container_width=True):
            if st.session_state.hitter_data and st.session_state.pitcher_data:
                # Blend projections (skip if only blended exists, e.g. sample data)
                with st.status("Blending projections..."):
                    init_db()
                    import sqlite3 as _sql

                    from src.database import DB_PATH as _dbp

                    _tc = _sql.connect(str(_dbp))
                    _non_blended = _tc.execute("SELECT COUNT(*) FROM projections WHERE system != 'blended'").fetchone()[
                        0
                    ]
                    _tc.close()
                    if _non_blended > 0:
                        create_blended_projections()
                st.session_state.setup_step = 2
                st.rerun()
            else:
                st.warning("Import both hitter and pitcher data first.")


# ── Step 2: League Settings ─────────────────────────────────────────


def render_step_league():
    sec("Step 2 — League Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("**SGP Denominators**")
        auto_sgp = st.toggle("Auto-compute SGP", value=st.session_state.auto_sgp, key="auto_sgp_toggle")
        st.session_state.auto_sgp = auto_sgp

        if not auto_sgp:
            sgp_r = st.number_input("R", value=32.0, step=1.0, key="sgp_r")
            sgp_hr = st.number_input("HR", value=12.0, step=1.0, key="sgp_hr")
            sgp_rbi = st.number_input("RBI", value=30.0, step=1.0, key="sgp_rbi")
            sgp_sb = st.number_input("SB", value=8.0, step=1.0, key="sgp_sb")
            sgp_avg = st.number_input("AVG", value=0.008, step=0.001, format="%.4f", key="sgp_avg")
            sgp_w = st.number_input("W", value=3.0, step=1.0, key="sgp_w")
            sgp_sv = st.number_input("SV", value=7.0, step=1.0, key="sgp_sv")
            sgp_k = st.number_input("K", value=25.0, step=1.0, key="sgp_k")
            sgp_era = st.number_input("ERA", value=0.30, step=0.01, format="%.3f", key="sgp_era")
            sgp_whip = st.number_input("WHIP", value=0.03, step=0.01, format="%.3f", key="sgp_whip")
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
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("← Back", use_container_width=True):
            st.session_state.setup_step = 1
            st.rerun()
    with c3:
        if st.button("Next →", type="primary", use_container_width=True):
            st.session_state.setup_step = 3
            st.rerun()


# ── Step 3: Connect Yahoo (Optional) ───────────────────────────────


def render_step_connect():
    sec("Step 3 — Connect Yahoo (Optional)")

    st.markdown(
        f'<div class="glass" style="text-align:center;padding:24px;">'
        f'<div style="font-family:Oswald,sans-serif;font-size:18px;color:{T["tx"]};'
        f'text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;">Yahoo Fantasy API</div>'
        f'<div style="color:{T["tx2"]};font-size:14px;margin-bottom:16px;">'
        f"Connect to auto-sync draft picks and league settings. You can skip this.</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if has_credentials():
        st.markdown('<span class="badge badge-value">Credentials Found ✓</span>', unsafe_allow_html=True)
        creds = load_credentials()
        st.text_input("Consumer Key", value=creds.get("consumer_key", ""), key="yk", type="password")
        st.text_input("Consumer Secret", value=creds.get("consumer_secret", ""), key="ys", type="password")
    else:
        st.text_input("Consumer Key", key="yk", type="password")
        st.text_input("Consumer Secret", key="ys", type="password")

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("Save & Test Connection", use_container_width=True):
            yk = st.session_state.get("yk", "")
            ys = st.session_state.get("ys", "")
            if yk and ys:
                save_credentials(yk, ys)
                try:
                    client = YahooFantasyClient()
                    st.session_state.yahoo_client = client
                    st.toast("Connected to Yahoo!", icon="✅")
                except Exception as e:
                    st.error(f"Connection failed: {e}")
            else:
                st.warning("Enter both keys.")

    # Nav
    st.markdown("")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("← Back", use_container_width=True, key="s3_back"):
            st.session_state.setup_step = 2
            st.rerun()
    with c2:
        if st.button("Skip →", use_container_width=True, key="s3_skip"):
            st.session_state.setup_step = 4
            st.rerun()
    with c3:
        if st.button("Next →", type="primary", use_container_width=True, key="s3_next"):
            st.session_state.setup_step = 4
            st.rerun()


# ── Step 4: Launch ──────────────────────────────────────────────────


def render_step_launch():
    sec("Step 4 — Ready to Launch")

    # Build league config + player pool
    pool = _build_player_pool()
    if pool is None:
        st.error("No player data found. Go back to Step 1.")
        if st.button("← Back to Import"):
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
        icon = "✅" if ok else "❌"
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
        if st.button("START DRAFT", type="primary", use_container_width=True, disabled=not all_ok):
            _start_new_draft(pool, practice, resume)

    # Back
    if st.button("← Back", key="s4_back"):
        st.session_state.setup_step = 3
        st.rerun()


def _build_player_pool():
    """Load player pool and run valuation pipeline."""
    try:
        init_db()
        pool = load_player_pool()
        if pool is None or pool.empty:
            return None

        # Build league config
        num_teams = st.session_state.get("num_teams", 12) if hasattr(st.session_state, "num_teams") else 12
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
    num_teams = st.session_state.get("num_teams", 12) if hasattr(st.session_state, "num_teams") else 12
    num_rounds = st.session_state.get("num_rounds", 23) if hasattr(st.session_state, "num_rounds") else 23
    draft_pos = st.session_state.get("draft_pos", 1) if hasattr(st.session_state, "draft_pos") else 1

    if resume:
        try:
            ds = DraftState.load(roster_config=ROSTER_CONFIG)
            st.toast("Resumed saved draft!", icon="✅")
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

    if ds is None or pool is None:
        st.error("Draft not initialized. Go back to setup.")
        if st.button("← Setup"):
            st.session_state.page = "setup"
            st.rerun()
        return

    inject_custom_css()

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

        if st.button("Undo Last Pick", use_container_width=True):
            ds.undo_last_pick()
            st.toast("Pick undone!", icon="↩️")
            st.rerun()

        if st.button("Save Draft", use_container_width=True):
            path = ds.save()
            st.toast("Saved!", icon="💾")

        st.markdown("---")

        # Yahoo sync
        if st.session_state.yahoo_client:
            if st.button("Sync Yahoo", use_container_width=True):
                try:
                    sync_draft_picks(st.session_state.yahoo_client, ds, pool)
                    st.session_state.last_yahoo_sync = time.time()
                    st.toast("Yahoo synced!", icon="🔄")
                    st.rerun()
                except Exception as e:
                    st.error(f"Sync error: {e}")

        st.markdown("---")
        if st.button("← Back to Setup", use_container_width=True):
            st.session_state.page = "setup"
            st.rerun()

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
        st.markdown(f'<div class="lock-in">✓ {st.session_state.last_drafted} — Locked In</div>', unsafe_allow_html=True)

    # ── Run recommendation engine ────────────────────────────────
    available = ds.available_players(pool)
    rec = None
    alts = []

    if ds.is_user_turn and len(available) > 0:
        with st.status("Analyzing best picks...", expanded=False) as status:
            try:
                sim = DraftSimulator(lc)
                candidates = sim.evaluate_candidates(pool, ds, n_simulations=st.session_state.num_sims)

                if candidates is not None and len(candidates) > 0:
                    rec = candidates.iloc[0]
                    alts = candidates.iloc[1:6] if len(candidates) > 1 else pd.DataFrame()

                # Position run detection
                pos_runs = detect_position_run(ds.pick_log)

                status.update(label="Ready", state="complete")
            except Exception as e:
                st.error(f"Engine error: {e}")
                status.update(label="Error", state="error")

    # ── 3-Column Layout ──────────────────────────────────────────
    left, center, right = st.columns([1.2, 3, 1.3])

    with left:
        render_roster_panel(ds, pool)
        render_scarcity_rings(ds, pool)

    with center:
        if ds.is_user_turn:
            if rec is not None:
                render_hero_pick(rec, ds, pool)
                if len(alts) > 0:
                    render_alternatives(alts)
            else:
                st.info("No recommendation available.")

            # Position run alerts
            if "pos_runs" in dir() and pos_runs:
                for pos, info in pos_runs.items():
                    severity = "critical" if info.get("severity", 0) > 0.7 else ""
                    st.markdown(
                        f'<div class="alert-card {severity}">'
                        f'<strong style="color:{T["warn"]};">⚡ {pos} Run!</strong> '
                        f"{info.get('message', 'Position being drafted heavily')}</div>",
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


def render_hero_pick(rec, ds, pool):
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
            vb = '<span class="badge badge-value">Value</span>'
        elif diff < -10:
            vb = '<span class="badge badge-reach">Reach</span>'
        else:
            vb = '<span class="badge badge-fair">Fair</span>'
    else:
        vb = ""

    # Tier
    tier = int(rec.get("tier", 1))
    tier = max(1, min(tier, 8))

    st.markdown(
        f'<div class="hero">'
        f'<div class="score-badge">{score:.1f}</div>'
        f'<div style="display:flex;align-items:center;gap:16px;">'
        f'<div class="surv-gauge" style="background:conic-gradient({surv_color} {surv_deg}deg, '
        f'{T["card_h"]} {surv_deg}deg);">'
        f'<div style="width:48px;height:48px;border-radius:50%;background:{T["card"]};'
        f'display:flex;align-items:center;justify-content:center;">{surv:.0f}%</div></div>'
        f"<div>"
        f'<div class="p-name">{name}</div>'
        f'<div class="p-meta">{pos} &middot; Tier {tier} {vb}</div>'
        f"</div></div>"
        f'<div class="reason">{reason}</div>'
        f'<div class="sgp-row">{chips_html}</div>'
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
                risk_badge = '<span class="badge badge-risk-low">Low Risk</span>'
            elif risk < 0.7:
                risk_badge = '<span class="badge badge-risk-med">Med Risk</span>'
            else:
                risk_badge = '<span class="badge badge-risk-high">High Risk</span>'

            st.markdown(
                f'<div class="alt tier-{tier}">'
                f'<div class="a-rank">#{i + 2}</div>'
                f'<div class="a-name">{name}</div>'
                f'<div class="a-meta">{pos} {risk_badge}</div>'
                f'<div class="a-score">{score:.1f}</div>'
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
                if st.button(btn_label, type="primary", use_container_width=True, key="draft_btn"):
                    _execute_pick(ds, p, pool)
            else:
                # Other team's pick
                team_idx = ds.picking_team_index()
                team_name = ds.teams[team_idx].team_name
                if st.button(f"Record for {team_name}", use_container_width=True, key="draft_btn"):
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
    ds.save()
    st.toast(f"Drafted {player_row['player_name']}!", icon="✅")
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
                st.toast(f"⚠️ {pos} scarce! Only {avail_count} left", icon="🚨")


# ═══════════════════════════════════════════════════════════════════
# DRAFT TABS
# ═══════════════════════════════════════════════════════════════════


def render_draft_tabs(ds, pool, lc, sgp):
    tabs = st.tabs(["Category Balance", "Available Players", "Draft Board", "Draft Log"])

    with tabs[0]:
        render_category_balance(ds, pool)

    with tabs[1]:
        render_available_players(ds, pool)

    with tabs[2]:
        render_draft_board(ds)

    with tabs[3]:
        render_draft_log(ds)


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
    st.plotly_chart(fig, use_container_width=True)


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
            if st.button(pos, key=f"pill_{pos}", type=btn_type, use_container_width=True):
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

    st.dataframe(display_df, use_container_width=True, hide_index=True, column_config=col_config, height=400)


# ── Draft Board ─────────────────────────────────────────────────────


def render_draft_board(ds):
    # Build HTML table
    headers = ""
    for i in range(ds.num_teams):
        team_name = ds.teams[i].team_name
        cls = 'class="user-col"' if i == ds.user_team_index else ""
        short_name = team_name[:8]
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
                short = name[:12] if len(name) > 12 else name
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

    if st.session_state.page == "setup":
        render_setup_page()
    elif st.session_state.page == "draft":
        render_draft_page()


if __name__ == "__main__":
    main()

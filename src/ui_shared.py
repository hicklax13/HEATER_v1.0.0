"""Shared UI constants, theme system, and CSS injection for all pages."""

import streamlit as st

# ── Theme Dictionaries ──────────────────────────────────────────────

DARK_THEME = {
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
    "border": "#1a1f2e",
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

LIGHT_THEME = {
    "bg": "#f8f9fc",
    "card": "#ffffff",
    "card_h": "#e2e4ea",
    "amber": "#f59e0b",
    "amber_l": "#fbbf24",
    "teal": "#06b6d4",
    "ok": "#84cc16",
    "danger": "#f43f5e",
    "warn": "#fb923c",
    "tx": "#1a1a2e",
    "tx2": "#5a6170",
    "border": "#1a1f2e",
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

# Backward compatibility: THEME always points to dark (import-time default)
THEME = DARK_THEME

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

HITTING_CATEGORIES = ["R", "HR", "RBI", "SB", "AVG"]
PITCHING_CATEGORIES = ["W", "SV", "K", "ERA", "WHIP"]
ALL_CATEGORIES = HITTING_CATEGORIES + PITCHING_CATEGORIES

# ── Metric Tooltips ─────────────────────────────────────────────────

METRIC_TOOLTIPS = {
    "sgp": (
        "Standings Gain Points (SGP) — how many roto standings points "
        "this stat is worth. Higher = more impact on your league rank."
    ),
    "vorp": (
        "Value Over Replacement Player — how much better this player is "
        "than the best freely available option at their position."
    ),
    "pick_score": (
        "Combined draft value score — factors in projected stats, "
        "positional scarcity, and how likely the player is to still "
        "be available on your next pick."
    ),
    "survival": (
        "Survival Probability — the percent chance this player will "
        "still be available when you pick next. Lower % = more urgency "
        "to draft now."
    ),
    "health_green": ("Low Risk — consistently healthy (90%+ games played over 3 seasons). No stat discount applied."),
    "health_yellow": (
        "Moderate Risk — some injury history (75-89% games played). Counting stats are adjusted down slightly."
    ),
    "health_red": (
        "High Risk — significant injury history (under 75% games played). Counting stats are heavily discounted."
    ),
    "trade_verdict": (
        "Trade verdict based on 200 Monte Carlo simulations. ACCEPT "
        "means your team improves on average. Confidence % shows how "
        "often the trade helps across all simulations."
    ),
    "mc_mean": (
        "Monte Carlo Mean — the average SGP change across 200 simulated seasons. Positive = trade helps your team."
    ),
    "mc_std": (
        "Monte Carlo Std Dev — how much the outcome varies across "
        "simulations. Lower = more predictable; higher = riskier trade."
    ),
    "z_score": (
        "Z-Score — how many standard deviations above or below "
        "league average. +1.0 = top ~16% of players; -1.0 = bottom ~16%."
    ),
    "marginal_value": (
        "Marginal Value — how much this free agent would improve your "
        "team's SGP compared to your current worst player at the position."
    ),
    "cat_targeting": (
        "Category Targeting Weight — higher weight means bigger "
        "standings impact. Higher = more room to gain in standings."
    ),
    "p10_p90": (
        "P10/P90 Range — the floor (10th percentile) and ceiling "
        "(90th percentile) projections. Wider range = more uncertain."
    ),
    "composite_score": (
        "Composite Score — weighted sum of Z-scores across all 10 "
        "fantasy categories. Higher = better overall fantasy value."
    ),
    "era": (
        "Earned Run Average — average earned runs allowed per 9 "
        "innings pitched. LOWER is better. League average is ~4.00."
    ),
    "whip": (
        "Walks + Hits per Inning Pitched — baserunners allowed per inning. LOWER is better. League average is ~1.25."
    ),
    "avg": ("Batting Average — hits divided by at-bats. Higher is better. League average is ~.250."),
    "adp_value": (
        "Value pick — this player's ADP is later than the current "
        "pick, meaning you are getting them earlier than most drafts."
    ),
    "adp_reach": (
        "Reach pick — this player's ADP is earlier than the current "
        "pick, meaning you are drafting them before most leagues would."
    ),
    "adp_fair": (
        "Fair pick — this player's ADP is close to the current pick, "
        "meaning this is roughly where they are expected to go."
    ),
}


# ── Theme Functions ─────────────────────────────────────────────────


def get_theme():
    """Return the active theme dict based on session state."""
    try:
        mode = st.session_state.get("theme_mode", "dark")
    except Exception:
        mode = "dark"
    return DARK_THEME if mode == "dark" else LIGHT_THEME


def render_theme_toggle():
    """Render a dark/light mode toggle in the sidebar."""
    try:
        with st.sidebar:
            current_mode = st.session_state.get("theme_mode", "dark")
            is_dark = current_mode == "dark"
            label = "🌙 Dark Mode" if is_dark else "☀️ Light Mode"
            new_is_dark = st.toggle(label, value=is_dark, key="_theme_toggle")
            new_mode = "dark" if new_is_dark else "light"
            if new_mode != current_mode:
                st.session_state["theme_mode"] = new_mode
                st.rerun()
    except Exception:
        pass


# ── Section Header Helper ──────────────────────────────────────────


def sec(title):
    """Render a styled section header (uppercase, Oswald font)."""
    st.markdown(f'<div class="sec-head">{title}</div>', unsafe_allow_html=True)


# ── Plotly Theme Helpers ───────────────────────────────────────────


def get_plotly_layout(theme=None):
    """Return Plotly layout kwargs for consistent chart theming."""
    t = theme or get_theme()
    return {
        "paper_bgcolor": t["bg"],
        "plot_bgcolor": t["card"],
        "font": {"color": t["tx"], "family": "DM Sans, sans-serif"},
        "margin": {"l": 60, "r": 60, "t": 40, "b": 40},
        "height": 350,
    }


def get_plotly_polar(theme=None):
    """Return polar/radar chart config for Player Compare."""
    t = theme or get_theme()
    return {
        "bgcolor": t["card"],
        "radialaxis": {
            "gridcolor": t["card_h"],
            "tickfont": {"color": t["tx2"], "size": 10},
            "visible": True,
        },
        "angularaxis": {
            "gridcolor": t["card_h"],
            "tickfont": {"color": t["tx"], "size": 11},
        },
    }


# ── CSS Injection ──────────────────────────────────────────────────


def inject_custom_css():
    """Inject the complete shared CSS for all pages.

    Reads the active theme (dark/light) and injects all styles.
    Call this once at the top of every page, after st.set_page_config().
    """
    t = get_theme()
    st.markdown(
        f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;500;600;700&family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ── BASE ─────────────────────────────────── */
    .stApp {{
        background: {t["bg"]};
        font-family: 'DM Sans', sans-serif;
        color: {t["tx"]};
    }}
    .block-container {{ padding-top: 1rem; padding-bottom: 1rem; }}

    /* ── TEXT PROTECTION (global) ─────────────── */
    .stApp p, .stApp span, .stApp td, .stApp th,
    .stApp label, .stApp .stMarkdown {{
        overflow-wrap: anywhere;
        word-break: break-word;
    }}
    h1, h2, h3 {{
        color: {t["amber"]};
        font-family: 'Oswald', sans-serif;
        text-transform: uppercase;
    }}

    /* ── SECTION HEADER ───────────────────────── */
    .sec-head {{
        font-family: 'Oswald', sans-serif;
        font-weight: 700;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 2.5px;
        color: {t["tx2"]};
        margin-bottom: 10px;
        padding-bottom: 6px;
        border-bottom: 1px solid {t["card_h"]};
    }}

    /* ── GLASS CARD ───────────────────────────── */
    .glass {{
        background: {t["card"]};
        border: 1px solid {t["card_h"]};
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }}
    .glass:hover {{
        background: {t["card_h"]};
        transform: translateY(-1px);
        transition: all 0.2s ease;
    }}

    /* ── COMMAND BAR ──────────────────────────── */
    .cmd-bar {{
        background: linear-gradient(135deg, {t["card"]} 0%, {t["bg"]} 100%);
        border: 1px solid {t["card_h"]};
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
        color: {t["tx"]};
    }}
    .cmd-center {{ display: flex; align-items: center; gap: 12px; }}
    .cmd-right {{
        display: flex; align-items: center; gap: 10px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        color: {t["tx2"]};
    }}

    /* ── YOUR TURN BADGE ─────────────────────── */
    .your-turn {{
        background: {t["amber"]};
        color: {t["bg"]};
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
        background: {t["card_h"]};
        color: {t["tx2"]};
        font-family: 'Oswald', sans-serif;
        font-weight: 600;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        padding: 8px 16px;
        border-radius: 8px;
    }}
    @keyframes amberGlow {{
        0%, 100% {{ box-shadow: 0 0 8px {t["amber"]}66; }}
        50% {{ box-shadow: 0 0 24px {t["amber"]}cc, 0 0 48px {t["amber"]}44; }}
    }}

    /* ── PROGRESS BAR ────────────────────────── */
    .prog-track {{
        width: 120px; height: 6px;
        background: {t["card_h"]};
        border-radius: 3px;
        overflow: hidden;
    }}
    .prog-fill {{
        height: 100%;
        background: linear-gradient(90deg, {t["amber"]}, {t["amber_l"]});
        border-radius: 3px;
        transition: width 0.5s ease;
    }}

    /* ── HERO PICK CARD ──────────────────────── */
    .hero {{
        background: {t["card"]};
        border: 2px solid {t["amber"]};
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        position: relative;
        box-shadow: 0 0 30px {t["amber"]}22, 0 8px 32px rgba(0,0,0,0.4);
    }}
    .hero .p-name {{
        font-family: 'Oswald', sans-serif;
        font-weight: 700;
        font-size: 36px;
        color: {t["tx"]};
        text-transform: uppercase;
        letter-spacing: 1px;
        line-height: 1.1;
        word-break: break-word;
    }}
    .hero .p-meta {{
        font-family: 'DM Sans', sans-serif;
        font-size: 14px;
        color: {t["tx2"]};
        margin-top: 4px;
    }}
    .hero .score-badge {{
        position: absolute;
        top: 16px; right: 20px;
        background: {t["amber"]};
        color: {t["bg"]};
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 22px;
        padding: 8px 14px;
        border-radius: 10px;
        cursor: help;
    }}
    .hero .reason {{
        font-family: 'DM Sans', sans-serif;
        font-size: 15px;
        color: {t["amber_l"]};
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
        color: {t["tx"]};
    }}

    /* ── SGP CHIPS ───────────────────────────── */
    .sgp-row {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }}
    .sgp-chip {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        padding: 4px 10px;
        border-radius: 6px;
        border: 1px solid {t["card_h"]};
        background: {t["bg"]};
        color: {t["tx2"]};
    }}
    .sgp-chip.pos {{ border-color: {t["teal"]}; color: {t["teal"]}; }}
    .sgp-chip.neg {{ border-color: {t["danger"]}; color: {t["danger"]}; }}

    /* ── ALTERNATIVE CARDS ───────────────────── */
    .alt {{
        background: {t["card"]};
        border: 1px solid {t["card_h"]};
        border-left: 4px solid {t["card_h"]};
        border-radius: 10px;
        padding: 12px 14px;
        transition: all 0.2s ease;
        cursor: default;
    }}
    .alt:hover {{
        background: {t["card_h"]};
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }}
    .alt .a-rank {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: {t["tx2"]};
    }}
    .alt .a-name {{
        font-family: 'Oswald', sans-serif;
        font-weight: 600;
        font-size: 16px;
        color: {t["tx"]};
        text-transform: uppercase;
        margin: 2px 0;
        word-break: break-word;
    }}
    .alt .a-meta {{
        font-family: 'DM Sans', sans-serif;
        font-size: 12px;
        color: {t["tx2"]};
    }}
    .alt .a-score {{
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        font-size: 14px;
        color: {t["amber"]};
        margin-top: 4px;
    }}

    /* ── TIER BORDERS ────────────────────────── */
    .tier-1 {{ border-left-color: {t["tiers"][0]}; }}
    .tier-2 {{ border-left-color: {t["tiers"][1]}; }}
    .tier-3 {{ border-left-color: {t["tiers"][2]}; }}
    .tier-4 {{ border-left-color: {t["tiers"][3]}; }}
    .tier-5 {{ border-left-color: {t["tiers"][4]}; }}
    .tier-6 {{ border-left-color: {t["tiers"][5]}; }}
    .tier-7 {{ border-left-color: {t["tiers"][6]}; }}
    .tier-8 {{ border-left-color: {t["tiers"][7]}; }}

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
    .badge-value {{ background: {t["ok"]}22; color: {t["ok"]}; border: 1px solid {t["ok"]}44; }}
    .badge-reach {{ background: {t["danger"]}22; color: {t["danger"]}; border: 1px solid {t["danger"]}44; }}
    .badge-fair {{ background: {t["card_h"]}; color: {t["tx2"]}; }}
    .badge-risk-low {{ background: {t["ok"]}22; color: {t["ok"]}; }}
    .badge-risk-med {{ background: {t["warn"]}22; color: {t["warn"]}; }}
    .badge-risk-high {{ background: {t["danger"]}22; color: {t["danger"]}; }}

    /* ── ROSTER GRID ─────────────────────────── */
    .roster-grid {{
        display: grid;
        gap: 6px;
    }}
    .roster-slot {{
        background: {t["card"]};
        border: 1px solid {t["card_h"]};
        border-radius: 8px;
        padding: 6px 10px;
        text-align: center;
        min-height: 48px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}
    .roster-slot.filled {{
        border-color: {t["ok"]}66;
        background: {t["card_h"]};
    }}
    .roster-slot.empty {{
        border-style: dashed;
        border-color: {t["tx2"]}44;
    }}
    .roster-slot .s-label {{
        font-family: 'Oswald', sans-serif;
        font-size: 10px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: {t["tx2"]};
    }}
    .roster-slot .s-player {{
        font-family: 'DM Sans', sans-serif;
        font-size: 12px;
        font-weight: 600;
        color: {t["tx"]};
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
        color: {t["tx"]};
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
        color: {t["tx2"]};
    }}

    /* ── DRAFT BOARD ─────────────────────────── */
    .draft-board {{
        width: 100%;
        border-collapse: collapse;
        font-family: 'DM Sans', sans-serif;
        font-size: 11px;
    }}
    .draft-board th {{
        background: {t["card"]};
        color: {t["tx2"]};
        font-family: 'Oswald', sans-serif;
        font-weight: 600;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        padding: 8px 6px;
        border-bottom: 2px solid {t["card_h"]};
        position: sticky;
        top: 0;
        z-index: 10;
    }}
    .draft-board td {{
        padding: 5px 6px;
        border-bottom: 1px solid {t["card_h"]}44;
        color: {t["tx2"]};
        max-width: 120px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }}
    .draft-board .user-col {{
        background: {t["amber"]}11;
        border-left: 2px solid {t["amber"]};
        border-right: 2px solid {t["amber"]};
    }}
    .draft-board .user-col th {{
        background: {t["amber"]};
        color: {t["bg"]};
    }}
    .draft-board .current-pick {{
        border: 2px solid {t["amber"]};
        animation: amberGlow 2s ease-in-out infinite;
    }}
    .draft-board .picked {{
        color: {t["tx"]};
        font-weight: 600;
    }}

    /* ── CATEGORY CARDS ──────────────────────── */
    .cat-card {{
        background: {t["card"]};
        border: 1px solid {t["card_h"]};
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
        color: {t["tx2"]};
    }}
    .cat-val {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 18px;
        font-weight: 700;
        color: {t["tx"]};
        margin-top: 2px;
    }}

    /* ── ALERTS, FEED, WIZARD ────────────────── */
    .alert-card {{
        background: {t["card"]};
        border-left: 4px solid {t["warn"]};
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 13px;
        color: {t["tx"]};
    }}
    .alert-card.critical {{ border-left-color: {t["danger"]}; }}

    .feed-card {{
        background: {t["card"]};
        border: 1px solid {t["card_h"]};
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 6px;
        font-size: 13px;
    }}
    .feed-card.user-pick {{
        border-left: 3px solid {t["amber"]};
    }}
    .feed-pick-num {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: {t["tx2"]};
    }}
    .feed-name {{
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        color: {t["tx"]};
        word-break: break-word;
    }}
    .feed-team {{
        font-size: 11px;
        color: {t["tx2"]};
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
        color: {t["tx2"]};
        background: {t["card"]};
        border: 1px solid {t["card_h"]};
    }}
    .wizard-step:first-child {{ border-radius: 8px 0 0 8px; }}
    .wizard-step:last-child {{ border-radius: 0 8px 8px 0; }}
    .wizard-step.active {{
        background: {t["amber"]};
        color: {t["bg"]};
        border-color: {t["amber"]};
    }}
    .wizard-step.done {{
        background: {t["ok"]}22;
        color: {t["ok"]};
        border-color: {t["ok"]}44;
    }}

    /* ── LOCK-IN BANNER ──────────────────────── */
    .lock-in {{
        background: linear-gradient(135deg, {t["ok"]}33, {t["ok"]}11);
        border: 1px solid {t["ok"]}66;
        border-radius: 10px;
        padding: 12px 20px;
        text-align: center;
        font-family: 'Oswald', sans-serif;
        font-weight: 600;
        font-size: 16px;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: {t["ok"]};
        animation: lockFade 3s ease-out forwards;
    }}
    @keyframes lockFade {{
        0% {{ opacity: 1; }}
        70% {{ opacity: 1; }}
        100% {{ opacity: 0; }}
    }}

    /* ── VERDICT BANNER (Trade Analyzer) ────── */
    .verdict-banner {{
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
        text-align: center;
    }}
    .verdict-banner .verdict-text {{
        font-family: 'Oswald', sans-serif;
        font-size: 28px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
    }}
    .verdict-banner .verdict-conf {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 18px;
        margin-top: 4px;
    }}

    /* ── METRIC CARD (In-Season) ─────────────── */
    .metric-card {{
        background: {t["card"]};
        border: 1px solid {t["card_h"]};
        border-radius: 10px;
        padding: 14px 16px;
        text-align: center;
    }}
    .metric-card .metric-label {{
        font-family: 'Oswald', sans-serif;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: {t["tx2"]};
    }}
    .metric-card .metric-value {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 22px;
        font-weight: 700;
        color: {t["tx"]};
        margin-top: 4px;
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
        background: {t["amber"]};
        color: {t["bg"]};
        border: none;
    }}
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {{
        background: {t["amber_l"]};
        box-shadow: 0 4px 16px {t["amber"]}44;
    }}
    .stButton > button[kind="secondary"],
    .stButton > button[data-testid="stBaseButton-secondary"] {{
        background: {t["card"]};
        color: {t["tx"]};
        border: 1px solid {t["card_h"]};
    }}
    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] > div {{
        background: {t["card"]} !important;
        color: {t["tx"]} !important;
        border-color: {t["card_h"]} !important;
        border-radius: 8px !important;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        background: {t["card"]};
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
        color: {t["tx2"]};
    }}
    .stTabs [aria-selected="true"] {{
        background: {t["amber"]} !important;
        color: {t["bg"]} !important;
    }}
    div[data-testid="stFileUploader"] {{
        background: {t["card"]};
        border: 2px dashed {t["card_h"]};
        border-radius: 12px;
        padding: 16px;
    }}
    div[data-testid="stFileUploader"]:hover {{
        border-color: {t["amber"]};
    }}
    div[data-testid="stDataFrame"] {{
        border: 1px solid {t["card_h"]};
        border-radius: 10px;
    }}
    .stSidebar {{
        background: {t["card"]} !important;
        border-right: 1px solid {t["card_h"]} !important;
    }}

    /* ── ADDITIONAL STREAMLIT WIDGET OVERRIDES ─ */
    /* Covers widgets that config.toml dark theme would otherwise force dark */
    div[data-testid="stExpander"] {{
        background: {t["card"]} !important;
        border-color: {t["card_h"]} !important;
    }}
    div[data-testid="stExpander"] summary {{
        color: {t["tx"]} !important;
    }}
    div[data-testid="stMetric"] {{
        background: transparent;
        color: {t["tx"]} !important;
    }}
    div[data-testid="stMetric"] label {{
        color: {t["tx2"]} !important;
    }}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: {t["tx"]} !important;
    }}
    div[data-testid="stMultiSelect"] > div {{
        background: {t["card"]} !important;
        color: {t["tx"]} !important;
        border-color: {t["card_h"]} !important;
    }}
    div[data-testid="stSlider"] {{
        color: {t["tx"]} !important;
    }}
    .stMarkdown, .stMarkdown p {{
        color: {t["tx"]};
    }}
    .stCaption, div[data-testid="stCaptionContainer"] {{
        color: {t["tx2"]} !important;
    }}
    div[data-testid="stCheckbox"] label {{
        color: {t["tx"]} !important;
    }}
    div[data-testid="stRadio"] label {{
        color: {t["tx"]} !important;
    }}
    .stAlert {{
        color: {t["tx"]} !important;
    }}

    /* ── TOOLTIP (CSS-based, on title attr) ─── */
    [title] {{
        cursor: help;
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )


# ── Lazy theme proxy ─────────────────────────────────────────
# T acts as a dict that always returns the *current* theme's values.
# This allows `T["amber"]` in inline HTML f-strings to be theme-aware
# without changing any of the 60+ call sites in app.py / pages.


class _ThemeProxy(dict):
    """Dict-like proxy that delegates all reads to the active theme.

    Subclasses dict for isinstance() compat, but overrides every read
    method so len(), bool(), keys(), values(), items(), and iteration
    all reflect the current theme — not the empty internal store.
    """

    def __getitem__(self, key):
        return get_theme()[key]

    def get(self, key, default=None):
        return get_theme().get(key, default)

    def __contains__(self, key):
        return key in get_theme()

    def __repr__(self):
        return repr(get_theme())

    def __len__(self):
        return len(get_theme())

    def __iter__(self):
        return iter(get_theme())

    def __bool__(self):
        return True

    def keys(self):
        return get_theme().keys()

    def values(self):
        return get_theme().values()

    def items(self):
        return get_theme().items()


T = _ThemeProxy()

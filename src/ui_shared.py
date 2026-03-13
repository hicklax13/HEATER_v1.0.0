"""Shared UI constants, theme system, CSS injection, and SVG icons for all pages."""

import streamlit as st

# ── Inline SVG Page Icons ─────────────────────────────────────────
# Unique vector icons for each page, replacing emoji. Each is a compact
# inline SVG string (24x24 viewBox) that can be embedded in st.markdown().
# Colors are applied via CSS `currentColor` so they respect the theme.

PAGE_ICONS = {
    "configurations": (
        '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:8px;">'
        '<circle cx="12" cy="12" r="3"/>'
        '<path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06'
        "a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09"
        "A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83"
        "l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09"
        "A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83"
        "l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09"
        "a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83"
        "l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09"
        'a1.65 1.65 0 0 0-1.51 1z"/></svg>'
    ),
    "my_team": (
        '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:8px;">'
        '<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>'
        '<circle cx="9" cy="7" r="4"/>'
        '<path d="M23 21v-2a4 4 0 0 0-3-3.87"/>'
        '<path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>'
    ),
    "trade_analyzer": (
        '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:8px;">'
        '<polyline points="17 1 21 5 17 9"/>'
        '<path d="M3 11V9a4 4 0 0 1 4-4h14"/>'
        '<polyline points="7 23 3 19 7 15"/>'
        '<path d="M21 13v2a4 4 0 0 1-4 4H3"/></svg>'
    ),
    "player_compare": (
        '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:8px;">'
        '<line x1="18" y1="20" x2="18" y2="10"/>'
        '<line x1="12" y1="20" x2="12" y2="4"/>'
        '<line x1="6" y1="20" x2="6" y2="14"/>'
        '<line x1="3" y1="20" x2="21" y2="20"/></svg>'
    ),
    "free_agents": (
        '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:8px;">'
        '<circle cx="11" cy="11" r="8"/>'
        '<line x1="21" y1="21" x2="16.65" y2="16.65"/>'
        '<line x1="11" y1="8" x2="11" y2="14"/>'
        '<line x1="8" y1="11" x2="14" y2="11"/></svg>'
    ),
    "lineup_optimizer": (
        '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:8px;">'
        '<rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>'
        '<line x1="3" y1="9" x2="21" y2="9"/>'
        '<line x1="3" y1="15" x2="21" y2="15"/>'
        '<line x1="9" y1="3" x2="9" y2="21"/></svg>'
    ),
    # Utility icons used across pages
    "refresh": (
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:4px;">'
        '<polyline points="23 4 23 10 17 10"/>'
        '<path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>'
    ),
    "check": (
        '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#84cc16" '
        'stroke-width="3" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:4px;">'
        '<polyline points="20 6 9 17 4 12"/></svg>'
    ),
    "x_mark": (
        '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#f43f5e" '
        'stroke-width="3" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:4px;">'
        '<line x1="18" y1="6" x2="6" y2="18"/>'
        '<line x1="6" y1="6" x2="18" y2="18"/></svg>'
    ),
    "warning": (
        '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#fb923c" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:4px;">'
        '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86'
        'a2 2 0 0 0-3.42 0z"/>'
        '<line x1="12" y1="9" x2="12" y2="13"/>'
        '<line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
    ),
    "alert": (
        '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#f43f5e" '
        'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:4px;">'
        '<circle cx="12" cy="12" r="10"/>'
        '<line x1="12" y1="8" x2="12" y2="12"/>'
        '<line x1="12" y1="16" x2="12.01" y2="16"/></svg>'
    ),
    "baseball": (
        '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" style="vertical-align:middle;margin-right:8px;">'
        '<circle cx="12" cy="12" r="10"/>'
        '<path d="M4.93 4.93c4.08 2.38 4.08 11.76 0 14.14"/>'
        '<path d="M19.07 4.93c-4.08 2.38-4.08 11.76 0 14.14"/></svg>'
    ),
    "bar_chart": (
        '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:4px;">'
        '<line x1="18" y1="20" x2="18" y2="10"/>'
        '<line x1="12" y1="20" x2="12" y2="4"/>'
        '<line x1="6" y1="20" x2="6" y2="14"/></svg>'
    ),
    "zap": (
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:4px;">'
        '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>'
    ),
    "calendar": (
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:4px;">'
        '<rect x="3" y="4" width="18" height="18" rx="2" ry="2"/>'
        '<line x1="16" y1="2" x2="16" y2="6"/>'
        '<line x1="8" y1="2" x2="8" y2="6"/>'
        '<line x1="3" y1="10" x2="21" y2="10"/></svg>'
    ),
    "accept": (
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#84cc16" '
        'stroke-width="3" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:6px;">'
        '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>'
        '<polyline points="22 4 12 14.01 9 11.01"/></svg>'
    ),
    "reject": (
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#f43f5e" '
        'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:6px;">'
        '<circle cx="12" cy="12" r="10"/>'
        '<line x1="4.93" y1="4.93" x2="19.07" y2="19.07"/></svg>'
    ),
    "fire": (
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="#f43f5e" stroke="none" '
        'style="vertical-align:middle;margin-right:3px;">'
        '<path d="M12 23c-3.866 0-7-3.134-7-7 0-3 2-5.5 4-8 .667 1.333 1.333 2 2 2'
        " 0-4 1.5-7.5 4-10 .667 2.667 2 5.333 4 8 1.333-1.333 2-3.333 2-6"
        ' 2 2 3 4.5 3 7 0 3.866-3.134 7-7 7z"/></svg>'
    ),
    "trending_up": (
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:3px;">'
        '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/>'
        '<polyline points="17 6 23 6 23 12"/></svg>'
    ),
    "minus": (
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2.5" stroke-linecap="round" style="vertical-align:middle;margin-right:3px;">'
        '<line x1="5" y1="12" x2="19" y2="12"/></svg>'
    ),
    "users": (
        '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:4px;">'
        '<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>'
        '<circle cx="9" cy="7" r="4"/>'
        '<path d="M23 21v-2a4 4 0 0 0-3-3.87"/>'
        '<path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>'
    ),
}

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
    "ink": "#0a0e1a",
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
    "border": "#d1d5db",
    "ink": "#1a1a2e",
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
            label = "Dark Mode" if is_dark else "Light Mode"
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
        color: {t["ink"]};
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
        color: {t["ink"]};
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
        word-break: break-word;
        overflow-wrap: anywhere;
        line-height: 1.3;
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
        max-width: 200px;
        overflow-wrap: anywhere;
        word-break: break-word;
    }}
    .draft-board .user-col {{
        background: {t["amber"]}11;
        border-left: 2px solid {t["amber"]};
        border-right: 2px solid {t["amber"]};
    }}
    .draft-board .user-col th {{
        background: {t["amber"]};
        color: {t["ink"]};
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
        color: {t["ink"]};
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
        color: {t["ink"]};
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
        color: {t["ink"]} !important;
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

    /* ── PAGE TITLE ────────────────────────────── */
    .page-title {{
        font-family: 'Oswald', sans-serif;
        font-weight: 700;
        font-size: 32px;
        color: {t["amber"]};
        text-transform: uppercase;
        letter-spacing: 3px;
        text-align: center;
        margin-bottom: 8px;
        word-break: break-word;
        overflow-wrap: anywhere;
    }}

    /* ── TOOLTIP (CSS-based, on title attr) ─── */
    [title] {{
        cursor: help;
    }}

    /* ── SIDEBAR NAV: rename "app" → "Configurations" ─── */
    [data-testid="stSidebarNav"] a[href="/"] span {{
        font-size: 0 !important;
    }}
    [data-testid="stSidebarNav"] a[href="/"] span::after {{
        content: "Configurations";
        font-size: 14px;
    }}

    /* ── RESPONSIVE: scale down large text on narrow viewports ─── */
    @media (max-width: 768px) {{
        .page-title {{
            font-size: 22px;
            letter-spacing: 1.5px;
        }}
        .hero .p-name {{
            font-size: 24px;
            letter-spacing: 0.5px;
        }}
        .hero .score-badge {{
            font-size: 16px;
            padding: 6px 10px;
        }}
        .verdict-banner .verdict-text {{
            font-size: 20px;
            letter-spacing: 1px;
        }}
        .wizard-step {{
            font-size: 10px;
            letter-spacing: 1px;
            padding: 8px 10px;
        }}
    }}
    @media (max-width: 480px) {{
        .page-title {{
            font-size: 18px;
            letter-spacing: 1px;
        }}
        .hero .p-name {{
            font-size: 20px;
            letter-spacing: 0px;
        }}
        .verdict-banner .verdict-text {{
            font-size: 16px;
            letter-spacing: 0.5px;
        }}
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

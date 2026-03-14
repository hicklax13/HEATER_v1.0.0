"""Shared UI constants, theme system, CSS injection, and SVG icons — Heater Edition."""

import streamlit as st

# ── Inline SVG Page Icons ─────────────────────────────────────────
# Unique vector icons for each page. Each is a compact inline SVG string
# (24x24 viewBox) that can be embedded in st.markdown().

PAGE_ICONS = {
    # ── Logo: realistic baseball with classic stitching + heat trail ──
    "logo": (
        '<svg width="40" height="40" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" '
        'style="vertical-align:middle;margin-right:8px;">'
        "<defs>"
        '<radialGradient id="bl" cx="40%" cy="35%" r="55%">'
        '<stop offset="0%" stop-color="#fff"/>'
        '<stop offset="60%" stop-color="#f5f0e8"/>'
        '<stop offset="100%" stop-color="#e8ddd0"/>'
        "</radialGradient>"
        '<linearGradient id="sg" x1="0" y1="0" x2="1" y2="0">'
        '<stop offset="0%" stop-color="#ff6d00" stop-opacity="0.7"/>'
        '<stop offset="100%" stop-color="#ff6d00" stop-opacity="0"/>'
        "</linearGradient>"
        "</defs>"
        "<!-- speed lines -->"
        '<line x1="2" y1="22" x2="18" y2="28" stroke="url(#sg)" stroke-width="2.5" stroke-linecap="round"/>'
        '<line x1="0" y1="32" x2="16" y2="33" stroke="url(#sg)" stroke-width="2" stroke-linecap="round"/>'
        '<line x1="4" y1="42" x2="17" y2="39" stroke="url(#sg)" stroke-width="1.5" stroke-linecap="round"/>'
        "<!-- ball body -->"
        '<circle cx="38" cy="32" r="18" fill="url(#bl)"/>'
        '<circle cx="38" cy="32" r="18" fill="none" stroke="#c8b8a8" stroke-width="0.8"/>'
        "<!-- shadow for 3D depth -->"
        '<ellipse cx="39" cy="50" rx="12" ry="2" fill="#00000010"/>'
        "<!-- classic V-stitching left arc -->"
        '<path d="M29 16 C24 22 22 27 23 32 C22 37 24 42 29 48" '
        'fill="none" stroke="#e63946" stroke-width="1.6" stroke-linecap="round"/>'
        "<!-- classic V-stitching right arc -->"
        '<path d="M47 16 C52 22 54 27 53 32 C54 37 52 42 47 48" '
        'fill="none" stroke="#e63946" stroke-width="1.6" stroke-linecap="round"/>'
        "<!-- stitch tick marks left -->"
        '<line x1="27" y1="19" x2="30" y2="20" stroke="#e63946" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="25" y1="23" x2="28" y2="23.5" stroke="#e63946" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="24" y1="27.5" x2="27" y2="27.5" stroke="#e63946" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="24" y1="36.5" x2="27" y2="36.5" stroke="#e63946" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="25" y1="41" x2="28" y2="40.5" stroke="#e63946" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="27" y1="45" x2="30" y2="44" stroke="#e63946" stroke-width="1" stroke-linecap="round"/>'
        "<!-- stitch tick marks right -->"
        '<line x1="49" y1="19" x2="46" y2="20" stroke="#e63946" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="51" y1="23" x2="48" y2="23.5" stroke="#e63946" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="52" y1="27.5" x2="49" y2="27.5" stroke="#e63946" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="52" y1="36.5" x2="49" y2="36.5" stroke="#e63946" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="51" y1="41" x2="48" y2="40.5" stroke="#e63946" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="49" y1="45" x2="46" y2="44" stroke="#e63946" stroke-width="1" stroke-linecap="round"/>'
        "<!-- heat glow ring -->"
        '<circle cx="38" cy="32" r="21" fill="none" stroke="#ff6d00" stroke-width="1.2" opacity="0.25"/>'
        "</svg>"
    ),
    "logo_lg": (
        '<svg width="64" height="64" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" '
        'style="vertical-align:middle;">'
        "<defs>"
        '<radialGradient id="bll" cx="40%" cy="35%" r="55%">'
        '<stop offset="0%" stop-color="#fff"/>'
        '<stop offset="60%" stop-color="#f5f0e8"/>'
        '<stop offset="100%" stop-color="#e8ddd0"/>'
        "</radialGradient>"
        '<linearGradient id="sgl" x1="0" y1="0" x2="1" y2="0">'
        '<stop offset="0%" stop-color="#ff6d00" stop-opacity="0.7"/>'
        '<stop offset="100%" stop-color="#ff6d00" stop-opacity="0"/>'
        "</linearGradient>"
        "</defs>"
        '<line x1="2" y1="22" x2="18" y2="28" stroke="url(#sgl)" stroke-width="2.5" stroke-linecap="round"/>'
        '<line x1="0" y1="32" x2="16" y2="33" stroke="url(#sgl)" stroke-width="2" stroke-linecap="round"/>'
        '<line x1="4" y1="42" x2="17" y2="39" stroke="url(#sgl)" stroke-width="1.5" stroke-linecap="round"/>'
        '<circle cx="38" cy="32" r="18" fill="url(#bll)"/>'
        '<circle cx="38" cy="32" r="18" fill="none" stroke="#c8b8a8" stroke-width="0.8"/>'
        '<ellipse cx="39" cy="50" rx="12" ry="2" fill="#00000010"/>'
        '<path d="M29 16 C24 22 22 27 23 32 C22 37 24 42 29 48" '
        'fill="none" stroke="#e63946" stroke-width="1.6" stroke-linecap="round"/>'
        '<path d="M47 16 C52 22 54 27 53 32 C54 37 52 42 47 48" '
        'fill="none" stroke="#e63946" stroke-width="1.6" stroke-linecap="round"/>'
        '<line x1="27" y1="19" x2="30" y2="20" stroke="#e63946" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="25" y1="23" x2="28" y2="23.5" stroke="#e63946" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="24" y1="27.5" x2="27" y2="27.5" stroke="#e63946" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="24" y1="36.5" x2="27" y2="36.5" stroke="#e63946" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="25" y1="41" x2="28" y2="40.5" stroke="#e63946" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="27" y1="45" x2="30" y2="44" stroke="#e63946" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="49" y1="19" x2="46" y2="20" stroke="#e63946" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="51" y1="23" x2="48" y2="23.5" stroke="#e63946" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="52" y1="27.5" x2="49" y2="27.5" stroke="#e63946" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="52" y1="36.5" x2="49" y2="36.5" stroke="#e63946" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="51" y1="41" x2="48" y2="40.5" stroke="#e63946" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="49" y1="45" x2="46" y2="44" stroke="#e63946" stroke-width="1" stroke-linecap="round"/>'
        '<circle cx="38" cy="32" r="21" fill="none" stroke="#ff6d00" stroke-width="1.2" opacity="0.25"/>'
        "</svg>"
    ),
    # ── Page navigation icons ──
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
    # ── Utility icons ──
    "refresh": (
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:4px;">'
        '<polyline points="23 4 23 10 17 10"/>'
        '<path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>'
    ),
    "check": (
        '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#2d6a4f" '
        'stroke-width="3" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:4px;">'
        '<polyline points="20 6 9 17 4 12"/></svg>'
    ),
    "x_mark": (
        '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#e63946" '
        'stroke-width="3" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:4px;">'
        '<line x1="18" y1="6" x2="6" y2="18"/>'
        '<line x1="6" y1="6" x2="18" y2="18"/></svg>'
    ),
    "warning": (
        '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#ff9f1c" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:4px;">'
        '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86'
        'a2 2 0 0 0-3.42 0z"/>'
        '<line x1="12" y1="9" x2="12" y2="13"/>'
        '<line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
    ),
    "alert": (
        '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#e63946" '
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
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#2d6a4f" '
        'stroke-width="3" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:6px;">'
        '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>'
        '<polyline points="22 4 12 14.01 9 11.01"/></svg>'
    ),
    "reject": (
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#e63946" '
        'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:6px;">'
        '<circle cx="12" cy="12" r="10"/>'
        '<line x1="4.93" y1="4.93" x2="19.07" y2="19.07"/></svg>'
    ),
    "fire": (
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="#e63946" stroke="none" '
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

# ── Theme (Light-only — Heater palette) ──────────────────────────

THEME = {
    "bg": "#f4f5f0",
    "card": "#ffffff",
    "card_h": "#e8e9e3",
    "primary": "#e63946",
    "primary_l": "#ff6b6b",
    "hot": "#ff6d00",
    "hot_l": "#ff9e40",
    "gold": "#ffd60a",
    "green": "#2d6a4f",
    "green_l": "#40916c",
    "sky": "#457b9d",
    "sky_l": "#a8dadc",
    "purple": "#6c63ff",
    "ok": "#2d6a4f",
    "danger": "#e63946",
    "warn": "#ff9f1c",
    "tx": "#1d1d1f",
    "tx2": "#6b7280",
    "border": "#d4d5cf",
    "ink": "#ffffff",
    "tiers": [
        "#e63946",
        "#ff6d00",
        "#ffd60a",
        "#2d6a4f",
        "#457b9d",
        "#6c63ff",
        "#8d99ae",
        "#adb5bd",
    ],
    # Backward compatibility aliases — old code uses T["amber"], T["teal"], etc.
    "amber": "#e63946",
    "amber_l": "#ff6b6b",
    "teal": "#457b9d",
}

# Simple alias — no proxy needed without dark mode.
# All existing T["key"] call sites continue to work.
T = THEME

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
        "Standings Gained Points — how many roto standings points "
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
        "Monte Carlo Mean — the average Standings Gained Points change across 200 simulated seasons. "
        "Positive = trade helps your team."
    ),
    "mc_std": (
        "Monte Carlo Standard Deviation — how much the outcome varies across "
        "simulations. Lower = more predictable; higher = riskier trade."
    ),
    "z_score": (
        "Z-Score — how many standard deviations above or below "
        "league average. +1.0 = top ~16% of players; -1.0 = bottom ~16%."
    ),
    "marginal_value": (
        "Marginal Value — how much this free agent would improve your "
        "team's Standings Gained Points compared to your current worst player at the position."
    ),
    "cat_targeting": (
        "Category Targeting Weight — higher weight means bigger "
        "standings impact. Higher = more room to gain in standings."
    ),
    "p10_p90": (
        "10th Percentile / 90th Percentile Range — the floor (10th percentile) and ceiling "
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
        "Value pick — this player's Average Draft Position is later than the current "
        "pick, meaning you are getting them earlier than most drafts."
    ),
    "adp_reach": (
        "Reach pick — this player's Average Draft Position is earlier than the current "
        "pick, meaning you are drafting them before most leagues would."
    ),
    "adp_fair": (
        "Fair pick — this player's Average Draft Position is close to the current pick, "
        "meaning this is roughly where they are expected to go."
    ),
}


# ── Section Header Helper ──────────────────────────────────────────


def sec(title):
    """Render a styled section header (uppercase, Bebas Neue font)."""
    st.markdown(f'<div class="sec-head">{title}</div>', unsafe_allow_html=True)


# ── Plotly Theme Helpers ───────────────────────────────────────────


def get_plotly_layout(theme=None):
    """Return Plotly layout kwargs for consistent chart theming."""
    t = theme or THEME
    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": t["tx"], "family": "Figtree, sans-serif"},
        "margin": {"l": 40, "r": 40, "t": 30, "b": 30},
        "height": 350,
    }


def get_plotly_polar(theme=None):
    """Return polar/radar chart config for Player Compare."""
    t = theme or THEME
    return {
        "bgcolor": "rgba(255,255,255,0.5)",
        "radialaxis": {
            "gridcolor": "rgba(0,0,0,0.06)",
            "tickfont": {"color": t["tx2"], "size": 10},
            "visible": True,
        },
        "angularaxis": {
            "gridcolor": "rgba(0,0,0,0.08)",
            "tickfont": {"color": t["tx"], "size": 11},
        },
    }


# ── Backward Compatibility Stubs ──────────────────────────────────
# Old code may call get_theme() or render_theme_toggle().


def get_theme():
    """Return the theme dict (light-only, no dark mode)."""
    return THEME


def render_theme_toggle():
    """No-op — dark mode removed. Kept for import compatibility."""
    pass


# ── CSS Injection ──────────────────────────────────────────────────


def inject_custom_css():
    """Inject the complete Heater CSS for all pages.

    Call this once at the top of every page, after st.set_page_config().
    Uses glassmorphism, 3D buttons, kinetic typography, and tactile animations.
    """
    t = THEME
    st.markdown(
        f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Figtree:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    /* ── BASE ─────────────────────────────────── */
    .stApp {{
        background: {t["bg"]};
        font-family: 'Figtree', sans-serif;
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
        color: {t["tx"]};
        font-family: 'Bebas Neue', sans-serif;
        text-transform: uppercase;
        letter-spacing: 2px;
    }}

    /* ── SECTION HEADER ───────────────────────── */
    .sec-head {{
        font-family: 'Bebas Neue', sans-serif;
        font-weight: 400;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 3px;
        color: {t["tx2"]};
        margin-bottom: 10px;
        padding-bottom: 6px;
        border-bottom: 2px solid {t["border"]};
    }}

    /* ── GLASS CARD (Glassmorphism) ───────────── */
    .glass {{
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 12px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08), 0 2px 8px rgba(0, 0, 0, 0.04);
        transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.3s ease;
    }}
    .glass:hover {{
        transform: translateY(-4px) scale(1.01);
        box-shadow: 0 16px 48px rgba(0, 0, 0, 0.12), 0 4px 16px rgba(0, 0, 0, 0.06);
    }}

    /* ── COMMAND BAR ──────────────────────────── */
    .cmd-bar {{
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(16px) saturate(150%);
        -webkit-backdrop-filter: blur(16px) saturate(150%);
        border: 1px solid rgba(255, 255, 255, 0.4);
        border-radius: 16px;
        padding: 14px 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 16px;
        flex-wrap: wrap;
        gap: 10px;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.06);
    }}
    .cmd-left {{
        font-family: 'Bebas Neue', sans-serif;
        font-size: 20px;
        text-transform: uppercase;
        letter-spacing: 3px;
        color: {t["tx"]};
    }}
    .cmd-center {{ display: flex; align-items: center; gap: 12px; }}
    .cmd-right {{
        display: flex; align-items: center; gap: 10px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13px;
        color: {t["tx2"]};
    }}

    /* ── YOUR TURN BADGE ─────────────────────── */
    .your-turn {{
        background: linear-gradient(135deg, {t["primary"]}, {t["hot"]});
        color: {t["ink"]};
        font-family: 'Bebas Neue', sans-serif;
        font-size: 16px;
        text-transform: uppercase;
        letter-spacing: 2px;
        padding: 8px 20px;
        border-radius: 10px;
        animation: heatPulse 2s ease-in-out infinite;
    }}
    .waiting {{
        background: {t["card_h"]};
        color: {t["tx2"]};
        font-family: 'Bebas Neue', sans-serif;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        padding: 8px 16px;
        border-radius: 10px;
    }}
    @keyframes heatPulse {{
        0%, 100% {{ box-shadow: 0 0 8px rgba(230, 57, 70, 0.4); }}
        50% {{ box-shadow: 0 0 24px rgba(230, 57, 70, 0.7), 0 0 48px rgba(255, 109, 0, 0.3); }}
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
        background: linear-gradient(90deg, {t["primary"]}, {t["hot"]});
        border-radius: 3px;
        transition: width 0.5s ease;
    }}

    /* ── HERO PICK CARD ──────────────────────── */
    .hero {{
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border: 2px solid transparent;
        border-image: linear-gradient(135deg, {t["primary"]}, {t["hot"]}) 1;
        border-radius: 0px;
        padding: 24px;
        margin-bottom: 16px;
        position: relative;
        box-shadow: 0 8px 40px rgba(230, 57, 70, 0.12), 0 4px 16px rgba(0, 0, 0, 0.06);
        transition: transform 0.4s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.3s ease;
    }}
    .hero:hover {{
        transform: perspective(1000px) rotateY(-1deg) rotateX(1deg) translateY(-4px);
        box-shadow: 0 20px 60px rgba(230, 57, 70, 0.15), 0 8px 24px rgba(0, 0, 0, 0.08);
    }}
    .hero .p-name {{
        font-family: 'Bebas Neue', sans-serif;
        font-size: 36px;
        color: {t["tx"]};
        text-transform: uppercase;
        letter-spacing: 2px;
        line-height: 1.1;
        word-break: break-word;
    }}
    .hero .p-meta {{
        font-family: 'Figtree', sans-serif;
        font-size: 14px;
        color: {t["tx2"]};
        margin-top: 4px;
    }}
    .hero .score-badge {{
        position: absolute;
        top: 16px; right: 20px;
        background: linear-gradient(135deg, {t["primary"]}, {t["hot"]});
        color: {t["ink"]};
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 700;
        font-size: 22px;
        padding: 8px 14px;
        border-radius: 12px;
        cursor: help;
        box-shadow: 0 4px 16px rgba(230, 57, 70, 0.3);
    }}
    .hero .reason {{
        font-family: 'Figtree', sans-serif;
        font-size: 15px;
        color: {t["hot"]};
        margin-top: 12px;
        font-style: italic;
    }}

    /* ── SURVIVAL GAUGE ──────────────────────── */
    .surv-gauge {{
        width: 64px; height: 64px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 700;
        font-size: 16px;
        color: {t["tx"]};
    }}

    /* ── SGP CHIPS ───────────────────────────── */
    .sgp-row {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }}
    .sgp-chip {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        padding: 4px 10px;
        border-radius: 20px;
        border: 1px solid {t["border"]};
        background: rgba(255, 255, 255, 0.5);
        color: {t["tx2"]};
    }}
    .sgp-chip.pos {{ border-color: {t["green"]}; color: {t["green"]}; background: rgba(45, 106, 79, 0.08); }}
    .sgp-chip.neg {{ border-color: {t["danger"]}; color: {t["danger"]}; background: rgba(230, 57, 70, 0.08); }}

    /* ── ALTERNATIVE CARDS ───────────────────── */
    .alt {{
        background: rgba(255, 255, 255, 0.5);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-left: 4px solid {t["border"]};
        border-radius: 12px;
        padding: 12px 14px;
        transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.3s ease;
        cursor: default;
    }}
    .alt:hover {{
        transform: translateY(-6px) scale(1.02);
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.1);
    }}
    .alt .a-rank {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        color: {t["tx2"]};
    }}
    .alt .a-name {{
        font-family: 'Bebas Neue', sans-serif;
        font-size: 16px;
        color: {t["tx"]};
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 2px 0;
        word-break: break-word;
    }}
    .alt .a-meta {{
        font-family: 'Figtree', sans-serif;
        font-size: 12px;
        color: {t["tx2"]};
    }}
    .alt .a-score {{
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        font-size: 14px;
        color: {t["primary"]};
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
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 20px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .badge-value {{ background: rgba(45, 106, 79, 0.1); color: {t["ok"]}; border: 1px solid rgba(45, 106, 79, 0.3); }}
    .badge-reach {{ background: rgba(230, 57, 70, 0.1); color: {t["danger"]}; border: 1px solid rgba(230, 57, 70, 0.3); }}
    .badge-fair {{ background: {t["card_h"]}; color: {t["tx2"]}; }}
    .badge-risk-low {{ background: rgba(45, 106, 79, 0.1); color: {t["ok"]}; }}
    .badge-risk-med {{ background: rgba(255, 159, 28, 0.1); color: {t["warn"]}; }}
    .badge-risk-high {{ background: rgba(230, 57, 70, 0.1); color: {t["danger"]}; }}

    /* ── ROSTER GRID ─────────────────────────── */
    .roster-grid {{
        display: grid;
        gap: 6px;
    }}
    .roster-slot {{
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 10px;
        padding: 6px 10px;
        text-align: center;
        min-height: 48px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: transform 0.2s ease;
    }}
    .roster-slot:hover {{
        transform: translateY(-2px);
    }}
    .roster-slot.filled {{
        border-color: rgba(45, 106, 79, 0.3);
        background: rgba(45, 106, 79, 0.05);
    }}
    .roster-slot.empty {{
        border-style: dashed;
        border-color: {t["border"]};
    }}
    .roster-slot .s-label {{
        font-family: 'Bebas Neue', sans-serif;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: {t["tx2"]};
    }}
    .roster-slot .s-player {{
        font-family: 'Figtree', sans-serif;
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
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        font-size: 13px;
        color: {t["tx"]};
        transition: transform 0.3s ease;
    }}
    .scar-ring:hover {{
        transform: scale(1.1);
    }}
    .scar-ring.critical {{
        animation: scarPulse 1.5s ease-in-out infinite;
    }}
    @keyframes scarPulse {{
        0%, 100% {{ transform: scale(1); box-shadow: 0 0 0 rgba(230, 57, 70, 0); }}
        50% {{ transform: scale(1.08); box-shadow: 0 0 16px rgba(230, 57, 70, 0.3); }}
    }}
    .scar-label {{
        font-family: 'Bebas Neue', sans-serif;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: {t["tx2"]};
    }}

    /* ── DRAFT BOARD ─────────────────────────── */
    .draft-board {{
        width: 100%;
        border-collapse: collapse;
        font-family: 'Figtree', sans-serif;
        font-size: 11px;
    }}
    .draft-board th {{
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(8px);
        color: {t["tx2"]};
        font-family: 'Bebas Neue', sans-serif;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        padding: 10px 6px;
        border-bottom: 2px solid {t["border"]};
        position: sticky;
        top: 0;
        z-index: 10;
    }}
    .draft-board td {{
        padding: 6px;
        border-bottom: 1px solid rgba(0, 0, 0, 0.04);
        color: {t["tx2"]};
        max-width: 200px;
        overflow-wrap: anywhere;
        word-break: break-word;
    }}
    .draft-board tr:nth-child(even) {{
        background: rgba(0, 0, 0, 0.015);
    }}
    .draft-board .user-col {{
        background: rgba(230, 57, 70, 0.04);
        border-left: 2px solid {t["primary"]};
        border-right: 2px solid {t["primary"]};
    }}
    .draft-board .user-col th {{
        background: linear-gradient(135deg, {t["primary"]}, {t["hot"]});
        color: {t["ink"]};
    }}
    .draft-board .current-pick {{
        border: 2px solid {t["primary"]};
        animation: heatPulse 2s ease-in-out infinite;
    }}
    .draft-board .picked {{
        color: {t["tx"]};
        font-weight: 600;
    }}

    /* ── CATEGORY CARDS ──────────────────────── */
    .cat-card {{
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 12px;
        padding: 12px 14px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
        transition: transform 0.3s ease;
    }}
    .cat-card:hover {{
        transform: translateY(-2px);
    }}
    .cat-name {{
        font-family: 'Bebas Neue', sans-serif;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: {t["tx2"]};
    }}
    .cat-val {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 20px;
        font-weight: 700;
        color: {t["tx"]};
        margin-top: 2px;
    }}

    /* ── ALERTS, FEED, WIZARD ────────────────── */
    .alert-card {{
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(8px);
        border-left: 4px solid {t["warn"]};
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 13px;
        color: {t["tx"]};
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
    }}
    .alert-card.critical {{ border-left-color: {t["danger"]}; }}

    .feed-card {{
        background: rgba(255, 255, 255, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 6px;
        font-size: 13px;
        transition: transform 0.2s ease;
    }}
    .feed-card:hover {{
        transform: translateX(4px);
    }}
    .feed-card.user-pick {{
        border-left: 3px solid {t["primary"]};
        background: rgba(230, 57, 70, 0.04);
    }}
    .feed-pick-num {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        color: {t["tx2"]};
    }}
    .feed-name {{
        font-family: 'Figtree', sans-serif;
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
        padding: 12px 24px;
        font-family: 'Bebas Neue', sans-serif;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: {t["tx2"]};
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }}
    .wizard-step:first-child {{ border-radius: 12px 0 0 12px; }}
    .wizard-step:last-child {{ border-radius: 0 12px 12px 0; }}
    .wizard-step.active {{
        background: linear-gradient(135deg, {t["primary"]}, {t["hot"]});
        color: {t["ink"]};
        border-color: transparent;
        box-shadow: 0 4px 16px rgba(230, 57, 70, 0.3);
    }}
    .wizard-step.done {{
        background: rgba(45, 106, 79, 0.1);
        color: {t["ok"]};
        border-color: rgba(45, 106, 79, 0.3);
    }}

    /* ── LOCK-IN BANNER ──────────────────────── */
    .lock-in {{
        background: linear-gradient(135deg, rgba(45, 106, 79, 0.15), rgba(45, 106, 79, 0.05));
        border: 1px solid rgba(45, 106, 79, 0.3);
        border-radius: 12px;
        padding: 12px 20px;
        text-align: center;
        font-family: 'Bebas Neue', sans-serif;
        font-size: 16px;
        text-transform: uppercase;
        letter-spacing: 3px;
        color: {t["ok"]};
        animation: lockSlideIn 3s ease-out forwards;
    }}
    @keyframes lockSlideIn {{
        0% {{ transform: translateY(-20px) scale(0.95); opacity: 0; }}
        20% {{ transform: translateY(0) scale(1.02); opacity: 1; }}
        30% {{ transform: scale(1); }}
        80% {{ opacity: 1; }}
        100% {{ opacity: 0; transform: translateY(-10px); }}
    }}

    /* ── VERDICT BANNER (Trade Analyzer) ────── */
    .verdict-banner {{
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        text-align: center;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
    }}
    .verdict-banner .verdict-text {{
        font-family: 'Bebas Neue', sans-serif;
        font-size: 32px;
        text-transform: uppercase;
        letter-spacing: 4px;
    }}
    .verdict-banner .verdict-conf {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 18px;
        margin-top: 4px;
    }}

    /* ── METRIC CARD (In-Season — 3D tilt) ─── */
    .metric-card {{
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.4);
        border-radius: 16px;
        padding: 16px 18px;
        text-align: center;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.06);
        transform: perspective(800px) rotateX(2deg);
        transition: transform 0.4s ease;
    }}
    .metric-card:hover {{
        transform: perspective(800px) rotateX(0deg) translateY(-2px);
    }}
    .metric-card .metric-label {{
        font-family: 'Bebas Neue', sans-serif;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: {t["tx2"]};
    }}
    .metric-card .metric-value {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 24px;
        font-weight: 700;
        color: {t["tx"]};
        margin-top: 4px;
    }}

    /* ── PLAYER CARD (for card-based selection) ─ */
    .player-card {{
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 14px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
        cursor: pointer;
    }}
    .player-card:hover {{
        transform: translateY(-6px) scale(1.03);
        box-shadow: 0 12px 40px rgba(230, 57, 70, 0.12);
        border-color: rgba(230, 57, 70, 0.3);
    }}
    .player-card.selected {{
        border: 2px solid {t["primary"]};
        background: rgba(230, 57, 70, 0.06);
        box-shadow: 0 4px 20px rgba(230, 57, 70, 0.15);
    }}
    .pc-name {{
        font-family: 'Bebas Neue', sans-serif;
        font-size: 15px;
        letter-spacing: 1px;
        color: {t["tx"]};
        text-transform: uppercase;
    }}
    .pc-pos {{
        font-family: 'Figtree', sans-serif;
        font-size: 11px;
        color: {t["tx2"]};
        margin-top: 2px;
    }}
    .pc-score {{
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        font-size: 14px;
        color: {t["primary"]};
        margin-top: 6px;
    }}

    /* ── BENTO GRID ──────────────────────────── */
    .bento-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 16px;
    }}
    .bento-wide {{ grid-column: span 2; }}
    .bento-tall {{ grid-row: span 2; }}

    /* ── SHIMMER LOADING ─────────────────────── */
    @keyframes shimmer {{
        0% {{ background-position: -200% 0; }}
        100% {{ background-position: 200% 0; }}
    }}
    .shimmer {{
        background: linear-gradient(90deg,
            rgba(0, 0, 0, 0.03) 25%,
            rgba(0, 0, 0, 0.06) 50%,
            rgba(0, 0, 0, 0.03) 75%
        );
        background-size: 200% 100%;
        animation: shimmer 1.5s infinite;
        border-radius: 10px;
    }}

    /* ── STAGGERED ENTRANCE ──────────────────── */
    @keyframes slideUp {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    .glass, .hero, .alt, .metric-card, .player-card, .cat-card {{
        animation: slideUp 0.4s ease-out both;
    }}

    /* ── KINETIC GRADIENT TEXT ────────────────── */
    @keyframes gradientShift {{
        0%, 100% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
    }}
    .page-title {{
        font-family: 'Bebas Neue', sans-serif !important;
        font-size: 28px !important;
        letter-spacing: 3px !important;
        text-transform: uppercase !important;
        text-align: center !important;
        margin-top: 8px !important;
        margin-bottom: 8px !important;
        word-break: break-word !important;
        overflow-wrap: anywhere !important;
        display: inline-block !important;
        padding: 8px 28px !important;
        border-radius: 50px !important;
        background: linear-gradient(135deg, #1a1a2e, #16213e) !important;
        box-shadow: 0 3px 14px rgba(22,33,62,0.3), inset 0 1px 0 rgba(255,255,255,0.08) !important;
        position: relative !important;
    }}
    .page-title-wrap {{
        text-align: center !important;
        margin-top: 4px !important;
        margin-bottom: 2px !important;
    }}
    .page-title span {{
        background: linear-gradient(135deg, {t["primary"]}, {t["hot"]}, {t["gold"]}) !important;
        background-size: 200% 200% !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        animation: gradientShift 4s ease infinite !important;
    }}

    /* ── SPLASH SCREEN TITLE ─────────────────── */
    @keyframes titleReveal {{
        0% {{ letter-spacing: 20px; opacity: 0; transform: scale(1.3); }}
        60% {{ letter-spacing: 6px; opacity: 1; }}
        80% {{ transform: scale(0.98); }}
        100% {{ transform: scale(1); letter-spacing: 6px; }}
    }}
    .splash-title {{
        font-family: 'Bebas Neue', sans-serif;
        font-size: 42px;
        text-align: center;
        background: linear-gradient(135deg, {t["primary"]}, {t["hot"]}, {t["gold"]});
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: titleReveal 1.2s ease-out both, gradientShift 4s ease 1.2s infinite;
    }}

    /* ── STREAMLIT OVERRIDES ─────────────────── */

    /* Buttons — 3D inflatable */
    .stButton > button {{
        font-family: 'Bebas Neue', sans-serif;
        font-size: 14px;
        letter-spacing: 2px;
        text-transform: uppercase;
        border-radius: 12px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        transition: all 0.15s ease;
    }}
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {{
        background: linear-gradient(135deg, {t["primary"]}, {t["hot"]});
        color: {t["ink"]};
        border: none;
        transform: translateY(-2px);
        box-shadow: 0 4px 0 #b71c1c, 0 6px 16px rgba(230, 57, 70, 0.3);
    }}
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {{
        transform: translateY(-3px);
        box-shadow: 0 6px 0 #b71c1c, 0 8px 24px rgba(230, 57, 70, 0.4);
    }}
    .stButton > button[kind="primary"]:active,
    .stButton > button[data-testid="stBaseButton-primary"]:active {{
        transform: translateY(0) scale(0.98);
        box-shadow: 0 1px 0 #b71c1c, 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.08s ease;
    }}
    /* Inputs */
    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input {{
        background: rgba(255, 255, 255, 0.7) !important;
        color: {t["tx"]} !important;
        border: 1px solid {t["border"]} !important;
        border-radius: 12px !important;
        font-family: 'Figtree', sans-serif !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }}
    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stNumberInput"] input:focus {{
        border-color: {t["primary"]} !important;
        box-shadow: 0 0 0 3px rgba(230, 57, 70, 0.1) !important;
    }}
    div[data-testid="stSelectbox"] > div {{
        background: rgba(255, 255, 255, 0.7) !important;
        color: {t["tx"]} !important;
        border-color: {t["border"]} !important;
        border-radius: 12px !important;
    }}

    /* Tabs — glassmorphic */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        background: rgba(255, 255, 255, 0.5);
        backdrop-filter: blur(8px);
        border-radius: 14px;
        padding: 4px;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }}
    .stTabs [data-baseweb="tab"] {{
        font-family: 'Bebas Neue', sans-serif;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-size: 13px;
        border-radius: 10px;
        color: {t["tx2"]};
        transition: all 0.2s ease;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {t["primary"]}, {t["hot"]}) !important;
        color: {t["ink"]} !important;
        box-shadow: 0 2px 8px rgba(230, 57, 70, 0.3);
    }}

    /* File uploader */
    div[data-testid="stFileUploader"] {{
        background: rgba(255, 255, 255, 0.5);
        border: 2px dashed {t["border"]};
        border-radius: 16px;
        padding: 16px;
        transition: border-color 0.2s ease;
    }}
    div[data-testid="stFileUploader"]:hover {{
        border-color: {t["primary"]};
    }}

    /* DataFrames — contrasting background + bold headers */
    div[data-testid="stDataFrame"] {{
        border: 2px solid #d4c5b0 !important;
        border-radius: 12px !important;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.06), 0 0 0 1px rgba(0,0,0,0.03) !important;
        background: #faf8f5 !important;
    }}
    div[data-testid="stDataFrame"] [data-testid="glideDataEditor"] {{
        background: #faf8f5 !important;
    }}
    /* Bold column headers */
    div[data-testid="stDataFrame"] th,
    div[data-testid="stDataFrame"] [role="columnheader"],
    div[data-testid="stDataFrame"] .gdg-header-cell {{
        font-weight: 700 !important;
        font-family: 'Figtree', sans-serif !important;
        background: #ede8e0 !important;
    }}
    /* Table cell background for contrast against page bg */
    div[data-testid="stDataFrame"] td,
    div[data-testid="stDataFrame"] [role="gridcell"] {{
        background: #faf8f5 !important;
    }}
    /* Left accent border for visual anchor */
    div[data-testid="stDataFrame"]::before {{
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: linear-gradient(180deg, #e65c00, #cc5200);
        border-radius: 12px 0 0 12px;
        z-index: 1;
    }}
    div[data-testid="stDataFrame"] {{
        position: relative;
    }}

    /* Sidebar */
    .stSidebar {{
        background: linear-gradient(180deg, #e65c00 0%, #cc5200 100%) !important;
        border-right: none !important;
    }}
    .stSidebar, .stSidebar * {{
        color: #ffffff !important;
    }}
    .stSidebar a {{
        color: rgba(255,255,255,0.9) !important;
        font-weight: 600 !important;
        font-size: 15px !important;
    }}
    .stSidebar a:hover {{
        color: #ffffff !important;
    }}
    .stSidebar [data-testid="stSidebarNav"] li a[aria-current="page"] {{
        background: rgba(0,0,0,0.15) !important;
        border-radius: 8px;
    }}
    .stSidebar [data-testid="stSidebarNav"] li a[aria-current="page"] span {{
        color: #ffffff !important;
        font-weight: 800 !important;
    }}
    .stSidebar [data-testid="stSidebarHeader"] {{
        background: linear-gradient(135deg, #1a1a2e, #16213e) !important;
        border-radius: 0 !important;
        margin: 0 -1rem !important;
        padding: 0 1rem !important;
    }}
    .stSidebar button[data-testid="stBaseButton-header"] {{
        color: #ffffff !important;
    }}
    .stSidebar button[data-testid="stBaseButton-header"] svg {{
        fill: #ffffff !important;
        stroke: #ffffff !important;
    }}

    /* Expanders */
    div[data-testid="stExpander"] {{
        background: rgba(255, 255, 255, 0.6) !important;
        border: 1px solid {t["border"]} !important;
        border-radius: 14px !important;
        backdrop-filter: blur(8px);
    }}
    div[data-testid="stExpander"] summary {{
        color: {t["tx"]} !important;
        font-family: 'Figtree', sans-serif;
    }}

    /* Metrics */
    div[data-testid="stMetric"] {{
        background: transparent;
        color: {t["tx"]} !important;
    }}
    div[data-testid="stMetric"] label {{
        color: {t["tx2"]} !important;
        font-family: 'Bebas Neue', sans-serif !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
    }}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: {t["tx"]} !important;
        font-family: 'IBM Plex Mono', monospace !important;
    }}

    /* MultiSelect */
    div[data-testid="stMultiSelect"] > div {{
        background: rgba(255, 255, 255, 0.7) !important;
        color: {t["tx"]} !important;
        border-color: {t["border"]} !important;
        border-radius: 12px !important;
    }}

    /* Slider, markdown, captions, checkboxes, radio, alerts */
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
        border-radius: 12px !important;
    }}

    /* Bold ALL titles — subheaders, headers, markdown bold */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    h1, h2, h3,
    [data-testid="stSubheader"],
    .stSubheader {{
        font-weight: 700 !important;
        font-family: 'Figtree', sans-serif !important;
        color: {t["tx"]} !important;
    }}
    .stMarkdown h1, h1 {{ font-size: 24px !important; }}
    .stMarkdown h2, h2, [data-testid="stSubheader"], .stSubheader {{ font-size: 18px !important; }}
    .stMarkdown h3, h3 {{ font-size: 15px !important; }}

    /* Orange action buttons — "Refresh Stats", "Sync Yahoo", etc. */
    .stButton > button[kind="secondary"],
    .stButton > button[data-testid="stBaseButton-secondary"] {{
        background: linear-gradient(135deg, #e65c00, #cc5200) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border: none !important;
        transform: translateY(-1px);
        box-shadow: 0 3px 0 #993d00, 0 4px 12px rgba(230, 92, 0, 0.25) !important;
    }}
    .stButton > button[kind="secondary"]:hover,
    .stButton > button[data-testid="stBaseButton-secondary"]:hover {{
        background: linear-gradient(135deg, #ff6d00, #e65c00) !important;
        color: #ffffff !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 0 #993d00, 0 6px 16px rgba(230, 92, 0, 0.35) !important;
    }}
    .stButton > button[kind="secondary"]:active,
    .stButton > button[data-testid="stBaseButton-secondary"]:active {{
        transform: translateY(0) scale(0.98);
        box-shadow: 0 1px 0 #993d00, 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    }}

    /* ── TOOLTIP (CSS-based, on title attr) ─── */
    [title] {{
        cursor: help;
    }}

    /* ── RESPONSIVE ──────────────────────────── */
    @media (max-width: 768px) {{
        .page-title {{
            font-size: 22px !important;
            letter-spacing: 2px !important;
            padding: 6px 20px !important;
        }}
        .splash-title {{
            font-size: 36px;
        }}
        .hero .p-name {{
            font-size: 24px;
            letter-spacing: 1px;
        }}
        .hero .score-badge {{
            font-size: 16px;
            padding: 6px 10px;
        }}
        .verdict-banner .verdict-text {{
            font-size: 22px;
            letter-spacing: 2px;
        }}
        .wizard-step {{
            font-size: 11px;
            letter-spacing: 1px;
            padding: 8px 12px;
        }}
    }}
    @media (max-width: 480px) {{
        .page-title {{
            font-size: 18px !important;
            letter-spacing: 1px !important;
            padding: 5px 14px !important;
        }}
        .splash-title {{
            font-size: 28px;
        }}
        .hero .p-name {{
            font-size: 20px;
            letter-spacing: 0px;
        }}
        .verdict-banner .verdict-text {{
            font-size: 18px;
            letter-spacing: 1px;
        }}
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Rename sidebar "app" → "Connect League" and inject logo + branding via JS
    import streamlit.components.v1 as components

    components.html(
        """<script>
        (function setup() {
            // Rename sidebar nav items
            const nav = parent.document.querySelector('[data-testid="stSidebarNav"]');
            if (!nav) { setTimeout(setup, 200); return; }
            const spans = nav.querySelectorAll('li a span');
            spans.forEach(function(span) {
                var t = span.textContent.trim();
                if (t === 'app' || t === 'Heater') { span.textContent = 'Connect League'; }
                if (t === 'Mock Draft') { span.textContent = 'Draft Simulator'; }
            });

            // Inject logo + HEATER branding into sidebar header
            const header = parent.document.querySelector('[data-testid="stSidebarHeader"]');
            if (header && !header.querySelector('.heater-logo')) {
                const logoDiv = parent.document.createElement('div');
                logoDiv.className = 'heater-logo';
                logoDiv.style.cssText = 'display:flex;align-items:center;justify-content:center;gap:6px;padding:14px 6px 10px 6px;';
                logoDiv.innerHTML = '<svg width="36" height="36" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0;">'
                    + '<defs><radialGradient id="sbl" cx="40%" cy="35%" r="55%">'
                    + '<stop offset="0%" stop-color="#fff"/><stop offset="60%" stop-color="#f5f0e8"/>'
                    + '<stop offset="100%" stop-color="#e8ddd0"/></radialGradient>'
                    + '<linearGradient id="ssl" x1="0" y1="0" x2="1" y2="0">'
                    + '<stop offset="0%" stop-color="rgba(255,255,255,0.9)"/>'
                    + '<stop offset="100%" stop-color="rgba(255,255,255,0)"/></linearGradient></defs>'
                    + '<line x1="2" y1="22" x2="18" y2="28" stroke="url(#ssl)" stroke-width="2.5" stroke-linecap="round"/>'
                    + '<line x1="0" y1="32" x2="16" y2="33" stroke="url(#ssl)" stroke-width="2" stroke-linecap="round"/>'
                    + '<line x1="4" y1="42" x2="17" y2="39" stroke="url(#ssl)" stroke-width="1.5" stroke-linecap="round"/>'
                    + '<circle cx="38" cy="32" r="18" fill="url(#sbl)"/>'
                    + '<circle cx="38" cy="32" r="18" fill="none" stroke="rgba(255,255,255,0.4)" stroke-width="0.8"/>'
                    + '<ellipse cx="39" cy="50" rx="12" ry="2" fill="rgba(0,0,0,0.1)"/>'
                    + '<path d="M29 16 C24 22 22 27 23 32 C22 37 24 42 29 48" fill="none" stroke="#e63946" stroke-width="1.6" stroke-linecap="round"/>'
                    + '<path d="M47 16 C52 22 54 27 53 32 C54 37 52 42 47 48" fill="none" stroke="#e63946" stroke-width="1.6" stroke-linecap="round"/>'
                    + '<line x1="27" y1="19" x2="30" y2="20" stroke="#e63946" stroke-width="1" stroke-linecap="round"/>'
                    + '<line x1="25" y1="23" x2="28" y2="23.5" stroke="#e63946" stroke-width="1" stroke-linecap="round"/>'
                    + '<line x1="24" y1="27.5" x2="27" y2="27.5" stroke="#e63946" stroke-width="1" stroke-linecap="round"/>'
                    + '<line x1="24" y1="36.5" x2="27" y2="36.5" stroke="#e63946" stroke-width="1" stroke-linecap="round"/>'
                    + '<line x1="25" y1="41" x2="28" y2="40.5" stroke="#e63946" stroke-width="1" stroke-linecap="round"/>'
                    + '<line x1="27" y1="45" x2="30" y2="44" stroke="#e63946" stroke-width="1" stroke-linecap="round"/>'
                    + '<line x1="49" y1="19" x2="46" y2="20" stroke="#e63946" stroke-width="1" stroke-linecap="round"/>'
                    + '<line x1="51" y1="23" x2="48" y2="23.5" stroke="#e63946" stroke-width="1" stroke-linecap="round"/>'
                    + '<line x1="52" y1="27.5" x2="49" y2="27.5" stroke="#e63946" stroke-width="1" stroke-linecap="round"/>'
                    + '<line x1="52" y1="36.5" x2="49" y2="36.5" stroke="#e63946" stroke-width="1" stroke-linecap="round"/>'
                    + '<line x1="51" y1="41" x2="48" y2="40.5" stroke="#e63946" stroke-width="1" stroke-linecap="round"/>'
                    + '<line x1="49" y1="45" x2="46" y2="44" stroke="#e63946" stroke-width="1" stroke-linecap="round"/>'
                    + '<circle cx="38" cy="32" r="21" fill="none" stroke="rgba(255,255,255,0.3)" stroke-width="1.2"/>'
                    + '</svg>'
                    + '<span style="font-family:Bebas Neue,sans-serif;font-size:28px;letter-spacing:4px;'
                    + 'color:#ffffff;font-weight:700;text-shadow:0 1px 4px rgba(0,0,0,0.25);line-height:1;">HEATER</span>';
                header.insertBefore(logoDiv, header.firstChild);
            }

            var firstSpan = nav.querySelector('li:first-child a span');
            if (!firstSpan || firstSpan.textContent.trim() === 'app') { setTimeout(setup, 200); }
        })();
        </script>""",
        height=0,
    )

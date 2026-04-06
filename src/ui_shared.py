"""Shared UI constants, theme system, CSS injection, and SVG icons — Heater Edition."""

from __future__ import annotations

import html as _html
import math as _math
import time as _time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

try:
    import streamlit as st
except ImportError:
    st = None  # type: ignore[assignment]

from src.valuation import LeagueConfig as _LC_Class

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

_LC = _LC_Class()
ROSTER_CONFIG = _LC.roster_slots
HITTING_CATEGORIES = _LC.hitting_categories
PITCHING_CATEGORIES = _LC.pitching_categories
ALL_CATEGORIES = _LC.all_categories

# ── Metric Tooltips ─────────────────────────────────────────────────

METRIC_TOOLTIPS = {
    "sgp": (
        "Standings Gained Points — how many standings points "
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
        "ACCEPT means your team improves overall across the 12 scoring categories. "
        "The grade (A+ to F) reflects surplus Standings Gained Points from the trade."
    ),
    "trade_verdict_legacy": (
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
        "Composite Score — weighted sum of Z-scores across all 12 "
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
    "obp": (
        "On-Base Percentage — fraction of plate appearances reaching base "
        "(hits + walks + hit-by-pitch). Higher is better. League average is ~.320."
    ),
    "l": ("Losses — pitcher losses. LOWER is better (inverse category)."),
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
    "replacement_penalty": (
        "Category Replacement Cost Penalty — measures how much production you lose "
        "in each category that cannot be recovered from the free agent pool. "
        "Scarce categories (like Saves) are penalized more heavily because there are "
        "fewer replacement options available. Discounted by 50% to account for "
        "free agent pool turnover over the season."
    ),
    "roster_move": (
        "Roster Move — in uneven trades, you must adjust your roster to stay at 23 players. "
        "When receiving more players than you give (e.g., 1-for-2), the system identifies "
        "the worst bench player to drop. When giving more than you receive (e.g., 2-for-1), "
        "the system identifies the best available free agent to pick up. "
        "Only starting lineup production counts — bench players contribute zero to standings."
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


# ── Styled Table Helper ────────────────────────────────────────────


def render_styled_table(df, hide_index=True, max_height=None):
    """Render a DataFrame as an HTML table with navy headers and full CSS control.

    Use this instead of st.dataframe() when navy column headers are needed.
    Glide Data Grid (used by st.dataframe) renders on canvas and can't be
    CSS-styled, so we render as an HTML table instead.

    Args:
        df: pandas DataFrame to render.
        hide_index: If True (default), hides the DataFrame index column.
        max_height: Optional max height in pixels for scrollable tables.
    """
    html = df.to_html(index=not hide_index, classes="heater-table", escape=False)
    scroll_style = f"max-height:{max_height}px;overflow-y:auto;" if max_height else ""
    st.markdown(
        f'<div class="heater-table-wrap" style="{scroll_style}">{html}</div>',
        unsafe_allow_html=True,
    )


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

    /* ── Glide Data Grid root-level theme overrides ── */
    :root {{
        --gdg-bg-header: #16213e !important;
        --gdg-bg-header-has-focus: #1a1a2e !important;
        --gdg-bg-header-hovered: #1e2a45 !important;
        --gdg-text-header: #ffffff !important;
        --gdg-bg-cell: #faf8f5 !important;
        --gdg-bg-cell-medium: #f5f2ed !important;
        --gdg-text-dark: #1d1d1f !important;
        --gdg-border-color: #d4c5b0 !important;
        --gdg-header-font-style: 700 14px Figtree, sans-serif !important;
    }}

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
    .sec-label {{
        font-size: 12px !important;
        font-weight: 700 !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        color: {t["tx2"]} !important;
        margin-bottom: 4px !important;
    }}

    /* ── GLASS CARD (Glassmorphism) ───────────── */
    .glass {{
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 12px;
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
        border-radius: 12px;
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
    .badge-last-chance {{
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        background: {t["danger"]};
        color: #ffffff;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-left: 8px;
        animation: pulse-lc 1.5s infinite;
    }}
    @keyframes pulse-lc {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.6; }}
    }}

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
        border-radius: 12px;
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
        border-radius: 12px;
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
        letter-spacing: 4px !important;
        font-style: italic !important;
        font-weight: 700 !important;
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
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        text-shadow: 0 1px 4px rgba(0,0,0,0.25) !important;
    }}
    .page-title-wrap {{
        text-align: center !important;
        margin-top: 4px !important;
        margin-bottom: 2px !important;
    }}
    .page-title span {{
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        text-shadow: 0 1px 4px rgba(0,0,0,0.25) !important;
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
        gap: 4px;
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
        padding: 8px 16px !important;
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
    /* Glide Data Grid canvas theming via CSS custom properties (belt-and-suspenders) */
    div[data-testid="stDataFrame"] [data-testid="glideDataEditor"],
    div[data-testid="stDataFrame"] .dvn-scroller,
    div[data-testid="stDataFrame"] {{
        --gdg-bg-header: #16213e !important;
        --gdg-bg-header-has-focus: #1a1a2e !important;
        --gdg-bg-header-hovered: #1e2a45 !important;
        --gdg-text-header: #ffffff !important;
        --gdg-bg-cell: #faf8f5 !important;
        --gdg-bg-cell-medium: #f5f2ed !important;
        --gdg-text-dark: #1d1d1f !important;
        --gdg-border-color: #d4c5b0 !important;
        --gdg-header-font-style: 700 14px Figtree, sans-serif !important;
    }}
    div[data-testid="stDataFrame"] [data-testid="glideDataEditor"] {{
        background: #faf8f5 !important;
    }}

    /* Override navy secondaryBackgroundColor on non-dataframe elements */
    pre, code, .stCodeBlock, [data-testid="stCodeBlock"] {{
        background: rgba(245, 242, 237, 0.9) !important;
        color: {t["tx"]} !important;
    }}
    .stToast, [data-testid="stToast"] {{
        background: #ffffff !important;
        color: {t["tx"]} !important;
    }}
    [data-testid="stPopover"], .stPopover {{
        background: #ffffff !important;
        color: {t["tx"]} !important;
    }}
    [data-testid="stChatMessage"], .stChatMessage {{
        background: rgba(255, 255, 255, 0.7) !important;
        color: {t["tx"]} !important;
    }}
    .stForm, [data-testid="stForm"] {{
        background: rgba(255, 255, 255, 0.5) !important;
        border: 1px solid {t["border"]} !important;
        border-radius: 14px !important;
    }}
    [data-testid="stVerticalBlockBorderWrapper"] {{
        background: transparent !important;
    }}
    div[data-testid="stColumn"] {{
        background: transparent !important;
    }}
    [data-testid="stBottomBlockContainer"] {{
        background: {t["bg"]} !important;
    }}
    /* Multiselect tags/pills — override navy bg */
    [data-baseweb="tag"] {{
        background: rgba(230, 57, 70, 0.1) !important;
        color: {t["primary"]} !important;
    }}
    /* Select/dropdown menus */
    [data-baseweb="popover"], [data-baseweb="menu"] {{
        background: #ffffff !important;
        color: {t["tx"]} !important;
    }}
    [data-baseweb="select"] [data-baseweb="input"] {{
        background: rgba(255, 255, 255, 0.7) !important;
    }}
    /* Number input steppers */
    div[data-testid="stNumberInput"] [data-testid="stNumberInputStepUp"],
    div[data-testid="stNumberInput"] [data-testid="stNumberInputStepDown"] {{
        background: rgba(255, 255, 255, 0.7) !important;
    }}
    /* Progress bars */
    .stProgress > div > div {{
        background: rgba(245, 242, 237, 0.5) !important;
    }}
    .stProgress > div > div > div {{
        background: linear-gradient(90deg, {t["primary"]}, {t["hot"]}) !important;
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

    /* Styled HTML tables (render_styled_table helper) */
    .heater-table-wrap {{
        border: 2px solid #d4c5b0;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.06), 0 0 0 1px rgba(0,0,0,0.03);
        background: #faf8f5;
        position: relative;
        border-left: 4px solid #e65c00;
    }}
    .heater-table {{
        width: 100%;
        border-collapse: collapse;
        font-family: 'Figtree', sans-serif;
        font-size: 13px;
        color: {t["tx"]};
    }}
    .heater-table thead th {{
        background: linear-gradient(135deg, #16213e, #1a1a2e) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-family: 'Figtree', sans-serif !important;
        font-size: 12px !important;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        padding: 10px 12px !important;
        border: none !important;
        text-align: left;
        white-space: nowrap;
        position: sticky;
        top: 0;
        z-index: 2;
    }}
    .heater-table thead th:first-child {{
        border-radius: 0;
    }}
    .heater-table thead th:last-child {{
        border-radius: 0;
    }}
    .heater-table tbody td {{
        padding: 8px 12px !important;
        border-bottom: 1px solid #e8e0d4 !important;
        border-top: none !important;
        border-left: none !important;
        border-right: none !important;
        background: #faf8f5 !important;
        font-family: 'Figtree', sans-serif;
        font-size: 13px !important;
        color: {t["tx"]};
    }}
    .heater-table tbody td:first-child {{
        white-space: nowrap !important;
        font-weight: 600 !important;
    }}
    .heater-table tbody tr:hover td {{
        background: #f0ece4 !important;
    }}
    .heater-table tbody tr:last-child td {{
        border-bottom: none !important;
    }}

    /* Sidebar — navy top for header, orange below */
    .stSidebar {{
        background: linear-gradient(180deg, #16213e 0%, #16213e 70px, #e65c00 70px, #cc5200 100%) !important;
        border-right: none !important;
    }}
    .stSidebar, .stSidebar * {{
        color: #ffffff !important;
    }}
    .stSidebar a {{
        color: rgba(255,255,255,0.9) !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }}
    .stSidebar a:hover {{
        color: #ffffff !important;
    }}
    .stSidebar [data-testid="stSidebarNav"] li {{
        margin-bottom: 4px !important;
    }}
    .stSidebar [data-testid="stSidebarNav"] li a {{
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;
        padding: 10px 12px !important;
        border-radius: 10px !important;
        transition: background 0.2s ease !important;
    }}
    .stSidebar [data-testid="stSidebarNav"] li a:hover {{
        background: rgba(255,255,255,0.1) !important;
    }}
    .stSidebar [data-testid="stSidebarNav"] li a[aria-current="page"] {{
        background: rgba(0,0,0,0.18) !important;
        border-radius: 10px !important;
    }}
    .stSidebar [data-testid="stSidebarNav"] li a[aria-current="page"] span {{
        color: #ffffff !important;
        font-weight: 800 !important;
    }}
    .stSidebar [data-testid="stSidebarNav"] li a .nav-icon {{
        flex-shrink: 0;
        display: inline-flex;
    }}
    .stSidebar [data-testid="stSidebarHeader"] {{
        background: transparent !important;
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

    /* ── RECOMMENDATION BANNER ────────────────── */
    .reco-banner {{
        background: rgba(255, 255, 255, 0.7) !important;
        backdrop-filter: blur(16px) saturate(150%) !important;
        -webkit-backdrop-filter: blur(16px) saturate(150%) !important;
        border: 1px solid rgba(255, 255, 255, 0.4) !important;
        border-left: 4px solid {t["hot"]} !important;
        border-radius: 12px !important;
        padding: 10px 16px !important;
        margin-bottom: 12px !important;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06) !important;
    }}
    .reco-banner-teaser {{
        font-family: 'Figtree', sans-serif !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        color: {t["tx"]} !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 8px !important;
        line-height: 1.4 !important;
    }}
    .reco-banner-detail {{
        font-family: 'Figtree', sans-serif !important;
        font-size: 13px !important;
        color: {t["tx2"]} !important;
        padding-top: 8px !important;
        line-height: 1.5 !important;
        animation: slideUp 0.3s ease-out !important;
    }}

    /* ── MATCHUP TICKER ──────────────────────── */
    .matchup-ticker {{
        background: linear-gradient(135deg, rgba(255,255,255,0.75), rgba(255,255,255,0.55)) !important;
        backdrop-filter: blur(16px) saturate(160%) !important;
        -webkit-backdrop-filter: blur(16px) saturate(160%) !important;
        border: 1px solid rgba(255,255,255,0.4) !important;
        border-left: 3px solid {t["hot"]} !important;
        border-radius: 10px !important;
        padding: 8px 14px !important;
        margin: 4px 0 12px !important;
        display: flex !important;
        align-items: center !important;
        gap: 12px !important;
        font-family: 'Figtree', sans-serif !important;
        font-size: 13px !important;
        color: {t["tx"]} !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05) !important;
        flex-wrap: wrap !important;
    }}
    .matchup-ticker-week {{
        font-family: 'Bebas Neue', sans-serif !important;
        font-size: 11px !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
        color: {t["tx2"]} !important;
        white-space: nowrap !important;
    }}
    .matchup-ticker-score {{
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        white-space: nowrap !important;
    }}
    .matchup-ticker-vs {{
        font-size: 12px !important;
        color: {t["tx2"]} !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }}
    .matchup-ticker-status {{
        font-size: 11px !important;
        font-weight: 600 !important;
        padding: 2px 8px !important;
        border-radius: 6px !important;
        white-space: nowrap !important;
    }}
    .matchup-ticker .cat-row {{
        display: flex !important;
        justify-content: space-between !important;
        padding: 2px 0 !important;
        font-size: 12px !important;
        font-family: 'IBM Plex Mono', monospace !important;
        border-bottom: 1px solid {t["border"]}33 !important;
    }}
    .matchup-ticker .cat-row:last-child {{
        border-bottom: none !important;
    }}
    .matchup-ticker .cat-win {{ color: {t["green"]} !important; font-weight: 600 !important; }}
    .matchup-ticker .cat-loss {{ color: {t["primary"]} !important; font-weight: 600 !important; }}
    .matchup-ticker .cat-tie {{ color: {t["tx2"]} !important; }}
    @media (max-width: 768px) {{
        .matchup-ticker {{
            flex-direction: column !important;
            align-items: flex-start !important;
            gap: 4px !important;
        }}
    }}

    /* ── CONTEXT CARD (sidebar panels) ────────── */
    .context-card {{
        background: rgba(255, 255, 255, 0.6) !important;
        backdrop-filter: blur(12px) saturate(140%) !important;
        -webkit-backdrop-filter: blur(12px) saturate(140%) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 10px !important;
        padding: 12px 14px !important;
        margin-bottom: 8px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
    }}
    .context-card-title {{
        font-family: 'Bebas Neue', sans-serif !important;
        font-size: 10px !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        color: {t["tx2"]} !important;
        margin-bottom: 6px !important;
        padding-bottom: 4px !important;
        border-bottom: 1px solid {t["border"]} !important;
    }}

    /* ── COMPACT TABLE (ESPN-style) ───────────── */
    .compact-table-wrap {{
        overflow-x: auto !important;
        overflow-y: auto !important;
        border-left: 3px solid {t["hot"]} !important;
        border-radius: 8px !important;
        background: {t["card"]} !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.06) !important;
        margin-bottom: 12px !important;
    }}
    .compact-table {{
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 13px !important;
        border-collapse: collapse !important;
        white-space: nowrap !important;
        width: 100% !important;
        color: {t["tx"]} !important;
    }}
    .compact-table th {{
        background: linear-gradient(135deg, #16213e, #1a1a2e) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 11px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        padding: 6px 10px !important;
        position: sticky !important;
        top: 0 !important;
        z-index: 2 !important;
        border-bottom: 2px solid {t["hot"]} !important;
    }}
    .compact-table td {{
        padding: 5px 10px !important;
        border-bottom: 1px solid {t["border"]} !important;
        font-size: 13px !important;
        font-variant-numeric: tabular-nums !important;
        background: {t["card"]} !important;
    }}
    .compact-table tr:hover td {{
        background: rgba(255, 109, 0, 0.06) !important;
    }}
    .th-hit {{
        border-bottom: 3px solid {t["hot"]} !important;
    }}
    .th-pit {{
        border-bottom: 3px solid {t["sky"]} !important;
    }}
    .col-name {{
        position: sticky !important;
        left: 0 !important;
        z-index: 3 !important;
        background: {t["card"]} !important;
        font-family: 'Figtree', sans-serif !important;
        font-weight: 600 !important;
        font-size: 12px !important;
        min-width: 140px !important;
        max-width: 200px !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
        border-right: 1px solid {t["border"]} !important;
        box-shadow: 2px 0 4px rgba(0, 0, 0, 0.06) !important;
    }}
    /* Headshot thumbnail in name column */
    .col-name img {{
        flex-shrink: 0 !important;
    }}
    /* Corner cell: sticky both top+left — needs highest z-index */
    .compact-table thead th.col-name {{
        z-index: 4 !important;
        background: linear-gradient(135deg, #16213e, #1a1a2e) !important;
        box-shadow: 2px 0 4px rgba(0, 0, 0, 0.12) !important;
    }}
    .compact-table tr:hover .col-name {{
        background: #ffffff !important;
    }}
    .row-start td {{
        background: rgba(45, 106, 79, 0.04) !important;
    }}
    .row-start td.col-name {{
        background: #f7fbf9 !important;
    }}
    .row-bench td {{
        background: rgba(230, 57, 70, 0.04) !important;
    }}
    .row-bench td.col-name {{
        background: #fdf7f7 !important;
    }}
    .health-dot {{
        display: inline-block !important;
        width: 8px !important;
        height: 8px !important;
        border-radius: 50% !important;
        margin-right: 4px !important;
        vertical-align: middle !important;
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
        .compact-table {{
            font-size: 11px !important;
        }}
        .compact-table th {{
            font-size: 10px !important;
            padding: 4px 8px !important;
        }}
        .compact-table td {{
            padding: 4px 8px !important;
            font-size: 11px !important;
        }}
        .col-name {{
            min-width: 100px !important;
            font-size: 11px !important;
        }}
        .reco-banner-teaser {{
            font-size: 13px !important;
        }}
    }}
    @media (max-width: 1024px) {{
        .col-name {{
            min-width: 120px !important;
            font-size: 11px !important;
        }}
        .compact-table tbody td {{
            font-size: 12px !important;
            padding: 4px 8px !important;
        }}
        .compact-table thead th {{
            font-size: 10px !important;
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
        .reco-banner-teaser svg {{
            display: none !important;
        }}
        .reco-banner {{
            padding: 8px 10px !important;
        }}
    }}

    /* ── PRINT ────────────────────────────────── */
    @media print {{
        .context-card, [data-testid="stSidebar"] {{
            display: none !important;
        }}
        .compact-table-wrap {{
            border-left: none !important;
            box-shadow: none !important;
        }}
        .compact-table {{
            font-size: 10px !important;
        }}
        .reco-banner {{
            border: 1px solid #ccc !important;
            box-shadow: none !important;
            backdrop-filter: none !important;
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
            // Rename sidebar nav items and inject page icons
            const nav = parent.document.querySelector('[data-testid="stSidebarNav"]');
            if (!nav) { setTimeout(setup, 200); return; }

            // Icon SVGs — 20x20 white stroke icons
            var icons = {
                'Connect League': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>',
                'My Team': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>',
                'Draft Simulator': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>',
                'Trade Analyzer': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/></svg>',
                'Player Compare': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/><line x1="3" y1="20" x2="21" y2="20"/></svg>',
                'Free Agents': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="8" y1="11" x2="14" y2="11"/><line x1="11" y1="8" x2="11" y2="14"/></svg>',
                'Lineup': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
                'Closer Monitor': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="7"/><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"/></svg>',
                'Standings': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>',
                'Leaders': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>',
                'Trade Finder': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>',
                'Matchup Planner': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>'
            };

            // Map original names → display names
            var nameMap = {
                'app': 'Connect League',
                'Heater': 'Connect League',
                'Mock Draft': 'Draft Simulator'
            };

            var links = nav.querySelectorAll('li a');
            links.forEach(function(link) {
                var span = link.querySelector('span');
                if (!span) return;
                var t = span.textContent.trim();

                // Rename if needed
                if (nameMap[t]) { span.textContent = nameMap[t]; t = nameMap[t]; }

                // Inject icon if not already present
                if (!link.querySelector('.nav-icon') && icons[t]) {
                    var iconSpan = parent.document.createElement('span');
                    iconSpan.className = 'nav-icon';
                    iconSpan.innerHTML = icons[t];
                    link.insertBefore(iconSpan, span);
                }
            });

            // Inject logo + HEATER branding into sidebar header
            const header = parent.document.querySelector('[data-testid="stSidebarHeader"]');
            if (header && !header.querySelector('.heater-logo')) {
                const logoDiv = parent.document.createElement('div');
                logoDiv.className = 'heater-logo';
                logoDiv.style.cssText = 'display:flex;flex-direction:column;align-items:center;padding:14px 6px 10px 6px;';
                logoDiv.innerHTML = '<div style="display:flex;align-items:center;gap:6px;"><svg width="36" height="36" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0;">'
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
                    + '<span style="font-family:Bebas Neue,sans-serif;font-size:28px;letter-spacing:4px;font-style:italic;'
                    + 'color:#ffffff;font-weight:700;text-shadow:0 1px 4px rgba(0,0,0,0.25);line-height:1;">HEATER</span></div>'
                    + '<div style="width:100%;height:3px;background:linear-gradient(90deg,#e65c00,#ff8c00);border-radius:2px;margin-top:3px;"></div>';
                header.insertBefore(logoDiv, header.firstChild);
            }

            // Inject orange bar under page title badges (with retry for timing)
            function injectTitleBars() {
                var pts = parent.document.querySelectorAll('.page-title');
                pts.forEach(function(pt) {
                    if (!pt.nextElementSibling || !pt.nextElementSibling.classList.contains('page-title-bar')) {
                        var bar = parent.document.createElement('div');
                        bar.className = 'page-title-bar';
                        bar.style.cssText = 'width:180px;height:3px;background:linear-gradient(90deg,#e65c00,#ff8c00);border-radius:2px;margin:3px auto 0;';
                        pt.parentNode.insertBefore(bar, pt.nextSibling);
                    }
                });
                if (pts.length === 0) { setTimeout(injectTitleBars, 300); }
            }
            injectTitleBars();

            var firstLink = nav.querySelector('li:first-child a');
            if (!firstLink || !firstLink.querySelector('.nav-icon')) { setTimeout(setup, 200); }
        })();
        </script>""",
        height=0,
    )


def build_category_heatmap_html(user_totals: dict, all_totals: list[dict]) -> str:
    """Build HTML heatmap grid for 12 scoring categories.

    Args:
        user_totals: Dict of category totals for the user's roster.
        all_totals: List of dicts (one per team) with category totals.

    Returns:
        HTML table string. Returns empty string when *all_totals* is empty.
    """
    if not all_totals:
        return ""

    cat_keys = ["R", "HR", "RBI", "SB", "AVG", "OBP", "W", "L", "SV", "K", "ERA", "WHIP"]
    cat_display = {
        "R": "Runs",
        "HR": "Home Runs",
        "RBI": "Runs Batted In",
        "SB": "Stolen Bases",
        "AVG": "Batting Average",
        "OBP": "On-Base Percentage",
        "W": "Wins",
        "L": "Losses",
        "SV": "Saves",
        "K": "Strikeouts",
        "ERA": "Earned Run Average",
        "WHIP": "Walks + Hits per Inning Pitched",
    }
    rate_fmt = {"AVG", "OBP", "ERA", "WHIP"}
    # Inverse stats: lower is better
    inverse_cats = {"L", "ERA", "WHIP"}

    num_teams = len(all_totals)

    rows_html = ""
    for cat in cat_keys:
        my_val = user_totals.get(cat, 0)

        # Compute rank (1 = best)
        if cat in inverse_cats:
            # Lower is better — count teams with strictly lower value
            # But if user has 0 IP, pitching inverse stats (ERA, WHIP, L) are
            # meaningless — treat as worst rank instead of best.
            pitching_inverse = {"ERA", "WHIP", "L"}
            if cat in pitching_inverse and "ip" in user_totals and user_totals["ip"] == 0:
                rank = num_teams  # worst rank
            else:
                rank = sum(1 for t in all_totals if t.get(cat, 0) < my_val) + 1
        else:
            # Higher is better — count teams with strictly higher value
            rank = sum(1 for t in all_totals if t.get(cat, 0) > my_val) + 1

        # Percentile thresholds based on rank among teams
        # Top 25% = green (Strong), Bottom 25% = red (Weak), Middle = yellow (Average)
        top_cutoff = max(int(num_teams * 0.25), 1)
        bottom_cutoff = num_teams - max(int(num_teams * 0.25), 1) + 1

        if rank <= top_cutoff:
            bg_color = f"{T['green']}22"
            status_color = T["green"]
            status_text = "Strong"
        elif rank >= bottom_cutoff:
            bg_color = f"{T['primary']}22"
            status_color = T["primary"]
            status_text = "Weak"
        else:
            bg_color = f"{T['hot']}22"
            status_color = T["hot"]
            status_text = "Average"

        # Format value
        if cat in rate_fmt:
            val_str = f"{my_val:.3f}"
        else:
            val_str = str(int(my_val))

        # Ordinal suffix for rank
        if rank % 10 == 1 and rank != 11:
            suffix = "st"
        elif rank % 10 == 2 and rank != 12:
            suffix = "nd"
        elif rank % 10 == 3 and rank != 13:
            suffix = "rd"
        else:
            suffix = "th"

        rows_html += (
            f'<tr style="background:{bg_color}">'
            f'<td style="padding:6px 12px;font-weight:600;">{cat_display[cat]}</td>'
            f'<td style="padding:6px 12px;text-align:right;">{val_str}</td>'
            f'<td style="padding:6px 12px;text-align:center;">{rank}{suffix}</td>'
            f'<td style="padding:6px 12px;text-align:center;color:{status_color};font-weight:700;">{status_text}</td>'
            f"</tr>"
        )

    html = (
        f'<table class="heatmap-grid" style="width:100%;border-collapse:collapse;'
        f"background:{T['card']};border-radius:12px;overflow:hidden;margin-top:16px;"
        f'font-family:Figtree,sans-serif;font-size:14px;color:{T["tx"]};">'
        f'<tr style="background:{T["bg"]};border-bottom:2px solid {T["card_h"]};">'
        f'<th style="padding:8px 12px;text-align:left;font-weight:700;">Category</th>'
        f'<th style="padding:8px 12px;text-align:right;font-weight:700;">My Total</th>'
        f'<th style="padding:8px 12px;text-align:center;font-weight:700;">Rank</th>'
        f'<th style="padding:8px 12px;text-align:center;font-weight:700;">Status</th>'
        f"</tr>"
        f"{rows_html}"
        f"</table>"
    )

    return html


# ── 3-Zone Layout: Constants & Helpers ─────────────────────────────

HITTING_STAT_COLS = {"R", "HR", "RBI", "SB", "AVG", "OBP"}
PITCHING_STAT_COLS = {"W", "L", "SV", "K", "ERA", "WHIP"}
_RATE_STAT_COLS = {"AVG", "OBP", "ERA", "WHIP"}
_RATE_3DP = {"AVG", "OBP", "avg", "obp"}  # 3 decimal places
_RATE_2DP = {"ERA", "WHIP", "era", "whip"}  # 2 decimal places

_HEALTH_DOT_COLORS = {
    "Healthy": THEME["green"],
    "Day-to-Day": THEME["warn"],
    "IL": THEME["danger"],
    "IL-60": THEME["danger"],
    "Out": THEME["danger"],
    "Low Risk": THEME["green"],
    "Moderate Risk": THEME["warn"],
    "Elevated Risk": THEME["hot"],
    "High Risk": THEME["danger"],
}


_MLB_HEADSHOT_URL = (
    "https://img.mlbstatic.com/mlb-photos/image/upload/"
    "d_people:generic:headshot:67:current.png/"
    "w_213,q_auto:best/v1/people/{mlb_id}/headshot/67/current"
)

# Columns auto-hidden from table display (used for headshot rendering)
_HIDDEN_META_COLS = {"mlb_id", "player_id"}


# Generic person silhouette SVG as data URI (used when no MLB headshot available)
_AVATAR_FALLBACK_SVG = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' "
    "fill='%23999'%3E%3Ccircle cx='12' cy='8' r='4' fill='%23bbb'/%3E"
    "%3Cpath d='M12 14c-5 0-8 2.5-8 5v1h16v-1c0-2.5-3-5-8-5z' fill='%23bbb'/%3E%3C/svg%3E"
)


def _headshot_img_html(mlb_id, size: int = 22) -> str:
    """Return a tiny circular headshot <img> tag.

    Always returns a visible circle:
    - If mlb_id is valid, shows the MLB headshot with fallback to generic avatar on error.
    - If mlb_id is missing/invalid, shows the generic avatar directly.
    """
    _style = (
        f"border-radius:50%;object-fit:cover;vertical-align:middle;"
        f"margin-right:5px;border:1px solid rgba(0,0,0,0.08);flex-shrink:0;"
        f"background:{THEME['card_h']};"
    )
    _onerror = f"this.onerror=null;this.src='{_AVATAR_FALLBACK_SVG}'"

    has_mlb_id = False
    if mlb_id is not None:
        try:
            mid = int(mlb_id)
            if mid != 0 and not _math.isnan(mid):
                has_mlb_id = True
        except (ValueError, TypeError):
            pass

    if has_mlb_id:
        url = _MLB_HEADSHOT_URL.format(mlb_id=mid)
    else:
        url = _AVATAR_FALLBACK_SVG

    return f'<img src="{url}" width="{size}" height="{size}" style="{_style}" onerror="{_onerror}" loading="lazy" />'


def build_compact_table_html(
    df,
    highlight_cols=None,
    row_classes=None,
    health_col=None,
    max_height=500,
    show_avatars=None,
):
    """Build ESPN-style compact HTML table string from a DataFrame.

    Pure function — no Streamlit dependency. Returns raw HTML string.

    If the DataFrame contains an ``mlb_id`` column, a circular headshot
    thumbnail is rendered next to the player name in the first column.
    The ``mlb_id`` and ``player_id`` columns are automatically hidden
    from display.

    Args:
        df: pandas DataFrame to render.
        highlight_cols: Optional dict mapping column names to CSS class
            (e.g. ``{"R": "th-hit", "ERA": "th-pit"}``).  When *None*,
            hitting/pitching classes are auto-assigned from
            ``HITTING_STAT_COLS`` / ``PITCHING_STAT_COLS``.
        row_classes: Optional dict mapping row index → CSS class string
            (e.g. ``{0: "row-start", 5: "row-bench"}``).
        health_col: Optional column name containing health status strings.
            When provided, a colored dot is prepended to the cell value.
        max_height: Max container height in pixels (0 for unlimited).

    Returns:
        HTML string with ``compact-table-wrap`` / ``compact-table`` classes.
    """
    if df is None or df.empty:
        return '<div class="compact-table-wrap"><p style="padding:16px;color:#6b7280;font-size:13px;">No data available.</p></div>'

    # Detect headshot column — extract mlb_id values before hiding
    has_mlb_col = "mlb_id" in df.columns
    mlb_ids = df["mlb_id"].tolist() if has_mlb_col else []
    # show_avatars: True = always show avatar circle (fallback SVG for missing mlb_id)
    # Default: auto-enable when mlb_id column is present
    if show_avatars is None:
        show_avatars = has_mlb_col

    # Determine visible columns (hide meta columns)
    visible_cols = [c for c in df.columns if c not in _HIDDEN_META_COLS]

    if highlight_cols is None:
        highlight_cols = {}
        for col in visible_cols:
            col_upper = str(col).upper()
            if col_upper in HITTING_STAT_COLS:
                highlight_cols[col] = "th-hit"
            elif col_upper in PITCHING_STAT_COLS:
                highlight_cols[col] = "th-pit"
    elif isinstance(highlight_cols, (list, set, tuple)):
        highlight_cols = {col: "th-hit" for col in highlight_cols}

    if row_classes is None:
        row_classes = {}

    # Determine the first visible column as the "name" column for sticky behavior
    first_col = visible_cols[0] if visible_cols else ""

    # Build header
    header_cells = []
    for col in visible_cols:
        cls_parts = []
        if col == first_col:
            cls_parts.append("col-name")
        if col in highlight_cols:
            cls_parts.append(highlight_cols[col])
        cls_attr = f' class="{" ".join(cls_parts)}"' if cls_parts else ""
        header_cells.append(f"<th{cls_attr}>{col}</th>")
    header_html = "<tr>" + "".join(header_cells) + "</tr>"

    # Build body rows
    body_rows = []
    for idx, (row_idx, row) in enumerate(df.iterrows()):
        row_cls = row_classes.get(idx, row_classes.get(row_idx, ""))
        tr_cls = f' class="{row_cls}"' if row_cls else ""
        cells = []
        for col in visible_cols:
            val = row[col]
            cls_parts = []
            is_name_col = col == first_col
            if is_name_col:
                cls_parts.append("col-name")

            # Health dot injection
            cell_html = ""
            if health_col and col == health_col and val:
                dot_color = _HEALTH_DOT_COLORS.get(str(val), THEME["tx2"])
                cell_html = f'<span class="health-dot" style="background:{dot_color};"></span>'

            # Headshot / avatar injection for name column
            if is_name_col and show_avatars:
                mid = mlb_ids[idx] if idx < len(mlb_ids) else None
                cell_html += _headshot_img_html(mid)

            # Format numeric values — skip first col and health col
            _skip_cols = {first_col}
            if health_col:
                _skip_cols.add(health_col)
            if col not in _skip_cols:
                try:
                    fv = float(val)
                    if _math.isnan(fv) or _math.isinf(fv):
                        cell_html += ""
                    elif str(col) in _RATE_3DP or str(col).upper() in {"AVG", "OBP"}:
                        cell_html += f"{fv:.3f}"
                    elif str(col) in _RATE_2DP or str(col).upper() in {"ERA", "WHIP"}:
                        cell_html += f"{fv:.2f}"
                    else:
                        cell_html += f"{fv:.2f}"
                except (ValueError, TypeError):
                    cell_html += str(val) if val is not None else ""
            else:
                cell_html += str(val) if val is not None else ""

            cls_attr = f' class="{" ".join(cls_parts)}"' if cls_parts else ""
            cells.append(f"<td{cls_attr}>{cell_html}</td>")
        body_rows.append(f"<tr{tr_cls}>{''.join(cells)}</tr>")

    body_html = "".join(body_rows)

    height_style = f"max-height:{max_height}px;" if max_height else ""
    return (
        f'<div class="compact-table-wrap" style="{height_style}">'
        f'<table class="compact-table">'
        f"<thead>{header_html}</thead>"
        f"<tbody>{body_html}</tbody>"
        f"</table></div>"
    )


# ── Roster Display Sorting ───────────────────────────────────────
# Yahoo Fantasy slot order for consistent roster display across all pages.

SLOT_ORDER_HITTERS = ["C", "1B", "2B", "3B", "SS", "OF", "LF", "CF", "RF", "Util", "DH"]
SLOT_ORDER_PITCHERS = ["SP", "RP", "P"]
SLOT_ORDER_BENCH = ["BN"]
SLOT_ORDER_IL = ["IL", "IL+", "NA"]

_SLOT_ORDER_MAP = {}
for _i, _slot in enumerate(SLOT_ORDER_HITTERS):
    _SLOT_ORDER_MAP[_slot] = _i
for _i, _slot in enumerate(SLOT_ORDER_PITCHERS, start=len(SLOT_ORDER_HITTERS)):
    _SLOT_ORDER_MAP[_slot] = _i
for _i, _slot in enumerate(SLOT_ORDER_BENCH, start=len(SLOT_ORDER_HITTERS) + len(SLOT_ORDER_PITCHERS)):
    _SLOT_ORDER_MAP[_slot] = _i
for _i, _slot in enumerate(
    SLOT_ORDER_IL, start=len(SLOT_ORDER_HITTERS) + len(SLOT_ORDER_PITCHERS) + len(SLOT_ORDER_BENCH)
):
    _SLOT_ORDER_MAP[_slot] = _i

# Also map common aliases (case-insensitive handled in function)
_SLOT_ORDER_MAP["BENCH"] = _SLOT_ORDER_MAP["BN"]


def sort_roster_for_display(roster_df: pd.DataFrame) -> pd.DataFrame:
    """Sort roster into Yahoo Fantasy slot order for display.

    Uses ``roster_slot`` or ``selected_position`` column from league_rosters
    (Yahoo's actual slot assignment).  Falls back to inferring from
    ``positions`` column if neither slot column is available.

    Order: Hitter starters (C,1B,2B,3B,SS,OF,Util) -> Pitcher starters
    (SP,RP,P) -> BN -> IL/IL+/NA.

    Returns a sorted **copy** of the DataFrame.  Never modifies in place.
    """
    import pandas as _pd

    if roster_df is None or roster_df.empty:
        return roster_df.copy() if roster_df is not None else _pd.DataFrame()

    df = roster_df.copy()

    # Determine which column carries the slot assignment
    slot_col = None
    for candidate in ("roster_slot", "selected_position", "Slot"):
        if candidate in df.columns:
            non_empty = df[candidate].dropna().astype(str).str.strip().replace("", _pd.NA).dropna()
            if len(non_empty) > 0:
                slot_col = candidate
                break

    if slot_col is None and "positions" not in df.columns:
        # Nothing to sort on — return as-is
        return df

    _unknown_sort = len(_SLOT_ORDER_MAP) + 10  # large value for unknown slots

    def _get_sort_key(row):
        """Return integer sort key for a single roster row."""
        slot_value = ""
        if slot_col is not None:
            raw = row.get(slot_col)
            if raw is not None:
                slot_value = str(raw).strip()

        if slot_value:
            key = _SLOT_ORDER_MAP.get(slot_value, _SLOT_ORDER_MAP.get(slot_value.upper(), -1))
            if key >= 0:
                return key

        # Fallback: infer from positions column
        positions_raw = row.get("positions", "") or row.get("Pos", "") or row.get("Position", "") or ""
        if not positions_raw:
            return _unknown_sort

        positions_str = str(positions_raw).strip()
        # Positions may be comma-separated or slash-separated
        parts = [p.strip() for p in positions_str.replace("/", ",").split(",") if p.strip()]

        # Find the first matching slot in our defined order
        all_ordered = SLOT_ORDER_HITTERS + SLOT_ORDER_PITCHERS
        for ordered_slot in all_ordered:
            for part in parts:
                if part.upper() == ordered_slot.upper():
                    return _SLOT_ORDER_MAP.get(ordered_slot, _unknown_sort)

        return _unknown_sort

    sort_keys = df.apply(_get_sort_key, axis=1)

    # Build a secondary sort key: player name (alphabetical within same slot)
    name_col = None
    for nc in ("player_name", "name", "Player"):
        if nc in df.columns:
            name_col = nc
            break

    df["_slot_sort_key"] = sort_keys
    sort_cols = ["_slot_sort_key"]
    if name_col:
        sort_cols.append(name_col)

    df = df.sort_values(by=sort_cols, ascending=True).reset_index(drop=True)
    df = df.drop(columns=["_slot_sort_key"])

    return df


def render_compact_table(df, highlight_cols=None, row_classes=None, health_col=None, max_height=500, show_avatars=None):
    """Render compact ESPN-style table via st.markdown().

    Thin wrapper around :func:`build_compact_table_html`.
    """
    html = build_compact_table_html(
        df,
        highlight_cols=highlight_cols,
        row_classes=row_classes,
        health_col=health_col,
        max_height=max_height,
        show_avatars=show_avatars,
    )
    st.markdown(html, unsafe_allow_html=True)


# ── Sortable Table (st.dataframe wrapper) ─────────────────────────

# Stat columns that get auto-formatted as numbers
_SORTABLE_STAT_COLS = (
    HITTING_STAT_COLS
    | PITCHING_STAT_COLS
    | {
        "PICK_SCORE",
        "SGP",
        "MARGINAL_VALUE",
        "COMPOSITE_SCORE",
        "TRADE_VALUE",
        "DOLLAR_VALUE",
        "VORP",
        "SURPLUS",
        "NET_SGP",
        "IMPACT",
        "ACCEPTANCE",
        "ADP",
        "ECR",
        "RANK",
        "IP",
        "PA",
        "AB",
        "H",
        "BB",
        "HBP",
        "SF",
        "ER",
        "BB_ALLOWED",
        "H_ALLOWED",
        "HEALTH_SCORE",
        "SCORE",
    }
)


_HIDDEN_META_COLS_UPPER = {c.upper() for c in _HIDDEN_META_COLS}


def _build_default_column_config(df):
    """Auto-detect stat columns and apply consistent 2-decimal formatting."""
    config = {}
    for col in df.columns:
        col_upper = str(col).upper().replace(" ", "_")
        if col_upper in _HIDDEN_META_COLS_UPPER:
            config[col] = st.column_config.NumberColumn(col, format="%d")
        elif col_upper in _SORTABLE_STAT_COLS:
            config[col] = st.column_config.NumberColumn(col, format="%.2f")
    return config


def render_sortable_table(
    df,
    column_config=None,
    height=400,
    hide_index=True,
    key=None,
    hide_columns=None,
):
    """Render a sortable/filterable table using st.dataframe().

    Use for data-heavy tables (player pools, rankings, rosters).
    Keep :func:`render_compact_table` for small display cards and banners.

    Parameters
    ----------
    df : DataFrame
        Data to display.
    column_config : dict | None
        Custom Streamlit column_config. Auto-generated if *None*.
    height : int
        Table height in pixels.
    hide_index : bool
        Whether to hide the DataFrame index.
    key : str | None
        Streamlit widget key for deduplication.
    hide_columns : list[str] | None
        Columns to exclude from display.
    """
    if df is None or df.empty:
        st.info("No data to display.")
        return

    display = df.copy()

    # Hide internal columns
    drop_cols = []
    for c in list(display.columns):
        if str(c).upper() in _HIDDEN_META_COLS:
            drop_cols.append(c)
    if hide_columns:
        drop_cols.extend([c for c in hide_columns if c in display.columns])
    if drop_cols:
        display = display.drop(columns=drop_cols, errors="ignore")

    if column_config is None:
        column_config = _build_default_column_config(display)

    st.dataframe(
        display,
        column_config=column_config,
        hide_index=hide_index,
        height=height,
        width="stretch",
        key=key,
    )


def render_reco_banner(teaser_text, expanded_html="", icon_key="zap"):
    """Render a collapsible recommendation banner at the top of a page.

    When *expanded_html* is empty the banner is a simple static line.
    When provided, the banner uses ``st.expander`` styled via CSS.
    """
    icon_svg = PAGE_ICONS.get(icon_key, "")
    safe_teaser = _html.escape(teaser_text)
    teaser_html = f'<div class="reco-banner"><span class="reco-banner-teaser">{icon_svg} {safe_teaser}</span></div>'

    if not expanded_html:
        st.markdown(teaser_html, unsafe_allow_html=True)
    else:
        st.markdown(teaser_html, unsafe_allow_html=True)
        with st.expander("Details", expanded=False):
            st.markdown(
                f'<div class="reco-banner-detail">{expanded_html}</div>',
                unsafe_allow_html=True,
            )


def render_context_card(title, content_html):
    """Render a single glassmorphic context card for the sidebar panel."""
    safe_title = _html.escape(title)
    st.markdown(
        f'<div class="context-card"><div class="context-card-title">{safe_title}</div>{content_html}</div>',
        unsafe_allow_html=True,
    )


_MATCHUP_TICKER_TTL_SECONDS = 300  # 5-minute cache


def _fetch_matchup_data() -> dict | None:
    """Fetch matchup data with session-state caching (5-min TTL)."""
    import time as _time

    cached = st.session_state.get("_matchup_ticker_data")
    cached_at = st.session_state.get("_matchup_ticker_ts", 0)
    if cached is not None and (_time.monotonic() - cached_at) < _MATCHUP_TICKER_TTL_SECONDS:
        return cached

    yahoo_client = st.session_state.get("yahoo_client")
    if yahoo_client is None:
        return None

    try:
        data = yahoo_client.get_current_matchup()
        st.session_state["_matchup_ticker_data"] = data
        st.session_state["_matchup_ticker_ts"] = _time.monotonic()
        return data
    except Exception:
        return cached  # return stale data on error


def render_matchup_ticker():
    """Render a compact matchup scoreboard bar.

    Only renders when Yahoo is connected and matchup data is available.
    Caches data in session_state for 5 minutes to avoid API spam.
    """
    if not st.session_state.get("yahoo_connected"):
        return

    data = _fetch_matchup_data()
    if data is None:
        return

    t = THEME
    week = data["week"]
    status = data["status"]
    opp_name = _html.escape(data["opp_name"])
    w, lo, ti = data["wins"], data["losses"], data["ties"]

    # Score color
    if w > lo:
        score_color = t["green"]
        status_label = "Winning"
        status_bg = f"{t['green']}18"
    elif lo > w:
        score_color = t["primary"]
        status_label = "Losing"
        status_bg = f"{t['primary']}18"
    else:
        score_color = t["hot"]
        status_label = "Tied"
        status_bg = f"{t['hot']}18"

    if status == "postevent":
        status_label = "Won" if w > lo else ("Lost" if lo > w else "Tied")
    elif status == "preevent":
        status_label = "Pre-game"
        score_color = t["tx2"]
        status_bg = f"{t['tx2']}18"

    score_str = f"{w}-{lo}-{ti}"

    # Build category detail rows for the expander
    cat_rows = ""
    for c in data.get("categories", []):
        cat = c["cat"]
        you_val = c["you"]
        opp_val = c["opp"]
        res = c["result"]
        css_cls = {"WIN": "cat-win", "LOSS": "cat-loss", "TIE": "cat-tie"}.get(res, "")
        marker = {"WIN": "+", "LOSS": "-", "TIE": "="}.get(res, "")
        cat_rows += (
            f'<div class="cat-row {css_cls}">'
            f'<span style="width:50px;display:inline-block;">{cat}</span>'
            f'<span style="width:60px;text-align:right;display:inline-block;">{you_val}</span>'
            f'<span style="width:20px;text-align:center;display:inline-block;opacity:0.4;">v</span>'
            f'<span style="width:60px;display:inline-block;">{opp_val}</span>'
            f'<span style="width:20px;text-align:right;display:inline-block;">{marker}</span>'
            f"</div>"
        )

    # Render the ticker bar
    st.markdown(
        f'<div class="matchup-ticker">'
        f'<span class="matchup-ticker-week">Week {week}</span>'
        f'<span class="matchup-ticker-score" style="color:{score_color};">{score_str}</span>'
        f'<span class="matchup-ticker-vs">vs {opp_name}</span>'
        f'<span class="matchup-ticker-status" style="background:{status_bg};color:{score_color};">'
        f"{status_label}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Expandable category breakdown
    if any(c["result"] != "-" for c in data.get("categories", [])):
        with st.expander("Category Breakdown", expanded=False):
            st.markdown(
                f'<div class="matchup-ticker" style="flex-direction:column;align-items:stretch;gap:0;">'
                f"{cat_rows}</div>",
                unsafe_allow_html=True,
            )


def render_page_layout(title, banner_teaser="", banner_detail="", banner_icon="zap"):
    """Render page title badge + recommendation banner + matchup ticker.

    Call at the top of every page, after ``inject_custom_css()``.
    """
    icon_svg = PAGE_ICONS.get(
        title.lower().replace(" ", "_"),
        PAGE_ICONS.get(title.lower(), ""),
    )
    safe_title = _html.escape(title)
    st.markdown(
        f'<div class="page-title">{icon_svg} {safe_title}</div>',
        unsafe_allow_html=True,
    )
    if banner_teaser:
        render_reco_banner(banner_teaser, banner_detail, banner_icon)
    render_matchup_ticker()


def render_context_columns(context_width=1):
    """Create a 2-column layout: narrow context panel + wide main area.

    Returns:
        Tuple of (context_col, main_col) Streamlit column objects.
    """
    return st.columns([context_width, 4])


# ── Player Card Dialog ────────────────────────────────────────────


def _render_player_card_header(profile: dict) -> None:
    """Render the player card header band: headshot + bio + tags."""
    t = THEME
    name = _html.escape(profile.get("name", ""))
    team = _html.escape(profile.get("team", ""))
    positions = _html.escape(profile.get("positions", ""))
    bats = profile.get("bats", "")
    throws = profile.get("throws", "")
    age = profile.get("age")
    age_str = f"Age {age}" if age else ""
    health_label = profile.get("health_label", "")
    health_score = profile.get("health_score", 0)
    headshot_url = profile.get("headshot_url", "")
    tags = profile.get("tags", [])

    # Health dot color
    if health_score >= 0.9:
        dot_color = t["green"]
    elif health_score >= 0.75:
        dot_color = t["warn"]
    else:
        dot_color = t["danger"]

    # Headshot or generic avatar placeholder
    _card_onerror = f"this.onerror=null;this.src='{_AVATAR_FALLBACK_SVG}'"
    src = headshot_url if headshot_url else _AVATAR_FALLBACK_SVG
    img_html = (
        f'<img src="{src}" '
        f'style="width:80px;height:80px;border-radius:50%;object-fit:cover;'
        f"border:3px solid {t['hot']};box-shadow:0 2px 8px rgba(0,0,0,0.15);"
        f'background:{t["bg"]};" '
        f'onerror="{_card_onerror}" />'
    )

    # Tag badges
    tag_colors = {
        "Sleeper": t["purple"],
        "Breakout": t["green"],
        "Target": t["sky"],
        "Bust": t["danger"],
        "Avoid": t["danger"],
    }
    tags_html = " ".join(
        f'<span style="display:inline-block;padding:2px 8px;border-radius:10px;'
        f"background:{tag_colors.get(tag, t['tx2'])};color:#fff;font-size:10px;"
        f'font-weight:700;letter-spacing:0.5px;text-transform:uppercase;">'
        f"{_html.escape(tag)}</span>"
        for tag in tags
    )

    # Bio line
    bio_parts = [p for p in [positions, team, age_str, f"B/T: {bats}/{throws}" if bats else ""] if p]
    bio_line = " | ".join(bio_parts)

    st.markdown(
        f'<div style="display:flex;align-items:center;gap:16px;padding:12px 16px;'
        f"background:rgba(255,255,255,0.6);backdrop-filter:blur(12px);"
        f"border:1px solid rgba(255,255,255,0.3);border-left:4px solid {t['hot']};"
        f'border-radius:12px;margin-bottom:12px;">'
        f"{img_html}"
        f'<div style="flex:1;">'
        f'<div style="font-family:Bebas Neue,sans-serif;font-size:24px;letter-spacing:2px;'
        f'color:{t["tx"]};line-height:1.2;">{name}</div>'
        f'<div style="font-family:Figtree,sans-serif;font-size:13px;color:{t["tx2"]};'
        f'margin-top:2px;">{bio_line}</div>'
        f'<div style="display:flex;align-items:center;gap:8px;margin-top:6px;">'
        f'<span class="health-dot" style="background:{dot_color};width:8px;height:8px;"></span>'
        f'<span style="font-size:12px;color:{t["tx2"]};">{_html.escape(health_label)}</span>'
        f"{tags_html}"
        f"</div></div></div>",
        unsafe_allow_html=True,
    )


def _render_radar_chart(radar: dict, is_hitter: bool) -> None:
    """Render a Plotly radar chart comparing player vs league/MLB averages."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.info("Plotly is required for radar charts.")
        return

    from src.valuation import LeagueConfig

    lc = LeagueConfig()
    cats = lc.hitting_categories if is_hitter else lc.pitching_categories

    player_vals = [radar.get("player", {}).get(c, 50) for c in cats]
    # Close the polygon
    cats_closed = cats + [cats[0]]
    player_closed = player_vals + [player_vals[0]]

    t = THEME
    fig = go.Figure()

    # Player trace
    fig.add_trace(
        go.Scatterpolar(
            r=player_closed,
            theta=cats_closed,
            name="Player",
            fill="toself",
            fillcolor="rgba(255,109,0,0.15)",
            line=dict(color=t["hot"], width=2),
        )
    )

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, gridcolor="rgba(0,0,0,0.08)"),
            angularaxis=dict(gridcolor="rgba(0,0,0,0.08)"),
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5, font=dict(size=11)),
        margin=dict(l=40, r=40, t=20, b=40),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Figtree, sans-serif", size=12),
    )

    st.plotly_chart(fig, width="stretch")


def _render_news_section(news: list[dict]) -> None:
    """Render deduplicated news items with full datetime."""
    t = THEME
    if not news:
        st.markdown(
            f'<div style="font-size:13px;color:{t["tx2"]};padding:8px 0;">No recent news for this player.</div>',
            unsafe_allow_html=True,
        )
        return

    for item in news:
        headline = _html.escape(item.get("headline", ""))
        source = _html.escape(item.get("source", ""))
        date_display = _html.escape(item.get("date_display", ""))
        sentiment = item.get("sentiment", 0.0) or 0.0
        news_type = item.get("news_type", "general")

        # Source badge color
        source_colors = {"espn": "#c41230", "rotowire": "#1a73e8", "mlb": "#002d72", "yahoo": "#6001d2"}
        src_bg = source_colors.get(source.lower(), t["tx2"])

        # Sentiment dot
        if sentiment >= 0.2:
            sent_color = t["green"]
        elif sentiment >= -0.2:
            sent_color = t["warn"]
        else:
            sent_color = t["danger"]

        # Type label
        type_colors = {"injury": t["danger"], "transaction": t["purple"], "callup": t["green"], "lineup": t["sky"]}
        type_color = type_colors.get(news_type, t["tx2"])

        st.markdown(
            f'<div style="padding:8px 12px;border-bottom:1px solid {t["border"]};">'
            f'<div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">'
            f'<span style="display:inline-block;padding:1px 6px;border-radius:8px;'
            f'background:{src_bg};color:#fff;font-size:10px;font-weight:700;">{source}</span>'
            f'<span style="font-size:10px;font-weight:600;color:{type_color};'
            f'text-transform:uppercase;letter-spacing:0.5px;">{news_type}</span>'
            f'<span style="font-size:10px;color:{t["tx2"]};font-family:IBM Plex Mono,monospace;">'
            f"{date_display}</span>"
            f'<span class="health-dot" style="background:{sent_color};width:6px;height:6px;"></span>'
            f"</div>"
            f'<div style="font-size:13px;font-weight:600;color:{t["tx"]};margin-top:4px;">'
            f"{headline}</div></div>",
            unsafe_allow_html=True,
        )


@st.dialog("Player Card", width="large")
def show_player_card_dialog(player_id: int):
    """Render the full player card as a modal dialog."""
    from src.player_card import build_player_card_data

    data = build_player_card_data(player_id)
    profile = data["profile"]
    is_hitter = bool(
        profile.get("positions", "") and not all(p in ("SP", "RP", "P") for p in profile["positions"].split("/"))
    )

    # 1. Header
    _render_player_card_header(profile)

    # 2. Radar chart
    radar = data.get("radar", {})
    if radar.get("player"):
        st.markdown('<div class="sec-head">Percentile Rankings</div>', unsafe_allow_html=True)
        _render_radar_chart(radar, is_hitter)

    # 3. Historical stats
    historical = data.get("historical", [])
    if historical:
        st.markdown('<div class="sec-head">Historical Stats (3 Years)</div>', unsafe_allow_html=True)
        cats = ["R", "HR", "RBI", "SB", "AVG", "OBP"] if is_hitter else ["W", "L", "SV", "K", "ERA", "WHIP"]
        rows = []
        for h in historical:
            row = {"Season": str(int(h["season"])) if h.get("season") else ""}
            row["GP"] = h.get("GAMES_PLAYED")
            if is_hitter:
                row["PA"] = h.get("PA")
            else:
                row["IP"] = h.get("IP")
            for c in cats:
                row[c] = h.get(c)
            rows.append(row)
        import pandas as pd

        render_compact_table(pd.DataFrame(rows), max_height=200)

    # 4. 2026 Projections
    projections = data.get("projections", {})
    blended = projections.get("blended", {})
    systems = projections.get("systems", {})
    if blended or systems:
        st.markdown('<div class="sec-head">2026 Projections</div>', unsafe_allow_html=True)
        cats = ["R", "HR", "RBI", "SB", "AVG", "OBP"] if is_hitter else ["W", "L", "SV", "K", "ERA", "WHIP"]
        rows = []
        if blended:
            row = {"Source": "Blended"}
            for c in cats:
                row[c] = blended.get(c)
            rows.append(row)
        for sys_name, sys_stats in sorted(systems.items()):
            row = {"Source": sys_name.title()}
            for c in cats:
                row[c] = sys_stats.get(c)
            rows.append(row)
        import pandas as pd

        # Bold the blended row
        row_cls = {0: "row-start"} if blended else {}
        render_compact_table(pd.DataFrame(rows), row_classes=row_cls, max_height=250)

    # 5. Advanced metrics (pitchers only)
    advanced = data.get("advanced", {})
    if advanced and any(v is not None for v in advanced.values()):
        st.markdown('<div class="sec-head">Advanced Metrics</div>', unsafe_allow_html=True)
        t = THEME
        metrics_html = '<div style="display:flex;flex-wrap:wrap;gap:12px;">'
        metric_labels = {
            "FIP": "FIP",
            "XFIP": "xFIP",
            "SIERA": "SIERA",
            "STUFF_PLUS": "Stuff+",
            "LOCATION_PLUS": "Location+",
            "PITCHING_PLUS": "Pitching+",
        }
        for key, label in metric_labels.items():
            val = advanced.get(key)
            if val is not None:
                try:
                    display = f"{float(val):.2f}" if "PLUS" not in key else f"{int(float(val))}"
                except (ValueError, TypeError):
                    display = str(val)
                metrics_html += (
                    f'<div style="background:{t["card"]};border:1px solid {t["border"]};'
                    f'border-radius:8px;padding:8px 14px;text-align:center;">'
                    f'<div style="font-size:10px;color:{t["tx2"]};text-transform:uppercase;'
                    f'letter-spacing:0.5px;">{label}</div>'
                    f'<div style="font-size:18px;font-weight:700;color:{t["tx"]};'
                    f'font-family:IBM Plex Mono,monospace;">{display}</div></div>'
                )
        metrics_html += "</div>"
        st.markdown(metrics_html, unsafe_allow_html=True)

    # 6. Rankings & ADP
    rankings = data.get("rankings", {})
    if rankings:
        st.markdown('<div class="sec-head">Rankings and Average Draft Position</div>', unsafe_allow_html=True)
        t = THEME
        rank_parts = []
        if rankings.get("consensus_rank"):
            rank_parts.append(f'<span style="font-weight:700;">Consensus:</span> #{rankings["consensus_rank"]}')
        if rankings.get("composite_adp"):
            try:
                rank_parts.append(
                    f'<span style="font-weight:700;">Composite ADP:</span> {float(rankings["composite_adp"]):.1f}'
                )
            except (ValueError, TypeError):
                pass
        if rankings.get("yahoo_adp"):
            rank_parts.append(f'<span style="font-weight:700;">Yahoo ADP:</span> {rankings["yahoo_adp"]}')
        if rankings.get("fantasypros_adp"):
            rank_parts.append(f'<span style="font-weight:700;">FantasyPros ADP:</span> {rankings["fantasypros_adp"]}')
        if rankings.get("nfbc_adp"):
            rank_parts.append(f'<span style="font-weight:700;">NFBC ADP:</span> {rankings["nfbc_adp"]}')

        if rank_parts:
            st.markdown(
                f'<div style="font-size:13px;color:{t["tx"]};line-height:1.8;'
                f'font-family:Figtree,sans-serif;">' + "<br>".join(rank_parts) + "</div>",
                unsafe_allow_html=True,
            )

    # 7. Injury history
    injury = data.get("injury_history", [])
    if injury:
        st.markdown('<div class="sec-head">Injury History</div>', unsafe_allow_html=True)
        import pandas as pd

        inj_rows = []
        inj_cls = {}
        for i, ih in enumerate(injury):
            inj_rows.append(
                {
                    "Season": str(ih["season"]),
                    "Games Played": ih["GP"],
                    "Games Available": ih["GA"],
                    "IL Stints": ih["IL_stints"],
                    "IL Days": ih["IL_days"],
                }
            )
            inj_cls[i] = "row-start" if ih["IL_stints"] == 0 else "row-bench"
        render_compact_table(pd.DataFrame(inj_rows), row_classes=inj_cls, max_height=180)

    # 8. News
    news = data.get("news", [])
    st.markdown('<div class="sec-head">Recent News</div>', unsafe_allow_html=True)
    _render_news_section(news)

    # 9. Prospect scouting (conditional)
    prospect = data.get("prospect")
    if prospect:
        st.markdown('<div class="sec-head">Prospect Scouting Report</div>', unsafe_allow_html=True)
        t = THEME

        # Scouting grades
        grade_pairs = [
            ("Hit", prospect.get("hit", {})),
            ("Game Power", prospect.get("game_power", {})),
            ("Raw Power", prospect.get("raw_power", {})),
            ("Speed", {"present": prospect.get("speed"), "future": prospect.get("speed")}),
            ("Field", {"present": prospect.get("field"), "future": prospect.get("field")}),
        ]
        if prospect.get("control", {}).get("present"):
            grade_pairs.append(("Control", prospect.get("control", {})))

        grades_html = '<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:8px;">'
        for label, grades in grade_pairs:
            present = grades.get("present", "")
            future = grades.get("future", "")
            if present or future:
                grades_html += (
                    f'<div style="background:{t["card"]};border:1px solid {t["border"]};'
                    f'border-radius:8px;padding:6px 10px;text-align:center;min-width:70px;">'
                    f'<div style="font-size:9px;color:{t["tx2"]};text-transform:uppercase;">{label}</div>'
                    f'<div style="font-size:14px;font-weight:700;color:{t["tx"]};">'
                    f"{present or '-'}/{future or '-'}</div></div>"
                )
        grades_html += "</div>"
        st.markdown(grades_html, unsafe_allow_html=True)

        # Meta info
        meta_parts = []
        if prospect.get("fg_rank"):
            meta_parts.append(f"FanGraphs Rank: #{_html.escape(str(prospect['fg_rank']))}")
        if prospect.get("fg_fv"):
            meta_parts.append(f"Future Value: {_html.escape(str(prospect['fg_fv']))}")
        if prospect.get("fg_eta"):
            meta_parts.append(f"ETA: {_html.escape(str(prospect['fg_eta']))}")
        if prospect.get("fg_risk"):
            meta_parts.append(f"Risk: {_html.escape(str(prospect['fg_risk']))}")
        if prospect.get("milb_level"):
            meta_parts.append(f"Level: {_html.escape(str(prospect['milb_level']))}")
        if meta_parts:
            st.markdown(
                f'<div style="font-size:12px;color:{t["tx2"]};margin-bottom:6px;">' + " | ".join(meta_parts) + "</div>",
                unsafe_allow_html=True,
            )

        # Scouting report
        tldr = prospect.get("tldr", "")
        if tldr:
            st.markdown(
                f'<div style="font-size:13px;color:{t["tx"]};font-style:italic;'
                f'line-height:1.5;">{_html.escape(str(tldr))}</div>',
                unsafe_allow_html=True,
            )


def render_player_select(player_names, player_ids, key_suffix="default"):
    """Render a selectbox that opens a player card dialog when a player is chosen."""
    if not player_ids:
        return

    selected = st.selectbox(
        "View player card",
        options=[""] + list(player_names),
        key=f"player_card_select_{key_suffix}",
        format_func=lambda x: "Select a player..." if x == "" else x,
    )
    if selected and selected != "":
        try:
            idx = list(player_names).index(selected)
            pid = player_ids[idx]
            show_player_card_dialog(int(pid))
        except (ValueError, IndexError):
            pass


# ── Page Load Timing ─────────────────────────────────────────────


def page_timer_start() -> None:
    """Record the page render start time in session state.

    Call at the very top of every page, before any heavy computation.
    """
    if st is None:
        return
    st.session_state["_page_timer_start"] = _time.perf_counter()


def page_timer_footer(page_name: str = "") -> None:
    """Render a subtle footer showing how long the page took to render.

    Call at the very bottom of every page. Stores the result in
    ``st.session_state["_page_load_times"]`` for cross-page comparison.
    """
    if st is None:
        return
    start = st.session_state.get("_page_timer_start")
    if start is None:
        return
    elapsed = _time.perf_counter() - start
    if "_page_load_times" not in st.session_state:
        st.session_state["_page_load_times"] = {}
    if page_name:
        st.session_state["_page_load_times"][page_name] = elapsed
    label = f"{page_name} loaded" if page_name else "Page loaded"
    st.markdown(
        f'<div style="text-align:right;padding:8px 12px 4px;'
        f'font-size:11px;color:{THEME["tx2"]};font-family:monospace;">'
        f"{label} in {elapsed:.2f}s</div>",
        unsafe_allow_html=True,
    )


# ── Session-Level Valuation Cache ────────────────────────────────


def get_session_config():
    """Return a cached LeagueConfig with computed SGP denominators.

    Computes once per session, reuses across all pages.
    """
    if st is None:
        from src.valuation import LeagueConfig

        return LeagueConfig()

    if "_session_config" not in st.session_state:
        from src.database import load_player_pool
        from src.valuation import LeagueConfig, compute_sgp_denominators

        pool = load_player_pool()
        lc = LeagueConfig()
        if not pool.empty:
            denoms = compute_sgp_denominators(pool, lc)
            lc.sgp_denominators.update(denoms)
        st.session_state["_session_config"] = lc

    return st.session_state["_session_config"]


def get_session_replacement_levels():
    """Return cached replacement levels computed once per session."""
    if st is None:
        return {}

    if "_session_replacement_levels" not in st.session_state:
        from src.database import load_player_pool
        from src.valuation import SGPCalculator, compute_replacement_levels

        pool = load_player_pool()
        lc = get_session_config()
        sgp = SGPCalculator(lc)
        if not pool.empty:
            st.session_state["_session_replacement_levels"] = compute_replacement_levels(pool, lc, sgp)
        else:
            st.session_state["_session_replacement_levels"] = {}

    return st.session_state["_session_replacement_levels"]


def invalidate_session_caches():
    """Clear all session-level valuation caches.

    Call after roster changes, data refresh, or config updates.
    """
    if st is None:
        return
    for key in ["_session_config", "_session_replacement_levels"]:
        st.session_state.pop(key, None)


# ── Data Freshness Widget ────────────────────────────────────────


def render_data_freshness_card():
    """Render a context card showing Yahoo data freshness per data type.

    Shows "Live", "Cached (Xm ago)", or "Offline" badges for each data type
    and provides a "Refresh All" button. Uses the YahooDataService singleton.

    Call inside a ``render_context_columns()`` context column.
    """
    if st is None:
        return

    try:
        from src.yahoo_data_service import get_yahoo_data_service

        yds = get_yahoo_data_service()
    except Exception:
        return

    t = THEME
    freshness = yds.get_data_freshness()
    connected = yds.is_connected()

    # Build status rows
    rows_html = ""
    status_icon = {
        "Live": f'<span style="color:{t["green"]}!important">LIVE</span>',
        "Cached": f'<span style="color:{t["amber"]}!important">CACHED</span>',
        "Stale": f'<span style="color:{t["hot"]}!important">STALE</span>',
        "Offline": f'<span style="color:{t["tx2"]}!important">OFFLINE</span>',
    }

    label_map = {
        "rosters": "Rosters",
        "standings": "Standings",
        "matchup": "Matchup",
        "free_agents": "Free Agents",
        "transactions": "Transactions",
        "settings": "Settings",
        "schedule": "Schedule",
    }

    for key, label in label_map.items():
        status_text = freshness.get(key, "Offline (DB)")
        # Determine badge type from status text
        if "Live" in status_text or "just now" in status_text:
            badge = status_icon["Live"]
        elif "Cached" in status_text:
            badge = status_icon["Cached"]
        elif "Stale" in status_text:
            badge = status_icon["Stale"]
        else:
            badge = status_icon["Offline"]

        rows_html += (
            f'<div style="display:flex;justify-content:space-between;'
            f'padding:2px 0;font-size:11px!important">'
            f"<span>{label}</span>{badge}</div>"
        )

    conn_status = (
        f'<div style="font-size:10px!important;color:{t["green"]}!important;'
        f'margin-bottom:4px!important">Yahoo Connected</div>'
        if connected
        else f'<div style="font-size:10px!important;color:{t["tx2"]}!important;'
        f'margin-bottom:4px!important">Yahoo Offline</div>'
    )

    render_context_card("Data Freshness", f"{conn_status}{rows_html}")

    if connected:
        if st.button("Refresh All Data", key="_yds_refresh_all", type="secondary"):
            with st.spinner("Refreshing..."):
                results = yds.force_refresh_all()
                refreshed = sum(1 for v in results.values() if v == "Refreshed")
                st.toast(f"Refreshed {refreshed}/{len(results)} data sources")
                st.rerun()

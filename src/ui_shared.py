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

from src.database import get_refresh_log_snapshot
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
        'fill="none" stroke="#ff6d00" stroke-width="1.6" stroke-linecap="round"/>'
        "<!-- classic V-stitching right arc -->"
        '<path d="M47 16 C52 22 54 27 53 32 C54 37 52 42 47 48" '
        'fill="none" stroke="#ff6d00" stroke-width="1.6" stroke-linecap="round"/>'
        "<!-- stitch tick marks left -->"
        '<line x1="27" y1="19" x2="30" y2="20" stroke="#ff6d00" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="25" y1="23" x2="28" y2="23.5" stroke="#ff6d00" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="24" y1="27.5" x2="27" y2="27.5" stroke="#ff6d00" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="24" y1="36.5" x2="27" y2="36.5" stroke="#ff6d00" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="25" y1="41" x2="28" y2="40.5" stroke="#ff6d00" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="27" y1="45" x2="30" y2="44" stroke="#ff6d00" stroke-width="1" stroke-linecap="round"/>'
        "<!-- stitch tick marks right -->"
        '<line x1="49" y1="19" x2="46" y2="20" stroke="#ff6d00" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="51" y1="23" x2="48" y2="23.5" stroke="#ff6d00" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="52" y1="27.5" x2="49" y2="27.5" stroke="#ff6d00" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="52" y1="36.5" x2="49" y2="36.5" stroke="#ff6d00" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="51" y1="41" x2="48" y2="40.5" stroke="#ff6d00" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="49" y1="45" x2="46" y2="44" stroke="#ff6d00" stroke-width="1" stroke-linecap="round"/>'
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
        'fill="none" stroke="#ff6d00" stroke-width="1.6" stroke-linecap="round"/>'
        '<path d="M47 16 C52 22 54 27 53 32 C54 37 52 42 47 48" '
        'fill="none" stroke="#ff6d00" stroke-width="1.6" stroke-linecap="round"/>'
        '<line x1="27" y1="19" x2="30" y2="20" stroke="#ff6d00" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="25" y1="23" x2="28" y2="23.5" stroke="#ff6d00" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="24" y1="27.5" x2="27" y2="27.5" stroke="#ff6d00" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="24" y1="36.5" x2="27" y2="36.5" stroke="#ff6d00" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="25" y1="41" x2="28" y2="40.5" stroke="#ff6d00" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="27" y1="45" x2="30" y2="44" stroke="#ff6d00" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="49" y1="19" x2="46" y2="20" stroke="#ff6d00" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="51" y1="23" x2="48" y2="23.5" stroke="#ff6d00" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="52" y1="27.5" x2="49" y2="27.5" stroke="#ff6d00" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="52" y1="36.5" x2="49" y2="36.5" stroke="#ff6d00" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="51" y1="41" x2="48" y2="40.5" stroke="#ff6d00" stroke-width="1" stroke-linecap="round"/>'
        '<line x1="49" y1="45" x2="46" y2="44" stroke="#ff6d00" stroke-width="1" stroke-linecap="round"/>'
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
        '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#e0492f" '
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
        '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#e0492f" '
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
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#e0492f" '
        'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:6px;">'
        '<circle cx="12" cy="12" r="10"/>'
        '<line x1="4.93" y1="4.93" x2="19.07" y2="19.07"/></svg>'
    ),
    "fire": (
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="#ff6d00" stroke="none" '
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
    "league_standings": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M7 17V9"/><path d="M11 17V5"/><path d="M15 17v-4"/><path d="M19 17v-8"/></svg>',
    "databank": '<svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="3" x2="9" y2="21"/><line x1="15" y1="3" x2="15" y2="21"/></svg>',
    # ── New page icons ──
    "weekly_dashboard": (
        '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:8px;">'
        '<rect x="3" y="3" width="8" height="9" rx="1"/><rect x="13" y="3" width="8" height="5" rx="1"/>'
        '<rect x="13" y="10" width="8" height="11" rx="1"/><rect x="3" y="14" width="8" height="7" rx="1"/></svg>'
    ),
    "trade_values": (
        '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:8px;">'
        '<line x1="12" y1="1" x2="12" y2="23"/>'
        '<path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>'
    ),
    "waiver_wire": (
        '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:8px;">'
        '<path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="8.5" cy="7" r="4"/>'
        '<line x1="20" y1="8" x2="20" y2="14"/><line x1="23" y1="11" x2="17" y2="11"/></svg>'
    ),
    "category_tracker": (
        '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:8px;">'
        '<path d="M9 11l3 3L22 4"/>'
        '<path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg>'
    ),
    "trends": (
        '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:8px;">'
        '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>'
    ),
    "playoff_odds": (
        '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:8px;">'
        '<path d="M21.21 15.89A10 10 0 1 1 8 2.83"/><path d="M22 12A10 10 0 0 0 12 2v10z"/></svg>'
    ),
    "bullpen": (
        '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:8px;">'
        '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>'
    ),
    "punt_analyzer": (
        '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:8px;">'
        '<circle cx="12" cy="12" r="10"/>'
        '<line x1="4.93" y1="4.93" x2="19.07" y2="19.07"/></svg>'
    ),
    "weekly_recap": (
        '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:8px;">'
        '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>'
        '<polyline points="14 2 14 8 20 8"/>'
        '<line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>'
    ),
}

# ── Theme (Light-only — Heater palette) ──────────────────────────

THEME = {
    # ── Combustion Index redesign (2026-06-08) ──
    # Off-white canvas, deep-navy chrome, hot-orange accent. No red in the brand
    # colorway (ember-red is reserved as a functional negative only). Values from
    # docs/design/mockup-myteam.html + mockup-player-popup.html :root + the
    # locked palette in docs/superpowers/plans/2026-06-08-combustion-redesign-plan.md.
    "bg": "#ffffff",  # pure white canvas (owner request 2026-06-10: off-white + blueprint grid removed)
    "card": "#f5f6f8",  # pale surface fill (owner request 2026-06-10: white elements re-filled)
    "card_h": "#ecebe7",
    "primary": "#ff6d00",  # hot orange — the headline brand change (was red)
    "primary_l": "#ff9a3c",
    "hot": "#ff6d00",
    "hot_l": "#ff9a3c",
    "gold": "#ffae42",
    "green": "#1f9d6b",
    "green_l": "#2bbd86",
    "sky": "#5f7d9c",  # muted steel (COLD/negative-leaning, minimal blue)
    "sky_l": "#8aa6c0",
    "purple": "#6c63ff",  # rarely used; kept
    "ok": "#1f9d6b",
    "danger": "#e0492f",  # ember-red — functional negative ONLY, not brand
    "warn": "#ff9f1c",
    "tx": "#1b1c20",  # charcoal primary body text
    "tx2": "#646a78",
    "border": "#e3e2de",
    # ink = text-on-accent (white on colored buttons). MUST stay white — the
    # mockup's dark "ink" maps to THEME["tx"], not this key.
    "ink": "#ffffff",
    # ── Neutrals + chrome (Combustion redesign) ──
    # "surface" is the card fill (== card); the sidebar pair drives the deep-navy
    # rail; divider/tx_muted/tx_subtle give the calmer text + line hierarchy.
    "surface": "#f5f6f8",
    "sidebar_bg": "#112744",  # deep navy chrome
    "sidebar_ink": "#eef1f6",
    "divider": "#edece8",
    "tx_muted": "#646a78",
    "tx_subtle": "#9aa0ac",
    "tiers": [
        "#ff6d00",
        "#ff9a3c",
        "#ffae42",
        "#1f9d6b",
        "#5f7d9c",
        "#8aa6c0",
        "#b0b5be",
        "#cdd1d8",
    ],
    # Backward compatibility aliases — old code uses T["amber"], T["teal"], etc.
    # Formerly aliased to red; now orange-family per the redesign.
    "amber": "#ff6d00",
    "amber_l": "#ff9a3c",
    "teal": "#5f7d9c",
    # ── New Combustion tokens (2026-06-08) ──
    "flame": "#ff9a3c",  # lighter orange (highlights, hover)
    "ember": "#e8480a",  # deep orange (gradient base, accent rules)
    "cold": "#5f7d9c",  # steel (COLD-only signal)
    "navy": "#112744",  # chrome base
    "navy2": "#0e2244",  # chrome gradient end
}

# Simple alias — no proxy needed without dark mode.
# All existing T["key"] call sites continue to work.
T = THEME

_LC_ONCE = _LC_Class()
ROSTER_CONFIG = _LC_ONCE.roster_slots
HITTING_CATEGORIES = _LC_ONCE.hitting_categories
PITCHING_CATEGORIES = _LC_ONCE.pitching_categories
ALL_CATEGORIES = _LC_ONCE.all_categories
del _LC_ONCE

# ── Team branding (Combustion redesign, 2026-06-08) ──────────────────
# MLB Stats API team id → {abbr, primary, secondary}. Primary = official
# primary brand color (used for team-themed chrome: popup headers, roster
# accents, chips). Logos render from mlbstatic by id. Anchors verified against
# the redesign plan; secondaries are the team's standard second brand color.
TEAM_BRAND: dict[int, dict] = {
    108: {"abbr": "LAA", "primary": "#BA0021", "secondary": "#003263"},  # Angels
    109: {"abbr": "ARI", "primary": "#A71930", "secondary": "#E3D4AD"},  # D-backs
    110: {"abbr": "BAL", "primary": "#DF4601", "secondary": "#000000"},  # Orioles
    111: {"abbr": "BOS", "primary": "#BD3039", "secondary": "#0C2340"},  # Red Sox
    112: {"abbr": "CHC", "primary": "#0E3386", "secondary": "#CC3433"},  # Cubs
    113: {"abbr": "CIN", "primary": "#C6011F", "secondary": "#000000"},  # Reds
    114: {"abbr": "CLE", "primary": "#0C2340", "secondary": "#E31937"},  # Guardians
    115: {"abbr": "COL", "primary": "#33006F", "secondary": "#C4CED4"},  # Rockies
    116: {"abbr": "DET", "primary": "#0C2340", "secondary": "#FA4616"},  # Tigers
    117: {"abbr": "HOU", "primary": "#002D62", "secondary": "#EB6E1F"},  # Astros
    118: {"abbr": "KC", "primary": "#004687", "secondary": "#BD9B60"},  # Royals
    119: {"abbr": "LAD", "primary": "#005A9C", "secondary": "#EF3E42"},  # Dodgers
    120: {"abbr": "WSH", "primary": "#AB0003", "secondary": "#14225A"},  # Nationals
    121: {"abbr": "NYM", "primary": "#002D72", "secondary": "#FF5910"},  # Mets
    133: {"abbr": "ATH", "primary": "#003831", "secondary": "#EFB21E"},  # Athletics
    134: {"abbr": "PIT", "primary": "#FDB827", "secondary": "#27251F"},  # Pirates
    135: {"abbr": "SD", "primary": "#2F241D", "secondary": "#FFC425"},  # Padres
    136: {"abbr": "SEA", "primary": "#0C2C56", "secondary": "#005C5C"},  # Mariners
    137: {"abbr": "SF", "primary": "#FD5A1E", "secondary": "#27251F"},  # Giants
    138: {"abbr": "STL", "primary": "#C41E3A", "secondary": "#0C2340"},  # Cardinals
    139: {"abbr": "TB", "primary": "#092C5C", "secondary": "#8FBCE6"},  # Rays
    140: {"abbr": "TEX", "primary": "#003278", "secondary": "#C0111F"},  # Rangers
    141: {"abbr": "TOR", "primary": "#134A8E", "secondary": "#1D2D5C"},  # Blue Jays
    142: {"abbr": "MIN", "primary": "#002B5C", "secondary": "#D31145"},  # Twins
    143: {"abbr": "PHI", "primary": "#E81828", "secondary": "#002D72"},  # Phillies
    144: {"abbr": "ATL", "primary": "#13274F", "secondary": "#CE1141"},  # Braves
    145: {"abbr": "CWS", "primary": "#27251F", "secondary": "#C4CED4"},  # White Sox
    146: {"abbr": "MIA", "primary": "#00A3E0", "secondary": "#EF3340"},  # Marlins
    147: {"abbr": "NYY", "primary": "#0C2340", "secondary": "#C4CED4"},  # Yankees
    158: {"abbr": "MIL", "primary": "#12284B", "secondary": "#FFC52F"},  # Brewers
}

# abbr → id reverse map so abbr-only callers can resolve branding/logos.
TEAM_ABBR_TO_ID: dict[str, int] = {v["abbr"]: k for k, v in TEAM_BRAND.items()}

_TEAM_FALLBACK_COLOR = "#ff6d00"  # orange — when a team can't be resolved


def _resolve_team_id(team) -> int | None:
    """Resolve an MLB Stats API team id from either an int id or an abbr str."""
    if team is None:
        return None
    if isinstance(team, bool):  # guard: bool is an int subclass
        return None
    if isinstance(team, int):
        return team if team in TEAM_BRAND else None
    try:
        # numeric-string ids (e.g. "117")
        as_int = int(team)
        if as_int in TEAM_BRAND:
            return as_int
    except (ValueError, TypeError):
        pass
    if isinstance(team, str):
        return TEAM_ABBR_TO_ID.get(team.strip().upper())
    return None


def team_logo_url(team) -> str:
    """Return the MLB team logo SVG URL for an id or abbr.

    Falls back to a generic id-0 path when the team is unknown (mlbstatic
    returns a transparent placeholder rather than 404ing the render).
    """
    tid = _resolve_team_id(team)
    return f"https://www.mlbstatic.com/team-logos/{tid if tid is not None else 0}.svg"


def team_color(team) -> str:
    """Return a team's official primary hex (id or abbr). Orange fallback."""
    tid = _resolve_team_id(team)
    if tid is None:
        return _TEAM_FALLBACK_COLOR
    return TEAM_BRAND[tid]["primary"]


def text_on(hex_color: str) -> str:
    """Pick readable text (#ffffff or charcoal) for a given background hex.

    WCAG-style relative luminance: dark text on light backgrounds, white text
    on dark. Keeps team-color labels legible regardless of the team color.
    """
    h = (hex_color or "").lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        return "#ffffff"
    try:
        r, g, b = (int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError:
        return "#ffffff"

    def _lin(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    lum = 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)
    # Threshold ~0.4 keeps mid-tone team colors (e.g. orange) on dark text only
    # when genuinely light; navy/maroon get white. 0.179 is the WCAG #767676
    # midpoint, but 0.4 reads better for saturated brand colors.
    return THEME["tx"] if lum > 0.4 else "#ffffff"


# 2026-05-17 Section 3 D10: canonical category short→long display-name
# map. Previously duplicated as `_CAT_DISPLAY` (uppercase) in 12_Trade_Finder
# + `CAT_DISPLAY_NAMES` (lowercase) in 2_Line-up_Optimizer. Centralized
# here in UPPERCASE; lowercase callers do `.get(cat.upper(), cat)`.
CAT_DISPLAY_NAMES: dict[str, str] = {
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
    """Render a styled section header (Archivo display, title case)."""
    st.markdown(f'<div class="sec-head">{title}</div>', unsafe_allow_html=True)


def build_empty_state_html(title: str, body: str = "", icon_key: str = "baseball") -> str:
    """Build the instrument-styled empty-data state panel (Combustion Finale).

    Replaces bare st.info() in data-empty contexts: dot-grid dashed panel,
    centered, orange SVG icon — never emoji.
    """
    icon = PAGE_ICONS.get(icon_key, PAGE_ICONS["baseball"])
    safe_title = _html.escape(str(title))
    safe_body = _html.escape(str(body)) if body else ""
    body_html = f'<div class="es-body">{safe_body}</div>' if safe_body else ""
    return (
        '<div class="empty-state">'
        f'<div class="es-icon">{icon}</div>'
        f'<div class="es-title">{safe_title}</div>'
        f"{body_html}"
        "</div>"
    )


def render_empty_state(title: str, body: str = "", icon_key: str = "baseball") -> None:
    """st.markdown wrapper for build_empty_state_html."""
    st.markdown(build_empty_state_html(title, body, icon_key), unsafe_allow_html=True)


def no_league_data_message(reason: str = "") -> str:
    """Member-facing message for the empty-league-data state, adapted to WHY it
    is empty (from ``YahooDataService.data_unavailable_reason()``).

    F3 (2026-06-02 silent-failure sweep): the four personalized pages used to show
    a single generic "No league data loaded — connect your Yahoo league" string,
    which implies the league was never connected even when the real cause is a
    transient Yahoo timeout (Yahoo slow → SQLite empty). This makes the message
    honest for the timeout/error cases while keeping the original guidance for the
    genuinely-not-loaded case.
    """
    if reason == "timeout":
        return (
            "Live league data timed out just now (Yahoo was slow to respond). It may "
            "still be warming up — use the Refresh Yahoo Data button, or reload in a moment."
        )
    if reason == "error":
        return (
            "League data is temporarily unavailable (a data fetch failed). Use the "
            "Refresh Yahoo Data button, or reload in a moment."
        )
    return (
        "No league data loaded yet. It loads automatically on app launch and may still "
        "be warming up; if this persists, reconnect your Yahoo league in Connect League."
    )


# ── Plotly Theme Helpers ───────────────────────────────────────────


def get_plotly_layout(theme=None):
    """Return Plotly layout kwargs for consistent chart theming.

    Combustion palette: transparent bg, Inter font, fine gridlines, orange colorway.
    Plotly 6 rejects 8-digit hex, so gridlines use rgba().
    """
    t = theme or THEME
    grid = "rgba(16,33,58,0.06)"
    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": t["tx"], "family": "Inter, sans-serif"},
        "margin": {"l": 40, "r": 40, "t": 30, "b": 30},
        "height": 350,
        "colorway": [t["primary"], t["flame"], t["gold"], t["green_l"], t["cold"]],
        "xaxis": {"gridcolor": grid, "zerolinecolor": grid, "linecolor": grid},
        "yaxis": {"gridcolor": grid, "zerolinecolor": grid, "linecolor": grid},
    }


def get_plotly_polar(theme=None):
    """Return polar/radar chart config for Player Compare."""
    t = theme or THEME
    return {
        "bgcolor": "rgba(0,0,0,0)",
        "radialaxis": {
            "gridcolor": "rgba(16,33,58,0.06)",
            "tickfont": {"color": t["tx_subtle"], "size": 10},
            "visible": True,
        },
        "angularaxis": {
            "gridcolor": "rgba(16,33,58,0.08)",
            "tickfont": {"color": t["tx"], "size": 11},
        },
    }


# ── Backward Compatibility Stubs ──────────────────────────────────
# Old code may call get_theme() or render_theme_toggle().


def get_theme():
    """Return the theme dict (light-only, no dark mode)."""
    return THEME


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


_HEATER_LOGO_B64: str | None = None


def _heater_logo_b64() -> str:
    """Base64 of the v2 HEATER logo PNG (assets/heater_logo_v2.png), cached per process.

    Returns '' if the asset is missing so the caller can fall back to a text wordmark.
    """
    global _HEATER_LOGO_B64
    if _HEATER_LOGO_B64 is None:
        import base64
        from pathlib import Path

        try:
            _p = Path(__file__).resolve().parent.parent / "assets" / "heater_logo_v2.png"
            _HEATER_LOGO_B64 = base64.b64encode(_p.read_bytes()).decode()
        except Exception:
            _HEATER_LOGO_B64 = ""
    return _HEATER_LOGO_B64


def inject_custom_css():
    """Inject the complete Heater CSS for all pages.

    Call this once at the top of every page, after st.set_page_config().
    Uses glassmorphism, 3D buttons, kinetic typography, and tactile animations.
    """
    t = THEME
    # Preconnect MUST live in its own markdown call: a <link> opener demotes the
    # whole block to a CommonMark type-6 HTML block, which ends at the first
    # blank line — the <style> sheet below would then render as literal text.
    # Standalone, the sheet starts with <style> and gets type-1 (raw-to-close).
    st.markdown(
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,500..900&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    /* ── Glide Data Grid root-level theme overrides — Combustion scoreboard:
          Archivo-800 uppercase headers, Archivo data cells, charcoal text. ── */
    :root {{
        --gdg-bg-header: #ffffff !important;
        --gdg-bg-header-has-focus: #f6f7f9 !important;
        --gdg-bg-header-hovered: #eef0f3 !important;
        --gdg-text-header: #2c2f36 !important;
        --gdg-bg-cell: #ffffff !important;
        --gdg-bg-cell-medium: #f6f7f9 !important;
        --gdg-text-dark: #1b1c20 !important;
        --gdg-border-color: rgba(24,26,32,.10) !important;
        --gdg-font-family: Archivo, system-ui, sans-serif !important;
        --gdg-header-font-style: 800 12px Archivo, sans-serif !important;
        --gdg-base-font-style: 700 13px Archivo, sans-serif !important;

        /* ── FP-revamp design tokens (revamp task 1) ──
           Pages + renderers reference these vars, not hex literals. Values are
           derived from THEME so the dict stays the single source of truth. */
        --fp-app-bg: {t["bg"]};
        --fp-surface: {t["surface"]};
        --fp-primary: {t["primary"]};
        --fp-amber: {t["warn"]};
        --fp-ink: {t["ink"]};
        --fp-tx: {t["tx"]};
        --fp-tx-muted: {t["tx_muted"]};
        --fp-tx-subtle: {t["tx_subtle"]};
        --fp-border: {t["border"]};
        --fp-divider: {t["divider"]};
        --fp-sidebar-bg: {t["sidebar_bg"]};
        --fp-sidebar-ink: {t["sidebar_ink"]};
        /* Combustion redesign derived tokens (2026-06-08) */
        --fp-flame: {t["flame"]};
        --fp-ember: {t["ember"]};
        --fp-cold: {t["cold"]};
        --fp-navy: {t["navy"]};
        --fp-navy2: {t["navy2"]};
        --fp-radius: 12px;
        --fp-radius-sm: 8px;
        --fp-shadow: 0 1px 3px rgba(16,33,58,.08);
        --font-body: 'Inter', system-ui, -apple-system, sans-serif;
        --font-display: 'Archivo', system-ui, sans-serif;
        --font-mono: 'IBM Plex Mono', monospace;
        --dur-1: 120ms;
        --dur-2: 180ms;
        --ease-snap: cubic-bezier(.2, .7, .3, 1);
    }}

    /* ── BASE ─────────────────────────────────── */
    /* Font-lock the bare document root too: Streamlit computes its default
       "Source Sans" on <body>, which leaks onto any node that doesn't inherit
       from .stApp (portals, tooltips, BaseWeb popovers mounted to <body>).
       Force Inter here so "these fonts only" holds at the very root. Headings
       (Archivo) + figures (mono) re-assert their own families later. */
    html, body {{ font-family: var(--font-body) !important; }}
    .stApp {{
        /* Owner request 2026-06-10: plain white canvas — the feTurbulence grain
           and blueprint-grid gradient layers were removed app-wide. */
        background: var(--fp-app-bg) !important;
        font-family: var(--font-body);
        color: {t["tx"]};
    }}
    .block-container {{ padding-top: 1rem; padding-bottom: 1rem; }}
    /* Force full-width content on EVERY render. When a page is reached via a
       direct URL (Streamlit pages/ auto-discovery) rather than through app.py's
       st.navigation, app.py's set_page_config(layout="wide") never runs and
       Streamlit falls back to its narrow centered max-width — leaving the page
       "condensed". inject_custom_css runs on every page, so overriding the
       container max-width here keeps the layout wide regardless of routing.
       Combustion redesign (2026-06-08): full-bleed instrument layout — content
       fills the whole canvas right of the navy rail, no 1180px clamp. Streamlit's
       main block-container already starts to the RIGHT of the sidebar, so this
       never slides under the rail. Comfortable fluid side padding. */
    [data-testid="stMainBlockContainer"],
    [data-testid="stAppViewBlockContainer"],
    .block-container {{
        max-width: none !important;
        margin-left: 0 !important;
        margin-right: 0 !important;
        padding-left: clamp(18px, 2.4vw, 48px) !important;
        padding-right: clamp(18px, 2.4vw, 48px) !important;
    }}

    /* ── HIDE STREAMLIT CHROME ────────────────── */
    /* Pure clutter (decoration strip + Deploy button) — hidden on all widths. */
    [data-testid="stDecoration"] {{ display: none !important; }}
    [data-testid="stAppDeployButton"] {{ display: none !important; }}
    /* DESKTOP: hide the header bar AND the toolbar ONLY WHILE THE SIDEBAR IS
       EXPANDED — the rail carries the nav, so Streamlit's chrome is redundant.
       The hide MUST be conditioned on body:has(sidebar[aria-expanded="true"]):
       the only reopen control (stExpandSidebarButton, ») lives inside that
       header, so an unconditional hide strands anyone who clicks the « collapse
       chevron (2026-06-10 prod report: owner collapsed the rail and had no way
       back; the desktop chrome must return whenever the rail is collapsed or
       absent, e.g. on the login page).
       PHONES: KEEP both always. The collapsed sidebar's open toggle is parented
       INSIDE stToolbar, so hiding the toolbar on mobile strands users with no
       way to open the nav (2026-06-01: this was the mobile-navigation bug). */
    @media (min-width: 768px) {{
        body:has(section[data-testid="stSidebar"][aria-expanded="true"]) header[data-testid="stHeader"] {{ display: none !important; }}
        body:has(section[data-testid="stSidebar"][aria-expanded="true"]) [data-testid="stToolbar"] {{ display: none !important; }}
        [data-testid="stExpandSidebarButton"] {{
            display: inline-flex !important;
            visibility: visible !important;
            align-items: center !important;
            justify-content: center !important;
            min-width: 2.5rem !important;
            min-height: 2.5rem !important;
        }}
    }}
    @media (max-width: 767px) {{
        header[data-testid="stHeader"] {{ display: flex !important; }}
        [data-testid="stToolbar"] {{ display: flex !important; }}
        [data-testid="stExpandSidebarButton"] {{
            display: inline-flex !important;
            visibility: visible !important;
            min-width: 2.5rem !important;
            min-height: 2.5rem !important;
            align-items: center !important;
            justify-content: center !important;
        }}
    }}

    /* ── TEXT PROTECTION (global) ─────────────── */
    .stApp p, .stApp span, .stApp td, .stApp th,
    .stApp label, .stApp .stMarkdown {{
        overflow-wrap: anywhere;
        word-break: break-word;
    }}
    /* ── SECTION HEADER (Combustion: Archivo display + orange accent rule) ── */
    .sec-head {{
        font-family: var(--font-display);
        font-weight: 800;
        font-size: 20px;
        text-transform: none;
        letter-spacing: -0.005em;
        color: {t["tx"]};
        margin-bottom: 10px;
        padding-bottom: 8px;
        border-bottom: 2px solid var(--fp-primary);
        position: relative;
    }}
    .sec-label {{
        font-size: 12px !important;
        font-weight: 700 !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        color: {t["tx2"]} !important;
        margin-bottom: 4px !important;
    }}

    /* ── TYPE SCALE (Combustion redesign) — Archivo display + Inter body ── */
    h1, .heater-h1 {{
        font-family: var(--font-display) !important;
        font-weight: 900 !important;
        font-size: 28px !important;
        letter-spacing: -0.02em !important;
        text-transform: none !important;
        color: {t["tx"]} !important;
    }}
    h2 {{ font-family: var(--font-display) !important; font-weight: 800 !important; font-size: 20px !important; letter-spacing: -0.01em !important; text-transform: none !important; }}
    h3 {{ font-family: var(--font-display) !important; font-weight: 800 !important; font-size: 16px !important; letter-spacing: -0.005em !important; text-transform: none !important; }}
    .stat, .mono, td.num {{ font-family: var(--font-mono) !important; }}

    /* ── INSTRUMENT PANEL (Combustion) — reusable flat panel matching
          mockup .panel: dot-grid texture, hairline border, soft layered
          shadow, optional orange accent + corner ticks. ── */
    .instr-panel {{
        position: relative;
        border: 1px solid var(--fp-border);
        border-radius: 14px;
        padding: 20px 22px;
        overflow: hidden;
        background:
            radial-gradient(rgba(24,26,32,.035) 1px, transparent 1.3px) 0 0 / 20px 20px,
            var(--fp-surface);
        box-shadow: inset 0 1px 0 rgba(255,255,255,.9), 0 1px 3px rgba(24,26,32,.06), 0 6px 20px rgba(24,26,32,.04);
    }}
    /* Owner request 2026-06-10: surface-filled elements (pale #f5f6f8 fill)
       carry darker, bolder text than the white canvas around them. Accent
       colors (orange links, status chips) keep their own rules. */
    .instr-panel, .glass, .alt, .context-card,
    [data-testid="stPopover"], [data-baseweb="popover"], [data-baseweb="menu"],
    [data-testid="stChatMessage"] {{
        color: #101114;
        font-weight: 600;
    }}
    /* Faint orange wash bleeding from the top-right, like the mockup. */
    .instr-panel::before {{
        content: "";
        position: absolute;
        inset: 0;
        background: radial-gradient(120% 80% at 100% 0%, rgba(255,109,0,.05), transparent 42%);
        pointer-events: none;
    }}
    /* Opt-in orange accent rules. */
    .instr-panel.accent-top {{ border-top: 2px solid var(--fp-primary); }}
    .instr-panel.accent-left {{ border-left: 3px solid var(--fp-primary); }}
    /* Corner ticks — add four <span class="pcorner tl|tr|bl|br"></span> children,
       or rely on ::after for a single TL tick when markup is minimal. */
    .instr-panel .pcorner {{
        position: absolute;
        width: 12px;
        height: 12px;
        border: 1px solid var(--fp-primary);
        opacity: .55;
        pointer-events: none;
    }}
    .instr-panel .pcorner.tl {{ top: 10px; left: 10px; border-right: none; border-bottom: none; }}
    .instr-panel .pcorner.tr {{ top: 10px; right: 10px; border-left: none; border-bottom: none; }}
    .instr-panel .pcorner.bl {{ bottom: 10px; left: 10px; border-right: none; border-top: none; }}
    .instr-panel .pcorner.br {{ bottom: 10px; right: 10px; border-left: none; border-top: none; }}

    /* ── GLASS CARD — Combustion: glassmorphism flattened to a flat instrument
          card (solid surface, hairline border, soft shadow). The translucent
          fill + blur are removed; layout/padding preserved. ── */
    .glass {{
        background: var(--fp-surface);
        border: 1px solid var(--fp-border);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 12px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.9), 0 1px 3px rgba(24,26,32,.06), 0 6px 20px rgba(24,26,32,.04);
        transition: box-shadow 0.2s ease;
    }}
    .glass:hover {{
        transform: translateY(-1px);
        box-shadow: inset 0 1px 0 rgba(255,255,255,.9), 0 4px 14px rgba(24,26,32,.10);
    }}

    /* ── COMMAND BAR — Combustion: flat instrument strip (no blur). ── */
    .cmd-bar {{
        background: var(--fp-surface);
        border: 1px solid var(--fp-border);
        border-radius: 12px;
        padding: 14px 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 16px;
        flex-wrap: wrap;
        gap: 10px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.9), 0 1px 3px rgba(24,26,32,.06);
    }}
    .cmd-left {{
        font-family: var(--font-body);
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
        font-family: var(--font-body);
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
        font-family: var(--font-body);
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        padding: 8px 16px;
        border-radius: 10px;
    }}
    @keyframes heatPulse {{
        0%, 100% {{ box-shadow: 0 0 8px rgba(255, 109, 0, 0.4); }}
        50% {{ box-shadow: 0 0 24px rgba(255, 109, 0, 0.7), 0 0 48px rgba(255, 154, 60, 0.35); }}
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

    /* ── HERO PICK CARD — Combustion: flat instrument card with a solid
          orange left accent (glassmorphism + perspective tilt removed). ── */
    .hero {{
        background: var(--fp-surface);
        border: 1px solid var(--fp-border);
        border-left: 3px solid var(--fp-primary);
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 16px;
        position: relative;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.9), 0 1px 3px rgba(24,26,32,.06), 0 6px 20px rgba(24,26,32,.04);
        transition: box-shadow 0.2s ease;
    }}
    .hero:hover {{
        transform: translateY(-1px);
        box-shadow: inset 0 1px 0 rgba(255,255,255,.9), 0 4px 14px rgba(24,26,32,.10);
    }}
    .hero .p-name {{
        font-family: var(--font-display);
        font-weight: 900;
        font-size: 36px;
        color: {t["tx"]};
        text-transform: none;
        letter-spacing: -0.01em;
        line-height: 1.1;
        word-break: break-word;
    }}
    .hero .p-meta {{
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        color: {t["tx2"]};
        margin-top: 4px;
    }}
    .hero .score-badge {{
        position: absolute;
        top: 16px; right: 20px;
        background: linear-gradient(135deg, {t["primary"]}, {t["ember"]});
        color: {t["ink"]};
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 700;
        font-size: 22px;
        padding: 8px 14px;
        border-radius: 12px;
        cursor: help;
        box-shadow: 0 4px 16px rgba(255, 109, 0, 0.30);
    }}
    .hero .reason {{
        font-family: 'Inter', sans-serif;
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
        background: var(--fp-surface);
        color: {t["tx2"]};
    }}
    .sgp-chip.pos {{ border-color: {t["green"]}; color: {t["green"]}; background: rgba(45, 106, 79, 0.08); }}
    .sgp-chip.neg {{ border-color: {t["danger"]}; color: {t["danger"]}; background: rgba(224, 73, 47, 0.08); }}

    /* ── ALTERNATIVE CARDS — Combustion: flat surface, hairline border,
          tier-colored left rail (glassmorphism + lift removed). ── */
    .alt {{
        background: var(--fp-surface);
        border: 1px solid var(--fp-border);
        border-left: 4px solid {t["border"]};
        border-radius: 12px;
        padding: 12px 14px;
        box-shadow: 0 1px 3px rgba(24,26,32,.06);
        transition: box-shadow 0.2s ease;
        cursor: default;
    }}
    .alt:hover {{
        transform: none;
        box-shadow: 0 4px 14px rgba(24,26,32,.10);
    }}
    .alt .a-rank {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        color: {t["tx2"]};
    }}
    .alt .a-name {{
        font-family: var(--font-display);
        font-weight: 800;
        font-size: 16px;
        color: {t["tx"]};
        text-transform: none;
        letter-spacing: -0.005em;
        margin: 2px 0;
        word-break: break-word;
    }}
    .alt .a-meta {{
        font-family: 'Inter', sans-serif;
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
    .badge-reach {{ background: rgba(224, 73, 47, 0.1); color: {t["danger"]}; border: 1px solid rgba(224, 73, 47, 0.3); }}
    .badge-fair {{ background: {t["card_h"]}; color: {t["tx2"]}; }}
    .badge-risk-low {{ background: rgba(45, 106, 79, 0.1); color: {t["ok"]}; }}
    .badge-risk-med {{ background: rgba(255, 159, 28, 0.1); color: {t["warn"]}; }}
    .badge-risk-high {{ background: rgba(224, 73, 47, 0.1); color: {t["danger"]}; }}
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
        background: var(--fp-surface);
        border: 1px solid var(--fp-border);
        border-radius: 10px;
        padding: 6px 10px;
        text-align: center;
        min-height: 48px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 1px 2px rgba(24,26,32,.05);
        transition: box-shadow 0.2s ease;
    }}
    .roster-slot:hover {{
        transform: none;
        box-shadow: 0 4px 14px rgba(24,26,32,.10);
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
        font-family: var(--font-body);
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: {t["tx2"]};
    }}
    .roster-slot .s-player {{
        font-family: 'Inter', sans-serif;
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
        0%, 100% {{ transform: scale(1); box-shadow: 0 0 0 rgba(224, 73, 47, 0); }}
        50% {{ transform: scale(1.08); box-shadow: 0 0 16px rgba(224, 73, 47, 0.3); }}
    }}
    .scar-label {{
        font-family: var(--font-body);
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: {t["tx2"]};
    }}

    /* ── DRAFT BOARD — Combustion scoreboard: Archivo headers + cells,
          hairline dividers, orange-tint hover (glassmorphism removed). ── */
    .draft-board {{
        width: 100%;
        border-collapse: collapse;
        font-family: var(--font-display);
        font-weight: 700;
        font-size: 11px;
        font-variant-numeric: tabular-nums;
    }}
    .draft-board th {{
        background: var(--fp-surface);
        color: #2c2f36;
        font-family: var(--font-display);
        font-weight: 800;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: .06em;
        padding: 10px 6px;
        border-bottom: 2px solid rgba(24,26,32,.18);
        position: sticky;
        top: 0;
        z-index: 10;
    }}
    .draft-board td {{
        padding: 6px;
        border-bottom: 1px solid rgba(24,26,32,.055);
        color: {t["tx"]};
        max-width: 200px;
        overflow-wrap: anywhere;
        word-break: break-word;
    }}
    .draft-board tr:nth-child(even) {{
        background: rgba(24,26,32,.015);
    }}
    .draft-board tr:hover td {{
        background: rgba(255, 109, 0, 0.06);
    }}
    .draft-board .user-col {{
        background: rgba(255, 109, 0, 0.05);
        border-left: 2px solid {t["primary"]};
        border-right: 2px solid {t["primary"]};
    }}
    .draft-board .user-col th {{
        background: linear-gradient(135deg, {t["primary"]}, {t["ember"]});
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

    /* ── CATEGORY CARDS — Combustion: flat instrument tile. ── */
    .cat-card {{
        background: var(--fp-surface);
        border: 1px solid var(--fp-border);
        border-radius: 12px;
        padding: 12px 14px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(24,26,32,.06);
        transition: box-shadow 0.2s ease;
    }}
    .cat-card:hover {{
        transform: none;
        box-shadow: 0 4px 14px rgba(24,26,32,.10);
    }}
    .cat-name {{
        font-family: var(--font-body);
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

    /* ── ALERTS, FEED, WIZARD — Combustion: flat surfaces, no blur. ── */
    .alert-card {{
        background: var(--fp-surface);
        border: 1px solid var(--fp-border);
        border-left: 4px solid {t["warn"]};
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 13px;
        color: {t["tx"]};
        box-shadow: 0 1px 3px rgba(24,26,32,.06);
    }}
    .alert-card.critical {{ border-left-color: {t["danger"]}; }}

    .feed-card {{
        background: var(--fp-surface);
        border: 1px solid var(--fp-border);
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 6px;
        font-size: 13px;
        box-shadow: 0 1px 2px rgba(24,26,32,.05);
        transition: box-shadow 0.2s ease;
    }}
    .feed-card:hover {{
        transform: none;
        box-shadow: 0 4px 14px rgba(24,26,32,.10);
    }}
    .feed-card.user-pick {{
        border-left: 3px solid {t["primary"]};
        background: rgba(255, 109, 0, 0.06);
    }}
    .feed-pick-num {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        color: {t["tx2"]};
    }}
    .feed-name {{
        font-family: 'Inter', sans-serif;
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
        font-family: var(--font-display);
        font-weight: 700;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: .04em;
        color: {t["tx2"]};
        background: var(--fp-surface);
        border: 1px solid var(--fp-border);
    }}
    .wizard-step:first-child {{ border-radius: 12px 0 0 12px; }}
    .wizard-step:last-child {{ border-radius: 0 12px 12px 0; }}
    .wizard-step.active {{
        background: linear-gradient(135deg, {t["primary"]}, {t["ember"]});
        color: {t["ink"]};
        border-color: transparent;
        box-shadow: 0 4px 16px rgba(255, 109, 0, 0.30);
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
        font-family: var(--font-body);
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

    /* ── VERDICT BANNER (Trade Analyzer) — Combustion: flat, no blur. ── */
    .verdict-banner {{
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
        text-align: center;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.9), 0 1px 3px rgba(24,26,32,.06), 0 6px 20px rgba(24,26,32,.04);
    }}
    .verdict-banner .verdict-text {{
        font-family: var(--font-display);
        font-weight: 900;
        font-size: 32px;
        text-transform: uppercase;
        letter-spacing: .02em;
    }}
    .verdict-banner .verdict-conf {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 18px;
        margin-top: 4px;
    }}

    /* ── METRIC CARD (In-Season) — Combustion: flat tile, no 3D tilt/blur.
          (Background/border/shadow also set by the late FP flatten block.) ── */
    .metric-card {{
        background: var(--fp-surface);
        border: 1px solid var(--fp-border);
        border-radius: 12px;
        padding: 16px 18px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(24,26,32,.06);
        transform: none;
        transition: box-shadow 0.2s ease;
    }}
    .metric-card:hover {{
        transform: none;
        box-shadow: 0 4px 14px rgba(24,26,32,.10);
    }}
    .metric-card .metric-label {{
        font-family: var(--font-body);
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

    /* ── PLAYER CARD (card-based selection) — Combustion: flat surface,
          orange hover/selected accents (glassmorphism + lift removed). ── */
    .player-card {{
        background: var(--fp-surface);
        border: 1px solid var(--fp-border);
        border-radius: 14px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(24,26,32,.06);
        transition: box-shadow 0.2s ease, border-color 0.2s ease;
        cursor: pointer;
    }}
    .player-card:hover {{
        transform: none;
        box-shadow: 0 4px 14px rgba(24,26,32,.10);
        border-color: rgba(255, 109, 0, 0.35);
    }}
    .player-card.selected {{
        border: 2px solid {t["primary"]};
        background: rgba(255, 109, 0, 0.06);
        box-shadow: 0 4px 16px rgba(255, 109, 0, 0.15);
    }}
    .pc-name {{
        font-family: var(--font-display);
        font-weight: 800;
        font-size: 15px;
        letter-spacing: -0.005em;
        color: {t["tx"]};
        text-transform: none;
    }}
    .pc-pos {{
        font-family: 'Inter', sans-serif;
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
    /* Streamlit remounts the DOM on every rerun, so entrance keyframes on
       persistent cards replay on every widget click. slideUp is kept ONLY
       for true entrances (.reco-banner-detail). */
    @keyframes slideUp {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    /* ── KINETIC GRADIENT TEXT ────────────────── */
    @keyframes gradientShift {{
        0%, 100% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
    }}
    /* ── SPLASH SCREEN TITLE ─────────────────── */
    @keyframes titleReveal {{
        0% {{ letter-spacing: 20px; opacity: 0; transform: scale(1.3); }}
        60% {{ letter-spacing: 6px; opacity: 1; }}
        80% {{ transform: scale(0.98); }}
        100% {{ transform: scale(1); letter-spacing: 6px; }}
    }}
    .splash-title {{
        font-family: 'Archivo', sans-serif;
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
        font-family: var(--font-body);
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
        background: linear-gradient(135deg, {t["primary"]}, {t["ember"]});
        color: {t["ink"]};
        border: none;
        transform: none;
        box-shadow: 0 4px 16px rgba(255, 109, 0, 0.30);
    }}
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {{
        transform: none;
        box-shadow: 0 6px 22px rgba(255, 109, 0, 0.42);
    }}
    .stButton > button[kind="primary"]:active,
    .stButton > button[data-testid="stBaseButton-primary"]:active {{
        transform: scale(0.99);
        box-shadow: 0 2px 8px rgba(255, 109, 0, 0.30);
        transition: all 0.08s ease;
    }}
    /* Inputs */
    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input {{
        background: rgba(255, 255, 255, 0.7) !important;
        color: {t["tx"]} !important;
        border: 1px solid {t["border"]} !important;
        border-radius: 12px !important;
        font-family: 'Inter', sans-serif !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }}
    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stNumberInput"] input:focus {{
        border-color: {t["primary"]} !important;
        box-shadow: 0 0 0 3px rgba(255, 109, 0, 0.12) !important;
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
        font-family: var(--font-body);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-size: 13px;
        border-radius: 10px;
        padding: 8px 16px !important;
        color: {t["tx2"]};
        transition: all 0.2s ease;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {t["primary"]}, {t["ember"]}) !important;
        color: {t["ink"]} !important;
        box-shadow: 0 2px 8px rgba(255, 109, 0, 0.30);
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

    /* DataFrames — Combustion scoreboard frame: hairline border, soft shadow,
       white surface (was a tan-bordered cream panel). */
    div[data-testid="stDataFrame"] {{
        border: 1px solid var(--fp-border) !important;
        border-radius: 12px !important;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(24,26,32,.06), 0 6px 20px rgba(24,26,32,.04) !important;
        background: var(--fp-surface) !important;
    }}
    /* Glide Data Grid canvas theming via CSS custom properties (belt-and-suspenders) */
    div[data-testid="stDataFrame"] [data-testid="glideDataEditor"],
    div[data-testid="stDataFrame"] .dvn-scroller,
    div[data-testid="stDataFrame"] {{
        --gdg-bg-header: #ffffff !important;
        --gdg-bg-header-has-focus: #f6f7f9 !important;
        --gdg-bg-header-hovered: #eef0f3 !important;
        --gdg-text-header: #2c2f36 !important;
        --gdg-bg-cell: #ffffff !important;
        --gdg-bg-cell-medium: #f6f7f9 !important;
        --gdg-text-dark: #1b1c20 !important;
        --gdg-border-color: rgba(24,26,32,.10) !important;
        --gdg-font-family: Archivo, system-ui, sans-serif !important;
        --gdg-header-font-style: 800 12px Archivo, sans-serif !important;
        --gdg-base-font-style: 700 13px Archivo, sans-serif !important;
    }}
    div[data-testid="stDataFrame"] [data-testid="glideDataEditor"] {{
        background: var(--fp-surface) !important;
    }}

    /* Override navy secondaryBackgroundColor on non-dataframe elements */
    pre, code, .stCodeBlock, [data-testid="stCodeBlock"] {{
        background: rgba(245, 242, 237, 0.9) !important;
        color: {t["tx"]} !important;
    }}
    [data-testid="stPopover"], .stPopover {{
        background: var(--fp-surface) !important;
        color: {t["tx"]} !important;
    }}
    [data-testid="stChatMessage"], .stChatMessage {{
        background: var(--fp-surface) !important;
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
    /* Multiselect tags/pills — override navy bg (Combustion orange tint) */
    [data-baseweb="tag"] {{
        background: rgba(255, 109, 0, 0.10) !important;
        color: {t["primary"]} !important;
    }}
    /* Select/dropdown menus */
    [data-baseweb="popover"], [data-baseweb="menu"] {{
        background: var(--fp-surface) !important;
        color: {t["tx"]} !important;
    }}
    [data-baseweb="select"] [data-baseweb="input"] {{
        background: var(--fp-surface) !important;
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
    /* Left accent border for visual anchor — Combustion orange. */
    div[data-testid="stDataFrame"]::before {{
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: linear-gradient(180deg, var(--fp-primary), var(--fp-ember));
        border-radius: 12px 0 0 12px;
        z-index: 1;
    }}
    div[data-testid="stDataFrame"] {{
        position: relative;
    }}

    /* Styled HTML tables (render_styled_table helper) — Combustion scoreboard:
       hairline frame, orange left rail, Archivo headers + cells, tabular nums. */
    .heater-table-wrap {{
        border: 1px solid var(--fp-border);
        border-radius: 12px;
        overflow: hidden;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.9), 0 1px 3px rgba(24,26,32,.06), 0 6px 20px rgba(24,26,32,.04);
        background: var(--fp-surface);
        position: relative;
        border-left: 4px solid var(--fp-primary);
    }}
    .heater-table {{
        width: 100%;
        border-collapse: collapse;
        font-family: var(--font-display);
        font-size: 13px;
        color: {t["tx"]};
    }}
    .heater-table thead th {{
        background: var(--fp-surface) !important;
        color: #2c2f36 !important;
        font-weight: 800 !important;
        font-family: var(--font-display) !important;
        font-size: 11px !important;
        letter-spacing: .06em !important;
        text-transform: uppercase !important;
        padding: 10px 12px !important;
        border: none !important;
        border-bottom: 2px solid rgba(24,26,32,.18) !important;
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
        border-bottom: 1px solid rgba(24,26,32,.055) !important;
        border-top: none !important;
        border-left: none !important;
        border-right: none !important;
        background: var(--fp-surface) !important;
        font-family: var(--font-display);
        font-weight: 700;
        font-size: 13.5px !important;
        font-variant-numeric: tabular-nums !important;
        color: {t["tx"]};
    }}
    .heater-table tbody td:first-child {{
        white-space: nowrap !important;
        font-weight: 700 !important;
    }}
    .heater-table tbody tr:hover td {{
        background: rgba(255, 109, 0, 0.06) !important;
    }}
    .heater-table tbody tr:last-child td {{
        border-bottom: none !important;
    }}

    /* ── FP SIDEBAR RAIL (Combustion): thin deep-navy gradient icon rail ── */
    .stSidebar {{
        background: linear-gradient(180deg, var(--fp-navy), var(--fp-navy2)) !important;
        border-right: none !important;
    }}
    /* Thin rail WIDTH only on desktop, and ONLY while EXPANDED. On phones
       Streamlit renders the sidebar as a slide-over drawer — forcing 100px
       there would cramp it — so mobile keeps the default full-width drawer
       (the dark rail styling still applies). The [aria-expanded="true"] scope
       matters: Streamlit collapses the rail by sliding the paint off-screen
       (translateX) while the element keeps its layout box — forcing width on
       the collapsed shell left a blank 100px strip (2026-06-10 prod report). */
    @media (min-width: 768px) {{
        section[data-testid="stSidebar"][aria-expanded="true"] {{
            width: 100px !important;
            min-width: 100px !important;
        }}
    }}
    .stSidebar [data-testid="stSidebarContent"] {{
        background: transparent !important;
    }}
    .stSidebar, .stSidebar * {{
        color: var(--fp-sidebar-ink) !important;
    }}
    .stSidebar a {{
        color: var(--fp-sidebar-ink) !important;
        font-weight: 600 !important;
        font-size: 9.5px !important;
        white-space: normal !important;
        overflow: visible !important;
        word-break: break-word !important;
        overflow-wrap: anywhere !important;
    }}
    /* nav label text: Streamlit renders the label as a <p> with nowrap +
       ellipsis (built for a wide sidebar). Force it to wrap onto 2 short
       lines, centered, so it fits the thin rail. */
    .stSidebar [data-testid="stSidebarNav"] li a p,
    .stSidebar [data-testid="stSidebarNav"] li a span:not(.nav-icon),
    .stSidebar [data-testid="stSidebarNavLink"] p,
    .stSidebar [data-testid="stSidebarNavLink"] span:not(.nav-icon) {{
        white-space: normal !important;
        word-break: normal !important;
        overflow-wrap: break-word !important;
        text-overflow: clip !important;
        overflow: visible !important;
        width: 100% !important;
        max-width: 100% !important;
        text-align: center !important;
        font-size: 9px !important;
        line-height: 1.12 !important;
    }}
    .stSidebar a:hover {{
        color: #ffffff !important;
    }}
    .stSidebar [data-testid="stSidebarNav"] li {{
        margin-bottom: 2px !important;
    }}
    .stSidebar [data-testid="stSidebarNav"] li a {{
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 3px !important;
        padding: 8px 1px !important;
        border-radius: 8px !important;
        text-align: center !important;
        line-height: 1.1 !important;
        transition: background 0.2s ease !important;
    }}
    .stSidebar [data-testid="stSidebarNav"] li a:hover {{
        background: rgba(255,255,255,0.07) !important;
    }}
    .stSidebar [data-testid="stSidebarNav"] li a[aria-current="page"] {{
        background: rgba(255,109,0,0.14) !important;
        box-shadow: inset 2px 0 0 {t["primary"]} !important;
        border-radius: 8px !important;
    }}
    .stSidebar [data-testid="stSidebarNav"] li a[aria-current="page"] span {{
        color: {t["primary"]} !important;
        font-weight: 700 !important;
    }}
    /* Active nav icon glows orange (mockup .navitem.active svg). */
    .stSidebar [data-testid="stSidebarNav"] li a[aria-current="page"] .nav-icon svg {{
        stroke: {t["primary"]} !important;
        fill: none !important;
    }}
    .stSidebar [data-testid="stSidebarNav"] li a .nav-icon {{
        flex-shrink: 0;
        display: inline-flex;
    }}
    .stSidebar [data-testid="stSidebarNav"] li a .nav-icon svg {{
        width: 22px !important;
        height: 22px !important;
    }}
    .stSidebar [data-testid="stSidebarHeader"] {{
        background: transparent !important;
        flex-direction: column !important;
        align-items: center !important;
        width: 100% !important;
        height: auto !important;
        padding-top: 4px !important;
        gap: 2px !important;
    }}
    .stSidebar button[data-testid="stBaseButton-header"] {{
        color: #ffffff !important;
    }}
    .stSidebar button[data-testid="stBaseButton-header"] svg {{
        fill: #ffffff !important;
        stroke: #ffffff !important;
    }}

    /* Expanders — Combustion: flat surface, no blur. */
    div[data-testid="stExpander"] {{
        background: var(--fp-surface) !important;
        border: 1px solid {t["border"]} !important;
        border-radius: 14px !important;
        box-shadow: 0 1px 3px rgba(24,26,32,.06) !important;
    }}
    div[data-testid="stExpander"] summary {{
        color: {t["tx"]} !important;
        font-family: 'Inter', sans-serif;
    }}

    /* Metrics */
    div[data-testid="stMetric"] {{
        background: transparent;
        color: {t["tx"]} !important;
    }}
    div[data-testid="stMetric"] label {{
        color: {t["tx2"]} !important;
        font-family: var(--font-body) !important;
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

    /* Bold ALL titles — subheaders, headers, markdown bold.
       Combustion: Archivo display so headings read as instrument labels. */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    h1, h2, h3,
    [data-testid="stSubheader"],
    .stSubheader {{
        font-weight: 800 !important;
        font-family: var(--font-display) !important;
        letter-spacing: -0.01em !important;
        color: {t["tx"]} !important;
    }}
    .stMarkdown h1, h1 {{ font-size: 24px !important; }}
    .stMarkdown h2, h2, [data-testid="stSubheader"], .stSubheader {{ font-size: 18px !important; }}
    .stMarkdown h3, h3 {{ font-size: 15px !important; }}

    /* Secondary action buttons (Refresh Stats, Sync Yahoo, Refresh All Data, ...)
       — FP flat-outline style, red on hover. Was an amber 3D gradient; the
       [kind="secondary"] specificity here outranks the base FP button rule, so
       this block now carries the FP look explicitly. */
    .stButton > button[kind="secondary"],
    .stButton > button[data-testid="stBaseButton-secondary"] {{
        background: var(--fp-surface) !important;
        color: var(--fp-tx) !important;
        font-weight: 600 !important;
        border: 1px solid var(--fp-border) !important;
        border-radius: var(--fp-radius-sm) !important;
        transform: none !important;
        box-shadow: none !important;
    }}
    .stButton > button[kind="secondary"]:hover,
    .stButton > button[data-testid="stBaseButton-secondary"]:hover {{
        background: var(--fp-surface) !important;
        color: var(--fp-primary) !important;
        border-color: var(--fp-primary) !important;
        transform: none !important;
        box-shadow: none !important;
    }}
    .stButton > button[kind="secondary"]:active,
    .stButton > button[data-testid="stBaseButton-secondary"]:active {{
        transform: none !important;
        box-shadow: none !important;
    }}

    /* ── TOOLTIP (CSS-based, on title attr) ─── */
    [title] {{
        cursor: help;
    }}

    /* ── RECOMMENDATION BANNER — Combustion: flat surface, orange left rail. ── */
    .reco-banner {{
        background: var(--fp-surface) !important;
        border: 1px solid var(--fp-border) !important;
        border-left: 4px solid var(--fp-primary) !important;
        border-radius: 12px !important;
        padding: 10px 16px !important;
        margin-bottom: 12px !important;
        box-shadow: 0 1px 3px rgba(24,26,32,.06) !important;
    }}
    .reco-banner-teaser {{
        font-family: 'Inter', sans-serif !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        color: {t["tx"]} !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 8px !important;
        line-height: 1.4 !important;
    }}
    .reco-banner-detail {{
        font-family: 'Inter', sans-serif !important;
        font-size: 13px !important;
        color: {t["tx2"]} !important;
        padding-top: 8px !important;
        line-height: 1.5 !important;
        animation: slideUp 0.3s ease-out !important;
    }}

    /* ── MATCHUP TICKER — Combustion: flat surface, orange left rail. ── */
    .matchup-ticker {{
        background: var(--fp-surface) !important;
        border: 1px solid var(--fp-border) !important;
        border-left: 3px solid var(--fp-primary) !important;
        border-radius: 10px !important;
        padding: 8px 14px !important;
        margin: 4px 0 12px !important;
        display: flex !important;
        align-items: center !important;
        gap: 12px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 13px !important;
        color: {t["tx"]} !important;
        box-shadow: 0 1px 3px rgba(24,26,32,.06) !important;
        flex-wrap: wrap !important;
    }}
    .matchup-ticker-week {{
        font-family: var(--font-body) !important;
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

    /* ── CONTEXT CARD (sidebar panels) — Combustion: flat surface, no blur. ── */
    .context-card {{
        background: var(--fp-surface) !important;
        border: 1px solid var(--fp-border) !important;
        border-radius: 10px !important;
        padding: 12px 14px !important;
        margin-bottom: 8px !important;
        box-shadow: 0 1px 3px rgba(24,26,32,.06) !important;
    }}
    .context-card-title {{
        font-family: var(--font-body) !important;
        font-size: 10px !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        color: {t["tx2"]} !important;
        margin-bottom: 6px !important;
        padding-bottom: 4px !important;
        border-bottom: 1px solid {t["border"]} !important;
    }}

    /* ── CONTEXT CARD STAT ROWS ───────────── */
    .context-stat-row {{
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
        margin-bottom: 4px !important;
        padding: 2px 0 !important;
    }}
    .context-stat-label {{
        font-family: 'Inter', sans-serif !important;
        font-size: 11px !important;
        color: {t["tx2"]} !important;
        font-weight: 500 !important;
        /* Short labels ("Roster") must never break mid-word in the narrow
           sidebar cards — exempt from the global overflow-wrap:anywhere. */
        overflow-wrap: normal !important;
        word-break: normal !important;
        white-space: nowrap !important;
    }}
    .context-stat-value {{
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 12px !important;
        color: {t["tx"]} !important;
        font-weight: 600 !important;
        text-align: right !important;
    }}

    /* ── COMPACT TABLE (ESPN-style) ───────────── */
    .compact-table-wrap {{
        overflow-x: auto !important;
        overflow-y: auto !important;
        border: 1px solid var(--fp-border) !important;
        border-radius: var(--fp-radius) !important;
        background: var(--fp-surface) !important;
        box-shadow: var(--fp-shadow) !important;
        margin-bottom: 12px !important;
    }}
    .compact-table {{
        font-family: var(--font-display) !important;
        font-weight: 700 !important;
        font-size: 13px !important;
        border-collapse: collapse !important;
        white-space: nowrap !important;
        width: 100% !important;
        color: {t["tx"]} !important;
    }}
    .compact-table th {{
        background: var(--fp-surface) !important;
        color: #2c2f36 !important;
        font-family: var(--font-display) !important;
        font-weight: 800 !important;
        font-size: 11px !important;
        text-transform: uppercase !important;
        letter-spacing: .06em !important;
        padding: 8px 10px !important;
        position: sticky !important;
        top: 0 !important;
        z-index: 2 !important;
        border-bottom: 2px solid rgba(24,26,32,.18) !important;
    }}
    .compact-table td {{
        padding: 7px 10px !important;
        border-bottom: 1px solid rgba(24,26,32,.055) !important;
        font-family: var(--font-display) !important;
        font-weight: 700 !important;
        font-size: 13.5px !important;
        font-variant-numeric: tabular-nums !important;
        background: var(--fp-surface) !important;
        color: {t["tx"]} !important;
    }}
    .compact-table tr:hover td {{
        background: rgba(255, 109, 0, 0.06) !important;
    }}
    .th-hit {{
        border-bottom: 3px solid {t["primary"]} !important;
    }}
    .th-pit {{
        border-bottom: 3px solid {t["sky"]} !important;
    }}
    .col-name {{
        position: sticky !important;
        left: 0 !important;
        z-index: 3 !important;
        background: {t["card"]} !important;
        font-family: 'Inter', sans-serif !important;
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
        background: var(--fp-surface) !important;
        box-shadow: 2px 0 4px rgba(0, 0, 0, 0.06) !important;
    }}
    .compact-table tr:hover .col-name {{
        background: var(--fp-surface) !important;
    }}
    .row-start td {{
        background: rgba(45, 106, 79, 0.04) !important;
    }}
    .row-start td.col-name {{
        background: #f7fbf9 !important;
    }}
    .row-bench td {{
        background: rgba(95, 125, 156, 0.05) !important;
    }}
    .row-bench td.col-name {{
        background: #f6f8fa !important;
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

    /* ═══ COMBUSTION FINALE — machined micro-detail layer (2026-06-10) ═══ */

    /* Interaction chrome */
    * {{ scrollbar-width: thin; scrollbar-color: #c9c8c3 transparent; }}
    .stSidebar * {{ scrollbar-color: rgba(238,241,246,.28) transparent; }}
    ::selection {{ background: var(--fp-navy); color: #ffffff; }}
    .stSidebar ::selection {{ background: var(--fp-primary); color: #ffffff; }}
    :focus-visible {{ outline: 2px solid var(--fp-primary); outline-offset: 2px; border-radius: 4px; }}
    :focus:not(:focus-visible) {{ outline: none; }}

    /* Numeric law — stat columns never shimmy */
    .stat, .mono, td.num, .heater-table td, .compact-table td, .draft-board td,
    [data-testid="stMetricValue"] {{
        font-variant-numeric: tabular-nums slashed-zero;
    }}

    /* Reading polish */
    h1, h2, h3, .sec-head {{ text-wrap: balance; }}
    .stMarkdown p {{ text-wrap: pretty; }}
    h1, h2, h3 {{ scroll-margin-top: 84px; }}

    /* Gradient hairline dividers */
    .hr-fade {{ height: 1px; border: 0; margin: 18px 0;
        background: linear-gradient(90deg, transparent, var(--fp-border) 15%, var(--fp-border) 85%, transparent); }}
    .hr-heat {{ height: 1px; border: 0; margin: 18px 0;
        background: linear-gradient(90deg, var(--fp-primary), rgba(255,109,0,.25) 45%, transparent 80%); }}
    [data-testid="stDivider"] hr {{ height: 1px; border: 0;
        background: linear-gradient(90deg, transparent, var(--fp-border) 15%, var(--fp-border) 85%, transparent); }}

    /* Hero numerals — Archivo width axis + orange gradient clip (orange ONLY, never red) */
    .hero-num {{
        font-family: var(--font-display);
        font-weight: 900;
        font-stretch: 112%;
        font-variant-numeric: tabular-nums;
        letter-spacing: -0.015em;
        background: linear-gradient(180deg, #ff8a2a, #ff6d00 55%, #e85f00);
        -webkit-background-clip: text;
        background-clip: text;
        /* !important + text-fill-color so the gradient clip survives the
           global charcoal text rules (which also carry !important). */
        color: transparent !important;
        -webkit-text-fill-color: transparent !important;
    }}

    /* Chip system — embossed token look; .dot-live strictly for live indicators */
    .chip {{ display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px;
        font: 600 11px/1.6 var(--font-body); letter-spacing: .05em; text-transform: uppercase;
        background: var(--fp-surface); border: 1px solid var(--fp-border); color: var(--fp-tx-muted);
        box-shadow: inset 0 0 0 1px rgba(255,255,255,.6), 0 1px 2px rgba(24,26,32,.05);
        transition: border-color var(--dur-1) var(--ease-snap), transform var(--dur-1) var(--ease-snap); }}
    .chip:hover {{ border-color: var(--fp-primary); transform: translateY(-1px); }}
    .chip.hot {{ color: var(--fp-primary); border-color: rgba(255,109,0,.45); background: rgba(255,109,0,.06); }}
    .chip.cold {{ color: var(--fp-cold); border-color: rgba(95,125,156,.45); background: rgba(95,125,156,.07); }}
    @keyframes dotPulse {{ 50% {{ box-shadow: 0 0 0 4px rgba(255,109,0,0); }} }}
    .chip .dot-live {{ width: 6px; height: 6px; border-radius: 50%; background: var(--fp-primary);
        box-shadow: 0 0 0 0 rgba(255,109,0,.45); animation: dotPulse 2.2s ease-out infinite; }}

    /* Conic heat wash — opt-in second light layer for hero panels */
    .instr-panel.heat::before {{
        background:
            radial-gradient(120% 80% at 100% 0%, rgba(255,109,0,.05), transparent 42%),
            conic-gradient(from 210deg at 108% -8%, rgba(255,154,60,.07), transparent 28%);
    }}

    /* Empty state — instrument-styled no-data panel */
    .empty-state {{
        position: relative;
        border: 1px dashed var(--fp-border);
        border-radius: 14px;
        padding: 34px 26px;
        margin: 8px 0 14px;
        text-align: center;
        background:
            radial-gradient(rgba(24,26,32,.03) 1px, transparent 1.3px) 0 0 / 20px 20px,
            var(--fp-surface);
    }}
    .empty-state .es-icon svg {{ width: 26px; height: 26px; stroke: var(--fp-primary); margin: 0 auto 8px; display: block; }}
    .empty-state .es-title {{ font-family: var(--font-display); font-weight: 800; font-size: 15px; color: var(--fp-tx); margin-bottom: 4px; }}
    .empty-state .es-body {{ font-family: var(--font-body); font-size: 12.5px; color: var(--fp-tx-muted); max-width: 460px; margin: 0 auto; line-height: 1.55; }}

    /* Type-scale utilities — stop minting inline sizes */
    .t-eyebrow {{ font-family: var(--font-body); font-weight: 700; font-size: 10px; letter-spacing: .28em; text-transform: uppercase; color: var(--fp-tx-muted); }}
    .t-fig {{ font-family: var(--font-mono); font-size: 10px; letter-spacing: .12em; color: var(--fp-tx-muted); }}
    .t-label {{ font-family: var(--font-body); font-weight: 600; font-size: 11px; letter-spacing: .05em; text-transform: uppercase; color: var(--fp-tx-muted); }}
    .t-caption {{ font-family: var(--font-body); font-size: 12px; color: var(--fp-tx-muted); }}

    /* Table micro-interaction — inset accent bar on row hover (no layout shift) */
    .heater-table tbody tr, .compact-table tbody tr, .draft-board tbody tr {{
        transition: background-color var(--dur-1) var(--ease-snap), box-shadow var(--dur-1) var(--ease-snap);
    }}
    .heater-table tbody tr:hover, .compact-table tbody tr:hover {{
        box-shadow: inset 3px 0 0 var(--fp-primary);
    }}

    /* Sticky header depth cue — hairline + soft scroll shadow */
    .heater-table thead th, .compact-table thead th {{
        box-shadow: 0 1px 0 var(--fp-border), 0 4px 10px rgba(24,26,32,.04);
    }}

    /* Animated tab underline — the highlight glides between tabs */
    .stTabs [data-baseweb="tab-highlight"] {{ background-color: var(--fp-primary); height: 2px; border-radius: 2px;
        transition: all 240ms cubic-bezier(.4, 0, .2, 1); }}
    .stTabs [data-baseweb="tab-border"] {{ background-color: var(--fp-divider); height: 1px; }}

    /* Expander chrome */
    [data-testid="stExpander"] summary {{ transition: background-color var(--dur-1) var(--ease-snap), color var(--dur-1) var(--ease-snap); }}
    [data-testid="stExpander"] summary:hover {{ background: #faf9f6; color: var(--fp-primary); }}
    [data-testid="stExpanderToggleIcon"] {{ transition: transform 180ms var(--ease-snap); }}

    /* Dialog chrome — player dossier */
    div[data-testid="stDialog"] div[role="dialog"] {{
        position: relative;
        width: min(92vw, 1100px);
        border-radius: 16px;
        border: 1px solid var(--fp-border);
        box-shadow: 0 24px 80px rgba(16,33,58,.38), inset 0 1px 0 rgba(255,255,255,.9);
    }}
    div[data-testid="stDialog"] div[role="dialog"]::before {{
        content: ""; position: absolute; top: 0; left: 24px; right: 24px; height: 2px;
        background: linear-gradient(90deg, var(--fp-primary), transparent 70%);
        pointer-events: none;
    }}

    /* Portal chrome — tooltips + toasts mount on <body>, outside .stApp */
    div[data-baseweb="tooltip"], [data-testid="stTooltipContent"] {{
        background: var(--fp-navy) !important; color: #eef1f6 !important;
        border-radius: 8px !important;
        font: 500 12.5px/1.45 var(--font-body) !important;
        box-shadow: 0 8px 24px rgba(16,33,58,.3) !important;
    }}
    [data-testid="stToast"] {{
        background: var(--fp-navy) !important; color: #eef1f6 !important;
        border-radius: 10px !important; border-left: 3px solid var(--fp-primary) !important;
        box-shadow: 0 12px 32px rgba(16,33,58,.35) !important;
    }}
    [data-testid="stToast"] * {{ color: #eef1f6 !important; }}

    /* Metric self-tinting via :has() */
    [data-testid="stMetric"]:has([data-testid="stMetricDeltaIcon-Up"]) {{
        background: rgba(31,157,107,.05); border-radius: 10px; padding: 6px 10px; }}
    [data-testid="stMetric"]:has([data-testid="stMetricDeltaIcon-Down"]) {{
        background: rgba(224,73,47,.04); border-radius: 10px; padding: 6px 10px; }}

    /* Glide grid — selection ring + hairline refinement */
    :root {{
        --gdg-accent-color: {t["primary"]} !important;
        --gdg-accent-light: rgba(255,109,0,.10) !important;
        --gdg-horizontal-border-color: rgba(24,26,32,.07) !important;
        --gdg-text-medium: {t["tx_muted"]} !important;
        --gdg-cell-horizontal-padding: 10px !important;
        --gdg-rounding-radius: 10px !important;
        --gdg-bg-search-result: rgba(255,174,66,.22) !important;
    }}

    /* Sidebar machining — inner hairline + grouped-nav section headers */
    .stSidebar {{ box-shadow: inset -1px 0 0 rgba(255,255,255,.06); }}
    [data-testid="stNavSectionHeader"] {{
        font-family: var(--font-body) !important; font-size: 9px !important; font-weight: 700 !important;
        letter-spacing: .14em !important; text-transform: uppercase !important;
        color: rgba(238,241,246,.55) !important; margin-top: 10px !important;
        justify-content: center !important;
    }}

    /* Shimmer — two-tone on-brand + no-motion fallback lives in the block below */
    .shimmer {{ background: linear-gradient(100deg, #efeeea 40%, #f7f6f3 50%, #efeeea 60%) 0 0 / 200% 100%; }}

    /* Reduced motion — keep as the LAST rules in the sheet */
    @media (prefers-reduced-motion: reduce) {{
        *, *::before, *::after {{
            animation-duration: .01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: .01ms !important;
            scroll-behavior: auto !important;
        }}
        .shimmer {{ animation: none; background: #efeeea; }}
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

    /* ── FP COMPONENTS (revamp task 3) — flat cards, clean buttons,
          underline tabs, rounded inputs. Placed late to win the cascade. ── */
    .glass, .fp-card, .metric-card, div[data-testid="stMetric"] {{
        background: var(--fp-surface) !important;
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
        border: 1px solid var(--fp-border) !important;
        border-radius: var(--fp-radius) !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.9), var(--fp-shadow) !important;
        padding: 20px !important;
    }}
    .glass:hover, .fp-card:hover {{
        transform: translateY(-1px) !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.9), 0 4px 14px rgba(16,33,58,.10) !important;
    }}
    /* Buttons (Combustion): flat white-outline base; solid-orange gradient
       primary with a subtle orange glow. Matches mockup .btn / .btn.primary. */
    .stButton > button {{
        border-radius: var(--fp-radius-sm) !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        border: 1px solid var(--fp-border) !important;
        background: var(--fp-surface) !important;
        color: var(--fp-tx) !important;
        box-shadow: 0 1px 2px rgba(24,26,32,.05) !important;
        transform: none !important;
        transition: background .15s ease, border-color .15s ease, color .15s ease, box-shadow .15s ease !important;
    }}
    .stButton > button:hover {{
        border-color: var(--fp-primary) !important;
        color: var(--fp-primary) !important;
        transform: none !important;
        box-shadow: 0 1px 2px rgba(24,26,32,.05) !important;
    }}
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {{
        background: linear-gradient(135deg, var(--fp-primary), var(--fp-ember)) !important;
        color: var(--fp-ink) !important;
        font-weight: 700 !important;
        border-color: transparent !important;
        box-shadow: 0 4px 16px rgba(255,109,0,.30) !important;
        transform: none !important;
    }}
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {{
        background: linear-gradient(135deg, var(--fp-flame), var(--fp-primary)) !important;
        border-color: transparent !important;
        color: var(--fp-ink) !important;
        transform: none !important;
        box-shadow: 0 6px 22px rgba(255,109,0,.42) !important;
    }}
    /* Tabs: FP underline strip (no glass pill) */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px !important;
        background: transparent !important;
        backdrop-filter: none !important;
        border: none !important;
        border-bottom: 1px solid var(--fp-divider) !important;
        border-radius: 0 !important;
        padding: 0 !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent !important;
        color: var(--fp-tx-muted) !important;
        font-weight: 600 !important;
        border-radius: 0 !important;
        padding: 8px 14px !important;
        box-shadow: none !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: transparent !important;
        color: var(--fp-primary) !important;
        box-shadow: inset 0 -2px 0 var(--fp-primary) !important;
    }}
    /* Inputs: FP rounded-sm */
    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] > div {{
        border-radius: var(--fp-radius-sm) !important;
        border-color: var(--fp-border) !important;
    }}
    /* Roomier top spacing */
    .block-container {{ padding-top: 1.5rem !important; }}

    /* ── Calm prose: no all-caps on prose headers/names/actions.
          Placed last so it wins the cascade over earlier uppercase rules.
          Micro-labels (position codes, category codes) keep their casing. ── */
    .cmd-left, .your-turn, .waiting,
    .hero .p-name, .alt .a-name, .pc-name,
    .verdict-banner .verdict-text,
    .stButton > button,
    .stTabs [data-baseweb="tab"],
    div[data-testid="stMetric"] label {{
        text-transform: none !important;
        letter-spacing: normal !important;
    }}

    /* ════════════════════════════════════════════════════════════════
       FONT-LOCK — "these fonts only": every character in the app must
       render in Archivo (--font-display), Inter (--font-body), or IBM
       Plex Mono (--font-mono). Streamlit ships its own font stacks on
       built-in widgets (BaseWeb select/input/tabs, metric, expander,
       dataframe chrome, etc.) that otherwise escape the base .stApp
       font. This block forces Inter as the body family on every widget
       container + its descendants; headings/.sec-head/.phead
       stay Archivo and stat figures/.mono/tables keep their existing
       Archivo/mono treatment (those rules are !important + later or more
       specific, so they win over this body default). Glide grid uses the
       --gdg-* Archivo tokens and Plotly uses Inter — both left alone.
       ════════════════════════════════════════════════════════════════ */
    .stButton, .stDownloadButton,
    [data-baseweb="select"], [data-baseweb="input"], [data-baseweb="tab"],
    [data-testid="stWidgetLabel"], [data-testid="stMarkdownContainer"],
    [data-testid="stCaptionContainer"],
    .stRadio, .stCheckbox, .stSelectbox, .stMultiSelect,
    .stTextInput, .stNumberInput, .stSlider,
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
    [data-testid="stExpander"], .stTabs, .stDataFrame, [data-testid="stTable"],
    .stButton *, .stDownloadButton *,
    [data-baseweb="select"] *, [data-baseweb="input"] *, [data-baseweb="tab"] *,
    [data-testid="stWidgetLabel"] *, [data-testid="stMarkdownContainer"] *,
    [data-testid="stCaptionContainer"] *,
    .stRadio *, .stCheckbox *, .stSelectbox *, .stMultiSelect *,
    .stTextInput *, .stNumberInput *, .stSlider *,
    [data-testid="stExpander"] *, .stTabs *, [data-testid="stTable"] * {{
        font-family: var(--font-body) !important;
    }}
    /* Headings + display surfaces stay on Archivo even inside the widget
       containers font-locked above (re-assert so the body default can't
       leak into a heading rendered via st.markdown/st.subheader). */
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4,
    .sec-head, .phead, .phead-title,
    .heater-h1, [data-testid="stMetricValue"] {{
        font-family: var(--font-display) !important;
    }}
    /* Mono figures stay on IBM Plex Mono. */
    .stat, .mono, td.num, .fig,
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
        font-family: var(--font-mono) !important;
    }}
    /* Material Symbols ligature spans are the ONE exception to the font
       lock: Streamlit renders widget glyphs (expander chevron, nav
       expand_more, dialog close) as icon-font ligature TEXT — forcing
       Inter on them shows the literal ligature name ("keyboard_arrow_right").
       Re-assert the icon font AFTER the lock so no container rule can
       break it. */
    span[data-testid="stIconMaterial"] {{
        font-family: "Material Symbols Rounded" !important;
        letter-spacing: normal !important;
        text-transform: none !important;
    }}

    /* ════════════════════════════════════════════════════════════════
       CLICKABLE / OPENS-A-WINDOW TEXT = orange + underline.
       Convention: any hyperlink or text that opens a window/dialog reads
       as interactive — orange (--fp-primary), underlined, pointer cursor,
       brightening to --fp-flame on hover. SCOPED TO MAIN CONTENT ONLY:
       prefixed with the main block container and explicitly NOT inside
       .stSidebar, so the sidebar nav links stay bone-on-navy. Buttons,
       download buttons, and st.page_link nav are deliberately EXCLUDED
       (they have their own flat/gradient button styling) via :not() and
       by only matching bare <a>/.heater-link/.clickable, never .stButton.
       ════════════════════════════════════════════════════════════════ */
    [data-testid="stMainBlockContainer"] [data-testid="stMarkdownContainer"] a,
    .stMain [data-testid="stMarkdownContainer"] a,
    [data-testid="stMainBlockContainer"] .stMarkdown a,
    .stMain .stMarkdown a,
    .stMain .heater-link, .stMain a.clickable,
    [data-testid="stMainBlockContainer"] .heater-link,
    [data-testid="stMainBlockContainer"] a.clickable {{
        color: var(--fp-primary) !important;
        text-decoration: underline !important;
        text-underline-offset: 2px !important;
        cursor: pointer !important;
    }}
    [data-testid="stMainBlockContainer"] [data-testid="stMarkdownContainer"] a:hover,
    .stMain [data-testid="stMarkdownContainer"] a:hover,
    [data-testid="stMainBlockContainer"] .stMarkdown a:hover,
    .stMain .stMarkdown a:hover,
    .stMain .heater-link:hover, .stMain a.clickable:hover,
    [data-testid="stMainBlockContainer"] .heater-link:hover,
    [data-testid="stMainBlockContainer"] a.clickable:hover {{
        color: var(--fp-flame) !important;
    }}
    /* Reusable helper classes (use anywhere a span/link should read as a
       link or window-opener): <a class="heater-link"> / class="clickable".
       The orange + underline convention is baked into the helper itself so it
       reads as interactive wherever it is dropped, not only when nested under
       the main block container. */
    .heater-link, .clickable {{
        color: var(--fp-primary);
        text-decoration: underline;
        text-underline-offset: 2px;
        cursor: pointer;
    }}
    /* Roster player-name link (build_roster_table_html emits
       <a href="?player=..."> wrapping the player cell): the NAME reads as
       "click to open dossier" — orange, underline on hover, pointer. The
       team-abbr sub-line below it is intentionally left muted/mono. */
    .rtbl td.ph a {{ cursor: pointer !important; }}
    .rtbl td.ph a b {{ color: var(--fp-primary) !important; }}
    .rtbl td.ph a:hover b {{ text-decoration: underline !important; text-underline-offset: 2px !important; }}
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Rename sidebar "app" → "Connect League" and inject logo + branding via JS
    import streamlit.components.v1 as components

    _logo_b64 = _heater_logo_b64()
    if _logo_b64:
        _logo_inner = (
            '<img src="data:image/png;base64,' + _logo_b64 + '" alt="HEATER" '
            'style="width:100%;height:auto;display:block;margin:0;border-radius:6px;">'
        )
    else:
        _logo_inner = (
            '<span style="font-family:Archivo,sans-serif;font-style:italic;font-weight:700;'
            'color:#fff;font-size:18px;letter-spacing:2px;">HEATER</span>'
        )

    # Global navy top chrome bar on every page — matches the sidebar fill (#112744)
    # and thickness, framing the app. Content + sidebar logo are offset below it.
    st.markdown(
        "<style>"
        ".stApp::before{content:'';position:fixed;top:0;left:0;right:0;height:90px;"
        "background:#112744;z-index:9998;}"
        'div[data-testid="stMainBlockContainer"]{padding-top:98px!important;}'
        'section[data-testid="stSidebar"] [data-testid="stSidebarHeader"]{margin-top:90px!important;}'
        # widen the sidebar content so the logo fills the rail (bigger logo)
        'section[data-testid="stSidebar"] [data-testid="stSidebarContent"]{padding-left:5px!important;padding-right:5px!important;}'
        "</style>",
        unsafe_allow_html=True,
    )

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
                'Player Databank': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>',
                'Draft Simulator': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>',
                'Trade Analyzer': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/></svg>',
                'Player Compare': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/><line x1="3" y1="20" x2="21" y2="20"/></svg>',
                'Free Agents': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="8" y1="11" x2="14" y2="11"/><line x1="11" y1="8" x2="11" y2="14"/></svg>',
                'Lineup': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
                'Line-up Optimizer': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
                'Closer Monitor': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="7"/><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"/></svg>',
                'Pitcher Streaming': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.59 4.59A2 2 0 1 1 11 8H2"/><path d="M12.59 19.41A2 2 0 1 0 14 16H2"/><path d="M17.73 7.73A2.5 2.5 0 1 1 19.5 12H2"/></svg>',
                'Standings': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>',
                'Leaders': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>',
                'Trade Finder': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>',
                'Matchup Planner': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>',
                'League Standings': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M7 17V9"/><path d="M11 17V5"/><path d="M15 17v-4"/><path d="M19 17v-8"/></svg>',
                'Weekly Dashboard': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="8" height="9" rx="1"/><rect x="13" y="3" width="8" height="5" rx="1"/><rect x="13" y="10" width="8" height="11" rx="1"/><rect x="3" y="14" width="8" height="7" rx="1"/></svg>',
                'Trade Values': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>',
                'Waiver Wire': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="8.5" cy="7" r="4"/><line x1="20" y1="8" x2="20" y2="14"/><line x1="23" y1="11" x2="17" y2="11"/></svg>',
                'Category Tracker': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 11l3 3L22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg>',
                'Trends': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
                'Playoff Odds': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.21 15.89A10 10 0 1 1 8 2.83"/><path d="M22 12A10 10 0 0 0 12 2v10z"/></svg>',
                'Bullpen': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
                'Punt Analyzer': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="4.93" y1="4.93" x2="19.07" y2="19.07"/></svg>',
                'Weekly Recap': '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>'
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
                logoDiv.style.cssText = 'display:flex;flex-direction:column;align-items:center;width:100%;padding:2px 0 4px;';
                logoDiv.innerHTML = '__LOGO_INNER__';
                header.insertBefore(logoDiv, header.firstChild);
            }
            // Tighten the sidebar header so the logo sits flush + fills the rail width.
            if (header) { header.style.padding = '6px 2px 2px'; }

            // Collapse zero-height helper blocks (injected <style>/<link>, height=0
            // component iframes, streamlit-float markers) so the main column's 16px
            // flex row-gap doesn't stack into a big empty band atop every page.
            function collapseHelpers() {
                const main = parent.document.querySelector('section[data-testid="stMain"]');
                if (!main) return;
                main.querySelectorAll('[data-testid="stElementContainer"]').forEach(function(el){
                    if (el.getAttribute('data-heater-collapsed')) return;
                    if (el.getBoundingClientRect().height > 0.5) return;  // never touch visible content
                    const md = el.querySelector('[data-testid="stMarkdownContainer"]');
                    const isHelper = el.querySelector('iframe')
                        || el.querySelector('.float, .elim')
                        || (md && (md.querySelector(':scope > style, :scope > link') || md.textContent.trim() === ''));
                    if (isHelper) { el.style.display = 'none'; el.setAttribute('data-heater-collapsed', '1'); }
                });
            }
            collapseHelpers();
            [250, 600, 1200, 2500].forEach(function(ms){ setTimeout(collapseHelpers, ms); });

            var firstLink = nav.querySelector('li:first-child a');
            if (!firstLink || !firstLink.querySelector('.nav-icon')) { setTimeout(setup, 200); }
        })();
        </script>""".replace("__LOGO_INNER__", _logo_inner),
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

    cat_keys = list(ALL_CATEGORIES)
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
    _lc = _LC_Class()
    rate_fmt = set(_lc.rate_stats)
    # Inverse stats: lower is better
    inverse_cats = set(_lc.inverse_stats)

    num_teams = len(all_totals)

    rows_html = ""
    for cat in cat_keys:
        my_val = user_totals.get(cat, 0)

        # Compute rank (1 = best)
        if cat in inverse_cats:
            # Lower is better — count teams with strictly lower value
            # But if user has 0 IP, pitching inverse stats are meaningless
            # — treat as worst rank instead of best.
            # 2026-05-19 D4: use LeagueConfig.inverse_stats — was {"ERA", "WHIP", "L"}.
            from src.valuation import LeagueConfig

            pitching_inverse = LeagueConfig().inverse_stats
            if cat in pitching_inverse and user_totals.get("IP", user_totals.get("ip")) == 0:
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
        f'font-family:Inter,sans-serif;font-size:14px;color:{T["tx"]};">'
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


# ── Combustion detail renderers (Phase B, 2026-06-08) ──────────────
# Reusable HTML-string builders + thin st.markdown wrappers that mirror the
# gold-standard mockups (docs/design/mockup-myteam.html + mockup-player-popup.html).
# All colors come from the --fp-* custom properties emitted by inject_custom_css()
# (or THEME), never raw brand hex. Each builder returns a single self-contained root
# element (Streamlit auto-closes tags, so no dangling markup). No emoji — icons are
# inline SVG / CSS elsewhere.


def build_eyebrow_html(text: str) -> str:
    """Build the mono uppercase letter-spaced eyebrow label (mockup ``.eyebrow``).

    Brightened to ``var(--fp-tx-muted)`` per the Combustion spec. Use it for the
    small label that sits above a section/page title.
    """
    safe = _html.escape(str(text))
    style = (
        "font-family:var(--font-body);font-weight:700;font-size:10px;"
        "letter-spacing:.28em;text-transform:uppercase;color:var(--fp-tx-muted);"
    )
    return f'<span class="eyebrow" style="{style}">{safe}</span>'


def render_eyebrow(text: str) -> None:
    """``st.markdown`` wrapper around :func:`build_eyebrow_html`."""
    st.markdown(build_eyebrow_html(text), unsafe_allow_html=True)


def render_page_header(
    title: str,
    *,
    eyebrow: str = "",
    fig: str = "",
    actions_html: str = "",
) -> None:
    """Render the Combustion page header (mockup ``.phead``).

    Replaces the old navy ``.page-title`` pill. Renders:
    - an optional mono ``eyebrow`` label + ``fig`` crumb (``FIG.NN — …``),
    - the page title in Archivo-900 with an orange ``.`` accent + a 2px
      orange→transparent underline rule,
    - an optional right-aligned ``actions_html`` block (caller-supplied markup,
      e.g. buttons / a live pill) embedded verbatim.

    Args:
        title: Page title text (escaped).
        eyebrow: Optional small uppercase eyebrow label (escaped).
        fig: Optional mono crumb, e.g. ``"FIG.01 — ROSTER CONTROL"`` (escaped).
        actions_html: Optional pre-built HTML for the right side (NOT escaped).
    """
    safe_title = _html.escape(str(title))

    crumb_html = ""
    if eyebrow or fig:
        parts = []
        if eyebrow:
            parts.append(build_eyebrow_html(eyebrow))
        if fig:
            safe_fig = _html.escape(str(fig))
            parts.append(
                f'<span class="fig" style="font-family:var(--font-mono);font-weight:500;'
                f"font-size:10px;letter-spacing:.12em;color:var(--fp-tx-muted);"
                f'">/ {safe_fig}</span>'
            )
        crumb_html = (
            '<div class="phead-crumb" style="display:flex;align-items:center;gap:10px;'
            'margin-bottom:9px;">' + "".join(parts) + "</div>"
        )

    title_html = (
        '<h1 class="phead-title" style="font-family:var(--font-display);font-weight:900;'
        "font-size:46px;letter-spacing:-.02em;line-height:.9;color:var(--fp-tx);margin:0;"
        f'">{safe_title}<span style="color:var(--fp-primary);">.</span></h1>'
    )

    actions_block = (
        f'<div class="phead-actions" style="display:flex;align-items:center;gap:12px;">{actions_html}</div>'
        if actions_html
        else ""
    )

    html = (
        '<div class="phead" style="display:flex;align-items:flex-end;'
        "justify-content:space-between;border-bottom:1px solid var(--fp-divider);"
        'padding-bottom:18px;position:relative;margin-bottom:8px;">'
        '<div class="phead-titlewrap" style="display:flex;flex-direction:column;gap:9px;">'
        f"{crumb_html}{title_html}"
        "</div>"
        f"{actions_block}"
        # orange underline accent rule (mockup .phead::after)
        '<span style="position:absolute;left:0;bottom:-1px;width:120px;height:2px;'
        'background:linear-gradient(90deg,var(--fp-primary),transparent);"></span>'
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def build_panel_html(
    title: str,
    body_html: str,
    *,
    fig_label: str = "",
    accent: str = "top",
) -> str:
    """Build a full ``.instr-panel`` instrument card (mockup ``.panel``).

    Renders four corner ticks (``.pcorner``), an Archivo-800 uppercase header
    with a 3px orange ``.accent`` bar + optional right ``fig_label``, then the
    caller's ``body_html`` (embedded verbatim — pre-built HTML).

    Args:
        title: Panel header text (escaped).
        body_html: Pre-built body markup (NOT escaped).
        fig_label: Optional mono label on the right of the header (escaped).
        accent: ``"top"`` (default) or ``"left"`` — selects the
            ``.instr-panel`` accent-rule variant. Any other value = no accent rule.

    Returns:
        Self-contained HTML string (single ``.instr-panel`` root element).
    """
    safe_title = _html.escape(str(title))
    accent_cls = {"top": " accent-top", "left": " accent-left"}.get(accent, "")

    corners = (
        '<span class="pcorner tl"></span>'
        '<span class="pcorner tr"></span>'
        '<span class="pcorner bl"></span>'
        '<span class="pcorner br"></span>'
    )

    fig_html = ""
    if fig_label:
        safe_fig = _html.escape(str(fig_label))
        fig_html = (
            '<span class="fig" style="font-family:var(--font-mono);font-weight:500;'
            "font-size:10px;letter-spacing:.12em;color:var(--fp-tx-muted);"
            f'">{safe_fig}</span>'
        )

    header = (
        '<div class="phdr" style="display:flex;align-items:center;'
        "justify-content:space-between;margin-bottom:16px;padding-bottom:13px;"
        'border-bottom:1px solid var(--fp-divider);">'
        '<div class="lhs" style="display:flex;align-items:center;gap:11px;">'
        # 3px orange accent bar (mockup .phdr .accent)
        '<span class="accent" style="width:3px;height:16px;background:var(--fp-primary);'
        'box-shadow:0 0 8px rgba(255,109,0,.5);border-radius:2px;"></span>'
        '<h3 style="font-family:var(--font-display);font-weight:800;font-size:15px;'
        "letter-spacing:.05em;text-transform:uppercase;color:var(--fp-tx);margin:0;"
        f'">{safe_title}</h3>'
        "</div>"
        f"{fig_html}"
        "</div>"
    )

    return f'<div class="instr-panel{accent_cls}">{corners}{header}<div class="panel-body">{body_html}</div></div>'


def render_panel(
    title: str,
    body_html: str,
    *,
    fig_label: str = "",
    accent: str = "top",
) -> None:
    """``st.markdown`` wrapper around :func:`build_panel_html`."""
    st.markdown(
        build_panel_html(title, body_html, fig_label=fig_label, accent=accent),
        unsafe_allow_html=True,
    )


def build_heatbar_html(pct: float, *, win: bool | None = None) -> str:
    """Build a mockup ``.cb``/``.cat`` heat bar: a track with a percentage fill.

    Orange gradient when ``win`` is truthy (high/winning), steel
    ``var(--fp-cold)`` when ``win`` is falsy (low/losing). When ``win`` is
    ``None`` the orange↔steel split is inferred from the value (``>= 50`` = hot).

    Args:
        pct: Fill percentage (clamped to ``[0, 100]``).
        win: Optional explicit win/high flag.

    Returns:
        Self-contained HTML string (single ``.cb`` root element).
    """
    try:
        width = float(pct)
    except (TypeError, ValueError):
        width = 0.0
    width = max(0.0, min(100.0, width))
    # Trim trailing ``.0`` so integer inputs read cleanly (62 not 62.0).
    width_str = f"{width:g}"

    is_hot = (width >= 50.0) if win is None else bool(win)
    fill_bg = "linear-gradient(90deg,var(--fp-ember),var(--fp-primary),var(--fp-flame))" if is_hot else "var(--fp-cold)"

    return (
        '<div class="cb" style="height:4px;border-radius:3px;'
        'background:rgba(24,26,32,.10);overflow:hidden;">'
        f'<i style="display:block;height:100%;border-radius:3px;width:{width_str}%;'
        f'background:{fill_bg};"></i></div>'
    )


def build_sparkline_html(values: list[float], *, tone: str = "hot") -> str:
    """Build a mockup ``.spark`` sparkline: a flex row of bars scaled to ``max``.

    Orange gradient for ``tone="hot"``, steel ``var(--fp-cold)`` for ``"cold"``.
    Guards empty / all-zero input (no divide-by-zero; renders the track only or
    flat bars).

    Args:
        values: Numeric series. Each value becomes one bar.
        tone: ``"hot"`` (orange) or ``"cold"`` (steel).

    Returns:
        Self-contained HTML string (single ``.spark`` root element).
    """
    nums: list[float] = []
    for v in values or []:
        try:
            f = float(v)
        except (TypeError, ValueError):
            f = 0.0
        if _math.isnan(f) or _math.isinf(f):
            f = 0.0
        nums.append(f)

    bar_bg = "linear-gradient(180deg,var(--fp-flame),var(--fp-ember))" if tone == "hot" else "var(--fp-cold)"

    peak = max((abs(n) for n in nums), default=0.0)
    bars = ""
    for n in nums:
        # All-zero (peak == 0) → a thin flat 8% bar so the sparkline stays visible.
        h = (abs(n) / peak * 100.0) if peak > 0 else 8.0
        h = max(0.0, min(100.0, h))
        bars += f'<i style="flex:1;border-radius:1px;height:{h:g}%;background:{bar_bg};"></i>'

    return f'<div class="spark" style="display:flex;align-items:flex-end;gap:2px;height:22px;">{bars}</div>'


def build_stat_readout_html(label, value, *, accent: bool = False, sub=None) -> str:
    """Build a mockup ``.stat`` readout: small uppercase label + big figure.

    The figure uses the Archivo display font with tabular numerals; it is colored
    orange (``var(--fp-primary)``) when ``accent`` is set, else the standard text
    token. An optional ``sub`` line renders muted beneath the value.

    Args:
        label: Small uppercase key (escaped).
        value: Big figure value (escaped; coerced to ``str``).
        accent: When True, color the figure orange.
        sub: Optional muted sub-line (escaped).

    Returns:
        Self-contained HTML string (single ``.stat`` root element).
    """
    safe_label = _html.escape(str(label))
    safe_value = _html.escape(str(value))
    value_color = "var(--fp-primary)" if accent else "var(--fp-tx)"

    sub_html = ""
    if sub is not None and str(sub) != "":
        safe_sub = _html.escape(str(sub))
        sub_html = (
            '<span class="stat-sub" style="font-family:var(--font-mono);font-size:10px;'
            f'color:var(--fp-tx-subtle);letter-spacing:.04em;">{safe_sub}</span>'
        )

    return (
        '<div class="stat" style="display:flex;flex-direction:column;gap:3px;">'
        '<span class="stat-k" style="font-size:9px;font-weight:800;letter-spacing:.2em;'
        f'text-transform:uppercase;color:var(--fp-tx-muted);">{safe_label}</span>'
        '<span class="stat-v" style="font-family:var(--font-display);font-weight:800;'
        f"font-size:20px;font-variant-numeric:tabular-nums;letter-spacing:-.01em;"
        f'color:{value_color};">{safe_value}</span>'
        f"{sub_html}"
        "</div>"
    )


# ── Player Dossier (Combustion redesign) — pure HTML builders ──────────
# These mirror the gold-standard mockup docs/design/mockup-player-popup.html
# (.mhead header band, .glog game-log table, .upc upcoming/projection cards).
# Kept pure (no Streamlit) so they are unit-testable; show_player_card_dialog
# assembles the data and calls them.

# Per-player-type game-log column spec: (header label, row-dict key).
# The leading Date/Opp/Res columns are rendered separately (they have logos
# + badges); these are the numeric stat columns.
_DOSSIER_HITTER_GLOG_COLS = [("AB", "ab"), ("H", "h"), ("HR", "hr"), ("RBI", "rbi"), ("AVG", "avg")]
_DOSSIER_PITCHER_GLOG_COLS = [("IP", "ip"), ("H", "h_allowed"), ("ER", "er"), ("K", "k"), ("ERA", "era")]

# Per-player-type projection fields shown on each upcoming card: (label, key).
_DOSSIER_HITTER_PROJ_FIELDS = [("H", "h"), ("HR", "hr"), ("RBI", "rbi"), ("R", "r")]
_DOSSIER_PITCHER_PROJ_FIELDS = [("K", "k"), ("ERA", "era"), ("IP", "ip"), ("W", "w")]


def _dossier_fmt(value, key: str) -> str:
    """Format a single game-log / projection cell for display.

    Rate stats (avg/obp → 3dp, era/whip → 2dp) use their canonical precision;
    IP shows 1dp; everything else is an integer-ish figure. ``None``/NaN → "—".
    """
    if value is None:
        return "—"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return _html.escape(str(value))
    if _math.isnan(f) or _math.isinf(f):
        return "—"
    k = (key or "").lower()
    if k in ("avg", "obp", "avg_calc", "obp_calc"):
        return f"{f:.3f}"
    if k in ("era", "whip", "era_calc", "whip_calc"):
        return f"{f:.2f}"
    if k in ("ip",):
        return f"{f:.1f}"
    # Counting figure — drop trailing ".0".
    if f == int(f):
        return str(int(f))
    return f"{f:.1f}"


def build_game_log_html(rows: list[dict], is_hitter: bool) -> str:
    """Build the ``.glog`` game-log table (mockup: Game Log — Last 10).

    Args:
        rows: Newest-first per-game dicts. Recognized keys:
            ``date`` (display string), ``opp`` (opponent abbr, optional),
            ``home`` (bool — True = "vs", False = "@"), ``result`` ("W"/"L" or
            ``None``), ``score`` (e.g. "6–3", optional), plus the per-game stat
            keys for the player type (ab/h/hr/rbi/avg or ip/h_allowed/er/k/era),
            ``hot`` (bool — tint the row), and ``form_pct`` (0–100 form bar).
        is_hitter: Selects the hitter vs pitcher column set.

    Returns:
        Self-contained HTML string (single ``<table class="glog">`` element).
        Empty / falsy ``rows`` yields a graceful empty-state ``<div>``.
    """
    cols = _DOSSIER_HITTER_GLOG_COLS if is_hitter else _DOSSIER_PITCHER_GLOG_COLS

    if not rows:
        return (
            '<div class="glog-empty" style="padding:18px 4px;font-family:var(--font-body);'
            'font-size:13px;color:var(--fp-tx-muted);text-align:center;">'
            "No game log available yet this season.</div>"
        )

    # Header row.
    head_cells = '<th class="l">Date</th><th class="l">Opp</th><th>Res</th>'
    head_cells += "".join(f"<th>{_html.escape(lbl)}</th>" for lbl, _ in cols)
    head_cells += '<th class="l">Form</th>'

    body = ""
    for r in rows:
        row_cls = ' class="hot"' if r.get("hot") else ""
        date_disp = _html.escape(str(r.get("date", "") or ""))

        # Opponent cell — logo (if abbr resolvable) + "vs"/"@" prefix.
        opp = r.get("opp")
        if opp:
            opp_abbr = _html.escape(str(opp).upper())
            prefix = "vs " if r.get("home", True) else "@ "
            logo = (
                f'<img class="tlogo" src="{team_logo_url(opp)}" '
                'style="width:15px;height:15px;margin-right:5px;vertical-align:-3px;" '
                'loading="lazy" />'
            )
            opp_cell = f"{logo}{prefix}{opp_abbr}"
        else:
            opp_cell = '<span style="color:var(--fp-tx-subtle);">—</span>'

        # Result badge.
        res = r.get("result")
        score = _html.escape(str(r.get("score", "") or ""))
        if res in ("W", "L"):
            res_cls = "w" if res == "W" else "l"
            res_txt = f"{res} {score}".strip()
            res_cell = f'<span class="res {res_cls}">{_html.escape(res_txt)}</span>'
        else:
            res_cell = '<span style="color:var(--fp-tx-subtle);">—</span>'

        stat_cells = "".join(f"<td>{_dossier_fmt(r.get(key), key)}</td>" for _, key in cols)

        # Form bar.
        try:
            form_pct = max(0.0, min(100.0, float(r.get("form_pct", 0.0))))
        except (TypeError, ValueError):
            form_pct = 0.0
        form_cell = (
            '<span class="bar" style="display:inline-block;width:42px;height:5px;border-radius:3px;'
            'background:var(--fp-divider);overflow:hidden;vertical-align:middle;">'
            '<i style="display:block;height:100%;border-radius:3px;'
            "background:linear-gradient(90deg,var(--fp-ember),var(--fp-flame));"
            f'width:{form_pct:g}%;"></i></span>'
        )

        body += (
            f"<tr{row_cls}>"
            f'<td class="l">{date_disp}</td>'
            f'<td class="l">{opp_cell}</td>'
            f"<td>{res_cell}</td>"
            f"{stat_cells}"
            f'<td class="l">{form_cell}</td>'
            "</tr>"
        )

    return f'<table class="glog"><thead><tr>{head_cells}</tr></thead><tbody>{body}</tbody></table>'


def build_upcoming_cards_html(games: list[dict], is_hitter: bool) -> str:
    """Build the ``.up`` stack of upcoming-game projection cards.

    Args:
        games: Per-game dicts. Recognized keys: ``date`` (display string),
            ``opp`` (opponent abbr, optional), ``home`` (bool), ``pitcher``
            (probable-starter label, optional), ``proj`` (dict of projected
            per-game stat values keyed by the player type's projection fields),
            ``conf_pct`` (0–100), ``conf_note`` (short label).
        is_hitter: Selects the projection field set.

    Returns:
        Self-contained HTML string (single ``.up`` wrapper). Empty / falsy
        ``games`` yields a graceful "No upcoming games" state.
    """
    fields = _DOSSIER_HITTER_PROJ_FIELDS if is_hitter else _DOSSIER_PITCHER_PROJ_FIELDS

    if not games:
        return (
            '<div class="up-empty" style="padding:18px 4px;font-family:var(--font-body);'
            'font-size:13px;color:var(--fp-tx-muted);text-align:center;">'
            "No upcoming games scheduled.</div>"
        )

    cards = ""
    for g in games:
        date_disp = _html.escape(str(g.get("date", "") or ""))

        opp = g.get("opp")
        if opp:
            opp_abbr = _html.escape(str(opp).upper())
            prefix = "vs " if g.get("home", True) else "@ "
            logo = (
                f'<img class="tlogo" src="{team_logo_url(opp)}" '
                'style="width:15px;height:15px;margin-right:5px;vertical-align:-3px;" '
                'loading="lazy" />'
            )
            opp_html = (
                '<span class="opp" style="font-family:var(--font-display);font-weight:700;'
                "font-size:12.5px;color:var(--fp-tx);display:inline-flex;align-items:center;"
                f'margin-left:8px;">{logo}{prefix}{opp_abbr}</span>'
            )
        else:
            opp_html = ""

        pitcher = g.get("pitcher")
        sp_html = ""
        if pitcher:
            sp_html = (
                '<span class="sp" style="font-family:var(--font-body);font-weight:600;'
                f'font-size:10.5px;color:var(--fp-tx-muted);">{_html.escape(str(pitcher))}</span>'
            )

        proj = g.get("proj", {}) or {}
        cells = ""
        for i, (lbl, key) in enumerate(fields):
            # Mockup .proj .p:last-child drops the divider — inline styles have no
            # :last-child, so suppress the border on the final cell explicitly.
            border = "" if i == len(fields) - 1 else "border-right:1px solid var(--fp-divider);"
            cells += (
                f'<div class="p" style="flex:1;text-align:center;{border}padding:2px 0;">'
                '<div class="pk" style="font-family:var(--font-display);font-weight:800;font-size:10px;'
                f'letter-spacing:.06em;color:var(--fp-tx);text-transform:uppercase;">{_html.escape(lbl)}</div>'
                '<div class="pv" style="font-family:var(--font-display);font-weight:700;font-size:19px;'
                'font-variant-numeric:tabular-nums;color:var(--fp-ember);letter-spacing:-.01em;">'
                f"{_dossier_fmt(proj.get(key), key)}</div>"
                "</div>"
            )
        proj_html = f'<div class="proj" style="display:flex;gap:0;">{cells}</div>'

        try:
            conf_pct = max(0.0, min(100.0, float(g.get("conf_pct", 0.0))))
        except (TypeError, ValueError):
            conf_pct = 0.0
        conf_note = _html.escape(str(g.get("conf_note", "") or ""))
        conf_label = f"{conf_pct:.0f}%"
        if conf_note:
            conf_label = f"{conf_label} · {conf_note}"
        conf_html = (
            '<div class="conf" style="margin-top:10px;display:flex;align-items:center;gap:8px;">'
            '<div class="t" style="flex:1;height:5px;border-radius:3px;background:var(--fp-divider);'
            'overflow:hidden;">'
            '<i style="display:block;height:100%;'
            "background:linear-gradient(90deg,var(--fp-ember),var(--fp-primary));"
            f'width:{conf_pct:g}%;"></i></div>'
            '<span class="cl" style="font-family:var(--font-mono);font-weight:500;font-size:9px;'
            f'color:var(--fp-tx-muted);">{conf_label}</span>'
            "</div>"
        )

        top_html = (
            '<div class="top" style="display:flex;align-items:center;justify-content:space-between;'
            'margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid var(--fp-divider);">'
            "<div>"
            '<span class="date" style="font-family:var(--font-display);font-weight:800;font-size:14px;'
            f'color:var(--fp-tx);letter-spacing:-.005em;">{date_disp}</span>'
            f"{opp_html}"
            "</div>"
            f"{sp_html}"
            "</div>"
        )

        cards += (
            '<div class="upc" style="position:relative;border:1px solid var(--fp-border);border-radius:10px;'
            "padding:12px 13px;background:linear-gradient(180deg,var(--fp-surface),var(--fp-app-bg));"
            'box-shadow:var(--fp-shadow);overflow:hidden;">'
            "<span style=\"content:'';position:absolute;left:0;top:0;width:26px;height:2px;"
            'background:var(--fp-primary);display:block;"></span>'
            f"{top_html}{proj_html}{conf_html}"
            "</div>"
        )

    return f'<div class="up" style="display:flex;flex-direction:column;gap:10px;">{cards}</div>'


def build_dossier_header_html(profile: dict, season_chips: list[dict]) -> str:
    """Build the team-color-tinted dossier header band (mockup ``.mhead``).

    Args:
        profile: Player profile dict — ``name``, ``team`` (abbr), ``positions``,
            ``bats``, ``jersey`` (optional), ``headshot_url``, ``roster_label``
            (optional eyebrow prefix, default "ROSTER").
        season_chips: Ordered list of ``{"label", "value", "accent"}`` season
            stat chips shown on the right (first chip is highlighted).

    Returns:
        Self-contained HTML string (single ``.mhead`` root element). Background
        is ``team_color(team)`` with ``text_on(...)`` for readable text.
    """
    name = _html.escape(str(profile.get("name", "") or ""))
    team = str(profile.get("team", "") or "")
    positions = _html.escape(str(profile.get("positions", "") or "").split("/")[0] if profile.get("positions") else "")
    bats = _html.escape(str(profile.get("bats", "") or ""))
    jersey = profile.get("jersey")
    roster_label = _html.escape(str(profile.get("roster_label", "ROSTER") or "ROSTER")).upper()
    headshot_url = profile.get("headshot_url") or _AVATAR_FALLBACK_SVG

    base = team_color(team)
    base_dark = _shade_hex(base, 0.55)  # darker companion for the gradient
    ink = text_on(base)
    # Muted variant of the readable ink for the eyebrow / meta lines.
    soft = "rgba(255,255,255,0.72)" if ink == "#ffffff" else "rgba(27,28,32,0.66)"
    logo_url = team_logo_url(team)

    # Eyebrow: ROSTER · POS · #num
    eb_parts = [roster_label]
    if positions:
        eb_parts.append(positions)
    if jersey not in (None, "", 0):
        eb_parts.append(f"#{_html.escape(str(jersey))}")
    eyebrow = " · ".join(eb_parts)

    # Meta line: TEAM · BATS L · 2026 SEASON
    team_full = _html.escape(str(profile.get("team_full", team) or team)).upper()
    meta_parts = [team_full]
    if bats:
        meta_parts.append(f"BATS {bats.upper()}")
    meta_parts.append("2026 SEASON")
    meta_line = " · ".join(meta_parts)
    logo_inline = (
        f'<img src="{logo_url}" style="width:17px;height:17px;vertical-align:-4px;margin-right:7px;" loading="lazy" />'
    )

    # Season stat chips.
    chips_html = ""
    for i, chip in enumerate(season_chips or []):
        lbl = _html.escape(str(chip.get("label", "") or ""))
        val = _html.escape(str(chip.get("value", "") or ""))
        hl = chip.get("accent", i == 0)
        chip_bg = "rgba(255,154,60,0.16)" if hl else "rgba(255,255,255,0.06)"
        v_color = "var(--fp-flame)" if hl else ink
        chips_html += (
            f'<div class="mchip" style="padding:5px 16px;border-left:1px solid rgba(255,255,255,0.16);'
            "display:flex;flex-direction:column;gap:3px;text-align:right;"
            f'background:{chip_bg};">'
            f'<span style="font-family:var(--font-display);font-size:10px;font-weight:800;'
            f'letter-spacing:.14em;text-transform:uppercase;color:{soft};">{lbl}</span>'
            f'<span style="font-family:var(--font-display);font-weight:800;font-size:21px;'
            f'font-variant-numeric:tabular-nums;color:{v_color};letter-spacing:-.01em;">{val}</span>'
            "</div>"
        )

    chips_block = (
        f'<div class="mchips" style="margin-left:auto;display:flex;gap:0;">{chips_html}</div>' if chips_html else ""
    )

    return (
        '<div class="mhead" style="position:relative;display:flex;align-items:center;gap:22px;'
        "padding:24px 28px 26px;border-radius:14px 14px 0 0;overflow:hidden;"
        f'background:linear-gradient(120deg,{base},{base_dark});">'
        # team-logo watermark
        f'<img src="{logo_url}" style="position:absolute;right:18px;top:50%;transform:translateY(-50%);'
        'width:200px;height:200px;opacity:.14;z-index:0;" loading="lazy" />'
        # orange glow
        '<span style="position:absolute;right:-60px;top:-60px;width:280px;height:280px;border-radius:50%;'
        'z-index:0;background:radial-gradient(circle,rgba(235,110,31,.32),transparent 60%);"></span>'
        # headshot
        f'<img src="{headshot_url}" onerror="this.onerror=null;this.src=\'{_AVATAR_FALLBACK_SVG}\'" '
        'style="width:92px;height:92px;border-radius:50%;object-fit:cover;'
        "border:2px solid rgba(255,255,255,.85);box-shadow:0 0 26px rgba(255,109,0,.4);"
        'flex-shrink:0;position:relative;z-index:1;" loading="lazy" />'
        # name block
        '<div class="mname" style="position:relative;z-index:1;">'
        f'<div style="font-family:var(--font-mono);font-size:10px;letter-spacing:.2em;'
        f'color:{soft};text-transform:uppercase;">{eyebrow}</div>'
        f'<div style="font-family:var(--font-display);font-weight:900;font-size:34px;color:{ink};'
        f'letter-spacing:-.01em;line-height:1;margin:5px 0 3px;">{name}</div>'
        '<div style="font-family:var(--font-body);font-weight:600;font-size:12.5px;'
        f'color:{ink};letter-spacing:.02em;display:flex;align-items:center;">{logo_inline}{meta_line}</div>'
        "</div>"
        f"{chips_block}"
        "</div>"
    )


def _shade_hex(hex_color: str, factor: float) -> str:
    """Darken a hex color toward black by ``factor`` (0=black, 1=unchanged)."""
    h = (hex_color or "").lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        return "#10203a"  # navy-ish fallback
    try:
        r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return "#10203a"
    f = max(0.0, min(1.0, factor))
    r, g, b = (int(round(c * f)) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


# ── 3-Zone Layout: Constants & Helpers ─────────────────────────────

HITTING_STAT_COLS = set(HITTING_CATEGORIES)
PITCHING_STAT_COLS = set(PITCHING_CATEGORIES)
# 2026-05-19 D6: snapshot from LeagueConfig (was {"AVG", "OBP", "ERA", "WHIP"} literal).
from src.valuation import LeagueConfig as _LC_FOR_RATES  # noqa: E402

_RATE_STAT_COLS = set(_LC_FOR_RATES().rate_stats)
_RATE_3DP = {"AVG", "OBP", "avg", "obp"}  # 3 decimal places
_RATE_2DP = {"ERA", "WHIP", "era", "whip"}  # 2 decimal places
_INT_COLS = {"ECR", "ecr", "consensus_rank", "GP", "gp", "ytd_gp"}  # integer display (no decimals)


# ── Position Filter (Section 5 helper extract — 2026-05-19) ─────────

# Canonical roster-order positions list, with "All" prepended. Used by the
# pill-button filter on Trade_Finder, Draft_Simulator, Free_Agents.
# (Leaders.py uses a different prospect-rank order intentionally.)
POSITIONS: list[str] = ["All", "C", "1B", "2B", "3B", "SS", "OF", "SP", "RP"]


def render_position_pills(
    key_prefix: str,
    session_key: str,
    default: str = "All",
    positions: list[str] | None = None,
) -> str:
    """Render a row of position-filter pill buttons; return the active filter.

    Identical pill-button widget previously duplicated in Trade_Finder.py:978
    and Draft_Simulator.py:527. Clicking a pill stores its value in
    ``st.session_state[session_key]`` and triggers ``st.rerun()``.

    Args:
        key_prefix: Per-page prefix for button keys (e.g. ``"tv_pill"`` →
            keys ``"tv_pill_All"``, ``"tv_pill_C"``, ...).
        session_key: Session-state slot for the active filter
            (e.g. ``"tv_pos_filter"``).
        default: Default filter when ``session_key`` is unset.
        positions: Optional override (default: module-level POSITIONS).

    Returns:
        The currently-active position filter string.
    """
    import streamlit as st

    pos_list = positions if positions is not None else POSITIONS
    cols = st.columns(len(pos_list))
    current = st.session_state.get(session_key, default)
    for i, pos in enumerate(pos_list):
        with cols[i]:
            btn_type = "primary" if current == pos else "secondary"
            if st.button(pos, key=f"{key_prefix}_{pos}", type=btn_type, width="stretch"):
                st.session_state[session_key] = pos
                st.rerun()
    return current


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


def _headshot_img_html(mlb_id, size: int = 22, team=None) -> str:
    """Return a tiny circular headshot <img> tag.

    Always returns a visible circle:
    - If mlb_id is valid, shows the MLB headshot with fallback to generic avatar on error.
    - If mlb_id is missing/invalid, shows the generic avatar directly.
    - When *team* (MLB id or abbr) RESOLVES, the circle is backed by the team's
      primary color — the MLB "spots" headshot PNGs are transparent, so the
      color shows through (2026-06-10 owner request). Missing/unresolvable
      team keeps the neutral card tint (never the orange team_color fallback,
      which would paint every unknown avatar brand-orange).
    """
    bg = THEME["card_h"]
    if team is not None and _resolve_team_id(team) is not None:
        bg = team_color(team)
    _style = (
        f"border-radius:50%;object-fit:cover;vertical-align:middle;"
        f"margin-right:5px;border:1px solid rgba(0,0,0,0.08);flex-shrink:0;"
        f"background:{bg};"
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


# Mockup .rtbl column sets — (df_key, header_label, stat_type_for_format).
# stat_type None → counting stat (int via format_stat default path).
_ROSTER_HITTER_COLS: list[tuple[str, str, str | None]] = [
    ("ab", "AB", None),
    ("r", "R", None),
    ("h", "H", None),
    ("hr", "HR", None),
    ("rbi", "RBI", None),
    ("sb", "SB", None),
    ("avg", "AVG", "AVG"),
    ("obp", "OBP", "OBP"),
]
_ROSTER_PITCHER_COLS: list[tuple[str, str, str | None]] = [
    ("ip", "IP", None),
    ("w", "W", None),
    ("l", "L", None),
    ("sv", "SV", None),
    ("k", "K", None),
    ("era", "ERA", "ERA"),
    ("whip", "WHIP", "WHIP"),
]


def _roster_status_dot_class(status) -> str:
    """Map a Yahoo roster status string to a ``.sdot`` modifier (ok/dtd/il)."""
    s = str(status or "").strip().lower()
    if not s or s in ("active", "ok", "starting", "started"):
        return "ok"
    if s in ("dtd", "day-to-day", "day to day", "gtd", "questionable", "probable"):
        return "dtd"
    # IL10 / IL15 / IL60 / IL / NA / OUT / DL / SUSP — anything else = injured/out.
    return "il"


def build_roster_table_html(df, *, is_hitter: bool = True, player_ids=None) -> str:
    """Build the Combustion ``.rtbl`` Active Roster table (mockup My Team v3).

    Renders the branded roster table with Archivo tabular figures, an Archivo-800
    uppercase header row with a 2px bottom border, a player cell (headshot +
    status dot + name + team logo/abbr), and a per-row ``--tc`` team-color custom
    property that drives the 3px left accent + 8% tint on the player column.

    Columns:
        - hitters: Player, Pos, AB, R, H, HR, RBI, SB, AVG, OBP
        - pitchers: Player, Pos, IP, W, L, SV, K, ERA, WHIP

    The MLB team for each row is resolved from the ``team`` (or ``mlb_team`` /
    ``editorial_team_abbr``) column for the logo + color; unresolved teams fall
    back to the orange brand color.

    Args:
        df: Roster slice. Expected columns: ``name``, ``positions`` (or
            ``Pos``), ``mlb_id``, ``status``, ``team``, plus the stat columns for
            the selected side. Missing stat cells render as ``—``.
        is_hitter: Select the hitter (True) or pitcher (False) column set.
        player_ids: Optional list aligned row-for-row with ``df``. When supplied,
            each player cell becomes an ``<a href="?player=<id>">`` so a click
            opens the existing player-card dialog.

    Returns:
        Self-contained HTML string (single ``.rtbl`` table element). Empty/None
        ``df`` still returns a valid header-only table shell.
    """
    cols = _ROSTER_HITTER_COLS if is_hitter else _ROSTER_PITCHER_COLS

    # ── header row ──
    th_common = (
        "font-family:var(--font-display);font-weight:800;font-size:10.5px;"
        "letter-spacing:.06em;text-transform:uppercase;color:#2c2f36;"
        "padding:9px 12px;border-bottom:2px solid rgba(24,26,32,.18);"
    )
    head = (
        f'<th class="ph" style="{th_common}text-align:left;">Player</th>'
        f'<th class="pp" style="{th_common}text-align:center;">Pos</th>'
    )
    for _key, label, _fmt in cols:
        head += f'<th style="{th_common}text-align:right;">{label}</th>'
    thead = f"<thead><tr>{head}</tr></thead>"

    n_cols = 2 + len(cols)

    if df is None or len(df) == 0:
        empty_row = (
            f'<tr><td colspan="{n_cols}" style="padding:18px 12px;text-align:center;'
            f'font-family:var(--font-mono);font-size:11px;color:var(--fp-tx-subtle);">'
            f"No players</td></tr>"
        )
        return (
            '<table class="rtbl" style="width:100%;border-collapse:collapse;margin-top:8px;">'
            f"{thead}<tbody>{empty_row}</tbody></table>"
        )

    ids_list = list(player_ids) if player_ids is not None else None

    # Resolve the team column once (first matching name in the row dict).
    team_col = next(
        (c for c in ("team", "mlb_team", "editorial_team_abbr", "Team") if c in df.columns),
        None,
    )
    pos_col = next((c for c in ("positions", "Pos", "pos") if c in df.columns), None)

    td_num = (
        "padding:8px 12px;border-bottom:1px solid var(--fp-divider);text-align:right;"
        "font-family:var(--font-display);font-weight:700;font-size:13.5px;"
        "font-variant-numeric:tabular-nums;color:var(--fp-tx);"
    )

    body_rows: list[str] = []
    for pos_i, (_idx, row) in enumerate(df.iterrows()):
        team = row.get(team_col) if team_col else None
        tc = team_color(team)
        logo = team_logo_url(team)
        abbr = ""
        if team is not None and not (isinstance(team, float) and _math.isnan(team)):
            abbr = _html.escape(str(team).strip().upper())

        dot = _roster_status_dot_class(row.get("status"))
        name = _html.escape(str(row.get("name", "")))
        mlb_id = row.get("mlb_id")
        headshot = _headshot_img_html(mlb_id, size=36, team=team)

        # Player cell inner (headshot + name w/ status dot + team logo + abbr).
        sdot = (
            f'<span class="sdot {dot}" style="width:7px;height:7px;border-radius:50%;'
            f"display:inline-block;margin-right:7px;flex-shrink:0;"
            f'background:{_SDOT_COLORS[dot]};"></span>'
        )
        tlogo = (
            f'<img class="tlogo" src="{logo}" width="13" height="13" '
            f'style="vertical-align:-3px;margin-right:5px;" loading="lazy" '
            f"onerror=\"this.style.display='none'\" />"
        )
        pcell_inner = (
            f'<span class="pcell" style="display:flex;align-items:center;gap:12px;">'
            f"{headshot}"
            f'<span class="pi" style="display:flex;flex-direction:column;gap:1px;min-width:0;">'
            f'<b style="font-family:var(--font-body);font-weight:600;font-size:13.5px;'
            f'color:var(--fp-tx);display:flex;align-items:center;">{sdot}{name}</b>'
            f'<span style="font-family:var(--font-mono);font-size:10px;color:var(--fp-tx-subtle);'
            f'letter-spacing:.04em;display:flex;align-items:center;">{tlogo}{abbr}</span>'
            f"</span></span>"
        )

        # Click wiring: wrap the player cell in an anchor to ?player=<id>.
        if ids_list is not None and pos_i < len(ids_list):
            pid = ids_list[pos_i]
            pcell_inner = (
                f'<a href="?player={_html.escape(str(pid))}" target="_self" '
                f'style="text-decoration:none;color:inherit;display:block;">{pcell_inner}</a>'
            )

        ph_td = (
            f'<td class="ph" style="padding:8px 12px;border-bottom:1px solid var(--fp-divider);'
            f"text-align:left;position:relative;"
            f"background:transparent;"
            f"background:color-mix(in srgb,{tc} 8%,transparent);"
            f'box-shadow:inset 3px 0 0 {tc};">{pcell_inner}</td>'
        )

        # Position chip.
        pos_raw = row.get(pos_col) if pos_col else ""
        pos_txt = _html.escape(str(pos_raw or "").split(",")[0].split("/")[0].strip())
        pp_td = (
            f'<td class="pp" style="padding:8px 12px;border-bottom:1px solid var(--fp-divider);'
            f'text-align:center;">'
            f'<span class="pos" style="display:inline-block;font-family:var(--font-mono);'
            f"font-size:9.5px;font-weight:600;color:var(--fp-tx-muted);background:var(--fp-surface);"
            f'border:1px solid var(--fp-border);border-radius:5px;padding:2px 7px;">{pos_txt}</span></td>'
        )

        # Stat cells + leader-highlight bookkeeping.
        stat_tds: list[str] = []
        for key, _label, fmt in cols:
            val = row.get(key)
            disp = _format_roster_stat(val, fmt)
            hl = _roster_cell_is_leader(df, key, val, is_hitter)
            cell_style = td_num + ("color:var(--fp-ember);" if hl else "")
            cls = ' class="hl"' if hl else ""
            stat_tds.append(f"<td{cls} style={chr(34)}{cell_style}{chr(34)}>{disp}</td>")

        body_rows.append(f'<tr style="--tc:{tc}">{ph_td}{pp_td}{"".join(stat_tds)}</tr>')

    return (
        '<table class="rtbl" style="width:100%;border-collapse:collapse;margin-top:8px;">'
        f"{thead}<tbody>{''.join(body_rows)}</tbody></table>"
    )


# Status-dot fill colors (mockup .sdot.ok/.dtd/.il), resolved from THEME.
_SDOT_COLORS = {
    "ok": THEME["green"],
    "dtd": THEME["gold"],
    "il": THEME["danger"],
}

# Stat keys eligible for the orange leader highlight (tasteful subset matching
# the mockup's HR / AVG / SB / OBP accents). Inverse / rate-noise stats excluded.
_ROSTER_HL_HITTER = {"hr", "avg", "sb", "obp"}
_ROSTER_HL_PITCHER = {"k", "sv"}


def _format_roster_stat(val, fmt: str | None) -> str:
    """Format a roster stat cell; missing/NaN → em dash."""
    if val is None:
        return "—"
    try:
        f = float(val)
    except (TypeError, ValueError):
        s = str(val).strip()
        return _html.escape(s) if s else "—"
    if _math.isnan(f) or _math.isinf(f):
        return "—"
    if fmt:
        return format_stat(f, fmt)
    # Counting stat — integer when whole, else 1 decimal (e.g. IP 120.1).
    return f"{int(f)}" if f == int(f) else f"{f:.1f}"


def _roster_cell_is_leader(df, key: str, val, is_hitter: bool) -> bool:
    """Return True when ``val`` is the column max for an HL-eligible stat.

    Used for the tasteful orange leader highlight on a small subset of columns
    (HR/AVG/SB/OBP for hitters; K/SV for pitchers). Single-row tables don't
    highlight (no leader to distinguish).
    """
    import pandas as _pd

    hl_set = _ROSTER_HL_HITTER if is_hitter else _ROSTER_HL_PITCHER
    if key not in hl_set or key not in getattr(df, "columns", []):
        return False
    try:
        col = _pd.to_numeric(df[key], errors="coerce")
    except Exception:
        return False
    if col.notna().sum() < 2:
        return False
    try:
        v = float(val)
    except (TypeError, ValueError):
        return False
    if _math.isnan(v):
        return False
    col_max = col.max(skipna=True)
    return bool(v >= col_max and col_max > 0)


def build_compact_table_html(
    df,
    highlight_cols=None,
    row_classes=None,
    health_col=None,
    max_height=500,
    show_avatars=None,
    html_cols=None,
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
        html_cols: Optional set of column names whose values are TRUSTED
            pre-built HTML (e.g. rank badges) and must not be escaped.
            All other cells stay escaped.

    Returns:
        HTML string with ``compact-table-wrap`` / ``compact-table`` classes.
    """
    if df is None or df.empty:
        return '<div class="compact-table-wrap"><p style="padding:16px;color:var(--fp-tx-muted);font-size:13px;">No data available.</p></div>'

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

            # Format numeric values — skip first col and health col.
            # html_cols columns bypass numeric formatting entirely and go
            # straight to the trusted-HTML branch (raw, unescaped).
            _skip_cols = {first_col}
            if health_col:
                _skip_cols.add(health_col)
            if html_cols and col in html_cols:
                cell_html += str(val) if val is not None else ""
            elif col not in _skip_cols:
                try:
                    fv = float(val)
                    if _math.isnan(fv) or _math.isinf(fv):
                        cell_html += ""
                    elif str(col) in _RATE_3DP or str(col).upper() in {"AVG", "OBP"}:
                        cell_html += f"{fv:.3f}"
                    elif str(col) in _RATE_2DP or str(col).upper() in {"ERA", "WHIP"}:
                        cell_html += f"{fv:.2f}"
                    elif str(col) in _INT_COLS:
                        cell_html += f"{int(fv)}"
                    else:
                        cell_html += f"{fv:.2f}"
                except (ValueError, TypeError):
                    cell_html += _html.escape(str(val)) if val is not None else ""
            else:
                cell_html += _html.escape(str(val)) if val is not None else ""

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


def render_roster_table(
    roster_df: pd.DataFrame,
    mode: str = "overview",
    health_col: str | None = "health_score",
) -> None:
    """S6: Shared roster table rendering for My Team and Lineup Optimizer.

    Sorts by Yahoo slot order, selects mode-appropriate columns, and
    renders via render_compact_table(). Both pages should call this
    instead of building their own render pipelines.

    Args:
        roster_df: Roster DataFrame with player_name/name, positions,
            selected_position, and stat columns.
        mode: "overview" (My Team — key stats) or "optimizer" (Lineup — projection focus).
        health_col: Column name for health badges (None to skip).
    """
    import streamlit as st

    if roster_df is None or roster_df.empty:
        st.info("No roster data available.")
        return

    df = sort_roster_for_display(roster_df)

    # Determine name column
    name_col = "player_name" if "player_name" in df.columns else "name"

    if mode == "overview":
        cols = [name_col, "positions", "team"]
        # Add key stats
        for c in ["r", "hr", "rbi", "sb", "avg", "w", "sv", "k", "era", "whip"]:
            if c in df.columns:
                cols.append(c)
    else:  # optimizer
        cols = [name_col, "positions"]
        for c in ["r", "hr", "rbi", "sb", "avg", "obp", "w", "l", "sv", "k", "era", "whip"]:
            if c in df.columns:
                cols.append(c)

    if health_col and health_col in df.columns:
        cols.append(health_col)

    display = df[[c for c in cols if c in df.columns]]
    render_compact_table(display, health_col=health_col if health_col in display.columns else None)


def render_compact_table(
    df,
    highlight_cols=None,
    row_classes=None,
    health_col=None,
    max_height=500,
    show_avatars=None,
    html_cols=None,
):
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
        html_cols=html_cols,
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
            f'<span style="width:50px;display:inline-block;">{_html.escape(str(cat))}</span>'
            f'<span style="width:60px;text-align:right;display:inline-block;">{_html.escape(str(you_val))}</span>'
            f'<span style="width:20px;text-align:center;display:inline-block;opacity:0.4;">v</span>'
            f'<span style="width:60px;display:inline-block;">{_html.escape(str(opp_val))}</span>'
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
    """Render page title header + recommendation banner + matchup ticker.

    Call at the top of every page, after ``inject_custom_css()``.

    Combustion redesign (2026-06-08, Phase D): the page title now renders the
    mockup ``.phead`` header (Archivo-900 title + orange underline rule, no navy
    pill) via :func:`render_page_header`, so every page that calls this gets the
    new look in one place. The banner-teaser + matchup-ticker behavior is kept.
    """
    render_page_header(title)
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
        font=dict(family="Inter, sans-serif", size=12),
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


def _dossier_is_hitter(positions: str) -> bool:
    """True when the player has any non-pitching eligibility."""
    toks = [p.strip().upper() for p in (positions or "").split("/") if p.strip()]
    if not toks:
        return True  # default to hitter layout when position unknown
    return not all(p in ("SP", "RP", "P") for p in toks)


def _dossier_season_chips(historical: list[dict], projections: dict, is_hitter: bool) -> list[dict]:
    """Build the header season-stat chips from the 2026 season line (fallback: blended proj).

    Hitters: AVG / HR / RBI / OBP / OPS. Pitchers: W / K / SV / ERA / WHIP.
    Returns ``[{"label","value","accent"}]``; missing values render as "—".
    """
    # Prefer the in-season (2026) actuals; fall back to blended projection.
    src: dict = {}
    for h in historical or []:
        try:
            if int(h.get("season", 0)) == 2026:
                src = h
                break
        except (TypeError, ValueError):
            continue
    if not src:
        src = (projections or {}).get("blended", {}) or {}

    def g(*keys):
        for k in keys:
            v = src.get(k)
            if v is not None:
                return v
        return None

    if is_hitter:
        avg = g("AVG", "avg")
        obp = g("OBP", "obp")
        # OPS = OBP + SLG; SLG rarely present, so approximate as OBP+? only when SLG known.
        slg = g("SLG", "slg")
        ops = None
        try:
            if obp is not None and slg is not None:
                ops = float(obp) + float(slg)
        except (TypeError, ValueError):
            ops = None
        return [
            {"label": "AVG", "value": _dossier_fmt(avg, "avg"), "accent": True},
            {"label": "HR", "value": _dossier_fmt(g("HR", "hr"), "hr"), "accent": False},
            {"label": "RBI", "value": _dossier_fmt(g("RBI", "rbi"), "rbi"), "accent": False},
            {"label": "OBP", "value": _dossier_fmt(obp, "obp"), "accent": False},
            {"label": "OPS", "value": _dossier_fmt(ops, "ops") if ops is not None else "—", "accent": False},
        ]
    return [
        {"label": "ERA", "value": _dossier_fmt(g("ERA", "era"), "era"), "accent": True},
        {"label": "W", "value": _dossier_fmt(g("W", "w"), "w"), "accent": False},
        {"label": "K", "value": _dossier_fmt(g("K", "k"), "k"), "accent": False},
        {"label": "SV", "value": _dossier_fmt(g("SV", "sv"), "sv"), "accent": False},
        {"label": "WHIP", "value": _dossier_fmt(g("WHIP", "whip"), "whip"), "accent": False},
    ]


def _dossier_game_log_rows(player_id: int, is_hitter: bool, limit: int = 10) -> list[dict]:
    """Assemble up-to-``limit`` newest-first game-log rows for the dossier table.

    Sources per-game counting lines from ``game_logs`` (via
    ``player_databank.load_game_logs``). When the row carries the opponent /
    result enrichment columns (``opponent_abbr``, ``is_home``, ``result``;
    populated by the game-log refresh) the opponent logo + "vs/@ ABBR" and a W/L
    badge render; rows that predate the enrichment (NULL columns) degrade
    gracefully to "—". The Form bar is derived from per-game performance vs the
    player's own window. Returns ``[]`` on any failure (caller renders the empty
    state).
    """
    try:
        from src.player_databank import load_game_logs
    except Exception:
        return []

    try:
        logs = load_game_logs([int(player_id)], season=datetime_now_year())
    except Exception:
        return []
    if logs is None or logs.empty:
        return []

    logs = logs.head(limit)

    def _num(row, col):
        if col not in row.index:
            return 0.0
        v = row[col]
        try:
            f = float(v)
            return 0.0 if (f != f) else f  # NaN guard
        except (TypeError, ValueError):
            return 0.0

    def _opp_result(row) -> dict:
        """Extract opponent / home / result enrichment from a game_logs row.

        Returns a dict with ``opp`` / ``home`` / ``result`` / ``score`` keys
        suitable for ``build_game_log_html``. Missing / NULL columns are omitted
        so the renderer shows a graceful "—" (old rows, pre-enrichment).
        """
        out: dict = {}
        # Opponent abbr (str). NULL/NaN/empty → omit (renders "—").
        if "opponent_abbr" in row.index:
            opp = row["opponent_abbr"]
            if isinstance(opp, str) and opp.strip():
                out["opp"] = opp.strip()
        # Home/away flag → "vs" vs "@". Default True only matters when ``opp``
        # is present; pandas may store the int as float/NaN.
        if "is_home" in row.index:
            ih = row["is_home"]
            try:
                if ih is not None and ih == ih:  # NaN guard
                    out["home"] = bool(int(ih))
            except (TypeError, ValueError):
                pass
        # Result badge ("W"/"L").
        if "result" in row.index:
            res = row["result"]
            if isinstance(res, str) and res.strip().upper() in ("W", "L"):
                out["result"] = res.strip().upper()
        # Optional score "team-opp" when both are present (currently always NULL
        # from statsapi — kept for forward-compat once a score source is wired).
        if "team_score" in row.index and "opp_score" in row.index:
            ts, os_ = row["team_score"], row["opp_score"]
            try:
                if (ts is not None and ts == ts) and (os_ is not None and os_ == os_):
                    out["score"] = f"{int(ts)}-{int(os_)}"
            except (TypeError, ValueError):
                pass
        return out

    rows: list[dict] = []
    # For the form bar: hitters use per-game total bases proxy (H + 2*HR + RBI),
    # pitchers use a quality proxy (K - ER). Scale to the window peak.
    raw_scores: list[float] = []
    parsed: list[dict] = []
    for _, r in logs.iterrows():
        gd = str(r.get("game_date", "") or "")
        # Display date "Jun 08".
        date_disp = gd
        try:
            from datetime import datetime as _dt

            date_disp = _dt.strptime(gd[:10], "%Y-%m-%d").strftime("%b %d")
        except (ValueError, TypeError):
            pass

        if is_hitter:
            ab = _num(r, "ab")
            h = _num(r, "h")
            row = {
                "date": date_disp,
                "ab": ab,
                "h": h,
                "hr": _num(r, "hr"),
                "rbi": _num(r, "rbi"),
                "avg": (h / ab) if ab > 0 else 0.0,
            }
            row.update(_opp_result(r))
            score = h + 2.0 * row["hr"] + row["rbi"]
        else:
            ip = _num(r, "ip")
            er = _num(r, "er")
            row = {
                "date": date_disp,
                "ip": ip,
                "h_allowed": _num(r, "h_allowed"),
                "er": er,
                "k": _num(r, "k"),
                "era": (er * 9.0 / ip) if ip > 0 else 0.0,
            }
            row.update(_opp_result(r))
            score = row["k"] - er
        raw_scores.append(score)
        parsed.append(row)

    if raw_scores:
        lo = min(raw_scores)
        hi = max(raw_scores)
        span = (hi - lo) or 1.0
        # "hot" = upper third of this player's own window.
        hot_cut = lo + span * 0.66
        for row, sc in zip(parsed, raw_scores):
            row["form_pct"] = max(8.0, min(100.0, (sc - lo) / span * 100.0))
            row["hot"] = sc >= hot_cut and hi > lo
            rows.append(row)
    return rows


def _dossier_upcoming_games(profile: dict, projections: dict, is_hitter: bool, limit: int = 4) -> list[dict]:
    """Best-effort upcoming-game projection cards.

    Sources the player's team's next MLB games + probable pitchers from the
    MLB Stats API (graceful: returns ``[]`` if statsapi is unavailable, the
    network is blocked, or the team can't be resolved). Per-game projections
    scale the blended season rate to a single game. The caller renders a
    "No upcoming games" state when this is empty.
    """
    team = profile.get("team", "") or ""
    blended = (projections or {}).get("blended", {}) or {}
    if not team or not blended:
        return []

    try:
        from datetime import datetime as _dt
        from datetime import timedelta as _td

        import statsapi
    except Exception:
        return []

    # Resolve team id → statsapi team lookup.
    try:
        tid = _resolve_team_id(team)
        if tid is None:
            return []
        start = _dt.now().strftime("%Y-%m-%d")
        end = (_dt.now() + _td(days=10)).strftime("%Y-%m-%d")
        sched = statsapi.schedule(start_date=start, end_date=end, team=tid)
    except Exception:
        return []
    if not sched:
        return []

    # Per-game scale factors from blended counting stats.
    def _per_game(stat_key: str, games_basis: float) -> float | None:
        v = blended.get(stat_key) or blended.get(stat_key.lower())
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return None
        return fv / games_basis if games_basis > 0 else None

    games: list[dict] = []
    for g in sched[:limit]:
        try:
            game_date = str(g.get("game_date") or g.get("game_datetime", "") or "")[:10]
            date_disp = _dt.strptime(game_date, "%Y-%m-%d").strftime("%b %d") if game_date else ""
        except (ValueError, TypeError):
            date_disp = ""

        home_id = g.get("home_id")
        is_home = home_id == tid
        opp_id = g.get("away_id") if is_home else home_id
        opp_abbr = TEAM_BRAND.get(opp_id, {}).get("abbr") if opp_id in TEAM_BRAND else None
        pitcher = g.get("away_probable_pitcher") if is_home else g.get("home_probable_pitcher")
        pitcher_txt = f"vs {pitcher}" if pitcher and str(pitcher).upper() != "TBD" else ""

        if is_hitter:
            # Hitters: counting stats over a ~150-game season → per-game line.
            gp_basis = 150.0
            proj = {
                "h": _per_game("H", gp_basis) if blended.get("H") else _hits_from_avg(blended),
                "hr": _per_game("HR", gp_basis),
                "rbi": _per_game("RBI", gp_basis),
                "r": _per_game("R", gp_basis),
            }
        else:
            # Pitchers: starters ~32 starts; per-start lines.
            gp_basis = 32.0
            proj = {
                "k": _per_game("K", gp_basis),
                "era": blended.get("ERA") or blended.get("era"),
                "ip": _per_game("IP", gp_basis),
                "w": _per_game("W", gp_basis),
            }

        games.append(
            {
                "date": date_disp,
                "opp": opp_abbr,
                "home": is_home,
                "pitcher": pitcher_txt,
                "proj": proj,
                "conf_pct": 60.0,  # neutral confidence baseline (no per-matchup model wired here)
                "conf_note": "home" if is_home else "away",
            }
        )
    return games


def _hits_from_avg(blended: dict) -> float | None:
    """Approximate per-game hits from AVG when H is missing (~4.1 AB/game)."""
    avg = blended.get("AVG") or blended.get("avg")
    try:
        return float(avg) * 4.1
    except (TypeError, ValueError):
        return None


def datetime_now_year() -> int:
    """Current year (small wrapper kept import-light for the dossier helpers)."""
    from datetime import UTC
    from datetime import datetime as _dt

    return _dt.now(UTC).year


@st.dialog("Player Card", width="large")
def show_player_card_dialog(player_id: int):
    """Render the player dossier modal (Combustion redesign).

    Layout mirrors the gold-standard mockup: a team-color header band, a
    "Game Log — Last 10" table, and "Upcoming · Projections" cards. The
    legacy News + Radar detail are preserved below the mockup content in
    collapsed expanders. Name + signature kept stable so existing wiring
    (``?player=<id>`` on My Team) keeps working.
    """
    from src.player_card import build_player_card_data

    data = build_player_card_data(player_id)
    profile = data["profile"]
    projections = data.get("projections", {})
    historical = data.get("historical", [])
    is_hitter = _dossier_is_hitter(profile.get("positions", ""))

    # ── 1. Header band (team-color tinted) ──────────────────────
    season_chips = _dossier_season_chips(historical, projections, is_hitter)
    st.markdown(build_dossier_header_html(profile, season_chips), unsafe_allow_html=True)

    # ── 2. Body: Game Log (left) | Upcoming·Projections (right) ──
    glog_rows = _dossier_game_log_rows(player_id, is_hitter)
    upcoming = _dossier_upcoming_games(profile, projections, is_hitter)

    # Form caption (last-6 hit/quality summary) for the game-log panel.
    glog_fig = f"LAST {len(glog_rows)}" if glog_rows else "NO DATA"
    upc_fig = f"NEXT {len(upcoming)}" if upcoming else "—"

    left, right = st.columns([1.45, 1])
    with left:
        st.markdown(
            build_panel_html(
                "Game Log — Last 10",
                build_game_log_html(glog_rows, is_hitter),
                fig_label=glog_fig,
                accent="left",
            ),
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            build_panel_html(
                "Upcoming · Projections",
                build_upcoming_cards_html(upcoming, is_hitter),
                fig_label=upc_fig,
                accent="left",
            ),
            unsafe_allow_html=True,
        )

    # ── 3. Secondary detail (preserved below the mockup content) ─
    news = data.get("news", [])
    with st.expander("Recent News", expanded=False):
        _render_news_section(news)

    radar = data.get("radar", {})
    if radar.get("player"):
        with st.expander("Percentile Rankings", expanded=False):
            _render_radar_chart(radar, is_hitter)

    # ── 4. Historical / projections / advanced / rankings detail ─
    with st.expander("Stats, Projections & Rankings", expanded=False):
        _render_dossier_detail(data, is_hitter)


def _render_dossier_detail(data: dict, is_hitter: bool) -> None:
    """Render the legacy stat/projection/advanced/rankings/injury/prospect detail.

    Kept as a secondary expander beneath the Combustion dossier layout so the
    rich data the card already loads is preserved without crowding the hero.
    """
    # 3. Historical stats
    historical = data.get("historical", [])
    if historical:
        st.markdown('<div class="sec-head">Historical Stats (3 Years)</div>', unsafe_allow_html=True)
        cats = list(HITTING_CATEGORIES) if is_hitter else list(PITCHING_CATEGORIES)
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
        cats = list(HITTING_CATEGORIES) if is_hitter else list(PITCHING_CATEGORIES)
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
                f'font-family:Inter,sans-serif;">' + "<br>".join(rank_parts) + "</div>",
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


def render_player_select(player_names, player_ids, key_suffix="default", label="View player card"):
    """Render a selectbox that opens a player card dialog when a player is chosen.

    Args:
        player_names: Selectable player display names.
        player_ids: IDs aligned row-for-row with ``player_names``.
        key_suffix: Disambiguates the widget key across pages.
        label: The selectbox label. Defaults to ``"View player card"`` so
            existing callers are unchanged; My Team passes ``"Open player
            dossier"`` (Combustion C3) to make the dossier opener obvious.
    """
    if not player_ids:
        return

    selected = st.selectbox(
        label,
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
        f'font-size:11px;color:{THEME["tx2"]};font-family:var(--font-mono);">'
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
    status = yds.connection_status()

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

    # Under MULTI_USER, member sessions never hold a live client, so report the
    # DATA status (server-served / warming), not the always-False session
    # connection. v1 keeps the literal connected/offline wording.
    _status_label = {
        "connected": ("Yahoo Connected", t["green"]),
        "server": ("Yahoo: Live (via server)", t["green"]),
        "warming": ("Yahoo: Warming up", t["amber"]),
        "offline": ("Yahoo Offline", t["tx2"]),
    }
    _label_text, _label_color = _status_label.get(status, ("Yahoo Offline", t["tx2"]))
    conn_status = (
        f'<div style="font-size:10px!important;color:{_label_color}!important;'
        f'margin-bottom:4px!important">{_label_text}</div>'
    )

    render_context_card("Data Freshness", f"{conn_status}{rows_html}")

    if connected:
        if st.button("Refresh All Data", key="_yds_refresh_all", type="secondary"):
            with st.spinner("Refreshing..."):
                results = yds.force_refresh_all()
                refreshed = sum(1 for v in results.values() if v == "Refreshed")
                st.toast(f"Refreshed {refreshed}/{len(results)} data sources")
                st.rerun()


# ── Unified Stat Formatting ──────────────────────────────────────────


# ── Data-Freshness Chip ───────────────────────────────────────────────────────
# Reuses get_refresh_log_snapshot() from src.database — no duplicate tracking.
# DataFreshnessTracker (src/optimizer/data_freshness.py) manages TTLs for the
# optimizer pipeline; the chip here surfaces per-source last_refresh to the UI.


def humanize_age(age_minutes: float) -> str:
    """Return a human-readable relative-time string for *age_minutes*.

    Examples:
        0  -> "just now"
        1  -> "1 minute ago"
        5  -> "5 minutes ago"
        60 -> "1 hour ago"
        90 -> "1 hour ago"
        120 -> "2 hours ago"
        1440 -> "1 day ago"
        2880 -> "2 days ago"
    """
    try:
        mins = float(age_minutes)
    except (TypeError, ValueError):
        return "unknown"

    if mins <= 0:
        return "just now"
    if mins < 2:
        return "1 minute ago"
    if mins < 60:
        return f"{int(mins)} minutes ago"
    hours = int(mins // 60)
    if hours < 24:
        if hours == 1:
            return "1 hour ago"
        return f"{hours} hours ago"
    days = int(hours // 24)
    if days == 1:
        return "1 day ago"
    return f"{days} days ago"


def render_data_freshness_chip(
    source: str | None = None,
    *,
    age_minutes: float | None = None,
) -> None:
    """Render a small inline chip showing how fresh the data for *source* is.

    Data source:
    - If *age_minutes* is given it is used directly (useful for tests / callers
      that already computed the age).
    - Otherwise, *source* is looked up in ``get_refresh_log_snapshot()`` and the
      age is derived from ``last_refresh``.
    - If neither is available the chip reads "freshness unknown".

    Visual treatment:
    - **Neutral** (surface fill, muted text) when age <= 1440 min (24 h).
    - **Amber/warn** (``var(--fp-amber)`` border + tint) when age > 1440 min.

    No emoji. Uses only THEME tokens / CSS vars.
    """
    from datetime import UTC, datetime

    if st is None:
        return  # Streamlit not available

    # ── Resolve age ──────────────────────────────────────────────────────────
    resolved_minutes: float | None = age_minutes

    if resolved_minutes is None and source is not None:
        try:
            snap = get_refresh_log_snapshot()
            for row in snap:
                if row.get("source") == source:
                    last_refresh = row.get("last_refresh")
                    if last_refresh:
                        try:
                            if isinstance(last_refresh, str):
                                # ISO format stored as "2026-06-13T10:00:00" or similar
                                ts = datetime.fromisoformat(last_refresh)
                                if ts.tzinfo is None:
                                    ts = ts.replace(tzinfo=UTC)
                            else:
                                ts = last_refresh
                            now = datetime.now(UTC)
                            resolved_minutes = (now - ts).total_seconds() / 60.0
                        except Exception:
                            pass
                    break
        except Exception:
            pass
    elif resolved_minutes is None and source is None:
        # No source, no age — still try snapshot so callers/tests can observe the call
        try:
            get_refresh_log_snapshot()
        except Exception:
            pass

    # ── Render ───────────────────────────────────────────────────────────────
    t = THEME
    if resolved_minutes is None:
        label = "Freshness unknown"
        border_color = f"var(--fp-border, {t['border']})"
        bg_color = f"var(--fp-surface, {t['surface']})"
        text_color = f"var(--fp-tx-subtle, {t['tx_subtle']})"
        dot_color = t["tx_subtle"]
    else:
        label = f"Updated {humanize_age(resolved_minutes)}"
        stale = resolved_minutes > 1440
        if stale:
            border_color = f"var(--fp-amber, {t['warn']})"
            bg_color = "rgba(255,159,28,0.08)"
            text_color = f"var(--fp-amber, {t['warn']})"
            dot_color = t["warn"]
        else:
            border_color = f"var(--fp-border, {t['border']})"
            bg_color = f"var(--fp-surface, {t['surface']})"
            text_color = f"var(--fp-tx-muted, {t['tx_muted']})"
            dot_color = t["green"]

    import html as _html_mod

    safe_label = _html_mod.escape(label)
    html = (
        f'<span style="'
        f"display:inline-flex;align-items:center;gap:5px;"
        f"font-family:var(--font-mono,'IBM Plex Mono',monospace);"
        f"font-size:11px;font-weight:500;"
        f"color:{text_color};"
        f"border:1px solid {border_color};"
        f"background:{bg_color};"
        f"border-radius:20px;padding:2px 10px;"
        f'">'
        f'<span style="width:6px;height:6px;border-radius:50%;'
        f"background:{dot_color};display:inline-block;flex-shrink:0;"
        f'"></span>'
        f"{safe_label}"
        f"</span>"
    )
    st.markdown(html, unsafe_allow_html=True)


# ── Glossary / Jargon Tooltips ────────────────────────────────────────────────


JARGON: dict[str, str] = {
    "SGP": "Standings Gained Points — how many standings spots a stat contributes. Higher = more league impact.",
    "DCV": "Daily Category Value — single-day scoring contribution across all 12 categories.",
    "VORP": "Value Over Replacement Player — how much better this player is than a freely available substitute.",
    "wRC+": "Weighted Runs Created Plus — overall offensive value vs league avg (100 = average; lower is worse).",
    "xFIP": "Expected Fielding Independent Pitching — ERA estimate using HR rate, walks, strikeouts; removes luck.",
    "SIERA": "Skill-Interactive ERA — ERA predictor that accounts for ground balls and strikeout/walk balance.",
    "Stuff+": "Pitch quality metric (100 = average); measures movement, velocity, and spin vs league baseline.",
    "Net SGP": "Trade SGP gain minus SGP lost — the net standings-point change from a proposed trade.",
    "Stream Score": "0-100 score for a streaming pitcher start: matchup, park, offense, form, and risk factors.",
    "Heat": "Hot-streak signal — player significantly outperforming recent rolling average across key categories.",
    "% JOB": "Closer Job % — probability this reliever gets save opportunities in the current closer role.",
    "Smash": "High-contact, high-power composite signal; elevated barrel rate and hard-hit rate vs league avg.",
    "Magic#": "Magic Number — wins + opponent losses needed to clinch a playoff spot or category lead.",
    "SOS": "Strength of Schedule — aggregate difficulty of upcoming matchups (pitching or offense faced).",
    "Sell-High": "Player whose recent production exceeds true talent; good trade-away candidate before regression.",
    "Buy-Low": "Player trading below true talent due to small sample or slump; target in trades or FA pickups.",
    "gmLI": "Game Leverage Index — average game-pressure weight of a reliever's appearances; 1.0 = average.",
    "ECR": "Expert Consensus Rank — average ranking across multiple fantasy experts (lower rank = better).",
    "ADP": "Average Draft Position — where this player typically gets selected across real fantasy drafts.",
}


def jargon_help(term: str) -> str:
    """Return the one-line definition for *term* from JARGON, or '' if unknown.

    Suitable for use as a ``help=`` tooltip string in any Streamlit widget.
    """
    return JARGON.get(term, "")


def render_glossary_expander(
    terms: list[str] | None = None,
    *,
    label: str = "What do these numbers mean?",
) -> None:
    """Render a collapsed st.expander listing plain-English definitions.

    Args:
        terms: List of JARGON keys to include. When None (default), all
               entries in JARGON are shown in insertion order.
        label: Expander header text (default "What do these numbers mean?").
    """
    if st is None:
        return

    keys = list(JARGON.keys()) if terms is None else list(terms)

    with st.expander(label, expanded=False):
        for key in keys:
            defn = JARGON.get(key, "")
            if defn:
                st.markdown(
                    f"**{_html.escape(key)}** — {_html.escape(defn)}",
                    unsafe_allow_html=False,
                )
            # Unknown terms are silently skipped so callers can pass a superset.


def format_stat(value: float, stat_type: str) -> str:
    """Unified stat formatting across all pages.

    AVG/OBP: .3f (e.g., .289)
    ERA/WHIP: .2f (e.g., 3.50)
    SGP: +.2f (e.g., +1.50)
    Counting stats: .0f (e.g., 25)
    Percentage: .1f% (e.g., 22.5%)
    """
    if stat_type in ("AVG", "OBP", "avg", "obp"):
        return f"{value:.3f}"[1:] if 0 < value < 1 else f"{value:.3f}"
    elif stat_type in ("ERA", "WHIP", "era", "whip"):
        return f"{value:.2f}"
    elif stat_type in ("SGP", "sgp"):
        return f"{value:+.2f}"
    elif stat_type in ("PCT", "pct", "K%", "BB%"):
        return f"{value:.1f}%"
    else:
        return f"{int(value)}" if value == int(value) else f"{value:.1f}"

"""Trends -- Hot/cold player detection using trend analysis and sustainability scoring."""

import logging

import pandas as pd
import streamlit as st

from src.database import init_db, load_player_pool, load_season_stats
from src.trend_tracker import (
    _HAS_KALMAN,
    _HAS_SUSTAINABILITY,
    compute_player_trends,
    detect_sell_high_candidates,
)
from src.ui_shared import THEME, inject_custom_css, render_styled_table
from src.valuation import LeagueConfig

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Heater | Trends", page_icon="", layout="wide")

init_db()

inject_custom_css()

st.markdown(
    '<div class="page-title-wrap"><div class="page-title"><span>PLAYER TRENDS</span></div></div>',
    unsafe_allow_html=True,
)

# ── Load Data ────────────────────────────────────────────────────────

pool = load_player_pool()
if pool.empty:
    st.warning("No player data loaded.")
    st.stop()

pool = pool.rename(columns={"name": "player_name"})
config = LeagueConfig()

season_stats = load_season_stats()

if season_stats.empty:
    st.info("Season stats not available yet. Stats will load once the MLB season begins.")
    st.stop()

# ── Compute Trends ───────────────────────────────────────────────────

trended = compute_player_trends(pool, season_stats, config)

if trended.empty:
    st.info("No player trend data available.")
    st.stop()

# ── Trend Color Styling ─────────────────────────────────────────────

_TREND_COLORS = {
    "HOT": THEME["green"],
    "COLD": THEME["danger"],
    "NEUTRAL": THEME["tx2"],
}


def _color_trend_label(label: str) -> str:
    """Wrap trend label in a colored span for HTML rendering."""
    color = _TREND_COLORS.get(label, THEME["tx2"])
    return f'<span style="color:{color};font-weight:700;">{label}</span>'


def _color_trend_delta(delta: float) -> str:
    """Format trend delta with color coding."""
    if delta > 0.10:
        color = THEME["green"]
    elif delta < -0.10:
        color = THEME["danger"]
    else:
        color = THEME["tx2"]
    sign = "+" if delta > 0 else ""
    return f'<span style="color:{color};font-weight:600;">{sign}{delta:.3f}</span>'


# ── Position Filter Pills ────────────────────────────────────────────

positions = ["All", "C", "1B", "2B", "3B", "SS", "OF", "SP", "RP"]
pill_cols = st.columns(len(positions))
pos_filter = st.session_state.get("trend_pos_filter", "All")

for i, pos in enumerate(positions):
    with pill_cols[i]:
        btn_type = "primary" if pos_filter == pos else "secondary"
        if st.button(pos, key=f"trend_pill_{pos}", type=btn_type, width="stretch"):
            st.session_state.trend_pos_filter = pos
            st.rerun()


def _filter_by_position(df: pd.DataFrame, pos: str) -> pd.DataFrame:
    """Filter DataFrame by position string matching."""
    if pos == "All":
        return df
    return df[df["positions"].apply(lambda p: pos in str(p).split(",") if pd.notna(p) else False)]


filtered = _filter_by_position(trended, pos_filter)

# ── Capability Info ──────────────────────────────────────────────────

capabilities = []
if _HAS_KALMAN:
    capabilities.append("Kalman filter")
if _HAS_SUSTAINABILITY:
    capabilities.append("Sustainability scoring")
if capabilities:
    cap_text = ", ".join(capabilities)
    st.caption(f"Active analytics: {cap_text}")

# ── Name Column Detection ───────────────────────────────────────────

name_col = "player_name" if "player_name" in filtered.columns else "name"

# ── Tabs ─────────────────────────────────────────────────────────────

tab_hot, tab_cold, tab_sell_high = st.tabs(["Hot List", "Cold List", "Sell-High Targets"])

# ── Hot List Tab ─────────────────────────────────────────────────────

with tab_hot:
    hot_players = filtered[filtered["trend_label"] == "HOT"].copy()
    hot_players = hot_players.sort_values("trend_delta", ascending=False).head(15)

    if hot_players.empty:
        if pos_filter != "All":
            st.info(f"No hot players found at position: {pos_filter}")
        else:
            st.info("No players are currently trending hot.")
    else:
        st.markdown(
            f'<p style="color:{THEME["tx2"]};margin-bottom:12px;">'
            f"Players performing significantly above their pre-season projections. "
            f"Positive trend delta indicates outperformance across key rate stats.</p>",
            unsafe_allow_html=True,
        )

        display_hot = pd.DataFrame()
        display_hot["Player"] = hot_players[name_col].values
        display_hot["Position"] = hot_players["positions"].values
        display_hot["Team"] = hot_players["team"].values
        display_hot["Trend Delta"] = [_color_trend_delta(d) for d in hot_players["trend_delta"]]
        display_hot["Trend"] = [_color_trend_label(t) for t in hot_players["trend_label"]]

        # Add key stat context
        if "is_hitter" in hot_players.columns:
            key_stats = []
            for _, row in hot_players.iterrows():
                is_h = int(row.get("is_hitter", 1) or 1)
                if is_h:
                    avg_val = float(row.get("avg", 0) or 0)
                    hr_val = int(row.get("hr", 0) or 0)
                    key_stats.append(f".{int(avg_val * 1000):03d} AVG / {hr_val} HR")
                else:
                    era_val = float(row.get("era", 0) or 0)
                    k_val = int(row.get("k", 0) or 0)
                    key_stats.append(f"{era_val:.2f} ERA / {k_val} K")
            display_hot["Key Stats"] = key_stats

        render_styled_table(display_hot, max_height=500)

# ── Cold List Tab ────────────────────────────────────────────────────

with tab_cold:
    cold_players = filtered[filtered["trend_label"] == "COLD"].copy()
    cold_players = cold_players.sort_values("trend_delta", ascending=True).head(15)

    if cold_players.empty:
        if pos_filter != "All":
            st.info(f"No cold players found at position: {pos_filter}")
        else:
            st.info("No players are currently trending cold.")
    else:
        st.markdown(
            f'<p style="color:{THEME["tx2"]};margin-bottom:12px;">'
            f"Players performing significantly below their pre-season projections. "
            f"Negative trend delta indicates underperformance across key rate stats.</p>",
            unsafe_allow_html=True,
        )

        display_cold = pd.DataFrame()
        display_cold["Player"] = cold_players[name_col].values
        display_cold["Position"] = cold_players["positions"].values
        display_cold["Team"] = cold_players["team"].values
        display_cold["Trend Delta"] = [_color_trend_delta(d) for d in cold_players["trend_delta"]]
        display_cold["Trend"] = [_color_trend_label(t) for t in cold_players["trend_label"]]

        if "is_hitter" in cold_players.columns:
            key_stats = []
            for _, row in cold_players.iterrows():
                is_h = int(row.get("is_hitter", 1) or 1)
                if is_h:
                    avg_val = float(row.get("avg", 0) or 0)
                    hr_val = int(row.get("hr", 0) or 0)
                    key_stats.append(f".{int(avg_val * 1000):03d} AVG / {hr_val} HR")
                else:
                    era_val = float(row.get("era", 0) or 0)
                    k_val = int(row.get("k", 0) or 0)
                    key_stats.append(f"{era_val:.2f} ERA / {k_val} K")
            display_cold["Key Stats"] = key_stats

        render_styled_table(display_cold, max_height=500)

# ── Sell-High Targets Tab ────────────────────────────────────────────

with tab_sell_high:
    sell_high = detect_sell_high_candidates(pool, season_stats, config)

    if pos_filter != "All":
        sell_high = _filter_by_position(sell_high, pos_filter)

    if sell_high.empty:
        if not _HAS_SUSTAINABILITY:
            st.info(
                "Sell-high detection requires sustainability scoring. "
                "Install the waiver wire module for full functionality."
            )
        elif pos_filter != "All":
            st.info(f"No sell-high candidates found at position: {pos_filter}")
        else:
            st.info("No sell-high candidates identified. Hot players appear to have sustainable performance.")
    else:
        st.markdown(
            f'<p style="color:{THEME["tx2"]};margin-bottom:12px;">'
            f"Players trending hot but with low sustainability scores. "
            f"High BABIP or unsustainable peripherals suggest regression is likely. "
            f"Consider trading these players at peak perceived value.</p>",
            unsafe_allow_html=True,
        )

        sell_high_top = sell_high.head(15)

        display_sell = pd.DataFrame()
        display_sell["Player"] = sell_high_top[name_col].values
        display_sell["Position"] = sell_high_top["positions"].values
        display_sell["Team"] = sell_high_top["team"].values
        display_sell["Trend Delta"] = [_color_trend_delta(d) for d in sell_high_top["trend_delta"]]
        display_sell["Trend"] = [_color_trend_label(t) for t in sell_high_top["trend_label"]]
        display_sell["Sustainability"] = [
            f'<span style="color:{THEME["danger"]};font-weight:600;">{s:.2f}</span>'
            for s in sell_high_top["sustainability_score"]
        ]

        if "is_hitter" in sell_high_top.columns:
            key_stats = []
            for _, row in sell_high_top.iterrows():
                is_h = int(row.get("is_hitter", 1) or 1)
                if is_h:
                    avg_val = float(row.get("avg", 0) or 0)
                    hr_val = int(row.get("hr", 0) or 0)
                    key_stats.append(f".{int(avg_val * 1000):03d} AVG / {hr_val} HR")
                else:
                    era_val = float(row.get("era", 0) or 0)
                    k_val = int(row.get("k", 0) or 0)
                    key_stats.append(f"{era_val:.2f} ERA / {k_val} K")
            display_sell["Key Stats"] = key_stats

        render_styled_table(display_sell, max_height=500)

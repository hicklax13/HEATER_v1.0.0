"""Punt Analyzer — Simulate different category punt strategies."""

import html as _html
import logging
import time

import pandas as pd
import streamlit as st

from src.ai.chat import render_chat_widget
from src.auth import multi_user_enabled, require_auth, resolve_viewer_team_name
from src.database import init_db, load_player_pool
from src.feature_flags import require_page_enabled
from src.feedback import render_feedback_widget
from src.standings_utils import filter_standings_to_valid_teams
from src.ui_shared import (
    _headshot_img_html,
    build_heatbar_html,
    build_stat_readout_html,
    format_stat,
    inject_custom_css,
    render_data_freshness_chip,
    render_empty_state,
    render_page_header,
    render_panel,
    team_color,
    team_logo_url,
)
from src.usage import log_page_view
from src.valuation import LeagueConfig, SGPCalculator
from src.yahoo_data_service import get_yahoo_data_service

_HAS_CATEGORY_ANALYSIS = True
try:
    from src.engine.portfolio.category_analysis import category_gap_analysis  # noqa: F401
except ImportError:
    _HAS_CATEGORY_ANALYSIS = False

logger = logging.getLogger(__name__)

if not multi_user_enabled():
    st.set_page_config(page_title="Heater | Punt Analyzer", page_icon="", layout="wide")

init_db()
inject_custom_css()
require_auth()
require_page_enabled("page:10_Punt_Analyzer")
log_page_view("Punt Analyzer")
render_chat_widget("Punt Analyzer")

render_page_header(
    "Punt Analyzer",
    eyebrow="STRATEGY",
    fig="FIG.10 — CATEGORY PUNT MODEL",
)
render_data_freshness_chip("projections")
st.markdown('<div class="hr-heat"></div>', unsafe_allow_html=True)

pool = load_player_pool()
if pool.empty:
    st.warning("No player data loaded.")
    st.stop()

pool = pool.rename(columns={"name": "player_name"})
config = LeagueConfig()


# ── Branded value-swing table (Combustion instrument look) ────────────────────
# build_compact_table_html escapes cell values, so logos/headshots can't be
# injected there. This renders an inline .rtbl-style table with a team-color
# accent + headshot + logo per row and Archivo tabular SGP figures.
def _value_swing_table_html(df: pd.DataFrame, *, gain: bool) -> str:
    """Build a branded player value-change table for the punt gainers/losers."""
    th = (
        "font-family:var(--font-display);font-weight:800;font-size:10.5px;"
        "letter-spacing:.06em;text-transform:uppercase;color:var(--fp-tx);"
        "padding:9px 12px;border-bottom:2px solid rgba(24,26,32,.18);"
    )
    head = (
        f'<th style="{th}text-align:left;">Player</th>'
        f'<th style="{th}text-align:center;">Pos</th>'
        f'<th style="{th}text-align:right;">Original</th>'
        f'<th style="{th}text-align:right;">Punt</th>'
        f'<th style="{th}text-align:right;">Change</th>'
    )
    td_num = (
        "padding:8px 12px;border-bottom:1px solid var(--fp-divider);text-align:right;"
        "font-family:var(--font-display);font-weight:700;font-size:13.5px;"
        "font-variant-numeric:tabular-nums;color:var(--fp-tx);"
    )
    delta_color = "var(--fp-ember)" if gain else "var(--fp-cold)"
    rows = ""
    for _, r in df.iterrows():
        team = r.get("team")
        tc = team_color(team)
        logo = team_logo_url(team)
        abbr = _html.escape(str(team or "").strip().upper())
        name = _html.escape(str(r.get("player_name", "")))
        headshot = _headshot_img_html(r.get("mlb_id"), size=34)
        pos_txt = _html.escape(str(r.get("positions") or "").split(",")[0].split("/")[0].strip())
        pcell = (
            f'<span style="display:flex;align-items:center;gap:11px;">{headshot}'
            f'<span style="display:flex;flex-direction:column;gap:1px;min-width:0;">'
            f'<b style="font-family:var(--font-body);font-weight:600;font-size:13.5px;color:var(--fp-tx);">{name}</b>'
            f'<span style="font-family:var(--font-mono);font-size:10px;color:var(--fp-tx-subtle);'
            f'letter-spacing:.04em;display:flex;align-items:center;gap:5px;">'
            f'<img src="{logo}" width="13" height="13" style="vertical-align:-2px;" loading="lazy" '
            f"onerror=\"this.style.display='none'\" />{abbr}</span></span></span>"
        )
        rows += (
            f'<tr><td style="padding:8px 12px;border-bottom:1px solid var(--fp-divider);'
            f"background:color-mix(in srgb,{tc} 8%,transparent);box-shadow:inset 3px 0 0 {tc};"
            f'">{pcell}</td>'
            f'<td style="padding:8px 12px;border-bottom:1px solid var(--fp-divider);text-align:center;">'
            f'<span style="font-family:var(--font-mono);font-size:9.5px;font-weight:600;color:var(--fp-tx-muted);'
            f'background:var(--fp-surface);border:1px solid var(--fp-border);border-radius:5px;padding:2px 7px;">{pos_txt}</span></td>'
            f'<td style="{td_num}color:var(--fp-tx-muted);">{format_stat(float(r.get("original_sgp", 0) or 0), "SGP")}</td>'
            f'<td style="{td_num}">{format_stat(float(r.get("punt_sgp", 0) or 0), "SGP")}</td>'
            f'<td style="{td_num}color:{delta_color};">{format_stat(float(r.get("value_change", 0) or 0), "SGP")}</td></tr>'
        )
    return (
        '<table class="rtbl" style="width:100%;border-collapse:collapse;margin-top:4px;">'
        f"<thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"
    )


# ── Category Selection ──────────────────────────────────────────────────────

st.markdown(
    "Select categories to **punt** (ignore). The tool will recompute player values "
    "and show how your roster and available players change in value."
)

all_cats = config.all_categories
punt_cats = st.multiselect(
    "Categories to punt:",
    options=all_cats,
    default=[],
    key="punt_cats",
    help="Select categories you want to intentionally concede in your H2H matchups.",
)

if not punt_cats:
    st.info("Select one or more categories above to see the punt analysis.")
    st.stop()

# ── Compute Original Values ─────────────────────────────────────────────────

progress = st.progress(0, text="Computing original player values...")

try:
    # Original values
    progress.progress(20, text="Computing baseline values...")
    sgp_orig = SGPCalculator(config)

    # Punt-adjusted values: zero out punted category denominators
    punt_config = LeagueConfig()
    # Set punted category weights to effectively zero by using huge denominators
    for cat in punt_cats:
        punt_config.sgp_denominators[cat] = 999999.0

    progress.progress(50, text="Computing punt-adjusted values...")
    sgp_punt = SGPCalculator(punt_config)

    # Compute total SGP for each player under both strategies (vectorized batch).
    pool = pool.copy()
    pool["original_sgp"] = sgp_orig.total_sgp_batch(pool)
    pool["punt_sgp"] = sgp_punt.total_sgp_batch(pool)
    pool["value_change"] = pool["punt_sgp"] - pool["original_sgp"]

    progress.progress(100, text="Analysis complete!")
except Exception as e:
    progress.empty()
    logger.exception("Punt analysis failed")
    st.error(f"Analysis failed: {e}")
    st.stop()

time.sleep(0.3)
progress.empty()

# ── Summary ─────────────────────────────────────────────────────────────────

active_cats = [c for c in all_cats if c not in punt_cats]
_punt_chips = "".join(
    f'<span class="chip cold" style="margin:0 6px 6px 0;">{_html.escape(c)}</span>' for c in punt_cats
)
_active_chips = "".join(
    f'<span class="chip hot" style="margin:0 6px 6px 0;">{_html.escape(c)}</span>' for c in active_cats
)
_readouts = (
    '<div style="display:flex;gap:32px;flex-wrap:wrap;margin-bottom:16px;">'
    + build_stat_readout_html("Punted", len(punt_cats), accent=True)
    + build_stat_readout_html("Active", len(active_cats))
    + build_stat_readout_html("Categories", len(all_cats))
    + "</div>"
)
_summary_body = (
    _readouts + '<div style="font-family:var(--font-display);font-size:9px;font-weight:800;letter-spacing:.2em;'
    'text-transform:uppercase;color:var(--fp-tx-muted);margin-bottom:6px;">Punted</div>'
    + f'<div style="margin-bottom:14px;">{_punt_chips}</div>'
    + '<div style="font-family:var(--font-display);font-size:9px;font-weight:800;letter-spacing:.2em;'
    'text-transform:uppercase;color:var(--fp-tx-muted);margin-bottom:6px;">Active</div>' + f"<div>{_active_chips}</div>"
)
render_panel(
    f"Punting {', '.join(punt_cats)}",
    _summary_body,
    fig_label="FIG.10.1 — STRATEGY",
    accent="top",
)

# ── Deep-link: browse Free Agents for active (non-punted) categories ─────────
# Stash active categories in session_state so the Free Agents page can read
# them and pre-filter recommendations.
st.session_state["_punt_active_cats"] = active_cats
st.page_link(
    "pages/14_Free_Agents.py",
    label=f"Browse Free Agents for active categories ({', '.join(active_cats[:4])}{'…' if len(active_cats) > 4 else ''}) →",
    help="Opens the Free Agents page. Your active (non-punted) categories are saved so you can filter recommendations there.",
)

# ── Task 3.5: "My Roster only" lens ──────────────────────────────────────────
# Fetch rosters early so they're available for the lens toggle and
# the standings-impact panel that follows.
_yds = get_yahoo_data_service()
_rosters_for_lens = _yds.get_rosters()
_user_team_for_lens = None
_user_roster_names: set[str] = set()
if not _rosters_for_lens.empty:
    _user_team_for_lens = resolve_viewer_team_name(_rosters_for_lens)
    if _user_team_for_lens:
        _user_team_for_lens = str(_user_team_for_lens)
        _my_rows = _rosters_for_lens[_rosters_for_lens["team_name"] == _user_team_for_lens]
        _name_col = "player_name" if "player_name" in _my_rows.columns else "name"
        if _name_col in _my_rows.columns:
            _user_roster_names = set(_my_rows[_name_col].dropna().astype(str))

_show_my_roster_only = st.toggle(
    "My Roster only",
    value=False,
    key="punt_my_roster_only",
    help="Show only players on your current roster in the value-swing tables below.",
)

# Apply roster filter when the lens is active
if _show_my_roster_only and _user_roster_names:
    _display_pool = pool[pool["player_name"].isin(_user_roster_names)].copy()
    if _display_pool.empty:
        st.info("No roster players found in the player pool for the value-swing tables.")
        _display_pool = pool  # fall back gracefully
else:
    _display_pool = pool

# ── Biggest Winners (value increases under punt) ────────────────────────────

_gainers_raw = _display_pool.nlargest(15, "value_change")[
    ["player_name", "positions", "team", "mlb_id", "original_sgp", "punt_sgp", "value_change"]
].copy()
gainers = _gainers_raw[_gainers_raw["value_change"] > 0]

if gainers.empty:
    render_panel(
        "Biggest Value Gainers Under Punt",
        '<div style="font-size:12px;color:var(--fp-tx-muted);margin-bottom:8px;">'
        "Players whose value increases most when the punted categories are removed.</div>",
        fig_label="FIG.10.2 — VALUE UP",
        accent="top",
    )
    render_empty_state(
        "No value gainers",
        "No players gain value when punting these categories — all players contribute equally (or less) without them.",
        icon_key="bar_chart",
    )
else:
    render_panel(
        "Biggest Value Gainers Under Punt",
        '<div style="font-size:12px;color:var(--fp-tx-muted);margin-bottom:8px;">'
        "Players whose value increases most when the punted categories are removed.</div>"
        + _value_swing_table_html(gainers, gain=True),
        fig_label="FIG.10.2 — VALUE UP",
        accent="top",
    )

# ── Biggest Losers (value decreases under punt) ────────────────────────────

losers = _display_pool.nsmallest(15, "value_change")[
    ["player_name", "positions", "team", "mlb_id", "original_sgp", "punt_sgp", "value_change"]
].copy()
render_panel(
    "Biggest Value Losers Under Punt",
    '<div style="font-size:12px;color:var(--fp-tx-muted);margin-bottom:8px;">'
    "Players whose value decreases when the punted categories are removed — potential trade targets.</div>"
    + _value_swing_table_html(losers, gain=False),
    fig_label="FIG.10.3 — VALUE DOWN",
    accent="top",
)

# ── Standings Impact ────────────────────────────────────────────────────────
# Task 3.6: wrap the entire compute in try/except and surface failures
# visibly via st.warning / render_empty_state instead of a bare pass.

if _HAS_CATEGORY_ANALYSIS:
    try:
        # Use rosters already fetched for the lens above (3-tier cache hit).
        standings = _yds.get_standings()
        rosters = _rosters_for_lens

        if standings.empty or rosters.empty:
            st.warning(
                "Standings Impact panel requires league standings and roster data. "
                "Sync your Yahoo league to enable this view."
            )
        else:
            user_team_name = _user_team_for_lens or resolve_viewer_team_name(rosters)
            if not user_team_name:
                st.warning(
                    "Standings Impact: could not identify your team. "
                    "Make sure your Yahoo account is connected and your team is assigned."
                )
            else:
                user_team_name = str(user_team_name)

                # MS-C5: drop ghost teams (present in standings cache but absent from
                # current rosters) so n_teams and per-category ranks are correct.
                _valid_teams = (
                    set(rosters["team_name"].astype(str).unique()) if "team_name" in rosters.columns else None
                )
                _standings_filtered = filter_standings_to_valid_teams(standings, _valid_teams)

                all_team_totals: dict[str, dict[str, float]] = {}
                if "category" in _standings_filtered.columns:
                    for _, row in _standings_filtered.iterrows():
                        team = str(row.get("team_name", ""))
                        cat = str(row.get("category", "")).strip()
                        val = float(row.get("total", 0))
                        all_team_totals.setdefault(team, {})[cat] = val

                user_totals = all_team_totals.get(user_team_name, {})

                if not user_totals or not all_team_totals:
                    render_empty_state(
                        "Standings data not available",
                        "Your team's category totals could not be found. Try refreshing Yahoo data via the sidebar.",
                        icon_key="bar_chart",
                    )
                else:
                    _n_teams = max(len(all_team_totals), 2)
                    impact_rows = []
                    impact_html_rows = ""
                    _th = (
                        "font-family:var(--font-display);font-weight:800;font-size:10.5px;"
                        "letter-spacing:.06em;text-transform:uppercase;color:var(--fp-tx);"
                        "padding:9px 12px;border-bottom:2px solid rgba(24,26,32,.18);"
                    )
                    for cat in all_cats:
                        rank_vals = sorted(
                            [t.get(cat, 0) for t in all_team_totals.values()],
                            reverse=(cat not in config.inverse_stats),
                        )
                        my_val = user_totals.get(cat, 0)
                        my_rank = 1
                        for v in rank_vals:
                            if cat in config.inverse_stats:
                                if v < my_val:
                                    my_rank += 1
                            else:
                                if v > my_val:
                                    my_rank += 1

                        is_punt = cat in punt_cats
                        status = "PUNT" if is_punt else "ACTIVE"
                        impact_rows.append({"Status": status, "Rank": f"{my_rank}/{_n_teams}"})

                        # Rank-strength heat bar: rank 1 = best = full bar.
                        _strength = max(0.0, (_n_teams - my_rank + 1) / _n_teams * 100.0)
                        _bar = build_heatbar_html(_strength, win=(not is_punt and my_rank <= _n_teams / 2))
                        _val_str = format_stat(my_val, cat) if cat in config.rate_stats else f"{my_val:.0f}"
                        _status_chip = f'<span class="chip {"cold" if is_punt else "hot"}">{status}</span>'
                        impact_html_rows += (
                            f'<tr><td style="padding:8px 12px;border-bottom:1px solid var(--fp-divider);'
                            f"font-family:var(--font-display);font-weight:800;font-size:13px;color:var(--fp-tx);"
                            f'">{_html.escape(cat)}</td>'
                            f'<td style="padding:8px 12px;border-bottom:1px solid var(--fp-divider);text-align:right;'
                            f"font-family:var(--font-display);font-weight:700;font-size:13px;"
                            f'font-variant-numeric:tabular-nums;color:var(--fp-tx);">{_val_str}</td>'
                            f'<td style="padding:8px 12px;border-bottom:1px solid var(--fp-divider);text-align:right;'
                            f"font-family:var(--font-mono);font-weight:600;font-size:12px;color:var(--fp-tx-muted);"
                            f'">{my_rank}/{_n_teams}</td>'
                            f'<td style="padding:8px 12px;border-bottom:1px solid var(--fp-divider);width:140px;">{_bar}</td>'
                            f'<td style="padding:8px 12px;border-bottom:1px solid var(--fp-divider);'
                            f'text-align:right;">{_status_chip}</td></tr>'
                        )

                    _impact_table = (
                        '<table class="rtbl" style="width:100%;border-collapse:collapse;margin-top:4px;">'
                        f'<thead><tr><th style="{_th}text-align:left;">Category</th>'
                        f'<th style="{_th}text-align:right;">Your Total</th>'
                        f'<th style="{_th}text-align:right;">Rank</th>'
                        f'<th style="{_th}text-align:left;">Strength</th>'
                        f'<th style="{_th}text-align:right;">Status</th></tr></thead>'
                        f"<tbody>{impact_html_rows}</tbody></table>"
                    )
                    render_panel(
                        "Standings Impact",
                        '<div style="font-size:12px;color:var(--fp-tx-muted);margin-bottom:8px;">'
                        "How your current standings ranks read when focusing only on non-punted "
                        "categories. The strength bar fills toward rank 1.</div>" + _impact_table,
                        fig_label="FIG.10.4 — RANKS",
                        accent="top",
                    )

                    # Compute effective standings points (only from active categories).
                    # Use (_n_teams + 1 - rank) so points scale correctly for 12 teams:
                    # rank 1 → _n_teams pts, rank _n_teams → 1 pt (never 0 for last place).
                    active_points = sum(
                        _n_teams + 1 - int(r["Rank"].split("/")[0]) for r in impact_rows if r["Status"] == "ACTIVE"
                    )
                    punt_points = sum(
                        _n_teams + 1 - int(r["Rank"].split("/")[0]) for r in impact_rows if r["Status"] == "PUNT"
                    )
                    total_points = active_points + punt_points

                    st.metric(
                        "Standings Points from Active Categories",
                        f"{active_points}",
                        help=f"Total points: {total_points} (active: {active_points}, punted: {punt_points})",
                    )
    except Exception as _si_exc:
        logger.warning("Standings Impact panel failed: %s", _si_exc)
        st.warning(
            f"Standings Impact panel could not be computed ({type(_si_exc).__name__}). "
            "This may indicate missing league data — try refreshing Yahoo data via the sidebar."
        )

render_feedback_widget("Punt Analyzer")

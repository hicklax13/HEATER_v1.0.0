"""Trade Analyzer -- Evaluate trade proposals with Phase 1 engine pipeline.

4-tab layout:
  1. Evaluate Trade  -- manual trade builder with category need + efficiency
  2. Target a Player -- lowball + fair value proposals for any league player
  3. Browse Partners  -- opponent roster + category comparison explorer
  4. Smart Recommendations -- auto-scanned need-efficient trades across league

Uses the new engine modules for marginal SGP elasticity, punt detection,
LP lineup optimization, and deterministic grading (A+ to F).
Falls back to the legacy analyzer if the engine is unavailable.
"""

import logging
import time

import pandas as pd
import streamlit as st

from src.database import init_db, load_player_pool
from src.in_season import _roster_category_totals
from src.injury_model import get_injury_badge
from src.league_manager import get_team_roster
from src.ui_shared import (
    METRIC_TOOLTIPS,
    PAGE_ICONS,
    T,
    inject_custom_css,
    page_timer_footer,
    page_timer_start,
    render_compact_table,
    render_context_card,
    render_context_columns,
    render_data_freshness_card,
    render_page_layout,
    render_player_select,
    render_sortable_table,
)
from src.valuation import LeagueConfig
from src.yahoo_data_service import get_yahoo_data_service

logger = logging.getLogger(__name__)

try:
    from src.trade_intelligence import (
        apply_scarcity_flags,
        compute_category_need_scores,
        get_health_adjusted_pool,
        score_trade_by_need_efficiency,
    )

    TRADE_INTEL_AVAILABLE = True
except ImportError:
    TRADE_INTEL_AVAILABLE = False

st.set_page_config(
    page_title="Heater | Trade Analyzer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Full display names for stat categories (no abbreviations per project rules)
_CAT_DISPLAY = {
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


# ── Helpers ────────────────────────────────────────────────────────────


def _standings_data_state() -> str:
    """Return quality state of league_standings table: empty / all_zero / ok."""
    try:
        yds = get_yahoo_data_service()
        standings = yds.get_standings()
        if standings.empty:
            return "empty"
        totals = standings.get("total", pd.Series(dtype=float))
        totals = pd.to_numeric(totals, errors="coerce").fillna(0)
        if (totals == 0).all():
            return "all_zero"
        return "ok"
    except Exception:
        return "empty"


def _build_all_team_totals(standings_df, config, league_rosters=None, pool=None):
    """Build {team_name: {cat: total}} from standings or roster projections."""
    all_team_totals: dict[str, dict[str, float]] = {}

    if not standings_df.empty and "team_name" in standings_df.columns and "category" in standings_df.columns:
        standings_df = standings_df.copy()
        standings_df["total"] = pd.to_numeric(standings_df["total"], errors="coerce").fillna(0)
        wide = standings_df.pivot_table(
            index="team_name", columns="category", values="total", aggfunc="first"
        ).reset_index()
        for _, row in wide.iterrows():
            team = row.get("team_name", "")
            if team:
                totals = {}
                for cat in config.all_categories:
                    totals[cat] = float(pd.to_numeric(row.get(cat, 0), errors="coerce") or 0)
                all_team_totals[team] = totals
    elif not standings_df.empty and "team_name" in standings_df.columns:
        for _, row in standings_df.iterrows():
            team = row.get("team_name", "")
            if team:
                totals = {}
                for cat in config.all_categories:
                    totals[cat] = float(pd.to_numeric(row.get(cat, 0), errors="coerce") or 0)
                all_team_totals[team] = totals
    elif league_rosters and pool is not None:
        for team_name, pids in league_rosters.items():
            all_team_totals[team_name] = _roster_category_totals(pids, pool)

    return all_team_totals


def _get_gap_analysis(user_team_name, all_team_totals, weeks_remaining):
    """Run category gap analysis for user, returning empty on failure."""
    if not user_team_name or not all_team_totals:
        return {}
    user_totals = all_team_totals.get(user_team_name, {})
    if not user_totals:
        return {}
    try:
        from src.engine.portfolio.category_analysis import category_gap_analysis

        return category_gap_analysis(
            your_totals=user_totals,
            all_team_totals=all_team_totals,
            your_team_id=user_team_name,
            weeks_remaining=weeks_remaining,
        )
    except Exception:
        return {}


def _build_league_rosters_dict(rosters_df):
    """Build {team_name: [player_ids]} from rosters DataFrame."""
    league_rosters: dict[str, list[int]] = {}
    for _, row in rosters_df.iterrows():
        team = row.get("team_name", "")
        pid = row.get("player_id")
        if team and pid is not None:
            league_rosters.setdefault(team, []).append(int(pid))
    return league_rosters


def _category_rank(cat, user_val, all_team_totals, config):
    """Compute user rank (1 = best) for a single category."""
    all_vals = sorted(
        [t.get(cat, 0) for t in all_team_totals.values()],
        reverse=(cat not in config.inverse_stats),
    )
    rank = 1
    for val in all_vals:
        if cat in config.inverse_stats:
            if val < user_val:
                rank += 1
        else:
            if val > user_val:
                rank += 1
    return rank


def _need_label_and_color(need_score):
    """Return (label, color) for a category need score 0-1."""
    if need_score >= 0.8:
        return "CRITICAL", T["danger"]
    elif need_score >= 0.5:
        return "COMPETITIVE", T["hot"]
    elif need_score >= 0.2:
        return "STRONG", T["ok"]
    else:
        return "PUNT", T["tx2"]


def _arrow_html(val):
    """Return colored arrow span for a category impact value."""
    if val > 0.05:
        return f'<span style="color:{T["ok"]};font-weight:700;">+{val:.2f}</span>'
    elif val < -0.05:
        return f'<span style="color:{T["danger"]};font-weight:700;">{val:.2f}</span>'
    return f'<span style="color:{T["tx2"]};">{val:.2f}</span>'


def _render_proposal_card(label, proposal, color, target_info, pool, config, category_needs):
    """Render a lowball or fair value proposal card."""
    if proposal is None:
        st.markdown(
            f'<div class="glass" style="padding:16px;text-align:center;border:1px solid {T["tx2"]};">'
            f'<div style="font-family:Bebas Neue,sans-serif;font-size:18px;color:{T["tx2"]};">'
            f"No {label} proposal found</div>"
            f'<div style="font-size:12px;color:{T["tx2"]};">No roster player matches the target '
            f"Standings Gained Points range for this tier.</div></div>",
            unsafe_allow_html=True,
        )
        return

    grade = proposal.get("grade", "?")
    efficiency = proposal.get("efficiency", {})
    eff_ratio = efficiency.get("efficiency_ratio", 0.0)
    acceptance = proposal.get("acceptance_probability", 0.0)
    adp_fair = proposal.get("adp_fairness", 0.5)
    ecr_fair = proposal.get("ecr_fairness", 0.5)
    giving_names = proposal.get("giving_names", [])

    # Grade color
    if grade.startswith("A"):
        grade_color = T["ok"]
    elif grade.startswith("B"):
        grade_color = T["sky"]
    elif grade.startswith("C"):
        grade_color = T["hot"]
    else:
        grade_color = T["danger"]

    # Header
    give_text = " + ".join(giving_names)
    st.markdown(
        f'<div class="glass" style="padding:16px;border:2px solid {color};">'
        f'<div style="font-family:Bebas Neue,sans-serif;font-size:20px;color:{color};'
        f'letter-spacing:2px;margin-bottom:8px;">{label}</div>'
        f'<div style="font-size:13px;color:{T["tx"]};margin-bottom:12px;">'
        f"Give: <strong>{give_text}</strong></div>"
        f'<div style="display:flex;gap:24px;flex-wrap:wrap;margin-bottom:12px;">'
        f'<div style="text-align:center;">'
        f'<div style="font-size:11px;color:{T["tx2"]};">Grade</div>'
        f'<div style="font-family:Bebas Neue,sans-serif;font-size:24px;color:{grade_color};">{grade}</div></div>'
        f'<div style="text-align:center;">'
        f'<div style="font-size:11px;color:{T["tx2"]};">Efficiency</div>'
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:16px;color:{T["tx"]};">{eff_ratio:.2f}x</div></div>'
        f'<div style="text-align:center;">'
        f'<div style="font-size:11px;color:{T["tx2"]};">Acceptance</div>'
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:16px;color:{T["tx"]};">{acceptance:.0%}</div></div>'
        f'<div style="text-align:center;">'
        f'<div style="font-size:11px;color:{T["tx2"]};">ADP Fair</div>'
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:16px;color:{T["tx"]};">{adp_fair:.0%}</div></div>'
        f'<div style="text-align:center;">'
        f'<div style="font-size:11px;color:{T["tx2"]};">ECR Fair</div>'
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:16px;color:{T["tx"]};">{ecr_fair:.0%}</div></div>'
        f"</div></div>",
        unsafe_allow_html=True,
    )

    # Category impact table
    cat_impact = proposal.get("category_impact", {})
    if cat_impact:
        rows = []
        for cat in config.all_categories:
            impact_val = cat_impact.get(cat, 0.0)
            need = category_needs.get(cat, 0.5)
            need_label, _ = _need_label_and_color(need)
            rows.append(
                {
                    "Category": _CAT_DISPLAY.get(cat, cat),
                    "Impact": f"{impact_val:+.2f}",
                    "Need": need_label,
                }
            )
        render_compact_table(pd.DataFrame(rows))

    # Historical stats
    hist = proposal.get("historical_stats", {})
    target_hist = target_info.get("historical_stats", {})
    _render_historical_section("Give", giving_names, proposal.get("giving_ids", []), hist, pool)
    _render_historical_section(
        "Receive",
        [target_info.get("name", "?")],
        [target_info.get("player_id", 0)],
        {target_info.get("player_id", 0): target_hist},
        pool,
    )


def _render_historical_section(side_label, names, pids, hist_data, pool):
    """Render 2025 + 2026 stats for a set of players."""
    rows_2026 = []
    rows_2025 = []
    for pid, name in zip(pids, names):
        seasons = hist_data.get(pid, {})
        for year, label_list in [(2026, rows_2026), (2025, rows_2025)]:
            stats = seasons.get(year, {})
            if stats:
                # Detect hitter vs pitcher
                p_row = pool[pool["player_id"] == pid]
                is_hitter = True
                if not p_row.empty:
                    is_hitter = bool(p_row.iloc[0].get("is_hitter", 1))
                if is_hitter:
                    label_list.append(
                        {
                            "Player": name,
                            "Runs": f"{stats.get('r', 0):.0f}",
                            "Home Runs": f"{stats.get('hr', 0):.0f}",
                            "Runs Batted In": f"{stats.get('rbi', 0):.0f}",
                            "Stolen Bases": f"{stats.get('sb', 0):.0f}",
                            "Batting Average": f"{stats.get('avg', 0):.3f}",
                        }
                    )
                else:
                    label_list.append(
                        {
                            "Player": name,
                            "Wins": f"{stats.get('w', 0):.0f}",
                            "Saves": f"{stats.get('sv', 0):.0f}",
                            "Strikeouts": f"{stats.get('k', 0):.0f}",
                            "Earned Run Average": f"{stats.get('era', 0):.2f}",
                            "Walks + Hits per Inning Pitched": f"{stats.get('whip', 0):.2f}",
                        }
                    )

    if rows_2026:
        st.caption(f"{side_label} -- 2026 Year to Date")
        render_compact_table(pd.DataFrame(rows_2026))
    if rows_2025:
        st.caption(f"{side_label} -- 2025 Historical")
        render_compact_table(pd.DataFrame(rows_2025))


# ── Main Page ──────────────────────────────────────────────────────────

init_db()
inject_custom_css()
page_timer_start()

render_page_layout("TRADE ANALYZER", banner_teaser="Analyze a trade below", banner_icon="trade")

# Load data
pool = load_player_pool()
if pool.empty:
    st.warning("No player data. Load sample data or import projections first.")
    st.stop()

pool = pool.rename(columns={"name": "player_name"})

# Apply trade intelligence: health-adjust projections and add scarcity flags
if TRADE_INTEL_AVAILABLE:
    try:
        pool = get_health_adjusted_pool(pool)
        pool = apply_scarcity_flags(pool)
    except Exception:
        pass

config = LeagueConfig()

# Get user team roster
yds = get_yahoo_data_service()
rosters = yds.get_rosters()
if rosters.empty:
    st.warning(
        "No league data loaded. Connect your Yahoo league in Connect League, "
        "or league data will load automatically on next app launch."
    )
    st.stop()

user_teams = rosters[rosters["is_user_team"] == 1]
if user_teams.empty:
    st.warning("No user team identified.")
    st.stop()

user_team_name = user_teams.iloc[0]["team_name"]
if isinstance(user_team_name, bytes):
    user_team_name = user_team_name.decode("utf-8", errors="replace")

user_roster = get_team_roster(user_team_name)

# Player selection -- build name-to-id mapping from both pool AND roster
player_names = pool[["player_id", "player_name"]].drop_duplicates()
name_to_id = dict(zip(player_names["player_name"], player_names["player_id"]))

# Remap roster IDs to player_pool IDs via name matching
user_roster_ids = []
if not user_roster.empty and "name" in user_roster.columns:
    for _, r in user_roster.iterrows():
        pname = r.get("name", "")
        pool_pid = name_to_id.get(pname)
        if pool_pid is not None:
            user_roster_ids.append(pool_pid)
        else:
            user_roster_ids.append(r["player_id"])
else:
    user_roster_ids = user_roster["player_id"].tolist() if not user_roster.empty else []

# Add roster player names to name_to_id
if not user_roster.empty and "name" in user_roster.columns:
    for _, r in user_roster.iterrows():
        pname = r.get("name", "")
        pid = r.get("player_id")
        if pname and pid and pname not in name_to_id:
            name_to_id[pname] = pid

# Build league-wide data
league_rosters = _build_league_rosters_dict(rosters)
standings_df = yds.get_standings()
all_team_totals = _build_all_team_totals(standings_df, config, league_rosters, pool)

# Weeks remaining
try:
    from src.validation.dynamic_context import compute_weeks_remaining

    weeks_remaining = compute_weeks_remaining()
except Exception:
    weeks_remaining = 22

# Category needs
gap_analysis = _get_gap_analysis(user_team_name, all_team_totals, weeks_remaining)
if TRADE_INTEL_AVAILABLE:
    category_needs = compute_category_need_scores(gap_analysis, config)
else:
    category_needs = {cat: 0.5 for cat in config.all_categories}

# Build other-team player options (for Evaluate Trade tab)
other_team_pids = set()
all_teams = rosters["team_name"].unique()
for tn in all_teams:
    if str(tn) == str(user_team_name):
        continue
    team_rows = rosters[rosters["team_name"] == tn]
    for _, tr in team_rows.iterrows():
        pname = tr.get("player_name") or tr.get("name", "")
        matched = name_to_id.get(pname)
        if matched is not None:
            other_team_pids.add(matched)
        else:
            other_team_pids.add(tr.get("player_id"))

# Build team-grouped player options for dropdowns
_team_player_map: dict[str, list[str]] = {}
for tn in all_teams:
    if str(tn) == str(user_team_name):
        continue
    team_rows = rosters[rosters["team_name"] == tn]
    for _, tr in team_rows.iterrows():
        pname = tr.get("player_name") or tr.get("name", "")
        if pname:
            _team_player_map.setdefault(str(tn), []).append(str(pname))

# Build "Team | Player" options sorted by team then player
_receive_options_grouped: list[str] = []
for team_name_key in sorted(_team_player_map.keys()):
    for pname in sorted(_team_player_map[team_name_key]):
        _receive_options_grouped.append(f"{team_name_key} | {pname}")


def _parse_team_player(option_str: str) -> tuple[str, str]:
    """Parse 'TeamName | PlayerName' back into components."""
    parts = option_str.split(" | ", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return "", option_str.strip()


# ── Layout ─────────────────────────────────────────────────────────────

ctx, main = render_context_columns()

with ctx:
    # Category Needs card
    needs_html = ""
    user_totals = all_team_totals.get(user_team_name, {})
    for cat in config.all_categories:
        need = category_needs.get(cat, 0.5)
        rank = _category_rank(cat, user_totals.get(cat, 0), all_team_totals, config) if all_team_totals else "?"
        label, color = _need_label_and_color(need)
        cat_name = _CAT_DISPLAY.get(cat, cat)
        needs_html += (
            f'<div style="font-size:11px!important;padding:2px 0!important;">'
            f'<span style="color:{color}!important;font-weight:600!important">{cat_name}</span> '
            f'<span style="color:{T["tx2"]}!important;">Rank {rank} -- {label}</span></div>'
        )
    render_context_card("Category Needs", needs_html)

    # Recent transactions card
    try:
        _txns = yds.get_transactions()
        if not _txns.empty:
            _txn_rows = ""
            _txn_limit = min(8, len(_txns))
            for _ti in range(_txn_limit):
                _txr = _txns.iloc[_ti]
                _txtype = str(_txr.get("type", "")).replace("_", " ").title()
                _txplayer = str(_txr.get("player_name", ""))[:20]
                _txdest = str(_txr.get("team_to", ""))[:15]
                _txlabel = f"{_txtype} > {_txdest}" if _txdest else _txtype
                _txn_rows += (
                    f'<div style="padding:2px 0!important;font-size:11px!important;'
                    f'font-family:IBM Plex Mono,monospace!important">'
                    f'<span style="color:{T["tx"]}!important;font-weight:600!important">'
                    f"{_txplayer}</span> "
                    f'<span style="color:{T["tx2"]}!important">{_txlabel}</span></div>'
                )
            render_context_card("Recent Transactions", _txn_rows)
    except Exception:
        pass

    render_data_freshness_card()

with main:
    tab1, tab2, tab3, tab4 = st.tabs(["Evaluate Trade", "Target a Player", "Browse Partners", "Smart Recommendations"])

    # ══════════════════════════════════════════════════════════════════
    # TAB 1: Evaluate Trade
    # ══════════════════════════════════════════════════════════════════
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("You Give")
            if not user_roster.empty and "name" in user_roster.columns:
                give_options = sorted(user_roster["name"].dropna().unique().tolist())
            else:
                give_options = sorted(pool[pool["player_id"].isin(user_roster_ids)]["player_name"].tolist())
            giving_names = st.multiselect("Select players to trade away", options=give_options, key="giving")

        with col2:
            st.subheader("You Receive")
            receiving_selections = st.multiselect(
                "Select players to receive (Team | Player)",
                options=_receive_options_grouped,
                key="receiving_grouped",
            )
            receiving_names = [_parse_team_player(s)[1] for s in receiving_selections]

        if st.button("Analyze Trade", type="primary", key="analyze_btn"):
            if not giving_names or not receiving_names:
                st.error("Select at least one player on each side.")
            else:
                giving_ids = [name_to_id[n] for n in giving_names if n in name_to_id]
                receiving_ids = [name_to_id[n] for n in receiving_names if n in name_to_id]

                if not giving_ids or not receiving_ids:
                    st.error("One or more selected players could not be matched. Please reselect.")
                    st.stop()

                trade_progress = st.progress(0, text="Computing category impacts...")

                # Try Phase 1 engine first, fall back to legacy
                try:
                    from src.engine.output.trade_evaluator import evaluate_trade

                    trade_progress.progress(20, text="Running marginal elasticity analysis...")
                    trade_progress.progress(40, text="Computing category gaps and punt detection...")
                    trade_progress.progress(60, text="Optimizing lineup assignments...")

                    result = evaluate_trade(
                        giving_ids=giving_ids,
                        receiving_ids=receiving_ids,
                        user_roster_ids=user_roster_ids,
                        player_pool=pool,
                        config=config,
                        user_team_name=user_team_name,
                        weeks_remaining=weeks_remaining,
                    )
                    engine_used = "phase1"
                except Exception as e:
                    logger.warning("Phase 1 engine failed, falling back to legacy: %s", e)
                    from src.in_season import analyze_trade

                    trade_progress.progress(40, text="Running Monte Carlo simulation (200 iterations)...")
                    result = analyze_trade(
                        giving_ids=giving_ids,
                        receiving_ids=receiving_ids,
                        user_roster_ids=user_roster_ids,
                        player_pool=pool,
                        config=config,
                    )
                    engine_used = "legacy"

                trade_progress.progress(100, text="Trade analysis complete!")
                time.sleep(0.3)
                trade_progress.empty()

                # Update context panel with trade summary
                with ctx:
                    verdict_color = T["ok"] if result["verdict"] == "ACCEPT" else T["danger"]
                    grade_val = result.get("grade", "")
                    grade_display = (
                        f'<div style="font-size:28px;font-family:Bebas Neue,sans-serif;'
                        f'color:{verdict_color};letter-spacing:2px;">{result["verdict"]}</div>'
                    )
                    if grade_val:
                        grade_display += (
                            f'<div style="font-size:20px;font-family:Bebas Neue,sans-serif;'
                            f'color:{verdict_color};">{grade_val}</div>'
                        )
                    grade_display += (
                        f'<div style="font-size:12px;color:#6b7280;margin-top:4px;">'
                        f"{result['confidence_pct']:.2f}% confidence</div>"
                    )
                    render_context_card("Trade Verdict", grade_display)

                    if engine_used == "phase1" and result.get("punt_categories"):
                        punt_list = "".join(
                            f'<div style="font-size:12px;color:#6b7280;padding:2px 0;">{cat}</div>'
                            for cat in result["punt_categories"]
                        )
                        render_context_card(
                            "Punted Categories",
                            f'<div style="font-size:11px;color:#9ca3af;margin-bottom:4px;">'
                            f"Zero-weighted in evaluation</div>{punt_list}",
                        )

                    surplus = (
                        result.get("surplus_sgp") if engine_used == "phase1" else result.get("total_sgp_change", 0)
                    )
                    surplus_label = (
                        "Surplus Standings Gained Points"
                        if engine_used == "phase1"
                        else "Total Standings Gained Points Change"
                    )
                    surplus_color = T["ok"] if (surplus or 0) >= 0 else T["danger"]
                    render_context_card(
                        surplus_label,
                        f'<div style="font-size:20px;font-family:Bebas Neue,sans-serif;'
                        f'color:{surplus_color};">{surplus:+.2f}</div>',
                    )

                # Verdict banner
                if result["verdict"] == "ACCEPT":
                    color = T["ok"]
                    icon = PAGE_ICONS["accept"]
                else:
                    color = T["danger"]
                    icon = PAGE_ICONS["reject"]

                grade_html = ""
                if "grade" in result:
                    grade = result["grade"]
                    if grade.startswith("A"):
                        grade_color = T["ok"]
                    elif grade.startswith("B"):
                        grade_color = T["sky"]
                    elif grade.startswith("C"):
                        grade_color = T["hot"]
                    else:
                        grade_color = T["danger"]
                    grade_html = (
                        f'<span style="font-family:Bebas Neue,sans-serif;font-size:36px;'
                        f"color:{grade_color};letter-spacing:3px;margin-left:16px;"
                        f'font-weight:bold;">{grade}</span>'
                    )

                st.markdown(
                    f'<div class="glass" style="border:2px solid {color};'
                    f"padding:20px;text-align:center;margin:16px 0;"
                    f'animation:slideUp 0.4s ease-out both;">'
                    f"{icon}"
                    f'<span style="font-family:Bebas Neue,sans-serif;font-size:28px;color:{color};'
                    f'letter-spacing:2px;margin-left:12px;">{result["verdict"]}</span>'
                    f"{grade_html}"
                    f'<span style="color:{T["tx2"]};margin-left:12px;font-size:18px;">'
                    f"{result['confidence_pct']:.2f}% confidence</span></div>",
                    unsafe_allow_html=True,
                )
                verdict_key = "trade_verdict" if engine_used == "phase1" else "trade_verdict_legacy"
                st.caption(METRIC_TOOLTIPS[verdict_key])

                # Standings data warning
                _standings_state = _standings_data_state()
                if engine_used == "phase1":
                    if _standings_state == "all_zero":
                        st.caption(
                            "League standings have not yet populated. Trade analysis is using "
                            "league-average category weights instead of standings-based weights. "
                            "Results will be more accurate after week 1."
                        )
                    elif _standings_state == "empty" or not result.get("category_analysis"):
                        st.warning(
                            "No league standings data available. All categories are weighted equally, "
                            "so the analysis cannot account for your team's strategic position "
                            "(punt detection, marginal elasticity). Sync your Yahoo league for "
                            "a more accurate evaluation."
                        )

                # Analytics transparency badge
                if engine_used == "phase1" and "analytics_context" in result:
                    try:
                        from src.ui_analytics_badge import render_analytics_badge

                        render_analytics_badge(result["analytics_context"])
                    except Exception:
                        pass

                # Metrics row
                if engine_used == "phase1":
                    mcols = st.columns(6)
                    mcols[0].metric("Trade Grade", result.get("grade", "N/A"))
                    mcols[1].metric(
                        "Surplus Standings Gained Points",
                        f"{result.get('surplus_sgp', 0):+.2f}",
                        help=METRIC_TOOLTIPS["sgp"],
                    )

                    roster_move = "None"
                    if result.get("drop_candidate"):
                        roster_move = f"Drop {result['drop_candidate']}"
                    elif result.get("fa_pickup"):
                        roster_move = f"Add {result['fa_pickup']}"
                    mcols[2].metric("Roster Move", roster_move, help=METRIC_TOOLTIPS.get("roster_move", ""))
                    mcols[3].metric(
                        "Replacement Penalty",
                        f"{result.get('replacement_penalty', 0):+.2f}",
                        help=METRIC_TOOLTIPS["replacement_penalty"],
                    )
                    mcols[4].metric(
                        "Punted Categories",
                        str(len(result.get("punt_categories", []))),
                        help="Categories where gaining standings positions is impossible. "
                        "These are zero-weighted in the trade evaluation.",
                    )

                    # Efficiency metric (new)
                    if TRADE_INTEL_AVAILABLE:
                        try:
                            eff_result = score_trade_by_need_efficiency(
                                result.get("category_impact", {}), category_needs, config
                            )
                            mcols[5].metric(
                                "Need Efficiency",
                                f"{eff_result.get('efficiency_ratio', 0):.2f}x",
                                help="How efficiently this trade targets your weak categories. "
                                "Values above 1.0 mean you gain more need-weighted value than you lose.",
                            )
                        except Exception:
                            mcols[5].metric("Need Efficiency", "N/A")
                    else:
                        mcols[5].metric("Need Efficiency", "N/A")

                    if result.get("punt_categories"):
                        punt_text = ", ".join(result["punt_categories"])
                        st.info(f"Punted categories (zero-weighted): {punt_text}")
                else:
                    col1, col2, col3 = st.columns(3)
                    col1.metric(
                        "Total Standings Gained Points Change",
                        f"{result['total_sgp_change']:+.2f}",
                        help=METRIC_TOOLTIPS["sgp"],
                    )
                    col2.metric("Monte Carlo Mean", f"{result['mc_mean']:+.2f}", help=METRIC_TOOLTIPS["mc_mean"])
                    col3.metric(
                        "Monte Carlo Standard Deviation", f"{result['mc_std']:.2f}", help=METRIC_TOOLTIPS["mc_std"]
                    )

                # Acceptance Analysis Panel
                try:
                    from src.trade_finder import (
                        acceptance_label,
                        compute_adp_fairness,
                        estimate_acceptance_probability,
                    )

                    st.subheader("Acceptance Analysis")
                    st.caption(
                        "Behavioral model: estimates whether the opponent would accept this trade "
                        "based on draft capital, expert rankings, loss aversion, and opponent needs."
                    )

                    adp_scores = []
                    for gid in giving_ids:
                        for rid in receiving_ids:
                            adp_scores.append(compute_adp_fairness(gid, rid, pool))
                    avg_adp_fairness = sum(adp_scores) / len(adp_scores) if adp_scores else 0.5

                    # ECR fairness
                    ecr_ranks_local = {}
                    try:
                        from src.database import get_connection as _gc_ecr

                        _ecr_conn = _gc_ecr()
                        try:
                            _ecr_df = pd.read_sql_query(
                                "SELECT player_id, consensus_rank FROM ecr_consensus", _ecr_conn
                            )
                            ecr_ranks_local = dict(
                                zip(_ecr_df["player_id"].astype(int), _ecr_df["consensus_rank"].astype(int))
                            )
                        finally:
                            _ecr_conn.close()
                    except Exception:
                        pass

                    ecr_scores = []
                    for gid in giving_ids:
                        for rid in receiving_ids:
                            g_ecr = ecr_ranks_local.get(gid)
                            r_ecr = ecr_ranks_local.get(rid)
                            if g_ecr and r_ecr:
                                ecr_scores.append((min(g_ecr, r_ecr) / max(g_ecr, r_ecr, 1)) ** 0.5)
                    avg_ecr_fairness = sum(ecr_scores) / len(ecr_scores) if ecr_scores else 0.5

                    opp_need_match = 0.5
                    opp_willingness = 0.5
                    opp_rank = None
                    try:
                        from src.opponent_trade_analysis import (
                            compute_opponent_needs,
                            get_opponent_archetype,
                        )

                        _recv_teams = set()
                        for rid in receiving_ids:
                            _rp = pool[pool["player_id"] == rid]
                            if not _rp.empty:
                                _rt = rosters[rosters["player_id"] == rid]
                                if _rt.empty:
                                    _pname = _rp.iloc[0].get("player_name", _rp.iloc[0].get("name", ""))
                                    if _pname:
                                        _name_col = (
                                            rosters["name"]
                                            if "name" in rosters.columns
                                            else rosters.get("player_name", pd.Series(dtype=str))
                                        )
                                        _rt = rosters[_name_col.str.strip() == str(_pname).strip()]
                                if not _rt.empty:
                                    _recv_teams.add(_rt.iloc[0].get("team_name", ""))

                        if _recv_teams:
                            opp_team = list(_recv_teams)[0]
                            if all_team_totals:
                                opp_needs = compute_opponent_needs(opp_team, all_team_totals)
                                arch = get_opponent_archetype(opp_team)
                                opp_willingness = arch.get("trade_willingness", 0.5)

                                opp_standings = standings_df[standings_df["team_name"] == opp_team]
                                if not opp_standings.empty:
                                    opp_rank = int(opp_standings.iloc[0].get("rank", 6))

                                opp_weak = [c for c, info in opp_needs.items() if info.get("rank", 6) >= 8]
                                if opp_weak:
                                    helps = 0
                                    for gid in giving_ids:
                                        gp = pool[pool["player_id"] == gid]
                                        if not gp.empty:
                                            for c in opp_weak:
                                                if float(gp.iloc[0].get(c.lower(), 0) or 0) > 0:
                                                    helps += 1
                                    opp_need_match = helps / (len(opp_weak) * len(giving_ids)) if opp_weak else 0.5
                    except Exception:
                        pass

                    user_sgp = result.get("surplus_sgp", result.get("total_sgp_change", 0))
                    opp_sgp = -user_sgp * 0.5
                    need_match = min(1.0, max(0.0, (opp_sgp + 1.0) / 2.0))

                    p_accept = estimate_acceptance_probability(
                        user_gain_sgp=user_sgp,
                        opponent_gain_sgp=opp_sgp,
                        need_match_score=need_match,
                        adp_fairness=avg_adp_fairness,
                        opponent_need_match=opp_need_match,
                        opponent_standings_rank=opp_rank,
                        opponent_trade_willingness=opp_willingness,
                    )

                    ac1, ac2, ac3, ac4 = st.columns(4)
                    ac1.metric("Acceptance Probability", f"{p_accept:.0%}")
                    ac2.metric("ADP Fairness", f"{avg_adp_fairness:.0%}")
                    ac3.metric("ECR Fairness", f"{avg_ecr_fairness:.0%}")
                    ac4.metric("Acceptance Tier", acceptance_label(p_accept))

                except ImportError:
                    pass
                except Exception as _acc_err:
                    logger.warning("Acceptance analysis failed: %s", _acc_err)

                # Category impact table -- enhanced with Need column
                st.subheader("Category Impact")
                if engine_used == "phase1" and result.get("category_analysis"):
                    impact_rows = []
                    for cat, sgp_val in result["category_impact"].items():
                        analysis = result["category_analysis"].get(cat, {})
                        rank = analysis.get("rank", "-")
                        is_punt = analysis.get("is_punt", False)
                        gap = analysis.get("gap_to_next", 0)
                        status = "PUNT" if is_punt else f"Rank {rank}"
                        need = category_needs.get(cat, 0.5)
                        need_label, _ = _need_label_and_color(need)

                        impact_rows.append(
                            {
                                "Category": _CAT_DISPLAY.get(cat, cat),
                                "Standings Gained Points Change": f"{sgp_val:+.2f}",
                                "Your Rank": status,
                                "Gap to Next": f"{gap:.2f}" if not is_punt else "-",
                                "Category Need": need_label,
                            }
                        )
                    render_compact_table(pd.DataFrame(impact_rows))
                else:
                    impact_df = pd.DataFrame(
                        [
                            {
                                "Category": _CAT_DISPLAY.get(cat, cat),
                                "Standings Gained Points Change": f"{val:+.2f}",
                                "Category Need": (
                                    _need_label_and_color(category_needs.get(cat, 0.5))[0]
                                    if TRADE_INTEL_AVAILABLE
                                    else "-"
                                ),
                            }
                            for cat, val in result["category_impact"].items()
                        ]
                    )
                    render_compact_table(impact_df)

                # Risk flags
                if result["risk_flags"]:
                    st.subheader("Risk Flags")
                    for flag in result["risk_flags"]:
                        st.warning(flag)

                # Player headshots + injury badges
                from src.ui_shared import _headshot_img_html

                trade_col1, trade_col2 = st.columns(2)
                with trade_col1:
                    st.markdown("**Giving:**")
                    for name in giving_names:
                        p = pool[pool["player_name"] == name]
                        if not p.empty:
                            mid = p.iloc[0].get("mlb_id")
                            hs = float(p.iloc[0].get("health_score", 0.85) or 0.85)
                            badge_icon, badge_label = get_injury_badge(hs)
                            shot = _headshot_img_html(mid, size=28)
                            st.markdown(f"{shot}{badge_icon} {name} -- {badge_label}", unsafe_allow_html=True)
                with trade_col2:
                    st.markdown("**Receiving:**")
                    for name in receiving_names:
                        p = pool[pool["player_name"] == name]
                        if not p.empty:
                            mid = p.iloc[0].get("mlb_id")
                            hs = float(p.iloc[0].get("health_score", 0.85) or 0.85)
                            badge_icon, badge_label = get_injury_badge(hs)
                            shot = _headshot_img_html(mid, size=28)
                            st.markdown(f"{shot}{badge_icon} {name} -- {badge_label}", unsafe_allow_html=True)

                # Player card selector
                _trade_all_names = list(giving_names) + list(receiving_names)
                _trade_all_ids = []
                for _tname in _trade_all_names:
                    _tp = pool[pool["player_name"] == _tname]
                    _trade_all_ids.append(int(_tp.iloc[0]["player_id"]) if not _tp.empty else 0)
                if any(pid != 0 for pid in _trade_all_ids):
                    render_player_select(_trade_all_names, _trade_all_ids, key_suffix="trade")

    # ══════════════════════════════════════════════════════════════════
    # TAB 2: Target a Player
    # ══════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("Target a Player")
        st.caption(
            "Select any player in the league and get two trade proposals: a lowball offer and a fair value package."
        )

        target_selection = st.selectbox(
            "Select target player (Team | Player)",
            options=[""] + _receive_options_grouped,
            key="target_player_select",
            index=0,
        )

        if target_selection:
            target_team, target_name = _parse_team_player(target_selection)
            target_pid = name_to_id.get(target_name)

            if target_pid is None:
                st.warning(f"Could not find player ID for {target_name}.")
            else:
                with st.spinner("Generating trade proposals..."):
                    try:
                        from src.trade_intelligence import generate_targeted_proposals

                        proposals = generate_targeted_proposals(
                            target_player_id=target_pid,
                            user_roster_ids=user_roster_ids,
                            player_pool=pool,
                            config=config,
                            all_team_totals=all_team_totals,
                            user_team_name=user_team_name,
                            opponent_team_name=target_team,
                            weeks_remaining=weeks_remaining,
                        )
                    except Exception as e:
                        logger.warning("Targeted proposals failed: %s", e)
                        proposals = {"target": {}, "lowball": None, "fair_value": None}

                target_info = proposals.get("target", {})
                target_display_name = target_info.get("name", target_name)
                target_pos = target_info.get("positions", "")
                target_team_name = target_info.get("team", target_team)
                target_sgp = target_info.get("sgp", 0.0)

                # Target player header
                st.markdown(
                    f'<div class="glass" style="padding:12px;margin-bottom:16px;">'
                    f'<div style="font-family:Bebas Neue,sans-serif;font-size:22px;'
                    f'color:{T["primary"]};letter-spacing:2px;">TARGET: {target_display_name}</div>'
                    f'<div style="font-size:13px;color:{T["tx2"]};">'
                    f"{target_team_name} -- {target_pos} -- "
                    f"Standings Gained Points: {target_sgp:.2f}</div></div>",
                    unsafe_allow_html=True,
                )

                # Two columns: Lowball | Fair Value
                pcol1, pcol2 = st.columns(2)
                with pcol1:
                    _render_proposal_card(
                        "LOWBALL",
                        proposals.get("lowball"),
                        T["hot"],
                        target_info,
                        pool,
                        config,
                        category_needs,
                    )
                with pcol2:
                    _render_proposal_card(
                        "FAIR VALUE",
                        proposals.get("fair_value"),
                        T["ok"],
                        target_info,
                        pool,
                        config,
                        category_needs,
                    )

    # ══════════════════════════════════════════════════════════════════
    # TAB 3: Browse Partners
    # ══════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("Browse Trade Partners")
        st.caption("Explore opponent rosters and compare category strengths to find trade fits.")

        team_names = sorted(rosters["team_name"].unique())
        opponent_teams = [t for t in team_names if str(t) != str(user_team_name)]

        selected_team = st.selectbox("Select opponent team", opponent_teams, key="browse_team_select")

        if selected_team:
            # Complementarity score
            try:
                from src.trade_finder import find_complementary_teams

                partners = find_complementary_teams(user_team_name, all_team_totals, config, top_n=11)
                comp_score = dict(partners).get(selected_team, 0.5)
            except Exception:
                comp_score = 0.5

            st.markdown(
                f'<div style="font-size:13px;color:{T["tx2"]};margin-bottom:12px;">'
                f'Complementarity score: <strong style="color:{T["tx"]};">{comp_score:.2f}</strong> '
                f"(higher = better trade fit)</div>",
                unsafe_allow_html=True,
            )

            # Category comparison table
            opp_totals = all_team_totals.get(selected_team, {})
            if opp_totals and user_totals:
                comp_rows = []
                for cat in config.all_categories:
                    u_val = user_totals.get(cat, 0)
                    o_val = opp_totals.get(cat, 0)
                    u_rank = _category_rank(cat, u_val, all_team_totals, config)
                    o_rank = _category_rank(cat, o_val, all_team_totals, config)
                    gap = u_rank - o_rank
                    # Opportunity: they are strong (low rank) where you are weak (high rank)
                    if gap > 3:
                        opportunity = "HIGH"
                    elif gap > 0:
                        opportunity = "MODERATE"
                    elif gap < -3:
                        opportunity = "THEY NEED"
                    elif gap < 0:
                        opportunity = "SLIGHT"
                    else:
                        opportunity = "EVEN"

                    comp_rows.append(
                        {
                            "Category": _CAT_DISPLAY.get(cat, cat),
                            "Your Rank": u_rank,
                            "Their Rank": o_rank,
                            "Rank Gap": gap,
                            "Opportunity": opportunity,
                        }
                    )
                st.markdown("**Category Comparison**")
                render_compact_table(pd.DataFrame(comp_rows))

            # Opponent roster table
            team_roster_df = rosters[rosters["team_name"] == selected_team].copy()
            if not team_roster_df.empty:
                st.markdown("**Opponent Roster**")
                # Merge with pool for stats
                name_col = "name" if "name" in team_roster_df.columns else "player_name"
                roster_names = team_roster_df[name_col].dropna().tolist()
                roster_pool = pool[pool["player_name"].isin(roster_names)].copy()

                if not roster_pool.empty:
                    display_cols = ["player_name", "positions"]
                    # Add relevant stat columns
                    for col in ["r", "hr", "rbi", "sb", "avg", "w", "sv", "k", "era", "whip"]:
                        if col in roster_pool.columns:
                            display_cols.append(col)

                    display_df = roster_pool[[c for c in display_cols if c in roster_pool.columns]].copy()

                    # Rename columns to full names
                    col_rename = {
                        "player_name": "Player",
                        "positions": "Positions",
                        "r": "Runs",
                        "hr": "Home Runs",
                        "rbi": "Runs Batted In",
                        "sb": "Stolen Bases",
                        "avg": "Batting Average",
                        "w": "Wins",
                        "sv": "Saves",
                        "k": "Strikeouts",
                        "era": "Earned Run Average",
                        "whip": "WHIP",
                    }
                    display_df = display_df.rename(columns=col_rename)

                    # Format rate stats
                    if "Batting Average" in display_df.columns:
                        display_df["Batting Average"] = pd.to_numeric(
                            display_df["Batting Average"], errors="coerce"
                        ).apply(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
                    if "Earned Run Average" in display_df.columns:
                        display_df["Earned Run Average"] = pd.to_numeric(
                            display_df["Earned Run Average"], errors="coerce"
                        ).apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
                    if "WHIP" in display_df.columns:
                        display_df["WHIP"] = pd.to_numeric(display_df["WHIP"], errors="coerce").apply(
                            lambda x: f"{x:.2f}" if pd.notna(x) else "-"
                        )

                    render_sortable_table(display_df, height=min(600, 50 + len(display_df) * 35))
                else:
                    st.caption("Roster player stats not available in player pool.")

    # ══════════════════════════════════════════════════════════════════
    # TAB 4: Smart Recommendations
    # ══════════════════════════════════════════════════════════════════
    with tab4:
        st.subheader("Smart Recommendations")
        st.caption(
            "Auto-scans all opponents to find trades that boost your weakest categories "
            "for the least cost. Ranked by need-weighted efficiency."
        )

        if st.button("Generate Recommendations", type="primary", key="smart_recs_btn"):
            with st.spinner("Scanning all opponents for need-efficient trades..."):
                t0 = time.time()
                try:
                    from src.trade_finder import find_trade_opportunities

                    opportunities = find_trade_opportunities(
                        user_roster_ids=user_roster_ids,
                        player_pool=pool,
                        config=config,
                        all_team_totals=all_team_totals,
                        user_team_name=user_team_name,
                        league_rosters=league_rosters,
                        max_results=30,
                        top_partners=5,
                    )
                except Exception as e:
                    logger.warning("Trade finder scan failed: %s", e)
                    opportunities = []
                scan_time = time.time() - t0

            if not opportunities:
                st.info(
                    "No profitable trade opportunities found. This can happen when:\n"
                    "- League standings data is missing (sync with Yahoo)\n"
                    "- Your roster is well-balanced (no clear upgrade paths)\n"
                    "- All opponents have weaker players at your need positions"
                )
            else:
                # Score each trade by need efficiency and sort
                scored = []
                for trade in opportunities:
                    # Compute category impact for this trade
                    try:
                        give_ids = trade.get("giving_ids", [])
                        recv_ids = trade.get("receiving_ids", [])
                        new_ids = [pid for pid in user_roster_ids if pid not in give_ids] + recv_ids
                        user_tots = _roster_category_totals(user_roster_ids, pool)
                        new_tots = _roster_category_totals(new_ids, pool)
                        cat_impact = {}
                        for cat in config.all_categories:
                            old_v = user_tots.get(cat, 0)
                            new_v = new_tots.get(cat, 0)
                            cat_impact[cat] = new_v - old_v

                        if TRADE_INTEL_AVAILABLE:
                            eff = score_trade_by_need_efficiency(cat_impact, category_needs, config)
                        else:
                            eff = {"efficiency_ratio": 0.0, "boosted_cats": [], "costly_cats": []}

                        trade_scored = dict(trade)
                        trade_scored["efficiency"] = eff
                        trade_scored["cat_impact"] = cat_impact
                        scored.append(trade_scored)
                    except Exception:
                        scored.append(trade)

                # Sort by efficiency ratio descending
                scored.sort(
                    key=lambda t: t.get("efficiency", {}).get("efficiency_ratio", 0),
                    reverse=True,
                )

                st.markdown(
                    f'<div style="font-size:12px;color:{T["tx2"]};margin-bottom:12px;">'
                    f"Found {len(scored)} trades in {scan_time:.1f}s. "
                    f"Ranked by need-weighted efficiency.</div>",
                    unsafe_allow_html=True,
                )

                for i, rec in enumerate(scored[:15]):
                    giving = ", ".join(rec.get("giving_names", []))
                    receiving = ", ".join(rec.get("receiving_names", []))
                    partner = rec.get("opponent_team", "?")
                    eff = rec.get("efficiency", {})
                    eff_ratio = eff.get("efficiency_ratio", 0.0)
                    boosted = eff.get("boosted_cats", [])
                    costly = eff.get("costly_cats", [])
                    grade = rec.get("grade", "?")
                    acceptance = rec.get("acceptance_label", "?")
                    user_gain = rec.get("user_sgp_gain", 0)

                    header_text = (
                        f"#{i + 1}: Give {giving} -- Get {receiving} ({partner}) [{grade}, {eff_ratio:.1f}x efficiency]"
                    )

                    with st.expander(header_text, expanded=(i < 3)):
                        rc1, rc2, rc3, rc4, rc5 = st.columns(5)
                        rc1.metric("Grade", grade)
                        rc2.metric("Efficiency", f"{eff_ratio:.2f}x")
                        rc3.metric("Acceptance", acceptance)
                        rc4.metric("Your Gain", f"{user_gain:+.2f}")
                        rc5.metric(
                            "ADP Fairness",
                            f"{rec.get('adp_fairness', 0):.0%}"
                            if isinstance(rec.get("adp_fairness"), (int, float))
                            else "N/A",
                        )

                        # Category impact summary
                        if boosted:
                            boosted_names = [_CAT_DISPLAY.get(c, c) for c in boosted]
                            st.markdown(
                                f'<div style="font-size:12px;color:{T["ok"]};">'
                                f"Boosted: {', '.join(boosted_names)}</div>",
                                unsafe_allow_html=True,
                            )
                        if costly:
                            costly_names = [_CAT_DISPLAY.get(c, c) for c in costly]
                            st.markdown(
                                f'<div style="font-size:12px;color:{T["hot"]};">Cost: {", ".join(costly_names)}</div>',
                                unsafe_allow_html=True,
                            )

                        # Compact category impact table
                        cat_impact = rec.get("cat_impact", {})
                        if cat_impact:
                            impact_rows = []
                            for cat in config.all_categories:
                                val = cat_impact.get(cat, 0)
                                if abs(val) > 0.01:
                                    need = category_needs.get(cat, 0.5)
                                    need_lbl, _ = _need_label_and_color(need)
                                    impact_rows.append(
                                        {
                                            "Category": _CAT_DISPLAY.get(cat, cat),
                                            "Impact": f"{val:+.2f}",
                                            "Need": need_lbl,
                                        }
                                    )
                            if impact_rows:
                                render_compact_table(pd.DataFrame(impact_rows))

page_timer_footer("Trade Analyzer")

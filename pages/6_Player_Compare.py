"""Player Compare — Head-to-head comparison with z-scores and radar chart."""

import time

import pandas as pd
import streamlit as st

from src.database import coerce_numeric_df, get_connection, init_db, load_league_rosters, load_player_pool
from src.in_season import compare_players
from src.injury_model import get_injury_badge
from src.ui_shared import (
    ALL_CATEGORIES,
    METRIC_TOOLTIPS,
    get_plotly_layout,
    get_plotly_polar,
    get_theme,
    inject_custom_css,
    page_timer_footer,
    page_timer_start,
    render_compact_table,
    render_context_card,
    render_context_columns,
    render_page_layout,
    render_player_select,
    render_styled_table,
)
from src.valuation import LeagueConfig, add_process_risk, compute_percentile_projections, compute_projection_volatility

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

st.set_page_config(page_title="Heater | Player Compare", page_icon="", layout="wide", initial_sidebar_state="collapsed")

init_db()

inject_custom_css()
page_timer_start()

render_page_layout("PLAYER COMPARE", banner_teaser="Select two players to compare", banner_icon="player_compare")

pool = load_player_pool()
if pool.empty:
    st.warning("No player data loaded.")
    st.stop()

pool = pool.rename(columns={"name": "player_name"})

# health_score and status are already in the enriched pool from load_player_pool()
# Player Compare does NOT need the destructive counting-stat reduction from
# get_health_adjusted_pool() — it only needs health_score for display badges.

rosters_df = load_league_rosters()
config = LeagueConfig()


def _get_roster_badge(player_id, rosters_df):
    """Return HTML badge showing roster status."""
    if rosters_df.empty:
        return ""
    match = rosters_df[rosters_df["player_id"] == player_id]
    if not match.empty:
        team = match.iloc[0].get("team_name", "Unknown")
        return (
            f'<span style="font-size:11px;padding:2px 6px;'
            f'background:#e8f5e9;border-radius:4px;">Rostered: {team}</span>'
        )
    return '<span style="font-size:11px;padding:2px 6px;background:#f5f5f5;border-radius:4px;">Free Agent</span>'


player_names = sorted(pool["player_name"].tolist())

# ── 3-zone hybrid layout ──────────────────────────────────────────────────────
ctx, main = render_context_columns()

# ── Main content panel ────────────────────────────────────────────────────────
with main:
    # Fix button text overflow for player name pills
    st.markdown(
        """<style>
        [data-testid="stHorizontalBlock"] button[kind] p {
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }
        </style>""",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        search_a = st.text_input("Search Player A", key="search_a", placeholder="Type a player name...")
        filtered_a = [n for n in player_names if search_a.lower() in n.lower()] if search_a else player_names
        if filtered_a:
            # Show top matches as selectable cards
            top_a = filtered_a[:3]
            a_cols = st.columns(min(len(top_a), 3))
            player_a_name = st.session_state.get("compare_a")
            for ai, aname in enumerate(top_a):
                with a_cols[ai]:
                    a_type = "primary" if player_a_name == aname else "secondary"
                    if st.button(aname, key=f"comp_a_{ai}", type=a_type, width="stretch"):
                        st.session_state.compare_a = aname
                        st.rerun()
            player_a_name = st.session_state.get("compare_a")
        else:
            st.info("No players match search.")
            player_a_name = None

    with col2:
        search_b = st.text_input("Search Player B", key="search_b", placeholder="Type a player name...")
        filtered_b = [n for n in player_names if search_b.lower() in n.lower()] if search_b else player_names
        if filtered_b:
            top_b = filtered_b[:3]
            b_cols = st.columns(min(len(top_b), 3))
            player_b_name = st.session_state.get("compare_b")
            for bi, bname in enumerate(top_b):
                with b_cols[bi]:
                    b_type = "primary" if player_b_name == bname else "secondary"
                    if st.button(bname, key=f"comp_b_{bi}", type=b_type, width="stretch"):
                        st.session_state.compare_b = bname
                        st.rerun()
            player_b_name = st.session_state.get("compare_b")
        else:
            st.info("No players match search.")
            player_b_name = None

    if player_a_name and player_b_name and player_a_name != player_b_name:
        match_a = pool[pool["player_name"] == player_a_name]
        match_b = pool[pool["player_name"] == player_b_name]
        if match_a.empty or match_b.empty:
            st.error("Could not find one or both selected players in the pool.")
            st.stop()
        id_a = match_a.iloc[0]["player_id"]
        id_b = match_b.iloc[0]["player_id"]

        # Roster status badges
        badge_col1, badge_col2 = st.columns(2)
        with badge_col1:
            st.markdown(_get_roster_badge(id_a, rosters_df), unsafe_allow_html=True)
        with badge_col2:
            st.markdown(_get_roster_badge(id_b, rosters_df), unsafe_allow_html=True)

        compare_progress = st.progress(0, text="Comparing players across 12 categories...")
        result = compare_players(int(id_a), int(id_b), pool, config)
        compare_progress.progress(100, text="Comparison complete!")
        time.sleep(0.3)
        compare_progress.empty()

        # Health scores are already in the pool from the enriched load_player_pool()
        # which accounts for both injury history AND current IL/DTD status.

        if "error" in result:
            st.error(result["error"])
        else:
            # Radar chart
            if HAS_PLOTLY:
                t = get_theme()
                cats = ALL_CATEGORIES
                cat_display_names = {
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
                cat_labels = [cat_display_names.get(c, c) for c in cats]
                z_a = [result["z_scores_a"].get(c, 0) for c in cats]
                z_b = [result["z_scores_b"].get(c, 0) for c in cats]

                fig = go.Figure()
                fig.add_trace(
                    go.Scatterpolar(
                        r=z_a + [z_a[0]],
                        theta=cat_labels + [cat_labels[0]],
                        name=result["player_a"],
                        line=dict(color=t["amber"]),
                        fill="toself",
                        fillcolor="rgba(230,57,70,0.15)",
                    )
                )
                fig.add_trace(
                    go.Scatterpolar(
                        r=z_b + [z_b[0]],
                        theta=cat_labels + [cat_labels[0]],
                        name=result["player_b"],
                        line=dict(color=t["teal"]),
                        fill="toself",
                        fillcolor="rgba(69,123,157,0.15)",
                    )
                )
                layout_kwargs = get_plotly_layout(t)
                layout_kwargs["polar"] = get_plotly_polar(t)
                layout_kwargs["legend"] = dict(font=dict(color=t["tx"]))
                fig.update_layout(**layout_kwargs)
                st.plotly_chart(fig, width="stretch")

            # Z-score comparison table
            st.subheader("Category Breakdown")
            cat_full_names = {
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
            rows = []
            for cat in ALL_CATEGORIES:
                za = result["z_scores_a"].get(cat, 0)
                zb = result["z_scores_b"].get(cat, 0)
                adv = result["advantages"].get(cat, "TIE")
                rows.append(
                    {
                        "Category": cat_full_names.get(cat, cat),
                        f"{result['player_a']} Z-Score": f"{za:+.2f}",
                        f"{result['player_b']} Z-Score": f"{zb:+.2f}",
                        "Advantage": adv,
                    }
                )
            render_compact_table(pd.DataFrame(rows))
            st.caption(METRIC_TOOLTIPS["z_score"])

            # N3: SGP Contribution Breakdown — shows concentrated vs diversified value
            try:
                from src.valuation import SGPCalculator

                _sgp_calc = SGPCalculator(config)
                _row_a = pool[pool["player_id"] == id_a]
                _row_b = pool[pool["player_id"] == id_b]
                if not _row_a.empty and not _row_b.empty:
                    sgp_a = _sgp_calc.player_sgp(_row_a.iloc[0])
                    sgp_b = _sgp_calc.player_sgp(_row_b.iloc[0])
                    st.subheader("Standings Gained Points Breakdown")
                    sgp_rows = []
                    for cat in ALL_CATEGORIES:
                        sa = sgp_a.get(cat, 0)
                        sb = sgp_b.get(cat, 0)
                        if abs(sa) > 0.001 or abs(sb) > 0.001:
                            sgp_rows.append({
                                "Category": cat_full_names.get(cat, cat),
                                f"{result['player_a']}": f"{sa:+.2f}",
                                f"{result['player_b']}": f"{sb:+.2f}",
                                "Delta": f"{sa - sb:+.2f}",
                            })
                    if sgp_rows:
                        # Summary row
                        total_a = sum(sgp_a.values())
                        total_b = sum(sgp_b.values())
                        sgp_rows.append({
                            "Category": "TOTAL",
                            f"{result['player_a']}": f"{total_a:+.2f}",
                            f"{result['player_b']}": f"{total_b:+.2f}",
                            "Delta": f"{total_a - total_b:+.2f}",
                        })
                        render_compact_table(pd.DataFrame(sgp_rows))
                        st.caption(
                            "Standings Gained Points: how many standings positions "
                            "each player's stats are worth. Concentrated value "
                            "(e.g., 3.0 from Home Runs/Runs Batted In) vs diversified "
                            "(0.5 across many) affects trade and roster strategy."
                        )
            except Exception:
                pass

            # YTD 2026 Stats comparison (from enriched pool)
            _has_ytd = any(c in pool.columns for c in ["ytd_pa", "ytd_avg", "ytd_hr"])
            if _has_ytd:
                _pa = pool[pool["player_id"] == id_a]
                _pb = pool[pool["player_id"] == id_b]
                if not _pa.empty and not _pb.empty:
                    _ra = _pa.iloc[0]
                    _rb = _pb.iloc[0]
                    _ytd_pa_a = int(_ra.get("ytd_pa", 0) or 0)
                    _ytd_pa_b = int(_rb.get("ytd_pa", 0) or 0)
                    if _ytd_pa_a > 0 or _ytd_pa_b > 0:
                        st.subheader("2026 Season Stats")
                        _ytd_cols = {
                            "ytd_pa": "Plate Appearances",
                            "ytd_avg": "Batting Average",
                            "ytd_hr": "Home Runs",
                            "ytd_rbi": "Runs Batted In",
                            "ytd_sb": "Stolen Bases",
                            "ytd_era": "Earned Run Average",
                            "ytd_whip": "Walks + Hits per Inning Pitched",
                            "ytd_sv": "Saves",
                            "ytd_k": "Strikeouts",
                        }
                        _ytd_rows = []
                        for _col, _label in _ytd_cols.items():
                            if _col in pool.columns:
                                _va = _ra.get(_col, 0) or 0
                                _vb = _rb.get(_col, 0) or 0
                                # Format rate stats
                                if _col in ("ytd_avg",):
                                    _va_s = f".{float(_va):.3f}"[1:] if float(_va) > 0 else "--"
                                    _vb_s = f".{float(_vb):.3f}"[1:] if float(_vb) > 0 else "--"
                                elif _col in ("ytd_era", "ytd_whip"):
                                    _va_s = f"{float(_va):.2f}" if float(_va) > 0 else "--"
                                    _vb_s = f"{float(_vb):.2f}" if float(_vb) > 0 else "--"
                                else:
                                    _va_s = str(int(float(_va))) if float(_va) > 0 else "--"
                                    _vb_s = str(int(float(_vb))) if float(_vb) > 0 else "--"
                                _ytd_rows.append(
                                    {
                                        "Stat": _label,
                                        result["player_a"]: _va_s,
                                        result["player_b"]: _vb_s,
                                    }
                                )
                        if _ytd_rows:
                            render_compact_table(pd.DataFrame(_ytd_rows))
                            st.caption("Actual 2026 stats from MLB Stats API. '--' = no data or zero.")

            # Player card selector for compared players
            render_player_select(
                [player_a_name, player_b_name],
                [int(id_a), int(id_b)],
                key_suffix="compare",
            )

            # Health comparison
            st.subheader("Health & Confidence")
            health_rows = []
            for name_col, pid_col in [(player_a_name, id_a), (player_b_name, id_b)]:
                p_match = pool[pool["player_id"] == pid_col]
                hs = float(p_match.iloc[0].get("health_score", 0.85) or 0.85) if not p_match.empty else 0.85
                icon, label = get_injury_badge(hs)
                health_rows.append({"Player": name_col, "Health": f"{icon} {label}", "Score": f"{hs:.2f}"})

            # Projection confidence: P10-P90 range width per player using key stat cols
            try:
                conn = get_connection()
                try:
                    systems = {}
                    for sys_name in ["steamer", "zips", "depthcharts"]:
                        df = pd.read_sql_query(
                            "SELECT * FROM projections WHERE system = ?",
                            conn,
                            params=(sys_name,),
                        )
                        if not df.empty:
                            df = coerce_numeric_df(df)
                            systems[sys_name] = df
                finally:
                    conn.close()

                if len(systems) >= 2:
                    vol = compute_projection_volatility(systems)
                    vol = add_process_risk(vol)
                    # pool uses 'player_name' column; pass the pool renamed back to 'name' for base
                    base_df = pool.rename(columns={"player_name": "name"})
                    pct = compute_percentile_projections(base=base_df, volatility=vol)
                    p10_df, p90_df = pct.get(10), pct.get(90)
                    stat_cols = ["r", "hr", "rbi", "sb", "w", "sv", "k"]
                    if p10_df is not None and p90_df is not None:
                        name_col_key = "name" if "name" in p10_df.columns else "player_name"
                        for row in health_rows:
                            name = row["Player"]
                            p10_row = p10_df[p10_df[name_col_key] == name]
                            p90_row = p90_df[p90_df[name_col_key] == name]
                            if not p10_row.empty and not p90_row.empty:
                                present = [c for c in stat_cols if c in p10_df.columns]
                                if present:
                                    p10_sum = p10_row.iloc[0][present].sum()
                                    p90_sum = p90_row.iloc[0][present].sum()
                                    width = p90_sum - p10_sum
                                    row["Confidence"] = f"±{width:.2f}" if width > 0 else "—"
                                else:
                                    row["Confidence"] = "—"
                            else:
                                row["Confidence"] = "—"
                else:
                    for row in health_rows:
                        row["Confidence"] = "—"
            except Exception:
                for row in health_rows:
                    row["Confidence"] = "—"

            render_styled_table(pd.DataFrame(health_rows))
    else:
        if player_a_name == player_b_name:
            st.info("Select two different players to compare.")

# ── Context panel (left) ──────────────────────────────────────────────────────
with ctx:
    # Determine whether a valid comparison result is available in session state
    _pa = st.session_state.get("compare_a")
    _pb = st.session_state.get("compare_b")
    _have_result = (
        _pa
        and _pb
        and _pa != _pb
        and not pool[pool["player_name"] == _pa].empty
        and not pool[pool["player_name"] == _pb].empty
    )

    if _have_result:
        # Re-run comparison (lightweight — result is fast and not cached)
        _ma = pool[pool["player_name"] == _pa].iloc[0]["player_id"]
        _mb = pool[pool["player_name"] == _pb].iloc[0]["player_id"]
        _r = compare_players(int(_ma), int(_mb), pool, config)
        if "error" not in _r:
            _ca = _r.get("composite_a", 0.0)
            _cb = _r.get("composite_b", 0.0)
            _winner = _pa if _ca >= _cb else _pb
            _margin = abs(_ca - _cb)
            _verdict = "Even matchup" if _margin < 0.5 else f"{_winner} leads"
            render_context_card(
                "Composite Scores",
                f'<div class="ctx-row"><span>{_pa}</span>'
                f'<span style="font-weight:700 !important;">{_ca:+.2f}</span></div>'
                f'<div class="ctx-row"><span>{_pb}</span>'
                f'<span style="font-weight:700 !important;">{_cb:+.2f}</span></div>',
            )
            _adv_a = sum(1 for v in _r.get("advantages", {}).values() if v == _pa)
            _adv_b = sum(1 for v in _r.get("advantages", {}).values() if v == _pb)
            _ties = sum(1 for v in _r.get("advantages", {}).values() if v == "TIE")
            render_context_card(
                "Quick Verdict",
                f'<div class="ctx-row"><span>Category wins</span>'
                f"<span>{_adv_a} — {_adv_b} ({_ties} ties)</span></div>"
                f'<div class="ctx-row"><span>Edge</span>'
                f'<span style="font-weight:700 !important;">{_verdict}</span></div>',
            )
        else:
            render_context_card(
                "Comparison",
                '<p style="color:var(--tx-muted,#888) !important;">Could not load scores.</p>',
            )
    else:
        render_context_card(
            "Select Players",
            '<p style="color:var(--tx-muted,#888) !important;">'
            "Search and select two different players to see composite scores and category breakdown."
            "</p>",
        )

page_timer_footer("Player Compare")

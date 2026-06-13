"""Player Compare — Head-to-head comparison with z-scores and radar chart."""

import time

import pandas as pd
import streamlit as st

from src.ai.chat import render_chat_widget
from src.auth import multi_user_enabled, require_auth, resolve_viewer_team_name
from src.database import coerce_numeric_df, get_connection, init_db, load_player_pool
from src.feature_flags import require_page_enabled
from src.feedback import render_feedback_widget
from src.in_season import compare_players, compute_category_fit
from src.injury_model import get_injury_badge
from src.usage import log_page_view
from src.yahoo_data_service import get_yahoo_data_service

try:
    from src.player_databank import compute_rolling_stats

    HAS_ROLLING_STATS = True
except Exception:
    HAS_ROLLING_STATS = False
from src.ui_shared import (
    ALL_CATEGORIES,
    HITTING_CATEGORIES,
    METRIC_TOOLTIPS,
    PITCHING_CATEGORIES,
    build_heatbar_html,
    build_stat_readout_html,
    format_stat,
    get_plotly_layout,
    get_plotly_polar,
    get_theme,
    inject_custom_css,
    jargon_help,
    page_timer_footer,
    page_timer_start,
    render_compact_table,
    render_context_card,
    render_context_columns,
    render_data_freshness_chip,
    render_glossary_expander,
    render_page_header,
    render_panel,
    render_player_select,
    render_reco_banner,
    render_styled_table,
    team_color,
    team_logo_url,
)
from src.valuation import LeagueConfig, add_process_risk, compute_percentile_projections, compute_projection_volatility

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

if not multi_user_enabled():
    st.set_page_config(
        page_title="Heater | Player Compare", page_icon="", layout="wide", initial_sidebar_state="collapsed"
    )

init_db()

inject_custom_css()
require_auth()
require_page_enabled("page:16_Player_Compare")
log_page_view("Player Compare")
render_chat_widget("Player Compare")
page_timer_start()

render_page_header(
    "Player Compare",
    eyebrow="RESEARCH",
    fig="FIG.16 — HEAD-TO-HEAD",
)
render_data_freshness_chip("player_pool")
render_reco_banner("Select two players to compare", "", "player_compare")


@st.cache_data(ttl=600, show_spinner=False)
def _load_per_system_projections(player_ids: tuple[int, ...]) -> dict[str, pd.DataFrame]:
    """Load raw per-system projections for the given players.

    The blended projection is already in ``load_player_pool()``. This helper exists
    only for cross-system volatility (P10/P90 confidence intervals) which needs the
    underlying steamer/zips/depthcharts rows the pool collapses into one. ECR rank,
    season stats, and Statcast all flow through ``load_player_pool()`` and must NOT
    be fetched directly here.
    """
    if not player_ids:
        return {}
    conn = get_connection()
    try:
        placeholders = ",".join("?" for _ in player_ids)
        out: dict[str, pd.DataFrame] = {}
        for sys_name in ("steamer", "zips", "depthcharts"):
            df = pd.read_sql_query(
                f"SELECT * FROM projections WHERE system = ? AND player_id IN ({placeholders})",
                conn,
                params=(sys_name, *player_ids),
            )
            if not df.empty:
                out[sys_name] = coerce_numeric_df(df)
        return out
    finally:
        conn.close()


pool = load_player_pool()
if pool.empty:
    st.warning("No player data loaded.")
    st.stop()

pool = pool.rename(columns={"name": "player_name"})

# health_score and status are already in the enriched pool from load_player_pool()
# Player Compare does NOT need the destructive counting-stat reduction from
# get_health_adjusted_pool() — it only needs health_score for display badges.

# ── Player universe filter (Wave 9 INFRA-F5) ─────────────────────────────────
# Filter by player level (MLB / AAA / AA / All). Minor leaguers lack Yahoo
# ownership data, so default to MLB-only to avoid accidentally comparing an
# MLB regular against an AAA prospect with no ownership context.
_LEVEL_OPTIONS = ["MLB only", "MLB + AAA", "MLB + AAA + AA", "All"]
_level_filter = st.selectbox(
    "Player universe",
    _LEVEL_OPTIONS,
    index=0,
    help="MLB-only is the default. Expanding to AAA/AA shows minor-league "
    "depth-chart candidates but they lack Yahoo ownership data.",
)

# Apply filter. Use pool["level"] (Series) — NOT pool.get("level") (would
# return a scalar default if column missing, breaking .isna()).
if "level" not in pool.columns:
    # Legacy DB without Wave 9 migration — show everything (no-op).
    pass
elif _level_filter == "MLB only":
    pool = pool[pool["level"].isna() | (pool["level"] == "MLB")]
elif _level_filter == "MLB + AAA":
    pool = pool[pool["level"].isna() | pool["level"].isin(["MLB", "AAA"])]
elif _level_filter == "MLB + AAA + AA":
    pool = pool[pool["level"].isna() | pool["level"].isin(["MLB", "AAA", "AA"])]
# else "All" → no filter

_yds = get_yahoo_data_service()
rosters_df = _yds.get_rosters()
config = LeagueConfig()
user_team_name = resolve_viewer_team_name(rosters_df)


def _get_roster_badge(player_id, rosters_df, viewer_team: str | None = None):
    """Return HTML badge showing roster status.

    Shows "My Team" for the viewer's own rostered players (using the emoji-
    reconciled name from resolve_viewer_team_name), "Rostered: {team}" for
    other teams, and "Free Agent" for unrostered players.
    """
    if rosters_df.empty:
        return ""
    match = rosters_df[rosters_df["player_id"] == player_id]
    if not match.empty:
        team = match.iloc[0].get("team_name", "Unknown")
        if viewer_team and str(team) == str(viewer_team):
            return (
                '<span style="font-size:11px;padding:2px 6px;'
                'background:rgba(255,109,0,.12);border-radius:4px;font-weight:600;">My Team</span>'
            )
        return (
            f'<span style="font-size:11px;padding:2px 6px;'
            f'background:rgba(31,157,107,.08);border-radius:4px;">Rostered: {team}</span>'
        )
    return (
        '<span style="font-size:11px;padding:2px 6px;background:var(--fp-divider);border-radius:4px;">Free Agent</span>'
    )


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
        search_a = st.text_input(
            "Search Player A",
            key="search_a",
            placeholder="Type a player name...",
            help=jargon_help("ECR"),
        )
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
            st.markdown(_get_roster_badge(id_a, rosters_df, viewer_team=user_team_name), unsafe_allow_html=True)
        with badge_col2:
            st.markdown(_get_roster_badge(id_b, rosters_df, viewer_team=user_team_name), unsafe_allow_html=True)

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
            # ── Player identity strip: logo + headshot + team-color accent ──
            import html as _cmp_html

            def _identity_card(_name, _pid, _accent):
                _prow = pool[pool["player_id"] == _pid]
                _team = str(_prow.iloc[0].get("team", "")) if not _prow.empty else ""
                _mlb = _prow.iloc[0].get("mlb_id") if not _prow.empty else None
                _pos = str(_prow.iloc[0].get("positions", "")) if not _prow.empty else ""
                _tc = team_color(_team) if _team and _team != "MLB" else _accent
                _logo = team_logo_url(_team) if _team and _team != "MLB" else ""
                _hs = ""
                try:
                    _mid = int(_mlb) if _mlb is not None else 0
                    if _mid > 0:
                        _hs = (
                            f'<img src="https://img.mlbstatic.com/mlb-photos/image/upload/'
                            f"d_people:generic:headshot:67:current.png/w_213,q_auto:best/"
                            f'v1/people/{_mid}/headshot/67/current" width="46" height="46" '
                            f'style="border-radius:50%;object-fit:cover;border:2px solid {_tc};'
                            f'flex-shrink:0;" onerror="this.style.display=\'none\'" loading="lazy" />'
                        )
                except (ValueError, TypeError):
                    pass
                _logo_html = (
                    f'<img src="{_logo}" width="15" height="15" style="vertical-align:-2px;'
                    f'margin-right:4px;" onerror="this.style.display=\'none\'" loading="lazy" />'
                    if _logo
                    else ""
                )
                _team_txt = _cmp_html.escape(_team.upper()) if _team and _team != "MLB" else "FA"
                _pos_txt = _cmp_html.escape(_pos.split(",")[0].split("/")[0].strip())
                return (
                    f'<div style="display:flex;align-items:center;gap:11px;background:var(--fp-surface);'
                    f"border:1px solid var(--fp-border);border-left:3px solid {_tc};border-radius:10px;"
                    f'padding:11px 13px;box-shadow:var(--fp-shadow);">'
                    f"{_hs}"
                    f'<div style="min-width:0;">'
                    f'<div style="font-family:var(--font-display);font-weight:800;font-size:16px;'
                    f"letter-spacing:-.01em;color:var(--fp-tx);white-space:nowrap;overflow:hidden;"
                    f'text-overflow:ellipsis;">{_cmp_html.escape(str(_name))}</div>'
                    f'<div style="font-family:var(--font-mono);font-size:10px;letter-spacing:.04em;'
                    f'color:var(--fp-tx-subtle);margin-top:2px;">{_logo_html}{_team_txt} · {_pos_txt}</div>'
                    f"</div></div>"
                )

            _id_c1, _id_c2 = st.columns(2)
            with _id_c1:
                st.markdown(_identity_card(player_a_name, id_a, "#ff6d00"), unsafe_allow_html=True)
            with _id_c2:
                st.markdown(_identity_card(player_b_name, id_b, "#5f7d9c"), unsafe_allow_html=True)

            # Determine player types (hitter vs pitcher) — needed for radar axis
            # filtering (4.6-B) and cross-type guard (4.6-C).
            _row_a_type = pool[pool["player_id"] == id_a]
            _row_b_type = pool[pool["player_id"] == id_b]
            is_hitter_a = bool(int(_row_a_type.iloc[0].get("is_hitter", 1))) if not _row_a_type.empty else True
            is_hitter_b = bool(int(_row_b_type.iloc[0].get("is_hitter", 1))) if not _row_b_type.empty else True
            _cross_type = is_hitter_a != is_hitter_b

            # 4.6-C — Cross-type guard: hitter vs pitcher comparison is not meaningful
            if _cross_type:
                st.warning(
                    "Comparing a hitter and a pitcher — category comparison isn't "
                    "meaningful across types. The z-scores, radar, and composite below "
                    "reflect league-wide norms within each player's position group only."
                )

            # Radar chart
            if HAS_PLOTLY:
                t = get_theme()
                # 4.6-B: same-type comparisons use only the 6 relevant axes so the
                # opposite-type axes (which are always 0) don't produce a dead flat ring.
                # Cross-type comparisons fall back to all 12 cats so both players show up.
                if not _cross_type and is_hitter_a:
                    cats = HITTING_CATEGORIES
                elif not _cross_type and not is_hitter_a:
                    cats = PITCHING_CATEGORIES
                else:
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
                        fillcolor="rgba(255,109,0,0.15)",
                    )
                )
                fig.add_trace(
                    go.Scatterpolar(
                        r=z_b + [z_b[0]],
                        theta=cat_labels + [cat_labels[0]],
                        name=result["player_b"],
                        line=dict(color=t["teal"]),
                        fill="toself",
                        fillcolor="rgba(95,125,156,0.15)",
                    )
                )
                layout_kwargs = get_plotly_layout(t)
                layout_kwargs["polar"] = get_plotly_polar(t)
                layout_kwargs["legend"] = dict(font=dict(color=t["tx"]))
                fig.update_layout(**layout_kwargs)
                st.plotly_chart(fig, width="stretch")

            # Z-score comparison table
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

            # ── Category edge heat-bar panel: per-category win/loss visual ──
            # Each bar's fill is the share of the pair's z-total held by player A
            # (orange = A leads, steel = B leads). Pure presentation of the same
            # advantages the compact table lists below.
            _edge_rows_html = ""
            for cat in ALL_CATEGORIES:
                za = result["z_scores_a"].get(cat, 0) or 0
                zb = result["z_scores_b"].get(cat, 0) or 0
                adv = result["advantages"].get(cat, "TIE")
                # Shift z-scores positive so the split is well-defined, then take
                # A's share of the combined magnitude as the fill percentage.
                _sa = za + 4.0
                _sb = zb + 4.0
                _tot = _sa + _sb
                _fill = (_sa / _tot * 100.0) if _tot > 0 else 50.0
                _a_leads = adv == result["player_a"]
                _bar = build_heatbar_html(_fill, win=_a_leads)
                _adv_safe = _cmp_html.escape(str(adv))
                _adv_color = (
                    "var(--fp-primary)"
                    if _a_leads
                    else ("var(--fp-cold)" if adv == result["player_b"] else "var(--fp-tx-subtle)")
                )
                _edge_rows_html += (
                    '<div style="display:grid;grid-template-columns:130px 1fr 96px;align-items:center;'
                    'gap:12px;padding:5px 0;">'
                    f'<span style="font-family:var(--font-mono);font-size:11px;letter-spacing:.03em;'
                    f'color:var(--fp-tx-muted);">{_cmp_html.escape(cat_full_names.get(cat, cat))}</span>'
                    f"<span>{_bar}</span>"
                    f'<span style="font-family:var(--font-mono);font-size:10.5px;font-weight:600;'
                    f"text-align:right;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"
                    f'color:{_adv_color};">{_adv_safe}</span>'
                    "</div>"
                )
            render_panel(
                "Category Edge",
                _edge_rows_html,
                fig_label="12 CATS · Z-SCORE",
            )

            st.subheader("Category Breakdown")
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
            render_compact_table(
                pd.DataFrame(rows),
            )
            st.caption(METRIC_TOOLTIPS["z_score"])
            # Task 3.3 — inverse-stat annotation: ERA/WHIP/L z-scores are
            # quality-adjusted (lower raw value = better = higher z-score).
            st.caption(
                "Inverse stats (ERA, WHIP, L): lower is better. "
                "Z-scores are already quality-adjusted so a higher z-score "
                "always means the player is farther above average, even for inverse categories."
            )
            render_glossary_expander(
                ["SGP", "VORP", "ECR", "ADP"],
                label="What do these numbers mean?",
            )

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
                            sgp_rows.append(
                                {
                                    "Category": cat_full_names.get(cat, cat),
                                    f"{result['player_a']}": format_stat(sa, "SGP"),
                                    f"{result['player_b']}": format_stat(sb, "SGP"),
                                    "Delta": format_stat(sa - sb, "SGP"),
                                }
                            )
                    if sgp_rows:
                        # Summary row
                        total_a = sum(sgp_a.values())
                        total_b = sum(sgp_b.values())
                        sgp_rows.append(
                            {
                                "Category": "TOTAL",
                                f"{result['player_a']}": format_stat(total_a, "SGP"),
                                f"{result['player_b']}": format_stat(total_b, "SGP"),
                                "Delta": format_stat(total_a - total_b, "SGP"),
                            }
                        )
                        render_compact_table(pd.DataFrame(sgp_rows))
                        st.caption(
                            "Standings Gained Points: how many standings positions "
                            "each player's stats are worth. Concentrated value "
                            "(e.g., 3.0 from Home Runs/Runs Batted In) vs diversified "
                            "(0.5 across many) affects trade and roster strategy."
                        )
            except Exception:
                pass

            # Task 3.5: Category Fit for Your Team
            # Wire compute_category_fit so the viewer sees which player better
            # fills their WEAK categories (e.g. "Helps your weak AVG and SB").
            # Build a simple team profile from standings data when available.
            # Falls back to a neutral "average" profile so the block is always
            # renderable even when rosters data is unavailable.
            try:
                from src.valuation import SGPCalculator as _SGPCalc

                _fit_calc = _SGPCalc(config)
                _fit_row_a = pool[pool["player_id"] == id_a]
                _fit_row_b = pool[pool["player_id"] == id_b]
                if not _fit_row_a.empty and not _fit_row_b.empty:
                    _fit_sgp_a = _fit_calc.player_sgp(_fit_row_a.iloc[0])
                    _fit_sgp_b = _fit_calc.player_sgp(_fit_row_b.iloc[0])

                    _team_profile: dict[str, str] = {}
                    _profile_source = "unknown"
                    if user_team_name:
                        try:
                            from src.standings_utils import get_team_totals

                            _all_totals = get_team_totals()
                            if _all_totals and user_team_name in _all_totals:
                                _user_totals = _all_totals[user_team_name]
                                _n_teams = len(_all_totals)
                                for _cat in config.all_categories:
                                    _uv = _user_totals.get(_cat, 0) or 0
                                    _vals = [t.get(_cat, 0) or 0 for t in _all_totals.values()]
                                    if _cat in set(config.inverse_stats):
                                        _rank = 1 + sum(1 for v in _vals if v < _uv and v > 0)
                                    else:
                                        _rank = 1 + sum(1 for v in _vals if v > _uv)
                                    if _rank <= max(2, _n_teams // 4):
                                        _team_profile[_cat] = "strong"
                                    elif _rank >= _n_teams - max(1, _n_teams // 4):
                                        _team_profile[_cat] = "weak"
                                    else:
                                        _team_profile[_cat] = "average"
                                _profile_source = f"standings ({user_team_name})"
                        except Exception:
                            pass

                    if not _team_profile:
                        _team_profile = {cat: "average" for cat in config.all_categories}
                        _profile_source = "no roster data"

                    _fit_a = compute_category_fit(_fit_sgp_a, _team_profile)
                    _fit_b = compute_category_fit(_fit_sgp_b, _team_profile)

                    _helps_a = _fit_a.get("helps", [])
                    _helps_b = _fit_b.get("helps", [])
                    _fit_score_a = _fit_a.get("fit_score", 0)
                    _fit_score_b = _fit_b.get("fit_score", 0)
                    _wastes_a = _fit_a.get("wastes", [])
                    _wastes_b = _fit_b.get("wastes", [])

                    if _helps_a or _helps_b or _wastes_a or _wastes_b:
                        st.subheader("Fits Your Team")

                        def _fit_line(player_name: str, helps: list, wastes: list, score: float) -> str:
                            parts = []
                            if helps:
                                cats_str = ", ".join(cat_full_names.get(c, c) for c in helps)
                                parts.append(f"Helps your weak **{cats_str}**")
                            if wastes:
                                cats_str = ", ".join(cat_full_names.get(c, c) for c in wastes)
                                parts.append(f"Redundant in **{cats_str}** (already strong)")
                            if not parts:
                                parts.append("No clear category fit signal")
                            return f"**{player_name}** — " + "; ".join(parts)

                        st.markdown(_fit_line(player_a_name, _helps_a, _wastes_a, _fit_score_a))
                        st.markdown(_fit_line(player_b_name, _helps_b, _wastes_b, _fit_score_b))
                        st.caption(
                            f"Category fit based on your team weak/strong categories "
                            f"(source: {_profile_source}). "
                            "Weak = ranked bottom third in the league. Strong = top third."
                        )
            except Exception:
                pass  # Graceful fallback — category fit is supplemental

            # N2: Schedule strength comparison — next 2-4 weeks of matchup quality
            try:
                from src.game_day import get_team_strength

                _team_a = str(pool[pool["player_id"] == id_a].iloc[0].get("team", ""))
                _team_b = str(pool[pool["player_id"] == id_b].iloc[0].get("team", ""))
                if _team_a and _team_b and _team_a != "MLB" and _team_b != "MLB":
                    _str_a = get_team_strength(_team_a)
                    _str_b = get_team_strength(_team_b)
                    st.subheader("Schedule Strength")
                    sched_rows = []
                    for label, key, fmt, better in [
                        ("Team wRC+ (offense)", "wrc_plus", ".0f", "higher"),
                        ("Team FIP (pitching)", "fip", ".2f", "lower"),
                        ("Team ERA (pitching)", "team_era", ".2f", "lower"),
                        ("Team K% (pitching)", "k_pct", ".1f", "higher"),
                    ]:
                        va = _str_a.get(key, 0)
                        vb = _str_b.get(key, 0)
                        sched_rows.append(
                            {
                                "Metric": label,
                                f"{result['player_a']} ({_team_a})": f"{va:{fmt}}",
                                f"{result['player_b']} ({_team_b})": f"{vb:{fmt}}",
                            }
                        )
                    render_compact_table(pd.DataFrame(sched_rows))
                    st.caption(
                        "Team strength context: higher wRC+ = better offense (good for hitters on that team). "
                        "Lower FIP/ERA = better pitching staff (good for pitchers)."
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
                                    _va_s = format_stat(float(_va), "AVG") if float(_va) > 0 else "--"
                                    _vb_s = format_stat(float(_vb), "AVG") if float(_vb) > 0 else "--"
                                elif _col in ("ytd_era", "ytd_whip"):
                                    _stat_type = "ERA" if _col == "ytd_era" else "WHIP"
                                    _va_s = format_stat(float(_va), _stat_type) if float(_va) > 0 else "--"
                                    _vb_s = format_stat(float(_vb), _stat_type) if float(_vb) > 0 else "--"
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

            # ── Recent Performance (L7 / L14 rolling stats) ────────────────
            if HAS_ROLLING_STATS:
                try:
                    _ids = [int(id_a), int(id_b)]
                    _l14 = compute_rolling_stats(_ids, days=14, stat_type="total")
                    _l7 = compute_rolling_stats(_ids, days=7, stat_type="total")

                    if not _l14.empty or not _l7.empty:
                        st.subheader("Recent Performance")

                        # Determine hitter vs pitcher per player
                        _row_a_pool = pool[pool["player_id"] == id_a]
                        _row_b_pool = pool[pool["player_id"] == id_b]
                        _is_hitter_a = (
                            bool(int(_row_a_pool.iloc[0].get("is_hitter", 1))) if not _row_a_pool.empty else True
                        )
                        _is_hitter_b = (
                            bool(int(_row_b_pool.iloc[0].get("is_hitter", 1))) if not _row_b_pool.empty else True
                        )

                        for _window_label, _wdf in [("Last 14 Days", _l14), ("Last 7 Days", _l7)]:
                            if _wdf.empty:
                                st.caption(f"{_window_label}: No recent game data available.")
                                continue

                            _wa = _wdf[_wdf["player_id"] == int(id_a)]
                            _wb = _wdf[_wdf["player_id"] == int(id_b)]

                            _recent_rows = []

                            # Games played
                            _gp_a = int(_wa.iloc[0].get("games_played", 0)) if not _wa.empty else 0
                            _gp_b = int(_wb.iloc[0].get("games_played", 0)) if not _wb.empty else 0
                            _recent_rows.append(
                                {
                                    "Stat": "Games Played",
                                    result["player_a"]: str(_gp_a) if _gp_a > 0 else "--",
                                    result["player_b"]: str(_gp_b) if _gp_b > 0 else "--",
                                }
                            )

                            # Hitter stats
                            if _is_hitter_a or _is_hitter_b:
                                for _scol, _slabel, _sfmt in [
                                    ("r", "Runs", None),
                                    ("hr", "Home Runs", None),
                                    ("rbi", "Runs Batted In", None),
                                    ("sb", "Stolen Bases", None),
                                    ("avg_calc", "Batting Average", "AVG"),
                                    ("obp_calc", "On-Base Percentage", "OBP"),
                                ]:
                                    _va_r = _wa.iloc[0].get(_scol, 0) if not _wa.empty else 0
                                    _vb_r = _wb.iloc[0].get(_scol, 0) if not _wb.empty else 0
                                    _va_r = float(_va_r) if _va_r is not None and _va_r == _va_r else 0
                                    _vb_r = float(_vb_r) if _vb_r is not None and _vb_r == _vb_r else 0
                                    if _sfmt:
                                        _va_s = format_stat(_va_r, _sfmt) if _va_r > 0 else "--"
                                        _vb_s = format_stat(_vb_r, _sfmt) if _vb_r > 0 else "--"
                                    else:
                                        _va_s = str(int(_va_r)) if _va_r > 0 else "--"
                                        _vb_s = str(int(_vb_r)) if _vb_r > 0 else "--"
                                    # Only show if at least one player is a hitter
                                    if _is_hitter_a or _is_hitter_b:
                                        _recent_rows.append(
                                            {
                                                "Stat": _slabel,
                                                result["player_a"]: _va_s if _is_hitter_a else "--",
                                                result["player_b"]: _vb_s if _is_hitter_b else "--",
                                            }
                                        )

                            # Pitcher stats
                            if not _is_hitter_a or not _is_hitter_b:
                                for _scol, _slabel, _sfmt in [
                                    ("ip", "Innings Pitched", None),
                                    ("w", "Wins", None),
                                    ("k", "Strikeouts", None),
                                    ("sv", "Saves", None),
                                    ("era_calc", "Earned Run Average", "ERA"),
                                    ("whip_calc", "WHIP", "WHIP"),
                                ]:
                                    _va_r = _wa.iloc[0].get(_scol, 0) if not _wa.empty else 0
                                    _vb_r = _wb.iloc[0].get(_scol, 0) if not _wb.empty else 0
                                    _va_r = float(_va_r) if _va_r is not None and _va_r == _va_r else 0
                                    _vb_r = float(_vb_r) if _vb_r is not None and _vb_r == _vb_r else 0
                                    if _sfmt:
                                        _va_s = format_stat(_va_r, _sfmt) if _va_r > 0 else "--"
                                        _vb_s = format_stat(_vb_r, _sfmt) if _vb_r > 0 else "--"
                                    elif _scol == "ip":
                                        _va_s = f"{_va_r:.1f}" if _va_r > 0 else "--"
                                        _vb_s = f"{_vb_r:.1f}" if _vb_r > 0 else "--"
                                    else:
                                        _va_s = str(int(_va_r)) if _va_r > 0 else "--"
                                        _vb_s = str(int(_vb_r)) if _vb_r > 0 else "--"
                                    # Only show if at least one player is a pitcher
                                    if not _is_hitter_a or not _is_hitter_b:
                                        _recent_rows.append(
                                            {
                                                "Stat": _slabel,
                                                result["player_a"]: _va_s if not _is_hitter_a else "--",
                                                result["player_b"]: _vb_s if not _is_hitter_b else "--",
                                            }
                                        )

                            if _recent_rows:
                                st.markdown(f"**{_window_label}**")
                                render_compact_table(pd.DataFrame(_recent_rows))

                        st.caption(
                            "Rolling totals from game logs. Rate stats (AVG, OBP, ERA, WHIP) "
                            "computed from weighted sums. '--' = no data or zero."
                        )
                except Exception:
                    pass  # Graceful fallback — don't crash if game logs unavailable

            # ── Statcast Profile Comparison ───────────────────────────────────
            try:
                _sc_a_pool = pool[pool["player_id"] == id_a]
                _sc_b_pool = pool[pool["player_id"] == id_b]
                if not _sc_a_pool.empty and not _sc_b_pool.empty:
                    _sc_ra = _sc_a_pool.iloc[0]
                    _sc_rb = _sc_b_pool.iloc[0]
                    _statcast_metrics = [
                        ("xwoba", "xwOBA", ".3f"),
                        ("barrel_pct", "Barrel %", ".1f"),
                        ("hard_hit_pct", "Hard Hit %", ".1f"),
                        ("stuff_plus", "Stuff+", ".0f"),
                    ]
                    _sc_rows = []
                    for _sc_col, _sc_label, _sc_fmt in _statcast_metrics:
                        if _sc_col in pool.columns:
                            _sc_va = float(_sc_ra.get(_sc_col, 0) or 0)
                            _sc_vb = float(_sc_rb.get(_sc_col, 0) or 0)
                            if _sc_va > 0 or _sc_vb > 0:
                                _sc_rows.append(
                                    {
                                        "Metric": _sc_label,
                                        result["player_a"]: f"{_sc_va:{_sc_fmt}}" if _sc_va > 0 else "--",
                                        result["player_b"]: f"{_sc_vb:{_sc_fmt}}" if _sc_vb > 0 else "--",
                                    }
                                )
                    if _sc_rows:
                        st.subheader("Statcast Profile")
                        render_compact_table(pd.DataFrame(_sc_rows))
                        st.caption(
                            "Statcast metrics from Baseball Savant. "
                            "xwOBA = expected weighted on-base average, "
                            "Barrel % = percentage of batted balls with ideal launch angle and exit velocity, "
                            "Hard Hit % = percentage of balls hit 95+ mph, "
                            "Stuff+ = pitch quality (100 = average)."
                        )
            except Exception:
                pass  # Graceful fallback — don't crash if Statcast data unavailable

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
            # NOTE: per-system projections are NOT in load_player_pool() (which only
            # carries the blended projection). The cached helper below scopes the
            # fetch to just the two compared players.
            try:
                systems = _load_per_system_projections((int(id_a), int(id_b)))

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

            # 4.6-A: use render_compact_table with html_cols so the colored dot <span>
            # in the Health column renders as HTML rather than escaped text.
            render_compact_table(pd.DataFrame(health_rows), html_cols={"Health"})

            # E10: Catcher Framing Value comparison
            try:
                from src.optimizer.matchup_adjustments import get_catcher_framing_data

                framing_data = get_catcher_framing_data()
                if framing_data:
                    framing_rows = []
                    for name_col, pid_col in [(player_a_name, id_a), (player_b_name, id_b)]:
                        p_match = pool[pool["player_id"] == pid_col]
                        if not p_match.empty:
                            positions = str(p_match.iloc[0].get("positions", ""))
                            if "C" in positions.split(","):
                                fd = framing_data.get(int(pid_col), {})
                                if fd:
                                    framing_runs = fd.get("framing_runs", 0)
                                    pop_time = fd.get("pop_time", 0)
                                    cs_pct = fd.get("cs_pct", 0)
                                    # ERA impact: ±0.01 per framing run
                                    era_impact = round(-0.01 * framing_runs, 2)
                                    tier = (
                                        "Elite"
                                        if framing_runs > 10
                                        else "Good"
                                        if framing_runs > 3
                                        else "Average"
                                        if framing_runs > -3
                                        else "Poor"
                                        if framing_runs > -10
                                        else "Liability"
                                    )
                                    framing_rows.append(
                                        {
                                            "Player": name_col,
                                            "Framing Runs": f"{framing_runs:+.1f}",
                                            "Pitcher ERA Impact": f"{era_impact:+.2f}",
                                            "Pop Time": f"{pop_time:.2f}s" if pop_time > 0 else "—",
                                            "CS%": f"{cs_pct:.0%}" if cs_pct > 0 else "—",
                                            "Tier": tier,
                                        }
                                    )
                    if framing_rows:
                        st.subheader("Catcher Framing Value")
                        render_styled_table(pd.DataFrame(framing_rows))
                        st.caption(
                            "Pitcher ERA differs 0.20-0.40 by catcher. Elite framers "
                            "generate 10-15 extra strikes per game."
                        )
            except Exception:
                pass  # Graceful fallback — don't crash page if framing data unavailable
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
            _readout = (
                '<div style="display:flex;gap:18px;flex-wrap:wrap;">'
                + build_stat_readout_html(_pa[:14], f"{_ca:+.2f}", accent=(_ca >= _cb))
                + build_stat_readout_html(_pb[:14], f"{_cb:+.2f}", accent=(_cb > _ca))
                + "</div>"
            )
            render_context_card("Composite Scores", _readout)
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
render_feedback_widget("Player Compare")

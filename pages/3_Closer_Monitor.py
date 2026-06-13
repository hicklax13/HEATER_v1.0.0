"""Closer Monitor — Track closer depth charts and job security across all 30 teams."""

from __future__ import annotations

import html as _html
import logging

import streamlit as st

from src.ai.chat import render_chat_widget
from src.auth import multi_user_enabled, require_auth, resolve_viewer_team_name
from src.closer_monitor import build_closer_grid, build_depth_data_from_db
from src.database import get_connection, init_db, load_player_pool
from src.feature_flags import require_page_enabled
from src.feedback import render_feedback_widget
from src.ui_shared import (
    THEME as T,
)
from src.ui_shared import (
    _headshot_img_html,
    build_heatbar_html,
    format_stat,
    inject_custom_css,
    jargon_help,
    page_timer_footer,
    page_timer_start,
    render_data_freshness_chip,
    render_glossary_expander,
    render_matchup_ticker,
    render_page_header,
    render_reco_banner,
    team_color,
    team_logo_url,
    text_on,
)
from src.usage import log_page_view

logger = logging.getLogger(__name__)

if not multi_user_enabled():
    st.set_page_config(
        page_title="Heater | Closer Monitor", page_icon="", layout="wide", initial_sidebar_state="collapsed"
    )

init_db()

inject_custom_css()
require_auth()
require_page_enabled("page:3_Closer_Monitor")
log_page_view("Closer Monitor")
render_chat_widget("Closer Monitor")
page_timer_start()

render_page_header(
    "Closer Monitor",
    eyebrow="BULLPEN",
    fig="FIG.03 — SAVE DEPTH CHART",
)
render_reco_banner("Closer depth chart", "", "closer")
render_data_freshness_chip("depth_charts")
render_glossary_expander(["% JOB", "gmLI"])
render_matchup_ticker()


@st.cache_data(ttl=300)
def _load_actual_sv_stats():
    """Load actual 2026 save stats for relievers."""
    import pandas as pd

    conn = get_connection()
    try:
        df = pd.read_sql_query(
            """SELECT p.name, ss.sv, ss.era, ss.whip, ss.games_played
               FROM season_stats ss
               JOIN players p ON ss.player_id = p.player_id
               WHERE ss.season = 2026 AND p.is_hitter = 0 AND ss.sv > 0
               ORDER BY ss.sv DESC""",
            conn,
        )
        if df.empty:
            return {}
        return {
            row["name"]: {
                "sv": int(row["sv"]),
                "era": float(row.get("era", 0) or 0),
                "whip": float(row.get("whip", 0) or 0),
            }
            for _, row in df.iterrows()
        }
    except Exception:
        return {}
    finally:
        conn.close()


# Load player pool for stat lookups
pool = load_player_pool()

# ── Viewer identity (MINE/FREE badge support) ─────────────────────────────────
# Fetch rosters once for team-resolver + ownership tagging.
_viewer_rosters = None
try:
    from src.yahoo_data_service import get_yahoo_data_service as _get_yds

    _viewer_rosters = _get_yds().get_rosters()
except Exception:
    pass

viewer_name = resolve_viewer_team_name(_viewer_rosters)

# Build the set of closer names rostered on the viewer's team.
# Used to render MINE vs FREE badges on each card.
_my_closer_names: set[str] = set()
if viewer_name and _viewer_rosters is not None and not _viewer_rosters.empty:
    try:
        _my_rows = _viewer_rosters[_viewer_rosters.get("team_name", _viewer_rosters.columns[0]) == viewer_name]
        for _col in ("name", "player_name"):
            if _col in _my_rows.columns:
                _my_closer_names = {str(n) for n in _my_rows[_col].dropna() if n}
                break
    except Exception:
        pass

# Normalize duplicate team abbreviations to canonical 30-team set
# Athletics: ATH is the 2026 canonical code (MLB Stats API + _PARK_FACTORS_EMERGENCY_2026
# + Wave 1 D1A-008 DB migration); fold legacy "OAK" → "ATH" on input.
_TEAM_NORMALIZE: dict[str, str] = {
    "OAK": "ATH",
    "AZ": "ARI",
    "WSN": "WSH",
    "CHW": "CWS",
    "TBR": "TB",
    "KCR": "KC",
    "SDP": "SD",
    "SFG": "SF",
}


def _normalize_team(abbr: str) -> str:
    return _TEAM_NORMALIZE.get(abbr.upper().strip(), abbr.upper().strip())


# Primary source: real bullpen-role classification from the DB
# (players.depth_chart_role, persisted by _persist_depth_chart_roles at bootstrap).
# DB-C2/DB-E1 fix — the page previously read st.session_state["closer_depth_data"],
# a key nothing ever wrote, so it always fell to the season-SV heuristic below.
_raw_depth_data: dict = build_depth_data_from_db()
# Normalize keys to canonical abbreviations
depth_data: dict = {}
for _k, _v in _raw_depth_data.items():
    _canon = _normalize_team(_k)
    if _canon not in depth_data or _v.get("closer_confidence", 0) > depth_data[_canon].get("closer_confidence", 0):
        depth_data[_canon] = _v

# Track whether we're showing real role data or an estimate, for an honest caption.
_using_role_data = bool(depth_data)

if not depth_data:
    # Fallback: no depth_chart_role rows in the DB (Roster Resource scrape is often
    # empty and the MLB Stats API fallback may not have run). Build a minimal,
    # clearly-flagged ESTIMATE from pitcher pool using sv projections.
    if not pool.empty:
        pitchers = pool[pool["is_hitter"] == 0].copy()
        if not pitchers.empty and "sv" in pitchers.columns:
            import pandas as pd

            pitchers["sv"] = pd.to_numeric(pitchers["sv"], errors="coerce").fillna(0)
            # Group by team, pick top SV pitcher as closer
            top_sv = pitchers[pitchers["sv"] > 0].sort_values("sv", ascending=False).drop_duplicates("team")
            if not top_sv.empty:
                for _, row in top_sv.iterrows():
                    team = _normalize_team(str(row.get("team", "")))
                    if not team:
                        continue
                    sv = float(row.get("sv", 0) or 0)
                    # Confidence heuristic: top SV earners get higher confidence
                    confidence = min(1.0, sv / 35.0)
                    if team not in depth_data:
                        depth_data[team] = {
                            "closer": str(row.get("name", "Unknown")),
                            "setup": [],
                            "closer_confidence": confidence,
                        }

if depth_data:
    if _using_role_data:
        st.info(
            "Closer depth charts are populated from the bullpen-role depth chart data "
            "(FanGraphs Roster Resource, with an MLB Stats API fallback) loaded at app launch. "
            "Run the app bootstrap to refresh."
        )
    else:
        st.warning(
            "No bullpen depth-chart roles available — showing an ESTIMATE: the highest "
            "projected-saves pitcher per team. Run the app bootstrap to load real depth "
            "chart roles (FanGraphs Roster Resource / MLB Stats API)."
        )

if not depth_data:
    st.warning(
        "No depth chart data available. Launch the app from the main page to bootstrap "
        "data from FanGraphs, or load sample data with: python load_sample_data.py"
    )
else:
    grid = build_closer_grid(depth_data, pool if not pool.empty else None)
    actual_sv_stats = _load_actual_sv_stats()

    if not grid:
        st.warning("No closer data to display.")
    else:
        # Column-header tooltips: use st.columns labels with help= so the user
        # can hover to understand the stat abbreviations.
        _hdr_c1, _hdr_c2, _hdr_c3 = st.columns([3, 3, 2])
        with _hdr_c1:
            st.caption(f"Showing {len(grid)} of 30 teams with closer data.")
        with _hdr_c2:
            _pct_job_help = jargon_help("% JOB")
            _gmli_help = jargon_help("gmLI")
            if viewer_name:
                _mine_count = sum(1 for it in grid if it["closer_name"] in _my_closer_names)
                st.caption(f"Your closers: {_mine_count}")
        with _hdr_c3:
            st.caption(
                "% JOB",
                help=jargon_help("% JOB"),
            )

        # Render in 5-column layout
        cols_per_row = 5
        for row_start in range(0, len(grid), cols_per_row):
            row_items = grid[row_start : row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col_idx, item in enumerate(row_items):
                with cols[col_idx]:
                    security = item["job_security"]
                    color = item["security_color"]
                    pct = int(security * 100)
                    closer_name = item["closer_name"]
                    actual = actual_sv_stats.get(closer_name, {})
                    actual_sv = actual.get("sv")
                    actual_era = actual.get("era")
                    actual_whip = actual.get("whip")

                    era_str = format_stat(item["era"], "ERA") if item["era"] else "—"
                    whip_str = format_stat(item["whip"], "WHIP") if item["whip"] else "—"
                    sv_str = f"{int(item['projected_sv'])}" if item["projected_sv"] else "—"
                    setup_str = _html.escape(", ".join(item["setup_names"])) if item["setup_names"] else ""
                    closer_name_safe = _html.escape(item["closer_name"])
                    headshot = _headshot_img_html(item.get("mlb_id"), size=34)

                    # MINE / FREE ownership badge
                    _is_mine = closer_name in _my_closer_names
                    if _is_mine:
                        _badge_html = (
                            '<span class="chip" style="font-size:9px;padding:1px 6px;'
                            "background:var(--fp-primary);color:#fff;"
                            'border-radius:4px;font-weight:700;letter-spacing:.05em;">MINE</span>'
                        )
                    elif viewer_name:
                        _badge_html = (
                            '<span class="chip" style="font-size:9px;padding:1px 6px;'
                            "background:var(--fp-surface);color:var(--fp-tx-muted);"
                            "border:1px solid var(--fp-divider);"
                            'border-radius:4px;font-weight:600;letter-spacing:.05em;">FREE</span>'
                        )
                    else:
                        _badge_html = ""

                    # Team identity: logo + official team color drive the card accent.
                    _team_abbr = str(item["team"])
                    _logo_url = team_logo_url(_team_abbr)
                    _tc = team_color(_team_abbr)
                    _team_label_ink = text_on(_tc)

                    # Recent form indicator from MatchupContextService
                    form_html = ""
                    try:
                        from src.matchup_context import get_matchup_context

                        _closer_ctx = get_matchup_context()
                        _closer_mlb = item.get("mlb_id")
                        if _closer_mlb is not None:
                            _closer_form = _closer_ctx.get_player_form(int(_closer_mlb))
                            _trend = _closer_form.get("trend", "neutral")
                            if _trend == "hot":
                                form_html = (
                                    '<div style="font-family:var(--font-mono);font-size:9px;'
                                    "letter-spacing:.1em;text-transform:uppercase;color:var(--fp-primary);"
                                    'font-weight:600;margin-top:3px;">Recent Form · HOT</div>'
                                )
                            elif _trend == "cold":
                                form_html = (
                                    '<div style="font-family:var(--font-mono);font-size:9px;'
                                    "letter-spacing:.1em;text-transform:uppercase;color:var(--fp-cold);"
                                    'font-weight:600;margin-top:3px;">Recent Form · COLD</div>'
                                )
                    except Exception:
                        pass

                    # gmLI trust indicator (data may come from session_state in future).
                    # Default is empty string — the row is omitted when no data exists.
                    gmli_html = ""
                    _gmli_data = st.session_state.get("closer_gmli_data", {})
                    _player_gmli = _gmli_data.get(closer_name, {})
                    _gmli_val = _player_gmli.get("gmli") if _player_gmli else None
                    _gmli_prev = _player_gmli.get("gmli_prev") if _player_gmli else None
                    if _gmli_val is not None:
                        if _gmli_val >= 1.8:
                            _trust_dot = T["green"]
                            _trust_label = "High Trust"
                        elif _gmli_val >= 1.0:
                            _trust_dot = T["warn"]
                            _trust_label = "Moderate"
                        else:
                            _trust_dot = T["danger"]
                            _trust_label = "Low Trust"
                        _trend_html = ""
                        if _gmli_prev is not None and (_gmli_prev - _gmli_val) > 0.5:
                            _trend_html = (
                                f' <span style="color:{T["danger"]};font-weight:700;">&#8595; Declining</span>'
                            )
                        gmli_html = (
                            f'<div style="font-size:0.6rem;margin-top:2px;white-space:nowrap;">'
                            f'<span style="display:inline-block;width:7px;height:7px;'
                            f"border-radius:50%;background:{_trust_dot};"
                            f'vertical-align:middle;margin-right:3px;"></span>'
                            f'<span style="color:{_trust_dot};font-weight:600;">'
                            f"gmLI {_gmli_val:.2f} - {_trust_label}</span>"
                            f"{_trend_html}</div>"
                        )

                    # Build actual stats line if available
                    actual_sv_html = ""
                    if actual_sv is not None:
                        actual_era_str = format_stat(actual_era, "ERA") if actual_era else "—"
                        actual_whip_str = format_stat(actual_whip, "WHIP") if actual_whip else "—"
                        actual_sv_html = (
                            f'<div style="font-family:var(--font-mono);font-size:9.5px;'
                            f"color:{T['green']};margin-top:3px;white-space:nowrap;font-weight:600;"
                            f'letter-spacing:.02em;">2026 ACTUAL · {actual_sv} SV · {actual_era_str} ERA · '
                            f"{actual_whip_str} WHIP</div>"
                        )

                    # SETUP row: only rendered when there are actual setup names.
                    setup_row_html = (
                        (
                            '<div style="font-family:var(--font-mono);font-size:9px;'
                            "color:var(--fp-tx-subtle);margin-top:4px;"
                            "white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"
                            'position:relative;z-index:1;letter-spacing:.02em;">'
                            f"SETUP · {setup_str}</div>"
                        )
                        if setup_str
                        else ""
                    )

                    # Job-security heat bar: orange (secure) → steel (shaky).
                    _security_bar = build_heatbar_html(pct, win=(security >= 0.5))
                    # Instrument card: white panel + 4 corner ticks + team-color
                    # left accent + a faint team-logo watermark, matching the
                    # dossier .panel look. Stat figures use Archivo tabular.
                    st.markdown(
                        f"""
<div class="instr-card" style="position:relative;background:var(--fp-surface);
     border:1px solid var(--fp-divider);border-left:4px solid {_tc};border-radius:10px;
     padding:11px 12px 10px;margin-bottom:9px;font-family:var(--font-body);
     min-height:172px;overflow:hidden;box-shadow:var(--fp-shadow);">
  <img src="{_logo_url}" alt="" aria-hidden="true" style="position:absolute;right:-14px;top:-12px;
       width:86px;height:86px;opacity:0.07;pointer-events:none;" />
  <span style="position:absolute;top:7px;left:7px;width:9px;height:9px;border-left:1px solid var(--fp-primary);border-top:1px solid var(--fp-primary);opacity:.55;"></span>
  <span style="position:absolute;top:7px;right:7px;width:9px;height:9px;border-right:1px solid var(--fp-primary);border-top:1px solid var(--fp-primary);opacity:.55;"></span>
  <span style="position:absolute;bottom:7px;left:7px;width:9px;height:9px;border-left:1px solid var(--fp-primary);border-bottom:1px solid var(--fp-primary);opacity:.55;"></span>
  <span style="position:absolute;bottom:7px;right:7px;width:9px;height:9px;border-right:1px solid var(--fp-primary);border-bottom:1px solid var(--fp-primary);opacity:.55;"></span>
  <div style="display:inline-flex;align-items:center;gap:5px;background:{_tc};color:{_team_label_ink};
       font-family:var(--font-display);font-size:9px;font-weight:800;letter-spacing:.1em;
       padding:2px 7px;border-radius:4px;position:relative;z-index:1;">
    <img src="{_logo_url}" alt="" style="width:12px;height:12px;vertical-align:middle;" />{_html.escape(_team_abbr)}
  </div>
  <div style="font-family:var(--font-display);font-size:0.92rem;font-weight:800;color:var(--fp-tx);
       margin:6px 0 3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
       display:flex;align-items:center;gap:5px;position:relative;z-index:1;">
    {headshot}{closer_name_safe}{_badge_html}
  </div>
  <div style="display:flex;align-items:center;gap:8px;margin:3px 0 5px;position:relative;z-index:1;">
    <div style="flex:1;">{_security_bar}</div>
    <span style="font-family:var(--font-mono);font-weight:600;font-size:11px;color:{color};white-space:nowrap;">{pct}% JOB</span>
  </div>
  <div style="display:flex;gap:0;border-top:1px solid var(--fp-divider);padding-top:5px;position:relative;z-index:1;">
    <div style="flex:1;text-align:center;border-right:1px solid var(--fp-divider);">
      <div style="font-family:var(--font-display);font-size:8.5px;font-weight:800;letter-spacing:.08em;color:var(--fp-tx-muted);">PROJ SV</div>
      <div style="font-family:var(--font-display);font-weight:800;font-size:15px;font-variant-numeric:tabular-nums;color:var(--fp-ember);">{sv_str}</div>
    </div>
    <div style="flex:1;text-align:center;border-right:1px solid var(--fp-divider);">
      <div style="font-family:var(--font-display);font-size:8.5px;font-weight:800;letter-spacing:.08em;color:var(--fp-tx-muted);">ERA</div>
      <div style="font-family:var(--font-display);font-weight:800;font-size:15px;font-variant-numeric:tabular-nums;color:var(--fp-tx);">{era_str}</div>
    </div>
    <div style="flex:1;text-align:center;">
      <div style="font-family:var(--font-display);font-size:8.5px;font-weight:800;letter-spacing:.08em;color:var(--fp-tx-muted);">WHIP</div>
      <div style="font-family:var(--font-display);font-weight:800;font-size:15px;font-variant-numeric:tabular-nums;color:var(--fp-tx);">{whip_str}</div>
    </div>
  </div>
  {actual_sv_html}
  {form_html}
  {gmli_html}
  {setup_row_html}
</div>
""",
                        unsafe_allow_html=True,
                    )

page_timer_footer("Closer Monitor")
render_feedback_widget("Closer Monitor")

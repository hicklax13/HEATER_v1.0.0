"""Bullpen Depth Chart — Closer/setup/committee status for all 30 MLB teams."""

import logging

import pandas as pd
import streamlit as st

from src.database import init_db, load_player_pool
from src.ui_shared import T, inject_custom_css, render_styled_table
from src.valuation import LeagueConfig

try:
    from src.contextual_factors import detect_closer_role

    _HAS_CLOSER_DETECTION = True
except ImportError:
    _HAS_CLOSER_DETECTION = False

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Heater | Bullpen", page_icon="", layout="wide")

init_db()
inject_custom_css()

st.markdown(
    '<div class="page-title-wrap"><div class="page-title"><span>BULLPEN DEPTH CHART</span></div></div>',
    unsafe_allow_html=True,
)

pool = load_player_pool()
if pool.empty:
    st.warning("No player data loaded.")
    st.stop()

pool = pool.rename(columns={"name": "player_name"})
config = LeagueConfig()

if not _HAS_CLOSER_DETECTION:
    st.error("Closer detection module not available.")
    st.stop()

# ── Filter to Relief Pitchers ───────────────────────────────────────────────

rp_pool = pool[
    (pool["is_hitter"] == 0) & (pool["positions"].apply(lambda p: "RP" in str(p) if pd.notna(p) else False))
].copy()

if rp_pool.empty:
    st.info("No relief pitchers found in the player pool.")
    st.stop()

# ── Detect Closer Roles ────────────────────────────────────────────────────

team_bullpens: dict[str, list[dict]] = {}

for _, player in rp_pool.iterrows():
    try:
        role_info = detect_closer_role(player)
    except Exception:
        role_info = {"role": "unknown", "confidence": 0.0, "draft_bonus": 0.0}

    team = str(player.get("team", "???"))
    entry = {
        "player_name": str(player.get("player_name", "Unknown")),
        "player_id": player.get("player_id"),
        "sv": float(player.get("sv", 0)),
        "era": float(player.get("era", 0)),
        "whip": float(player.get("whip", 0)),
        "role": role_info.get("role", "unknown"),
        "confidence": role_info.get("confidence", 0.0),
        "draft_bonus": role_info.get("draft_bonus", 0.0),
    }
    team_bullpens.setdefault(team, []).append(entry)

# ── Search/Filter ───────────────────────────────────────────────────────────

search = st.text_input("Search by team:", key="bullpen_search", placeholder="e.g. NYY, LAD, ATL...")

# ── Build Display Table ─────────────────────────────────────────────────────

rows = []
for team in sorted(team_bullpens.keys()):
    if search and search.upper() not in team.upper():
        continue

    relievers = team_bullpens[team]
    # Sort by saves (desc) then role confidence
    relievers.sort(key=lambda r: (-r["sv"], -r["confidence"]))

    closer = None
    setup = None
    is_committee = False

    for r in relievers:
        role = r["role"]
        if role == "closer" and closer is None:
            closer = r
        elif role in ("setup", "high_leverage") and setup is None:
            setup = r

    # Detect committee: no clear closer or low confidence
    if closer is None or closer["confidence"] < 0.4:
        is_committee = True
        # Pick highest SV pitcher as de facto closer
        if relievers:
            closer = relievers[0]

    closer_name = closer["player_name"] if closer else "—"
    closer_sv = f"{closer['sv']:.0f}" if closer else "—"
    closer_era = f"{closer['era']:.2f}" if closer else "—"

    setup_name = setup["player_name"] if setup else "—"
    setup_sv = f"{setup['sv']:.0f}" if setup else "—"

    conf = closer["confidence"] if closer else 0.0
    if conf >= 0.7:
        conf_label = "HIGH"
    elif conf >= 0.4:
        conf_label = "MEDIUM"
    else:
        conf_label = "LOW"

    rows.append(
        {
            "Team": team,
            "Closer": closer_name,
            "SV": closer_sv,
            "ERA": closer_era,
            "Setup": setup_name,
            "Setup SV": setup_sv,
            "Committee?": "Yes" if is_committee else "No",
            "Confidence": conf_label,
        }
    )

if not rows:
    st.info("No teams match your search.")
else:
    st.markdown(f"**{len(rows)} teams displayed**")

    # Color-code committee situations
    df = pd.DataFrame(rows)
    render_styled_table(df)

    # Committee alerts
    committees = [r for r in rows if r["Committee?"] == "Yes"]
    if committees:
        st.markdown("### Committee Alerts")
        st.caption("Teams without a clear closer — volatile saves category.")
        for c in committees:
            st.markdown(
                f'<div class="glass" style="padding:12px;margin:6px 0;border-left:4px solid {T["hot"]};">'
                f"<strong>{c['Team']}</strong> — {c['Closer']} ({c['SV']} SV) "
                f"| Confidence: {c['Confidence']}"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Handcuff targets (setup men on teams with LOW confidence closers)
    handcuffs = []
    for team, relievers in team_bullpens.items():
        if search and search.upper() not in team.upper():
            continue
        for r in relievers:
            if r["role"] in ("setup", "high_leverage") and r["sv"] >= 1:
                # Check if team's closer has low confidence
                team_closer = next((x for x in relievers if x["role"] == "closer"), None)
                if team_closer is None or team_closer["confidence"] < 0.5:
                    handcuffs.append(
                        {
                            "Team": team,
                            "Handcuff": r["player_name"],
                            "SV": f"{r['sv']:.0f}",
                            "ERA": f"{r['era']:.2f}",
                        }
                    )

    if handcuffs:
        st.markdown("### Handcuff Targets")
        st.caption("Setup men on teams with shaky closer situations — potential closer-in-waiting.")
        render_styled_table(pd.DataFrame(handcuffs[:10]))

"""Weekly Dashboard — Matchup heatmap, start/sit advisor, and two-start pitcher planner."""

import logging
import time

import pandas as pd
import streamlit as st

from src.database import init_db, load_league_rosters, load_player_pool
from src.league_manager import get_team_roster
from src.ui_shared import METRIC_TOOLTIPS, T, inject_custom_css, render_styled_table
from src.valuation import LeagueConfig

logger = logging.getLogger(__name__)

# ── Optional imports with graceful degradation ────────────────────

try:
    from src.matchup_planner import color_tier, compute_weekly_matchup_ratings

    MATCHUP_AVAILABLE = True
except ImportError:
    MATCHUP_AVAILABLE = False

try:
    from src.start_sit import start_sit_recommendation

    START_SIT_AVAILABLE = True
except ImportError:
    START_SIT_AVAILABLE = False

try:
    from src.two_start import identify_two_start_pitchers

    TWO_START_AVAILABLE = True
except ImportError:
    TWO_START_AVAILABLE = False

try:
    from src.data_bootstrap import PARK_FACTORS
except ImportError:
    PARK_FACTORS = {}


# ── Page Config ───────────────────────────────────────────────────

st.set_page_config(page_title="Heater | Weekly Dashboard", page_icon="", layout="wide")

init_db()

inject_custom_css()

st.markdown(
    '<div class="page-title-wrap"><div class="page-title"><span>WEEKLY DASHBOARD</span></div></div>',
    unsafe_allow_html=True,
)


# ── Data Loading ──────────────────────────────────────────────────

pool = load_player_pool()
if pool.empty:
    st.warning("No player data loaded. Launch the app from the main page to bootstrap data.")
    st.stop()

pool = pool.rename(columns={"name": "player_name"})
config = LeagueConfig()

# Standard roster loading with Yahoo sync fallback
rosters = load_league_rosters()
if rosters.empty:
    if st.session_state.get("yahoo_connected"):
        st.warning("Yahoo is connected but no roster data found in the database. Try syncing:")
        if st.button("Sync League Data Now", key="sync_league_weekly"):
            client = st.session_state.get("yahoo_client")
            if client:
                progress = st.progress(0, text="Connecting to Yahoo Fantasy...")
                try:
                    progress.progress(30, text="Fetching league standings...")
                    sync_result = client.sync_to_db()
                    progress.progress(100, text="Sync complete!")
                    standings_count = sync_result.get("standings", 0) if sync_result else 0
                    rosters_count = sync_result.get("rosters", 0) if sync_result else 0
                    if rosters_count > 0:
                        st.success(f"Synced {rosters_count} roster entries and {standings_count} standing entries.")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.warning(
                            f"Sync completed but Yahoo returned no roster data "
                            f"(standings: {standings_count}). This may mean the league "
                            f"season hasn't started yet on Yahoo, or rosters haven't been set."
                        )
                except Exception as e:
                    progress.empty()
                    st.error(f"Sync failed: {e}")
            else:
                st.error("Yahoo client not found in session. Return to Connect League and reconnect.")
    else:
        st.warning(
            "No league data loaded. Connect your Yahoo league in Connect League, "
            "or league data will load automatically on next app launch."
        )
    st.stop()

user_teams = rosters[rosters["is_user_team"] == 1]
if user_teams.empty:
    st.warning("No user team identified in roster data.")
    st.stop()

user_team_name = user_teams.iloc[0]["team_name"]
user_roster = get_team_roster(user_team_name)

# Build name-to-pool-id mapping for roster ID resolution
name_to_pool_id = dict(zip(pool["player_name"], pool["player_id"])) if not pool.empty else {}
user_roster_ids: list[int] = []
if not user_roster.empty and "name" in user_roster.columns:
    for _, r in user_roster.iterrows():
        pname = r.get("name", "")
        pool_pid = name_to_pool_id.get(pname)
        if pool_pid is not None:
            user_roster_ids.append(pool_pid)
        else:
            user_roster_ids.append(r["player_id"])
else:
    user_roster_ids = user_roster["player_id"].tolist() if not user_roster.empty else []

# Build roster DataFrame from player pool for downstream consumers
roster_pool = pool[pool["player_id"].isin(user_roster_ids)].copy() if user_roster_ids else pd.DataFrame()


# ── Color helpers ─────────────────────────────────────────────────

_TIER_COLORS: dict[str, str] = {
    "smash": T["green"],
    "favorable": T["sky"],
    "neutral": "#6b7280",
    "unfavorable": T["hot"],
    "avoid": T["primary"],
}

_MATCHUP_SCORE_COLORS: dict[str, str] = {
    "high": T["green"],
    "mid": T["warn"],
    "low": T["primary"],
}


def _matchup_score_color(score: float) -> str:
    """Return a hex color based on matchup score (0-10)."""
    if score >= 7.0:
        return _MATCHUP_SCORE_COLORS["high"]
    if score >= 4.0:
        return _MATCHUP_SCORE_COLORS["mid"]
    return _MATCHUP_SCORE_COLORS["low"]


def _tier_badge_html(tier: str, percentile_rank: float | None = None) -> str:
    """Return an HTML badge for a matchup tier."""
    # Use color_tier() for authoritative tier→color mapping when available
    if MATCHUP_AVAILABLE and percentile_rank is not None:
        resolved_tier = color_tier(percentile_rank)
        color = _TIER_COLORS.get(resolved_tier, "#6b7280")
    else:
        color = _TIER_COLORS.get(tier, "#6b7280")
    label = tier.replace("_", " ").title()
    return (
        f'<span style="background:{color};color:#ffffff;padding:2px 10px;'
        f'border-radius:12px;font-size:0.82em;font-weight:600;">{label}</span>'
    )


def _score_badge_html(score: float) -> str:
    """Return an HTML badge for a numeric matchup score."""
    color = _matchup_score_color(score)
    return (
        f'<span style="background:{color};color:#ffffff;padding:2px 10px;'
        f'border-radius:12px;font-size:0.82em;font-weight:700;">{score:.1f}</span>'
    )


def _positions_for_player(row: pd.Series) -> list[str]:
    """Parse comma-separated positions from a roster row."""
    raw = row.get("positions", "")
    if pd.isna(raw) or not raw:
        return []
    return [p.strip() for p in str(raw).split(",") if p.strip()]


# ── Tabs ──────────────────────────────────────────────────────────

tab_heatmap, tab_start_sit, tab_two_start = st.tabs(["Matchup Heatmap", "Start / Sit Advisor", "Two-Start Pitchers"])


# ── Tab 1: Matchup Heatmap ───────────────────────────────────────

with tab_heatmap:
    if not MATCHUP_AVAILABLE:
        st.warning("Matchup planner module is unavailable. Check installation.")
    elif roster_pool.empty:
        st.info("No roster players found in the player pool.")
    else:
        try:
            with st.spinner("Computing weekly matchup ratings..."):
                ratings_df = compute_weekly_matchup_ratings(
                    roster=roster_pool,
                    park_factors=PARK_FACTORS or None,
                    config=config,
                )
        except Exception as exc:
            logger.exception("Matchup rating computation failed")
            st.error(f"Schedule data unavailable. Try again later. ({exc})")
            ratings_df = pd.DataFrame()

        if ratings_df.empty:
            st.info("No matchup ratings available for this week.")
        else:
            # Split into hitters and pitchers
            hitters_df = ratings_df[ratings_df["is_hitter"] == 1].copy()
            pitchers_df = ratings_df[ratings_df["is_hitter"] == 0].copy()

            for section_label, section_df in [("Hitters", hitters_df), ("Pitchers", pitchers_df)]:
                st.subheader(section_label)
                if section_df.empty:
                    st.info(f"No {section_label.lower()} matchup data available.")
                    continue

                # Sort by rating descending
                section_df = section_df.sort_values("weekly_matchup_rating", ascending=False).reset_index(drop=True)

                # Build display table
                display_rows = []
                for _, row in section_df.iterrows():
                    tier = str(row.get("matchup_tier", "neutral")).lower()
                    rating = float(row.get("weekly_matchup_rating", 5.0))
                    games_count = int(row.get("games_count", 0))
                    display_rows.append(
                        {
                            "Player": str(row.get("name", "")),
                            "Position": str(row.get("positions", "")),
                            "Games": games_count,
                            "Rating": _score_badge_html(rating),
                            "Matchup Tier": _tier_badge_html(tier),
                        }
                    )

                display_df = pd.DataFrame(display_rows)
                render_styled_table(display_df, max_height=420)

            st.caption(
                METRIC_TOOLTIPS.get(
                    "matchup_rating",
                    "Weekly matchup rating (1-10) based on opponent quality, park factors, "
                    "and platoon splits. Higher is better.",
                )
            )


# ── Tab 2: Start/Sit Advisor ─────────────────────────────────────

with tab_start_sit:
    if not START_SIT_AVAILABLE:
        st.warning("Start/sit advisor module is unavailable. Check installation.")
    elif roster_pool.empty:
        st.info("No roster players found in the player pool.")
    else:
        # Gather all positions on the roster
        all_positions: set[str] = set()
        for _, row in roster_pool.iterrows():
            all_positions.update(_positions_for_player(row))

        position_list = sorted(all_positions) if all_positions else ["C", "1B", "2B", "3B", "SS", "OF", "SP", "RP"]

        selected_position = st.selectbox(
            "Choose position slot",
            options=position_list,
            key="ss_position",
        )

        # Filter roster to eligible players at that position
        eligible_ids: list[int] = []
        eligible_names: dict[int, str] = {}
        for _, row in roster_pool.iterrows():
            player_positions = _positions_for_player(row)
            if selected_position in player_positions:
                pid = int(row["player_id"])
                eligible_ids.append(pid)
                eligible_names[pid] = str(row.get("player_name", row.get("name", f"ID {pid}")))

        if len(eligible_ids) < 2:
            st.info(f"Need at least 2 eligible players at {selected_position} to compare. Found {len(eligible_ids)}.")
        else:
            # Multiselect for 2-4 players
            default_selection = eligible_ids[: min(4, len(eligible_ids))]
            chosen_ids = st.multiselect(
                f"Select 2-4 players at {selected_position}",
                options=eligible_ids,
                default=default_selection[:4],
                format_func=lambda pid: eligible_names.get(pid, str(pid)),
                key="ss_players",
            )

            if len(chosen_ids) < 2:
                st.info("Select at least 2 players to compare.")
            elif len(chosen_ids) > 4:
                st.warning("Maximum 4 players. Please deselect some.")
            else:
                if st.button("Compare", key="ss_compare", type="primary"):
                    try:
                        with st.spinner("Evaluating start/sit decision..."):
                            result = start_sit_recommendation(
                                player_ids=chosen_ids,
                                player_pool=pool,
                                config=config,
                                park_factors=PARK_FACTORS or None,
                            )
                    except Exception as exc:
                        logger.exception("Start/sit recommendation failed")
                        st.error(f"Schedule data unavailable. Try again later. ({exc})")
                        result = None

                    if result and result.get("players"):
                        winner_id = result.get("recommendation")
                        confidence_label = result.get("confidence_label", "")
                        confidence_val = result.get("confidence", 0.0)

                        st.markdown(
                            f'<p style="font-size:0.95em;color:{T["tx2"]};">'
                            f"Confidence: <strong>{confidence_label}</strong> ({confidence_val:.0%})</p>",
                            unsafe_allow_html=True,
                        )

                        for player_info in result["players"]:
                            pid = player_info.get("player_id")
                            pname = player_info.get("name", f"Player {pid}")
                            score = player_info.get("start_score", 0.0)
                            floor_val = player_info.get("floor", 0.0)
                            ceiling_val = player_info.get("ceiling", 0.0)
                            reasoning = player_info.get("reasoning", [])

                            is_winner = pid == winner_id
                            border_color = T["green"] if is_winner else T["border"]
                            border_width = "3px" if is_winner else "1px"
                            winner_tag = (
                                f'<span style="background:{T["green"]};color:#ffffff;'
                                f"padding:2px 8px;border-radius:8px;font-size:0.78em;"
                                f'font-weight:700;margin-left:8px;">START</span>'
                                if is_winner
                                else ""
                            )

                            reasoning_html = ""
                            if reasoning:
                                bullets = "".join(f"<li>{r}</li>" for r in reasoning if isinstance(r, str))
                                reasoning_html = (
                                    f'<ul style="margin:6px 0 0 16px;padding:0;'
                                    f'color:{T["tx2"]};font-size:0.85em;">{bullets}</ul>'
                                )

                            st.markdown(
                                f'<div style="border:{border_width} solid {border_color};'
                                f"border-radius:12px;padding:16px;margin-bottom:12px;"
                                f'background:{T["card"]};">'
                                f'<div style="font-size:1.05em;font-weight:700;'
                                f'color:{T["tx"]};">{pname}{winner_tag}</div>'
                                f'<div style="display:flex;gap:24px;margin-top:8px;'
                                f'font-size:0.9em;color:{T["tx2"]};">'
                                f"<span>Score: <strong>{score:.2f}</strong></span>"
                                f"<span>Floor: {floor_val:.2f}</span>"
                                f"<span>Ceiling: {ceiling_val:.2f}</span>"
                                f"</div>"
                                f"{reasoning_html}"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                    elif result is not None:
                        st.info("No recommendation could be generated for these players.")


# ── Tab 3: Two-Start Pitchers ────────────────────────────────────

with tab_two_start:
    if not TWO_START_AVAILABLE:
        st.warning("Two-start pitcher module is unavailable. Check installation.")
    else:
        # Compute team-level ERA/WHIP/IP from user roster pitchers for rate damage context
        team_era = 4.00
        team_whip = 1.25
        team_ip = 55.0
        if not roster_pool.empty:
            pitchers = roster_pool[roster_pool["is_hitter"] == 0]
            if not pitchers.empty:
                total_ip = pd.to_numeric(pitchers.get("ip", pd.Series(dtype=float)), errors="coerce").sum()
                if total_ip > 0:
                    total_er = pd.to_numeric(pitchers.get("er", pd.Series(dtype=float)), errors="coerce").sum()
                    total_bb = pd.to_numeric(pitchers.get("bb_allowed", pd.Series(dtype=float)), errors="coerce").sum()
                    total_ha = pd.to_numeric(pitchers.get("h_allowed", pd.Series(dtype=float)), errors="coerce").sum()
                    team_era = (total_er * 9.0) / total_ip if total_ip > 0 else 4.00
                    team_whip = (total_bb + total_ha) / total_ip if total_ip > 0 else 1.25
                    team_ip = total_ip

        try:
            with st.spinner("Scanning upcoming schedule for two-start pitchers..."):
                two_starters = identify_two_start_pitchers(
                    days_ahead=7,
                    team_era=team_era,
                    team_whip=team_whip,
                    team_ip=team_ip,
                    player_pool=pool if not pool.empty else None,
                )
        except Exception as exc:
            logger.exception("Two-start pitcher identification failed")
            st.error(f"Schedule data unavailable. Try again later. ({exc})")
            two_starters = []

        if not two_starters:
            st.info("No two-start pitchers found for this week. This is expected during the offseason.")
        else:
            # Sort by avg matchup score descending
            two_starters.sort(key=lambda x: x.get("avg_matchup_score", 0.0), reverse=True)

            display_rows = []
            for sp in two_starters:
                avg_score = float(sp.get("avg_matchup_score", 0.0))
                era_dmg = sp.get("rate_damage_weekly", {}).get("era_change", 0.0)
                whip_dmg = sp.get("rate_damage_weekly", {}).get("whip_change", 0.0)
                streaming_val = float(sp.get("streaming_value", 0.0))

                # Format rate damage as signed string
                era_str = f"{era_dmg:+.3f}" if era_dmg else "0.000"
                whip_str = f"{whip_dmg:+.3f}" if whip_dmg else "0.000"

                display_rows.append(
                    {
                        "Pitcher": str(sp.get("pitcher_name", "")),
                        "Team": str(sp.get("team", "")),
                        "Starts": int(sp.get("num_starts", 2)),
                        "Avg Matchup Score": _score_badge_html(avg_score),
                        "ERA Impact": era_str,
                        "WHIP Impact": whip_str,
                        "Streaming Value": f"{streaming_val:.2f}",
                    }
                )

            display_df = pd.DataFrame(display_rows)
            render_styled_table(display_df, max_height=500)

            st.caption(
                "Matchup score ranges from 0 (worst) to 10 (best) based on opponent quality, "
                "park factors, and pitcher skill. ERA/WHIP impact shows the expected change to "
                "your team's rate stats from starting this pitcher. Streaming value measures the "
                "net Standings Gained Points contribution."
            )

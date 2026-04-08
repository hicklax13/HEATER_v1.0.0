"""Proactive Alert System — AVIS Section 6 communication.

Monitors for roster-impacting events and generates actionable alerts.
Analyst tone, not cheerleader. Data-driven recommendations.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta, timezone

import pandas as pd

ET = timezone(timedelta(hours=-4))


def _fmt_et(dt_obj: datetime) -> str:
    """Format a datetime in ET as 'Apr 05, 7:34 PM ET' (cross-platform)."""
    dt_et = dt_obj.astimezone(ET)
    hour = dt_et.hour % 12 or 12
    ampm = "AM" if dt_et.hour < 12 else "PM"
    return f"{dt_et.strftime('%b %d')}, {hour}:{dt_et.strftime('%M')} {ampm} ET"


logger = logging.getLogger(__name__)

# AVIS Section 7: IL stash players — do NOT drop or trade within 2 weeks of return
IL_STASH_NAMES: set[str] = {"Shane Bieber", "Spencer Strider"}


def generate_roster_alerts(
    roster: pd.DataFrame,
    player_news: pd.DataFrame | None = None,
    closer_data: dict | None = None,
    max_roster_size: int = 23,
    fa_pool: pd.DataFrame | None = None,
    user_roster_ids: list | None = None,
    player_pool: pd.DataFrame | None = None,
    transactions: pd.DataFrame | None = None,
    user_team_name: str = "",
) -> list[dict]:
    """Generate proactive alerts based on current roster state.

    Checks:
    1. Empty roster spots (AVIS Rule #4)
    2. Injured players not on IL
    3. Closer role changes
    4. Low closer count (AVIS Rule #2)
    5. League trade monitoring (opponent acquisitions)

    Returns list of alert dicts: {type, severity, title, message, action}.
    """
    alerts = []

    if roster.empty:
        return alerts

    roster_size = len(roster)

    # Alert 1: Empty roster spots
    if roster_size < max_roster_size:
        empty = max_roster_size - roster_size
        # Auto-compute top 3 FA fill recommendations if data is available
        top_fills_str = ""
        if fa_pool is not None and not fa_pool.empty and user_roster_ids is not None and player_pool is not None:
            try:
                from src.database import get_all_rostered_player_ids
                from src.in_season import rank_free_agents

                # Ensure fa_pool only contains true free agents (not opponents' players)
                _all_rostered = get_all_rostered_player_ids()
                if _all_rostered:
                    fa_pool = fa_pool[~fa_pool["player_id"].isin(_all_rostered)]
                ranked = rank_free_agents(user_roster_ids, fa_pool, player_pool)
                if not ranked.empty:
                    top3 = ranked.head(3)
                    names = [str(r.get("player_name", r.get("name", "?"))) for _, r in top3.iterrows()]
                    top_fills_str = f" Top fills: {', '.join(names)}."
            except Exception:
                pass
        alerts.append(
            {
                "type": "empty_roster",
                "severity": "critical",
                "title": f"EMPTY ROSTER SPOTS ({empty})",
                "message": f"You have {empty} empty roster slot(s). An empty spot is a zero — any player is better than nothing.{top_fills_str}",
                "action": "Go to Free Agents and add the top-ranked available player immediately.",
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

    # Alert 2: Injured starters not on IL (roster-only)
    if player_news is not None and not player_news.empty:
        injury_news = player_news[player_news["news_type"] == "injury"]
        if not injury_news.empty:
            # Filter to only players on this roster
            roster_pids = set()
            roster_names = set()
            if roster is not None and not roster.empty:
                if "player_id" in roster.columns:
                    roster_pids = set(roster["player_id"].dropna().astype(int).tolist())
                if "name" in roster.columns:
                    roster_names = {str(n).strip().lower() for n in roster["name"].dropna()}

            recent = injury_news.sort_values("fetched_at", ascending=False).head(20)
            shown = 0
            _alerted_players: set[str] = set()
            for _, news in recent.iterrows():
                if shown >= 5:
                    break
                # Check if this injury is for a rostered player
                news_pid = news.get("player_id")
                news_name = str(news.get("player_name", "")).strip().lower()
                on_roster = False
                if news_pid is not None:
                    try:
                        on_roster = int(news_pid) in roster_pids
                    except (ValueError, TypeError):
                        pass
                if not on_roster and news_name:
                    on_roster = news_name in roster_names

                if not on_roster:
                    continue

                # Deduplicate: only show the most recent alert per player
                if news_name in _alerted_players:
                    continue

                il_status = news.get("il_status", "")
                if il_status and "IL" in str(il_status).upper():
                    _alerted_players.add(news_name)
                    alerts.append(
                        {
                            "type": "injury",
                            "severity": "warning",
                            "title": f"INJURY: {news.get('headline', 'Player injured')}",
                            "message": f"Status: {il_status}. Check if IL slot is available.",
                            "action": "Move to IL and pick up a replacement from free agents.",
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                    )
                    shown += 1

    # Alert 3: Closer count (check actual SV OR projected SV for early-season)
    closer_count = 0
    closer_names = []
    proj_sv_map = {}
    try:
        from src.database import get_connection

        conn = get_connection()
        try:
            roster_ids = roster["player_id"].dropna().astype(int).tolist()
            if roster_ids:
                placeholders = ",".join("?" * len(roster_ids))
                proj_df = pd.read_sql_query(
                    f"SELECT player_id, sv FROM projections WHERE system='blended' AND player_id IN ({placeholders})",
                    conn,
                    params=roster_ids,
                )
                proj_sv_map = dict(zip(proj_df["player_id"], proj_df["sv"]))
        finally:
            conn.close()
    except Exception:
        pass

    for _, p in roster.iterrows():
        actual_sv = float(p.get("sv", 0) or 0)
        pid = p.get("player_id")
        proj_sv = float(proj_sv_map.get(pid, 0) or 0)
        if actual_sv >= 5 or proj_sv >= 5:
            closer_count += 1
            closer_names.append(p.get("name", "?"))

    if closer_count < 2:
        alerts.append(
            {
                "type": "closer_shortage",
                "severity": "warning",
                "title": f"CLOSER ALERT: Only {closer_count} closer(s)",
                "message": f"Current closers: {', '.join(closer_names) if closer_names else 'None'}. AVIS requires minimum 2 closers at all times.",
                "action": "Check Waiver Wire for available closers (sorted by projected SV).",
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

    # Alert 5: League trade monitoring — flag opponent acquisitions
    if transactions is not None and not transactions.empty:
        _user_lower = user_team_name.strip().lower() if user_team_name else ""
        trade_txns = transactions[transactions["type"].str.lower() == "trade"]
        if not trade_txns.empty:
            # Group by transaction_id to show each trade once
            shown_txns: set[str] = set()
            for _, tx in trade_txns.iterrows():
                txid = str(tx.get("transaction_id", ""))
                if txid in shown_txns:
                    continue
                team_to = str(tx.get("team_to", "")).strip()
                team_from = str(tx.get("team_from", "")).strip()
                player_name = str(tx.get("player_name", ""))
                # Skip trades involving the user's own team
                if _user_lower and (team_to.lower() == _user_lower or team_from.lower() == _user_lower):
                    continue
                if not team_to or not player_name:
                    continue
                shown_txns.add(txid)
                alerts.append(
                    {
                        "type": "league_trade",
                        "severity": "info",
                        "title": f"LEAGUE TRADE: {player_name} to {team_to}",
                        "message": f"{team_from} traded {player_name} to {team_to}. Monitor impact on opponent strength.",
                        "action": "Check if this changes your upcoming matchup strategy.",
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                )

    # Alert 4: IL stash return watch — dynamic date-based protection
    _return_dates = _get_il_return_dates(roster)
    _now = datetime.now(UTC)
    _two_weeks = timedelta(weeks=2)

    for _, p in roster.iterrows():
        name = p.get("name", "")
        pid = p.get("player_id")

        # Check dynamic return-date window from ESPN injury data
        return_date = _return_dates.get(pid) or _return_dates.get(name)
        if return_date and (return_date - _now) <= _two_weeks:
            days_away = max(0, (return_date - _now).days)
            alerts.append(
                {
                    "type": "il_watch",
                    "severity": "warning",
                    "title": f"IL STASH — PROTECTED: {name}",
                    "message": f"{name} expected to return in ~{days_away} day(s) ({return_date.strftime('%b %d')}). Do NOT drop.",
                    "action": "Hold this player — return is imminent. Clear an IL slot if needed.",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
        elif name in IL_STASH_NAMES:
            # Fallback: named stash players always get a reminder even without date data
            alerts.append(
                {
                    "type": "il_watch",
                    "severity": "info",
                    "title": f"IL STASH: {name}",
                    "message": f"{name} is on your IL. Monitor return timeline — playoff weapon if healthy by August.",
                    "action": "Do NOT drop within 2 weeks of expected return date.",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

    # Alert 6: M5 — IL slot utilization
    # If team has empty IL slots, suggest stashing injured players.
    # IL slots = free storage for upside players. Empty IL slot = wasted value.
    try:
        il_capacity = 4  # Standard Yahoo leagues have 4 IL slots
        # Count players currently in IL slots
        il_occupied = 0
        for _, row in roster.iterrows():
            sel_pos = str(row.get("selected_position", "")).upper()
            status = str(row.get("status", "")).lower()
            if sel_pos in ("IL", "IL+", "DL") or status in ("il10", "il15", "il60", "dl"):
                il_occupied += 1

        empty_il = il_capacity - il_occupied
        if empty_il > 0 and fa_pool is not None and not fa_pool.empty:
            # Find IL-eligible FAs with highest ROS SGP
            il_candidates = fa_pool[
                fa_pool.get("status", pd.Series("", index=fa_pool.index)).str.lower().isin(
                    ["il10", "il15", "il60", "dl", "out"]
                )
            ]
            if not il_candidates.empty:
                # Sort by projected value (SGP or marginal_value)
                val_col = "marginal_value" if "marginal_value" in il_candidates.columns else "adp"
                if val_col == "adp":
                    il_candidates = il_candidates.sort_values("adp", ascending=True)
                else:
                    il_candidates = il_candidates.sort_values(val_col, ascending=False)
                top = il_candidates.head(3)
                names = [str(r.get("name", r.get("player_name", "?"))) for _, r in top.iterrows()]
                alerts.append(
                    {
                        "type": "il_slot",
                        "severity": "warning",
                        "title": f"EMPTY IL SLOT ({empty_il} available)",
                        "message": (
                            f"You have {empty_il} empty IL slot(s). "
                            f"Stash injured upside: {', '.join(names)}. "
                            "Free storage — no roster cost until they return."
                        ),
                        "action": f"Add one of: {', '.join(names)} to IL slot.",
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                )
    except Exception:
        pass

    return alerts


def generate_opponent_move_alerts(
    transactions: pd.DataFrame,
    opponent_team_name: str,
    days_back: int = 7,
) -> list[dict]:
    """M3: Track opponent's recent roster moves (adds, drops, trades).

    Surfaces mid-week streaming adds, pitcher pickups, and trades that
    could change the opponent's strategy for this week's matchup.

    Args:
        transactions: All league transactions (from yds.get_transactions()).
        opponent_team_name: Current week's opponent team name.
        days_back: How many days back to look (default 7).

    Returns:
        List of alert dicts with opponent move details.
    """
    alerts = []
    if transactions.empty or not opponent_team_name:
        return alerts

    opp_lower = opponent_team_name.strip().lower()
    cutoff = (datetime.now(UTC) - timedelta(days=days_back)).isoformat()

    for _, tx in transactions.iterrows():
        team_to = str(tx.get("team_to", "")).strip().lower()
        team_from = str(tx.get("team_from", "")).strip().lower()
        ts = str(tx.get("timestamp", ""))
        player = str(tx.get("player_name", "Unknown"))
        tx_type = str(tx.get("type", ""))

        # Filter to opponent's moves within time window
        if ts < cutoff:
            continue
        if team_to != opp_lower and team_from != opp_lower:
            continue

        if team_to == opp_lower and tx_type in ("add", "trade"):
            alerts.append({
                "type": "opponent_add",
                "severity": "info",
                "title": f"OPP ADD: {player}",
                "message": f"{opponent_team_name} added {player} ({tx_type}). Check if this targets your weak categories.",
                "action": "Review their streaming strategy and adjust.",
                "timestamp": ts,
            })
        elif team_from == opp_lower and tx_type in ("drop", "trade"):
            alerts.append({
                "type": "opponent_drop",
                "severity": "info",
                "title": f"OPP DROP: {player}",
                "message": f"{opponent_team_name} dropped {player}. Potential waiver target if valuable.",
                "action": f"Check if {player} is worth claiming.",
                "timestamp": ts,
            })

    return alerts


def compute_swap_impacts(
    roster: pd.DataFrame,
    player_pool: pd.DataFrame,
    config=None,
) -> list[dict]:
    """M2: Compute category impact of benching one player and starting another.

    Finds bench players who could start and shows the marginal impact
    on each counting category. Helps answer "should I start X over Y?"

    Returns:
        List of dicts with bench_player, start_player, category impacts.
    """
    if roster.empty or player_pool.empty:
        return []

    try:
        from src.valuation import LeagueConfig, SGPCalculator

        if config is None:
            config = LeagueConfig()
        sgp_calc = SGPCalculator(config)

        # Identify starters and bench
        starters = []
        bench = []
        for _, row in roster.iterrows():
            sel_pos = str(row.get("selected_position", "")).upper()
            pid = int(row.get("player_id", 0))
            if sel_pos in ("BN", "IL", "IL+", "NA", "DL"):
                bench.append(pid)
            elif sel_pos and pid:
                starters.append(pid)

        if not bench or not starters:
            return []

        swaps = []
        for bench_pid in bench:
            bp = player_pool[player_pool["player_id"] == bench_pid]
            if bp.empty:
                continue
            bp_row = bp.iloc[0]
            bp_sgp = sgp_calc.player_sgp(bp_row)
            bp_name = str(bp_row.get("name", bp_row.get("player_name", "?")))
            bp_is_hitter = int(bp_row.get("is_hitter", 1))

            # Find worst starter of same type to swap with
            best_swap = None
            best_delta = -999
            for start_pid in starters:
                sp = player_pool[player_pool["player_id"] == start_pid]
                if sp.empty:
                    continue
                sp_row = sp.iloc[0]
                if int(sp_row.get("is_hitter", 1)) != bp_is_hitter:
                    continue
                sp_sgp = sgp_calc.player_sgp(sp_row)
                total_delta = sum(bp_sgp.values()) - sum(sp_sgp.values())
                if total_delta > best_delta:
                    best_delta = total_delta
                    sp_name = str(sp_row.get("name", sp_row.get("player_name", "?")))
                    impacts = {cat: round(bp_sgp.get(cat, 0) - sp_sgp.get(cat, 0), 3) for cat in config.all_categories}
                    best_swap = {
                        "bench_player": bp_name,
                        "start_player": sp_name,
                        "total_sgp_delta": round(total_delta, 3),
                        "category_impacts": impacts,
                    }
            if best_swap and best_swap["total_sgp_delta"] > 0.1:
                swaps.append(best_swap)

        swaps.sort(key=lambda x: x["total_sgp_delta"], reverse=True)
        return swaps[:5]
    except Exception:
        return []


def _get_il_return_dates(roster: pd.DataFrame) -> dict:
    """Fetch ESPN injury return dates for rostered players.

    Returns:
        dict mapping player_id (int) AND player name (str) to datetime.
    """
    result: dict = {}
    try:
        from src.espn_injuries import fetch_espn_injuries

        injuries = fetch_espn_injuries()
        if not injuries:
            return result

        roster_names = set()
        if not roster.empty and "name" in roster.columns:
            roster_names = {str(n).strip().lower() for n in roster["name"].dropna()}

        for inj in injuries:
            raw_date = inj.get("return_date", "")
            if not raw_date:
                continue
            pname = inj.get("player_name", "")
            if pname.strip().lower() not in roster_names:
                continue
            try:
                # ESPN returns ISO-8601 or "YYYY-MM-DDTHH:MM:SSZ"
                dt = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                continue

            result[pname] = dt
            # Also map by player_id if we can resolve it
            try:
                from src.live_stats import match_player_id

                pid = match_player_id(pname, inj.get("team", ""))
                if pid is not None:
                    result[pid] = dt
            except Exception:
                pass
    except Exception:
        logger.debug("Could not fetch ESPN injury return dates", exc_info=True)

    return result


def generate_regression_alerts(
    roster: pd.DataFrame,
    min_pa: int = 50,
) -> list[dict]:
    """Generate sell-high / buy-low regression alerts from expected stats.

    Compares actual performance to Statcast expected stats (xwOBA, xBA).
    Flags players with >1.5 SD divergence.

    Args:
        roster: DataFrame with player rows containing obp, xwoba, ytd_pa columns.
        min_pa: Minimum plate appearances to qualify.

    Returns:
        List of alert dicts sorted by divergence magnitude (descending).
    """
    alerts: list[dict] = []
    for _, row in roster.iterrows():
        pa = int(row.get("ytd_pa", 0) or 0)
        if pa < min_pa:
            continue

        # Check xwOBA vs approximate wOBA (OBP * 1.15 proxy)
        xwoba = float(row.get("xwoba", 0) or 0)
        obp = float(row.get("obp", 0) or 0)
        if xwoba > 0 and obp > 0:
            woba_approx = obp * 1.15
            delta = xwoba - woba_approx
            sd = 0.020  # Approximate SD for xwOBA-wOBA gap
            if abs(delta) > 1.5 * sd:
                name = str(row.get("name", row.get("player_name", "Unknown")))
                if delta > 0:
                    alerts.append(
                        {
                            "player_name": name,
                            "alert_type": "BUY_LOW",
                            "stat": "xwOBA",
                            "actual": round(woba_approx, 3),
                            "expected": round(xwoba, 3),
                            "divergence_sd": round(delta / sd, 1),
                            "message": (
                                f"{name}: xwOBA ({xwoba:.3f}) >> actual wOBA "
                                f"({woba_approx:.3f}) — underperforming contact quality"
                            ),
                        }
                    )
                else:
                    alerts.append(
                        {
                            "player_name": name,
                            "alert_type": "SELL_HIGH",
                            "stat": "xwOBA",
                            "actual": round(woba_approx, 3),
                            "expected": round(xwoba, 3),
                            "divergence_sd": round(abs(delta) / sd, 1),
                            "message": (
                                f"{name}: actual wOBA ({woba_approx:.3f}) >> xwOBA "
                                f"({xwoba:.3f}) — overperforming contact quality"
                            ),
                        }
                    )

    alerts.sort(key=lambda x: abs(x.get("divergence_sd", 0)), reverse=True)
    return alerts


def generate_ratio_lock_alert(
    current_era: float,
    opp_era: float,
    current_whip: float,
    opp_whip: float,
    banked_ip: float,
    remaining_starts: int = 0,
) -> dict | None:
    """Generate ratio lock alert when you should bench pitchers.

    When you have a comfortable lead in ERA and WHIP with enough innings
    banked, benching remaining starters locks 2 category wins.

    Args:
        current_era: Your team's current ERA.
        opp_era: Opponent's current ERA.
        current_whip: Your team's current WHIP.
        opp_whip: Opponent's current WHIP.
        banked_ip: Total innings pitched so far this matchup week.
        remaining_starts: Number of scheduled starts remaining.

    Returns:
        Alert dict or None if no lock opportunity.
    """
    era_lead = opp_era - current_era  # Positive = you're winning
    whip_lead = opp_whip - current_whip  # Positive = you're winning

    if era_lead > 0.50 and whip_lead > 0.10 and banked_ip >= 30:
        return {
            "type": "ratio_lock",
            "severity": "info",
            "title": "Ratio Lock Opportunity",
            "message": (
                f"Winning ERA by {era_lead:.2f} and WHIP by {whip_lead:.2f} "
                f"with {banked_ip:.0f} IP banked. Consider benching remaining "
                f"{remaining_starts} starts to lock 2 category wins."
            ),
            "action": f"Bench remaining {remaining_starts} scheduled starters.",
            "era_lead": round(era_lead, 2),
            "whip_lead": round(whip_lead, 2),
            "timestamp": datetime.now(UTC).isoformat(),
        }
    return None


def render_alerts_html(alerts: list[dict], theme: dict) -> str:
    """Render alerts as HTML cards for Streamlit st.markdown().

    Args:
        alerts: List of alert dicts from generate_roster_alerts.
        theme: THEME dict from ui_shared.

    Returns:
        HTML string ready for st.markdown(unsafe_allow_html=True).
    """
    if not alerts:
        return ""

    # Sort alerts by timestamp descending (newest first)
    alerts = sorted(alerts, key=lambda a: a.get("timestamp", ""), reverse=True)

    severity_colors = {
        "critical": theme.get("danger", "#e63946"),
        "warning": theme.get("warn", "#ff9f1c"),
        "info": theme.get("sky", "#457b9d"),
    }

    cards = []
    for alert in alerts:
        color = severity_colors.get(alert["severity"], theme.get("tx2", "#6b7280"))

        # Format timestamp for display in ET
        ts_html = ""
        raw_ts = alert.get("timestamp", "")
        if raw_ts:
            try:
                dt_utc = datetime.fromisoformat(raw_ts)
                ts_html = (
                    f'<span style="float:right;font-size:10px;color:{theme.get("tx2", "#6b7280")};">'
                    f"{_fmt_et(dt_utc)}</span>"
                )
            except (ValueError, TypeError):
                pass

        cards.append(
            f'<div style="background:{theme.get("card", "#fff")};'
            f"border-left:4px solid {color};"
            f"padding:8px 12px;border-radius:6px;margin-bottom:6px;font-size:12px;"
            f'font-family:IBM Plex Mono,monospace;">'
            f'<b style="color:{color};">{alert["title"]}</b>{ts_html}<br>'
            f'<span style="color:{theme.get("tx2", "#6b7280")};">{alert["message"]}</span><br>'
            f'<span style="color:{theme.get("tx", "#1d1d1f")};font-weight:600;">'
            f"Action: {alert['action']}</span>"
            f"</div>"
        )

    return "\n".join(cards)

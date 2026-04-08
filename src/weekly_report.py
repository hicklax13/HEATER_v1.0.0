"""Weekly Report Generator — AVIS Section 5 operating cadence.

Produces Monday matchup reports, Thursday mid-week checkpoints,
and daily lineup checks.
"""

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

logger = logging.getLogger(__name__)


def generate_monday_report(
    user_roster: pd.DataFrame,
    opponent_roster: pd.DataFrame | None,
    opponent_profile: dict,
    week: int,
    config=None,
) -> dict:
    """Generate the Monday matchup report per AVIS Section 2.1.

    Returns dict with sections: opponent_summary, category_projections,
    exploit_weaknesses, protect_floor, streaming_targets.
    """
    from src.valuation import LeagueConfig, SGPCalculator

    if config is None:
        config = LeagueConfig()
    sgp_calc = SGPCalculator(config)

    report = {
        "week": week,
        "opponent": opponent_profile.get("name", "Unknown"),
        "tier": opponent_profile.get("tier", 3),
        "threat": opponent_profile.get("threat", "Unknown"),
        "manager": opponent_profile.get("manager", "Unknown"),
        "notes": opponent_profile.get("notes", ""),
        "sections": [],
    }

    # Section 1: Opponent summary
    opp_strengths = opponent_profile.get("strengths", [])
    opp_weaknesses = opponent_profile.get("weaknesses", [])
    report["opponent_strengths"] = opp_strengths
    report["opponent_weaknesses"] = opp_weaknesses

    # Section 2: Category-by-category projection
    cat_projections = []
    if not user_roster.empty:
        all_cats = config.all_categories
        for cat in all_cats:
            col = cat.lower()
            user_val = 0
            if col in user_roster.columns:
                user_val = pd.to_numeric(user_roster[col], errors="coerce").fillna(0).sum()

            # Determine expected outcome
            if cat in opp_weaknesses:
                outlook = "LIKELY WIN"
            elif cat in opp_strengths:
                outlook = "LIKELY LOSS"
            else:
                outlook = "TOSS-UP"

            cat_projections.append(
                {
                    "category": cat,
                    "team_hickey_total": round(float(user_val), 1),
                    "outlook": outlook,
                }
            )

    report["category_projections"] = cat_projections

    # Section 3: Exploit opponent weaknesses
    exploit = []
    for cat in opp_weaknesses:
        exploit.append(f"Target {cat} — opponent is structurally weak here. Stream/start players who contribute {cat}.")
    report["exploit_weaknesses"] = exploit

    # Section 4: Protect your floor
    # Team Hickey structural weaknesses from AVIS Section 2.2
    hickey_weak = ["SB"]  # Per AVIS manual
    protect = []
    for cat in hickey_weak:
        if cat not in opp_weaknesses:  # Only protect if opponent isn't also weak
            protect.append(f"Shore up {cat} — your structural weakness. Consider streaming speed if SB is a toss-up.")
    report["protect_floor"] = protect

    # Section 5: Streaming targets (top-level guidance)
    streaming = []
    if "K" in opp_weaknesses or "W" in opp_weaknesses:
        streaming.append("Stream SP for K/W — opponent is weak in pitching counting stats.")
    if "SV" in opp_weaknesses:
        streaming.append("Add RP closers — opponent may be punting saves.")
    if "SB" in opp_weaknesses:
        streaming.append("Bench speed — no need to stream SB specialists this week.")
    if not streaming:
        streaming.append("No clear streaming edge this week. Stick with your starters.")
    report["streaming_guidance"] = streaming

    return report


def generate_thursday_checkpoint(
    user_roster: pd.DataFrame,
    matchup_score: dict | None = None,
    ip_projected: float = 0.0,
) -> dict:
    """Thursday mid-week checkpoint per AVIS Section 5.

    Args:
        user_roster: Current roster DataFrame.
        matchup_score: Dict with category-by-category current standings
            (from Yahoo scoreboard). None if not available.
        ip_projected: Projected IP for the rest of the week.

    Returns:
        Dict with: categories_at_risk, ip_status, recommendations.
    """
    checkpoint = {
        "day": "Thursday",
        "categories_at_risk": [],
        "ip_status": "",
        "recommendations": [],
    }

    # IP check
    MIN_IP = 20.0
    if ip_projected < MIN_IP:
        checkpoint["ip_status"] = f"DANGER: {ip_projected:.2f} IP projected, need {MIN_IP:.0f}. Add streaming SP NOW."
        checkpoint["recommendations"].append("Stream a SP with a start this weekend to hit 20 IP minimum.")
    else:
        checkpoint["ip_status"] = f"On pace: {ip_projected:.2f} IP projected (minimum {MIN_IP:.0f})."

    # Matchup score analysis (if available from Yahoo)
    if matchup_score:
        winning = []
        losing = []
        close = []
        for cat, data in matchup_score.items():
            margin = data.get("margin", 0)
            if margin > 0.5:
                winning.append(cat)
            elif margin < -0.5:
                losing.append(cat)
            else:
                close.append(cat)

        checkpoint["categories_at_risk"] = close
        if close:
            checkpoint["recommendations"].append(
                f"Toss-up categories: {', '.join(close)}. Target streaming/lineup moves here."
            )

        # ERA/WHIP risk management
        if "ERA" in winning and "WHIP" in winning:
            checkpoint["recommendations"].append(
                "Winning ERA + WHIP — consider benching risky SP starts to protect ratios."
            )
        if "K" in losing or "W" in losing:
            checkpoint["recommendations"].append("Losing K or W — add streaming SP for counting stats.")

    return checkpoint


def check_daily_lineup(
    roster: pd.DataFrame,
    todays_games: list[str] | None = None,
) -> list[dict]:
    """Daily lineup check — flag off-day starters and benchable players.

    Args:
        roster: Roster DataFrame with columns: name, positions, team, roster_slot.
        todays_games: List of team abbreviations playing today.
            If None, assumes all teams play.

    Returns:
        List of alert dicts: {player, issue, recommendation}.
    """
    alerts = []

    if todays_games is None:
        return alerts  # Can't check without schedule

    playing_teams = set(t.upper() for t in todays_games)

    for _, player in roster.iterrows():
        name = player.get("name", "?")
        team = str(player.get("team", "")).upper()
        slot = str(player.get("roster_slot", "")).upper()
        is_bench = "BN" in slot or "IL" in slot

        if not team:
            continue

        if team not in playing_teams and not is_bench:
            # Player is in a starting slot but their team isn't playing
            alerts.append(
                {
                    "player": name,
                    "issue": f"OFF-DAY — {team} not playing today but in starting lineup slot",
                    "recommendation": f"Move {name} to bench and start an active player.",
                    "severity": "warning",
                }
            )

        if team in playing_teams and is_bench and "IL" not in slot:
            # Player is on bench but their team IS playing
            alerts.append(
                {
                    "player": name,
                    "issue": f"BENCHED — {team} playing today but {name} is on the bench",
                    "recommendation": f"Consider starting {name} if a starter is on an off-day.",
                    "severity": "info",
                }
            )

    return alerts


def _position_eligible(player_positions: str, slot: str) -> bool:
    """Check if a player is eligible for a given roster slot.

    Args:
        player_positions: Comma-separated position string (e.g. "1B,OF,DH").
        slot: Roster slot to check eligibility for (e.g. "1B", "OF", "Util").
    """
    slot_upper = slot.strip().upper()
    positions = {p.strip().upper() for p in str(player_positions).split(",") if p.strip()}

    # Util slots accept any player
    if slot_upper in ("UTIL", "UT"):
        return True

    # P slots accept SP and RP
    if slot_upper == "P":
        return bool(positions & {"SP", "RP", "P"})

    # OF slots accept LF, CF, RF, OF
    if slot_upper == "OF":
        return bool(positions & {"LF", "CF", "RF", "OF"})

    # Direct position match
    return slot_upper in positions


def _find_replacements(
    roster: pd.DataFrame,
    off_day_player: pd.Series,
    playing_teams: set[str],
    max_suggestions: int = 2,
) -> list[str]:
    """Find bench players who can replace a starting player on an off-day.

    Returns a list of player names (up to max_suggestions) who are:
    - On the bench (BN slot)
    - Playing today (team is in playing_teams)
    - Position-eligible for the starter's slot
    """
    starter_slot = str(off_day_player.get("roster_slot", "")).strip()
    starter_is_hitter = off_day_player.get("is_hitter", 1)
    replacements = []

    for _, bench_player in roster.iterrows():
        bench_slot = str(bench_player.get("roster_slot", "")).upper()
        if "BN" not in bench_slot:
            continue
        bench_team = str(bench_player.get("team", "")).upper()
        if bench_team not in playing_teams:
            continue
        # Check type match (hitter for hitter, pitcher for pitcher)
        bench_is_hitter = bench_player.get("is_hitter", 1)
        if bench_is_hitter != starter_is_hitter:
            continue
        # Check position eligibility
        bench_positions = str(bench_player.get("positions", ""))
        if _position_eligible(bench_positions, starter_slot):
            replacements.append(str(bench_player.get("name", "?")))
            if len(replacements) >= max_suggestions:
                break

    return replacements


def validate_daily_lineup(
    roster: pd.DataFrame,
    todays_games: list[str] | None = None,
) -> list[dict]:
    """Enhanced daily lineup validation with replacement suggestions.

    Wraps check_daily_lineup() and enriches each alert with position-eligible
    bench replacements who are playing today.

    Args:
        roster: Roster DataFrame with columns: name, positions, team,
            roster_slot, is_hitter.
        todays_games: List of team abbreviations playing today.

    Returns:
        List of dicts: {player, issue, severity, replacements: [name, ...]}.
    """
    base_alerts = check_daily_lineup(roster, todays_games)
    if not base_alerts or todays_games is None:
        return base_alerts

    playing_teams = {t.upper() for t in todays_games}
    enriched = []

    for alert in base_alerts:
        player_name = alert["player"]
        # Find the matching roster row for this player
        match = roster[roster["name"] == player_name]
        replacements = []
        if not match.empty and alert["severity"] == "warning":
            player_row = match.iloc[0]
            replacements = _find_replacements(roster, player_row, playing_teams)

        enriched.append(
            {
                "player": alert["player"],
                "issue": alert["issue"],
                "severity": alert["severity"],
                "recommendation": alert.get("recommendation", ""),
                "replacements": replacements,
            }
        )

    return enriched


def get_todays_mlb_games() -> list[str]:
    """Fetch today's MLB schedule and return list of team abbreviations playing."""
    try:
        import statsapi

        # MLB schedule dates are in US Eastern time, not UTC
        _ET = timezone(timedelta(hours=-4))  # EDT (summer)
        today = datetime.now(_ET).strftime("%Y-%m-%d")
        schedule = statsapi.schedule(start_date=today, end_date=today)
        teams_playing = set()
        for game in schedule:
            away = game.get("away_name", "")
            home = game.get("home_name", "")
            # statsapi returns full names, we need abbreviations
            # Use the teams endpoint for mapping
            away_id = game.get("away_id")
            home_id = game.get("home_id")
            if away_id:
                teams_playing.add(str(away_id))
            if home_id:
                teams_playing.add(str(home_id))

        # Convert team IDs to abbreviations using the cached team map
        from src.live_stats import _build_team_id_map

        team_map = _build_team_id_map(2026)
        abbrs = []
        for tid_str in teams_playing:
            try:
                abbr = team_map.get(int(tid_str), "")
                if abbr:
                    abbrs.append(abbr)
            except (ValueError, TypeError):
                pass

        return abbrs
    except Exception:
        logger.debug("Could not fetch today's MLB schedule", exc_info=True)
        return []

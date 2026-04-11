"""Opponent Intelligence — profiles, schedule, and matchup context.

Provides opponent threat levels, category strengths/weaknesses,
and the full 24-week schedule.
"""

import logging
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

# ── Schedule: Team Hickey's opponents by week ────────────────────────

TEAM_HICKEY_SCHEDULE = {
    1: "The Good The Vlad The Ugly",
    2: "Baty Babies",
    3: "Jonny Jockstrap",
    4: "Over the Rembow",
    5: "Going…Going…Gonorrhea",
    6: "On a Twosday",
    7: "Twigs",
    8: "Cyrus The Greats",
    9: "Go yanks",
    10: "BUBBA CROSBY",
    11: "HUMAN INTELLIGENCE",
    12: "The Good The Vlad The Ugly",
    13: "Baty Babies",
    14: "Jonny Jockstrap",
    15: "Over the Rembow",
    16: "Going…Going…Gonorrhea",
    17: "On a Twosday",
    18: "Twigs",
    19: "Cyrus The Greats",
    20: "Go yanks",
    21: "BUBBA CROSBY",
    22: "HUMAN INTELLIGENCE",
    23: "The Good The Vlad The Ugly",
    24: "Baty Babies",
}

# ── Opponent Profiles ────────────────────────────────────────────────

OPPONENT_PROFILES = {
    "Over the Rembow": {
        "tier": 1,
        "threat": "High",
        "manager": "Alex",
        "strengths": ["SB", "K", "ERA"],
        "weaknesses": ["RBI"],
        "notes": "Skenes + Ohtani(P) best pitching ceiling. Jazz = elite SB. Most active manager.",
    },
    "Twigs": {
        "tier": 1,
        "threat": "High",
        "manager": "Nick",
        "strengths": ["HR", "R", "SB", "K"],
        "weaknesses": ["SV"],
        "notes": "Judge + Duran + Cruz = best OF. Yamamoto, Cease, Mason Miller. Deep and balanced.",
    },
    "BUBBA CROSBY": {
        "tier": 1,
        "threat": "High",
        "manager": "Elias",
        "strengths": ["HR", "RBI", "AVG", "SB"],
        "weaknesses": ["ERA", "WHIP"],
        "notes": "Acuña, Harper, Freeman, Caminero = elite bats. IL-heavy but high upside.",
    },
    "Cyrus The Greats": {
        "tier": 2,
        "threat": "Medium-High",
        "manager": "Montezuma",
        "strengths": ["HR", "RBI", "SB", "R"],
        "weaknesses": ["SV", "W"],
        "notes": "J-Ram + Gunnar + Trea Turner absurd infield. Empty C slot = sloppy management.",
    },
    "Baty Babies": {
        "tier": 2,
        "threat": "Medium",
        "manager": "Dandre",
        "strengths": ["HR", "AVG", "SB"],
        "weaknesses": ["W", "K", "ERA"],
        "notes": "Ohtani(B), Devers, Chourio, Carroll = elite bats. Pitching disaster — 4 empty SP slots.",
    },
    "Jonny Jockstrap": {
        "tier": 2,
        "threat": "Medium-High",
        "manager": "Jon",
        "strengths": ["HR", "SB", "K"],
        "weaknesses": ["AVG", "OBP"],
        "notes": "Tucker, J-Rod, James Wood = elite OF. deGrom + Ragans + Ryan + Bradish strong.",
    },
    "Going…Going…Gonorrhea": {
        "tier": 3,
        "threat": "Medium",
        "manager": "Sam",
        "strengths": ["K", "ERA"],
        "weaknesses": ["HR", "RBI", "SV"],
        "notes": "Skubal is best single pitcher. Rest is mid. Lives and dies by Skubal starts.",
    },
    "Go yanks": {
        "tier": 3,
        "threat": "Medium",
        "manager": "Ben",
        "strengths": ["SB", "W", "K"],
        "weaknesses": ["SV"],
        "notes": "Witt Jr. elite. SP-heavy (Woo, Gilbert, Webb, Woodruff). Zero closers — punting SV.",
    },
    "The Good The Vlad The Ugly": {
        "tier": 3,
        "threat": "Medium-Low",
        "manager": "Ricky",
        "strengths": ["HR", "RBI"],
        "weaknesses": ["SB", "K", "ERA"],
        "notes": "Vlad Jr. and Alonso on bench is bizarre. Pitching thin (Fried healthy, rest IL).",
    },
    "On a Twosday": {
        "tier": 3,
        "threat": "Medium-Low",
        "manager": "Matt",
        "strengths": ["SB", "HR"],
        "weaknesses": ["W", "K", "ERA", "WHIP"],
        "notes": "Elly, Tatis, Betts = stars & scrubs. Empty OF slot, thin pitching, no ace.",
    },
    "HUMAN INTELLIGENCE": {
        "tier": 4,
        "threat": "Low",
        "manager": "Spread Love",
        "strengths": ["OBP"],
        "weaknesses": ["HR", "RBI", "K", "W"],
        "notes": "Defending champ but regressed. Soto elite; everything else replacement-level.",
    },
}

# MLB season start date for week calculation
_SEASON_START = datetime(2026, 3, 23, tzinfo=UTC)  # Week 1 starts March 23


def get_week_number() -> int:
    """Compute the current fantasy week number (1-based)."""
    now = datetime.now(UTC)
    delta = now - _SEASON_START
    week = max(1, (delta.days // 7) + 1)
    return min(week, 26)  # Cap at playoff end


def get_current_opponent(yds=None) -> dict:
    """Get this week's opponent profile.

    When a YahooDataService is provided, uses live data (schedule from
    Yahoo matchups, profile from live standings). Falls back to
    hardcoded data when live data is unavailable.

    Args:
        yds: Optional YahooDataService instance. If None, uses fallback data only.

    Returns:
        Dict with: name, tier, threat, manager, strengths, weaknesses, notes, week.
        Returns empty dict if schedule data not available.
    """
    week = get_week_number()

    # Try live schedule from Yahoo first
    opponent_name = None
    if yds is not None:
        try:
            schedule = yds.get_schedule()
            if schedule:
                opponent_name = schedule.get(week)
        except Exception:
            logger.debug("Live schedule unavailable, falling back to hardcoded data")

    # Fall back to hardcoded schedule
    if not opponent_name:
        opponent_name = TEAM_HICKEY_SCHEDULE.get(week)
    if not opponent_name:
        return {}

    # Try live profile from Yahoo standings
    if yds is not None:
        try:
            live_profile = yds.get_opponent_profile(opponent_name)
            if live_profile and live_profile.get("threat") != "Unknown":
                return {
                    "name": opponent_name,
                    "week": week,
                    **live_profile,
                }
        except Exception:
            logger.debug("Live profile unavailable for %s, falling back to hardcoded data", opponent_name)

    # Fall back to hardcoded profiles
    profile = OPPONENT_PROFILES.get(opponent_name, {})
    return {
        "name": opponent_name,
        "week": week,
        "tier": profile.get("tier", 3),
        "threat": profile.get("threat", "Unknown"),
        "manager": profile.get("manager", "Unknown"),
        "strengths": profile.get("strengths", []),
        "weaknesses": profile.get("weaknesses", []),
        "notes": profile.get("notes", ""),
    }


def get_opponent_for_week(week: int, yds=None) -> dict:
    """Get opponent profile for a specific week.

    Args:
        week: Fantasy week number (1-24).
        yds: Optional YahooDataService instance for live data.
    """
    # Try live schedule first
    opponent_name = None
    if yds is not None:
        try:
            schedule = yds.get_schedule()
            if schedule:
                opponent_name = schedule.get(week)
        except Exception:
            pass

    if not opponent_name:
        opponent_name = TEAM_HICKEY_SCHEDULE.get(week)
    if not opponent_name:
        return {}

    # Try live profile
    if yds is not None:
        try:
            live_profile = yds.get_opponent_profile(opponent_name)
            if live_profile and live_profile.get("threat") != "Unknown":
                return {"name": opponent_name, "week": week, **live_profile}
        except Exception:
            pass

    profile = OPPONENT_PROFILES.get(opponent_name, {})
    return {
        "name": opponent_name,
        "week": week,
        **profile,
    }


def get_schedule_difficulty(weeks: range | None = None) -> list[dict]:
    """Get schedule difficulty overview.

    Returns list of dicts with week, opponent, tier, threat for each week.
    """
    if weeks is None:
        weeks = range(1, 25)
    result = []
    for w in weeks:
        opp = TEAM_HICKEY_SCHEDULE.get(w, "BYE")
        profile = OPPONENT_PROFILES.get(opp, {})
        result.append(
            {
                "week": w,
                "opponent": opp,
                "tier": profile.get("tier", 0),
                "threat": profile.get("threat", ""),
            }
        )
    return result


def analyze_weekly_matchup(
    user_roster,
    opponent_profile: dict,
    week: int,
    config=None,
) -> dict:
    """Category-by-category matchup analysis with wins/losses/toss-ups.

    Combines Team Hickey's roster projections with the opponent's known
    strengths/weaknesses to project category-level outcomes.

    Args:
        user_roster: DataFrame with Team Hickey's projected stats.
        opponent_profile: Dict from get_current_opponent().
        week: Fantasy week number.
        config: LeagueConfig (defaults to standard).

    Returns:
        Dict with: week, opponent, tier, category_results (list of per-cat dicts),
        likely_wins, likely_losses, toss_ups, exploit_targets, vulnerabilities,
        streaming_recommendations.
    """
    import pandas as pd

    from src.valuation import LeagueConfig

    if config is None:
        config = LeagueConfig()

    opp_strengths = set(opponent_profile.get("strengths", []))
    opp_weaknesses = set(opponent_profile.get("weaknesses", []))

    category_results = []
    likely_wins = []
    likely_losses = []
    toss_ups = []

    for cat in config.all_categories:
        col = cat.lower()
        team_total = 0.0
        if not user_roster.empty and col in user_roster.columns:
            team_total = float(pd.to_numeric(user_roster[col], errors="coerce").fillna(0).sum())

        # Determine outlook based on opponent profile
        if cat in opp_weaknesses:
            outlook = "LIKELY WIN"
            likely_wins.append(cat)
        elif cat in opp_strengths:
            outlook = "LIKELY LOSS"
            likely_losses.append(cat)
        else:
            outlook = "TOSS-UP"
            toss_ups.append(cat)

        category_results.append(
            {
                "category": cat,
                "team_total": round(team_total, 2),
                "outlook": outlook,
                "opp_strong": cat in opp_strengths,
                "opp_weak": cat in opp_weaknesses,
            }
        )

    # Exploitation targets: opponent weaknesses that align with our strengths
    # Team Hickey structural strengths: HR, RBI, K
    hickey_strengths = {"HR", "RBI", "K"}
    exploit_targets = [cat for cat in opp_weaknesses if cat in hickey_strengths or cat in toss_ups]

    # Vulnerabilities: opponent strengths that overlap with our weaknesses
    hickey_weaknesses = {"SB"}  # Structural weakness based on draft profile
    vulnerabilities = [cat for cat in opp_strengths if cat in hickey_weaknesses or cat in toss_ups]

    # Streaming recommendations based on matchup
    streaming_recs = []
    if opp_weaknesses & {"K", "W"}:
        streaming_recs.append("Stream SP for K/W — opponent is weak in pitching counting stats.")
    if "SV" in opp_weaknesses:
        streaming_recs.append("Add RP closers — exploit opponent's saves weakness.")
    if "SB" in toss_ups:
        streaming_recs.append("Consider speed hitters — SB is a toss-up this week.")
    if opp_strengths & {"ERA", "WHIP"} and {"K", "W"} & set(toss_ups):
        streaming_recs.append("Stream volume SP (high K upside) — accept rate-stat risk for counting gains.")
    if not streaming_recs:
        streaming_recs.append("No clear streaming edge — play your best lineup.")

    return {
        "week": week,
        "opponent": opponent_profile.get("name", "Unknown"),
        "tier": opponent_profile.get("tier", 3),
        "threat": opponent_profile.get("threat", "Unknown"),
        "category_results": category_results,
        "likely_wins": likely_wins,
        "likely_losses": likely_losses,
        "toss_ups": toss_ups,
        "exploit_targets": exploit_targets,
        "vulnerabilities": vulnerabilities,
        "streaming_recommendations": streaming_recs,
        "projected_record": f"{len(likely_wins)}-{len(likely_losses)}-{len(toss_ups)}",
    }

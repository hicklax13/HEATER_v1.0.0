"""Opponent Intelligence — profiles, schedule, and matchup context.

Populated from the AVIS Operations Manual. Provides opponent threat levels,
category strengths/weaknesses, and the full 24-week schedule.
"""

import logging
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

# ── Schedule: Team Hickey's opponents by week (AVIS Section 4) ────────

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

# ── Opponent Profiles (AVIS Section 3) ────────────────────────────────

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


def get_current_opponent() -> dict:
    """Get this week's opponent profile.

    Returns:
        Dict with: name, tier, threat, manager, strengths, weaknesses, notes, week.
        Returns empty dict if schedule data not available.
    """
    week = get_week_number()
    opponent_name = TEAM_HICKEY_SCHEDULE.get(week)
    if not opponent_name:
        return {}

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


def get_opponent_for_week(week: int) -> dict:
    """Get opponent profile for a specific week."""
    opponent_name = TEAM_HICKEY_SCHEDULE.get(week)
    if not opponent_name:
        return {}
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

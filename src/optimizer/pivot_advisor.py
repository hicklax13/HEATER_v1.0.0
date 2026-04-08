"""Mid-Week Pivot Advisor — category flip probability for H2H matchups.

Identifies WON/LOST/CONTESTED categories mid-week so the manager can
redirect resources from lost causes to contested categories.  Research
shows identifying 2-3 lost causes by Wednesday and pivoting swings
+1-2 category wins per week (24+ over a season).

Uses Normal CDF on the remaining-games-adjusted standard deviation to
compute the probability that a category result flips before week's end.
"""

from __future__ import annotations

import logging
import math

from scipy.stats import norm

logger = logging.getLogger(__name__)

# Weekly category standard deviations in RAW stat units, calibrated from
# FanGraphs research (48 twelve-team leagues).  Counting stats are weekly
# totals; rate stats are weekly averages.  Scaled by sqrt(games_remaining/7).
WEEKLY_SD: dict[str, float] = {
    # Counting stats — typical weekly SD between two H2H teams
    "R": 8.5,
    "HR": 3.5,
    "RBI": 9.0,
    "SB": 2.5,
    "W": 2.0,
    "L": 2.0,
    "SV": 2.5,
    "K": 15.0,
    # Rate stats — weekly SD in raw stat units
    "AVG": 0.025,
    "OBP": 0.020,
    "ERA": 1.20,
    "WHIP": 0.18,
}

# Thresholds for classification
_WON_THRESHOLD = 0.15  # flip_prob below this -> WON
_LOST_THRESHOLD = 0.15  # catch-up prob below this -> LOST


def compute_category_flip_probabilities(
    my_totals: dict[str, float],
    opp_totals: dict[str, float],
    games_remaining: int,
    config=None,
) -> dict[str, dict]:
    """Per-category flip probability and WON/LOST/CONTESTED classification.

    For each scoring category:
    1. Compute margin (my - opp), flipped for inverse stats (L/ERA/WHIP).
    2. Scale the weekly SD by sqrt(games_remaining / 7).
    3. Compute flip probability via Normal CDF.
    4. Classify as WON / LOST / CONTESTED.
    5. Assign a recommended action string.

    Args:
        my_totals: My team's category totals so far this matchup week.
        opp_totals: Opponent's category totals so far this matchup week.
        games_remaining: Games left in the matchup week (0-7).
        config: LeagueConfig instance (default created if None).

    Returns:
        Dict keyed by category name, each value a dict with keys:
        margin, flip_prob, classification, action.
    """
    if config is None:
        from src.valuation import LeagueConfig

        config = LeagueConfig()

    inverse_stats = config.inverse_stats  # {"L", "ERA", "WHIP"}
    rate_stats = getattr(config, "rate_stats", {"AVG", "OBP", "ERA", "WHIP"})
    results: dict[str, dict] = {}

    for cat in config.all_categories:
        my_val = float(my_totals.get(cat, my_totals.get(cat.lower(), 0)) or 0)
        opp_val = float(opp_totals.get(cat, opp_totals.get(cat.lower(), 0)) or 0)

        # For inverse stats, lower is better — flip the sign so positive
        # margin means "I am winning".
        if cat in inverse_stats:
            margin = opp_val - my_val
        else:
            margin = my_val - opp_val

        sd_base = WEEKLY_SD.get(cat, 2.0)

        # --- Edge case: no games remaining ---
        if games_remaining <= 0:
            # Result is locked in
            if margin > 0:
                flip_prob = 0.0
                classification = "WON"
            elif margin < 0:
                flip_prob = 0.0
                classification = "LOST"
            else:
                flip_prob = 0.5
                classification = "CONTESTED"
        else:
            remaining_sd = sd_base * math.sqrt(games_remaining / 7.0)

            if remaining_sd <= 0:
                # Degenerate — treat like 0 games remaining
                if margin > 0:
                    flip_prob = 0.0
                    classification = "WON"
                elif margin < 0:
                    flip_prob = 0.0
                    classification = "LOST"
                else:
                    flip_prob = 0.5
                    classification = "CONTESTED"
            elif margin > 0:
                # Currently winning — probability of losing the lead
                flip_prob = float(norm.cdf(-margin / remaining_sd))
                if flip_prob < _WON_THRESHOLD:
                    classification = "WON"
                else:
                    classification = "CONTESTED"
            elif margin < 0:
                # Currently losing — probability of catching up
                # margin is negative, so -margin/sd gives positive z
                flip_prob = float(norm.cdf(margin / remaining_sd))
                if flip_prob < _LOST_THRESHOLD:
                    classification = "LOST"
                else:
                    classification = "CONTESTED"
            else:
                # Tied
                flip_prob = 0.5
                classification = "CONTESTED"

        # Determine action
        if classification == "WON":
            if cat in rate_stats:
                action = "Protect -- consider benching risky pitchers"
            else:
                action = "Maintain -- keep current lineup"
        elif classification == "LOST":
            action = "Concede -- redirect resources to contested categories"
        else:
            action = "Pursue -- prioritize this category in lineup decisions"

        results[cat] = {
            "margin": round(margin, 4),
            "flip_prob": round(flip_prob, 4),
            "classification": classification,
            "action": action,
        }

    return results


def get_pivot_summary(
    my_totals: dict[str, float],
    opp_totals: dict[str, float],
    games_remaining: int,
    config=None,
) -> dict:
    """High-level pivot summary grouping categories by classification.

    Returns:
        Dict with keys: won, lost, contested (each a list of category
        names), and recommended_actions (list of action strings for
        contested and lost categories).
    """
    probs = compute_category_flip_probabilities(my_totals, opp_totals, games_remaining, config)

    won: list[str] = []
    lost: list[str] = []
    contested: list[str] = []
    actions: list[str] = []

    for cat, info in probs.items():
        cls = info["classification"]
        if cls == "WON":
            won.append(cat)
        elif cls == "LOST":
            lost.append(cat)
        else:
            contested.append(cat)

        # Surface actions for non-maintenance categories
        if cls in ("LOST", "CONTESTED"):
            actions.append(f"{cat}: {info['action']}")

    return {
        "won": won,
        "lost": lost,
        "contested": contested,
        "recommended_actions": actions,
    }


# ── Ratio Protection Calculator ─────────────────────────────────────


def compute_ratio_protection(
    current_era: float,
    current_whip: float,
    banked_ip: float,
    pitcher_proj_era: float,
    pitcher_proj_whip: float = 1.30,
    pitcher_proj_ip: float = 6.0,
    era_lead: float = 0.0,
    whip_lead: float = 0.0,
) -> dict:
    """Compute marginal ERA/WHIP risk from one additional pitcher start.

    When you are winning ERA/WHIP, each start risks destroying the lead.
    This function quantifies that risk so the manager can decide whether
    to bench a pitcher to protect rate-stat leads.

    The math:
        new_era = (current_era * banked_ip + pitcher_era * pitcher_ip)
                  / (banked_ip + pitcher_ip)
        era_risk = new_era - current_era  (positive = ERA gets worse)

    Args:
        current_era: Team's current ERA for the matchup week.
        current_whip: Team's current WHIP for the matchup week.
        banked_ip: Innings pitched so far this matchup week.
        pitcher_proj_era: Projected ERA for the pitcher being considered.
        pitcher_proj_whip: Projected WHIP for the pitcher being considered.
        pitcher_proj_ip: Expected innings for this start (default 6.0).
        era_lead: Current ERA lead over opponent (positive = winning).
        whip_lead: Current WHIP lead over opponent (positive = winning).

    Returns:
        Dict with keys: era_risk, whip_risk, recommend, era_after, whip_after.
        recommend is one of "START", "BENCH", or "RISKY".
    """
    total_ip = banked_ip + pitcher_proj_ip
    if total_ip <= 0:
        return {
            "era_risk": 0.0,
            "whip_risk": 0.0,
            "recommend": "START",
            "era_after": current_era,
            "whip_after": current_whip,
        }

    # Blend current rate stats with pitcher's projected rate stats by IP
    era_after = (current_era * banked_ip + pitcher_proj_era * pitcher_proj_ip) / total_ip
    era_risk = era_after - current_era

    whip_after = (current_whip * banked_ip + pitcher_proj_whip * pitcher_proj_ip) / total_ip
    whip_risk = whip_after - current_whip

    # Recommend based on risk vs lead
    if era_lead > 0 and era_risk > era_lead * 0.5:
        recommend = "BENCH"  # Risk losing >50% of ERA lead
    elif whip_lead > 0 and whip_risk > whip_lead * 0.5:
        recommend = "BENCH"  # Risk losing >50% of WHIP lead
    elif era_risk > 0.30 or whip_risk > 0.10:
        recommend = "RISKY"
    else:
        recommend = "START"

    return {
        "era_risk": round(era_risk, 3),
        "whip_risk": round(whip_risk, 3),
        "recommend": recommend,
        "era_after": round(era_after, 3),
        "whip_after": round(whip_after, 3),
    }

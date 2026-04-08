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


def recommend_ip_management_mode(urgency_data: dict) -> str:
    """Recommend IP management mode from category urgency state.

    "CHASE_KW": Losing K and/or W -- stream aggressively, accept ratio risk
    "PROTECT_RATIOS": Winning ERA/WHIP -- bench risky pitchers
    "BALANCED": Mixed -- standard approach

    Returns: mode string
    """
    summary = urgency_data.get("summary", {})
    winning = summary.get("winning", [])
    losing = summary.get("losing", [])

    era_safe = "ERA" in winning
    whip_safe = "WHIP" in winning

    k_losing = "K" in losing
    w_losing = "W" in losing

    if era_safe and whip_safe:
        return "PROTECT_RATIOS"
    elif k_losing or w_losing:
        return "CHASE_KW"
    return "BALANCED"

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


# ── Punt Mode Optimizer ───────────────────────────────────────────────

# Mean absolute correlation of each category with all other 11 categories.
# Lower values = more independent = safer to punt without hurting other cats.
# Derived from 10-year FanGraphs H2H category correlation analysis.
CATEGORY_MEAN_CORRELATION: dict[str, float] = {
    "SB": 0.12,
    "W": 0.17,
    "L": 0.17,
    "SV": 0.18,
    "K": 0.25,
    "AVG": 0.40,
    "OBP": 0.42,
    "ERA": 0.45,
    "WHIP": 0.45,
    "HR": 0.55,
    "R": 0.60,
    "RBI": 0.62,
}

# Pairwise correlation clusters — punting one category in a cluster
# drags down the others.  Keys are frozensets of correlated categories.
_CORRELATION_CLUSTERS: list[tuple[set[str], float]] = [
    ({"HR", "R", "RBI"}, 0.83),  # Power cluster
    ({"AVG", "OBP"}, 0.90),  # Contact cluster
    ({"ERA", "WHIP"}, 0.88),  # Pitching rate cluster
    ({"W", "K"}, 0.45),  # Workload cluster (moderate)
]

# Minimum rank threshold to consider a category punt-worthy.
_PUNT_RANK_THRESHOLD = 9


def recommend_punt_targets(
    category_ranks: dict[str, int],
    config: object | None = None,
    max_punts: int = 2,
) -> list[dict]:
    """Recommend categories to punt based on isolation + current weakness.

    A category is a good punt candidate if:
    1. You're already bad at it (rank >= 9 out of 12)
    2. It has low mean correlation with other categories (independent)
    3. Punting it frees resources for correlated categories you CAN win

    Args:
        category_ranks: Dict mapping category name to standings rank (1=best, 12=worst).
        config: LeagueConfig instance (default created if None).
        max_punts: Maximum number of punt recommendations (0 returns empty).

    Returns:
        List of dicts sorted by punt_value descending:
        [{category, rank, isolation_score, punt_value, reason}]
    """
    if max_punts <= 0:
        return []

    if config is None:
        from src.valuation import LeagueConfig

        config = LeagueConfig()

    all_cats = config.all_categories
    candidates: list[dict] = []

    for cat in all_cats:
        rank = category_ranks.get(cat, 6)  # Default to middle if unknown
        if rank < _PUNT_RANK_THRESHOLD:
            continue  # Only punt categories where you're already weak

        # Isolation score: 1 - mean_correlation (higher = more independent)
        mean_corr = CATEGORY_MEAN_CORRELATION.get(cat, 0.40)
        isolation_score = 1.0 - mean_corr

        # Cluster penalty: if punting this cat drags down a cluster partner
        # that you're competitive in, reduce the punt value.
        cluster_penalty = 0.0
        for cluster_cats, cluster_corr in _CORRELATION_CLUSTERS:
            if cat in cluster_cats:
                # Check if any cluster partner is competitive (rank <= 6)
                partners = cluster_cats - {cat}
                for partner in partners:
                    partner_rank = category_ranks.get(partner, 6)
                    if partner_rank <= 6:
                        # Punting this cat would hurt a competitive partner
                        cluster_penalty = max(cluster_penalty, cluster_corr * 0.5)

        # Weakness bonus: the worse you are, the less you lose by punting
        weakness_bonus = (rank - _PUNT_RANK_THRESHOLD) * 0.10  # 0.0 at rank 9, 0.3 at rank 12

        # Punt value: isolation + weakness - cluster drag
        punt_value = round(isolation_score + weakness_bonus - cluster_penalty, 3)

        # Build reason string
        reason_parts: list[str] = []
        if isolation_score >= 0.75:
            reason_parts.append(f"highly independent (correlation {mean_corr:.2f})")
        elif isolation_score >= 0.50:
            reason_parts.append(f"moderately independent (correlation {mean_corr:.2f})")
        else:
            reason_parts.append(f"somewhat correlated (correlation {mean_corr:.2f})")

        reason_parts.append(f"rank {rank} of 12")

        if cluster_penalty > 0:
            reason_parts.append("cluster drag on competitive categories")

        candidates.append(
            {
                "category": cat,
                "rank": rank,
                "isolation_score": round(isolation_score, 3),
                "punt_value": punt_value,
                "reason": "; ".join(reason_parts),
            }
        )

    # Sort by punt_value descending, then by rank descending (worst first)
    candidates.sort(key=lambda x: (-x["punt_value"], -x["rank"]))

    return candidates[:max_punts]


def compute_punt_redistribution_value(
    punt_cats: list[str],
    category_ranks: dict[str, int],
    config: object | None = None,
) -> dict:
    """Estimate standings positions gained if resources from punted categories
    were redistributed to remaining categories.

    The model assumes punting a category frees ~1.0 standings positions of
    resources that can be spread across non-punted categories proportional
    to how close they are to gaining a position (i.e., categories ranked
    5-8 benefit most, already-strong categories benefit less).

    Args:
        punt_cats: List of category names to punt.
        category_ranks: Dict mapping category name to standings rank (1-12).
        config: LeagueConfig instance (default created if None).

    Returns:
        {positions_gained: float, helped_categories: list[str],
         per_category_gain: dict[str, float]}
    """
    if config is None:
        from src.valuation import LeagueConfig

        config = LeagueConfig()

    if not punt_cats:
        return {
            "positions_gained": 0.0,
            "helped_categories": [],
            "per_category_gain": {},
        }

    all_cats = config.all_categories
    non_punted = [c for c in all_cats if c not in punt_cats]

    if not non_punted:
        return {
            "positions_gained": 0.0,
            "helped_categories": [],
            "per_category_gain": {},
        }

    # Total freed resources: ~1.0 standings position per punted category
    # scaled by how independent the punted category is (more independent
    # = more resources freed without collateral damage).
    total_freed = 0.0
    for cat in punt_cats:
        mean_corr = CATEGORY_MEAN_CORRELATION.get(cat, 0.40)
        independence = 1.0 - mean_corr
        total_freed += 0.8 + 0.4 * independence  # 0.8-1.2 range

    # Distribute freed resources to non-punted categories weighted by
    # how improvable they are.  Middle ranks (4-8) benefit most.
    improvability: dict[str, float] = {}
    for cat in non_punted:
        rank = category_ranks.get(cat, 6)
        if rank <= 2:
            # Already elite, diminishing returns
            improvability[cat] = 0.2
        elif rank <= 5:
            # Competitive, moderate room
            improvability[cat] = 0.6
        elif rank <= 8:
            # Middle tier, most improvable
            improvability[cat] = 1.0
        else:
            # Weak but not punted, some room
            improvability[cat] = 0.5

    total_weight = sum(improvability.values())
    if total_weight <= 0:
        return {
            "positions_gained": 0.0,
            "helped_categories": [],
            "per_category_gain": {},
        }

    # Allocate freed resources proportionally
    per_category_gain: dict[str, float] = {}
    helped: list[str] = []
    total_gain = 0.0

    for cat in non_punted:
        share = (improvability[cat] / total_weight) * total_freed
        # Cap at realistic gain per category (~1.5 positions max)
        gain = min(share, 1.5)
        if gain >= 0.05:
            per_category_gain[cat] = round(gain, 2)
            helped.append(cat)
            total_gain += gain

    return {
        "positions_gained": round(total_gain, 2),
        "helped_categories": helped,
        "per_category_gain": per_category_gain,
    }

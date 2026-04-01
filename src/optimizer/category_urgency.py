"""Category Urgency — sigmoid-based urgency scoring for H2H matchups.

Replaces static SGP weighting with dynamic, matchup-aware urgency.
Categories you're close to winning/losing get highest weight.
Categories already decided (winning or losing by a lot) get low weight.

Used by the Daily Optimizer (Stage 10 of the pipeline).
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)


def compute_category_urgency(
    my_totals: dict[str, float],
    opp_totals: dict[str, float],
    config=None,
) -> dict[str, float]:
    """Compute urgency weight per scoring category based on H2H gap.

    Uses a sigmoid function: urgency peaks at 0.5 when tied, approaches
    1.0 when losing, approaches 0.0 when winning comfortably.

    Args:
        my_totals: My team's category totals for the current matchup week.
        opp_totals: Opponent's category totals for the current matchup week.
        config: LeagueConfig (uses default if None).

    Returns:
        Dict mapping category -> urgency weight (0.0 to 1.0).
    """
    if config is None:
        from src.valuation import LeagueConfig

        config = LeagueConfig()

    urgency: dict[str, float] = {}
    inverse_stats = config.inverse_stats  # L, ERA, WHIP — lower is better
    sgp_denoms = config.sgp_denominators

    for cat in config.all_categories:
        my_val = float(my_totals.get(cat, my_totals.get(cat.lower(), 0)) or 0)
        opp_val = float(opp_totals.get(cat, opp_totals.get(cat.lower(), 0)) or 0)
        denom = sgp_denoms.get(cat, 1.0)
        if abs(denom) < 1e-9:
            denom = 1.0

        # For inverse stats (L, ERA, WHIP), lower is better
        # So "losing" means my value is HIGHER than opponent's
        if cat in inverse_stats:
            gap = (my_val - opp_val) / denom  # Positive = I'm losing (my ERA > their ERA)
        else:
            gap = (opp_val - my_val) / denom  # Positive = I'm losing (their HR > my HR)

        # Sigmoid: approaches 1.0 when losing (gap > 0), 0.0 when winning (gap < 0)
        # k controls steepness: 2.0 for counting stats, 3.0 for rate stats
        rate_stats = getattr(config, "rate_stats", {"AVG", "OBP", "ERA", "WHIP"})
        k = 3.0 if cat in rate_stats else 2.0

        urgency[cat] = 1.0 / (1.0 + math.exp(-k * gap))

    return urgency


def classify_rate_stat_mode(
    my_rate: float,
    opp_rate: float,
    cat: str,
    config=None,
) -> str:
    """Classify rate stat strategy: protect, compete, or abandon.

    Args:
        my_rate: My team's rate stat value (e.g., ERA 2.50).
        opp_rate: Opponent's rate stat value.
        cat: Category name (AVG, OBP, ERA, WHIP).
        config: LeagueConfig.

    Returns:
        One of: "protect", "compete", "abandon".
    """
    if config is None:
        from src.valuation import LeagueConfig

        config = LeagueConfig()

    inverse_stats = config.inverse_stats

    # For inverse stats: my lower value = I'm winning
    if cat in inverse_stats:
        gap = opp_rate - my_rate  # Positive = I'm winning (their ERA > mine)
    else:
        gap = my_rate - opp_rate  # Positive = I'm winning (my AVG > theirs)

    # Thresholds calibrated from research:
    # ERA: 0.30 gap is meaningful (e.g., 2.50 vs 2.80)
    # WHIP: 0.05 gap is meaningful (e.g., 1.10 vs 1.15)
    # AVG: 0.020 gap is meaningful (e.g., .265 vs .245)
    # OBP: 0.020 gap is meaningful
    thresholds = {
        "ERA": (0.30, -0.50),  # (protect_threshold, abandon_threshold)
        "WHIP": (0.05, -0.08),
        "AVG": (0.020, -0.030),
        "OBP": (0.020, -0.030),
    }
    protect_thresh, abandon_thresh = thresholds.get(cat.upper(), (0.30, -0.50))

    if gap >= protect_thresh:
        return "protect"
    elif gap <= abandon_thresh:
        return "abandon"
    else:
        return "compete"


def compute_urgency_weights(
    matchup: dict | None,
    config=None,
) -> dict:
    """Convert a Yahoo matchup dict into urgency weights + rate stat modes.

    Args:
        matchup: Dict from yds.get_matchup() with keys: week, opp_name,
            categories (list of {cat, you, opp, result}).
        config: LeagueConfig.

    Returns:
        Dict with:
            urgency: {category: float 0-1}
            rate_modes: {category: "protect"/"compete"/"abandon"}
            summary: {winning: [...], losing: [...], tied: [...]}
    """
    if config is None:
        from src.valuation import LeagueConfig

        config = LeagueConfig()

    if not matchup or not isinstance(matchup, dict):
        # No matchup data — return equal urgency
        equal = {cat: 0.5 for cat in config.all_categories}
        return {
            "urgency": equal,
            "rate_modes": {},
            "summary": {"winning": [], "losing": [], "tied": []},
        }

    # Extract per-category scores from Yahoo matchup format
    categories = matchup.get("categories", [])
    my_totals: dict[str, float] = {}
    opp_totals: dict[str, float] = {}

    for cat_dict in categories:
        cat_name = str(cat_dict.get("cat", "")).upper()
        try:
            you_val = float(cat_dict.get("you", 0)) if cat_dict.get("you") not in ("-", "", None) else 0.0
            opp_val = float(cat_dict.get("opp", 0)) if cat_dict.get("opp") not in ("-", "", None) else 0.0
        except (ValueError, TypeError):
            you_val = 0.0
            opp_val = 0.0
        my_totals[cat_name] = you_val
        opp_totals[cat_name] = opp_val

    # Compute urgency
    urgency = compute_category_urgency(my_totals, opp_totals, config)

    # Classify rate stat modes
    rate_stats = getattr(config, "rate_stats", {"AVG", "OBP", "ERA", "WHIP"})
    rate_modes: dict[str, str] = {}
    for cat in rate_stats:
        my_val = my_totals.get(cat, 0.0)
        opp_val = opp_totals.get(cat, 0.0)
        if my_val > 0 or opp_val > 0:
            rate_modes[cat] = classify_rate_stat_mode(my_val, opp_val, cat, config)

    # Summary
    winning = []
    losing = []
    tied = []
    for cat_dict in categories:
        result = cat_dict.get("result", "")
        cat_name = str(cat_dict.get("cat", "")).upper()
        if result == "WIN":
            winning.append(cat_name)
        elif result == "LOSS":
            losing.append(cat_name)
        else:
            tied.append(cat_name)

    return {
        "urgency": urgency,
        "rate_modes": rate_modes,
        "summary": {"winning": winning, "losing": losing, "tied": tied},
    }

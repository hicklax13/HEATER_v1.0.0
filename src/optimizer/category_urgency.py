"""Category Urgency — sigmoid-based urgency scoring for H2H matchups.

Replaces static SGP weighting with dynamic, matchup-aware urgency.
Categories you're close to winning/losing get highest weight.
Categories already decided (winning or losing by a lot) get low weight.

Used by the Daily Optimizer (Stage 10 of the pipeline).
"""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager

from src.optimizer.constants_registry import CONSTANTS_REGISTRY

logger = logging.getLogger(__name__)


def _get_counting_k() -> float:
    """Read sigmoid k for counting stats from the registry at call time.

    Reading at call time (rather than caching at module-level) ensures that
    calibration changes via scripts/calibrate_sigmoid.py take effect on the
    next urgency computation without requiring a process restart.
    """
    return CONSTANTS_REGISTRY["sigmoid_k_counting"].value


def _get_rate_k() -> float:
    """Read sigmoid k for rate stats from the registry at call time."""
    return CONSTANTS_REGISTRY["sigmoid_k_rate"].value


# Sigmoid steepness parameters (validated via ROADMAP B6)
# Higher k = steeper transition near toss-up
# k=2.0 for counting stats: moderate sensitivity to gaps
# k=3.0 for rate stats: higher sensitivity (rate stats are noisier)
# Calibrate with: python scripts/calibrate_sigmoid.py
#
# These module-level aliases preserve the legacy import surface; production
# code paths inside this module read from the registry on every call so
# calibration changes take effect immediately.
COUNTING_STAT_K = _get_counting_k()
RATE_STAT_K = _get_rate_k()


@contextmanager
def patch_sigmoid_k(counting_k: float | None = None, rate_k: float | None = None):
    """Test/calibration helper: temporarily override sigmoid k values in
    CONSTANTS_REGISTRY (the production read-path) for the duration of a
    ``with`` block. Restores original values on exit.

    Used by sigmoid_calibrator.py and sensitivity_analysis.py to perturb
    these constants in production-equivalent fashion. The earlier approach
    of ``unittest.mock.patch("...COUNTING_STAT_K", value)`` patched module-
    level aliases that the runtime read-path no longer consults — every
    grid point produced identical urgency (BUG-006).

    ConstantEntry is a frozen dataclass, so mutation goes through
    ``object.__setattr__`` to bypass the frozen check.

    Args:
        counting_k: override for sigmoid_k_counting; None = leave unchanged
        rate_k: override for sigmoid_k_rate; None = leave unchanged
    """
    saved: dict[str, float] = {}
    try:
        if counting_k is not None:
            entry = CONSTANTS_REGISTRY["sigmoid_k_counting"]
            saved["sigmoid_k_counting"] = entry.value
            object.__setattr__(entry, "value", float(counting_k))
        if rate_k is not None:
            entry = CONSTANTS_REGISTRY["sigmoid_k_rate"]
            saved["sigmoid_k_rate"] = entry.value
            object.__setattr__(entry, "value", float(rate_k))
        yield
    finally:
        for name, val in saved.items():
            object.__setattr__(CONSTANTS_REGISTRY[name], "value", val)


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

    # 2026-05-19 D6: config is always a LeagueConfig; getattr fallback was defensive.
    rate_stats = config.rate_stats

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
        # k controls steepness: 2.0 for counting stats, 3.0 for rate stats.
        # Read from the registry on every call so calibration changes via
        # scripts/calibrate_sigmoid.py take effect immediately.
        k = _get_rate_k() if cat in rate_stats else _get_counting_k()

        exponent = -k * gap
        exponent = max(-500, min(500, exponent))
        raw = 1.0 / (1.0 + math.exp(exponent))
        # Category-aware urgency floor — OQ-6 resolution (2026-05-15).
        #
        # Without a floor, large positive gaps collapse `raw` to ~0.00005
        # and the post-hoc DCV multiplication crushes pitcher counting
        # stats (already small after /162 daily-fraction scaling) below
        # rounding noise. With a floor that's too high, categories you're
        # winning comfortably get over-weighted and the LP ignores
        # legitimate losing-category urgency.
        #
        # Three tiers calibrated by the H2H-flip-risk literature
        # (RotoGraphs / FantasyPros / CBS strategy; Razzball variance
        # tables 2020-2024):
        #   - HIGH-variance counting cats (SV, SB, W, L) → 0.20 floor.
        #     These flip late-week most often even at apparent comfort
        #     (closer hot-streaks, theft binges, win/loss timing).
        #   - Counting cats (R, HR, RBI, K) → 0.15 floor. Moderate
        #     mid-week swing potential.
        #   - Rate stats (AVG, OBP, ERA, WHIP) → 0.10 floor. After
        #     mid-week samples are large the rates are near-locked, so
        #     the floor can be lower without losing real signal.
        _HIGH_VARIANCE_CATS = {"SV", "SB", "W", "L"}
        if cat in _HIGH_VARIANCE_CATS:
            floor = 0.20
        elif cat in rate_stats:
            floor = 0.10
        else:
            floor = 0.15
        urgency[cat] = max(floor, raw)

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

    # Wave 11B DCV-A5-001: thresholds calibrated to category-typical
    # season-end standings spread in a 12-team H2H league. The asymmetry
    # (protect threshold smaller than the abandon threshold's absolute
    # value) is deliberate risk aversion: it's easier to commit to a
    # category we're slightly ahead in than to give up on one we're
    # slightly behind in (because forfeit means losing each remaining
    # matchup in that cat, not just this week).
    #
    # ERA: 0.30 ≈ ~1 win in standings (e.g., 3.50 vs 3.80); abandon at
    #      -0.50 ≈ 2 wins behind, where catching up via single-week
    #      maneuvers becomes infeasible.
    # WHIP: 0.05 ≈ ~1 win (1.20 vs 1.25); abandon at -0.08.
    # AVG/OBP: 0.020 ≈ ~1 win (.265 vs .245); abandon at -0.030.
    # Source: 2022-2024 FourzynBurn-equivalent league standings analysis.
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
    # 2026-05-19 D6: config is always a LeagueConfig; getattr fallback was defensive.
    rate_stats = config.rate_stats
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

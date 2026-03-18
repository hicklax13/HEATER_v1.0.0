"""Draft strategy analytics — category balance, opportunity cost, streaming, BUY/FAIR/AVOID.

Provides four standalone functions consumed by DraftRecommendationEngine
(src/draft_engine.py) and the draft UI (app.py).  Each function is pure —
no database or Streamlit dependency — so it is trivially testable.

Wires into:
  - src/valuation.py: LeagueConfig (categories, inverse_stats, rate_stats)
  - src/draft_engine.py: enhance_player_pool() calls these functions
  - src/simulation.py: survival probability feeds opportunity cost
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import norm

if TYPE_CHECKING:
    from src.valuation import LeagueConfig

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

# Per-category sigma for the Normal PDF weighting in category balance.
# Sigma represents one "standings-place equivalent" spread for each stat.
DEFAULT_SIGMAS: dict[str, float] = {
    "r": 25.0,
    "hr": 10.0,
    "rbi": 25.0,
    "sb": 10.0,
    "avg": 0.010,
    "obp": 0.012,
    "w": 3.0,
    "l": 2.0,
    "sv": 7.0,
    "k": 35.0,
    "era": 0.30,
    "whip": 0.025,
}

# Inverse categories (lower is better)
_INVERSE_CATS: set[str] = {"l", "era", "whip"}

# Draft progress thresholds for category balance scaling
_EARLY_PROGRESS: float = 0.35  # rounds 1-8  of 23
_LATE_PROGRESS: float = 0.70  # rounds 17+  of 23

# Streaming draft value constants
_ELITE_ERA_THRESHOLD: float = 2.80
_SAVES_STREAMABLE_THRESHOLD: int = 10
_STREAMING_PENALTY_MILD: float = -0.3
_STREAMING_PENALTY_HARSH: float = -0.5

# BUY/FAIR/AVOID thresholds by draft phase
_BFA_EARLY_GAP: int = 20  # picks 1-100
_BFA_MID_GAP: int = 15  # picks 100-200
_BFA_LATE_GAP: int = 10  # picks 200+


# ── Function 1: Category Balance ──────────────────────────────────────


def compute_category_balance(
    roster_totals: dict,
    all_team_totals: list[dict],
    config: LeagueConfig | None = None,
    draft_progress: float = 0.0,
) -> dict[str, float]:
    """Per-category balance multiplier for draft picks.

    Algorithm
    ---------
    1. Compute MEDIAN value per category across *all_team_totals*.
    2. For each category compute ``gap = your_total - median``.
    3. Weight via Normal PDF: ``phi(gap / sigma) / sigma`` — peaks at tie.
    4. Below-median categories get a 1.2x boost; above-median get 0.9x.
    5. Normalise so weights mean == 1.0.
    6. Draft progress scaling:
       - Early (progress < 0.35): compress toward 1.0 (BPA strategy).
       - Mid (0.35-0.70): full strength.
       - Late (> 0.70): amplify 1.5x (urgent gap-filling).

    Parameters
    ----------
    roster_totals : dict
        User's current projected totals, keyed by lowercase category name
        (e.g. ``{"r": 320, "hr": 100, ...}``).  Missing keys default to 0.
    all_team_totals : list[dict]
        Every team's projected totals (same key format).
    config : LeagueConfig, optional
        Provides ``hitting_categories`` and ``pitching_categories`` lists.
        When *None*, all 12 default H2H categories are used.
    draft_progress : float
        0.0 (pick 1) to 1.0 (last pick).

    Returns
    -------
    dict[str, float]
        Mapping lowercase category name -> weight multiplier.
    """
    # Determine active categories
    if config is not None:
        cats = [c.lower() for c in config.hitting_categories + config.pitching_categories]
    else:
        cats = list(DEFAULT_SIGMAS.keys())

    # Graceful fallback: no opponents data -> equal weights
    if not all_team_totals:
        return {cat: 1.0 for cat in cats}

    # Step 1: median per category
    medians: dict[str, float] = {}
    for cat in cats:
        values = [float(t.get(cat, t.get(cat.upper(), 0)) or 0) for t in all_team_totals]
        medians[cat] = float(np.median(values)) if values else 0.0

    # Steps 2-4: gap -> Normal CDF need score -> directional weight
    raw_weights: dict[str, float] = {}
    for cat in cats:
        my_val = float(roster_totals.get(cat, roster_totals.get(cat.upper(), 0)) or 0)
        med_val = medians[cat]
        sigma = DEFAULT_SIGMAS.get(cat, 1.0)

        if sigma <= 0:
            sigma = 1.0

        gap = my_val - med_val

        # For inverse stats, flip the gap so "below median" means "good"
        if cat in _INVERSE_CATS:
            gap = -gap

        # Normalised gap in sigma units (negative = below median = need help)
        z = gap / sigma

        # Hybrid PDF + need weighting:
        # - PDF component: peaks at z=0 (tied = max marginal value), captures closeness
        # - Need component: below-median -> boost, above-median -> reduce
        # Combined: categories near the median AND below it get highest weight,
        # while dominant categories get reduced weight.
        pdf_val = float(norm.pdf(z))  # [0, 0.399], peaks at z=0
        cdf_val = float(norm.cdf(z))  # [0, 1], 0.5 at z=0

        # Need-based with PDF closeness bonus:
        # - Need: below-median categories get higher base weight
        # - Closeness bonus: categories near tie get marginal-value bump
        need_weight = 1.0 + 0.8 * (0.5 - cdf_val)  # below median -> > 1.0
        closeness_bonus = 0.2 * (pdf_val / 0.3989)  # peak 0.2 at z=0

        weight = need_weight + closeness_bonus

        raw_weights[cat] = max(0.1, weight)

    # Step 5: normalise to mean == 1.0
    if raw_weights:
        mean_w = np.mean(list(raw_weights.values()))
        if mean_w > 0:
            raw_weights = {k: v / mean_w for k, v in raw_weights.items()}
        else:
            raw_weights = {k: 1.0 for k in raw_weights}

    # Step 6: draft progress scaling
    progress = max(0.0, min(1.0, draft_progress))
    if progress < _EARLY_PROGRESS:
        # Compress toward 1.0 — BPA dominates early
        compress = 0.3 + 0.7 * (progress / _EARLY_PROGRESS)
        weights = {k: 1.0 + (v - 1.0) * compress for k, v in raw_weights.items()}
    elif progress > _LATE_PROGRESS:
        # Amplify 1.5x — gap-filling dominates late
        weights = {k: 1.0 + (v - 1.0) * 1.5 for k, v in raw_weights.items()}
    else:
        # Middle rounds — full strength
        weights = dict(raw_weights)

    return weights


# ── Function 2: Opportunity Cost ──────────────────────────────────────


def compute_opportunity_cost(
    candidate: pd.Series,
    available_pool: pd.DataFrame,
    survival: float = 0.5,
) -> float:
    """How much value is lost by NOT picking this player now.

    Computes the maximum positional gap between *candidate* and the
    next-best available player at each position the candidate fills,
    weighted by the probability the candidate will be taken before the
    user's next pick.

    Formula
    -------
    ``OC = max_position_gap * (1 - survival_probability)``

    For each position the player fills:
        ``gap = candidate_score - next_best_at_position``
    Use the MAX gap (the position where they are most irreplaceable).
    Multi-position players distribute risk → lower OC.

    Parameters
    ----------
    candidate : pd.Series
        Row from the player pool; must contain ``positions`` (str) and
        a score column (``enhanced_pick_score`` or ``pick_score``).
    available_pool : pd.DataFrame
        Other available (undrafted) players.
    survival : float
        Probability (0-1) that the candidate will still be available at
        the user's next pick.

    Returns
    -------
    float
        Non-negative SGP cost of skipping this candidate.
    """
    # Determine score column
    score_col = "enhanced_pick_score" if "enhanced_pick_score" in candidate.index else "pick_score"
    candidate_score = float(candidate.get(score_col, 0) or 0)

    # Parse positions
    positions_str = str(candidate.get("positions", "Util") or "Util")
    positions = [p.strip() for p in positions_str.split(",") if p.strip()]
    if not positions:
        positions = ["Util"]

    # Handle empty pool — no alternatives at all
    if available_pool.empty or "positions" not in available_pool.columns:
        survival_clamped = max(0.0, min(1.0, survival))
        return max(0.0, candidate_score * (1.0 - survival_clamped))

    # Determine score column in pool
    pool_score_col = "enhanced_pick_score" if "enhanced_pick_score" in available_pool.columns else "pick_score"
    if pool_score_col not in available_pool.columns:
        return 0.0

    # Exclude the candidate from the pool
    candidate_id = candidate.get("player_id")
    if candidate_id is not None and "player_id" in available_pool.columns:
        pool = available_pool[available_pool["player_id"] != candidate_id]
    else:
        pool = available_pool

    if pool.empty:
        return max(0.0, candidate_score * (1.0 - max(0.0, min(1.0, survival))))

    # Find max gap across all candidate positions
    max_gap = 0.0
    for pos in positions:
        eligible = pool[
            pool["positions"].apply(lambda p: pos in [x.strip() for x in str(p).split(",")] if pd.notna(p) else False)
        ]
        if eligible.empty:
            # No alternatives at this position — candidate is irreplaceable
            next_best = 0.0
        else:
            next_best = float(eligible[pool_score_col].max())
        gap = max(0.0, candidate_score - next_best)
        max_gap = max(max_gap, gap)

    # Multi-position discount: each extra position reduces OC
    if len(positions) > 1:
        multi_discount = 1.0 / (1.0 + 0.2 * (len(positions) - 1))
        max_gap *= multi_discount

    # Weight by probability of being taken
    survival_clamped = max(0.0, min(1.0, survival))
    oc = max_gap * (1.0 - survival_clamped)

    return max(0.0, oc)


# ── Function 3: Streaming Draft Value ─────────────────────────────────


def compute_streaming_draft_value(
    pitcher: pd.Series,
    config: LeagueConfig | None = None,
) -> float:
    """Discount pitchers whose value is mostly streamable.

    In H2H Categories leagues, managers add/drop SPs weekly to chase
    Wins and Strikeouts.  A pitcher whose value comes primarily from
    counting stats (W, K) rather than rate-stat anchoring (low ERA/WHIP)
    is replaceable through streaming — therefore worth less in a draft.

    Rules
    -----
    - RP / closers (SV > 10): return 0.0 — saves are not streamable.
    - Elite SP (ERA < 2.80): return 0.0 — keep elite rate anchors.
    - Others: penalty = -0.3 to -0.5 based on streamable fraction.

    The *streamable fraction* is the share of the pitcher's projected
    counting SGP (W + K) relative to total SGP.

    Parameters
    ----------
    pitcher : pd.Series
        Row with at least ``sv``, ``era``, ``w``, ``k``, ``ip`` columns.
    config : LeagueConfig, optional
        Used for SGP denominators.  Falls back to default LeagueConfig.

    Returns
    -------
    float
        Penalty <= 0.0.  Closer to -0.5 for highly streamable pitchers.
    """
    # Non-pitchers: no penalty
    is_hitter = pitcher.get("is_hitter", True)
    if is_hitter is True or is_hitter == 1:
        return 0.0

    sv = float(pitcher.get("sv", 0) or 0)
    era = float(pitcher.get("era", 0) or 0)
    ip = float(pitcher.get("ip", 0) or 0)
    w = float(pitcher.get("w", 0) or 0)
    k = float(pitcher.get("k", 0) or 0)

    # RP / closers — saves are scarce and not streamable
    if sv > _SAVES_STREAMABLE_THRESHOLD:
        return 0.0

    # Elite ERA anchor — keep
    if era > 0 and era < _ELITE_ERA_THRESHOLD:
        return 0.0

    # Low IP → barely rosterable anyway, small penalty
    if ip < 30:
        return _STREAMING_PENALTY_MILD

    # Compute streamable fraction: W + K counting value vs total value
    # Use SGP denominators for scaling
    if config is not None:
        sgp_denoms = config.sgp_denominators
    else:
        sgp_denoms = {
            "W": 3.5,
            "K": 45.0,
            "ERA": 0.20,
            "WHIP": 0.020,
            "L": 3.0,
            "SV": 9.0,
        }

    w_denom = sgp_denoms.get("W", sgp_denoms.get("w", 3.5))
    k_denom = sgp_denoms.get("K", sgp_denoms.get("k", 45.0))
    era_denom = sgp_denoms.get("ERA", sgp_denoms.get("era", 0.20))
    whip_denom = sgp_denoms.get("WHIP", sgp_denoms.get("whip", 0.020))

    whip = float(pitcher.get("whip", 0) or 0)

    # Counting SGP: W and K contributions
    counting_sgp = abs(w / w_denom) + abs(k / k_denom) if w_denom and k_denom else 0.0

    # Rate SGP: ERA and WHIP contributions (only positive = better than avg)
    # Only pitchers who are BETTER than league average anchor your ratios;
    # a 4.80 ERA pitcher does not provide rate anchoring value.
    rate_sgp = 0.0
    if era > 0 and era_denom:
        era_contrib = (4.50 - era) / era_denom  # positive when ERA < 4.50
        rate_sgp += max(0.0, era_contrib)
    if whip > 0 and whip_denom:
        whip_contrib = (1.30 - whip) / whip_denom  # positive when WHIP < 1.30
        rate_sgp += max(0.0, whip_contrib)

    total_sgp = counting_sgp + rate_sgp
    if total_sgp <= 0:
        return _STREAMING_PENALTY_MILD

    streamable_fraction = counting_sgp / total_sgp

    # Interpolate penalty: 0.3 at fraction=0.5, 0.5 at fraction=1.0
    if streamable_fraction <= 0.5:
        return 0.0  # rate-dominant pitcher — not streamable
    penalty = _STREAMING_PENALTY_MILD + (
        (_STREAMING_PENALTY_HARSH - _STREAMING_PENALTY_MILD) * (streamable_fraction - 0.5) / 0.5
    )
    return max(_STREAMING_PENALTY_HARSH, min(0.0, penalty))


# ── Function 4: BUY / FAIR / AVOID ───────────────────────────────────


def compute_buy_fair_avoid(
    enhanced_rank: int,
    adp_rank: int,
    current_pick: int,
    total_picks: int = 276,
) -> str:
    """Classify a player as BUY, FAIR, or AVOID based on value vs ADP.

    Compares the player's *enhanced_rank* (model rank after all
    adjustments) against their *adp_rank* (consensus market rank).
    A large positive gap (model ranks player much higher than ADP)
    signals a buy; a large negative gap signals avoid.

    The required gap threshold scales with draft progress because
    rankings compress as the talent pool narrows.

    Parameters
    ----------
    enhanced_rank : int
        Player's rank based on enhanced_pick_score (1 = best).
    adp_rank : int
        Player's rank based on ADP (1 = first pick).
    current_pick : int
        The current overall pick number (1-based).
    total_picks : int
        Total picks in the draft (default 276 = 12 teams x 23 rounds).

    Returns
    -------
    str
        ``"BUY"`` | ``"FAIR"`` | ``"AVOID"``
    """
    # Validate inputs — ranks must be positive
    if enhanced_rank < 1 or adp_rank < 1:
        return "FAIR"

    # gap > 0 means ADP ranks the player later (lower) than our model
    # i.e. our model thinks the player is better -> BUY
    gap = adp_rank - enhanced_rank

    # Threshold scales with draft progress
    if current_pick <= 100:
        threshold = _BFA_EARLY_GAP
    elif current_pick <= 200:
        threshold = _BFA_MID_GAP
    else:
        threshold = _BFA_LATE_GAP

    if gap >= threshold:
        return "BUY"
    elif gap <= -threshold:
        return "AVOID"
    return "FAIR"

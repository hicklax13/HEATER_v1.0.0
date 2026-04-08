"""Dual objective blending for H2H weekly + season-long optimization.

H2H category leagues benefit from balancing two competing objectives:

  1. Win THIS WEEK's H2H matchup (short-term tactics).
  2. Maximize season-long category totals across the whole season
     (long-term strategy).

The ``alpha`` parameter controls the blend:
  - alpha = 0.0 -> pure season-long focus
  - alpha = 0.5 -> balanced
  - alpha = 1.0 -> pure H2H (weekly focus)

``recommend_alpha()`` auto-selects alpha based on the current situation:
weeks remaining, standings rank, and H2H record.

This module has no Streamlit dependency and no external API calls.
"""

from __future__ import annotations

import logging

from src.validation.constant_optimizer import load_constants

logger = logging.getLogger(__name__)

_CONSTANTS = load_constants()

# ── Constants ────────────────────────────────────────────────────────

ALL_CATEGORIES: list[str] = [
    "r",
    "hr",
    "rbi",
    "sb",
    "avg",
    "obp",
    "w",
    "l",
    "sv",
    "k",
    "era",
    "whip",
]

INVERSE_CATS: set[str] = {"l", "era", "whip"}

# Small epsilon to avoid division by zero.
_EPSILON: float = 1e-12


# ── Dual Objective Blending ──────────────────────────────────────────


def blend_h2h_roto_weights(
    h2h_weights: dict[str, float],
    roto_weights: dict[str, float],
    alpha: float = 0.5,
) -> dict[str, float]:
    """Blend H2H weekly and roto season-long category weights.

    Core formula per category c:
      blended_c = alpha * h2h_c + (1 - alpha) * roto_c

    Missing categories in one source are filled from the other source.

    Args:
        h2h_weights: Per-category weights optimized for this week's H2H
            matchup.
        roto_weights: Per-category weights optimized for season-long roto
            standings.
        alpha: Blend parameter in [0.0, 1.0].
            0.0 = pure roto, 1.0 = pure H2H, 0.5 = balanced.

    Returns:
        Dict mapping category name to blended weight, normalized so that
        the mean across all categories equals 1.0.
    """
    alpha = max(0.0, min(1.0, alpha))

    # Union of all categories from both sources
    all_cats = set(h2h_weights.keys()) | set(roto_weights.keys())

    if not all_cats:
        return {}

    blended: dict[str, float] = {}
    for cat in all_cats:
        h2h_val = h2h_weights.get(cat)
        roto_val = roto_weights.get(cat)

        if h2h_val is not None and roto_val is not None:
            blended[cat] = alpha * h2h_val + (1.0 - alpha) * roto_val
        elif h2h_val is not None:
            # Only H2H available for this category
            blended[cat] = alpha * h2h_val
        elif roto_val is not None:
            # Only roto available for this category
            blended[cat] = (1.0 - alpha) * roto_val

    # Normalize so mean = 1.0
    if blended:
        vals = list(blended.values())
        mean_val = sum(vals) / len(vals)
        if mean_val > _EPSILON:
            blended = {cat: v / mean_val for cat, v in blended.items()}
        else:
            blended = {cat: 1.0 for cat in blended}

    return blended


# ── Alpha Recommendation ─────────────────────────────────────────────


def recommend_alpha(
    weeks_remaining: int,
    roto_rank: int | None = None,
    h2h_record_wins: int | None = None,
    h2h_record_losses: int | None = None,
    num_teams: int = 12,
    playoff_cutoff: int = 6,
    desperation_level: float = 0.0,
) -> float:
    """Auto-recommend the H2H/roto blend alpha based on league situation.

    In H2H category leagues, winning each weekly matchup is always the
    primary objective.  The base alpha therefore never drops below 0.5:

      - Early season (>16 weeks remaining): 0.55 (slight season-long tilt)
      - Mid season (8-16 weeks): 0.65 (balanced, H2H leaning)
      - Late season (3-8 weeks): 0.85 (strong H2H focus)
      - Playoff weeks (<3 weeks): 0.85 (maximum H2H focus)

    Adjustments are made based on standings:
      - Very bad roto rank (bottom 3 of league): alpha += 0.1
        (punt roto, focus on salvaging H2H wins)
      - Very bad H2H record (below .400 win rate): alpha -= 0.1
        (H2H is lost cause, focus roto)
      - Desperation level (0.0-1.0): boosts alpha by up to +0.25
        for teams that must win now.

    The result is always clamped to [0.5, 1.0] — H2H weekly matchups
    always matter at least as much as season-long in H2H leagues.

    Args:
        weeks_remaining: Number of weeks left in the regular season.
        roto_rank: Current roto standings rank (1 = first place).
            If None, no roto adjustment is applied.
        h2h_record_wins: Total H2H wins so far.  If None, no H2H
            adjustment is applied.
        h2h_record_losses: Total H2H losses so far.  If None, no H2H
            adjustment is applied.
        num_teams: Number of teams in the league (default 12).
        playoff_cutoff: Number of teams that make playoffs (default 6).
        desperation_level: Float in [0.0, 1.0] indicating how desperate
            the team is (e.g. bubble team near playoffs).  Boosts alpha
            by up to +0.25.  Default 0.0 for backward compatibility.

    Returns:
        Recommended alpha in [0.5, 1.0].
    """
    # Base alpha from time remaining — H2H floor of 0.55
    if weeks_remaining < 3:
        alpha = 0.85
    elif weeks_remaining < 8:
        alpha = 0.85
    elif weeks_remaining <= 16:
        alpha = 0.65
    else:
        alpha = 0.55

    # Desperation boost: up to +0.25 for desperate teams
    desperation_level = max(0.0, min(1.0, desperation_level))
    alpha += desperation_level * 0.25

    # Roto rank adjustment: bottom 3 teams shift toward H2H
    if roto_rank is not None and num_teams > 0:
        bottom_3_threshold = num_teams - 2  # e.g. 10, 11, 12 for 12-team
        if roto_rank >= bottom_3_threshold:
            alpha += 0.1

    # H2H record adjustment: below .400 win rate shifts toward roto
    if h2h_record_wins is not None and h2h_record_losses is not None:
        total_games = h2h_record_wins + h2h_record_losses
        if total_games > 0:
            win_rate = h2h_record_wins / total_games
            if win_rate < 0.4:
                alpha -= 0.1

    # Clamp to [0.5, 1.0] — H2H always matters at least 50% in H2H leagues
    return max(0.5, min(1.0, alpha))

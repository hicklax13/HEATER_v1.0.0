"""War Room Briefing — dynamic matchup intelligence for My Team page.

Replaces static alerts with actionable, day-aware insights:
1. Matchup Pulse — live W-L-T score with category breakdown
2. Flippable Categories — close categories with specific move suggestions
3. Today's Actions — schedule-aware roster decisions (war_room_actions.py)
4. Hot/Cold Report — recent performance deviations (war_room_hotcold.py)
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Categories where lower is better
INVERSE_CATS = {"L", "ERA", "WHIP"}

# Rate stats (non-counting)
RATE_CATS = {"AVG", "OBP", "ERA", "WHIP"}

# Flippable thresholds per category type
_COUNTING_THRESHOLD = 3
_L_THRESHOLD = 2
_RATE_THRESHOLD = 0.015  # AVG, OBP
_INVERSE_RATE_THRESHOLD = 0.30  # ERA, WHIP

# Suggestion templates keyed by (category_set, direction)
_SUGGESTIONS: dict[tuple[frozenset[str], str], str] = {
    (frozenset({"SB"}), "flip_to_win"): "Start speed-eligible bench players to close the {gap} SB gap",
    (frozenset({"HR", "RBI", "R"}), "flip_to_win"): "Maximize active lineup slots with power hitters",
    (frozenset({"K", "W"}), "flip_to_win"): "Stream a SP with favorable matchup this week",
    (frozenset({"SV"}), "flip_to_win"): "Add a closer from FA if available",
    (frozenset({"AVG", "OBP"}), "flip_to_win"): "Start high-AVG bench bats to lift rate stats",
    (frozenset({"AVG", "OBP"}), "at_risk"): "Bench low-AVG hitters to protect rate stat lead",
    (frozenset({"ERA", "WHIP"}), "flip_to_win"): "Stream a high-floor SP to improve ratios",
    (frozenset({"ERA", "WHIP"}), "at_risk"): "Bench risky SP starts to protect ratio lead",
    (frozenset({"L"}), "at_risk"): "Bench SP in tough matchups to avoid accumulating losses",
}


def _get_suggestion(cat: str, direction: str, gap: float) -> str:
    """Return a context-specific suggestion for a flippable category."""
    for cat_set, dir_key in _SUGGESTIONS:
        if cat in cat_set and direction == dir_key:
            template = _SUGGESTIONS[(cat_set, dir_key)]
            return template.format(gap=int(gap)) if "{gap}" in template else template

    # Generic fallbacks
    if direction == "flip_to_win":
        return f"Focus roster moves on closing the {cat} gap"
    return f"Protect your {cat} lead with conservative lineup choices"


def _parse_cat_value(raw: str, cat: str) -> float:
    """Parse a category value string to float."""
    try:
        return float(raw)
    except (ValueError, TypeError):
        logger.warning("Could not parse value %r for category %s", raw, cat)
        return 0.0


def _get_threshold(cat: str) -> float:
    """Return the flippable threshold for a given category."""
    if cat == "L":
        return _L_THRESHOLD
    if cat in {"ERA", "WHIP"}:
        return _INVERSE_RATE_THRESHOLD
    if cat in {"AVG", "OBP"}:
        return _RATE_THRESHOLD
    return _COUNTING_THRESHOLD


def compute_matchup_pulse(
    matchup: dict[str, Any] | None,
    opponent_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute live matchup pulse from Yahoo matchup data.

    Parameters
    ----------
    matchup : dict or None
        The matchup dict from ``yds.get_matchup()``. Contains ``week``,
        ``opp_name``, ``wins``, ``losses``, ``ties``, and ``categories``.
    opponent_profile : dict or None
        Optional opponent profile (unused currently, reserved for future
        enrichment such as opponent tier or trade willingness).

    Returns
    -------
    dict
        Matchup pulse with score breakdown, verdict, and per-category
        win/loss/tie lists. Returns ``{"available": False, ...}`` when
        no matchup data is present.
    """
    empty: dict[str, Any] = {
        "available": False,
        "week": 0,
        "opponent": "",
        "score": "0-0-0",
        "verdict": "Tied",
        "winning_cats": [],
        "losing_cats": [],
        "tied_cats": [],
        "margin": 0,
    }

    if matchup is None:
        return empty

    wins = matchup.get("wins", 0) or 0
    losses = matchup.get("losses", 0) or 0
    ties = matchup.get("ties", 0) or 0

    if wins > losses:
        verdict = "Leading"
    elif losses > wins:
        verdict = "Trailing"
    else:
        verdict = "Tied"

    winning_cats: list[str] = []
    losing_cats: list[str] = []
    tied_cats: list[str] = []

    for entry in matchup.get("categories", []):
        cat = entry.get("cat", "")
        result = entry.get("result", "").upper()
        if result == "WIN":
            winning_cats.append(cat)
        elif result == "LOSS":
            losing_cats.append(cat)
        else:
            tied_cats.append(cat)

    return {
        "available": True,
        "week": matchup.get("week", 0),
        "opponent": matchup.get("opp_name", ""),
        "score": f"{wins}-{losses}-{ties}",
        "verdict": verdict,
        "winning_cats": winning_cats,
        "losing_cats": losing_cats,
        "tied_cats": tied_cats,
        "margin": wins - losses,
    }


def get_flippable_categories(
    matchup: dict[str, Any] | None,
    roster: pd.DataFrame | None = None,
    fa_pool: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Identify categories closest to flipping with actionable suggestions.

    Scans both losing categories (``flip_to_win``) and winning categories
    (``at_risk``) to find those within the flippable threshold. Results
    are sorted by absolute gap ascending, capped at 3.

    Parameters
    ----------
    matchup : dict or None
        The matchup dict from ``yds.get_matchup()``.
    roster : DataFrame or None
        Current user roster. Reserved for future enrichment (e.g.,
        identifying specific bench players to activate). Not used in
        the current implementation.
    fa_pool : DataFrame or None
        Free agent pool. Reserved for future enrichment (e.g.,
        suggesting specific FA pickups). Not used in the current
        implementation.

    Returns
    -------
    list[dict]
        Up to 3 flippable category dicts, each containing the category
        name, direction, values, gap info, and a suggestion string.
        Returns an empty list when no matchup data is present.
    """
    if matchup is None:
        return []

    categories = matchup.get("categories", [])
    if not categories:
        return []

    candidates: list[dict[str, Any]] = []

    for entry in categories:
        cat = entry.get("cat", "")
        result = entry.get("result", "").upper()
        you_raw = entry.get("you", "0")
        opp_raw = entry.get("opp", "0")

        you_val = _parse_cat_value(you_raw, cat)
        opp_val = _parse_cat_value(opp_raw, cat)

        threshold = _get_threshold(cat)
        is_inverse = cat in INVERSE_CATS

        if result == "LOSS":
            # We're losing — compute gap to flip
            if is_inverse:
                # For inverse cats, we're losing means our value is higher
                # (worse). Gap = our value - their value.
                gap = you_val - opp_val
            else:
                # For normal cats, we're losing means our value is lower.
                # Gap = their value - our value.
                gap = opp_val - you_val

            abs_gap = abs(gap)
            if abs_gap <= threshold:
                # Compute gap_pct relative to opponent value (avoid div/0)
                denom = max(abs(opp_val), 0.001)
                gap_pct = round((abs_gap / denom) * 100, 1)

                candidates.append(
                    {
                        "category": cat,
                        "direction": "flip_to_win",
                        "you": you_val,
                        "opp": opp_val,
                        "gap": abs_gap,
                        "gap_pct": gap_pct,
                        "suggestion": _get_suggestion(cat, "flip_to_win", abs_gap),
                    }
                )

        elif result == "WIN":
            # We're winning — check if at risk of flipping
            if is_inverse:
                # For inverse cats, winning means our value is lower (better).
                # Gap = their value - our value.
                gap = opp_val - you_val
            else:
                # For normal cats, winning means our value is higher.
                # Gap = our value - their value.
                gap = you_val - opp_val

            abs_gap = abs(gap)
            if abs_gap <= threshold:
                denom = max(abs(you_val), 0.001)
                gap_pct = round((abs_gap / denom) * 100, 1)

                candidates.append(
                    {
                        "category": cat,
                        "direction": "at_risk",
                        "you": you_val,
                        "opp": opp_val,
                        "gap": abs_gap,
                        "gap_pct": gap_pct,
                        "suggestion": _get_suggestion(cat, "at_risk", abs_gap),
                    }
                )

        # TIE results: technically flippable in either direction, treat as
        # flip_to_win with gap=0
        elif result == "TIE":
            candidates.append(
                {
                    "category": cat,
                    "direction": "flip_to_win",
                    "you": you_val,
                    "opp": opp_val,
                    "gap": 0,
                    "gap_pct": 0.0,
                    "suggestion": _get_suggestion(cat, "flip_to_win", 0),
                }
            )

    # Sort by gap ascending (closest to flipping first), cap at 3
    candidates.sort(key=lambda c: c["gap"])
    return candidates[:3]

"""Standings engine — shared computation for League Standings + Matchup Planner.

Pure functions (no Streamlit dependency). All Yahoo/DB I/O happens in callers.

This module provides:
- Schedule helpers: parse scoreboard matchups, find user opponents
- Category result parsing: W/L/T determination per category
- (Future) Category win probabilities, MC season simulation, magic numbers
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# ── Schedule helpers ────────────────────────────────────────────────


def parse_scoreboard_matchups(
    matchups: list[dict],
) -> list[tuple[str, str]]:
    """Extract (team_a, team_b) pairs from a scoreboard response.

    Args:
        matchups: List of dicts with ``team_a`` and ``team_b`` keys.

    Returns:
        List of (team_a_name, team_b_name) tuples.
    """
    result: list[tuple[str, str]] = []
    for m in matchups:
        team_a = str(m.get("team_a", ""))
        team_b = str(m.get("team_b", ""))
        if team_a and team_b:
            result.append((team_a, team_b))
    return result


def find_user_opponent(
    schedule: dict[int, list[tuple[str, str]]],
    week: int,
    user_team_name: str,
) -> str | None:
    """Find the user's opponent for a specific week.

    Args:
        schedule: ``{week: [(team_a, team_b), ...]}``
        week: Week number to look up.
        user_team_name: User's team name.

    Returns:
        Opponent team name, or ``None`` if not found.
    """
    matchups = schedule.get(week, [])
    for team_a, team_b in matchups:
        if team_a == user_team_name:
            return team_b
        if team_b == user_team_name:
            return team_a
    return None


def parse_week_category_results(
    categories: list[dict],
) -> list[dict]:
    """Parse category-level W/L/T results from a completed matchup.

    Args:
        categories: List of dicts with keys ``name``, ``user_val``,
            ``opp_val``, ``is_inverse``.

    Returns:
        List of dicts with keys ``name``, ``user_val``, ``opp_val``,
        ``result`` (one of ``"W"``, ``"L"``, ``"T"``).
    """
    results: list[dict] = []
    for cat in categories:
        name = cat["name"]
        user_val = float(cat["user_val"])
        opp_val = float(cat["opp_val"])
        is_inverse = bool(cat.get("is_inverse", False))

        if user_val == opp_val:
            result = "T"
        elif is_inverse:
            result = "W" if user_val < opp_val else "L"
        else:
            result = "W" if user_val > opp_val else "L"

        results.append(
            {
                "name": name,
                "user_val": user_val,
                "opp_val": opp_val,
                "result": result,
            }
        )
    return results

# src/pick_predictor.py
"""Pick Predictor — survival probability at each future user pick."""

from __future__ import annotations

import math

from scipy.stats import norm

POSITION_SHAPE_PARAMS: dict[str, float] = {
    "C": 1.4,
    "SS": 1.3,
    "2B": 1.2,
    "3B": 1.1,
    "1B": 0.9,
    "OF": 0.8,
    "SP": 1.0,
    "RP": 1.2,
    "Util": 0.7,
    "P": 1.0,
    "DH": 0.9,
}

_SIGMA = 10.0  # ADP standard deviation (matches simulation.py default)


def weibull_survival(
    picks_remaining: float,
    adp_distance: float,  # unused, kept for API compat
    shape: float,
    scale: float,
) -> float:
    """Weibull survival S(t) = exp(-(t/lambda)^k), clamped [0,1]."""
    if scale <= 0 or picks_remaining <= 0:
        return 1.0
    return max(0.0, min(1.0, math.exp(-((picks_remaining / scale) ** shape))))


def _get_primary_position(positions: str) -> str:
    """Extract primary position from comma-separated positions string."""
    first = positions.split(",")[0].strip() if positions else "Util"
    return first if first in POSITION_SHAPE_PARAMS else "Util"


def _future_user_picks(
    current_pick: int,
    user_team_index: int,
    num_teams: int,
    num_rounds: int,
) -> list[dict]:
    """Compute all future picks for the user in snake draft order."""
    total_picks = num_teams * num_rounds
    picks = []
    for overall in range(current_pick + 1, total_picks + 1):
        rnd = (overall - 1) // num_teams  # 0-indexed round
        pos_in_round = (overall - 1) % num_teams
        if rnd % 2 == 0:
            team_idx = pos_in_round
        else:
            team_idx = num_teams - 1 - pos_in_round
        if team_idx == user_team_index:
            picks.append(
                {
                    "pick": overall,
                    "round": rnd + 1,
                    "pick_in_round": pos_in_round + 1,
                }
            )
    return picks


def compute_survival_curve(
    player_adp: float,
    player_positions: str,
    current_pick: int,
    user_team_index: int,
    num_teams: int = 12,
    num_rounds: int = 23,
) -> list[dict]:
    """Compute survival probability at each future user pick.

    Returns list of {pick, round, p_available, p_normal, p_weibull}.
    """
    future = _future_user_picks(current_pick, user_team_index, num_teams, num_rounds)
    primary_pos = _get_primary_position(player_positions)
    shape = POSITION_SHAPE_PARAMS.get(primary_pos, 1.0)

    results = []
    for fp in future:
        picks_between = fp["pick"] - current_pick
        # Normal CDF component (mirrors simulation.py)
        z = (player_adp - fp["pick"]) / (_SIGMA * max(1, picks_between**0.3))
        p_normal = float(norm.cdf(z))
        p_normal = max(0.01, min(0.99, p_normal))
        # Weibull component
        adp_dist = max(1.0, player_adp - current_pick)
        scale = adp_dist * 1.5
        p_weibull = weibull_survival(picks_between, adp_dist, shape, scale)
        # Blend
        p_blended = 0.6 * p_normal + 0.4 * p_weibull
        results.append(
            {
                "pick": fp["pick"],
                "round": fp["round"],
                "pick_in_round": fp["pick_in_round"],
                "p_available": round(p_blended, 4),
                "p_normal": round(p_normal, 4),
                "p_weibull": round(p_weibull, 4),
            }
        )
    return results

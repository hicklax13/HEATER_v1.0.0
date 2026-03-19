# src/power_rankings.py
"""5-factor composite league power rankings with bootstrap CI."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Component weights (sum to 1.0)
WEIGHTS: dict[str, float] = {
    "roster_quality": 0.40,
    "category_balance": 0.25,
    "schedule_strength": 0.15,
    "injury_exposure": 0.10,
    "momentum": 0.10,
}


def compute_roster_quality(team_trade_values: float, max_trade_values: float) -> float:
    """Roster quality as fraction of league-best total trade value [0, 1]."""
    if max_trade_values <= 0:
        return 0.0
    return max(0.0, min(1.0, team_trade_values / max_trade_values))


def compute_category_balance_score(
    team_zscores: dict[str, float],
) -> float:
    """Category balance = 1 / (1 + CV(z_scores)) where CV = std/|mean|.

    Returns [0, 1]. Higher = more balanced.
    """
    if not team_zscores:
        return 0.5
    values = list(team_zscores.values())
    std = float(np.std(values))
    mean = float(np.mean(values))
    if abs(mean) < 1e-9:
        # Near-zero mean: use just std-based score
        return max(0.0, min(1.0, 1.0 / (1.0 + std)))
    cv = std / abs(mean)
    return max(0.0, min(1.0, 1.0 / (1.0 + cv)))


def compute_schedule_strength_index(
    team_name: str,
    roster_qualities: dict[str, float],
    schedule: list[list[tuple[str, str]]] | None = None,
) -> float:
    """Mean opponent roster quality, normalized [0, 1].

    Falls back to league average if no schedule provided.
    """
    if not roster_qualities:
        return 0.5

    if schedule:
        opponents = []
        for week in schedule:
            for a, b in week:
                if a == team_name:
                    opponents.append(roster_qualities.get(b, 0.5))
                elif b == team_name:
                    opponents.append(roster_qualities.get(a, 0.5))
        if opponents:
            return round(float(np.mean(opponents)), 4)

    # Fallback: average of all other teams
    others = [v for k, v in roster_qualities.items() if k != team_name]
    return round(float(np.mean(others)), 4) if others else 0.5


def compute_injury_exposure(
    roster_trade_values: list[float],
    roster_health_scores: list[float],
) -> float:
    """Injury risk weighted by player value [0, 1]. Higher = more exposed.

    Risk = 1 - sum(tv_i * hs_i) / sum(tv_i) for top-10 players.
    """
    if not roster_trade_values or not roster_health_scores:
        return 0.0

    # Use top-10 by trade value
    paired = sorted(
        zip(roster_trade_values, roster_health_scores),
        key=lambda x: x[0],
        reverse=True,
    )[:10]

    tv_sum = sum(tv for tv, _ in paired)
    if tv_sum <= 0:
        return 0.0

    weighted_health = sum(tv * hs for tv, hs in paired)
    return max(0.0, min(1.0, 1.0 - weighted_health / tv_sum))


def compute_momentum(
    recent_win_rate: float | None = None,
    season_win_rate: float | None = None,
) -> float:
    """Momentum = recent/season win rate, clipped [0.5, 2.0]. Default 1.0."""
    if recent_win_rate is None or season_win_rate is None:
        return 1.0
    if season_win_rate <= 0:
        return 1.0
    ratio = recent_win_rate / season_win_rate
    return max(0.5, min(2.0, ratio))


def compute_power_rating(
    roster_quality: float,
    category_balance: float,
    schedule_strength: float,
    injury_exposure: float,
    momentum: float,
) -> float:
    """Composite power rating [0, 100].

    100 * (0.40*roster + 0.25*balance + 0.15*sos + 0.10*(1-risk) + 0.10*momentum_norm)
    """
    # Normalize momentum from [0.5, 2.0] to [0, 1]
    momentum_norm = (momentum - 0.5) / 1.5

    return round(
        100
        * (
            WEIGHTS["roster_quality"] * roster_quality
            + WEIGHTS["category_balance"] * category_balance
            + WEIGHTS["schedule_strength"] * schedule_strength
            + WEIGHTS["injury_exposure"] * (1.0 - injury_exposure)
            + WEIGHTS["momentum"] * momentum_norm
        ),
        1,
    )


def compute_power_rankings(
    team_data: list[dict],
) -> pd.DataFrame:
    """Compute power rankings for all teams.

    Args:
        team_data: List of dicts with keys:
            team_name, roster_quality, category_balance,
            schedule_strength, injury_exposure, momentum

    Returns:
        DataFrame sorted by power_rating descending with rank column.
    """
    results = []
    for td in team_data:
        rating = compute_power_rating(
            roster_quality=td.get("roster_quality", 0.5),
            category_balance=td.get("category_balance", 0.5),
            schedule_strength=td.get("schedule_strength", 0.5),
            injury_exposure=td.get("injury_exposure", 0.0),
            momentum=td.get("momentum", 1.0),
        )
        results.append(
            {
                "team_name": td["team_name"],
                "power_rating": rating,
                "roster_quality": td.get("roster_quality", 0.5),
                "category_balance": td.get("category_balance", 0.5),
                "schedule_strength": td.get("schedule_strength", 0.5),
                "injury_exposure": td.get("injury_exposure", 0.0),
                "momentum": td.get("momentum", 1.0),
            }
        )

    df = pd.DataFrame(results)
    df = df.sort_values("power_rating", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


def bootstrap_confidence_interval(
    team_rating: float,
    component_stds: dict[str, float] | None = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap 90% CI for a team's power rating.

    Resamples component values using provided standard deviations.

    Returns:
        (p5, p95) tuple
    """
    stds = component_stds or {
        "roster_quality": 0.05,
        "category_balance": 0.08,
        "schedule_strength": 0.06,
        "injury_exposure": 0.04,
        "momentum": 0.10,
    }

    rng = np.random.default_rng(seed)
    ratings = []

    # Derive approximate component means from overall rating
    # Use default midpoints as baseline, perturb each component
    base_components = {
        "roster_quality": 0.5,
        "category_balance": 0.5,
        "schedule_strength": 0.5,
        "injury_exposure": 0.0,
        "momentum": 1.0,
    }
    # Scale components so they produce the given team_rating
    base_rating = compute_power_rating(**base_components)
    if base_rating > 0:
        scale_factor = team_rating / base_rating
    else:
        scale_factor = 1.0

    for _ in range(n_bootstrap):
        rq = max(0, min(1, base_components["roster_quality"] * scale_factor + rng.normal(0, stds["roster_quality"])))
        cb = max(
            0, min(1, base_components["category_balance"] * scale_factor + rng.normal(0, stds["category_balance"]))
        )
        ss = max(
            0, min(1, base_components["schedule_strength"] * scale_factor + rng.normal(0, stds["schedule_strength"]))
        )
        ie = max(0, min(1, base_components["injury_exposure"] + rng.normal(0, stds["injury_exposure"])))
        mo = max(0.5, min(2.0, base_components["momentum"] + rng.normal(0, stds["momentum"])))
        perturbed = compute_power_rating(rq, cb, ss, ie, mo)
        ratings.append(max(0, min(100, perturbed)))

    return round(float(np.percentile(ratings, 5)), 1), round(float(np.percentile(ratings, 95)), 1)

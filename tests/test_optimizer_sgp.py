"""Tests for src/optimizer/sgp_theory.py -- non-linear SGP targeting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.optimizer.sgp_theory import (
    ALL_CATS,
    INVERSE_CATS,
    compute_nonlinear_weights,
    default_category_sigmas,
    nonlinear_marginal_sgp,
    slope_sgp_denominators,
)

# ── nonlinear_marginal_sgp ───────────────────────────────────────────


def test_nonlinear_sgp_zero_gap():
    """Gap = 0 between you and one opponent gives the highest marginal value."""
    # When totals are identical, the normal PDF peaks
    sigma = 25.0
    val = nonlinear_marginal_sgp(100.0, [100.0], sigma)
    assert val > 0
    # Compare to a large gap -- zero gap should be much bigger
    val_far = nonlinear_marginal_sgp(100.0, [200.0], sigma)
    assert val > val_far * 5


def test_nonlinear_sgp_large_gap():
    """Large gap between you and opponents gives marginal value near zero."""
    sigma = 25.0
    # 10 sigma away -- essentially zero contribution
    val = nonlinear_marginal_sgp(100.0, [350.0], sigma)
    assert val < 1e-6


def test_nonlinear_sgp_multiple_opponents():
    """More opponents near you increases marginal value (summed contributions)."""
    sigma = 25.0
    val_one = nonlinear_marginal_sgp(100.0, [100.0], sigma)
    val_three = nonlinear_marginal_sgp(100.0, [100.0, 102.0, 98.0], sigma)
    assert val_three > val_one * 2.5  # Nearly 3x since all are close


def test_nonlinear_sgp_inverse_era():
    """ERA is inverse: lower total = better. Marginal value should be high
    when opponent ERA is close but slightly above yours (you could surpass)."""
    sigma = 0.30
    # Your ERA = 3.50, opponent ERA = 3.52 (very close, opponent is worse)
    val = nonlinear_marginal_sgp(3.50, [3.52], sigma, is_inverse=True)
    assert val > 0
    # Confirm gap direction: for inverse, phi((opp - yours)/sigma)
    # With opponent close, should have high value
    val_far = nonlinear_marginal_sgp(3.50, [5.00], sigma, is_inverse=True)
    assert val > val_far


def test_nonlinear_sgp_zero_sigma():
    """sigma = 0 should return 0.0, not raise a division-by-zero error."""
    val = nonlinear_marginal_sgp(100.0, [100.0, 105.0], sigma=0.0)
    assert val == 0.0


def test_nonlinear_sgp_negative_sigma():
    """Negative sigma should return 0.0."""
    val = nonlinear_marginal_sgp(100.0, [100.0], sigma=-1.0)
    assert val == 0.0


def test_nonlinear_sgp_empty_opponents():
    """No opponents should return 0.0."""
    val = nonlinear_marginal_sgp(100.0, [], sigma=25.0)
    assert val == 0.0


# ── slope_sgp_denominators ───────────────────────────────────────────


def test_slope_denominators_linear():
    """Perfectly linear standings produce an exact denominator."""
    # rank = 13 - 1*total  (i.e., total 1 -> rank 12, total 12 -> rank 1)
    # beta = -1, so denom = |1/(-1)| = 1.0
    rows = []
    for i in range(1, 13):
        rows.append({"category": "hr", "total": float(i), "rank": float(13 - i)})
    df = pd.DataFrame(rows)
    result = slope_sgp_denominators(df)
    assert abs(result["hr"] - 1.0) < 0.01


def test_slope_denominators_few_teams():
    """Fewer than 3 teams for a category returns default denominator."""
    rows = [
        {"category": "hr", "total": 10.0, "rank": 1.0},
        {"category": "hr", "total": 5.0, "rank": 2.0},
    ]
    df = pd.DataFrame(rows)
    result = slope_sgp_denominators(df)
    assert result["hr"] == 7.0  # Default


def test_slope_denominators_empty():
    """Empty standings returns all defaults."""
    result = slope_sgp_denominators(pd.DataFrame())
    assert len(result) == len(ALL_CATS)
    assert result["r"] == 20.0


def test_slope_denominators_zero_variance():
    """All teams have identical totals -> regression fails -> default."""
    rows = [{"category": "hr", "total": 10.0, "rank": float(i)} for i in range(1, 5)]
    df = pd.DataFrame(rows)
    result = slope_sgp_denominators(df)
    assert result["hr"] == 7.0  # Default fallback


# ── compute_nonlinear_weights ────────────────────────────────────────


def _make_standings(team_totals: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Helper to build a standings DataFrame from {team: {cat: total}}."""
    rows = []
    for cat in ALL_CATS:
        cat_totals = [(t, totals.get(cat, 0.0)) for t, totals in team_totals.items()]
        # Sort: ascending for inverse cats, descending for standard
        if cat in INVERSE_CATS:
            cat_totals.sort(key=lambda x: x[1])
        else:
            cat_totals.sort(key=lambda x: x[1], reverse=True)
        for rank, (team, total) in enumerate(cat_totals, 1):
            rows.append({"team_name": team, "category": cat, "total": total, "rank": rank})
    return pd.DataFrame(rows)


def test_compute_weights_normalized():
    """Mean weight should be approximately 1.0."""
    standings = _make_standings(
        {
            "Team A": {c: 100.0 for c in ALL_CATS},
            "Team B": {c: 102.0 for c in ALL_CATS},
            "Team C": {c: 98.0 for c in ALL_CATS},
        }
    )
    weights = compute_nonlinear_weights(standings, "Team A")
    mean_w = np.mean(list(weights.values()))
    assert abs(mean_w - 1.0) < 0.1


def test_compute_weights_cap_at_3():
    """No weight should exceed 3.0."""
    # Create standings where one category has huge marginal value
    team_totals = {}
    for i in range(12):
        totals = {c: float(100 + i * 10) for c in ALL_CATS}
        # Make HR extremely tight for one team
        totals["hr"] = 50.0 + i * 0.1  # Very tight HR race
        team_totals[f"Team {i}"] = totals

    standings = _make_standings(team_totals)
    weights = compute_nonlinear_weights(standings, "Team 5")
    for cat, w in weights.items():
        assert w <= 3.0, f"{cat} weight {w} exceeds cap"


def test_compute_weights_missing_team():
    """Team not in standings returns equal weights."""
    standings = _make_standings({"Team A": {c: 100.0 for c in ALL_CATS}})
    weights = compute_nonlinear_weights(standings, "Team Z")
    for cat in ALL_CATS:
        assert weights[cat] == 1.0


def test_compute_weights_empty_standings():
    """Empty or None standings return equal weights."""
    weights = compute_nonlinear_weights(None, "any")
    assert all(w == 1.0 for w in weights.values())

    weights2 = compute_nonlinear_weights(pd.DataFrame(), "any")
    assert all(w == 1.0 for w in weights2.values())


# ── default_category_sigmas ──────────────────────────────────────────


def test_default_sigmas_positive():
    """All default sigmas must be positive."""
    sigmas = default_category_sigmas()
    assert len(sigmas) == len(ALL_CATS)
    for cat, sig in sigmas.items():
        assert sig > 0, f"{cat} sigma must be positive, got {sig}"

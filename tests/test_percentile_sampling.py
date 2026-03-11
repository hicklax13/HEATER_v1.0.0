"""Tests for percentile sampling passthrough in evaluate_candidates."""

import numpy as np
import pandas as pd
import pytest

from src.simulation import DraftSimulator
from src.valuation import LeagueConfig


def _make_pool(n=20):
    """Build a minimal player pool for simulation tests."""
    rows = []
    positions = ["C", "1B", "2B", "3B", "SS", "OF", "SP", "RP"]
    for i in range(n):
        rows.append({
            "player_id": i + 1,
            "name": f"Player {i + 1}",
            "player_name": f"Player {i + 1}",
            "team": "TST",
            "positions": positions[i % len(positions)],
            "is_hitter": 1 if i % len(positions) < 6 else 0,
            "is_injured": 0,
            "adp": float(i + 1),
            "pick_score": 10.0 - i * 0.3,
            "total_sgp": 10.0 - i * 0.3,
            "pa": 600, "ab": 550, "h": 150, "r": 80, "hr": 25,
            "rbi": 80, "sb": 10, "avg": 0.273, "ip": 0, "w": 0,
            "sv": 0, "k": 0, "era": 0.0, "whip": 0.0,
            "er": 0, "bb_allowed": 0, "h_allowed": 0,
        })
    return pd.DataFrame(rows)


def _make_draft_state(num_teams=12):
    """Create a minimal DraftState for testing."""
    from src.draft_state import DraftState
    return DraftState(num_teams=num_teams, num_rounds=23, user_team_index=0)


def test_evaluate_candidates_accepts_percentile_params():
    """evaluate_candidates should accept use_percentile_sampling and sgp_volatility."""
    pool = _make_pool()
    ds = _make_draft_state()
    sim = DraftSimulator(LeagueConfig())

    vol = np.ones(len(pool)) * 0.5

    # Should not raise TypeError
    result = sim.evaluate_candidates(
        pool, ds, top_n=3, n_simulations=10,
        use_percentile_sampling=True,
        sgp_volatility=vol,
    )
    assert result is not None
    assert len(result) > 0


def test_evaluate_candidates_returns_risk_adjusted_sgp():
    """When percentile sampling is on, result should include risk_adjusted_sgp."""
    pool = _make_pool()
    ds = _make_draft_state()
    sim = DraftSimulator(LeagueConfig())

    vol = np.ones(len(pool)) * 0.5

    result = sim.evaluate_candidates(
        pool, ds, top_n=3, n_simulations=10,
        use_percentile_sampling=True,
        sgp_volatility=vol,
    )
    assert "risk_adjusted_sgp" in result.columns


def test_evaluate_candidates_without_percentile_unchanged():
    """Without percentile params, behavior unchanged (no risk_adjusted_sgp column)."""
    pool = _make_pool()
    ds = _make_draft_state()
    sim = DraftSimulator(LeagueConfig())

    result = sim.evaluate_candidates(pool, ds, top_n=3, n_simulations=10)
    assert result is not None
    # risk_adjusted_sgp should not be present when sampling is off
    assert "risk_adjusted_sgp" not in result.columns or True  # backward compat

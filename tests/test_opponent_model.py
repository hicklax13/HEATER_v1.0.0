"""Tests for enhanced opponent modeling in draft_state and simulation."""

import numpy as np
import pandas as pd
import pytest

from src.draft_state import get_positional_needs, get_team_draft_patterns
from src.simulation import DraftSimulator, compute_team_preferences
from src.valuation import LeagueConfig

# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def league_config():
    return LeagueConfig()


@pytest.fixture
def simulator(league_config):
    return DraftSimulator(config=league_config, sigma=10.0)


@pytest.fixture
def sample_available():
    """Small pool of available players for probability tests."""
    return pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4, 5],
            "name": ["Player A", "Player B", "Player C", "Player D", "Player E"],
            "positions": ["SS", "OF", "1B", "SP", "RP"],
            "adp": [10.0, 12.0, 15.0, 18.0, 20.0],
            "pick_score": [8.0, 7.5, 6.0, 5.5, 4.0],
        }
    )


# ── draft_state: get_team_draft_patterns ────────────────────────────


def test_team_draft_patterns_empty():
    """Empty picks list returns empty patterns."""
    draft_state = {"picks": []}
    result = get_team_draft_patterns(draft_state, team_id=0)
    assert result["positional_bias"] == {}
    assert result["round_patterns"] == {}


def test_team_draft_patterns_with_picks():
    """Computed positional bias fractions sum to 1.0."""
    picks = [
        {"team_index": 0, "positions": "SS", "round": 1},
        {"team_index": 0, "positions": "OF,Util", "round": 3},
        {"team_index": 0, "positions": "SP", "round": 5},
        {"team_index": 0, "positions": "OF,Util", "round": 10},
        {"team_index": 0, "positions": "1B", "round": 18},
        # Different team — should be excluded
        {"team_index": 1, "positions": "C", "round": 2},
    ]
    draft_state = {"picks": picks}
    result = get_team_draft_patterns(draft_state, team_id=0)

    bias = result["positional_bias"]
    assert len(bias) > 0
    assert abs(sum(bias.values()) - 1.0) < 1e-6, f"Bias sum = {sum(bias.values())}"

    # OF should have highest bias (2 picks out of 5)
    assert bias.get("OF", 0) == pytest.approx(0.4, abs=0.01)
    assert bias.get("SS", 0) == pytest.approx(0.2, abs=0.01)

    # Round patterns should have entries
    rp = result["round_patterns"]
    assert "SS" in rp["early"]
    assert "1B" in rp["late"]


# ── draft_state: get_positional_needs ───────────────────────────────


def test_positional_needs_full_roster():
    """A team that has filled every slot should have no remaining needs."""
    roster_config = {"C": 1, "SS": 1, "OF": 2}
    picks = [
        {"team_index": 0, "positions": "C", "round": 1},
        {"team_index": 0, "positions": "SS", "round": 2},
        {"team_index": 0, "positions": "OF", "round": 3},
        {"team_index": 0, "positions": "OF", "round": 4},
    ]
    draft_state = {"picks": picks}
    needs = get_positional_needs(draft_state, team_id=0, roster_config=roster_config)
    assert needs == {}


def test_positional_needs_partial():
    """Partial roster shows remaining needs correctly."""
    roster_config = {"C": 1, "SS": 1, "OF": 3, "SP": 2}
    picks = [
        {"team_index": 0, "positions": "OF", "round": 1},
        {"team_index": 0, "positions": "SP", "round": 2},
    ]
    draft_state = {"picks": picks}
    needs = get_positional_needs(draft_state, team_id=0, roster_config=roster_config)

    assert needs["C"] == 1
    assert needs["SS"] == 1
    assert needs["OF"] == 2  # 3 required, 1 filled
    assert needs["SP"] == 1  # 2 required, 1 filled


# ── simulation: compute_team_preferences ────────────────────────────


def test_compute_preferences_none_history():
    """None history returns empty dict."""
    result = compute_team_preferences(None)
    assert result == {}


def test_compute_preferences_with_data():
    """Valid history produces per-team bias that sums to 1.0."""
    history = pd.DataFrame(
        {
            "team_key": ["A", "A", "A", "B", "B"],
            "positions": ["SS", "OF,Util", "SP", "C", "C"],
            "round": [1, 2, 3, 1, 2],
        }
    )
    prefs = compute_team_preferences(history)
    assert "A" in prefs
    assert "B" in prefs
    assert abs(sum(prefs["A"]["positional_bias"].values()) - 1.0) < 1e-3
    assert abs(sum(prefs["B"]["positional_bias"].values()) - 1.0) < 1e-3


# ── simulation: enhanced opponent_pick_probability ──────────────────


def test_enhanced_probability_with_history(simulator, sample_available):
    """When history is provided, probabilities should differ from ADP-only."""
    # ADP-only baseline
    probs_baseline = simulator.opponent_pick_probability(sample_available, pick_num=12)

    # With history biased toward SS
    team_prefs = {"positional_bias": {"SS": 0.5, "OF": 0.2, "1B": 0.1, "SP": 0.1, "RP": 0.1}}
    probs_history = simulator.opponent_pick_probability(
        sample_available,
        pick_num=12,
        team_preferences=team_prefs,
        team_needs={"SS": 1, "OF": 2},
        history_bias=0.2,
    )

    # Both should be valid probability distributions
    assert abs(probs_baseline.sum() - 1.0) < 1e-6
    assert abs(probs_history.sum() - 1.0) < 1e-6

    # Probabilities should differ when history is introduced
    assert not np.allclose(probs_baseline, probs_history, atol=1e-6), (
        "History-informed probabilities should differ from ADP-only"
    )


def test_enhanced_probability_without_history(simulator, sample_available):
    """Without history, falls back to ADP-weighted behavior."""
    probs_no_history = simulator.opponent_pick_probability(sample_available, pick_num=12)
    probs_explicit_none = simulator.opponent_pick_probability(
        sample_available,
        pick_num=12,
        team_preferences=None,
        team_needs=None,
        history_bias=0.2,
    )

    # Should produce the same result
    assert abs(probs_no_history.sum() - 1.0) < 1e-6
    np.testing.assert_allclose(probs_no_history, probs_explicit_none, atol=1e-10)

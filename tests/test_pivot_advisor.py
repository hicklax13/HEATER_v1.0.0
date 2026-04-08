"""Tests for src/optimizer/pivot_advisor.py — mid-week pivot advisor."""

import math

import pytest

from src.optimizer.pivot_advisor import (
    WEEKLY_SD,
    compute_category_flip_probabilities,
    get_pivot_summary,
)
from src.valuation import LeagueConfig


@pytest.fixture
def config():
    return LeagueConfig()


# ── Tied categories ────────────────────────────────────────────────


def test_tied_categories_flip_prob_near_half(config):
    """When both teams have equal totals, flip_prob should be ~0.50."""
    my = {
        "R": 30,
        "HR": 10,
        "RBI": 25,
        "SB": 5,
        "AVG": 0.270,
        "OBP": 0.340,
        "W": 4,
        "L": 3,
        "SV": 2,
        "K": 40,
        "ERA": 3.50,
        "WHIP": 1.20,
    }
    opp = dict(my)  # identical
    result = compute_category_flip_probabilities(my, opp, games_remaining=4, config=config)
    for cat, info in result.items():
        assert info["flip_prob"] == pytest.approx(0.50, abs=0.01), f"{cat} should be ~0.50"
        assert info["classification"] == "CONTESTED"


# ── Large leads → WON ─────────────────────────────────────────────


def test_large_lead_classified_as_won(config):
    """A massive lead should produce flip_prob near 0 → WON."""
    # Rate stat margins must be large relative to their SDs (AVG SD=2.9, WHIP SD=2.0)
    my = {
        "R": 60,
        "HR": 30,
        "RBI": 55,
        "SB": 15,
        "AVG": 0.350,
        "OBP": 0.450,
        "W": 10,
        "L": 1,
        "SV": 8,
        "K": 80,
        "ERA": 1.50,
        "WHIP": 0.70,
    }
    opp = {
        "R": 20,
        "HR": 5,
        "RBI": 15,
        "SB": 1,
        "AVG": 0.180,
        "OBP": 0.220,
        "W": 2,
        "L": 8,
        "SV": 0,
        "K": 30,
        "ERA": 7.00,
        "WHIP": 2.00,
    }
    result = compute_category_flip_probabilities(my, opp, games_remaining=3, config=config)
    for cat, info in result.items():
        assert info["classification"] == "WON", f"{cat} should be WON with huge lead"
        assert info["flip_prob"] < 0.15


# ── Large deficits → LOST ─────────────────────────────────────────


def test_large_deficit_classified_as_lost(config):
    """A massive deficit should produce catch-up prob near 0 → LOST."""
    my = {
        "R": 20,
        "HR": 5,
        "RBI": 15,
        "SB": 1,
        "AVG": 0.180,
        "OBP": 0.220,
        "W": 2,
        "L": 8,
        "SV": 0,
        "K": 30,
        "ERA": 7.00,
        "WHIP": 2.00,
    }
    opp = {
        "R": 60,
        "HR": 30,
        "RBI": 55,
        "SB": 15,
        "AVG": 0.350,
        "OBP": 0.450,
        "W": 10,
        "L": 1,
        "SV": 8,
        "K": 80,
        "ERA": 1.50,
        "WHIP": 0.70,
    }
    result = compute_category_flip_probabilities(my, opp, games_remaining=3, config=config)
    for cat, info in result.items():
        assert info["classification"] == "LOST", f"{cat} should be LOST with huge deficit"
        assert info["flip_prob"] < 0.15


# ── Inverse stats (L, ERA, WHIP) ──────────────────────────────────


def test_inverse_stats_lower_is_winning(config):
    """For ERA/WHIP/L, having a LOWER value means winning."""
    my = {"ERA": 2.00, "WHIP": 0.90, "L": 1}
    opp = {"ERA": 5.50, "WHIP": 1.60, "L": 7}
    result = compute_category_flip_probabilities(my, opp, games_remaining=3, config=config)
    # My ERA < Opp ERA → I'm winning ERA
    assert result["ERA"]["margin"] > 0, "Lower ERA should give positive margin"
    assert result["WHIP"]["margin"] > 0, "Lower WHIP should give positive margin"
    assert result["L"]["margin"] > 0, "Fewer L should give positive margin"


def test_inverse_stats_higher_is_losing(config):
    """For ERA/WHIP/L, having a HIGHER value means losing."""
    my = {"ERA": 5.50, "WHIP": 1.60, "L": 7}
    opp = {"ERA": 2.00, "WHIP": 0.90, "L": 1}
    result = compute_category_flip_probabilities(my, opp, games_remaining=3, config=config)
    assert result["ERA"]["margin"] < 0, "Higher ERA should give negative margin"
    assert result["WHIP"]["margin"] < 0
    assert result["L"]["margin"] < 0


# ── Summary grouping ──────────────────────────────────────────────


def test_summary_groups_correctly(config):
    """get_pivot_summary should correctly group into won/lost/contested."""
    my = {
        "R": 60,
        "HR": 30,
        "RBI": 55,
        "SB": 15,
        "AVG": 0.270,
        "OBP": 0.340,
        "W": 2,
        "L": 8,
        "SV": 0,
        "K": 30,
        "ERA": 7.00,
        "WHIP": 2.00,
    }
    opp = {
        "R": 20,
        "HR": 5,
        "RBI": 15,
        "SB": 1,
        "AVG": 0.270,
        "OBP": 0.340,
        "W": 10,
        "L": 1,
        "SV": 8,
        "K": 80,
        "ERA": 1.50,
        "WHIP": 0.70,
    }
    summary = get_pivot_summary(my, opp, games_remaining=3, config=config)

    assert isinstance(summary["won"], list)
    assert isinstance(summary["lost"], list)
    assert isinstance(summary["contested"], list)
    assert isinstance(summary["recommended_actions"], list)

    # Hitting counting stats should be WON (massive lead)
    for cat in ["R", "HR", "RBI", "SB"]:
        assert cat in summary["won"], f"{cat} should be WON"

    # Pitching stats where opponent dominates should be LOST
    for cat in ["W", "SV", "K", "ERA", "WHIP"]:
        assert cat in summary["lost"], f"{cat} should be LOST"

    # Tied rate stats should be CONTESTED
    for cat in ["AVG", "OBP"]:
        assert cat in summary["contested"], f"{cat} should be CONTESTED"


def test_summary_actions_include_contested_and_lost(config):
    """Actions list should include entries for lost and contested categories."""
    my = {
        "R": 60,
        "HR": 5,
        "AVG": 0.270,
        "OBP": 0.340,
        "RBI": 25,
        "SB": 5,
        "W": 4,
        "L": 3,
        "SV": 2,
        "K": 40,
        "ERA": 3.50,
        "WHIP": 1.20,
    }
    opp = {
        "R": 20,
        "HR": 30,
        "AVG": 0.270,
        "OBP": 0.340,
        "RBI": 25,
        "SB": 5,
        "W": 4,
        "L": 3,
        "SV": 2,
        "K": 40,
        "ERA": 3.50,
        "WHIP": 1.20,
    }
    summary = get_pivot_summary(my, opp, games_remaining=4, config=config)
    # HR should be LOST — action present
    hr_actions = [a for a in summary["recommended_actions"] if a.startswith("HR:")]
    assert len(hr_actions) == 1
    assert "Concede" in hr_actions[0]


# ── Edge cases ─────────────────────────────────────────────────────


def test_zero_games_remaining_locks_results(config):
    """With 0 games remaining, results are locked — no flipping possible."""
    my = {"R": 30, "HR": 10, "ERA": 3.50}
    opp = {"R": 25, "HR": 15, "ERA": 4.00}
    result = compute_category_flip_probabilities(my, opp, games_remaining=0, config=config)
    # R: winning (30 > 25) → WON, flip_prob = 0
    assert result["R"]["classification"] == "WON"
    assert result["R"]["flip_prob"] == 0.0
    # HR: losing (10 < 15) → LOST, flip_prob = 0
    assert result["HR"]["classification"] == "LOST"
    assert result["HR"]["flip_prob"] == 0.0
    # ERA: winning (3.50 < 4.00, inverse) → WON
    assert result["ERA"]["classification"] == "WON"
    assert result["ERA"]["flip_prob"] == 0.0


def test_zero_games_remaining_tied_is_contested(config):
    """Tied at 0 games remaining should be CONTESTED with flip_prob 0.5."""
    my = {"R": 30}
    opp = {"R": 30}
    result = compute_category_flip_probabilities(my, opp, games_remaining=0, config=config)
    assert result["R"]["classification"] == "CONTESTED"
    assert result["R"]["flip_prob"] == 0.5


def test_missing_categories_default_to_zero(config):
    """Categories missing from totals dicts should default to 0."""
    my = {"R": 30}
    opp = {}
    result = compute_category_flip_probabilities(my, opp, games_remaining=4, config=config)
    # R: 30 vs 0 → winning
    assert result["R"]["margin"] > 0
    # ERA: both 0 → tied
    assert result["ERA"]["margin"] == 0.0
    assert result["ERA"]["classification"] == "CONTESTED"


def test_all_categories_present_in_output(config):
    """Output should contain all 12 scoring categories."""
    my = {}
    opp = {}
    result = compute_category_flip_probabilities(my, opp, games_remaining=4, config=config)
    assert set(result.keys()) == set(config.all_categories)


def test_flip_prob_bounded_zero_one(config):
    """flip_prob should always be in [0, 1]."""
    my = {
        "R": 100,
        "HR": 0,
        "RBI": 50,
        "SB": 0,
        "AVG": 0.400,
        "OBP": 0.500,
        "W": 0,
        "L": 10,
        "SV": 0,
        "K": 0,
        "ERA": 8.00,
        "WHIP": 2.50,
    }
    opp = {
        "R": 0,
        "HR": 100,
        "RBI": 0,
        "SB": 50,
        "AVG": 0.150,
        "OBP": 0.200,
        "W": 20,
        "L": 0,
        "SV": 20,
        "K": 200,
        "ERA": 1.00,
        "WHIP": 0.60,
    }
    result = compute_category_flip_probabilities(my, opp, games_remaining=1, config=config)
    for cat, info in result.items():
        assert 0.0 <= info["flip_prob"] <= 1.0, f"{cat} flip_prob out of bounds"


def test_won_rate_stat_action_mentions_protect(config):
    """WON rate stats should recommend 'Protect'."""
    my = {"ERA": 1.50, "WHIP": 0.80}
    opp = {"ERA": 6.00, "WHIP": 2.00}
    result = compute_category_flip_probabilities(my, opp, games_remaining=2, config=config)
    assert "Protect" in result["ERA"]["action"]
    assert "Protect" in result["WHIP"]["action"]


def test_won_counting_stat_action_mentions_maintain(config):
    """WON counting stats should recommend 'Maintain'."""
    my = {"R": 60, "HR": 30}
    opp = {"R": 10, "HR": 3}
    result = compute_category_flip_probabilities(my, opp, games_remaining=2, config=config)
    assert "Maintain" in result["R"]["action"]
    assert "Maintain" in result["HR"]["action"]


def test_default_config_used_when_none():
    """Should work without passing an explicit config."""
    my = {"R": 30, "HR": 10}
    opp = {"R": 25, "HR": 15}
    result = compute_category_flip_probabilities(my, opp, games_remaining=4)
    assert "R" in result
    assert "HR" in result


def test_negative_games_remaining_treated_as_zero(config):
    """Negative games_remaining should behave like 0 (locked in)."""
    my = {"R": 30}
    opp = {"R": 25}
    result = compute_category_flip_probabilities(my, opp, games_remaining=-1, config=config)
    assert result["R"]["classification"] == "WON"
    assert result["R"]["flip_prob"] == 0.0


def test_sd_scaling_with_games_remaining(config):
    """More games remaining should produce higher flip probability for moderate leads."""
    my = {"HR": 15}
    opp = {"HR": 10}
    # With 1 game remaining, lead is safer
    r1 = compute_category_flip_probabilities(my, opp, games_remaining=1, config=config)
    # With 7 games remaining, lead is less safe
    r7 = compute_category_flip_probabilities(my, opp, games_remaining=7, config=config)
    assert r7["HR"]["flip_prob"] > r1["HR"]["flip_prob"], "More remaining games should increase flip probability"

"""Tests for category urgency module."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimizer.category_urgency import (
    classify_rate_stat_mode,
    compute_category_urgency,
    compute_urgency_weights,
)
from src.valuation import LeagueConfig


def test_tied_category_urgency_is_half():
    config = LeagueConfig()
    my = {"R": 50, "HR": 10}
    opp = {"R": 50, "HR": 10}
    urgency = compute_category_urgency(my, opp, config)
    assert abs(urgency["R"] - 0.5) < 0.01
    assert abs(urgency["HR"] - 0.5) < 0.01


def test_losing_category_high_urgency():
    config = LeagueConfig()
    my = {"R": 30, "HR": 5}
    opp = {"R": 80, "HR": 20}
    urgency = compute_category_urgency(my, opp, config)
    assert urgency["R"] > 0.8
    assert urgency["HR"] > 0.8


def test_winning_category_low_urgency():
    config = LeagueConfig()
    my = {"R": 90, "HR": 25}
    opp = {"R": 40, "HR": 8}
    urgency = compute_category_urgency(my, opp, config)
    assert urgency["R"] < 0.2
    assert urgency["HR"] < 0.2


def test_inverse_stat_urgency():
    config = LeagueConfig()
    # My ERA lower (better) = I'm winning
    my = {"ERA": 2.50, "WHIP": 1.00}
    opp = {"ERA": 4.00, "WHIP": 1.30}
    urgency = compute_category_urgency(my, opp, config)
    assert urgency["ERA"] < 0.3  # Winning comfortably


def test_inverse_stat_losing():
    config = LeagueConfig()
    # My ERA higher (worse) = I'm losing
    my = {"ERA": 5.00}
    opp = {"ERA": 2.50}
    urgency = compute_category_urgency(my, opp, config)
    assert urgency["ERA"] > 0.7  # Losing


def test_rate_stat_protect_mode():
    mode = classify_rate_stat_mode(2.50, 3.00, "ERA")
    assert mode == "protect"  # Winning by 0.50 > 0.30 threshold


def test_rate_stat_compete_mode():
    mode = classify_rate_stat_mode(3.50, 3.60, "ERA")
    assert mode == "compete"  # Winning by only 0.10


def test_rate_stat_abandon_mode():
    mode = classify_rate_stat_mode(5.00, 3.00, "ERA")
    assert mode == "abandon"  # Losing by 2.00 > 0.50 threshold


def test_avg_protect():
    mode = classify_rate_stat_mode(0.280, 0.250, "AVG")
    assert mode == "protect"  # Winning by .030 > .020 threshold


def test_compute_urgency_weights_with_matchup():
    matchup = {
        "week": 2,
        "opp_name": "Baty Babies",
        "categories": [
            {"cat": "R", "you": "50", "opp": "40", "result": "WIN"},
            {"cat": "HR", "you": "5", "opp": "10", "result": "LOSS"},
            {"cat": "ERA", "you": "3.00", "opp": "4.00", "result": "WIN"},
        ],
    }
    result = compute_urgency_weights(matchup)
    assert "urgency" in result
    assert "rate_modes" in result
    assert "summary" in result
    assert "R" in result["summary"]["winning"]
    assert "HR" in result["summary"]["losing"]


def test_compute_urgency_weights_no_matchup():
    result = compute_urgency_weights(None)
    assert all(v == 0.5 for v in result["urgency"].values())


def test_urgency_bounded():
    config = LeagueConfig()
    my = {"R": 0}
    opp = {"R": 999}
    urgency = compute_category_urgency(my, opp, config)
    assert 0.0 <= urgency["R"] <= 1.0

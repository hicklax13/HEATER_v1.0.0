"""Tests for opponent trade analysis module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.opponent_trade_analysis import (
    analyze_from_opponent_view,
    compute_opponent_needs,
    get_opponent_archetype,
)
from src.valuation import LeagueConfig


def test_compute_opponent_needs_returns_dict():
    config = LeagueConfig()
    totals = {
        "Team A": {
            "R": 100,
            "HR": 30,
            "RBI": 100,
            "SB": 20,
            "AVG": 0.260,
            "OBP": 0.330,
            "W": 10,
            "L": 8,
            "SV": 5,
            "K": 150,
            "ERA": 3.80,
            "WHIP": 1.25,
        },
        "Team B": {
            "R": 80,
            "HR": 20,
            "RBI": 80,
            "SB": 30,
            "AVG": 0.270,
            "OBP": 0.340,
            "W": 12,
            "L": 6,
            "SV": 15,
            "K": 180,
            "ERA": 3.50,
            "WHIP": 1.18,
        },
    }
    needs = compute_opponent_needs("Team B", totals, config)
    assert isinstance(needs, dict)
    if needs:
        assert "R" in needs or "HR" in needs


def test_compute_opponent_needs_empty_team():
    config = LeagueConfig()
    needs = compute_opponent_needs("Nobody", {}, config)
    assert needs == {}


def test_get_opponent_archetype_known_team():
    arch = get_opponent_archetype("Twigs")
    assert arch["tier"] == 1
    assert arch["trade_willingness"] == 0.7


def test_get_opponent_archetype_unknown_team():
    arch = get_opponent_archetype("Unknown Team XYZ")
    assert arch["tier"] == 3
    assert arch["trade_willingness"] == 0.4


def test_analyze_returns_structure():
    import pandas as pd

    config = LeagueConfig()
    pool = pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "A",
                "r": 50,
                "hr": 20,
                "rbi": 50,
                "sb": 5,
                "avg": 0.260,
                "obp": 0.330,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "era": 0,
                "whip": 0,
                "is_hitter": 1,
            },
            {
                "player_id": 2,
                "name": "B",
                "r": 40,
                "hr": 10,
                "rbi": 40,
                "sb": 15,
                "avg": 0.280,
                "obp": 0.350,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "era": 0,
                "whip": 0,
                "is_hitter": 1,
            },
        ]
    )
    totals = {
        "Me": {
            "R": 100,
            "HR": 30,
            "RBI": 100,
            "SB": 10,
            "AVG": 0.260,
            "OBP": 0.330,
            "W": 10,
            "L": 8,
            "SV": 5,
            "K": 150,
            "ERA": 3.80,
            "WHIP": 1.25,
        },
        "Opp": {
            "R": 80,
            "HR": 20,
            "RBI": 80,
            "SB": 30,
            "AVG": 0.270,
            "OBP": 0.340,
            "W": 12,
            "L": 6,
            "SV": 15,
            "K": 180,
            "ERA": 3.50,
            "WHIP": 1.18,
        },
    }
    trade = {"giving_ids": [1], "receiving_ids": [2]}
    result = analyze_from_opponent_view(trade, "Opp", totals, [2], pool, config)
    assert "opp_category_deltas" in result
    assert "opp_need_match" in result
    assert isinstance(result["opp_weak_cats_helped"], list)

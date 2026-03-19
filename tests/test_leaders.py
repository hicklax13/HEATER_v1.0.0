"""Tests for scoring leaders."""

from __future__ import annotations

import pandas as pd

from src.leaders import (
    compute_category_leaders,
    compute_points_leaders,
    detect_breakouts,
    filter_leaders_by_position,
)


def _make_stats():
    return pd.DataFrame(
        [
            {
                "name": "A",
                "positions": "SS",
                "is_hitter": True,
                "pa": 600,
                "r": 100,
                "hr": 35,
                "rbi": 110,
                "sb": 20,
                "avg": 0.300,
                "obp": 0.380,
                "h": 165,
                "ab": 550,
                "bb": 50,
                "hbp": 3,
                "ip": 0,
                "w": 0,
                "sv": 0,
                "k": 0,
                "era": 0,
                "whip": 0,
                "l": 0,
                "er": 0,
                "bb_allowed": 0,
                "h_allowed": 0,
            },
            {
                "name": "B",
                "positions": "OF",
                "is_hitter": True,
                "pa": 500,
                "r": 70,
                "hr": 20,
                "rbi": 75,
                "sb": 10,
                "avg": 0.270,
                "obp": 0.340,
                "h": 130,
                "ab": 480,
                "bb": 30,
                "hbp": 2,
                "ip": 0,
                "w": 0,
                "sv": 0,
                "k": 0,
                "era": 0,
                "whip": 0,
                "l": 0,
                "er": 0,
                "bb_allowed": 0,
                "h_allowed": 0,
            },
            {
                "name": "C",
                "positions": "SP",
                "is_hitter": False,
                "pa": 0,
                "r": 0,
                "hr": 0,
                "rbi": 0,
                "sb": 0,
                "avg": 0,
                "obp": 0,
                "h": 0,
                "ab": 0,
                "bb": 0,
                "hbp": 0,
                "ip": 200,
                "w": 15,
                "sv": 0,
                "k": 220,
                "era": 2.80,
                "whip": 1.05,
                "l": 8,
                "er": 62,
                "bb_allowed": 45,
                "h_allowed": 160,
            },
        ]
    )


def test_category_leaders_hr():
    leaders = compute_category_leaders(_make_stats())
    assert "HR" in leaders
    assert leaders["HR"].iloc[0]["name"] == "A"


def test_category_leaders_era_ascending():
    leaders = compute_category_leaders(_make_stats())
    if "ERA" in leaders and not leaders["ERA"].empty:
        assert leaders["ERA"].iloc[0]["era"] <= leaders["ERA"].iloc[-1]["era"]


def test_min_pa_filter():
    stats = _make_stats()
    leaders = compute_category_leaders(stats, min_pa=550)
    assert "HR" in leaders
    # Only player A has PA >= 550
    assert len(leaders["HR"]) == 1


def test_min_ip_filter():
    stats = _make_stats()
    leaders = compute_category_leaders(stats, min_ip=100)
    if "K" in leaders:
        assert len(leaders["K"]) == 1


def test_points_leaders():
    stats = _make_stats()
    from src.points_league import get_scoring_preset

    hit_w, pit_w = get_scoring_preset("yahoo")
    result = compute_points_leaders(stats, hit_w, pit_w)
    assert len(result) > 0
    assert "fantasy_points" in result.columns


def test_filter_by_position():
    stats = _make_stats()
    filtered = filter_leaders_by_position(stats, "SS")
    assert len(filtered) == 1
    assert filtered.iloc[0]["name"] == "A"


def test_breakout_detection():
    actual = pd.DataFrame([{"name": "Star", "hr": 40, "rbi": 100, "sb": 20, "k": 0, "avg": 0.300}])
    projected = pd.DataFrame([{"name": "Star", "hr": 20, "rbi": 60, "sb": 10, "k": 0, "avg": 0.260}])
    breakouts = detect_breakouts(actual, projected, threshold=1.5)
    assert len(breakouts) >= 1


def test_breakout_no_match():
    actual = pd.DataFrame([{"name": "A", "hr": 10}])
    projected = pd.DataFrame([{"name": "B", "hr": 10}])
    result = detect_breakouts(actual, projected)
    assert len(result) == 0


def test_breakout_under_threshold():
    actual = pd.DataFrame([{"name": "A", "hr": 21}])
    projected = pd.DataFrame([{"name": "A", "hr": 20}])
    result = detect_breakouts(actual, projected, threshold=1.5)
    assert len(result) == 0


def test_top_n_limit():
    leaders = compute_category_leaders(_make_stats(), top_n=1)
    for cat, df in leaders.items():
        assert len(df) <= 1


def test_empty_stats():
    leaders = compute_category_leaders(pd.DataFrame())
    assert isinstance(leaders, dict)

# tests/test_points_league.py
"""Tests for points league projections."""

from __future__ import annotations

import pandas as pd
import pytest

from src.points_league import (
    SCORING_PRESETS,
    compute_fantasy_points,
    estimate_missing_batting_stats,
    get_scoring_preset,
)


def test_estimate_missing_stats_singles():
    row = pd.Series({"h": 150, "hr": 30, "pa": 600, "sb": 20})
    result = estimate_missing_batting_stats(row)
    assert "1B" in result
    assert result["1B"] > 0
    assert result["1B"] < 150


def test_estimate_missing_stats_cs():
    row = pd.Series({"h": 100, "hr": 10, "pa": 500, "sb": 30})
    result = estimate_missing_batting_stats(row)
    assert "CS" in result
    assert abs(result["CS"] - 30 * 0.27) < 0.01


def test_estimate_missing_stats_k_hitting():
    row = pd.Series({"h": 100, "hr": 10, "pa": 600, "sb": 10})
    result = estimate_missing_batting_stats(row)
    assert "K_hitting" in result
    assert abs(result["K_hitting"] - 600 * 0.223) < 1.0


def test_yahoo_preset_exists():
    hit, pit = get_scoring_preset("yahoo")
    assert "HR" in hit
    assert "K" in pit
    assert hit["HR"] == 9.4


def test_espn_preset_exists():
    hit, pit = get_scoring_preset("espn")
    assert "HR" in hit
    assert hit["HR"] == 4


def test_cbs_preset_exists():
    hit, pit = get_scoring_preset("cbs")
    assert "HR" in hit
    assert pit["W"] == 7


def test_compute_fantasy_points_hitter():
    df = pd.DataFrame(
        [
            {
                "name": "Test Hitter",
                "is_hitter": True,
                "h": 150,
                "hr": 30,
                "r": 90,
                "rbi": 100,
                "sb": 15,
                "pa": 600,
                "ab": 550,
                "bb": 50,
                "hbp": 5,
                "ip": 0,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "era": 0,
                "whip": 0,
                "er": 0,
                "bb_allowed": 0,
                "h_allowed": 0,
            }
        ]
    )
    hit_w, pit_w = get_scoring_preset("yahoo")
    result = compute_fantasy_points(df, hit_w, pit_w)
    assert "fantasy_points" in result.columns
    assert result.iloc[0]["fantasy_points"] > 0


def test_compute_fantasy_points_pitcher():
    df = pd.DataFrame(
        [
            {
                "name": "Test Pitcher",
                "is_hitter": False,
                "h": 0,
                "hr": 0,
                "r": 0,
                "rbi": 0,
                "sb": 0,
                "pa": 0,
                "ab": 0,
                "bb": 0,
                "hbp": 0,
                "ip": 200,
                "w": 15,
                "l": 8,
                "sv": 0,
                "k": 200,
                "era": 3.50,
                "whip": 1.20,
                "er": 78,
                "bb_allowed": 50,
                "h_allowed": 180,
            }
        ]
    )
    hit_w, pit_w = get_scoring_preset("yahoo")
    result = compute_fantasy_points(df, hit_w, pit_w)
    assert result.iloc[0]["fantasy_points"] > 0


def test_hitter_gets_zero_pitching_points():
    df = pd.DataFrame(
        [
            {
                "name": "Pure Hitter",
                "is_hitter": True,
                "h": 100,
                "hr": 20,
                "r": 60,
                "rbi": 70,
                "sb": 10,
                "pa": 500,
                "ab": 450,
                "bb": 40,
                "hbp": 3,
                "ip": 0,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "era": 0,
                "whip": 0,
                "er": 0,
                "bb_allowed": 0,
                "h_allowed": 0,
            }
        ]
    )
    hit_w, pit_w = get_scoring_preset("espn")
    result = compute_fantasy_points(df, hit_w, pit_w)
    assert result.iloc[0]["fantasy_points"] > 0


def test_all_presets_nonzero():
    for name in SCORING_PRESETS:
        hit_w, pit_w = get_scoring_preset(name)
        assert len(hit_w) > 0
        assert len(pit_w) > 0
        assert any(v != 0 for v in hit_w.values())
        assert any(v != 0 for v in pit_w.values())


def test_invalid_preset():
    with pytest.raises(KeyError):
        get_scoring_preset("nonexistent_platform")

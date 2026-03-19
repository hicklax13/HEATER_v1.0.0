# tests/test_start_sit_widget.py
"""Tests for WSIS quick compare widget."""
from __future__ import annotations

import pandas as pd

from src.start_sit_widget import (
    compute_fantasy_points_distribution,
    compute_overlap_probability,
    generate_density_data,
    quick_start_sit,
)


def _make_pool() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": 1, "name": "Star Hitter", "is_hitter": True,
                "r": 100, "hr": 35, "rbi": 110, "sb": 15, "pa": 600,
            },
            {
                "player_id": 2, "name": "OK Hitter", "is_hitter": True,
                "r": 60, "hr": 15, "rbi": 60, "sb": 5, "pa": 500,
            },
            {
                "player_id": 3, "name": "Bad Hitter", "is_hitter": True,
                "r": 40, "hr": 8, "rbi": 35, "sb": 2, "pa": 400,
            },
            {
                "player_id": 4, "name": "Ace Pitcher", "is_hitter": False,
                "w": 15, "sv": 0, "k": 220, "ip": 200, "era": 2.80,
            },
        ]
    )


def test_basic_2_player_compare():
    pool = _make_pool()
    result = quick_start_sit([1, 2], pool)
    assert result["recommendation"] == 1  # Star > OK
    assert len(result["players"]) == 2


def test_3_player_compare():
    pool = _make_pool()
    result = quick_start_sit([1, 2, 3], pool)
    assert len(result["players"]) == 3
    assert result["recommendation"] == 1


def test_4_player_compare():
    pool = _make_pool()
    result = quick_start_sit([1, 2, 3, 4], pool)
    assert len(result["players"]) == 4


def test_overlap_identical_distributions():
    """Identical distributions should have overlap near 1.0."""
    ovl = compute_overlap_probability(5.0, 1.0, 5.0, 1.0)
    assert ovl > 0.99


def test_overlap_distant_distributions():
    """Very different distributions should have overlap near 0.0."""
    ovl = compute_overlap_probability(0.0, 0.5, 100.0, 0.5)
    assert ovl < 0.01


def test_overlap_symmetry():
    """OVL(A,B) == OVL(B,A)."""
    ovl1 = compute_overlap_probability(3.0, 1.0, 5.0, 1.5)
    ovl2 = compute_overlap_probability(5.0, 1.5, 3.0, 1.0)
    assert abs(ovl1 - ovl2) < 0.001


def test_density_data_shape():
    x, y = generate_density_data(5.0, 1.0)
    assert len(x) == 200
    assert len(y) == 200
    assert y.max() > 0


def test_distribution_positive_std():
    pool = _make_pool()
    mu, sigma = compute_fantasy_points_distribution(pool.iloc[0])
    assert sigma > 0
    assert mu > 0


def test_pairwise_overlaps_computed():
    pool = _make_pool()
    result = quick_start_sit([1, 2, 3], pool)
    # 3 players = 3 pairwise combinations
    assert len(result["overlaps"]) == 3


def test_single_player_returns_id():
    pool = _make_pool()
    result = quick_start_sit([1], pool)
    assert result["recommendation"] == 1

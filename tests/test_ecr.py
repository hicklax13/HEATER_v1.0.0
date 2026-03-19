"""Tests for ECR integration."""

from __future__ import annotations

import pandas as pd

from src.ecr import (
    blend_ecr_with_projections,
    compute_ecr_disagreement,
    fetch_ecr_extended,
    load_ecr_rankings,
    store_ecr_rankings,
)


def _make_pool():
    return pd.DataFrame(
        [
            {"name": "Player A", "pick_score": 10.0},
            {"name": "Player B", "pick_score": 8.0},
            {"name": "Player C", "pick_score": 6.0},
        ]
    )


def _make_ecr():
    return pd.DataFrame(
        [
            {"player_name": "Player A", "ecr_rank": 1, "best_rank": 1, "worst_rank": 3, "avg_rank": 1.5},
            {"player_name": "Player C", "ecr_rank": 2, "best_rank": 1, "worst_rank": 5, "avg_rank": 2.5},
        ]
    )


def test_blend_basic():
    pool = _make_pool()
    ecr = _make_ecr()
    result = blend_ecr_with_projections(pool, ecr)
    assert "blended_rank" in result.columns
    assert "ecr_badge" in result.columns


def test_blend_empty_ecr():
    pool = _make_pool()
    result = blend_ecr_with_projections(pool, pd.DataFrame())
    assert len(result) == 3


def test_disagreement_ecr_higher():
    badge = compute_ecr_disagreement(50, 10, threshold=20)
    assert badge == "ECR Higher"


def test_disagreement_proj_higher():
    badge = compute_ecr_disagreement(10, 50, threshold=20)
    assert badge == "Proj Higher"


def test_disagreement_within_threshold():
    badge = compute_ecr_disagreement(25, 30, threshold=20)
    assert badge is None


def test_fetch_ecr_returns_dataframe():
    result = fetch_ecr_extended()
    assert isinstance(result, pd.DataFrame)


def test_store_ecr_empty():
    assert store_ecr_rankings(pd.DataFrame()) == 0


def test_store_ecr_count():
    ecr = _make_ecr()
    assert store_ecr_rankings(ecr) == 2


def test_load_ecr_empty():
    result = load_ecr_rankings()
    assert isinstance(result, pd.DataFrame)


def test_blend_weight_effect():
    pool = _make_pool()
    ecr = _make_ecr()
    r1 = blend_ecr_with_projections(pool, ecr, ecr_weight=0.0)
    r2 = blend_ecr_with_projections(pool, ecr, ecr_weight=0.5)
    # With 0 weight, blended should equal proj rank
    assert r1.iloc[0]["blended_rank"] == 1


def test_blend_preserves_columns():
    pool = _make_pool()
    ecr = _make_ecr()
    result = blend_ecr_with_projections(pool, ecr)
    assert "pick_score" in result.columns


def test_ecr_disagreement_zero_diff():
    badge = compute_ecr_disagreement(5, 5)
    assert badge is None

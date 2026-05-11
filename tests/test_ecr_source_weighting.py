"""Test BUG-014 fix: ECR consensus actually applies in-season source weights."""

import pandas as pd
import pytest


def test_compute_player_consensus_applies_source_weights():
    """_compute_player_consensus should weight sources by _SOURCE_WEIGHTS_INSEASON.
    Pass keys WITHOUT the '_rank' suffix (matching the weight dict keys)."""
    from src.ecr import _SOURCE_WEIGHTS_INSEASON, _compute_player_consensus

    assert "fantasypros" in _SOURCE_WEIGHTS_INSEASON
    assert "yahoo" in _SOURCE_WEIGHTS_INSEASON
    assert _SOURCE_WEIGHTS_INSEASON["fantasypros"] > _SOURCE_WEIGHTS_INSEASON["yahoo"]

    a = _compute_player_consensus({"fantasypros": 50, "yahoo": 100})
    b = _compute_player_consensus({"fantasypros": 100, "yahoo": 50})
    assert a["consensus_avg"] < b["consensus_avg"], (
        "BUG-014: ECR source weighting is broken — "
        f"player with fantasypros=50/yahoo=100 should have LOWER consensus "
        f"(better rank) than fantasypros=100/yahoo=50, "
        f"but got A={a['consensus_avg']:.2f} vs B={b['consensus_avg']:.2f}"
    )


def test_refresh_ecr_strips_rank_suffix_from_keys():
    """Passing _rank-suffixed keys should give EQUAL weighting (since lookup
    misses _SOURCE_WEIGHTS_INSEASON and falls back to 1.0); bare keys give
    differential weighting."""
    from src.ecr import _compute_player_consensus

    bare = _compute_player_consensus({"fantasypros": 30, "yahoo": 60})
    suffixed = _compute_player_consensus({"fantasypros_rank": 30, "yahoo_rank": 60})
    assert bare["consensus_avg"] < suffixed["consensus_avg"], (
        "BUG-014: passing _rank-suffixed keys does NOT trigger in-season "
        "weighting (all weights default to 1.0). Caller must strip the suffix."
    )

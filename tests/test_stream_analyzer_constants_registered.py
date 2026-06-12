"""Structural guards: Stream Analyzer constants live in CONSTANTS_REGISTRY.

Phase 0 of the Pitcher Streaming Analyzer plan
(docs/superpowers/plans/2026-06-09-pitcher-streaming-analyzer.md).

Every Stream Score weight and risk threshold must be registered with value,
citation, bounds, and sensitivity BEFORE any scoring code consumes it — the
same discipline PR #107 imposed on the FA engine. The six component weights
must sum to 1.0 so the composite stays a convex blend.
"""

from __future__ import annotations

import pytest

from src.optimizer.constants_registry import CONSTANTS_REGISTRY

_EXPECTED_CONSTANTS = {
    # Stream Score component weights (convex blend, sum == 1.0)
    "stream_score_w_matchup": 0.35,
    "stream_score_w_sgp": 0.25,
    "stream_score_w_form": 0.15,
    "stream_score_w_lineup": 0.10,
    "stream_score_w_env": 0.10,
    "stream_score_w_winprob": 0.05,
    # Risk-flag thresholds
    "stream_risk_short_leash_ip": 4.5,
    "stream_risk_elite_offense_wrc": 110,
    "stream_risk_hitter_park": 1.08,
    "stream_risk_wind_out_mph": 12.0,
}

_WEIGHT_KEYS = [k for k in _EXPECTED_CONSTANTS if k.startswith("stream_score_w_")]


@pytest.mark.parametrize("key", sorted(_EXPECTED_CONSTANTS))
def test_constant_registered(key):
    assert key in CONSTANTS_REGISTRY, (
        f"{key} missing from CONSTANTS_REGISTRY — stream_analyzer tunables must be "
        "registered (citation, bounds, sensitivity) before code consumes them"
    )


@pytest.mark.parametrize("key", sorted(_EXPECTED_CONSTANTS))
def test_constant_value_and_metadata(key):
    entry = CONSTANTS_REGISTRY[key]
    assert entry.value == pytest.approx(_EXPECTED_CONSTANTS[key])
    assert entry.lower_bound <= entry.value <= entry.upper_bound, (
        f"{key}: value {entry.value} outside [{entry.lower_bound}, {entry.upper_bound}]"
    )
    assert entry.citation.strip(), f"{key}: citation must be non-empty"
    assert entry.sensitivity in {"HIGH", "MEDIUM", "LOW"}
    assert "stream_analyzer" in entry.module, f"{key}: module should reference stream_analyzer.py, got {entry.module!r}"


def test_score_weights_sum_to_one():
    total = sum(CONSTANTS_REGISTRY[k].value for k in _WEIGHT_KEYS)
    assert total == pytest.approx(1.0, abs=1e-9), (
        f"Stream Score component weights must sum to 1.0 (got {total}); the composite is a convex blend"
    )


def test_six_component_weights_exist():
    assert len(_WEIGHT_KEYS) == 6

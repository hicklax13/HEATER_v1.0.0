"""Tests for src/optimizer/streaming.py -- pitcher streaming and two-start valuation."""

from __future__ import annotations

import pytest

from src.optimizer.streaming import (
    compute_streaming_value,
    optimal_streaming_schedule,
    quantify_two_start_value,
    rank_streaming_candidates,
)

# ── compute_streaming_value ──────────────────────────────────────────


def test_streaming_value_good_pitcher():
    """A low-ERA, high-K pitcher should have a positive net value."""
    pitcher = {"k": 8.0, "w": 0.5, "era": 2.50, "whip": 0.95, "ip": 6.0}
    result = compute_streaming_value(pitcher)
    assert result["net_value"] > 0
    assert result["counting_sgp"] > 0
    assert result["rate_impact"] > 0  # ERA well below baseline -> helps


def test_streaming_value_bad_pitcher():
    """A high-ERA pitcher can have negative net value."""
    pitcher = {"k": 3.0, "w": 0.2, "era": 6.50, "whip": 1.70, "ip": 4.0}
    result = compute_streaming_value(pitcher)
    # Rate damage should outweigh small counting stats
    assert result["rate_impact"] < 0
    # Net could be negative
    assert result["net_value"] < result["counting_sgp"]


def test_streaming_value_season_totals():
    """Season totals (IP > 30) are normalised to per-start rates."""
    pitcher = {"k": 180.0, "w": 12.0, "era": 3.20, "whip": 1.10, "ip": 165.0}
    result = compute_streaming_value(pitcher)
    # Should still produce reasonable per-start K value (not 180)
    assert result["k_per_start"] < 10.0
    assert result["k_per_start"] > 3.0


def test_streaming_value_park_factor():
    """Hitter-friendly park factor inflates ERA, reducing net value."""
    pitcher = {"k": 7.0, "w": 0.4, "era": 3.50, "whip": 1.15, "ip": 6.0}
    neutral = compute_streaming_value(pitcher, team_park_factor=1.0)
    coors = compute_streaming_value(pitcher, team_park_factor=1.38)
    # Coors field should produce worse (lower) net value
    assert coors["net_value"] < neutral["net_value"]


def test_counting_sgp_positive():
    """K + W contribution is always >= 0 (counting stats are additive)."""
    pitcher = {"k": 5.0, "w": 0.3, "era": 5.00, "whip": 1.40, "ip": 5.0}
    result = compute_streaming_value(pitcher)
    assert result["counting_sgp"] >= 0


# ── quantify_two_start_value ─────────────────────────────────────────


def test_two_start_below_team_era():
    """Pitcher with ERA below team ERA -> positive rate_impact for 2nd start."""
    pitcher = {"k": 7.0, "w": 0.5, "era": 3.00, "whip": 1.05, "ip": 6.0}
    result = quantify_two_start_value(pitcher, team_era=4.20, team_whip=1.30)
    assert result["rate_impact"] > 0


def test_two_start_above_team_era():
    """Pitcher with ERA above team ERA -> negative rate_impact for 2nd start."""
    pitcher = {"k": 4.0, "w": 0.2, "era": 5.80, "whip": 1.55, "ip": 5.0}
    result = quantify_two_start_value(pitcher, team_era=3.80, team_whip=1.20)
    assert result["rate_impact"] < 0


def test_two_start_recommendation():
    """Net value > 0 -> 'Start'; < 0 -> 'Sit'."""
    ace = {"k": 9.0, "w": 0.6, "era": 2.50, "whip": 0.90, "ip": 7.0}
    result_ace = quantify_two_start_value(ace, team_era=4.00, team_whip=1.25)
    assert result_ace["recommendation"] == "Start"
    assert result_ace["net_value"] > 0

    scrub = {"k": 3.0, "w": 0.1, "era": 6.50, "whip": 1.70, "ip": 4.0}
    result_scrub = quantify_two_start_value(scrub, team_era=3.50, team_whip=1.10)
    assert result_scrub["recommendation"] == "Sit"
    assert result_scrub["net_value"] < 0


def test_two_start_counting_always_positive():
    """Counting SGP from the 2nd start is always >= 0."""
    pitcher = {"k": 4.0, "w": 0.2, "era": 6.00, "whip": 1.60, "ip": 4.5}
    result = quantify_two_start_value(pitcher)
    assert result["counting_sgp"] >= 0


# ── rank_streaming_candidates ────────────────────────────────────────


def test_rank_candidates_sorted():
    """Candidates should be sorted by net_value descending."""
    pitchers = [
        {"player_name": "Ace", "team": "NYY", "k": 9.0, "w": 0.6, "era": 2.50, "whip": 0.90, "ip": 6.5},
        {"player_name": "Average", "team": "CHC", "k": 6.0, "w": 0.4, "era": 4.00, "whip": 1.25, "ip": 5.5},
        {"player_name": "Bad", "team": "COL", "k": 3.0, "w": 0.2, "era": 6.00, "whip": 1.60, "ip": 4.0},
    ]
    ranked = rank_streaming_candidates(pitchers)
    assert len(ranked) == 3
    assert ranked[0]["player_name"] == "Ace"
    # Values should be monotonically decreasing
    for i in range(len(ranked) - 1):
        assert ranked[i]["net_value"] >= ranked[i + 1]["net_value"]


def test_streaming_empty_roster():
    """Empty pitcher list returns empty result."""
    result = rank_streaming_candidates([])
    assert result == []


def test_rank_candidates_max_results():
    """max_results limits the output size."""
    pitchers = [
        {"player_name": f"P{i}", "team": "NYY", "k": 6.0, "w": 0.4, "era": 3.50, "whip": 1.15, "ip": 5.5}
        for i in range(20)
    ]
    ranked = rank_streaming_candidates(pitchers, max_results=5)
    assert len(ranked) <= 5


# ── optimal_streaming_schedule ───────────────────────────────────────


def test_optimal_schedule_respects_limit():
    """max_adds limits the number of recommended pickups."""
    candidates = [{"player_name": f"P{i}", "net_value": 0.5 - i * 0.05} for i in range(10)]
    schedule = optimal_streaming_schedule(candidates, max_adds=3)
    assert len(schedule) <= 3


def test_optimal_schedule_skips_negative():
    """Candidates with negative net value are not included."""
    candidates = [
        {"player_name": "Good", "net_value": 0.3},
        {"player_name": "Bad", "net_value": -0.1},
    ]
    schedule = optimal_streaming_schedule(candidates, max_adds=7)
    assert len(schedule) == 1
    assert schedule[0]["player_name"] == "Good"


def test_optimal_schedule_empty():
    """Empty candidate list returns empty schedule."""
    assert optimal_streaming_schedule([]) == []


def test_optimal_schedule_deduplicates():
    """Same player appearing twice is only picked once."""
    candidates = [
        {"player_name": "Ace", "net_value": 0.5},
        {"player_name": "Ace", "net_value": 0.4},
        {"player_name": "Other", "net_value": 0.3},
    ]
    schedule = optimal_streaming_schedule(candidates, max_adds=5)
    names = [c["player_name"] for c in schedule]
    assert names.count("Ace") == 1

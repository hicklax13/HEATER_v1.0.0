"""Tests for IL manager."""

from __future__ import annotations

import pandas as pd

from src.il_manager import (
    IL_DURATION_ESTIMATES,
    ILAlert,
    classify_il_type,
    compute_lost_sgp,
    detect_il_changes,
    estimate_il_duration,
    find_best_replacement,
    generate_il_alert,
)


def test_classify_il10():
    assert classify_il_type("10-Day IL") == "IL10"


def test_classify_il15():
    assert classify_il_type("15-Day Injured List") == "IL15"


def test_classify_il60():
    assert classify_il_type("60-Day IL") == "IL60"


def test_classify_dtd():
    assert classify_il_type("Day-To-Day") == "DTD"


def test_classify_generic_il():
    assert classify_il_type("IL") == "IL15"


def test_estimate_duration_hitter():
    d = estimate_il_duration("IL10", "SS")
    assert d == 2.0


def test_estimate_duration_pitcher():
    d = estimate_il_duration("IL10", "SP")
    assert d > 2.0  # Pitchers slightly longer


def test_lost_sgp_half_season():
    lost = compute_lost_sgp(10.0, 11.0, 22.0)
    assert abs(lost - 5.0) < 0.01


def test_lost_sgp_zero_remaining():
    assert compute_lost_sgp(10.0, 5.0, 0.0) == 0.0


def test_find_replacement_basic():
    bench = pd.DataFrame(
        [
            {"player_id": 10, "name": "Bench Guy", "positions": "SS,2B", "pick_score": 3.0},
            {"player_id": 11, "name": "Bench Guy 2", "positions": "OF", "pick_score": 2.0},
        ]
    )
    result = find_best_replacement(["SS"], bench)
    assert result is not None
    assert result["player_id"] == 10


def test_find_replacement_no_eligible():
    bench = pd.DataFrame(
        [
            {"player_id": 10, "name": "Bench Guy", "positions": "OF", "pick_score": 3.0},
        ]
    )
    result = find_best_replacement(["C"], bench)
    assert result is None


def test_find_replacement_empty_bench():
    result = find_best_replacement(["SS"], pd.DataFrame())
    assert result is None


def test_find_replacement_picks_highest_sgp():
    bench = pd.DataFrame(
        [
            {"player_id": 10, "name": "Low", "positions": "SS", "pick_score": 1.0},
            {"player_id": 11, "name": "High", "positions": "SS", "pick_score": 5.0},
        ]
    )
    result = find_best_replacement(["SS"], bench)
    assert result["player_id"] == 11


def test_detect_il_changes():
    roster = pd.DataFrame(
        [
            {"player_id": 1, "name": "Hurt Player", "positions": "SS", "status": "10-Day IL"},
            {"player_id": 2, "name": "Healthy", "positions": "OF", "status": ""},
        ]
    )
    changes = detect_il_changes(roster, {1: "", 2: ""})
    assert len(changes) == 1
    assert changes[0]["il_type"] == "IL10"


def test_detect_no_changes():
    roster = pd.DataFrame(
        [
            {"player_id": 1, "name": "Same", "positions": "SS", "status": "10-Day IL"},
        ]
    )
    changes = detect_il_changes(roster, {1: "10-Day IL"})
    assert len(changes) == 0


def test_generate_alert():
    il_player = {"player_id": 1, "player_name": "Test", "il_type": "IL10", "positions": "SS"}
    bench = pd.DataFrame([{"player_id": 10, "name": "Sub", "positions": "SS", "pick_score": 2.0}])
    alert = generate_il_alert(il_player, bench, player_sgp=5.0, weeks_remaining=22.0)
    assert isinstance(alert, ILAlert)
    assert alert.il_type == "IL10"
    assert alert.lost_sgp > 0
    assert alert.recommended_replacement_name == "Sub"


def test_il_durations_complete():
    for key in ["IL10", "IL15", "IL60", "DTD"]:
        assert key in IL_DURATION_ESTIMATES

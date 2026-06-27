import math

import pandas as pd

from src.points_scoring import (
    PointsResult,
    PointsScoringConfig,
    project_player_points,
)


def _hitter(**over):
    row = {
        "player_id": 1,
        "is_hitter": 1,
        "r": 90,
        "hr": 30,
        "rbi": 100,
        "sb": 10,
        "h": 150,
        "bb": 60,
        "hbp": 5,
        "ab": 550,
        "sf": 4,
        "avg": 0.273,
        "obp": 0.350,
        "ip": 0,
    }
    row.update(over)
    return row


def _cfg(hit=None, pit=None):
    return PointsScoringConfig(hitter_weights=hit or {}, pitcher_weights=pit or {})


def test_hitter_points_is_weighted_sum():
    cfg = _cfg(hit={"HR": 4.0, "R": 1.0, "RBI": 1.0, "SB": 2.0, "BB": 1.0})
    res = project_player_points(_hitter(), cfg)
    # 30*4 + 90*1 + 100*1 + 10*2 + 60*1 = 120 + 90 + 100 + 20 + 60 = 390
    assert isinstance(res, PointsResult)
    assert res.points == 390.0
    assert res.breakdown["HR"] == 120.0


def test_unprojected_hitter_stat_is_uncovered_not_zeroed_silently():
    # Hitter strikeouts ("K") are NOT projected → flagged uncovered, not scored.
    cfg = _cfg(hit={"HR": 4.0, "K": -1.0})
    res = project_player_points(_hitter(), cfg)
    assert res.points == 120.0  # only HR scored
    assert "K" in res.uncovered
    assert "K" not in res.breakdown


def test_weight_keys_are_case_insensitive():
    cfg = _cfg(hit={"hr": 4.0})
    assert project_player_points(_hitter(), cfg).points == 120.0


def test_nan_and_missing_values_score_zero_never_raise():
    cfg = _cfg(hit={"HR": 4.0, "RBI": 1.0})
    row = _hitter(hr=float("nan"))
    del row["rbi"]  # missing column entirely
    res = project_player_points(row, cfg)
    assert res.points == 0.0
    assert math.isfinite(res.points)

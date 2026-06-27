import math

import pandas as pd

from src.points_scoring import (
    PointsResult,
    PointsScoringConfig,
    project_player_points,
    rank_players_by_points,
    roster_points,
    uncovered_stats,
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


def _pitcher(**over):
    row = {
        "player_id": 2,
        "is_hitter": 0,
        "w": 15,
        "l": 8,
        "sv": 0,
        "k": 220,
        "ip": 195,
        "er": 70,
        "bb_allowed": 50,
        "h_allowed": 165,
        "era": 3.23,
        "whip": 1.10,
    }
    row.update(over)
    return row


def test_pitcher_uses_pitcher_columns_for_h_and_bb():
    # A pitcher's "H" must resolve to h_allowed (165), NOT a hitter h.
    cfg = _cfg(pit={"K": 1.0, "W": 5.0, "ER": -2.0, "H": -0.5, "BB": -0.5})
    res = project_player_points(_pitcher(), cfg)
    # 220*1 + 15*5 + 70*-2 + 165*-0.5 + 50*-0.5 = 220 + 75 - 140 - 82.5 - 25 = 47.5
    assert res.points == 47.5


def test_two_way_player_scores_both_halves():
    cfg = _cfg(hit={"HR": 4.0}, pit={"K": 1.0})
    # Ohtani-like: hits AND pitches.
    row = _hitter(player_id=3, hr=50, ip=130, k=180, is_hitter=1)
    res = project_player_points(row, cfg)
    # hitter: 50*4 = 200 ; pitcher: 180*1 = 180 → 380
    assert res.points == 380.0


def test_pure_pitcher_is_not_scored_with_hitter_weights():
    cfg = _cfg(hit={"HR": 4.0}, pit={"K": 1.0})
    res = project_player_points(_pitcher(k=200), cfg)
    assert res.points == 200.0  # only the pitcher half


def _pool():
    return pd.DataFrame(
        [
            _hitter(player_id=1, hr=30),
            _hitter(player_id=2, hr=10),
            _pitcher(player_id=3, k=250),
        ]
    )


def test_uncovered_stats_reports_per_type_unprojected_stats():
    cfg = _cfg(hit={"HR": 4.0, "K": -1.0, "DOUBLES": 2.0}, pit={"K": 1.0, "HLD": 5.0})
    unc = uncovered_stats(cfg, _pool())
    assert unc["hitter"] == {"K", "DOUBLES"}
    assert unc["pitcher"] == {"HLD"}


def test_rank_orders_by_points_desc_and_does_not_mutate_input():
    cfg = _cfg(hit={"HR": 4.0})
    pool = _pool()
    before = pool.copy(deep=True)
    ranked = rank_players_by_points(pool, cfg)
    assert "points" in ranked.columns
    hitters = ranked[ranked["is_hitter"] == 1]
    assert list(hitters["player_id"])[:2] == [1, 2]  # hr 30 (120) before hr 10 (40)
    assert "points" not in pool.columns  # input untouched
    pd.testing.assert_frame_equal(pool, before)


def test_roster_points_sums_named_players_unknown_ids_zero():
    cfg = _cfg(hit={"HR": 4.0})
    total = roster_points([1, 2, 999], _pool(), cfg)
    assert total == 120.0 + 40.0  # ids 1 and 2; 999 unknown → 0


def test_roster_points_empty_roster_is_zero():
    cfg = _cfg(hit={"HR": 4.0})
    assert roster_points([], _pool(), cfg) == 0.0

"""leaders/overall: helper unit tests + fake-service contract test."""

from __future__ import annotations

import pandas as pd

from api.services.leaders_overall_service import (
    _LENS_META,
    _norm_delta,
    _norm_z,
    _overall_stats,
    _to_overall_row,
)


def test_norm_z_maps_and_clamps():
    assert _norm_z(0.0) == 50.0  # (0+4)/8*100
    assert _norm_z(4.0) == 100.0
    assert _norm_z(-4.0) == 0.0
    assert _norm_z(99.0) == 100.0  # clamp
    assert _norm_z(float("nan")) == 0.0  # NaN-safe


def test_norm_delta_maps_and_clamps():
    assert _norm_delta(0.0) == 50.0  # (0+3)/6*100
    assert _norm_delta(3.0) == 100.0
    assert _norm_delta(-3.0) == 0.0
    assert _norm_delta(float("nan")) == 0.0


def test_overall_stats_hitter_and_pitcher():
    hrow = {"ytd_hr": 24, "ytd_r": 58, "ytd_avg": 0.322}
    assert _overall_stats(hrow, True) == ["24 HR", "58 R", ".322 AVG"]
    prow = {"ytd_k": 180, "ytd_era": 3.21, "ytd_whip": 1.05}
    assert _overall_stats(prow, False) == ["180 K", "3.21 ERA", "1.05 WHIP"]


def test_lens_meta_covers_all_five():
    assert set(_LENS_META) == {"overall", "hot", "cold", "breakout", "sell"}
    assert _LENS_META["overall"] == ("", "flat", _LENS_META["overall"][2])  # tag empty for overall
    assert _LENS_META["hot"][0] == "hot" and _LENS_META["hot"][1] == "up"
    assert _LENS_META["cold"][1] == "down"


def test_to_overall_row_enriches_and_stamps_lens():
    pool = pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "Aaron Judge",
                "positions": "OF",
                "mlb_id": 592450,
                "team": "NYY",
                "is_hitter": True,
                "ytd_hr": 30,
                "ytd_r": 70,
                "ytd_avg": 0.31,
            }
        ]
    )
    row = {"player_id": 1, "_value": 88.0}
    item = _to_overall_row(2, row, pool, "hot")
    assert item.rank == 2
    assert item.value == 88.0
    assert item.player.mlb_id == 592450
    assert item.player.team_id == 147  # NYY
    assert item.hitter is True
    assert item.tag == "hot"
    assert item.trend == "up"
    assert item.note == _LENS_META["hot"][2]
    assert item.stats == ["30 HR", "70 R", ".310 AVG"]


def test_to_overall_row_missing_pool_row_degrades():
    item = _to_overall_row(1, {"player_id": 999, "_value": 50.0}, pd.DataFrame(), "overall")
    assert item.player.name == "Player 999"
    assert item.stats == []  # no pool row → no stats
    assert item.tag == ""
    assert item.value == 50.0

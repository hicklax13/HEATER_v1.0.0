"""HIGH-2: pool is_hitter is authoritative; on a pool miss, classify by ELIGIBLE
positions (SP/RP/P → pitcher), NOT the assigned slot (a benched pitcher's slot is
'BN'/'IL'). Default unknown → hitter, with a debug log."""

import logging

import pandas as pd

from api.services.matchup_service import _is_pitcher


def _pool(rows):
    return pd.DataFrame(rows)


def test_pool_hit_pitcher_flag_wins():
    pool = _pool([{"player_id": 5, "is_hitter": False}])
    assert _is_pitcher(5, "BN", pool) is True  # pool says pitcher even though slot is BN


def test_pool_hit_hitter_flag_wins():
    pool = _pool([{"player_id": 6, "is_hitter": True}])
    assert _is_pitcher(6, "SP,RP", pool) is False  # pool says hitter; slot ignored


def test_pool_miss_eligible_sp_is_pitcher():
    pool = _pool([{"player_id": 6, "is_hitter": True}])  # 99 absent
    assert _is_pitcher(99, "SP,RP", pool) is True


def test_pool_miss_benched_pitcher_classified_by_eligibility():
    # selected_position would be 'BN' but eligible carries 'SP' → pitcher.
    assert _is_pitcher(99, "SP", None) is True


def test_pool_miss_eligible_hitter_is_hitter():
    assert _is_pitcher(99, "2B,SS", None) is False


def test_pool_miss_unknown_defaults_hitter_and_logs(caplog):
    with caplog.at_level(logging.DEBUG, logger="api.services.matchup_service"):
        assert _is_pitcher(99, "", None) is False
    assert any("unresolved" in r.message for r in caplog.records)

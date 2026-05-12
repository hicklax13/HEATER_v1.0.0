"""BUG-017 fix: compute_fa_comparisons handles negative-SGP players."""

import pandas as pd


def test_compute_fa_comparisons_finds_alternative_for_negative_sgp():
    """A bad reliever (negative target SGP) should still get a non-empty
    fa_name and non-zero fa_value — the comparison should seed from the
    first candidate, not skip everything below the 0.0 default."""
    from src.trade_intelligence import compute_fa_comparisons
    from src.valuation import LeagueConfig

    cfg = LeagueConfig()

    # Build a tiny pool with bad relievers (will produce negative total SGP).
    hitter_baseline = {
        "r": 0,
        "hr": 0,
        "rbi": 0,
        "sb": 0,
        "ab": 0,
        "h": 0,
        "bb": 0,
        "hbp": 0,
        "sf": 0,
        "avg": 0.0,
        "obp": 0.0,
        "pa": 0,
    }
    pool = pd.DataFrame(
        [
            # Opponent's bad reliever (target_sgp very negative)
            {
                "player_id": 1,
                "name": "BadReliever",
                "player_name": "BadReliever",
                "positions": "RP",
                "is_hitter": 0,
                **hitter_baseline,
                "w": 0,
                "l": 8,
                "sv": 2,
                "k": 30,
                "ip": 50.0,
                "era": 6.0,
                "whip": 1.8,
                "er": 33,
                "bb_allowed": 30,
                "h_allowed": 60,
            },
            # FA #1 — also bad but slightly less so
            {
                "player_id": 100,
                "name": "BadFA1",
                "player_name": "BadFA1",
                "positions": "RP",
                "is_hitter": 0,
                **hitter_baseline,
                "w": 0,
                "l": 6,
                "sv": 1,
                "k": 25,
                "ip": 40.0,
                "era": 5.5,
                "whip": 1.7,
                "er": 24,
                "bb_allowed": 22,
                "h_allowed": 46,
            },
            # FA #2 — the least-bad option
            {
                "player_id": 101,
                "name": "OkFA",
                "player_name": "OkFA",
                "positions": "RP",
                "is_hitter": 0,
                **hitter_baseline,
                "w": 1,
                "l": 4,
                "sv": 5,
                "k": 35,
                "ip": 45.0,
                "era": 4.2,
                "whip": 1.3,
                "er": 21,
                "bb_allowed": 15,
                "h_allowed": 43,
            },
        ]
    )
    fa_pool = pool[pool["player_id"] != 1].copy()
    result = compute_fa_comparisons(
        opponent_player_ids=[1],
        user_roster_ids=[],
        fa_pool=fa_pool,
        player_pool=pool,
        config=cfg,
    )
    assert 1 in result, f"Missing pid=1 in result: {result}"
    r = result[1]
    assert r["fa_name"] != "", f"BUG-017: best FA name empty for negative-SGP player. Got: {r}"
    assert abs(r["fa_value"]) > 0.001, f"BUG-017: best FA value is ~0, indicating no alternative was found. Got: {r}"

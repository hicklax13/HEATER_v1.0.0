"""SF-25 parity test: in_season raw_change/denom pattern matches SGPCalculator."""

from src.valuation import LeagueConfig, SGPCalculator


def test_in_season_raw_change_parity():
    cfg = LeagueConfig()
    raw_change = {"R": 50.0, "ERA": -0.30, "WHIP": -0.05}
    expected = 0.0
    for cat, val in raw_change.items():
        denom = cfg.sgp_denominators.get(cat, 1.0)
        if denom == 0:
            continue
        sign = -1.0 if cat in cfg.inverse_stats else 1.0
        expected += sign * val / denom
    actual = SGPCalculator(cfg).totals_sgp(raw_change)
    assert abs(actual - expected) < 1e-9

"""SF-25 parity test: trade_simulator's _simulate_roster_sgp matches SGPCalculator.totals_sgp."""

from src.valuation import LeagueConfig, SGPCalculator


def test_simulator_sgp_parity():
    cfg = LeagueConfig()
    totals = {cat: float(i + 1) * 10 for i, cat in enumerate(cfg.all_categories)}
    expected = 0.0
    for cat in cfg.all_categories:
        val = totals.get(cat, 0.0)
        denom = cfg.sgp_denominators.get(cat, 1.0)
        if denom == 0:
            continue
        sign = -1.0 if cat in cfg.inverse_stats else 1.0
        expected += sign * val / denom
    actual = SGPCalculator(cfg).totals_sgp(totals)
    assert abs(actual - expected) < 1e-9

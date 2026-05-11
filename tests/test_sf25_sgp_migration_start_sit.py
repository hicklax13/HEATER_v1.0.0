"""SF-25 parity test: start_sit (proj-baseline)/|denom| pattern matches SGPCalculator."""

from src.valuation import LeagueConfig, SGPCalculator


def test_start_sit_parity():
    cfg = LeagueConfig()
    proj = {"HR": 30.0, "ERA": 3.50}
    baseline = {"HR": 25.0, "ERA": 4.00}
    delta = {cat: proj[cat] - baseline[cat] for cat in proj}
    expected = 0.0
    for cat, d in delta.items():
        denom = abs(cfg.sgp_denominators.get(cat, 1.0))
        if denom == 0:
            continue
        sign = -1.0 if cat in cfg.inverse_stats else 1.0
        expected += sign * d / denom
    actual = SGPCalculator(cfg).totals_sgp(delta)
    assert abs(actual - expected) < 1e-9, f"actual={actual}, expected={expected}"

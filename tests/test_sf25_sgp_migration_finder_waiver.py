"""SF-25 parity tests: SGPCalculator.totals_sgp must produce identical
output to the local helpers it replaces."""

from src.valuation import LeagueConfig, SGPCalculator


def test_trade_finder_totals_sgp_parity():
    """SGPCalculator.totals_sgp matches the OLD _totals_sgp formula."""
    cfg = LeagueConfig()
    totals = {
        "R": 800,
        "HR": 250,
        "RBI": 800,
        "SB": 100,
        "AVG": 0.265,
        "OBP": 0.335,
        "W": 75,
        "L": 60,
        "SV": 50,
        "K": 1400,
        "ERA": 3.85,
        "WHIP": 1.21,
    }
    # Compute the OLD formula manually (matches trade_finder._totals_sgp)
    expected = 0.0
    for cat in cfg.all_categories:
        denom = cfg.sgp_denominators.get(cat, 1.0)
        if abs(denom) < 1e-9:
            denom = 1.0
        val = totals.get(cat, 0)
        if cat in cfg.inverse_stats:
            expected -= val / denom
        else:
            expected += val / denom
    actual = SGPCalculator(cfg).totals_sgp(totals)
    assert abs(actual - expected) < 1e-9, f"expected {expected}, got {actual}"


def test_trade_finder_weighted_sgp_parity():
    """SGPCalculator.totals_sgp(weights=...) matches OLD _weighted_totals_sgp."""
    cfg = LeagueConfig()
    totals = {
        "R": 800,
        "HR": 250,
        "RBI": 800,
        "SB": 100,
        "AVG": 0.265,
        "OBP": 0.335,
        "W": 75,
        "L": 60,
        "SV": 50,
        "K": 1400,
        "ERA": 3.85,
        "WHIP": 1.21,
    }
    weights = {
        "HR": 2.0,
        "ERA": 0.5,
        "K": 1.5,
        "SB": 0.8,
        "WHIP": 1.2,
    }
    # OLD formula (trade_finder._weighted_totals_sgp body)
    expected = 0.0
    for cat in cfg.all_categories:
        denom = cfg.sgp_denominators.get(cat, 1.0)
        if abs(denom) < 1e-9:
            denom = 1.0
        val = totals.get(cat, 0)
        w = weights.get(cat, 1.0)
        if cat in cfg.inverse_stats:
            expected -= (val / denom) * w
        else:
            expected += (val / denom) * w
    actual = SGPCalculator(cfg).totals_sgp(totals, weights=weights)
    assert abs(actual - expected) < 1e-9, f"expected {expected}, got {actual}"


def test_trade_finder_weighted_sgp_no_weights_parity():
    """Weighted helper with weights=None must equal unweighted helper."""
    cfg = LeagueConfig()
    totals = {"HR": 50, "ERA": 4.0, "K": 1500}
    # OLD: _weighted_totals_sgp(totals, config, None) returns _totals_sgp(totals, config)
    expected_unweighted = 0.0
    for cat in cfg.all_categories:
        denom = cfg.sgp_denominators.get(cat, 1.0)
        if abs(denom) < 1e-9:
            denom = 1.0
        val = totals.get(cat, 0)
        if cat in cfg.inverse_stats:
            expected_unweighted -= val / denom
        else:
            expected_unweighted += val / denom
    actual = SGPCalculator(cfg).totals_sgp(totals)
    assert abs(actual - expected_unweighted) < 1e-9


def test_waiver_wire_totals_to_sgp_parity():
    """SGPCalculator.totals_sgp matches OLD waiver_wire._totals_to_sgp."""
    cfg = LeagueConfig()
    totals = {
        "K": 1500,
        "WHIP": 1.20,
        "R": 750,
        "HR": 200,
        "AVG": 0.275,
        "ERA": 3.75,
        "L": 55,
    }
    # OLD formula (identical body to trade_finder's _totals_sgp)
    expected = 0.0
    for cat in cfg.all_categories:
        denom = cfg.sgp_denominators.get(cat, 1.0)
        if abs(denom) < 1e-9:
            denom = 1.0
        val = totals.get(cat, 0)
        if cat in cfg.inverse_stats:
            expected -= val / denom
        else:
            expected += val / denom
    actual = SGPCalculator(cfg).totals_sgp(totals)
    assert abs(actual - expected) < 1e-9, f"expected {expected}, got {actual}"


def test_inverse_stat_sign_flip():
    """Inverse cats (L, ERA, WHIP) must subtract; positive cats add."""
    cfg = LeagueConfig()
    # Pure inverse total: only ERA contributes negatively
    inverse_totals = {"ERA": 4.00}
    pos_totals = {"K": 1000}
    inverse_sgp = SGPCalculator(cfg).totals_sgp(inverse_totals)
    pos_sgp = SGPCalculator(cfg).totals_sgp(pos_totals)
    assert inverse_sgp < 0, f"ERA-only SGP must be negative; got {inverse_sgp}"
    assert pos_sgp > 0, f"K-only SGP must be positive; got {pos_sgp}"


def test_empty_totals_returns_zero():
    """An empty totals dict must produce exactly 0.0 SGP."""
    cfg = LeagueConfig()
    assert SGPCalculator(cfg).totals_sgp({}) == 0.0

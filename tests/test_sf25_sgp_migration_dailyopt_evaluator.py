"""SF-25 parity tests for daily_optimizer + trade_evaluator migrations.

These tests pin the behaviour of the inline ``(value/denom) * sign`` SGP
patterns at:

  - ``src/optimizer/daily_optimizer.py:935-940`` — DCV per-category SGP scaling
    for counting stats.
  - ``src/engine/output/trade_evaluator.py:908-927`` — Phase 1 weighted SGP
    delta per category.

Both patterns are mathematically equivalent to
``SGPCalculator.totals_sgp(totals, weights)``.  These tests assert that
equivalence so the migrations can be done with confidence.
"""

from src.valuation import LeagueConfig, SGPCalculator


def test_daily_optimizer_dcv_sgp_parity():
    """daily_optimizer.py's weighted_dcv/denom pattern matches SGPCalculator."""
    cfg = LeagueConfig()
    weighted_dcv = {"R": 12.0, "HR": 5.0, "ERA": -0.20}
    expected = 0.0
    for cat, val in weighted_dcv.items():
        denom = cfg.sgp_denominators.get(cat, 1.0)
        if denom == 0:
            continue
        sign = -1.0 if cat in cfg.inverse_stats else 1.0
        expected += sign * val / denom
    actual = SGPCalculator(cfg).totals_sgp(weighted_dcv)
    assert abs(actual - expected) < 1e-9


def test_trade_evaluator_phase1_sgp_delta_parity():
    """Phase 1 SGP delta in trade_evaluator matches SGPCalculator."""
    cfg = LeagueConfig()
    # Trade delta: receiving HR=10, giving up R=15, ERA goes up 0.20 (worse)
    raw_change = {"R": -15.0, "HR": 10.0, "ERA": 0.20}
    expected = 0.0
    for cat, val in raw_change.items():
        denom = cfg.sgp_denominators.get(cat, 1.0)
        if denom == 0:
            continue
        sign = -1.0 if cat in cfg.inverse_stats else 1.0
        expected += sign * val / denom
    actual = SGPCalculator(cfg).totals_sgp(raw_change)
    assert abs(actual - expected) < 1e-9


def test_daily_optimizer_per_category_breakdown_parity():
    """Per-category dcv values must equal weighted/denom * sign per cat.

    The daily_optimizer needs the per-cat values for the ``dcv_<cat>`` columns
    in its output. This test pins the per-category math (which is what we'll
    extract into a helper that uses SGPCalculator's denom/sign convention).
    """
    cfg = LeagueConfig()
    calc = SGPCalculator(cfg)
    weighted_dcv_by_cat = {"R": 12.0, "HR": 5.0, "L": 2.0, "K": 30.0}
    for cat, val in weighted_dcv_by_cat.items():
        # Old formula
        denom = cfg.sgp_denominators.get(cat, 1.0)
        sign = -1.0 if cat in cfg.inverse_stats else 1.0
        expected = sign * val / denom
        # New formula (single-cat totals dict)
        actual = calc.totals_sgp({cat: val})
        assert abs(actual - expected) < 1e-9, f"mismatch for {cat}"


def test_trade_evaluator_weighted_phase1_parity():
    """Phase 1 SGP delta with per-cat weights matches totals_sgp(weights=...).

    The trade evaluator multiplies each cat's SGP by ``category_weights[cat]``
    after sign-flipping. ``SGPCalculator.totals_sgp`` accepts a ``weights``
    kwarg that does the same thing.
    """
    cfg = LeagueConfig()
    calc = SGPCalculator(cfg)
    raw_change = {"R": 20.0, "HR": -3.0, "WHIP": -0.05, "L": -1.0}
    weights = {"R": 1.5, "HR": 0.8, "WHIP": 2.0, "L": 1.0, "ERA": 0.5}

    expected = 0.0
    for cat, val in raw_change.items():
        denom = cfg.sgp_denominators.get(cat, 1.0)
        if abs(denom) < 1e-9:
            denom = 1.0
        sign = -1.0 if cat in cfg.inverse_stats else 1.0
        sgp_change = sign * val / denom
        weighted_sgp = sgp_change * weights.get(cat, 1.0)
        expected += weighted_sgp
    actual = calc.totals_sgp(raw_change, weights=weights)
    assert abs(actual - expected) < 1e-9


def test_daily_optimizer_inverse_l_sign_flip():
    """L (Losses) is an inverse counting stat — sign must flip in DCV math."""
    cfg = LeagueConfig()
    calc = SGPCalculator(cfg)
    # 5 losses worth of weighted DCV — bad outcome, negative SGP expected
    actual = calc.totals_sgp({"L": 5.0})
    expected = -5.0 / cfg.sgp_denominators["L"]
    assert actual < 0
    assert abs(actual - expected) < 1e-9


def test_trade_evaluator_zero_denom_skip():
    """Pathological zero denoms must contribute zero, not divide-by-zero."""
    cfg = LeagueConfig()
    cfg.sgp_denominators["R"] = 0.0  # pathological
    calc = SGPCalculator(cfg)
    # No exception; R contributes 0
    actual = calc.totals_sgp({"R": 100.0, "HR": 5.0})
    expected = 5.0 / cfg.sgp_denominators["HR"]
    assert abs(actual - expected) < 1e-9

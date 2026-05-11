"""SGPCalculator.totals_sgp — single source of truth for SGP from totals (SF-25).

These tests pin the contract that the centralized method must satisfy:

* Inverse categories (L, ERA, WHIP) flip sign.
* Counting categories add positively.
* ``weights`` scale per-category contribution multiplicatively.
* Pathological zero-denom categories are SKIPPED (no division by zero).
* Categories absent from the totals dict contribute 0.
* Categories present in totals but absent from ``LeagueConfig.all_categories``
  are silently ignored (no KeyError).

Parity tests live with each migrated call site and confirm the centralized
method produces identical values to the prior local code for the same inputs.
"""

from __future__ import annotations

import pytest

from src.valuation import LeagueConfig, SGPCalculator


@pytest.fixture
def config() -> LeagueConfig:
    return LeagueConfig()


@pytest.fixture
def calc(config: LeagueConfig) -> SGPCalculator:
    return SGPCalculator(config)


# ── Sign flip / direction ───────────────────────────────────────────────


def test_totals_sgp_inverse_sign_flip(config: LeagueConfig, calc: SGPCalculator):
    """ERA total above zero produces negative SGP (lower-is-better)."""
    totals = {cat: 0.0 for cat in config.all_categories}
    totals["ERA"] = 4.5
    sgp = calc.totals_sgp(totals)
    assert sgp < 0, f"Higher ERA must produce negative SGP, got {sgp}"
    expected = -4.5 / config.sgp_denominators["ERA"]
    assert sgp == pytest.approx(expected)


def test_totals_sgp_counting_stat_positive(config: LeagueConfig, calc: SGPCalculator):
    totals = {cat: 0.0 for cat in config.all_categories}
    totals["HR"] = 50.0
    sgp = calc.totals_sgp(totals)
    assert sgp > 0
    assert sgp == pytest.approx(50.0 / config.sgp_denominators["HR"])


def test_totals_sgp_losses_inverse_counting(config: LeagueConfig, calc: SGPCalculator):
    """L is an inverse counting stat — should also flip sign."""
    totals = {cat: 0.0 for cat in config.all_categories}
    totals["L"] = 10.0
    sgp = calc.totals_sgp(totals)
    assert sgp < 0
    assert sgp == pytest.approx(-10.0 / config.sgp_denominators["L"])


def test_totals_sgp_whip_inverse(config: LeagueConfig, calc: SGPCalculator):
    totals = {cat: 0.0 for cat in config.all_categories}
    totals["WHIP"] = 1.30
    sgp = calc.totals_sgp(totals)
    assert sgp < 0


# ── Weights ─────────────────────────────────────────────────────────────


def test_totals_sgp_with_weights_scales(config: LeagueConfig, calc: SGPCalculator):
    """Per-category weight multiplies that category's contribution."""
    totals = {cat: 0.0 for cat in config.all_categories}
    totals["HR"] = 50.0
    base = calc.totals_sgp(totals)
    weighted = calc.totals_sgp(totals, weights={"HR": 2.0})
    assert weighted == pytest.approx(base * 2.0)


def test_totals_sgp_weight_default_one(config: LeagueConfig, calc: SGPCalculator):
    """Categories absent from weights default to 1.0 (no scaling)."""
    totals = {cat: 0.0 for cat in config.all_categories}
    totals["HR"] = 50.0
    totals["R"] = 80.0
    base = calc.totals_sgp(totals)
    # Pass a weights dict that only mentions HR; R should still contribute fully.
    weighted = calc.totals_sgp(totals, weights={"HR": 1.0})
    assert weighted == pytest.approx(base)


def test_totals_sgp_weight_zero_zeros_category(config: LeagueConfig, calc: SGPCalculator):
    """A zero weight should fully zero out a category — used for punt logic."""
    totals = {cat: 0.0 for cat in config.all_categories}
    totals["SB"] = 25.0
    totals["HR"] = 50.0
    sgp_punt_sb = calc.totals_sgp(totals, weights={"SB": 0.0, "HR": 1.0})
    sgp_only_hr = calc.totals_sgp({"HR": 50.0})
    assert sgp_punt_sb == pytest.approx(sgp_only_hr)


def test_totals_sgp_inverse_with_weight(config: LeagueConfig, calc: SGPCalculator):
    """Weighted inverse stat: weight scales the (already negative) contribution."""
    totals = {cat: 0.0 for cat in config.all_categories}
    totals["ERA"] = 4.0
    base = calc.totals_sgp(totals)
    weighted = calc.totals_sgp(totals, weights={"ERA": 0.5})
    assert weighted == pytest.approx(base * 0.5)
    assert weighted < 0


# ── Edge cases ──────────────────────────────────────────────────────────


def test_totals_sgp_zero_denom_safe(config: LeagueConfig):
    """Pathological zero denom: skip the category, do not raise."""
    config.sgp_denominators["HR"] = 0.0
    calc = SGPCalculator(config)
    totals = {cat: 1.0 for cat in config.all_categories}
    result = calc.totals_sgp(totals)
    assert isinstance(result, float)
    # HR contribution skipped; the result should NOT include 1/0.
    # Verify the returned value matches the same totals minus the HR cat.
    config.sgp_denominators["HR"] = 13.0  # restore for next computation
    calc2 = SGPCalculator(config)
    totals_no_hr = {cat: 1.0 for cat in config.all_categories if cat != "HR"}
    expected = calc2.totals_sgp(totals_no_hr)
    assert result == pytest.approx(expected)


def test_totals_sgp_missing_category(config: LeagueConfig, calc: SGPCalculator):
    """Categories absent from totals are treated as 0."""
    sgp = calc.totals_sgp({"HR": 30.0})
    assert sgp == pytest.approx(30.0 / config.sgp_denominators["HR"])


def test_totals_sgp_extra_keys_ignored(config: LeagueConfig, calc: SGPCalculator):
    """Keys in totals not in ``config.all_categories`` are ignored (e.g. raw 'h')."""
    totals = {"HR": 30.0, "h": 150.0, "ab": 500.0, "not_a_cat": 999.0}
    sgp = calc.totals_sgp(totals)
    assert sgp == pytest.approx(30.0 / config.sgp_denominators["HR"])


def test_totals_sgp_none_value_treated_as_zero(config: LeagueConfig, calc: SGPCalculator):
    """A None value (e.g. NaN-ish) should be treated as 0 not blow up."""
    totals = {"HR": None, "R": 100.0}
    sgp = calc.totals_sgp(totals)
    assert sgp == pytest.approx(100.0 / config.sgp_denominators["R"])


def test_totals_sgp_empty_totals(config: LeagueConfig, calc: SGPCalculator):
    """Empty dict returns 0.0."""
    assert calc.totals_sgp({}) == 0.0


def test_totals_sgp_full_roster(config: LeagueConfig, calc: SGPCalculator):
    """Realistic full roster totals produce sensible aggregate SGP."""
    totals = {
        "R": 700,
        "HR": 200,
        "RBI": 700,
        "SB": 100,
        "AVG": 0.265,
        "OBP": 0.330,
        "W": 80,
        "L": 60,
        "SV": 50,
        "K": 1200,
        "ERA": 3.80,
        "WHIP": 1.25,
    }
    sgp = calc.totals_sgp(totals)
    # Sanity: positive contributions should outweigh inverse penalties.
    assert sgp > 0


def test_totals_sgp_uses_constructor_denominators(config: LeagueConfig):
    """The override-denominators arg to SGPCalculator must flow through."""
    custom = dict(config.sgp_denominators)
    custom["HR"] = 26.0  # double the default
    calc = SGPCalculator(config, denominators=custom)
    totals = {cat: 0.0 for cat in config.all_categories}
    totals["HR"] = 26.0
    sgp = calc.totals_sgp(totals)
    assert sgp == pytest.approx(1.0)  # 26 / 26 = 1.0


def test_totals_sgp_weights_only_inverse(config: LeagueConfig, calc: SGPCalculator):
    """When weights dict is passed but only contains an inverse cat, others use 1.0."""
    totals = {cat: 0.0 for cat in config.all_categories}
    totals["HR"] = 50.0
    totals["ERA"] = 4.0
    # weight ERA at 2.0, leave HR at default
    sgp = calc.totals_sgp(totals, weights={"ERA": 2.0})
    expected = (50.0 / config.sgp_denominators["HR"]) + (2.0 * -1 * 4.0 / config.sgp_denominators["ERA"])
    assert sgp == pytest.approx(expected)

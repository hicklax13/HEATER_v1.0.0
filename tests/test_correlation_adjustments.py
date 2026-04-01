"""Tests for category correlation adjustments."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trade_intelligence import (
    CONTACT_CLUSTER_DISCOUNT,
    PITCHING_RATE_DISCOUNT,
    POWER_CLUSTER_DISCOUNT,
    SB_INDEPENDENCE_PREMIUM,
    apply_correlation_adjustments,
)


def test_power_cluster_discount_applied():
    sgp = {"HR": 10.0, "R": 8.0, "RBI": 9.0, "SB": 5.0}
    adj = apply_correlation_adjustments(sgp)
    assert adj["HR"] < 10.0
    assert adj["HR"] == 10.0 * POWER_CLUSTER_DISCOUNT


def test_sb_premium_applied():
    sgp = {"HR": 10.0, "SB": 5.0}
    adj = apply_correlation_adjustments(sgp)
    assert adj["SB"] > 5.0
    assert adj["SB"] == 5.0 * SB_INDEPENDENCE_PREMIUM


def test_contact_cluster_discount():
    sgp = {"AVG": 3.0, "OBP": 2.5}
    adj = apply_correlation_adjustments(sgp)
    assert adj["AVG"] == 3.0 * CONTACT_CLUSTER_DISCOUNT
    assert adj["OBP"] == 2.5 * CONTACT_CLUSTER_DISCOUNT


def test_pitching_rate_discount():
    sgp = {"ERA": -4.0, "WHIP": -3.0}
    adj = apply_correlation_adjustments(sgp)
    assert adj["ERA"] == -4.0 * PITCHING_RATE_DISCOUNT
    assert adj["WHIP"] == -3.0 * PITCHING_RATE_DISCOUNT


def test_unclustered_categories_unchanged():
    sgp = {"W": 5.0, "L": -2.0, "SV": 3.0, "K": 8.0}
    adj = apply_correlation_adjustments(sgp)
    assert adj["W"] == 5.0
    assert adj["L"] == -2.0
    assert adj["SV"] == 3.0
    assert adj["K"] == 8.0


def test_adjustments_preserve_sign():
    sgp = {"HR": -5.0, "SB": -3.0, "ERA": 2.0}
    adj = apply_correlation_adjustments(sgp)
    assert adj["HR"] < 0  # Negative * positive discount = negative
    assert adj["SB"] < 0  # Negative * positive premium = negative
    assert adj["ERA"] > 0  # Positive * positive discount = positive


def test_empty_dict():
    adj = apply_correlation_adjustments({})
    assert adj == {}

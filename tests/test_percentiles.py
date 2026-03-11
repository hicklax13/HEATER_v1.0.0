"""Tests for percentile forecast functions in src/valuation.py."""

import numpy as np
import pandas as pd
import pytest

from src.valuation import (
    add_process_risk,
    compute_percentile_projections,
    compute_projection_volatility,
)

# ── Helpers ──────────────────────────────────────────────────────────


def _make_projection_df(player_id, **stats):
    """Build a single-row projection DataFrame with sensible defaults."""
    row = {
        "player_id": player_id,
        "r": 80,
        "hr": 25,
        "rbi": 75,
        "sb": 10,
        "avg": 0.270,
        "w": 0,
        "sv": 0,
        "k": 0,
        "era": 0.0,
        "whip": 0.0,
    }
    row.update(stats)
    return pd.DataFrame([row])


def _make_multi_system(overrides_by_system: dict[str, dict]) -> dict[str, pd.DataFrame]:
    """Build a dict of DataFrames keyed by system name.

    Each system gets a single player (player_id=1) with the specified
    stat overrides applied on top of a common baseline.
    """
    result = {}
    for sys_name, overrides in overrides_by_system.items():
        result[sys_name] = _make_projection_df(1, **overrides)
    return result


# ── Tests ────────────────────────────────────────────────────────────


def test_volatility_single_system():
    """A single projection system should produce zero volatility."""
    systems = {"steamer": _make_projection_df(1, hr=30, sb=15)}
    vol = compute_projection_volatility(systems)

    assert len(vol) == 1
    assert vol.iloc[0]["player_id"] == 1
    for col in ["r", "hr", "rbi", "sb", "avg", "w", "sv", "k", "era", "whip"]:
        assert vol.iloc[0][col] == 0.0


def test_volatility_multiple_systems():
    """Multiple systems with different projections should produce positive StdDev."""
    systems = _make_multi_system(
        {
            "steamer": {"hr": 20, "sb": 10, "avg": 0.260},
            "zips": {"hr": 30, "sb": 20, "avg": 0.280},
            "depth_charts": {"hr": 25, "sb": 15, "avg": 0.270},
        }
    )
    vol = compute_projection_volatility(systems)

    assert len(vol) == 1
    # HR values are 20, 30, 25 → std > 0
    assert vol.iloc[0]["hr"] > 0
    # SB values are 10, 20, 15 → std > 0
    assert vol.iloc[0]["sb"] > 0
    # AVG values are 0.260, 0.280, 0.270 → std > 0
    assert vol.iloc[0]["avg"] > 0


def test_percentile_bounds():
    """P10 < P50 < P90 for counting stats when volatility is positive."""
    base = _make_projection_df(1, hr=25, rbi=75, sb=12)
    vol = pd.DataFrame(
        [
            {
                "player_id": 1,
                "r": 5,
                "hr": 4,
                "rbi": 8,
                "sb": 3,
                "avg": 0.01,
                "w": 0,
                "sv": 0,
                "k": 0,
                "era": 0,
                "whip": 0,
            }
        ]
    )
    pcts = compute_percentile_projections(base, vol, percentiles=[10, 50, 90])

    for stat in ["hr", "rbi", "sb"]:
        p10_val = pcts[10].iloc[0][stat]
        p50_val = pcts[50].iloc[0][stat]
        p90_val = pcts[90].iloc[0][stat]
        assert p10_val < p50_val, f"P10 should be < P50 for {stat}"
        assert p50_val < p90_val, f"P50 should be < P90 for {stat}"


def test_percentile_floor():
    """Counting stats should never go negative, even at P10 with high volatility."""
    base = _make_projection_df(1, hr=2, sb=1, sv=0)
    vol = pd.DataFrame(
        [
            {
                "player_id": 1,
                "r": 50,
                "hr": 50,
                "rbi": 50,
                "sb": 50,
                "avg": 0,
                "w": 50,
                "sv": 50,
                "k": 50,
                "era": 0,
                "whip": 0,
            }
        ]
    )
    pcts = compute_percentile_projections(base, vol, percentiles=[10])

    for stat in ["r", "hr", "rbi", "sb", "w", "sv", "k"]:
        assert pcts[10].iloc[0][stat] >= 0.0, f"{stat} should be >= 0 at P10"


def test_percentile_rate_bounds():
    """Rate stats should stay within physical bounds even at extreme percentiles."""
    base = _make_projection_df(1, avg=0.270, era=3.50, whip=1.20)
    # Large volatility to push toward limits
    vol = pd.DataFrame(
        [
            {
                "player_id": 1,
                "r": 0,
                "hr": 0,
                "rbi": 0,
                "sb": 0,
                "avg": 0.10,
                "w": 0,
                "sv": 0,
                "k": 0,
                "era": 2.0,
                "whip": 0.50,
            }
        ]
    )
    pcts = compute_percentile_projections(base, vol, percentiles=[10, 90])

    # AVG bounds: [0.150, 0.400]
    assert pcts[10].iloc[0]["avg"] >= 0.150
    assert pcts[90].iloc[0]["avg"] <= 0.400

    # ERA bounds: [1.50, 7.00]
    assert pcts[10].iloc[0]["era"] >= 1.50
    assert pcts[90].iloc[0]["era"] <= 7.00

    # WHIP bounds: [0.80, 2.00]
    assert pcts[10].iloc[0]["whip"] >= 0.80
    assert pcts[90].iloc[0]["whip"] <= 2.00


def test_process_risk_widens_low_correlation():
    """Stats with low year-to-year correlation should get wider intervals."""
    vol = pd.DataFrame(
        [
            {
                "player_id": 1,
                "r": 5.0,
                "hr": 5.0,
                "rbi": 5.0,
                "sb": 5.0,
                "avg": 0.01,
                "w": 5.0,
                "sv": 5.0,
                "k": 5.0,
                "era": 0.2,
                "whip": 0.02,
            }
        ]
    )

    adjusted = add_process_risk(vol)

    # W has correlation 0.30 (low) → adjusted should be wider
    # HR has correlation 0.72 (high) → adjusted should be narrower
    # Both start at 5.0, so adjusted_w > adjusted_hr
    assert adjusted.iloc[0]["w"] > adjusted.iloc[0]["hr"], (
        "Low-correlation stat (W, r=0.30) should have wider adjusted volatility than high-correlation stat (HR, r=0.72)"
    )

    # Specifically: vol / sqrt(0.30) > vol / sqrt(0.72) since sqrt(0.30) < sqrt(0.72)
    expected_w = 5.0 / np.sqrt(0.30)
    expected_hr = 5.0 / np.sqrt(0.72)
    assert np.isclose(adjusted.iloc[0]["w"], expected_w, atol=1e-6)
    assert np.isclose(adjusted.iloc[0]["hr"], expected_hr, atol=1e-6)


def test_process_risk_default_correlations():
    """Default correlation dict should have all 10 standard 5x5 categories."""
    vol = pd.DataFrame(
        [{"player_id": 1, "r": 1, "hr": 1, "rbi": 1, "sb": 1, "avg": 1, "w": 1, "sv": 1, "k": 1, "era": 1, "whip": 1}]
    )
    adjusted = add_process_risk(vol)

    # All 10 categories should be present and adjusted (> 1.0 since all
    # default correlations are < 1.0 → division by sqrt(c) > 1)
    for col in ["r", "hr", "rbi", "sb", "avg", "w", "sv", "k", "era", "whip"]:
        assert adjusted.iloc[0][col] > 1.0, (
            f"Default correlation for {col} should be < 1.0, making adjusted volatility > original"
        )

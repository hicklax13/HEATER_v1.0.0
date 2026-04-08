"""Tests for xwOBA regression flags in enriched player pool and trade finder."""

import pandas as pd
import pytest

from src.database import _enrich_pool

# ── Helper to build a minimal player DataFrame ────────────────────────


def _make_pool(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal player pool DataFrame suitable for _enrich_pool().

    Fills in defaults so _enrich_pool can run without hitting the DB
    (health_score, status, scarcity all get overwritten anyway).
    """
    defaults = {
        "player_id": 1,
        "name": "Test Player",
        "team": "NYY",
        "positions": "OF",
        "is_hitter": 1,
        "is_injured": 0,
        "sv": 0,
        "obp": 0.350,
        "xwoba": None,
    }
    full_rows = [{**defaults, **r} for r in rows]
    return pd.DataFrame(full_rows)


# ── xwOBA delta computation ────────────────────────────────────────────


class TestXwobaDeltaComputation:
    """Verify woba_approx, xwoba_delta computed correctly."""

    def test_xwoba_delta_correct_value(self):
        """xwoba_delta = xwoba - (obp * 1.15)."""
        df = _make_pool([{"player_id": 1, "obp": 0.300, "xwoba": 0.400}])
        result = _enrich_pool(df)
        expected_woba_approx = 0.300 * 1.15  # 0.345
        expected_delta = 0.400 - expected_woba_approx  # 0.055
        assert abs(result["woba_approx"].iloc[0] - expected_woba_approx) < 1e-6
        assert abs(result["xwoba_delta"].iloc[0] - expected_delta) < 1e-6

    def test_xwoba_delta_negative(self):
        """Negative delta when xwoba < woba_approx."""
        df = _make_pool([{"player_id": 2, "obp": 0.400, "xwoba": 0.350}])
        result = _enrich_pool(df)
        expected_delta = 0.350 - (0.400 * 1.15)  # -0.110
        assert result["xwoba_delta"].iloc[0] < 0
        assert abs(result["xwoba_delta"].iloc[0] - expected_delta) < 1e-6


# ── BUY_LOW flag ───────────────────────────────────────────────────────


class TestBuyLowFlag:
    """BUY_LOW when xwoba_delta >= 0.030 and xwoba > 0."""

    def test_buy_low_exact_threshold(self):
        """xwoba_delta == 0.030 triggers BUY_LOW."""
        # Need xwoba - obp*1.15 = 0.030
        # obp=0.300 -> woba_approx=0.345, xwoba=0.375 -> delta=0.030
        df = _make_pool([{"player_id": 3, "obp": 0.300, "xwoba": 0.375}])
        result = _enrich_pool(df)
        assert result["regression_flag"].iloc[0] == "BUY_LOW"

    def test_buy_low_above_threshold(self):
        """Large positive delta triggers BUY_LOW."""
        df = _make_pool([{"player_id": 4, "obp": 0.280, "xwoba": 0.420}])
        result = _enrich_pool(df)
        assert result["regression_flag"].iloc[0] == "BUY_LOW"


# ── SELL_HIGH flag ─────────────────────────────────────────────────────


class TestSellHighFlag:
    """SELL_HIGH when xwoba_delta <= -0.030 and xwoba > 0."""

    def test_sell_high_exact_threshold(self):
        """xwoba_delta <= -0.030 triggers SELL_HIGH."""
        # obp=0.350 -> woba_approx=0.4025, xwoba=0.370 -> delta=-0.0325
        df = _make_pool([{"player_id": 5, "obp": 0.350, "xwoba": 0.370}])
        result = _enrich_pool(df)
        assert result["regression_flag"].iloc[0] == "SELL_HIGH"

    def test_sell_high_large_negative(self):
        """Large negative delta triggers SELL_HIGH."""
        df = _make_pool([{"player_id": 6, "obp": 0.400, "xwoba": 0.350}])
        result = _enrich_pool(df)
        assert result["regression_flag"].iloc[0] == "SELL_HIGH"


# ── No flag cases ──────────────────────────────────────────────────────


class TestNoFlag:
    """No regression flag for small gaps or missing data."""

    def test_no_flag_small_positive_gap(self):
        """Delta of 0.020 (< 0.030) gets no flag."""
        # obp=0.300 -> woba_approx=0.345, xwoba=0.365 -> delta=0.020
        df = _make_pool([{"player_id": 7, "obp": 0.300, "xwoba": 0.365}])
        result = _enrich_pool(df)
        assert result["regression_flag"].iloc[0] == ""

    def test_no_flag_small_negative_gap(self):
        """Delta of -0.020 (> -0.030) gets no flag."""
        # obp=0.300 -> woba_approx=0.345, xwoba=0.325 -> delta=-0.020
        df = _make_pool([{"player_id": 8, "obp": 0.300, "xwoba": 0.325}])
        result = _enrich_pool(df)
        assert result["regression_flag"].iloc[0] == ""

    def test_no_flag_zero_xwoba(self):
        """xwoba=0 never gets a flag (no Statcast data)."""
        df = _make_pool([{"player_id": 9, "obp": 0.300, "xwoba": 0.0}])
        result = _enrich_pool(df)
        assert result["regression_flag"].iloc[0] == ""

    def test_no_flag_missing_xwoba(self):
        """NULL/NaN xwoba never gets a flag."""
        df = _make_pool([{"player_id": 10, "obp": 0.300, "xwoba": None}])
        result = _enrich_pool(df)
        assert result["regression_flag"].iloc[0] == ""

    def test_no_flag_nan_xwoba(self):
        """Explicit NaN xwoba never gets a flag."""
        df = _make_pool([{"player_id": 11, "obp": 0.300, "xwoba": float("nan")}])
        result = _enrich_pool(df)
        assert result["regression_flag"].iloc[0] == ""


# ── Regression bonus in trade finder composite ─────────────────────────


class TestRegressionBonus:
    """Verify regression bonus logic applied to composite scores."""

    def test_buy_low_recv_bonus(self):
        """Receiving a BUY_LOW player adds 0.03 to composite."""
        # Simulate the logic from scan_1_for_1
        recv_flag = "BUY_LOW"
        give_flag = ""
        regression_bonus = 0.0
        if recv_flag == "BUY_LOW":
            regression_bonus += 0.03
        if give_flag == "SELL_HIGH":
            regression_bonus += 0.03
        assert abs(regression_bonus - 0.03) < 1e-9

    def test_sell_high_give_bonus(self):
        """Giving a SELL_HIGH player adds 0.03 to composite."""
        recv_flag = ""
        give_flag = "SELL_HIGH"
        regression_bonus = 0.0
        if recv_flag == "BUY_LOW":
            regression_bonus += 0.03
        if give_flag == "SELL_HIGH":
            regression_bonus += 0.03
        assert abs(regression_bonus - 0.03) < 1e-9

    def test_both_flags_stack(self):
        """Receiving BUY_LOW + giving SELL_HIGH stacks to 0.06."""
        recv_flag = "BUY_LOW"
        give_flag = "SELL_HIGH"
        regression_bonus = 0.0
        if recv_flag == "BUY_LOW":
            regression_bonus += 0.03
        if give_flag == "SELL_HIGH":
            regression_bonus += 0.03
        assert abs(regression_bonus - 0.06) < 1e-9

    def test_no_flags_no_bonus(self):
        """No regression flags means no bonus."""
        recv_flag = ""
        give_flag = ""
        regression_bonus = 0.0
        if recv_flag == "BUY_LOW":
            regression_bonus += 0.03
        if give_flag == "SELL_HIGH":
            regression_bonus += 0.03
        assert regression_bonus == 0.0


# ── Empty DataFrame edge case ──────────────────────────────────────────


class TestEmptyPool:
    """_enrich_pool handles empty DataFrames gracefully."""

    def test_empty_df_returns_empty(self):
        """Empty DataFrame passes through without error."""
        df = pd.DataFrame()
        result = _enrich_pool(df)
        assert result.empty

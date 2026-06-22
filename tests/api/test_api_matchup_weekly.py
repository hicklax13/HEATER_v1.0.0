"""Matchup weekly-scaling: the projected stat line is ROS ÷ weeks_remaining, so a
weekly H2H matchup shows weekly-scale numbers (rates unchanged). Counting stats scale;
omitting `weeks` keeps the raw projection (back-compat for the existing unit tests)."""

import pandas as pd

from api.services.matchup_service import _aggregate_totals, _fmt_hitter_stats, _fmt_pitcher_stats, _weekly_divisor


def test_hitter_stats_weekly_scaled():
    row = {"h": 70, "ab": 280, "r": 56, "hr": 14, "rbi": 56, "sb": 7, "avg": 0.250, "obp": 0.320}
    # weeks=7 → counting ÷7 (h=10, ab=40, r=8, hr=2, rbi=8, sb=1); rates unchanged.
    result = _fmt_hitter_stats(row, weeks=7)
    values = [s.value for s in result]
    assert values == ["10/40", "8", "2", "8", "1", ".250", ".320"]


def test_pitcher_stats_weekly_scaled():
    row = {"ip": 70.0, "w": 7, "l": 0, "sv": 0, "k": 84, "era": 3.00, "whip": 1.10}
    # weeks=7 → ip 10.0, k 12, w 1; era/whip unchanged.
    result = _fmt_pitcher_stats(row, weeks=7)
    values = [s.value for s in result]
    assert values == ["10.0", "1", "0", "0", "12", "3.00", "1.10"]


def test_totals_weekly_scaled():
    rows = pd.DataFrame([{"h": 140, "ab": 560, "r": 112, "hr": 28, "rbi": 112, "sb": 14, "bb": 0, "hbp": 0, "sf": 0}])
    out = _aggregate_totals(rows, hitter=True, weeks=14)
    assert out[0].value == "10/40"  # 140/14 over 560/14
    assert out[1].value == "8" and out[2].value == "2"
    assert out[5].value == ".250"  # AVG = 140/560 — ratio unchanged by scaling


def test_no_weeks_is_raw_backcompat():
    row = {"h": 70, "ab": 280, "r": 56, "hr": 14, "rbi": 56, "sb": 7, "avg": 0.250, "obp": 0.320}
    result = _fmt_hitter_stats(row)
    values = [s.value for s in result]
    assert values == ["70/280", "56", "14", "56", "7", ".250", ".320"]


def test_weekly_divisor_is_weeks_remaining():
    assert _weekly_divisor(12) == 14  # 26-week season, week 12 → 14 left
    assert _weekly_divisor(0) >= 1  # no week → full-season-scale per week, never 0
    assert _weekly_divisor(99) == 1  # past the season → clamp to 1 (raw)

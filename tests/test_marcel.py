"""Tests for Marcel projection system (src/marcel.py).

Covers:
- 3-year weighted averaging (5/4/3)
- PA-weighted rate stat averaging
- Regression toward league mean
- Age adjustment curves (hitters and pitchers)
- Edge cases: missing seasons, zero PA, single season
- Full player projection via project_player_marcel
- Batch projection
- Reliability calculation
"""

import math

import pytest

from src.marcel import (
    HITTER_DECLINE_RATE,
    HITTER_PEAK_AGE,
    LEAGUE_AVERAGES,
    PITCHER_DECLINE_RATE,
    PITCHER_PEAK_AGE,
    RATE_STATS,
    REGRESSION_PA,
    YEAR_WEIGHTS,
    compute_marcel_projection,
    compute_marcel_reliability,
    marcel_age_adjustment,
    project_batch_marcel,
    project_player_marcel,
)

# ── Age Adjustment Tests ──────────────────────────────────────────────


class TestMarcelAgeAdjustment:
    """Tests for marcel_age_adjustment()."""

    def test_peak_age_hitter_returns_one(self):
        """At peak age (27), hitter adjustment should be exactly 1.0."""
        assert marcel_age_adjustment(27, is_hitter=True) == 1.0

    def test_peak_age_pitcher_returns_one(self):
        """At peak age (26), pitcher adjustment should be exactly 1.0."""
        assert marcel_age_adjustment(26, is_hitter=False) == 1.0

    def test_young_hitter_above_one(self):
        """A 24-year-old hitter (3 years before peak) should get a boost."""
        adj = marcel_age_adjustment(24, is_hitter=True)
        assert adj > 1.0
        # 3 * 0.005 = 0.015 improvement
        assert adj == pytest.approx(1.015)

    def test_young_hitter_capped_at_1_02(self):
        """Very young players should have their improvement capped at 2%."""
        adj = marcel_age_adjustment(20, is_hitter=True)
        # 7 years * 0.005 = 0.035, but capped at 0.02
        assert adj == pytest.approx(1.02)

    def test_old_hitter_declines(self):
        """A 32-year-old hitter (5 years past peak) should decline."""
        adj = marcel_age_adjustment(32, is_hitter=True)
        expected = 1.0 - 5 * HITTER_DECLINE_RATE  # 1.0 - 0.025 = 0.975
        assert adj == pytest.approx(expected)

    def test_old_pitcher_declines_faster(self):
        """A 32-year-old pitcher (6 years past peak 26) declines faster than hitter."""
        adj = marcel_age_adjustment(32, is_hitter=False)
        expected = 1.0 - 6 * PITCHER_DECLINE_RATE  # 1.0 - 0.042 = 0.958
        assert adj == pytest.approx(expected)

    def test_pitcher_declines_faster_than_hitter_same_age(self):
        """At age 33, pitchers should have a lower adjustment than hitters."""
        hitter_adj = marcel_age_adjustment(33, is_hitter=True)
        pitcher_adj = marcel_age_adjustment(33, is_hitter=False)
        assert pitcher_adj < hitter_adj


# ── Counting Stat Projection Tests ───────────────────────────────────


class TestCountingStatProjection:
    """Tests for counting stat projections via compute_marcel_projection."""

    def test_three_year_weighted_average(self):
        """Verify 5/4/3 weighting with full regression disabled (high PA)."""
        # With very high PA, reliability ~ 1.0, so projection ~ weighted avg
        hr_values = [30.0, 25.0, 20.0]  # Most recent first
        pa_values = [600.0, 600.0, 600.0]
        proj = compute_marcel_projection(
            historical_stats=hr_values,
            stat="hr",
            pa_values=pa_values,
            is_hitter=True,
        )
        # Weighted avg = (30*5 + 25*4 + 20*3) / (5+4+3) = (150+100+60)/12 = 25.833
        weighted_avg = 25.833333
        # Reliability = 1800 / (1800 + 1200) = 0.6
        reliability = 1800 / (1800 + REGRESSION_PA)
        league_avg = LEAGUE_AVERAGES["hr"]
        expected = reliability * weighted_avg + (1.0 - reliability) * league_avg
        assert proj == pytest.approx(expected, abs=0.01)

    def test_single_season_heavy_regression(self):
        """With only 1 season of data, regression should pull hard toward mean."""
        proj = compute_marcel_projection(
            historical_stats=[40.0],
            stat="hr",
            pa_values=[300.0],
            is_hitter=True,
        )
        # Rate-normalized: rate = 40/300, weighted_rate = 40/300 (single season)
        # Reliability = 300 / (300 + 1200) = 0.2
        # league_avg_rate = 18 / 600 = 0.03
        # regressed_rate = 0.2 * (40/300) + 0.8 * 0.03
        # projected = regressed_rate * 600
        reliability = 300.0 / (300.0 + REGRESSION_PA)
        league_avg = LEAGUE_AVERAGES["hr"]
        league_avg_rate = league_avg / 600.0
        weighted_rate = 40.0 / 300.0
        regressed_rate = reliability * weighted_rate + (1.0 - reliability) * league_avg_rate
        expected = regressed_rate * 600.0
        assert proj == pytest.approx(expected, abs=0.01)

    def test_no_history_returns_league_avg(self):
        """No historical data should return the league average."""
        proj = compute_marcel_projection(
            historical_stats=[],
            stat="hr",
            pa_values=[],
            is_hitter=True,
        )
        assert proj == pytest.approx(LEAGUE_AVERAGES["hr"])

    def test_none_values_skipped(self):
        """None values in history should be skipped gracefully."""
        proj = compute_marcel_projection(
            historical_stats=[25.0, None, 20.0],
            stat="hr",
            pa_values=[500.0, None, 400.0],
            is_hitter=True,
        )
        # Only seasons 0 and 2 are valid (weights 5 and 3)
        # Rate-normalized: rate0 = 25/500 = 0.05, rate2 = 20/400 = 0.05
        # weighted_rate = (0.05*5 + 0.05*3) / 8 = 0.05
        # Reliability = 900 / (900 + 1200) = 0.4286
        # league_avg_rate = 18 / 600 = 0.03
        # regressed_rate = 0.4286 * 0.05 + 0.5714 * 0.03
        # projected = regressed_rate * 600
        reliability = 900.0 / (900.0 + REGRESSION_PA)
        league_avg = LEAGUE_AVERAGES["hr"]
        league_avg_rate = league_avg / 600.0
        rate0 = 25.0 / 500.0
        rate2 = 20.0 / 400.0
        weighted_rate = (rate0 * 5 + rate2 * 3) / (5 + 3)
        regressed_rate = reliability * weighted_rate + (1.0 - reliability) * league_avg_rate
        expected = regressed_rate * 600.0
        assert proj == pytest.approx(expected, abs=0.01)


# ── Rate Stat Projection Tests ───────────────────────────────────────


class TestRateStatProjection:
    """Tests for rate stat projections (PA-weighted averaging)."""

    def test_avg_is_year_weighted(self):
        """AVG projection should use PA-weighted year-weighted averaging with regression."""
        # Season 1: .300 in 600 PA, Season 2: .250 in 200 PA
        proj = compute_marcel_projection(
            historical_stats=[0.300, 0.250],
            stat="avg",
            is_rate=True,
            pa_values=[600.0, 200.0],
            is_hitter=True,
        )
        # PA-weighted year-weighted avg: (0.300*600*5 + 0.250*200*4) / (600*5 + 200*4)
        # = (900 + 200) / (3000 + 800) = 1100 / 3800 = 0.28947
        weighted_avg = (0.300 * 600 * 5 + 0.250 * 200 * 4) / (600 * 5 + 200 * 4)
        # Reliability uses raw PA: 600 + 200 = 800
        reliability = 800.0 / (800.0 + REGRESSION_PA)
        league_avg = LEAGUE_AVERAGES["avg"]
        expected = reliability * weighted_avg + (1.0 - reliability) * league_avg
        assert proj == pytest.approx(expected, abs=0.001)

    def test_era_is_rate_stat(self):
        """ERA should be treated as a rate stat automatically."""
        proj = compute_marcel_projection(
            historical_stats=[3.50, 4.00],
            stat="era",
            pa_values=[180.0, 160.0],
            is_hitter=False,
        )
        # Should use year-weighted averaging, with raw IP for regression
        # Year-weighted: (3.50*5 + 4.00*4) / (5 + 4) = 33.5/9 = 3.7222
        weighted_avg = (3.50 * 5 + 4.00 * 4) / (5 + 4)
        reliability = 340.0 / (340.0 + REGRESSION_PA)
        league_avg = LEAGUE_AVERAGES["era"]
        expected = reliability * weighted_avg + (1.0 - reliability) * league_avg
        assert proj == pytest.approx(expected, abs=0.01)

    def test_rate_stat_zero_pa_returns_league_avg(self):
        """Rate stat with zero PA should return league average."""
        proj = compute_marcel_projection(
            historical_stats=[0.300],
            stat="avg",
            is_rate=True,
            pa_values=[0.0],
            is_hitter=True,
        )
        assert proj == pytest.approx(LEAGUE_AVERAGES["avg"])

    def test_whip_auto_detected_as_rate(self):
        """WHIP should be auto-detected as a rate stat via RATE_STATS."""
        assert "whip" in RATE_STATS
        proj = compute_marcel_projection(
            historical_stats=[1.10, 1.30, 1.20],
            stat="whip",
            pa_values=[200.0, 180.0, 150.0],
            is_hitter=False,
        )
        # Should not crash and should return a reasonable WHIP
        assert 0.8 < proj < 1.6


# ── Full Player Projection Tests ─────────────────────────────────────


class TestProjectPlayerMarcel:
    """Tests for project_player_marcel full stat line projection."""

    def test_hitter_returns_all_hitting_cats(self):
        """Hitter projection should include all 6 hitting categories."""
        hist = [
            {"r": 80, "hr": 25, "rbi": 75, "sb": 12, "avg": 0.270, "obp": 0.340, "pa": 550},
            {"r": 70, "hr": 20, "rbi": 65, "sb": 15, "avg": 0.260, "obp": 0.330, "pa": 500},
        ]
        proj = project_player_marcel(hist, age=27, is_hitter=True)
        for cat in ["r", "hr", "rbi", "sb", "avg", "obp"]:
            assert cat in proj, f"Missing category: {cat}"
        assert "pa" in proj  # PA should also be projected

    def test_pitcher_returns_all_pitching_cats(self):
        """Pitcher projection should include all 6 pitching categories."""
        hist = [
            {"w": 12, "l": 8, "sv": 0, "k": 180, "era": 3.40, "whip": 1.15, "ip": 190},
            {"w": 10, "l": 9, "sv": 0, "k": 160, "era": 3.80, "whip": 1.22, "ip": 175},
        ]
        proj = project_player_marcel(hist, age=26, is_hitter=False)
        for cat in ["w", "l", "sv", "k", "era", "whip"]:
            assert cat in proj, f"Missing category: {cat}"
        assert "ip" in proj

    def test_age_adjustment_applied_to_counting(self):
        """Counting stats should be scaled by age adjustment."""
        hist = [
            {"r": 80, "hr": 30, "rbi": 90, "sb": 10, "avg": 0.280, "obp": 0.350, "pa": 600},
            {"r": 80, "hr": 30, "rbi": 90, "sb": 10, "avg": 0.280, "obp": 0.350, "pa": 600},
            {"r": 80, "hr": 30, "rbi": 90, "sb": 10, "avg": 0.280, "obp": 0.350, "pa": 600},
        ]
        proj_peak = project_player_marcel(hist, age=27, is_hitter=True)
        proj_old = project_player_marcel(hist, age=35, is_hitter=True)
        # Older player should have lower counting stats
        assert proj_old["hr"] < proj_peak["hr"]
        assert proj_old["rbi"] < proj_peak["rbi"]

    def test_age_adjustment_applied_to_rate_stats(self):
        """Rate stats should regress more for older players."""
        hist = [
            {"r": 80, "hr": 30, "rbi": 90, "sb": 10, "avg": 0.300, "obp": 0.380, "pa": 600},
            {"r": 80, "hr": 30, "rbi": 90, "sb": 10, "avg": 0.300, "obp": 0.380, "pa": 600},
            {"r": 80, "hr": 30, "rbi": 90, "sb": 10, "avg": 0.300, "obp": 0.380, "pa": 600},
        ]
        proj_peak = project_player_marcel(hist, age=27, is_hitter=True)
        proj_old = project_player_marcel(hist, age=35, is_hitter=True)
        # AVG above league average should be closer to league avg for older player
        assert proj_old["avg"] < proj_peak["avg"]

    def test_empty_history_returns_league_averages(self):
        """Empty history should give league-average projections (at peak age)."""
        # Use peak age (27) so age adjustment is exactly 1.0
        proj = project_player_marcel([], age=HITTER_PEAK_AGE, is_hitter=True)
        assert proj["hr"] == pytest.approx(LEAGUE_AVERAGES["hr"])
        assert proj["avg"] == pytest.approx(LEAGUE_AVERAGES["avg"])


# ── Reliability Tests ─────────────────────────────────────────────────


class TestMarcelReliability:
    """Tests for the reliability (regression) factor."""

    def test_zero_pa_zero_reliability(self):
        """Zero PA should give zero reliability."""
        assert compute_marcel_reliability(0.0) == 0.0

    def test_negative_pa_zero_reliability(self):
        """Negative PA should give zero reliability."""
        assert compute_marcel_reliability(-100.0) == 0.0

    def test_1200_pa_fifty_percent(self):
        """At exactly 1200 PA, reliability should be 50%."""
        assert compute_marcel_reliability(1200.0) == pytest.approx(0.5)

    def test_high_pa_approaches_one(self):
        """Very high PA should give reliability near 1.0."""
        rel = compute_marcel_reliability(12000.0)
        assert rel > 0.9
        assert rel < 1.0


# ── Batch Projection Tests ───────────────────────────────────────────


class TestBatchProjection:
    """Tests for project_batch_marcel."""

    def test_batch_returns_correct_count(self):
        """Batch projection should return one result per player."""
        players = [
            {
                "age": 27,
                "is_hitter": True,
                "history": [
                    {"r": 80, "hr": 25, "rbi": 70, "sb": 15, "avg": 0.270, "obp": 0.340, "pa": 550},
                ],
            },
            {
                "age": 30,
                "is_hitter": False,
                "history": [
                    {"w": 10, "l": 8, "sv": 0, "k": 170, "era": 3.60, "whip": 1.18, "ip": 185},
                ],
            },
        ]
        results = project_batch_marcel(players)
        assert len(results) == 2
        # First should have hitting stats, second pitching
        assert "hr" in results[0]
        assert "era" in results[1]

    def test_batch_empty_list(self):
        """Empty player list should return empty results."""
        assert project_batch_marcel([]) == []


# ── Constants Tests ───────────────────────────────────────────────────


class TestMarcelConstants:
    """Verify Marcel constants match documented values."""

    def test_year_weights(self):
        """Year weights should be 5/4/3."""
        assert YEAR_WEIGHTS == (5, 4, 3)

    def test_regression_pa(self):
        """Regression PA should be 1200."""
        assert REGRESSION_PA == 1200

    def test_hitter_peak_age(self):
        """Hitter peak age should be 27."""
        assert HITTER_PEAK_AGE == 27

    def test_pitcher_peak_age(self):
        """Pitcher peak age should be 26."""
        assert PITCHER_PEAK_AGE == 26

    def test_rate_stats_set(self):
        """RATE_STATS should contain avg, obp, era, whip."""
        assert RATE_STATS == {"avg", "obp", "era", "whip"}

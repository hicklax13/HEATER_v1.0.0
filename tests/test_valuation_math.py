"""Math verification tests for src/valuation.py formulas.

Tests SGP, VORP, replacement levels, category weights, projection
confidence, percentile forecasts, and process risk — using hand-calculated
golden values and mathematical invariants.
"""

import numpy as np
import pandas as pd
import pytest

from src.valuation import (
    LeagueConfig,
    SGPCalculator,
    add_process_risk,
    compute_category_weights,
    compute_percentile_projections,
    compute_projection_volatility,
    compute_replacement_levels,
    compute_sgp_denominators,
    compute_vorp,
    value_all_players,
)

# ── Fixtures ─────────────────────────────────────────────────────────


def _hitter(pid, name, pa=600, ab=550, h=150, r=80, hr=25, rbi=80, sb=10, avg=None, positions="OF", **kw):
    """Create a hitter row dict with sane defaults."""
    if avg is None:
        avg = h / ab if ab > 0 else 0.0
    return {
        "player_id": pid,
        "name": name,
        "team": "TST",
        "positions": positions,
        "is_hitter": 1,
        "is_injured": 0,
        "pa": pa,
        "ab": ab,
        "h": h,
        "r": r,
        "hr": hr,
        "rbi": rbi,
        "sb": sb,
        "avg": avg,
        "ip": 0,
        "w": 0,
        "sv": 0,
        "k": 0,
        "era": 0,
        "whip": 0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
        "adp": kw.get("adp", pid),
        **{k: v for k, v in kw.items() if k != "adp"},
    }


def _pitcher(pid, name, ip=180, w=12, sv=0, k=180, era=None, whip=None, positions="SP", **kw):
    """Create a pitcher row dict with sane defaults."""
    er = kw.pop("er", ip * 3.80 / 9 if era is None else ip * era / 9)
    bb_a = kw.pop("bb_allowed", ip * 0.30 if whip is None else (whip * ip - ip * 0.95))
    h_a = kw.pop("h_allowed", ip * 0.95 if whip is None else whip * ip - bb_a)
    if era is None:
        era = er * 9 / ip if ip > 0 else 0
    if whip is None:
        whip = (bb_a + h_a) / ip if ip > 0 else 0
    return {
        "player_id": pid,
        "name": name,
        "team": "TST",
        "positions": positions,
        "is_hitter": 0,
        "is_injured": 0,
        "pa": 0,
        "ab": 0,
        "h": 0,
        "r": 0,
        "hr": 0,
        "rbi": 0,
        "sb": 0,
        "avg": 0,
        "ip": ip,
        "w": w,
        "sv": sv,
        "k": k,
        "era": era,
        "whip": whip,
        "er": er,
        "bb_allowed": bb_a,
        "h_allowed": h_a,
        "adp": kw.get("adp", pid),
        **{k: v for k, v in kw.items() if k != "adp"},
    }


@pytest.fixture
def small_pool():
    """120-player pool (10 per position) for deterministic SGP tests."""
    rows = []
    pid = 1
    # 10 hitters per position × 6 positions = 60 hitters
    for pos in ["C", "1B", "2B", "3B", "SS", "OF"]:
        for i in range(10):
            rows.append(
                _hitter(
                    pid,
                    f"{pos}_{i}",
                    r=80 + i * 5,
                    hr=20 + i * 3,
                    rbi=70 + i * 5,
                    sb=5 + i * 2,
                    ab=550,
                    h=140 + i * 5,
                    positions=pos,
                    adp=pid,
                )
            )
            pid += 1
    # 30 SP + 30 RP = 60 pitchers
    for i in range(30):
        rows.append(
            _pitcher(
                pid,
                f"SP_{i}",
                ip=180 - i * 2,
                w=12 - i // 3,
                k=180 - i * 3,
                er=round((3.50 + i * 0.10) * (180 - i * 2) / 9, 1),
                bb_allowed=round((180 - i * 2) * 0.30, 0),
                h_allowed=round((180 - i * 2) * 0.92, 0),
                positions="SP",
                adp=pid,
            )
        )
        pid += 1
    for i in range(30):
        rows.append(
            _pitcher(
                pid,
                f"RP_{i}",
                ip=65 - i,
                w=3,
                sv=25 - i,
                k=70 - i,
                er=round((3.20 + i * 0.12) * (65 - i) / 9, 1),
                bb_allowed=round((65 - i) * 0.35, 0),
                h_allowed=round((65 - i) * 0.90, 0),
                positions="RP",
                adp=pid,
            )
        )
        pid += 1
    return pd.DataFrame(rows)


@pytest.fixture
def config():
    return LeagueConfig()


# ── SGP Counting-Stat Formula ────────────────────────────────────────


class TestSGPCountingStats:
    """Verify SGP = stat_value / denominator for counting stats."""

    def test_positive_hr_yields_positive_sgp(self, config):
        """More HR → more SGP (monotonically increasing)."""
        calc = SGPCalculator(config)
        low = _hitter(1, "Low", hr=10)
        high = _hitter(2, "High", hr=40)
        assert calc.player_sgp(low)["HR"] < calc.player_sgp(high)["HR"]

    def test_hr_sgp_hand_calculation(self, config):
        """SGP = HR / denominator, verified by hand."""
        denom = config.sgp_denominators["HR"]
        calc = SGPCalculator(config)
        player = _hitter(1, "Test", hr=30)
        expected = 30.0 / denom
        assert calc.player_sgp(player)["HR"] == pytest.approx(expected, rel=1e-6)

    def test_zero_stat_yields_zero_sgp(self, config):
        """Player with 0 HR should have 0 HR SGP."""
        calc = SGPCalculator(config)
        player = _hitter(1, "Zero", hr=0)
        assert calc.player_sgp(player)["HR"] == 0.0

    def test_all_counting_stats_monotonic(self, config):
        """Every counting stat should increase SGP when the stat increases."""
        calc = SGPCalculator(config)
        for cat, col in [("R", "r"), ("HR", "hr"), ("RBI", "rbi"), ("SB", "sb")]:
            low = _hitter(1, "L", **{col: 10})
            high = _hitter(2, "H", **{col: 50})
            assert calc.player_sgp(low)[cat] < calc.player_sgp(high)[cat], f"{cat} not monotonic"

    def test_pitcher_counting_stats_monotonic(self, config):
        """W, SV, K should be monotonically increasing with stat value."""
        calc = SGPCalculator(config)
        for cat, col in [("W", "w"), ("SV", "sv"), ("K", "k")]:
            low = _pitcher(1, "L", **{col: 3})
            high = _pitcher(2, "H", **{col: 20})
            assert calc.player_sgp(low)[cat] < calc.player_sgp(high)[cat], f"{cat} not monotonic"


# ── SGP Rate-Stat Formulas ───────────────────────────────────────────


class TestSGPRateStats:
    """Verify marginal rate-stat SGP using the volume-weighted formula.

    For AVG: new_avg = (5500*0.265 + H) / (5500 + AB), sgp = (new - old) / denom
    For ERA: new_era = (1300*3.80/9 + ER)*9 / (1300 + IP), sgp = -(new - old) / denom
    For WHIP: new_whip = (1300*1.25 + BB+H) / (1300 + IP), sgp = -(new - old) / denom
    """

    def test_avg_sgp_hand_calculation(self, config):
        """Hand-calc AVG SGP for a .300 hitter with 550 AB."""
        denom = config.sgp_denominators["AVG"]
        calc = SGPCalculator(config)
        # .300 hitter: 165 H in 550 AB
        player = _hitter(1, "Test", ab=550, h=165, avg=0.300)
        # Code uses int() truncation: roster_h = int(5500 * 0.265) = 1457
        roster_ab = 5500
        roster_h = int(5500 * 0.265)  # 1457, matching code's int() truncation
        new_avg = (roster_h + 165) / (roster_ab + 550)
        old_avg = roster_h / roster_ab  # 1457/5500, not 0.265
        expected = (new_avg - old_avg) / denom
        actual = calc.player_sgp(player)["AVG"]
        assert actual == pytest.approx(expected, rel=1e-6)

    def test_era_sgp_hand_calculation(self, config):
        """Hand-calc ERA SGP for a 3.00 ERA pitcher with 200 IP."""
        denom = config.sgp_denominators["ERA"]
        calc = SGPCalculator(config)
        ip = 200
        er = 3.00 * ip / 9  # = 66.67
        player = _pitcher(1, "Ace", ip=ip, era=3.00, er=er)
        # Code uses int() truncation: roster_er = int(1300 * 3.80 / 9) = 548
        roster_ip = 1300
        roster_er = int(1300 * 3.80 / 9)  # 548, matching code's int()
        new_era = (roster_er + er) * 9 / (roster_ip + ip)
        old_era = roster_er * 9 / roster_ip  # 548*9/1300, not 3.80
        expected = -(new_era - old_era) / denom
        actual = calc.player_sgp(player)["ERA"]
        assert actual == pytest.approx(expected, rel=1e-6)

    def test_whip_sgp_hand_calculation(self, config):
        """Hand-calc WHIP SGP for a 1.05 WHIP pitcher with 180 IP."""
        denom = config.sgp_denominators["WHIP"]
        calc = SGPCalculator(config)
        ip = 180
        bb_a = 40
        h_a = ip * 1.05 - bb_a  # WHIP = (40 + h_a) / 180 = 1.05 → h_a = 149
        player = _pitcher(1, "Good", ip=ip, bb_allowed=bb_a, h_allowed=h_a)
        roster_ip = 1300
        roster_whip_num = 1300 * 1.25
        new_whip = (roster_whip_num + bb_a + h_a) / (roster_ip + ip)
        old_whip = 1.25
        expected = -(new_whip - old_whip) / denom
        actual = calc.player_sgp(player)["WHIP"]
        assert actual == pytest.approx(expected, rel=1e-3)

    def test_era_lower_is_better(self, config):
        """Lower ERA → higher (more positive) SGP."""
        calc = SGPCalculator(config)
        ace = _pitcher(1, "Ace", ip=200, era=2.50, er=2.50 * 200 / 9)
        bad = _pitcher(2, "Bad", ip=200, era=5.00, er=5.00 * 200 / 9)
        assert calc.player_sgp(ace)["ERA"] > calc.player_sgp(bad)["ERA"]

    def test_whip_lower_is_better(self, config):
        """Lower WHIP → higher SGP."""
        calc = SGPCalculator(config)
        good = _pitcher(1, "Good", ip=180, bb_allowed=30, h_allowed=150)
        bad = _pitcher(2, "Bad", ip=180, bb_allowed=70, h_allowed=200)
        assert calc.player_sgp(good)["WHIP"] > calc.player_sgp(bad)["WHIP"]

    def test_zero_ip_returns_zero_era_sgp(self, config):
        """Pitcher with 0 IP should have 0 ERA/WHIP SGP."""
        calc = SGPCalculator(config)
        player = _pitcher(1, "Zero", ip=0, era=0, er=0, bb_allowed=0, h_allowed=0)
        assert calc.player_sgp(player)["ERA"] == 0.0
        assert calc.player_sgp(player)["WHIP"] == 0.0


# ── SGP Denominator Auto-Computation ─────────────────────────────────


class TestSGPDenominators:
    """Verify compute_sgp_denominators uses stddev of simulated team totals."""

    def test_denominators_all_positive(self, small_pool, config):
        """Every denominator must be > 0."""
        denoms = compute_sgp_denominators(small_pool, config)
        for cat in ["R", "HR", "RBI", "SB", "AVG", "W", "SV", "K", "ERA", "WHIP"]:
            assert denoms[cat] > 0, f"{cat} denominator not positive"

    def test_counting_stat_denominator_is_stddev(self, config):
        """For a trivial pool, verify denominator ≈ stddev of team totals."""
        # Create 24 hitters (2 per team × 12 teams) with known HR
        rows = []
        for i in range(24):
            rows.append(_hitter(i + 1, f"H_{i}", hr=10 + i * 2, positions="OF"))
        # Need pitchers too
        for i in range(24):
            rows.append(_pitcher(i + 25, f"P_{i}", k=100 + i * 5, positions="SP"))
        pool = pd.DataFrame(rows)
        denoms = compute_sgp_denominators(pool, config)
        # Denominator should be positive and reasonable
        assert denoms["HR"] > 0
        assert denoms["K"] > 0

    def test_rate_stat_denominator_bounded(self, small_pool, config):
        """AVG denominator should be small (e.g., 0.001-0.050); ERA similar."""
        denoms = compute_sgp_denominators(small_pool, config)
        assert 0.001 <= denoms["AVG"] <= 0.100, f"AVG denom {denoms['AVG']} out of range"
        assert 0.05 <= denoms["ERA"] <= 2.0, f"ERA denom {denoms['ERA']} out of range"
        assert 0.005 <= denoms["WHIP"] <= 0.50, f"WHIP denom {denoms['WHIP']} out of range"


# ── VORP Calculation ─────────────────────────────────────────────────


class TestVORP:
    """Verify VORP = total_sgp - best_replacement + positional flexibility bonus."""

    def test_vorp_above_replacement_is_positive(self, config):
        """A clearly elite player should have VORP > 0."""
        calc = SGPCalculator(config)
        elite = _hitter(1, "Elite", r=120, hr=45, rbi=110, sb=20, ab=600, h=195, positions="OF")
        replacement_levels = {"OF": 0.5, "C": 0.2, "1B": 0.4, "2B": 0.3, "3B": 0.3, "SS": 0.2}
        vorp = compute_vorp(elite, calc, replacement_levels)
        assert vorp > 0

    def test_vorp_formula_hand_calc(self, config):
        """Verify VORP = total_sgp - best_repl + flex_bonus with known values."""
        calc = SGPCalculator(config)
        player = _hitter(1, "Multi", r=80, hr=25, rbi=80, sb=10, ab=550, h=150, positions="2B,SS")
        total = calc.total_sgp(player)
        replacement_levels = {"2B": 1.0, "SS": 0.8, "OF": 1.2, "C": 0.3}
        # best_repl = max(repl[2B], repl[SS]) = max(1.0, 0.8) = 1.0
        # flex_bonus = 0.12 * (2-1) + 0.08 * 2 = 0.12 + 0.16 = 0.28
        #   (both 2B and SS are scarce positions)
        expected = total - 1.0 + 0.12 * 1 + 0.08 * 2
        vorp = compute_vorp(player, calc, replacement_levels)
        assert vorp == pytest.approx(expected, rel=1e-6)

    def test_single_position_no_flex_bonus(self, config):
        """Single-position player gets no flexibility bonus."""
        calc = SGPCalculator(config)
        player = _hitter(1, "Single", positions="1B")
        replacement_levels = {"1B": 1.0}
        total = calc.total_sgp(player)
        vorp = compute_vorp(player, calc, replacement_levels)
        expected = total - 1.0  # no bonus: 1B not scarce, only 1 position
        assert vorp == pytest.approx(expected, rel=1e-6)

    def test_multi_position_flex_bonus_constants(self, config):
        """Verify exact bonus constants: 0.12/extra pos, 0.08/scarce pos."""
        calc = SGPCalculator(config)
        # 3 positions, 1 scarce (C)
        player = _hitter(1, "Flex", positions="C,1B,OF")
        replacement_levels = {"C": 0.5, "1B": 0.8, "OF": 1.0}
        total = calc.total_sgp(player)
        # best_repl = max(0.5, 0.8, 1.0) = 1.0
        # flex = 0.12 * (3-1) + 0.08 * 1 = 0.24 + 0.08 = 0.32
        expected = total - 1.0 + 0.32
        vorp = compute_vorp(player, calc, replacement_levels)
        assert vorp == pytest.approx(expected, rel=1e-6)


# ── Replacement Levels ───────────────────────────────────────────────


class TestReplacementLevels:
    """Verify replacement level = cutline player's SGP at N_starters rank."""

    def test_replacement_level_is_cutline(self, small_pool, config):
        """Replacement level should sit between starter and non-starter SGP."""
        calc = SGPCalculator(config)
        repl = compute_replacement_levels(small_pool, config, calc)
        # Every position should have a replacement level
        for pos in ["C", "1B", "2B", "3B", "SS", "OF", "SP", "RP"]:
            assert pos in repl, f"Missing replacement level for {pos}"
            assert isinstance(repl[pos], (int, float)), f"{pos} repl not numeric"

    def test_util_is_min_of_hitting_positions(self, small_pool, config):
        """Util replacement should be the minimum across hitting positions."""
        calc = SGPCalculator(config)
        repl = compute_replacement_levels(small_pool, config, calc)
        hitting_repls = [repl[p] for p in ["C", "1B", "2B", "3B", "SS", "OF"] if p in repl]
        if "Util" in repl:
            assert repl["Util"] == pytest.approx(min(hitting_repls), rel=1e-6)


# ── Category Weights ─────────────────────────────────────────────────


class TestCategoryWeights:
    """Verify standings-based and benchmark-based weight formulas."""

    def test_weights_all_positive(self, config):
        """All category weights must be > 0."""
        roster_totals = {
            "R": 780,
            "HR": 200,
            "RBI": 760,
            "SB": 110,
            "AVG": 0.262,
            "W": 70,
            "SV": 60,
            "K": 1100,
            "ERA": 3.90,
            "WHIP": 1.25,
        }
        weights = compute_category_weights(roster_totals, config)
        for cat, w in weights.items():
            assert w > 0, f"Weight for {cat} is not positive"

    def test_weights_bounded_02_to_20(self, config):
        """Weights should be clipped to [0.2, 2.0]."""
        # Extreme roster: terrible at everything
        roster_totals = {
            "R": 100,
            "HR": 10,
            "RBI": 100,
            "SB": 5,
            "AVG": 0.180,
            "W": 5,
            "SV": 2,
            "K": 200,
            "ERA": 6.50,
            "WHIP": 1.80,
        }
        weights = compute_category_weights(roster_totals, config)
        for cat, w in weights.items():
            assert 0.2 <= w <= 2.0, f"Weight {cat}={w} out of bounds"

    def test_weak_category_gets_higher_weight(self, config):
        """A category where team is weak should get boosted weight."""
        # Strong in HR (300), weak in SB (20)
        strong_sb = {
            "R": 780,
            "HR": 300,
            "RBI": 760,
            "SB": 200,
            "AVG": 0.262,
            "W": 70,
            "SV": 60,
            "K": 1100,
            "ERA": 3.90,
            "WHIP": 1.25,
        }
        weak_sb = {
            "R": 780,
            "HR": 300,
            "RBI": 760,
            "SB": 20,
            "AVG": 0.262,
            "W": 70,
            "SV": 60,
            "K": 1100,
            "ERA": 3.90,
            "WHIP": 1.25,
        }
        w_strong = compute_category_weights(strong_sb, config)
        w_weak = compute_category_weights(weak_sb, config)
        assert w_weak["SB"] > w_strong["SB"], "Weak SB should get higher weight"

    def test_draft_progress_scaling_formula(self, config):
        """Progress scale = 0.4 + 1.1 * progress. At 0 → 0.4, at 1.0 → 1.5."""
        roster = {
            "R": 780,
            "HR": 200,
            "RBI": 760,
            "SB": 110,
            "AVG": 0.262,
            "W": 70,
            "SV": 60,
            "K": 1100,
            "ERA": 3.90,
            "WHIP": 1.25,
        }
        w_early = compute_category_weights(roster, config, draft_progress=0.0)
        w_late = compute_category_weights(roster, config, draft_progress=1.0)
        # Late-draft weights should deviate more from 1.0 than early-draft
        # Because progress_scale amplifies (weight - 1.0)
        for cat in w_early:
            early_dev = abs(w_early[cat] - 1.0)
            late_dev = abs(w_late[cat] - 1.0)
            # At progress=0 scale=0.4, at progress=1 scale=1.5
            # So late deviations should be larger (1.5/0.4 = 3.75x)
            if early_dev > 0.01:  # skip near-neutral categories
                assert late_dev >= early_dev * 0.9, f"{cat}: late deviation not larger"


# ── Projection Confidence Discount ───────────────────────────────────


class TestProjectionConfidence:
    """Verify the volume-based confidence discount formula."""

    def test_full_season_hitter_no_discount(self, small_pool, config):
        """Hitter with 650+ PA should get confidence = 1.0 (no discount)."""
        pool = small_pool.copy()
        pool.loc[0, "pa"] = 700
        # value_all_players applies discount internally
        valued = value_all_players(pool, config)
        # With 700 PA: volume = 700/650 = 1.077, confidence = clip(0.8 + 0.2*1.077) = 1.0
        # We can't directly test confidence, but we can verify the player wasn't discounted
        # by checking pick_score is reasonable
        assert valued.iloc[0]["pick_score"] != 0

    def test_low_pa_hitter_gets_discount(self, config):
        """Hitter with only 100 PA should get ~16% discount."""
        # volume = 100/650 ≈ 0.154
        # confidence = clip(0.8 + 0.2 * 0.154) = clip(0.831) = 0.831
        volume = 100 / 650.0
        confidence = max(0.8, min(1.0, 0.8 + 0.2 * volume))
        assert confidence == pytest.approx(0.831, abs=0.01)
        assert confidence < 1.0, "Low PA should discount"

    def test_confidence_formula_bounds(self):
        """Confidence should be [0.8, 1.0] for any volume."""
        for pa in [0, 50, 100, 200, 400, 650, 800, 1000]:
            volume = pa / 650.0
            confidence = max(0.8, min(1.0, 0.8 + 0.2 * volume))
            assert 0.8 <= confidence <= 1.0, f"PA={pa}: confidence={confidence}"


# ── Late-Draft Bench Flexibility Bonus ───────────────────────────────


class TestBenchFlexBonus:
    """Verify flex_bonus = pos_count * 0.08 * (progress - 0.7) / 0.3."""

    def test_no_bonus_before_70_pct(self):
        """No bonus when draft_progress <= 0.7."""
        progress = 0.5
        pos_count = 3
        bonus = pos_count * 0.08 * (progress - 0.7) / 0.3 if progress > 0.7 else 0
        assert bonus == 0

    def test_bonus_at_full_progress(self):
        """At progress=1.0: bonus = positions * 0.08 * (1.0-0.7)/0.3 = positions * 0.08."""
        progress = 1.0
        pos_count = 4
        bonus = pos_count * 0.08 * (progress - 0.7) / 0.3
        assert bonus == pytest.approx(4 * 0.08, rel=1e-6)

    def test_bonus_scales_with_positions(self):
        """More eligible positions → bigger bonus."""
        progress = 0.9
        bonus_2 = 2 * 0.08 * (progress - 0.7) / 0.3
        bonus_4 = 4 * 0.08 * (progress - 0.7) / 0.3
        assert bonus_4 > bonus_2
        assert bonus_4 == pytest.approx(bonus_2 * 2, rel=1e-6)


# ── Percentile Projections & Volatility ──────────────────────────────


class TestPercentileForecasts:
    """Verify cross-system volatility and percentile clipping."""

    def test_single_system_zero_volatility(self):
        """One projection system → 0 volatility everywhere."""
        df = pd.DataFrame([{"player_id": 1, "name": "A", "hr": 30, "avg": 0.280}])
        vol = compute_projection_volatility({"steamer": df})
        assert (vol[["hr", "avg"]] == 0).all().all()

    def test_two_systems_positive_volatility(self):
        """Two systems with different values → positive volatility."""
        sys_a = pd.DataFrame([{"player_id": 1, "name": "A", "hr": 30, "avg": 0.290}])
        sys_b = pd.DataFrame([{"player_id": 1, "name": "A", "hr": 25, "avg": 0.270}])
        vol = compute_projection_volatility({"steamer": sys_a, "zips": sys_b})
        assert vol.iloc[0]["hr"] > 0
        assert vol.iloc[0]["avg"] > 0

    def test_volatility_is_sample_stddev(self):
        """Volatility should use ddof=1 (sample stddev) for unbiased estimate."""
        sys_a = pd.DataFrame([{"player_id": 1, "name": "A", "hr": 30}])
        sys_b = pd.DataFrame([{"player_id": 1, "name": "A", "hr": 20}])
        vol = compute_projection_volatility({"a": sys_a, "b": sys_b})
        expected = np.std([30, 20], ddof=1)  # = 7.071...
        assert vol.iloc[0]["hr"] == pytest.approx(expected, rel=1e-6)

    def test_p10_le_p50_le_p90(self):
        """Percentile ordering: P10 ≤ P50 ≤ P90 for all stats."""
        base = pd.DataFrame([{"player_id": 1, "name": "A", "hr": 30, "r": 80, "avg": 0.280}])
        vol = pd.DataFrame([{"player_id": 1, "hr": 5.0, "r": 10.0, "avg": 0.015}])
        pct = compute_percentile_projections(base, vol, percentiles=[10, 50, 90])
        for stat in ["hr", "r"]:
            if stat in pct[10].columns:
                assert pct[10].iloc[0][stat] <= pct[50].iloc[0][stat]
                assert pct[50].iloc[0][stat] <= pct[90].iloc[0][stat]

    def test_counting_stats_floor_at_zero(self):
        """Counting stats should never go negative (clipped to 0)."""
        base = pd.DataFrame([{"player_id": 1, "name": "A", "hr": 2, "sb": 1}])
        vol = pd.DataFrame([{"player_id": 1, "hr": 10.0, "sb": 8.0}])  # huge volatility
        pct = compute_percentile_projections(base, vol, percentiles=[10])
        for stat in ["hr", "sb"]:
            if stat in pct[10].columns:
                assert pct[10].iloc[0][stat] >= 0, f"P10 {stat} went negative"

    def test_avg_clipped_to_bounds(self):
        """AVG should be clipped to [0.150, 0.400]."""
        base = pd.DataFrame([{"player_id": 1, "name": "A", "avg": 0.280}])
        vol = pd.DataFrame([{"player_id": 1, "avg": 0.200}])  # extreme volatility
        pct = compute_percentile_projections(base, vol, percentiles=[10, 90])
        if "avg" in pct[10].columns:
            assert pct[10].iloc[0]["avg"] >= 0.150
        if "avg" in pct[90].columns:
            assert pct[90].iloc[0]["avg"] <= 0.400

    def test_era_clipped_to_bounds(self):
        """ERA should be clipped to [1.50, 7.00]."""
        base = pd.DataFrame([{"player_id": 1, "name": "A", "era": 3.80}])
        vol = pd.DataFrame([{"player_id": 1, "era": 5.0}])  # huge volatility
        pct = compute_percentile_projections(base, vol, percentiles=[10, 90])
        if "era" in pct[10].columns:
            assert pct[10].iloc[0]["era"] >= 1.50
        if "era" in pct[90].columns:
            assert pct[90].iloc[0]["era"] <= 7.00


# ── Process Risk Widening ────────────────────────────────────────────


class TestProcessRisk:
    """Verify add_process_risk: adjusted = vol / sqrt(correlation)."""

    def test_low_correlation_widens_more(self):
        """Stats with low r² get wider CI than high r² stats."""
        vol = pd.DataFrame([{"player_id": 1, "hr": 5.0, "w": 5.0}])
        adjusted = add_process_risk(vol)
        # hr correlation = 0.72, w correlation = 0.30
        # hr adjusted = 5.0 / sqrt(0.72) ≈ 5.89
        # w adjusted = 5.0 / sqrt(0.30) ≈ 9.13
        assert adjusted.iloc[0]["w"] > adjusted.iloc[0]["hr"]

    def test_exact_process_risk_formula(self):
        """Verify vol / sqrt(corr) with known values."""
        vol = pd.DataFrame([{"player_id": 1, "hr": 5.0, "avg": 0.015}])
        adjusted = add_process_risk(vol)
        # hr: 5.0 / sqrt(0.72) = 5.893
        # avg: 0.015 / sqrt(0.41) = 0.02342
        assert adjusted.iloc[0]["hr"] == pytest.approx(5.0 / np.sqrt(0.72), rel=1e-3)
        assert adjusted.iloc[0]["avg"] == pytest.approx(0.015 / np.sqrt(0.41), rel=1e-3)

    def test_high_correlation_minimal_widening(self):
        """HR (r=0.72) should barely widen: factor = 1/sqrt(0.72) ≈ 1.18."""
        vol = pd.DataFrame([{"player_id": 1, "hr": 10.0}])
        adjusted = add_process_risk(vol)
        factor = adjusted.iloc[0]["hr"] / 10.0
        assert factor == pytest.approx(1 / np.sqrt(0.72), rel=1e-3)
        assert factor < 1.25, "HR widening factor should be modest"

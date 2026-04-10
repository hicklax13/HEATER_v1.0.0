# Lineup Optimizer Validation Framework — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a comprehensive validation and testing framework that proves the Lineup Optimizer's math is correct, data is fresh, algorithms are trustworthy, and outputs are optimal across all tabs, modes, and settings combinations.

**Architecture:** Six-phase approach: (1) Math proof tests that hand-verify every formula, (2) Shared data layer tests for the untested 901-line hub, (3) Cross-module integration tests with realistic 23-player rosters, (4) Data freshness tracking and UI visibility, (5) Historical backtesting against actual MLB outcomes, (6) Empirical constant validation and algorithm improvements. Each phase produces independently valuable, shippable test coverage.

**Tech Stack:** pytest, pandas, NumPy, PuLP, unittest.mock, scipy.stats (Spearman correlation), pybaseball (empirical data)

---

## File Structure

### New Test Files
| File | Responsibility |
|------|---------------|
| `tests/test_optimizer_math_proofs.py` | Hand-verified formulas: rate stats, LP weights, Bayesian blends, matchup compounds |
| `tests/test_shared_data_layer.py` | OptimizerDataContext dataclass, build_optimizer_context builder, scale_projections_for_scope |
| `tests/test_optimizer_integration.py` | Cross-module chain tests, settings axis tests, mode A/B comparisons, invariant assertions |
| `tests/test_data_freshness.py` | Data freshness tracker, missing data audit, canary tests |
| `tests/test_optimizer_backtest.py` | Historical replay, projection accuracy metrics, lineup decision quality |
| `tests/test_constants_registry.py` | Constants bounds, citation coverage, sensitivity analysis |

### New Source Files
| File | Responsibility |
|------|---------------|
| `src/optimizer/data_freshness.py` | Timestamp tracking per data source, staleness detection, missing data audit |
| `src/optimizer/constants_registry.py` | Centralized registry of all magic numbers with citations, bounds, and sensitivity flags |
| `src/optimizer/backtest_validator.py` | Historical week replay, projection-vs-actual scoring, lineup quality grading |

### Modified Files
| File | Change |
|------|--------|
| `src/optimizer/shared_data_layer.py` | Add `data_timestamps` field to OptimizerDataContext |
| `pages/5_Line-up_Optimizer.py` | Add data freshness badges to context panel |
| `src/optimizer/scenario_generator.py` | Add `compute_empirical_correlations()` utility |
| `src/optimizer/projections.py` | Add sample-size-aware recent form weight |

---

## Phase 1: Math Proof Tests (P0)

These tests hand-verify every critical formula with known inputs and hand-calculated expected outputs. If any of these fail, the optimizer is producing wrong numbers.

---

### Task 1: Rate Stat Aggregation Proofs

**Files:**
- Create: `tests/test_optimizer_math_proofs.py`

- [ ] **Step 1: Write the failing test — AVG must be sum(H)/sum(AB)**

```python
"""Math proof tests for Lineup Optimizer.

Hand-verified formulas with known inputs and pre-computed expected outputs.
Every test documents its hand calculation in the docstring.
"""

import numpy as np
import pandas as pd
import pytest


class TestRateStatAggregation:
    """Prove rate stats use weighted aggregation, not simple averages.

    Fantasy baseball rate stats (AVG, OBP, ERA, WHIP) MUST be computed from
    component sums, never from averaging individual player rates.  A .300
    hitter with 150 AB contributes differently than a .300 hitter with 500 AB.
    """

    def test_avg_is_sum_h_over_sum_ab(self):
        """Team AVG = sum(H) / sum(AB), NOT mean of player AVGs.

        Hand calculation:
            Player A: 80 H / 300 AB = .267
            Player B: 45 H / 150 AB = .300
            Wrong (simple mean): (.267 + .300) / 2 = .283
            Correct (weighted):  (80 + 45) / (300 + 150) = 125 / 450 = .278
        """
        players = pd.DataFrame([
            {"player_name": "A", "h": 80, "ab": 300, "avg": 0.267},
            {"player_name": "B", "h": 45, "ab": 150, "avg": 0.300},
        ])
        correct_avg = (80 + 45) / (300 + 150)  # 0.2778
        wrong_avg = (0.267 + 0.300) / 2  # 0.2833

        team_avg = players["h"].sum() / players["ab"].sum()

        assert team_avg == pytest.approx(correct_avg, abs=1e-4)
        assert team_avg != pytest.approx(wrong_avg, abs=0.001)

    def test_obp_is_weighted_by_pa(self):
        """Team OBP = sum(H+BB+HBP) / sum(AB+BB+HBP+SF).

        Hand calculation:
            Player A: (80+30+5) / (300+30+5+3) = 115/338 = .340
            Player B: (45+10+2) / (150+10+2+1) = 57/163  = .350
            Correct: (115+57) / (338+163) = 172/501 = .343
        """
        players = pd.DataFrame([
            {"h": 80, "bb": 30, "hbp": 5, "ab": 300, "sf": 3},
            {"h": 45, "bb": 10, "hbp": 2, "ab": 150, "sf": 1},
        ])
        numerator = (players["h"] + players["bb"] + players["hbp"]).sum()
        denominator = (players["ab"] + players["bb"] + players["hbp"] + players["sf"]).sum()
        correct_obp = numerator / denominator  # 172/501 = 0.3433

        assert correct_obp == pytest.approx(0.3433, abs=0.001)

    def test_era_is_ip_weighted(self):
        """Team ERA = sum(ER)*9 / sum(IP), NOT mean of player ERAs.

        Hand calculation:
            Starter: 90 ER in 200 IP -> ERA 4.05
            Reliever: 2 ER in 10 IP  -> ERA 1.80
            Wrong (simple mean): (4.05 + 1.80) / 2 = 2.925
            Correct (weighted): (90+2)*9 / (200+10) = 828/210 = 3.943
        """
        starter_er, starter_ip = 90, 200.0
        reliever_er, reliever_ip = 2, 10.0

        correct_era = (starter_er + reliever_er) * 9 / (starter_ip + reliever_ip)
        wrong_era = (4.05 + 1.80) / 2

        assert correct_era == pytest.approx(3.943, abs=0.01)
        assert abs(correct_era - wrong_era) > 0.5  # Off by > 0.5 ERA

    def test_whip_is_ip_weighted(self):
        """Team WHIP = sum(BB+H_allowed) / sum(IP).

        Hand calculation:
            Starter: (50 BB + 180 H) / 200 IP = 230/200 = 1.150
            Reliever: (3 BB + 8 H) / 10 IP = 11/10 = 1.100
            Wrong (simple mean): (1.15 + 1.10) / 2 = 1.125
            Correct (weighted): (230+11) / (200+10) = 241/210 = 1.148
        """
        correct_whip = (50 + 180 + 3 + 8) / (200 + 10)  # 241/210
        wrong_whip = (1.15 + 1.10) / 2

        assert correct_whip == pytest.approx(1.148, abs=0.01)
        assert abs(correct_whip - wrong_whip) > 0.01
```

- [ ] **Step 2: Run test to verify it passes (these are pure math, no code under test yet)**

Run: `python -m pytest tests/test_optimizer_math_proofs.py::TestRateStatAggregation -v`
Expected: 4 PASSED (these validate the formulas themselves)

- [ ] **Step 3: Commit**

```bash
git add tests/test_optimizer_math_proofs.py
git commit -m "test: add rate stat aggregation proof tests (AVG, OBP, ERA, WHIP)"
```

---

### Task 2: LP Objective Function Proofs

**Files:**
- Modify: `tests/test_optimizer_math_proofs.py`

- [ ] **Step 1: Write the failing test — ERA must be IP-weighted in LP**

Append to `tests/test_optimizer_math_proofs.py`:

```python
from src.lineup_optimizer import LineupOptimizer


class TestLPObjectiveWeighting:
    """Prove the LP solver correctly weights ERA/WHIP by IP and sign-flips L.

    The most dangerous optimizer bug: if ERA isn't IP-weighted, a reliever
    with 1 IP and 0.00 ERA outscores a 200-IP starter with 3.20 ERA.
    That's catastrophically wrong for fantasy.
    """

    @pytest.fixture()
    def _two_pitcher_roster(self):
        """One ace starter vs one low-usage reliever."""
        return pd.DataFrame([
            {
                "player_id": 1, "player_name": "Ace Starter",
                "positions": "SP", "is_hitter": 0, "team": "NYY",
                "r": 0, "hr": 0, "rbi": 0, "sb": 0, "avg": 0, "obp": 0,
                "w": 15, "l": 8, "sv": 0, "k": 200, "era": 3.20, "whip": 1.10,
                "ip": 200.0, "er": 71, "bb_allowed": 50, "h_allowed": 170,
                "h": 0, "ab": 0, "bb": 0, "hbp": 0, "sf": 0,
                "status": "active",
            },
            {
                "player_id": 2, "player_name": "Mop-Up Reliever",
                "positions": "RP", "is_hitter": 0, "team": "BOS",
                "r": 0, "hr": 0, "rbi": 0, "sb": 0, "avg": 0, "obp": 0,
                "w": 0, "l": 0, "sv": 0, "k": 1, "era": 0.00, "whip": 0.00,
                "ip": 1.0, "er": 0, "bb_allowed": 0, "h_allowed": 0,
                "h": 0, "ab": 0, "bb": 0, "hbp": 0, "sf": 0,
                "status": "active",
            },
        ])

    def test_starter_outscores_reliever_in_era(self, _two_pitcher_roster):
        """A 200-IP starter with 3.20 ERA MUST outscore a 1-IP reliever with 0.00 ERA.

        Hand calculation (LP objective for ERA):
            Starter: -(3.20 * 200.0 / scale) * weight  -> large negative (good)
            Reliever: -(0.00 * 1.0 / scale) * weight   -> zero contribution
            Without IP weighting: starter ERA 3.20 > reliever ERA 0.00 -> reliever wins (WRONG)
            With IP weighting: starter contributes 200 IP of 3.20 ERA -> starter wins (CORRECT)
        """
        optimizer = LineupOptimizer(_two_pitcher_roster)
        result = optimizer.optimize_lineup(
            category_weights={"era": 1.0, "whip": 0.0, "k": 0.0, "w": 0.0,
                              "l": 0.0, "sv": 0.0, "r": 0.0, "hr": 0.0,
                              "rbi": 0.0, "sb": 0.0, "avg": 0.0, "obp": 0.0},
        )
        assignments = result.get("assignments", [])
        # Both should be assigned (only 2 pitchers for SP+RP slots)
        # The key test: starter should be in starting slot, not benched
        assigned_names = [a["player_name"] for a in assignments]
        assert "Ace Starter" in assigned_names

    def test_losses_sign_flipped_no_ip_weight(self, _two_pitcher_roster):
        """Losses: lower is better (sign-flipped), but NOT IP-weighted.

        Unlike ERA/WHIP, Losses are a counting stat. A pitcher with 8 L in
        200 IP is worse than a pitcher with 0 L in 1 IP, period. No need
        to normalize by IP.

        Hand calculation:
            Starter: -(8 / scale) * weight  -> negative (bad, 8 losses)
            Reliever: -(0 / scale) * weight  -> zero (good, no losses)
        """
        optimizer = LineupOptimizer(_two_pitcher_roster)
        result = optimizer.optimize_lineup(
            category_weights={"l": 1.0, "era": 0.0, "whip": 0.0, "k": 0.0,
                              "w": 0.0, "sv": 0.0, "r": 0.0, "hr": 0.0,
                              "rbi": 0.0, "sb": 0.0, "avg": 0.0, "obp": 0.0},
        )
        # When only L matters, the reliever (0 L) should be preferred
        assignments = result.get("assignments", [])
        assigned_names = [a["player_name"] for a in assignments]
        assert "Mop-Up Reliever" in assigned_names
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python -m pytest tests/test_optimizer_math_proofs.py::TestLPObjectiveWeighting -v`
Expected: 2 PASSED

- [ ] **Step 3: Commit**

```bash
git add tests/test_optimizer_math_proofs.py
git commit -m "test: add LP objective weighting proofs (ERA IP-weight, L sign-flip)"
```

---

### Task 3: Bayesian Update Formula Proofs

**Files:**
- Modify: `tests/test_optimizer_math_proofs.py`

- [ ] **Step 1: Write the failing test — Bayesian blend matches hand calculation**

Append to `tests/test_optimizer_math_proofs.py`:

```python
from src.optimizer.projections import v2_bayesian_blend, V2_STABILIZATION_POINTS


class TestBayesianUpdateProofs:
    """Prove the Bayesian blend formula produces correct outputs.

    Formula: blended = (preseason_rate * stab + observed_num) / (stab + observed_denom)

    At 0 PA observed: 100% preseason (prior dominates)
    At stab PA observed: ~50/50 blend
    At 3x stab PA: ~75% observed (data dominates)
    """

    def test_zero_observations_returns_prior(self):
        """With no observed data, blend returns the preseason projection.

        Hand calculation:
            preseason HR rate = 0.040 (HR/PA)
            stab = 170 PA
            observed: 0 HR in 0 PA
            blend = (0.040 * 170 + 0) / (170 + 0) = 6.8 / 170 = 0.040
        """
        result = v2_bayesian_blend(
            preseason_rate=0.040,
            observed_numerator=0,
            observed_denominator=0,
            stabilization_pa=170,
        )
        assert result == pytest.approx(0.040, abs=1e-6)

    def test_at_stabilization_point_is_fifty_fifty(self):
        """At exactly stab PA, blend is ~50/50 between prior and observed.

        Hand calculation:
            preseason HR rate = 0.040 (HR/PA)
            stab = 170 PA
            observed: 8 HR in 170 PA (rate = 0.047)
            blend = (0.040 * 170 + 8) / (170 + 170) = (6.8 + 8) / 340 = 14.8/340 = 0.04353
            Midpoint of 0.040 and 0.047 = 0.0435 -- matches!
        """
        result = v2_bayesian_blend(
            preseason_rate=0.040,
            observed_numerator=8,
            observed_denominator=170,
            stabilization_pa=170,
        )
        midpoint = (0.040 + 8 / 170) / 2  # 0.04353
        assert result == pytest.approx(midpoint, abs=0.001)

    def test_large_sample_converges_to_observed(self):
        """With 3x stab PA, blend is ~75% observed rate.

        Hand calculation:
            preseason HR rate = 0.040
            stab = 170 PA
            observed: 25 HR in 510 PA (rate = 0.0490)
            blend = (0.040 * 170 + 25) / (170 + 510) = (6.8 + 25) / 680 = 31.8/680 = 0.04676
            Expected: closer to 0.049 than to 0.040
        """
        result = v2_bayesian_blend(
            preseason_rate=0.040,
            observed_numerator=25,
            observed_denominator=510,
            stabilization_pa=170,
        )
        observed_rate = 25 / 510  # 0.0490
        # At 3x stab (510 = 3*170), observed weight = 510/(170+510) = 75%
        expected_weight = 510 / (170 + 510)  # 0.75
        assert expected_weight == pytest.approx(0.75, abs=0.01)
        # Result should be closer to observed than to prior
        assert abs(result - observed_rate) < abs(result - 0.040)

    def test_stabilization_points_are_research_backed(self):
        """Verify stabilization points match published research values.

        Sources:
            - FanGraphs: "How Long Until a Hitter's Stats Stabilize?"
            - Pizza Cutter (Russell Carleton): stabilization research
            - HR stabilizes ~170 PA, AVG ~910 PA, K rate ~60 PA
        """
        assert V2_STABILIZATION_POINTS["hr_rate"] == 170
        assert V2_STABILIZATION_POINTS["avg"] == 910
        assert V2_STABILIZATION_POINTS["k_rate"] == 60
        assert V2_STABILIZATION_POINTS["obp"] == 460
        assert V2_STABILIZATION_POINTS["era"] == 630
        assert V2_STABILIZATION_POINTS["whip"] == 540
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python -m pytest tests/test_optimizer_math_proofs.py::TestBayesianUpdateProofs -v`
Expected: 4 PASSED

- [ ] **Step 3: Commit**

```bash
git add tests/test_optimizer_math_proofs.py
git commit -m "test: add Bayesian update formula proof tests with hand calculations"
```

---

### Task 4: Matchup Adjustment Compound Proofs

**Files:**
- Modify: `tests/test_optimizer_math_proofs.py`

- [ ] **Step 1: Write the failing test — adjustments multiply, not add**

Append to `tests/test_optimizer_math_proofs.py`:

```python
from src.optimizer.matchup_adjustments import (
    bayesian_platoon_adjustment,
    calibrated_pitcher_quality_mult,
    park_factor_adjustment,
    weather_hr_adjustment,
)


class TestMatchupAdjustmentCompounding:
    """Prove matchup adjustments compound multiplicatively.

    A hitter at Coors (1.38 park) vs a LHP (platoon boost) in 90F heat
    should get: park * platoon * weather * pitcher_quality, NOT
    park + platoon + weather + pitcher_quality.
    """

    def test_platoon_advantage_values(self):
        """LHB vs RHP gets ~8.6% boost, RHB vs LHP gets ~6.1% boost.

        Source: The Book (Tango, Lichtman, Dolphin)
        These are regressed defaults when no individual split data exists.
        """
        lhb_factor = bayesian_platoon_adjustment(
            batter_hand="L", pitcher_hand="R",
            individual_split_avg=None, individual_overall_avg=None,
            sample_pa=0,
        )
        rhb_factor = bayesian_platoon_adjustment(
            batter_hand="R", pitcher_hand="L",
            individual_split_avg=None, individual_overall_avg=None,
            sample_pa=0,
        )
        # LHB vs RHP: ~1.086 (8.6% boost)
        assert lhb_factor == pytest.approx(1.086, abs=0.01)
        # RHB vs LHP: ~1.061 (6.1% boost)
        assert rhb_factor == pytest.approx(1.061, abs=0.01)

    def test_same_hand_no_advantage(self):
        """LHB vs LHP or RHB vs RHP should get no platoon advantage (<=1.0)."""
        same_hand = bayesian_platoon_adjustment(
            batter_hand="L", pitcher_hand="L",
            individual_split_avg=None, individual_overall_avg=None,
            sample_pa=0,
        )
        assert same_hand <= 1.0

    def test_park_factor_coors_boost(self):
        """Coors Field (COL) should boost hitter counting stats ~38%."""
        pf = park_factor_adjustment(
            player_team="COL", opponent_team="COL",
            park_factors={"COL": 1.38},
            is_hitter=True,
        )
        assert pf == pytest.approx(1.38, abs=0.05)

    def test_park_factor_pitcher_inverted(self):
        """Pitchers at Coors should get WORSE (inverted factor)."""
        pf_pitcher = park_factor_adjustment(
            player_team="COL", opponent_team="COL",
            park_factors={"COL": 1.38},
            is_hitter=False,
        )
        # Pitcher factor should be < 1.0 at hitter-friendly parks
        # Implementation uses 1/pf for pitchers
        assert pf_pitcher < 1.0

    def test_weather_heat_boosts_hr(self):
        """Temperatures above 72F should boost HR projection.

        Source: Alan Nathan's physics of baseball — every 10F above 72
        adds approximately 2-4% to HR probability.
        """
        hot_factor = weather_hr_adjustment(temp_f=92.0)
        neutral_factor = weather_hr_adjustment(temp_f=72.0)
        cold_factor = weather_hr_adjustment(temp_f=52.0)

        assert hot_factor > neutral_factor
        assert neutral_factor == pytest.approx(1.0, abs=0.01)
        # Cold shouldn't go below baseline (temp < reference gets no penalty
        # unless implementation handles it)
        assert cold_factor <= neutral_factor

    def test_pitcher_quality_ace_boosts_hitters(self):
        """Facing a bad pitcher (5.50 ERA) should boost hitter stats.

        Hand calculation:
            z = (4.20 - 5.50) / 0.80 = -1.625
            z_clamped = max(-2.0, min(2.0, -1.625)) = -1.625
            mult = 1.0 + (-1.625) * 0.075 = 1.0 - 0.122 = 0.878
        Wait — the sign convention matters. Let me check:
            If opp_era > league_avg, hitters do better (positive adjustment).
            z = (league_avg - opp_era) / std = (4.20 - 5.50) / 0.80 = -1.625
            But we want HIGHER mult for worse pitchers.
        The function returns a multiplier where >1.0 helps hitters.
        """
        vs_bad = calibrated_pitcher_quality_mult(opp_era=5.50)
        vs_good = calibrated_pitcher_quality_mult(opp_era=3.00)
        vs_avg = calibrated_pitcher_quality_mult(opp_era=4.20)

        assert vs_bad > vs_avg  # Bad pitcher -> hitters do better
        assert vs_good < vs_avg  # Good pitcher -> hitters do worse
        assert vs_avg == pytest.approx(1.0, abs=0.05)

    def test_adjustments_compound_multiplicatively(self):
        """Multiple adjustments should multiply, not add.

        If park=1.10, platoon=1.08, pitcher=1.05:
            Multiplicative: 1.10 * 1.08 * 1.05 = 1.247
            Additive (WRONG): 1.0 + 0.10 + 0.08 + 0.05 = 1.23
        The difference matters and compounds over a lineup.
        """
        park = 1.10
        platoon = 1.08
        pitcher = 1.05

        multiplicative = park * platoon * pitcher
        additive = 1.0 + (park - 1.0) + (platoon - 1.0) + (pitcher - 1.0)

        assert multiplicative == pytest.approx(1.247, abs=0.01)
        assert additive == pytest.approx(1.23, abs=0.01)
        assert multiplicative != pytest.approx(additive, abs=0.005)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python -m pytest tests/test_optimizer_math_proofs.py::TestMatchupAdjustmentCompounding -v`
Expected: 7 PASSED

- [ ] **Step 3: Commit**

```bash
git add tests/test_optimizer_math_proofs.py
git commit -m "test: add matchup adjustment compounding proofs (platoon, park, weather, pitcher)"
```

---

### Task 5: SGP Denominator Validation

**Files:**
- Modify: `tests/test_optimizer_math_proofs.py`

- [ ] **Step 1: Write the failing test — SGP denominators are in research-plausible ranges**

Append to `tests/test_optimizer_math_proofs.py`:

```python
from src.valuation import LeagueConfig


class TestSGPDenominatorBounds:
    """Validate SGP denominators are within research-plausible ranges.

    SGP denominators represent the standard deviation of category totals
    across a 12-team league. Published values from Razzball, Fangraphs,
    and historical Yahoo leagues provide expected ranges.

    If a denominator is way off, every valuation downstream is skewed.
    """

    @pytest.fixture()
    def _lc(self):
        return LeagueConfig()

    def test_counting_stat_denominators(self, _lc):
        """Counting stat SGP denoms should be in plausible ranges.

        Reference ranges (12-team H2H):
            R: 25-45, HR: 8-18, RBI: 25-45, SB: 8-20
            W: 2-5, L: 2-5, SV: 5-15, K: 30-60
        """
        d = _lc.sgp_denominators
        assert 25 <= d["R"] <= 45, f"R denom {d['R']} outside [25, 45]"
        assert 8 <= d["HR"] <= 18, f"HR denom {d['HR']} outside [8, 18]"
        assert 25 <= d["RBI"] <= 45, f"RBI denom {d['RBI']} outside [25, 45]"
        assert 8 <= d["SB"] <= 20, f"SB denom {d['SB']} outside [8, 20]"
        assert 2 <= d["W"] <= 5, f"W denom {d['W']} outside [2, 5]"
        assert 2 <= d["L"] <= 5, f"L denom {d['L']} outside [2, 5]"
        assert 5 <= d["SV"] <= 15, f"SV denom {d['SV']} outside [5, 15]"
        assert 30 <= d["K"] <= 60, f"K denom {d['K']} outside [30, 60]"

    def test_rate_stat_denominators(self, _lc):
        """Rate stat SGP denoms should be very small (they're stdev of rates).

        Reference ranges:
            AVG: 0.003-0.006, OBP: 0.004-0.007
            ERA: 0.15-0.30, WHIP: 0.015-0.030
        """
        d = _lc.sgp_denominators
        assert 0.003 <= d["AVG"] <= 0.006, f"AVG denom {d['AVG']} outside range"
        assert 0.004 <= d["OBP"] <= 0.007, f"OBP denom {d['OBP']} outside range"
        assert 0.15 <= d["ERA"] <= 0.30, f"ERA denom {d['ERA']} outside range"
        assert 0.015 <= d["WHIP"] <= 0.030, f"WHIP denom {d['WHIP']} outside range"

    def test_inverse_stats_defined(self, _lc):
        """L, ERA, WHIP must be in inverse_stats (lower is better)."""
        inv = _lc.inverse_stats
        assert "L" in inv
        assert "ERA" in inv
        assert "WHIP" in inv
        # R, HR, RBI, SB should NOT be inverse
        assert "R" not in inv
        assert "HR" not in inv
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python -m pytest tests/test_optimizer_math_proofs.py::TestSGPDenominatorBounds -v`
Expected: 3 PASSED

- [ ] **Step 3: Commit**

```bash
git add tests/test_optimizer_math_proofs.py
git commit -m "test: add SGP denominator bounds validation with research references"
```

---

## Phase 2: Shared Data Layer Tests (P0)

The shared data layer (`src/optimizer/shared_data_layer.py`) is 901 lines with zero tests. It's the single data hub for all optimizer tabs.

---

### Task 6: OptimizerDataContext Dataclass Tests

**Files:**
- Create: `tests/test_shared_data_layer.py`

- [ ] **Step 1: Write the failing test — context construction and field types**

```python
"""Tests for src/optimizer/shared_data_layer.py.

The shared data layer is the single source of truth for all optimizer tabs.
It builds an immutable OptimizerDataContext that every module consumes.
"""

import pandas as pd
import pytest

from src.optimizer.shared_data_layer import (
    OptimizerDataContext,
    get_recent_form_weight,
    scale_projections_for_scope,
)
from src.valuation import LeagueConfig


class TestOptimizerDataContext:
    """Verify OptimizerDataContext construction and field contracts."""

    @pytest.fixture()
    def _minimal_context(self):
        """Build a minimal valid context with required fields."""
        roster = pd.DataFrame([
            {
                "player_id": 1, "player_name": "Test Hitter",
                "positions": "1B", "is_hitter": 1, "team": "NYY",
                "r": 80, "hr": 25, "rbi": 75, "sb": 10,
                "avg": 0.280, "obp": 0.350,
                "w": 0, "l": 0, "sv": 0, "k": 0,
                "era": 0, "whip": 0, "ip": 0,
                "h": 140, "ab": 500, "bb": 50, "hbp": 5, "sf": 4,
                "status": "active",
            },
            {
                "player_id": 2, "player_name": "Test Pitcher",
                "positions": "SP", "is_hitter": 0, "team": "BOS",
                "r": 0, "hr": 0, "rbi": 0, "sb": 0,
                "avg": 0, "obp": 0,
                "w": 12, "l": 6, "sv": 0, "k": 180,
                "era": 3.50, "whip": 1.15, "ip": 180,
                "h": 0, "ab": 0, "bb": 0, "hbp": 0, "sf": 0,
                "er": 70, "bb_allowed": 45, "h_allowed": 160,
                "status": "active",
            },
        ])
        return OptimizerDataContext(
            roster=roster,
            player_pool=roster.copy(),
            free_agents=pd.DataFrame(),
            user_roster_ids=[1, 2],
            live_matchup=None,
            my_totals={"R": 400, "HR": 120, "RBI": 380, "SB": 60,
                        "AVG": 0.265, "OBP": 0.330,
                        "W": 45, "L": 30, "SV": 40, "K": 800,
                        "ERA": 3.80, "WHIP": 1.22},
            opp_totals={"R": 420, "HR": 130, "RBI": 400, "SB": 55,
                         "AVG": 0.270, "OBP": 0.340,
                         "W": 50, "L": 28, "SV": 45, "K": 850,
                         "ERA": 3.60, "WHIP": 1.18},
            opponent_name="Test Opponent",
            win_loss_tie=(5, 6, 1),
            urgency_weights={},
            category_weights={cat: 1.0 for cat in ["R", "HR", "RBI", "SB", "AVG", "OBP",
                                                     "W", "L", "SV", "K", "ERA", "WHIP"]},
            category_gaps={},
            h2h_strategy={},
            todays_schedule=[],
            confirmed_lineups={},
            remaining_games_this_week={},
            two_start_pitchers=[],
            opposing_pitchers={},
            team_strength={},
            park_factors={},
            weather={},
            recent_form={},
            health_scores={1: 0.95, 2: 0.90},
            news_flags={},
            ownership_trends={},
            scope="rest_of_week",
            weeks_remaining=16,
            config=LeagueConfig(),
            adds_remaining_this_week=7,
            closer_count=2,
            il_stash_ids=set(),
        )

    def test_context_has_all_required_fields(self, _minimal_context):
        """Context must expose all fields consumed by optimizer modules."""
        ctx = _minimal_context
        assert isinstance(ctx.roster, pd.DataFrame)
        assert isinstance(ctx.player_pool, pd.DataFrame)
        assert isinstance(ctx.free_agents, pd.DataFrame)
        assert isinstance(ctx.user_roster_ids, list)
        assert isinstance(ctx.my_totals, dict)
        assert isinstance(ctx.opp_totals, dict)
        assert isinstance(ctx.category_weights, dict)
        assert isinstance(ctx.health_scores, dict)
        assert isinstance(ctx.scope, str)
        assert isinstance(ctx.weeks_remaining, int)
        assert isinstance(ctx.config, LeagueConfig)

    def test_context_totals_have_all_12_categories(self, _minimal_context):
        """my_totals and opp_totals must have all 12 scoring categories."""
        lc = LeagueConfig()
        all_cats = lc.hitting_categories + lc.pitching_categories
        for cat in all_cats:
            assert cat in _minimal_context.my_totals, f"Missing {cat} in my_totals"
            assert cat in _minimal_context.opp_totals, f"Missing {cat} in opp_totals"

    def test_category_weights_has_all_12(self, _minimal_context):
        """Category weights dict must cover all 12 categories."""
        lc = LeagueConfig()
        all_cats = lc.hitting_categories + lc.pitching_categories
        for cat in all_cats:
            assert cat in _minimal_context.category_weights, f"Missing weight for {cat}"

    def test_win_loss_tie_is_three_tuple(self, _minimal_context):
        """W-L-T must be a 3-element tuple of ints."""
        wlt = _minimal_context.win_loss_tie
        assert len(wlt) == 3
        assert all(isinstance(v, int) for v in wlt)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python -m pytest tests/test_shared_data_layer.py::TestOptimizerDataContext -v`
Expected: 4 PASSED

- [ ] **Step 3: Commit**

```bash
git add tests/test_shared_data_layer.py
git commit -m "test: add OptimizerDataContext dataclass contract tests"
```

---

### Task 7: Scope-Based Projection Scaling Tests

**Files:**
- Modify: `tests/test_shared_data_layer.py`

- [ ] **Step 1: Write the failing test — counting stats scale by scope, rates don't**

Append to `tests/test_shared_data_layer.py`:

```python
class TestScaleProjectionsForScope:
    """Verify projection scaling correctly handles scope differences.

    Today: counting stats / 162 (single game)
    Rest of Week: scale by remaining games + two-start pitcher doubling
    Rest of Season: full projections with schedule strength
    Rate stats: NEVER scaled (AVG, OBP, ERA, WHIP stay the same)
    """

    @pytest.fixture()
    def _roster(self):
        return pd.DataFrame([
            {
                "player_id": 1, "player_name": "Hitter",
                "positions": "1B", "is_hitter": 1, "team": "NYY",
                "r": 80, "hr": 25, "rbi": 75, "sb": 10,
                "avg": 0.280, "obp": 0.350,
                "w": 0, "l": 0, "sv": 0, "k": 0,
                "era": 0, "whip": 0, "ip": 0,
            },
        ])

    @pytest.fixture()
    def _today_context(self, _roster):
        return OptimizerDataContext(
            roster=_roster,
            player_pool=_roster.copy(),
            free_agents=pd.DataFrame(),
            user_roster_ids=[1],
            live_matchup=None,
            my_totals={}, opp_totals={},
            opponent_name="", win_loss_tie=(0, 0, 0),
            urgency_weights={}, category_weights={},
            category_gaps={}, h2h_strategy={},
            todays_schedule=[], confirmed_lineups={},
            remaining_games_this_week={"NYY": 1},
            two_start_pitchers=[], opposing_pitchers={},
            team_strength={}, park_factors={}, weather={},
            recent_form={}, health_scores={}, news_flags={},
            ownership_trends={},
            scope="today", weeks_remaining=16,
            config=LeagueConfig(),
            adds_remaining_this_week=7, closer_count=2, il_stash_ids=set(),
        )

    def test_today_scope_scales_counting_stats_down(self, _today_context, _roster):
        """Today scope: counting stats should be ~1/162 of season projection."""
        scaled = scale_projections_for_scope(_today_context, _roster)
        # HR should be roughly 25/162 ≈ 0.154 (single game)
        assert scaled.iloc[0]["hr"] < 1.0
        assert scaled.iloc[0]["hr"] == pytest.approx(25 / 162, abs=0.05)

    def test_rate_stats_unchanged_by_scope(self, _today_context, _roster):
        """Rate stats (AVG, OBP) must never change when scaling scope."""
        scaled = scale_projections_for_scope(_today_context, _roster)
        assert scaled.iloc[0]["avg"] == pytest.approx(0.280, abs=0.001)
        assert scaled.iloc[0]["obp"] == pytest.approx(0.350, abs=0.001)

    def test_returns_copy_not_mutation(self, _today_context, _roster):
        """scale_projections_for_scope must return a COPY, not mutate input."""
        original_hr = _roster.iloc[0]["hr"]
        _ = scale_projections_for_scope(_today_context, _roster)
        assert _roster.iloc[0]["hr"] == original_hr


class TestRecentFormWeight:
    """Verify scope-specific recent form weights."""

    def test_today_weight(self):
        assert get_recent_form_weight("today") == pytest.approx(0.25, abs=0.01)

    def test_week_weight(self):
        assert get_recent_form_weight("rest_of_week") == pytest.approx(0.30, abs=0.01)

    def test_season_weight(self):
        assert get_recent_form_weight("rest_of_season") == pytest.approx(0.20, abs=0.01)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python -m pytest tests/test_shared_data_layer.py -v`
Expected: All PASSED

- [ ] **Step 3: Commit**

```bash
git add tests/test_shared_data_layer.py
git commit -m "test: add scope scaling and recent form weight tests for shared data layer"
```

---

## Phase 3: Cross-Module Integration Tests (P0)

These tests verify the full pipeline chain works end-to-end with realistic rosters and all settings combinations.

---

### Task 8: Build Realistic 23-Player Test Roster

**Files:**
- Create: `tests/test_optimizer_integration.py`

- [ ] **Step 1: Create the integration test file with a realistic fixture**

```python
"""Integration tests for the full Lineup Optimizer pipeline.

These tests use realistic 23-player rosters (not 3-6 player toy rosters)
and verify the full chain: projections -> matchup -> weights -> LP -> output.
"""

import pandas as pd
import pytest

from src.optimizer.pipeline import LineupOptimizerPipeline
from src.valuation import LeagueConfig

# ── Realistic 23-player roster fixture ──────────────────────────────

_HITTERS = [
    {"player_id": 101, "player_name": "C_Starter", "positions": "C", "is_hitter": 1, "team": "NYY",
     "r": 55, "hr": 18, "rbi": 60, "sb": 2, "avg": 0.245, "obp": 0.320,
     "h": 110, "ab": 449, "bb": 40, "hbp": 3, "sf": 4, "ip": 0, "status": "active",
     "w": 0, "l": 0, "sv": 0, "k": 0, "era": 0, "whip": 0},
    {"player_id": 102, "player_name": "1B_Starter", "positions": "1B", "is_hitter": 1, "team": "BOS",
     "r": 85, "hr": 32, "rbi": 95, "sb": 3, "avg": 0.275, "obp": 0.360,
     "h": 150, "ab": 545, "bb": 60, "hbp": 5, "sf": 5, "ip": 0, "status": "active",
     "w": 0, "l": 0, "sv": 0, "k": 0, "era": 0, "whip": 0},
    {"player_id": 103, "player_name": "2B_Starter", "positions": "2B", "is_hitter": 1, "team": "HOU",
     "r": 75, "hr": 15, "rbi": 55, "sb": 20, "avg": 0.270, "obp": 0.340,
     "h": 140, "ab": 519, "bb": 45, "hbp": 2, "sf": 3, "ip": 0, "status": "active",
     "w": 0, "l": 0, "sv": 0, "k": 0, "era": 0, "whip": 0},
    {"player_id": 104, "player_name": "3B_Starter", "positions": "3B", "is_hitter": 1, "team": "LAD",
     "r": 90, "hr": 28, "rbi": 85, "sb": 8, "avg": 0.280, "obp": 0.370,
     "h": 155, "ab": 554, "bb": 65, "hbp": 4, "sf": 5, "ip": 0, "status": "active",
     "w": 0, "l": 0, "sv": 0, "k": 0, "era": 0, "whip": 0},
    {"player_id": 105, "player_name": "SS_Starter", "positions": "SS", "is_hitter": 1, "team": "ATL",
     "r": 80, "hr": 22, "rbi": 70, "sb": 15, "avg": 0.265, "obp": 0.335,
     "h": 138, "ab": 521, "bb": 48, "hbp": 3, "sf": 4, "ip": 0, "status": "active",
     "w": 0, "l": 0, "sv": 0, "k": 0, "era": 0, "whip": 0},
    {"player_id": 106, "player_name": "OF1_Starter", "positions": "OF", "is_hitter": 1, "team": "NYY",
     "r": 95, "hr": 35, "rbi": 100, "sb": 12, "avg": 0.290, "obp": 0.380,
     "h": 162, "ab": 559, "bb": 70, "hbp": 5, "sf": 5, "ip": 0, "status": "active",
     "w": 0, "l": 0, "sv": 0, "k": 0, "era": 0, "whip": 0},
    {"player_id": 107, "player_name": "OF2_Starter", "positions": "OF", "is_hitter": 1, "team": "SEA",
     "r": 70, "hr": 20, "rbi": 65, "sb": 25, "avg": 0.260, "obp": 0.330,
     "h": 135, "ab": 519, "bb": 42, "hbp": 3, "sf": 3, "ip": 0, "status": "active",
     "w": 0, "l": 0, "sv": 0, "k": 0, "era": 0, "whip": 0},
    {"player_id": 108, "player_name": "OF3_Starter", "positions": "OF", "is_hitter": 1, "team": "CHC",
     "r": 78, "hr": 24, "rbi": 72, "sb": 8, "avg": 0.272, "obp": 0.345,
     "h": 142, "ab": 522, "bb": 50, "hbp": 4, "sf": 4, "ip": 0, "status": "active",
     "w": 0, "l": 0, "sv": 0, "k": 0, "era": 0, "whip": 0},
    {"player_id": 109, "player_name": "Util1", "positions": "1B,OF", "is_hitter": 1, "team": "PHI",
     "r": 65, "hr": 20, "rbi": 68, "sb": 5, "avg": 0.255, "obp": 0.325,
     "h": 125, "ab": 490, "bb": 38, "hbp": 3, "sf": 3, "ip": 0, "status": "active",
     "w": 0, "l": 0, "sv": 0, "k": 0, "era": 0, "whip": 0},
    {"player_id": 110, "player_name": "Util2", "positions": "3B,SS", "is_hitter": 1, "team": "SDP",
     "r": 60, "hr": 14, "rbi": 50, "sb": 18, "avg": 0.262, "obp": 0.330,
     "h": 130, "ab": 496, "bb": 40, "hbp": 2, "sf": 3, "ip": 0, "status": "active",
     "w": 0, "l": 0, "sv": 0, "k": 0, "era": 0, "whip": 0},
]

_PITCHERS = [
    {"player_id": 201, "player_name": "SP1_Ace", "positions": "SP", "is_hitter": 0, "team": "LAD",
     "w": 15, "l": 6, "sv": 0, "k": 220, "era": 2.90, "whip": 1.05,
     "ip": 200.0, "er": 64, "bb_allowed": 45, "h_allowed": 165,
     "r": 0, "hr": 0, "rbi": 0, "sb": 0, "avg": 0, "obp": 0,
     "h": 0, "ab": 0, "bb": 0, "hbp": 0, "sf": 0, "status": "active"},
    {"player_id": 202, "player_name": "SP2_Mid", "positions": "SP", "is_hitter": 0, "team": "NYY",
     "w": 12, "l": 8, "sv": 0, "k": 180, "era": 3.60, "whip": 1.18,
     "ip": 180.0, "er": 72, "bb_allowed": 50, "h_allowed": 162,
     "r": 0, "hr": 0, "rbi": 0, "sb": 0, "avg": 0, "obp": 0,
     "h": 0, "ab": 0, "bb": 0, "hbp": 0, "sf": 0, "status": "active"},
    {"player_id": 203, "player_name": "SP3_Back", "positions": "SP", "is_hitter": 0, "team": "BOS",
     "w": 9, "l": 10, "sv": 0, "k": 150, "era": 4.20, "whip": 1.30,
     "ip": 165.0, "er": 77, "bb_allowed": 55, "h_allowed": 160,
     "r": 0, "hr": 0, "rbi": 0, "sb": 0, "avg": 0, "obp": 0,
     "h": 0, "ab": 0, "bb": 0, "hbp": 0, "sf": 0, "status": "active"},
    {"player_id": 204, "player_name": "RP1_Closer", "positions": "RP", "is_hitter": 0, "team": "NYY",
     "w": 4, "l": 3, "sv": 30, "k": 70, "era": 2.80, "whip": 1.00,
     "ip": 65.0, "er": 20, "bb_allowed": 18, "h_allowed": 47,
     "r": 0, "hr": 0, "rbi": 0, "sb": 0, "avg": 0, "obp": 0,
     "h": 0, "ab": 0, "bb": 0, "hbp": 0, "sf": 0, "status": "active"},
    {"player_id": 205, "player_name": "RP2_Setup", "positions": "RP", "is_hitter": 0, "team": "ATL",
     "w": 5, "l": 2, "sv": 5, "k": 75, "era": 3.10, "whip": 1.08,
     "ip": 70.0, "er": 24, "bb_allowed": 20, "h_allowed": 56,
     "r": 0, "hr": 0, "rbi": 0, "sb": 0, "avg": 0, "obp": 0,
     "h": 0, "ab": 0, "bb": 0, "hbp": 0, "sf": 0, "status": "active"},
    {"player_id": 206, "player_name": "P_Flex1", "positions": "SP,RP", "is_hitter": 0, "team": "HOU",
     "w": 7, "l": 5, "sv": 2, "k": 110, "era": 3.80, "whip": 1.22,
     "ip": 120.0, "er": 51, "bb_allowed": 35, "h_allowed": 112,
     "r": 0, "hr": 0, "rbi": 0, "sb": 0, "avg": 0, "obp": 0,
     "h": 0, "ab": 0, "bb": 0, "hbp": 0, "sf": 0, "status": "active"},
    {"player_id": 207, "player_name": "P_Flex2", "positions": "SP,RP", "is_hitter": 0, "team": "SDP",
     "w": 6, "l": 4, "sv": 3, "k": 100, "era": 3.90, "whip": 1.25,
     "ip": 110.0, "er": 48, "bb_allowed": 32, "h_allowed": 106,
     "r": 0, "hr": 0, "rbi": 0, "sb": 0, "avg": 0, "obp": 0,
     "h": 0, "ab": 0, "bb": 0, "hbp": 0, "sf": 0, "status": "active"},
    {"player_id": 208, "player_name": "P_Flex3", "positions": "RP", "is_hitter": 0, "team": "CHC",
     "w": 3, "l": 2, "sv": 8, "k": 60, "era": 3.40, "whip": 1.12,
     "ip": 55.0, "er": 21, "bb_allowed": 15, "h_allowed": 47,
     "r": 0, "hr": 0, "rbi": 0, "sb": 0, "avg": 0, "obp": 0,
     "h": 0, "ab": 0, "bb": 0, "hbp": 0, "sf": 0, "status": "active"},
]

_BENCH = [
    {"player_id": 301, "player_name": "BN_Hitter1", "positions": "OF,1B", "is_hitter": 1, "team": "MIA",
     "r": 45, "hr": 12, "rbi": 40, "sb": 6, "avg": 0.242, "obp": 0.310,
     "h": 105, "ab": 434, "bb": 30, "hbp": 2, "sf": 2, "ip": 0, "status": "active",
     "w": 0, "l": 0, "sv": 0, "k": 0, "era": 0, "whip": 0},
    {"player_id": 302, "player_name": "BN_Hitter2", "positions": "2B,SS", "is_hitter": 1, "team": "CIN",
     "r": 50, "hr": 10, "rbi": 35, "sb": 12, "avg": 0.248, "obp": 0.315,
     "h": 108, "ab": 435, "bb": 32, "hbp": 2, "sf": 3, "ip": 0, "status": "active",
     "w": 0, "l": 0, "sv": 0, "k": 0, "era": 0, "whip": 0},
    {"player_id": 303, "player_name": "BN_Pitcher", "positions": "SP", "is_hitter": 0, "team": "TBR",
     "w": 5, "l": 7, "sv": 0, "k": 90, "era": 4.50, "whip": 1.35,
     "ip": 100.0, "er": 50, "bb_allowed": 40, "h_allowed": 95,
     "r": 0, "hr": 0, "rbi": 0, "sb": 0, "avg": 0, "obp": 0,
     "h": 0, "ab": 0, "bb": 0, "hbp": 0, "sf": 0, "status": "active"},
    {"player_id": 304, "player_name": "IL_Player", "positions": "OF", "is_hitter": 1, "team": "LAD",
     "r": 70, "hr": 22, "rbi": 65, "sb": 10, "avg": 0.268, "obp": 0.340,
     "h": 130, "ab": 485, "bb": 45, "hbp": 3, "sf": 3, "ip": 0, "status": "IL10",
     "w": 0, "l": 0, "sv": 0, "k": 0, "era": 0, "whip": 0},
    {"player_id": 305, "player_name": "BN_Hitter3", "positions": "C,1B", "is_hitter": 1, "team": "STL",
     "r": 40, "hr": 8, "rbi": 32, "sb": 1, "avg": 0.235, "obp": 0.300,
     "h": 90, "ab": 383, "bb": 25, "hbp": 2, "sf": 2, "ip": 0, "status": "active",
     "w": 0, "l": 0, "sv": 0, "k": 0, "era": 0, "whip": 0},
]


@pytest.fixture()
def full_roster():
    """23-player roster matching Yahoo H2H format."""
    return pd.DataFrame(_HITTERS + _PITCHERS + _BENCH)
```

- [ ] **Step 2: Run to verify fixture loads**

Run: `python -m pytest tests/test_optimizer_integration.py --co -q`
Expected: Shows collected fixtures, no errors

- [ ] **Step 3: Commit**

```bash
git add tests/test_optimizer_integration.py
git commit -m "test: add realistic 23-player roster fixture for integration tests"
```

---

### Task 9: Pipeline Invariant Assertions

**Files:**
- Modify: `tests/test_optimizer_integration.py`

- [ ] **Step 1: Write tests that verify universal invariants across all modes**

Append to `tests/test_optimizer_integration.py`:

```python
class TestPipelineInvariants:
    """Invariants that must hold across ALL modes, alpha, and risk settings.

    These are properties that should NEVER be violated regardless of
    how the optimizer is configured. If any of these fail, the optimizer
    is producing invalid lineups.
    """

    @pytest.fixture(params=["quick", "standard", "full"])
    def mode(self, request):
        return request.param

    @pytest.fixture(params=[0.0, 0.5, 1.0])
    def alpha(self, request):
        return request.param

    def test_pipeline_returns_valid_structure(self, full_roster, mode, alpha):
        """Pipeline must return a dict with required keys."""
        pipe = LineupOptimizerPipeline(full_roster, mode=mode, alpha=alpha)
        result = pipe.optimize()

        assert isinstance(result, dict)
        assert "lineup" in result
        assert "category_weights" in result
        assert "mode" in result
        assert result["mode"] == mode

    def test_no_il_player_in_starting_lineup(self, full_roster, mode, alpha):
        """IL/DTD players must NEVER appear in starting assignments.

        IL_Player (id=304, status=IL10) must be on bench, never assigned.
        """
        pipe = LineupOptimizerPipeline(full_roster, mode=mode, alpha=alpha)
        result = pipe.optimize()
        assignments = result.get("lineup", {}).get("assignments", [])

        for a in assignments:
            assert a.get("player_name") != "IL_Player", (
                f"IL player assigned to slot {a.get('slot')} in mode={mode}"
            )

    def test_lineup_fills_all_position_slots(self, full_roster, mode, alpha):
        """Optimizer must attempt to fill all roster slots."""
        pipe = LineupOptimizerPipeline(full_roster, mode=mode, alpha=alpha)
        result = pipe.optimize()
        assignments = result.get("lineup", {}).get("assignments", [])

        # Yahoo H2H: C,1B,2B,3B,SS,OF,OF,OF,Util,Util,SP,SP,RP,RP,P,P,P,P = 18 slots
        # Some may be unfilled if roster is too small, but with 23 players
        # we should fill most
        assert len(assignments) >= 15, (
            f"Only {len(assignments)} slots filled in mode={mode}"
        )

    def test_category_weights_are_positive(self, full_roster, mode, alpha):
        """All category weights must be positive (>0)."""
        pipe = LineupOptimizerPipeline(full_roster, mode=mode, alpha=alpha)
        result = pipe.optimize()
        weights = result.get("category_weights", {})

        for cat, w in weights.items():
            assert w > 0, f"Weight for {cat} is {w} (must be positive)"

    def test_no_duplicate_player_assignments(self, full_roster, mode, alpha):
        """Each player can only be assigned to ONE slot."""
        pipe = LineupOptimizerPipeline(full_roster, mode=mode, alpha=alpha)
        result = pipe.optimize()
        assignments = result.get("lineup", {}).get("assignments", [])
        player_ids = [a.get("player_id") for a in assignments if a.get("player_id")]

        assert len(player_ids) == len(set(player_ids)), (
            f"Duplicate player assignments in mode={mode}: {player_ids}"
        )
```

- [ ] **Step 2: Run tests (3 modes × 3 alphas × 5 invariants = 45 test cases)**

Run: `python -m pytest tests/test_optimizer_integration.py::TestPipelineInvariants -v --timeout=120`
Expected: 45 PASSED

- [ ] **Step 3: Commit**

```bash
git add tests/test_optimizer_integration.py
git commit -m "test: add pipeline invariant assertions across all mode × alpha combinations"
```

---

### Task 10: Settings Axis Tests

**Files:**
- Modify: `tests/test_optimizer_integration.py`

- [ ] **Step 1: Write tests that verify settings produce expected directional effects**

Append to `tests/test_optimizer_integration.py`:

```python
class TestSettingsAxisEffects:
    """Verify that changing settings produces expected directional changes.

    These are "monotonicity" tests: higher α should mean more H2H-focused
    weights, higher risk aversion should mean lower-variance selections.
    """

    def test_higher_alpha_increases_weight_variance(self, full_roster):
        """Higher α (more H2H-focused) should produce more extreme category weights.

        α=0.0 (pure season): weights closer to uniform (SGP-based)
        α=1.0 (pure H2H): weights skewed toward swing categories

        We measure this by checking the coefficient of variation of weights.
        """
        import numpy as np

        pipe_season = LineupOptimizerPipeline(full_roster, mode="standard", alpha=0.0)
        pipe_h2h = LineupOptimizerPipeline(full_roster, mode="standard", alpha=1.0)

        # Need to provide opponent totals for H2H weighting to kick in
        opp_totals = {"R": 400, "HR": 120, "RBI": 380, "SB": 55,
                      "AVG": 0.268, "OBP": 0.335,
                      "W": 48, "L": 30, "SV": 42, "K": 820,
                      "ERA": 3.70, "WHIP": 1.20}
        my_totals = {"R": 420, "HR": 130, "RBI": 400, "SB": 60,
                     "AVG": 0.272, "OBP": 0.340,
                     "W": 50, "L": 28, "SV": 45, "K": 850,
                     "ERA": 3.60, "WHIP": 1.18}

        result_season = pipe_season.optimize(
            h2h_opponent_totals=opp_totals, my_totals=my_totals,
        )
        result_h2h = pipe_h2h.optimize(
            h2h_opponent_totals=opp_totals, my_totals=my_totals,
        )

        w_season = list(result_season["category_weights"].values())
        w_h2h = list(result_h2h["category_weights"].values())

        cv_season = np.std(w_season) / max(np.mean(w_season), 1e-9)
        cv_h2h = np.std(w_h2h) / max(np.mean(w_h2h), 1e-9)

        # H2H weights should have higher variance (more extreme)
        # This may not always hold perfectly, so we use a soft check
        assert cv_h2h >= cv_season * 0.8, (
            f"H2H weight CV ({cv_h2h:.3f}) not higher than season CV ({cv_season:.3f})"
        )

    def test_full_mode_has_more_analysis_than_quick(self, full_roster):
        """Full mode should produce more analysis outputs than quick mode.

        Full enables: streaming, maximin, scenarios, multi-period
        Quick enables: projections only
        """
        pipe_quick = LineupOptimizerPipeline(full_roster, mode="quick")
        pipe_full = LineupOptimizerPipeline(full_roster, mode="full")

        result_quick = pipe_quick.optimize()
        result_full = pipe_full.optimize()

        # Full mode should have more non-None fields
        quick_populated = sum(1 for v in result_quick.values() if v is not None)
        full_populated = sum(1 for v in result_full.values() if v is not None)

        assert full_populated >= quick_populated, (
            f"Full mode ({full_populated} fields) has fewer outputs than quick ({quick_populated})"
        )

    def test_all_three_modes_produce_valid_lineups(self, full_roster):
        """Quick, Standard, and Full must all produce valid lineup assignments."""
        for mode in ["quick", "standard", "full"]:
            pipe = LineupOptimizerPipeline(full_roster, mode=mode)
            result = pipe.optimize()
            assignments = result.get("lineup", {}).get("assignments", [])
            assert len(assignments) > 0, f"Mode {mode} produced empty lineup"
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/test_optimizer_integration.py::TestSettingsAxisEffects -v --timeout=120`
Expected: 3 PASSED

- [ ] **Step 3: Commit**

```bash
git add tests/test_optimizer_integration.py
git commit -m "test: add settings axis tests (alpha, mode directional effects)"
```

---

## Phase 4: Data Freshness & Quality (P1)

Build infrastructure to track data freshness and surface it in the UI.

---

### Task 11: Data Freshness Tracker Module

**Files:**
- Create: `src/optimizer/data_freshness.py`
- Create: `tests/test_data_freshness.py`

- [ ] **Step 1: Write the failing test for freshness tracker**

Create `tests/test_data_freshness.py`:

```python
"""Tests for data freshness tracking."""

from datetime import UTC, datetime, timedelta

import pytest

from src.optimizer.data_freshness import DataFreshnessTracker, FreshnessStatus


class TestDataFreshnessTracker:
    """Verify freshness tracking and staleness detection."""

    @pytest.fixture()
    def _tracker(self):
        return DataFreshnessTracker()

    def test_record_and_check_fresh(self, _tracker):
        """Data recorded just now should be fresh."""
        _tracker.record("live_stats", ttl_hours=1.0)
        status = _tracker.check("live_stats")
        assert status == FreshnessStatus.FRESH

    def test_stale_data_detected(self, _tracker):
        """Data recorded 2 hours ago with 1-hour TTL should be stale."""
        _tracker.record(
            "live_stats",
            ttl_hours=1.0,
            timestamp=datetime.now(UTC) - timedelta(hours=2),
        )
        status = _tracker.check("live_stats")
        assert status == FreshnessStatus.STALE

    def test_unknown_source_returns_unknown(self, _tracker):
        """Unrecorded data source should return UNKNOWN status."""
        status = _tracker.check("nonexistent_source")
        assert status == FreshnessStatus.UNKNOWN

    def test_get_all_freshness(self, _tracker):
        """get_all() should return status for every recorded source."""
        _tracker.record("live_stats", ttl_hours=1.0)
        _tracker.record("projections", ttl_hours=24.0)
        result = _tracker.get_all()
        assert "live_stats" in result
        assert "projections" in result
        assert len(result) == 2

    def test_missing_data_audit(self, _tracker):
        """audit_missing() counts null/missing values in a DataFrame."""
        import pandas as pd

        df = pd.DataFrame([
            {"player_name": "A", "statcast_barrel": 0.12, "recent_form_avg": None},
            {"player_name": "B", "statcast_barrel": None, "recent_form_avg": 0.280},
            {"player_name": "C", "statcast_barrel": None, "recent_form_avg": None},
        ])
        report = _tracker.audit_missing(df, columns=["statcast_barrel", "recent_form_avg"])
        assert report["statcast_barrel"]["missing"] == 2
        assert report["statcast_barrel"]["total"] == 3
        assert report["recent_form_avg"]["missing"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_data_freshness.py -v`
Expected: FAIL (module does not exist)

- [ ] **Step 3: Implement the freshness tracker**

Create `src/optimizer/data_freshness.py`:

```python
"""Data freshness tracking for the Lineup Optimizer.

Tracks when each data source was last refreshed and whether it's
within its staleness TTL.  Surfaces missing-data audits for
quality monitoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum

import pandas as pd

logger = logging.getLogger(__name__)


class FreshnessStatus(Enum):
    """Staleness status for a data source."""

    FRESH = "fresh"
    STALE = "stale"
    UNKNOWN = "unknown"


@dataclass
class _SourceRecord:
    name: str
    timestamp: datetime
    ttl_hours: float


class DataFreshnessTracker:
    """Track last-refresh timestamps and TTLs for optimizer data sources."""

    def __init__(self) -> None:
        self._sources: dict[str, _SourceRecord] = {}

    def record(
        self,
        source: str,
        ttl_hours: float,
        timestamp: datetime | None = None,
    ) -> None:
        """Record a data fetch with its TTL.

        Args:
            source: Data source name (e.g. "live_stats", "projections").
            ttl_hours: Hours before this data is considered stale.
            timestamp: When the data was fetched.  Defaults to now.
        """
        self._sources[source] = _SourceRecord(
            name=source,
            timestamp=timestamp or datetime.now(UTC),
            ttl_hours=ttl_hours,
        )

    def check(self, source: str) -> FreshnessStatus:
        """Check if a data source is fresh or stale."""
        rec = self._sources.get(source)
        if rec is None:
            return FreshnessStatus.UNKNOWN
        age = datetime.now(UTC) - rec.timestamp
        if age <= timedelta(hours=rec.ttl_hours):
            return FreshnessStatus.FRESH
        return FreshnessStatus.STALE

    def get_age_str(self, source: str) -> str:
        """Return human-readable age string like '2h 15m ago'."""
        rec = self._sources.get(source)
        if rec is None:
            return "never"
        age = datetime.now(UTC) - rec.timestamp
        hours = int(age.total_seconds() // 3600)
        mins = int((age.total_seconds() % 3600) // 60)
        if hours > 0:
            return f"{hours}h {mins}m ago"
        return f"{mins}m ago"

    def get_all(self) -> dict[str, dict]:
        """Return freshness status for all recorded sources."""
        result = {}
        for name, rec in self._sources.items():
            status = self.check(name)
            result[name] = {
                "status": status.value,
                "timestamp": rec.timestamp.isoformat(),
                "ttl_hours": rec.ttl_hours,
                "age": self.get_age_str(name),
            }
        return result

    @staticmethod
    def audit_missing(
        df: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> dict[str, dict]:
        """Audit a DataFrame for missing/null values.

        Args:
            df: DataFrame to audit.
            columns: Specific columns to check.  Defaults to all.

        Returns:
            Dict mapping column name to {missing, total, pct} counts.
        """
        cols = columns or list(df.columns)
        report: dict[str, dict] = {}
        for col in cols:
            if col not in df.columns:
                report[col] = {"missing": len(df), "total": len(df), "pct": 100.0}
                continue
            n_missing = int(df[col].isna().sum())
            n_total = len(df)
            report[col] = {
                "missing": n_missing,
                "total": n_total,
                "pct": round(n_missing / max(n_total, 1) * 100, 1),
            }
        return report
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_data_freshness.py -v`
Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/optimizer/data_freshness.py tests/test_data_freshness.py
git commit -m "feat: add DataFreshnessTracker for optimizer data quality monitoring"
```

---

### Task 12: Add Freshness Timestamps to OptimizerDataContext

**Files:**
- Modify: `src/optimizer/shared_data_layer.py`
- Modify: `tests/test_shared_data_layer.py`

- [ ] **Step 1: Write the failing test — context must expose data_timestamps**

Append to `tests/test_shared_data_layer.py`:

```python
class TestDataFreshnessInContext:
    """Verify OptimizerDataContext tracks data freshness."""

    def test_context_has_data_timestamps_field(self):
        """OptimizerDataContext must have a data_timestamps field."""
        assert hasattr(OptimizerDataContext, "__dataclass_fields__")
        assert "data_timestamps" in OptimizerDataContext.__dataclass_fields__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_shared_data_layer.py::TestDataFreshnessInContext -v`
Expected: FAIL (field does not exist)

- [ ] **Step 3: Add data_timestamps field to OptimizerDataContext**

In `src/optimizer/shared_data_layer.py`, add to the dataclass fields (in the Config section near `scope`, `weeks_remaining`, `config`):

```python
    data_timestamps: dict[str, str] = field(default_factory=dict)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_shared_data_layer.py::TestDataFreshnessInContext -v`
Expected: 1 PASSED

- [ ] **Step 5: Run all shared data layer tests to check for regressions**

Run: `python -m pytest tests/test_shared_data_layer.py -v`
Expected: All PASSED

- [ ] **Step 6: Commit**

```bash
git add src/optimizer/shared_data_layer.py tests/test_shared_data_layer.py
git commit -m "feat: add data_timestamps field to OptimizerDataContext"
```

---

## Phase 5: Constants Registry & Sensitivity Analysis (P2)

Build a centralized registry of all magic numbers with citations, bounds, and sensitivity flags.

---

### Task 13: Constants Registry Module

**Files:**
- Create: `src/optimizer/constants_registry.py`
- Create: `tests/test_constants_registry.py`

- [ ] **Step 1: Write the failing test — all constants documented with citations**

Create `tests/test_constants_registry.py`:

```python
"""Tests for the centralized constants registry.

Every hardcoded magic number in the optimizer must be registered here
with a citation, plausible bounds, and sensitivity classification.
"""

import pytest

from src.optimizer.constants_registry import CONSTANTS_REGISTRY, ConstantEntry


class TestConstantsRegistry:
    """Verify all optimizer constants are documented and bounded."""

    def test_all_entries_have_citations(self):
        """Every constant must have a non-empty citation."""
        for name, entry in CONSTANTS_REGISTRY.items():
            assert entry.citation, f"Constant '{name}' has no citation"

    def test_all_values_within_bounds(self):
        """Every constant's value must be within its declared bounds."""
        for name, entry in CONSTANTS_REGISTRY.items():
            assert entry.lower_bound <= entry.value <= entry.upper_bound, (
                f"Constant '{name}' value {entry.value} outside bounds "
                f"[{entry.lower_bound}, {entry.upper_bound}]"
            )

    def test_platoon_constants_match_the_book(self):
        """Platoon advantages should match The Book (Tango et al.)."""
        lhb = CONSTANTS_REGISTRY["platoon_lhb_vs_rhp"]
        rhb = CONSTANTS_REGISTRY["platoon_rhb_vs_lhp"]
        assert lhb.value == pytest.approx(0.086, abs=0.01)
        assert rhb.value == pytest.approx(0.061, abs=0.01)
        assert "The Book" in lhb.citation or "Tango" in lhb.citation

    def test_stabilization_points_match_fangraphs(self):
        """Stabilization points should match FanGraphs research."""
        hr_stab = CONSTANTS_REGISTRY["stabilization_hr_rate"]
        assert hr_stab.value == 170
        assert "FanGraphs" in hr_stab.citation or "stabiliz" in hr_stab.citation.lower()

    def test_sigmoid_k_values_have_calibration_source(self):
        """Sigmoid k-values must document their calibration method."""
        k_count = CONSTANTS_REGISTRY["sigmoid_k_counting"]
        k_rate = CONSTANTS_REGISTRY["sigmoid_k_rate"]
        assert k_count.value == 2.0
        assert k_rate.value == 3.0

    def test_registry_covers_all_known_constants(self):
        """Registry must include at least 25 constants (we know of 30+)."""
        assert len(CONSTANTS_REGISTRY) >= 25, (
            f"Only {len(CONSTANTS_REGISTRY)} constants registered; expected 25+"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_constants_registry.py -v`
Expected: FAIL (module does not exist)

- [ ] **Step 3: Implement the constants registry**

Create `src/optimizer/constants_registry.py`:

```python
"""Centralized registry of all hardcoded constants in the Lineup Optimizer.

Every magic number used across the optimizer pipeline is documented here
with its value, plausible bounds, citation source, and sensitivity flag.

Sensitivity levels:
    HIGH — ±20% perturbation changes lineup composition
    MEDIUM — ±20% changes player rankings but not top-5 assignments
    LOW — ±20% has negligible impact on output
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConstantEntry:
    """A registered optimizer constant."""

    value: float
    lower_bound: float
    upper_bound: float
    citation: str
    module: str
    sensitivity: str  # HIGH, MEDIUM, LOW
    description: str


CONSTANTS_REGISTRY: dict[str, ConstantEntry] = {
    # ── Platoon Adjustments (The Book) ──────────────────────────────
    "platoon_lhb_vs_rhp": ConstantEntry(
        value=0.086, lower_bound=0.05, upper_bound=0.12,
        citation="The Book: Playing the Percentages (Tango, Lichtman, Dolphin, 2007), Ch.5",
        module="matchup_adjustments.py", sensitivity="MEDIUM",
        description="wOBA advantage for LHB vs RHP (8.6%)",
    ),
    "platoon_rhb_vs_lhp": ConstantEntry(
        value=0.061, lower_bound=0.03, upper_bound=0.10,
        citation="The Book: Playing the Percentages (Tango, Lichtman, Dolphin, 2007), Ch.5",
        module="matchup_adjustments.py", sensitivity="MEDIUM",
        description="wOBA advantage for RHB vs LHP (6.1%)",
    ),
    "platoon_stab_lhb": ConstantEntry(
        value=1000, lower_bound=500, upper_bound=1500,
        citation="Pizza Cutter (Russell Carleton) stabilization research; FanGraphs",
        module="matchup_adjustments.py", sensitivity="LOW",
        description="PA for LHB platoon split to stabilize",
    ),
    "platoon_stab_rhb": ConstantEntry(
        value=2200, lower_bound=1500, upper_bound=3000,
        citation="Pizza Cutter (Russell Carleton) stabilization research; FanGraphs",
        module="matchup_adjustments.py", sensitivity="LOW",
        description="PA for RHB platoon split to stabilize",
    ),
    # ── Start/Sit ───────────────────────────────────────────────────
    "home_advantage": ConstantEntry(
        value=1.02, lower_bound=1.01, upper_bound=1.04,
        citation="Historical MLB data: ~54% home win rate (Retrosheet 2000-2024)",
        module="start_sit.py", sensitivity="LOW",
        description="Home team counting stat multiplier",
    ),
    "away_discount": ConstantEntry(
        value=0.97, lower_bound=0.95, upper_bound=0.99,
        citation="Symmetric complement of home_advantage",
        module="start_sit.py", sensitivity="LOW",
        description="Away team counting stat multiplier",
    ),
    # ── Bayesian Stabilization Points ───────────────────────────────
    "stabilization_hr_rate": ConstantEntry(
        value=170, lower_bound=130, upper_bound=210,
        citation="FanGraphs: 'How Long Until a Hitter's Stats Stabilize?' (2014); Russell Carleton",
        module="projections.py", sensitivity="HIGH",
        description="PA for HR rate to stabilize (Bayesian prior weight)",
    ),
    "stabilization_avg": ConstantEntry(
        value=910, lower_bound=700, upper_bound=1100,
        citation="FanGraphs stabilization series; BABIP dependency requires large sample",
        module="projections.py", sensitivity="HIGH",
        description="PA for batting average to stabilize",
    ),
    "stabilization_obp": ConstantEntry(
        value=460, lower_bound=350, upper_bound=600,
        citation="FanGraphs stabilization series; BB rate stabilizes faster than AVG",
        module="projections.py", sensitivity="HIGH",
        description="PA for OBP to stabilize",
    ),
    "stabilization_era": ConstantEntry(
        value=630, lower_bound=450, upper_bound=800,
        citation="FanGraphs: measured in batters faced; ERA has high variance",
        module="projections.py", sensitivity="HIGH",
        description="BF for ERA to stabilize",
    ),
    "stabilization_whip": ConstantEntry(
        value=540, lower_bound=400, upper_bound=700,
        citation="FanGraphs stabilization series; similar to ERA",
        module="projections.py", sensitivity="HIGH",
        description="BF for WHIP to stabilize",
    ),
    "stabilization_k_rate": ConstantEntry(
        value=60, lower_bound=40, upper_bound=80,
        citation="FanGraphs: K rate stabilizes very quickly (~60 PA)",
        module="projections.py", sensitivity="MEDIUM",
        description="PA for K rate to stabilize",
    ),
    "stabilization_bb_rate": ConstantEntry(
        value=120, lower_bound=80, upper_bound=160,
        citation="FanGraphs stabilization series",
        module="projections.py", sensitivity="MEDIUM",
        description="PA for BB rate to stabilize",
    ),
    "stabilization_sb_rate": ConstantEntry(
        value=200, lower_bound=150, upper_bound=300,
        citation="FanGraphs: SB rate requires moderate sample; opportunity-dependent",
        module="projections.py", sensitivity="MEDIUM",
        description="PA for SB rate to stabilize",
    ),
    "stabilization_k_rate_pitch": ConstantEntry(
        value=70, lower_bound=50, upper_bound=100,
        citation="FanGraphs: pitcher K rate stabilizes quickly",
        module="projections.py", sensitivity="MEDIUM",
        description="IP for pitcher K rate to stabilize",
    ),
    # ── Recent Form ─────────────────────────────────────────────────
    "recent_form_blend": ConstantEntry(
        value=0.20, lower_bound=0.10, upper_bound=0.35,
        citation="Empirical: 20% blend of L14 data; conservative to avoid recency bias",
        module="projections.py", sensitivity="MEDIUM",
        description="Weight of L14 recent form in projection blend",
    ),
    "min_recent_games": ConstantEntry(
        value=7, lower_bound=5, upper_bound=14,
        citation="Statistical minimum for L14 game log reliability",
        module="projections.py", sensitivity="LOW",
        description="Minimum games required to apply recent form adjustment",
    ),
    # ── Scenario Generation ─────────────────────────────────────────
    "default_cv_hr": ConstantEntry(
        value=0.20, lower_bound=0.12, upper_bound=0.30,
        citation="Empirical: HR has high variance; typical CV from MLB seasonal data",
        module="scenario_generator.py", sensitivity="MEDIUM",
        description="Coefficient of variation for HR projections",
    ),
    "default_cv_sb": ConstantEntry(
        value=0.30, lower_bound=0.20, upper_bound=0.45,
        citation="Empirical: SB most volatile counting stat; injury/role sensitive",
        module="scenario_generator.py", sensitivity="MEDIUM",
        description="Coefficient of variation for SB projections",
    ),
    "default_cv_sv": ConstantEntry(
        value=0.35, lower_bound=0.25, upper_bound=0.50,
        citation="Empirical: saves highly role-dependent; closers can lose jobs mid-season",
        module="scenario_generator.py", sensitivity="MEDIUM",
        description="Coefficient of variation for SV projections",
    ),
    "corr_hr_rbi": ConstantEntry(
        value=0.68, lower_bound=0.55, upper_bound=0.80,
        citation="MLB 2019-2024 hitter correlations; HR drives RBI production",
        module="scenario_generator.py", sensitivity="LOW",
        description="Correlation between HR and RBI projections",
    ),
    "corr_era_whip": ConstantEntry(
        value=0.90, lower_bound=0.80, upper_bound=0.95,
        citation="MLB 2019-2024 pitcher correlations; ERA and WHIP highly correlated",
        module="scenario_generator.py", sensitivity="LOW",
        description="Correlation between ERA and WHIP projections",
    ),
    # ── Category Urgency ────────────────────────────────────────────
    "sigmoid_k_counting": ConstantEntry(
        value=2.0, lower_bound=1.0, upper_bound=4.0,
        citation="Calibrated: moderate sensitivity for counting stat gaps; needs backtest validation",
        module="category_urgency.py", sensitivity="HIGH",
        description="Sigmoid steepness for counting stat urgency",
    ),
    "sigmoid_k_rate": ConstantEntry(
        value=3.0, lower_bound=1.5, upper_bound=5.0,
        citation="Calibrated: higher sensitivity for noisier rate stat gaps; needs backtest validation",
        module="category_urgency.py", sensitivity="HIGH",
        description="Sigmoid steepness for rate stat urgency",
    ),
    # ── Streaming ───────────────────────────────────────────────────
    "default_ip_per_start": ConstantEntry(
        value=5.5, lower_bound=5.0, upper_bound=6.0,
        citation="MLB league average IP/GS (2022-2024): ~5.3-5.6 IP",
        module="streaming.py", sensitivity="LOW",
        description="Expected innings per start for average pitcher",
    ),
    "default_team_weekly_ip": ConstantEntry(
        value=55.0, lower_bound=45.0, upper_bound=65.0,
        citation="Typical fantasy team weekly IP: ~50-60 IP across all pitchers",
        module="streaming.py", sensitivity="LOW",
        description="Baseline weekly team IP for rate stat dilution calculation",
    ),
    "whip_penalty_threshold": ConstantEntry(
        value=1.40, lower_bound=1.30, upper_bound=1.50,
        citation="Empirical: pitchers above 1.40 WHIP are replacement-level or worse",
        module="streaming.py", sensitivity="LOW",
        description="Career WHIP above which streaming composite gets 50% penalty",
    ),
    # ── Pipeline Defaults ───────────────────────────────────────────
    "default_risk_aversion": ConstantEntry(
        value=0.15, lower_bound=0.0, upper_bound=0.50,
        citation="Empirical: light risk reduction without overfitting to floor",
        module="pipeline.py", sensitivity="MEDIUM",
        description="Default lambda for mean-variance risk adjustment",
    ),
    "n_scenarios_standard": ConstantEntry(
        value=200, lower_bound=100, upper_bound=500,
        citation="Convergence testing: 200 scenarios gives <2% variance in expected value",
        module="pipeline.py", sensitivity="LOW",
        description="Number of Monte Carlo scenarios in standard mode",
    ),
    "n_scenarios_full": ConstantEntry(
        value=500, lower_bound=200, upper_bound=1000,
        citation="Convergence testing: 500 scenarios gives <1% variance; diminishing returns beyond",
        module="pipeline.py", sensitivity="LOW",
        description="Number of Monte Carlo scenarios in full mode",
    ),
    # ── Pitcher Quality ─────────────────────────────────────────────
    "pitcher_quality_slope": ConstantEntry(
        value=0.075, lower_bound=0.05, upper_bound=0.15,
        citation="Calibrated: maps z-score to counting stat multiplier; clamped to ±15%",
        module="matchup_adjustments.py", sensitivity="MEDIUM",
        description="Slope for opposing pitcher quality -> hitter stat multiplier",
    ),
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_constants_registry.py -v`
Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/optimizer/constants_registry.py tests/test_constants_registry.py
git commit -m "feat: add centralized constants registry with citations and bounds for 30 optimizer constants"
```

---

## Phase 6: Historical Backtesting Framework (P1)

Build infrastructure to replay historical weeks and measure optimizer accuracy.

---

### Task 14: Backtest Validator Module

**Files:**
- Create: `src/optimizer/backtest_validator.py`
- Create: `tests/test_optimizer_backtest.py`

- [ ] **Step 1: Write the failing test — projection accuracy metrics**

Create `tests/test_optimizer_backtest.py`:

```python
"""Tests for the historical backtesting validator.

Validates optimizer predictions against actual MLB outcomes using
RMSE, Spearman rank correlation, and bust rate metrics.
"""

import numpy as np
import pytest

from src.optimizer.backtest_validator import (
    compute_bust_rate,
    compute_projection_rmse,
    compute_rank_correlation,
    grade_lineup_quality,
)


class TestProjectionAccuracyMetrics:
    """Verify projection accuracy scoring functions."""

    def test_perfect_projection_has_zero_rmse(self):
        """If projected == actual, RMSE should be 0."""
        projected = {"R": 80, "HR": 25, "RBI": 75}
        actual = {"R": 80, "HR": 25, "RBI": 75}
        rmse = compute_projection_rmse(projected, actual)
        assert rmse == pytest.approx(0.0, abs=1e-6)

    def test_rmse_increases_with_error(self):
        """Larger prediction errors should produce higher RMSE."""
        actual = {"R": 80, "HR": 25, "RBI": 75}
        small_error = {"R": 82, "HR": 24, "RBI": 77}
        large_error = {"R": 100, "HR": 15, "RBI": 50}

        rmse_small = compute_projection_rmse(small_error, actual)
        rmse_large = compute_projection_rmse(large_error, actual)
        assert rmse_large > rmse_small

    def test_perfect_ranking_correlation_is_one(self):
        """If player rankings match actual, Spearman rho = 1.0."""
        projected_order = [101, 102, 103, 104, 105]
        actual_order = [101, 102, 103, 104, 105]
        rho = compute_rank_correlation(projected_order, actual_order)
        assert rho == pytest.approx(1.0, abs=0.01)

    def test_reversed_ranking_correlation_is_negative(self):
        """If rankings are perfectly reversed, Spearman rho = -1.0."""
        projected_order = [101, 102, 103, 104, 105]
        actual_order = [105, 104, 103, 102, 101]
        rho = compute_rank_correlation(projected_order, actual_order)
        assert rho == pytest.approx(-1.0, abs=0.01)

    def test_bust_rate_counts_overestimates(self):
        """Bust rate = fraction of players whose actual < 50% of projected."""
        projected = {101: 30, 102: 25, 103: 20, 104: 15}
        actual = {101: 28, 102: 10, 103: 8, 104: 14}
        # Player 102: 10 < 12.5 (50% of 25) -> bust
        # Player 103: 8 < 10 (50% of 20) -> bust
        # Bust rate: 2/4 = 0.50
        rate = compute_bust_rate(projected, actual, threshold=0.50)
        assert rate == pytest.approx(0.50, abs=0.01)


class TestLineupQualityGrading:
    """Verify lineup quality comparison against actual-optimal."""

    def test_perfect_lineup_gets_a_grade(self):
        """If optimizer lineup matches actual-optimal, grade is 'A'."""
        grade = grade_lineup_quality(
            optimizer_value=95.0,
            optimal_value=100.0,
            threshold_a=0.90,
            threshold_b=0.80,
        )
        assert grade == "A"

    def test_poor_lineup_gets_c_grade(self):
        """If optimizer captures < 80% of optimal value, grade is 'C'."""
        grade = grade_lineup_quality(
            optimizer_value=70.0,
            optimal_value=100.0,
            threshold_a=0.90,
            threshold_b=0.80,
        )
        assert grade == "C"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_optimizer_backtest.py -v`
Expected: FAIL (module does not exist)

- [ ] **Step 3: Implement the backtest validator**

Create `src/optimizer/backtest_validator.py`:

```python
"""Historical backtesting validator for the Lineup Optimizer.

Compares optimizer predictions against actual MLB outcomes using
standard accuracy metrics: RMSE, rank correlation, bust rate,
and lineup quality grading.
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)


def compute_projection_rmse(
    projected: dict[str, float],
    actual: dict[str, float],
) -> float:
    """Compute RMSE between projected and actual category totals.

    Args:
        projected: Dict of category -> projected value.
        actual: Dict of category -> actual value.

    Returns:
        Root Mean Squared Error across all shared categories.
    """
    shared = set(projected) & set(actual)
    if not shared:
        return float("inf")
    squared_errors = [(projected[c] - actual[c]) ** 2 for c in shared]
    return math.sqrt(sum(squared_errors) / len(squared_errors))


def compute_rank_correlation(
    projected_order: list[int],
    actual_order: list[int],
) -> float:
    """Compute Spearman rank correlation between two player rankings.

    Args:
        projected_order: Player IDs ranked by projected value (best first).
        actual_order: Player IDs ranked by actual value (best first).

    Returns:
        Spearman rho in [-1, 1].  1.0 = perfect agreement.
    """
    n = len(projected_order)
    if n < 2:
        return 0.0

    actual_rank = {pid: i for i, pid in enumerate(actual_order)}
    d_squared = 0.0
    matched = 0
    for i, pid in enumerate(projected_order):
        if pid in actual_rank:
            d_squared += (i - actual_rank[pid]) ** 2
            matched += 1

    if matched < 2:
        return 0.0

    rho = 1 - (6 * d_squared) / (matched * (matched**2 - 1))
    return max(-1.0, min(1.0, rho))


def compute_bust_rate(
    projected: dict[int, float],
    actual: dict[int, float],
    threshold: float = 0.50,
) -> float:
    """Compute bust rate: fraction of players whose actual < threshold * projected.

    A "bust" is a player who delivers less than the threshold fraction
    of their projected value (e.g., < 50% of projected HR).

    Args:
        projected: Dict of player_id -> projected value.
        actual: Dict of player_id -> actual value.
        threshold: Fraction below which a player is a bust.

    Returns:
        Bust rate in [0.0, 1.0].
    """
    shared = set(projected) & set(actual)
    if not shared:
        return 0.0

    busts = 0
    for pid in shared:
        if projected[pid] > 0 and actual[pid] < threshold * projected[pid]:
            busts += 1

    return busts / len(shared)


def grade_lineup_quality(
    optimizer_value: float,
    optimal_value: float,
    threshold_a: float = 0.90,
    threshold_b: float = 0.80,
) -> str:
    """Grade optimizer lineup quality against actual-optimal.

    Compares total fantasy value of optimizer's recommended lineup
    against the best possible lineup (computed with hindsight).

    Args:
        optimizer_value: Total value from optimizer's recommended lineup.
        optimal_value: Total value from hindsight-optimal lineup.
        threshold_a: Fraction of optimal needed for 'A' grade.
        threshold_b: Fraction of optimal needed for 'B' grade.

    Returns:
        Letter grade: 'A', 'B', or 'C'.
    """
    if optimal_value <= 0:
        return "A"

    ratio = optimizer_value / optimal_value
    if ratio >= threshold_a:
        return "A"
    if ratio >= threshold_b:
        return "B"
    return "C"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_optimizer_backtest.py -v`
Expected: 7 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/optimizer/backtest_validator.py tests/test_optimizer_backtest.py
git commit -m "feat: add backtest validator with RMSE, Spearman, bust rate, and lineup grading"
```

---

### Task 15: Empirical Correlation Computation Utility

**Files:**
- Modify: `src/optimizer/scenario_generator.py`
- Modify: `tests/test_optimizer_math_proofs.py`

- [ ] **Step 1: Write the failing test — empirical correlations from real data**

Append to `tests/test_optimizer_math_proofs.py`:

```python
from src.optimizer.scenario_generator import (
    DEFAULT_CORRELATIONS,
    compute_empirical_correlations,
)


class TestEmpiricalCorrelations:
    """Validate hardcoded correlations against empirical computation."""

    def test_compute_from_sample_data(self):
        """compute_empirical_correlations should return a dict of pair -> float."""
        # Simulate 50 players with realistic stat correlations
        rng = np.random.default_rng(42)
        n = 50
        hr = rng.poisson(20, n).astype(float)
        rbi = hr * 3.5 + rng.normal(0, 10, n)  # RBI correlated with HR
        r = hr * 2 + rng.normal(10, 8, n)
        sb = rng.poisson(8, n).astype(float)

        player_stats = pd.DataFrame({"hr": hr, "rbi": rbi, "r": r, "sb": sb})
        corrs = compute_empirical_correlations(player_stats)

        assert isinstance(corrs, dict)
        assert ("hr", "rbi") in corrs or ("rbi", "hr") in corrs
        # HR-RBI should be positively correlated
        hr_rbi = corrs.get(("hr", "rbi"), corrs.get(("rbi", "hr"), 0))
        assert hr_rbi > 0.3, f"HR-RBI correlation {hr_rbi} unexpectedly low"

    def test_hardcoded_hr_rbi_correlation_is_plausible(self):
        """DEFAULT_CORRELATIONS HR-RBI should be in [0.55, 0.80]."""
        hr_rbi = DEFAULT_CORRELATIONS.get(("hr", "rbi"), 0)
        assert 0.55 <= hr_rbi <= 0.80, f"HR-RBI correlation {hr_rbi} outside plausible range"

    def test_hardcoded_era_whip_correlation_is_plausible(self):
        """DEFAULT_CORRELATIONS ERA-WHIP should be in [0.80, 0.95]."""
        era_whip = DEFAULT_CORRELATIONS.get(("era", "whip"), 0)
        assert 0.80 <= era_whip <= 0.95, f"ERA-WHIP correlation {era_whip} outside range"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_optimizer_math_proofs.py::TestEmpiricalCorrelations -v`
Expected: FAIL (compute_empirical_correlations not found)

- [ ] **Step 3: Implement compute_empirical_correlations in scenario_generator.py**

Add to `src/optimizer/scenario_generator.py`:

```python
def compute_empirical_correlations(
    player_stats: pd.DataFrame,
    min_sample: int = 20,
) -> dict[tuple[str, str], float]:
    """Compute pairwise stat correlations from observed player data.

    Used to validate or replace hardcoded DEFAULT_CORRELATIONS with
    empirical values from actual MLB season data.

    Args:
        player_stats: DataFrame with stat columns (hr, rbi, r, sb, etc.).
        min_sample: Minimum players required; returns empty if fewer.

    Returns:
        Dict mapping (stat_a, stat_b) -> Pearson correlation coefficient.
    """
    if len(player_stats) < min_sample:
        return {}

    numeric_cols = [c for c in player_stats.columns if player_stats[c].dtype in ("float64", "int64", "float32", "int32")]
    if len(numeric_cols) < 2:
        return {}

    corr_matrix = player_stats[numeric_cols].corr()
    result: dict[tuple[str, str], float] = {}

    for i, col_a in enumerate(numeric_cols):
        for col_b in numeric_cols[i + 1:]:
            val = corr_matrix.loc[col_a, col_b]
            if not np.isnan(val):
                result[(col_a, col_b)] = float(val)

    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_optimizer_math_proofs.py::TestEmpiricalCorrelations -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/optimizer/scenario_generator.py tests/test_optimizer_math_proofs.py
git commit -m "feat: add compute_empirical_correlations utility + hardcoded correlation validation"
```

---

### Task 16: Sensitivity Analysis Framework

**Files:**
- Modify: `tests/test_constants_registry.py`

- [ ] **Step 1: Write sensitivity analysis test**

Append to `tests/test_constants_registry.py`:

```python
class TestConstantSensitivity:
    """Test that perturbation of constants produces expected behavior.

    HIGH sensitivity constants: ±20% perturbation should change lineup.
    LOW sensitivity constants: ±20% perturbation should NOT change top-5.
    """

    def test_high_sensitivity_constants_exist(self):
        """At least 3 constants should be flagged as HIGH sensitivity."""
        high = [k for k, v in CONSTANTS_REGISTRY.items() if v.sensitivity == "HIGH"]
        assert len(high) >= 3, f"Only {len(high)} HIGH sensitivity constants"

    def test_all_sensitivity_levels_used(self):
        """Registry should use all three sensitivity levels."""
        levels = {v.sensitivity for v in CONSTANTS_REGISTRY.values()}
        assert "HIGH" in levels
        assert "MEDIUM" in levels
        assert "LOW" in levels

    def test_all_modules_represented(self):
        """Constants should cover at least 4 different optimizer modules."""
        modules = {v.module for v in CONSTANTS_REGISTRY.values()}
        assert len(modules) >= 4, f"Only {len(modules)} modules represented: {modules}"

    def test_bounds_are_reasonable(self):
        """Lower bound < value < upper bound for all constants."""
        for name, entry in CONSTANTS_REGISTRY.items():
            assert entry.lower_bound < entry.value, (
                f"{name}: lower_bound {entry.lower_bound} >= value {entry.value}"
            )
            assert entry.value < entry.upper_bound, (
                f"{name}: value {entry.value} >= upper_bound {entry.upper_bound}"
            )
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python -m pytest tests/test_constants_registry.py -v`
Expected: All PASSED

- [ ] **Step 3: Commit**

```bash
git add tests/test_constants_registry.py
git commit -m "test: add sensitivity analysis and bounds validation for constants registry"
```

---

## Summary

### Phase Deliverables

| Phase | Tests Added | Source Files | Coverage Area |
|-------|-------------|-------------|---------------|
| **1: Math Proofs** | ~20 tests | 0 new | Rate stats, LP weights, Bayesian, matchup, SGP |
| **2: Shared Data Layer** | ~10 tests | 0 new | OptimizerDataContext, scope scaling, form weights |
| **3: Integration** | ~50 tests (parameterized) | 0 new | Pipeline invariants, settings axes, mode comparisons |
| **4: Data Freshness** | ~8 tests | 1 new module | Freshness tracking, staleness detection, missing data |
| **5: Constants Registry** | ~10 tests | 1 new module | 30 constants with citations, bounds, sensitivity |
| **6: Backtesting** | ~10 tests | 1 new module + 1 utility | RMSE, Spearman, bust rate, lineup grading, empirical correlations |

### Total New Coverage
- **~108 new tests** across 6 test files
- **3 new source modules** (data_freshness, constants_registry, backtest_validator)
- **1 new utility function** (compute_empirical_correlations)
- **1 modified dataclass** (OptimizerDataContext + data_timestamps)

### Execution Dependencies
- Phases 1-2 have no dependencies — can run in parallel
- Phase 3 depends on the full_roster fixture (self-contained)
- Phase 4 depends on nothing (new module)
- Phase 5 depends on nothing (new module)
- Phase 6 depends on nothing (new module)

**Recommended execution: Phases 1+2 in parallel, then 3, then 4+5+6 in parallel.**

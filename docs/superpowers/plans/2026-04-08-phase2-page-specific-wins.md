# Phase 2: Page-Specific Wins — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the 9 highest-impact page-specific improvements from ROADMAP Tier 2. These are the biggest accuracy wins per page after the core engine was unified in Phase 1.

**Architecture:** 9 tasks organized into 4 parallel waves. Each wave's tasks target different pages so agents don't conflict on the same files.

**Tech Stack:** Python 3.14, pandas, NumPy, SciPy, pybaseball, PuLP, SQLite, Streamlit, pytest, XGBoost (optional)

**Prerequisite:** Phase 1 complete (`7975db5` or later). All Tier 1 items verified.

**Total: 9 tasks, 4 waves, estimated 4-6 hours with parallel agents.**

---

## Wave 5: Lineup Optimizer + Trade Finder — PARALLEL (3 agents)

These target different files so they can run simultaneously.

---

### Task 1: I1+I4 — Mid-Week Pivot Advisor + Category Flip Probability

**Agent:** `lineup-pivot`

**Problem:** No mid-week category state tracking. Optimizer runs once at start of week. Identifying 2-3 lost causes by Wednesday and redirecting resources swings +1-2 category wins/week.

**Files:**
- Create: `src/optimizer/pivot_advisor.py` (~150 lines)
- Modify: `src/optimizer/category_urgency.py:123-200` — add flip_probability to return dict
- Modify: `pages/5_Line-up_Optimizer.py` — add pivot advisory panel
- Test: `tests/test_pivot_advisor.py` (new)

**Algorithm:**
```python
def compute_category_flip_probabilities(
    my_totals: dict, opp_totals: dict,
    games_remaining: int, config: LeagueConfig,
) -> dict[str, dict]:
    """Per-category flip probability + WON/LOST/CONTESTED classification.

    For each category:
    1. current_margin = my_val - opp_val (flipped for inverse)
    2. daily_stdev = WEEKLY_SD[cat] / sqrt(7) * sqrt(games_remaining)
    3. flip_prob = norm.cdf(-abs(current_margin) / daily_stdev)
    4. Classify: WON (P(flip)<15%), LOST (P(win)<15%), CONTESTED (30-70%)

    Returns: {cat: {margin, flip_prob, classification, action}}
    """
```

**Weekly SDs** (from ROADMAP R2 research, FanGraphs 48-league study):
```python
WEEKLY_SD = {
    "R": 1.6, "HR": 2.1, "RBI": 2.3, "SB": 2.3,
    "AVG": 2.9, "OBP": 2.5,
    "W": 1.7, "L": 1.7, "SV": 1.8, "K": 1.2,
    "ERA": 2.2, "WHIP": 2.0,
}
```

**Steps:**
- [ ] Write test: flip_prob for tied categories ≈ 0.5, large leads ≈ 0.0
- [ ] Write test: classification thresholds (WON/LOST/CONTESTED)
- [ ] Write test: action recommendations (bench pitchers for WON ratios)
- [ ] Implement `compute_category_flip_probabilities()`
- [ ] Add `flip_probabilities` key to `compute_urgency_weights()` return dict
- [ ] Wire into Lineup Optimizer page as an advisory panel
- [ ] Run tests, lint, commit

---

### Task 2: G1 — xwOBA Regression Flags (Buy-Low / Sell-High)

**Agent:** `trade-regression`

**Problem:** No expected stats in trade valuation. The xwOBA-wOBA gap is the single most actionable regression signal — a gap ≥0.030 predicts 40-80 point wOBA surge.

**Files:**
- Modify: `src/database.py:1011-1050` (`_enrich_pool()`) — add xwoba_delta column
- Modify: `src/database.py:1053-1152` (`_load_player_pool_impl()`) — LEFT JOIN statcast_archive for xwoba
- Modify: `src/trade_finder.py` — add regression bonus to composite score
- Test: `tests/test_xwoba_regression.py` (new)

**Algorithm:**
1. In SQL queries, add: `LEFT JOIN statcast_archive sa ON p.player_id = sa.player_id AND sa.season = 2026`
2. Select: `sa.xwoba, sa.xba, sa.barrel_pct, sa.hard_hit_pct, sa.ev_mean`
3. In `_enrich_pool()`: compute `xwoba_delta = xwoba - woba` (derive wOBA from AVG/OBP/SLG or use OBP as proxy)
4. Flag: `regression_flag = "BUY_LOW" if xwoba_delta >= 0.030 else "SELL_HIGH" if xwoba_delta <= -0.030 else ""`
5. In Trade Finder composite: when receiving a BUY_LOW player, add +0.03 bonus. When giving a SELL_HIGH, add +0.03 bonus.

**Steps:**
- [ ] Write test: xwoba_delta computed correctly from enriched pool
- [ ] Write test: BUY_LOW flag at ≥0.030 gap, SELL_HIGH at ≤-0.030
- [ ] Write test: composite bonus applied in trade scoring
- [ ] Add LEFT JOIN statcast_archive to all 3 SQL queries in _load_player_pool_impl()
- [ ] Add xwoba_delta + regression_flag to _enrich_pool()
- [ ] Add regression bonus to scan_1_for_1() composite
- [ ] Run tests, lint, commit

---

### Task 3: I2 — Swing-Category Weighting

**Agent:** `lineup-swing`

**Problem:** H2H weights use Normal PDF proximity to toss-up. Does not distinguish "swing" categories (30-70% P(win)) from locked/lost categories. Research shows maximizing investment in swing categories yields +0.5-1 category wins/week.

**Files:**
- Modify: `src/optimizer/h2h_engine.py:83-142` — add swing emphasis multiplier
- Test: `tests/test_swing_weights.py` (new)

**Algorithm:**
After computing `z = gap / sigma` and `p_win = norm.cdf(z)`:
```python
# Emphasize swing categories (30-70% P(win)) where marginal investment matters most
if 0.30 <= p_win <= 0.70:
    emphasis = 1.5  # Toss-up: high marginal value
elif 0.15 <= p_win < 0.30 or 0.70 < p_win <= 0.85:
    emphasis = 1.0  # Moderate: standard weight
else:
    emphasis = 0.5  # Locked win (>85%) or lost cause (<15%): deprioritize
raw_weights[cat] = float(norm.pdf(z)) / sigma * emphasis
```

**Steps:**
- [ ] Write test: swing categories (P(win)=50%) get 1.5x weight
- [ ] Write test: locked categories (P(win)=90%) get 0.5x weight
- [ ] Write test: total weights still normalize to mean=1.0
- [ ] Implement emphasis multiplier in compute_h2h_category_weights()
- [ ] Run tests, lint, commit

---

## Wave 6: My Team + Leaders + Closer Monitor — PARALLEL (3 agents)

---

### Task 4: M1 — Category Flip Analyzer on My Team

**Agent:** `myteam-flip`

**Problem:** My Team War Room shows matchup pulse but doesn't compute which categories are close enough to flip with a single roster move.

**Files:**
- Modify: `src/war_room.py` — integrate flip probabilities from pivot_advisor
- Modify: `pages/1_My_Team.py:760-766` — render flip probability cards
- Test: `tests/test_myteam_flip.py` (new)

**Depends on:** Task 1 (I1+I4) — uses `compute_category_flip_probabilities()`

**Steps:**
- [ ] Write test: flip analyzer returns categories with probability
- [ ] Import and call `compute_category_flip_probabilities()` in War Room
- [ ] Render "Categories Near Flipping" card with probability bars and action suggestions
- [ ] Run tests, lint, commit

---

### Task 5: O1 — Statcast Breakout Score

**Agent:** `leaders-breakout`

**Problem:** Current breakout detection uses simple z-scores of actual vs projected. Doesn't incorporate Statcast process metrics (barrel rate, xwOBA, bat speed) that predict breakouts before outcomes show.

**Files:**
- Modify: `src/leaders.py:151-200` — enhance `detect_breakouts()` with Statcast composite
- Test: `tests/test_breakout_score.py` (new)

**Depends on:** G1 (Task 2) — needs xwoba/barrel in enriched pool

**Algorithm:**
```python
def compute_breakout_score(player_row, config) -> float:
    """Composite breakout score 0-100.

    Hitters: EV50 percentile (30%) + barrel_rate YoY change (25%) +
             xwOBA-wOBA gap (20%) + bat_speed percentile (15%) +
             launch_angle change (10%)

    Pitchers: SwStr% percentile (30%) + Stuff+ (25%) +
              SIERA-ERA gap (25%) + K-BB% (20%)
    """
```

**Steps:**
- [ ] Write test: breakout score for player with elite Statcast = high score
- [ ] Write test: breakout score for average Statcast = medium score
- [ ] Write test: pitcher breakout uses Stuff+ and SwStr%
- [ ] Implement `compute_breakout_score()` in leaders.py
- [ ] Wire into `detect_breakouts()` as alternative/enhanced scoring
- [ ] Display in Leaders page Breakout tab
- [ ] Run tests, lint, commit

---

### Task 6: L1 — gmLI Closer Trust Tracking

**Agent:** `closer-gmli`

**Problem:** Closer monitor uses basic 0.6×hierarchy + 0.4×saves formula. gmLI (game-entry Leverage Index) dropping from 2.0+ to <1.5 is a 1-2 week earlier warning of closer demotion than counting blown saves.

**Files:**
- Modify: `src/closer_monitor.py:8-15` — add gmLI component to job security
- Modify: `pages/7_Closer_Monitor.py` — display gmLI trend indicator
- Test: `tests/test_closer_gmli.py` (new)

**Depends on:** T5 data fetch (gmLI per reliever from pybaseball). If data not yet fetched, use fallback default.

**Algorithm:**
```python
def compute_job_security(hierarchy_confidence, projected_sv, gmli=None, gmli_trend=None):
    """Enhanced job security with gmLI trust signal.

    New formula: 0.45 * hierarchy + 0.30 * saves_component + 0.25 * gmli_component
    gmli_component = 1.0 if gmli >= 1.8, scales down linearly to 0.0 at gmli = 0.5
    gmli_trend penalty: -0.15 if gmli dropped >0.5 in last 2 weeks
    """
```

**Steps:**
- [ ] Write test: high gmLI (2.0+) produces high security
- [ ] Write test: dropping gmLI triggers trend penalty
- [ ] Write test: missing gmLI falls back to original formula
- [ ] Implement enhanced compute_job_security() with gmLI
- [ ] Add gmLI indicator to Closer Monitor cards
- [ ] Run tests, lint, commit

---

## Wave 7: Global ML Models — PARALLEL (2 agents)

---

### Task 7: D1 — Statcast XGBoost Regression Model

**Agent:** `statcast-ml`

**Problem:** XGBoost stub exists but has no training pipeline. Falls back to zeros. The model should learn what projections miss using Statcast features.

**Files:**
- Modify: `src/ml_ensemble.py` — implement training + prediction
- Modify: `src/data_bootstrap.py` — add training step to bootstrap
- Test: `tests/test_xgboost_regression.py` (new)

**Algorithm:**
- Features: exit_velocity, barrel_rate, hard_hit_pct, xwOBA, xBA, sprint_speed, age, health_score
- Target: (actual_fantasy_value_30d_later - projected_value) from 2025 historical data
- Model: XGBoost with max_depth=4, eta=0.05, n_estimators=100 (conservative)
- Output: correction per player, clamped to ±2.0 SGP, blended at ML_WEIGHT=0.1
- Graceful: if xgboost not installed, return zeros (existing fallback)

**Steps:**
- [ ] Write test: model trains on synthetic data without error
- [ ] Write test: predictions are clamped to ±2.0
- [ ] Write test: graceful fallback when xgboost unavailable
- [ ] Implement `train_ensemble()` and `predict_corrections()`
- [ ] Add training step to bootstrap (after Statcast data loaded)
- [ ] Run tests (skip if xgboost not installed), lint, commit

---

### Task 8: D2 — Playing Time Prediction Model

**Agent:** `playtime-ml`

**Problem:** Static preseason PA/IP from Steamer/ZiPS, never updated. OOPSY dominated 2025 projection accuracy mainly via playing time. Playing time is the #1 source of projection error.

**Files:**
- Create: `src/playing_time_model.py` (~120 lines)
- Modify: `src/data_bootstrap.py` — add weekly PT update
- Test: `tests/test_playing_time.py` (new)

**Algorithm:**
```python
def predict_remaining_pa(
    player_id, recent_pa_rate, depth_chart_slot,
    health_score, age, is_platoon, weeks_remaining,
) -> float:
    """Ridge regression: remaining_PA = f(features).

    Features: recent_PA/game (L14), depth_chart position (starter=1.0, bench=0.3),
    health_score, age_factor, platoon_pct, weeks_remaining.

    Apply as multiplier: adjusted_counting_stat = projected * (predicted_PA / projected_PA)
    """
```

**Steps:**
- [ ] Write test: starter with high recent PA gets high prediction
- [ ] Write test: injured player gets reduced PA
- [ ] Write test: prediction scales with weeks_remaining
- [ ] Implement `predict_remaining_pa()` with ridge regression
- [ ] Add `compute_all_playing_time()` batch function
- [ ] Run tests, lint, commit

---

## Wave 8: Trade Finder LP Fix — Single Agent

---

### Task 9: U2 — LP-Constrained Totals in Trade Finder By Value Tab

**Agent:** `trade-lp`

**Problem:** `scan_1_for_1()` uses `_roster_category_totals()` which counts ALL players including bench. The Trade Analyzer engine uses LP-constrained 18-starter totals. By Value tab overcounts bench production, making trades look better than they are.

**Files:**
- Modify: `src/trade_finder.py:690-693` — replace raw totals with LP-constrained totals
- Modify: `src/trade_finder.py` — add LP pre-computation for user baseline
- Test: `tests/test_trade_lp.py` (new)

**Algorithm:**
1. Before the scan loop, compute LP-optimal starting lineup for user's current roster
2. Use LP-constrained totals (18 starters only) for the baseline
3. For each candidate trade: recompute LP-optimal lineup with new roster
4. Compare LP-constrained before vs after totals
5. Performance optimization: LP solve once for baseline, then only for promising trades (filter first by raw SGP, then LP-validate top candidates)

**Steps:**
- [ ] Write test: LP totals exclude bench players
- [ ] Write test: LP totals ≤ raw totals (bench contributes ≥0 SGP)
- [ ] Write test: LP-based composite score differs from raw-based
- [ ] Pre-compute LP baseline in find_trade_opportunities()
- [ ] Add LP recomputation for trade candidates in scan_1_for_1()
- [ ] Optimize: LP-validate only top 50 candidates by raw SGP
- [ ] Run tests, lint, commit

---

## Verification (After All Waves)

- [ ] `python -m pytest tests/ -x -q` — all pass (2500+ expected)
- [ ] `python -m ruff check src/ tests/` — clean
- [ ] Verify: enriched pool has xwoba_delta and regression_flag columns
- [ ] Verify: Lineup Optimizer shows flip probabilities
- [ ] Verify: Leaders page shows Statcast breakout scores
- [ ] Verify: Closer Monitor shows gmLI indicators
- [ ] Verify: Trade Finder By Value uses LP-constrained totals

---

## Agent Dispatch Summary

| Wave | Agent | Task | ROADMAP | Files | Time Est |
|------|-------|------|---------|-------|----------|
| **5** | `lineup-pivot` | Mid-week pivot + flip probability | I1+I4 | pivot_advisor.py (new), category_urgency.py | 45 min |
| **5** | `trade-regression` | xwOBA regression flags | G1 | database.py, trade_finder.py | 40 min |
| **5** | `lineup-swing` | Swing-category weighting | I2 | h2h_engine.py | 20 min |
| **6** | `myteam-flip` | Category flip on My Team | M1 | war_room.py, 1_My_Team.py | 30 min |
| **6** | `leaders-breakout` | Statcast breakout score | O1 | leaders.py, 9_Leaders.py | 35 min |
| **6** | `closer-gmli` | gmLI trust tracking | L1 | closer_monitor.py, 7_Closer_Monitor.py | 25 min |
| **7** | `statcast-ml` | XGBoost regression model | D1 | ml_ensemble.py, data_bootstrap.py | 45 min |
| **7** | `playtime-ml` | Playing time prediction | D2 | playing_time_model.py (new) | 35 min |
| **8** | `trade-lp` | LP-constrained By Value | U2 | trade_finder.py | 40 min |

**Dependencies:**
- Wave 6 (Tasks 4, 5) depends on Wave 5 (Tasks 1, 2) completing first
- Wave 7 is independent of Waves 5-6
- Wave 8 depends on Phase 1 V2 (already complete)

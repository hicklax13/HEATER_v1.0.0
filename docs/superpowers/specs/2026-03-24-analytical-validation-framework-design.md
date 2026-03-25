# HEATER Analytical Validation Framework — Design Spec

**Date:** 2026-03-24
**Status:** Foundation implemented, integration pending
**Scope:** Fix the 4 systemic root causes found in the full analytical audit

## Problem Statement

HEATER has ~40,000 lines of analytical code with **zero empirical validation**. The 1,956 existing tests verify code runs without errors — not that recommendations are accurate. An audit of all 60+ source files found ~80 distinct analytical failures across 4 systemic root causes:

1. **Zero empirical validation** — ~30 magic numbers, 0 backtests
2. **Silent degradation architecture** — ~20 silent fallbacks shown as green checkmarks
3. **Disconnected pipelines** — ~15 modules that compute but don't influence decisions
4. **Missing real-world context** — ~15 hardcoded defaults where live data should flow

## Solution Architecture

### Layer 1: AnalyticsContext (Transparency Spine)

**File:** `src/analytics_context.py` (implemented)

A dataclass that flows through every pipeline, tracking:
- Which modules executed vs. fell back vs. errored
- Data source freshness and quality
- Auto-computed quality score (0-1) and confidence tier
- User-facing warnings

**Key design decisions:**
- Exceptions inside `track_module()` are swallowed (preserving graceful degradation) but **recorded** — so the UI can show what happened
- Quality score is input-based (data + module execution), NOT prediction-based — because we can't measure prediction accuracy without the validation harness
- Skipped modules (intentional, mode-based) don't penalize quality — only disabled/errored ones do
- Returns `None` for missing context (schedule_strength, momentum) instead of fake defaults

### Layer 2: Validation Harness

**Files:** `src/validation/` (implemented)

Infrastructure to answer "does this recommendation actually work?"

- `calibration_data.py` — Fetches historical drafts, trades, matchups from Yahoo
- `survival_calibrator.py` — Compares predicted survival probability vs. actual pick availability using Brier score; finds optimal sigma via `scipy.optimize.minimize_scalar`
- `constant_optimizer.py` — Registry of all 32 magic numbers with bounds, source files, and calibration infrastructure
- `dynamic_context.py` — Computes real-world values that are currently hardcoded (weeks_remaining, schedule_strength, injury_exposure, momentum)

### Layer 3: Module Triage

**File:** `src/validation/module_triage.py` (implemented)

Every analytical module gets a verdict:

| Verdict | Count | Modules |
|---------|-------|---------|
| KILL | 4 | Regime detection, Bellman DP, adverse selection, multi-period optimization |
| FIX | 8 | Matchup adjustments, Start/Sit, Waiver Wire weeks_remaining, Power Rankings, injury badges, trade confidence, bootstrap failures, Phase 1 vs legacy |
| KEEP | 5 | Core LP, H2H weights, SGP theory, Phase 1 trade eval, maximin LP |
| CONNECT | 3 | CVaR → LP, streaming → lineup swaps, MC → deterministic side-by-side |
| DEMOTE | 2 | Scenario generation, "Bayesian" update (actually Marcel) |

### Layer 4: UI Analytics Badge

**File:** `src/ui_analytics_badge.py` (implemented)

Renders AnalyticsContext as a collapsible panel on every page:
- Color-coded confidence tier dot
- Module execution summary
- Data freshness warnings
- Expandable detail table

## Files Created

| File | Lines | Tests | Purpose |
|------|-------|-------|---------|
| `src/analytics_context.py` | 230 | 27 | Transparency spine |
| `src/validation/__init__.py` | 15 | — | Package |
| `src/validation/calibration_data.py` | 280 | — | Yahoo historical data fetcher |
| `src/validation/survival_calibrator.py` | 200 | — | Survival probability calibration |
| `src/validation/constant_optimizer.py` | 350 | 19 | Magic number registry + calibration |
| `src/validation/dynamic_context.py` | 160 | 17 | Dynamic weeks_remaining etc. |
| `src/validation/module_triage.py` | 300 | — | Kill/fix/keep decisions |
| `src/ui_analytics_badge.py` | 160 | — | UI transparency component |
| `tests/test_analytics_context.py` | 170 | 27 | Context tests |
| `tests/test_dynamic_context.py` | 120 | 17 | Dynamic context tests |
| `tests/test_constant_optimizer.py` | 120 | 19 | Constant registry tests |

**Total: 97 new tests, all passing. Existing 1956 tests unaffected.**

## Integration Plan (Next Steps)

### Phase 1: Kill Dead Code (Effort: trivial, 1-2 hours)
1. Delete `_apply_regime_adjustment()` from projections.py
2. Delete `game_theory/dynamic_programming.py`
3. Delete `game_theory/adverse_selection.py`
4. Delete `optimizer/multi_period.py`
5. Remove references from pipeline.py and trade_evaluator.py
6. Update CLAUDE.md module counts

### Phase 2: Wire AnalyticsContext (Effort: medium, 4-6 hours)
1. Add `ctx = AnalyticsContext(pipeline="...")` to each pipeline entry point
2. Replace every `try/except: logger.warning(...)` with `ctx.track_module(...)`
3. Add `ctx.stamp_data(...)` in bootstrap for each data source
4. Call `render_analytics_badge(ctx)` on each page after recommendations
5. Pass `ctx` through trade_evaluator, draft_engine, lineup_optimizer

### Phase 3: Fix weeks_remaining and Power Rankings (Effort: small, 2-3 hours)
1. Replace `weeks_remaining=16` with `compute_weeks_remaining()` everywhere
2. Replace hardcoded power ranking components with `compute_schedule_strength()`, `compute_injury_exposure()`, `compute_momentum()` — returning `None` when data unavailable
3. Show "N/A" in UI instead of fake values

### Phase 4: Wire Missing Inputs to Start/Sit and Matchup (Effort: medium, 4-6 hours)
1. Load opponent weekly totals in pages/11 and pass to start_sit_recommendation()
2. Load park_factors in pages/11 and pass through
3. Wire schedule_grid.py output into pages/6 for matchup adjustments
4. Add AnalyticsContext stamps for which inputs were available

### Phase 5: Calibrate Constants (Effort: large, ongoing)
1. Connect Yahoo client to calibration_data.py
2. Fetch 2025 season historical data
3. Run survival_calibrator to find optimal sigma
4. Build trade_calibrator and lineup_calibrator
5. Replace hardcoded constants with ConstantSet.get() calls
6. Store calibrated values in data/calibrated_constants.json

### Phase 6: Rename "Bayesian" to "Marcel" (Effort: trivial, 30 min)
1. Rename all UI references from "Bayesian" to "Marcel Stabilization"
2. Update CLAUDE.md to be honest about what runs in production
3. If real Bayesian is wanted, make PyMC a hard dependency

## Design Principles

1. **Return None, not fake defaults** — `compute_schedule_strength()` returns `None` when data is unavailable, not `0.5`. The UI shows "N/A" instead of a lie.

2. **Record everything, hide nothing** — `AnalyticsContext` records every module status. The UI expander shows it all. Users can trust the system because it doesn't pretend.

3. **Calibrate from outcomes, not intuition** — Every magic number has bounds and a calibration function. The goal is to replace `sigma=10.0 # guess` with `sigma=constants.get("survival_sigma") # 13.7, calibrated from 2025 Yahoo draft data`.

4. **Kill dead code, don't comment it out** — Modules that don't work should be deleted, not wrapped in `if False:`. The "11-module pipeline" should honestly report however many modules actually run.

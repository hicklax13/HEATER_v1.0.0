# Implementation Plan: Lineup-Constrained Trade Evaluation

**Spec:** `2026-03-16-lineup-constrained-trade-eval-design.md`
**Date:** 2026-03-16

## Step 1: Add helper functions to trade_evaluator.py

Add three new private functions to `src/engine/output/trade_evaluator.py`:

1. `_lineup_constrained_totals(roster_ids, player_pool, config)` — LP wrapper
2. `_find_drop_candidate(bench_ids, player_pool, sgp_calc)` — lowest SGP bench player
3. `_find_fa_pickup(player_pool, sgp_calc, need_hitter, median_sgp_cap)` — best FA

Add imports: `LineupOptimizer`, `ROSTER_SLOTS`, `PULP_AVAILABLE` from `src/lineup_optimizer`.

## Step 2: Rewire evaluate_trade() core logic

Replace lines 244-245 (`_roster_category_totals` calls) with `_lineup_constrained_totals`.
Replace lines 297-335 (bench_cost/bench_bonus block) with roster cap enforcement:
- Detect net_roster_growth
- If positive: run LP, find drop candidates, remove them
- If negative: find FA pickups, add them, then run LP
- If zero: just run LP
Set `bench_cost = 0.0` always. Add `drop_candidate`, `fa_pickup`, `lineup_constrained` to result dict.

## Step 3: Update Trade Analyzer UI

Update `pages/3_Trade_Analyzer.py`:
- Replace "Bench Cost" metric with "Roster Move" showing drop/pickup info
- Add tooltip for the new metric

Update `src/ui_shared.py`:
- Add `roster_move` tooltip to `METRIC_TOOLTIPS`

## Step 4: Write tests

Add 9 functional tests in `TestLineupConstrainedEval` class in `tests/test_trade_engine.py`.
Add 3 math verification tests in `TestLineupConstraintMath` class in `tests/test_trade_engine_math.py`.

## Step 5: Update CLAUDE.md

Update API signatures, gotchas, test counts, and architecture description.

## Step 6: Run full test suite + lint

Verify all tests pass and code is lint-clean.

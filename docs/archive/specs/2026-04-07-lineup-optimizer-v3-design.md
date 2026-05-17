# Line-up Optimizer V3 — Unified Optimizer with FA Intelligence

**Date:** 2026-04-07
**Status:** Approved, implementing

## Problem

The Line-up Optimizer page has 6 tabs with significant data inconsistencies (4 different category weight systems, health scores computed 2 ways, MatchupContextService only wired into 1 tab), the H2H tab is broken when standings are empty, and no tab considers free agent upgrades in the context of the current matchup.

## Solution

Approach B: Shared Data Layer + Scope Router. Keep LP and DCV as separate engines, feed them identical enriched data through `OptimizerDataContext`.

## Architecture

- `src/optimizer/shared_data_layer.py` — `OptimizerDataContext` dataclass + `build_optimizer_context()` builder
- `src/optimizer/fa_recommender.py` — Post-optimization FA add/drop with AVIS constraints
- `pages/5_Line-up_Optimizer.py` — Unified "Optimizer" tab with 3 scopes, context panel fixes, H2H fix, tab consistency

## Key Design Decisions

1. **Separate engines, shared data** — LP (binary assignment) and DCV (daily scoring) solve different problems. Unifying them would compromise mathematical correctness.
2. **Correlation-aware weights** — HR/R/RBI cluster dampened when one is already strong. Prevents double-counting power production.
3. **AVIS hard constraints** — FA recommendations enforce: 10 adds/week, min 2 closers, IL stash protection, 3+ category worsening rejection.
4. **Category-win-maximizing LP** — Weight categories by win probability sensitivity (40-60% = highest weight) instead of raw SGP.

## Implementation Plan

See `C:\Users\conno\.claude\plans\rustling-watching-zephyr.md`

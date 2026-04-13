# Player Databank Integration — Design Spec

**Date:** 2026-04-13
**Scope:** Top 5 pages × Player Databank data gaps
**Audit:** 42 findings from 5 page agents + 1 critique agent

## Problem

The Player Databank page provides rich data (game logs, rolling stats, Statcast, multi-season history) that the rest of the app doesn't leverage. Five high-impact pages use stale, incomplete, or entirely absent data for decisions that would meaningfully improve with Databank integration.

One critical bug was discovered: the DCV engine's recent form adjustments have never worked due to a key name mismatch.

## Tier 1 Implementation (This Sprint)

### T1-1: Fix DCV recent_form key bug
- **File:** `src/optimizer/daily_optimizer.py` line 531
- **Change:** `"last_14"` → `"l14"`
- **Also:** Fix docstring at lines 343-346
- **Risk:** Zero — enables a feature that was always intended but silently broken

### T1-2: Add bats/throws to player pool SQL + enable platoon splits
- **Files:** `src/database.py` (both pool SQL queries), `src/optimizer/daily_optimizer.py`
- **Change:** Add `p.bats, p.throws` to SELECT; pass actual handedness to `compute_matchup_multiplier()`
- **Impact:** Platoon adjustment (~7-10% on contact stats) currently a complete no-op

### T1-3: Add ecr_rank_stddev to player pool SQL
- **File:** `src/database.py` (both pool SQL queries)
- **Change:** Add `ecr.rank_stddev AS ecr_rank_stddev` to SELECT
- **Impact:** Unlocks ECR confidence display on FA, Compare, Trade Finder, Optimizer

### T1-4: Surface Statcast columns already in pool
- **Files:** `pages/5_Free_Agents.py`, `pages/7_Player_Compare.py`, `pages/1_My_Team.py`
- **Change:** Add xwOBA, barrel%, stuff+ to table renders (columns already in DataFrame)

### T1-5: Fix rank_free_agents() to preserve enriched columns
- **File:** `src/in_season.py` `rank_free_agents()`
- **Change:** Merge regression flags, Statcast, health score from full_pool onto output
- **Impact:** BUY_LOW/SELL_HIGH flags, Statcast quality visible in FA rankings

## Tier 2 Implementation (Next Sprint)

T2-1 through T2-7 as documented in audit report. Key items:
- L7/L14 rolling stats in Player Compare and Free Agents
- Hot/cold batch query replacing per-player API calls
- DCV counting stats form adjustment
- Regression flag badges in FA table

## Tier 3 (Backlog)

T3-1 through T3-7 deferred per critique agent's analysis.

## Success Criteria

- DCV "Today" scope applies recent form adjustments (verified via score diff)
- Platoon splits produce non-1.0 matchup multipliers for LHB vs LHP matchups
- Free Agents table shows regression flags and Statcast columns
- Player Compare shows L7/L14 rolling stats
- All existing tests pass (3180+)

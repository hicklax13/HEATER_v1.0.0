# Lineup-Constrained Trade Evaluation

**Date:** 2026-03-16
**Status:** Approved
**Scope:** Replace raw stat summation in trade evaluator with LP-optimized lineup-constrained totals + roster cap enforcement (forced drop / FA pickup)

## Problem Statement

The trade evaluator (`src/engine/output/trade_evaluator.py`) computes roster totals by calling `_roster_category_totals()`, which sums **all** players' raw projected stats — starters and bench alike. This creates a systematic "phantom production" bias in uneven trades:

- In a 1-for-2 trade, both received players are counted at full season production even though only one can start (18 starting slots: C/1B/2B/3B/SS/3OF/2Util/2SP/2RP/4P).
- Two average hitters generate ~12 SGP in counting stats, overwhelming the ~6 SGP lost from an elite closer. The bench cost (~2.65 SGP) and replacement penalty (~0.3 SGP) are insufficient corrections.
- Result: a Devin Williams (elite closer) for Jacob Wilson + Dylan Crews (two average hitters) trade grades "A" when it should grade "D" or worse.

The root cause is that the system treats roster value as a simple sum of all player projections, ignoring positional constraints and lineup slot limits.

## Solution: LP-Optimized Lineup Totals + Roster Cap Enforcement

### Core Architecture Change

Replace the two `_roster_category_totals()` calls in `evaluate_trade()` (lines 244-245) with a new `_lineup_constrained_totals()` function that:

1. Filters `player_pool` to the given roster IDs
2. Renames `name` → `player_name` if needed (LP expects `player_name` column; `player_pool` from DB has `name`)
3. Runs the existing `LineupOptimizer` (PuLP LP solver) to assign the best 18 players to starting slots
4. Extracts assigned starters' player IDs from the LP `assignments` list: `starter_ids = {a["player_id"] for a in assignments}`
5. Derives bench IDs by set difference: `bench_ids = [id for id in roster_ids if id not in starter_ids]`
6. Calls `_roster_category_totals(list(starter_ids), player_pool)` on just those starters

The LP optimizer already handles multi-position eligibility, Util slot flexibility, and P flex slots. No new optimization infrastructure needed.

**Graceful fallback:** If PuLP is not installed (`PULP_AVAILABLE = False`) or the LP solver fails, fall back to `_roster_category_totals()` with a logged warning.

### Roster Cap Enforcement

The Yahoo league has a 23-man roster limit (18 starting + 5 bench). Uneven trades violate this limit, requiring roster management moves that the evaluator must model.

#### When roster grows (receiving more than giving, e.g., 1-for-2):

1. Run LP on the oversized roster (e.g., 24 players) to identify 18 starters and 6 bench players
2. Derive bench IDs: `bench_ids = [id for id in after_ids if id not in starter_ids]`
3. Rank bench players by `SGPCalculator.total_sgp()` — lowest total SGP = drop candidate
4. Remove the drop candidate(s) from the after-roster (1 drop per excess player)
5. **No re-run needed** — dropping a bench player cannot change the LP's optimal starter assignment (the dropped player was already excluded from the objective). The starters from step 1 are the final answer.
6. Call `_roster_category_totals()` on the starter IDs from step 1
7. **No bench_cost penalty** — roster returns to 23, no lost bench slot

#### When roster shrinks (giving more than receiving, e.g., 2-for-1):

1. Query `get_free_agents(player_pool)` for available free agents
2. For each open slot, find the best FA by `total_sgp` that matches the type lost (hitter if a hitter was lost, pitcher if a pitcher was lost)
3. Add the FA(s) to the after-roster to restore it to 23 players
4. Run LP on the restored roster to get final starters
5. **No bench_bonus** — the FA pickup's production IS the bonus (counted in lineup stats if they start, or on bench if they don't)
6. **Fallback when no roster data loaded:** `get_free_agents()` returns the full player pool. To prevent picking up elite "FAs" who are actually rostered, cap the FA pickup at the **median SGP of the relevant subset** (hitter-only median when `need_hitter=True`, pitcher-only median otherwise). This approximates a replacement-level pickup.
7. **FA pool represents pre-trade state** — `get_free_agents()` reads from the DB `league_rosters` table, which hasn't been updated with the trade yet. This is intentionally correct: the FA pool at the time of the trade decision is the pre-trade pool.

#### Equal trades (1-for-1, 2-for-2):

No drop or pickup needed. Run LP directly on the 23-player after-roster.

### What This Eliminates

The entire `bench_cost` / `bench_bonus` calculation (lines 297-335 of `trade_evaluator.py`) is replaced by the drop/pickup model. The `bench_option_value()` and `enhanced_bench_option_value()` functions remain available for other consumers but are no longer called by the trade evaluator.

The `bench_cost` key in the result dict is retained (set to 0.0) for backward compatibility with the UI.

## File Changes

### Modified Files

1. **`src/engine/output/trade_evaluator.py`** — Primary changes:
   - Add new `_lineup_constrained_totals()` function (~80 lines)
   - Add new `_find_drop_candidate()` helper (~20 lines)
   - Add new `_find_fa_pickup()` helper (~30 lines)
   - Replace `_roster_category_totals()` calls with `_lineup_constrained_totals()`
   - Remove bench_cost/bench_bonus calculation block (lines 297-335)
   - Add imports: `LineupOptimizer`, `ROSTER_SLOTS`, `PULP_AVAILABLE` from `src/lineup_optimizer`
   - Add import: `SGPCalculator` usage for drop candidate ranking
   - Update result dict: `bench_cost` always 0.0, add `drop_candidate` and `fa_pickup` keys

2. **`pages/3_Trade_Analyzer.py`** — UI updates:
   - Replace "Bench Cost" metric with "Roster Move" indicator (shows drop candidate name or FA pickup name)
   - Remove bench_cost from the 5-column metrics row or repurpose it

3. **`src/ui_shared.py`** — Add tooltip for new roster move metric

4. **`CLAUDE.md`** — Update documentation (API signatures, gotchas, architecture description)

### New Tests

5. **`tests/test_trade_engine.py`** — Add tests in `TestLineupConstrainedEval` class:
   - `test_lineup_constrained_totals_excludes_bench` — verify bench players don't inflate stats
   - `test_forced_drop_in_2for1_trade` — verify worst bench player is dropped, result has `drop_candidate`
   - `test_fa_pickup_in_1for2_trade` — verify FA is added when roster shrinks, result has `fa_pickup`
   - `test_fallback_when_pulp_unavailable` — mock `PULP_AVAILABLE=False`, verify graceful degradation to raw sums
   - `test_equal_trade_no_drop_no_pickup` — verify 1-for-1 trades skip roster moves, both keys are None
   - `test_drop_candidate_is_lowest_sgp` — verify `_find_drop_candidate` returns the player with lowest total SGP
   - `test_multi_player_drop_3for1` — 3-for-1 trade requires dropping 2 bench players
   - `test_lp_solver_failure_fallback` — mock LP returning non-Optimal status, verify fallback
   - `test_column_rename_name_to_player_name` — verify LP works when pool has `name` not `player_name`

6. **`tests/test_trade_engine_math.py`** — Add math verification in `TestLineupConstraintMath` class:
   - `test_lineup_constraint_prevents_phantom_production` — hand-calculated: 2 hitters received but only 1 can start (all hitter slots full), verify SGP delta reflects only the marginal starter improvement
   - `test_closer_trade_grades_correctly` — Williams-like closer traded for 2 hitters, verify grade reflects real lineup impact (not A/A+)
   - `test_fa_pickup_fills_roster_gap` — 2-for-1 trade adds FA, verify after-totals include FA production

## API Changes

### New Function: `_lineup_constrained_totals()`

```python
def _lineup_constrained_totals(
    roster_ids: list[int],
    player_pool: pd.DataFrame,
    config: LeagueConfig | None = None,
) -> tuple[dict[str, float], list[dict]]:
    """Compute category totals using only LP-optimized starters.

    Args:
        roster_ids: Player IDs on the roster.
        player_pool: Full player pool DataFrame.
        config: League configuration.

    Returns:
        Tuple of (totals_dict, assignments_list) where:
          - totals_dict: Same format as _roster_category_totals() output
          - assignments_list: LP assignment details [{slot, player_name, player_id}]

    Falls back to _roster_category_totals() when PuLP unavailable.
    """
```

### New Function: `_find_drop_candidate()`

```python
def _find_drop_candidate(
    bench_ids: list[int],
    player_pool: pd.DataFrame,
    sgp_calc: SGPCalculator,
) -> int | None:
    """Find the worst bench player to drop (lowest total SGP).

    Returns player_id of the drop candidate, or None if bench is empty.
    """
```

### New Function: `_find_fa_pickup()`

```python
def _find_fa_pickup(
    player_pool: pd.DataFrame,
    sgp_calc: SGPCalculator,
    need_hitter: bool = True,
    median_sgp_cap: float | None = None,
) -> int | None:
    """Find the best FA pickup to fill an open roster slot.

    Args:
        player_pool: Full player pool DataFrame.
        sgp_calc: SGP calculator for ranking.
        need_hitter: True if the open slot should be filled by a hitter.
        median_sgp_cap: When set, FAs above this SGP are excluded
            (prevents picking elite "FAs" when no roster data loaded).

    Returns player_id of the best FA, or None if no suitable FA found.
    """
```

### Modified Result Dict Keys

```python
# Existing key, changed behavior:
"bench_cost": 0.0,  # Always 0.0 (replaced by drop/pickup model)

# New keys:
"drop_candidate": str | None,    # Name of player auto-dropped (1-for-2 trades)
"fa_pickup": str | None,         # Name of FA auto-picked-up (2-for-1 trades)
"lineup_constrained": bool,      # True if LP was used, False if fallback
```

## Edge Cases

1. **PuLP not installed** — Fallback to `_roster_category_totals()` (raw sums). Log warning. `lineup_constrained` = False in result.
2. **LP solver fails** — Same fallback. Log warning with solver status.
3. **No league rosters loaded** — `get_free_agents()` returns full pool. FA pickup capped at median SGP to prevent unrealistic pickups.
4. **Roster already under 23** — Can happen with partial rosters during setup. Skip drop logic, just run LP.
5. **All bench players have equal SGP** — Drop the first one (arbitrary but consistent).
6. **Trade involves bench-only players** — LP correctly handles this: if you trade a bench player for a better one, the starters may reshuffle.
7. **3-for-1 or 1-for-3 trades** — Drop/pickup logic handles any imbalance via the `abs(net_roster_growth)` loop.

## Phase 2 MC Interaction (Intentionally Deferred)

Phase 2's `_run_mc_overlay()` calls `build_roster_stats()` which sums ALL player stats (not just starters). This means Phase 2 still has the phantom production problem. However, Phase 2 is gated behind `enable_mc=True` (off by default) and its grade **overwrites** the Phase 1 grade when active. Updating Phase 2's stat aggregation to use LP-constrained totals is a follow-up task — it requires changes to `build_roster_stats()` in `src/engine/monte_carlo/trade_simulator.py` and the copula sampling pipeline, which is a separate scope.

For now, the Phase 1 grade (which most users see) is corrected. Phase 2 MC, when enabled, will be conservative (slightly optimistic for 2-for-1 trades) until the follow-up update.

## Performance

- LP solver runs in <100ms for a 23-player roster (well within interactive budget)
- Single LP run per roster (no re-run needed after drop — bench player removal doesn't affect starter assignment)
- Phase 5 sensitivity analysis calls `evaluate_fn` N times with `enable_mc=False, enable_context=False`. With ~10 iterations and 2 LP runs each (before + after), that's ~20 LP runs at ~100ms = ~2 seconds. Within interactive budget.

## Backward Compatibility

- `bench_cost` key retained at 0.0 in result dict
- `bench_option_detail` key retained as None
- UI code that reads `bench_cost` continues to work (shows 0.0)
- Legacy `analyze_trade()` in `src/in_season.py` unchanged (still uses raw sums)
- `bench_option_value()` and `enhanced_bench_option_value()` functions unchanged (other consumers)

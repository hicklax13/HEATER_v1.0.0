# Lineup Optimizer V2 — Implementation Plan

## Context

The existing 8-stage lineup optimizer works well for weekly optimization but lacks daily matchup intelligence. Today's manual lineup analysis revealed critical gaps: it used 5-day YTD stats at face value instead of Bayesian-blended projections, didn't account for matchup-specific factors (opposing pitcher, park, weather, batting order), and had no concept of category urgency (which categories to attack vs protect based on the current H2H matchup score).

The V2 research framework (52 variables, 6 formulas, 100+ academic/industry sources) identified the optimal approach: a Daily Category Value (DCV) system that blends multi-year projections with matchup factors and H2H urgency, feeding into the existing LP solver for optimal slot assignment.

**Key design principle:** Build ON TOP of the existing optimizer (11 modules, ~5,500 lines, production-grade). Add 3 new pipeline stages + 2 new modules. Don't replace anything.

## Architecture

```
Existing Pipeline (stages 1-9, untouched):
  projections → matchup → weights → risk → LP → H2H → streaming → scenarios → maximin

V2 Additions (stages 10-12, new):
  Stage 10: Category Urgency (sigmoid from live matchup gap)
  Stage 11: Daily Category Value (DCV per player per category)
  Stage 12: Daily Lineup Assignment (LP with urgency-weighted DCV)
```

New `"daily"` mode preset activates stages 10-12. Existing modes unchanged.

---

## Data Sources (What Comes From Where)

| Data | Source | Already Available? |
|---|---|---|
| Rosters + IL/DTD/NA status | Yahoo API via `yds.get_rosters()` | YES |
| Current matchup category scores | Yahoo API via `yds.get_matchup()` | YES |
| Free agents with ownership % | Yahoo API via `yds.get_free_agents()` | YES |
| Transactions (for budget tracking) | Yahoo API via `yds.get_transactions()` | YES |
| Schedule + opponent | Yahoo API via `yds.get_schedule()` | YES |
| ROS projections (Bayesian) | `ros_projections` table | YES |
| 2024/2025 season stats | `season_stats` table | YES |
| 2026 YTD stats | `season_stats` table (4h refresh) | YES |
| Probable pitchers | MLB Stats API via `fetch_probable_pitchers()` | YES |
| Park factors (stat-specific) | `PARK_FACTORS` from bootstrap | YES |
| Platoon splits | `matchup_adjustments.platoon_adjustment()` | YES |
| H2H category weights | `h2h_engine.compute_h2h_category_weights()` | YES |
| SGP denominators | `LeagueConfig.sgp_denominators` | YES |
| Injury health scores | `injury_model.compute_health_score()` | YES |
| IP tracking | `ip_tracker.compute_weekly_ip_projection()` | YES |
| Statcast (xwOBA, barrel rate) | `statcast_archive` table via pybaseball | YES |
| Weather (temp/wind) | `matchup_adjustments.weather_hr_adjustment()` | YES (basic) |
| Lineup confirmations | MLB Stats API `schedule(hydrate=lineups)` | NEW (1 API call) |
| Doubleheader detection | MLB Stats API `schedule()` | NEW (parse from existing) |
| Batting order position | MLB Stats API `schedule(hydrate=lineups)` | NEW (same call as above) |

**Gaps filled:** Only 2 new MLB Stats API calls needed (lineup confirmation + doubleheader detection) — both from the same `statsapi.schedule()` endpoint already used in `matchup_adjustments.py`.

---

## Implementation Tasks (6 parallel-ready subagent tasks)

### Task 1: Category Urgency Module [NEW FILE]
**File:** `src/optimizer/category_urgency.py` (~150 lines)
**Tests:** `tests/test_category_urgency.py` (~100 lines)

Functions:
- `compute_category_urgency(my_totals, opp_totals, config)` — sigmoid urgency per category
- `classify_rate_stat_mode(my_rate, opp_rate, gap_threshold)` — protect/compete/abandon
- `compute_urgency_weights(matchup_dict, config)` — converts Yahoo matchup to urgency dict

Formulas:
```python
urgency = 1 / (1 + exp(-k * (opp_total - my_total) / sgp_denom))
# k=2.0 for counting stats, k=3.0 for rate stats
# L uses inverse: (my_total - opp_total)
```

Reuses: `LeagueConfig` from `src/valuation.py`, matchup data from `yds.get_matchup()`

### Task 2: Daily Optimizer Module [NEW FILE]
**File:** `src/optimizer/daily_optimizer.py` (~400 lines)
**Tests:** `tests/test_daily_optimizer.py` (~200 lines)

Functions:
- `compute_blended_projection(player, stab_points)` — Bayesian blend formula
- `compute_matchup_multiplier(player, game_info, park_factors)` — stat-specific multipliers applied to COMPONENTS
- `compute_health_factor(player_status)` — 0.00 for IL/DTD/NA, 1.00 for active
- `compute_volume_factor(player, schedule, lineups_posted)` — 0.0/0.3/0.9/1.0/2.0
- `build_daily_dcv_table(roster, matchup, schedule, config)` — master DCV per player per category
- `apply_stud_floor(dcv_table, top_n=30)` — prevent benching elite players
- `check_ip_override(dcv_table, weekly_ip, ip_minimum=20)` — force-start pitcher if needed
- `detect_doubleheaders(schedule_date)` — 2.0x volume for confirmed DH starters
- `fetch_confirmed_lineups(date)` — MLB Stats API for lineup cards
- `compute_pitcher_blowup_risk(pitcher, game_log_variance)` — variance penalty
- `check_transaction_budget(transactions_this_week, weekly_limit=7)` — discount streaming

Reuses:
- `matchup_adjustments.platoon_adjustment()`, `park_factor_adjustment()`, `weather_hr_adjustment()`
- `injury_model.compute_health_score()` for 3-year health history
- `ip_tracker.compute_weekly_ip_projection()` for IP minimum
- `trade_intelligence._load_roster_statuses()` for IL/DTD/NA
- `bayesian.STABILIZATION_POINTS` for blend weights

### Task 3: Pipeline Extension [MODIFY]
**File:** `src/optimizer/pipeline.py` (~80 lines added)

Changes:
- Add `"daily"` to `MODE_PRESETS` with `enable_daily_dcv=True`
- Add Stage 10: Category Urgency (calls `compute_urgency_weights()`)
- Add Stage 11: Daily DCV Table (calls `build_daily_dcv_table()`)
- Add Stage 12: Daily Lineup (feeds DCV into existing LP solver)
- Add 4 new keys to result dict: `daily_dcv`, `urgency_weights`, `rate_stat_modes`, `daily_lineup`

Reuses: All existing stages 1-9 untouched. Same `optimize()` method signature.

### Task 4: Lineup Page Tab [MODIFY]
**File:** `pages/5_Lineup.py` (~200 lines added)

Changes:
- Add 7th tab "Daily Optimize" to existing `st.tabs()` call
- "Daily Optimize" tab shows: date picker, DCV table (sortable), urgency indicators, recommended lineup with slot assignments, FA comparison column
- Uses existing UI patterns: `render_sortable_table()`, `render_context_card()`
- Data: `yds.get_matchup()` for category scores, `yds.get_rosters()` for roster

### Task 5: Streaming + Projections Enhancement [MODIFY]
**File:** `src/optimizer/streaming.py` (~15 lines added)
**File:** `src/optimizer/projections.py` (~25 lines added)

Changes to streaming:
- Add `adds_used_this_week` parameter to `rank_streaming_candidates()`
- Discount streaming value when transaction budget nearly exhausted

Changes to projections:
- Add `v2_bayesian_blend()` helper (standalone, no side effects)

### Task 6: Tests + Integration [NEW FILES]
**Files:** `tests/test_category_urgency.py`, `tests/test_daily_optimizer.py`
**Modify:** `tests/test_optimizer_pipeline.py` (add daily mode regression test)

Test cases:
- Urgency at gap=0 → 0.5 (tied category)
- Urgency when losing big → approaches 1.0
- Rate stat protect mode at 0.30+ ERA gap
- DCV math: Bayesian blend at PA=0 → 100% preseason
- Health factor: IL status → 0.00 excluded
- Volume factor: off-day → 0.00, confirmed lineup → 1.00
- Stud floor: top-30 player not benched despite bad matchup
- IP override: force-start when below 20 IP
- Doubleheader: volume = 2.0
- Existing modes produce identical output (regression guard)

---

## Execution Plan

```
Parallel Group 1 (no dependencies):
  Task 1: Category Urgency Module (pure math, standalone)
  Task 5: Streaming + Projections Enhancement (small, independent)

Sequential (dependency chain):
  Task 2: Daily Optimizer Module (depends on Task 1)
  Task 3: Pipeline Extension (depends on Tasks 1 + 2)
  Task 4: Lineup Page Tab (depends on Task 3)
  Task 6: Tests (runs after Tasks 1-5)
```

**Estimated total:** ~1,000 new lines across 2 new files + 4 modified files + 3 test files.

---

## Files Summary

| File | Action | Lines |
|---|---|---|
| `src/optimizer/category_urgency.py` | NEW | ~150 |
| `src/optimizer/daily_optimizer.py` | NEW | ~400 |
| `src/optimizer/pipeline.py` | MODIFY | +80 |
| `src/optimizer/streaming.py` | MODIFY | +15 |
| `src/optimizer/projections.py` | MODIFY | +25 |
| `pages/5_Lineup.py` | MODIFY | +200 |
| `tests/test_category_urgency.py` | NEW | ~100 |
| `tests/test_daily_optimizer.py` | NEW | ~200 |
| `tests/test_optimizer_pipeline.py` | MODIFY | +30 |

---

## Verification Plan

1. `python -m pytest tests/test_category_urgency.py tests/test_daily_optimizer.py -v` — all new tests green
2. `python -m pytest tests/test_optimizer_pipeline.py -v` — existing + new tests pass (regression guard)
3. `python -m pytest -x -q` — full suite passes (2151+ tests, 0 failures)
4. `python -m ruff check . && python -m ruff format --check .` — lint clean
5. Start app → Lineup page → "Daily Optimize" tab → verify DCV table renders with live data
6. Verify urgency indicators change based on current Yahoo matchup scores
7. Verify IL/DTD players excluded, stud floor active, doubleheaders detected
8. `git push origin master` — all 5 CI jobs pass

# HEATER Plan 1 — Post-Draft In-Season Polish

**Created:** 2026-03-25
**Context:** MLB Opening Day. Fantasy draft completed. App is v1.0.0 with 1991 passing tests.
**Goal:** Transform from draft tool to polished in-season manager.

---

## Phase A: Bug Fixes (from manual UI testing)

### A1. ECR Consensus Parsing Error (FIXED in working tree)
- **Bug:** `invalid literal for int() with base 10: '469.p.9877'` — Yahoo player keys (`469.p.XXXX`) were passed to `int()` without guard.
- **Fix:** `src/ecr.py` lines 836-847 now catch `ValueError` and fall back to name-based resolution.
- **Status:** Fixed, needs commit.

### A2. Verify All Players Loaded & Selectable Across Every Feature
Pages that surface player selection (dropdowns, search, multiselect):
- [ ] **Player Compare** (`pages/4_Player_Compare.py`) — two `st.selectbox` dropdowns. Confirm full 9,226-player pool available, not truncated.
- [ ] **Trade Analyzer** (`pages/3_Trade_Analyzer.py`) — giving/receiving multiselects. Must show all rostered + all pool players respectively.
- [ ] **Start/Sit Advisor** (`pages/11_Start_Sit.py`) — player multiselect. Must show full roster (or full pool if no roster loaded).
- [ ] **Free Agents** (`pages/5_Free_Agents.py`) — position filter + table. Confirm all non-rostered players visible.
- [ ] **Waiver Wire** (`pages/10_Waiver_Wire.py`) — add/drop candidates. Confirm all FA candidates surface.
- [ ] **Draft Simulator** (`pages/2_Draft_Simulator.py`) — recommendation cards. Confirm all undrafted players eligible.
- [ ] **Matchup Planner** (`pages/12_Matchup_Planner.py`) — roster-based. Confirm all roster players rated.
- [ ] **Lineup Optimizer** (`pages/6_Lineup_Optimizer.py`) — roster-based. Confirm full roster feeds optimizer.
- [ ] **Leaders** (`pages/9_Leaders.py`) — category leaders from season_stats. Confirm no artificial truncation.

**Audit method:** For each page, trace the data flow from `load_player_pool()` or `league_rosters` through to the UI widget. Confirm no `.head(N)` truncation, no silent filter that drops players, and that the widget's `options` parameter receives the full eligible set.

---

## Phase B: Player Headshot Photos

### B1. Data Source — MLB Stats API Headshot URL
MLB provides official headshots at:
```
https://img.mlb.com/mlb-photos/image/upload/d_people:generic:headshot:67:current.png/w_213,q_auto:best/v1/people/{mlb_id}/headshot/67/current
```
- Requires `mlb_id` (MLB Stats API person ID), already stored in `player_id_map` table.
- Fallback: monogram initials (already implemented in player card).

### B2. Implementation Plan
1. **Add `get_headshot_url(player_id)` utility** to `src/player_card.py`
   - Look up `mlb_id` from `player_id_map` table
   - Return MLB headshot URL or `None` if no mapping
   - Handle `float('nan')` from SQLite NULL (known gotcha)

2. **Add headshot column to `build_compact_table_html()`** in `src/ui_shared.py`
   - New optional parameter: `show_headshots=False`
   - When enabled, prepend a 24x24px `<img>` tag before the player name in the `.col-name` cell
   - Use `onerror="this.style.display='none'"` for graceful fallback
   - CSS: `.player-headshot { width:24px; height:24px; border-radius:50%; vertical-align:middle; margin-right:4px; }`

3. **Enable headshots on all pages that show player tables:**
   - My Team roster table
   - Draft Simulator recommendation cards
   - Trade Analyzer giving/receiving panels
   - Player Compare header area
   - Free Agents ranking table
   - Lineup Optimizer roster/lineup tables
   - Leaders category leader tables
   - Waiver Wire add/drop tables
   - Start/Sit recommendation results
   - Matchup Planner ratings table
   - Standings team rosters (if expanded)

4. **Player card dialog** — already has monogram; replace with headshot `<img>` when `mlb_id` available.

5. **Caching:** Browser caches MLB CDN images. No server-side caching needed. Add `loading="lazy"` to `<img>` tags.

### B3. Gotchas
- `mlb_id` can be `float('nan')` from SQLite NULL — use `pd.notna()` check
- Not all players have MLB IDs (minor leaguers, international) — graceful fallback to no image
- MLB CDN rate limits: lazy loading + browser cache should suffice for normal use
- Streamlit `st.dataframe()` cannot render HTML — headshots only work in `st.markdown()` tables (our compact tables already use this)

---

## Phase C: Load Time Recording & Benchmarking

### C1. Benchmark Results (March 25, 2026 — 3 runs averaged)

| Function | Avg Time | Classification |
|---|---|---|
| DraftEngine.recommend() [standard, 100 sims] | 213.3s | Very Slow |
| DraftEngine.recommend() [quick, 50 sims] | 107.5s | Very Slow |
| bootstrap_all_data() [no force] | 14.3s | Very Slow |
| load_player_pool() | 6.5s | Very Slow |
| compute_trade_values() | 2.9s | Very Slow |
| value_all_players() | 1.7s | Slow |
| compute_add_drop_recommendations() | 1.2s | Slow |
| DraftEngine.enhance_player_pool() | 1.0s | Slow |
| compute_replacement_levels() | 0.6s | Slow |
| compute_weekly_matchup_ratings() | 0.5s | Slow |
| simulate_season() [100 sims] | 0.4s | OK |
| LineupOptimizerPipeline.optimize() [quick] | 0.4s | OK |
| evaluate_trade() [Phase 1] | 0.2s | OK |
| evaluate_trade() [Phase 1+context] | 0.2s | OK |
| DB: load season_stats | 0.04s | Fast |
| compare_players() | 0.03s | Fast |
| compute_sgp_denominators() | 0.03s | Fast |
| rank_free_agents() | 0.02s | Fast |
| get_prospect_rankings() | 0.02s | Fast |
| compute_category_leaders() | 0.02s | Fast |
| start_sit_recommendation() | 0.003s | Fast |
| load_ecr_consensus() | 0.003s | Fast |
| DB: load league_standings | 0.003s | Fast |
| compute_power_rankings() | 0.001s | Fast |

**Pool size:** 9,226 players | **Total benchmark time:** ~360s

### C2. In-App Timing (to be added)
- Add `st.session_state["_page_load_times"]` dict
- Each page records `time.perf_counter()` at start/end
- Display in footer or dev panel: "Page loaded in X.Xs"

---

## Phase D: Performance Optimization Plan

### D1. Critical Path: `load_player_pool()` — 6.5s -> target <1s
**Root cause:** Loads 9,226 rows with all columns, joins projections, runs `pd.to_numeric()` coercion on every column.
**Fixes:**
1. **Add SQLite indexes** on `player_id`, `is_hitter`, `positions` columns
2. **Lazy column loading** — only load columns needed by the calling page
3. **Cache with `@st.cache_data(ttl=300)`** — pool changes rarely within 5 minutes
4. **Pre-compute blended projections** at bootstrap time, store as materialized view

### D2. Critical Path: Draft Engine — 107-213s -> target <15s
**Root cause:** MC simulation iterates 50-100 sims x 6 rounds x 12 teams with full pool scoring each iteration.
**Fixes:**
1. **Pre-filter pool** — only simulate top-300 ADP players (covers 6 rounds x 12 teams = 72 picks with 4x buffer)
2. **Vectorize scoring** — replace per-player Python loops with NumPy array operations
3. **Cache SGP denominators** between simulations (they don't change)
4. **Reduce simulation horizon** — 4 rounds instead of 6 (diminishing info returns)
5. **Parallel simulations** — `concurrent.futures.ProcessPoolExecutor` for MC sims
6. **Pre-compute survival probabilities** — batch CDF calculations instead of per-player

### D3. Medium Priority: `compute_trade_values()` — 2.9s -> target <0.5s
**Root cause:** Calls `value_all_players()` internally for all 9,226 players.
**Fixes:**
1. **Cache `value_all_players()` result** with `@st.cache_data`
2. **Only recompute on roster change**, not every page load

### D4. Medium Priority: `value_all_players()` — 1.7s -> target <0.3s
**Root cause:** Iterates all 9,226 players computing SGP per-player.
**Fixes:**
1. **Vectorize SGP computation** — replace row-by-row with DataFrame operations
2. **Pre-compute denominators once** at session start

### D5. Low Priority: Bootstrap — 14s (acceptable for startup)
**Current behavior:** Staleness-checked, only refreshes stale data.
**Optional improvements:**
1. **Parallel phase execution** — players + park_factors can run concurrently
2. **Background refresh** — don't block UI while refreshing non-critical data
3. **Progressive rendering** — show cached data immediately, refresh in background

### D6. Low Priority: Waiver Wire / Matchup Ratings — 0.5-1.2s
**Fixes:**
1. **Limit FA candidates** in waiver wire to top-200 by ADP
2. **Cache matchup schedule** per week (doesn't change within a day)

### D7. Architecture-Level Optimization
1. **Session-level caches** — compute `player_pool`, `valued_pool`, `sgp_denominators`, `replacement_levels` ONCE at session start, store in `st.session_state`
2. **Page-specific pool slices** — each page only loads the columns + rows it needs
3. **Background data refresh** — use `st.spinner` with async refresh, show stale data immediately
4. **Database query optimization** — add composite indexes for common join patterns

---

## Execution Order

1. **Phase A** (bug fixes) — commit existing fixes, audit player availability
2. **Phase B** (headshots) — implement `get_headshot_url()`, integrate into compact tables
3. **Phase C** (timing) — add in-app page load timing
4. **Phase D** (performance) — D1 first (biggest user impact), then D2, D3, D4

---

## Success Criteria

- [ ] All 13 pages load without errors
- [ ] All players selectable in every dropdown/multiselect
- [ ] Player headshots visible next to names in all tables
- [ ] Page load times displayed in UI
- [ ] `load_player_pool()` under 1 second
- [ ] Draft engine under 15 seconds (quick mode)
- [ ] All 1991+ tests passing, CI green

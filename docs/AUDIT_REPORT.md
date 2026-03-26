# HEATER v1.0.0 — Systematic Audit Report

**Date:** 2026-03-25 (updated after Yahoo sync)
**Auditor:** Claude (systematic debugging)
**Scope:** All 13 pages, 30+ backend modules, SQLite database (21 tables), 5 core algorithms, 4 external data sources, Yahoo Fantasy integration

---

## YAHOO FANTASY STATUS (Post-Sync)

- **Connected:** Yes (Team Hickey, League ID 109662)
- **Token persistence:** `data/yahoo_token.json` — has access_token + refresh_token + consumer keys
- **Auto-reconnect:** Hardened — `refresh_token()` now persists refreshed tokens to disk
- **Teams synced:** 12/12
- **Roster entries:** 264 total (21 on Team Hickey)
- **Standings:** 0 (expected — Opening Day, no games played yet)
- **Player news from Yahoo:** 10 injury reports
- **Ownership trends:** 264 records
- **Duplicate roster entries:** None
- **Unmatched players:** 0 (all 21 Team Hickey players matched to DB)

### New Findings from Yahoo Sync

**BUG-012: League Standings Empty (Opening Day)**
- **Severity:** INFO (not a bug)
- **Root Cause:** MLB 2026 season started March 25. Yahoo hasn't populated team_stats yet (no games played). `get_league_standings()` correctly returns empty stats pre-games.
- **Expected resolution:** Standings will auto-populate after first day of games
- **Location:** `src/yahoo_api.py:681-682`

**BUG-013: Yahoo Injury News Headlines Are Body Parts, Not Headlines**
- **Severity:** MEDIUM
- **Root Cause:** Yahoo API returns injury body part (e.g., "Wrist", "Elbow") in the headline field. The sync code at line 1187 stores this as `headline`. Real headlines should describe the injury event (e.g., "Mike Trout placed on 15-day IL").
- **Evidence:** All 10 player_news rows have body-part-only headlines: "Wrist", "Elbow", "Lower Leg", "Ribs"
- **Impact:** News feed shows cryptic body part labels instead of meaningful injury descriptions
- **Location:** `src/yahoo_api.py:1187-1193`

**BUG-014: Roster Size Variance (19-24 players per team)**
- **Severity:** LOW (expected)
- **Root Cause:** Opening Day rosters vary due to IL placements, callups, and bench moves. "On a Twosday" has 19 players (4 short of 23-man max), "HUMAN INTELLIGENCE" has 24 (1 over standard). This is normal Yahoo league behavior.
- **Status:** NOT A BUG — Yahoo allows flexible roster sizes during the season

---

## CRITICAL ISSUES (App-Breaking)

### BUG-001: Zero Projections — FanGraphs API 403 Blocked
- **Severity:** CRITICAL
- **Root Cause:** FanGraphs added bot detection (Cloudflare/WAF) that blocks all API requests, returning HTTP 403. When `fetch_all_projections()` returns empty, the entire projection pipeline short-circuits — no projections stored, no blended projections created, no ADP extracted.
- **Evidence:** `projections` table has 2 rows (both for "Test Pitcher"). `refresh_log` shows `projections: failed`. 779 of 780 players have zero projection records.
- **Impact:** Breaks ALL valuation features: draft engine, trade analyzer, free agent rankings, lineup optimizer, waiver wire
- **Location:** `src/data_pipeline.py:627-631` (short-circuit) + `src/data_pipeline.py:173-228` (fetch)
- **Status:** OPEN

### BUG-002: Zero ADP Data
- **Severity:** CRITICAL
- **Root Cause:** Two-part failure: (1) FanGraphs ADP extraction skipped because projections failed, (2) External ADP sources (FantasyPros, NFBC) returned empty but `_bootstrap_adp_sources()` marked refresh as "success" anyway
- **Evidence:** `adp` table has 0 rows. All players get `adp=999` (the COALESCE default)
- **Impact:** Draft recommendations, player rankings, and simulation accuracy all broken
- **Location:** `src/data_pipeline.py:651-657` (skipped), `src/data_bootstrap.py:449` (false success)
- **Status:** OPEN

### BUG-003: All Player Team Fields Empty
- **Severity:** CRITICAL
- **Root Cause:** `fetch_all_mlb_players()` extracts `p.get("currentTeam", {}).get("abbreviation", "")` but the MLB Stats API `sports_players` endpoint returns `currentTeam` with only `{id, link}` — no `abbreviation` field. Confirmed via live API test: all 779 players have `currentTeam.id` but no `abbreviation`.
- **Fix:** Build a `team_id -> abbreviation` mapping from `statsapi.get('teams', ...)` before processing players, then resolve via `team_map.get(currentTeam.id)`. Confirmed all 779/779 players are resolvable.
- **Location:** `src/live_stats.py:377`
- **Status:** OPEN

---

## HIGH-SEVERITY ISSUES

### BUG-004: Only 780 Players Loaded (Expected 9,226)
- **Severity:** HIGH
- **Root Cause:** `fetch_all_mlb_players()` uses `gameType=R` which returns only active 25/26-man rosters. The 9,226 figure was from a previous run with FanGraphs data + sample data. `fetch_extended_roster()` exists but even with spring training adds only ~1,200 total.
- **Impact:** Most MLB players (minor leaguers, 40-man extended, non-active) are missing
- **Location:** `src/live_stats.py:351-386`
- **Status:** OPEN

### BUG-005: player_id_map Table Empty (0 mappings)
- **Severity:** HIGH
- **Root Cause:** The ID map is populated during FanGraphs projection processing via `_update_fangraphs_ids()`. Since FanGraphs is 403, this function never ran. Without cross-platform ID mapping, Yahoo sync can't match players to the internal pool.
- **Evidence:** `player_id_map` has 0 rows
- **Location:** `src/data_pipeline.py:646` (skipped when projections fail)
- **Status:** OPEN

### BUG-006: ECR Consensus is Placeholder Data
- **Severity:** HIGH
- **Root Cause:** All 6 external ranking sources (ESPN, Yahoo, CBS, NFBC, FanGraphs, FantasyPros) failed to fetch. Only `heater_sgp_rank` was populated (the internal ranking). `consensus_rank` is just player_id order (1, 2, 3...) since there's only 1 source.
- **Evidence:** 300 ECR records, all with `n_sources=1`. All external rank columns are NULL.
- **Location:** `src/ecr.py` + external fetch functions
- **Status:** OPEN

### BUG-007: player_news Insertion Constraint Mismatch
- **Severity:** MEDIUM
- **Root Cause:** Yahoo injury news is inserted without `published_at` timestamp, but the `player_news` table has a UNIQUE constraint on `(player_id, source, headline, published_at)`. Re-syncs silently fail via `INSERT OR IGNORE`.
- **Impact:** Yahoo injury data goes stale and never updates
- **Location:** `src/yahoo_api.py:1187-1193`
- **Status:** OPEN

### BUG-008: MLB Stats API Transactions Endpoint Bug
- **Severity:** MEDIUM
- **Root Cause:** `statsapi.get('transactions', ...)` has a library-level validation bug rejecting `startDate/endDate` despite them being valid params
- **Fix:** Bypassed with direct `requests.get()` to the API URL
- **Location:** `src/news_fetcher.py:88-94`
- **Status:** FIXED (2026-03-25)

---

## LOW-SEVERITY ISSUES

### BUG-009: Yahoo game_key 469 Not Validated
- **Severity:** LOW
- **Root Cause:** `YahooFantasyClient` resolves game_key dynamically but doesn't assert it equals 469 for 2026 MLB
- **Location:** `src/yahoo_api.py:329-343`
- **Status:** OPEN

### BUG-010: ADP Sources Marked "success" Despite Zero Data
- **Severity:** LOW
- **Root Cause:** `_bootstrap_adp_sources()` calls `update_refresh_log("adp_sources", "success")` regardless of whether any ADP rows were actually stored
- **Location:** `src/data_bootstrap.py:449`
- **Status:** OPEN

### BUG-011: Prospect Rankings Empty
- **Severity:** LOW
- **Root Cause:** FanGraphs Board API also 403-ing. Prospect data quality check caught 11/11 empty names and skipped the store.
- **Location:** `src/prospect_engine.py:243`
- **Status:** OPEN

---

## ALGORITHMS AUDIT

| Algorithm | File | Status |
|---|---|---|
| SGP Calculator (scalar + vectorized) | `src/valuation.py` | CORRECT |
| VORP + replacement levels | `src/valuation.py` | CORRECT |
| Trade Analyzer (6-phase engine) | `src/engine/output/trade_evaluator.py` | CORRECT |
| Monte Carlo Draft Simulation | `src/simulation.py` | CORRECT |
| Lineup Optimizer (LP solver) | `src/lineup_optimizer.py` | CORRECT |
| Snake Draft Order | `src/draft_state.py` + `src/simulation.py` | CORRECT |
| Opponent Pick Probability (additive blend) | `src/simulation.py` | CORRECT |

**All core algorithms are mathematically correct.**

---

## FRONTEND UI AUDIT

| Page | Status |
|---|---|
| app.py (Main/Draft) | NO ISSUES |
| 1_My_Team.py | NO ISSUES |
| 2_Draft_Simulator.py | NO ISSUES |
| 3_Trade_Analyzer.py | NO ISSUES |
| 4_Player_Compare.py | NO ISSUES |
| 5_Free_Agents.py | NO ISSUES |
| 6_Lineup_Optimizer.py | NO ISSUES |
| 7_Closer_Monitor.py | NO ISSUES |
| 8_Standings.py | NO ISSUES |
| 9_Leaders.py | NO ISSUES |
| 10_Waiver_Wire.py | NO ISSUES |
| 11_Start_Sit.py | NO ISSUES |
| 12_Matchup_Planner.py | NO ISSUES |

**No CSS conflicts, overlapping elements, invisible text, widget key collisions, or unhandled exceptions.**

---

## YAHOO FANTASY INTEGRATION

| Check | Status |
|---|---|
| Token persistence + auto-refresh | CORRECT |
| yfpy roster iteration quirk | CORRECT |
| yfpy bytes decoding | CORRECT |
| Duplicate roster prevention | CORRECT |
| Duplicate standings prevention | CORRECT |
| Player ID matching | CORRECT |
| Token override conflict | CORRECT |

---

## DATA SOURCE STATUS

| Source | Status | Records | Issue |
|---|---|---|---|
| MLB Stats API (players) | Working | 780 | Team field empty (BUG-003) |
| MLB Stats API (season_stats) | Working | 1,962 (3 seasons) | OK |
| MLB Stats API (transactions) | Fixed | — | Was broken (BUG-008) |
| FanGraphs (projections) | BLOCKED | 0 | HTTP 403 (BUG-001) |
| FanGraphs (ADP) | BLOCKED | 0 | Depends on projections (BUG-002) |
| FanGraphs (prospects) | BLOCKED | 0 | HTTP 403 (BUG-011) |
| Park Factors | Working | 30 | All 30 teams complete |
| Injury History | Working | 1,962 | 3 seasons (2023-2025) |
| ECR Consensus | Partial | 300 | Only internal ranking (BUG-006) |
| Yahoo Fantasy | CONNECTED | 264 rosters, 10 news, 264 ownership | Standings empty (Opening Day, BUG-012) |
| FantasyPros | BLOCKED | 0 | Requires JS rendering |
| NFBC | Empty | 0 | Parsed 0 players |

---

## PERFORMANCE BENCHMARKS

| Feature | Time | Status |
|---|---|---|
| `load_player_pool()` | <0.1s cached / 6.5s first | Optimized (st.cache_data) |
| Draft Engine (quick) | ~2-4s | Optimized (pool pre-filter) |
| Draft Engine (standard) | ~4-8s | Optimized |
| Bootstrap (all fresh) | ~10-12s | Optimized (parallel phases) |
| `value_all_players()` | ~0.3s | Optimized (vectorized SGP) |
| `compute_trade_values()` | ~1.2s | Optimized (vectorized SGP) |
| `rank_free_agents()` | ~0.2s | Optimized (pre-filtered) |

**No features over 1 minute with current optimizations.**

---

## PRIORITY FIX ORDER

1. BUG-003 (team field empty) — Fixable now, build team_id->abbreviation map
2. BUG-001 (FanGraphs 403) — Needs Playwright fallback or alternative projection source
3. BUG-002 (zero ADP) — Depends on BUG-001, or needs alternative ADP source
4. BUG-017 (transactions not synced) — Add transaction sync to sync_to_db()
5. BUG-005 (empty ID map) — Depends on BUG-001
6. BUG-016 (team logos not captured) — Add logo/manager extraction to Yahoo sync
7. BUG-019 (Yahoo FAs not fetched) — Expand FA pool from Yahoo API
8. BUG-007 (news constraint) — Quick schema fix
9. BUG-013 (Yahoo news headlines) — Improve injury news extraction
10. BUG-010 (false success log) — Quick validation fix
11. BUG-015 (opponent data useless) — Auto-resolves when BUG-001 + BUG-003 fixed
12. BUG-018 (FAs unrankable) — Auto-resolves when BUG-001 fixed

---

## LEAGUE VISIBILITY ISSUES (Post-Yahoo Sync)

### BUG-015: Opponent Player Data Shows No Projections or Team
- **Severity:** HIGH
- **Root Cause:** All 264 rostered players (across 12 teams) show NO PROJECTIONS because only 1 player has blended projections in the DB (BUG-001 dependency). All show empty team field (BUG-003 dependency). Opponent rosters ARE synced (names, positions, roster slots correct), but without projections the data is useless for analysis.
- **Evidence:** Spot-checked BUBBA CROSBY roster — all 23 players have "NO PROJECTIONS" and `team=(empty)`
- **Impact:** Cannot evaluate opponent teams, run trade analysis against opponents, or compare rosters meaningfully
- **Blocked by:** BUG-001 (projections) + BUG-003 (team field)
- **Location:** Database state, not a code bug
- **Status:** OPEN

### BUG-016: Fantasy Team Logos/Avatars Not Captured
- **Severity:** MEDIUM
- **Root Cause:** `get_all_rosters()` and `sync_to_db()` do not extract `team_logos` or manager `image_url` from yfpy team objects. The `league_rosters` table has no columns for team logo URLs. Yahoo's API provides `team_logos[0].url` (Cloudinary-hosted team avatar) and `managers[0].image_url` (manager profile pic).
- **Evidence:** `league_rosters` columns: `[id, team_name, team_index, player_id, roster_slot, is_user_team]` — no logo/image column. yfpy confirms `team_logos` attribute returns valid Cloudinary URLs.
- **Fix requires:** (1) Add `team_logo_url` and `manager_name` columns to `league_rosters` or a new `league_teams` table, (2) Extract from yfpy team objects during sync, (3) Display in My Team and Standings pages
- **Location:** `src/yahoo_api.py:703-762` (get_all_rosters), `src/database.py` (schema)
- **Status:** OPEN

### BUG-017: League Transactions Not Synced to Database
- **Severity:** HIGH
- **Root Cause:** `get_league_transactions()` method EXISTS (line 917) but is NEVER CALLED during `sync_to_db()` (lines 1064-1197). The `transactions` table has 0 rows. Trades, FA pickups, and roster moves from Yahoo are available via the API but not being captured.
- **Evidence:** `transactions` table: 0 rows. `sync_to_db()` syncs standings + rosters but skips transactions entirely.
- **Fix requires:** Add a transactions sync step to `sync_to_db()` that calls `get_league_transactions()` and stores results in the `transactions` table
- **Location:** `src/yahoo_api.py:1064-1197` (sync_to_db missing transaction sync)
- **Status:** OPEN

### BUG-018: Free Agents Have No Projections (Unrankable)
- **Severity:** HIGH
- **Root Cause:** Of 516 free agents (players in DB but not rostered), only 1 has any projection data. The Free Agents page uses `rank_free_agents()` which computes marginal SGP from projections — with zero projections, all FAs get 0 marginal value and can't be meaningfully ranked.
- **Evidence:** `fa_with_proj = 1` out of 516 available FAs
- **Blocked by:** BUG-001 (projections)
- **Location:** Database state, not a code bug
- **Status:** OPEN

### BUG-019: Yahoo Free Agents Not Fetched
- **Severity:** MEDIUM
- **Root Cause:** The app defines FAs as "players in DB not on any roster." But only 780 MLB players are in the DB. Yahoo leagues have access to a much larger FA pool (~1,500+ players). Yahoo's API provides `get_league_free_agents()` which returns all available FAs with ownership percentages, but this is not called during sync.
- **Evidence:** Only 516 FAs available (780 total - 264 rostered). Many real FA candidates (minor leaguers, bench players) are missing from the DB entirely.
- **Fix requires:** Call Yahoo's free agent endpoint to identify the full FA pool, or expand the MLB Stats API player fetch to include 40-man rosters
- **Location:** `src/yahoo_api.py` (missing free agent fetch), `src/live_stats.py:351` (limited to active rosters)
- **Status:** OPEN

---

## TOTAL BUG COUNT

| Severity | Count | Fixed | Open |
|----------|-------|-------|------|
| CRITICAL | 3 | 0 | 3 |
| HIGH | 6 | 0 | 6 |
| MEDIUM | 4 | 1 | 3 |
| LOW | 3 | 0 | 3 |
| INFO | 3 | 0 | 3 |
| **TOTAL** | **19** | **1** | **18** |

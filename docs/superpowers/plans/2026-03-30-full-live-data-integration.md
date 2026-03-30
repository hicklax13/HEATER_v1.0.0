# Full Live Data Integration Plan — 3 Sub-Projects

## Context

The audit revealed 4 of 11 pages don't use live Yahoo data (Draft Simulator, Player Compare, Closer Monitor, Leaders). Additionally, the Yahoo roster fetch extracts only basic fields despite yfpy returning rich Player objects with team abbreviations, lineup slots, and ownership data. External data sources (ESPN injuries, MLB Stats API probable pitchers, pybaseball Statcast) are available but not integrated. This plan closes all gaps so every feature runs on live data.

## Execution Order

```
Sub-Project 1 (foundation) ─────────────────────────────┐
    │                                                    │
    ├──→ Sub-Project 2C (Leaders team fix, HARD dep)     │
    ├──→ Sub-Project 2A (Player Compare, SOFT dep)       │
    │                                                    │
Sub-Project 3A (ESPN injuries, independent) ──→ Sub-Project 2B (Closer Monitor)
Sub-Project 3B (MLB Stats API enhanced, independent)
Sub-Project 3C (Statcast leaderboards, independent)
```

**Parallelization:** Sub-Projects 1, 3A, 3B, 3C can all start simultaneously. Sub-Project 2 waits for 1 and 3A.

---

## Sub-Project 1: Yahoo Roster Enrichment

**Goal:** Extract richer data from yfpy Player objects that we already fetch but don't use. Fixes BUG-003 (empty team column), adds lineup slot tracking, and enriches team metadata.

### Changes

**`src/yahoo_api.py` — `get_all_rosters()` (lines 782-823)**
- Extract 4 new fields from Player objects using existing `_safe_attr` helper:
  - `editorial_team_abbr` → MLB team abbreviation (fixes empty team bug)
  - `selected_position.position` → fantasy lineup slot (C, BN, IL, etc.)
  - `has_player_notes` → boolean flag for player news
  - `has_recent_player_notes` → boolean flag for recent news
- Add all 4 to the DataFrame output

**`src/yahoo_api.py` — `sync_to_db()` (line 1309)**
- Replace `team_abbr = ""` with `team_abbr = row.get("editorial_team_abbr", "")` so `match_player_id()` can match by team
- Pass `selected_position` and `editorial_team_abbr` to `upsert_league_roster_entry()`

**`src/database.py` — Schema migration + upsert**
- Add columns via `_safe_add_column`: `league_rosters.selected_position`, `league_rosters.editorial_team_abbr`
- Add columns to `league_teams`: `faab_balance`, `waiver_priority`, `number_of_moves`, `number_of_trades`
- Update `upsert_league_roster_entry()` to accept and persist new fields

**`src/yahoo_api.py` — New `get_team_details()` method**
- Extract FAAB balance, waiver priority, move/trade counts from Team objects
- Store in `league_teams` table during `sync_to_db()`

**`src/database.py` — New `compute_ownership_deltas()` function**
- Compute 7-day ownership % change from `ownership_trends` snapshots
- Called at end of roster sync

### Files

| File | Action |
|------|--------|
| `src/yahoo_api.py` | MODIFY — extract new fields, fix team matching |
| `src/database.py` | MODIFY — schema migration, new functions |
| `src/yahoo_data_service.py` | MINOR — no changes (write-through handles it) |
| `tests/test_yahoo_api.py` | ADD — test new field extraction |

---

## Sub-Project 2: Wire Remaining Pages to Live Data

**Goal:** Add Yahoo/live context to Player Compare, Closer Monitor, and Leaders. Draft Simulator stays static (correct for mock drafts).

### 2A: Player Compare (`pages/6_Player_Compare.py`)
- Add roster lookup: "Rostered by: Team X" or "Free Agent" badge for each compared player
- Add ownership % display from `ownership_trends` table
- Import `load_league_rosters` from database, add context after player selection

### 2B: Closer Monitor (`pages/7_Closer_Monitor.py`)
- Add "Refresh Depth Charts" button that re-runs bootstrap depth chart fetch
- Add actual 2026 SV stats from `season_stats` table alongside projected SV
- New cached function `_load_actual_sv_stats()` queries `season_stats` for relievers
- Integrate ESPN injury data (Sub-Project 3A dependency) for closer role status

### 2C: Leaders (`pages/9_Leaders.py`)
- Fix empty team column: use `COALESCE(lr.editorial_team_abbr, p.team)` in SQL JOIN (depends on Sub-Project 1)
- Add ownership % column from `ownership_trends`
- Add "Rostered" indicator showing which team owns the player

### Files

| File | Action |
|------|--------|
| `pages/6_Player_Compare.py` | MODIFY — add roster/ownership context |
| `pages/7_Closer_Monitor.py` | MODIFY — add refresh + live SV stats |
| `pages/9_Leaders.py` | MODIFY — fix team column + add ownership |
| `src/closer_monitor.py` | MODIFY — add `injury_data` param to `build_closer_grid()` |

---

## Sub-Project 3: External Data Sources Integration

**Goal:** Add ESPN injuries API, MLB Stats API enhanced endpoints, and pybaseball Statcast leaderboards.

### 3A: ESPN Injuries API (new module)

**New file: `src/espn_injuries.py`** (~100 lines)
- `fetch_espn_injuries()` → GET `https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/injuries` (free, no auth, JSON)
- `save_espn_injuries_to_db()` → store in existing `player_news` table with `source='espn_injuries'`
- `refresh_espn_injuries()` → orchestrator
- Add to bootstrap pipeline with 1-hour staleness

### 3B: MLB Stats API Enhanced (`src/live_stats.py`)

**New functions:**
- `fetch_probable_pitchers(date)` → MLB Stats API `/schedule?hydrate=probablePitcher` (who's pitching today/this week)
- `fetch_mlb_transactions(start, end)` → `/transactions` endpoint for call-ups, IL moves, options
- `save_mlb_transactions(df)` → store in existing `transactions` table

**Modify:**
- `refresh_all_stats()` — change 24h threshold to 4h for fresher season stats

### 3C: pybaseball Statcast Leaderboards (`src/live_stats.py`)

**New functions:**
- `fetch_statcast_leaderboards(season)` → `statcast_batter_expected_stats()` + `statcast_batter_exitvelo_barrels()` + `statcast_pitcher_expected_stats()` (all from pybaseball, already installed)
- `save_statcast_leaderboards_to_db(df)` → store in existing `statcast_archive` table
- Add to bootstrap pipeline with 24-hour staleness

### Files

| File | Action |
|------|--------|
| `src/espn_injuries.py` | NEW — ESPN injury fetcher |
| `src/live_stats.py` | MODIFY — add probable pitchers, transactions, 4h refresh, Statcast |
| `src/data_bootstrap.py` | MODIFY — add ESPN + Statcast bootstrap phases |
| `src/database.py` | MINOR — add `save_mlb_transactions()` |
| `tests/test_espn_injuries.py` | NEW |
| `tests/test_live_stats_enhanced.py` | NEW |

---

## Verification Plan

1. **Sub-Project 1:** `python -m pytest tests/test_yahoo_api.py -v` — new field extraction tests pass. Verify `editorial_team_abbr` populated in DB.
2. **Sub-Project 2:** Start app → navigate to Player Compare, Closer Monitor, Leaders. Verify roster badges, actual SV stats, team names populated.
3. **Sub-Project 3:** `python -m pytest tests/test_espn_injuries.py tests/test_live_stats_enhanced.py -v`. Verify ESPN injuries in `player_news` table. Verify Statcast metrics in `statcast_archive`.
4. **Full suite:** `python -m pytest -x -q` — 2151+ tests pass, 0 failures.
5. **Lint:** `python -m ruff check . && python -m ruff format --check .`
6. **CI:** `git push origin master` — all 5 jobs pass.

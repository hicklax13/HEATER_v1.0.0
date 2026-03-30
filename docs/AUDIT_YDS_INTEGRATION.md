# YahooDataService Integration Audit Report

**Date:** 2026-03-30
**Scope:** Code-level analysis of live-first Yahoo architecture (commit 7aa6d25)

## Issue Catalog

### CRITICAL

**YDS-001: Matchup dict key mismatch — My Team matchup card renders empty**
- File: `pages/1_My_Team.py` (~line 1010)
- The page reads `matchup.get("opponent")`, `matchup.get("user_stats")`, `matchup.get("opp_stats")`
- Yahoo API returns: `opp_name`, `categories` (list of dicts with `cat`, `you`, `opp` keys)
- Result: Opponent shows "Unknown", W-L-T shows 0-0-12, no category rows display
- Fix: Remap keys from Yahoo's actual response format

### HIGH

**YDS-002: `get_standings()` returns different schemas — wide vs long format**
- File: `src/yahoo_data_service.py` `_fetch_and_sync_standings()`
- When Yahoo is live: returns wide-format DataFrame (columns: team_name, rank, r, hr, rbi...)
- When DB fallback: returns long-format (columns: team_name, category, total, rank)
- Impact: Pages check `"category" in standings_df.columns` — fails on cached Yahoo data
- Affected: Standings page, Trade Finder, Lineup Optimizer, Trade Analyzer
- Fix: Have `_fetch_and_sync_standings()` return from DB after write-through (like rosters does)

**YDS-003: Opponent profiles have empty strengths/weaknesses from live data**
- File: `src/yahoo_data_service.py` `_build_live_profile()`
- Caused by: ISSUE-002 — wide-format standings have no "category" column
- Fix: Resolves automatically when YDS-002 is fixed

### MEDIUM

**YDS-004: Transaction card accesses non-existent "team" column**
- File: `pages/3_Trade_Analyzer.py` (~line 160)
- Yahoo returns `team_from`, `team_to` — page reads `team` (always empty string)
- Fix: Use `_txr.get("team_to", "")` instead

**YDS-005: Bootstrap and YDS race on `clear_league_rosters()`**
- Files: `src/data_bootstrap.py`, `src/yahoo_data_service.py`
- Both call `sync_to_db()` which does DELETE + INSERT on league_rosters
- Brief empty-table window possible if two tabs are open
- Fix: Add staleness check in YDS — skip sync if bootstrap ran within TTL

**YDS-006: `load_league_schedule()` fallback returns garbled multi-team data**
- File: `src/database.py` `load_league_schedule()`
- When no `is_user_team=1` row exists, returns all teams' schedules mixed
- Fix: Return empty dict in the fallback branch

**YDS-007: `_fetch_and_sync_standings()` casing mismatch (lowercase vs uppercase)**
- File: `src/yahoo_data_service.py`
- Yahoo returns lowercase columns (`r`, `hr`), write-through stores uppercase (`R`, `HR`)
- Pages with config.all_categories use uppercase — mismatch on cached Yahoo path
- Fix: Resolves with YDS-002 fix (always return from DB)

### LOW

**YDS-008: `st.success()` followed by `st.rerun()` — message never visible**
- File: `src/ui_shared.py` `render_data_freshness_card()`
- Fix: Use `st.toast()` instead

**YDS-009: Freshness widget omits settings and schedule labels**
- File: `src/ui_shared.py` `render_data_freshness_card()` label_map
- Fix: Add `"settings": "Settings"` and `"schedule": "Schedule"`

**YDS-010: `force_refresh_all()` shows "Empty" for null matchup (expected None)**
- File: `src/yahoo_data_service.py`
- Fix: Special-case matchup to show "No active matchup"

**YDS-011: `weekly_report.py` accepts `opponent_roster` param but never uses it**
- File: `src/weekly_report.py`
- Feature gap, not a bug — quantitative projections not implemented

## Priority Fix Order

1. YDS-002 (fixes YDS-003 and YDS-007 automatically) — normalize standings format
2. YDS-001 — fix matchup key mapping
3. YDS-004 — fix transaction column name
4. YDS-006 — fix schedule fallback
5. YDS-005 — add staleness guard for race condition
6. YDS-008/009/010 — cosmetic fixes

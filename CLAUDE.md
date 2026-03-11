# Fantasy Baseball Draft Tool

## Project Status: Draft Tool + In-Season Management Complete ✅

**All phases complete.** UI/UX design plan: `.claude/plans/optimized-wiggling-meerkat.md`. In-season plan: `docs/plans/2026-03-10-in-season-management-plan.md`

The project has two major pillars:
1. **Draft Tool** — "Broadcast Booth" dark-theme Streamlit app (`app.py`, ~1900 lines): 4-step setup wizard, 3-column draft page, Monte Carlo recommendations.
2. **In-Season Management** — 4 Streamlit pages (`pages/`) covering team overview, trade analysis, player comparison, and free agent rankings; powered by live MLB Stats API data and ROS projections.

## What's Been Done (UI Redesign)

### Phase 0: Infrastructure ✅
- `.streamlit/config.toml` dark theme created
- `THEME` dict and `inject_custom_css()` extracted to `src/ui_shared.py`
- Google Fonts, component styles, animations (~400 lines CSS)

### Phase 1: Setup Wizard ✅
- 4-step wizard: Import → Settings → League Import → Launch
- Wizard progress bar with amber active / green completed states
- Load Sample Data button with in-process import
- Step 4 replaced Yahoo OAuth with League Import CSV upload

### Phase 2: Draft Page Layout ✅
- 3-column layout: My Roster | Center (hero pick + alternatives) | Enter Pick
- Command bar: Round/Pick, YOUR TURN badge, progress bar

### Phases 3-6: Components + Animations + Streamlit Features ✅
- Hero pick card, alternative cards, roster diamond, scarcity rings
- Pick entry panel, recent picks feed, category radar chart
- Draft board, draft log, available players table
- Amber glow pulse, card hover lift, skeleton loading
- `@st.fragment`, `st.toast()`, `st.status()`

### Phase 7: Verification ✅
11 bugs found and fixed during verification. All 15 checklist items passed.

<details>
<summary>Bugs fixed (click to expand)</summary>

| # | Bug | Root Cause | Fix |
|---|-----|-----------|-----|
| 1 | "No player data found" on Step 4 | `create_blended_projections()` deletes `system='blended'` rows, but sample data only has 'blended' | Skip blend when no non-blended projections exist |
| 2 | `SGPCalculator.compute_sgp_denominators` AttributeError | `compute_sgp_denominators` is a standalone function, not a method | Import and call as standalone function |
| 3 | `compute_replacement_levels()` missing args | Function signature is `(pool, config, sgp_calc)` but was called with `(pool)` | Fixed to `(pool, lc, sgp)` |
| 4 | `value_all_players()` wrong args | Signature is `(pool, config, ...)` but was called with `(pool, sgp, repl, lc)` | Fixed to `(pool, lc, replacement_levels=repl)` |
| 5 | Hitter/pitcher count shows 0/190 | Code checked `player_type` column but DB uses `is_hitter` | Changed to `pool["is_hitter"].sum()` |
| 6 | `KeyError: 'player_name'` on draft page | DB column is `name` but UI code uses `player_name` | Added `rename(columns={"name": "player_name"})` in `_build_player_pool()` |
| 7 | Hero pick card rendering error | Missing column references after rename | Fixed column mappings in hero card rendering |
| 8 | Alternative cards not displaying | Score calculation used wrong column names | Aligned with renamed columns |
| 9 | Radar chart Plotly fillcolor error | Plotly 6.x rejects 8-digit hex colors (`#RRGGBBAA`) | Changed to `rgba(r,g,b,a)` format |
| 10 | Radar chart Plotly not installed | Streamlit's Python (3.13 Store) didn't have plotly | Installed plotly in correct Python environment |
| 11 | Draft board snake order wrong | `overall` pick number used display column index instead of pick position | Introduced `pos_in_round` variable for correct snake reversal |

</details>

### Verification Checklist ✅
- [x] 1. Setup wizard step navigation (forward/back/skip)
- [x] 2. Data import through new UI (Load Sample Data works, 190 players)
- [x] 3. Draft page 3-column layout renders
- [x] 4. Hero pick card: correct player + stats + survival gauge
- [x] 5. Alternative cards: 5 players with scores
- [x] 6. Roster diamond: filled/empty slots correct
- [x] 7. Scarcity rings: correct color thresholds
- [x] 8. Pick entry search + draft button works
- [x] 9. Practice mode auto-pick functions
- [x] 10. Undo reverses correctly
- [x] 11. Radar chart renders (or graceful fallback)
- [x] 12. Draft board dark theme + snake order
- [x] 13. Text contrast passes on dark background
- [x] 14. MC simulation <8 seconds (4.45s with 6-round horizon optimization)
- [x] 15. `@st.fragment` — verified unnecessary (architecture avoids redundant reruns)
- [x] Debug logging cleaned up from Load Sample Data handler
- [x] `?skip=1` debug shortcut removed

## In-Season Management (Added Post-Draft)

Four Streamlit pages extend the tool into the regular season. All data comes from public sources (MLB Stats API, pybaseball/FanGraphs) — no Yahoo API required.

### Pages
- **My Team** (`pages/1_My_Team.py`) — Roster overview, season stats to date, category standings position
- **Trade Analyzer** (`pages/2_Trade_Analyzer.py`) — Proposal builder + MC-backed trade analysis with verdict and confidence %
- **Player Compare** (`pages/3_Player_Compare.py`) — Head-to-head player comparison across all 10 roto categories
- **Free Agents** (`pages/4_Free_Agents.py`) — FA rankings by net marginal SGP value relative to user's roster

### Key In-Season Algorithms

**Trade Analyzer**
- Before/after roster swap → projected season totals (actual YTD + ROS projections)
- Park factor adjustment per player
- Live SGP delta computation
- MC simulation (200 sims) → positional scarcity check
- Outputs verdict with confidence %

**Player Comparison**
- ROS projections → Z-score normalization across all 10 roto categories
- Composite weighted score
- Optional team impact analysis via marginal SGP

**Free Agent Ranker**
- Marginal SGP per FA relative to user's current roster
- Category-need weighting (boosts scarce categories)
- Identifies which rostered player the FA would replace
- Sorts by net marginal value

**Live Stats Pipeline** (`src/live_stats.py`)
- MLB Stats API for current season stats (daily auto-refresh + on-demand)
- pybaseball for ROS projections (FanGraphs Depth Charts) + park factors
- Refresh log tracks staleness per data source

### New Database Tables (6)

| Table | Description |
|-------|-------------|
| `season_stats` | Actual 2026 stats from MLB Stats API |
| `ros_projections` | Rest-of-season projections from FanGraphs |
| `league_rosters` | All 12 teams' rosters (manually uploaded CSV) |
| `league_standings` | Current roto standings |
| `park_factors` | Stadium hitting/pitching adjustments |
| `refresh_log` | Tracks when each data source was last updated |

### Data Model
- **Hybrid approach:** Manual CSV upload for league-specific data (rosters, standings); auto-pull for player stats/projections
- **No Yahoo dependency** — all data from public MLB Stats API and FanGraphs via pybaseball

## Purpose

A live fantasy baseball draft assistant and in-season manager for a Yahoo Sports snake draft league. During the draft, recommends the optimal player to select each round. Post-draft, tracks roster health, evaluates trades, and surfaces waiver wire pickups.

## League Context

- **Provider:** Yahoo Sports
- **League:** FourzynBurn | **Team:** Team Hickey
- **Format:** 12-team snake draft, 23 rounds
- **Scoring:** 5x5 roto categories (R, HR, RBI, SB, AVG / W, SV, K, ERA, WHIP)
- **Manager skill:** Opponents are extremely high-skilled; user is a novice

## Roster Positions

| Slot | Count | Slot | Count |
|------|-------|------|-------|
| C    | 1     | Util | 2     |
| 1B   | 1     | P    | 4     |
| 2B   | 1     | SP   | 2     |
| 3B   | 1     | RP   | 2     |
| SS   | 1     | BN   | 5     |
| OF   | 3     | MI/CI | 0   |

**Total roster size:** 23 (matches draft rounds)

## Tech Stack

- **Framework:** Streamlit (Python), multi-page app
- **Database:** SQLite (`data/draft_tool.db`)
- **Core libs:** pandas, NumPy, SciPy, Plotly
- **Live data:** MLB-StatsAPI (current season stats), pybaseball (ROS projections + park factors)

## File Structure

```
app.py                  — Draft tool: 4-step setup wizard + 3-column draft page (~1900 lines)
.streamlit/config.toml  — Dark theme configuration
.claude/launch.json     — Dev server config for preview tools
pages/
  1_My_Team.py          — In-season: team overview, roster, category standings
  2_Trade_Analyzer.py   — In-season: trade proposal builder + MC analysis
  3_Player_Compare.py   — In-season: head-to-head player comparison
  4_Free_Agents.py      — In-season: free agent rankings by marginal value
src/
  database.py           — SQLite schema (12 tables), CSV import, projection blending, player pool + in-season queries
  valuation.py          — SGP calculator, replacement levels, VORP, category weights, full valuation
  draft_state.py        — Draft state management, roster tracking, snake pick order, JSON persistence
  simulation.py         — Monte Carlo draft simulation, opponent modeling, survival probability, urgency
  in_season.py          — Trade analyzer, player comparison engine, FA ranker
  live_stats.py         — MLB Stats API + pybaseball data fetcher, daily refresh logic
  league_manager.py     — League roster/standings management, CSV import for all 12 teams
  ui_shared.py          — Shared THEME dict + inject_custom_css() used by app.py and pages/
  validation.py         — Validation utilities
tests/
  test_database_schema.py   — DB schema and table existence tests
  test_database_queries.py  — Query function tests
  test_in_season.py         — Trade analyzer, comparison, FA ranker tests
  test_league_manager.py    — League roster/standings management tests
  test_live_stats.py        — Live stats pipeline tests (22 tests total across suite)
  profile_latency.py        — Performance profiling utility
load_sample_data.py     — Generates ~190 sample players for testing
docs/
  plans/
    2026-03-10-in-season-management-plan.md — In-season feature design plan
data/
  draft_tool.db         — SQLite database (created at runtime)
  backups/              — Draft state JSON backups
```

## Architecture & Key Algorithms

### Valuation Pipeline
1. **Projection blending** — Weighted average of multiple projection systems (Steamer, ZiPS, etc.)
2. **SGP (Standings Gain Points)** — Converts raw stats to standings-point movement; auto-computed denominators available
3. **VORP** — Value Over Replacement Player with multi-position flexibility premium
4. **Category weights** — Dynamic: standings-based when league data available, draft-progress-aware scaling
5. **pick_score** = weighted SGP + positional scarcity + VORP bonus

### Draft Engine
- **Monte Carlo simulation** (100-300 sims) with opponent roster tracking
- **Survival probability** — Normal CDF with positional scarcity adjustment
- **Urgency** = (1 - P_survive) * positional_dropoff
- **Combined score** = MC mean SGP + urgency * 0.4
- **Tier assignment** via natural breaks algorithm

### Nine Accuracy Improvements (Implemented)
1. Dynamic replacement levels (recalculated as players are drafted)
2. Actual roster totals in rate stat SGP calculations
3. Opponent roster tracking in simulations
4. Multi-position eligibility bonus in VORP (+0.12 per extra position, +0.08 for scarce positions)
5. Dynamic category targets from standings data
6. Full Monte Carlo integration into pick recommendations
7. Projection confidence discount (low PA/IP players get up to 20% discount)
8. Bench slot optimization (late-draft bonus for multi-position flexibility)
9. Draft position awareness in urgency calculations

## Running the App

```bash
# First time: load sample data
python load_sample_data.py

# Launch the app (multi-page: draft tool on main page, in-season pages in sidebar)
streamlit run app.py
```

The sidebar will show navigation links to all 4 in-season pages once the app is running. In-season pages require live data — run a manual refresh from the My Team page or wait for the daily auto-refresh.

## Key Technical Notes

- Python 3.14 + SQLite returns bytes for some integer columns; fixed with explicit CAST in SQL and `pd.to_numeric()` coercion
- Snake draft pick order: `round % 2 == 0` forward, `round % 2 == 1` reverse
- MC simulation uses NumPy vectorized availability tracking (`is_available` boolean mask)
- SGP denominators stored in `LeagueConfig` dataclass, user-configurable in UI
- Auto-SGP denominators computed by distributing top players across simulated teams, measuring stddev
- Draft state persists to JSON in `data/backups/` for crash recovery
- Sample data uses `system='blended'` — do NOT call `create_blended_projections()` when only blended data exists
- DB column is `name` but UI expects `player_name` — normalized via rename in `_build_player_pool()`
- `compute_replacement_levels(pool, config, sgp_calc)` and `compute_sgp_denominators(pool, config)` are standalone functions in `src/valuation.py`, NOT methods on SGPCalculator
- MC simulation uses 6-round horizon limit for performance (4.45s vs 10.42s without limit)
- Plotly 6.x does NOT accept 8-digit hex colors; use `rgba(r,g,b,a)` format instead
- `src/yahoo_api.py` still exists in the repo but is no longer used; in-season data comes from MLB Stats API + pybaseball

## Data Sources

### Draft
- **Projections:** FanGraphs CSV exports (Steamer, ZiPS, Depth Charts)
- **ADP:** FantasyPros consensus ADP

### In-Season
- **Current stats:** MLB Stats API (via `MLB-StatsAPI` Python package, auto-refreshed daily)
- **ROS projections:** FanGraphs Depth Charts (via `pybaseball`, on-demand or daily)
- **Park factors:** pybaseball / FanGraphs
- **League rosters/standings:** Manual CSV upload via League Import step in setup wizard

## GitHub Repository

- **Repo:** https://github.com/hicklax13/fantasy-baseball-draft-tool (public)
- **CI/CD:** GitHub Actions — lint (ruff), test (Python 3.11-3.13), build verification
- **Release workflow:** Auto-creates GitHub releases on `v*.*.*` tags
- **Current release:** v1.0.0

### Dependabot
Configured in `.github/dependabot.yml` for weekly pip and GitHub Actions updates.

## Progress Log

### 2026-03-10: In-Season Management — Implementation Complete
- 10 commits implementing full in-season management system (commits `8fe65b0`..`a26b169`)
- 4 Streamlit pages: My Team, Trade Analyzer, Player Compare, Free Agents
- 3 new backend modules: `live_stats.py`, `in_season.py`, `league_manager.py`
- 6 new database tables, 22 tests passing
- Yahoo API eliminated; replaced by MLB Stats API + pybaseball
- Setup wizard Step 4 changed from Yahoo OAuth to League Import CSV

### 2026-03-10: Cleanup & Verification Pass (`f87451b`)
- Fixed `datetime.utcnow()` deprecation → `datetime.now(UTC)` (3 sites in database.py)
- Fixed 30 ruff lint errors (unused imports, sort order, UP017, F541) across 11 files
- Auto-formatted 14 files to pass `ruff format --check`
- Added `if/else` guard pattern to pages 1/2/4 (prevents `st.stop()` no-op crash outside Streamlit)
- Replaced deprecated `use_container_width=True` → `width="stretch"` (18 sites across 4 files + app.py)
- Updated CLAUDE.md with full in-season documentation
- **Final state:** 22/22 tests pass, 0 lint errors, 0 format issues, all 4 pages import cleanly, draft engine verified (190 players valued)

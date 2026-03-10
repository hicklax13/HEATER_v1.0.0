# In-Season Management: Live Stats, Trade Analyzer, Player Comparison, Free Agent Analysis

## Context

The Fantasy Baseball Draft Tool is currently a draft-only assistant. The 2026 MLB season is underway and the user needs the app to function as a full in-season manager: pulling live player stats without Yahoo, analyzing trade proposals, comparing players/teams, and ranking free agents. The Yahoo Sports API is being eliminated entirely — replaced by free public data sources (MLB Stats API, pybaseball, FanGraphs).

## Approach Summary

- **4 new Streamlit pages** alongside the existing draft page (multi-page app via `pages/` directory)
- **3 new backend modules**: `src/live_stats.py`, `src/in_season.py`, `src/league_manager.py`
- **6 new database tables** in SQLite
- **Hybrid data model**: Manual upload for league-specific data (rosters, standings), auto-pull for all player stats/projections
- **Daily auto-refresh** + on-demand button for MLB stats
- **Full MC + Live SGP** trade analysis reusing existing valuation engine

## New File Structure

```
pages/
  1_My_Team.py           — Team overview, roster, category standings
  2_Trade_Analyzer.py    — Trade proposal builder + MC analysis
  3_Player_Compare.py    — Head-to-head player comparison
  4_Free_Agents.py       — Free agent rankings by marginal value
src/
  live_stats.py          — NEW: MLB Stats API + pybaseball data fetcher
  in_season.py           — NEW: Trade analyzer, comparison engine, FA ranker
  league_manager.py      — NEW: League roster/standings management, CSV import
  ui_shared.py           — NEW: Extracted THEME dict + inject_custom_css() from app.py
  database.py            — MODIFIED: 6 new tables added to init_db()
  valuation.py           — UNCHANGED: reused by all new features
  simulation.py          — UNCHANGED: MC pattern referenced by in_season.py
  draft_state.py         — UNCHANGED
  yahoo_api.py           — REMOVED: no longer needed
```

## Critical Files to Modify

| File | Action | Details |
|------|--------|---------|
| `src/database.py` | Extend | Add 6 new tables to init_db(), new query functions |
| `app.py` | Refactor | Extract THEME + CSS into `src/ui_shared.py`, remove yahoo_api imports |
| `src/yahoo_api.py` | Remove | Replaced by live_stats.py + league_manager.py |
| `requirements.txt` | Update | Add MLB-StatsAPI, pybaseball |

## Database Schema (6 New Tables)

### season_stats
Actual 2026 stats from MLB Stats API. Same stat columns as `projections` table.
```sql
CREATE TABLE IF NOT EXISTS season_stats (
    player_id INTEGER NOT NULL,
    season INTEGER NOT NULL DEFAULT 2026,
    pa INTEGER DEFAULT 0, ab INTEGER DEFAULT 0, h INTEGER DEFAULT 0,
    r INTEGER DEFAULT 0, hr INTEGER DEFAULT 0, rbi INTEGER DEFAULT 0,
    sb INTEGER DEFAULT 0, avg REAL DEFAULT 0,
    ip REAL DEFAULT 0, w INTEGER DEFAULT 0, sv INTEGER DEFAULT 0,
    k INTEGER DEFAULT 0, era REAL DEFAULT 0, whip REAL DEFAULT 0,
    er INTEGER DEFAULT 0, bb_allowed INTEGER DEFAULT 0, h_allowed INTEGER DEFAULT 0,
    games_played INTEGER DEFAULT 0,
    last_updated TEXT,
    PRIMARY KEY (player_id, season),
    FOREIGN KEY (player_id) REFERENCES players(player_id)
);
```

### ros_projections
Rest-of-season projections from FanGraphs. Same stat columns as `projections`.
```sql
CREATE TABLE IF NOT EXISTS ros_projections (
    player_id INTEGER NOT NULL,
    system TEXT NOT NULL,  -- 'steamer_ros', 'zips_ros', 'the_bat_ros', 'blended_ros'
    pa INTEGER DEFAULT 0, ab INTEGER DEFAULT 0, h INTEGER DEFAULT 0,
    r INTEGER DEFAULT 0, hr INTEGER DEFAULT 0, rbi INTEGER DEFAULT 0,
    sb INTEGER DEFAULT 0, avg REAL DEFAULT 0,
    ip REAL DEFAULT 0, w INTEGER DEFAULT 0, sv INTEGER DEFAULT 0,
    k INTEGER DEFAULT 0, era REAL DEFAULT 0, whip REAL DEFAULT 0,
    er INTEGER DEFAULT 0, bb_allowed INTEGER DEFAULT 0, h_allowed INTEGER DEFAULT 0,
    updated_at TEXT,
    PRIMARY KEY (player_id, system),
    FOREIGN KEY (player_id) REFERENCES players(player_id)
);
```

### league_rosters
All 12 teams' rosters (manually uploaded).
```sql
CREATE TABLE IF NOT EXISTS league_rosters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_name TEXT NOT NULL,
    team_index INTEGER NOT NULL,  -- 0-11
    player_id INTEGER NOT NULL,
    roster_slot TEXT,  -- C, 1B, SP, BN, etc.
    is_user_team INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (player_id) REFERENCES players(player_id)
);
```

### league_standings
Current roto standings (manually entered or computed from stats).
```sql
CREATE TABLE IF NOT EXISTS league_standings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_name TEXT NOT NULL,
    category TEXT NOT NULL,  -- R, HR, RBI, SB, AVG, W, SV, K, ERA, WHIP
    total REAL DEFAULT 0,
    rank INTEGER,
    points REAL
);
```

### park_factors
Stadium hitting/pitching adjustments (loaded once yearly).
```sql
CREATE TABLE IF NOT EXISTS park_factors (
    team_code TEXT PRIMARY KEY,
    factor_hitting REAL DEFAULT 1.0,
    factor_pitching REAL DEFAULT 1.0
);
```

### refresh_log
Tracks when data sources were last updated.
```sql
CREATE TABLE IF NOT EXISTS refresh_log (
    source TEXT PRIMARY KEY,  -- 'mlb_stats', 'ros_projections', 'park_factors'
    last_refresh TEXT,
    status TEXT DEFAULT 'success'
);
```

## New Module: `src/live_stats.py`

Data fetcher using MLB Stats API + pybaseball. No Yahoo dependency.

### Key Functions

```python
def fetch_season_stats(season: int = 2026) -> pd.DataFrame:
    """Pull current season stats for all MLB players from MLB Stats API.
    Uses MLB-StatsAPI library. Returns DataFrame matching season_stats schema."""

def fetch_ros_projections(system: str = "steamer") -> pd.DataFrame:
    """Pull ROS projections from FanGraphs via pybaseball.
    Returns DataFrame matching ros_projections schema."""

def fetch_park_factors() -> pd.DataFrame:
    """Pull park factors from Baseball Savant via pybaseball.
    Returns DataFrame matching park_factors schema."""

def refresh_all_stats(force: bool = False) -> dict:
    """Orchestrator: checks refresh_log, pulls data if stale.
    Daily auto-refresh for season_stats, weekly for ROS projections.
    Returns dict of {source: status}."""

def match_player_ids(external_name: str, external_team: str) -> int | None:
    """Match MLB Stats API player names to our players table.
    Uses fuzzy matching (existing pattern from yahoo_api.py sync_draft_picks)."""
```

## New Module: `src/league_manager.py`

Handles league-specific data that can't come from public APIs.

### Key Functions

```python
def import_league_rosters_csv(csv_path: str, user_team_name: str):
    """Import all 12 teams' rosters from CSV.
    Expected format: team_name, player_name, position, roster_slot"""

def import_standings_csv(csv_path: str):
    """Import current standings from CSV.
    Expected format: team_name, R, HR, RBI, SB, AVG, W, SV, K, ERA, WHIP"""

def add_player_to_roster(team_name: str, player_id: int, roster_slot: str):
    """Manual roster update (for trades/adds)."""

def remove_player_from_roster(team_name: str, player_id: int):
    """Manual roster update (for trades/drops)."""

def get_free_agents(player_pool: pd.DataFrame) -> pd.DataFrame:
    """Return all players NOT on any league_rosters team."""

def get_team_roster(team_name: str) -> pd.DataFrame:
    """Return roster for a specific team with stats."""

def compute_standings_from_stats() -> pd.DataFrame:
    """Compute roto standings from season_stats + league_rosters."""
```

## New Module: `src/in_season.py`

Core algorithms for trade analysis, player comparison, and free agent ranking.

### Trade Analyzer: `analyze_trade()`

**Inputs:** giving_away (player_ids), receiving (player_ids), config (LeagueConfig)

**Algorithm:**
1. Build before/after rosters by swapping players
2. Compute projected season totals: `actual_stats + ros_projection` for each player
3. Apply park factor adjustments
4. Sum category totals for before-roster and after-roster
5. Live SGP: `sgp_change[cat] = (after_total[cat] - before_total[cat]) / sgp_denominator[cat]`
6. MC simulation (200 sims): model opponent performance with normal variance around ROS projections, compute final standings position before and after trade
7. Positional scarcity check via `compute_replacement_levels()`
8. Injury risk flags from players table

**Outputs:** TradeResult dataclass with verdict, category_impact (dict), mc_standings_before/after (arrays), confidence_pct, risk_flags (list)

**Reuses:** `SGPCalculator` (valuation.py), `compute_replacement_levels()` (valuation.py), `compute_category_weights()` (valuation.py), MC simulation pattern (simulation.py)

### Player Comparison: `compare_players()`

**Inputs:** player_id_a, player_id_b, config, user_roster (optional)

**Algorithm:**
1. Load ROS projections for both players
2. Z-score normalization: `z[cat] = (stat - league_mean) / league_stddev` across all 10 categories
3. Composite score: weighted sum of z-scores using `compute_category_weights()`
4. If user_roster provided: compute team impact of swapping A for B using marginal SGP

**Outputs:** CompareResult with z_scores (dict per player), composite_scores, team_impact (optional)

### Team Comparison: `compare_teams()`

**Inputs:** team_a_name, team_b_name

**Algorithm:**
1. Load both team rosters with projected season totals
2. Z-score per category at team level
3. Category surplus/deficit for each team

**Outputs:** TeamCompareResult with category_totals, z_scores, surplus_deficit per category

### Free Agent Ranker: `rank_free_agents()`

**Inputs:** user_roster, free_agent_pool, config

**Algorithm:**
1. For each FA: compute `marginal_sgp` using `SGPCalculator.marginal_sgp()` relative to user roster
2. Weight by category needs via `compute_category_weights()`
3. For each FA: identify which current roster player they'd replace (lowest marginal value at same position)
4. Sort by net marginal value (FA marginal SGP - replaced player marginal SGP)

**Outputs:** DataFrame sorted by marginal_value with columns: player info, marginal_sgp, replaces_player, category_impact

## New Module: `src/ui_shared.py`

Extracted from app.py to share across all pages.

```python
# Extracted from app.py lines ~54-76
THEME = { ... }

# Extracted from app.py lines ~98-656
def inject_custom_css(): ...

# Common UI helpers
def render_page_header(title: str, subtitle: str = ""): ...
def render_category_table(data: dict, categories: list): ...
def render_radar_chart(data: dict, title: str): ...  # existing plotly pattern
```

## UI Page Layouts

### pages/1_My_Team.py
- Header: Team name + standings position + "Last updated" badge
- Refresh Stats button (calls `refresh_all_stats()`)
- 3 columns: Roster table | Category standings heatmap (10 cats × rank 1-12) | Strengths/weaknesses
- Bottom: Full league standings table

### pages/2_Trade_Analyzer.py
- Trade builder: "You Give" + "You Receive" player selectors (multiselect dropdowns)
- "Analyze Trade" button → `st.status()` progress → `analyze_trade()` call
- Results: Verdict banner (green ACCEPT / red DECLINE + confidence %), category impact table, MC histogram, risk flags

### pages/3_Player_Compare.py
- Two player search boxes
- Plotly radar chart overlay (existing pattern)
- Side-by-side stats table with color-coded advantages
- Team impact section (if comparing for a roster swap)

### pages/4_Free_Agents.py
- Position filter + min PA/IP filter + Refresh button
- Ranked table: Name, Positions, Key Stats, Marginal SGP, Replaces, Category Impact
- Click-to-expand: full comparison vs replaced player

### Setup Wizard Extension (in app.py)
- Add Step 5: "Import League" — upload roster CSV + standings CSV, set user team name
- League data management UI for manual roster updates

## Dependencies to Add

```
MLB-StatsAPI>=1.7
pybaseball>=2.3
```

## Implementation Order

1. **Phase 1: Foundation** — `src/ui_shared.py` extraction, database schema, `src/league_manager.py`
2. **Phase 2: Data Pipeline** — `src/live_stats.py`, MLB Stats API integration, auto-refresh
3. **Phase 3: Core Algorithms** — `src/in_season.py` (trade analyzer, comparison, FA ranker)
4. **Phase 4: UI Pages** — All 4 new Streamlit pages
5. **Phase 5: Integration** — Setup wizard Step 5, remove yahoo_api.py, update CLAUDE.md
6. **Phase 6: Verification** — End-to-end testing with sample data

## Verification Plan

1. **Data pipeline**: Run `refresh_all_stats()`, verify season_stats table has 2026 data for 500+ players
2. **League import**: Upload test CSV with 12 team rosters, verify league_rosters populated correctly
3. **Trade analyzer**: Propose a known trade (e.g., top hitter for top pitcher), verify:
   - Category impact table shows correct before/after
   - MC simulation completes in <10s
   - Verdict aligns with intuition (trading elite hitter hurts HR/RBI/R, helps pitching)
4. **Player comparison**: Compare two known players, verify z-scores and radar chart render
5. **Free agent ranker**: Verify free agents sorted by marginal value, top FA addresses team's weakest category
6. **UI**: All 4 pages render in Broadcast Booth theme, no broken styles
7. **Auto-refresh**: Verify refresh_log updates, stale data triggers re-fetch
8. **Existing draft page**: Verify draft tool still works exactly as before (no regressions)

# League Standings + Matchup Planner Redesign

**Date:** 2026-04-05
**Status:** Design approved, pending implementation
**Scope:** Two pages + shared engine module + Yahoo data enhancement

---

## Context

The existing `pages/8_Standings.py` has a Projected Standings tab (MC simulation) and a Power Rankings tab. Both have significant limitations:

- MC simulation uses generic round-robin schedule instead of actual Yahoo schedule
- Simulation requires manual "Run Simulation" button click each time
- Power Rankings has 2/5 factors permanently N/A (injury exposure, momentum)
- No view of actual live Yahoo H2H standings (W-L-T records)
- No per-category win probabilities for matchups
- Matchup Planner (page 11) only shows per-game player ratings, not H2H category analysis

This redesign replaces both pages and adds a shared computation engine.

---

## Deliverables

| Deliverable | Type | Description |
|-------------|------|-------------|
| `src/standings_engine.py` | New module | Shared computation engine (pure functions, no Streamlit) |
| `pages/8_League_Standings.py` | Replace `pages/8_Standings.py` | 2-tab page: Current Standings + Season Projections |
| `pages/11_Matchup_Planner.py` | Rewrite | Add Category Probabilities tab + week navigator |
| `yahoo_data_service.py` | Enhancement | `get_full_league_schedule()` method |
| `database.py` | Enhancement | `league_schedule_full` table + CRUD |
| `tests/test_standings_engine.py` | New tests | Full coverage of engine functions |

---

## 1. Shared Engine: `src/standings_engine.py`

All computation lives here. Pure functions, no Streamlit dependency, fully testable.

### 1.1 Full League Schedule Fetcher

```python
def fetch_full_league_schedule(
    yahoo_client,
    total_weeks: int = 24,
) -> dict[int, list[tuple[str, str]]]:
    """Fetch all matchups for all weeks from Yahoo API.

    Returns: {week_num: [(team_a, team_b), ...]} -- 6 matchups per week.
    Uses _request_with_backoff() for 429 protection.
    Stores in league_schedule_full table (24h TTL).
    Falls back to hardcoded AVIS schedule for user's team if Yahoo fails.
    """
```

**Implementation:**
- Loop `get_league_scoreboard_by_week(chosen_week=week)` for weeks 1 through `total_weeks`
- Parse `scoreboard.matchups` to extract all team pairs
- Store in `league_schedule_full` DB table
- Cache in `st.session_state["_full_league_schedule"]`
- 24h TTL — schedule doesn't change during season
- ~24 API calls, ~12 seconds with rate limiting
- Fallback: `TEAM_HICKEY_SCHEDULE` from `opponent_intel.py` (user's team only)

**New DB table:**
```sql
CREATE TABLE league_schedule_full (
    week INTEGER NOT NULL,
    team_a TEXT NOT NULL,
    team_b TEXT NOT NULL,
    PRIMARY KEY (week, team_a, team_b)
);
```

### 1.2 Category Win Probability Engine

```python
def compute_category_win_probabilities(
    user_roster_ids: list[int],
    opp_roster_ids: list[int],
    player_pool: pd.DataFrame,
    config: LeagueConfig,
    weeks_played: int = 0,
    season_stats: pd.DataFrame | None = None,
) -> dict:
    """Compute per-category P(user wins) for a head-to-head matchup.

    Returns: {
        "overall_win_pct": float,      # 0-1
        "overall_tie_pct": float,      # 0-1
        "overall_loss_pct": float,     # 0-1
        "projected_score": {"W": float, "L": float, "T": float},
        "categories": [
            {
                "name": str,           # e.g. "HR"
                "user_proj": float,    # projected weekly total
                "opp_proj": float,
                "win_pct": float,      # 0-1
                "confidence": str,     # "high" / "medium" / "low"
                "is_inverse": bool,
            },
            ...
        ],
    }
    """
```

**Algorithm — Bayesian-Updated Normal + Gaussian Copula:**

**Step 1: Estimate weekly production per team per category**

For counting stats (R, HR, RBI, SB, W, L, SV, K):
- Sum ROS projections for active starters (exclude IL/BN)
- Scale to weekly: `weekly_proj = season_proj / weeks_remaining`
- Active starters determined by roster slot (not bench)

For rate stats (AVG, OBP):
- Weighted by projected PA: `AVG = sum(H) / sum(AB)` across starters
- Weekly PA estimated from season PA / weeks_remaining

For rate stats (ERA, WHIP):
- Weighted by projected IP: `ERA = sum(ER)*9 / sum(IP)` across starters
- Weekly IP estimated from season IP / weeks_remaining

**Step 2: Estimate weekly variance per team per category**

Base variance from existing `WEEKLY_TAU` constants:
```python
WEEKLY_TAU = {
    "R": 6.0, "HR": 2.5, "RBI": 6.5, "SB": 1.5,
    "AVG": 0.020, "OBP": 0.025, "W": 1.0, "L": 1.0,
    "SV": 1.5, "K": 8.0, "ERA": 0.80, "WHIP": 0.15,
}
```

Bayesian shrinkage (when `weeks_played > 0` and `season_stats` provided):
```python
# Observed variance from season-to-date weekly performance
obs_var = observed_weekly_variance(team_stats, category)
# Prior variance
prior_var = WEEKLY_TAU[category] ** 2
# Posterior variance (shrink toward observed as more data accumulates)
posterior_var = 1.0 / (1.0 / prior_var + weeks_played / obs_var)
posterior_std = sqrt(posterior_var)
```

Roster-adjusted: teams with IL starters get +15% variance (less predictable).

Confidence label:
- `weeks_played >= 8`: "high"
- `weeks_played >= 4`: "medium"
- `weeks_played < 4`: "low"

**Step 3: Independent category win probabilities**

For counting stats and inverse counting stats (L):
```python
mu_diff = mu_user - mu_opp  # (flipped for inverse: mu_opp - mu_user)
sigma_diff = sqrt(var_user + var_opp)
p_win = norm.cdf(mu_diff / sigma_diff)
```

For rate stats (AVG, OBP, ERA, WHIP):
- Use delta method for ratio estimator variance
- AVG variance: `var(H/AB) ~ AVG*(1-AVG)/PA` (binomial approximation)
- ERA variance: `var(ER*9/IP) ~ (ERA_std)^2` from tau
- For inverse rate stats (ERA, WHIP): P(win) = P(user < opponent)

**Step 4: Correlated overall matchup probability (Gaussian copula)**

Category correlation matrix (estimated from historical league data):
```python
CATEGORY_CORRELATIONS = {
    ("HR", "R"): 0.72,    ("HR", "RBI"): 0.68,   ("R", "RBI"): 0.65,
    ("AVG", "OBP"): 0.85, ("ERA", "WHIP"): 0.78,
    ("W", "K"): 0.45,     ("SB", "AVG"): -0.15,
    # All other pairs default to 0.0 (independent)
}
```

Simulation (10,000 draws):
```python
# Build 12x12 correlation matrix from pairs above
# Generate 10K correlated Normal(0,1) samples via Cholesky decomposition
# Transform to category-specific distributions using marginal CDFs
# For each draw: count wins per category, determine matchup outcome
# overall_win_pct = fraction of draws where user wins > 6 categories
# overall_tie_pct = fraction where exactly 6-6
# overall_loss_pct = 1 - win - tie
```

Projected score: `E[categories_won]`, `E[categories_lost]`, `E[ties]` from the 10K sims.

### 1.3 Enhanced MC Season Simulation

```python
def simulate_season_enhanced(
    current_standings: dict[str, dict],  # {team: {W, L, T}}
    team_rosters: dict[str, list[int]],  # {team: [player_ids]}
    full_schedule: dict[int, list[tuple[str, str]]],
    player_pool: pd.DataFrame,
    config: LeagueConfig,
    current_week: int = 1,
    n_sims: int = 1000,
    seed: int = 42,
    injury_data: pd.DataFrame | None = None,
    momentum_data: dict[str, float] | None = None,
) -> dict:
    """Schedule-aware MC season simulation.

    Returns: {
        "projected_records": {team: {W, L, T, win_pct}},
        "playoff_probability": {team: float},
        "magic_numbers": {team: int | None},
        "elimination_numbers": {team: int | None},
        "strength_of_schedule": {team: float},
        "confidence_intervals": {team: {p5_wins, p95_wins}},
        "team_strength": {team: {power_rating, roster_quality, ...}},
    }
    """
```

**Key differences from existing `simulate_season()`:**

| Aspect | Old | New |
|--------|-----|-----|
| Schedule | Round-robin (generic) | Actual Yahoo schedule for remaining weeks |
| Starting point | Week 1, no existing results | Current W-L-T record carried forward |
| Team means | Fixed for all sims | Bayesian-updated from season-to-date |
| Variance | Fixed `WEEKLY_TAU` | Dynamic: shrinks with more data, widens for injured teams |
| Category correlations | Independent | Gaussian copula (same as 1.2) |
| Injury impact | None | IL players excluded from projections, +15% variance |
| Momentum | None | Last 4 weeks W-L ratio adjusts mean by +/-5% |
| Output | Just W-L-T + playoff% | + magic numbers, elimination numbers, SOS, team strength |

**Algorithm:**

1. Compute weekly projections per team (same as 1.2 Step 1)
2. Build correlation matrix (same as 1.2 Step 4)
3. For each of `n_sims` iterations:
   a. Start with actual W-L-T record from `current_standings`
   b. For each remaining week (from `current_week` to end of season):
      - Get matchups from `full_schedule[week]`
      - For each matchup: simulate correlated category outcomes
      - Award W/L/T based on category wins
   c. Record final W-L-T per team
4. Aggregate across sims:
   - Mean/p5/p95 wins per team
   - Playoff probability (top 4 finish)
   - Magic/elimination numbers (see 1.4)
   - Strength of schedule (from team_strength profiles)

### 1.4 Magic Number Computation

```python
def compute_magic_numbers(
    current_standings: dict[str, dict],
    sim_results: np.ndarray,  # (n_sims, n_teams) final win counts
    team_names: list[str],
    playoff_spots: int = 4,
) -> dict[str, int | None]:
    """Compute magic number (wins to clinch) per team.

    Magic number: the number of additional wins needed such that
    P(playoffs | wins >= current + magic) >= 95%.

    Returns: {team: magic_number} where None = already eliminated.
    """
```

**Algorithm:**
- For each team, binary search on additional wins [0, remaining_games]
- For each candidate X: filter sims where team wins >= current_wins + X
- If P(playoffs) >= 95% in that filtered set, X is sufficient
- Magic number = smallest such X
- If no X achieves 95%, team is effectively eliminated (return None)

### 1.5 Team Strength Profiles

```python
def compute_team_strength_profiles(
    team_rosters: dict[str, list[int]],
    standings: pd.DataFrame,
    player_pool: pd.DataFrame,
    config: LeagueConfig,
    full_schedule: dict[int, list[tuple[str, str]]] | None = None,
    current_week: int = 1,
    injury_data: pd.DataFrame | None = None,
    recent_records: dict[str, dict] | None = None,
) -> list[dict]:
    """Compute 5-factor team strength profiles.

    Uses existing compute_power_rankings() from src/power_rankings.py
    but now wires up ALL 5 factors:

    - roster_quality (40%): Z-score of team's avg category rank (existing)
    - category_balance (25%): Fraction of cats beating median (existing)
    - schedule_strength (15%): Avg opponent roster_quality for REMAINING schedule (NEW)
    - injury_exposure (10%): Weighted health scores from injury_model (NEW)
    - momentum (10%): Last 4 weeks W-L vs season W-L ratio (NEW)
    """
```

**Injury exposure wiring:**
```python
from src.injury_model import compute_health_score
# For each team's top 10 players (by SGP):
# health_score = compute_health_score(player_id)
# injury_exposure = 1 - weighted_avg(health_scores, weights=sgp_values)
```

**Momentum wiring:**
```python
# recent_records: {team: {recent_W, recent_L, season_W, season_L}}
# recent_win_rate = recent_W / (recent_W + recent_L)
# season_win_rate = season_W / (season_W + season_L)
# momentum = recent_win_rate / season_win_rate  # clipped to [0.5, 2.0]
```

---

## 2. League Standings Page: `pages/8_League_Standings.py`

Replaces `pages/8_Standings.py`. Deletes the old file.

### 2.1 Page Config

- Title: "LEAGUE STANDINGS"
- Icon: "standings" (from PAGE_ICONS)
- Layout: wide, sidebar collapsed
- Uses: `render_page_layout()`, `render_context_columns()`, `render_context_card()`, `render_compact_table()`, `inject_custom_css()`

### 2.2 Recommendation Banner

**Source:** `yds.get_matchup()` + `compute_category_win_probabilities()`

**Template:**
```
This week vs {opponent}: leading {W}-{L}-{T} in categories.
Close battles: {cat1} ({user} vs {opp}), {cat2} ({user} vs {opp}).
Focus areas: {weakest_winnable_cats}.
```

**Fallback (no Yahoo):** "Connect your Yahoo league to see live matchup analysis."

### 2.3 Context Panel (Left ~20%)

Four glassmorphic cards:

**Card 1: YOUR POSITION**
- Rank (ordinal: "3rd")
- W-L-T record
- GB from 1st place
- GB from playoff line (4th place) or games ahead of 5th

**Card 2: THIS WEEK**
- Opponent name
- Live category score (W-L-T) from `yds.get_matchup()`
- Win probability % from engine

**Card 3: PLAYOFF ODDS**
- Playoff probability % from MC simulation
- Magic number (wins to clinch)
- Elimination number (losses to be eliminated)

**Card 4: DATA FRESHNESS**
- Standard `render_data_freshness_card()` widget

### 2.4 Tab 1: Current Standings

**Section A: Head-to-Head Record Table**

Columns: Rank, Team, W, L, T, Win%, GB, Streak

Data source: `yds.get_standings()` — the Yahoo API returns wins, losses, ties, percentage, streak in the meta columns that the current `_fetch_and_sync_standings()` filters out. Enhancement: capture these meta columns too.

Rendering:
- `build_compact_table_html()` with custom styling
- User's team row highlighted with `T["primary"]` tint
- Dashed red border between rank 4 and 5 (playoff cutoff)
- Sorted by Win% descending (same as Yahoo)

**Section B: Category Standings Grid**

A 12-team x 12-category matrix showing each team's **rank** per category.

Data source: `yds.get_standings()` in long format (team_name, category, total, rank).
Pivot to wide: teams as rows, categories as columns, cells show rank.

Rendering:
- `build_compact_table_html()` with custom `highlight_cols`
- Rank badges: color-coded by tier
  - Rank 1-4: `T["green"]` background
  - Rank 5-8: `T["sky"]` background
  - Rank 9-12: `T["danger"]` / `T["primary"]` background
- User's team row highlighted
- Hit categories with `.th-hit` header, pitch with `.th-pit`
- Hovering a cell shows the actual stat total (tooltip or expander)

**Fallback (no Yahoo):** "Connect your Yahoo league to see live standings. Run the app bootstrap first."

### 2.5 Tab 2: Season Projections

**Auto-run:** Simulation runs on page load (cached in `session_state`). No button click required. Re-runs when "Refresh" is clicked or data changes.

**Section A: Projected Final Standings Table**

Columns: Rank, Team, Projected W, Projected L, Projected T, Win%, Playoff%, Magic#, SOS

Data source: `simulate_season_enhanced()` output.

Rendering:
- `build_compact_table_html()` with custom styling
- Playoff% column: green gradient for high probability, red for low
- Magic# column: bold number, "CLINCHED" if 0, "--" if eliminated
- User's team row highlighted

**Section B: Team Strength Profiles (merged from Power Rankings)**

Expandable section showing all 5 factors per team:
- Power Rating (0-100)
- Roster Quality, Category Balance, Schedule Strength, Injury Exposure, Momentum
- Bootstrap confidence intervals (p5, p95)

Rendered as compact table with color-coded component scores.

**Section C: Scenario Explorer**

Interactive: "What if I go ___-___-___ this week?"
- Three number inputs (W, L, T) constrained to sum to 12
- Re-runs simulation with modified current-week result
- Shows delta: "Your playoff odds change from 78% to 84% (+6%)"
- Modifies the user's W-L-T for the selected week, then re-runs MC simulation for remaining weeks from that point
- Uses cached team projections and schedule — only the starting W-L-T changes
- Target latency: <2s for re-simulation

---

## 3. Matchup Planner Redesign: `pages/11_Matchup_Planner.py`

Extends existing page with new primary tab and week navigation.

### 3.1 Page Config

Same as current: title "MATCHUP PLANNER", icon "calendar", wide layout, sidebar collapsed.

### 3.2 Week Navigator (Context Panel)

**Left/right arrow buttons** for browsing weeks 1 through final week.
- Default: current week (auto-detected from `yds.get_matchup()`)
- Stored in `st.session_state["matchup_week"]`
- Shows: week number, date range, opponent name
- Label: "Current week" / "Past" / "Future"

**Data for each week:**
- Past weeks (week < current): Actual results from Yahoo (`get_league_scoreboard_by_week(week)`)
- Current week: Live data from `yds.get_matchup()` + projections for remaining days
- Future weeks: Projected from rosters + ROS projections

**Opponent identification:** From `full_schedule[week]` — find the matchup containing user's team.

### 3.3 Context Panel Cards

**Card 1: WEEK NAVIGATOR** (arrow buttons + week info)
**Card 2: WIN PROBABILITY** (overall W/T/L percentages + stacked bar)
**Card 3: PROJECTED SCORE** (expected category W-L-T)
**Card 4: FILTERS** (player type: All/Hitters/Pitchers, team selector)

### 3.4 Recommendation Banner

**Template:**
```
Week {N} vs {opponent}: {win_pct}% chance to win (projected {W}-{L}-{T}).
Best odds: {top_cat} ({pct}%), {second_cat} ({pct}%).
Toss-ups: {tossup_cat} ({pct}%), {tossup_cat2} ({pct}%).
```

### 3.5 New Tab: Category Probabilities

**Per-category win probability display:**

For each of the 12 scoring categories, show:
- Category name (color-coded: hit=blue header, pitch=red header)
- Horizontal probability bar (0-100%)
- Bar color: green (>=70%), blue (55-70%), orange (45-55%), red (<45%)
- User projected total vs opponent projected total
- Confidence indicator (high/medium/low based on weeks played)

**Rendering:**
- Custom HTML via `st.markdown(unsafe_allow_html=True)`
- Each category is a row: `[Cat Name] [====Bar====] [pct%] [user_val vs opp_val]`
- Sorted by probability descending (strongest advantages first)

**Past week behavior:**
- When viewing a completed week, show actual W/L result per category instead of probability
- Green checkmark for categories won, red X for categories lost, gray dash for ties
- Show actual totals instead of projections

**Future week behavior:**
- Show probabilities from `compute_category_win_probabilities()`
- Uses current rosters + ROS projections
- Note: "Based on current rosters as of {date}. Roster changes will affect projections."

### 3.6 Existing Tabs (Preserved)

The existing tabs are kept with minor renames:
- "Player Matchups" (was "Summary") — per-player matchup rating table
- "Per-Game Detail" — expandable per-game breakdown
- "Hitters Only" — filtered hitter view
- "Pitchers Only" — filtered pitcher view

These tabs continue to show the per-game player ratings from `compute_weekly_matchup_ratings()`. They always operate on the **current real-world MLB game schedule** (next 7/10/14 days of actual baseball games) regardless of which fantasy week is selected in the navigator. The week navigator only affects the Category Probabilities tab, which deals with fantasy H2H matchups, not individual MLB game schedules.

---

## 4. Yahoo Data Enhancement

### 4.1 `get_full_league_schedule()` in `yahoo_data_service.py`

```python
def get_full_league_schedule(self, force_refresh: bool = False) -> dict[int, list[tuple[str, str]]]:
    """Fetch all matchups for all weeks.

    Returns: {week: [(team_a, team_b), ...]}
    TTL: 86400 seconds (24 hours)
    Cache: session_state["_full_league_schedule"]
    Storage: league_schedule_full table
    """
```

### 4.2 Capture W-L-T Meta Columns from Standings

The current `_fetch_and_sync_standings()` filters out meta columns (wins, losses, ties, percentage, streak). Enhancement: store these in a new `league_records` table or extend `league_standings` with a separate record view.

**New DB table:**
```sql
CREATE TABLE league_records (
    team_name TEXT PRIMARY KEY,
    wins INTEGER DEFAULT 0,
    losses INTEGER DEFAULT 0,
    ties INTEGER DEFAULT 0,
    win_pct REAL DEFAULT 0,
    points_for REAL DEFAULT 0,
    points_against REAL DEFAULT 0,
    streak TEXT DEFAULT '',
    rank INTEGER DEFAULT 0,
    updated_at TEXT
);
```

### 4.3 Historical Scoreboard Fetch (for past week results)

```python
def get_week_scoreboard(self, week: int, force_refresh: bool = False) -> dict:
    """Fetch scoreboard for a specific week.

    Returns: {
        "week": int,
        "matchups": [
            {
                "team_a": str, "team_b": str,
                "team_a_wins": int, "team_b_wins": int,
                "categories": [{name, team_a_val, team_b_val, winner}, ...]
            },
            ...
        ]
    }
    For past weeks: cached permanently (results don't change).
    For current week: 5min TTL (live updating).
    """
```

---

## 5. Data Flow Summary

```
Yahoo API
  |
  ├─ get_standings() ──────────> league_standings + league_records tables
  ├─ get_full_league_schedule() ──> league_schedule_full table
  ├─ get_matchup() ────────────> current week live scores
  ├─ get_week_scoreboard(N) ───> past week results (permanent cache)
  └─ get_rosters() ────────────> league_rosters table
          |
          v
  standings_engine.py
  ├─ compute_category_win_probabilities() ──> per-category P(win)
  ├─ simulate_season_enhanced() ────────────> projected records + playoff odds
  ├─ compute_magic_numbers() ───────────────> wins to clinch
  └─ compute_team_strength_profiles() ──────> 5-factor power ratings
          |
          v
  ┌─────────────────────────────────────┐
  │ pages/8_League_Standings.py         │
  │ Tab 1: Current Standings            │
  │   - H2H record table (W-L-T)       │
  │   - Category rank grid (12x12)     │
  │ Tab 2: Season Projections           │
  │   - Projected final standings       │
  │   - Playoff odds + magic numbers    │
  │   - Team strength profiles          │
  │   - Scenario explorer               │
  └─────────────────────────────────────┘
  ┌─────────────────────────────────────┐
  │ pages/11_Matchup_Planner.py         │
  │ Week Navigator (1-24)               │
  │ Tab 1: Category Probabilities       │
  │   - Per-category win% bars          │
  │   - Past: actual results            │
  │   - Future: projected probabilities │
  │ Tab 2-5: Existing player matchups   │
  └─────────────────────────────────────┘
```

---

## 6. Testing Strategy

### `tests/test_standings_engine.py`

| Test Area | Cases |
|-----------|-------|
| `compute_category_win_probabilities` | Equal teams -> ~50%. Dominant team -> >90%. Inverse categories correct direction. Rate stat weighting. Zero-variance edge case. |
| `simulate_season_enhanced` | Deterministic seed reproducibility. Current W-L-T carried forward. Schedule-aware (specific opponents). Injury impact reduces projections. Momentum adjustment. |
| `compute_magic_numbers` | Already clinched -> 0. Already eliminated -> None. Mid-pack -> positive integer. |
| `compute_team_strength_profiles` | All 5 factors populated. Missing data handled gracefully. Score in [0, 100]. |
| `fetch_full_league_schedule` | Returns 6 matchups per week. All teams appear exactly once per week. Fallback to AVIS schedule. |
| Gaussian copula | Correlation between HR/R/RBI reflected in joint samples. Independent categories stay independent. |

### Page integration tests

- League Standings: page loads with Yahoo data, shows W-L-T table, category grid renders correctly
- Matchup Planner: week navigator arrows work, past weeks show actual results, future weeks show probabilities
- Fallback paths: both pages degrade gracefully when Yahoo is offline

---

## 7. UI Components Used

All rendering uses existing HEATER UI system:

- `inject_custom_css()` — full glassmorphic stylesheet
- `render_page_layout()` — page title badge + SVG icon
- `render_reco_banner()` — collapsible matchup preview banner
- `render_context_card()` — glassmorphic sidebar cards
- `render_context_columns()` — 1:4 column split
- `render_compact_table()` / `build_compact_table_html()` — ESPN-style tables
- `render_data_freshness_card()` — data freshness widget
- `THEME` dict — all colors from existing palette
- `PAGE_ICONS` — inline SVG icons (no emoji)
- Category probability bars: custom HTML matching existing style conventions

New icon needed in `PAGE_ICONS`: ensure "standings" icon exists (it does — already in the dict).

---

## 8. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Yahoo API rate limiting (24 calls for full schedule) | `_request_with_backoff()` + 24h cache. One-time cost per session. |
| Copula computation slow (10K sims x 12 categories) | NumPy vectorized. Pre-compute Cholesky decomposition once. Target: <500ms. |
| MC simulation slow (1000 sims x 10 remaining weeks) | NumPy vectorized inner loop. Cache in session_state. Target: <2s. |
| Yahoo scoreboard API unreliable for past weeks | Cache permanently once fetched. Fallback: show "Results unavailable" gracefully. |
| `yfpy` parsing bugs on completed weeks | Use `_get_team_week_stats_raw()` REST endpoint as fallback (already exists). |
| Existing 2300 tests must still pass | Old `pages/8_Standings.py` deleted, but `standings_projection.py` and `power_rankings.py` kept intact (engine calls them). No test breakage. |

---

## 9. Files Modified/Created Summary

| File | Action | Description |
|------|--------|-------------|
| `src/standings_engine.py` | CREATE | Shared computation engine |
| `pages/8_League_Standings.py` | CREATE | New League Standings page |
| `pages/8_Standings.py` | DELETE | Replaced by League Standings |
| `pages/11_Matchup_Planner.py` | REWRITE | Add category probabilities + week navigator |
| `src/yahoo_data_service.py` | MODIFY | Add `get_full_league_schedule()`, `get_week_scoreboard()`, capture W-L-T meta |
| `src/database.py` | MODIFY | Add `league_schedule_full` + `league_records` tables + CRUD |
| `src/ui_shared.py` | MODIFY | Add sidebar icon entry if missing for "League Standings" nav label |
| `tests/test_standings_engine.py` | CREATE | Engine unit tests |
| `tests/test_league_standings_page.py` | CREATE | Page integration tests |

# Fantasy Baseball Draft Tool

## Overview

A fantasy baseball draft assistant + in-season manager for a 12-team Yahoo Sports 5x5 roto snake draft league. Two pillars:

1. **Draft Tool** (`app.py`, ~2450 lines) — "Broadcast Booth" dark-theme Streamlit app: 4-step setup wizard (with optional Yahoo OAuth), 3-column draft page with injury badges, percentile ranges, opponent intel tab, and practice mode. Monte Carlo recommendations with percentile sampling.
2. **In-Season Management** (`pages/`) — 5 Streamlit pages: team overview, trade analysis, player comparison, free agent rankings, lineup optimizer. Powered by MLB Stats API + pybaseball + optional Yahoo Fantasy API.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Load sample data (first time / testing)
python load_sample_data.py

# Run the app
streamlit run app.py

# Lint
ruff check .

# Format
ruff format .

# Run all tests (188 collected, 187 pass, 1 skipped for PyMC)
python -m pytest

# Run with verbose output
python -m pytest -v

# Run a single test file
python -m pytest tests/test_in_season.py -v

# Run a specific test
python -m pytest tests/test_in_season.py::test_trade_analyzer -v
```

## League Context

- **League:** FourzynBurn (Yahoo Sports) | **Team:** Team Hickey
- **Format:** 12-team snake draft, 23 rounds
- **Scoring:** 5x5 roto (R, HR, RBI, SB, AVG / W, SV, K, ERA, WHIP)
- **Roster:** C/1B/2B/3B/SS/3OF/2Util/2SP/2RP/4P/5BN = 23 slots
- **Manager skill:** Opponents extremely high-skilled; user is a novice

## Tech Stack

- **Framework:** Streamlit (Python), multi-page app
- **Database:** SQLite (`data/draft_tool.db`) — 14 tables total
- **Core libs:** pandas, NumPy, SciPy, Plotly
- **Analytics:** PyMC 5 (Bayesian), PuLP (LP optimizer), arviz (posterior analysis)
- **Live data:** MLB-StatsAPI, pybaseball (FanGraphs Depth Charts + park factors)
- **Yahoo API:** yfpy + streamlit-oauth (optional OAuth integration)
- **Linter:** ruff (lint + format)
- **CI:** GitHub Actions — ruff lint/format, pytest (Python 3.11-3.13), build check
- **Python:** Local dev uses 3.14; CI tests 3.11-3.13

## File Structure

```
app.py                  — Draft tool: 4-step setup wizard + 3-column draft page
requirements.txt        — pip dependencies (streamlit, pandas, numpy, scipy, plotly, MLB-StatsAPI, pybaseball, pymc, PuLP, yfpy)
load_sample_data.py     — Generates ~190 sample players + injury history for testing
.streamlit/config.toml  — Dark theme configuration
.claude/launch.json     — Dev server config for preview tools
pages/
  1_My_Team.py          — In-season: team overview, roster, category standings
  2_Trade_Analyzer.py   — In-season: trade proposal builder + MC analysis
  3_Player_Compare.py   — In-season: head-to-head player comparison
  4_Free_Agents.py      — In-season: free agent rankings by marginal value
  5_Lineup_Optimizer.py — In-season: PuLP LP solver for start/sit + category targeting
src/
  database.py           — SQLite schema (14 tables), CSV import, projection blending, player pool + in-season queries
  valuation.py          — SGP calculator, replacement levels, VORP, category weights, percentile forecasts
  draft_state.py        — Draft state management, roster tracking, snake pick order, opponent patterns
  simulation.py         — Monte Carlo draft simulation, opponent modeling (history-aware), survival probability
  in_season.py          — Trade analyzer, player comparison engine, FA ranker
  live_stats.py         — MLB Stats API + pybaseball data fetcher, daily refresh logic
  league_manager.py     — League roster/standings management, CSV import for all 12 teams
  ui_shared.py          — Shared THEME dict + inject_custom_css() used by app.py and pages/
  data_2026.py          — Hardcoded 2026 projections (~200 hitters, ~80 pitchers) for sample data
  validation.py         — Validation utilities
  bayesian.py           — Bayesian projection updater: PyMC hierarchical model + Marcel regression fallback
  injury_model.py       — Health scores, age-risk curves, workload flags, injury-adjusted projections
  lineup_optimizer.py   — PuLP LP solver: lineup optimization, category targeting, two-start SP detection
  yahoo_api.py          — Yahoo Fantasy API: OAuth integration via yfpy, league sync, roster import
  data_pipeline.py      — FanGraphs auto-fetch: Steamer/ZiPS/Depth Charts JSON API, normalize, upsert, ADP extraction
tests/
  test_database_schema.py   — DB schema and table existence tests
  test_database_queries.py  — Query function tests
  test_in_season.py         — Trade analyzer, comparison, FA ranker tests
  test_league_manager.py    — League roster/standings management tests
  test_live_stats.py        — Live stats pipeline tests
  test_bayesian.py          — Bayesian updater: regression, age curves, stabilization, batch update (31 tests)
  test_injury_model.py      — Injury model: health scores, age risk, workload flags (17 tests)
  test_lineup_optimizer.py  — Lineup optimizer: LP solver, constraints, category targeting (20 tests)
  test_yahoo_api.py         — Yahoo API: OAuth, oob flow, sync, mock endpoints (40 tests)
  test_percentiles.py       — Percentile forecasts: volatility, P10/P50/P90 bounds (7 tests)
  test_opponent_model.py    — Enhanced opponent modeling: preferences, needs, history (8 tests)
  test_percentile_sampling.py — Percentile sampling passthrough in evaluate_candidates (4 tests)
  test_data_pipeline.py     — FanGraphs auto-fetch: normalization, fetch, storage, ADP, orchestration (28 tests)
  test_integration.py       — End-to-end pipeline: injury → Bayesian → percentiles → valuation (11 tests)
  profile_latency.py        — Performance profiling utility
data/
  draft_tool.db         — SQLite database (created at runtime)
  backups/              — Draft state JSON backups
docs/plans/             — Implementation plan archives
.github/
  workflows/ci.yml      — CI pipeline (lint + test + build)
  dependabot.yml        — Weekly pip + GitHub Actions updates
```

## Architecture

### Draft Valuation Pipeline
1. **Projection blending** — Weighted average of multiple systems (Steamer, ZiPS, etc.)
2. **SGP** — Converts raw stats to standings-point movement; auto-computed denominators
3. **VORP** — Value Over Replacement Player with multi-position flexibility premium (+0.12/extra position, +0.08/scarce position)
4. **Category weights** — Dynamic: standings-based when available, draft-progress-aware scaling
5. **pick_score** = weighted SGP + positional scarcity + VORP bonus

### Draft Engine
- **Monte Carlo simulation** (100-300 sims, 6-round horizon) with opponent roster tracking
- **Survival probability** — Normal CDF with positional scarcity adjustment
- **Urgency** = (1 - P_survive) * positional_dropoff (draft-position-aware)
- **Combined score** = MC mean SGP + urgency * 0.4
- **Tier assignment** via natural breaks algorithm
- Dynamic replacement levels recalculated as players are drafted
- Projection confidence discount (low PA/IP get up to 20% discount)
- Bench slot optimization (late-draft bonus for multi-position flexibility)
- **Percentile sampling** — MC sims sample from P10-P90 distributions when multiple projection systems available
- **Injury badges** — Hero card shows 🟢/🟡/🔴 health icons, age flags for aging curves, workload flags for IP spikes
- **P10/P90 range bars** — Floor/ceiling projections displayed on hero pick and alternatives
- **Opponent intel tab** — Threat alerts when opponents need your target position, plus full opponent roster/needs breakdown
- **Practice mode** — Ephemeral DraftState clone for what-if scenarios, resets on refresh or button click
- **Yahoo OAuth** — Setup wizard Step 1 shows Yahoo connect card when env vars are set. Uses out-of-band (oob) flow: user clicks link, authorizes on Yahoo, pastes verification code back in the app.

### Auto-Fetch Pipeline
- **FanGraphs JSON API** — Fetches Steamer, ZiPS, Depth Charts projections (3 systems × bat/pit = 6 endpoints) on app startup
- **Normalization** — Maps FG JSON fields to DB schema; SP/RP classification mirrors CSV import logic (GS≥5→SP, SV≥3→RP)
- **Staleness check** — `refresh_if_stale()` skips fetch if projections table already has data; `force=True` overrides
- **ADP extraction** — Pulls ADP from Steamer JSON responses, resolves names→player_ids with fuzzy fallback
- **Graceful degradation** — Partial system failures proceed with available data; total failure falls back to CSV upload
- **Rate limiting** — 0.5s between requests, User-Agent header to avoid CloudFlare blocks

### In-Season Algorithms
- **Trade Analyzer:** Roster swap → projected season totals (YTD + ROS) → park-adjusted SGP delta → MC simulation (200 sims) → verdict with confidence %. Now includes injury badges for both sides and P10/P90 risk assessment.
- **Player Compare:** ROS projections → Z-score normalization across 10 categories → composite weighted score, optional marginal SGP team impact. Now includes health badges and projection confidence (P10-P90 range width).
- **FA Ranker:** Marginal SGP per FA vs. user's roster → category-need weighting → replacement target identification → sort by net marginal value
- **Live Stats:** MLB Stats API (daily auto-refresh) + pybaseball ROS projections, staleness tracking via `refresh_log` table
- **Lineup Optimizer:** PuLP LP solver with binary assignment per player/slot, category targeting based on standings gaps, two-start SP detection. Now includes health-adjusted SGP penalties and auto-detected two-start SP highlights.
- **My Team:** Roster display with injury badges, Bayesian projection indicators, Yahoo sync button (when connected)

### Advanced Analytics (Plan 3)
- **Bayesian Updater:** PyMC 5 hierarchical beta-binomial model for in-season stat updates; Marcel regression fallback when PyMC unavailable. Uses FanGraphs stabilization thresholds (K=60 PA, AVG=910 AB, ERA=70 IP). Aging curves on logit scale.
- **Injury Model:** `health_score = avg(GP/games_available)` over 3 seasons. Age-risk curves: hitters +2%/yr after 30, pitchers +3%/yr after 28. Workload flag for >40 IP increase. Counting stats scaled by health score.
- **Percentile Forecasts:** Inter-projection volatility (StdDev across Steamer/ZiPS/Depth Charts). P10/P50/P90 using ±1.28σ. Process risk widening for low-correlation stats (AVG r²=0.41 vs HR r²=0.72).
- **Enhanced Opponent Model:** `P(pick) = 0.5*ADP + 0.3*team_need + 0.2*historical_preference`. Computes per-team positional bias from draft history. Falls back to ADP-only when no history available.
- **Yahoo API:** OAuth 2.0 via yfpy v17+, out-of-band (oob) flow. User clicks link → authorizes on Yahoo → pastes verification code. League settings, rosters, standings, FA pool, draft results sync. Graceful degradation when not connected.

### Database Tables (14)

| Group | Tables |
|-------|--------|
| Draft | `projections`, `adp`, `league_config`, `draft_picks`, `blended_projections`, `player_pool` |
| In-Season | `season_stats`, `ros_projections`, `league_rosters`, `league_standings`, `park_factors`, `refresh_log` |
| Plan 3 | `injury_history`, `transactions` |

## Data Sources

- **Draft projections:** Auto-fetched from FanGraphs JSON API (Steamer, ZiPS, Depth Charts) on startup; CSV upload as manual fallback
- **ADP:** Extracted from FanGraphs Steamer JSON (filters ADP ≥ 999 and nulls); FantasyPros consensus as fallback
- **Current stats:** MLB Stats API (`MLB-StatsAPI` package, auto-refreshed daily)
- **ROS projections:** FanGraphs Depth Charts (`pybaseball`, on-demand or daily)
- **Park factors:** pybaseball / FanGraphs
- **League rosters/standings:** Manual CSV upload via League Import wizard step

## Key API Signatures

These are commonly called wrong — always double-check:

```python
# STANDALONE functions in src/valuation.py — NOT methods on SGPCalculator
compute_replacement_levels(pool, config, sgp_calc)   # valuation.py:209
compute_sgp_denominators(pool, config)                # valuation.py:446

# Keyword args required for optional params
value_all_players(pool, config, roster_totals=None, category_weights=None,
                  replacement_levels=None, current_round=None, num_rounds=23)

# load_player_pool() columns:
# player_id, name, team, positions, is_hitter, is_injured,
# pa, ab, h, r, hr, rbi, sb, avg, ip, w, sv, k, era, whip, er, bb_allowed, h_allowed, adp

# --- Plan 3 new APIs ---

# Bayesian updater (src/bayesian.py)
BayesianUpdater(prior_weight=0.6)
updater.regressed_rate(observed_rate, sample_size, league_mean, stabilization_point)
updater.batch_update_projections(season_stats_df, preseason_df, config=None)
updater.age_adjustment(age, stat)  # returns float multiplier

# Injury model (src/injury_model.py)
compute_health_score(games_played_3yr: list, games_available_3yr: list) -> float
apply_injury_adjustment(projections_df, health_scores_df) -> pd.DataFrame
get_injury_badge(health_score) -> tuple[str, str]  # (icon, label)

# Percentiles (src/valuation.py)
compute_projection_volatility(projections_by_system: dict[str, DataFrame]) -> DataFrame
add_process_risk(volatility_df) -> DataFrame  # widens CI for low-correlation stats
compute_percentile_projections(base_df, volatility_df, percentiles=[10,50,90]) -> dict[int, DataFrame]

# Opponent model (src/simulation.py, src/draft_state.py)
compute_team_preferences(draft_history_df)  # needs columns: team_key, positions, round
get_team_draft_patterns(draft_state_dict, team_id: int)  # team_id is 0-based index
get_positional_needs(draft_state_dict, team_id: int, roster_config: dict)

# Lineup optimizer (src/lineup_optimizer.py)
LineupOptimizer(roster_df, config_dict)
optimizer.optimize_lineup()  # returns {slot: player_name, projected_stats: {...}}
optimizer.category_targeting(standings_df, team_name)  # returns {cat: weight}

# Data pipeline (src/data_pipeline.py)
# Yahoo OAuth helpers (src/yahoo_api.py)
build_oauth_url(consumer_key, redirect_uri="oob")  # returns auth URL string
exchange_code_for_token(consumer_key, consumer_secret, code)  # returns token dict or None

refresh_if_stale(force=False)  # returns bool; auto-skips if projections table has data
fetch_projections(system, stats)  # returns (DataFrame, raw_json_list)
SYSTEM_MAP = {"steamer": "steamer", "zips": "zips", "fangraphsdc": "depthcharts"}
```

## Gotchas

- **DB column `name` vs UI `player_name`** — Normalized via `rename(columns={"name": "player_name"})` in `_build_player_pool()`. Always check which context you're in.
- **Sample data is `system='blended'` only** — Do NOT call `create_blended_projections()` when only blended data exists; it deletes `system='blended'` rows first.
- **Python 3.14 + SQLite bytes** — Returns bytes for some integer columns. Fixed with explicit `CAST` in SQL and `pd.to_numeric()` coercion.
- **Plotly 6.x hex colors** — Does NOT accept 8-digit hex (`#RRGGBBAA`). Use `rgba(r,g,b,a)` format.
- **Snake draft order** — `round % 2 == 0` forward, `round % 2 == 1` reverse.
- **MC horizon limit** — 6-round horizon for performance (4.45s vs 10.42s without).
- **Streamlit pages outside runtime** — Pages use `if/else` guard pattern around `st.stop()` to prevent crashes when imported outside Streamlit.
- **`datetime.now(UTC)`** — Used everywhere; `datetime.utcnow()` is deprecated.
- **`width="stretch"`** — Replaces deprecated `use_container_width=True` in Streamlit.
- **PyMC/PuLP optional deps** — `PYMC_AVAILABLE` and `PULP_AVAILABLE` flags in bayesian.py and lineup_optimizer.py. Always use `try/except ImportError` pattern. CI skips tests requiring these with `@pytest.mark.skipif`.
- **`compute_team_preferences()` column names** — Expects `team_key` and `positions`, NOT `team_id`/`position`. Integration tests caught this.
- **`get_team_draft_patterns()` uses int team_id** — `team_id` is 0-based team index, not team name string. Filters on `pick.get("team_index")`.
- **health_score range** — Always 0.0 to 1.0. Missing seasons default to 0.85 (league average). Empty history returns 0.85.
- **Yahoo API graceful degradation** — All Yahoo features wrapped in `YFPY_AVAILABLE` checks. App works fully without Yahoo credentials. Setup wizard Yahoo connect requires `YAHOO_CLIENT_ID` and `YAHOO_CLIENT_SECRET` env vars.
- **Yahoo OAuth uses oob (out-of-band)** — Yahoo Fantasy API requires `redirect_uri=oob`. Do NOT use `http://localhost:8501/` or any redirect-based flow — Yahoo rejects all non-oob redirect URIs for fantasy apps. The `yahoo-oauth` library hardcodes `CALLBACK_URI='oob'`. Our Streamlit UI mirrors this: user clicks a link, authorizes on Yahoo, pastes the verification code back.
- **yfpy `--no-deps` install needs extras** — Installing `yfpy>=17.0 --no-deps` strips `stringcase`, `yahoo-oauth`, and their transitive deps. Must also `pip install stringcase yahoo-oauth==2.1.1`. Similarly, `streamlit-oauth --no-deps` needs `httpx-oauth`. The python-dotenv version mismatch (1.2.2 vs pinned 1.1.1) is a pip warning but works at runtime.
- **In-season page warnings** — All pages that require league data use the standardized message: "No league data loaded. Import your league rosters in the main app (Setup Step 3)."
- **FanGraphs API SYSTEM_MAP** — FG API uses `"fangraphsdc"` for Depth Charts, but DB stores as `"depthcharts"`. Always use `SYSTEM_MAP` dict, never hardcode system names. `SYSTEMS = list(SYSTEM_MAP.keys())` derives the list.
- **FanGraphs `minpos` field** — Returns primary position only (e.g., "SS"), not multi-position eligibility. Sometimes returns `"0"` or `"-"` meaning no position; these are guarded to fall back to `"Util"`.
- **Auto-fetch session state** — `st.session_state.projections_fetched` gates the fetch-once logic. Delete the key to allow retry. `HAS_DATA_PIPELINE` flag handles import failure gracefully.
- **sgp_volatility alignment** — When `evaluate_candidates()` receives `sgp_volatility`, the array is aligned to the full pool. After filtering drafted players, it must be re-indexed via `pd.Series.reindex()` to match the filtered `available` pool.
- **Practice mode isolation** — Uses separate `st.session_state["practice_draft_state"]` DraftState, never persisted to disk/DB. Resets on page refresh or "Reset Practice" button click.
- **Percentile pipeline ordering** — `compute_projection_volatility()` → `add_process_risk()` → `compute_percentile_projections()`. Skip entirely when only one projection system exists (zero variance). All 3 consumers (app.py, Trade Analyzer, Player Compare) must follow this exact ordering.
- **Connection leak pattern in pages** — Always wrap `get_connection()` + queries in `try/finally` with `conn.close()` in the `finally` block. Do NOT put `conn.close()` inline after queries — if a query throws, the connection leaks.
- **Scarcity toast dedup** — `st.toast()` fires on every Streamlit rerender. Use `st.session_state` keys to deduplicate (e.g., `f"scarcity_toast_{pos}_{count}"`).

## GitHub

- **Repo:** https://github.com/hicklax13/fantasy-baseball-draft-tool (public)
- **Current release:** v1.0.0
- **Release workflow:** Auto-creates GitHub releases on `v*.*.*` tags

## Testing Status

- **Unit tests:** 188 collected, 187 passed, 1 skipped (PyMC optional dep)
- **Test files:** 14 test files across draft engine, in-season, analytics, data pipeline, and integration
- **CI:** GitHub Actions runs ruff lint/format + pytest on Python 3.11, 3.12, 3.13
- **Coverage:** 64% (below 75% CI threshold; pre-existing, no regressions)
- **Systematic code reviews:** Two rounds of full codebase review completed; all bugs fixed and pushed

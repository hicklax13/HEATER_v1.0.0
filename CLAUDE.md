# Fantasy Baseball Draft Tool

## Overview

A fantasy baseball draft assistant + in-season manager for a 12-team Yahoo Sports 5x5 roto snake draft league. Two pillars:

1. **Draft Tool** (`app.py`, ~1600 lines) — "Broadcast Booth" themed Streamlit app with dark/light mode toggle: splash screen data bootstrap + 2-step setup wizard (Settings, Launch), 3-column draft page with SVG injury badges, percentile ranges, opponent intel tab, and practice mode. Monte Carlo recommendations with percentile sampling. Zero CSV uploads — all data auto-fetched from MLB Stats API + FanGraphs on every launch.
2. **In-Season Management** (`pages/`) — 6 Streamlit pages: team overview, mock draft simulator, trade analysis, player comparison, free agent rankings, lineup optimizer. Powered by MLB Stats API + pybaseball + optional Yahoo Fantasy API. All pages share centralized theme system with dark/light toggle.

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

# Run all tests (330 collected, 329 pass, 1 skipped for PyMC)
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
app.py                  — Draft tool: splash screen bootstrap + 2-step setup wizard + 3-column draft page
requirements.txt        — pip dependencies (streamlit, pandas, numpy, scipy, plotly, MLB-StatsAPI, pybaseball, pymc, PuLP, yfpy)
load_sample_data.py     — Generates ~190 sample players + injury history for testing
.streamlit/config.toml  — Dark theme configuration
.claude/launch.json     — Dev server config for preview tools
pages/
  1_My_Team.py          — In-season: team overview, roster, category standings
  2_Mock_Draft.py       — Standalone mock draft simulator (AI opponents, MC recommendations, draft board)
  3_Trade_Analyzer.py   — In-season: trade proposal builder + MC analysis
  4_Player_Compare.py   — In-season: head-to-head player comparison
  5_Free_Agents.py      — In-season: free agent rankings by marginal value
  6_Lineup_Optimizer.py — In-season: PuLP LP solver for start/sit + category targeting
src/
  database.py           — SQLite schema (14 tables), CSV import, projection blending, player pool + in-season queries
  valuation.py          — SGP calculator, replacement levels, VORP, category weights, percentile forecasts
  draft_state.py        — Draft state management, roster tracking, snake pick order, opponent patterns
  simulation.py         — Monte Carlo draft simulation, opponent modeling (history-aware), survival probability
  in_season.py          — Trade analyzer, player comparison engine, FA ranker
  live_stats.py         — MLB Stats API data fetcher: roster, season/historical stats, injury data extraction
  data_bootstrap.py     — Zero-interaction bootstrap orchestrator: staleness-based refresh of all data sources on app launch
  league_manager.py     — League roster/standings management, CSV import for all 12 teams
  ui_shared.py          — Centralized theme system (dark/light), PAGE_ICONS (inline SVGs), inject_custom_css(), METRIC_TOOLTIPS
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
  test_data_bootstrap.py    — Bootstrap pipeline: staleness, bulk upserts, orchestrator (30 tests)
  test_bayesian.py          — Bayesian updater: regression, age curves, stabilization, batch update (31 tests)
  test_injury_model.py      — Injury model: health scores, age risk, workload flags (17 tests)
  test_lineup_optimizer.py  — Lineup optimizer: LP solver, constraints, category targeting (20 tests)
  test_yahoo_api.py         — Yahoo API: OAuth, oob flow, sync, mock endpoints (40 tests)
  test_percentiles.py       — Percentile forecasts: volatility, P10/P50/P90 bounds (7 tests)
  test_opponent_model.py    — Enhanced opponent modeling: preferences, needs, history (8 tests)
  test_percentile_sampling.py — Percentile sampling passthrough in evaluate_candidates (4 tests)
  test_data_pipeline.py     — FanGraphs auto-fetch: normalization, fetch, storage, ADP, orchestration (28 tests)
  test_integration.py       — End-to-end pipeline: injury → Bayesian → percentiles → valuation (11 tests)
  test_valuation_math.py    — Math verification: SGP, VORP, replacement levels, percentiles, process risk (40 tests)
  test_simulation_math.py   — Math verification: survival probability, urgency, combined score, tiers, MC convergence (37 tests)
  test_trade_math.py        — Math verification: trade SGP delta, MC noise, verdict, z-scores, rate stats (35 tests)
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
- **Injury badges** — Hero card shows CSS dot health indicators (green/yellow/red), age flags for aging curves, workload flags for IP spikes. No emoji — all icons are inline SVGs from `PAGE_ICONS` dict.
- **P10/P90 range bars** — Floor/ceiling projections displayed on hero pick and alternatives
- **Opponent intel tab** — Threat alerts when opponents need your target position, plus full opponent roster/needs breakdown
- **Practice mode** — Ephemeral DraftState clone for what-if scenarios, resets on refresh or button click
- **Yahoo OAuth** — Setup wizard Settings step shows Yahoo connect expander when env vars are set. Uses out-of-band (oob) flow: user clicks link, authorizes on Yahoo, pastes verification code back in the app.

### Bootstrap Pipeline (`src/data_bootstrap.py`)
- **Splash screen** — Every app launch shows progress bar while `bootstrap_all_data()` runs all 7 data phases
- **Staleness-based refresh** — Each data source has its own max-age threshold (`StalenessConfig`): 1h live stats, 6h Yahoo, 7d players/projections, 30d historical/park factors
- **Phase ordering** — Players first (other phases need player_ids), then park factors, projections, live stats, historical, injury data, Yahoo sync
- **Zero interaction** — No CSV uploads required; all data fetched from free APIs automatically

### Auto-Fetch Pipeline (`src/data_pipeline.py`)
- **FanGraphs JSON API** — Fetches Steamer, ZiPS, Depth Charts projections (3 systems × bat/pit = 6 endpoints) on app startup
- **Normalization** — Maps FG JSON fields to DB schema; SP/RP classification mirrors CSV import logic (GS≥5→SP, SV≥3→RP)
- **Staleness check** — `refresh_if_stale()` uses `check_staleness("fangraphs_projections", 168)` to skip if fresh; `force=True` overrides
- **ADP extraction** — Pulls ADP from Steamer JSON responses, resolves names→player_ids with fuzzy fallback
- **Graceful degradation** — Partial system failures proceed with available data
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

- **Players:** Auto-fetched from MLB Stats API on every launch (750+ active players); staleness: 7 days
- **Draft projections:** Auto-fetched from FanGraphs JSON API (Steamer, ZiPS, Depth Charts) on startup; staleness: 7 days
- **ADP:** Extracted from FanGraphs Steamer JSON (filters ADP ≥ 999 and nulls); FantasyPros consensus as fallback
- **Current stats:** MLB Stats API (`MLB-StatsAPI` package, auto-refreshed on launch); staleness: 1 hour
- **Historical stats:** 3 years (2023-2025) from MLB Stats API for injury modeling; staleness: 30 days
- **Park factors:** Hardcoded FanGraphs 2024 values (30 teams) in `data_bootstrap.py`; staleness: 30 days
- **Injury history:** Derived from historical stats (games played vs available); staleness: 30 days
- **League rosters/standings:** Yahoo Fantasy API sync (optional, auto-syncs when connected); staleness: 6 hours

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
get_injury_badge(health_score) -> tuple[str, str]  # (css_dot_html, label) — returns <span> with colored dot, NOT emoji

# UI shared (src/ui_shared.py)
PAGE_ICONS: dict[str, str]  # ~20 inline SVG icons keyed by name ("baseball", "fire", "accept", etc.)
METRIC_TOOLTIPS: dict[str, str]  # Educational tooltip text for every metric (sgp, vorp, survival, etc.)
DARK_THEME: dict  # bg=#0a0e1a, ink=#0a0e1a, card=#1a1f2e, amber=#f59e0b, ...
LIGHT_THEME: dict  # bg=#f8f9fc, ink=#1a1a2e, card=#ffffff, amber=#f59e0b, ...
T = _ThemeProxy()  # Dict-like proxy — reads delegate to active theme via get_theme()
get_theme() -> dict  # Returns DARK_THEME or LIGHT_THEME based on session_state["theme_mode"]
render_theme_toggle()  # Renders dark/light toggle in sidebar
inject_custom_css()  # Injects full CSS (1000+ lines) — call once per page

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

# Data bootstrap (src/data_bootstrap.py)
bootstrap_all_data(yahoo_client=None, on_progress=None, force=False, staleness=None)  # returns dict[str, str]
StalenessConfig(players_hours=168, live_stats_hours=1, projections_hours=168, historical_hours=720, park_factors_hours=720, yahoo_hours=6)
BootstrapProgress(phase="", detail="", pct=0.0)  # callback dataclass for splash screen
PARK_FACTORS: dict[str, float]  # 30-team dict, e.g. {"COL": 1.38, "MIA": 0.88}

# Database bulk helpers (src/database.py)
check_staleness(source, max_age_hours)  # returns bool — True if needs refresh
upsert_player_bulk(players: list[dict])  # dict needs: name, team, positions, is_hitter; optional: mlb_id
upsert_injury_history_bulk(records: list[dict])  # dict needs: player_id, season, games_played, games_available
upsert_park_factors(factors: list[dict])  # dict needs: team_code, factor_hitting, factor_pitching

# Live stats extensions (src/live_stats.py)
fetch_all_mlb_players(season=2026)  # returns DataFrame with mlb_id, name, team, positions, is_hitter
fetch_historical_stats(seasons=[2023,2024,2025])  # returns {year: DataFrame}
fetch_injury_data_bulk(historical_stats)  # returns list[dict] with player_name, team, season, games_played, games_available

# Data pipeline (src/data_pipeline.py)
# Yahoo OAuth helpers (src/yahoo_api.py)
build_oauth_url(consumer_key, redirect_uri="oob")  # returns auth URL string
exchange_code_for_token(consumer_key, consumer_secret, code)  # returns token dict or None

refresh_if_stale(force=False)  # returns bool; uses check_staleness() with 168h threshold
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
- **In-season page warnings** — All pages that require league data use the standardized message: "No league data loaded. Connect your Yahoo league in Settings, or league data will load automatically on next app launch."
- **FanGraphs API SYSTEM_MAP** — FG API uses `"fangraphsdc"` for Depth Charts, but DB stores as `"depthcharts"`. Always use `SYSTEM_MAP` dict, never hardcode system names. `SYSTEMS = list(SYSTEM_MAP.keys())` derives the list.
- **FanGraphs `minpos` field** — Returns primary position only (e.g., "SS"), not multi-position eligibility. Sometimes returns `"0"` or `"-"` meaning no position; these are guarded to fall back to `"Util"`.
- **Bootstrap session state** — `st.session_state.bootstrap_complete` gates the bootstrap-once logic. Delete the key to force re-bootstrap. Results stored in `st.session_state.bootstrap_results`.
- **Bootstrap lazy imports** — `data_bootstrap.py` imports from `database.py` and `live_stats.py` inside functions to avoid circular imports and enable graceful degradation.
- **Players table has no UNIQUE constraint on name** — `upsert_player_bulk()` uses SELECT-first pattern (check if name exists, then INSERT or UPDATE). Do NOT use `ON CONFLICT(name)`. However, `injury_history` DOES have a UNIQUE index on `(player_id, season)` so `ON CONFLICT` is safe there.
- **Park factors schema** — Uses columns `team_code`, `factor_hitting`, `factor_pitching` (not just `team` and `park_factor`). `upsert_park_factors()` accepts both naming conventions via flexible getter.
- **sgp_volatility alignment** — When `evaluate_candidates()` receives `sgp_volatility`, the array is aligned to the full pool. After filtering drafted players, it must be re-indexed via `pd.Series.reindex()` to match the filtered `available` pool.
- **Practice mode isolation** — Uses separate `st.session_state["practice_draft_state"]` DraftState, never persisted to disk/DB. Resets on page refresh or "Reset Practice" button click.
- **Mock Draft session state** — All keys prefixed `mock_` (`mock_ds`, `mock_pool`, `mock_lc`, `mock_sgp`, `mock_started`, `mock_draft_pos`, `mock_num_sims`). Ephemeral — never saved to DB/disk. Resets on "Reset Draft" button click.
- **`evaluate_candidates()` returns `name`, not `player_name`** — The result DataFrame uses `"name"` from the pool. Always alias to `"player_name"` after calling: `recs["player_name"] = recs["name"]`.
- **Percentile pipeline ordering** — `compute_projection_volatility()` → `add_process_risk()` → `compute_percentile_projections()`. Skip entirely when only one projection system exists (zero variance). All 3 consumers (app.py, Trade Analyzer, Player Compare) must follow this exact ordering.
- **Connection leak pattern in pages** — Always wrap `get_connection()` + queries in `try/finally` with `conn.close()` in the `finally` block. Do NOT put `conn.close()` inline after queries — if a query throws, the connection leaks.
- **Scarcity toast dedup** — `st.toast()` fires on every Streamlit rerender. Use `st.session_state` keys to deduplicate (e.g., `f"scarcity_toast_{pos}_{count}"`).
- **`T["ink"]` vs `T["bg"]` for text-on-accent** — `T["bg"]` is for page backgrounds. `T["ink"]` is for dark text on colored surfaces (amber buttons, badges, tabs). Never use `T["bg"]` as a text color — it's invisible on amber in light mode.
- **No emoji in the codebase** — All icons are inline SVGs from `PAGE_ICONS` dict in `ui_shared.py`. Injury badges use CSS dots (`border-radius:50%`), not emoji. Do NOT re-introduce emoji.
- **No abbreviations in UI text** — All user-facing text uses full terms: "Standings Gained Points" (not SGP), "Value Over Replacement Player" (not VORP), "Average Draft Position" (not ADP), "Monte Carlo" (not MC), "Runs Batted In" (not RBI), etc. Variable names and DB columns stay abbreviated. Category display uses `cat_names`/`cat_display_names` mapping dicts.
- **Draft Settings live on Mock Draft page** — Number of teams, Rounds, and Draft position inputs are on `pages/2_Mock_Draft.py`, not on the Configurations page. Stored in `mock_num_teams`, `mock_num_rounds`, `mock_draft_pos` session state. Also written to shared `num_teams`/`num_rounds`/`draft_pos` keys for the main draft.
- **Yahoo sync across pages** — Pages reuse `st.session_state.yahoo_client` for Yahoo sync instead of creating new clients. Token data stored in `st.session_state.yahoo_token_data` for re-authentication. Pages show "Sync League Data Now" button when Yahoo connected but DB empty.
- **`_ThemeProxy` dict subclass** — `T` in `ui_shared.py` is a `_ThemeProxy` that delegates all reads to `get_theme()`. This makes `T["amber"]` theme-aware without changing call sites. Do not replace `T` with a plain dict.
- **Sidebar nav rename via CSS** — The sidebar "app" label is renamed to "Configurations" using `font-size:0` + `::after { content: "Configurations" }` pseudo-element trick in `inject_custom_css()`.
- **Rate-stat aggregation** — AVG=sum(h)/sum(ab), ERA=sum(er)*9/sum(ip), WHIP=sum(bb+h)/sum(ip). Weighted averages, NOT simple averages. `_fix_rate_stats()` in `lineup_optimizer.py` recalculates these after LP solves.
- **Injury model scales rate stats** — `apply_injury_adjustment()` scales ER, BB_allowed, H_allowed by `_combined_factor` (health×age×workload), not just counting stats. Without this, injured pitchers show artificially low ERA/WHIP.
- **LP inverse stat weighting** — ERA/WHIP in lineup optimizer LP objective must be weighted by IP: `player_value -= val * ip * weight`. Without IP weighting, a 1-IP reliever with 0.00 ERA dominates a 200-IP starter.
- **`compare_players()` peer-group filtering** — Z-scores computed against `is_hitter`-filtered pool only. Hitter HR z-score uses hitter pool mean/std, not full pool (which includes pitchers with HR=0).
- **`check_staleness()` edge case** — `max_age_hours <= 0` returns `True` (always stale). Prevents division-by-zero and logical errors.
- **`ruff` command on Windows** — Use `python -m ruff check .` and `python -m ruff format .` instead of bare `ruff` — the binary may not be on PATH.

## GitHub

- **Repo:** https://github.com/hicklax13/fantasy-baseball-draft-tool (public)
- **Current release:** v1.0.0
- **Release workflow:** Auto-creates GitHub releases on `v*.*.*` tags

## Testing Status

- **Unit tests:** 330 collected, 329 passed, 1 skipped (PyMC optional dep)
- **Test files:** 18 test files across draft engine, in-season, analytics, data pipeline, bootstrap, integration, and math verification
- **Math verification suite:** 112 tests across 3 files (valuation, simulation, trade) — hand-calculated expected values verified against code formulas
- **CI:** GitHub Actions runs ruff lint/format + pytest on Python 3.11, 3.12, 3.13
- **Coverage:** 64% (below 75% CI threshold; pre-existing, no regressions)
- **Systematic code reviews:** Three rounds of full codebase review completed (including parallel 7-agent sweep); all bugs fixed and pushed

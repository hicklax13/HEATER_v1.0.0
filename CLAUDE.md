# HEATER — Fantasy Baseball Draft Tool

## Overview

A fantasy baseball draft assistant + in-season manager for a 12-team Yahoo Sports H2H Categories snake draft league.

1. **Draft Tool** (`app.py`) — "Heater" themed Streamlit app with glassmorphic design, splash screen bootstrap, 2-step setup wizard, 3-column draft page, Monte Carlo recommendations with percentile sampling. Zero CSV uploads — all data auto-fetched.
2. **In-Season Management** (`pages/`) — 12 pages: team overview, draft simulator, trade analysis, player comparison, free agents, lineup optimizer, closer monitor, standings/power rankings, leaders/prospects, waiver wire, start/sit advisor, matchup planner.
3. **Trade Analyzer Engine** (`src/engine/`) — 6-phase pipeline: deterministic SGP → stochastic MC → signal intelligence → contextual adjustments → game theory → production convergence/caching.
4. **Enhanced Lineup Optimizer** (`src/optimizer/`) — 10-module pipeline: enhanced projections, matchup adjustments, H2H weights, non-linear SGP, streaming, scenarios, dual objective, advanced LP.
5. **Draft Recommendation Engine** (`src/draft_engine.py`) — 8-stage enhancement chain with 3 execution modes (Quick/Standard/Full).
6. **In-Season Analytics** (`src/`) — Trade value chart, two-start planner, start/sit advisor, matchup planner, waiver wire, trade finder, draft grader, prospect rankings, ECR consensus, player news.

## Commands

```bash
pip install -r requirements.txt        # Install deps
python load_sample_data.py             # Load sample data (first time/testing)
streamlit run app.py                   # Run the app
ruff check .                           # Lint
ruff format .                          # Format
python -m pytest                       # Run all tests (1956 pass, 4 skipped)
python -m pytest tests/test_foo.py -v  # Run single test file
```

## League Context

- **League:** FourzynBurn (Yahoo Sports) | **Team:** Team Hickey
- **Format:** 12-team snake draft, 23 rounds, Head-to-Head Categories
- **Hitting cats (6):** R, HR, RBI, SB, AVG, OBP
- **Pitching cats (6):** W, L, SV, K, ERA, WHIP
- **Inverse cats:** L, ERA, WHIP (lower is better)
- **Rate stats:** AVG, OBP, ERA, WHIP
- **Roster:** C/1B/2B/3B/SS/3OF/2Util/2SP/2RP/4P/5BN = 23 slots
- **Manager skill:** Opponents extremely high-skilled; user is a novice

## Tech Stack

- **Framework:** Streamlit (Python), multi-page app
- **Database:** SQLite (`data/draft_tool.db`) — 21 tables
- **Core libs:** pandas, NumPy, SciPy, Plotly
- **Analytics:** PyMC 5 (Bayesian), PuLP (LP optimizer), arviz
- **Live data:** MLB-StatsAPI, pybaseball (FanGraphs Depth Charts + park factors)
- **Yahoo API:** yfpy + streamlit-oauth (optional OAuth)
- **Linter:** ruff (lint + format)
- **CI:** GitHub Actions — ruff lint/format, pytest (Python 3.11-3.13)
- **Python:** Local dev uses 3.14; CI tests 3.11-3.13

## File Structure

```
app.py                  — Draft tool: splash screen bootstrap + setup wizard + draft page
requirements.txt        — pip dependencies
load_sample_data.py     — Generates ~190 sample players + injury history for testing
.streamlit/config.toml  — Light theme configuration (Heater palette)
pages/
  1_My_Team.py          — Team overview, roster, category standings, Yahoo sync
  2_Draft_Simulator.py  — Standalone draft simulator with AI opponents, MC recommendations
  3_Trade_Analyzer.py   — Trade proposal builder + Phase 1 SGP engine with legacy fallback
  4_Player_Compare.py   — Head-to-head player comparison
  5_Free_Agents.py      — Free agent rankings by marginal value
  6_Lineup_Optimizer.py — 5-tab lineup optimizer powered by LineupOptimizerPipeline
  7_Closer_Monitor.py   — 30-team closer depth chart grid
  8_Standings.py        — Projected standings + power rankings
  9_Leaders.py          — Category leaders, points leaders, breakout detection, prospects
  10_Waiver_Wire.py     — Add/drop recommendations based on roster gaps (powered by src/waiver_wire.py)
  11_Start_Sit.py       — Weekly start/sit advisor with matchup analysis (powered by src/start_sit.py)
  12_Matchup_Planner.py — Per-game matchup ratings with color-coded quality tiers (powered by src/matchup_planner.py)
src/
  database.py           — SQLite schema (21 tables), player pool + in-season queries
  valuation.py          — SGP calculator, replacement levels, VORP, category weights, percentiles
  draft_state.py        — Draft state management, roster tracking, snake pick order
  simulation.py         — Monte Carlo draft simulation, opponent modeling, survival probability
  in_season.py          — Legacy trade analyzer, player comparison, FA ranker
  live_stats.py         — MLB Stats API data fetcher
  data_bootstrap.py     — Zero-interaction bootstrap orchestrator (staleness-based refresh)
  league_manager.py     — League roster/standings management
  ui_shared.py          — THEME dict, PAGE_ICONS (inline SVGs), glassmorphic CSS, sidebar branding, 3-zone layout system, player card dialog
  player_card.py        — Player card data assembly (pure function, no Streamlit dependency)
  data_2026.py          — Hardcoded 2026 projections for sample data
  bayesian.py           — PyMC hierarchical model + Marcel regression fallback
  injury_model.py       — Health scores, age-risk curves, injury-adjusted projections
  lineup_optimizer.py   — PuLP LP solver for lineup optimization
  yahoo_api.py          — Yahoo Fantasy API OAuth integration via yfpy
  yahoo_data_service.py — Live-first data layer: 3-tier cache (session_state → Yahoo → SQLite)
  trade_intelligence.py — Trade valuation layer: health, scarcity, FA gating, Trade Readiness
  trade_signals.py      — Kalman + regime trend adjustment for trade valuations
  data_pipeline.py      — FanGraphs auto-fetch (Steamer/ZiPS/Depth Charts)
  draft_engine.py       — DraftRecommendationEngine (3-mode, 8-stage pipeline)
  draft_analytics.py    — Category balance, opportunity cost, BUY/FAIR/AVOID
  contextual_factors.py — Closer hierarchy, platoon risk, lineup protection, schedule strength
  ml_ensemble.py        — XGBoost ensemble (optional, graceful fallback)
  news_sentiment.py     — Keyword-based news sentiment scoring
  trade_value.py        — Universal trade value chart (0-100) with G-Score adjustment
  two_start.py          — Two-start pitcher planner
  start_sit.py          — Start/sit advisor (3-layer decision model)
  matchup_planner.py    — Weekly matchup planner with percentile color tiers
  waiver_wire.py        — Waiver wire LP-verified add/drop recommendations
  trade_finder.py       — Cosine dissimilarity trade finder
  draft_grader.py       — Post-draft grader (3-component)
  ecr.py                — Multi-platform ECR consensus (7 sources, Trimmed Borda Count)
  prospect_engine.py    — FanGraphs Board API + MLB Stats API MiLB stats
  player_news.py        — 4-source news aggregation with template summaries
  closer_monitor.py     — 30-team job security scoring
  points_league.py      — Yahoo/ESPN/CBS scoring presets
  leaders.py            — Category/points leaders, breakout detection
  cheat_sheet.py        — HTML/PDF export with HEATER theme
  live_draft_sync.py    — Yahoo draft polling and sync
  pick_predictor.py     — Normal CDF + Weibull survival blend
  player_tags.py        — Sleeper/Target/Avoid/Breakout/Bust tags with DB persistence
  schedule_grid.py      — 7-day schedule grid with matchup colors
  standings_projection.py — MC season simulation
  power_rankings.py     — 5-factor power rankings with bootstrap CI
  il_manager.py         — IL detection and replacement selection
  league_registry.py    — Multiple league support
  optimizer/            — Enhanced Lineup Optimizer (11 modules)
    pipeline.py         — Master orchestrator (Quick/Standard/Full modes)
    projections.py      — Enhanced projections (Bayesian/Kalman/regime/injury)
    matchup_adjustments.py — Park factors, platoon splits, weather
    h2h_engine.py       — H2H category weights + win probability
    sgp_theory.py       — Non-linear marginal SGP
    streaming.py        — Pitcher streaming + Bayesian stream scoring
    scenario_generator.py — Gaussian copula scenarios + CVaR
    dual_objective.py   — H2H/Roto weight blending
    advanced_lp.py      — Maximin, epsilon-constraint, stochastic MIP
  engine/               — Trade Analyzer Engine (6 phases)
    portfolio/          — Z-scores, SGP, category analysis, lineup optimizer, copula
    projections/        — ROS projections, BMA, KDE marginals
    monte_carlo/        — Paired MC simulator (10K sims, variance reduction)
    signals/            — Statcast, decay, Kalman filter, regime detection
    context/            — Log5 matchup, injury process, bench value, concentration risk
    game_theory/        — Opponent valuation, adverse selection, Bellman DP, sensitivity
    production/         — Convergence diagnostics, cache, adaptive sim scaling
    output/             — Master trade orchestrator (evaluate_trade)
tests/                  — 83 test files, 1956 passing tests
data/
  draft_tool.db         — SQLite database (created at runtime)
  backups/              — Draft state JSON backups
docs/
  ROADMAP.md            — Phase history and future directions
.github/
  workflows/ci.yml      — CI pipeline (lint + test + build)
  workflows/refresh.yml — Scheduled daily data refresh (9:17 UTC)
```

## Architecture

### Core Valuation Pipeline
1. **Projection blending** — Weighted average of 7 systems (Steamer, ZiPS, Depth Charts, ATC, THE BAT, THE BAT X, Marcel)
2. **SGP** — Converts raw stats to standings-point movement; auto-computed denominators
3. **VORP** — Value Over Replacement Player with multi-position flexibility premium
4. **pick_score** = weighted SGP + positional scarcity + VORP bonus

### Draft Engine
- Monte Carlo simulation (100-300 sims, 6-round horizon) with opponent roster tracking
- Survival probability (Normal CDF) + Urgency = (1 - P_survive) * positional_dropoff
- Combined score = MC mean SGP + urgency * 0.4, tier assignment via natural breaks

### Trade Analyzer Engine (6 phases)
- **Phase 1** (default): Peer-group z-scores → standings-based SGP → marginal elasticity → punt detection → LP-constrained lineup totals → roster cap enforcement → weighted SGP delta → grade (A+ to F)
- **Phase 2** (`enable_mc=True`): BMA → KDE marginals → Gaussian copula → paired MC (10K sims)
- **Phase 3**: Statcast → signal decay → Kalman filter → regime detection (BOCPD/HMM)
- **Phase 4** (`enable_context=True`): Log5 matchup → injury process → concentration risk (HHI)
- **Phase 5** (`enable_game_theory=True`): Opponent valuations → adverse selection → Bellman DP → sensitivity/counter-offers
- **Phase 6**: Convergence diagnostics (ESS/R̂) → cache → adaptive sim scaling

### UI Layer — Hybrid 3-Zone Layout
All 10 pages use a consistent 3-zone layout pattern:
1. **Recommendation banner** (top) — One-line teaser, expandable for detail. Uses `render_page_layout()` + `render_reco_banner()`.
2. **Context panel** (left ~20%) — Compact glassmorphic cards with summary data (category totals, filters, settings). Uses `render_context_card()` inside `render_context_columns()`.
3. **Main content** (right ~80%) — Full data tables, charts, interactive controls. Uses `render_compact_table()` for ESPN-style 11px monospace tables with sticky player names, hit/pitch header colors, health dots, and starter/bench row tinting.

Key layout functions in `src/ui_shared.py`:
- `build_compact_table_html(df, ...)` — Pure function, returns HTML string. Unit-testable (39 tests).
- `render_compact_table(df, ...)` — Thin Streamlit wrapper around `build_compact_table_html()`.
- `render_reco_banner(teaser, detail, icon)` — Collapsible banner with `st.expander`.
- `render_context_card(title, content_html)` — Glassmorphic sidebar card.
- `render_page_layout(title, banner_teaser, ...)` — Page title badge + optional banner.
- `render_context_columns(context_width)` — Returns `(ctx, main)` column pair.

CSS classes: `.reco-banner`, `.context-card`, `.compact-table`, `.th-hit`/`.th-pit`, `.col-name`, `.row-start`/`.row-bench`, `.health-dot`. Responsive at 768px/480px. Print-friendly.

Sidebar is collapsed by default on all pages (hamburger menu for nav only).

### Yahoo Data Service (Live-First Architecture)
All league data flows through `src/yahoo_data_service.py`, a three-tier cache:
1. **Tier 1: `st.session_state`** — survives Streamlit reruns, ~0ms latency
2. **Tier 2: Yahoo Fantasy API** — live source of truth, 0.5s per call
3. **Tier 3: SQLite fallback** — used when Yahoo is offline

TTLs: Rosters 30m, Standings 30m, Matchup 5m, Free Agents 1h, Transactions 15m, Settings/Schedule 24h.

Write-through: every Yahoo fetch writes to SQLite so un-migrated code (trade engine, league_manager) automatically gets fresher data.

Singleton accessor: `get_yahoo_data_service()` stored in `st.session_state["_yahoo_data_service"]`.

Pages wired: My Team, Trade Analyzer, Free Agents, Lineup Optimizer, Standings, Trade Finder, Matchup Planner.

New UI features: live matchup score card (My Team), recent transactions feed (Trade Analyzer), data freshness widget (My Team, Trade Analyzer, Standings).

### Trade Intelligence Layer
All trade valuations (Trade Finder + Trade Analyzer) flow through `src/trade_intelligence.py`:
1. **Health adjustment** — IL15=0.84x, IL60=0.55x, DTD=0.95x, NA excluded. Uses `injury_model.compute_health_score()` with 3-year history + Yahoo IL/DTD status from `league_rosters.status` column.
2. **Category-weighted SGP** — `category_gap_analysis()` computes marginal weights per category. Categories where you can gain standings positions weighted higher. Punted categories get zero weight. Max weight capped at 3x average to prevent single-category dominance.
3. **FA gating** — Dynamic threshold (85% early season -> 60% late season) flags trades where a comparable FA exists. Prevents wasting trade capital.
4. **Closer scarcity premium** — SV >= 5 players get 1.3x multiplier. C/SS/2B positions get 1.15x VORP premium.
5. **Elite player protection** — Players in top 20% by raw SGP require return >= 50% as valuable.
6. **Trade Readiness tab** — 0-100 composite: 40% category fit + 25% projection confidence + 15% health + 10% scarcity + 10% FA advantage.

### Bootstrap Pipeline
- Every app launch: splash screen → `bootstrap_all_data()` with staleness-based refresh
- Staleness thresholds: 1h live stats, 6h Yahoo, 7d players/projections, 30d historical/park factors
- Phase order: players → park factors → projections → live stats → historical → injury → Yahoo
- Bootstrap primes the SQLite cache; the YahooDataService handles ongoing freshness per-page

### Database Tables (21)

| Group | Tables |
|-------|--------|
| Draft | `projections`, `adp`, `league_config`, `draft_picks`, `blended_projections`, `player_pool` |
| In-Season | `season_stats`, `ros_projections`, `league_rosters`, `league_standings`, `park_factors`, `refresh_log` |
| Analytics | `injury_history`, `transactions` |
| Features | `player_tags`, `leagues` |
| Intelligence | `prospect_rankings`, `ecr_consensus`, `player_id_map`, `player_news`, `ownership_trends` |

## Key API Signatures

Only the commonly-misused ones. For others, read the source files.

```python
# YahooDataService (src/yahoo_data_service.py) — singleton accessor
from src.yahoo_data_service import get_yahoo_data_service
yds = get_yahoo_data_service()  # creates or retrieves from session_state
yds.get_rosters(force_refresh=False) -> pd.DataFrame     # 30min TTL
yds.get_standings(force_refresh=False) -> pd.DataFrame    # 30min TTL
yds.get_matchup(force_refresh=False) -> dict | None       # 5min TTL
yds.get_free_agents(max_players=500) -> pd.DataFrame      # 1hr TTL
yds.get_transactions(force_refresh=False) -> pd.DataFrame # 15min TTL
yds.get_schedule(force_refresh=False) -> dict             # 24hr TTL
yds.get_opponent_profile(team_name) -> dict               # from live standings
yds.force_refresh_all() -> dict[str, str]                 # invalidate + refetch
yds.get_data_freshness() -> dict[str, str]                # "Live (2m ago)" per key
yds.is_connected() -> bool                                # Yahoo auth status

# Freshness widget (src/ui_shared.py) — call inside context column
from src.ui_shared import render_data_freshness_card
render_data_freshness_card()  # shows badges + "Refresh All" button

# Opponent intel with live data (src/opponent_intel.py)
get_current_opponent(yds=yds)       # live schedule + profile when yds provided
get_opponent_for_week(week, yds=yds)  # same, for specific week

# Trade Intelligence (src/trade_intelligence.py)
get_health_adjusted_pool(player_pool, config) -> pd.DataFrame  # IL/DTD/NA adjusted
get_category_weights(user_team_name, all_team_totals, config) -> dict[str, float]
compute_fa_comparisons(opp_ids, user_ids, fa_pool, pool, config) -> dict[int, dict]
apply_scarcity_flags(player_pool) -> pd.DataFrame  # adds is_closer, scarcity_mult
compute_trade_readiness(player_id, ...) -> dict  # 0-100 composite score
compute_trade_readiness_batch(player_ids, ...) -> pd.DataFrame  # batch scoring

# STANDALONE functions in src/valuation.py — NOT methods on SGPCalculator
compute_replacement_levels(pool, config, sgp_calc)   # valuation.py
compute_sgp_denominators(pool, config)                # valuation.py

# Keyword args required for optional params
value_all_players(pool, config, roster_totals=None, category_weights=None,
                  replacement_levels=None, current_round=None, num_rounds=23)

# load_player_pool() columns:
# player_id, name, team, positions, is_hitter, is_injured,
# pa, ab, h, r, hr, rbi, sb, avg, obp, bb, hbp, sf, ip, w, l, sv, k, era, whip, er, bb_allowed, h_allowed, adp

# Trade evaluator (src/engine/output/trade_evaluator.py)
evaluate_trade(giving_ids, receiving_ids, user_roster_ids, player_pool,
               config=None, user_team_name=None, weeks_remaining=16,
               enable_mc=False, enable_context=True, enable_game_theory=True)
# Returns: {grade, surplus_sgp, category_impact, category_analysis, punt_categories,
#           replacement_penalty, risk_flags, verdict, confidence_pct,
#           before_totals, after_totals, giving_players, receiving_players,
#           total_sgp_change, mc_mean, mc_std, lineup_constrained, drop_candidate, fa_pickup, ...}

# DraftRecommendationEngine (src/draft_engine.py)
DraftRecommendationEngine(config: LeagueConfig, mode: str = "standard")
# .enhance_player_pool(player_pool, draft_state, park_factors=None) -> DataFrame
# .recommend(player_pool, draft_state, top_n=8, n_simulations=300, park_factors=None) -> DataFrame

# LineupOptimizerPipeline (src/optimizer/pipeline.py)
LineupOptimizerPipeline(roster, mode="standard", alpha=0.5, weeks_remaining=16, config=None)
# .optimize(...) -> {lineup, category_weights, h2h_analysis, streaming_suggestions, ...}

# Bootstrap (src/data_bootstrap.py)
bootstrap_all_data(yahoo_client=None, on_progress=None, force=False, staleness=None)

# Data pipeline (src/data_pipeline.py)
SYSTEM_MAP = {"steamer": "steamer", "zips": "zips", "fangraphsdc": "depthcharts"}
refresh_if_stale(force=False)

# UI shared (src/ui_shared.py)
THEME: dict  # Single light-mode palette: bg=#f4f5f0, primary=#e63946, hot=#ff6d00, etc.
T = THEME    # Backward compat: T["amber"]→#e63946, T["teal"]→#457b9d
PAGE_ICONS: dict[str, str]  # ~22 inline SVG icons
inject_custom_css()  # Full CSS injection (~1500 lines)

# Opponent model (src/simulation.py, src/draft_state.py)
compute_team_preferences(draft_history_df)  # needs columns: team_key, positions, round (NOT team_id/position)
get_team_draft_patterns(draft_state_dict, team_id: int)  # team_id is 0-based index

# Injury model (src/injury_model.py)
get_injury_badge(health_score) -> tuple[str, str]  # returns <span> with CSS dot, NOT emoji
# 4-tier thresholds: >= 0.85 Low Risk, >= 0.65 Moderate Risk, >= 0.40 Elevated Risk, < 0.40 High Risk
```

## Gotchas

### Data & Schema
- **DB column `name` vs UI `player_name`** — Normalized via `rename(columns={"name": "player_name"})` in `_build_player_pool()`. `evaluate_candidates()` returns `name`, not `player_name` — always alias after calling.
- **Sample data is `system='blended'` only** — Do NOT call `create_blended_projections()` when only blended data exists; it deletes `system='blended'` rows first.
- **Python 3.14 + SQLite bytes** — Returns bytes for some integer columns. Fixed with explicit `CAST` in SQL and `pd.to_numeric()` coercion. Also affects yfpy: `team.name` may return `bytes` — always guard with decode.
- **Players table has no UNIQUE constraint on name** — `upsert_player_bulk()` uses SELECT-first pattern. Do NOT use `ON CONFLICT(name)`. However, `injury_history` has UNIQUE on `(player_id, season)`.
- **Park factors schema** — Columns: `team_code`, `factor_hitting`, `factor_pitching` (not `team`/`park_factor`).
- **FanGraphs API SYSTEM_MAP** — FG API uses `"fangraphsdc"` for Depth Charts, DB stores as `"depthcharts"`. Always use `SYSTEM_MAP` dict.
- **LeagueConfig is the single source of truth** — All category definitions live in `LeagueConfig` in `src/valuation.py`. Do NOT add local hardcoded category lists — import from LeagueConfig.

### Yahoo API
- **Yahoo OAuth uses oob (out-of-band)** — `redirect_uri=oob` required. Do NOT use localhost redirects.
- **Yahoo auto-reconnect** — `_try_reconnect_yahoo()` loads `data/yahoo_token.json` on restart. Token refresh is automatic.
- **yfpy Roster iteration** — `Roster.__iter__()` yields attribute NAMES (strings), NOT players. Use `getattr(roster, "players", None) or []`.
- **yfpy token override** — When `yahoo_access_token_json` is provided, do NOT also pass `consumer_key`/`consumer_secret`.
- **Yahoo game_key** — MLB 2026 = 469. Must set BOTH `self._query.game_id` AND `self._query.league_key`.

### Streamlit UI
- **No emoji in the codebase** — All icons are inline SVGs from `PAGE_ICONS`. Injury badges use CSS dots. Do NOT introduce emoji.
- **No abbreviations in UI text** — Use full terms: "Standings Gained Points" not "SGP", "Value Over Replacement Player" not "VORP", etc.
- **CSS `!important` required** — Streamlit's inline CSS has high specificity. ALL custom CSS needs `!important`.
- **HTML sanitization** — `st.markdown('<div>')` auto-closes tags. Cannot use split open/close patterns with widgets between. Use self-contained HTML blocks.
- **Health badges in st.dataframe()** — Renders raw text only. Use text labels for dataframes, HTML badges only in `st.markdown(unsafe_allow_html=True)`.
- **`T["ink"]` for text-on-accent** — Use `T["ink"]` for dark text on colored surfaces, not `T["bg"]` (invisible on amber).
- **Connection leak pattern** — Always wrap `get_connection()` in `try/finally` with `conn.close()` in `finally`.
- **Draft Settings live on Draft Simulator page** — Not on Connect League page. Stored in `mock_num_teams`, `mock_num_rounds`, `mock_draft_pos`.

### Visual Audit Gotchas (discovered via Playwright)
- **THEME keys** — Use `T["tx"]` not `T["tx1"]`. There is no `tx1` key. This crashed the Start/Sit page.
- **Player Compare must not auto-select** — Default `compare_a`/`compare_b` to `None`, not `top_a[0]`. Otherwise the page loads with two random nobodies showing all-zero stats.
- **Matchup Planner empty schedule** — When `games_count` is 0 for all players, show empty state instead of fake "Avoid" ratings. Check `ratings_df["games_count"].sum() == 0` before rendering.
- **Sidebar icon map** — New pages need entries in the JS `icons` dict in `inject_custom_css()`. Keys must match the sidebar link text exactly (e.g., `'Waiver Wire'`, `'Start Sit'`, `'Matchup Planner'`).
- **Leaders Team column empty** — Known data-quality issue: `players.team` is "MLB" or NULL for many players. Not a page bug — the MLB Stats API doesn't always return team abbreviations.
- **Closer Monitor only 6 teams** — Known: FanGraphs depth chart data is incomplete in the bootstrap. Not a page bug.

### Player Card & News
- **`player_news` table has no UNIQUE constraint** — Yahoo sync can insert duplicate rows. `_render_news_tab` deduplicates by `player_name + headline` (case-insensitive) before rendering.
- **Bayesian table must filter to roster only** — `season_stats` has ALL players across ALL seasons. Query with `WHERE player_id IN (roster_pids)` and `drop_duplicates(subset=["player_id"], keep="first")` after sorting by season DESC.
- **`mlb_id` can be `float('nan')` from SQLite NULL** — NaN is truthy in Python. Use `mlb_id is not None` + `try: int(mlb_id)` instead of `if mlb_id:`.
- **Yahoo team names may contain emoji** — Strip non-alpha chars before extracting initials for monogram fallback.
- **`league_rosters`/`league_standings` need UNIQUE constraints** — Schema in `database.py` has them but old DBs may not. Drop and recreate tables if `ON CONFLICT` fails.

### Algorithms & Math
- **Rate-stat aggregation** — AVG=sum(h)/sum(ab), OBP=sum(h+bb+hbp)/sum(ab+bb+hbp+sf), ERA=sum(er)*9/sum(ip), WHIP=sum(bb+h)/sum(ip). Weighted averages, NOT simple averages.
- **LP inverse stat weighting** — ERA/WHIP must be weighted by IP in LP objective. Without it, 1-IP relievers dominate 200-IP starters.
- **L (Losses) is an inverse counting stat** — Lower-is-better but counting. Sign-flip in SGP, no IP weighting in LP.
- **Punt detection requires BOTH conditions** — `gainable_positions == 0 AND rank >= 10`.
- **LP-constrained lineup totals** — Only 18 starters' stats feed SGP. Bench excluded. Prevents phantom production in uneven trades.
- **Roster cap enforcement (ROSTER_CAP=23)** — 2-for-1: drop lowest-SGP bench. 1-for-2: add FA capped at median SGP.
- **Percentile pipeline ordering** — `compute_projection_volatility()` → `add_process_risk()` → `compute_percentile_projections()`. Skip when only one projection system.
- **P90 ERA/WHIP inversion** — P90=optimistic=lowest ERA, P10=pessimistic=highest ERA. Uses `z_eff = -z`.
- **Kalman prior_var = process variance only** — NOT `proc_var + obs_var` (double-counts noise).
- **Paired MC seed discipline** — Same seed for before/after rosters. NEVER use different seeds.
- **Maximin LP excludes inverse cats** — L, ERA, WHIP auto-filtered (incompatible SGP scale).
- **Opponent pick probability is additive blend** — Not multiplicative (collapses to near-zero).
- **Trade analyzer needs standings for accuracy** — Without standings: no elasticity, no punts, no strategic context.

### Dependencies & Platform
- **Plotly 6.x hex colors** — Does NOT accept 8-digit hex (`#RRGGBBAA`). Use `rgba(r,g,b,a)`.
- **PyMC/PuLP optional deps** — `PYMC_AVAILABLE`/`PULP_AVAILABLE` flags. Use `try/except ImportError`. CI skips with `@pytest.mark.skipif`.
- **XGBoost optional** — `XGBOOST_AVAILABLE` flag. Returns zeros when unavailable.
- **`ruff` on Windows** — Use `python -m ruff check .` instead of bare `ruff`.
- **Cheat sheet PDF requires weasyprint** — Returns None when not installed.
- **Snake draft order** — `round % 2 == 0` forward, `round % 2 == 1` reverse.
- **`datetime.now(UTC)`** — Used everywhere; `datetime.utcnow()` is deprecated.

### Trade Engine Specifics
- **Trade engine fallback** — Page wraps engine import in try/except, falls back to legacy `analyze_trade()`.
- **`enable_mc=True` activates Phase 2** — Default is Phase 1 only (deterministic).
- **`bench_cost` is always 0.0** — Replaced by explicit roster cap modeling. Key kept for backward compat.
- **Sensitivity calls use `enable_mc=False, enable_context=False`** — Avoids expensive calls during N-player analysis.
- **FA pickup median SGP cap** — Prevents unrealistic elite "FA" acquisitions when no rosters loaded.
- **Replacement penalty skips rate stats** — AVG, ERA, WHIP excluded (roster-aggregate, not simple counting gaps).

## Data Sources

- **Players:** MLB Stats API (750+ active + 40-man); staleness: 7 days
- **Projections:** 7 systems via FanGraphs JSON API + Marcel (local); staleness: 7 days
- **ADP:** FanGraphs Steamer + FantasyPros ECR + NFBC; staleness: 24 hours
- **Current stats:** MLB Stats API; staleness: 1 hour
- **Historical stats:** 3 years (2023-2025) from MLB Stats API; staleness: 30 days
- **Park factors:** Hardcoded FanGraphs 2024 values (30 teams); staleness: 30 days
- **League data:** Yahoo Fantasy API (optional); staleness: 6 hours
- **News:** ESPN + RotoWire RSS + MLB Stats API + Yahoo; staleness: 1 hour
- **Prospects:** FanGraphs Board API + MLB Stats API MiLB; staleness: 7 days
- **ECR:** 7-source aggregation; staleness: 24 hours

## GitHub

- **Repo:** https://github.com/hicklax13/fantasy-baseball-draft-tool (public)
- **Current release:** v1.0.0

## Testing

- **2139 passing tests** across 84+ test files, 4 skipped (PyMC/xgboost optional deps)
- **CI:** GitHub Actions — ruff lint/format + pytest on Python 3.11, 3.12, 3.13
- **Coverage:** 64% (above 60% CI threshold)
- **8 rounds of systematic debugging** (207 bugs fixed) + **data pipeline audit** (32 issues fixed), all CI green
- **Full system audit** (March 26, 2026) — 19 bugs cataloged, all critical/high fixed
- **Manual UI testing** — All 13 pages tested via Playwright + Claude in Chrome (March 2026)

## Current Implementation Plan

**Read `The_Last_Plan.md` at the start of every session.** It is the active implementation plan.

- **The Last Plan:** AVIS-driven in-season optimization — Yahoo exhaustive FA sync, AVIS hard constraints, opponent intelligence, weekly automation
- **Status:** All 4 phases implemented and tested (2022 pass, 0 fail)
- **Previous plans:** `plan_1.md` (completed), `Full_Debug_Plan.md` (completed)
- **AVIS Manual:** `AVIS_FANTASY_BASEBALL_OPS_MANUAL_2026.md` — the "bible" for league rules, scoring, team analysis
- **Season context:** MLB 2026 season started March 25. Fantasy draft completed. App is now in-season mode.

## New Modules (Added March 26, 2026)

- **`src/alerts.py`** — AVIS-enforced decision rules: auto-IL, closer handcuff, streaming SP, sell-high/buy-low triggers
- **`src/opponent_intel.py`** — Opponent roster analysis, trade target finder, weakness detection
- **`src/weekly_report.py`** — Monday morning briefing: roster health, matchup preview, waiver targets, trade opportunities
- **`src/ip_tracker.py`** — Weekly IP tracking toward 1,400 IP target (AVIS requirement)

## Key Fixes (March 26, 2026)

- **BUG-001 FIXED:** FanGraphs 403 → pybaseball + Marcel fallback pipeline
- **BUG-003 FIXED:** Empty team fields → team_id-to-abbreviation mapping from MLB API
- **BUG-004 FIXED:** 780 players → 9,213 via expanded roster fetch (40-man + spring training)
- **Yahoo exhaustive FA sync** — Paginated fetch of ALL league free agents (not just 50)
- **Start/Sit scoring** — ERA/WHIP now use delta-from-baseline SGP (was all-negative)
- **Trade Analyzer** — Dropdown shows tradeable players from other teams (was 9K player pool)
- **Lineup Optimizer** — Pitchers now counted in starting lineup (was hitter-only)
- **Points Leaders** — Fixed `list.empty` AttributeError, returns DataFrame

## Season State (2026)

- **MLB game_key:** 469
- **Draft:** Completed (snake, 23 rounds)
- **Mode:** In-season management (rosters, trades, waivers, matchups)
- **Yahoo token:** Auto-refreshes via `data/yahoo_token.json` (persistent across sessions)
- **Yahoo auto-reconnect:** `_try_reconnect_yahoo()` loads token on every app launch
- **Player pool:** 9,213 players loaded from MLB Stats API + pybaseball + Marcel
- **League rosters:** All 12 teams synced (264 players)
- **Free agents:** Exhaustive pagination from Yahoo API

## Resume Checklist (New Session)

1. Read `CLAUDE.md` (this file) and `The_Last_Plan.md`
2. Read `AVIS_FANTASY_BASEBALL_OPS_MANUAL_2026.md` (league bible)
3. Check `docs/AUDIT_REPORT.md` for any remaining bugs
4. Run `python -m pytest -x -q` to verify all tests pass
5. Run app with `streamlit run app.py` and verify Yahoo auto-reconnect works
6. Continue with any remaining tasks from The Last Plan or user requests

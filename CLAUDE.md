# HEATER — Fantasy Baseball Draft Tool

## Overview

A fantasy baseball draft assistant + in-season manager for a 12-team Yahoo Sports H2H Categories snake draft league. Two pillars:

1. **Draft Tool** (`app.py`, ~1800 lines) — "Heater" themed Streamlit app with light-mode-only glassmorphic design system: splash screen data bootstrap + 2-step setup wizard (Settings, Launch), 3-column draft page with SVG injury badges, percentile ranges, opponent intel tab, and practice mode. Monte Carlo recommendations with percentile sampling. Zero CSV uploads — all data auto-fetched from MLB Stats API + FanGraphs on every launch. Pill button navigation replaces dropdowns. Search + card grids for player selection.
2. **In-Season Management** (`pages/`) — 9 Streamlit pages: team overview, draft simulator, trade analysis, player comparison, free agent rankings, lineup optimizer, closer monitor, standings/power rankings, leaders/prospects. Powered by MLB Stats API + pybaseball + optional Yahoo Fantasy API. All pages share centralized single-theme system (light mode only) with glassmorphic design, orange sidebar branding, and bold Heater identity.
2b. **In-Season Analytics** (`src/` 7 new modules) — Trade value chart (0-100 universal values with G-Score H2H variance adjustment), two-start pitcher planner, start/sit 3-layer advisor, weekly matchup planner with percentile color tiers, waiver wire LP-verified add/drop recommender, cosine-dissimilarity trade finder, post-draft grader. 276 dedicated tests across 7 test files.
3. **Trade Analyzer Engine** (`src/engine/`) — 6-phase pipeline: Phase 1 deterministic SGP with LP-constrained lineup totals, Phase 2 stochastic MC (10K sims), Phase 3 signal intelligence (Statcast/Kalman/BOCPD), Phase 4 contextual adjustments (matchups/injuries/concentration), Phase 5 game theory (opponent modeling/adverse selection/Bellman), Phase 6 production (convergence/caching/adaptive scaling). 11 modules, 219 dedicated tests.
4. **Enhanced Lineup Optimizer** (`src/optimizer/`) — 11-module pipeline with 20 mathematical techniques: enhanced projections (Bayesian/Kalman/regime/injury), weekly matchup adjustments (park/platoon/weather), H2H category weights (Normal PDF), non-linear SGP (bell-curve proximity), pitcher streaming, stochastic scenarios (copula/CVaR), multi-period planning, dual H2H/Roto objective, advanced LP (maximin/epsilon-constraint/stochastic MIP). Three modes: Quick (<1s), Standard (2-3s), Full (5-10s). 204 dedicated tests across 10 test files.
5. **Draft Recommendation Engine** (`src/draft_engine.py` + 4 support modules) — 25-feature enhanced draft pipeline: 3 execution modes (Quick <1s, Standard 2-3s, Full 5-10s), 8-stage enhancement chain (park factors → Bayesian blend → injury probability → Statcast delta → FIP correction → contextual factors → category balance → ML ensemble). Multiplicative + additive scoring formula with clipping. Category-aware recommendations via Normal PDF weighting. Contextual factors: closer hierarchy, platoon risk, lineup protection, schedule strength, contract year boost. ML ensemble (XGBoost, optional) + news sentiment scoring. BUY/FAIR/AVOID classification. 270 dedicated tests across 5 test files.
6. **Gap Closure Data Layer** (`src/` new modules) — 14 new modules closing spec gaps: extended roster (40-man + spring training via MLB Stats API), 7 projection systems (Steamer/ZiPS/Depth Charts/ATC/THE BAT/THE BAT X/Marcel), multi-source ADP (FanGraphs + FantasyPros ECR + NFBC), depth chart scraping with role classification, contract year data from BB-Ref, news fetcher from MLB transactions API, background refresh scheduler, GitHub Actions daily cron. Engine output enrichment adds composite value score, position/overall ranks, confidence level, LAST CHANCE badge. 153 dedicated tests across 14 test files.
7. **FantasyPros Parity** (`src/` 15 new modules + 3 new pages) — 17 features closing FantasyPros competitive gaps: draft order generator, player tags (Sleeper/Target/Avoid/Breakout/Bust) with DB persistence, pick predictor (Normal CDF + Weibull survival blend), closer depth chart monitor (30-team job security grid), points league projections (Yahoo/ESPN/CBS presets), Bayesian stream scoring with matchup grades, category/points leaders with breakout detection, ECR integration (15% blend + disagreement badges), cheat sheet export (HTML/PDF with HEATER theme), live draft assistant (Yahoo real-time sync), 7-day schedule grid with matchup color-coding, WSIS quick compare (density overlap), projected season standings (copula-based MC simulation), 5-factor league power rankings (bootstrap CI), auto-swap IL players, multiple league support, prospect rankings. 184 dedicated tests across 16 test files.

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

# Run all tests (1774 pass, 4 skipped for PyMC/xgboost)
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
- **Format:** 12-team snake draft, 23 rounds, Head-to-Head Categories
- **Hitting cats (6):** R, HR, RBI, SB, AVG, OBP
- **Pitching cats (6):** W, L, SV, K, ERA, WHIP
- **Inverse cats:** L, ERA, WHIP (lower is better)
- **Rate stats:** AVG, OBP, ERA, WHIP
- **Roster:** C/1B/2B/3B/SS/3OF/2Util/2SP/2RP/4P/5BN = 23 slots
- **Manager skill:** Opponents extremely high-skilled; user is a novice

## Tech Stack

- **Framework:** Streamlit (Python), multi-page app
- **Database:** SQLite (`data/draft_tool.db`) — 16 tables total
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
.streamlit/config.toml  — Light theme configuration (Heater palette)
.claude/launch.json     — Dev server config for preview tools
pages/
  1_My_Team.py          — In-season: team overview with monogram avatar, roster, category standings, Yahoo sync
  2_Draft_Simulator.py  — Standalone draft simulator (AI opponents, MC recommendations, pill filters, card-based picks)
  3_Trade_Analyzer.py   — In-season: trade proposal builder + Phase 1 SGP engine (grade A+ to F, LP-constrained lineup totals, roster cap enforcement, punt detection, marginal elasticity, category replacement cost penalty) with legacy fallback
  4_Player_Compare.py   — In-season: head-to-head player comparison with dual search + card pickers
  5_Free_Agents.py      — In-season: free agent rankings by marginal value, position pill filters
  6_Lineup_Optimizer.py — In-season: 5-tab lineup optimizer (Optimize, H2H Matchup, Streaming, Category Analysis, Roster) powered by LineupOptimizerPipeline
  7_Closer_Monitor.py   — 30-team closer depth chart grid with job security meters
  8_Standings.py        — Projected season standings (MC simulation) + 5-factor league power rankings (bootstrap CI)
  9_Leaders.py          — Category leaders, points leaders, breakout detection, prospect rankings
src/
  database.py           — SQLite schema (16 tables), CSV import, projection blending, player pool + in-season queries
  valuation.py          — SGP calculator, replacement levels, VORP, category weights, percentile forecasts
  draft_state.py        — Draft state management, roster tracking, snake pick order, opponent patterns
  simulation.py         — Monte Carlo draft simulation, opponent modeling (history-aware), survival probability
  in_season.py          — Trade analyzer, player comparison engine, FA ranker
  live_stats.py         — MLB Stats API data fetcher: roster, season/historical stats, injury data extraction
  data_bootstrap.py     — Zero-interaction bootstrap orchestrator: staleness-based refresh of all data sources on app launch
  league_manager.py     — League roster/standings management, CSV import for all 12 teams
  ui_shared.py          — Heater design system: single THEME dict, PAGE_ICONS (inline SVGs), glassmorphic CSS injection (~1500 lines), metric tooltips, sidebar branding
  data_2026.py          — Hardcoded 2026 projections (~200 hitters, ~80 pitchers) for sample data
  validation.py         — Validation utilities
  bayesian.py           — Bayesian projection updater: PyMC hierarchical model + Marcel regression fallback
  injury_model.py       — Health scores, age-risk curves, workload flags, injury-adjusted projections
  lineup_optimizer.py   — PuLP LP solver: lineup optimization, category targeting, two-start SP detection
  yahoo_api.py          — Yahoo Fantasy API: OAuth integration via yfpy, league sync, roster import
  data_pipeline.py      — FanGraphs auto-fetch: Steamer/ZiPS/Depth Charts JSON API, normalize, upsert, ADP extraction
  draft_engine.py       — DraftRecommendationEngine: 3-mode orchestrator (Quick/Standard/Full), 8-stage enhancement pipeline, multiplicative+additive pick_score formula
  draft_analytics.py    — Category balance (Normal PDF weighting), opportunity cost, streaming draft value, BUY/FAIR/AVOID classification
  contextual_factors.py — Closer hierarchy detection, platoon risk (The Book), lineup protection (PA bonus), schedule strength, contract year boost
  ml_ensemble.py        — XGBoost ensemble model for residual prediction (optional dep, graceful fallback to 0.0)
  news_sentiment.py     — Keyword-based news sentiment scoring (-1.0 to +1.0), high-impact flags, batch processing
  news_fetcher.py       — MLB Stats API transaction fetcher, fuzzy name matching, player news aggregation
  adp_sources.py        — Multi-source ADP: FantasyPros ECR scraper + NFBC ADP scraper, name→player_id resolution
  contract_data.py      — Contract year detection from Baseball Reference free agent list scraper
  depth_charts.py       — FanGraphs depth chart scraper, role classification (starter/platoon/closer/setup/committee)
  marcel.py             — Marcel projection system: 3yr weighted avg (5/4/3) with regression to mean + age adjustment
  backtesting.py        — Draft engine accuracy backtesting: projected vs actual comparison, RMSE, rank correlation, value capture rate
  scheduler.py          — Background data refresh daemon thread (5-min interval, staleness-aware, idempotent start/stop)
  trade_value.py        — Universal trade value chart (0-100): SGP surplus + G-Score H2H variance adjustment (arXiv:2307.02188), dollar values, contextual overlays, 5 tiers (Elite/Star/Solid/Flex/Replacement)
  two_start.py          — Two-start pitcher planner: schedule scanning, pitcher matchup scoring (K-BB%/xFIP/CSW%), rate stat damage analysis, streaming value with park factor adjustment, thread-safe team batting cache
  start_sit.py          — Start/sit advisor: 3-layer decision model (H2H-weighted weekly projection, risk-adjusted scoring by matchup state, per-category SGP impact), platoon/park/home-away factors
  matchup_planner.py    — Weekly matchup planner: per-game hitter/pitcher ratings with percentile-based color tiers (smash/favorable/neutral/unfavorable/avoid), multiplicative matchup model, park-adjusted projections
  waiver_wire.py        — Waiver wire + drop suggestions: LP-verified add/drop pairs, 7-stage pipeline (category gap → FA pre-filter → drop cost → net swap → sustainability BABIP filter → greedy optimization → annotate)
  trade_finder.py       — Trade finder: cosine dissimilarity team pairing (arXiv:2111.02859), 1-for-1 scan with acceptance probability (loss aversion behavioral model), grade integration
  draft_grader.py       — Post-draft grader: 3-component grading (40% team value, 35% pick efficiency, 25% category balance), steal/reach detection (AND logic: SGP surplus + ADP gap), position-adjusted thresholds
  draft_order.py        — Randomized draft order generator with seed-based reproducibility
  player_tags.py        — Player tag CRUD (Sleeper/Target/Avoid/Breakout/Bust) with SQLite persistence and badge rendering
  pick_predictor.py     — Pick predictor: Normal CDF + Weibull survival blend per position, future user pick computation
  closer_monitor.py     — Closer depth chart monitor: 30-team job security scoring, color-coded grid builder
  points_league.py      — Points league projections: Yahoo/ESPN/CBS scoring presets, missing stat estimation, points VORP
  leaders.py            — Category/points leaders with min PA/IP thresholds, breakout detection (z > 1.5)
  ecr.py                — ECR integration: FantasyPros fetch extension, 15% blend with SGP, disagreement badges, prospect rankings
  cheat_sheet.py        — Cheat sheet HTML/PDF export with HEATER theme, positional tiers, health badges, print CSS
  live_draft_sync.py    — Live draft assistant: Yahoo draft polling, diff detection, DraftState sync, YOUR TURN detection
  schedule_grid.py      — 7-day schedule grid with matchup color-coding (smash/favorable/neutral/unfavorable/avoid)
  start_sit_widget.py   — Quick WSIS compare (2-4 players): density overlap coefficient, Normal PDF visualization data
  standings_projection.py — Projected season standings: round-robin schedule, Normal CDF category win probability, MC simulation
  power_rankings.py     — 5-factor league power rankings: roster quality, category balance, SOS, injury exposure, momentum with bootstrap CI
  il_manager.py         — IL detection: status classification, duration estimation, lost SGP computation, replacement selection
  league_registry.py    — Multiple league support: register/get/list/set_active/delete with SQLite persistence
Research.md             — FantasyPros competitive gap analysis: 74 FP features vs 33 HEATER comparable features, 30 actionable gaps in 3 priority tiers, recommendations for next builds
  optimizer/            — Enhanced Lineup Optimizer (11 modules, 20 mathematical techniques)
    __init__.py         — Package exports with lazy import documentation
    pipeline.py         — Master orchestrator: 9-stage chain, Quick/Standard/Full modes, LineupOptimizerPipeline class
    projections.py      — Enhanced projections: Bayesian→Kalman→regime→decay→Statcast→injury availability chain
    matchup_adjustments.py — Weekly matchup adjustments: park factors, platoon splits (The Book regression), weather HR model
    h2h_engine.py       — H2H category weights (Normal PDF peaks at tie) + per-category win probability (Normal CDF)
    sgp_theory.py       — Non-linear marginal SGP (bell-curve proximity), SLOPE regression SGP denominators
    streaming.py        — Pitcher streaming value (counting SGP − rate damage), two-start quantification, optimal schedule, Bayesian stream scoring with matchup grades (A+ through C)
    scenario_generator.py — Gaussian copula correlated scenarios, mean-variance adjustments, CVaR linearization data
    multi_period.py     — Rolling horizon optimization with discount factor, season balance urgency weights
    dual_objective.py   — H2H/season-long weight blending (alpha parameter), auto-alpha recommendation
    advanced_lp.py      — Maximin LP (balanced worst-category), epsilon-constraint (Pareto frontier), stochastic MIP
  engine/               — Trade Analyzer Engine (Phase 1-4)
    __init__.py
    portfolio/
      __init__.py
      valuation.py      — Z-score + SGP valuation: peer-group z-scores, standings-based SGP denominators, VORP
      category_analysis.py — Marginal SGP elasticity (1/gap), category gap analysis, punt detection, category weights
      lineup_optimizer.py — LP optimizer wrapper: optimal lineup value, pre/post trade lineup delta, bench option value
      copula.py         — Gaussian copula: correlated stat sampling, empirical 12×12 correlation matrix, nearest-PD correction
    projections/
      __init__.py
      projection_client.py — ROS projections loader, 3-pass fuzzy name matching, trade player resolution
      bayesian_blend.py — BMA: posterior-weighted projection blending, system forecast sigmas, total variance decomposition
      marginals.py      — KDE marginals: kernel density estimation with Normal fallback, stat bounds clipping, ppf for copula
    monte_carlo/
      __init__.py
      trade_simulator.py — Paired MC: 10K sims, variance reduction via identical seeds, VaR/CVaR/Sharpe/CI metrics
    signals/
      __init__.py
      statcast.py       — Statcast harvesting: pitch-level data → rolling features (EV, barrel%, xwOBA, whiff%)
      decay.py          — Exponential decay weighting: signal-specific half-lives, recency-weighted mean/variance
      kalman.py         — Kalman filter: true talent estimation, sample-size-aware observation variance
      regime.py         — Regime detection: BOCPD changepoint detection (run length mode) + 4-state HMM
    context/
      __init__.py
      matchup.py        — Log5 matchup engine: batter-pitcher odds ratio, park/weather adjustments, game-level projections
      injury_process.py — Injury stochastic process: Weibull duration sampling, frailty multipliers, season availability MC
      bench_value.py    — Enhanced bench option value: streaming + hot FA + flexibility premium + injury replacement cushion
      concentration.py  — Roster concentration risk: HHI scoring, diversification delta, team exposure breakdown
    game_theory/
      __init__.py
      opponent_valuation.py — Opponent willingness-to-pay + Nash equilibrium market clearing price (L8A)
      adverse_selection.py — Bayesian adverse selection discount from manager trade history (L8B)
      dynamic_programming.py — Bellman rollout for future trade option value, playoff-aware discounting (L9)
      sensitivity.py     — Category/player sensitivity analysis, breakeven, counter-offer generation (L11)
    production/
      __init__.py
      convergence.py     — MC convergence diagnostics: ESS via FFT, split-R̂, running mean stability
      cache.py           — Precomputation cache: TTL staleness, get_or_compute pattern, singleton
      sim_config.py      — Adaptive simulation scaling: 1K-100K by trade complexity + time budget
    output/
      __init__.py
      trade_evaluator.py — Master trade orchestrator: Phase 1-5 + LP-constrained lineup totals + roster cap enforcement + enable_mc/enable_context/enable_game_theory flags
tests/
  test_database_schema.py   — DB schema and table existence tests
  test_database_queries.py  — Query function tests
  test_in_season.py         — Trade analyzer, comparison, FA ranker tests
  test_league_manager.py    — League roster/standings management tests
  test_live_stats.py        — Live stats pipeline tests
  test_data_bootstrap.py    — Bootstrap pipeline: staleness, bulk upserts, orchestrator (30 tests)
  test_bayesian.py          — Bayesian updater: regression, age curves, stabilization, batch update (31 tests)
  test_injury_model.py      — Injury model: health scores, age risk, workload flags (17 tests)
  test_lineup_optimizer.py  — Lineup optimizer: LP solver, constraints, category targeting, scale normalization (24 tests)
  test_optimizer_projections.py — Enhanced projections pipeline: Bayesian, Kalman, regime, injury, integration (28 tests)
  test_optimizer_matchups.py — Matchup adjustments: platoon, park factors, weather, schedule pipeline (19 tests)
  test_optimizer_h2h.py    — H2H engine: category weights, win probability, inverse cats, edge cases (18 tests)
  test_optimizer_sgp.py    — Non-linear SGP: bell-curve proximity, SLOPE denominators, weight normalization (16 tests)
  test_optimizer_streaming.py — Streaming: pitcher value, two-start, optimal schedule, rate damage, Bayesian stream scoring (22 tests)
  test_optimizer_scenarios.py — Scenarios: copula generation, mean-variance, CVaR constraints, risk metrics (19 tests)
  test_optimizer_multi_period.py — Multi-period: rolling horizon, season balance, urgency weights (16 tests)
  test_optimizer_dual_objective.py — Dual objective: alpha blending, auto-alpha, edge cases (21 tests)
  test_optimizer_advanced_lp.py — Advanced LP: maximin, epsilon-constraint, stochastic MIP (25 tests)
  test_optimizer_pipeline.py — Pipeline orchestrator: init, optimize, weights, risk metrics, integration (26 tests)
  test_yahoo_api.py         — Yahoo API: OAuth, oob flow, sync, mock endpoints (40 tests)
  test_deduplication.py     — Player deduplication: merge duplicates, remap foreign keys, case-insensitive matching (12 tests)
  test_percentiles.py       — Percentile forecasts: volatility, P10/P50/P90 bounds (7 tests)
  test_opponent_model.py    — Enhanced opponent modeling: preferences, needs, history (8 tests)
  test_percentile_sampling.py — Percentile sampling passthrough in evaluate_candidates (4 tests)
  test_data_pipeline.py     — FanGraphs auto-fetch: normalization, fetch, storage, ADP, orchestration (28 tests)
  test_data_foundation.py   — Phase 2 data foundation: LeagueConfig updates, Yahoo ADP, draft_state enhancements, sample data (30 tests)
  test_draft_engine.py      — DraftRecommendationEngine: all 8 stages, 3 modes, enhanced_pick_score formula, integration (50 tests)
  test_draft_analytics.py   — Category balance, opportunity cost, streaming value, BUY/FAIR/AVOID (35 tests)
  test_contextual_factors.py — Closer hierarchy, platoon risk, lineup protection, schedule strength, contract year (25 tests)
  test_ml_ensemble.py       — ML ensemble + news sentiment: XGBoost fallback, feature prep, keyword scoring (40 tests)
  test_risk_score.py        — Spring training K-rate signal + composite risk score (0-100) (16 tests)
  test_schema_persistence.py — Statcast archive, FG ID cross-ref, computed field persistence (10 tests)
  test_yahoo_adp.py         — Yahoo ADP extraction and bootstrap integration (5 tests)
  test_heatmap.py           — Category heatmap grid: color coding, inverse stats, rendering (5 tests)
  test_backtesting.py       — Draft engine backtesting: RMSE, rank correlation, value capture, bust rate (15 tests)
  test_integration.py       — End-to-end pipeline: injury → Bayesian → percentiles → valuation (11 tests)
  test_trade_value.py       — Trade value chart: G-Score adjustment, tier assignment, dollar values, contextual overlay, position filter (40 tests)
  test_two_start.py         — Two-start planner: schedule parsing, matchup scoring, rate damage, confidence tiers, streaming integration (43 tests)
  test_start_sit.py         — Start/sit advisor: matchup classification, risk adjustment, weekly projections, category impact, confidence labels (50 tests)
  test_matchup_planner.py   — Matchup planner: hitter/pitcher game ratings, percentile ranking, weekly ratings, park adjustments, color tiers (38 tests)
  test_waiver_wire.py       — Waiver wire: BABIP, category priority, drop cost, net swap, sustainability, add/drop recommendations (32 tests)
  test_trade_finder.py      — Trade finder: cosine dissimilarity, team vectors, complementary teams, acceptance probability, 1-for-1 scan, trade opportunities (28 tests)
  test_draft_grader.py      — Draft grader: grade mapping, expected SGP curve, pick classification, category balance, category projections, full grading (45 tests)
  test_draft_order.py       — Draft order generator: length, contents, seed reproducibility, formatting (8 tests)
  test_player_tags.py       — Player tags: add/remove/get CRUD, duplicate handling, DB persistence, badge rendering (12 tests)
  test_pick_predictor.py    — Pick predictor: Weibull survival, curve monotonicity, position shapes, past-ADP behavior (8 tests)
  test_closer_monitor.py    — Closer monitor: job security bounds, color thresholds, grid building, committee handling (10 tests)
  test_points_league.py     — Points league: missing stat estimation, Yahoo/ESPN/CBS scoring, hitter-only, points VORP (11 tests)
  test_leaders.py           — Leaders: category leaders, ascending for ERA/WHIP, min thresholds, breakout detection (11 tests)
  test_ecr.py               — ECR: fetch extension, blend computation, disagreement badges, round-trip storage (12 tests)
  test_cheat_sheet.py       — Cheat sheet: HTML generation, positional sections, tiers, health badges, print CSS, PDF fallback (16 tests)
  test_live_draft_sync.py   — Live draft sync: new pick detection, incremental sync, user turn detection, error handling (12 tests)
  test_schedule_grid.py     — Schedule grid: 7-day structure, off days, empty roster, tier colors, HTML output (10 tests)
  test_start_sit_widget.py  — WSIS widget: 2-4 player compare, overlap coefficient, identical/distant distributions (10 tests)
  test_standings_projection.py — Standings projection: round-robin schedule, category win probability, MC simulation, CI ordering (9 tests)
  test_power_rankings.py    — Power rankings: roster quality, category balance, weights sum, bootstrap CI, sorted descending (11 tests)
  test_il_manager.py        — IL manager: classify types, duration estimates, lost SGP, eligible replacements, alert generation (17 tests)
  test_league_registry.py   — League registry: register/get/list/set_active/delete, first-becomes-active, UUID generation (15 tests)
  test_prospect_rankings.py — Prospect rankings: fetch returns DataFrame, top-N limit, position filter (6 tests)
  test_valuation_math.py    — Math verification: SGP, VORP, replacement levels, percentiles, process risk (40 tests)
  test_simulation_math.py   — Math verification: survival probability, urgency, combined score, tiers, MC convergence (37 tests)
  test_trade_math.py        — Math verification: trade SGP delta, MC noise, verdict, z-scores, rate stats (35 tests)
  test_trade_engine_math.py — Math verification: all 6 phases hand-calculated — SGP, BMA, copula, decay, Kalman, HHI, Bayes, Vickrey, Bellman, ESS, R̂, replacement cost, lineup constraint (56 tests)
  test_trade_engine.py      — Trade engine Phase 1: marginal SGP, punt detection, z-scores, grading, fuzzy match, replacement cost penalty, lineup-constrained eval, integration (47 tests)
  test_trade_engine_phase2.py — Trade engine Phase 2: BMA, KDE marginals, copula, paired MC, integration (33 tests)
  test_trade_engine_phase3.py — Trade engine Phase 3: Statcast aggregation, signal decay, Kalman filter, BOCPD, HMM regime, rolling features (32 tests)
  test_trade_engine_phase4.py — Trade engine Phase 4: Log5 matchup, Weibull injury, enhanced bench, HHI concentration, context integration (40 tests)
  test_trade_engine_phase5.py — Trade engine Phase 5: opponent valuations, adverse selection, Bellman rollout, sensitivity, counter-offers (38 tests)
  test_trade_engine_phase6.py — Trade engine Phase 6: ESS convergence, split-R̂, cache TTL, adaptive sim scaling (32 tests)
  profile_latency.py        — Performance profiling utility
data/
  draft_tool.db         — SQLite database (created at runtime)
  backups/              — Draft state JSON backups
docs/plans/             — Implementation plan archives
docs/
  ROADMAP.md            — Phase history, acceptance criteria, infeasible items, future directions
  Research.md           — FantasyPros vs. HEATER gap analysis: 20 missing features, prioritized implementation roadmap
.github/
  workflows/ci.yml      — CI pipeline (lint + test + build)
  workflows/refresh.yml — Scheduled daily data refresh (9:17 UTC) + manual trigger
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

### Draft Recommendation Engine (`src/draft_engine.py`)
- **DraftRecommendationEngine** orchestrator with 3 modes (Quick/Standard/Full)
- **8-stage enhancement pipeline:** park factors → Bayesian blend → injury probability → Statcast delta → FIP correction → contextual factors → category balance → ML ensemble
- **Enhanced pick_score** = base_sgp × multiplicative_factors + additive_bonuses
  - Multiplicative (clamped [0.5, 1.5]): category_balance × park_factor × (1 - injury_prob × 0.3) × (1 + statcast_delta × 0.15) × platoon_factor × contract_year
  - Additive: streaming_penalty + lineup_protection + closer_bonus + ml_correction × 0.1 + flex_bonus
- **Category balance** — Normal PDF weighting per category; draft progress scales: early compresses toward 1.0 (BPA), late amplifies gaps (fill needs)
- **Contextual factors** — Closer hierarchy (SV-based role + scarcity bonus), platoon risk (The Book: LHB 0.90), lineup protection (PA bonus by batting order), schedule strength (division park factors), contract year (+2% hitters)
- **BUY/FAIR/AVOID** — Classification based on enhanced_rank vs ADP_rank gap, threshold scales by draft phase
- **ML ensemble** — Optional XGBoost residual prediction (actual - projected), 10% weight, graceful fallback
- **News sentiment** — Keyword-based scoring (-1.0 to +1.0), ready for news API integration
- **Timing instrumentation** — Per-stage timing via `engine.timing` dict
- **Injury badges** — Hero card shows CSS dot health indicators (green/yellow/red), age flags for aging curves, workload flags for IP spikes. No emoji — all icons are inline SVGs from `PAGE_ICONS` dict.
- **P10/P90 range bars** — Floor/ceiling projections displayed on hero pick and alternatives
- **Opponent intel tab** — Threat alerts when opponents need your target position, plus full opponent roster/needs breakdown
- **Practice mode** — Ephemeral DraftState clone for what-if scenarios, resets on refresh or button click
- **Yahoo OAuth** — Setup wizard Settings step shows Yahoo connect expander when env vars are set. Uses out-of-band (oob) flow: user clicks link, authorizes on Yahoo, pastes verification code back in the app.
- **Yahoo auto-reconnect** — On every app restart, `_try_reconnect_yahoo()` loads the saved token from `data/yahoo_token.json`, recreates `YahooFantasyClient`, and authenticates automatically. Token auto-refresh is handled by yfpy/yahoo-oauth (refresh_token never expires). Users authenticate once and never need to reconnect.

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

### Trade Analyzer Engine (`src/engine/`)
Phase 1 SGP-based evaluation pipeline (7 modules):
1. **Peer-group z-scores** — Hitters scored against hitters, pitchers against pitchers (IP>0). ERA/WHIP sign-flipped.
2. **Standings-based SGP** — Median gap between adjacent teams replaces static denominators. Falls back to defaults when standings unavailable.
3. **Marginal elasticity** — `1/gap_to_next_team` per category. Close gap = high marginal value, dominant = near-zero.
4. **Punt detection** — Category is PUNT when: (a) cannot gain any standings position in remaining weeks, AND (b) ranked 10th or worse. Punted categories get zero weight.
5. **LP-constrained lineup totals** — PuLP `LineupOptimizer` assigns best players to starting slots (C/1B/2B/3B/SS/3OF/2Util/2SP/2RP/4P = 18 starters). Only starters' stats feed SGP delta — bench players are excluded, preventing phantom production inflation in uneven trades. Falls back to `_roster_category_totals()` when PuLP unavailable.
5b. **Roster cap enforcement** — Yahoo 23-man limit modeled: 2-for-1 trades trigger forced drop (lowest-SGP bench player removed); 1-for-2 trades trigger FA pickup (best available FA added, capped at type-specific median SGP when no league rosters loaded). Ensures realistic post-trade rosters.
5c. **Category replacement cost penalty** — Scans FA pool for each counting stat where trade is negative. `unrecoverable = max(0, raw_loss - best_FA)`, penalty = `(unrecoverable / SGP_denom) * 0.5`. Rate stats (AVG/ERA/WHIP) excluded. Punted categories skipped.
6. **Weighted SGP delta** — Per-category `(after - before) / SGP_denom * marginal_weight`. Inverse stats (ERA/WHIP) sign-flipped.
7. **Grade** — Surplus SGP maps to A+ (>2.0) through F (≤-1.0). Verdict: ACCEPT if surplus > 0.
- **Graceful fallback** — Trade Analyzer page tries Phase 1 engine first, falls back to legacy `analyze_trade()` from `src/in_season.py`
- **Risk flags** — Injured players received, elite players given away (SGP>3.0), weak non-punt categories (rank≥8)

**Phase 2 (stochastic, when `enable_mc=True`):**
8. **Bayesian Model Averaging** — Posterior weights for each projection system based on P(YTD | System_i). Systems closer to reality get higher weight. Variance = within-model + between-model.
9. **KDE marginals** — Non-Gaussian stat distributions via kernel density estimation. Normal fallback when <20 historical data points. Stat bounds clipping (AVG: 0.100–0.400, ERA: 0.50–12.00).
10. **Gaussian copula** — Correlated stat sampling via Cholesky decomposition of empirical 12×12 correlation matrix (HR↔RBI: 0.85, ERA↔WHIP: 0.90, OBP↔AVG: 0.80, L↔ERA: 0.50). Nearest-PD correction for non-PD matrices.
11. **Paired Monte Carlo** — 10K sims with identical seeds for before/after (variance reduction). Produces: mc_mean, mc_std, p5/p25/median/p75/p95, prob_positive, VaR5, CVaR5, Sharpe ratio, 95% CI. Grade via composite = mean*0.4 + sharpe*0.3 + kelly_approx*0.3.

**Phase 3 (signal intelligence, `src/engine/signals/`):**
12. **Statcast harvesting** — Pitch-level data from Baseball Savant via pybaseball. Aggregated into rolling 14-day features: exit velocity (mean/p90), barrel%, hard hit%, xBA, xwOBA, whiff%, chase rate (batters); fastball speed/spin, K%, BB%, GB% (pitchers).
13. **Signal decay** — Exponential recency weighting with sport-specific half-lives: EV/barrel (35d), spin rate (46d), plate discipline (58d), traditional stats (87d), sprint speed (139d). λ=0 for counting stats (no decay).
14. **Kalman filter** — State-space model separating true talent from observation noise. Sample-size-aware observation variance (more PAs → lower noise → trust data more). Process variance models how much true talent can drift per time step.
15. **Regime detection** — BOCPD (Adams & MacKay 2007) detects structural changepoints via run length mode drops. 4-state HMM (Elite/Above/Below/Replacement) provides soft regime probabilities for projection blending. Graceful fallback when hmmlearn unavailable.

**Phase 4 (context engine, `src/engine/context/`, `enable_context=True`):**
16. **Log5 matchup** — Odds-ratio method for batter-pitcher matchup prediction. Combines batter rate, pitcher rate, and league average via `M_odds = (B_odds * P_odds) / L_odds`. Park factor and weather (temp/wind) adjustments. Game-level stat projections by lineup slot (PA scaling: leadoff 4.65 → 9-hole 3.85).
17. **Injury stochastic process** — Weibull-distributed injury durations by body part (hamstring shape=1.8/scale=18, UCL shape=3.0/scale=365). Frailty multiplier from health score (`1/max(hs, 0.5)`, max 2.0). Season availability sampling for MC integration via `sample_season_availability()`.
18. **Enhanced bench option value** — Extends simple streaming value (0.166 SGP/week) with 4 components: streaming, hot FA pickup (15%/week × 0.5 SGP), roster flexibility premium (multi-position eligibility normalized to [0,1]), and injury replacement cushion. Returns component breakdown dict.
19. **Roster concentration risk** — Herfindahl-Hirschman Index (HHI = Σshare²) measuring team exposure. PA-weighted for hitters, IP-weighted for pitchers. Penalty = `(HHI - 0.15) * 3.0` when above 0.15 threshold. `compute_concentration_delta()` compares before/after trade rosters. Team alias normalization (WSN→WSH, AZ→ARI).

**Phase 5 (game theory, `src/engine/game_theory/`, `enable_game_theory=True`):**
20. **Opponent valuations** — Estimates each opponent's willingness-to-pay based on THEIR category needs. Market clearing price via Vickrey auction (second-highest bidder = Nash equilibrium). Player demand count (teams valuing player above 0.5 SGP threshold).
21. **Adverse selection** — Bayesian P(flaw|offered) discount. Prior P(flaw)=0.15, P(offered|flaw)=0.60, P(offered|ok)=0.20. Calibrated from manager trade history when ≥3 trades available. Discount factor capped at 0.75 (max 25% haircut).
22. **Dynamic programming** — Bellman rollout approximates future trade option value via MC. Discount factor γ depends on playoff probability (contending=0.98, bubble=0.95, rebuilding=0.85). Roster balance score affects future trade opportunity probability.
23. **Sensitivity + counter-offers** — Category sensitivity ranking by absolute SGP impact. Breakeven gap analysis with vulnerability classification (robust/moderate/fragile/razor-thin). Counter-offer generation tries swapping each given player with roster alternatives, returning top 3 improvements.

**Phase 6 (production, `src/engine/production/`):**
24. **Convergence diagnostics** — Effective Sample Size (ESS) via FFT autocorrelation, split-R̂ (Gelman-Rubin), running mean stability normalized by sample std. Quality classification: excellent (ESS>1000 + all pass), good, marginal, poor.
25. **Precomputation cache** — In-memory cache with TTL staleness tracking. `get_or_compute()` pattern for lazy refresh. Default TTLs: copula 24h, SGP 1h, gap analysis 1h, market values 30min. Module-level singleton via `get_trade_cache()`.
26. **Adaptive simulation scaling** — 1K (quick) → 10K (standard) → 50K (production) → 100K (full). Scales by trade complexity (+5K per extra player beyond 1-for-1). Time budget cap ensures interactive responsiveness. `recommend_n_sims()` uses current ESS to suggest optimal count.

### In-Season Algorithms
- **Trade Analyzer (legacy):** Roster swap → projected season totals (YTD + ROS) → park-adjusted SGP delta → MC simulation (200 sims) → verdict with confidence %. Now includes injury badges for both sides and P10/P90 risk assessment.
- **Player Compare:** ROS projections → Z-score normalization across 12 categories → composite weighted score, optional marginal SGP team impact. Now includes health badges and projection confidence (P10-P90 range width).
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

### Database Tables (16)

| Group | Tables |
|-------|--------|
| Draft | `projections`, `adp`, `league_config`, `draft_picks`, `blended_projections`, `player_pool` |
| In-Season | `season_stats`, `ros_projections`, `league_rosters`, `league_standings`, `park_factors`, `refresh_log` |
| Plan 3 | `injury_history`, `transactions` |
| FantasyPros Parity | `player_tags`, `leagues` |

## Data Sources

- **Players:** Auto-fetched from MLB Stats API on every launch (750+ active + extended 40-man/spring training); staleness: 7 days
- **Draft projections:** 7 systems — Steamer, ZiPS, Depth Charts (FanGraphs JSON API) + ATC, THE BAT, THE BAT X (FanGraphs JSON API) + Marcel (local computation); staleness: 7 days
- **ADP:** 3 sources — FanGraphs Steamer (primary), FantasyPros ECR (scraped), NFBC (scraped); staleness: 24 hours
- **Current stats:** MLB Stats API (`MLB-StatsAPI` package, auto-refreshed on launch); staleness: 1 hour
- **Historical stats:** 3 years (2023-2025) from MLB Stats API for injury modeling; staleness: 30 days
- **Park factors:** Hardcoded FanGraphs 2024 values (30 teams) in `data_bootstrap.py`; staleness: 30 days
- **Injury history:** Derived from historical stats (games played vs available); staleness: 30 days
- **League rosters/standings:** Yahoo Fantasy API sync (optional, auto-syncs when connected); staleness: 6 hours
- **Depth charts:** FanGraphs depth chart scraper with role classification (starter/platoon/closer/setup/committee); staleness: 7 days
- **Contract data:** Baseball Reference free agent list scraper for contract year detection; staleness: 30 days
- **News/transactions:** MLB Stats API transaction feed (7-day window), mapped to player_ids via fuzzy matching; staleness: 6 hours

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
# pa, ab, h, r, hr, rbi, sb, avg, obp, bb, hbp, sf, ip, w, l, sv, k, era, whip, er, bb_allowed, h_allowed, adp

# --- Trade Engine Phase 1 APIs (src/engine/) ---

# Trade evaluator (src/engine/output/trade_evaluator.py)
evaluate_trade(giving_ids, receiving_ids, user_roster_ids, player_pool,
               config=None, user_team_name=None, weeks_remaining=16,
               enable_mc=False, enable_context=True, enable_game_theory=True)
# Returns: {grade, surplus_sgp, category_impact, category_analysis, punt_categories,
#           bench_cost, replacement_penalty, replacement_detail,
#           risk_flags, verdict, confidence_pct, before_totals, after_totals,
#           giving_players, receiving_players, total_sgp_change, mc_mean, mc_std,
#           concentration_hhi_before, concentration_hhi_after, concentration_delta,
#           concentration_penalty, bench_option_detail,
#           adverse_selection, market_values, sensitivity_report,
#           lineup_constrained, drop_candidate, fa_pickup}
grade_trade(surplus_sgp: float) -> str  # A+ through F

# LP-constrained lineup helpers (src/engine/output/trade_evaluator.py)
_lineup_constrained_totals(roster_ids, player_pool, config) -> tuple[dict, list[dict]]
# Returns: (totals_dict, starter_details_list)  — only starters' stats, bench excluded
_find_drop_candidate(bench_ids, player_pool, sgp_calc) -> int | None
# Returns: player_id of lowest-SGP bench player to drop in 2-for-1 trades
_find_fa_pickup(player_pool, sgp_calc, need_hitter, median_sgp_cap, exclude_ids) -> int | None
# Returns: player_id of best FA to add in 1-for-2 trades, capped at median SGP
_compute_median_sgp_cap(player_pool, sgp_calc, is_hitter) -> float
# Returns: median SGP for hitter/pitcher pool (cap for FA pickup when no rosters loaded)
ROSTER_CAP: int = 23  # Yahoo 23-man roster limit

# Category replacement cost penalty (src/engine/output/trade_evaluator.py)
_compute_replacement_penalty(before_totals, after_totals, player_pool, config, category_weights, punt_categories)
# Returns: (total_penalty: float, detail: dict[str, dict])
#   detail per-cat: {raw_loss, best_fa, best_fa_name, unrecoverable, sgp_penalty} or {skipped: str}
COUNTING_CATEGORIES: set[str] = {"R", "HR", "RBI", "SB", "W", "SV", "K"}
FA_TURNOVER_DISCOUNT: float = 0.5

# Z-score + SGP valuation (src/engine/portfolio/valuation.py)
compute_player_zscores(player_pool, config=None) -> DataFrame  # adds z_{cat} + z_composite columns
compute_sgp_from_standings(standings, config=None) -> dict[str, float]  # standings-based SGP denoms
compute_player_vorp(player_pool, config=None) -> DataFrame  # adds vorp column
build_valuation_context(config=None) -> dict  # full context from DB state

# Category analysis (src/engine/portfolio/category_analysis.py)
compute_marginal_sgp(your_totals, all_team_totals, categories=None) -> dict[str, float]
category_gap_analysis(your_totals, all_team_totals, your_team_id, weeks_remaining=16) -> dict[str, dict]
build_standings_totals(standings: DataFrame) -> dict[str, dict[str, float]]
compute_category_weights_from_analysis(analysis) -> dict[str, float]  # punts = 0.0

# Lineup optimizer wrapper (src/engine/portfolio/lineup_optimizer.py)
compute_optimal_lineup_value(roster, config, category_weights=None) -> float
compute_lineup_delta(before_roster, after_roster, config, category_weights=None) -> float
bench_option_value(weeks_remaining=16, streaming_sgp_per_week=0.15) -> float

# Projection client (src/engine/projections/projection_client.py)
fuzzy_match_player(name, candidates, name_col="name", threshold=0.7) -> str | None
get_ros_projections(player_ids) -> DataFrame
resolve_trade_players(giving_names, receiving_names, player_pool) -> tuple[list[int], list[int]]

# --- Trade Engine Phase 2 APIs (src/engine/) ---

# BMA (src/engine/projections/bayesian_blend.py)
bayesian_model_average(ytd_stats, projections, prior_weights=None)
# Returns: (posterior_weights, blended_projection, blended_variance)
compute_bma_for_player(player_ytd, system_projections) -> dict[str, float]
compute_bma_variance(player_ytd, system_projections) -> dict[str, float]
SYSTEM_FORECAST_SIGMA: dict[str, dict[str, float]]  # steamer/zips/depthcharts × 10 stats

# KDE marginals (src/engine/projections/marginals.py)
PlayerMarginal(stat_name, projected_value, variance, historical_values=None)
# .ppf(quantile) -> stat_value  (inverse CDF for copula integration)
# .sample(n, rng) -> np.ndarray
# .uses_kde -> bool
build_player_marginals(projected_stats, variances, historical_by_stat=None)

# Copula (src/engine/portfolio/copula.py)
GaussianCopula(correlation=None)  # defaults to 12×12 DEFAULT_CORRELATION
# .sample(n, rng) -> np.ndarray of shape (n, 10) in [0, 1]
sample_correlated_stats(copula, player_marginals, n=1, rng=None) -> np.ndarray
fit_copula_from_data(player_seasons) -> GaussianCopula

# Paired MC (src/engine/monte_carlo/trade_simulator.py)
run_paired_monte_carlo(before_roster_stats, after_roster_stats,
    before_marginals=None, after_marginals=None, copula=None,
    all_team_totals=None, sgp_denominators=None,
    n_sims=10000, seed=42, weeks_remaining=16)
# Returns: {mc_mean, mc_std, mc_median, p5..p95, prob_positive,
#           var5, cvar5, sharpe, grade, verdict, confidence_pct,
#           confidence_interval, surplus_distribution, n_sims}
build_roster_stats(player_ids, player_pool) -> dict[str, dict[str, float]]

# --- Trade Engine Phase 3 APIs (src/engine/signals/) ---

# Statcast harvesting (src/engine/signals/statcast.py)
fetch_batter_statcast(player_name, start_date, end_date) -> pd.DataFrame
fetch_pitcher_statcast(player_name, start_date, end_date) -> pd.DataFrame
aggregate_batter_statcast(df) -> dict[str, float]  # ev_mean, ev_p90, barrel_pct, hard_hit_pct, xba, xwoba, whiff_pct, chase_rate
aggregate_pitcher_statcast(df) -> dict[str, float]  # ff_avg_speed, ff_spin_rate, k_pct, bb_pct, gb_pct, xba_against, xwoba_against
compute_rolling_features(df, window_days=14, step_days=7, is_pitcher=False) -> list[dict]

# Signal decay (src/engine/signals/decay.py)
DECAY_LAMBDAS: dict[str, float]  # 7 categories: batted_ball=0.020, spin=0.015, discipline=0.012, ...
FEATURE_DECAY_MAP: dict[str, str]  # feature_name -> decay category
decay_weight(obs_date, reference_date, lambda_param) -> float  # e^(-λ * days_diff)
apply_decay_weights(observations, feature_name, reference_date) -> tuple[np.ndarray, np.ndarray]
weighted_mean(values, weights) -> float
weighted_variance(values, weights) -> float
get_feature_lambda(feature_name) -> float
half_life_days(lambda_param) -> float  # ln(2) / lambda

# Kalman filter (src/engine/signals/kalman.py)
kalman_true_talent(observations, obs_variance, process_variance, prior_mean, prior_variance)
# Returns: (filtered_means, filtered_variances) — np.ndarray pair
observation_variance(feature_name, sample_size) -> float
get_process_variance(feature_name) -> float
run_kalman_for_feature(rolling_data, feature_name, prior_mean=None)
# Returns: {filtered_mean, filtered_var, kalman_gain_final}

# Regime detection (src/engine/signals/regime.py)
BOCPD(hazard_lambda=200, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
# .update(x) -> (changepoint_prob, run_length_probs)
# .reset() -> None
detect_changepoints(time_series, hazard_lambda=200, threshold=0.7)
# Returns: {changepoint_indices, changepoint_probs, last_changepoint, current_regime_length}
fit_player_hmm(obs_matrix, n_states=4) -> tuple[model | None, state_probs]
regime_conditional_projection(current_probs, projections_by_state) -> dict[str, float]
classify_regime_simple(recent_xwoba, season_xwoba, league_avg_xwoba=0.315) -> tuple[str, np.ndarray]
REGIME_STATES: list[str]  # ["Elite", "Above-avg", "Below-avg", "Replacement"]
HMM_AVAILABLE: bool  # True if hmmlearn is installed

# --- Trade Engine Phase 4 APIs (src/engine/context/) ---

# Log5 matchup (src/engine/context/matchup.py)
log5_matchup(batter_rate, pitcher_rate, league_avg) -> float  # odds-ratio matchup rate
park_adjust(stat_value, park_factor) -> float  # multiplicative park adjustment
game_projection(batter_xwoba, pitcher_xwoba_against=None, park_factor=1.0,
    lineup_slot=5, league_avg_woba=0.315, temp_f=None, wind_out_mph=None) -> dict
# Returns: {hr, r, rbi, sb, h, ab, pa}
matchup_adjustment_factor(player_xwoba=None, opponent_xwoba_against=None,
    park_factor=1.0, league_avg=0.315) -> float  # multiplicative factor (1.0=neutral)
LINEUP_SLOT_PA: dict[int, float]  # 1:4.65 → 9:3.85

# Injury stochastic process (src/engine/context/injury_process.py)
sample_injury_duration(injury_type="other", frailty=1.0, rng=None) -> int  # days
frailty_from_health_score(health_score) -> float  # 1.0/max(hs, 0.5)
estimate_injury_probability(health_score, age=None, is_pitcher=False, horizon_days=30) -> float
sample_season_availability(health_score, age=None, is_pitcher=False,
    weeks_remaining=16, rng=None) -> float  # fraction in [0, 1]
sample_availability_batch(health_scores, ages=None, is_pitcher_flags=None,
    weeks_remaining=16, n_sims=1, rng=None) -> np.ndarray  # (n_sims, n_players)
INJURY_DURATION: dict[str, dict[str, float]]  # 6 body parts: hamstring, oblique, ucl, shoulder, back, other

# Enhanced bench value (src/engine/context/bench_value.py)
enhanced_bench_option_value(weeks_remaining=16, streaming_sgp_per_week=0.15,
    roster_flexibility=0.0, injury_replacement_value=0.0) -> dict
# Returns: {streaming, hot_fa, flexibility, injury_cushion, total}
compute_roster_flexibility(roster_df) -> float  # normalized [0, 1]
compute_injury_replacement_value(roster_df, bench_count=5, avg_health_score=0.85,
    weeks_remaining=16) -> float  # expected SGP saved

# Concentration risk (src/engine/context/concentration.py)
roster_concentration_hhi(roster_df) -> float  # HHI in [0, 1]
concentration_risk_penalty(hhi, threshold=0.15, scale=3.0) -> float  # SGP penalty
compute_concentration_delta(before_df, after_df) -> dict
# Returns: {before_hhi, after_hhi, delta, penalty_before, penalty_after, penalty_delta}
team_exposure_breakdown(roster_df) -> list[dict]  # [{team, count, share}, ...]
TEAM_ALIASES: dict[str, str]  # WSN→WSH, AZ→ARI, etc.

# --- Trade Engine Phase 5 APIs (src/engine/game_theory/) ---

# Opponent valuation (src/engine/game_theory/opponent_valuation.py)
estimate_opponent_valuations(player_projections, all_team_totals, your_team_id,
    sgp_denominators=None) -> dict[str, float]  # {team_name: valuation_sgp}
market_clearing_price(valuations) -> float  # Nash equilibrium = 2nd highest bid
player_market_value(player_projections, all_team_totals, your_team_id,
    sgp_denominators=None) -> dict  # {valuations, market_price, max_bidder, max_bid, demand}
get_player_projections_from_pool(player_id, player_pool) -> dict[str, float]

# Adverse selection (src/engine/game_theory/adverse_selection.py)
adverse_selection_discount(offering_manager_history=None, p_flaw_prior=0.15) -> float  # (0.75, 1.0]
compute_discount_for_trade(receiving_player_count=1, offering_manager_history=None) -> dict
# Returns: {discount_factor, p_flaw, p_flaw_given_offered, sgp_adjustment, total_sgp_adjustment, risk_level}

# Dynamic programming (src/engine/game_theory/dynamic_programming.py)
get_gamma(playoff_probability) -> float  # 0.85-0.98
estimate_playoff_probability(standings_rank, num_teams=12, weeks_remaining=16) -> float
bellman_rollout(immediate_surplus, weeks_remaining=16, playoff_probability=0.50,
    roster_balance_before=0.0, roster_balance_after=0.0,
    n_lookahead=2, n_sims=200, seed=42) -> dict
# Returns: {immediate, future_before, future_after, option_value, gamma, total_value}
compute_roster_balance(roster_category_ranks, num_teams=12) -> float  # [-1, 1]

# Sensitivity (src/engine/game_theory/sensitivity.py)
category_sensitivity(category_impact, category_weights=None) -> list[dict]  # sorted by |impact|
player_sensitivity(giving_ids, receiving_ids, ..., evaluate_fn, base_surplus) -> list[dict]
suggest_counter_offers(giving_ids, receiving_ids, ..., evaluate_fn, base_surplus,
    max_suggestions=3) -> list[dict]  # [{swap_out, swap_in, improvement, new_grade}]
trade_sensitivity_report(category_impact, category_weights=None, surplus_sgp=0.0) -> dict
# Returns: {category_ranking, biggest_driver, biggest_drag, breakeven_gap, vulnerability}

# --- Trade Engine Phase 6 APIs (src/engine/production/) ---

# Convergence diagnostics (src/engine/production/convergence.py)
effective_sample_size(samples) -> float  # ESS via FFT autocorrelation
split_rhat(samples) -> float  # split-R̂ (near 1.0 = converged)
running_mean_stability(samples, window=500) -> float  # lower = more stable
check_convergence(samples) -> dict  # {ess, rhat, stability, converged, quality}
recommend_n_sims(current_ess, target_ess=500, current_n=10000) -> int  # capped at 100K

# Cache (src/engine/production/cache.py)
TradeEvalCache()  # In-memory cache with TTL
# .get(key) -> value | None
# .set(key, value, ttl=3600)
# .get_or_compute(key, compute_fn, ttl=None) -> value
# .invalidate(key) -> bool
# .clear() -> int
# .stats() -> dict
get_trade_cache() -> TradeEvalCache  # Module-level singleton
reset_trade_cache() -> None

# Simulation config (src/engine/production/sim_config.py)
compute_adaptive_n_sims(n_giving=1, n_receiving=1, mode="standard",
    time_budget_s=None) -> int  # 1K-100K
get_sim_mode(interactive=True) -> str  # "standard" or "production"
estimate_runtime_seconds(n_sims) -> float
sim_config_summary(n_giving, n_receiving, mode, time_budget_s=None) -> dict

# --- Enhanced Lineup Optimizer APIs (src/optimizer/) ---

# Pipeline orchestrator (src/optimizer/pipeline.py)
LineupOptimizerPipeline(roster, mode="standard", alpha=0.5, weeks_remaining=16, config=None)
# .optimize(standings=None, team_name=None, h2h_opponent_totals=None,
#           my_totals=None, week_schedule=None, park_factors=None,
#           free_agents=None, ytd_totals=None, target_totals=None,
#           weekly_rates=None, roto_rank=None, h2h_wins=None, h2h_losses=None) -> dict
# Returns: {lineup, category_weights, h2h_analysis, streaming_suggestions,
#           risk_metrics, maximin_comparison, recommendations, timing, mode, matchup_adjusted}
MODE_PRESETS: dict[str, dict]  # quick/standard/full with enable flags + n_scenarios + risk_aversion

# Enhanced projections (src/optimizer/projections.py)
build_enhanced_projections(roster, config=None, enable_bayesian=True, enable_kalman=True,
    enable_regime=True, enable_statcast=False, enable_injury=True, weeks_remaining=16) -> DataFrame

# Matchup adjustments (src/optimizer/matchup_adjustments.py)
get_weekly_schedule(days_ahead=7) -> list[dict]  # MLB Stats API schedule
platoon_adjustment(batter_hand, pitcher_hand, batter_avg=0.260, regression_pa=1000) -> float
park_factor_adjustment(player_team, opponent_team, park_factors, is_hitter=True) -> float
weather_hr_adjustment(temp_f) -> float  # +0.9%/degree above 72°F
compute_weekly_matchup_adjustments(roster, week_schedule, park_factors) -> DataFrame

# H2H engine (src/optimizer/h2h_engine.py)
compute_h2h_category_weights(my_totals, opp_totals, category_variances=None) -> dict[str, float]
estimate_h2h_win_probability(my_totals, opp_totals, category_variances=None) -> dict
# Returns: {per_category: {cat: p_win}, expected_wins, overall_win_prob}
default_category_variances() -> dict[str, float]

# Non-linear SGP (src/optimizer/sgp_theory.py)
nonlinear_marginal_sgp(your_total, opponent_totals, sigma, is_inverse=False) -> float
compute_nonlinear_weights(standings, team_name, sigmas=None) -> dict[str, float]
slope_sgp_denominators(standings) -> dict[str, float]
default_category_sigmas() -> dict[str, float]

# Streaming (src/optimizer/streaming.py)
compute_streaming_value(pitcher, weekly_games, team_park_factor=1.0, ...) -> dict
rank_streaming_candidates(free_agent_pitchers, park_factors=None, ...) -> list[dict]
quantify_two_start_value(pitcher_stats, team_era=4.0, team_whip=1.25, ...) -> dict
optimal_streaming_schedule(candidates, max_adds=7) -> list[dict]

# Scenario generator (src/optimizer/scenario_generator.py)
generate_stat_scenarios(roster, n_scenarios=200, seed=42) -> np.ndarray  # (n, players, 10)
mean_variance_adjustments(roster, lambda_risk=0.15) -> dict[int, float]
build_cvar_constraints(scenarios, player_indices, category_weights, alpha=0.05) -> dict
compute_scenario_lineup_values(scenarios, assignments, category_weights) -> np.ndarray

# Multi-period (src/optimizer/multi_period.py)
rolling_horizon_optimization(weekly_projections, category_weights, horizon_weeks=4, discount=0.95) -> dict
season_balance_weights(ytd_totals, target_totals, weeks_remaining, weekly_rates) -> dict[str, float]

# Dual objective (src/optimizer/dual_objective.py)
blend_h2h_roto_weights(h2h_weights, roto_weights, alpha=0.5) -> dict[str, float]
recommend_alpha(weeks_remaining, roto_rank=None, h2h_record_wins=None, h2h_record_losses=None) -> float

# Advanced LP (src/optimizer/advanced_lp.py)
maximin_lineup(roster, scale_factors, category_weights, active_categories=None) -> dict
epsilon_constraint_lineup(roster, scale_factors, primary_category, epsilon_bounds, ...) -> dict
stochastic_mip(roster, scenarios, category_weights, scale_factors, ...) -> dict

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
PAGE_ICONS: dict[str, str]  # ~22 inline SVG icons keyed by name ("logo", "logo_lg", "baseball", "fire", "accept", etc.)
METRIC_TOOLTIPS: dict[str, str]  # Educational tooltip text for every metric (sgp, vorp, survival, etc.)
THEME: dict  # Single light-mode palette: bg=#f4f5f0, primary=#e63946, hot=#ff6d00, gold=#ffd60a, green=#2d6a4f, sky=#457b9d, purple=#6c63ff
T = THEME  # Direct dict alias — no proxy needed without dark mode. Backward compat: T["amber"]→#e63946, T["teal"]→#457b9d
get_theme() -> dict  # Stub — always returns THEME (kept for backward compat)
render_theme_toggle()  # No-op stub (kept for backward compat)
inject_custom_css()  # Injects full CSS (1500+ lines) with glassmorphism, 3D buttons, kinetic typography, 7 animations, orange sidebar, bold titles, contrasting data tables

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

# --- Draft Recommendation Engine APIs ---

# DraftRecommendationEngine (src/draft_engine.py)
DraftRecommendationEngine(config: LeagueConfig, mode: str = "standard")
# .enhance_player_pool(player_pool, draft_state, park_factors=None) -> DataFrame  # adds 15 columns
# .recommend(player_pool, draft_state, top_n=8, n_simulations=300, park_factors=None) -> DataFrame
# .timing -> dict  # per-stage timing breakdown
MODE_PRESETS: dict[str, dict]  # quick/standard/full with enable flags for all 8 stages

# Draft analytics (src/draft_analytics.py)
compute_category_balance(roster_totals, all_team_totals, draft_progress, config=None) -> dict[str, float]
compute_opportunity_cost(candidate, available_pool, survival=0.5) -> float
compute_streaming_draft_value(pitcher, config=None) -> float  # -0.5 to 0.0 penalty
compute_buy_fair_avoid(enhanced_rank, adp_rank, current_pick, total_picks=276) -> str  # "BUY"|"FAIR"|"AVOID"
DEFAULT_SIGMAS: dict[str, float]  # per-category standard deviations for Normal PDF weighting

# Contextual factors (src/contextual_factors.py)
detect_closer_role(player) -> dict  # {role, confidence, draft_bonus}
compute_platoon_risk(player_bats: str) -> float  # 0.90-1.0
compute_lineup_protection(player) -> float  # 0.0-0.3 SGP bonus
compute_schedule_strength(player_team, park_factors) -> float  # ~0.95-1.05
contract_year_boost(is_contract_year, is_hitter) -> float  # 1.0 or 1.02

# ML ensemble (src/ml_ensemble.py)
DraftMLEnsemble(model_path=None)
# .predict_batch(player_pool) -> Series  # ml_correction column
# .train(historical_data, target_col="residual") -> dict  # {status, metrics}
# .is_ready -> bool
get_ml_ensemble(model_path=None) -> DraftMLEnsemble  # factory

# News sentiment (src/news_sentiment.py)
compute_news_sentiment(news_items: list[str]) -> float  # -1.0 to 1.0
analyze_news_sentiment(news_items) -> SentimentResult  # {score, positive_count, negative_count, confidence, high_impact_flags}
sentiment_adjustment(score, weight=0.05) -> float  # multiplicative factor
batch_sentiment(player_news: dict[int, list[str]]) -> dict[int, float]

# --- In-Season Analytics APIs ---

# Trade value chart (src/trade_value.py)
compute_trade_values(player_pool, config=None, standings=None, weeks_remaining=16) -> DataFrame
# Returns: player_id, name, team, positions, is_hitter, total_sgp, vorp, g_score, sgp_surplus, trade_value (0-100), dollar_value, tier, rank, pos_rank
compute_contextual_values(trade_values, user_totals, all_team_totals, user_team_name, config=None) -> DataFrame
# Returns: trade_values + contextual_value, contextual_tier columns
filter_by_position(trade_values, position) -> DataFrame
assign_tier(trade_value) -> str  # "Elite", "Star", "Solid Starter", "Flex", "Replacement"
compute_g_score_adjustment(per_cat_sgp, pool_sigma, config=None) -> float

# Two-start pitcher planner (src/two_start.py)
identify_two_start_pitchers(days_ahead=7, team_era=4.00, team_whip=1.25, team_ip=55.0, player_pool=None) -> list[dict]
# Returns: [{pitcher_name, team, num_starts, starts, avg_matchup_score, rate_damage_per_start, rate_damage_weekly, two_start_value, streaming_value}]
compute_pitcher_matchup_score(pitcher_stats, opponent_team_stats=None, park_factor=1.0, is_home=True) -> float  # 0-10
rate_stat_damage(pitcher_era, pitcher_whip, pitcher_ip, team_era, team_whip, team_ip) -> dict  # {era_change, whip_change}
fetch_team_batting_stats(season=None) -> dict[str, dict[str, float]]  # thread-safe cached
clear_team_batting_cache() -> None

# Start/sit advisor (src/start_sit.py)
start_sit_recommendation(player_ids, player_pool, config=None, weekly_schedule=None, park_factors=None,
    my_weekly_totals=None, opp_weekly_totals=None, standings=None, team_name=None) -> dict
# Returns: {recommendation: player_id, confidence: float, confidence_label: str, players: list[dict]}
classify_matchup_state(my_weekly_totals, opp_weekly_totals, config=None) -> str  # "winning", "close", "losing"
risk_adjusted_score(expected_value, p10_value, p90_value, matchup_state) -> float
compute_weekly_projection(player, weekly_schedule=None, park_factors=None) -> dict[str, float]

# Weekly matchup planner (src/matchup_planner.py)
compute_weekly_matchup_ratings(roster, weekly_schedule=None, park_factors=None, team_batting_stats=None, config=None) -> DataFrame
# Returns: player_id, name, team, positions, is_hitter, games, weekly_matchup_rating (1-10), matchup_tier, games_count, projected_stats_adjusted
compute_hitter_game_rating(player_stats, opposing_pitcher_stats, park_factor, is_home, batter_hand=None, pitcher_hand=None) -> dict
compute_pitcher_game_rating(pitcher_stats, opponent_team_stats, park_factor, is_home) -> dict
compute_all_ratings_with_percentiles(ratings_list) -> list[dict]  # [{percentile_rank, rating, tier}]
color_tier(percentile_rank) -> str  # "smash", "favorable", "neutral", "unfavorable", "avoid"

# Waiver wire (src/waiver_wire.py)
compute_add_drop_recommendations(user_roster_ids, player_pool, config=None, standings_totals=None,
    user_team_name=None, weeks_remaining=16, max_moves=3, max_fa_candidates=30, max_drop_candidates=5) -> list[dict]
# Returns: [{add_player_id, add_name, drop_player_id, drop_name, net_sgp_delta, category_impact, sustainability_score, reasoning}]
compute_drop_cost(player_id, roster_ids, player_pool, config=None) -> float
compute_net_swap_value(add_id, drop_id, roster_ids, player_pool, config=None) -> dict  # {net_sgp, category_deltas}
compute_sustainability_score(player) -> float  # 0.0-1.0
compute_babip(h, hr, ab, k, sf=0) -> float
classify_category_priority(user_totals, all_team_totals, user_team_name, weeks_remaining=16, config=None) -> dict[str, str]  # {cat: ATTACK|DEFEND|IGNORE}

# Trade finder (src/trade_finder.py)
find_trade_opportunities(user_roster_ids, player_pool, config=None, all_team_totals=None,
    user_team_name=None, league_rosters=None, weeks_remaining=16, max_results=20, top_partners=5) -> list[dict]
# Returns: [{giving_ids, receiving_ids, giving_names, receiving_names, user_sgp_gain, opponent_sgp_gain, acceptance_probability, composite_score, grade, opponent_team, complementarity}]
scan_1_for_1(user_roster_ids, opponent_roster_ids, player_pool, config=None) -> list[dict]
find_complementary_teams(user_team, all_team_totals, config=None, top_n=5) -> list[tuple[str, float]]
compute_team_vectors(all_team_totals, config=None) -> dict[str, np.ndarray]
cosine_dissimilarity(vec_a, vec_b) -> float  # [0, 2]
estimate_acceptance_probability(user_gain_sgp, opponent_gain_sgp, need_match_score=0.5) -> float

# Draft grader (src/draft_grader.py)
grade_draft(draft_picks, player_pool, config=None) -> dict
# Returns: {overall_grade, overall_score, team_value_score, pick_efficiency_score, category_balance_score, picks, steals, reaches, category_projections, strengths, weaknesses, total_sgp, expected_sgp}
classify_pick(player_sgp, expected_sgp, player_adp, pick_number, num_teams=12, primary_position="Util") -> tuple[str, float, float]
build_expected_sgp_curve(player_pool, config=None) -> list[float]
category_balance_score(category_projections) -> float  # [0, 1]
compute_category_projections(roster_ids, player_pool, config=None) -> dict[str, dict]

# --- FantasyPros Parity APIs ---

# Draft order (src/draft_order.py)
generate_draft_order(team_names: list[str], seed: int | None = None) -> list[str]
format_draft_order(order: list[str]) -> str

# Player tags (src/player_tags.py)
VALID_TAGS: list[str]  # ["Sleeper", "Target", "Avoid", "Breakout", "Bust"]
TAG_COLORS: dict[str, str]  # tag → hex color
add_tag(player_id: int, tag: str, note: str = "") -> bool
remove_tag(player_id: int, tag: str) -> bool
get_tags(player_id: int) -> list[dict]
get_all_tagged_players(tag_filter: str | None = None) -> pd.DataFrame
render_tag_badges_html(tags: list[dict]) -> str

# Pick predictor (src/pick_predictor.py)
POSITION_SHAPE_PARAMS: dict[str, float]  # C=1.4, SS=1.3, ..., OF=0.8
weibull_survival(picks_remaining, adp_distance, shape, scale) -> float
compute_survival_curve(player_adp, player_positions, current_pick, user_team_index, num_teams=12, num_rounds=23) -> list[dict]
# Returns: [{pick, round, p_available, p_normal, p_weibull}]

# Closer monitor (src/closer_monitor.py)
compute_job_security(hierarchy_confidence: float, projected_sv: float) -> float  # [0, 1]
get_security_color(security: float) -> str  # green/yellow/red hex
build_closer_grid(depth_data: dict, player_pool=None) -> list[dict]

# Points league (src/points_league.py)
SCORING_PRESETS: dict[str, dict]  # yahoo/espn/cbs with hitting + pitching weights
estimate_missing_batting_stats(row: pd.Series) -> dict
compute_fantasy_points(projections_df, hitting_weights, pitching_weights) -> pd.DataFrame
get_scoring_preset(name: str) -> tuple[dict, dict]

# Leaders (src/leaders.py)
compute_category_leaders(stats_df, config, min_pa=50, min_ip=20.0, top_n=20) -> dict
compute_points_leaders(stats_df, hitting_weights, pitching_weights, top_n=20) -> list
detect_breakouts(season_stats_df, preseason_df, threshold=1.5) -> pd.DataFrame

# ECR (src/ecr.py)
fetch_ecr_extended(position="overall") -> pd.DataFrame
blend_ecr_with_projections(valued_pool, ecr_df, ecr_weight=0.15) -> pd.DataFrame
compute_ecr_disagreement(proj_rank, ecr_rank, threshold=20) -> str | None
fetch_prospect_rankings(top_n=100) -> pd.DataFrame
filter_prospects_by_position(prospects_df, position) -> pd.DataFrame

# Cheat sheet (src/cheat_sheet.py)
CheatSheetConfig(title, sort_by, positions, show_tiers, top_n)
generate_cheat_sheet_html(player_pool, config=None) -> str
generate_cheat_sheet_pdf(html_string) -> bytes | None

# Live draft sync (src/live_draft_sync.py)
LiveDraftSyncer(yahoo_client, draft_state, player_pool, user_team_key="")
# .poll_and_sync() -> dict  # {new_picks, is_user_turn, error, draft_complete}

# Schedule grid (src/schedule_grid.py)
TIER_COLORS: dict[str, str]  # smash/favorable/neutral/unfavorable/avoid
build_schedule_grid(roster, weekly_schedule=None, matchup_ratings=None) -> dict
render_schedule_html(grid: dict) -> str

# Start/sit widget (src/start_sit_widget.py)
quick_start_sit(player_ids, player_pool, config=None) -> dict
compute_overlap_probability(mu1, sigma1, mu2, sigma2) -> float  # OVL coefficient

# Standings projection (src/standings_projection.py)
INVERSE_CATS: set[str]  # {"ERA", "WHIP", "L"}
generate_round_robin_schedule(team_names, n_weeks=22) -> list[list[tuple]]
compute_category_win_probability(mu_a, mu_b, sigma_a, sigma_b, is_inverse=False) -> float
simulate_season(team_totals, n_sims=1000, seed=42) -> pd.DataFrame
# Returns: team_name, mean_wins, mean_losses, mean_ties, win_p5, win_p95, playoff_pct

# Power rankings (src/power_rankings.py)
WEIGHTS: dict[str, float]  # roster_quality=0.40, category_balance=0.25, schedule_strength=0.15, injury_exposure=0.10, momentum=0.10
compute_roster_quality(team_trade_values, max_trade_values) -> float
compute_category_balance_score(team_zscores) -> float
compute_schedule_strength_index(team_name, roster_qualities, schedule=None) -> float
compute_injury_exposure(roster_trade_values, roster_health_scores) -> float
compute_momentum(recent_win_rate=None, season_win_rate=None) -> float
compute_power_rating(roster_quality, category_balance, schedule_strength, injury_exposure, momentum) -> float
compute_power_rankings(team_data) -> pd.DataFrame
bootstrap_confidence_interval(team_rating, component_stds=None, n_bootstrap=1000) -> tuple[float, float]

# IL manager (src/il_manager.py)
IL_DURATION_ESTIMATES: dict[str, float]  # IL10=2.0, IL15=3.5, IL60=10.0, DTD=0.5
classify_il_type(status_string) -> str
estimate_il_duration(il_type, position="") -> float
compute_lost_sgp(player_sgp, duration_weeks, weeks_remaining=22.0) -> float
find_best_replacement(vacated_positions, bench_players) -> dict | None
generate_il_alert(player, il_type, bench_players) -> dict

# League registry (src/league_registry.py)
register_league(platform, league_name, num_teams=12, scoring_format="h2h_categories", yahoo_league_id=None) -> str
get_league(league_id) -> LeagueInfo | None
list_leagues() -> list[LeagueInfo]
set_active_league(league_id) -> bool
get_active_league_id() -> str
delete_league(league_id) -> bool

# Bayesian streaming (src/optimizer/streaming.py — new functions)
compute_bayesian_stream_score(pitcher_era, pitcher_k9, pitcher_fip, opp_k_pct, opp_woba, ...) -> dict
# Returns: {stream_score, expected_k, expected_ip, expected_er, win_prob, risk_penalty, matchup_grade}
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
- **Yahoo auto-reconnect on restart** — `_try_reconnect_yahoo()` in `app.py` loads `data/yahoo_token.json` on every Streamlit restart, recreates `YahooFantasyClient`, and authenticates. Token refresh is automatic (yahoo-oauth checks `token_is_valid()` at 59-min mark and refreshes using `refresh_token`). Users connect once and never need to re-authorize.
- **Yahoo game_key resolution** — `_resolve_game_key()` in `yahoo_api.py` uses 3 strategies: (1) `get_game_key_by_season()`, (2) `get_all_yahoo_fantasy_game_keys()` enumeration, (3) `get_current_game_metadata()` fallback. MLB 2026 game_key = 469. Must set BOTH `self._query.game_id` AND `self._query.league_key` — belt and suspenders.
- **yfpy Roster iteration** — `Roster.__iter__()` yields attribute NAMES (strings), NOT players. Always use `getattr(roster, "players", None) or []` to get the actual player list. Never iterate a Roster object directly.
- **Python 3.14 bytes in yfpy** — `team.name` and `player.name.full` may return `bytes` instead of `str`. Always guard with `raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)`.
- **Health badges in st.dataframe()** — `get_injury_badge()` returns HTML `<span>` tags, but `st.dataframe()` renders raw text only. My Team page uses text labels ("Low Risk", "Moderate Risk", "High Risk") for the dataframe Health column. HTML badges are used only in `st.markdown(unsafe_allow_html=True)` contexts like the draft hero card.
- **yfpy `--no-deps` install needs extras** — Installing `yfpy>=17.0 --no-deps` strips `stringcase`, `yahoo-oauth`, and their transitive deps. Must also `pip install stringcase yahoo-oauth==2.1.1`. Similarly, `streamlit-oauth --no-deps` needs `httpx-oauth`. The python-dotenv version mismatch (1.2.2 vs pinned 1.1.1) is a pip warning but works at runtime.
- **yfpy token override warning** — When `yahoo_access_token_json` is provided to `YahooFantasySportsQuery`, do NOT also pass `yahoo_consumer_key`/`yahoo_consumer_secret` — the token already contains them. Passing both triggers a yfpy warning. `src/yahoo_api.py` conditionally excludes consumer_key/secret when a token dict is present.
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
- **Draft Settings live on Draft Simulator page** — Number of teams, Rounds, and Draft position inputs are on `pages/2_Draft_Simulator.py`, not on the Connect League page. Stored in `mock_num_teams`, `mock_num_rounds`, `mock_draft_pos` session state. Also written to shared `num_teams`/`num_rounds`/`draft_pos` keys for the main draft.
- **Yahoo sync across pages** — Pages reuse `st.session_state.yahoo_client` for Yahoo sync instead of creating new clients. Token data stored in `st.session_state.yahoo_token_data` for re-authentication. Pages show "Sync League Data Now" button when Yahoo connected but DB empty.
- **`T = THEME` is now a plain dict** — `T` in `ui_shared.py` is just `THEME` (the single color palette dict). The old `_ThemeProxy` class was removed along with dark mode. Backward-compat aliases: `T["amber"]`→`"#e63946"` (primary red), `T["teal"]`→`"#457b9d"` (sky blue).
- **Sidebar nav rename via JS** — The sidebar "app" label is renamed to "Connect League" and "Mock Draft" to "Draft Simulator" using JS `textContent` replacement in `inject_custom_css()`. Same JS block also injects the HEATER logo + text into the sidebar header.
- **Streamlit CSS `!important` requirement** — Streamlit's React-based renderer applies high-specificity inline CSS. ALL custom CSS properties on injected HTML (`.page-title`, `.metric-card`, etc.) need `!important` to override. Without it, custom backgrounds, font-sizes, and display properties silently revert to Streamlit defaults.
- **Streamlit HTML sanitization** — `st.markdown('<div class="glass">')` immediately auto-closes the tag. Cannot use split open/close div patterns with Streamlit widgets between them. Content renders outside the wrapper. Use self-contained HTML blocks only.
- **Orange sidebar + white text** — Sidebar background is `linear-gradient(180deg, #e65c00, #cc5200)`. All sidebar text forced to `#ffffff !important`. Active nav item uses dark overlay `rgba(0,0,0,0.15)` instead of white highlight for contrast.
- **Title badge deep navy gradient** — Page title badges use `linear-gradient(135deg, #1a1a2e, #16213e)` background with gradient text overlay (red > orange > gold via `background-clip: text`).
- **All buttons are orange** — Secondary buttons globally styled with `linear-gradient(135deg, #e65c00, #cc5200)` + white bold text. This matches the sidebar branding and creates visual consistency.
- **Data table white background** — All `st.dataframe()` tables get `background: #ffffff !important` to contrast against the page's `#f4f5f0` chalk background. Column headers get `font-weight: 700 !important`.
- **Trade analyzer needs standings for accuracy** — Without `league_standings` data, `evaluate_trade()` uses equal category weights (1.0 for all 12 categories). This means no marginal elasticity, no punt detection, and no strategic context. The trade may grade as A+ purely from raw stat exchange without knowing if those stat gains actually help in standings. UI shows a warning when standings are missing.
- **Rate-stat aggregation** — AVG=sum(h)/sum(ab), OBP=sum(h+bb+hbp)/sum(ab+bb+hbp+sf), ERA=sum(er)*9/sum(ip), WHIP=sum(bb+h)/sum(ip). Weighted averages, NOT simple averages. `_fix_rate_stats()` in `lineup_optimizer.py` recalculates these after LP solves.
- **Injury model scales rate stats** — `apply_injury_adjustment()` scales ER, BB_allowed, H_allowed by `_combined_factor` (health×age×workload), not just counting stats. Without this, injured pitchers show artificially low ERA/WHIP.
- **LP inverse stat weighting** — ERA/WHIP in lineup optimizer LP objective must be weighted by IP: `player_value -= val * ip * weight`. Without IP weighting, a 1-IP reliever with 0.00 ERA dominates a 200-IP starter.
- **`compare_players()` peer-group filtering** — Z-scores computed against `is_hitter`-filtered pool only. Hitter HR z-score uses hitter pool mean/std, not full pool (which includes pitchers with HR=0).
- **`check_staleness()` edge case** — `max_age_hours <= 0` returns `True` (always stale). Prevents division-by-zero and logical errors.
- **Trade engine graceful fallback** — `pages/3_Trade_Analyzer.py` wraps `from src.engine.output.trade_evaluator import evaluate_trade` in try/except. If the engine module is missing or broken, falls back to legacy `analyze_trade()` from `src/in_season.py`. Both code paths produce compatible output dicts.
- **Trade engine backward compat keys** — `evaluate_trade()` returns both new keys (`grade`, `surplus_sgp`, `category_analysis`) AND legacy keys (`total_sgp_change`, `mc_mean`, `mc_std`) so existing UI code doesn't break.
- **`compute_marginal_sgp()` uses `.get()` for missing categories** — Team totals may not have all 12 categories. Always use `team_totals.get(cat, 0.0)`, never `team_totals[cat]`.
- **LeagueConfig is the single source of truth** — All category definitions (hitting/pitching categories, rate stats, inverse stats, counting stats, STAT_MAP, SGP denominators) live in `LeagueConfig` in `src/valuation.py`. All other files import from there. Do NOT add local hardcoded category lists to any file — import from LeagueConfig instead.
- **OBP requires component stats** — OBP = (H+BB+HBP)/(AB+BB+HBP+SF). DB stores both the rate (`obp`) and intermediate columns (`bb`, `hbp`, `sf`) for proper roster-level aggregation. Same pattern as AVG=H/AB.
- **L (Losses) is an inverse counting stat** — Unlike ERA/WHIP (inverse rate stats), L is counting but lower-is-better. Handled by sign-flip in SGP, no IP weighting needed in LP objective.
- **Punt detection requires BOTH conditions** — A category is punt only when `gainable_positions == 0 AND rank >= 10`. Being rank 10 alone isn't enough if positions are still gainable; having 0 gainable alone isn't enough if you're already ranked high.
- **LP-constrained lineup totals replace raw roster totals** — `evaluate_trade()` uses `_lineup_constrained_totals()` (PuLP LP solver) to compute before/after category totals from only the 18 starting slots. Bench players are excluded from SGP calculations. This prevents "phantom production" where bench players inflate trade value in uneven trades. Falls back to `_roster_category_totals()` when PuLP is unavailable.
- **Roster cap enforcement (ROSTER_CAP=23)** — Uneven trades model forced drops and FA pickups. 2-for-1: `_find_drop_candidate()` removes lowest-SGP bench player. 1-for-2: `_find_fa_pickup()` adds best available FA, capped at type-specific median SGP when no league rosters loaded (prevents picking elite "FAs" who are actually rostered). Equal trades skip both.
- **`bench_cost` is always 0.0 now** — The old bench option value formula (`streaming_sgp_per_week * weeks_remaining * net_roster_change`) was replaced by explicit roster cap modeling. `bench_cost` key remains in the result dict for backward compatibility but is always 0.0. The real roster adjustment happens via `drop_candidate` and `fa_pickup` keys.
- **LP `<= 1` slot constraints mean slots can be empty** — The lineup optimizer uses `<= 1` (not `== 1`) constraints. If filling a slot hurts the objective (e.g., a pitcher with terrible ERA), the LP leaves it empty. This is correct behavior — a bad player's stats shouldn't count toward team totals.
- **FA pickup median SGP cap** — When no league rosters are loaded, `get_free_agents()` returns the full player pool. Without a cap, `_find_fa_pickup()` would add elite "FAs" (actually rostered players). `_compute_median_sgp_cap()` caps the pickup at the type-specific (hitter/pitcher) median SGP to prevent unrealistic acquisitions.
- **Replacement cost penalty uses `get_free_agents()`** — `_compute_replacement_penalty()` calls `get_free_agents()` from `src/league_manager.py`, which internally calls `load_league_rosters()` from the DB. When no rosters are loaded, `get_free_agents()` returns the full `player_pool` as the FA pool (graceful degradation → penalty ≈ 0 for most categories). The penalty only becomes meaningful when league roster data exists to identify which players are actually unavailable.
- **Replacement penalty skips rate stats** — AVG, ERA, WHIP are excluded from the replacement cost penalty because they are roster-aggregate stats. Adding a .300 hitter matters differently with 600 AB vs 100 AB, so "best FA replacement" doesn't translate to a simple counting gap. The `replacement_detail` dict shows `"skipped": "rate_stat"` for these categories.
- **`_roster_category_totals()` lives in `src/in_season.py`** — Used as fallback when PuLP is unavailable. The primary path uses `_lineup_constrained_totals()` which runs PuLP's LP solver. Don't duplicate `_roster_category_totals()`.
- **`enable_mc=True` activates Phase 2** — By default, `evaluate_trade()` runs Phase 1 only (deterministic). Set `enable_mc=True` for MC overlay. MC gracefully falls back to Phase 1 on failure.
- **BMA sigma difference → unequal weights** — Even when all systems project the same value, posterior weights differ slightly because forecast sigmas differ by system (Steamer HR sigma=5.2 vs ZiPS sigma=5.5). Tighter sigma → higher likelihood density → more weight.
- **KDE needs ≥20 samples** — `PlayerMarginal` falls back to Normal when `len(historical_values) < MIN_KDE_SAMPLES`. KDE with too few points produces unreliable density estimates.
- **Gaussian copula nearest-PD** — The `_nearest_pd()` function clips negative eigenvalues to 1e-10 and re-normalizes the diagonal to 1.0. This ensures Cholesky decomposition always succeeds, even with user-provided non-PD correlation matrices.
- **Paired MC seed discipline** — `run_paired_monte_carlo` uses a master RNG to generate per-sim seeds. Each seed is used identically for both before/after rosters. NEVER use different seeds for before vs after — this breaks the variance reduction.
- **MC roster stats use string keys** — `build_roster_stats()` converts int player_ids to str keys (e.g., `"1"`, `"42"`). This avoids int/str key mismatches between the pool DataFrame and the MC simulator.
- **Copula inverse categories** — ERA/WHIP correlations in `DEFAULT_CORRELATION` are already negative against W/K (e.g., ERA↔K = -0.65). In `sample_correlated_stats()`, inverse categories use `ppf(1-u)` to flip the quantile direction.
- **`ruff` command on Windows** — Use `python -m ruff check .` and `python -m ruff format .` instead of bare `ruff` — the binary may not be on PATH.
- **Statcast requires pybaseball** — `PYBASEBALL_AVAILABLE` flag in `statcast.py`. Functions return empty results when pybaseball not installed. All rolling feature computation works without it (just no raw data to aggregate).
- **Signal decay λ=0 means no decay** — `decay_weight()` returns 1.0 when `lambda_param=0`. Used for counting stats (HR, RBI) that don't lose relevance over time.
- **Kalman observation variance is sample-size-aware** — `observation_variance("ba", 20)` >> `observation_variance("ba", 500)`. This is critical: early-season observations with 20 PA get high noise → filter trusts prior. Late-season with 500 PA → filter trusts data.
- **BOCPD cp_prob is bounded by 1/hazard_lambda** — The raw `P(r_t=0)` returned by `BOCPD.update()` can never exceed ~`1/hazard_lambda` (e.g., 0.005 for λ=200). The real detection signal is the run length distribution's **mode** dropping from a long run (mode ≈ 30) to near zero (mode ≤ 2). `detect_changepoints()` uses this mode-drop approach.
- **HMM graceful fallback** — `fit_player_hmm()` returns `(None, DEFAULT_STATE_PROBS)` when: (a) hmmlearn not installed, (b) fewer than 10 observations, or (c) fitting raises an exception. Always usable.
- **`classify_regime_simple()` is the fallback** — When no Statcast data exists for HMM, this rule-based function uses just `recent_xwoba` and `season_xwoba` to estimate regime probabilities. Good enough for non-Savant players.
- **`enable_context=True` activates Phase 4** — By default, `evaluate_trade()` runs with context analysis enabled. Set `enable_context=False` to skip concentration risk and enhanced bench value. Phase 4 gracefully wraps in try/except with fallback to Phase 1 behavior.
- **Log5 rate clamping** — `RATE_FLOOR=0.001`, `RATE_CEILING=0.999` prevent division-by-zero in the odds-ratio formula. Applied via `np.clip()` to all three inputs (batter, pitcher, league avg).
- **`matchup_adjustment_factor()` returns 1.0 gracefully** — When `player_xwoba is None`, returns 1.0 (neutral). When `opponent_xwoba_against is None`, returns just `park_factor`. Never crashes with missing schedule data.
- **Weibull frailty cap at 2.0** — `frailty_from_health_score()` uses `1/max(hs, 0.5)`. Health scores below 0.5 are floored, so frailty never exceeds 2.0× normal duration. This prevents unrealistically long injury predictions.
- **HHI concentration threshold is 0.15** — Penalty only applies when `HHI > 0.15`. A perfectly diversified 10-team roster has HHI=0.1. The `scale=3.0` parameter maps `HHI=0.25` to a 0.3 SGP penalty — meaningful but not dominant.
- **Team alias normalization** — `TEAM_ALIASES` maps 8 common variants (WSN→WSH, AZ→ARI, CHW→CWS, etc.). Applied in `roster_concentration_hhi()` before grouping. Without this, "WSN" and "WSH" are counted as separate teams.
- **`concentration_risk_penalty` applied to `total_surplus`** — The penalty is subtracted from the trade surplus AFTER all other SGP calculations. This means concentration risk can flip a marginal "Accept" to "Reject" but won't dominate large surplus trades.
- **Enhanced bench value removed from trade evaluator** — Bench option value (both simple and enhanced) has been replaced by LP-constrained lineup totals + roster cap enforcement. The `bench_option_detail` key in the result dict is always `None`. Phase 4 context engine still handles concentration risk but no longer adjusts bench value.
- **Injury process imports from `src/injury_model.py`** — `injury_process.py` reuses `age_risk_adjustment()` from the existing module rather than duplicating the age-risk curve logic.
- **`enable_game_theory=True` activates Phase 5** — By default, `evaluate_trade()` runs game theory analysis. Set `enable_game_theory=False` to skip adverse selection, market values, and sensitivity. Phase 5 wraps in try/except for graceful fallback.
- **Market clearing price = second-highest bid** — Vickrey auction principle: the fair trade price is what the second-most-interested team would pay, not the top bidder. `market_clearing_price()` sorts valuations descending and returns index [1].
- **Adverse selection MAX_DISCOUNT = 0.25** — The discount factor never goes below 0.75 (1.0 - 0.25). Even with the worst possible manager history, received player value is reduced by at most 25%. This prevents the model from overreacting to small sample sizes.
- **Adverse selection needs ≥3 trades** — `MIN_HISTORY_FOR_CALIBRATION = 3`. With fewer trades, falls back to default prior (P_flaw=0.15). This prevents volatile estimates from 1-2 data points.
- **Bellman rollout uses paired scenarios** — For each sim, BOTH "no trade" and "trade" scenarios are evaluated with the same RNG state, ensuring variance reduction identical to the paired MC technique in Phase 2.
- **`compute_roster_balance()` normalizes by median rank** — Deviation is `|rank - median| / median`, not raw difference. This makes the score invariant to league size.
- **Sensitivity `player_sensitivity()` calls `evaluate_fn` with `enable_mc=False, enable_context=False`** — To avoid expensive MC/context calls during sensitivity analysis (which may call evaluate_trade N times for N players). Only the deterministic Phase 1 surplus is used for marginal impact computation.
- **Counter-offer `MIN_SWAP_IMPROVEMENT = 0.2`** — Only suggests player swaps that improve trade surplus by at least 0.2 SGP. Prevents suggesting trivial lateral moves.
- **Running mean stability normalizes by sample std, not mean** — Using `std(running_means) / std(samples)` instead of coefficient of variation (`std/mean`) avoids division-by-near-zero for zero-mean distributions (which are common in trade surplus samples centered at 0).
- **ESS via FFT** — `effective_sample_size()` uses `np.fft.fft()` for fast autocorrelation computation. O(N log N) instead of O(N²) for the naive approach. Uses Geyer's initial positive sequence estimator: stops summing ACF when it drops below 0.05.
- **Cache TTL = 0 means immediately stale** — Setting `ttl=0` in `TradeEvalCache.set()` creates an entry that `is_stale` returns True for on the next `.get()`. Used in tests but should never be used in production.
- **`get_trade_cache()` is a module-level singleton** — All trade evaluations within a Streamlit session share the same cache. Call `reset_trade_cache()` to force a full refresh (e.g., after standings update).
- **Adaptive sim scaling caps at 100K** — `compute_adaptive_n_sims()` never returns more than 100K regardless of trade complexity. The minimum is 1K (quick mode). The `time_budget_s` parameter allows capping by estimated runtime.
- **`DraftRecommendationEngine` swaps `pick_score`** — `recommend()` temporarily replaces `pick_score` with `enhanced_pick_score` before calling `DraftSimulator.evaluate_candidates()`. The original is saved to `_original_pick_score`. This ensures MC simulation uses the enhanced values without modifying the simulator.
- **Enhanced pick_score multiplier clamped [0.5, 1.5]** — The multiplicative component (park × injury × Statcast × platoon × contract_year × category_balance) is clipped to prevent extreme values. Additive bonuses (streaming penalty, closer bonus, flex bonus) are unclamped.
- **Category balance uses draft_progress scaling** — Early draft (rounds 1-8) compresses weights toward 1.0 (BPA dominates). Mid-draft full strength. Late draft (rounds 17+) amplifies gaps to 1.5x (gap-filling urgent). `draft_progress` = `current_round / num_rounds`.
- **BUY/FAIR/AVOID threshold scales** — Early picks (1-100): gap >= 20 = BUY. Mid (100-200): gap >= 15. Late (200+): gap >= 10. Invalid ranks (0 or negative) return "FAIR".
- **Streaming penalty only for non-elite non-closers** — Closers (SV > 10) and elite SP (ERA < 2.80) get zero penalty. Others penalized proportional to streaming-replaceable stat fraction (W, K).
- **XGBoost is optional** — `XGBOOST_AVAILABLE` flag in `ml_ensemble.py`. When unavailable, `predict_batch()` returns Series of zeros. Training requires `MIN_TRAINING_SAMPLES = 50`.
- **News sentiment not yet integrated** — Module is complete and tested but not wired into `DraftRecommendationEngine`. Ready for future news API integration.
- **`compute_category_balance()` empty all_teams returns all 1.0** — Graceful degradation when no opponent data available (e.g., before draft starts).
- **Platoon risk uses The Book research** — LHB vs LHP: 0.90 (10% discount). Based on Tom Tango's "The Book" (2007): ~40 fewer PA and ~15 wRC+ loss for LHB facing same-side pitching.
- **`compute_trade_values()` time decay before tiers** — Time decay is applied BEFORE tier assignment so tiers reflect actual trade values. `weeks_remaining` is clamped to minimum 1 to avoid degenerate scaling.
- **G-Score surplus uses scaled replacement levels** — `sgp_surplus = g_score - replacement_level * gscore_ratio`. The ratio converts raw SGP replacement levels to G-Score units for apples-to-apples comparison.
- **`identify_two_start_pitchers()` accepts `player_pool`** — When provided, uses actual pitcher stats (ERA, WHIP, K-BB%, xFIP, CSW%) for matchup scoring and streaming value. Without it, falls back to league-average defaults. Pass it for meaningful differentiation between pitchers.
- **`rate_damage_per_start` vs `rate_damage_weekly`** — Two-start results include both per-start and cumulative weekly rate damage. Consumers should use `rate_damage_weekly` for total impact assessment.
- **`_team_batting_cache` is thread-safe** — Uses `threading.Lock` around all reads/writes. Transient API failures do NOT cache empty dict (so retries are possible); only permanent import failures cache empty.
- **`_confidence_tier()` returns LOW for past dates** — `days_ahead < 0` returns "LOW". This handles edge cases where schedule data includes past games.
- **Start/sit `_HOME_ADVANTAGE = 1.02`** — Home field provides a 2% boost to counting stats. Rate stats (AVG, OBP, ERA, WHIP) are NOT scaled by matchup volume factor in `_compute_start_score()`.
- **Start/sit park factors inverted for pitchers** — `pf = 2.0 - pf` with floor at 0.5, consistent with `two_start.py`'s pitcher park adjustment. Hitter-friendly parks hurt pitchers.
- **Matchup planner percentile formula** — Uses `count(arr < val) / n * 100`, not `/ (n-1)`. Division by n prevents values exceeding 100% at the maximum.
- **`_projected_hitter_adjusted()` returns weekly-scaled totals** — Divides season projections by 140 games to get per-game rates, then multiplies by actual games this week with per-game park factors. NOT season totals.
- **`compute_pitcher_game_rating()` inverse_park floor** — `max(2.0 - pf, 0.5)` prevents negative raw scores for extreme park factors.
- **Waiver wire `classify_category_priority()` rate stat handling** — Rate stats (AVG, OBP, ERA, WHIP) classified by rank only (ATTACK/DEFEND/IGNORE) since weekly production rates don't apply to rate stats.
- **`compute_sustainability_score()` hitter K vs pitcher K** — For BABIP calculation, only hitter strikeouts are used in the denominator. The `hitter_k` variable is set to 0 for pitchers to avoid contamination.
- **Trade finder loss aversion** — `LOSS_AVERSION = 1.5` multiplies opponent LOSSES (making them feel 1.5x worse), not divides gains. This matches Kahneman & Tversky's prospect theory correctly.
- **Trade finder composite score** — `0.50 * user_delta + 0.25 * p_accept * 3.0 + 0.15 * max(opp_delta, 0) + 0.10 * need_match * 2.0`. Single-counted user_delta (not double-counted).
- **Draft grader `classify_pick()` uses AND logic** — Both SGP surplus AND ADP gap must agree for STEAL/REACH classification. Prevents misclassifying a player as STEAL purely from ADP deviation when SGP says otherwise.
- **Draft grader `category_balance_score()` uses std, not CV** — Standard deviation of z-scores mapped to [0, 1] via `1 - std/3.0`. CV (std/mean) is unstable when mean is near zero (common for balanced teams).
- **Pick predictor Weibull scale** — `scale = adp_distance * 1.5`, so `weibull_survival(picks, adp_dist, shape, scale)` with `adp_dist = max(1.0, ADP - current_pick)`. If `scale <= 0` or `picks <= 0`, returns 1.0 (available).
- **Pick predictor blended probability** — `P = 0.6 × P_normal + 0.4 × P_weibull`. Position shape params: C=1.4 (scarce, steep hazard), OF=0.8 (abundant, gentle hazard).
- **ECR blend uses Int64Dtype for nullable ecr_rank** — `df["ecr_rank"] = pd.array([None] * len(df), dtype=pd.Int64Dtype())` prevents FutureWarning when assigning int values to a column initialized with None.
- **ECR disagreement threshold is 20** — `|proj_rank - ecr_rank| > 20` triggers badge. Returns "ECR Higher" when projection undervalues (proj_rank > ecr_rank), "Proj Higher" when projection overvalues.
- **Power rankings momentum normalization** — Momentum [0.5, 2.0] normalized to [0, 1] via `(momentum - 0.5) / 1.5` before applying 10% weight in composite.
- **Standings projection INVERSE_CATS** — `{"ERA", "WHIP", "L"}` — lower-is-better categories. `compute_category_win_probability()` flips `mu_a`/`mu_b` for these.
- **Player tags DB uses CHECK constraint** — `CHECK(tag IN ('Sleeper','Target','Avoid','Breakout','Bust'))`. Invalid tags return False from `add_tag()`, not raise.
- **League registry uses UUID4** — `register_league()` generates `str(uuid.uuid4())` for league_id. First registered league auto-becomes active.
- **IL duration estimates are rough heuristics** — IL10=2 weeks, IL15=3.5 weeks, IL60=10 weeks, DTD=0.5 weeks. Position-specific adjustments not yet implemented.
- **Cheat sheet PDF requires weasyprint** — `generate_cheat_sheet_pdf()` returns None when weasyprint is not installed. HTML export always works.
- **Live draft sync poll interval** — 8-second polling via `time.time() - last_poll_time >= 8.0`. After 3 consecutive API errors, disables auto-sync. `LiveDraftSyncer.poll_and_sync()` returns `{new_picks, is_user_turn, error, draft_complete}`.
- **Bayesian stream score formula** — `stream_score = E[K]/sgp_K + W_prob×0.5/sgp_W - E[ER]×risk_penalty/sgp_ERA`. Matchup grade: top 10%→A+, 25%→A, 40%→B+, 60%→B, rest→C.
- **Prospect rankings are static** — `fetch_prospect_rankings()` returns a hardcoded curated top-20 list. No external scraping — avoids anti-scraping issues with FanGraphs.

## GitHub

- **Repo:** https://github.com/hicklax13/fantasy-baseball-draft-tool (public)
- **Current release:** v1.0.0
- **Release workflow:** Auto-creates GitHub releases on `v*.*.*` tags

## Testing Status

- **Unit tests:** 1777 collected, 1774 passed, 3 skipped (PyMC/xgboost optional deps)
- **Test files:** 76 test files across draft engine, trade engine (Phase 1-6), lineup optimizer (10 files), draft recommendation engine (5 files), gap closure (14 files), in-season analytics (7 files), FantasyPros parity (16 files), in-season, analytics, data pipeline, bootstrap, integration, backtesting, and math verification
- **Gap closure tests:** 153 tests total — extended roster (6), LAST CHANCE badge (8), Marcel projections (12), contract data (10), depth charts (12), news fetcher (18), ADP sources (15), extended projections (16), engine output (14), data pipeline schema (12), scheduler (5), bootstrap integration (25)
- **Spec completion tests:** 51 tests total — risk score + ST signal (16), schema persistence + Statcast archive + FG IDs (10), Yahoo ADP (5), category heatmap (5), backtesting harness (15)
- **Math verification suite:** 168 tests across 4 files (valuation, simulation, trade, trade engine math) — hand-calculated expected values verified against code formulas
- **Draft recommendation engine tests:** 270 tests total — DraftRecommendationEngine (50): all 8 stages, 3 modes, enhanced_pick_score formula, timing, integration. Draft analytics (35): category balance, opportunity cost, streaming value, BUY/FAIR/AVOID. Contextual factors (25): closer hierarchy, platoon risk, lineup protection, schedule strength, contract year. ML ensemble + sentiment (40): XGBoost fallback, feature prep, keyword scoring. Data foundation (30): LeagueConfig, Yahoo ADP, draft_state enhancements.
- **Trade engine tests:** 228 tests total — Phase 1 (47): marginal SGP, punt detection, z-scores, grading, fuzzy match, replacement cost penalty (6), lineup-constrained eval (9), integration. Phase 2 (33): BMA, KDE marginals, Gaussian copula, paired MC, correlated sampling, distributional metrics, integration. Phase 3 (32): Statcast aggregation, signal decay, Kalman filter, BOCPD changepoint detection, HMM regime classification, rolling features. Phase 4 (40): Log5 matchup math, Weibull injury duration, frailty, season availability, enhanced bench value, roster flexibility, HHI concentration, penalty thresholds, trade context integration. Phase 5 (38): opponent valuations, market clearing price, adverse selection Bayesian discount, Bellman rollout, roster balance, sensitivity ranking, counter-offers, game theory integration. Phase 6 (32): ESS convergence, split-R̂, running mean stability, cache TTL/invalidation/get_or_compute, adaptive sim scaling, time budget caps. Math (6): replacement cost formula hand-calcs (3) + lineup constraint math (3).
- **Lineup optimizer tests:** 204 tests total across 10 files — projections (28), matchups (19), H2H engine (18), SGP theory (16), streaming (16), scenarios (19), multi-period (16), dual objective (21), advanced LP (25), pipeline orchestrator (26)
- **In-season analytics tests:** 276 tests total across 7 files — trade value (40), two-start planner (43), start/sit advisor (50), matchup planner (38), waiver wire (32), trade finder (28), draft grader (45)
- **FantasyPros parity tests:** 184 tests total across 16 files — draft order (8), player tags (12), pick predictor (8), closer monitor (10), points league (11), leaders (11), ECR (12), cheat sheet (16), live draft sync (12), schedule grid (10), WSIS widget (10), standings projection (9), power rankings (11), IL manager (17), league registry (15), prospect rankings (6), plus 6 new Bayesian streaming tests in existing optimizer streaming file
- **CI:** GitHub Actions runs ruff lint/format + pytest on Python 3.11, 3.12, 3.13
- **Coverage:** 64% (above 60% CI threshold; pre-existing, no regressions)
- **Systematic code reviews:** Six rounds of full codebase debugging (149 bugs fixed): Round 1 (10 bugs — data pipeline, Yahoo API, lineup optimizer, CI), Round 2 (9 bugs — MC rate stats, regime detection, bellman DP, convergence, lineup optimizer UI, trade analyzer HTML), Round 3 (8 bugs — SGP denominators, survival gauge, injury persistence, percentile volatility, player pool duplicates), Round 4 (3 bugs — connection leak, player_name alias, export buttons), Round 5 (3 bugs — standings long-format parsing, maximin display format, emoji in H2H subheader), Round 6 (116 bugs — 7 new in-season modules: trade_value.py, two_start.py, start_sit.py, matchup_planner.py, waiver_wire.py, trade_finder.py, draft_grader.py — time-decay ordering, G-Score/replacement level mismatch, confidence tier logic, thread safety, loss aversion formula, STEAL/REACH AND-logic, category balance CV→std, park factor inversion for pitchers, rate stat scaling, sustainability interpolation slopes). All pushed to master, all CI green.

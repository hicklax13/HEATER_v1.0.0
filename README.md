# HEATER — Fantasy Baseball Draft Tool

A fantasy baseball draft assistant + in-season manager for Yahoo Sports H2H Categories leagues. Features the **Heater** UI — a modern glassmorphic design system with vibrant thermal color palette, 3D inflatable buttons, kinetic gradient typography, and pill-based navigation. Monte Carlo draft recommendations, Bayesian projection updates, and full in-season roster management.

Built for the **FourzynBurn** league (12-team H2H Categories, snake draft, 23 rounds, 12 scoring categories). ~30,000 lines of Python across 65+ source files with 1,113 tests (41 test files). Zero CSV uploads — all data auto-fetched from MLB Stats API + FanGraphs on every launch. Zero emoji — all icons are inline SVGs.

## Overview

The app features the **Heater** design system — a light-mode-only interface with glassmorphic cards, stadium red + thermal orange + scoreboard gold accents, and modern typography (Bebas Neue headers, Figtree body, IBM Plex Mono stats). All dropdowns replaced with pill button groups and search + card grids.

- **Splash Screen** — Zero-interaction data bootstrap on every launch: 750+ MLB players, 3 years of history, projections, park factors, live stats — all auto-fetched with staleness-based refresh
- **Setup Wizard** — 2-step guided setup (Settings > Launch). No CSV uploads needed — all data loaded automatically
- **Draft Page** — 3-column layout: MC recommendations with hero card (CSS dot health badges, P10/P90 ranges, survival probability), draft board, player search + opponent intel
- **In-Season Pages** — 6 Streamlit pages with shared theme system: team overview, draft simulator, trade analyzer, player compare, free agent rankings, lineup optimizer
- **Glassmorphic UI** — translucent cards with backdrop blur, 3D inflatable buttons, staggered entrance animations, kinetic gradient page titles

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) Yahoo Fantasy API
pip install python-dotenv stringcase yahoo-oauth==2.1.1
pip install yfpy>=17.0 --no-deps
pip install httpx-oauth
pip install streamlit-oauth>=0.1.14 --no-deps

# Generate sample data for testing
python load_sample_data.py

# Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Features

### Draft Tool (`app.py`)

- **Splash screen bootstrap** — auto-loads all data on every launch with progress bar
- **2-step setup wizard** (Settings > Launch) with optional Yahoo OAuth
- **Monte Carlo simulation** (100-300 sims, 6-round horizon) for pick recommendations
- **Hero card** with survival probability, CSS dot health badges, P10/P90 range bars, SGP chips
- **Search + card grid** player selection — replaces traditional dropdowns
- **Opponent intel** — threat alerts when opponents need your target position
- **Practice mode** — ephemeral draft state for what-if scenarios
- **Snake draft engine** with dynamic replacement levels and positional scarcity
- **Percentile sampling** from P10-P90 distributions across multiple projection systems

### In-Season Management (`pages/`)

| Page | Description |
|------|-------------|
| **My Team** | Roster display with injury badges, team monogram avatar, category standings, Bayesian-adjusted projections, Yahoo sync |
| **Draft Simulator** | Standalone mock draft with AI opponents, MC recommendations, position pill filters, card-based player selection |
| **Trade Analyzer** | 6-phase engine: Phase 1 deterministic SGP grading (A+ to F), Phase 2 MC (10K paired sims), game theory opponent modeling, Bayesian model averaging, Gaussian copula correlated sampling, glassmorphic verdict banners |
| **Player Compare** | Z-score normalization, dual search + card pickers, Plotly radar charts, health badges |
| **Free Agents** | Marginal SGP rankings by category need, position pill filters |
| **Lineup Optimizer** | 11-module pipeline with 20 mathematical techniques: enhanced projections (Bayesian/Kalman/regime/injury), weekly matchup adjustments (park/platoon/weather), H2H category weights (Normal PDF), non-linear SGP (bell-curve proximity), pitcher streaming, stochastic scenarios (copula/CVaR), multi-period planning, dual H2H/Roto objective, advanced LP (maximin/epsilon-constraint/stochastic MIP). 5-tab UI: Optimize, H2H Matchup, Streaming, Category Analysis, Roster. Three modes: Quick (<1s), Standard (2-3s), Full (5-10s) |

### Heater Design System (`src/ui_shared.py`)

- **Light-mode-only** — single vibrant `THEME` dict with stadium red, thermal orange, scoreboard gold, outfield green, sky blue accents
- **Glassmorphic cards** — `backdrop-filter: blur(20px) saturate(180%)` with translucent borders, hover lift + spring-back
- **3D inflatable buttons** — bottom shadow creates depth illusion, `:active` press-down effect
- **Kinetic gradient text** — animated `background-clip: text` on page titles (red > orange > gold gradient shift)
- **Deep navy title badges** — rounded pill-shaped header badges with gradient fill
- **Orange sidebar** — bold gradient sidebar with baseball logo + HEATER branding
- **Pill button navigation** — position filters use styled button groups instead of dropdowns
- **Centralized CSS** — 1500+ lines of shared styles injected via `inject_custom_css()`
- **Inline SVG icon system** — ~20 Feather-style icons in `PAGE_ICONS` dict (no emoji anywhere)
- **Educational tooltips** — `METRIC_TOOLTIPS` dict with explanations for SGP, VORP, survival %, health badges, and more
- **Fonts:** Bebas Neue (headers), Figtree (body), IBM Plex Mono (stats/metrics)

### Data Pipeline & Analytics

- **Zero-interaction bootstrap** — `bootstrap_all_data()` orchestrates 7 data phases with staleness-based smart refresh (1h live stats, 6h Yahoo, 7d projections, 30d historical/park factors)
- **FanGraphs auto-fetch** — Steamer, ZiPS, Depth Charts projections (3 systems x bat/pit = 6 endpoints) on app startup
- **MLB Stats API** — 750+ active player roster fetch, current season stats, 3-year historical stats for injury modeling
- **Yahoo Fantasy API** — OAuth 2.0 out-of-band flow with persistent auto-reconnect. Multi-strategy game key resolution handles pre-season edge cases. Syncs settings, rosters (12 teams x 23 players), standings, FA pool. Token auto-refreshes on every restart
- **Bayesian updater** — PyMC 5 hierarchical beta-binomial model for in-season stat updates; Marcel regression fallback when PyMC unavailable. Uses FanGraphs stabilization thresholds (K=60 PA, AVG=910 AB, ERA=70 IP)
- **Injury model** — 3-year health scores, age-risk curves (hitters +2%/yr after 30, pitchers +3%/yr after 28), workload flags for IP spikes
- **Percentile forecasts** — P10/P50/P90 floor/ceiling projections using inter-system volatility with process risk widening for low-correlation stats
- **Enhanced opponent modeling** — History-aware pick probability combining ADP, team needs, and draft history preferences

### Trade Analyzer Engine (`src/engine/`)

A 6-phase pipeline for rigorous trade evaluation:

| Phase | Module | What It Does |
|-------|--------|-------------|
| **1. Deterministic SGP** | `portfolio/` | Peer-group z-scores, standings-based SGP denominators, marginal elasticity (1/gap), punt detection, LP lineup delta, weighted SGP delta → grade (A+ to F) |
| **2. Stochastic MC** | `monte_carlo/` | Bayesian model averaging, KDE marginals, Gaussian copula (12×12 correlation matrix), 10K paired Monte Carlo sims → VaR, CVaR, Sharpe, confidence intervals |
| **3. Signal Intelligence** | `signals/` | Statcast harvesting (barrel%, xwOBA), exponential decay weighting, Kalman filter (true talent vs noise), BOCPD changepoint detection, HMM regime classification |
| **4. Contextual Adjustments** | `context/` | Log5 matchup probabilities, Weibull injury duration modeling, enhanced bench option value, HHI roster concentration risk |
| **5. Game Theory** | `game_theory/` | Vickrey auction opponent valuation, Bayesian adverse selection discount, Bellman DP rollout (future trade option value), sensitivity analysis + counter-offer generation |
| **6. Production** | `production/` | Monte Carlo convergence diagnostics (ESS, split-R̂), adaptive simulation scaling, precomputation cache with TTL |

219 dedicated tests across 6 test files verify each phase independently and in integration.

### Enhanced Lineup Optimizer (`src/optimizer/`)

An 11-module pipeline with 20 mathematical techniques for lineup optimization:

| Module | What It Does |
|--------|-------------|
| **Pipeline** | Master orchestrator: 9-stage chain, Quick/Standard/Full modes |
| **Projections** | Enhanced projections: Bayesian → Kalman → regime → decay → Statcast → injury availability chain |
| **Matchup Adjustments** | Weekly matchup adjustments: park factors, platoon splits (The Book regression), weather HR model |
| **H2H Engine** | H2H category weights (Normal PDF peaks at tie) + per-category win probability (Normal CDF) |
| **SGP Theory** | Non-linear marginal SGP (bell-curve proximity), SLOPE regression SGP denominators |
| **Streaming** | Pitcher streaming value (counting SGP - rate damage), two-start quantification, optimal schedule |
| **Scenario Generator** | Gaussian copula correlated scenarios, mean-variance adjustments, CVaR linearization |
| **Multi-Period** | Rolling horizon optimization with discount factor, season balance urgency weights |
| **Dual Objective** | H2H/season-long weight blending (alpha parameter), auto-alpha recommendation |
| **Advanced LP** | Maximin LP (balanced worst-category), epsilon-constraint (Pareto frontier), stochastic MIP |

204 dedicated tests across 10 test files. 5-tab Streamlit UI: Optimize, H2H Matchup, Streaming, Category Analysis, Roster.

### Draft Recommendation Engine (`src/draft_engine.py`)

A 25-feature enhanced draft pipeline with 3 execution modes:

| Mode | Time | Features |
|------|------|----------|
| **Quick** | <1s | Park factors + category balance only |
| **Standard** | 2-3s | + Bayesian blend, injury probability, Statcast delta, streaming penalty, contextual factors |
| **Full** | 5-10s | + ML ensemble (XGBoost), schedule strength, contract year boost |

**8-stage enhancement pipeline:** park factors → Bayesian blend → injury probability → Statcast delta → FIP correction → contextual factors → category balance → ML ensemble

**Enhanced scoring formula:**
```
enhanced = base_sgp × mult + additive
  mult = category_balance × park × (1 - injury×0.3) × (1 + statcast×0.15) × platoon × contract_year
  additive = streaming_penalty + lineup_protection + closer_bonus + ml×0.1 + flex_bonus
```

**Key features:**
- **Category balance** — Normal PDF weighting per category; early draft favors best-player-available, late draft amplifies gap-filling
- **Contextual factors** — Closer hierarchy (SV-based scarcity premium), platoon risk (The Book: LHB -10%), lineup protection (PA bonus by batting order), schedule strength (division park factors), contract year (+2% hitters)
- **BUY/FAIR/AVOID** — Classification based on model rank vs ADP gap, threshold scales by draft phase
- **ML ensemble** — Optional XGBoost residual correction (graceful fallback when unavailable)
- **News sentiment** — Keyword-based scoring ready for news API integration

270 dedicated tests across 5 test files.

## Setup

### 1. Launch

On every launch, the splash screen automatically fetches all data:
- **750+ MLB players** from MLB Stats API
- **3 projection systems** (Steamer, ZiPS, Depth Charts) from FanGraphs
- **3 years of historical stats** for injury modeling
- **Current season stats** from MLB Stats API
- **30-team park factors** for park-adjusted analysis

For testing without API access, run `python load_sample_data.py` first.

### 2. Configure League Settings

- Number of teams, rounds, your draft position
- SGP denominators (auto-computed or manual)
- Risk tolerance slider

### 3. (Optional) Connect Yahoo Fantasy

Set environment variables and the app shows a Yahoo connect card in Settings:

```bash
export YAHOO_CLIENT_ID=<your consumer key>
export YAHOO_CLIENT_SECRET=<your consumer secret>
export YAHOO_LEAGUE_ID=<your numeric league ID>
```

The OAuth flow uses out-of-band (oob): click the authorization link, approve on Yahoo, paste the verification code back in the app. **You only need to do this once** — the token is saved to `data/yahoo_token.json` and the app auto-reconnects on every restart.

**Auto-reconnect:** On each app launch, the splash screen automatically loads the saved Yahoo token, refreshes it if expired, resolves the correct game key for the current season, and syncs league data. No manual re-authorization needed.

**Multi-strategy game key resolution:** The app uses 3 fallback strategies to find the correct Yahoo game key for the current MLB season, handling pre-season edge cases where Yahoo may not have registered the new game key yet.

See [Yahoo Developer Apps](https://developer.yahoo.com/apps/) to create credentials (select "Fantasy Sports" API).

## During the Draft

1. **START DRAFT** from the setup wizard (toggle Practice Mode for mock drafts)
2. The **hero card** shows the top recommendation with survival probability, injury badge, and P10/P90 ranges
3. Search for a player or select from the card grid and click **DRAFT** to lock in your pick
4. In practice mode, opponents auto-pick based on ADP + team needs
5. Use the **tabs** to view category balance, available players, draft board, draft log, and opponent intel

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | Streamlit (Python), multi-page app |
| Database | SQLite (14 tables) |
| Core libs | pandas, NumPy, SciPy, Plotly |
| Analytics | PyMC 5 (Bayesian), PuLP (LP optimizer), arviz |
| Live data | MLB-StatsAPI, pybaseball |
| Yahoo API | yfpy v17+ + streamlit-oauth |
| Design | Glassmorphism, Bebas Neue/Figtree/IBM Plex Mono |
| Linting | ruff (lint + format) |
| CI | GitHub Actions (ruff, pytest on Python 3.11-3.13) |

## File Structure

```
app.py                  — Draft tool (~1835 lines): splash screen bootstrap + 2-step setup wizard + 3-column draft page
requirements.txt        — pip dependencies
load_sample_data.py     — Generates ~190 sample players + injury history for testing
.streamlit/config.toml  — Light theme configuration (Heater palette)
pages/
  1_My_Team.py          — Team overview, roster, injury badges, Bayesian projections, Yahoo sync
  2_Draft_Simulator.py  — Standalone draft simulator with AI opponents, pill filters, card-based picks
  3_Trade_Analyzer.py   — Trade proposal builder + MC analysis + glassmorphic verdict banners
  4_Player_Compare.py   — Head-to-head comparison with dual search + radar charts + health badges
  5_Free_Agents.py      — Free agent rankings by marginal value, position pill filters
  6_Lineup_Optimizer.py — 5-tab lineup optimizer powered by LineupOptimizerPipeline (Optimize, H2H Matchup, Streaming, Category Analysis, Roster)
src/
  database.py           — SQLite schema (14 tables), CSV import, queries
  valuation.py          — SGP, VORP, replacement levels, percentile forecasts
  draft_state.py        — Draft state management, roster tracking, opponent patterns
  simulation.py         — Monte Carlo simulation, history-aware opponent modeling
  in_season.py          — Trade analyzer, player comparison, FA ranker
  live_stats.py         — MLB Stats API: roster fetch, season/historical stats, injury data extraction
  data_bootstrap.py     — Zero-interaction bootstrap orchestrator: staleness-based refresh of all data sources
  league_manager.py     — League roster/standings management
  bayesian.py           — PyMC 5 hierarchical model + Marcel regression fallback
  injury_model.py       — Health scores, age-risk curves, workload flags
  lineup_optimizer.py   — PuLP LP solver, category targeting, two-start SP detection
  yahoo_api.py          — Yahoo Fantasy OAuth (oob flow) + auto-reconnect + multi-strategy game key resolution + league sync
  data_pipeline.py      — FanGraphs auto-fetch pipeline (Steamer/ZiPS/Depth Charts)
  draft_engine.py       — DraftRecommendationEngine: 3-mode orchestrator, 8-stage enhancement pipeline
  draft_analytics.py    — Category balance, opportunity cost, streaming value, BUY/FAIR/AVOID
  contextual_factors.py — Closer hierarchy, platoon risk, lineup protection, schedule strength, contract year
  ml_ensemble.py        — XGBoost ensemble for residual prediction (optional, graceful fallback)
  news_sentiment.py     — Keyword-based news sentiment scoring (-1.0 to +1.0)
  ui_shared.py          — Heater design system: single THEME dict, PAGE_ICONS (inline SVGs), glassmorphic CSS injection, metric tooltips
  validation.py         — Validation utilities
  data_2026.py          — Hardcoded 2026 projections for sample data
  optimizer/            — Enhanced Lineup Optimizer (11 modules, 20 mathematical techniques)
    pipeline.py         — Master orchestrator: 9-stage chain, Quick/Standard/Full modes
    projections.py      — Enhanced projections: Bayesian→Kalman→regime→decay→Statcast→injury chain
    matchup_adjustments.py — Weekly matchup adjustments: park factors, platoon splits, weather HR model
    h2h_engine.py       — H2H category weights (Normal PDF) + per-category win probability (Normal CDF)
    sgp_theory.py       — Non-linear marginal SGP (bell-curve proximity), SLOPE regression denominators
    streaming.py        — Pitcher streaming value, two-start quantification, optimal schedule
    scenario_generator.py — Gaussian copula scenarios, mean-variance adjustments, CVaR linearization
    multi_period.py     — Rolling horizon optimization with discount factor, season balance urgency
    dual_objective.py   — H2H/season-long weight blending (alpha parameter), auto-alpha recommendation
    advanced_lp.py      — Maximin LP, epsilon-constraint (Pareto frontier), stochastic MIP
  engine/               — Trade Analyzer Engine (6 phases, 11 modules)
    portfolio/          — Z-score valuation, category analysis, LP lineup optimization, Gaussian copula
    projections/        — ROS projection client, Bayesian model averaging, KDE marginals
    monte_carlo/        — Paired MC simulation (10K sims, variance reduction)
    output/             — Master trade orchestrator (Phase 1 + Phase 2 overlay)
    signals/            — Statcast harvesting, exponential decay, Kalman filter, BOCPD regime detection
    context/            — Log5 matchups, Weibull injury process, bench value, concentration risk
    game_theory/        — Opponent valuation, adverse selection, Bellman DP rollout, sensitivity analysis
    production/         — Convergence diagnostics, adaptive sim scaling, precomputation cache
tests/                  — 1,113 tests (41 test files), 1,108 pass, 3 skipped, 2 pre-existing failures
data/                   — SQLite DB + draft state backups (gitignored)
.github/
  workflows/ci.yml      — CI pipeline (lint + test + build)
  dependabot.yml        — Weekly dependency updates
```

## How It Works

### Valuation Pipeline

1. **Projection blending** — weighted average of Steamer, ZiPS, Depth Charts
2. **SGP** — converts raw stats to standings-point movement
3. **VORP** — value over replacement with multi-position flexibility premium
4. **Category weights** — dynamic: standings-based when available, draft-progress-aware
5. **pick_score** = weighted SGP + positional scarcity + VORP bonus

### Recommendation Scoring

```
combined_score = MC_mean_SGP + urgency * 0.4
urgency = (1 - P_survive) * positional_dropoff
```

Where survival probability uses Normal CDF with positional scarcity adjustment.

### Enhanced Draft Engine (25-Feature Pipeline)

The `DraftRecommendationEngine` wraps the base MC simulation with 8 enhancement stages:

1. **Park factors** — Coors 1.38x hitter boost, Marlins 0.88x
2. **Bayesian blend** — Marcel regression toward preseason projections
3. **Injury probability** — Health score + age + workload → P(injury over season)
4. **Statcast delta** — xwOBA vs projected wOBA (positive = undervalued skill)
5. **FIP correction** — `0.6*FIP + 0.4*ERA` replaces raw ERA for pitchers
6. **Contextual factors** — Closer role bonus, platoon discount, lineup protection, schedule strength, contract year
7. **Category balance** — Normal PDF weights by roster gaps (early draft: BPA, late: fill needs)
8. **ML ensemble** — Optional XGBoost residual correction (10% weight)

### Percentile Forecasts

Floor/ceiling projections using cross-system volatility:

1. Compute StdDev across Steamer, ZiPS, Depth Charts for each player/stat
2. Apply process risk widening for low year-over-year correlation stats (e.g., AVG r^2=0.41 vs HR r^2=0.72)
3. P10 (floor) = base - 1.28 * volatility, P90 (ceiling) = base + 1.28 * volatility
4. Bounded by physical limits (AVG in [.150, .400], ERA in [1.50, 7.00])

### Opponent Modeling

```
P(pick) = 0.5 * ADP + 0.3 * team_need + 0.2 * historical_preference
```

Falls back to ADP-only when no draft history is available.

## Development

```bash
# Lint
ruff check .

# Format
ruff format .

# Run all tests (843 pass, 1 skipped for PyMC)
python -m pytest

# Run with verbose output
python -m pytest -v

# Run with coverage
python -m pytest tests/ -v --cov=src --cov-report=term-missing

# Run a specific test file
python -m pytest tests/test_yahoo_api.py -v
```

Python 3.11+ required. CI tests on 3.11, 3.12, and 3.13.

## Optional Dependencies

| Package | Purpose | Install |
|---------|---------|---------|
| yfpy | Yahoo Fantasy API | `pip install yfpy>=17.0 --no-deps` + extras (see Quick Start) |
| streamlit-oauth | Browser-based Yahoo OAuth | `pip install streamlit-oauth>=0.1.14 --no-deps` + `httpx-oauth` |
| pymc | Bayesian projection updates | `pip install pymc>=5.0` |
| openpyxl | Excel cheat sheet export | `pip install openpyxl>=3.1.0` |

All optional features degrade gracefully when dependencies are missing.

## League Context

- **League:** FourzynBurn (Yahoo Sports)
- **Team:** Team Hickey
- **Format:** 12-team snake draft, 23 rounds, Head-to-Head Categories
- **Hitting cats (6):** R, HR, RBI, SB, AVG, OBP
- **Pitching cats (6):** W, L, SV, K, ERA, WHIP
- **Roster:** C/1B/2B/3B/SS/3OF/2Util/2SP/2RP/4P/5BN = 23 slots

## License

MIT

# HEATER — Fantasy Baseball Draft Tool

A fantasy baseball draft assistant + in-season manager for Yahoo Sports H2H Categories leagues. Features the **Heater** UI — a modern glassmorphic design system with vibrant thermal color palette, 3D inflatable buttons, kinetic gradient typography, and pill-based navigation. Monte Carlo draft recommendations, Bayesian projection updates, and full in-season roster management.

Built for the **FourzynBurn** league (12-team H2H Categories, snake draft, 23 rounds, 12 scoring categories). ~35,000 lines of Python across 75+ source files with 1,312 tests (53 test files). Zero CSV uploads — all data auto-fetched from MLB Stats API + FanGraphs on every launch. Zero emoji — all icons are inline SVGs.

## Overview

- **Splash Screen** — Zero-interaction data bootstrap on every launch: 1,000+ MLB players (active + 40-man + spring training), 3 years of history, 7 projection systems, park factors, live stats — all auto-fetched with staleness-based refresh
- **Setup Wizard** — 2-step guided setup (Settings > Launch). No CSV uploads needed
- **Draft Page** — 3-column layout: MC recommendations with hero card (BUY/FAIR/AVOID badges, risk score 0-100, category balance meter, LAST CHANCE flag, P10/P90 ranges), draft board, player search + opponent intel
- **In-Season Pages** — 6 Streamlit pages: team overview, draft simulator, trade analyzer, player compare, free agent rankings, lineup optimizer
- **Draft Recommendation Engine** — 25-feature 8-stage enhancement pipeline with 3 execution modes (Quick <1s, Standard 2-3s, Full 5-10s)
- **Trade Analyzer Engine** — 6-phase pipeline: deterministic SGP, stochastic MC (10K sims), signal intelligence (Statcast/Kalman/BOCPD), contextual adjustments, game theory (opponent modeling/adverse selection/Bellman), production (convergence/caching)
- **Enhanced Lineup Optimizer** — 11-module pipeline with 20 mathematical techniques
- **Backtesting Harness** — Evaluate engine accuracy vs historical outcomes (RMSE, rank correlation, value capture rate)

## Quick Start

```bash
pip install -r requirements.txt
python load_sample_data.py    # Sample data for testing
streamlit run app.py          # Opens at http://localhost:8501
```

## Data Sources (Auto-Fetched)

| Source | What | Staleness |
|--------|------|-----------|
| **MLB Stats API** | 1,000+ players (active + 40-man + spring training), current stats, 3yr history, injuries, transactions | 1h-7d |
| **FanGraphs JSON API** | 7 projection systems (Steamer, ZiPS, Depth Charts, ATC, THE BAT, THE BAT X, Marcel) | 7d |
| **Multi-Source ADP** | FanGraphs Steamer + FantasyPros ECR + NFBC + Yahoo (when connected) | 24h |
| **FanGraphs** | Depth charts with role classification (starter/platoon/closer/setup/committee) | 7d |
| **Baseball Reference** | Contract year detection from free agent lists | 30d |
| **Baseball Savant** | Statcast advanced metrics (xwOBA, barrel%, exit velocity) via pybaseball | On-demand |
| **Yahoo Fantasy API** | League rosters, standings, FA pool, draft results, Yahoo ADP (optional) | 6h |
| **Background Refresh** | Daemon thread checks staleness every 5 minutes + GitHub Actions daily cron | Continuous |

## Draft Recommendation Engine

A 25-feature enhanced draft pipeline with 3 execution modes:

**8-stage enhancement chain:** park factors → Bayesian blend → injury probability → spring training signal → Statcast delta → FIP correction → contextual factors → category balance → ML ensemble

**Key features:**
- **Category balance** — Normal PDF weighting per category; early draft favors BPA, late draft fills gaps
- **BUY/FAIR/AVOID** badges — classification based on model rank vs ADP gap
- **Risk score (0-100)** — Composite from health (40%), projection uncertainty (30%), age (15%), role stability (15%)
- **Confidence level** (HIGH/MEDIUM/LOW) from Monte Carlo variance
- **LAST CHANCE** flag when survival probability < 20%
- **Category heatmap** — green/yellow/red grid showing standings position per category
- **Contextual factors** — Closer hierarchy, platoon risk, lineup protection, schedule strength, contract year boost, spring training K-rate signal
- **ML ensemble** — Optional XGBoost residual correction (graceful fallback)

270 dedicated tests across 5 test files + 51 spec completion tests.

## Trade Analyzer Engine (`src/engine/`)

A 6-phase pipeline for rigorous trade evaluation:

| Phase | What It Does |
|-------|-------------|
| **1. Deterministic SGP** | Z-scores, standings-based SGP, marginal elasticity, punt detection, LP lineup delta, grade (A+ to F) |
| **2. Stochastic MC** | BMA, KDE marginals, Gaussian copula (12×12), 10K paired sims → VaR, CVaR, Sharpe |
| **3. Signal Intelligence** | Statcast harvesting, exponential decay, Kalman filter, BOCPD changepoint, HMM regime |
| **4. Context** | Log5 matchup, Weibull injury duration, bench value, HHI concentration risk |
| **5. Game Theory** | Vickrey auction, Bayesian adverse selection, Bellman DP rollout, counter-offer generation |
| **6. Production** | ESS convergence, split-R̂, adaptive scaling (1K-100K sims), precomputation cache |

228 dedicated tests across 6 test files.

## Enhanced Lineup Optimizer (`src/optimizer/`)

11-module pipeline with 20 mathematical techniques: enhanced projections (Bayesian/Kalman/regime/injury), weekly matchup adjustments (park/platoon/weather), H2H category weights, non-linear SGP, pitcher streaming, stochastic scenarios (copula/CVaR), multi-period planning, dual H2H/Roto objective, advanced LP (maximin/epsilon-constraint/stochastic MIP). 204 tests across 10 files.

## Database Schema

| Table | Purpose |
|-------|---------|
| `players` | Master player data (1,000+ players with mlb_id, fangraphs_id, positions, depth_chart_role, contract_year, news_sentiment) |
| `projections` | 7 projection systems + blended composite |
| `adp` | Multi-source ADP (FanGraphs, FantasyPros, NFBC, Yahoo) |
| `season_stats` | Current season statistics |
| `ros_projections` | Rest-of-season projections |
| `statcast_archive` | Per-year Statcast metrics (18 columns: EV, barrel%, xwOBA, sprint speed, stuff+, etc.) |
| `injury_history` | 3-year injury data (games played, IL stints, IL days) |
| `park_factors` | 30-team park factors (hitting + pitching) |
| `league_rosters` | 12-team league rosters (Yahoo sync) |
| `league_standings` | Category standings and scores |
| `transactions` | MLB transaction feed (7-day window) |
| `refresh_log` | Data staleness tracking |

## Development

```bash
ruff check .                          # Lint
ruff format .                         # Format
python -m pytest                      # 1,312 pass, 3 skipped, 2 pre-existing
python -m pytest -v                   # Verbose
python -m pytest tests/test_xxx.py    # Single file
```

Python 3.11+ required. CI tests on 3.11, 3.12, 3.13. Local dev uses 3.14.

## Testing

- **1,317 tests collected**, 1,312 passed, 3 skipped (PyMC/xgboost optional deps), 2 pre-existing failures
- **53 test files** covering: draft engine (5), trade engine phases 1-6 (6), lineup optimizer (10), gap closure (14), backtesting, math verification (4), integration, data pipeline, bootstrap, Yahoo API
- **Math verification suite** — 168 hand-calculated tests across 4 files
- **CI** — GitHub Actions: ruff lint/format + pytest on Python 3.11-3.13 + daily data refresh cron

## League Context

- **League:** FourzynBurn (Yahoo Sports) | **Team:** Team Hickey
- **Format:** 12-team snake draft, 23 rounds, Head-to-Head Categories
- **Hitting cats (6):** R, HR, RBI, SB, AVG, OBP
- **Pitching cats (6):** W, L, SV, K, ERA, WHIP
- **Roster:** C/1B/2B/3B/SS/3OF/2Util/2SP/2RP/4P/5BN = 23 slots

## License

MIT

# HEATER — Fantasy Baseball In-Season Manager

A fantasy baseball draft assistant and in-season manager for Yahoo Sports H2H Categories leagues. Powered by Monte Carlo simulation, Bayesian projection updating, LP-constrained lineup optimization, and a 6-phase trade analysis engine.

Built for the **FourzynBurn** league (12-team H2H Categories, snake draft, 23 rounds, 12 scoring categories). ~125,000 lines of Python with ~3,900 tests. Zero CSV uploads — all data auto-fetched from MLB Stats API, FanGraphs, Yahoo Fantasy, and 10+ other sources on every launch.

## Quick Start

```bash
pip install -r requirements.txt
python scripts/install-hooks.py    # Install pre-commit hook (first time)
python load_sample_data.py         # Sample data for testing (optional)
streamlit run app.py               # Opens at http://localhost:8501
```

## What It Does

### Draft Tool
- **Splash screen bootstrap** — 33-phase auto-fetch pipeline loads 9,000+ players, 7 projection systems, park factors, Statcast data, and more
- **Setup wizard** — 2-step guided setup, no CSV uploads needed
- **Draft page** — 3-column layout with Monte Carlo recommendations, BUY/FAIR/AVOID badges, risk scores, P10/P90 ranges, survival probability, and LAST CHANCE flags
- **AI opponents** — Draft simulator with marginal-SGP-based AI, position run detection, per-player ADP variance

### 20 In-Season Pages

Pages are ordered for daily workflow (status → daily action → strategy → trades → wire → research → preseason):

| # | Page | What It Does |
|---|------|-------------|
| 1 | **My Team** | War Room with category flip analysis, regression alerts, ratio lock detection, IL slot utilization, opponent lineup tracking |
| 2 | **Lineup Optimizer** | 6-tab optimizer: Start/Sit, Weekly Optimize, Manual, Streaming, Daily DCV, Roster view. LP-constrained with category urgency and mid-week pivot |
| 3 | **Closer Monitor** | 30-team closer depth chart with gmLI trust tracking, K% skill decay alerts, committee risk scores, opener detection |
| 4 | **Bullpen** | Reliever workload tracker and handcuff identification |
| 5 | **Matchup Planner** | Per-category win probabilities (Bayesian + copula), player matchup ratings, per-game detail with umpire/weather/PvB adjustments |
| 6 | **League Standings** | Live Yahoo H2H records, 12x12 category rank grid, MC season projections, magic numbers, power rankings |
| 7 | **Playoff Odds** | Top-4 playoff probability simulation |
| 8 | **Weekly Dashboard** | Weekly category trend summary |
| 9 | **Weekly Recap** | Post-week breakdown / win-loss attribution |
| 10 | **Punt Analyzer** | Punt-category strategy recommender |
| 11 | **Trade Analyzer** | 6-phase trade evaluation: deterministic SGP, stochastic MC (10K sims), Statcast signals, contextual adjustments, game theory, convergence diagnostics |
| 12 | **Trade Finder** | Multi-tab: Smart Recs, Target Player, Browse Partners, Trade Readiness. Cosine partner matching, behavioral acceptance model with loss aversion |
| 13 | **Trade Values** | Universal trade value chart reference |
| 14 | **Free Agents** | FA rankings by marginal value, ownership heat index, breakout candidate filter |
| 15 | **Waiver Wire** | Add/drop + waiver-priority tool |
| 16 | **Player Compare** | Head-to-head comparison with category fit, schedule strength, SGP breakdown |
| 17 | **Leaders** | Category leaders, Statcast breakout scores, prospect fantasy relevance, 40-man call-up alerts |
| 18 | **Trends** | Player trend / regression analysis |
| 19 | **Player Databank** | Historical multi-year player lookup |
| 20 | **Draft Simulator** | Standalone draft sim with AI opponents, MC recommendations (preseason) |

## Architecture

### Core Engines

- **Valuation Pipeline** — Ridge regression weighted stacking of 7 projection systems, Bayesian-updated SGP denominators, graduated positional scarcity, dynamic replacement levels
- **Trade Analyzer** (6 phases) — Deterministic SGP with LP-constrained lineup totals, paired Monte Carlo with true antithetic variates + Gaussian copula, Statcast/Kalman/regime signal intelligence, Log5 matchup + injury process context, Bellman DP game theory, convergence diagnostics
- **Trade Finder** (V4) — 3-tier scan: cosine dissimilarity partner matching, 1-for-1 + 2-for-1 scanning, behavioral acceptance model (loss aversion 1.8), category fit scoring, cross-type trades
- **Lineup Optimizer** (21 modules) — Enhanced projections with recent form, park/platoon/weather/umpire/catcher framing matchup adjustments, sigmoid category urgency, DCV daily scoring, FA recommender, pivot advisor, punt mode
- **Draft Engine** — 8-stage enhancement chain, 3 execution modes (Quick/Standard/Full), Monte Carlo simulation with opponent modeling
- **War Room** — Mid-week pivot analysis, category flip probability, hot/cold detection, ratio protection, conditional swap impact
- **Backtesting** — Historical replay framework validating engine recommendations against actual outcomes

### Unified Services
All pages consume from shared services (no divergent implementations):
- **MatchupContextService** — Single source for category weights (matchup/standings/blended modes)
- **SGPCalculator** — Sole SGP computation path with proper rate-stat volume weighting
- **YahooDataService** — 3-tier cache (session_state / Yahoo API / SQLite), 6 TTL-managed data types
- **CONSTANTS_REGISTRY** — Single source for tunable optimizer constants with citations and bounds
- **LeagueConfig** — Sole source for category lists, inverse stats, rate stats, SGP denominators

## Data Sources

| Source | What | Refresh |
|--------|------|---------|
| MLB Stats API | 9,000+ players, live stats, 3yr history, injuries, lineups, game logs | 1h-7d |
| FanGraphs | 7 projection systems, ROS projections, depth charts, Statcast (Stuff+, barrel%, xwOBA) | 24h |
| pybaseball | Park factors, team strength (wRC+/FIP), sprint speed, catcher framing, pitch mix | 24h |
| Yahoo Fantasy API | Rosters, standings, matchups, free agents, transactions, draft results | 5m-24h |
| FantasyPros | ECR consensus rankings (part of 6-source aggregation) | 24h |
| Open-Meteo | 30-stadium weather: temperature, wind speed/direction, precipitation | 2h |
| ESPN | Real-time injury feed (IL/DTD/NA status) | 1h |
| Baseball Savant | Umpire tendencies, bat speed, pitcher-batter matchup history | 7d |

## Tech Stack

- **Framework:** Streamlit (Python), multi-page app
- **Database:** SQLite (24+ tables, created at runtime)
- **Core:** pandas, NumPy, SciPy, Plotly, PuLP
- **Analytics:** PyMC 5 (Bayesian, optional), XGBoost (optional), arviz
- **Live data:** MLB-StatsAPI, pybaseball, Open-Meteo, feedparser, BeautifulSoup
- **Yahoo:** yfpy + streamlit-oauth (optional)
- **Linting:** ruff (lint + format), pre-commit hook enforced
- **CI:** GitHub Actions — ruff + sharded pytest (4 shards × 2 xdist workers, Python 3.12) + 60% coverage floor + build check

## Testing

- **~3,900 tests** across 165+ test files, ~14 skipped (PyMC/XGBoost optional deps)
- **~65% coverage** (60% CI floor)
- **Pre-commit hook** enforces ruff format + lint on every commit
- **Structural-invariant guards** (~70 patterns) prevent regression of architectural decisions
- **Backtesting framework** validates engine accuracy against historical outcomes

```bash
python -m pytest                                                      # Run all tests
python -m pytest tests/test_foo.py -v                                 # Single file
python -m pytest tests/ -n auto --dist loadfile                       # Parallel
python -m pytest tests/ --splits 4 --group 1 -n 2 --dist loadfile     # CI shard 1
python -m ruff check .                                                # Lint
python -m ruff format .                                               # Format
```

## League Context

- **League:** FourzynBurn (Yahoo Sports)
- **Team:** Team Hickey
- **Format:** 12-team snake draft, 23 rounds, Head-to-Head Categories, 26-week season
- **Hitting (6):** R, HR, RBI, SB, AVG, OBP
- **Pitching (6):** W, L, SV, K, ERA, WHIP (L/ERA/WHIP are inverse — lower is better)
- **Roster:** C/1B/2B/3B/SS/3OF/2Util/2SP/2RP/4P/6BN/4IL = 28 slots

## License

MIT

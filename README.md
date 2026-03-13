# Fantasy Baseball Draft Tool

A fantasy baseball draft assistant + in-season manager for Yahoo Sports 5x5 roto leagues. Features a "Broadcast Booth" UI with dark/light theme toggle, Monte Carlo draft recommendations, Bayesian projection updates, and full in-season roster management.

Built for the **FourzynBurn** league (12-team snake draft, 23 rounds). ~10,000 lines of Python across 20 source files with 300 tests. Zero emoji — all icons are inline SVGs.

## Overview

The app features a navy + amber "Broadcast Booth" theme with dark/light mode toggle and sports broadcast typography (Oswald + DM Sans). All icons are custom inline SVGs — no emoji.

- **Setup Wizard** — 4-step guided setup: auto-fetch projections from FanGraphs, configure league, import rosters, launch draft
- **Draft Page** — 3-column layout: MC recommendations with hero card (CSS dot health badges, P10/P90 ranges, survival probability), draft board, player search + opponent intel
- **In-Season Pages** — 5 Streamlit pages with shared theme system: team overview, trade analyzer, player compare, free agent rankings, lineup optimizer
- **Dark/Light Toggle** — one-click theme switch that persists across pages; navy borders + amber accent in both modes

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

- **4-step setup wizard** with optional Yahoo OAuth and FanGraphs auto-fetch
- **Monte Carlo simulation** (100-300 sims, 6-round horizon) for pick recommendations
- **Hero card** with survival probability, CSS dot health badges, P10/P90 range bars, SGP chips
- **Opponent intel** — threat alerts when opponents need your target position
- **Practice mode** — ephemeral draft state for what-if scenarios
- **Snake draft engine** with dynamic replacement levels and positional scarcity
- **Percentile sampling** from P10-P90 distributions across multiple projection systems

### In-Season Management (`pages/`)

| Page | Description |
|------|-------------|
| **My Team** | Roster display with injury badges, category standings, Bayesian-adjusted projections, Yahoo sync |
| **Trade Analyzer** | MC simulation (200 sims), park-adjusted SGP delta, confidence %, P10/P90 risk assessment |
| **Player Compare** | Z-score normalization, composite scores, Plotly radar charts, health badges, projection confidence |
| **Free Agents** | Marginal SGP rankings by category need, position filtering |
| **Lineup Optimizer** | PuLP LP solver, category targeting, two-start SP detection, health-adjusted SGP |

### UI & Theme System (`src/ui_shared.py`)

- **Dark/light theme toggle** — one-click switch persists across all pages via `st.session_state`
- **Centralized CSS** — 1100+ lines of shared styles injected via `inject_custom_css()`
- **Inline SVG icon system** — ~20 Feather-style icons in `PAGE_ICONS` dict (no emoji anywhere)
- **CSS dot health badges** — colored circles for injury indicators instead of emoji
- **Dual theme dictionaries** — `DARK_THEME` and `LIGHT_THEME` with `ink` key for text-on-accent contrast
- **`_ThemeProxy`** — dict subclass that makes `T["color"]` always resolve to the active theme
- **Educational tooltips** — `METRIC_TOOLTIPS` dict with explanations for SGP, VORP, survival %, health badges, and more

### Analytics & Data Pipeline

- **FanGraphs auto-fetch** — Steamer, ZiPS, Depth Charts projections (3 systems x bat/pit = 6 endpoints) on app startup
- **MLB Stats API** — daily auto-refresh of current season stats
- **Yahoo Fantasy API** — OAuth 2.0 out-of-band flow for league sync (settings, rosters, standings, FA pool)
- **Bayesian updater** — PyMC 5 hierarchical beta-binomial model for in-season stat updates; Marcel regression fallback when PyMC unavailable. Uses FanGraphs stabilization thresholds (K=60 PA, AVG=910 AB, ERA=70 IP)
- **Injury model** — 3-year health scores, age-risk curves (hitters +2%/yr after 30, pitchers +3%/yr after 28), workload flags for IP spikes
- **Percentile forecasts** — P10/P50/P90 floor/ceiling projections using inter-system volatility with process risk widening for low-correlation stats
- **Enhanced opponent modeling** — History-aware pick probability combining ADP, team needs, and draft history preferences

## Setup

### 1. Import Projections

On first launch, the app auto-fetches projections from FanGraphs (Steamer, ZiPS, Depth Charts). Alternatively:
- Upload CSV files from [FanGraphs Projections](https://www.fangraphs.com/projections)
- Click "Load Sample Data" for testing with ~190 sample players

### 2. Configure League Settings

- Number of teams, rounds, your draft position
- SGP denominators (auto-computed or manual)
- Risk tolerance slider

### 3. (Optional) Import League Data

Upload league rosters and standings CSVs to enable in-season pages (My Team, Trade Analyzer, Free Agents, Lineup Optimizer).

### 4. (Optional) Connect Yahoo Fantasy

Set environment variables and the app shows a Yahoo connect card on Step 1:

```bash
export YAHOO_CLIENT_ID=<your consumer key>
export YAHOO_CLIENT_SECRET=<your consumer secret>
export YAHOO_LEAGUE_ID=<your numeric league ID>
```

The OAuth flow uses out-of-band (oob): click the authorization link, approve on Yahoo, paste the verification code back in the app.

See [Yahoo Developer Apps](https://developer.yahoo.com/apps/) to create credentials (select "Fantasy Sports" API).

## During the Draft

1. **START DRAFT** from the setup wizard (toggle Practice Mode for mock drafts)
2. The **hero card** shows the top recommendation with survival probability, injury badge, and P10/P90 ranges
3. Select a player from the dropdown and click **DRAFT** to lock in your pick
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
| Linting | ruff (lint + format) |
| CI | GitHub Actions (ruff, pytest on Python 3.11-3.13) |

## File Structure

```
app.py                  — Draft tool (~1890 lines): 4-step setup wizard + 3-column draft page
requirements.txt        — pip dependencies
load_sample_data.py     — Generates ~190 sample players + injury history for testing
.streamlit/config.toml  — Dark theme configuration
pages/
  1_My_Team.py          — Team overview, roster, injury badges, Bayesian projections
  2_Trade_Analyzer.py   — Trade proposal builder + MC analysis + P10/P90 risk
  3_Player_Compare.py   — Head-to-head comparison with radar charts + health badges
  4_Free_Agents.py      — Free agent rankings by marginal value
  5_Lineup_Optimizer.py — PuLP LP solver for start/sit + category targeting
src/
  database.py           — SQLite schema (14 tables), CSV import, queries
  valuation.py          — SGP, VORP, replacement levels, percentile forecasts
  draft_state.py        — Draft state management, roster tracking, opponent patterns
  simulation.py         — Monte Carlo simulation, history-aware opponent modeling
  in_season.py          — Trade analyzer, player comparison, FA ranker
  live_stats.py         — MLB Stats API + pybaseball data fetcher
  league_manager.py     — League roster/standings management
  bayesian.py           — PyMC 5 hierarchical model + Marcel regression fallback
  injury_model.py       — Health scores, age-risk curves, workload flags
  lineup_optimizer.py   — PuLP LP solver, category targeting, two-start SP detection
  yahoo_api.py          — Yahoo Fantasy OAuth (oob flow) + league sync
  data_pipeline.py      — FanGraphs auto-fetch pipeline (Steamer/ZiPS/Depth Charts)
  ui_shared.py          — Centralized theme system (dark/light), PAGE_ICONS (inline SVGs), CSS injection, metric tooltips
  validation.py         — Validation utilities
  data_2026.py          — Hardcoded 2026 projections for sample data
tests/                  — 300 tests (17 test files), 299 pass, 1 skipped (PyMC)
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

# Run all tests (300 collected, 299 pass, 1 skipped for PyMC)
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
- **Format:** 12-team snake draft, 23 rounds
- **Scoring:** 5x5 roto (R, HR, RBI, SB, AVG / W, SV, K, ERA, WHIP)
- **Roster:** C/1B/2B/3B/SS/3OF/2Util/2SP/2RP/4P/5BN = 23 slots

## License

MIT

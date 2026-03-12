# Fantasy Baseball Draft Tool

A fantasy baseball draft assistant + in-season manager for Yahoo Sports 5x5 roto leagues. Features a dark-themed "Broadcast Booth" UI, Monte Carlo draft recommendations, Bayesian projection updates, and full in-season roster management.

Built for the **FourzynBurn** league (12-team snake draft, 23 rounds).

## Screenshots

The app features a dark navy + amber "Broadcast Booth" theme with sports broadcast typography.

- **Setup Wizard** — 4-step guided setup: import projections, configure league, import rosters, launch draft
- **Draft Page** — 3-column layout: MC recommendations with hero card, draft board, player search + opponent intel
- **In-Season Pages** — Team overview, trade analyzer, player compare, free agent rankings, lineup optimizer

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
- **Hero card** with survival probability, injury badges, P10/P90 range bars, SGP chips
- **Opponent intel** — threat alerts when opponents need your target position
- **Practice mode** — ephemeral draft state for what-if scenarios
- **Snake draft engine** with dynamic replacement levels and positional scarcity
- **Percentile sampling** from P10-P90 distributions across multiple projection systems

### In-Season Management (`pages/`)

| Page | Description |
|------|-------------|
| **My Team** | Roster display, category standings, Yahoo sync |
| **Trade Analyzer** | MC simulation (200 sims), park-adjusted SGP delta, confidence % |
| **Player Compare** | Z-score normalization, composite scores, Plotly radar charts |
| **Free Agents** | Marginal SGP rankings by category need |
| **Lineup Optimizer** | PuLP LP solver, category targeting, two-start SP detection |

### Data Pipeline

- **FanGraphs auto-fetch** — Steamer, ZiPS, Depth Charts projections on app startup
- **MLB Stats API** — daily auto-refresh of current season stats
- **Yahoo Fantasy API** — OAuth 2.0 league sync (settings, rosters, standings, FA pool)
- **Bayesian updater** — PyMC 5 hierarchical model for in-season stat updates (Marcel regression fallback)
- **Injury model** — health scores, age-risk curves, workload flags

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

Set environment variables and the app shows an "Authorize with Yahoo" button on Step 1:

```bash
export YAHOO_CLIENT_ID=<your consumer key>
export YAHOO_CLIENT_SECRET=<your consumer secret>
export YAHOO_LEAGUE_ID=<your numeric league ID>
```

See [Yahoo Developer Apps](https://developer.yahoo.com/apps/) to create credentials.

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
app.py                  — Draft tool: 4-step setup wizard + 3-column draft page
requirements.txt        — pip dependencies
load_sample_data.py     — Generates ~190 sample players + injury history
.streamlit/config.toml  — Dark theme configuration
pages/
  1_My_Team.py          — Team overview, roster, category standings
  2_Trade_Analyzer.py   — Trade proposal builder + MC analysis
  3_Player_Compare.py   — Head-to-head player comparison
  4_Free_Agents.py      — Free agent rankings by marginal value
  5_Lineup_Optimizer.py — PuLP LP solver for start/sit + category targeting
src/
  database.py           — SQLite schema (14 tables), CSV import, queries
  valuation.py          — SGP, VORP, replacement levels, percentile forecasts
  draft_state.py        — Draft state management, roster tracking
  simulation.py         — Monte Carlo simulation, opponent modeling
  in_season.py          — Trade analyzer, player comparison, FA ranker
  live_stats.py         — MLB Stats API + pybaseball data fetcher
  league_manager.py     — League roster/standings management
  bayesian.py           — PyMC hierarchical model + Marcel regression fallback
  injury_model.py       — Health scores, age-risk curves, workload flags
  lineup_optimizer.py   — PuLP LP solver, category targeting
  yahoo_api.py          — Yahoo Fantasy OAuth + league sync
  data_pipeline.py      — FanGraphs auto-fetch pipeline
  ui_shared.py          — Shared theme constants + CSS injection
  validation.py         — Validation utilities
  data_2026.py          — Hardcoded 2026 projections for sample data
tests/                  — 186 tests (14 test files, 64% coverage)
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

# Run all tests
python -m pytest

# Run with coverage
python -m pytest tests/ -v --cov=src --cov-report=term-missing

# Run a specific test file
python -m pytest tests/test_yahoo_api.py -v
```

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

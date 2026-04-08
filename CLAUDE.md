# HEATER — Fantasy Baseball In-Season Manager

## Overview

A fantasy baseball draft assistant + in-season manager for a 12-team Yahoo Sports H2H Categories snake draft league. Built with Streamlit, powered by Monte Carlo simulation, Bayesian projection updating, and LP-constrained lineup optimization.

1. **Draft Tool** (`app.py`) — Heater-themed Streamlit app with glassmorphic design, splash screen bootstrap, setup wizard, 3-column draft page, Monte Carlo recommendations with percentile sampling.
2. **In-Season Management** (`pages/`) — 11 pages: team overview, draft simulator, trade analysis, player comparison, free agents, lineup optimizer, closer monitor, league standings, leaders/prospects, trade finder, matchup planner.
3. **Trade Analyzer Engine** (`src/engine/`) — 6-phase pipeline: deterministic SGP → stochastic MC → signal intelligence → contextual adjustments → game theory → production convergence/caching.
4. **Lineup Optimizer** (`src/optimizer/`) — 15-module pipeline: enhanced projections, matchup adjustments, H2H weights, non-linear SGP, streaming, scenarios, dual objective, advanced LP, category urgency, daily optimizer (DCV), FA recommender, pivot advisor, shared data layer.
5. **Draft Recommendation Engine** (`src/draft_engine.py`) — 8-stage enhancement chain with 3 execution modes (Quick/Standard/Full).
6. **War Room** (`src/war_room*.py`) — Mid-week pivot analysis, category flip probability, hot/cold detection, action recommendations.
7. **Backtesting** (`src/backtesting*.py`) — Historical replay framework for validating engine recommendations against actual outcomes.

## Commands

```bash
pip install -r requirements.txt        # Install deps
python scripts/install-hooks.py        # Install pre-commit hook (first time)
python load_sample_data.py             # Load sample data (first time/testing)
streamlit run app.py                   # Run the app
python -m ruff check .                 # Lint
python -m ruff format .                # Format
python -m pytest                       # Run all tests (~2960 pass, ~14 skipped)
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
- **Season state:** MLB 2026 season active. Draft completed. App is in in-season mode.
- **Yahoo game_key:** 469

## Tech Stack

- **Framework:** Streamlit (Python), multi-page app
- **Database:** SQLite (`data/draft_tool.db`)
- **Core libs:** pandas, NumPy, SciPy, Plotly
- **Analytics:** PyMC 5 (Bayesian, optional), PuLP (LP optimizer), arviz
- **Live data:** MLB-StatsAPI, pybaseball (FanGraphs), Open-Meteo (weather)
- **Yahoo API:** yfpy + streamlit-oauth (optional OAuth)
- **Linter:** ruff (lint + format), pre-commit hook enforced
- **CI:** GitHub Actions — ruff lint/format, pytest (Python 3.11-3.13), 60% coverage floor
- **Python:** Local dev uses 3.14; CI tests 3.11-3.13

## File Structure

```
app.py                      — Main Streamlit app: splash screen, bootstrap, setup wizard, draft page
requirements.txt            — pip dependencies
load_sample_data.py         — Sample data generator for testing
.streamlit/config.toml      — Light theme (Heater palette)

pages/
  1_My_Team.py              — War Room, roster, category standings, alerts, Yahoo sync
  2_Draft_Simulator.py      — Draft simulator with AI opponents, MC recommendations
  3_Trade_Analyzer.py       — Trade proposal builder + 6-phase engine
  4_Free_Agents.py          — Free agent rankings by marginal value + ownership heat
  5_Line-up_Optimizer.py    — 6-tab optimizer: Start/Sit, Optimize, Manual, Streaming, Daily, Roster
  6_Player_Compare.py       — Head-to-head comparison with category fit + schedule strength
  7_Closer_Monitor.py       — 30-team closer depth chart grid with job security scoring
  8_League_Standings.py     — Current standings + MC season projections + power rankings
  9_Leaders.py              — Category leaders, breakout detection, prospects
  10_Trade_Finder.py        — Multi-tab trade finder: Smart Recs, Target Player, Browse Partners, Readiness
  11_Matchup_Planner.py     — Category probabilities, player matchups, per-game detail

src/
  # Core engines
  valuation.py              — SGP calculator, replacement levels, VORP, LeagueConfig (source of truth)
  database.py               — SQLite schema, player pool + in-season queries
  data_bootstrap.py         — 21-phase bootstrap orchestrator (staleness-based refresh)
  data_pipeline.py          — FanGraphs auto-fetch (Steamer/ZiPS/Depth Charts)
  yahoo_api.py              — Yahoo Fantasy API OAuth integration (429 backoff, ghost filtering)
  yahoo_data_service.py     — 3-tier cache: session_state → Yahoo API → SQLite

  # Draft
  draft_engine.py           — DraftRecommendationEngine (3-mode, 8-stage)
  draft_state.py            — Draft state management, roster tracking, snake pick order
  draft_analytics.py        — Category balance, opportunity cost, BUY/FAIR/AVOID
  draft_grader.py           — Post-draft grader (3-component)
  draft_order.py            — Draft order utilities
  simulation.py             — MC draft simulation, AI opponent modeling, position run detection
  pick_predictor.py         — Normal CDF + Weibull survival blend

  # Trade system
  trade_finder.py           — V4 trade finder: cosine→1v1→2v1 scan, 52 variables, category fit
  trade_intelligence.py     — Health/scarcity/FA gating/correlation adjustments/schedule urgency
  trade_value.py            — Universal trade value chart (0-100)
  trade_signals.py          — Kalman + regime trend adjustment
  opponent_trade_analysis.py — Opponent needs, archetype detection, acceptance modeling
  in_season.py              — Legacy trade analyzer, player comparison, FA ranker

  # Lineup optimization
  lineup_optimizer.py       — PuLP LP solver (DTD/IL exclusion, IP weighting)
  lineup_rl.py              — Category-aware reinforcement learning optimizer

  # Matchup & strategy
  matchup_context.py        — MatchupContextService: unified category weights + opponent intel
  matchup_planner.py        — Weekly matchup planner with percentile color tiers
  standings_engine.py       — Bayesian category probabilities, MC season simulation, magic numbers
  standings_projection.py   — MC season simulation (legacy)
  standings_utils.py        — Shared standings utilities with session cache
  weekly_h2h_strategy.py    — Weekly H2H strategic planning
  war_room.py               — Mid-week pivot analysis
  war_room_actions.py       — War room action tracking
  war_room_hotcold.py       — Hot/cold player analysis
  start_sit.py              — Start/sit advisor (3-layer decision model)
  start_sit_widget.py       — Start/sit UI widget

  # Player intelligence
  bayesian.py               — PyMC hierarchical model + Marcel regression fallback
  injury_model.py           — Health scores, age-risk curves, position-specific aging, injury-type
  ml_ensemble.py            — XGBoost Statcast regression (optional)
  playing_time_model.py     — Ridge regression playing time predictions
  projection_stacking.py    — Ridge regression projection weighting
  ecr.py                    — 6-source ECR consensus (Trimmed Borda Count, temporal weighting)
  prospect_engine.py        — FanGraphs Board API + MLB Stats API MiLB stats
  player_news.py            — 2-source news aggregation (MLB + Yahoo)
  news_fetcher.py           — News fetching utilities
  news_sentiment.py         — Keyword-based news sentiment scoring
  espn_injuries.py          — Real-time ESPN injury feed for IL/DTD status
  player_card.py            — Player card data assembly (pure function)
  player_tags.py            — Sleeper/Target/Avoid/Breakout/Bust tags

  # Game-day intelligence
  game_day.py               — Weather (Open-Meteo), opposing pitchers, team strength, lineups, form
  closer_monitor.py         — 30-team job security scoring (gmLI, K%, committee risk)
  schedule_grid.py          — 7-day schedule grid with matchup colors
  contextual_factors.py     — Closer hierarchy, platoon risk, lineup protection

  # In-season management
  alerts.py                 — AVIS decision rules: auto-IL, closer handcuff, regression alerts
  opponent_intel.py         — Opponent roster analysis, weakness detection
  waiver_wire.py            — LP-verified add/drop recommendations
  weekly_report.py          — Monday morning briefing
  ip_tracker.py             — Weekly IP tracking toward 1,400 IP target
  il_manager.py             — IL detection and replacement selection
  two_start.py              — Two-start pitcher planner
  power_rankings.py         — 5-factor power rankings with bootstrap CI
  leaders.py                — Category/points leaders, breakout detection

  # Data sources
  live_stats.py             — MLB Stats API data fetcher
  adp_sources.py            — ADP aggregation from multiple sources
  depth_charts.py           — FanGraphs depth chart data
  contract_data.py          — Contract year player detection
  marcel.py                 — Marcel projection system (local computation)
  data_2026.py              — Hardcoded 2026 projections for sample data

  # Backtesting & validation
  backtesting.py            — Historical replay engine
  backtesting_framework.py  — Backtesting framework infrastructure
  validation/               — Calibration, constant optimization, dynamic context, survival

  # UI
  ui_shared.py              — THEME, PAGE_ICONS, CSS, 3-zone layout, compact tables, formatting
  ui_analytics_badge.py     — Analytics badge rendering
  cheat_sheet.py            — HTML/PDF export

  # Misc
  league_manager.py         — League roster/standings management
  league_registry.py        — Multiple league support
  live_draft_sync.py        — Yahoo draft polling and sync
  points_league.py          — Yahoo/ESPN/CBS scoring presets
  scheduler.py              — Task scheduling utilities
  analytics_context.py      — Analytics context tracking

  # Sub-packages
  optimizer/                — Enhanced Lineup Optimizer (15 modules)
    pipeline.py             — Master orchestrator (Quick/Standard/Full modes)
    projections.py          — Enhanced projections (Bayesian/Kalman/regime/injury/recent form)
    matchup_adjustments.py  — Park factors, platoon splits, weather, umpire, catcher framing, PvB
    h2h_engine.py           — H2H category weights + win probability
    sgp_theory.py           — Non-linear marginal SGP
    streaming.py            — Pitcher streaming + Bayesian stream scoring
    scenario_generator.py   — Gaussian copula scenarios + CVaR
    dual_objective.py       — H2H/Roto weight blending
    advanced_lp.py          — Maximin, epsilon-constraint, stochastic MIP
    category_urgency.py     — Sigmoid urgency for H2H matchup gaps
    daily_optimizer.py      — Daily Category Value (DCV) per-player per-day scoring
    fa_recommender.py       — Free agent recommendation engine
    pivot_advisor.py        — Mid-week pivot recommendations
    shared_data_layer.py    — Unified data infrastructure across optimizer modules
  engine/                   — Trade Analyzer Engine (6 phases, 7 sub-packages)
    portfolio/              — Z-scores, SGP, category analysis, lineup optimizer, copula
    projections/            — ROS projections, BMA, KDE marginals
    monte_carlo/            — Paired MC simulator (10K sims, variance reduction)
    signals/                — Statcast, decay, Kalman filter, regime detection
    context/                — Log5 matchup, injury process, bench value, concentration risk
    game_theory/            — Opponent valuation, adverse selection, Bellman DP, sensitivity
    production/             — Convergence diagnostics, cache, adaptive sim scaling
    output/                 — Master trade orchestrator (evaluate_trade)

scripts/
  install-hooks.py          — Installs git pre-commit hook
  pre-commit                — Hook: ruff format + lint check on staged .py files
tests/                      — 140 test files, ~2960 passing tests
data/
  draft_tool.db             — SQLite database (created at runtime)
  backups/                  — Draft state JSON backups
docs/
  ROADMAP.md                — 115/115 improvement items completed
  AUDIT_REPORT.md           — Bug audit history
  VERIFICATION_LOG.md       — Browser testing verification log
.github/
  workflows/ci.yml          — CI pipeline (lint + test + build)
  workflows/refresh.yml     — Scheduled daily data refresh (9:17 UTC)
```

## Architecture

### Core Valuation Pipeline
1. **Projection blending** — Ridge regression weighted stacking of 7 systems (Steamer, ZiPS, Depth Charts, ATC, THE BAT, THE BAT X, Marcel)
2. **Dynamic SGP** — Converts raw stats to standings-point movement; Bayesian-updated denominators from league data
3. **VORP** — Value Over Replacement Player with graduated positional scarcity (C=1.20x, 2B=1.15x, SS=1.10x)
4. **pick_score** = weighted SGP + positional scarcity + VORP bonus

### Unified Services (V1-V6 unification)
- **MatchupContextService** (`matchup_context.py`) — Single source for category weights with 3 modes: "matchup" (H2H urgency), "standings" (gap analysis), "blended" (alpha-weighted). All pages consume from ONE source.
- **SGPCalculator** — Sole path for SGP computation. No local `_totals_sgp()` variants.
- **`standings_utils.get_team_totals()`** — Session-cached roster totals. One computation, all pages.
- **`get_fa_pool()`** — Single FA pool accessor with session cache.
- **`format_stat(value, stat_type)`** — Enforced stat formatting: AVG/OBP=`.3f`, ERA/WHIP=`.2f`, SGP=`+.2f`.

### Yahoo Data Service (Live-First Architecture)
3-tier cache: `st.session_state` → Yahoo Fantasy API → SQLite fallback.
TTLs: Rosters 30m, Standings 30m, Matchup 5m, Free Agents 1h, Transactions 15m, Settings/Schedule 24h.
Write-through: every Yahoo fetch writes to SQLite. Singleton via `get_yahoo_data_service()`.

### Bootstrap Pipeline (21 phases)
Staleness-based refresh on every app launch. Force Refresh sidebar button overrides all checks.
Key thresholds: 1h live stats/news, 30min Yahoo, 2h game-day, 24h projections/ADP/ECR, 7d players/prospects.

## Key API Signatures

```python
# LeagueConfig — source of truth for all category definitions
from src.valuation import LeagueConfig, SGPCalculator
# Do NOT hardcode category lists. Import from LeagueConfig.

# Yahoo Data Service
from src.yahoo_data_service import get_yahoo_data_service
yds = get_yahoo_data_service()
yds.get_rosters() -> pd.DataFrame          # 30min TTL
yds.get_standings() -> pd.DataFrame         # 30min TTL
yds.get_matchup() -> dict | None            # 5min TTL
yds.get_free_agents(max_players=500)        # 1hr TTL
yds.get_transactions() -> pd.DataFrame      # 15min TTL
yds.get_full_league_schedule() -> dict      # 24hr TTL
yds.is_connected() -> bool

# Trade evaluator
from src.engine.output.trade_evaluator import evaluate_trade
evaluate_trade(giving_ids, receiving_ids, user_roster_ids, player_pool,
               config=None, enable_mc=False, enable_context=True, enable_game_theory=True)

# Trade finder
from src.trade_finder import find_trade_opportunities
find_trade_opportunities(user_roster_ids, player_pool, config=None,
    all_team_totals=None, league_rosters=None, max_results=20)

# Lineup optimizer
from src.optimizer.pipeline import LineupOptimizerPipeline
LineupOptimizerPipeline(roster, mode="standard", alpha=0.5, weeks_remaining=16, config=None)

# Bootstrap
from src.data_bootstrap import bootstrap_all_data
bootstrap_all_data(yahoo_client=None, on_progress=None, force=False)

# Valuation (standalone functions, NOT methods on SGPCalculator)
from src.valuation import compute_replacement_levels, compute_sgp_denominators, value_all_players

# Roster display sort
from src.ui_shared import sort_roster_for_display
sort_roster_for_display(roster_df: pd.DataFrame) -> pd.DataFrame  # Yahoo slot order, returns copy

# Injury badges
from src.injury_model import get_injury_badge
get_injury_badge(health_score) -> tuple[str, str]  # CSS dot, NOT emoji
```

## Gotchas

### Data & Schema
- **DB column `name` vs UI `player_name`** — Normalized via `rename(columns={"name": "player_name"})` in `_build_player_pool()`.
- **Sample data is `system='blended'` only** — Do NOT call `create_blended_projections()` when only blended data exists.
- **Python 3.14 + SQLite bytes** — Returns bytes for some integer columns. Fixed with `CAST` in SQL and `pd.to_numeric()`.
- **Players table has no UNIQUE constraint on name** — Uses SELECT-first pattern. Do NOT use `ON CONFLICT(name)`.
- **Park factors schema** — Columns: `team_code`, `factor_hitting`, `factor_pitching`.
- **FanGraphs API SYSTEM_MAP** — FG API uses `"fangraphsdc"`, DB stores `"depthcharts"`. Always use `SYSTEM_MAP`.
- **LeagueConfig is the single source of truth** — All category definitions. Do NOT hardcode category lists.

### Yahoo API
- **OAuth uses oob** — `redirect_uri=oob` required. No localhost redirects.
- **Auto-reconnect** — `_try_reconnect_yahoo()` loads `data/yahoo_token.json`. Token refresh automatic.
- **yfpy Roster iteration** — `Roster.__iter__()` yields attribute NAMES (strings), NOT players. Use `getattr(roster, "players", None) or []`.
- **Yahoo game_key** — MLB 2026 = 469. Must set BOTH `self._query.game_id` AND `self._query.league_key`.
- **429 backoff** — `_request_with_backoff()` retries 3x with exponential delay.

### Streamlit UI
- **No emoji** — All icons are inline SVGs from `PAGE_ICONS`. Injury badges use CSS dots.
- **CSS `!important` required** — Streamlit's inline CSS has high specificity.
- **HTML sanitization** — `st.markdown('<div>')` auto-closes tags. Use self-contained HTML blocks.
- **`T["ink"]` for text-on-accent** — Not `T["bg"]` (invisible on amber).
- **Connection leak** — Always wrap `get_connection()` in `try/finally` with `conn.close()`.
- **`T["tx"]` not `T["tx1"]`** — There is no `tx1` key.

### Algorithms & Math
- **Rate-stat aggregation** — AVG=sum(h)/sum(ab), OBP=sum(h+bb+hbp)/sum(ab+bb+hbp+sf), ERA=sum(er)*9/sum(ip), WHIP=sum(bb+h)/sum(ip). Weighted averages, NOT simple averages.
- **LP inverse stat weighting** — ERA/WHIP must be weighted by IP. Without it, 1-IP relievers dominate starters.
- **L (Losses) is an inverse counting stat** — Sign-flip in SGP, no IP weighting in LP.
- **Punt detection** — Requires BOTH: `gainable_positions == 0 AND rank >= 10`.
- **LP-constrained lineup totals** — Only 18 starters' stats feed SGP. Bench excluded.
- **Paired MC seed discipline** — Same seed for before/after rosters. NEVER different seeds.
- **Trade analyzer needs standings** — Without standings: no elasticity, no punts, no strategic context.

### Lineup Optimizer
- **DTD/IL exclusion** — `_il_statuses` includes `dtd` and `day-to-day`.
- **Recent form blend** — L14 game logs at 20% weight. Requires >= 7 games. First run ~20-30s, then 2h cache.
- **Matchup data before state classification** — `yds.get_matchup()` must be called before `classify_matchup_state()`.
- **IP budget uses projected IP** — Not actual IP pitched.
- **Weekly scaling** — Counting stats divided by 26 weeks. Rate stats unchanged.
- **`sort_roster_for_display()` returns a copy** — Never modifies input DataFrame.

### Dependencies
- **Pre-commit hook** — `scripts/pre-commit` runs ruff format + lint on staged files. Install with `python scripts/install-hooks.py`.
- **PyMC/PuLP/XGBoost optional** — `PYMC_AVAILABLE`/`PULP_AVAILABLE`/`XGBOOST_AVAILABLE` flags. CI skips with `@pytest.mark.skipif`.
- **Plotly 6.x** — Does NOT accept 8-digit hex (`#RRGGBBAA`). Use `rgba(r,g,b,a)`.
- **`ruff` on Windows** — Use `python -m ruff check .` instead of bare `ruff`.
- **`datetime.now(UTC)`** — Used everywhere; `datetime.utcnow()` deprecated.
- **Snake draft order** — `round % 2 == 0` forward, `round % 2 == 1` reverse.

## Data Sources

| Source | API | Staleness |
|--------|-----|-----------|
| Players (9K+) | MLB Stats API (40-man + spring training) | 7 days |
| Projections (7 systems) | FanGraphs JSON API + Marcel local | 24 hours |
| ROS Projections | FanGraphs (Steamer/ZiPS/DC ROS) | 24 hours |
| ADP | FanGraphs + FantasyPros ECR + NFBC | 24 hours |
| Current stats | MLB Stats API | 1 hour |
| Historical (3 years) | MLB Stats API | 30 days |
| Park factors | pybaseball (dynamic mid-season refresh) | 30 days |
| League data | Yahoo Fantasy API (optional) | 30 minutes |
| News | MLB Stats API + Yahoo | 1 hour |
| Prospects | FanGraphs Board API + MiLB stats | 7 days |
| ECR (6 sources) | Multi-platform aggregation (Trimmed Borda) | 24 hours |
| Weather | Open-Meteo (30 stadiums, dome detection) | 2 hours |
| Team strength | pybaseball wRC+/FIP | 24 hours |
| Opposing pitchers | MLB Stats API + platoon splits | 2 hours |
| Statcast | pybaseball (EV, barrel, xwOBA, Stuff+, sprint speed) | 24 hours |
| Umpire tendencies | Baseball Savant + Retrosheet | 7 days |
| Catcher framing | Baseball Savant | 7 days |

## GitHub

- **Repo:** https://github.com/hicklax13/HEATER_v1.0.0
- **CI:** All green (ruff + pytest across Python 3.11-3.13 + build check)
- **ROADMAP:** 115/115 improvement items completed (6 phases)

## Testing

- **~2960 passing tests** across 140 test files, ~14 skipped (PyMC/XGBoost optional)
- **CI:** GitHub Actions — ruff lint/format, pytest (3.11, 3.12, 3.13), build check
- **Coverage:** ~65% (60% CI floor)
- **Pre-commit hook:** Enforces `ruff format` + `ruff check` on every commit
- **Backtesting framework:** Historical replay validates engine recommendations vs actual outcomes

## Resume Checklist (New Session)

1. Read `CLAUDE.md` (this file)
2. Run `python -m pytest -x -q` to verify tests pass
3. Run `streamlit run app.py` and verify Yahoo auto-reconnect
4. Check `docs/ROADMAP.md` for current state (115/115 complete)

# HEATER — Fantasy Baseball In-Season Manager

## Overview

A fantasy baseball draft assistant + in-season manager for a 12-team Yahoo Sports H2H Categories snake draft league. Built with Streamlit, powered by Monte Carlo simulation, Bayesian projection updating, and LP-constrained lineup optimization.

1. **Draft Tool** (`app.py`) — Heater-themed Streamlit app with glassmorphic design, splash screen bootstrap, setup wizard, 3-column draft page, Monte Carlo recommendations with percentile sampling.
2. **In-Season Management** (`pages/`) — 11 pages: team overview, draft simulator, trade analysis, player comparison, free agents, lineup optimizer, closer monitor, league standings, leaders/prospects, trade finder, matchup planner.
3. **Trade Analyzer Engine** (`src/engine/`) — 6-phase pipeline: deterministic SGP → stochastic MC → signal intelligence → contextual adjustments → game theory → production convergence/caching.
4. **Lineup Optimizer** (`src/optimizer/`) — 21-module pipeline: enhanced projections, matchup adjustments, H2H weights, non-linear SGP, streaming, scenarios, dual objective, advanced LP, category urgency, daily optimizer (DCV), FA recommender, pivot advisor, shared data layer, data freshness tracking, constants registry (30 constants w/ citations), backtest validator, backtest runner, sensitivity analysis, sigmoid calibrator.
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
python -m pytest                       # Run all tests (~3385 pass, ~13 skipped)
python -m pytest tests/test_foo.py -v  # Run single test file

# Optimizer validation tools
python scripts/run_backtest.py --quick          # Replay 3 historical MLB weeks, score accuracy
python scripts/compute_empirical_stats.py       # Compare correlation/CV defaults vs real MLB data
python scripts/calibrate_sigmoid.py --full      # Grid-search optimal sigmoid k-values
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
  6_Line-up_Optimizer.py    — 6-tab optimizer: Start/Sit, Optimize, Manual, Streaming, Daily, Roster
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
  alerts.py                 — Proactive alerts: auto-IL, closer handcuff, regression alerts
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
  optimizer/                — Enhanced Lineup Optimizer (21 modules)
    pipeline.py             — Master orchestrator (Quick/Standard/Full modes)
    projections.py          — Enhanced projections (Bayesian/Kalman/regime/injury/recent form)
    matchup_adjustments.py  — Park factors, platoon splits, weather, umpire, catcher framing, PvB
    h2h_engine.py           — H2H category weights + win probability
    sgp_theory.py           — Non-linear marginal SGP
    streaming.py            — Pitcher streaming + Bayesian stream scoring + two-start fatigue
    scenario_generator.py   — Gaussian copula scenarios + CVaR + empirical correlation utils
    dual_objective.py       — H2H/Roto weight blending
    advanced_lp.py          — Maximin, epsilon-constraint, stochastic MIP
    category_urgency.py     — Sigmoid urgency for H2H matchup gaps
    daily_optimizer.py      — Daily Category Value (DCV) per-player per-day scoring
    fa_recommender.py       — Free agent recommendation engine
    pivot_advisor.py        — Mid-week pivot recommendations
    shared_data_layer.py    — Unified data infrastructure + data timestamps across optimizer modules
    data_freshness.py       — Per-source staleness tracking with TTLs + UI badge support
    constants_registry.py   — 30 optimizer constants with citations, bounds, sensitivity levels
    backtest_validator.py   — RMSE, Spearman rank correlation, bust rate, lineup grading
    backtest_runner.py      — Historical replay via statsapi game logs (20-player roster, 10 weeks)
    sensitivity_analysis.py — Mock.patch perturbation of 11 constants, lineup/weight diff scoring
    sigmoid_calibrator.py   — Grid-search calibration of sigmoid k-values for H2H urgency
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
  run_backtest.py           — CLI: replay historical MLB weeks through optimizer (--quick/--weeks/--verbose)
  compute_empirical_stats.py — CLI: compute Spearman correlations + CVs from pybaseball 2022-2024 data
  calibrate_sigmoid.py      — CLI: grid-search optimal sigmoid k-values (--quick/--full/--verbose)
tests/                      — 150 test files, ~3180 passing tests
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
**Post-2026-04-17 audit:** bootstrap runs with `force=True` on every new browser session (per-session guard `bootstrap_complete` prevents re-run on intra-session page navigation). The "Refresh All Data" sidebar button also passes `force=True`, clears `st.cache_data`, and preserves `bootstrap_results` for the Data Status panel.
Key thresholds (still used inside force-refresh to estimate ETA): 1h live stats/news, 30min Yahoo, 2h game-day, 24h projections/ADP/ECR, 7d players/prospects.
Per-phase timeouts: default 180s, 300s for game_logs + ROS projections (PyMC), 240s for ECR consensus.

### Bootstrap Timing (measured 2026-04-12, Python 3.14, Windows)
- **Cold start (all stale):** ~30 minutes total
  - Players (MLB Stats API 40-man + spring): ~2-3 min
  - Projections (FanGraphs 7 systems): ~3-5 min
  - Live Stats (9K+ player season stats): ~8-12 min (largest phase, network-bound)
  - Deduplication: <10 seconds (batch SQL via temp table)
  - Yahoo sync (rosters, standings, FA, transactions): ~3-5 min
  - News Intelligence (ESPN + RotoWire + MLB): ~2-3 min
  - Depth Charts + ADP + ECR: ~2-3 min
  - Remaining phases (prospects, park factors, etc.): ~1-2 min
- **Warm start (most data cached, <1h old):** ~2-5 minutes (only stale sources refresh)
- **Hot start (all data fresh):** ~5-10 seconds (staleness checks only, no fetches)
- **Bottleneck:** Live Stats fetch is the largest phase — MLB Stats API returns all 9K+ players' current season stats in one large response.
- **Rate limits:** Yahoo ~2K-3K API calls/day. Safe to do 10-15 full refreshes/day. Don't spam Force Refresh.

### Player Databank Integration (2026-04-13)
- **Game logs table** (`game_logs`): Per-game stats (PA, H, HR, RBI, SB, IP, K, ERA per game) for rolling L7/L14/L30 windows.
- **`compute_rolling_stats()`** in `src/player_databank.py`: Batch SQL query for rolling totals, averages, or standard deviations. Used by Player Compare (L7/L14 display), Free Agents (L14 columns), War Room hot/cold (replaces per-player API calls).
- **Statcast columns in pool**: `xwoba`, `barrel_pct`, `hard_hit_pct`, `stuff_plus`, `babip`, `ev_mean` flow through `load_player_pool()` to all pages. Regression flags (`regression_flag`, `babip_regression_flag`, `stuff_regression_flag`) computed in `_enrich_pool()`.
- **`bats`/`throws`** columns now in player pool SQL — enables platoon splits in DCV matchup multiplier.
- **`ecr_rank_stddev`** now in player pool SQL — enables ECR confidence display.
- **DCV recent form**: Fixed `"last_14"` → `"l14"` key bug. Now adjusts counting stats (HR/RBI/SB/K/W/SV) in addition to rate stats.
- **Statcast in Standard mode**: `enable_statcast` now fires in Standard (not just Full) optimizer mode.
- **Auto-advance date**: `get_target_game_date()` in `src/game_day.py` returns tomorrow's date when all today's games are final. Used by optimizer, game_day, shared_data_layer, war_room_actions, weekly_report.
- **Design spec**: `docs/superpowers/specs/2026-04-13-databank-integration-design.md`

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

# Optimizer validation tools
from src.optimizer.data_freshness import DataFreshnessTracker
tracker = DataFreshnessTracker()
tracker.record("projections")                # Record data was refreshed now
tracker.check("projections") -> FreshnessStatus  # FRESH/STALE/UNKNOWN

from src.optimizer.constants_registry import CONSTANTS_REGISTRY
CONSTANTS_REGISTRY["sigmoid_k_counting"].value  # 2.0, with citation + bounds

from src.optimizer.sensitivity_analysis import run_sensitivity_analysis, summarize_results
results = run_sensitivity_analysis(roster, constants_to_test=["sigmoid_k_counting"])
summarize_results(results) -> pd.DataFrame  # With mismatch column

from src.optimizer.backtest_runner import run_backtest, format_report
report = run_backtest(roster, weeks=[(date(2025,4,7), date(2025,4,13))])
print(format_report(report))  # RMSE, Spearman, bust rate, grades

from src.optimizer.sigmoid_calibrator import calibrate_sigmoid_k, recommend_k_values
result = calibrate_sigmoid_k(roster)  # Grid search over k-value pairs
print(recommend_k_values(roster))     # Human-readable recommendation

from src.optimizer.scenario_generator import load_cached_empirical_stats, compare_to_defaults
cached = load_cached_empirical_stats("data/empirical_stats.json")
divergences = compare_to_defaults(cached)  # Flags >20% divergence

# Player Databank
from src.player_databank import compute_rolling_stats, load_databank
rolling = compute_rolling_stats([player_id], days=14, stat_type="total")  # L14 totals
rolling_avg = compute_rolling_stats([player_id], days=14, stat_type="avg")  # Per-game avg
rolling_std = compute_rolling_stats([player_id], days=14, stat_type="stddev")  # Consistency
df = load_databank("S_L14")  # Full databank view for L14

# Auto-advance game date
from src.game_day import get_target_game_date
target = get_target_game_date()  # Today, or tomorrow if all games final
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
- **SQLite WAL mode + busy_timeout** (2026-04-17 trust audit) — `get_connection()` in `src/database.py` sets `PRAGMA journal_mode=WAL`, `PRAGMA busy_timeout=30000`, `PRAGMA synchronous=NORMAL` on every connection. Root-cause fix for 6 bootstrap phases that failed with "database is locked" when the `ThreadPoolExecutor` blocks ran concurrent writes to per-team tables. Do NOT call `sqlite3.connect()` directly elsewhere — always use `get_connection()`.
- **Game logs scoped to rostered/40-man players** — `fetch_game_logs_from_api` in `src/player_databank.py` joins `league_rosters` first (drops from ~9K to ~240), falls back to 40-man / active roster. Prevents the 90-second timeout caused by iterating all players.
- **FanGraphs 403 handling** — `leaders-legacy.aspx` returns 403 to non-browser scrapers; `_format_fetch_error` in `src/data_bootstrap.py` translates 403/429/timeout signatures to "Skipped: ..." messages so Data Status shows known limitations clearly. Team strength has a built-in `_fetch_team_strength_statsapi` fallback to MLB Stats API.

### Yahoo API
- **OAuth uses oob** — `redirect_uri=oob` required. No localhost redirects.
- **Auto-reconnect** — `_try_reconnect_yahoo()` loads `data/yahoo_token.json`. Token refresh automatic.
- **yfpy Roster iteration** — `Roster.__iter__()` yields attribute NAMES (strings), NOT players. Use `getattr(roster, "players", None) or []`.
- **Yahoo game_key** — MLB 2026 = 469. Must set BOTH `self._query.game_id` AND `self._query.league_key`.
- **429 backoff** — `_request_with_backoff()` retries 3x with exponential delay.
- **yfpy eligible_positions bug** — yfpy's `Player.eligible_positions` attribute drops multi-position data (e.g., returns `["2B"]` instead of `["2B", "OF"]`). Workaround: read from `player._extracted_data["eligible_positions"]` which preserves the full list. Fixed in both `get_all_rosters()` and `get_team_roster()` methods. Separator is `,` (not `/`).

### Streamlit UI
- **No emoji** — All icons are inline SVGs from `PAGE_ICONS`. Injury badges use CSS dots.
- **CSS `!important` required** — Streamlit's inline CSS has high specificity.
- **HTML sanitization** — `st.markdown('<div>')` auto-closes tags. Use self-contained HTML blocks.
- **`T["ink"]` for text-on-accent** — Not `T["bg"]` (invisible on amber).
- **Connection leak** — Always wrap `get_connection()` in `try/finally` with `conn.close()`.
- **`T["tx"]` not `T["tx1"]`** — There is no `tx1` key.
- **Splash screen load timer** (2026-04-17) — HH:MM:SS live counter below the progress bar. Start recorded in `st.session_state["bootstrap_start_time"]`. Final elapsed stored in `bootstrap_elapsed_secs` / `bootstrap_elapsed_hms`. Use `_format_elapsed_hms(secs)` helper in `app.py`. Total Load Time is the last item in the Data Status expander on the Connect League page.
- **Force Refresh button behavior** — Renamed to "Refresh All Data". Calls `st.cache_data.clear()` BEFORE bootstrap, passes `force=True`, assigns the returned results back to `bootstrap_results` (prior bug: set to `None`, wiping the Data Status panel), and records a new elapsed timer.

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
- **Recent form blend** — Dynamic weight: linear interpolation from 0.10 (7 games) to scope max (14+ games). Below 7 games = 0 weight.
- **Matchup data before state classification** — `yds.get_matchup()` must be called before `classify_matchup_state()`.
- **IP budget uses projected IP** — Not actual IP pitched.
- **Weekly scaling** — Counting stats divided by 26 weeks. Rate stats unchanged.
- **`sort_roster_for_display()` returns a copy** — Never modifies input DataFrame.
- **Two-start pitcher fatigue** — 0.93x rate stat quality on 2nd start (~7% ERA/WHIP decay). Applied in streaming module.
- **Platoon splits (2020-2024)** — LHB vs RHP: 7.5% wOBA advantage (was 8.6%). RHB vs LHP: 5.8% (was 6.1%).
- **Sigmoid urgency** — `COUNTING_STAT_K=2.0`, `RATE_STAT_K=3.0`. Calibrate with `python scripts/calibrate_sigmoid.py`.
- **Data freshness UI** — Colored CSS dots (green=fresh, orange=stale) in optimizer context panel expander.
- **Constants registry** — 30 constants in `constants_registry.py` with citations, bounds, and sensitivity levels. Run sensitivity analysis to verify declared vs actual sensitivity.
- **DCV gate — non-probable pure SPs get volume=0.0** (2026-04-17 trust audit). Previously a pure SP whose team played today but who wasn't a probable starter got `volume_factor=0.9` (ghost value), landing in P slots with DCV 1-2 and flagged START. Now `volume=0.0` for pure SPs absent from `probable_starters`. SP/RP hybrids keep `volume=0.9` (can still relieve). Pure RPs still default to 0.9.
- **Pitcher probable-starter name matching** — uses `_normalize_pitcher_name` (accents, suffixes, punctuation stripped) so "José Ramírez" matches "Jose Ramirez" and "Luis Garcia Jr." matches "Luis Garcia". Prevents silent gate misses from API/roster string drift.
- **Matchup multiplier resolves opponent + venue from schedule** (2026-04-17 trust audit). Previously the caller passed `opponent_team=""`, which made `park_factor_adjustment` fall back to the player's own team park regardless of home/away — Moniak (COL) always got 1.38. The caller now walks `schedule_today`, determines if the player is home or away, and passes the **home team code** as `opponent_team` (the convention used by `park_factor_adjustment`). Opposing pitcher throws + xFIP are also resolved from the roster pool and passed through.
- **Pitcher leave-empty threshold** — `_PITCHER_EMPTY_THRESHOLD = _pitcher_median * 0.20` in `pages/6_Line-up_Optimizer.py` (was 0.05). Raised so marginal SP/RP hybrids with weak matchups go BENCH rather than filling open P slots.
- **FA streaming drop-candidate is slot-aware** — `_pick_drop(fa_is_hitter, fa_positions)` filters roster players by position overlap with the FA's positions. Prior bug: globally-worst player per side, which caused every batter stream to target the same drop candidate regardless of slot. Cross-swap (drop batter for pitcher stream) still uses global worst by design.

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

- **~3385 passing tests** across 151 test files, ~13 skipped (PyMC/XGBoost optional)
- **CI:** GitHub Actions — ruff lint/format, pytest (3.11, 3.12, 3.13), build check
- **Coverage:** ~65% (60% CI floor)
- **Pre-commit hook:** Enforces `ruff format` + `ruff check` on every commit
- **Backtesting framework:** Historical replay validates engine recommendations vs actual outcomes
- **Optimizer validation suite** (220 tests across 10 files):
  - `test_optimizer_math_proofs.py` — 29 hand-verified formula proofs (rate stats, LP, Bayesian, SGP, correlations)
  - `test_shared_data_layer.py` — 18 data context contract tests (fields, scaling, form weight, timestamps)
  - `test_optimizer_integration.py` — 48 cross-module tests (3 modes × 3 alphas × 5 invariants + settings axis)
  - `test_data_freshness.py` — 6 freshness tracker tests
  - `test_constants_registry.py` — 10 constants validation tests (citations, bounds, sensitivity)
  - `test_optimizer_backtest.py` — 10 accuracy metric tests (RMSE, Spearman, bust rate, grading)
  - `test_sensitivity_analysis.py` — 32 perturbation framework tests
  - `test_backtest_runner.py` — 34 historical replay tests (IP parsing, aggregation, scaling, API fallback)
  - `test_empirical_stats.py` — 15 correlation/CV computation tests
  - `test_sigmoid_calibrator.py` — 18 calibration framework tests (H2H sim, urgency scoring, grid search)

## Resume Checklist (New Session)

1. Read `CLAUDE.md` (this file)
2. Run `python -m pytest -x -q` to verify tests pass
3. Run `streamlit run app.py` and verify Yahoo auto-reconnect
4. Check `docs/ROADMAP.md` for current state (115/115 complete)

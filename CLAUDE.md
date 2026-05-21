# HEATER — Fantasy Baseball In-Season Manager

## Overview

A fantasy-baseball draft assistant + in-season manager for a 12-team Yahoo Sports H2H Categories snake-draft league. Built with Streamlit, powered by Monte Carlo simulation, Bayesian projection updating, and LP-constrained lineup optimization. **2026 MLB season is active; the app is in in-season mode** — draft completed, all components running against live league + Yahoo data.

The codebase is organized around 7 feature surfaces:

1. **Draft Tool** (`app.py`) — Heater-themed splash + bootstrap + setup wizard + 3-column draft page with Monte Carlo recommendations.
2. **In-Season Pages** (`pages/`) — 13 pages, consolidated 2026-05-18 via Section 5 audit: My Team, Lineup Optimizer, Closer Monitor, Matchup Planner, League Standings (+ Playoff Odds tab), Punt Analyzer, Trade Analyzer, Trade Finder (+ Value Chart tab), Free Agents, Player Compare, Leaders (+ Hot/Cold/Sell-High tabs), Player Databank, Draft Simulator. Bullpen, Weekly_Dashboard, Weekly_Recap, Waiver_Wire pages removed — their underlying engine modules (`src/contextual_factors.py`, `src/weekly_report.py`, `src/waiver_wire.py`) remain callable.
3. **Trade Analyzer Engine** (`src/engine/`) — 6-phase pipeline: deterministic SGP → stochastic MC (paired, true antithetic) → signal intelligence → contextual adjustments → game theory → production convergence/caching.
4. **Lineup Optimizer** (`src/optimizer/`) — 21-module pipeline with PuLP LP, daily category value (DCV) scoring, sigmoid urgency, FA recommender, sensitivity analysis, backtest framework.
5. **Draft Recommendation Engine** (`src/draft_engine.py`) — 8-stage enhancement chain with 3 execution modes (Quick/Standard/Full).
6. **War Room** (`src/war_room*.py`) — Mid-week pivot analysis, category flip probability, hot/cold detection.
7. **Backtesting** (`src/backtesting*.py` + `src/optimizer/backtest_runner.py`) — Historical replay framework validating recommendations against actual outcomes.

## Local Environment

- **Project root (local):** `C:\Users\conno\Code\HEATER_v1.0.1` (note: local folder is `v1.0.1`; GitHub repo NAME remains `HEATER_v1.0.0`).
- **Relocated 2026-05-17** from `C:\Users\conno\OneDrive\Desktop\HEATER_v1.0.0`. OneDrive's Cloud Files API conflicts with Cowork's FUSE/virtiofs mount layer (anthropics/claude-code issues #25293 and #40973), corrupts SQLite WAL files when streamlit runs locally, locks `.venv` operations, and conflicts with `.git` writes. Never store this project under OneDrive, Dropbox, iCloud, or any cloud-sync directory.
- **Local Python:** 3.14 preferred, 3.12 acceptable. Recreate venv with `py -3.14 -m venv .venv` (or `py -3.12 -m venv .venv`).
- **Yahoo OAuth deps:** `yfpy` and `streamlit-oauth` must be installed with `--no-deps` after `pip install -r requirements.txt`, per the comment in `requirements.txt` (python-dotenv pin conflict on 3.14).
- **Hooks:** Reinstall after any fresh venv via `python scripts/install-hooks.py`.

## Commands

```bash
pip install -r requirements.txt        # Install deps (or `uv pip install --system -r requirements.txt` for ~10× faster)
python scripts/install-hooks.py        # Install pre-commit + pre-push hooks (first time, per machine)
python load_sample_data.py             # Load sample data (first time/testing)
streamlit run app.py                   # Run the app
python -m ruff check .                 # Lint
python -m ruff format .                # Format
python -m pytest --ignore=tests/test_cheat_sheet.py  # Full suite (~3900 pass; cheat_sheet skipped on Windows)
python -m pytest tests/test_foo.py -v  # Single test file

# Parallel test execution (matches CI's sharded layout)
python -m pytest tests/ -n auto --dist loadfile      # Run all in parallel locally (~4× faster)
python -m pytest tests/ --splits 4 --group 1 -n 2 --dist loadfile  # Run shard 1 of 4
python -m pytest tests/ --store-durations             # Regenerate .test_durations (weekly cadence)

# Optimizer validation tools
python scripts/run_backtest.py --quick          # Replay historical MLB weeks, score accuracy
python scripts/compute_empirical_stats.py       # Compare correlation/CV defaults vs real MLB data
python scripts/calibrate_sigmoid.py --full      # Grid-search optimal sigmoid k-values
python scripts/calibrate_repl_baselines.py      # Refresh _REPL_* from FourzynBurn standings (annual)
```

## League Context

- **League:** FourzynBurn (Yahoo Sports) | **Team:** Team Hickey
- **Format:** 12-team snake draft, 23 rounds, Head-to-Head Categories
- **Hitting cats (6):** R, HR, RBI, SB, AVG, OBP
- **Pitching cats (6):** W, L, SV, K, ERA, WHIP
- **Inverse cats:** L, ERA, WHIP (lower is better)
- **Rate stats:** AVG, OBP, ERA, WHIP
- **Roster:** C/1B/2B/3B/SS/3OF/2Util/2SP/2RP/4P/6BN/4IL = 28 slots
- **Yahoo game_key:** 469
- **Transactions:** 10 adds+trades combined per matchup week (FCFS waivers)
- **Can't-drop:** Players drafted in rounds 1-3 per team (36 league-wide)

## Tech Stack

- **Framework:** Streamlit (multi-page)
- **Database:** SQLite (`data/draft_tool.db`) with WAL + 60s busy_timeout via `get_connection()`
- **Core libs:** pandas, NumPy, SciPy, Plotly
- **Analytics:** PyMC 5 (Bayesian, optional), PuLP (LP optimizer), arviz
- **Live data:** MLB-StatsAPI, pybaseball (FanGraphs/Statcast), Open-Meteo (weather), Baseball Savant (catcher framing)
- **Yahoo API:** yfpy + streamlit-oauth (optional OAuth)
- **Linter:** ruff (lint + format), pre-commit + pre-push hooks enforced
- **CI:** GitHub Actions — ruff lint/format + pytest sharded (4 groups × 2 xdist workers, Python 3.12) + 60% coverage floor; uv-based dep install; concurrency cancels in-flight PR runs; expected ~5-7 min total wall time
- **Python:** Local dev uses 3.14; CI tests on 3.12 (sharded 4 ways via pytest-split)

## File Structure

```
app.py                      — Splash, bootstrap, setup wizard, draft page
requirements.txt            — pip dependencies
load_sample_data.py         — Sample data generator for testing
.streamlit/config.toml      — Light theme (Heater palette)

pages/ — 13 in-season pages, ordered for daily workflow (status → daily action → strategy → trades → wire → research → preseason):
  # Status & this-week context
  1_My_Team.py              — War Room, roster, category standings, alerts, Yahoo sync
  2_Line-up_Optimizer.py    — 6-tab optimizer: Start/Sit, Optimize, Manual, Streaming, Daily, Roster
  3_Closer_Monitor.py       — 30-team closer depth chart grid
  5_Matchup_Planner.py      — Category probabilities, player matchups, per-game detail
  6_League_Standings.py     — 3 tabs: Current Standings / Season Projections / Playoff Odds
  10_Punt_Analyzer.py       — Punt-category strategy recommender
  # Trades
  11_Trade_Analyzer.py      — Trade proposal builder + 6-phase engine
  12_Trade_Finder.py        — 5 tabs: Recs / Target / Browse Partners / Readiness / Value Chart
  # Wire
  14_Free_Agents.py         — FA rankings by marginal value + ownership heat
  # Research
  16_Player_Compare.py      — Head-to-head comparison with category fit
  17_Leaders.py             — 7 tabs: Leaders / Value / Breakouts / Prospects / Hot / Cold / Sell-High
  19_Player_Databank.py     — Historical multi-year player lookup
  # Preseason
  20_Draft_Simulator.py     — Draft simulator with AI opponents, MC recommendations

# Section 5 (2026-05-18) consolidated 20 → 13 pages.
# Removed pages: 4_Bullpen, 7_Playoff_Odds, 8_Weekly_Dashboard,
# 9_Weekly_Recap, 13_Trade_Values, 15_Waiver_Wire, 18_Trends.
# Their core engine logic lives in src/ and remains importable.

src/
  # Core
  valuation.py              — LeagueConfig (sole source of truth), SGPCalculator, VORP
  database.py               — SQLite schema, _build_player_pool (full pool with all ytd_*, statcast)
  data_bootstrap.py         — 33-phase bootstrap orchestrator (3-tier waterfall on most phases)
  data_pipeline.py          — FanGraphs auto-fetch (Steamer/ZiPS/Depth Charts)
  data_fetch_utils.py       — fetch_with_fallback + browser-headers helpers
  yahoo_api.py              — Yahoo Fantasy API OAuth (429 backoff, ghost filtering)
  yahoo_data_service.py     — 3-tier cache: session_state → Yahoo API → SQLite

  # Draft
  draft_engine.py / draft_state.py / draft_analytics.py / draft_grader.py
  draft_order.py / simulation.py / pick_predictor.py

  # Trade system
  trade_finder.py           — V4 trade finder (cosine→1v1→2v1)
  trade_intelligence.py     — Health/scarcity/FA gating/correlation
  trade_value.py            — Universal trade value chart (0-100)
  trade_signals.py          — Kalman + regime trend
  opponent_trade_analysis.py — Opponent needs, archetype detection
  in_season.py              — Legacy trade analyzer, player comparison

  # Lineup optimization
  lineup_optimizer.py       — PuLP LP solver (sole PULP_AVAILABLE source)
  lineup_rl.py              — Category-aware RL optimizer

  # Matchup & strategy
  matchup_context.py        — MatchupContextService (canonical category weights)
  matchup_planner.py / standings_engine.py / standings_utils.py
  weekly_h2h_strategy.py / war_room*.py
  start_sit.py / start_sit_widget.py

  # Player intelligence
  bayesian.py / injury_model.py / ml_ensemble.py / playing_time_model.py
  projection_stacking.py / ecr.py / prospect_engine.py
  player_news.py / news_fetcher.py / news_sentiment.py
  espn_injuries.py / player_card.py / player_tags.py

  # Game-day intelligence
  game_day.py               — Weather, opposing pitchers, team strength, lineups
  closer_monitor.py / schedule_grid.py / contextual_factors.py

  # In-season management
  alerts.py / opponent_intel.py / waiver_wire.py / weekly_report.py
  ip_tracker.py / il_manager.py / two_start.py / power_rankings.py / leaders.py
  league_rules.py           — Undroppable list + 10 txn/week + FCFS

  # Data sources
  live_stats.py / adp_sources.py / depth_charts.py / contract_data.py
  marcel.py / data_2026.py

  # Backtesting & validation
  backtesting.py / backtesting_framework.py
  validation/               — Calibration, constant optimization, dynamic context, survival

  # UI
  ui_shared.py              — THEME, PAGE_ICONS, format_stat (canonical formatter)
  ui_analytics_badge.py / cheat_sheet.py

  # Misc
  league_manager.py / league_registry.py / live_draft_sync.py
  points_league.py / scheduler.py / analytics_context.py / player_databank.py

  # Sub-packages
  optimizer/                — Enhanced Lineup Optimizer (21 modules)
    pipeline.py             — Master orchestrator (Quick/Standard/Full modes)
    projections.py          — Enhanced projections (Bayesian/Kalman/regime/injury/recent form)
    matchup_adjustments.py  — Park, platoon, weather, umpire, catcher framing, PvB
    h2h_engine.py           — H2H category weights + win probability
    sgp_theory.py           — Non-linear marginal SGP
    streaming.py            — Pitcher streaming + two-start fatigue (uses constants_registry)
    scenario_generator.py   — Gaussian copula + CVaR + empirical correlation
    dual_objective.py       — H2H/Roto weight blending
    advanced_lp.py          — Maximin, epsilon-constraint, stochastic MIP (imports PULP_AVAILABLE)
    category_urgency.py     — Sigmoid urgency (reads k-values from CONSTANTS_REGISTRY at runtime)
    daily_optimizer.py      — Daily Category Value (DCV) scoring + Stuff+/FIP K-boost
    fa_recommender.py       — FA recommendation engine
    pivot_advisor.py        — Mid-week pivot recommendations
    shared_data_layer.py    — Unified data infrastructure + freshness tracker calls
    data_freshness.py       — Per-source staleness tracking with TTLs + UI badge
    constants_registry.py   — 30+ optimizer constants with citations, bounds, sensitivity
    backtest_validator.py   — RMSE, Spearman, bust rate, lineup grading
    backtest_runner.py      — Historical replay via statsapi game logs
    sensitivity_analysis.py — Mock.patch perturbation of constants
    sigmoid_calibrator.py   — Grid-search calibration
  engine/                   — Trade Analyzer Engine (6 phases, 7 sub-packages)
    portfolio/              — Z-scores, SGP via SGPCalculator (no local _LC singleton)
    projections/            — ROS projections, BMA, KDE marginals
    monte_carlo/            — Paired MC simulator (TRUE antithetic via negate-uniforms)
    signals/                — Statcast, decay, Kalman filter, regime detection
    context/                — Log5 matchup, injury process, bench value
    game_theory/            — opponent_valuation (sgp_denominators required), Bellman DP, sensitivity
    production/             — Convergence diagnostics, cache, adaptive sim scaling
    output/                 — Master trade orchestrator (evaluate_trade)

scripts/
  install-hooks.py / pre-commit
  run_backtest.py / compute_empirical_stats.py / calibrate_sigmoid.py
  draft_vs_current.py / extract_trade_data.py / optimal_roster_sim.py  (all use get_connection)

  # FA engine diagnostics (preserved 2026-05-20 → 2026-05-21 for ongoing debug)
  diag_fa_cross_type.py            — Traces _allow_cross_type for known bad rec pairs
  diag_fa_with_filter.py           — Verifies user-roster slicing fix end-to-end
  diag_fa_full_trace.py            — Stage-by-stage pipeline trace (drops, FAs, evaluate_swaps)
  diag_fa_score_plateau.py         — Inspects _compute_base_value internals for inverse-stat bug
  diag_fa_stage_by_stage.py        — Full recommend_fa_moves trace with intermediate counts
  diag_fa_adds_budget.py           — ctx.adds_remaining_this_week + transactions check
  diag_muncy_data.py               — DB column inspection for Muncy DNA collision
  diag_roster_data_audit.py        — Audits ALL rostered players for DNA collisions + stale YTD
  diag_fa_score_plateau.py         — Why unknown minor leaguers score composite=20.6

  # One-time DB migration (2026-05-21)
  migrate_muncy_dna_2026_05_21.py  — Insert LAD Muncy + repoint league_rosters (idempotent)

tests/                      — 195+ test files, ~4200 passing tests
  conftest.py               — Session-scoped league_standings fixture (autouse)
  test_no_*.py              — Structural-invariant guards (see Structural Invariants section)
  test_sf*.py               — TDD tests for SF-1..SF-28 fixes
data/
  draft_tool.db             — SQLite database (created at runtime)
  seed/                     — 2024 seed JSONs (catcher_framing, umpire_tendencies) for SF-7 Tier 3
  backups/                  — Draft state JSON backups
  logs/bootstrap.log        — Persistent bootstrap log (SF-14)
docs/
  architecture.md           — Layered system architecture reference
  Research.md               — Competitive gap analysis vs FantasyPros
  VERIFICATION_LOG.md       — V-001..V-017 browser-verified validation sessions
  2026-04-08-heater-postmortem.md  — Early-season recommendation-failure RCA
  legal/                    — LLC formation, ToS, privacy policy, data sources, etc.
  archive/                  — Stale historical docs (ROADMAP, AUDIT_REPORT, BETA_*)
.github/
  workflows/ci.yml          — CI pipeline (lint + sharded pytest 4×2 workers on 3.12 + coverage floor + build check)
  workflows/refresh.yml     — Scheduled daily data refresh
```

## Architecture

### Core Valuation Pipeline
1. **Projection blending** — Ridge regression weighted stacking of 7 systems (Steamer, ZiPS, Depth Charts, ATC, THE BAT, THE BAT X, Marcel).
2. **Dynamic SGP** — Converts raw stats to standings-point movement; Bayesian-updated denominators from league data.
3. **VORP** — Value Over Replacement Player with graduated positional scarcity (C=1.20×, 2B=1.15×, SS=1.10×).
4. **pick_score** = weighted SGP + positional scarcity + VORP bonus.

### Unified Services (sole sources of truth)
- **`LeagueConfig`** (`src/valuation.py`) — Category lists, inverse stats, rate stats, SGP denominators. **Never hardcode category lists** — structural test guards this in src/, pages/, and tests.
- **`SGPCalculator`** (`src/valuation.py`) — Sole path for SGP computation. The `totals_sgp(totals, weights=None)` method consolidates 7 prior local reinventions across trade_finder, waiver_wire, daily_optimizer, trade_evaluator, trade_simulator, in_season, start_sit. **`marginal_sgp(player, roster_totals, weights)`** handles inverse-stat signs correctly (PR #99 / FA P3.7) — consumed via `_compute_base_value` in `fa_recommender.py`.
- **`MatchupContextService`** (`src/matchup_context.py`) — Single source for category weights with 3 modes: "matchup" (H2H urgency), "standings" (gap analysis), "blended" (alpha-weighted).
- **`standings_utils.get_team_totals()`** — Session-cached roster totals.
- **`get_yahoo_data_service()`** — Singleton; 3-tier cache: session_state → Yahoo API → SQLite. All 11+ pages route through it (no raw `load_league_rosters/standings` in pages).
- **`format_stat(value, stat_type)`** (`src/ui_shared.py`) — Enforced display: AVG/OBP=`.3f`, ERA/WHIP=`.2f`, SGP=`+.2f`.
- **`get_connection()`** (`src/database.py`) — Only sanctioned SQLite access. Sets `PRAGMA journal_mode=WAL`, `busy_timeout=60000`, `synchronous=NORMAL`. Direct `sqlite3.connect` is structurally guarded against.
- **`PULP_AVAILABLE`** — Defined ONCE in `src/lineup_optimizer.py`. `src/optimizer/advanced_lp.py` imports it (no duplicate flag).
- **`CONSTANTS_REGISTRY`** (`src/optimizer/constants_registry.py`) — 50+ optimizer/FA-engine constants with citations, bounds, sensitivity tags. **Never hardcode tunable thresholds in `fa_recommender.py` / `waiver_wire.py`** — register them. After PR #107 the FA engine reads all major thresholds (streaming, playing-time, ownership, floor, position-cap, urgency) from the registry at import time.
- **`compute_positional_scarcity_factor(positions, replacement_levels)`** (`src/valuation.py`) — Shared scarcity math (PR #96). Both `fa_recommender._score_fa_candidates` (adds) and `waiver_wire.compute_drop_cost` (drops) consume it for symmetric scarcity treatment. Moved here from `fa_recommender.py` to break the `waiver_wire ← fa_recommender` import cycle.
- **`get_il_stash_names()`** (`src/alerts.py`) — Dynamic derivation of IL stash names from `league_rosters.status IN (IL10, IL15, IL60, IL, DTD)` (PR #102). The module-level `IL_STASH_NAMES` constant calls this at import time. Pages should call the function directly for fresh data each render. Now surfaces ~52 IL players league-wide vs the previously hardcoded 2 (Bieber + Strider).

### Yahoo Data Service (Live-First Architecture)
3-tier cache: `st.session_state` → Yahoo Fantasy API → SQLite fallback.
TTLs: Rosters 30m, Standings 30m, Matchup 5m, Free Agents 1h, Transactions 15m, Settings/Schedule 24h.
Write-through: every Yahoo fetch writes to SQLite. Singleton via `get_yahoo_data_service()`.

### Bootstrap Pipeline (33 phases, 3-tier waterfall)
Bootstrap runs with `force=True` on every new browser session (per-session guard `bootstrap_complete` prevents re-run on intra-session navigation). The "Refresh All Data" sidebar button passes `force=True`, clears `st.cache_data`, preserves `bootstrap_results` for the Data Status panel, and records a new elapsed timer.

Most phases follow a 3-tier waterfall:
- **Tier 1 (primary)** — Live API (pybaseball / FanGraphs / Yahoo / statsapi)
- **Tier 2 (fallback)** — Browser-headers scraper (where applicable, e.g. Savant for SF-7)
- **Tier 3 (emergency)** — Shipped seed file or hardcoded baseline (catcher_framing, umpire_tendencies, park_factors)

`refresh_log` records `tier='primary'/'fallback'/'emergency'` per phase. Operators can query the full snapshot via `get_refresh_log_snapshot()` (added 2026-05-11).

Per-phase timeouts: default 180s, 300s for game_logs + ROS projections (PyMC), 240s for ECR consensus.

### Player Pool (`_build_player_pool`)
The canonical enriched pool returned by `load_player_pool()` includes:
- Projection-blended stats (R/HR/RBI/SB/AB/H/BB/HBP/SF/AVG/OBP for hitters; W/L/SV/K/IP/ER/BB_allowed/H_allowed/ERA/WHIP for pitchers; FIP/xFIP/SIERA)
- ECR rank + consensus + sources + stddev
- ADP from FanGraphs/FantasyPros/NFBC
- Statcast: xwoba, xba, barrel_pct, hard_hit_pct, ev_mean, stuff_plus, babip, **sprint_speed**
- YTD stats (full set, supports weighted aggregation): `ytd_r/hr/rbi/sb/avg/obp/ab/h/bb/hbp/sf/pa/gp` + `ytd_w/l/sv/k/era/whip/ip/er/bb_allowed/h_allowed`
- Regression flags (`regression_flag`, `babip_regression_flag`, `stuff_regression_flag`)
- `bats`/`throws` (enables platoon splits)
- `percent_owned` (ownership trends)

Pages should consume the pool — direct SQL is structurally guarded against in `pages/1_My_Team.py`, `pages/11_Trade_Analyzer.py`, `pages/16_Player_Compare.py`.

## Key API Signatures

```python
# LeagueConfig + SGPCalculator — sole source of truth
from src.valuation import LeagueConfig, SGPCalculator
cfg = LeagueConfig()
calc = SGPCalculator(cfg)
calc.totals_sgp(totals_dict, weights=None) -> float   # Sole path for roster-totals SGP
calc.player_sgp(player_series) -> dict                # Per-player per-category SGP

# Yahoo Data Service
from src.yahoo_data_service import get_yahoo_data_service
yds = get_yahoo_data_service()
yds.get_rosters() -> pd.DataFrame                     # 30min TTL
yds.get_standings() -> pd.DataFrame                   # 30min TTL
yds.get_matchup() -> dict | None                      # 5min TTL
yds.get_free_agents(max_players=500)                  # 1hr TTL
yds.get_transactions() -> pd.DataFrame                # 15min TTL
yds.is_connected() -> bool

# Player pool
from src.database import load_player_pool, load_season_stats
pool = load_player_pool()                             # Full enriched pool with all ytd_*

# Trade evaluator
from src.engine.output.trade_evaluator import evaluate_trade
evaluate_trade(giving_ids, receiving_ids, user_roster_ids, player_pool,
               config=None, enable_mc=False, enable_context=True, enable_game_theory=True)

# Trade finder
from src.trade_finder import find_trade_opportunities
find_trade_opportunities(user_roster_ids, player_pool, config=None,
    all_team_totals=None, league_rosters=None, max_results=20)

# Engine portfolio (config required for live denominators)
from src.engine.portfolio.category_analysis import compute_marginal_sgp, category_gap_analysis
compute_marginal_sgp(player_totals, config)           # config now required (no module _LC)
category_gap_analysis(user_totals, all_team_totals, user_team_name, weeks_remaining, config)

# Engine game theory (sgp_denominators required)
from src.engine.game_theory.opponent_valuation import estimate_opponent_valuations, player_market_value
estimate_opponent_valuations(rosters, sgp_denominators=cfg.sgp_denominators, ...)

# Lineup optimizer
from src.optimizer.pipeline import LineupOptimizerPipeline
LineupOptimizerPipeline(roster, mode="standard", alpha=0.5, weeks_remaining=16, config=None)

# Bootstrap
from src.data_bootstrap import bootstrap_all_data
bootstrap_all_data(yahoo_client=None, on_progress=None, force=False)

# Refresh log snapshot (ops/debug visibility)
from src.database import get_refresh_log_snapshot
get_refresh_log_snapshot() -> list[dict]              # source, status, tier, message, last_refresh

# MatchupContextService
from src.matchup_context import get_matchup_context
get_matchup_context().get_category_weights(mode="matchup"|"standings"|"blended", ...)

# Auto-advance game date
from src.game_day import get_target_game_date
get_target_game_date()  # Today, or tomorrow if all today's games final

# Stat formatting
from src.ui_shared import format_stat
format_stat(0.275, "AVG")  # "0.275"
format_stat(3.85, "ERA")   # "3.85"
format_stat(2.34, "SGP")   # "+2.34"

# Optimizer validation tools
from src.optimizer.data_freshness import DataFreshnessTracker
from src.optimizer.constants_registry import CONSTANTS_REGISTRY
from src.optimizer.sensitivity_analysis import run_sensitivity_analysis
from src.optimizer.backtest_runner import run_backtest, format_report
from src.optimizer.sigmoid_calibrator import calibrate_sigmoid_k

# FA recommender (canonical engine for Free Agents page)
from src.optimizer.fa_recommender import recommend_fa_moves, recommend_streaming_moves
from src.optimizer.shared_data_layer import build_optimizer_context

# build_optimizer_context accepts level_filter (default "MLB only") and
# auto-filters multi-team rosters by user_team_name (PR #98 / FA P3.6)
ctx = build_optimizer_context(
    scope="rest_of_season", yds=yds, config=config,
    weeks_remaining=compute_weeks_remaining(),
    user_team_name=user_team_name,
    roster=user_roster,                     # pre-filtered single-team slice preferred
    level_filter="MLB only",                # filters minor leaguers from ctx.player_pool
)
recommend_fa_moves(ctx, max_moves=5)        # post-PR #110 engine

# Shared scarcity factor (used by both add-side and drop-side scoring)
from src.valuation import compute_positional_scarcity_factor
compute_positional_scarcity_factor(positions_str, replacement_levels_dict) -> float  # 1.0 - 1.25x

# Dynamic IL stash (replaces hardcoded {"Bieber", "Strider"})
from src.alerts import get_il_stash_names
get_il_stash_names() -> set[str]            # Fresh from league_rosters.status

# News-derived suspension end-dates (PR #109 / FA P5d)
from src.news_sentiment import parse_suspension_days
parse_suspension_days("Suspended through 5/26", reference_date=datetime.now(UTC)) -> int | None

# IL weighting with return-date awareness (PR #95 + PR #105)
from src.in_season import _il_weight_from_status, _return_date_weight
_il_weight_from_status(status: str, expected_return_days: float | None = None) -> float
# Curve: 0d→1.0, 1d→0.95, 7d→0.85, 14d→0.70 (= IL10), 21d→0.55, 45d→0.30, 60d→0.20
# Also parses day counts inline from status strings (PR #105): "IL10 - 3 days" → 3
```

## Gotchas

### Data & Schema
- **`LeagueConfig` is the single source of truth** — Never hardcode category lists. Structurally guarded by `tests/test_no_hardcoded_categories_in_src.py` + sibling per-file guards.
- **DB column `name` vs UI `player_name`** — Normalized via `rename(columns={"name": "player_name"})` in `_build_player_pool()`.
- **Sample data is `system='blended'` only** — Do NOT call `create_blended_projections()` when only blended data exists.
- **Python 3.14 + SQLite bytes** — Returns bytes for some integer columns. Fixed with `CAST` in SQL and `pd.to_numeric()`.
- **Players table has no UNIQUE constraint on name** — Uses SELECT-first pattern. Do NOT use `ON CONFLICT(name)`.
- **Park factors schema** — Columns: `team_code`, `factor_hitting`, `factor_pitching`. Tier 1 (pybaseball) honored when available; emergency dict when not (SF-22).
- **FanGraphs API SYSTEM_MAP** — FG API uses `"fangraphsdc"`, DB stores `"depthcharts"`. Always use `SYSTEM_MAP`.
- **SQLite WAL mode + busy_timeout** — `get_connection()` sets `PRAGMA journal_mode=WAL`, `busy_timeout=60000`, `synchronous=NORMAL`. Direct `sqlite3.connect()` is structurally guarded against (`tests/test_no_direct_sqlite_connect_in_scripts.py`). Bumped from 30s → 60s in PR #69 after parallel-write contention hit the lower ceiling.
- **FanGraphs 403 handling** — `leaders-legacy.aspx` returns 403 to non-browser scrapers. SF-6 wired browser-headers scraper as Tier 2 (probe still 403). FIP/xFIP proxy serves as K-boost fallback when Stuff+ unavailable.
- **`statcast_archive.sprint_speed` column** — Required for SF-19 player pool SELECT. `init_db` creates it; legacy DBs get it via `_safe_add_column` migration.
- **Player pool ytd_* columns are complete** — Includes all hitting + pitching counting + rate components needed for weighted aggregation. Consumers no longer need separate `load_season_stats(year)` calls.

### Yahoo API
- **OAuth uses oob** — `redirect_uri=oob` required. No localhost redirects.
- **Auto-reconnect** — `_try_reconnect_yahoo()` loads `data/yahoo_token.json`. Token refresh automatic.
- **yfpy Roster iteration** — `Roster.__iter__()` yields attribute NAMES (strings), NOT players. Use `getattr(roster, "players", None) or []`.
- **Yahoo game_key** — MLB 2026 = 469. Must set BOTH `self._query.game_id` AND `self._query.league_key`.
- **429 backoff** — `_request_with_backoff()` retries 3× with exponential delay.
- **yfpy eligible_positions bug** — yfpy's `Player.eligible_positions` drops multi-position data. Workaround: read from `player._extracted_data["eligible_positions"]` which preserves the full list. Separator is `,` (not `/`).

### Streamlit UI
- **No emoji** — All icons are inline SVGs from `PAGE_ICONS`. Injury badges use CSS dots.
- **CSS `!important` required** — Streamlit's inline CSS has high specificity.
- **HTML sanitization** — `st.markdown('<div>')` auto-closes tags. Use self-contained HTML blocks.
- **`T["ink"]` for text-on-accent** — Not `T["bg"]` (invisible on amber). `T["tx"]` not `T["tx1"]`.
- **Connection leak** — Always wrap `get_connection()` in `try/finally` with `conn.close()`.
- **Splash-screen load timer** — HH:MM:SS live counter. Start in `st.session_state["bootstrap_start_time"]`. Final in `bootstrap_elapsed_secs` / `bootstrap_elapsed_hms`.
- **Format precision** — Use `format_stat(value, stat_type)` for all rate stats. Inline `f"{x:.3f}"` for ERA/WHIP is structurally guarded against (`tests/test_pages_format_compliance.py`).

### Algorithms & Math
- **Rate-stat aggregation** — AVG=sum(h)/sum(ab), OBP=sum(h+bb+hbp)/sum(ab+bb+hbp+sf), ERA=sum(er)*9/sum(ip), WHIP=sum(bb+h)/sum(ip). Weighted, NEVER simple averages.
- **LP inverse stat weighting** — ERA/WHIP weighted by IP. Without it, 1-IP relievers dominate starters.
- **L (Losses) is an inverse counting stat** — Sign-flip in SGP, no IP weighting in LP. Inverse-stat sets must include L (`{"L", "ERA", "WHIP"}`); guards prevent accidental `{"ERA", "WHIP"}` regressions.
- **Punt detection** — Requires BOTH: `gainable_positions == 0 AND rank >= 10`.
- **LP-constrained lineup totals** — Only 18 starters' stats feed SGP. Bench excluded.
- **Paired MC seed discipline** — Same seed for before/after rosters. NEVER different seeds.
- **True antithetic variates** — `trade_simulator.py` negates uniform quantiles (`u → 1-u`, `z → -z`) for paired arms; ~25% variance reduction.
- **SGP through `SGPCalculator.totals_sgp` only** — All 7 prior local reinventions migrated. Structurally guarded.
- **Sigmoid urgency reads from registry at runtime** — `category_urgency.py` reads `sigmoid_k_counting` / `sigmoid_k_rate` from `CONSTANTS_REGISTRY` per call, so `calibrate_sigmoid.py` updates take effect without restart.

### FA Recommender Engine (post-2026-05-21 overhaul)
The Free Agents page (`pages/14_Free_Agents.py`) is the sole consumer of `recommend_fa_moves` in production. The engine's correctness depends on these invariants — guard tests prevent regression:

- **`ctx.user_roster_ids` is the USER's team, not the whole league** — Pages pass a single-team roster slice to `build_optimizer_context`. The builder auto-filters multi-team rosters when `user_team_name` is set (PR #98 / FA P3.6). Without this filter, drop candidates would be drawn from all 12 teams.
- **`ctx.player_pool` respects `level_filter`** — Default `"MLB only"` filters minor leaguers (PR #98). Pages can opt into `"All"` for prospect lookups (Player Databank, Draft Simulator).
- **`_compute_base_value` delegates to canonical `SGPCalculator.marginal_sgp`** (PR #99 / FA P3.7) — Handles inverse-stat signs (ERA, WHIP, L are NEGATIVE contributions) and rate-stat volume weighting. The earlier inline formula `value += (stat/denom) * weight` had the wrong sign for inverse stats — unknown pitchers with default `era=9.0/whip=2.0` scored base=+38.8 (should be -38.8).
- **Position cap boundary semantics** — Block when post-swap count `> cap` (PR #100). Having exactly `cap` players at a position is healthy (`starter + 1 backup`), not a violation. Cap = Yahoo starting slots + 1. Off-by-one (`>=`) was blocking legitimate same-position upgrade swaps.
- **Roster-construction floors** — Never drop below 10 active hitters or 8 active pitchers (FourzynBurn starting lineup count). IL players count toward position caps but NOT toward active-roster floors (PR #97 / FA P3.5).
- **Symmetric positional scarcity** — `_compute_base_value` (adds) AND `compute_drop_cost` (drops) both multiply by `compute_positional_scarcity_factor` (PR #96 / FA P3.5). Without symmetry, backup catchers got over-boosted on the add side.
- **Playing-time gate** — FA composite score multiplied by [0.30, 1.0] based on `ytd_gp / expected_gp` ratio for hitters (or `ytd_ip / expected_ip` for pitchers). Grace period: first 30 days of season inactive. Curve: 0 GP → 0.30x, <30% → 0.60x, 30-59% → 0.85x, ≥60% → 1.0x (PR #101 / FA P3.9).
- **ROS × playing-time scaling** — Before blending, ROS counting stats are scaled by `max(0.20, ytd_gp / expected_gp)` for low-volume players (PR #110 / FA P5c). Stops IL stash phantoms (e.g. Westburg, 0 YTD GP) from riding inflated preseason ROS projections.
- **Canonical blend weights** — 0.70 ROS + 0.20 YTD + 0.10 L14 in `_blend_fa_row` (PRs #92 + #110/P5b). L14 wired from `l14_*` columns when `l14_pa >= 20` (hitters) / `l14_ip >= 5` (pitchers); below the volume gate, ROS+YTD renormalize.
- **Regression flag nudge** — `_score_fa_candidates` multiplies composite by 1.05× for `BUY_LOW`, 0.95× for `SELL_HIGH`, 1.0× for empty/None (PR #106 / FA P10).
- **Punt-category awareness** — When `ctx.h2h_strategy.get("punt", [])` lists a category OR `ctx.urgency_weights["per_cat"][cat]["win_prob"] < 0.10`, its weight is overridden to 0.05× for the duration of FA scoring (PR #110 / FA P5f). Prevents engine from rewarding contributions to conceded cats.
- **Sustainability: counting vs rate** — `compute_sustainability_score` in `waiver_wire.py` combines rate-stat signals (xwOBA-wOBA, SIERA-ERA) with counting-stat signals (HR/FB% for hitters, LOB% for pitchers). PR #108 / FA P5e. Backward compat: missing counting cols → rate-only behavior.
- **Drop candidate diversity** — `_deduplicate_and_limit` walks sorted swaps and surfaces distinct (add, drop) pairs. When all top swaps share a single drop, output collapses to 1; when alternative drops exist, output diversifies (PR #110 / FA P5a is a regression-guard test, no impl change).
- **Roster sync: mlb_id matching** — `src/live_stats.py::match_player_id` emits a WARNING when name-only match resolves to a different team than the caller's `team_abbr` (PR #104 / FA P4). Prevents Muncy-DNA-class bugs (one MLB roster row → two players with the same name).
- **News-derived suspension dates** — `parse_suspension_days()` in `news_sentiment.py` recognizes "Suspended through MM/DD" and "Suspended for N games" patterns (PR #109 / FA P5d). Wired into `expected_return_days` column on `ctx.roster` + `ctx.player_pool` so short suspensions (Valdez-class, ~1-7 days) get appropriate IL-weight curves, not the 0.0 indefinite default.
- **`_il_weight_from_status` parses day counts inline** — Status strings like `"IL10 - 3 days"` now extract the day count before falling through to the IL10/IL15/IL60 string defaults (PR #105 / FA PR8).
- **Legacy fallback is now visible** — When the new `recommend_fa_moves` engine throws, `pages/14_Free_Agents.py` falls back to the legacy `compute_add_drop_recommendations` AND shows a `st.warning(...)` banner with the exception type and message (PR #98). Prevents silent degradation masking new-engine bugs.

### Lineup Optimizer
- **DTD/IL exclusion** — `_il_statuses` includes `dtd` and `day-to-day`.
- **Recent form blend** — Dynamic weight: linear interpolation from 0.10 (7 games) to scope max (14+ games). Below 7 games = 0 weight.
- **Matchup data before state classification** — `yds.get_matchup()` must be called before `classify_matchup_state()`.
- **IP budget uses projected IP** — Not actual IP pitched.
- **Weekly scaling** — Counting stats divided by 26 weeks. Rate stats unchanged.
- **`sort_roster_for_display()` returns a copy** — Never modifies input DataFrame.
- **Two-start pitcher fatigue** — 0.93× rate stat quality on 2nd start (~7% ERA/WHIP decay).
- **Platoon splits** — LHB vs RHP: 7.5% wOBA advantage. RHB vs LHP: 5.8%.
- **Sigmoid urgency** — `COUNTING_STAT_K=2.0`, `RATE_STAT_K=3.0` (configurable via constants_registry).
- **DCV gate — pure SPs not in probables get volume=0.0** — SP/RP hybrids keep 0.9 (can still relieve). Pure-`P` positions (no SP/RP qualifier) ALSO gate via token-set membership; if positions=`{P}` and player isn't a probable, `volume=0`. Closes the SP-gate-was-dead-code bug.
- **Pitcher probable-starter name matching** — `_normalize_pitcher_name` (accents/suffixes/punctuation stripped).
- **Matchup multiplier resolves opponent + venue from schedule** — Walks `schedule_today`, determines home/away, passes the **home team code** as `opponent_team` to `park_factor_adjustment` (the convention used by that fn). Opposing pitcher throws + xFIP also resolved.
- **Pitcher leave-empty threshold** — `_PITCHER_EMPTY_THRESHOLD = _pitcher_median * 0.20` (was 0.05).
- **FA streaming drop-candidate is slot-aware** — Filters roster by position overlap with FA's positions.
- **Already-played games zero out DCV** — `locked_teams` set (status "in progress"/"final" or game_datetime ≤ now). `volume_factor=0.0`. UI renders `—` (not `0.00`) and a "Reason" column shows LOCKED/IL/OFF_DAY/NOT_PROBABLE.
- **Forced-start flag on lineup output** — `forced_start=True` when LP started a player but `matchup_mult < 0.70` OR `total_dcv < median × 0.5`. Renders as "START ⚠" with orange background.
- **Dynamic streaming SGP threshold for pitchers** — `_STREAM_NET_SGP_RELAXED=0.40` replaces `_STREAM_NET_SGP_MIN=0.70` for PITCHER streams when projected weekly IP < 75% of `_STREAM_IP_TARGET=54`.
- **Yahoo lineup mismatch banner** — Captures user's Yahoo `selected_position` into `yahoo_slot` BEFORE LP overwrites; orange banner if LP wants to start a BN player.
- **Pitcher matchup multiplier includes opposing offense wRC+** — League-avg 100 → 1.0, 120 → ~0.75× (capped 0.80), 80 → ~1.25× (capped 1.20). Sourced from `ctx.team_strength[opp_team]['wrc_plus']`.
- **Post-LP IP budget** — After LP assigns starters, recomputes IP from LP-selected pitchers only into `st.session_state["_post_lp_ip"]`. Header card shows "Post-LP Starters X / 54 IP".
- **SP slot reordering after LP** — Post-processes `_lp_slot_map` so highest-DCV SP-eligible pitchers land in SP slots.

### Dependencies
- **Pre-commit hook** — `scripts/pre-commit` runs ruff format + lint on staged files. Install with `python scripts/install-hooks.py`.
- **PyMC/PuLP/XGBoost optional** — `PYMC_AVAILABLE`/`PULP_AVAILABLE` (canonical in `src/lineup_optimizer.py`)/`XGBOOST_AVAILABLE` flags. CI skips with `@pytest.mark.skipif`.
- **Plotly 6.x** — Does NOT accept 8-digit hex (`#RRGGBBAA`). Use `rgba(r,g,b,a)`.
- **`ruff` on Windows** — Use `python -m ruff check .` instead of bare `ruff`.
- **`datetime.now(UTC)`** — Used everywhere; `datetime.utcnow()` deprecated.
- **Snake draft order** — `round % 2 == 0` forward, `round % 2 == 1` reverse.

## Data Sources

| Source | API | Staleness | Tier 2 fallback | Tier 3 fallback |
|--------|-----|-----------|-----------------|-----------------|
| Players (9K+) | MLB Stats API (40-man + spring training) | 7 days | — | — |
| Projections (7 systems) | FanGraphs JSON API + Marcel local | 24h | — | Marcel local |
| ROS Projections | FanGraphs (Steamer/ZiPS/DC ROS) | 4h | — | — |
| ADP | FanGraphs + FantasyPros ECR + NFBC | 24h | — | — |
| Current stats | MLB Stats API | 1h (15min during games) | — | — |
| Historical (3 years) | MLB Stats API | 30 days | — | — |
| Park factors | pybaseball | 30 days | — | Emergency dict (FG 5yr) |
| League data | Yahoo Fantasy API (optional) | 30 minutes | — | SQLite |
| News | MLB Stats API + Yahoo | 1 hour | — | — |
| Prospects | FanGraphs Board API + MiLB stats | 7 days | — | — |
| ECR (6 sources) | Multi-platform aggregation (Trimmed Borda) | 24 hours | — | Cached |
| Weather | Open-Meteo (30 stadiums, dome detection) | 2 hours | — | — |
| Team strength | pybaseball wRC+/FIP | 24 hours | statsapi fallback | — |
| Opposing pitchers | MLB Stats API + platoon splits | 2 hours | — | — |
| Statcast | pybaseball (EV, barrel, xwOBA, Stuff+, sprint speed) | 24 hours | — | — |
| Stuff+ / Batting Stats | FanGraphs (currently 403) | 24 hours | Browser-headers (still 403) | Neutral defaults; FIP/xFIP K-boost proxy active |
| Catcher framing | pybaseball Savant | 7 days | Browser-headers Savant scrape (200 OK) | `data/seed/catcher_framing_2024.json` (32 catchers) |
| Umpire tendencies | statsapi boxscore HP umpire | 24 hours | Savant `/umpire` (404 — no public endpoint) | `data/seed/umpire_tendencies_2024.json` (34 umpires) |
| Depth charts | Roster Resource scrape (often empty) | 7 days | MLB Stats API team_roster + GS≥5/SV≥5 heuristic | — |
| Game logs | statsapi `person+hydrate(stats)` (per SF-1 fix) | 1 hour | — | — |

## Structural Invariants (machine-checked)

These tests guard against regression of the cleanup work. Adding new code that violates any of these will fail CI:

| Test | Guards |
|------|--------|
| `test_no_hardcoded_categories_in_src.py` | No literal category lists in 13 src/ files; must derive from `LeagueConfig` |
| `test_app_no_hardcoded_categories.py` | Same for `app.py` |
| `test_my_team_no_hardcoded_categories.py` | Same for `pages/1_My_Team.py` |
| `test_sensitivity_categories_canonical.py` | `engine/game_theory/sensitivity.py` `CATEGORIES` from LeagueConfig |
| `test_no_merge_conflict_markers.py` | No leftover `<<<<<<<`/`=======`/`>>>>>>>` in any source file |
| `test_pages_yahoo_compliance.py` | 7 named pages don't call `load_league_rosters/standings` directly (must use `yahoo_data_service`) |
| `test_pages_format_compliance.py` | No inline `f"{x:.3f}"` near ERA/WHIP context (must use `format_stat`) |
| `test_no_direct_sqlite_connect_in_scripts.py` | Scripts use `get_connection`, never raw `sqlite3.connect` |
| `test_refresh_log_status_validity.py` | All bootstrap-emitted statuses (`success`/`partial`/`cached`/`skipped`/`no_data`/`error`) round-trip without downgrade |
| `test_my_team_uses_pool.py`, `test_player_compare_uses_pool.py`, `test_trade_analyzer_uses_pool.py` | No raw `SELECT FROM season_stats/ecr_rankings/statcast_archive` in those pages |
| `test_pulp_availability_consolidation.py` | `PULP_AVAILABLE` defined ONCE (`src/lineup_optimizer.py`); other modules import |
| `test_engine_no_fallback_singletons.py` | No `_LC = ...` module-level singletons in 6 engine modules (SF-21 + BUG-010 Wave 6 expansion: 3 original + trade_evaluator + engine/portfolio/valuation + engine/game_theory/sensitivity) |
| `test_opponent_valuation_no_default_denoms.py` | No `DEFAULT_SGP_DENOMS` constant in opponent_valuation; functions require `sgp_denominators` |
| `test_pool_ytd_columns_complete.py` | Pool surfaces all `ytd_*` columns needed for weighted rate-stat aggregation |
| `test_standings_adapter_contract.py` | Standings page adapter shape preserved for `simulate_season_enhanced` + `compute_team_strength_profiles` |
| `test_no_shadow_player_rows.py` | No shadow rows in `players` (team='MLB' + fake mlb_id range); no rostered NULL mlb_ids |
| `test_no_lr_name_column_refs.py` | No `lr.name` joins; no `UPDATE league_rosters WHERE name = ?` patterns |
| `test_dynamic_park_factors_removed.py` | `_bootstrap_dynamic_park_factors` does not exist; orchestrator does not dispatch this source |
| `test_no_ip_decimal_parse.py` | No direct `float(...inningsPitched...)` parses in src/ or scripts/ — must use `_ip_outs_to_decimal()` |
| `test_no_ecr_rank_suffix_in_consensus_keys.py` | `refresh_ecr_consensus` strips `_rank` suffix from `sources` dict keys before calling `_compute_player_consensus` |
| `test_engine_signals_architectural_boundary.py` | Core trade-engine modules do NOT import from `src.engine.signals` (those are UI-helper-only) |
| `test_sigmoid_calibrator_patches_registry.py` | Sigmoid calibrator + sensitivity_analysis patch `CONSTANTS_REGISTRY` (the runtime read-path), not legacy module aliases |
| `test_mc_convergence_wired.py` | `_run_mc_overlay` calls `check_convergence` and returns `convergence_quality`/`convergence_ess`/`convergence_rhat` |
| `test_calibration_data_client_methods.py` | Every method called on `yahoo_client` from calibration code exists on `YahooFantasyClient`; calls pass `league_id` |
| `test_lineup_optimizer_26_weeks.py` | `pages/2_Line-up_Optimizer.py` uses `WEEKS_IN_SEASON = 26.0` (canonical), not 24 |
| `test_lineup_optimizer_team_map_consistent.py` | Both "Athletics" and "Oakland Athletics" map to "ATH" (canonical 2026 MLB Stats API code) in `pages/2_Line-up_Optimizer.py` |
| `test_pages_use_yds_for_rosters.py` | `pages/6_League_Standings.py` and `pages/16_Player_Compare.py` do not call `load_league_rosters` directly |
| `test_optimizer_pipeline_forwards_context.py` | `LineupOptimizerPipeline` forwards `confirmed_lineups`/`recent_form`/`team_strength` to `build_daily_dcv_table` (AST-based check) |
| `test_matchup_planner_demo_banner.py` | `pages/5_Matchup_Planner.py` `pool.head(...)` demo-roster fallback is immediately preceded by `st.warning`/`st.info`/`st.error` (within 3 lines) |
| `test_compute_fa_comparisons_negative_sgp.py` | `compute_fa_comparisons` seeds `best_fa_value` from first candidate (handles negative-SGP players) |
| `test_closer_monitor_name_normalization.py` | `closer_monitor.build_closer_grid` normalizes names (accents, suffixes, punctuation) before matching pool |
| `test_playoff_sim_uses_league_config.py` | `playoff_sim` uses `season_weeks=26` and `_PLAYOFF_SPOTS=4` (FourzynBurn canonical) |
| `test_yahoo_reconnect_in_yahoo_api.py` | `try_reconnect_yahoo` lives in `src/yahoo_api`; `app.py` keeps a thin wrapper; no top-level streamlit import in yahoo_api |
| `test_streamlit_security_settings.py` | `.streamlit/config.toml` has `enableXsrfProtection=true` (CORS default-true also acceptable) |
| `test_no_lc_singletons_in_optimizer.py` | No module-level `_LC = ...` in 8 optimizer modules (SF-21 + BUG-010 Wave 6 expansion) |
| `test_no_lc_singletons_in_strategy.py` | No module-level `_LC = ...` in 8 strategy/UI modules (SF-21 + BUG-010 Wave 6 expansion) |
| `test_survival_calibrator_skips_on_missing_adp.py` | `survival_calibrator._build_prediction_pairs` skips with WARNING when no ADP column (no `actual_pick` → ADP data leakage) |
| `test_no_unguarded_update_refresh_log.py` | Bootstrap phases use `update_refresh_log_auto` (with row-count) not plain `update_refresh_log(..., "success")` (SF-54 / INFRA-F6) |
| `test_umpire_phase_has_timeout.py` | `_bootstrap_umpire_tendencies` Tier 1 uses a wall-clock budget (`time.monotonic()` + elapsed/timeout sentinel) (SF-55 / INFRA-F3) |
| `test_data_freshness_reads_refresh_log.py` | `DataFreshnessTracker` hydrates `_sources` from `refresh_log` on `__init__`; non-success rows skipped (SF-56 / INFRA-F4) |
| `test_no_oak_in_source_modules.py` | `src/player_databank.py`, `src/data_2026.py`, `src/ecr.py`, `src/prospect_engine.py` emit "ATH" not "OAK" (SF-57 / D1A-008 follow-up) |
| `test_wave9_level_column_migration.py` | `players.level` column exists after `init_db()` (SF-78 / INFRA-F5) |
| `test_wave9_minor_league_fetch.py` | `fetch_minor_league_players` caps at `top_n_per_team`, sets `level`, handles failures (SF-79) |
| `test_wave9_bootstrap_phase.py` | `_bootstrap_minor_league_rosters` writes rows with `level` + updates `refresh_log` (SF-80) |
| `test_wave9_pool_includes_level.py` | `load_player_pool()` exposes `level` column from all 3 SELECT paths (SF-81) |
| `test_wave9_free_agents_level_filter.py` | `pages/14_Free_Agents.py` has level selectbox defaulting to "MLB only" (SF-82) |
| `test_wave9_player_compare_level_filter.py` | `pages/16_Player_Compare.py` has level selectbox (SF-83) |
| `test_bootstrap_log_isolation.py` | `src/data_bootstrap.py` skips the RotatingFileHandler when running under pytest — prevents test mocks (e.g. `RuntimeError("DB out")`) from polluting `data/logs/bootstrap.log` (SFH L) |
| `test_bootstrap_refresh_log_on_error.py` | Every outermost `except` in `_bootstrap_*` functions in `src/data_bootstrap.py` either calls `update_refresh_log[_auto]`, re-raises, or catches ImportError-only. AST-checked; ≥25-phase sentinel (SFH D companion to `test_no_unguarded_update_refresh_log.py`) |
| `test_sfh_m3_partial_cached_reads.py` | `check_staleness` force-refreshes only `error`/`unknown`; `partial`/`no_data`/`cached`/`skipped` honor TTL. `DataFreshnessTracker` hydrates from `partial`/`cached`/`skipped` rows (SFH M-3) |
| `test_players_name_lookup_case_insensitive.py` | Every `WHERE name = ?` query against the `players` table in `src/` uses `COLLATE NOCASE`. Prevents Muncy-DNA dedup misses from case-variant duplicates (SFH LOW-2) |
| `test_refresh_log_resilience.py` | `_try_write_refresh_log` retries 3× on `OperationalError("database is locked")`; `_run_with_timeout(source=)` writes "timeout" to refresh_log on budget exceeded; `_reconcile_results_to_refresh_log` overwrites stale-success rows for "Error:"/"Timeout" results (SFH H2 + H3) |
| `test_pvb_splits_commit_per_batter.py` | `_bootstrap_pvb_splits` calls `conn.commit()` INSIDE the outer batter loop, NOT outside — required to release the write lock between Statcast fetches so parallel writers don't starve (SFH M1, root-cause fix for #69's band-aid) |
| `test_pybaseball_batting_stats_no_pos_kwarg.py` | No `batting_stats(..., pos=...)` calls in `src/data_bootstrap.py` — pybaseball 2.2.7 rejects `pos=` with TypeError (SFH M2) |
| `test_yahoo_player_key_column.py` | `league_rosters` has `yahoo_player_key` column after `init_db()`; `upsert_league_roster_entry` writes it; TWP-on-different-teams persists both rows distinguished by yahoo_player_key; UNIQUE(team_name, player_id) constraint preserved (SFH M4) |
| `test_draft_pick_name_resolution.py` | `resolve_player_names_by_keys` batch-fetches Yahoo player names (25 per call); `get_draft_results` uses it for entries where yfpy didn't expand the player resource; still-unresolved keys fall back to legacy `"Player {key}"` placeholder (SFH L8) |

**FA Engine Overhaul guards (PRs #89-#110, 2026-05-20 → 2026-05-21):**

| Test | Guards |
|------|--------|
| `test_fa_il_stash_unified_protection.py` | FA page UI-layer IL filter applies to BOTH top "Recommended Adds/Drops" and bottom "Recommended Drops" tables. Imports + uses `get_il_stash_names()` not the static set (FA PR #89 + PR #102) |
| `test_roster_category_totals_il_weighted.py` | `_roster_category_totals` no longer zeroes IL players — they keep their projection × `_il_weight_from_status` weight (FA PR #90) |
| `test_fa_page_uses_recommend_fa_moves.py` | FA page imports + calls `recommend_fa_moves`, not legacy `compute_add_drop_recommendations` (FA PR #91) |
| `test_fa_blending_ros_ytd.py` | `_blend_fa_row` uses canonical 0.70/0.20 ROS/YTD blend with sample-size gating (FA PR #92) |
| `test_sustainability_calibrated.py` | `compute_sustainability_score` is a calibrated sigmoid, not a 5-bucket step function (FA PR #93) |
| `test_fa_composite_multiplicative.py` | Composite formula is purely multiplicative (urgency applied per-category inside `_compute_base_value`, not additively outside) (FA PR #94) |
| `test_il_weight_return_date_aware.py` + `test_roster_category_totals_return_date_aware.py` | `_il_weight_from_status(status, expected_return_days)` uses the piecewise-linear curve when ESPN/Yahoo provides return-date data (FA PR #95) |
| `test_drop_cost_positional_scarcity.py` | `compute_drop_cost(player_id, ..., replacement_levels=...)` multiplies base_cost by symmetric scarcity factor. `compute_positional_scarcity_factor` importable from `src.valuation` (FA PR #96) |
| `test_fa_recommender_positional_scarcity.py` | `_compute_base_value` multiplies by positional scarcity factor (top-2 catcher with raw SGP X outranks 25th OF with same raw SGP) (FA PR #91/PR3 + PR #96) |
| `test_fa_roster_construction_guard.py` + `test_fa_position_cap_at_boundary.py` | `_passes_roster_construction_guard` blocks: 3rd catcher (at-cap position), drops below 10 active hitters or 8 active pitchers (FourzynBurn starting-lineup minimums). Boundary semantics: block when count `> cap`, NOT `>= cap` (FA PR #97 + PR #100) |
| `test_build_optimizer_context_user_roster_filter.py` + `test_fa_page_passes_user_roster.py` | `build_optimizer_context` filters multi-team rosters by `user_team_name`; FA page passes `user_roster` not `rosters` (FA PR #98 / FA P3.6) |
| `test_build_optimizer_context_level_filter.py` | `level_filter` param defaults to `"MLB only"`; minor leaguers filtered from `ctx.player_pool` (FA PR #98) |
| `test_compute_base_value_inverse_stats.py` | `_compute_base_value` delegates to `SGPCalculator.marginal_sgp` so inverse stats (ERA/WHIP/L) get negative contribution. Unknown pitchers with default-bad stats score LOWER than real MLB hitters (FA PR #99) |
| `test_fa_playing_time_gate.py` | `_playing_time_multiplier` penalizes low-GP FAs after 30-day grace period; curve 0.30x/0.60x/0.85x/1.0x (FA PR #101) |
| `test_dynamic_il_stash_list.py` + `test_no_hardcoded_il_stash.py` | `IL_STASH_NAMES` derived dynamically from `league_rosters`; no hardcoded set literal allowed in `src/alerts.py` (FA PR #102) |
| `test_fa_page_ui_polish.py` | ECR formatted as integer; position string dedup; "Other categories" reconciliation line in why-expander; "Show all NNN" pagination toggle present (FA PR #103) |
| `test_roster_sync_mlb_id_matching.py` + `test_no_name_only_roster_matching.py` | `match_player_id` emits DNA-collision WARNING when name-only resolves to a different team than caller's `team_abbr`; no new name-only roster matching code in `src/live_stats.py` / `src/yahoo_data_service.py` / `src/league_manager.py` / `src/data_bootstrap.py` (FA PR #104) |
| `test_il_return_date_multi_source.py` | `_parse_return_days_from_text` extracts day count from Yahoo status strings like `"IL10 - 3 days"` (FA PR #105) |
| `test_regression_flag_wiring.py` + `test_rate_stat_volume_in_targeted_adds.py` | `_score_fa_candidates` multiplies composite by `regression_mult` (BUY_LOW=1.05, SELL_HIGH=0.95). `compute_matchup_targeted_adds` does NOT use naive `team_totals["AVG"] +=` rate-stat addition (FA PR #106) |
| `test_constants_registry_coverage.py` | 23 FA-engine constants registered in `CONSTANTS_REGISTRY` with `value`, `citation`, `lower_bound`, `upper_bound`, `sensitivity`. Includes streaming, playing-time, ownership, floor, position-cap thresholds (FA PR #107) |
| `test_sustainability_counting_vs_rate.py` | `compute_sustainability_score` incorporates HR/FB% z-score (hitters) and LOB% z-score (pitchers) alongside existing rate-stat signals. Backward compat: missing counting cols → rate-only (FA PR #108) |
| `test_news_suspension_parser.py` | `parse_suspension_days` recognizes "Suspended through MM/DD" and "Suspended for N games" patterns; returns None for non-suspension text; clamps past-dates to 0 (FA PR #109) |
| `test_l14_blend_wiring.py` | `_blend_fa_row` consumes `l14_*` columns when `l14_pa >= 20` (hitters) or `l14_ip >= 5` (pitchers); below gate, falls back to renormalized ROS+YTD (FA PR #110 / P5b) |
| `test_ros_projection_playing_time_scale.py` | `_scale_ros_by_playing_time` discounts ROS counting stats by `max(0.20, ytd_gp/expected_gp)` for low-volume FAs after 30-day grace period (FA PR #110 / P5c) |
| `test_drop_candidate_diversity.py` | `_deduplicate_and_limit` walks sorted swaps and produces distinct (add, drop) pairs when alternatives exist (FA PR #110 / P5a — regression guard) |
| `test_punt_category_awareness.py` | `_score_fa_candidates` overrides `category_weights` to 0.05× for punted categories (from `ctx.h2h_strategy.get("punt", [])` or per-cat `win_prob < 0.10`) (FA PR #110 / P5f) |
| `test_fa_nan_guard.py` | `_score_fa_candidates` must not raise `ValueError` when FA rows contain NaN `player_id` or NaN `is_hitter` (post-pool-merge left-join artefact). `_is_hitter_safe()` helper guards all 5 int-cast sites; NaN `player_id` rows silently skipped; valid FA rows still score correctly. `compute_sustainability_score` in `waiver_wire.py` also guarded. |

## Audit History

The data + analytics pipeline has been audited across 13 waves (April–May 2026) covering ~105 silent-failure / drift / dead-code / type-design / migration bugs. All HIGH-severity findings are resolved across PRs #7–#23 + the 2026-05-17 deep-audit cleanup (PRs #29–#46) + 2026-05-19 deep-audit completion + 2026-05-19 follow-up cleanups (PRs #47–#52) + 2026-05-20 silent-failure sweep (PRs #59–#72).

The cumulative structural-invariant guard set in `tests/test_no_*.py`, `test_pages_*.py`, `test_wave*.py`, and `test_sf*.py` covers ~75 patterns that were silent-failure-prone, duplication-prone, or schema-evolution-prone before audit. See `docs/archive/2026-05-17-deep-audit-punchlist.md` for the historical deep-audit punchlist (now fully shipped) and `docs/archive/specs/` for shipped design docs from earlier waves.

**2026-05-19 follow-up PRs (post deep-audit completion):**

| PR | Title | Highlights |
|----|-------|------------|
| #47 | Deep-audit punchlist completion + repo hygiene + preflight tooling | 17 commits squashed; Sections 2-8 shipped; 6 new structural-invariant guard tests; preflight env+access verification script |
| #48 | Research.md gap-status reconciliation against PRs #18-#47 | 20 numbered gaps reclassified: 13 shipped, 3 partial, 4 still gap |
| #49 | silent-failure-hunter follow-up: H1 + M1/M2/M3/L1 + H2 regression lock | Found 2 HIGH + 3 MED + 1 LOW post-merge; H1 case-insensitive SQL was a real bug; H2 was a false positive locked with regression test |
| #50 | News_fetcher O(N*M) → 3-tier lookup | Caused CI shard 1 hang in PR #47; now <50ms exact / <5s at-scale |
| #51 | Un-skip 3 bootstrap integration tests via auto-mock fixture | Replaces 8-of-30-phases mock list with introspection-based fixture |
| #52 | Test isolation fixes for 4 local-only failures | DataFreshnessTracker hydration + YahooDataService singleton mocked; tests pass locally and on CI |

Milestone tag `milestone/2026-05-19-deep-audit-complete` marks SHA after PR #49.

**2026-05-20 silent-failure sweep (PRs #59–#72):**

| PR | Title | Highlights |
|----|-------|------------|
| #59 | fix(bootstrap) add 120s timeout to Yahoo FA phase | Prevent indefinite hang on Yahoo rate-limit (480-FA pagination × backoff) |
| #60 | fix(app) session-start bootstrap uses staleness check not force=True | Stops every new tab from re-running the full 33-phase bootstrap |
| #61 | fix SFH HIGH findings (M-1 + H-1 + H-2) | M-1: news_fetcher canonical-name collisions (Muncy DNA — "Will Smith" + "Will Smith Jr." silently overwrote). H-1: PR #59 FA-timeout signal clobbered by empty-fa_df else branch. H-2: standings_engine.py:206 dead `float(hitters["pa"].sum())` |
| #62 | fix SFH MED/LOW polish | M-2 news_fetcher prefix fallback, M-4 lineup_optimizer SIM114 parens, L-1 dual_objective dead branch, L-2 live_stats COLLATE NOCASE on remaining 2 sites, K CLAUDE.md "7" → "6" |
| #63 | fix(logging) isolate bootstrap.log from pytest test runs | RotatingFileHandler gated on `pytest not in sys.modules`. Test mock noise (e.g. `RuntimeError("DB out")`) no longer leaks into production log file |
| #64 | fix(refresh_log) honor partial/cached/skipped statuses on read (M-3) | `check_staleness` force-refreshes only {error, unknown}; partial/no_data/cached/skipped honor TTL. `data_freshness` hydrates from {success, partial, cached, skipped} |
| #65 | fix(refresh_log) record error status on exception path (D) | 9 phases (ecr_consensus, adp_sources, contracts, news, depth_charts, prospects, news_intel, bat_speed, forty_man) silently swallowed exceptions. AST-checked guard prevents regression |
| #66 | fix(bootstrap) surface Yahoo-skipped state in injury_writeback (B) | `lr_count == 0` skip path now appends `[yahoo skipped: rosters empty]` to refresh_log message instead of looking identical to a full run |
| #67 | fix(refresh_log) treat 0 >= 0 as success (idempotent no-op phases) (F) | `update_refresh_log_auto` was labeling `_enrich_pitcher_positions` (writes 0 rows when already enriched) as "no_data" instead of "success" |
| #68 | fix(season_stats) lower expected_min threshold 70% → 25% (G) | Realistic match rate — MLB API returns stats for all players (incl. minor-league callups) but `players` table covers ~1100 |
| #69 | fix(db) bump SQLite busy_timeout 30s → 60s (A) | 7 "database is locked" errors on 2026-05-20 across parallel ThreadPoolExecutor group (umpire + catcher_framing + pvb_splits) |
| #70 | test(perf) relax test_prefix_pruning_speed thresholds (J) | CI "Coverage Floor" was actually a flaky timing test (1.2s > 1s on GitHub Actions); coverage was 70.57% all along |
| #71, #72 | Follow-ups | Updated `test_check_staleness_treats_non_success_as_stale` for M-3's new contract; ruff format on the renamed test |
| #74 | fix SFH MED-1 + LOW-1 + LOW-3 follow-ups | MED-1: `_enrich_pitcher_positions` error path silent-skip (same DNA as #65, missed by the `_bootstrap_*`-prefixed structural guard). LOW-1: CLAUDE.md busy_timeout doc drift after #69. LOW-3: split injured_count into fresh-ESPN vs stale-yahoo when skipped path taken |
| #75 | fix case-insensitive `WHERE name = ?` cleanup (LOW-2) | 12 case-sensitive `WHERE name = ?` queries against `players` across data_bootstrap, data_pipeline, database, league_manager, live_stats. Pre-existing Muncy-DNA gap. Added structural guard `test_players_name_lookup_case_insensitive.py` (no allowlist) |

Milestone tag `milestone/2026-05-20-silent-failure-sweep-complete` marks SHA after PR #75 (re-tagged 3x as the docs/follow-ups landed).

**2026-05-20 afternoon "fix everything" sweep (PRs #77–#86):**

Bootstrap diagnostic surfaced 12 distinct issues across HIGH/MEDIUM/LOW severity. The user directive was "everything must be fixed, no matter how small"; after triage 7 confirmed-bug PRs shipped, 3 investigation PRs shipped (each found a real bug under the LOW category), and 5 items were documented as design choices (see _Known Design Choices_ below) rather than re-flagged on future audits.

| PR | Title | Highlights |
|----|-------|------------|
| #77 | fix(live_stats) TWP routing in save_season_stats_to_db (H1) | Ohtani's 2026 pitching-only row was rejected by the Bellinger is_hitter type guard — fetched as is_hitter=False, stored as is_hitter=1 → type_mismatch → zero 2026 season_stats. Fix: route TWP players through stat-type-specific UPSERTs (hitting cols + games OR pitching cols + games), preserving the off-type half. Bellinger protection preserved verbatim for non-TWP |
| #78 | fix(bootstrap) refresh_log resilience for lock + timeout failures (H2 + H3) | Two silent-failure modes: per-phase except handlers attempted update_refresh_log during a parallel DB lock and the inner `except Exception: pass` swallowed it (H2); _run_with_timeout returned "Timeout after Ns" but never wrote refresh_log (H3). Fix: _try_write_refresh_log helper with 3-attempt 0.5s/1s backoff; _run_with_timeout accepts source= kwarg; end-of-bootstrap _reconcile_results_to_refresh_log walks results dict and overwrites stale-success rows. Added "timeout" to _VALID_REFRESH_STATUSES |
| #79 | fix(bootstrap) pvb_splits commit per-batter (M1) | Root cause of the parallel-write contention PR #69 band-aided. pvb_splits held a single write transaction across 50 batters × 1-3s Statcast fetches → blew through 60s busy_timeout, starving umpire_tendencies + catcher_framing in the same ThreadPoolExecutor group. Move conn.commit() inside the outer batter loop. AST guard ensures it stays inside |
| #80 | fix(bootstrap) drop pybaseball batting_stats pos= kwarg (M2) | pybaseball 2.2.7 rejects pos=. Drop the kwarg, filter by Pos column post-hoc. AST guard prevents regression |
| #81 | fix(db) yahoo_player_key column on league_rosters (M4) | Ohtani-Pitcher and Ohtani-Batter are TWO Yahoo entities (different player_keys, both → HEATER player_id=2). Without yahoo_player_key, queries can't distinguish which Yahoo entity each team rosters. Foundation step: add the nullable column, populate it in Yahoo sync, preserve UNIQUE(team_name, player_id). Downstream stats routing left for a follow-up once consumers need it |
| #82 | test(isolation) fix 4 test failures from production-DB pollution (G) | TestRollingStats::test_compute_total + 3 TestComputeHotColdReport pitcher tests were reading production game_logs (8347 rows mid-season) because setup_method used INSERT OR IGNORE and assumed empty tables. Fix: DELETE-before-INSERT + INSERT OR REPLACE for rolling stats; patch src.player_databank.compute_rolling_stats to empty DataFrame for war_room tests so the mocked get_player_recent_form_cached actually runs |
| #83 | perf(ecr) parallelize 6 source fetches via ThreadPoolExecutor (L1) | ECR consensus routinely hit its 240s budget — 6 sequential HTTP scrapes summed to ~240s. Parallel with max_workers=6 drops to max(individual), roughly 4× speedup. Per-source try/except preserved verbatim |
| #84 | fix(bootstrap) use paginated get_all_free_agents (L6) | Bootstrap called get_free_agents(count=200) but Yahoo's API caps single calls at 25 results per page regardless of `count`. get_all_free_agents already existed and paginates start=0,25,50,... Switched the call. Stored FAs jumps from 25 → ~500 |
| #85 | fix(yahoo) batch-resolve draft pick names via Yahoo Players API (L8) | ~25 rounds 1-3 picks (70%) came back as "Player 469.p.XXXX" placeholders because yfpy's get_league_draft_results doesn't expand the player resource. New resolve_player_names_by_keys helper batch-fetches names (25 per call, Yahoo's cap). Still-unresolved keys keep the legacy placeholder so the NOT NULL constraint doesn't break |
| #86 | docs(claude.md) document not-bugs + audit history append | This PR. Adds the _Known Design Choices_ section so future audits don't re-flag L2/L3/L4/L5/L7/M3 |

**2026-05-20 → 2026-05-21 FA Engine Overhaul (PRs #89-#110)** — 22 PRs landing the most invasive engine rewrite in the codebase's history. Triggered by the Crochet/Kirk live-validation report ("Drop top-30 SP on IL15 for a backup catcher"). The plan in `docs/2026-05-20-fa-engine-overhaul-plan.md` + `docs/2026-05-20-fa-engine-p3.5-plan.md` covers the architectural decisions.

**Phase 1 — Critical defense-in-depth (PRs #89, #90, #91):**

| PR | Title | Highlights |
|----|-------|------------|
| #89 | fix(fa_page) unified IL-stash protection across top + bottom drop tables | UI-layer filter expands IL_STASH_NAMES with news-keyword matches + league_rosters.status. Both the headline "Top pickup" callout and the bottom "Recommended Drops" table now apply the same filter |
| #90 | fix(in_season) IL players keep weighted projection instead of being zeroed | Root cause of Crochet/Kirk class — `_roster_category_totals` was zeroing IL contributions, making drops of IL aces look "free" in the SGP math. Replaced with `_il_weight_from_status(status)` returning a [0, 1] weight scaled by expected return window |
| #91 | fix(fa_page) wire to recommend_fa_moves + positional scarcity | Free Agents page switched from legacy `compute_add_drop_recommendations` to `recommend_fa_moves`. PR1 + PR3 combined |

**Phase 2 — Engine quality (PRs #92, #93, #94):**

| PR | Title | Highlights |
|----|-------|------------|
| #92 | fix(fa_recommender) blend ROS + YTD when scoring FAs | `_blend_fa_row` implements canonical 0.70 ROS + 0.20 YTD blend with sample-size gating (skips when ytd_gp < 30). L14 placeholder noted as future work |
| #93 | fix(waiver_wire) rewrite compute_sustainability_score as calibrated sigmoid | Replaces 5-bucket step function with logistic on xwOBA-wOBA gap (hitters) + SIERA-ERA gap (pitchers). Pitcher inversion bug fixed |
| #94 | fix(fa_recommender) composite formula no longer double-counts urgency | Urgency now applied multiplicatively per-category inside `_compute_base_value` (via `ctx.category_weights`), not additively in the composite line. The additive boost summed weights across 4-6 cats which dwarfed base_value for marginal FAs |

**Phase 3.5 — Live-validation surfaced bugs (PRs #95, #96, #97):**

| PR | Title | Highlights |
|----|-------|------------|
| #95 | fix(in_season) `_il_weight_from_status` uses return-date curve for short suspensions | Framber Valdez (6-day suspension ending today) was getting weight 0.0. New piecewise-linear curve consumes `expected_return_days` from ESPN + Yahoo news |
| #96 | fix(waiver_wire) symmetric positional scarcity on `compute_drop_cost` | PR3 added scarcity to add side; this PR matches on drop side. Hoisted `compute_positional_scarcity_factor` to `src/valuation.py` to break import cycle |
| #97 | fix(fa_recommender) add roster-construction guard | New `_passes_roster_construction_guard` blocks 3rd catcher adds + drops below starting-lineup minimums |

**Phase 3.6/3.7/3.8/3.9 — Root-cause cascade discovered via diagnostic (PRs #98, #99, #100, #101):**

| PR | Title | Highlights |
|----|-------|------------|
| #98 | fix(fa_recommender) pass user roster + filter MLB-only pool | THREE compounding bugs: (a) FA page passed full 12-team rosters to `build_optimizer_context`, making `ctx.user_roster_ids` contain all 317 league players; (b) `load_player_pool()` loaded minor leaguers, polluting top FA candidates; (c) silent legacy fallback masked engine errors. Fix: defensive multi-team filter in builder + `level_filter` param + visible `st.warning` on fallback |
| #99 | fix(fa_recommender) delegate `_compute_base_value` to canonical `marginal_sgp` | The earlier inline formula `value += (stat/denom) * weight` had WRONG SIGN for inverse stats (ERA, WHIP, L). Unknown pitchers with default `era=9.0/whip=2.0` scored base=+38.8 (should be -38.8). Brandon Marsh (real hitter, .325 AVG) scored 18.3 — half the unknown phantom. Delegated to `SGPCalculator.marginal_sgp` which handles signs and rate-stat volume correctly |
| #100 | fix(fa_recommender) position-cap boundary off-by-one | PR #97's check used `cnt >= cap` (block at or above). Cap = max allowed depth; having exactly `cap` is healthy. Changed to `cnt > cap` (block only when EXCEEDING) — was blocking legitimate same-position upgrade swaps |
| #101 | fix(fa_recommender) playing-time gate de-weights IL-stash phantoms | Jordan Westburg (0 YTD GP, post-wrist-surgery IL stash) was top FA pickup because his ROS projection assumed regular playing time. New `_playing_time_multiplier` curve (0 GP → 0.30x) penalizes low-volume players with 30-day grace period |

**Phase 4 — DNA collision fixes (PR #104):**

| PR | Title | Highlights |
|----|-------|------------|
| #104 | fix(roster_sync) DNA collision warning + Muncy migration | Two Max Muncys in MLB (LAD veteran mlb_id=571970 + ATH rookie mlb_id=691777). Yahoo correctly identified user's LAD Muncy via `editorial_team_abbr`, but `match_player_id` did name-only lookup → mapped to ATH Muncy's `player_id=71`. Engine read ATH Muncy's stats (.239/2HR/26GP) for user's LAD Muncy (.263/12HR/47GP). Fix: WARNING when name-only resolves to a different team than caller's `team_abbr`. One-time `migrate_muncy_dna_2026_05_21.py` inserted LAD Muncy as new row + repointed user's league_rosters |

**Phase P3+P4 cleanup (PRs #102, #103, #105, #106, #107):**

| PR | Title | Highlights |
|----|-------|------------|
| #102 | fix(alerts) derive IL_STASH_NAMES dynamically from league_rosters (FA PR7) | Hardcoded {Bieber, Strider} set went stale weekly. New `get_il_stash_names()` queries `league_rosters.status IN (IL10, IL15, IL60, IL, DTD)`. Surfaces ~52 IL players vs 2 hardcoded |
| #103 | FA Page UI Polish (PR11+12+13+14) | ECR shown as integer; position string dedup (`"2B/3B,3B"` → `"2B,3B"`); "Other categories" reconciliation line; "Show all NNN" pagination toggle |
| #105 | fix(in_season) `_il_weight_from_status` parses day-count from status string (FA PR8) | Yahoo status strings like `"IL10 - 3 days"` now extract the day count via 4 regex patterns; falls through to existing string lookup when no pattern matches |
| #106 | fix(fa_recommender, waiver_wire) wire regression_flag + fix rate-stat volume (FA PR10) | `regression_flag` column (BUY_LOW / SELL_HIGH) was loaded but never consumed. Wired as 5% nudge in `_score_fa_candidates`. Rate-stat volume guard test added (no bug found, but locked against future regression) |
| #107 | fix(constants_registry) consolidate FA-engine magic constants (FA PR9) | 23 magic constants migrated from `fa_recommender.py` + `waiver_wire.py` into `CONSTANTS_REGISTRY` with citations, bounds, sensitivity tags. Includes streaming thresholds (Razzball / Pitcher List), playing-time gate (PR21 calibration), ownership boost, floor multipliers |

**Phase P5 — Additional engine quality (PRs #108, #109, #110):**

| PR | Title | Highlights |
|----|-------|------------|
| #108 | feat(waiver_wire) sustainability differentiates counting vs rate regression (FA P5e) | `compute_sustainability_score` adds HR/FB% z-score (hitters) and LOB% strand-rate z-score (pitchers). A 30% HR/FB hitter now gets lower sustainability than 13% peer; high-LOB pitcher rides regression risk |
| #109 | feat(news_sentiment) parse_suspension_days extracts return date from news (FA P5d) | Recognizes "Suspended through MM/DD" (date parse) and "Suspended for N games" (game-count → day estimate). Wired into `expected_return_days` after ESPN merge (ESPN priority) |
| #110 | FA Engine P5 Bundle: L14 + ROS-scale + diversity + punt (P5b+c+a+f) | Four sequential commits in one PR. L14 wired into `_blend_fa_row` (canonical 0.70/0.20/0.10); ROS scaled by playing-time ratio before blending; drop-candidate diversity regression-guard test; punt-category weight override (`0.05x` for categories with win_prob<10% or explicit punt tag) |

Milestone tag `milestone/2026-05-21-fa-engine-overhaul-complete` marks SHA after PR #110.

## Known Design Choices & External Limitations

These items have surfaced in audits as "broken" or "flagged" but are NOT bugs. They are either external limitations we can't fix from our side, deliberate design choices, or correctly-handled edge cases. **Do not flag any of these as bugs in future audits without re-reading this section first.**

| Item | Why it's not a bug |
|------|---------------------|
| **FanGraphs `leaders-legacy.aspx` 403** | External anti-scrape policy. We have Tier 2 (browser-headers — also 403) and Tier 3 (neutral defaults + FIP/xFIP K-boost proxy). No code on our side can change FanGraphs' policy. Documented in Data Sources table |
| **`bat_speed` phase skipped** | pybaseball doesn't expose the Statcast bat-tracking endpoint (no `statcast_batter_bat_tracking` in 2.2.7). The phase correctly returns "Skipped: pybaseball bat tracking not available". Cannot fix without upstream library support |
| **`pitcher_positions: 0 rows enriched`** | Idempotent no-op when all pitchers already have SP/RP enrichment from a prior run. PR #67 made this correctly report as "success" (not "no_data"). Adding pitchers to enrich next time would change the count — the 0 is the **correct** result for the cached state |
| **News fetcher canonical-name collisions** | There are genuinely 3 different active MLB players named "Luis Garcia", 2 named "Will Smith", etc. PR #61 made the warning explicit (`canonical-name collision 'luis garcia' maps to player_ids [...] - keeping first`). The fix path is team-based disambiguation at every news caller — a multi-module refactor for a known, accepted limitation. The current keep-first behavior is documented as deliberate |
| **`season_stats: partial 2497/7498`** | MLB Stats API returns ~7500 player-seasons including all 40-man + spring training + minor-league callups. Our `players` table is curated to ~1100 active MLB rosters. The ~5000 unmatched rows are players we don't track by design (not data loss). PR #68 lowered expected_min threshold 70%→25% to reflect this reality. The "partial" status reflects intentional scope, not pipeline failure |
| **pybaseball `statcast_catcher_framing` CSV tokenization** | Upstream pybaseball/Savant integration issue — Savant's response can't be parsed by pybaseball 2.2.7. Our 3-tier waterfall already catches this gracefully: Tier 2 (browser-headers Savant scrape) covers, then Tier 3 seed file. The pybaseball failure is logged as WARNING, not ERROR. No code on our side fixes the upstream parser; removing the call would lose resilience when pybaseball updates |
| **Playing-time gate 30-day grace period** | First 30 days of season, `_playing_time_multiplier` returns 1.0 regardless of GP/IP. Avoids penalizing late-March / early-April call-ups whose YTD samples are too small to interpret. After Day 30, the curve kicks in (PR #101) |
| **ROS playing-time scale floor (0.20x)** | `_scale_ros_by_playing_time` floors at 0.20× even for 0-GP players. Hot rookies just called up shouldn't lose ALL their ROS projection — 0.20 preserves a meaningful baseline while still discounting IL phantoms (PR #110 / P5c) |
| **Punt-category weight = 0.05x (not 0)** | When a category is detected as punted (h2h_strategy or win_prob < 10%), weight overridden to 0.05 not exactly 0. Avoids divide-by-zero edge cases in downstream rate-stat math while effectively zeroing the contribution (PR #110 / P5f) |
| **L14 volume gate (20 PA hitters, 5 IP pitchers)** | Below this, `_blend_fa_row` skips L14 and renormalizes to ROS+YTD. 20 PA / 5 IP is the empirical "signal noise floor" — below this, single hot/cold games dominate (PR #110 / P5b) |
| **Drop candidate dedup collapses to 1 when single drop dominates** | When all top FAs share a single best drop, `_deduplicate_and_limit` correctly surfaces only 1 rec (each drop used at most once). Adding alternative drops requires the engine to see additional viable swap pairs in `_evaluate_swaps`, not a dedup-logic change (PR #110 / P5a) |
| **ATH Max Muncy (player_id=71) preserved post-migration** | The Muncy DNA migration (PR #104 + `scripts/migrate_muncy_dna_2026_05_21.py`) inserted LAD Muncy as player_id=9864 (mlb_id=571970) and repointed Team Hickey's league_rosters to player_id=9864 — it did NOT delete or update the ATH Muncy row (player_id=71, mlb_id=691777). The ATH row remains for any other team in the league that might roster him |

When a future audit flags one of these, the correct response is: confirm it matches the entry above, then move on.

## GitHub

- **Repo:** https://github.com/hicklax13/HEATER_v1.0.0
- **CI:** GitHub Actions — ruff lint/format + sharded pytest (4 shards × 2 xdist workers, Python 3.12, ~5–7 min wall time) + 60% coverage floor + structural-invariant guards

## Testing

- **~4200 passing tests** across 195+ test files, 15 skipped (PyMC/XGBoost/WeasyPrint optional)
- **CI:** GitHub Actions (Python 3.12, sharded 4 ways)
- **Coverage:** ~71% (60% CI floor — the 5-week "Coverage Floor red" was actually a flaky timing test, fixed 2026-05-20 by PR #70)
- **Pre-commit hook:** Enforces `ruff format` + `ruff check` on every commit
- **Pre-push hook:** Runs structural-invariant suite (~210 tests) — catches regressions on the locked design choices before they reach CI
- **Backtesting framework:** Historical replay validates engine recommendations vs actual outcomes
- **FA engine diagnostic toolkit:** 9 `scripts/diag_fa_*.py` scripts preserved as evidence + future debug tools (cross-type trace, score plateau inspection, roster data audit, etc.)

## Resume Checklist (New Session)

1. Confirm shell is in `C:\Users\conno\Code\HEATER_v1.0.1` (NOT the deprecated `OneDrive\Desktop` path; note the local folder is `v1.0.1` while GitHub repo NAME remains `HEATER_v1.0.0`).
2. Read `CLAUDE.md` (this file)
3. Check git status: `git status`, `git log --oneline -10`. Master should be at or beyond SHA marked by tag `milestone/2026-05-21-fa-engine-overhaul-complete` (PR #110)
4. Run `python -m pytest --ignore=tests/test_cheat_sheet.py -x -q` to verify tests pass (~4200 tests, ~3-5 min)
5. Run `streamlit run app.py` and verify Yahoo auto-reconnect
6. Inspect refresh_log: `python -c "from src.database import get_refresh_log_snapshot; import json; print(json.dumps(get_refresh_log_snapshot(), indent=2))"`
7. (If continuing FA engine work) Read `docs/2026-05-20-fa-engine-overhaul-plan.md` + `docs/2026-05-20-fa-engine-p3.5-plan.md` for full design rationale of PRs #89-#110
8. (If debugging FA recommendations) Use `scripts/diag_fa_stage_by_stage.py` to see what the engine sees stage-by-stage. `scripts/diag_roster_data_audit.py` flags DNA collisions across the roster

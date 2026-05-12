# HEATER — Fantasy Baseball In-Season Manager

## Overview

A fantasy-baseball draft assistant + in-season manager for a 12-team Yahoo Sports H2H Categories snake-draft league. Built with Streamlit, powered by Monte Carlo simulation, Bayesian projection updating, and LP-constrained lineup optimization. **2026 MLB season is active; the app is in in-season mode** — draft completed, all components running against live league + Yahoo data.

The codebase is organized around 7 feature surfaces:

1. **Draft Tool** (`app.py`) — Heater-themed splash + bootstrap + setup wizard + 3-column draft page with Monte Carlo recommendations.
2. **In-Season Pages** (`pages/`) — 11 pages: My Team / War Room, Draft Simulator, Trade Analyzer, Free Agents, Lineup Optimizer, Player Compare, Closer Monitor, League Standings, Leaders/Prospects, Trade Finder, Matchup Planner.
3. **Trade Analyzer Engine** (`src/engine/`) — 6-phase pipeline: deterministic SGP → stochastic MC (paired, true antithetic) → signal intelligence → contextual adjustments → game theory → production convergence/caching.
4. **Lineup Optimizer** (`src/optimizer/`) — 21-module pipeline with PuLP LP, daily category value (DCV) scoring, sigmoid urgency, FA recommender, sensitivity analysis, backtest framework.
5. **Draft Recommendation Engine** (`src/draft_engine.py`) — 8-stage enhancement chain with 3 execution modes (Quick/Standard/Full).
6. **War Room** (`src/war_room*.py`) — Mid-week pivot analysis, category flip probability, hot/cold detection.
7. **Backtesting** (`src/backtesting*.py` + `src/optimizer/backtest_runner.py`) — Historical replay framework validating recommendations against actual outcomes.

## Commands

```bash
pip install -r requirements.txt        # Install deps
python scripts/install-hooks.py        # Install pre-commit hook (first time)
python load_sample_data.py             # Load sample data (first time/testing)
streamlit run app.py                   # Run the app
python -m ruff check .                 # Lint
python -m ruff format .                # Format
python -m pytest --ignore=tests/test_cheat_sheet.py  # Full suite (~3700 pass; cheat_sheet skipped on Windows)
python -m pytest tests/test_foo.py -v  # Single test file

# Optimizer validation tools
python scripts/run_backtest.py --quick          # Replay historical MLB weeks, score accuracy
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
- **Roster:** C/1B/2B/3B/SS/3OF/2Util/2SP/2RP/4P/6BN/4IL = 28 slots
- **Yahoo game_key:** 469
- **Transactions:** 10 adds+trades combined per matchup week (FCFS waivers)
- **Can't-drop:** Players drafted in rounds 1-3 per team (36 league-wide)

## Tech Stack

- **Framework:** Streamlit (multi-page)
- **Database:** SQLite (`data/draft_tool.db`) with WAL + 30s busy_timeout via `get_connection()`
- **Core libs:** pandas, NumPy, SciPy, Plotly
- **Analytics:** PyMC 5 (Bayesian, optional), PuLP (LP optimizer), arviz
- **Live data:** MLB-StatsAPI, pybaseball (FanGraphs/Statcast), Open-Meteo (weather), Baseball Savant (catcher framing)
- **Yahoo API:** yfpy + streamlit-oauth (optional OAuth)
- **Linter:** ruff (lint + format), pre-commit hook enforced
- **CI:** GitHub Actions — ruff lint/format + pytest (Python 3.11/3.12/3.13) + 60% coverage floor
- **Python:** Local dev uses 3.14; CI tests 3.11-3.13

## File Structure

```
app.py                      — Splash, bootstrap, setup wizard, draft page
requirements.txt            — pip dependencies
load_sample_data.py         — Sample data generator for testing
.streamlit/config.toml      — Light theme (Heater palette)

pages/
  1_My_Team.py              — War Room, roster, category standings, alerts, Yahoo sync
  2_Draft_Simulator.py      — Draft simulator with AI opponents, MC recommendations
  3_Trade_Analyzer.py       — Trade proposal builder + 6-phase engine
  4_Free_Agents.py          — FA rankings by marginal value + ownership heat
  6_Line-up_Optimizer.py    — 6-tab optimizer: Start/Sit, Optimize, Manual, Streaming, Daily, Roster
  6_Player_Compare.py       — Head-to-head comparison with category fit
  7_Closer_Monitor.py       — 30-team closer depth chart grid
  8_League_Standings.py     — Current standings + MC season projections + power rankings
  9_Leaders.py              — Category leaders, breakout detection, prospects
  10_Trade_Finder.py        — Smart Recs / Target Player / Browse Partners / Readiness
  11_Matchup_Planner.py     — Category probabilities, player matchups, per-game detail

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
tests/                      — 165+ test files, ~3700 passing tests
  conftest.py               — Session-scoped league_standings fixture (autouse)
  test_no_*.py              — Structural-invariant guards (see Structural Invariants section)
  test_sf*.py               — TDD tests for SF-1..SF-28 fixes
data/
  draft_tool.db             — SQLite database (created at runtime)
  seed/                     — 2024 seed JSONs (catcher_framing, umpire_tendencies) for SF-7 Tier 3
  backups/                  — Draft state JSON backups
  logs/bootstrap.log        — Persistent bootstrap log (SF-14)
docs/
  ROADMAP.md                — Improvement history
  superpowers/plans/        — Implementation plans (e.g. 2026-05-10-finish-data-audit-cleanup.md)
.github/
  workflows/ci.yml          — CI pipeline (lint + 3.11/3.12/3.13 test + coverage floor)
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
- **`SGPCalculator`** (`src/valuation.py`) — Sole path for SGP computation. The `totals_sgp(totals, weights=None)` method consolidates 7 prior local reinventions across trade_finder, waiver_wire, daily_optimizer, trade_evaluator, trade_simulator, in_season, start_sit.
- **`MatchupContextService`** (`src/matchup_context.py`) — Single source for category weights with 3 modes: "matchup" (H2H urgency), "standings" (gap analysis), "blended" (alpha-weighted).
- **`standings_utils.get_team_totals()`** — Session-cached roster totals.
- **`get_yahoo_data_service()`** — Singleton; 3-tier cache: session_state → Yahoo API → SQLite. All 11+ pages route through it (no raw `load_league_rosters/standings` in pages).
- **`format_stat(value, stat_type)`** (`src/ui_shared.py`) — Enforced display: AVG/OBP=`.3f`, ERA/WHIP=`.2f`, SGP=`+.2f`.
- **`get_connection()`** (`src/database.py`) — Only sanctioned SQLite access. Sets `PRAGMA journal_mode=WAL`, `busy_timeout=30000`, `synchronous=NORMAL`. Direct `sqlite3.connect` is structurally guarded against.
- **`PULP_AVAILABLE`** — Defined ONCE in `src/lineup_optimizer.py`. `src/optimizer/advanced_lp.py` imports it (no duplicate flag).

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

### Bootstrap Timing (measured 2026-04-12, Python 3.14, Windows)
- **Cold start (all stale):** ~30 minutes — Live Stats fetch is the largest phase (~8-12 min, network-bound)
- **Warm start (most data cached, <1h old):** ~2-5 minutes
- **Hot start (all data fresh):** ~5-10 seconds
- **Rate limits:** Yahoo ~2K-3K calls/day. Safe to do 10-15 full refreshes/day.

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

Pages should consume the pool — direct SQL is structurally guarded against in `pages/1_My_Team.py`, `pages/4_Trade_Analyzer.py`, `pages/7_Player_Compare.py`.

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
- **SQLite WAL mode + busy_timeout** — `get_connection()` sets `PRAGMA journal_mode=WAL`, `busy_timeout=30000`, `synchronous=NORMAL`. Direct `sqlite3.connect()` is structurally guarded against (`tests/test_no_direct_sqlite_connect_in_scripts.py`).
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
| `test_engine_no_fallback_singletons.py` | No `_LC = _LC_Class()` module singletons in engine portfolio modules |
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

## Data Audit History (compressed)

The data pipeline has been audited twice with all findings resolved.

**2026-04-18 audit (SF-1 → SF-14)** — Triggered by user-reported confusing DCV results. 4 parallel investigation agents found 14 silent-fail patterns. Headlines: SF-1 game_logs dead, SF-2 season_stats discarding 86%, SF-3 Two-Way Players skipped, SF-4 team_strength race condition, SF-5 depth_charts empty, SF-7 catcher_framing/umpire_tendencies empty, SF-13 pvb_splits status, SF-14 no persistent logging. All resolved in `feat/resilient-data-pipeline` (PR merged 2026-04-22) plus follow-on commits.

**2026-05-10 follow-up audit (SF-15 → SF-28)** — Proactive 5-agent sweep across bootstrap completeness, optimizer formulas, engine formulas, page-level data flow, cross-codebase consistency. Found 14 NEW silent-fail patterns. Headlines: SF-19 sprint_speed read but not in pool, SF-20 simple-mean rate-stat fallback, SF-21 stale `_LC` singleton, SF-22 park_factors Tier 1 ignored, SF-23 7 pages bypass Yahoo service, SF-25 7 SGP local reinventions consolidated to `SGPCalculator.totals_sgp`, SF-26 12+ hardcoded category lists, SF-27 MatchupContextService bypasses, SF-28 scripts using direct sqlite3.connect. All resolved across PRs #7-#11 (2026-05-10/11). Plan: [docs/superpowers/plans/2026-05-10-finish-data-audit-cleanup.md](docs/superpowers/plans/2026-05-10-finish-data-audit-cleanup.md).

**2026-05-11 whole-repo audit (SF-29 → SF-32)** — Triggered by a 24-agent parallel bug audit (22 code reviewers + Stream B player verification + Stream C refresh infrastructure). Found ~569 actionable findings; first wave (4 HIGH-severity data-correction bugs) resolved here. Headlines: SF-29 `league_rosters.name` SQL bugs in `injury_writeback` + `draft_results` (0 IL flagged, 0 R1-3 picks flagged undroppable for 8+ days), SF-30 `_bootstrap_dynamic_park_factors` used team OPS+ as park proxy (silent corruption every 7 days), SF-31 33 shadow rows with fake mlb_ids in [600000, 601999] (4 rostered IL players resolved to DSL prospects), SF-32 3 rostered players with NULL mlb_id (invisible to live-stats). All resolved in Wave 1 (PR #13).

**2026-05-11 Wave 2 (SF-33 → SF-37)** — Silent-failure elimination wave. SF-33 IP outs notation parsed as decimal (every ERA/WHIP off by ~1.5%; live DB confirmed all 536 pitcher IP values ended in .0/.1/.2 only). SF-34 Bayesian batch_update produced h/ab ≠ avg + hardcoded 35/65 BB:H split ignoring observed ratio. SF-35 Two-way player injury history overwritten by pitching gamesPlayed (Ohtani health_score 0.06 instead of 0.98). SF-36 ECR consensus source-weight lookups silently no-op due to `_rank`-suffix key mismatch. SF-37 IL-change detector flagged every existing IL player as "new" on cold start. All resolved in Wave 2 (PR will be filled in below).

**2026-05-11 Wave 3 (SF-38 → SF-41)** — Architectural cleanup wave. SF-38 `engine/signals/*` modules (decay/kalman/regime/statcast) are consumed only by the `src/trade_signals.py` UI helper, not by the core trade evaluator despite the Phase 3 architecture diagram's claim — added structural boundary guard. SF-39 `sigmoid_calibrator` and `sensitivity_analysis` patched dead module aliases (`COUNTING_STAT_K`, `RATE_STAT_K`) instead of the runtime read-path `CONSTANTS_REGISTRY` — calibration was a silent no-op; fixed via new `patch_sigmoid_k` context manager. SF-40 `engine/production/convergence.py` (effective_sample_size, split_rhat, check_convergence) was orphaned despite MC results displaying confidence_pct — wired `check_convergence` into `_run_mc_overlay` so trade-engine MC results now expose `convergence_quality`/`convergence_ess`/`convergence_rhat`. SF-41 `validation/calibration_data.py` + `calibrate_constants.py` called non-existent YahooFantasyClient methods (`get_transactions`/`get_matchups`/`get_standings`) and instantiated the client without `league_id` — calibration ran against empty data, silently. Rewrote against the actual YahooFantasyClient surface (`get_league_transactions`, `get_league_standings`; matchups defer until a per-week scoreboard wrapper exists). BUG-010 (`_LC` singletons in 11 files) and BUG-015 (survival calibrator data leakage) deferred to Wave 4/5.

**Session totals (2026-05-10/11):** ~50+ commits across 5 sequential PRs (#7-#11), 264+ new tests, 22 pre-existing tests rescued via `tests/conftest.py` `league_standings` fixture, 18 dispatched parallel-agent worktrees (all merged then cleaned up), zero regressions.

## What Was Done This Session (2026-05-10/11)

**HIGH-impact behavior fixes:**
- **SP gate now handles pure-`P` positions** — was dead code, now token-set membership catches `positions=={P}` case (audit's "unresolved contradiction")
- **Sprint speed boost no longer silently 0** — `sprint_speed` added to all 3 pool SELECT paths
- **Database fallback uses weighted rate aggregation** — replaced `AVG(proj.avg)` with `SUM(h)/SUM(ab)` etc.
- **`compute_marginal_sgp` reads live denominators** — was stuck on pre-season defaults via stale module singleton; now `config` parameter required, threaded through 5 trade-engine callers
- **Park factors Tier 1 honored** — was unconditionally overwriting with emergency dict; now `tier='primary'` when pybaseball returns data
- **7 pages now show fresh Yahoo data** — was reading raw SQL with no TTL
- **5+ pages format ERA/WHIP at correct precision** — was showing `3.850`, now `3.85`
- **K-boost FIP/xFIP proxy active** — was neutral 1.0× when Stuff+ unavailable; now derives from FIP (`(4.0-fip)/10` clipped to [0.85, 1.15])
- **catcher_framing + umpire_tendencies tables now have data** — 31 catchers + 34 umpires live in production via 3-tier waterfall (primary → Savant scrape → 2024 seed JSON)
- **Depth chart roles persisted** — 657 roles in production via MLB Stats API fallback (was 0)
- **My_Team L-drop bug fixed** — `_INVERSE_CATS` was `{"ERA","WHIP"}`, dropped L; Category Gaps card was treating Losses as positive-good

**Architectural consolidations:**
- **`SGPCalculator.totals_sgp` is sole SGP path** — 7 prior local reinventions deleted/migrated
- **13 src/ files derive categories from `LeagueConfig`** — 12+ hardcoded copies eliminated
- **`PULP_AVAILABLE` defined once** in `src/lineup_optimizer.py`
- **`_LC` singletons removed** from `engine/portfolio/category_analysis.py`, `copula.py`, `opponent_valuation.py`
- **`DEFAULT_SGP_DENOMS` deleted** — `opponent_valuation` functions require explicit `sgp_denominators`
- **Player pool has all `ytd_*` columns** for weighted aggregation (13 added)
- **True antithetic variates** in `trade_simulator.py` (negate uniforms; ~25% variance reduction)
- **MatchupContextService consumed by all pages** (no local weight computations)
- **`get_yahoo_data_service` consumed by all pages** that show live league data
- **3 scripts use `get_connection()`** — no direct `sqlite3.connect`
- **Bootstrap status snapshot** — `get_refresh_log_snapshot()` for ops/debug visibility
- **`league_standings` fixture** in `tests/conftest.py` — unblocked 22 pre-existing trade-engine test failures

**New infrastructure:**
- **`data/seed/`** — 2024 seed JSONs for catcher_framing (32 catchers) + umpire_tendencies (34 umpires); annual refresh procedure documented in `data/seed/README.md`
- **15 structural-test guards** in `tests/test_no_*.py`, `test_pages_*.py`, etc. — wired into CI
- **`data/logs/bootstrap.log`** + `bootstrap_results.json` for persistent diagnostics

## GitHub

- **Repo:** https://github.com/hicklax13/HEATER_v1.0.0
- **CI:** GitHub Actions — ruff lint/format + pytest (Python 3.11/3.12/3.13) + 60% coverage floor + structural-invariant guards
- **Recent PRs:** #7-#11 (2026-05-10/11) all merged to master

## Testing

- **~3700 passing tests** across 165+ test files, 13 skipped (PyMC/XGBoost/WeasyPrint optional)
- **CI:** GitHub Actions (3.11/3.12/3.13)
- **Coverage:** ~65% (60% CI floor)
- **Pre-commit hook:** Enforces `ruff format` + `ruff check` on every commit
- **Backtesting framework:** Historical replay validates engine recommendations vs actual outcomes

## Resume Checklist (New Session)

1. Read `CLAUDE.md` (this file)
2. Check git status: `git status`, `git log --oneline -10`
3. Run `python -m pytest --ignore=tests/test_cheat_sheet.py -x -q` to verify tests pass
4. Run `streamlit run app.py` and verify Yahoo auto-reconnect
5. Inspect refresh_log: `python -c "from src.database import get_refresh_log_snapshot; import json; print(json.dumps(get_refresh_log_snapshot(), indent=2))"`
6. Latest implementation plan: `docs/superpowers/plans/`

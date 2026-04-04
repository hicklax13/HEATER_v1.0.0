# HEATER — Fantasy Baseball Draft Tool

## Overview

A fantasy baseball draft assistant + in-season manager for a 12-team Yahoo Sports H2H Categories snake draft league.

1. **Draft Tool** (`app.py`) — "Heater" themed Streamlit app with glassmorphic design, splash screen bootstrap, 2-step setup wizard, 3-column draft page, Monte Carlo recommendations with percentile sampling. Zero CSV uploads — all data auto-fetched.
2. **In-Season Management** (`pages/`) — 13 pages: team overview, draft simulator, trade analysis, player comparison, free agents, lineup optimizer, closer monitor, standings/power rankings, leaders/prospects, waiver wire, start/sit advisor, matchup planner, trade finder.
3. **Trade Analyzer Engine** (`src/engine/`) — 6-phase pipeline: deterministic SGP → stochastic MC → signal intelligence → contextual adjustments → game theory → production convergence/caching.
4. **Enhanced Lineup Optimizer** (`src/optimizer/`) — 11-module pipeline: enhanced projections, matchup adjustments, H2H weights, non-linear SGP, streaming, scenarios, dual objective, advanced LP, category urgency, daily optimizer (DCV).
5. **Draft Recommendation Engine** (`src/draft_engine.py`) — 8-stage enhancement chain with 3 execution modes (Quick/Standard/Full).
6. **In-Season Analytics** (`src/`) — Trade value chart, two-start planner, start/sit advisor, matchup planner, waiver wire, trade finder, draft grader, prospect rankings, ECR consensus, player news.

## Commands

```bash
pip install -r requirements.txt        # Install deps
python load_sample_data.py             # Load sample data (first time/testing)
streamlit run app.py                   # Run the app
ruff check .                           # Lint
ruff format .                          # Format
python -m pytest                       # Run all tests (2300 pass, 4 skipped)
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
- **Manager skill:** Opponents extremely high-skilled; user is a novice

## Tech Stack

- **Framework:** Streamlit (Python), multi-page app
- **Database:** SQLite (`data/draft_tool.db`) — 24 tables
- **Core libs:** pandas, NumPy, SciPy, Plotly
- **Analytics:** PyMC 5 (Bayesian), PuLP (LP optimizer), arviz
- **Live data:** MLB-StatsAPI, pybaseball (FanGraphs Depth Charts + park factors)
- **Yahoo API:** yfpy + streamlit-oauth (optional OAuth)
- **Linter:** ruff (lint + format)
- **CI:** GitHub Actions — ruff lint/format, pytest (Python 3.11-3.13)
- **Python:** Local dev uses 3.14; CI tests 3.11-3.13

## File Structure

```
app.py                  — Draft tool: splash screen bootstrap + setup wizard + draft page
requirements.txt        — pip dependencies
load_sample_data.py     — Generates ~190 sample players + injury history for testing
.streamlit/config.toml  — Light theme configuration (Heater palette)
pages/
  1_My_Team.py          — Team overview, roster, category standings, Yahoo sync
  2_Draft_Simulator.py  — Standalone draft simulator with AI opponents, MC recommendations
  3_Trade_Analyzer.py   — Trade proposal builder + Phase 1 SGP engine with legacy fallback
  4_Player_Compare.py   — Head-to-head player comparison
  5_Free_Agents.py      — Free agent rankings by marginal value
  6_Lineup_Optimizer.py — 5-tab lineup optimizer powered by LineupOptimizerPipeline
  7_Closer_Monitor.py   — 30-team closer depth chart grid
  8_Standings.py        — Projected standings + power rankings
  9_Leaders.py          — Category leaders, points leaders, breakout detection, prospects
  10_Waiver_Wire.py     — Add/drop recommendations based on roster gaps (powered by src/waiver_wire.py)
  11_Start_Sit.py       — Weekly start/sit advisor with matchup analysis (powered by src/start_sit.py)
  12_Matchup_Planner.py — Per-game matchup ratings with color-coded quality tiers (powered by src/matchup_planner.py)
  10_Trade_Finder.py    — 4-tab trade finder: By Partner, By Category Need, By Value, Trade Readiness (powered by src/trade_finder.py)
src/
  database.py           — SQLite schema (24 tables), player pool + in-season queries
  valuation.py          — SGP calculator, replacement levels, VORP, category weights, percentiles
  draft_state.py        — Draft state management, roster tracking, snake pick order
  simulation.py         — Monte Carlo draft simulation, opponent modeling, survival probability
  in_season.py          — Legacy trade analyzer, player comparison, FA ranker
  live_stats.py         — MLB Stats API data fetcher
  data_bootstrap.py     — Zero-interaction bootstrap orchestrator (staleness-based refresh)
  league_manager.py     — League roster/standings management
  ui_shared.py          — THEME dict, PAGE_ICONS (inline SVGs), glassmorphic CSS, sidebar branding, 3-zone layout system, player card dialog
  player_card.py        — Player card data assembly (pure function, no Streamlit dependency)
  data_2026.py          — Hardcoded 2026 projections for sample data
  bayesian.py           — PyMC hierarchical model + Marcel regression fallback
  injury_model.py       — Health scores, age-risk curves, injury-adjusted projections
  lineup_optimizer.py   — PuLP LP solver for lineup optimization
  yahoo_api.py          — Yahoo Fantasy API OAuth integration via yfpy
  yahoo_data_service.py — Live-first data layer: 3-tier cache (session_state → Yahoo → SQLite)
  trade_intelligence.py — Trade valuation layer: health, scarcity, FA gating, Trade Readiness, correlation adjustments, schedule urgency
  trade_signals.py      — Kalman + regime trend adjustment for trade valuations
  data_pipeline.py      — FanGraphs auto-fetch (Steamer/ZiPS/Depth Charts)
  draft_engine.py       — DraftRecommendationEngine (3-mode, 8-stage pipeline)
  draft_analytics.py    — Category balance, opportunity cost, BUY/FAIR/AVOID
  contextual_factors.py — Closer hierarchy, platoon risk, lineup protection, schedule strength
  ml_ensemble.py        — XGBoost ensemble (optional, graceful fallback)
  news_sentiment.py     — Keyword-based news sentiment scoring
  trade_value.py        — Universal trade value chart (0-100) with G-Score adjustment
  two_start.py          — Two-start pitcher planner
  start_sit.py          — Start/sit advisor (3-layer decision model)
  matchup_planner.py    — Weekly matchup planner with percentile color tiers
  waiver_wire.py        — Waiver wire LP-verified add/drop recommendations
  trade_finder.py       — V3 trade finder: 3-tier scan (cosine→1v1→2v1), ADP/ECR/YTD/opponent needs, 52 variables
  opponent_trade_analysis.py — Opponent perspective: needs analysis, archetype detection, acceptance modeling
  game_day.py           — Game-day intelligence: weather (Open-Meteo), opposing pitchers, team strength, lineups, recent form, pitch mix
  espn_injuries.py      — Real-time ESPN injury feed for IL/DTD status
  draft_grader.py       — Post-draft grader (3-component)
  ecr.py                — Multi-platform ECR consensus (7 sources, Trimmed Borda Count, auto in-season swap to ROS rankings)
  prospect_engine.py    — FanGraphs Board API + MLB Stats API MiLB stats
  player_news.py        — 2-source news aggregation (MLB Stats API + Yahoo) with template summaries
  closer_monitor.py     — 30-team job security scoring
  points_league.py      — Yahoo/ESPN/CBS scoring presets
  leaders.py            — Category/points leaders, breakout detection
  cheat_sheet.py        — HTML/PDF export with HEATER theme
  live_draft_sync.py    — Yahoo draft polling and sync
  pick_predictor.py     — Normal CDF + Weibull survival blend
  player_tags.py        — Sleeper/Target/Avoid/Breakout/Bust tags with DB persistence
  schedule_grid.py      — 7-day schedule grid with matchup colors
  standings_projection.py — MC season simulation
  power_rankings.py     — 5-factor power rankings with bootstrap CI
  il_manager.py         — IL detection and replacement selection
  league_registry.py    — Multiple league support
  optimizer/            — Enhanced Lineup Optimizer (11 modules)
    pipeline.py         — Master orchestrator (Quick/Standard/Full modes)
    projections.py      — Enhanced projections (Bayesian/Kalman/regime/injury)
    matchup_adjustments.py — Park factors, platoon splits, weather
    h2h_engine.py       — H2H category weights + win probability
    sgp_theory.py       — Non-linear marginal SGP
    streaming.py        — Pitcher streaming + Bayesian stream scoring
    scenario_generator.py — Gaussian copula scenarios + CVaR
    dual_objective.py   — H2H/Roto weight blending
    advanced_lp.py      — Maximin, epsilon-constraint, stochastic MIP
    category_urgency.py — Sigmoid urgency for H2H matchup gaps
    daily_optimizer.py  — Daily Category Value (DCV) per-player per-day scoring
  engine/               — Trade Analyzer Engine (6 phases)
    portfolio/          — Z-scores, SGP, category analysis, lineup optimizer, copula
    projections/        — ROS projections, BMA, KDE marginals
    monte_carlo/        — Paired MC simulator (10K sims, variance reduction)
    signals/            — Statcast, decay, Kalman filter, regime detection
    context/            — Log5 matchup, injury process, bench value, concentration risk
    game_theory/        — Opponent valuation, adverse selection, Bellman DP, sensitivity
    production/         — Convergence diagnostics, cache, adaptive sim scaling
    output/             — Master trade orchestrator (evaluate_trade)
tests/                  — 101 test files, 2300 passing tests
data/
  draft_tool.db         — SQLite database (created at runtime)
  backups/              — Draft state JSON backups
docs/
  ROADMAP.md            — Phase history and future directions
.github/
  workflows/ci.yml      — CI pipeline (lint + test + build)
  workflows/refresh.yml — Scheduled daily data refresh (9:17 UTC)
```

## Architecture

### Core Valuation Pipeline
1. **Projection blending** — Weighted average of 7 systems (Steamer, ZiPS, Depth Charts, ATC, THE BAT, THE BAT X, Marcel)
2. **SGP** — Converts raw stats to standings-point movement; auto-computed denominators
3. **VORP** — Value Over Replacement Player with multi-position flexibility premium
4. **pick_score** = weighted SGP + positional scarcity + VORP bonus

### Draft Engine
- Monte Carlo simulation (100-300 sims, 6-round horizon) with opponent roster tracking
- Survival probability (Normal CDF) + Urgency = (1 - P_survive) * positional_dropoff
- Combined score = MC mean SGP + urgency * 0.4, tier assignment via natural breaks

### Trade Analyzer Engine (6 phases)
- **Phase 1** (default): Peer-group z-scores → standings-based SGP → marginal elasticity → punt detection → LP-constrained lineup totals → roster cap enforcement → weighted SGP delta → grade (A+ to F)
- **Phase 2** (`enable_mc=True`): BMA → KDE marginals → Gaussian copula → paired MC (10K sims)
- **Phase 3**: Statcast → signal decay → Kalman filter → regime detection (BOCPD/HMM)
- **Phase 4** (`enable_context=True`): Log5 matchup → injury process → concentration risk (HHI)
- **Phase 5** (`enable_game_theory=True`): Opponent valuations → adverse selection → Bellman DP → sensitivity/counter-offers
- **Phase 6**: Convergence diagnostics (ESS/R̂) → cache → adaptive sim scaling

### UI Layer — Hybrid 3-Zone Layout
All 10 pages use a consistent 3-zone layout pattern:
1. **Recommendation banner** (top) — One-line teaser, expandable for detail. Uses `render_page_layout()` + `render_reco_banner()`.
2. **Context panel** (left ~20%) — Compact glassmorphic cards with summary data (category totals, filters, settings). Uses `render_context_card()` inside `render_context_columns()`.
3. **Main content** (right ~80%) — Full data tables, charts, interactive controls. Uses `render_compact_table()` for ESPN-style 11px monospace tables with sticky player names, hit/pitch header colors, health dots, and starter/bench row tinting.

Key layout functions in `src/ui_shared.py`:
- `build_compact_table_html(df, ...)` — Pure function, returns HTML string. Unit-testable (39 tests).
- `render_compact_table(df, ...)` — Thin Streamlit wrapper around `build_compact_table_html()`.
- `render_reco_banner(teaser, detail, icon)` — Collapsible banner with `st.expander`.
- `render_context_card(title, content_html)` — Glassmorphic sidebar card.
- `render_page_layout(title, banner_teaser, ...)` — Page title badge + optional banner.
- `render_context_columns(context_width)` — Returns `(ctx, main)` column pair.

CSS classes: `.reco-banner`, `.context-card`, `.compact-table`, `.th-hit`/`.th-pit`, `.col-name`, `.row-start`/`.row-bench`, `.health-dot`. Responsive at 768px/480px. Print-friendly.

Sidebar is collapsed by default on all pages (hamburger menu for nav only).

### Yahoo Data Service (Live-First Architecture)
All league data flows through `src/yahoo_data_service.py`, a three-tier cache:
1. **Tier 1: `st.session_state`** — survives Streamlit reruns, ~0ms latency
2. **Tier 2: Yahoo Fantasy API** — live source of truth, 0.5s per call
3. **Tier 3: SQLite fallback** — used when Yahoo is offline

TTLs: Rosters 30m, Standings 30m, Matchup 5m, Free Agents 1h, Transactions 15m, Settings/Schedule 24h.

Write-through: every Yahoo fetch writes to SQLite so un-migrated code (trade engine, league_manager) automatically gets fresher data.

Singleton accessor: `get_yahoo_data_service()` stored in `st.session_state["_yahoo_data_service"]`.

Pages wired: My Team, Trade Analyzer, Free Agents, Lineup Optimizer, Standings, Trade Finder, Matchup Planner.

New UI features: live matchup score card (My Team), recent transactions feed (Trade Analyzer), data freshness widget (My Team, Trade Analyzer, Standings).

### Trade Intelligence Layer
All trade valuations (Trade Finder + Trade Analyzer) flow through `src/trade_intelligence.py`:
1. **Health adjustment** — IL15=0.84x, IL60=0.55x, DTD=0.95x, NA excluded. Uses `injury_model.compute_health_score()` with 3-year history + Yahoo IL/DTD status from `league_rosters.status` column.
2. **Category-weighted SGP** — `category_gap_analysis()` computes marginal weights per category. Categories where you can gain standings positions weighted higher. Punted categories get zero weight. Max weight capped at **1.5x** average (reduced from 3x to prevent single-category dominance in season-long trade decisions).
3. **FA gating** — Dynamic threshold (85% early season -> 60% late season) flags trades where a comparable FA exists. Prevents wasting trade capital.
4. **Closer scarcity premium** — SV >= 5 players get 1.3x multiplier. C/SS/2B positions get 1.15x VORP premium.
5. **Elite player protection** — Players in top 20% by raw SGP require return >= **75%** as valuable (raised from 50%).
6. **Trade Readiness tab** — 0-100 composite: 40% category fit + 25% projection confidence + 15% health + 10% scarcity + 10% FA advantage.
7. **Correlation adjustments** — Power cluster (HR/R/RBI) 0.83x, SB 1.08x premium, Contact (AVG/OBP) 0.90x, Pitching rate (ERA/WHIP) 0.88x.
8. **Schedule urgency** — 3-week lookahead multiplier [0.85, 1.25] based on opponent difficulty tiers.

### Trade Finder V3 Architecture
`src/trade_finder.py` — 3-tier scanning pipeline with 52 variables:
1. **Tier 1: Cosine dissimilarity** → finds top 5 complementary trade partners (opposite category strengths).
2. **Tier 2: 1-for-1 scan** → every user-opponent player pair through 8 hard filters + 6-component composite scoring.
3. **Tier 3: 2-for-1 greedy expansion** → top 15 seeds expanded by adding a second give player.

**Composite score (1-for-1):** `30% SGP*YTD + 15% ADP fair + 15% ECR fair + 20% acceptance + 10% opp benefit + 10% need match`

**Hard filters:** NA/minors exclusion, same-type (H↔H, P↔P), elite protection (P80 @ 75% floor), league draft gap ≤ 8 rounds, generic ADP ratio ≤ 2.5x, category breadth ≥ 2, min user gain ≥ 0.3 SGP, max opponent loss ≥ -0.5 SGP.

**Acceptance model (behavioral):** Sigmoid with loss aversion λ=1.8 (Brown 2024 meta-analysis), ADP penalty, bubble team bonus (rank 4-8), opponent need match, archetype willingness (0.3-0.7).

**Data sources:** Yahoo live rosters + standings, ECR consensus (551 players, 7-source Trimmed Borda Count), YTD 2026 stats, league draft picks, generic ADP (558 entries), opponent profiles + archetypes, free agent pool.

### Opponent Trade Analysis
`src/opponent_trade_analysis.py` — models the other side of the trade:
1. **compute_opponent_needs()** — Runs `category_gap_analysis()` from opponent's perspective to find their weak categories (rank ≥ 8th).
2. **get_opponent_archetype()** — Maps `OPPONENT_PROFILES` tier to trade willingness (0.3 passive → 0.7 active).
3. **analyze_from_opponent_view()** — Per-category impact assessment for opponent.

### Bootstrap Pipeline (21 phases)
- Every app launch: splash screen → `bootstrap_all_data()` with staleness-based refresh
- Staleness thresholds: 1h live stats/news, 30min Yahoo, 2h game-day, 24h projections/ADP/ECR/team strength, 7d players/prospects/depth charts, 30d historical/park factors
- Phase order: players → park factors → projections → live stats → historical → injury → Yahoo → extended roster → ADP → depth charts → contracts → news → dedup → prospects → news intel → ECR → player IDs → Yahoo transactions → Yahoo FAs → ROS projections → **game-day intelligence → team strength** (last 2 in parallel)
- Force Refresh sidebar button: `bootstrap_all_data(force=True)` overrides all staleness checks
- Bootstrap primes the SQLite cache; the YahooDataService handles ongoing freshness per-page

### Database Tables (24)

| Group | Tables |
|-------|--------|
| Draft | `projections`, `adp`, `league_config`, `draft_picks`, `blended_projections`, `player_pool` |
| In-Season | `season_stats`, `ros_projections`, `league_rosters`, `league_standings`, `park_factors`, `refresh_log` |
| Analytics | `injury_history`, `transactions` |
| Features | `player_tags`, `leagues` |
| Intelligence | `prospect_rankings`, `ecr_consensus`, `player_id_map`, `player_news`, `ownership_trends` |
| Game-Day | `game_day_weather`, `team_strength`, `opp_pitcher_stats` |

## Key API Signatures

Only the commonly-misused ones. For others, read the source files.

```python
# Game-Day Intelligence (src/game_day.py)
from src.game_day import (
    fetch_game_day_intelligence,    # Master: weather + pitchers + lineups + team strength
    fetch_game_day_weather,         # Open-Meteo → game_day_weather table
    fetch_opposing_pitchers,        # statsapi → opp_pitcher_stats table
    fetch_team_strength,            # pybaseball/statsapi → team_strength table
    get_team_strength,              # Read from DB, neutral defaults if missing
    get_player_recent_form,         # statsapi lastXGames (L7/L14/L30)
    get_player_recent_form_cached,  # Session-cached wrapper (2h TTL)
    fetch_pitcher_pitch_mix,        # pybaseball Statcast pitch type usage
    get_todays_lineups,             # statsapi boxscore_data batting orders
    STADIUM_COORDS,                 # 30-team lat/lon dict
    DOME_TEAMS,                     # 8 indoor/retractable stadiums
)

# YahooDataService (src/yahoo_data_service.py) — singleton accessor
from src.yahoo_data_service import get_yahoo_data_service
yds = get_yahoo_data_service()  # creates or retrieves from session_state
yds.get_rosters(force_refresh=False) -> pd.DataFrame     # 30min TTL
yds.get_standings(force_refresh=False) -> pd.DataFrame    # 30min TTL
yds.get_matchup(force_refresh=False) -> dict | None       # 5min TTL
yds.get_free_agents(max_players=500) -> pd.DataFrame      # 1hr TTL
yds.get_transactions(force_refresh=False) -> pd.DataFrame # 15min TTL
yds.get_schedule(force_refresh=False) -> dict             # 24hr TTL
yds.get_opponent_profile(team_name) -> dict               # from live standings
yds.force_refresh_all() -> dict[str, str]                 # invalidate + refetch
yds.get_data_freshness() -> dict[str, str]                # "Live (2m ago)" per key
yds.is_connected() -> bool                                # Yahoo auth status

# Freshness widget (src/ui_shared.py) — call inside context column
from src.ui_shared import render_data_freshness_card
render_data_freshness_card()  # shows badges + "Refresh All" button

# Opponent intel with live data (src/opponent_intel.py)
get_current_opponent(yds=yds)       # live schedule + profile when yds provided
get_opponent_for_week(week, yds=yds)  # same, for specific week

# Trade Intelligence (src/trade_intelligence.py)
get_health_adjusted_pool(player_pool, config) -> pd.DataFrame  # IL/DTD/NA adjusted
get_category_weights(user_team_name, all_team_totals, config) -> dict[str, float]
compute_fa_comparisons(opp_ids, user_ids, fa_pool, pool, config) -> dict[int, dict]
apply_scarcity_flags(player_pool) -> pd.DataFrame  # adds is_closer, scarcity_mult
compute_trade_readiness(player_id, ...) -> dict  # 0-100 composite score
compute_trade_readiness_batch(player_ids, ...) -> pd.DataFrame  # batch scoring

# STANDALONE functions in src/valuation.py — NOT methods on SGPCalculator
compute_replacement_levels(pool, config, sgp_calc)   # valuation.py
compute_sgp_denominators(pool, config)                # valuation.py

# Keyword args required for optional params
value_all_players(pool, config, roster_totals=None, category_weights=None,
                  replacement_levels=None, current_round=None, num_rounds=23)

# load_player_pool() columns:
# player_id, name, team, positions, is_hitter, is_injured,
# pa, ab, h, r, hr, rbi, sb, avg, obp, bb, hbp, sf, ip, w, l, sv, k, era, whip, er, bb_allowed, h_allowed, adp

# Trade Finder V3 (src/trade_finder.py)
find_trade_opportunities(user_roster_ids, player_pool, config=None,
    all_team_totals=None, user_team_name=None, league_rosters=None,
    weeks_remaining=16, max_results=20, top_partners=5) -> list[dict]
# Returns: [{giving_ids, receiving_ids, giving_names, receiving_names,
#   user_sgp_gain, opponent_sgp_gain, acceptance_probability, acceptance_label,
#   composite_score, trade_type, is_closer_trade, give_adp_round, recv_adp_round,
#   adp_fairness, ecr_fairness, give_ecr_rank, recv_ecr_rank, ytd_modifier,
#   opp_need_match, opponent_team, complementarity, grade, health_risk, ...}]

compute_adp_fairness(give_id, recv_id, player_pool) -> float  # 0-1, uses league draft round (primary) + ADP (fallback)
estimate_acceptance_probability(user_gain, opp_gain, need_match,
    adp_fairness=0.5, opponent_need_match=0.5, opponent_standings_rank=None,
    opponent_trade_willingness=0.5) -> float  # 0-1 sigmoid with loss aversion 1.8

# Opponent Trade Analysis (src/opponent_trade_analysis.py)
compute_opponent_needs(opp_team, all_team_totals, config) -> dict  # per-cat rank + gap
get_opponent_archetype(team_name) -> dict  # trade_willingness 0.3-0.7

# Trade evaluator (src/engine/output/trade_evaluator.py)
evaluate_trade(giving_ids, receiving_ids, user_roster_ids, player_pool,
               config=None, user_team_name=None, weeks_remaining=16,
               enable_mc=False, enable_context=True, enable_game_theory=True)
# Returns: {grade, surplus_sgp, category_impact, category_analysis, punt_categories,
#           replacement_penalty, risk_flags, verdict, confidence_pct,
#           before_totals, after_totals, giving_players, receiving_players,
#           total_sgp_change, mc_mean, mc_std, lineup_constrained, drop_candidate, fa_pickup, ...}

# DraftRecommendationEngine (src/draft_engine.py)
DraftRecommendationEngine(config: LeagueConfig, mode: str = "standard")
# .enhance_player_pool(player_pool, draft_state, park_factors=None) -> DataFrame
# .recommend(player_pool, draft_state, top_n=8, n_simulations=300, park_factors=None) -> DataFrame

# LineupOptimizerPipeline (src/optimizer/pipeline.py)
LineupOptimizerPipeline(roster, mode="standard", alpha=0.5, weeks_remaining=16, config=None)
# .optimize(...) -> {lineup, category_weights, h2h_analysis, streaming_suggestions, ...}

# Bootstrap (src/data_bootstrap.py)
bootstrap_all_data(yahoo_client=None, on_progress=None, force=False, staleness=None)

# Data pipeline (src/data_pipeline.py)
SYSTEM_MAP = {"steamer": "steamer", "zips": "zips", "fangraphsdc": "depthcharts"}
refresh_if_stale(force=False)

# UI shared (src/ui_shared.py)
THEME: dict  # Single light-mode palette: bg=#f4f5f0, primary=#e63946, hot=#ff6d00, etc.
T = THEME    # Backward compat: T["amber"]→#e63946, T["teal"]→#457b9d
PAGE_ICONS: dict[str, str]  # ~22 inline SVG icons
inject_custom_css()  # Full CSS injection (~1500 lines)

# Opponent model (src/simulation.py, src/draft_state.py)
compute_team_preferences(draft_history_df)  # needs columns: team_key, positions, round (NOT team_id/position)
get_team_draft_patterns(draft_state_dict, team_id: int)  # team_id is 0-based index

# Injury model (src/injury_model.py)
get_injury_badge(health_score) -> tuple[str, str]  # returns <span> with CSS dot, NOT emoji
# 4-tier thresholds: >= 0.85 Low Risk, >= 0.65 Moderate Risk, >= 0.40 Elevated Risk, < 0.40 High Risk
```

## Gotchas

### Data & Schema
- **DB column `name` vs UI `player_name`** — Normalized via `rename(columns={"name": "player_name"})` in `_build_player_pool()`. `evaluate_candidates()` returns `name`, not `player_name` — always alias after calling.
- **Sample data is `system='blended'` only** — Do NOT call `create_blended_projections()` when only blended data exists; it deletes `system='blended'` rows first.
- **Python 3.14 + SQLite bytes** — Returns bytes for some integer columns. Fixed with explicit `CAST` in SQL and `pd.to_numeric()` coercion. Also affects yfpy: `team.name` may return `bytes` — always guard with decode.
- **Players table has no UNIQUE constraint on name** — `upsert_player_bulk()` uses SELECT-first pattern. Do NOT use `ON CONFLICT(name)`. However, `injury_history` has UNIQUE on `(player_id, season)`.
- **Park factors schema** — Columns: `team_code`, `factor_hitting`, `factor_pitching` (not `team`/`park_factor`).
- **FanGraphs API SYSTEM_MAP** — FG API uses `"fangraphsdc"` for Depth Charts, DB stores as `"depthcharts"`. Always use `SYSTEM_MAP` dict.
- **LeagueConfig is the single source of truth** — All category definitions live in `LeagueConfig` in `src/valuation.py`. Do NOT add local hardcoded category lists — import from LeagueConfig.

### Yahoo API
- **Yahoo OAuth uses oob (out-of-band)** — `redirect_uri=oob` required. Do NOT use localhost redirects.
- **Yahoo auto-reconnect** — `_try_reconnect_yahoo()` loads `data/yahoo_token.json` on restart. Token refresh is automatic.
- **yfpy Roster iteration** — `Roster.__iter__()` yields attribute NAMES (strings), NOT players. Use `getattr(roster, "players", None) or []`.
- **yfpy token override** — When `yahoo_access_token_json` is provided, do NOT also pass `consumer_key`/`consumer_secret`.
- **Yahoo game_key** — MLB 2026 = 469. Must set BOTH `self._query.game_id` AND `self._query.league_key`.

### Streamlit UI
- **No emoji in the codebase** — All icons are inline SVGs from `PAGE_ICONS`. Injury badges use CSS dots. Do NOT introduce emoji.
- **No abbreviations in UI text** — Use full terms: "Standings Gained Points" not "SGP", "Value Over Replacement Player" not "VORP", etc.
- **CSS `!important` required** — Streamlit's inline CSS has high specificity. ALL custom CSS needs `!important`.
- **HTML sanitization** — `st.markdown('<div>')` auto-closes tags. Cannot use split open/close patterns with widgets between. Use self-contained HTML blocks.
- **Health badges in st.dataframe()** — Renders raw text only. Use text labels for dataframes, HTML badges only in `st.markdown(unsafe_allow_html=True)`.
- **`T["ink"]` for text-on-accent** — Use `T["ink"]` for dark text on colored surfaces, not `T["bg"]` (invisible on amber).
- **Connection leak pattern** — Always wrap `get_connection()` in `try/finally` with `conn.close()` in `finally`.
- **Draft Settings live on Draft Simulator page** — Not on Connect League page. Stored in `mock_num_teams`, `mock_num_rounds`, `mock_draft_pos`.

### Visual Audit Gotchas (discovered via Playwright)
- **THEME keys** — Use `T["tx"]` not `T["tx1"]`. There is no `tx1` key. This crashed the Start/Sit page.
- **Player Compare must not auto-select** — Default `compare_a`/`compare_b` to `None`, not `top_a[0]`. Otherwise the page loads with two random nobodies showing all-zero stats.
- **Matchup Planner empty schedule** — When `games_count` is 0 for all players, show empty state instead of fake "Avoid" ratings. Check `ratings_df["games_count"].sum() == 0` before rendering.
- **Sidebar icon map** — New pages need entries in the JS `icons` dict in `inject_custom_css()`. Keys must match the sidebar link text exactly (e.g., `'Waiver Wire'`, `'Start Sit'`, `'Matchup Planner'`).
- **Leaders Team column empty** — Known data-quality issue: `players.team` is "MLB" or NULL for many players. Not a page bug — the MLB Stats API doesn't always return team abbreviations.
- **Closer Monitor only 6 teams** — Known: FanGraphs depth chart data is incomplete in the bootstrap. Not a page bug.

### Player Card & News
- **`player_news` table has no UNIQUE constraint** — Yahoo sync can insert duplicate rows. `_render_news_tab` deduplicates by `player_name + headline` (case-insensitive) before rendering.
- **Bayesian table must filter to roster only** — `season_stats` has ALL players across ALL seasons. Query with `WHERE player_id IN (roster_pids)` and `drop_duplicates(subset=["player_id"], keep="first")` after sorting by season DESC.
- **`mlb_id` can be `float('nan')` from SQLite NULL** — NaN is truthy in Python. Use `mlb_id is not None` + `try: int(mlb_id)` instead of `if mlb_id:`.
- **Yahoo team names may contain emoji** — Strip non-alpha chars before extracting initials for monogram fallback.
- **`league_rosters`/`league_standings` need UNIQUE constraints** — Schema in `database.py` has them but old DBs may not. Drop and recreate tables if `ON CONFLICT` fails.

### Algorithms & Math
- **Rate-stat aggregation** — AVG=sum(h)/sum(ab), OBP=sum(h+bb+hbp)/sum(ab+bb+hbp+sf), ERA=sum(er)*9/sum(ip), WHIP=sum(bb+h)/sum(ip). Weighted averages, NOT simple averages.
- **LP inverse stat weighting** — ERA/WHIP must be weighted by IP in LP objective. Without it, 1-IP relievers dominate 200-IP starters.
- **L (Losses) is an inverse counting stat** — Lower-is-better but counting. Sign-flip in SGP, no IP weighting in LP.
- **Punt detection requires BOTH conditions** — `gainable_positions == 0 AND rank >= 10`.
- **LP-constrained lineup totals** — Only 18 starters' stats feed SGP. Bench excluded. Prevents phantom production in uneven trades.
- **Roster cap enforcement (ROSTER_CAP=23)** — 2-for-1: drop lowest-SGP bench. 1-for-2: add FA capped at median SGP.
- **Percentile pipeline ordering** — `compute_projection_volatility()` → `add_process_risk()` → `compute_percentile_projections()`. Skip when only one projection system.
- **P90 ERA/WHIP inversion** — P90=optimistic=lowest ERA, P10=pessimistic=highest ERA. Uses `z_eff = -z`.
- **Kalman prior_var = process variance only** — NOT `proc_var + obs_var` (double-counts noise).
- **Paired MC seed discipline** — Same seed for before/after rosters. NEVER use different seeds.
- **Maximin LP excludes inverse cats** — L, ERA, WHIP auto-filtered (incompatible SGP scale).
- **Opponent pick probability is additive blend** — Not multiplicative (collapses to near-zero).
- **Trade analyzer needs standings for accuracy** — Without standings: no elasticity, no punts, no strategic context.

### Dependencies & Platform
- **Plotly 6.x hex colors** — Does NOT accept 8-digit hex (`#RRGGBBAA`). Use `rgba(r,g,b,a)`.
- **PyMC/PuLP optional deps** — `PYMC_AVAILABLE`/`PULP_AVAILABLE` flags. Use `try/except ImportError`. CI skips with `@pytest.mark.skipif`.
- **XGBoost optional** — `XGBOOST_AVAILABLE` flag. Returns zeros when unavailable.
- **`ruff` on Windows** — Use `python -m ruff check .` instead of bare `ruff`.
- **Cheat sheet PDF requires weasyprint** — Returns None when not installed.
- **Snake draft order** — `round % 2 == 0` forward, `round % 2 == 1` reverse.
- **`datetime.now(UTC)`** — Used everywhere; `datetime.utcnow()` is deprecated.

### Trade Engine Specifics
- **Trade engine fallback** — Page wraps engine import in try/except, falls back to legacy `analyze_trade()`.
- **`enable_mc=True` activates Phase 2** — Default is Phase 1 only (deterministic).
- **`bench_cost` is always 0.0** — Replaced by explicit roster cap modeling. Key kept for backward compat.
- **Sensitivity calls use `enable_mc=False, enable_context=False`** — Avoids expensive calls during N-player analysis.
- **FA pickup median SGP cap** — Prevents unrealistic elite "FA" acquisitions when no rosters loaded.
- **Replacement penalty skips rate stats** — AVG, ERA, WHIP excluded (roster-aggregate, not simple counting gaps).

### Trade Finder V3 Specifics
- **LOSS_AVERSION = 1.8** — Up from 1.5 per Brown 2024 meta-analysis. Must match in tests.
- **MAX_WEIGHT_RATIO = 1.5** — Category weight cap in `scan_1_for_1()`. Was 3.0 then 2.0. Prevents AVG-only specialists dominating when AVG is weighted.
- **ELITE_RETURN_FLOOR = 0.75** — P80 players require return >= 75% SGP. Was 50% (let Story trade for Alvarez).
- **ADP fairness uses league draft round first** — `get_player_draft_round()` from `league_draft_picks` table. Falls back to generic ADP (`adp` table, 558 entries). Unknown ADP returns 0.5 (neutral).
- **ECR fairness sqrt-softened** — `(min_rank / max_rank) ^ 0.5`. Without sqrt, small rank gaps would be over-penalized.
- **YTD modifier clamped ±10%** — `max(0.90, min(1.10, ytd_avg / proj_avg))`. Requires >= 10 PA. Prevents chasing 3-game streaks.
- **Multi-player ADP guard** — Added player in 2-for-1 must have `add_adp >= recv_adp * 0.5`. Prevents bundling elite players as "throw-ins".
- **Slot reservation** — 50% of `max_results` reserved for 1-for-1, 50% for multi-player. Without this, 2-for-1 trades (with ROSTER_SPOT_SGP bonus) crowd out 1-for-1.
- **Category dominance check (40%) exists but NOT active** — `_check_category_dominance()` was written but rejected: when AVG weighted 2x, every hitter exceeds 40% threshold. Kept as utility function.
- **Opponent needs default to 0.5** — When `opponent_trade_analysis` fails to load, all opponent scores are neutral. Never crashes the scanner.

## Data Sources

- **Players:** MLB Stats API (750+ active + 40-man); staleness: 7 days
- **Projections:** 7 systems via FanGraphs JSON API + Marcel (local); staleness: 24 hours
- **ROS Projections:** Steamer ROS + ZiPS ROS + Depth Charts ROS via FanGraphs; staleness: 24 hours
- **ADP:** FanGraphs Steamer + FantasyPros ECR + NFBC; staleness: 24 hours
- **Current stats:** MLB Stats API; staleness: 1 hour
- **Historical stats:** 3 years (2023-2025) from MLB Stats API; staleness: 30 days
- **Park factors:** Hardcoded FanGraphs 2024 values (30 teams); staleness: 30 days
- **League data:** Yahoo Fantasy API (optional); staleness: 30 minutes
- **News:** MLB Stats API + Yahoo (2 active sources); staleness: 1 hour
- **Prospects:** FanGraphs Board API + MLB Stats API MiLB; staleness: 7 days
- **ECR:** 7-source aggregation (auto-swaps to ROS rankings in-season); staleness: 24 hours
- **Weather:** Open-Meteo API (30 stadium coords, dome detection); staleness: 2 hours
- **Team strength:** pybaseball wRC+/FIP (FanGraphs) with statsapi fallback; staleness: 24 hours
- **Opposing pitchers:** MLB Stats API season stats + platoon splits; staleness: 2 hours
- **Recent form:** MLB Stats API lastXGames (L7/L14/L30); on-demand, 2h session cache
- **Pitch mix:** pybaseball Statcast pitch type usage; on-demand, 24h session cache
- **Lineups:** MLB Stats API boxscore_data (confirmed lineups); staleness: 2 hours

## GitHub

- **Repo:** https://github.com/hicklax13/fantasy-baseball-draft-tool (public)
- **Current release:** v1.0.0 (live data pipeline added April 3, 2026)

## Testing

- **2300 passing tests** across 101 test files, 4 skipped (PyMC/xgboost optional deps)
- **CI:** GitHub Actions — ruff lint/format + pytest on Python 3.11, 3.12, 3.13
- **Coverage:** 64% (above 60% CI threshold)
- **8 rounds of systematic debugging** (207 bugs fixed) + **data pipeline audit** (32 issues fixed), all CI green
- **Full system audit** (March 26, 2026) — 19 bugs cataloged, all critical/high fixed
- **Manual UI testing** — All 13 pages tested via Playwright + Claude in Chrome (March 2026)
- **Trade Engine V3** (March 31, 2026) — 5-agent deep research, 52-variable algorithm, 5+ rounds of iterative testing
- **Live Data Pipeline** (April 3, 2026) — 10-agent parallel implementation, 15 data gaps filled, 39 new tests
- **Deep Verification** (April 3, 2026) — All 13 pages tested via Claude in Chrome. 17 issues found, 14 fixed (V-001 through V-017). See `docs/VERIFICATION_LOG.md`

## Current Implementation Plan

**Read `The_Last_Plan.md` at the start of every session.** It is the active implementation plan.

- **The Last Plan:** AVIS-driven in-season optimization — Yahoo exhaustive FA sync, AVIS hard constraints, opponent intelligence, weekly automation
- **Status:** All 4 phases implemented and tested (2022 pass, 0 fail)
- **Trade Engine V3:** Completed — ADP-weighted scoring, multi-player trades, opponent modeling, schedule urgency, category correlations, ECR/YTD integration
- **Lineup Optimizer V2:** Completed — DCV daily optimizer, category urgency, enhanced projections
- **Previous plans:** `plan_1.md` (completed), `Full_Debug_Plan.md` (completed)
- **AVIS Manual:** `AVIS_FANTASY_BASEBALL_OPS_MANUAL_2026.md` — the "bible" for league rules, scoring, team analysis
- **Season context:** MLB 2026 season started March 25. Fantasy draft completed. App is now in-season mode.
- **Systematic Debug (April 1, 2026):** 49 bugs fixed across 21 files, acceptance panel added to Trade Analyzer, all pages audited for upstream consistency
- **Design docs:** `docs/superpowers/specs/2026-03-31-trade-engine-v3-research.md` (deep research), `docs/superpowers/specs/2026-03-29-lineup-optimizer-v2-research.md` (optimizer research)

## New Modules (Added March 26, 2026)

- **`src/alerts.py`** — AVIS-enforced decision rules: auto-IL, closer handcuff, streaming SP, sell-high/buy-low triggers
- **`src/opponent_intel.py`** — Opponent roster analysis, trade target finder, weakness detection
- **`src/weekly_report.py`** — Monday morning briefing: roster health, matchup preview, waiver targets, trade opportunities
- **`src/ip_tracker.py`** — Weekly IP tracking toward 1,400 IP target (AVIS requirement)

## New Modules (Added March 31, 2026)

- **`src/opponent_trade_analysis.py`** — Opponent perspective modeling: needs analysis, archetype detection, acceptance scoring
- **`src/espn_injuries.py`** — Real-time ESPN injury feed for IL/DTD/NA status
- **`src/optimizer/category_urgency.py`** — Sigmoid urgency for H2H matchup gaps (k=2.0 counting, k=3.0 rate)
- **`src/optimizer/daily_optimizer.py`** — Daily Category Value (DCV) per-player per-day scoring with team name mapping

## New Modules (Added April 3, 2026)

- **`src/game_day.py`** — Game-day intelligence: 30-stadium weather (Open-Meteo), opposing pitcher stats + platoon splits, team strength (pybaseball wRC+/FIP), confirmed lineups (statsapi boxscore), recent player form (L7/L14/L30 game logs), Statcast pitch mix, session_state caching

## Key Fixes (March 26, 2026)

- **BUG-001 FIXED:** FanGraphs 403 → pybaseball + Marcel fallback pipeline
- **BUG-003 FIXED:** Empty team fields → team_id-to-abbreviation mapping from MLB API
- **BUG-004 FIXED:** 780 players → 9,213 via expanded roster fetch (40-man + spring training)
- **Yahoo exhaustive FA sync** — Paginated fetch of ALL league free agents (not just 50)
- **Start/Sit scoring** — ERA/WHIP now use delta-from-baseline SGP (was all-negative)
- **Trade Analyzer** — Dropdown shows tradeable players from other teams (was 9K player pool)
- **Lineup Optimizer** — Pitchers now counted in starting lineup (was hitter-only)
- **Points Leaders** — Fixed `list.empty` AttributeError, returns DataFrame

## Key Fixes (April 1, 2026)

- **Systematic Debug** — 9 parallel agents audited 16 files, found 49 bugs (1 critical, 16 high, 24 medium, 8 low), all fixed
- **TI-001 CRITICAL:** ERA/WHIP recalculated with unreduced er/bb_allowed/h_allowed — injured pitchers got worse rate stats
- **TI-002 HIGH:** Health score double-dip: IL15 players lost 36% instead of 16%
- **TI-003 HIGH:** "C" in "CF" substring match gave center fielders catcher scarcity premium
- **TI-004 HIGH:** OBP never recalculated after IL counting stat adjustment
- **TF-001/TF-002 HIGH:** 2-for-1 drop cost always zero — opponent roster squeeze never penalized
- **TF-004 HIGH:** Closer scarcity 1.3x multiplied entire SGP delta instead of saves only
- **OP-001 HIGH:** Statcast step called with player name instead of mlb_id — entire step was dead code
- **OP-002 HIGH:** Injury history query had no ORDER BY — health scores used arbitrary season order
- **OP-003 HIGH:** Rate stats filled with 0 for NaN — ERA=0.00 made pitchers look perfect
- **CU-001 HIGH:** math.exp() overflow on extreme WHIP gaps crashed optimizer
- **DO-001 HIGH:** Arizona="AZ"/Oakland="ATH" abbreviations broke park factor lookups
- **DO-002 HIGH:** STUD_FLOOR_TOP_N=30 > 23-player roster — all players became studs
- **DO-001/LP HIGH:** Single-source categories bypassed alpha blending in dual objective
- **LP-001 HIGH:** Daily Optimize tab read park_factors from empty session_state
- **LP-003 HIGH:** park_factor_adjustment() returned player's own team PF instead of venue
- **OPT-005/006 HIGH:** Streaming relievers with 15-30 IP had inflated values; 0-game pitchers not excluded
- **Acceptance Analysis Panel** — Trade Analyzer now shows ADP Fairness, ECR Fairness, Acceptance Probability, Acceptance Tier
- **Page Audit** — 9 page bugs fixed across Trade Analyzer, Lineup Optimizer, My Team, Player Compare, Start/Sit
- **Smart Drop Candidate** — 5-factor scoring replaces pure SGP: roster balance, DH-only penalty, 0-SB dead weight, AVG drag, base SGP
- **UI Sizing Overhaul** — Tables 11→13px, tabular-nums, unified 12px border-radius, 1024px breakpoint, rate stat formatting (AVG 3dp, ERA 2dp)
- **Browser Testing** — 4 parallel agents tested all pages with real clicks; found and fixed 9 additional UI bugs
- **Daily Lineup Validation** — My Team page now shows lineup issues (off-day starters) with position-eligible bench replacement suggestions
- **Ownership Heat Index** — Free Agents page shows Heat Score (1-10) per player based on ownership trends and recent adds; breakout candidate filter
- **Monday Morning Report** — Weekly report auto-expands on Mondays with opponent analysis, category projections, action items, streaming targets

## Key Fixes (April 3, 2026)

- **TRADED-PLAYER GHOST BUG:** Yahoo `get_team_roster_by_week()` returns players traded away mid-week with `position: None`. Fixed both `get_all_rosters()` and `get_team_roster()` to filter these ghosts. Corey Seager (traded for Harper + Stott) was appearing on Team Hickey's roster.
- **Live Data Pipeline** — 10-agent parallel implementation filling 15 data gaps:
  - Weather integration (Open-Meteo, 30 stadiums, dome detection) → wired into optimizer DCV
  - Team strength (pybaseball wRC+/FIP with statsapi fallback)
  - Opposing pitcher stats + platoon splits (vs LHB/RHB)
  - Recent player form (L7/L14/L30 game log aggregation, session cached)
  - FanGraphs ROS projections (steamerr/rzips/rfangraphsdc)
  - ECR in-season auto-swap (dead sources → FantasyPros ROS)
  - Confirmed lineups via statsapi boxscore_data
  - Statcast pitch mix (pybaseball, 24h session cache)
  - News cleanup (2 active sources documented, dead stubs removed from aggregation)
- **Staleness Tightened** — Projections 7d→24h, Yahoo 6h→30min, new game-day 2h, team strength 24h
- **Bootstrap Phases 20-21** — Game-day intelligence + team strength run in parallel at startup
- **Force Refresh Button** — Sidebar button calls `bootstrap_all_data(force=True)`
- **3 New DB Tables** — `game_day_weather`, `team_strength`, `opp_pitcher_stats`

## Deep Verification Fixes (April 3, 2026)

- **V-004/V-007 FIXED:** Closer Alert used `sv >= 5` on actual saves — early-season closers have <5. Now checks projected SV too (`pages/1_My_Team.py`, `src/alerts.py`).
- **V-006 FIXED:** Injury alerts showed league-wide injuries (Kirby Yates, Nick Lodolo) instead of roster-only. Added roster filtering by player_id and name (`src/alerts.py`).
- **V-009/V-010 FIXED:** Shohei Ohtani appeared twice in Draft Simulator (TWP dual-entry) causing `StreamlitDuplicateElementKey` crash. Deduplicated by player_id + appended index to button key (`pages/2_Draft_Simulator.py`).
- **V-011/V-016 FIXED:** IL stash players (Bieber, Strider) suggested for drops/trades. Promoted `IL_STASH_NAMES` to module-level constant in `src/alerts.py`. Added guard in `scan_1_for_1()` and `scan_2_for_1()` (`src/trade_finder.py`).
- **V-015 FIXED:** All complementarity scores = 1.00. Root cause: z-score vectors had near-zero norms hitting the 1e-9 guard. Fixed std floor (1e-6→0.01), added raw deviation fallback, changed zero-norm default from 1.0 to 0.5 (neutral), clamped to [0,1] (`src/trade_finder.py`).
- **V-001 FIXED:** HTML `<div style="f` leak in Closer Monitor cards (ARI, ATH, ATL). Replaced empty `actual_sv_html=""` with HTML comment + `html.escape()` for closer names (`pages/7_Closer_Monitor.py`).
- **V-013 FIXED:** Closer Monitor showed 34 teams. Added `_TEAM_NORMALIZE` dict (ATH→OAK, AZ→ARI, etc.) (`pages/7_Closer_Monitor.py`).
- **V-014 FIXED:** Power Rankings Schedule Strength was N/A. Now computes as avg opponent roster quality in projection-only path (`pages/8_Standings.py`).
- **V-012 FIXED:** Player Compare buttons showed "A..." — reduced from 5 to 3 per row (`pages/6_Player_Compare.py`).
- **V-017 FIXED:** Trade Finder BY VALUE tab truncated. Reduced to 8 essential columns + explicit widths (`pages/10_Trade_Finder.py`).

## Season State (2026)

- **MLB game_key:** 469
- **Draft:** Completed (snake, 23 rounds)
- **Mode:** In-season management (rosters, trades, waivers, matchups)
- **Yahoo token:** Auto-refreshes via `data/yahoo_token.json` (persistent across sessions)
- **Yahoo auto-reconnect:** `_try_reconnect_yahoo()` loads token on every app launch
- **Player pool:** 9,213 players loaded from MLB Stats API + pybaseball + Marcel
- **League rosters:** All 12 teams synced (264 players)
- **Free agents:** Exhaustive pagination from Yahoo API

## Resume Checklist (New Session)

1. Read `CLAUDE.md` (this file) and `The_Last_Plan.md`
2. Read `AVIS_FANTASY_BASEBALL_OPS_MANUAL_2026.md` (league bible)
3. Check `docs/AUDIT_REPORT.md` for any remaining bugs
4. Run `python -m pytest -x -q` to verify all tests pass
5. Run app with `streamlit run app.py` and verify Yahoo auto-reconnect works
6. Continue with any remaining tasks from The Last Plan or user requests

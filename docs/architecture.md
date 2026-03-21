# HEATER Architecture

> System architecture for the HEATER Fantasy Baseball Draft Tool & In-Season Manager.
> Last updated: 2026-03-20

---

## System Overview

HEATER is a Streamlit-based fantasy baseball application with two pillars: a draft assistant and an in-season manager. The codebase is organized into 9 architectural layers, 60+ source modules, 83 test files (1956+ tests), and integrates with 12 external APIs — all with graceful degradation.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Layer 1: Entry Points                        │
│  app.py (draft tool) + 11 Streamlit pages                      │
├─────────────────────────────────────────────────────────────────┤
│                    Layer 2: Database                            │
│  src/database.py — 21 SQLite tables, bulk upserts              │
├─────────────────────────────────────────────────────────────────┤
│                    Layer 3: Core Analytics                      │
│  valuation · simulation · draft_state · injury · bayesian       │
│  in_season · lineup_optimizer · ui_shared                       │
├─────────────────────────────────────────────────────────────────┤
│                    Layer 4: Data Ingestion                      │
│  data_bootstrap (9 phases) · data_pipeline · live_stats         │
│  yahoo_api · adp_sources · depth_charts · news_fetcher          │
├─────────────────────────────────────────────────────────────────┤
│                    Layer 5: Trade Engine                        │
│  6-phase pipeline: SGP → MC → Signals → Context → Game Theory  │
│  → Production (src/engine/ — 20 modules)                       │
├─────────────────────────────────────────────────────────────────┤
│                    Layer 6: Enhanced Lineup Optimizer            │
│  11-module pipeline: projections → matchups → H2H → SGP →      │
│  streaming → scenarios → multi-period → dual-obj → advanced LP  │
├─────────────────────────────────────────────────────────────────┤
│                    Layer 7: Draft Recommendation Engine          │
│  8-stage enhancement: park → Bayesian → injury → Statcast →    │
│  FIP → contextual → category balance → ML ensemble             │
├─────────────────────────────────────────────────────────────────┤
│                    Layer 8: In-Season Analytics                 │
│  trade_value · two_start · start_sit · matchup_planner          │
│  waiver_wire · trade_finder · draft_grader                      │
├─────────────────────────────────────────────────────────────────┤
│                    Layer 9: Feature Modules                     │
│  FP Parity (15 modules) + FP Edge Intelligence (3 modules)     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Layer 1: Entry Points

### `app.py` (~1800 lines)
The main draft tool. Three-phase UX flow:

1. **Splash screen** — Calls `bootstrap_all_data()` with a progress bar. Reconnects Yahoo OAuth token automatically via `_try_reconnect_yahoo()`.
2. **Setup wizard** — Two steps: Settings (league config, Yahoo connect) and Launch (confirmation).
3. **Draft page** — Three-column layout: hero pick card (left), recommendation alternatives (center), category/opponent intel tabs (right). Six tabs: Category Balance, Available Players, Draft Board, Draft Log, Opponent Intel, Pick Predictor. Draft controls (Practice Mode, Undo, Save, Reset) live in a context panel via `render_context_card()`.

Key session state: `draft_state` (DraftState), `valued_pool` (DataFrame), `sgp_calc` (SGPCalculator), `yahoo_client` (optional), `bootstrap_complete` (bool).

### Pages (`pages/`)

| Page | File | Purpose | Context Panel |
|------|------|---------|---------------|
| My Team | `1_My_Team.py` | Roster display, standings, Yahoo sync | Hitting/pitching totals, IL alerts |
| Draft Simulator | `2_Draft_Simulator.py` | Standalone mock draft with AI opponents | Draft settings, roster, recent picks |
| Trade Analyzer | `3_Trade_Analyzer.py` | Trade evaluation (Phase 1 engine → legacy fallback) | Trade verdict, punts, surplus SGP |
| Player Compare | `4_Player_Compare.py` | Head-to-head z-score comparison | Composite scores, quick verdict |
| Free Agents | `5_Free_Agents.py` | Marginal SGP FA rankings | Position filter pills |
| Lineup Optimizer | `6_Lineup_Optimizer.py` | 5-tab optimizer (Optimize, H2H, Streaming, Category, Roster) | Mode/alpha settings, H2H win prob |
| Closer Monitor | `7_Closer_Monitor.py` | 30-team closer depth chart grid | (uses existing card grid) |
| Standings | `8_Standings.py` | Projected standings + power rankings | Simulation controls |
| Leaders | `9_Leaders.py` | Category/points leaders, breakouts, prospects | Category/preset filters |
| Waiver Wire | `10_Waiver_Wire.py` | Add/drop recommendations via LP-verified SGP swaps | Roster summary, position filter |
| Start/Sit Advisor | `11_Start_Sit.py` | Weekly lineup recommendations with matchup analysis | Team info, matchup state |

### Hybrid 3-Zone Layout Pattern

All pages share a consistent ESPN/Yahoo-style layout powered by `src/ui_shared.py`:

```
┌──────────────────────────────────────────────────┐
│  Recommendation Banner (collapsible teaser)       │
├────────────┬─────────────────────────────────────┤
│  Context   │  Main Content                        │
│  Panel     │  (compact tables, charts, controls)  │
│  (~20%)    │  (~80%)                              │
│  ─ cards   │                                      │
│  ─ filters │                                      │
│  ─ alerts  │                                      │
└────────────┴─────────────────────────────────────┘
```

- **Banner**: `render_page_layout()` + `render_reco_banner()` — page title badge + one-line teaser, expandable for detail
- **Context panel**: `render_context_columns()` returns `(ctx, main)` → `render_context_card()` for compact glassmorphic cards
- **Main content**: Full data tables via `render_compact_table()` — ESPN-style 11px IBM Plex Mono, sticky player names, hit/pitch header colors, health dots, starter/bench row tinting
- **Sidebar**: Collapsed by default on all pages (`initial_sidebar_state="collapsed"`) — hamburger menu for navigation only

---

## Layer 2: Database

### `src/database.py`
SQLite database at `data/draft_tool.db` with 21 tables across 5 groups:

| Group | Tables | Purpose |
|-------|--------|---------|
| Draft | `projections`, `adp`, `league_config`, `draft_picks`, `blended_projections`, `player_pool` | Draft-time data |
| In-Season | `season_stats`, `ros_projections`, `league_rosters`, `league_standings`, `park_factors`, `refresh_log` | Live league data |
| Analytics | `injury_history`, `transactions` | Historical analytics |
| FP Parity | `player_tags`, `leagues` | Tags and multi-league |
| FP Edge | `prospect_rankings`, `ecr_consensus`, `player_id_map`, `player_news`, `ownership_trends` | Intelligence data |

**Key patterns:**
- `check_staleness(source, max_age_hours)` — Returns True when data needs refresh. Powers the bootstrap pipeline's selective refresh.
- `upsert_player_bulk(players)` — SELECT-first pattern (no UNIQUE on `name`). `injury_history` DOES have a UNIQUE index so it can use `ON CONFLICT`.
- `load_player_pool()` — The central query: JOINs `players` + `projections(system='blended')` + `adp`. Returns `name` column (pages rename to `player_name`).
- `create_blended_projections()` — Averages across projection systems, recomputing rate stats from components (never averaging rates directly).
- `get_connection()` — All callers MUST use `try/finally: conn.close()` to prevent leaks.

---

## Layer 3: Core Analytics

### `src/valuation.py` — SGP & VORP Engine
The mathematical foundation. `LeagueConfig` is the **single source of truth** for all category definitions.

- **SGP (Standings Gain Points):** Converts raw stats to standings movement. Auto-computes denominators from standings gaps or uses defaults.
- **VORP:** Value Over Replacement Player with multi-position premium (+0.12/extra position, +0.08/scarce position).
- **Replacement levels:** Per-position starter thresholds determine replacement-level SGP.
- **`value_all_players()`:** Adds `pick_score`, `vorp`, `tier` to each player.
- **Percentile projections:** Inter-system volatility → P10/P50/P90 projections. Process risk widens CI for low-correlation stats (AVG r²=0.41 vs HR r²=0.72). Inverse rate stats (ERA, WHIP) use negated z-multiplier.

### `src/simulation.py` — Monte Carlo Draft Simulation
- **`evaluate_candidates()`:** Runs N Monte Carlo simulations (100-300) over a 6-round horizon with opponent roster tracking.
- **Survival probability:** Normal CDF with σ=10.0 and positional scarcity adjustment.
- **Opponent modeling:** Additive blend of three independently normalized distributions: `0.5×ADP + 0.3×need + 0.2×history`. Each distribution sums to 1.0 before blending.
- **Urgency:** `(1 - P_survive) × positional_dropoff`, draft-position-aware.
- **Combined score:** `MC_mean_SGP + urgency × 0.4`.
- **Tier assignment:** Natural breaks algorithm.

### `src/draft_state.py` — Draft State Machine
Snake draft state management: roster tracking per team, pick order computation, positional need detection. `from_yahoo_draft()` classmethod enables replaying Yahoo draft results.

### `src/injury_model.py` — Health & Injury Modeling
- `health_score = avg(GP/games_available)` over 3 seasons (default 0.85 for missing data).
- Age-risk curves: hitters +2%/yr after 30, pitchers +3%/yr after 28.
- Workload flag: >40 IP increase year-over-year.
- `get_injury_badge()` returns HTML `<span>` with CSS dot (green/yellow/red) — no emoji.

### `src/bayesian.py` — Bayesian Projection Updates
PyMC 5 hierarchical beta-binomial model for in-season stat updates. Marcel regression fallback when PyMC unavailable. FanGraphs stabilization thresholds (K=60 PA, AVG=910 AB, ERA=70 IP). Aging curves on logit scale. Stabilization points for rate stats: R=320 PA, RBI=300 PA, SB=100 PA.

### `src/ui_shared.py` — Design System & Layout Engine
Single light-mode palette: `THEME` dict with bg=#f4f5f0, primary=#e63946, hot=#ff6d00, gold=#ffd60a, green=#2d6a4f, sky=#457b9d, purple=#6c63ff. `T = THEME` is a direct dict alias. `PAGE_ICONS` contains ~22 inline SVG icons. `inject_custom_css()` injects 1700+ lines of glassmorphism, 3D buttons, kinetic typography, 7 animations, orange sidebar, bold titles, plus the 3-zone layout classes.

**3-Zone Layout Functions:**
- `build_compact_table_html(df, highlight_cols, row_classes, health_col, max_height)` — Pure function returning ESPN-style HTML table string. Auto-detects hitting vs pitching columns for `.th-hit`/`.th-pit` header classes. First column gets `.col-name` (sticky on horizontal scroll). Unit-testable (39 tests in `test_compact_table.py`).
- `render_compact_table()` — Thin Streamlit wrapper calling `build_compact_table_html()` → `st.markdown()`.
- `render_reco_banner(teaser_text, expanded_html, icon_key)` — Collapsible recommendation banner. Static when no detail; uses `st.expander` when expandable.
- `render_context_card(title, content_html)` — Glassmorphic card for the context panel. Bebas Neue 10px uppercase title.
- `render_page_layout(title, banner_teaser, banner_detail, banner_icon)` — Page title badge + optional banner. Call at top of every page.
- `render_context_columns(context_width=1)` — Returns `(ctx, main)` from `st.columns([1, 4])`.

**CSS Classes:** `.reco-banner`, `.reco-banner-teaser`, `.reco-banner-detail`, `.context-card`, `.context-card-title`, `.compact-table-wrap`, `.compact-table`, `.th-hit` (orange border), `.th-pit` (blue border), `.col-name` (sticky left), `.row-start` (green tint), `.row-bench` (red tint), `.health-dot` (6px circle). Responsive at 768px (compact font) and 480px (icon-free banner). Print CSS hides context panel.

---

## Layer 4: Data Ingestion Pipeline

### `src/player_card.py` — Player Card Data Assembly
Pure function `build_player_card_data(player_id)` assembles all data for the interactive player card dialog: profile (name, team, positions, age, headshot URL), 3-year historical stats, 6-system projections, radar percentiles (vs league and MLB averages), injury history, ADP/ECR rankings, deduplicated news with full datetime, and prospect scouting grades. No Streamlit dependency — fully unit-testable (76 tests). The rendering counterpart `show_player_card_dialog()` in `ui_shared.py` uses `@st.dialog` to display the card as a modal overlay. Accessible from all pages via `render_player_select()`.

### `src/data_bootstrap.py` — Bootstrap Orchestrator
Zero-interaction bootstrap runs on every app launch. 9 phases with staleness-based refresh:

| Phase | Source | Staleness | Target Table |
|-------|--------|-----------|-------------|
| 1. Players | MLB Stats API | 7 days | `players` |
| 2. Park Factors | Hardcoded FG 2024 | 30 days | `park_factors` |
| 3. Projections | FanGraphs JSON API (7 systems) | 7 days | `projections` |
| 4. Live Stats | MLB Stats API | 1 hour | `season_stats` |
| 5. Historical | MLB Stats API (2023-2025) | 30 days | `season_stats` |
| 6. Injury Data | Derived from historical | 30 days | `injury_history` |
| 7. Yahoo Sync | Yahoo Fantasy API | 6 hours | `league_rosters`, `league_standings` |
| 8. News | 4-source aggregation | 1 hour | `player_news` |
| 9. ECR | Multi-platform scraping | 24 hours | `ecr_consensus` |

Players phase runs first — all others need player_ids.

### `src/data_pipeline.py` — FanGraphs Auto-Fetch
Fetches Steamer, ZiPS, Depth Charts, ATC, THE BAT, THE BAT X from FanGraphs JSON API. 0.5s rate limiting between requests. `SYSTEM_MAP` dict translates FG names to DB names (e.g., `"fangraphsdc"` → `"depthcharts"`).

### `src/live_stats.py` — MLB Stats API Integration
`fetch_all_mlb_players()` returns 750+ active players with mlb_id, name, team, positions. `fetch_historical_stats()` pulls 3 years for injury modeling. `match_player_id()` resolves external names to internal player_ids with fuzzy matching.

### `src/yahoo_api.py` — Yahoo Fantasy API
OAuth 2.0 via yfpy with out-of-band (oob) flow only — Yahoo rejects all redirect URIs for fantasy apps. Auto-reconnect loads token from `data/yahoo_token.json` on every restart. Token refresh is automatic (yahoo-oauth checks validity at 59-min mark).

### Other Ingest Modules
- `src/adp_sources.py` — FantasyPros ECR scraper + NFBC ADP scraper
- `src/depth_charts.py` — FanGraphs depth chart scraper, role classification (starter/platoon/closer/setup/committee)
- `src/contract_data.py` — Baseball Reference free agent list scraper
- `src/news_fetcher.py` — MLB Stats API transaction fetcher
- `src/marcel.py` — Marcel projection system: 3yr weighted avg (5/4/3) with regression to mean + age adjustment

---

## Layer 5: Trade Engine (`src/engine/`)

6-phase pipeline across 20 modules in 8 subpackages:

```
Phase 1 (Deterministic)          Phase 2 (Stochastic)
 portfolio/                       projections/
   valuation.py                     bayesian_blend.py (BMA)
   category_analysis.py             marginals.py (KDE)
   lineup_optimizer.py            monte_carlo/
   copula.py                        trade_simulator.py (10K paired MC)

Phase 3 (Signals)                Phase 4 (Context)
 signals/                         context/
   statcast.py (Baseball Savant)    matchup.py (Log5)
   decay.py (exponential)           injury_process.py (Weibull)
   kalman.py (state-space)          bench_value.py
   regime.py (BOCPD + HMM)         concentration.py (HHI)

Phase 5 (Game Theory)            Phase 6 (Production)
 game_theory/                     production/
   opponent_valuation.py (Vickrey)  convergence.py (ESS, split-R̂)
   adverse_selection.py (Bayes)     cache.py (TTL)
   dynamic_programming.py (Bellman) sim_config.py (adaptive 1K-100K)
   sensitivity.py

                 output/
                   trade_evaluator.py (master orchestrator)
```

### Phase 1: Deterministic SGP Evaluation
1. Load standings → `build_standings_totals()`
2. `category_gap_analysis()` → punt detection → category weights
3. LP-constrained lineup totals (PuLP) — only starters' stats, bench excluded
4. Roster cap enforcement: 2-for-1 → drop lowest-SGP bench; 1-for-2 → add FA capped at median SGP
5. `Σ (after-before)/sgp_denom × marginal_weight` → `surplus_sgp`
6. `_compute_replacement_penalty()` → counting stat FA replacement cost
7. `grade_trade(surplus_sgp)` → A+ through F

### Phase 2: Stochastic MC Overlay (`enable_mc=True`)
- **BMA:** Posterior-weighted projection blending across systems. Tighter σ → higher weight.
- **KDE marginals:** Non-Gaussian distributions via kernel density (≥20 samples, Normal fallback).
- **Gaussian copula:** Correlated sampling via Cholesky decomposition of 12×12 matrix (HR↔RBI: 0.85, ERA↔WHIP: 0.90).
- **Paired MC (10K):** Identical per-sim seeds for before/after → variance reduction. Produces VaR, CVaR, Sharpe, confidence intervals.

### Phase 3: Signal Intelligence
- **Statcast:** Pitch-level data → rolling 14-day features (EV, barrel%, xwOBA, whiff%).
- **Exponential decay:** Signal-specific half-lives (batted ball: 35d, spin: 46d, discipline: 58d, traditional: 87d).
- **Kalman filter:** True talent estimation with sample-size-aware observation variance.
- **Regime detection:** BOCPD changepoint detection + 4-state HMM (Elite/Above/Below/Replacement).

### Phase 4: Context Engine (`enable_context=True`)
- **Log5 matchup:** Odds-ratio batter-pitcher matchup with park/weather adjustments.
- **Injury process:** Weibull-distributed durations by body part, frailty multipliers.
- **HHI concentration:** `Σ(share²)`, penalty when > 0.15 threshold.

### Phase 5: Game Theory (`enable_game_theory=True`)
- **Vickrey auction:** Market price = second-highest bidder (Nash equilibrium).
- **Adverse selection:** Bayesian P(flaw|offered) discount, max 25% haircut.
- **Bellman rollout:** Future trade option value with playoff-aware discounting (γ=0.85-0.98).
- **Sensitivity:** Category/player impact ranking, breakeven gap, counter-offer generation.

### Phase 6: Production
- **Convergence:** ESS via FFT, split-R̂, running mean stability.
- **Cache:** In-memory with TTL (copula 24h, SGP 1h, market values 30min).
- **Adaptive scaling:** 1K (quick) → 10K (standard) → 50K (production) → 100K (full).

---

## Layer 6: Enhanced Lineup Optimizer (`src/optimizer/`)

11-module pipeline with 20 mathematical techniques, orchestrated by `LineupOptimizerPipeline`:

```
                    pipeline.py (orchestrator)
                         │
    ┌────────────────────┼────────────────────┐
    │                    │                    │
projections.py    matchup_adjustments.py   h2h_engine.py
(Bayesian/Kalman/    (park/platoon/         (Normal PDF
 regime/injury)       weather)               weights)
    │                    │                    │
    └────────────────────┼────────────────────┘
                         │
    ┌────────────────────┼────────────────────┐
    │                    │                    │
sgp_theory.py     streaming.py          scenario_generator.py
(non-linear SGP,   (pitcher value,        (Gaussian copula
 SLOPE regression)  Bayesian scoring)      scenarios, CVaR)
    │                    │                    │
    └────────────────────┼────────────────────┘
                         │
    ┌────────────────────┼────────────────────┐
    │                    │                    │
multi_period.py   dual_objective.py      advanced_lp.py
(rolling horizon,  (H2H/Roto blend,       (maximin, epsilon-
 urgency weights)   auto-alpha)            constraint, MIP)
```

Three modes: Quick (<1s), Standard (2-3s), Full (5-10s) — controlled by `MODE_PRESETS`.

### Key Techniques
- **H2H weights:** Normal PDF peaks where category gap = 0 (close races matter most).
- **Non-linear SGP:** Bell-curve proximity to standings positions, not linear slope.
- **Streaming:** `stream_score = E[K]/sgp_K + W_prob×0.5/sgp_W - E[ER]×risk_penalty/sgp_ERA`. Matchup grades A+ through C.
- **Maximin LP:** Maximizes worst non-inverse category (inverse cats excluded — incompatible scale factors).
- **Stochastic MIP:** CVaR-constrained optimization over 200 correlated scenarios.

---

## Layer 7: Draft Recommendation Engine (`src/draft_engine.py`)

`DraftRecommendationEngine` orchestrates an 8-stage enhancement pipeline:

```
Base SGP → park_factor → bayesian_blend → injury_prob → statcast_delta
    → fip_correction → contextual_factors → category_balance → ml_ensemble
```

**Enhanced pick score formula:**
```
enhanced_pick_score = base_sgp × multiplicative_factors + additive_bonuses

Multiplicative (clamped [0.5, 1.5]):
  category_balance × park_factor × (1 - injury_prob × 0.3)
  × (1 + statcast_delta × 0.15) × platoon_factor × contract_year

Additive:
  streaming_penalty + lineup_protection + closer_bonus + ml_correction × 0.1 + flex_bonus
```

**Supporting modules:**
- `src/draft_analytics.py` — Category balance (Normal PDF weighting), BUY/FAIR/AVOID classification
- `src/contextual_factors.py` — Closer hierarchy, platoon risk (The Book), lineup protection, schedule strength
- `src/ml_ensemble.py` — Optional XGBoost residual prediction (10% weight)
- `src/news_sentiment.py` — Keyword-based scoring (-1.0 to +1.0)

Three execution modes (Quick/Standard/Full) toggle stages on/off for performance.

---

## Layer 8: In-Season Analytics

Seven modules for in-season decision support:

| Module | Algorithm | Key Formula |
|--------|-----------|-------------|
| `trade_value.py` | G-Score adjustment (arXiv:2307.02188) | `trade_value = 100 × (sgp_surplus / max_surplus)` |
| `two_start.py` | Pitcher matchup scoring + rate damage | `matchup_score = f(K-BB%, xFIP, CSW%, opp_stats, park)` |
| `start_sit.py` | 3-layer decision model | `risk_adjusted = f(EV, P10, P90, matchup_state)` |
| `matchup_planner.py` | Log5 + percentile tiers | 5 tiers: smash/favorable/neutral/unfavorable/avoid |
| `waiver_wire.py` | 7-stage LP-verified add/drop | BABIP sustainability filter + category priority |
| `trade_finder.py` | Cosine dissimilarity (arXiv:2111.02859) | Loss aversion: `LOSS_AVERSION = 1.5` (Kahneman) |
| `draft_grader.py` | 3-component grading | 40% team value + 35% pick efficiency + 25% category balance |

---

## Layer 9: Feature Modules

### FantasyPros Parity (15 modules)

| Module | Feature |
|--------|---------|
| `draft_order.py` | Randomized draft order with seed reproducibility |
| `player_tags.py` | Sleeper/Target/Avoid/Breakout/Bust CRUD with SQLite persistence |
| `pick_predictor.py` | Normal CDF + Weibull survival blend by position |
| `closer_monitor.py` | 30-team job security grid |
| `points_league.py` | Yahoo/ESPN/CBS scoring presets + points VORP |
| `leaders.py` | Category/points leaders with breakout detection (z > 1.5) |
| `cheat_sheet.py` | HTML/PDF export with HEATER theme |
| `live_draft_sync.py` | Yahoo draft polling (8s interval) with YOUR TURN detection |
| `schedule_grid.py` | 7-day calendar with matchup color-coding |
| `start_sit_widget.py` | Quick 2-4 player compare with density overlap |
| `standings_projection.py` | Copula-based MC season simulation |
| `power_rankings.py` | 5-factor composite (roster/balance/SOS/injury/momentum) with bootstrap CI |
| `il_manager.py` | IL detection, duration estimation, replacement selection |
| `league_registry.py` | Multi-league support with context switching |

### FP Edge Intelligence (3 modules)

| Module | Feature | Key Innovation |
|--------|---------|----------------|
| `prospect_engine.py` | Prospect Rankings Engine | FG Board API + MiLB stats → MLB Readiness Score (0-100) |
| `ecr.py` | Multi-Platform ECR Consensus | 7 ranking sources + Trimmed Borda Count algorithm |
| `player_news.py` | News/Transaction Intelligence | 4-source aggregation + template-based analytical summaries |

---

## Data Flow Diagrams

### Bootstrap → Draft Flow

```
App Launch
  ↓
_try_reconnect_yahoo()        [restore token from data/yahoo_token.json]
  ↓
bootstrap_all_data()           [src/data_bootstrap.py — 9 phases]
  ├── _bootstrap_players()     → MLB Stats API → players table
  ├── _bootstrap_park_factors()→ hardcoded dict → park_factors table
  ├── _bootstrap_projections() → FanGraphs JSON API → projections table
  ├── _bootstrap_live_stats()  → MLB Stats API → season_stats table
  ├── _bootstrap_historical()  → MLB Stats API (3yr) → season_stats table
  ├── _bootstrap_injury_data() → derived from historical → injury_history table
  ├── _bootstrap_yahoo_sync()  → Yahoo API → league_rosters, league_standings
  ├── _bootstrap_news()        → 4 sources → player_news table
  └── _bootstrap_ecr()         → multi-platform → ecr_consensus table
  ↓
create_blended_projections()   [average across systems, recompute rate stats]
  ↓
load_player_pool()             [JOIN players + blended projections + adp]
  ↓
compute_sgp_denominators()     [standings-based or defaults]
  ↓
compute_replacement_levels()   [per-position starter thresholds]
  ↓
value_all_players()            [adds pick_score, vorp, tier]
  ↓
DraftRecommendationEngine      [8-stage enhancement → enhanced_pick_score]
  ↓
DraftSimulator                 [N MC sims → combined_score ranking]
  ↓
UI: hero card + alternatives + opponent intel
```

### Trade Evaluation Flow (Phases 1-5)

```
User inputs giving_ids, receiving_ids
  ↓
evaluate_trade()               [src/engine/output/trade_evaluator.py]
  │
  ├── Phase 1 (Deterministic):
  │   ├── load_league_standings() → build_standings_totals()
  │   ├── category_gap_analysis() → punt detection → category weights
  │   ├── _lineup_constrained_totals(before) → LP solve → starters-only totals
  │   ├── _lineup_constrained_totals(after)  → LP solve → starters-only totals
  │   ├── Roster cap: 2-for-1 → drop; 1-for-2 → pickup
  │   ├── Σ (after-before)/sgp_denom × marginal_weight → surplus_sgp
  │   ├── _compute_replacement_penalty() → FA replacement cost
  │   └── grade_trade(surplus_sgp) → A+ through F
  │
  ├── Phase 2 (enable_mc=True):
  │   ├── bayesian_model_average() → posterior-weighted projections
  │   ├── build_player_marginals() → KDE or Gaussian per stat
  │   ├── GaussianCopula.sample() → correlated uniform variates
  │   └── run_paired_monte_carlo(10K) → distributional metrics overlay
  │
  ├── Phase 4 (enable_context=True):
  │   ├── compute_concentration_delta() → HHI penalty
  │   └── enhanced_bench_option_value()
  │
  └── Phase 5 (enable_game_theory=True):
      ├── compute_discount_for_trade() → adverse selection
      ├── player_market_value() → Vickrey auction
      └── trade_sensitivity_report() → counter-offers
```

---

## Mathematical Models Inventory

| Module | Model | Reference |
|--------|-------|-----------|
| `valuation.py` | SGP (Standings Gain Points) | Rate stat weighted-average formula |
| `valuation.py` | VORP + multi-position premium | +0.12/extra, +0.08/scarce |
| `simulation.py` | Survival probability (Normal CDF) | σ=10.0, positional scarcity |
| `bayesian.py` | Marcel regression + PyMC hierarchical | FanGraphs stabilization thresholds |
| `injury_model.py` | Health score + age-risk curves | GP/GA 3-year average, logit aging |
| `engine/portfolio/copula.py` | Gaussian copula | Cholesky decomposition, 12×12 matrix |
| `engine/projections/bayesian_blend.py` | Bayesian Model Averaging | Posterior weights via P(YTD\|System) |
| `engine/projections/marginals.py` | KDE marginals | Gaussian KDE, ≥20 samples required |
| `engine/signals/kalman.py` | Kalman filter | Sample-size-aware observation variance |
| `engine/signals/regime.py` | BOCPD + 4-state HMM | Adams & MacKay 2007 |
| `engine/signals/decay.py` | Exponential decay | e^(-λ×days), signal-specific λ |
| `engine/context/matchup.py` | Log5 odds ratio | Tango (The Book) |
| `engine/context/injury_process.py` | Weibull injury duration | Body-part-specific shape/scale |
| `engine/context/concentration.py` | HHI concentration | Σ(share²), threshold=0.15 |
| `engine/game_theory/adverse_selection.py` | Bayes' theorem | Akerlof lemons model |
| `engine/game_theory/dynamic_programming.py` | Bellman rollout | γ=0.85-0.98 by playoff prob |
| `engine/game_theory/opponent_valuation.py` | Vickrey auction | 2nd-highest bid = Nash equilibrium |
| `engine/monte_carlo/trade_simulator.py` | Paired MC (10K sims) | Variance reduction via identical seeds |
| `engine/production/convergence.py` | ESS (FFT), split-R̂ | Gelman-Rubin, Geyer's sequence |
| `optimizer/h2h_engine.py` | Normal PDF weighting | φ(gap/σ)/σ peaks at tie |
| `optimizer/sgp_theory.py` | Non-linear SGP | SLOPE regression denominators |
| `optimizer/advanced_lp.py` | Maximin + epsilon-constraint + MIP | CVaR (Rockafellar-Uryasev) |
| `optimizer/scenario_generator.py` | Copula scenarios + CVaR | 200-scenario risk framework |
| `trade_value.py` | G-Score adjustment | arXiv:2307.02188 |
| `trade_finder.py` | Cosine dissimilarity | arXiv:2111.02859 |
| `pick_predictor.py` | Weibull survival blend | Position-dependent shape params |
| `prospect_engine.py` | MLB Readiness Score | wOBA proxy: OBP×0.92 + SLG×0.47 - 0.08 |
| `ecr.py` | Trimmed Borda Count | Drop min/max when ≥4 sources |

---

## External API Integrations

All external dependencies use graceful degradation — `try/except ImportError` with `MODULE_AVAILABLE` flags.

| API | Package | Fallback Behavior |
|-----|---------|-------------------|
| MLB Stats API | `statsapi` | Empty DataFrames |
| FanGraphs JSON API | `requests` | Skip projection phase, use blended |
| Statcast (Baseball Savant) | `pybaseball` | `PYBASEBALL_AVAILABLE=False` → 0.0 delta |
| Yahoo Fantasy API | `yfpy` | `YFPY_AVAILABLE=False` → full app works |
| FantasyPros ECR | `requests` + `bs4` | Empty DataFrame |
| NFBC ADP | `requests` + `bs4` | Empty DataFrame |
| Baseball Reference | `requests` + `bs4` | Empty set for contract years |
| RotoWire RSS | `feedparser` | `FEEDPARSER_AVAILABLE=False` → skip |
| ESPN News | `requests` | Silently skip source |
| PyMC | `pymc` (optional) | Marcel regression fallback |
| XGBoost | `xgboost` (optional) | Zero corrections (ml_correction=0.0) |
| PuLP | `pulp` | Fallback to `_roster_category_totals()` |
| hmmlearn | `hmmlearn` (optional) | `classify_regime_simple()` rule-based |

---

## CI/CD Pipeline

### `.github/workflows/ci.yml`
- **Lint job:** `ruff check` + `ruff format --check` (Python 3.12)
- **Test job:** pytest with coverage (Python 3.11, 3.12, 3.13 matrix), `--cov-fail-under=60`
- **Build job:** Verifies all 30+ modules import cleanly, generates sample data, verifies DB

### `.github/workflows/refresh.yml`
Daily cron at 9:17 UTC + manual trigger for data refresh.

### Local Development
- Python 3.14 (Windows). CI tests 3.11-3.13.
- `python -m ruff` on Windows (bare `ruff` may not be on PATH).

---

## Test Architecture

83 test files, 1956+ passing tests, 4 skipped (PyMC/XGBoost optional deps).

| Category | Files | Tests |
|----------|-------|-------|
| Core math verification | 4 | 168 |
| Database | 3 | ~50 |
| Data pipeline/bootstrap | 3 | 85 |
| Trade engine (Phase 1-6) | 7 | 283 |
| Lineup optimizer | 10 | 204 |
| Draft recommendation engine | 5 | 270 |
| In-season analytics | 7 | 276 |
| FantasyPros parity | 16 | 184 |
| FP Edge intelligence | 4 | ~69 |
| Live stats/Yahoo/injury | 4 | ~130 |
| UI layout (compact table) | 1 | 44 |
| Player card data assembly | 1 | 76 |
| Integration/misc | ~17 | ~180 |

Eight rounds of systematic code reviews have fixed 207 bugs total, plus a data pipeline audit (32 issues fixed). Math verification tests validate hand-calculated expected values against code formulas. The compact table tests validate `build_compact_table_html()` output (pure function, no Streamlit runtime required).

---

## Architecture Invariants

These are the non-negotiable rules that all code must follow:

1. **`LeagueConfig` is the single source of truth** for all category definitions. Never hardcode category lists elsewhere.

2. **Rate stats computed from components**, never averaged: `AVG=H/AB`, `OBP=(H+BB+HBP)/(AB+BB+HBP+SF)`, `ERA=ER×9/IP`, `WHIP=(BB+H)/IP`.

3. **`load_player_pool()` returns `name` column.** Pages must rename to `player_name` for display.

4. **Connection leak prevention:** Always `try/finally: conn.close()`.

5. **No emoji anywhere.** All icons are inline SVGs from `PAGE_ICONS`. Health badges use CSS dots.

6. **LP constraints are `≤ 1`, not `== 1`** — allows empty slots when filling hurts the objective.

7. **Paired MC seed discipline:** Before and after rosters use identical per-sim seeds.

8. **Inverse stat handling varies by context:**
   - SGP: sign-flipped
   - LP maximin: excluded (incompatible scale factors)
   - Percentiles: negated z-multiplier
   - Standings: `P(A wins) = Φ((μ_B - μ_A) / √(σ²_A + σ²_B))`

9. **`check_staleness(source, 0)` always returns True** (force refresh).

10. **`T["amber"]` = `"#e63946"`, `T["teal"]` = `"#457b9d"`** — backward compat aliases on the flat THEME dict.

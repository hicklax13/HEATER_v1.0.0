# HEATER — Fantasy Baseball Draft Tool

## Overview

A fantasy baseball draft assistant + in-season manager for a 12-team Yahoo Sports 5x5 roto snake draft league. Two pillars:

1. **Draft Tool** (`app.py`, ~1800 lines) — "Heater" themed Streamlit app with light-mode-only glassmorphic design system: splash screen data bootstrap + 2-step setup wizard (Settings, Launch), 3-column draft page with SVG injury badges, percentile ranges, opponent intel tab, and practice mode. Monte Carlo recommendations with percentile sampling. Zero CSV uploads — all data auto-fetched from MLB Stats API + FanGraphs on every launch. Pill button navigation replaces dropdowns. Search + card grids for player selection.
2. **In-Season Management** (`pages/`) — 6 Streamlit pages: team overview, draft simulator, trade analysis, player comparison, free agent rankings, lineup optimizer. Powered by MLB Stats API + pybaseball + optional Yahoo Fantasy API. All pages share centralized single-theme system (light mode only) with glassmorphic design, orange sidebar branding, and bold Heater identity.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Load sample data (first time / testing)
python load_sample_data.py

# Run the app
streamlit run app.py

# Lint
ruff check .

# Format
ruff format .

# Run all tests (587 collected, 586 pass, 1 skipped for PyMC)
python -m pytest

# Run with verbose output
python -m pytest -v

# Run a single test file
python -m pytest tests/test_in_season.py -v

# Run a specific test
python -m pytest tests/test_in_season.py::test_trade_analyzer -v
```

## League Context

- **League:** FourzynBurn (Yahoo Sports) | **Team:** Team Hickey
- **Format:** 12-team snake draft, 23 rounds
- **Scoring:** 5x5 roto (R, HR, RBI, SB, AVG / W, SV, K, ERA, WHIP)
- **Roster:** C/1B/2B/3B/SS/3OF/2Util/2SP/2RP/4P/5BN = 23 slots
- **Manager skill:** Opponents extremely high-skilled; user is a novice

## Tech Stack

- **Framework:** Streamlit (Python), multi-page app
- **Database:** SQLite (`data/draft_tool.db`) — 14 tables total
- **Core libs:** pandas, NumPy, SciPy, Plotly
- **Analytics:** PyMC 5 (Bayesian), PuLP (LP optimizer), arviz (posterior analysis)
- **Live data:** MLB-StatsAPI, pybaseball (FanGraphs Depth Charts + park factors)
- **Yahoo API:** yfpy + streamlit-oauth (optional OAuth integration)
- **Linter:** ruff (lint + format)
- **CI:** GitHub Actions — ruff lint/format, pytest (Python 3.11-3.13), build check
- **Python:** Local dev uses 3.14; CI tests 3.11-3.13

## File Structure

```
app.py                  — Draft tool: splash screen bootstrap + 2-step setup wizard + 3-column draft page
requirements.txt        — pip dependencies (streamlit, pandas, numpy, scipy, plotly, MLB-StatsAPI, pybaseball, pymc, PuLP, yfpy)
load_sample_data.py     — Generates ~190 sample players + injury history for testing
.streamlit/config.toml  — Light theme configuration (Heater palette)
.claude/launch.json     — Dev server config for preview tools
pages/
  1_My_Team.py          — In-season: team overview with monogram avatar, roster, category standings, Yahoo sync
  2_Draft_Simulator.py  — Standalone draft simulator (AI opponents, MC recommendations, pill filters, card-based picks)
  3_Trade_Analyzer.py   — In-season: trade proposal builder + Phase 1 SGP engine (grade A+ to F, punt detection, marginal elasticity) with legacy fallback
  4_Player_Compare.py   — In-season: head-to-head player comparison with dual search + card pickers
  5_Free_Agents.py      — In-season: free agent rankings by marginal value, position pill filters
  6_Lineup_Optimizer.py — In-season: PuLP LP solver for start/sit + category targeting
src/
  database.py           — SQLite schema (14 tables), CSV import, projection blending, player pool + in-season queries
  valuation.py          — SGP calculator, replacement levels, VORP, category weights, percentile forecasts
  draft_state.py        — Draft state management, roster tracking, snake pick order, opponent patterns
  simulation.py         — Monte Carlo draft simulation, opponent modeling (history-aware), survival probability
  in_season.py          — Trade analyzer, player comparison engine, FA ranker
  live_stats.py         — MLB Stats API data fetcher: roster, season/historical stats, injury data extraction
  data_bootstrap.py     — Zero-interaction bootstrap orchestrator: staleness-based refresh of all data sources on app launch
  league_manager.py     — League roster/standings management, CSV import for all 12 teams
  ui_shared.py          — Heater design system: single THEME dict, PAGE_ICONS (inline SVGs), glassmorphic CSS injection (~1500 lines), metric tooltips, sidebar branding
  data_2026.py          — Hardcoded 2026 projections (~200 hitters, ~80 pitchers) for sample data
  validation.py         — Validation utilities
  bayesian.py           — Bayesian projection updater: PyMC hierarchical model + Marcel regression fallback
  injury_model.py       — Health scores, age-risk curves, workload flags, injury-adjusted projections
  lineup_optimizer.py   — PuLP LP solver: lineup optimization, category targeting, two-start SP detection
  yahoo_api.py          — Yahoo Fantasy API: OAuth integration via yfpy, league sync, roster import
  data_pipeline.py      — FanGraphs auto-fetch: Steamer/ZiPS/Depth Charts JSON API, normalize, upsert, ADP extraction
  engine/               — Trade Analyzer Engine (Phase 1-4)
    __init__.py
    portfolio/
      __init__.py
      valuation.py      — Z-score + SGP valuation: peer-group z-scores, standings-based SGP denominators, VORP
      category_analysis.py — Marginal SGP elasticity (1/gap), category gap analysis, punt detection, category weights
      lineup_optimizer.py — LP optimizer wrapper: optimal lineup value, pre/post trade lineup delta, bench option value
      copula.py         — Gaussian copula: correlated stat sampling, empirical 10×10 correlation matrix, nearest-PD correction
    projections/
      __init__.py
      projection_client.py — ROS projections loader, 3-pass fuzzy name matching, trade player resolution
      bayesian_blend.py — BMA: posterior-weighted projection blending, system forecast sigmas, total variance decomposition
      marginals.py      — KDE marginals: kernel density estimation with Normal fallback, stat bounds clipping, ppf for copula
    monte_carlo/
      __init__.py
      trade_simulator.py — Paired MC: 10K sims, variance reduction via identical seeds, VaR/CVaR/Sharpe/CI metrics
    signals/
      __init__.py
      statcast.py       — Statcast harvesting: pitch-level data → rolling features (EV, barrel%, xwOBA, whiff%)
      decay.py          — Exponential decay weighting: signal-specific half-lives, recency-weighted mean/variance
      kalman.py         — Kalman filter: true talent estimation, sample-size-aware observation variance
      regime.py         — Regime detection: BOCPD changepoint detection (run length mode) + 4-state HMM
    context/
      __init__.py
      matchup.py        — Log5 matchup engine: batter-pitcher odds ratio, park/weather adjustments, game-level projections
      injury_process.py — Injury stochastic process: Weibull duration sampling, frailty multipliers, season availability MC
      bench_value.py    — Enhanced bench option value: streaming + hot FA + flexibility premium + injury replacement cushion
      concentration.py  — Roster concentration risk: HHI scoring, diversification delta, team exposure breakdown
    game_theory/
      __init__.py
      opponent_valuation.py — Opponent willingness-to-pay + Nash equilibrium market clearing price (L8A)
      adverse_selection.py — Bayesian adverse selection discount from manager trade history (L8B)
      dynamic_programming.py — Bellman rollout for future trade option value, playoff-aware discounting (L9)
      sensitivity.py     — Category/player sensitivity analysis, breakeven, counter-offer generation (L11)
    production/
      __init__.py
      convergence.py     — MC convergence diagnostics: ESS via FFT, split-R̂, running mean stability
      cache.py           — Precomputation cache: TTL staleness, get_or_compute pattern, singleton
      sim_config.py      — Adaptive simulation scaling: 1K-100K by trade complexity + time budget
    output/
      __init__.py
      trade_evaluator.py — Master trade orchestrator: Phase 1-5 + enable_mc/enable_context/enable_game_theory flags
tests/
  test_database_schema.py   — DB schema and table existence tests
  test_database_queries.py  — Query function tests
  test_in_season.py         — Trade analyzer, comparison, FA ranker tests
  test_league_manager.py    — League roster/standings management tests
  test_live_stats.py        — Live stats pipeline tests
  test_data_bootstrap.py    — Bootstrap pipeline: staleness, bulk upserts, orchestrator (30 tests)
  test_bayesian.py          — Bayesian updater: regression, age curves, stabilization, batch update (31 tests)
  test_injury_model.py      — Injury model: health scores, age risk, workload flags (17 tests)
  test_lineup_optimizer.py  — Lineup optimizer: LP solver, constraints, category targeting (20 tests)
  test_yahoo_api.py         — Yahoo API: OAuth, oob flow, sync, mock endpoints (40 tests)
  test_percentiles.py       — Percentile forecasts: volatility, P10/P50/P90 bounds (7 tests)
  test_opponent_model.py    — Enhanced opponent modeling: preferences, needs, history (8 tests)
  test_percentile_sampling.py — Percentile sampling passthrough in evaluate_candidates (4 tests)
  test_data_pipeline.py     — FanGraphs auto-fetch: normalization, fetch, storage, ADP, orchestration (28 tests)
  test_integration.py       — End-to-end pipeline: injury → Bayesian → percentiles → valuation (11 tests)
  test_valuation_math.py    — Math verification: SGP, VORP, replacement levels, percentiles, process risk (40 tests)
  test_simulation_math.py   — Math verification: survival probability, urgency, combined score, tiers, MC convergence (37 tests)
  test_trade_math.py        — Math verification: trade SGP delta, MC noise, verdict, z-scores, rate stats (35 tests)
  test_trade_engine_math.py — Math verification: all 6 phases hand-calculated — SGP, BMA, copula, decay, Kalman, HHI, Bayes, Vickrey, Bellman, ESS, R̂ (50 tests)
  test_trade_engine.py      — Trade engine Phase 1: marginal SGP, punt detection, z-scores, grading, fuzzy match, integration (32 tests)
  test_trade_engine_phase2.py — Trade engine Phase 2: BMA, KDE marginals, copula, paired MC, integration (33 tests)
  test_trade_engine_phase3.py — Trade engine Phase 3: Statcast aggregation, signal decay, Kalman filter, BOCPD, HMM regime, rolling features (32 tests)
  test_trade_engine_phase4.py — Trade engine Phase 4: Log5 matchup, Weibull injury, enhanced bench, HHI concentration, context integration (40 tests)
  test_trade_engine_phase5.py — Trade engine Phase 5: opponent valuations, adverse selection, Bellman rollout, sensitivity, counter-offers (38 tests)
  test_trade_engine_phase6.py — Trade engine Phase 6: ESS convergence, split-R̂, cache TTL, adaptive sim scaling (32 tests)
  profile_latency.py        — Performance profiling utility
data/
  draft_tool.db         — SQLite database (created at runtime)
  backups/              — Draft state JSON backups
docs/plans/             — Implementation plan archives
.github/
  workflows/ci.yml      — CI pipeline (lint + test + build)
  dependabot.yml        — Weekly pip + GitHub Actions updates
```

## Architecture

### Draft Valuation Pipeline
1. **Projection blending** — Weighted average of multiple systems (Steamer, ZiPS, etc.)
2. **SGP** — Converts raw stats to standings-point movement; auto-computed denominators
3. **VORP** — Value Over Replacement Player with multi-position flexibility premium (+0.12/extra position, +0.08/scarce position)
4. **Category weights** — Dynamic: standings-based when available, draft-progress-aware scaling
5. **pick_score** = weighted SGP + positional scarcity + VORP bonus

### Draft Engine
- **Monte Carlo simulation** (100-300 sims, 6-round horizon) with opponent roster tracking
- **Survival probability** — Normal CDF with positional scarcity adjustment
- **Urgency** = (1 - P_survive) * positional_dropoff (draft-position-aware)
- **Combined score** = MC mean SGP + urgency * 0.4
- **Tier assignment** via natural breaks algorithm
- Dynamic replacement levels recalculated as players are drafted
- Projection confidence discount (low PA/IP get up to 20% discount)
- Bench slot optimization (late-draft bonus for multi-position flexibility)
- **Percentile sampling** — MC sims sample from P10-P90 distributions when multiple projection systems available
- **Injury badges** — Hero card shows CSS dot health indicators (green/yellow/red), age flags for aging curves, workload flags for IP spikes. No emoji — all icons are inline SVGs from `PAGE_ICONS` dict.
- **P10/P90 range bars** — Floor/ceiling projections displayed on hero pick and alternatives
- **Opponent intel tab** — Threat alerts when opponents need your target position, plus full opponent roster/needs breakdown
- **Practice mode** — Ephemeral DraftState clone for what-if scenarios, resets on refresh or button click
- **Yahoo OAuth** — Setup wizard Settings step shows Yahoo connect expander when env vars are set. Uses out-of-band (oob) flow: user clicks link, authorizes on Yahoo, pastes verification code back in the app.
- **Yahoo auto-reconnect** — On every app restart, `_try_reconnect_yahoo()` loads the saved token from `data/yahoo_token.json`, recreates `YahooFantasyClient`, and authenticates automatically. Token auto-refresh is handled by yfpy/yahoo-oauth (refresh_token never expires). Users authenticate once and never need to reconnect.

### Bootstrap Pipeline (`src/data_bootstrap.py`)
- **Splash screen** — Every app launch shows progress bar while `bootstrap_all_data()` runs all 7 data phases
- **Staleness-based refresh** — Each data source has its own max-age threshold (`StalenessConfig`): 1h live stats, 6h Yahoo, 7d players/projections, 30d historical/park factors
- **Phase ordering** — Players first (other phases need player_ids), then park factors, projections, live stats, historical, injury data, Yahoo sync
- **Zero interaction** — No CSV uploads required; all data fetched from free APIs automatically

### Auto-Fetch Pipeline (`src/data_pipeline.py`)
- **FanGraphs JSON API** — Fetches Steamer, ZiPS, Depth Charts projections (3 systems × bat/pit = 6 endpoints) on app startup
- **Normalization** — Maps FG JSON fields to DB schema; SP/RP classification mirrors CSV import logic (GS≥5→SP, SV≥3→RP)
- **Staleness check** — `refresh_if_stale()` uses `check_staleness("fangraphs_projections", 168)` to skip if fresh; `force=True` overrides
- **ADP extraction** — Pulls ADP from Steamer JSON responses, resolves names→player_ids with fuzzy fallback
- **Graceful degradation** — Partial system failures proceed with available data
- **Rate limiting** — 0.5s between requests, User-Agent header to avoid CloudFlare blocks

### Trade Analyzer Engine (`src/engine/`)
Phase 1 SGP-based evaluation pipeline (7 modules):
1. **Peer-group z-scores** — Hitters scored against hitters, pitchers against pitchers (IP>0). ERA/WHIP sign-flipped.
2. **Standings-based SGP** — Median gap between adjacent teams replaces static denominators. Falls back to defaults when standings unavailable.
3. **Marginal elasticity** — `1/gap_to_next_team` per category. Close gap = high marginal value, dominant = near-zero.
4. **Punt detection** — Category is PUNT when: (a) cannot gain any standings position in remaining weeks, AND (b) ranked 10th or worse. Punted categories get zero weight.
5. **Bench option value** — `streaming_sgp_per_week * weeks_remaining` (~0.166 SGP/week). Penalizes 2-for-1 trades, rewards 1-for-2.
6. **Weighted SGP delta** — Per-category `(after - before) / SGP_denom * marginal_weight`. Inverse stats (ERA/WHIP) sign-flipped.
7. **Grade** — Surplus SGP maps to A+ (>2.0) through F (≤-1.0). Verdict: ACCEPT if surplus > 0.
- **Graceful fallback** — Trade Analyzer page tries Phase 1 engine first, falls back to legacy `analyze_trade()` from `src/in_season.py`
- **Risk flags** — Injured players received, elite players given away (SGP>3.0), weak non-punt categories (rank≥8)

**Phase 2 (stochastic, when `enable_mc=True`):**
8. **Bayesian Model Averaging** — Posterior weights for each projection system based on P(YTD | System_i). Systems closer to reality get higher weight. Variance = within-model + between-model.
9. **KDE marginals** — Non-Gaussian stat distributions via kernel density estimation. Normal fallback when <20 historical data points. Stat bounds clipping (AVG: 0.100–0.400, ERA: 0.50–12.00).
10. **Gaussian copula** — Correlated stat sampling via Cholesky decomposition of empirical 10×10 correlation matrix (HR↔RBI: 0.85, ERA↔WHIP: 0.90, SB↔HR: -0.15). Nearest-PD correction for non-PD matrices.
11. **Paired Monte Carlo** — 10K sims with identical seeds for before/after (variance reduction). Produces: mc_mean, mc_std, p5/p25/median/p75/p95, prob_positive, VaR5, CVaR5, Sharpe ratio, 95% CI. Grade via composite = mean*0.4 + sharpe*0.3 + kelly_approx*0.3.

**Phase 3 (signal intelligence, `src/engine/signals/`):**
12. **Statcast harvesting** — Pitch-level data from Baseball Savant via pybaseball. Aggregated into rolling 14-day features: exit velocity (mean/p90), barrel%, hard hit%, xBA, xwOBA, whiff%, chase rate (batters); fastball speed/spin, K%, BB%, GB% (pitchers).
13. **Signal decay** — Exponential recency weighting with sport-specific half-lives: EV/barrel (35d), spin rate (46d), plate discipline (58d), traditional stats (87d), sprint speed (139d). λ=0 for counting stats (no decay).
14. **Kalman filter** — State-space model separating true talent from observation noise. Sample-size-aware observation variance (more PAs → lower noise → trust data more). Process variance models how much true talent can drift per time step.
15. **Regime detection** — BOCPD (Adams & MacKay 2007) detects structural changepoints via run length mode drops. 4-state HMM (Elite/Above/Below/Replacement) provides soft regime probabilities for projection blending. Graceful fallback when hmmlearn unavailable.

**Phase 4 (context engine, `src/engine/context/`, `enable_context=True`):**
16. **Log5 matchup** — Odds-ratio method for batter-pitcher matchup prediction. Combines batter rate, pitcher rate, and league average via `M_odds = (B_odds * P_odds) / L_odds`. Park factor and weather (temp/wind) adjustments. Game-level stat projections by lineup slot (PA scaling: leadoff 4.65 → 9-hole 3.85).
17. **Injury stochastic process** — Weibull-distributed injury durations by body part (hamstring shape=1.8/scale=18, UCL shape=3.0/scale=365). Frailty multiplier from health score (`1/max(hs, 0.5)`, max 2.0). Season availability sampling for MC integration via `sample_season_availability()`.
18. **Enhanced bench option value** — Extends simple streaming value (0.166 SGP/week) with 4 components: streaming, hot FA pickup (15%/week × 0.5 SGP), roster flexibility premium (multi-position eligibility normalized to [0,1]), and injury replacement cushion. Returns component breakdown dict.
19. **Roster concentration risk** — Herfindahl-Hirschman Index (HHI = Σshare²) measuring team exposure. PA-weighted for hitters, IP-weighted for pitchers. Penalty = `(HHI - 0.15) * 3.0` when above 0.15 threshold. `compute_concentration_delta()` compares before/after trade rosters. Team alias normalization (WSN→WSH, AZ→ARI).

**Phase 5 (game theory, `src/engine/game_theory/`, `enable_game_theory=True`):**
20. **Opponent valuations** — Estimates each opponent's willingness-to-pay based on THEIR category needs. Market clearing price via Vickrey auction (second-highest bidder = Nash equilibrium). Player demand count (teams valuing player above 0.5 SGP threshold).
21. **Adverse selection** — Bayesian P(flaw|offered) discount. Prior P(flaw)=0.15, P(offered|flaw)=0.60, P(offered|ok)=0.20. Calibrated from manager trade history when ≥3 trades available. Discount factor capped at 0.75 (max 25% haircut).
22. **Dynamic programming** — Bellman rollout approximates future trade option value via MC. Discount factor γ depends on playoff probability (contending=0.98, bubble=0.95, rebuilding=0.85). Roster balance score affects future trade opportunity probability.
23. **Sensitivity + counter-offers** — Category sensitivity ranking by absolute SGP impact. Breakeven gap analysis with vulnerability classification (robust/moderate/fragile/razor-thin). Counter-offer generation tries swapping each given player with roster alternatives, returning top 3 improvements.

**Phase 6 (production, `src/engine/production/`):**
24. **Convergence diagnostics** — Effective Sample Size (ESS) via FFT autocorrelation, split-R̂ (Gelman-Rubin), running mean stability normalized by sample std. Quality classification: excellent (ESS>1000 + all pass), good, marginal, poor.
25. **Precomputation cache** — In-memory cache with TTL staleness tracking. `get_or_compute()` pattern for lazy refresh. Default TTLs: copula 24h, SGP 1h, gap analysis 1h, market values 30min. Module-level singleton via `get_trade_cache()`.
26. **Adaptive simulation scaling** — 1K (quick) → 10K (standard) → 50K (production) → 100K (full). Scales by trade complexity (+5K per extra player beyond 1-for-1). Time budget cap ensures interactive responsiveness. `recommend_n_sims()` uses current ESS to suggest optimal count.

### In-Season Algorithms
- **Trade Analyzer (legacy):** Roster swap → projected season totals (YTD + ROS) → park-adjusted SGP delta → MC simulation (200 sims) → verdict with confidence %. Now includes injury badges for both sides and P10/P90 risk assessment.
- **Player Compare:** ROS projections → Z-score normalization across 10 categories → composite weighted score, optional marginal SGP team impact. Now includes health badges and projection confidence (P10-P90 range width).
- **FA Ranker:** Marginal SGP per FA vs. user's roster → category-need weighting → replacement target identification → sort by net marginal value
- **Live Stats:** MLB Stats API (daily auto-refresh) + pybaseball ROS projections, staleness tracking via `refresh_log` table
- **Lineup Optimizer:** PuLP LP solver with binary assignment per player/slot, category targeting based on standings gaps, two-start SP detection. Now includes health-adjusted SGP penalties and auto-detected two-start SP highlights.
- **My Team:** Roster display with injury badges, Bayesian projection indicators, Yahoo sync button (when connected)

### Advanced Analytics (Plan 3)
- **Bayesian Updater:** PyMC 5 hierarchical beta-binomial model for in-season stat updates; Marcel regression fallback when PyMC unavailable. Uses FanGraphs stabilization thresholds (K=60 PA, AVG=910 AB, ERA=70 IP). Aging curves on logit scale.
- **Injury Model:** `health_score = avg(GP/games_available)` over 3 seasons. Age-risk curves: hitters +2%/yr after 30, pitchers +3%/yr after 28. Workload flag for >40 IP increase. Counting stats scaled by health score.
- **Percentile Forecasts:** Inter-projection volatility (StdDev across Steamer/ZiPS/Depth Charts). P10/P50/P90 using ±1.28σ. Process risk widening for low-correlation stats (AVG r²=0.41 vs HR r²=0.72).
- **Enhanced Opponent Model:** `P(pick) = 0.5*ADP + 0.3*team_need + 0.2*historical_preference`. Computes per-team positional bias from draft history. Falls back to ADP-only when no history available.
- **Yahoo API:** OAuth 2.0 via yfpy v17+, out-of-band (oob) flow. User clicks link → authorizes on Yahoo → pastes verification code. League settings, rosters, standings, FA pool, draft results sync. Graceful degradation when not connected.

### Database Tables (14)

| Group | Tables |
|-------|--------|
| Draft | `projections`, `adp`, `league_config`, `draft_picks`, `blended_projections`, `player_pool` |
| In-Season | `season_stats`, `ros_projections`, `league_rosters`, `league_standings`, `park_factors`, `refresh_log` |
| Plan 3 | `injury_history`, `transactions` |

## Data Sources

- **Players:** Auto-fetched from MLB Stats API on every launch (750+ active players); staleness: 7 days
- **Draft projections:** Auto-fetched from FanGraphs JSON API (Steamer, ZiPS, Depth Charts) on startup; staleness: 7 days
- **ADP:** Extracted from FanGraphs Steamer JSON (filters ADP ≥ 999 and nulls); FantasyPros consensus as fallback
- **Current stats:** MLB Stats API (`MLB-StatsAPI` package, auto-refreshed on launch); staleness: 1 hour
- **Historical stats:** 3 years (2023-2025) from MLB Stats API for injury modeling; staleness: 30 days
- **Park factors:** Hardcoded FanGraphs 2024 values (30 teams) in `data_bootstrap.py`; staleness: 30 days
- **Injury history:** Derived from historical stats (games played vs available); staleness: 30 days
- **League rosters/standings:** Yahoo Fantasy API sync (optional, auto-syncs when connected); staleness: 6 hours

## Key API Signatures

These are commonly called wrong — always double-check:

```python
# STANDALONE functions in src/valuation.py — NOT methods on SGPCalculator
compute_replacement_levels(pool, config, sgp_calc)   # valuation.py:209
compute_sgp_denominators(pool, config)                # valuation.py:446

# Keyword args required for optional params
value_all_players(pool, config, roster_totals=None, category_weights=None,
                  replacement_levels=None, current_round=None, num_rounds=23)

# load_player_pool() columns:
# player_id, name, team, positions, is_hitter, is_injured,
# pa, ab, h, r, hr, rbi, sb, avg, ip, w, sv, k, era, whip, er, bb_allowed, h_allowed, adp

# --- Trade Engine Phase 1 APIs (src/engine/) ---

# Trade evaluator (src/engine/output/trade_evaluator.py)
evaluate_trade(giving_ids, receiving_ids, user_roster_ids, player_pool,
               config=None, user_team_name=None, weeks_remaining=16,
               enable_mc=False, enable_context=True, enable_game_theory=True)
# Returns: {grade, surplus_sgp, category_impact, category_analysis, punt_categories,
#           bench_cost, risk_flags, verdict, confidence_pct, before_totals, after_totals,
#           giving_players, receiving_players, total_sgp_change, mc_mean, mc_std,
#           concentration_hhi_before, concentration_hhi_after, concentration_delta,
#           concentration_penalty, bench_option_detail,
#           adverse_selection, market_values, sensitivity_report}
grade_trade(surplus_sgp: float) -> str  # A+ through F

# Z-score + SGP valuation (src/engine/portfolio/valuation.py)
compute_player_zscores(player_pool, config=None) -> DataFrame  # adds z_{cat} + z_composite columns
compute_sgp_from_standings(standings, config=None) -> dict[str, float]  # standings-based SGP denoms
compute_player_vorp(player_pool, config=None) -> DataFrame  # adds vorp column
build_valuation_context(config=None) -> dict  # full context from DB state

# Category analysis (src/engine/portfolio/category_analysis.py)
compute_marginal_sgp(your_totals, all_team_totals, categories=None) -> dict[str, float]
category_gap_analysis(your_totals, all_team_totals, your_team_id, weeks_remaining=16) -> dict[str, dict]
build_standings_totals(standings: DataFrame) -> dict[str, dict[str, float]]
compute_category_weights_from_analysis(analysis) -> dict[str, float]  # punts = 0.0

# Lineup optimizer wrapper (src/engine/portfolio/lineup_optimizer.py)
compute_optimal_lineup_value(roster, config, category_weights=None) -> float
compute_lineup_delta(before_roster, after_roster, config, category_weights=None) -> float
bench_option_value(weeks_remaining=16, streaming_sgp_per_week=0.166) -> float

# Projection client (src/engine/projections/projection_client.py)
fuzzy_match_player(name, candidates, name_col="name", threshold=0.7) -> str | None
get_ros_projections(player_ids) -> DataFrame
resolve_trade_players(giving_names, receiving_names, player_pool) -> tuple[list[int], list[int]]

# --- Trade Engine Phase 2 APIs (src/engine/) ---

# BMA (src/engine/projections/bayesian_blend.py)
bayesian_model_average(ytd_stats, projections, prior_weights=None)
# Returns: (posterior_weights, blended_projection, blended_variance)
compute_bma_for_player(player_ytd, system_projections) -> dict[str, float]
compute_bma_variance(player_ytd, system_projections) -> dict[str, float]
SYSTEM_FORECAST_SIGMA: dict[str, dict[str, float]]  # steamer/zips/depthcharts × 10 stats

# KDE marginals (src/engine/projections/marginals.py)
PlayerMarginal(stat_name, projected_value, variance, historical_values=None)
# .ppf(quantile) -> stat_value  (inverse CDF for copula integration)
# .sample(n, rng) -> np.ndarray
# .uses_kde -> bool
build_player_marginals(projected_stats, variances, historical_by_stat=None)

# Copula (src/engine/portfolio/copula.py)
GaussianCopula(correlation=None)  # defaults to 10×10 DEFAULT_CORRELATION
# .sample(n, rng) -> np.ndarray of shape (n, 10) in [0, 1]
sample_correlated_stats(copula, player_marginals, n=1, rng=None) -> np.ndarray
fit_copula_from_data(player_seasons) -> GaussianCopula

# Paired MC (src/engine/monte_carlo/trade_simulator.py)
run_paired_monte_carlo(before_roster_stats, after_roster_stats,
    before_marginals=None, after_marginals=None, copula=None,
    all_team_totals=None, sgp_denominators=None,
    n_sims=10000, seed=42, weeks_remaining=16)
# Returns: {mc_mean, mc_std, mc_median, p5..p95, prob_positive,
#           var5, cvar5, sharpe, grade, verdict, confidence_pct,
#           confidence_interval, surplus_distribution, n_sims}
build_roster_stats(player_ids, player_pool) -> dict[str, dict[str, float]]

# --- Trade Engine Phase 3 APIs (src/engine/signals/) ---

# Statcast harvesting (src/engine/signals/statcast.py)
fetch_batter_statcast(player_name, start_date, end_date) -> pd.DataFrame
fetch_pitcher_statcast(player_name, start_date, end_date) -> pd.DataFrame
aggregate_batter_statcast(df) -> dict[str, float]  # ev_mean, ev_p90, barrel_pct, hard_hit_pct, xba, xwoba, whiff_pct, chase_rate
aggregate_pitcher_statcast(df) -> dict[str, float]  # ff_avg_speed, ff_spin_rate, k_pct, bb_pct, gb_pct, xba_against, xwoba_against
compute_rolling_features(df, window_days=14, step_days=7, is_pitcher=False) -> list[dict]

# Signal decay (src/engine/signals/decay.py)
DECAY_LAMBDAS: dict[str, float]  # 7 categories: batted_ball=0.020, spin=0.015, discipline=0.012, ...
FEATURE_DECAY_MAP: dict[str, str]  # feature_name -> decay category
decay_weight(obs_date, reference_date, lambda_param) -> float  # e^(-λ * days_diff)
apply_decay_weights(observations, feature_name, reference_date) -> tuple[np.ndarray, np.ndarray]
weighted_mean(values, weights) -> float
weighted_variance(values, weights) -> float
get_feature_lambda(feature_name) -> float
half_life_days(lambda_param) -> float  # ln(2) / lambda

# Kalman filter (src/engine/signals/kalman.py)
kalman_true_talent(observations, obs_variance, process_variance, prior_mean, prior_variance)
# Returns: (filtered_means, filtered_variances) — np.ndarray pair
observation_variance(feature_name, sample_size) -> float
get_process_variance(feature_name) -> float
run_kalman_for_feature(rolling_data, feature_name, prior_mean=None)
# Returns: {filtered_mean, filtered_var, kalman_gain_final}

# Regime detection (src/engine/signals/regime.py)
BOCPD(hazard_lambda=200, mu0=0.0, kappa0=1.0, alpha0=1.0, beta0=1.0)
# .update(x) -> (changepoint_prob, run_length_probs)
# .reset() -> None
detect_changepoints(time_series, hazard_lambda=200, threshold=0.7)
# Returns: {changepoint_indices, changepoint_probs, last_changepoint, current_regime_length}
fit_player_hmm(obs_matrix, n_states=4) -> tuple[model | None, state_probs]
regime_conditional_projection(current_probs, projections_by_state) -> dict[str, float]
classify_regime_simple(recent_xwoba, season_xwoba, league_avg_xwoba=0.315) -> tuple[str, np.ndarray]
REGIME_STATES: list[str]  # ["Elite", "Above-avg", "Below-avg", "Replacement"]
HMM_AVAILABLE: bool  # True if hmmlearn is installed

# --- Trade Engine Phase 4 APIs (src/engine/context/) ---

# Log5 matchup (src/engine/context/matchup.py)
log5_matchup(batter_rate, pitcher_rate, league_avg) -> float  # odds-ratio matchup rate
park_adjust(stat_value, park_factor) -> float  # multiplicative park adjustment
game_projection(batter_xwoba, pitcher_xwoba_against=None, park_factor=1.0,
    lineup_slot=5, league_avg_woba=0.315, temp_f=None, wind_out_mph=None) -> dict
# Returns: {hr, r, rbi, sb, h, ab, pa}
matchup_adjustment_factor(player_xwoba=None, opponent_xwoba_against=None,
    park_factor=1.0, league_avg=0.315) -> float  # multiplicative factor (1.0=neutral)
LINEUP_SLOT_PA: dict[int, float]  # 1:4.65 → 9:3.85

# Injury stochastic process (src/engine/context/injury_process.py)
sample_injury_duration(injury_type="other", frailty=1.0, rng=None) -> int  # days
frailty_from_health_score(health_score) -> float  # 1.0/max(hs, 0.5)
estimate_injury_probability(health_score, age=None, is_pitcher=False, horizon_days=30) -> float
sample_season_availability(health_score, age=None, is_pitcher=False,
    weeks_remaining=16, rng=None) -> float  # fraction in [0, 1]
sample_availability_batch(health_scores, ages=None, is_pitcher_flags=None,
    weeks_remaining=16, n_sims=1, rng=None) -> np.ndarray  # (n_sims, n_players)
INJURY_DURATION: dict[str, dict[str, float]]  # 6 body parts: hamstring, oblique, ucl, shoulder, back, other

# Enhanced bench value (src/engine/context/bench_value.py)
enhanced_bench_option_value(weeks_remaining=16, streaming_sgp_per_week=0.15,
    roster_flexibility=0.0, injury_replacement_value=0.0) -> dict
# Returns: {streaming, hot_fa, flexibility, injury_cushion, total}
compute_roster_flexibility(roster_df) -> float  # normalized [0, 1]
compute_injury_replacement_value(roster_df, bench_count=5, avg_health_score=0.85,
    weeks_remaining=16) -> float  # expected SGP saved

# Concentration risk (src/engine/context/concentration.py)
roster_concentration_hhi(roster_df) -> float  # HHI in [0, 1]
concentration_risk_penalty(hhi, threshold=0.15, scale=3.0) -> float  # SGP penalty
compute_concentration_delta(before_df, after_df) -> dict
# Returns: {before_hhi, after_hhi, delta, penalty_before, penalty_after, penalty_delta}
team_exposure_breakdown(roster_df) -> list[dict]  # [{team, count, share}, ...]
TEAM_ALIASES: dict[str, str]  # WSN→WSH, AZ→ARI, etc.

# --- Trade Engine Phase 5 APIs (src/engine/game_theory/) ---

# Opponent valuation (src/engine/game_theory/opponent_valuation.py)
estimate_opponent_valuations(player_projections, all_team_totals, your_team_id,
    sgp_denominators=None) -> dict[str, float]  # {team_name: valuation_sgp}
market_clearing_price(valuations) -> float  # Nash equilibrium = 2nd highest bid
player_market_value(player_projections, all_team_totals, your_team_id,
    sgp_denominators=None) -> dict  # {valuations, market_price, max_bidder, max_bid, demand}
get_player_projections_from_pool(player_id, player_pool) -> dict[str, float]

# Adverse selection (src/engine/game_theory/adverse_selection.py)
adverse_selection_discount(offering_manager_history=None, p_flaw_prior=0.15) -> float  # (0.75, 1.0]
compute_discount_for_trade(receiving_player_count=1, offering_manager_history=None) -> dict
# Returns: {discount_factor, p_flaw, p_flaw_given_offered, sgp_adjustment, total_sgp_adjustment, risk_level}

# Dynamic programming (src/engine/game_theory/dynamic_programming.py)
get_gamma(playoff_probability) -> float  # 0.85-0.98
estimate_playoff_probability(standings_rank, num_teams=12, weeks_remaining=16) -> float
bellman_rollout(immediate_surplus, weeks_remaining=16, playoff_probability=0.50,
    roster_balance_before=0.0, roster_balance_after=0.0,
    n_lookahead=2, n_sims=200, seed=42) -> dict
# Returns: {immediate, future_before, future_after, option_value, gamma, total_value}
compute_roster_balance(roster_category_ranks, num_teams=12) -> float  # [-1, 1]

# Sensitivity (src/engine/game_theory/sensitivity.py)
category_sensitivity(category_impact, category_weights=None) -> list[dict]  # sorted by |impact|
player_sensitivity(giving_ids, receiving_ids, ..., evaluate_fn, base_surplus) -> list[dict]
suggest_counter_offers(giving_ids, receiving_ids, ..., evaluate_fn, base_surplus,
    max_suggestions=3) -> list[dict]  # [{swap_out, swap_in, improvement, new_grade}]
trade_sensitivity_report(category_impact, category_weights=None, surplus_sgp=0.0) -> dict
# Returns: {category_ranking, biggest_driver, biggest_drag, breakeven_gap, vulnerability}

# --- Trade Engine Phase 6 APIs (src/engine/production/) ---

# Convergence diagnostics (src/engine/production/convergence.py)
effective_sample_size(samples) -> float  # ESS via FFT autocorrelation
split_rhat(samples) -> float  # split-R̂ (near 1.0 = converged)
running_mean_stability(samples, window=500) -> float  # lower = more stable
check_convergence(samples) -> dict  # {ess, rhat, stability, converged, quality}
recommend_n_sims(current_ess, target_ess=500, current_n=10000) -> int  # capped at 100K

# Cache (src/engine/production/cache.py)
TradeEvalCache()  # In-memory cache with TTL
# .get(key) -> value | None
# .set(key, value, ttl=3600)
# .get_or_compute(key, compute_fn, ttl=None) -> value
# .invalidate(key) -> bool
# .clear() -> int
# .stats() -> dict
get_trade_cache() -> TradeEvalCache  # Module-level singleton
reset_trade_cache() -> None

# Simulation config (src/engine/production/sim_config.py)
compute_adaptive_n_sims(n_giving=1, n_receiving=1, mode="standard",
    time_budget_s=None) -> int  # 1K-100K
get_sim_mode(interactive=True) -> str  # "standard" or "production"
estimate_runtime_seconds(n_sims) -> float
sim_config_summary(n_giving, n_receiving, mode, time_budget_s=None) -> dict

# --- Plan 3 new APIs ---

# Bayesian updater (src/bayesian.py)
BayesianUpdater(prior_weight=0.6)
updater.regressed_rate(observed_rate, sample_size, league_mean, stabilization_point)
updater.batch_update_projections(season_stats_df, preseason_df, config=None)
updater.age_adjustment(age, stat)  # returns float multiplier

# Injury model (src/injury_model.py)
compute_health_score(games_played_3yr: list, games_available_3yr: list) -> float
apply_injury_adjustment(projections_df, health_scores_df) -> pd.DataFrame
get_injury_badge(health_score) -> tuple[str, str]  # (css_dot_html, label) — returns <span> with colored dot, NOT emoji

# UI shared (src/ui_shared.py)
PAGE_ICONS: dict[str, str]  # ~22 inline SVG icons keyed by name ("logo", "logo_lg", "baseball", "fire", "accept", etc.)
METRIC_TOOLTIPS: dict[str, str]  # Educational tooltip text for every metric (sgp, vorp, survival, etc.)
THEME: dict  # Single light-mode palette: bg=#f4f5f0, primary=#e63946, hot=#ff6d00, gold=#ffd60a, green=#2d6a4f, sky=#457b9d, purple=#6c63ff
T = THEME  # Direct dict alias — no proxy needed without dark mode. Backward compat: T["amber"]→#e63946, T["teal"]→#457b9d
get_theme() -> dict  # Stub — always returns THEME (kept for backward compat)
render_theme_toggle()  # No-op stub (kept for backward compat)
inject_custom_css()  # Injects full CSS (1500+ lines) with glassmorphism, 3D buttons, kinetic typography, 7 animations, orange sidebar, bold titles, contrasting data tables

# Percentiles (src/valuation.py)
compute_projection_volatility(projections_by_system: dict[str, DataFrame]) -> DataFrame
add_process_risk(volatility_df) -> DataFrame  # widens CI for low-correlation stats
compute_percentile_projections(base_df, volatility_df, percentiles=[10,50,90]) -> dict[int, DataFrame]

# Opponent model (src/simulation.py, src/draft_state.py)
compute_team_preferences(draft_history_df)  # needs columns: team_key, positions, round
get_team_draft_patterns(draft_state_dict, team_id: int)  # team_id is 0-based index
get_positional_needs(draft_state_dict, team_id: int, roster_config: dict)

# Lineup optimizer (src/lineup_optimizer.py)
LineupOptimizer(roster_df, config_dict)
optimizer.optimize_lineup()  # returns {slot: player_name, projected_stats: {...}}
optimizer.category_targeting(standings_df, team_name)  # returns {cat: weight}

# Data bootstrap (src/data_bootstrap.py)
bootstrap_all_data(yahoo_client=None, on_progress=None, force=False, staleness=None)  # returns dict[str, str]
StalenessConfig(players_hours=168, live_stats_hours=1, projections_hours=168, historical_hours=720, park_factors_hours=720, yahoo_hours=6)
BootstrapProgress(phase="", detail="", pct=0.0)  # callback dataclass for splash screen
PARK_FACTORS: dict[str, float]  # 30-team dict, e.g. {"COL": 1.38, "MIA": 0.88}

# Database bulk helpers (src/database.py)
check_staleness(source, max_age_hours)  # returns bool — True if needs refresh
upsert_player_bulk(players: list[dict])  # dict needs: name, team, positions, is_hitter; optional: mlb_id
upsert_injury_history_bulk(records: list[dict])  # dict needs: player_id, season, games_played, games_available
upsert_park_factors(factors: list[dict])  # dict needs: team_code, factor_hitting, factor_pitching

# Live stats extensions (src/live_stats.py)
fetch_all_mlb_players(season=2026)  # returns DataFrame with mlb_id, name, team, positions, is_hitter
fetch_historical_stats(seasons=[2023,2024,2025])  # returns {year: DataFrame}
fetch_injury_data_bulk(historical_stats)  # returns list[dict] with player_name, team, season, games_played, games_available

# Data pipeline (src/data_pipeline.py)
# Yahoo OAuth helpers (src/yahoo_api.py)
build_oauth_url(consumer_key, redirect_uri="oob")  # returns auth URL string
exchange_code_for_token(consumer_key, consumer_secret, code)  # returns token dict or None

refresh_if_stale(force=False)  # returns bool; uses check_staleness() with 168h threshold
fetch_projections(system, stats)  # returns (DataFrame, raw_json_list)
SYSTEM_MAP = {"steamer": "steamer", "zips": "zips", "fangraphsdc": "depthcharts"}
```

## Gotchas

- **DB column `name` vs UI `player_name`** — Normalized via `rename(columns={"name": "player_name"})` in `_build_player_pool()`. Always check which context you're in.
- **Sample data is `system='blended'` only** — Do NOT call `create_blended_projections()` when only blended data exists; it deletes `system='blended'` rows first.
- **Python 3.14 + SQLite bytes** — Returns bytes for some integer columns. Fixed with explicit `CAST` in SQL and `pd.to_numeric()` coercion.
- **Plotly 6.x hex colors** — Does NOT accept 8-digit hex (`#RRGGBBAA`). Use `rgba(r,g,b,a)` format.
- **Snake draft order** — `round % 2 == 0` forward, `round % 2 == 1` reverse.
- **MC horizon limit** — 6-round horizon for performance (4.45s vs 10.42s without).
- **Streamlit pages outside runtime** — Pages use `if/else` guard pattern around `st.stop()` to prevent crashes when imported outside Streamlit.
- **`datetime.now(UTC)`** — Used everywhere; `datetime.utcnow()` is deprecated.
- **`width="stretch"`** — Replaces deprecated `use_container_width=True` in Streamlit.
- **PyMC/PuLP optional deps** — `PYMC_AVAILABLE` and `PULP_AVAILABLE` flags in bayesian.py and lineup_optimizer.py. Always use `try/except ImportError` pattern. CI skips tests requiring these with `@pytest.mark.skipif`.
- **`compute_team_preferences()` column names** — Expects `team_key` and `positions`, NOT `team_id`/`position`. Integration tests caught this.
- **`get_team_draft_patterns()` uses int team_id** — `team_id` is 0-based team index, not team name string. Filters on `pick.get("team_index")`.
- **health_score range** — Always 0.0 to 1.0. Missing seasons default to 0.85 (league average). Empty history returns 0.85.
- **Yahoo API graceful degradation** — All Yahoo features wrapped in `YFPY_AVAILABLE` checks. App works fully without Yahoo credentials. Setup wizard Yahoo connect requires `YAHOO_CLIENT_ID` and `YAHOO_CLIENT_SECRET` env vars.
- **Yahoo OAuth uses oob (out-of-band)** — Yahoo Fantasy API requires `redirect_uri=oob`. Do NOT use `http://localhost:8501/` or any redirect-based flow — Yahoo rejects all non-oob redirect URIs for fantasy apps. The `yahoo-oauth` library hardcodes `CALLBACK_URI='oob'`. Our Streamlit UI mirrors this: user clicks a link, authorizes on Yahoo, pastes the verification code back.
- **Yahoo auto-reconnect on restart** — `_try_reconnect_yahoo()` in `app.py` loads `data/yahoo_token.json` on every Streamlit restart, recreates `YahooFantasyClient`, and authenticates. Token refresh is automatic (yahoo-oauth checks `token_is_valid()` at 59-min mark and refreshes using `refresh_token`). Users connect once and never need to re-authorize.
- **Yahoo game_key resolution** — `_resolve_game_key()` in `yahoo_api.py` uses 3 strategies: (1) `get_game_key_by_season()`, (2) `get_all_yahoo_fantasy_game_keys()` enumeration, (3) `get_current_game_metadata()` fallback. MLB 2026 game_key = 469. Must set BOTH `self._query.game_id` AND `self._query.league_key` — belt and suspenders.
- **yfpy Roster iteration** — `Roster.__iter__()` yields attribute NAMES (strings), NOT players. Always use `getattr(roster, "players", None) or []` to get the actual player list. Never iterate a Roster object directly.
- **Python 3.14 bytes in yfpy** — `team.name` and `player.name.full` may return `bytes` instead of `str`. Always guard with `raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)`.
- **Health badges in st.dataframe()** — `get_injury_badge()` returns HTML `<span>` tags, but `st.dataframe()` renders raw text only. My Team page uses text labels ("Low Risk", "Moderate Risk", "High Risk") for the dataframe Health column. HTML badges are used only in `st.markdown(unsafe_allow_html=True)` contexts like the draft hero card.
- **yfpy `--no-deps` install needs extras** — Installing `yfpy>=17.0 --no-deps` strips `stringcase`, `yahoo-oauth`, and their transitive deps. Must also `pip install stringcase yahoo-oauth==2.1.1`. Similarly, `streamlit-oauth --no-deps` needs `httpx-oauth`. The python-dotenv version mismatch (1.2.2 vs pinned 1.1.1) is a pip warning but works at runtime.
- **In-season page warnings** — All pages that require league data use the standardized message: "No league data loaded. Connect your Yahoo league in Settings, or league data will load automatically on next app launch."
- **FanGraphs API SYSTEM_MAP** — FG API uses `"fangraphsdc"` for Depth Charts, but DB stores as `"depthcharts"`. Always use `SYSTEM_MAP` dict, never hardcode system names. `SYSTEMS = list(SYSTEM_MAP.keys())` derives the list.
- **FanGraphs `minpos` field** — Returns primary position only (e.g., "SS"), not multi-position eligibility. Sometimes returns `"0"` or `"-"` meaning no position; these are guarded to fall back to `"Util"`.
- **Bootstrap session state** — `st.session_state.bootstrap_complete` gates the bootstrap-once logic. Delete the key to force re-bootstrap. Results stored in `st.session_state.bootstrap_results`.
- **Bootstrap lazy imports** — `data_bootstrap.py` imports from `database.py` and `live_stats.py` inside functions to avoid circular imports and enable graceful degradation.
- **Players table has no UNIQUE constraint on name** — `upsert_player_bulk()` uses SELECT-first pattern (check if name exists, then INSERT or UPDATE). Do NOT use `ON CONFLICT(name)`. However, `injury_history` DOES have a UNIQUE index on `(player_id, season)` so `ON CONFLICT` is safe there.
- **Park factors schema** — Uses columns `team_code`, `factor_hitting`, `factor_pitching` (not just `team` and `park_factor`). `upsert_park_factors()` accepts both naming conventions via flexible getter.
- **sgp_volatility alignment** — When `evaluate_candidates()` receives `sgp_volatility`, the array is aligned to the full pool. After filtering drafted players, it must be re-indexed via `pd.Series.reindex()` to match the filtered `available` pool.
- **Practice mode isolation** — Uses separate `st.session_state["practice_draft_state"]` DraftState, never persisted to disk/DB. Resets on page refresh or "Reset Practice" button click.
- **Mock Draft session state** — All keys prefixed `mock_` (`mock_ds`, `mock_pool`, `mock_lc`, `mock_sgp`, `mock_started`, `mock_draft_pos`, `mock_num_sims`). Ephemeral — never saved to DB/disk. Resets on "Reset Draft" button click.
- **`evaluate_candidates()` returns `name`, not `player_name`** — The result DataFrame uses `"name"` from the pool. Always alias to `"player_name"` after calling: `recs["player_name"] = recs["name"]`.
- **Percentile pipeline ordering** — `compute_projection_volatility()` → `add_process_risk()` → `compute_percentile_projections()`. Skip entirely when only one projection system exists (zero variance). All 3 consumers (app.py, Trade Analyzer, Player Compare) must follow this exact ordering.
- **Connection leak pattern in pages** — Always wrap `get_connection()` + queries in `try/finally` with `conn.close()` in the `finally` block. Do NOT put `conn.close()` inline after queries — if a query throws, the connection leaks.
- **Scarcity toast dedup** — `st.toast()` fires on every Streamlit rerender. Use `st.session_state` keys to deduplicate (e.g., `f"scarcity_toast_{pos}_{count}"`).
- **`T["ink"]` vs `T["bg"]` for text-on-accent** — `T["bg"]` is for page backgrounds. `T["ink"]` is for dark text on colored surfaces (amber buttons, badges, tabs). Never use `T["bg"]` as a text color — it's invisible on amber in light mode.
- **No emoji in the codebase** — All icons are inline SVGs from `PAGE_ICONS` dict in `ui_shared.py`. Injury badges use CSS dots (`border-radius:50%`), not emoji. Do NOT re-introduce emoji.
- **No abbreviations in UI text** — All user-facing text uses full terms: "Standings Gained Points" (not SGP), "Value Over Replacement Player" (not VORP), "Average Draft Position" (not ADP), "Monte Carlo" (not MC), "Runs Batted In" (not RBI), etc. Variable names and DB columns stay abbreviated. Category display uses `cat_names`/`cat_display_names` mapping dicts.
- **Draft Settings live on Draft Simulator page** — Number of teams, Rounds, and Draft position inputs are on `pages/2_Draft_Simulator.py`, not on the Connect League page. Stored in `mock_num_teams`, `mock_num_rounds`, `mock_draft_pos` session state. Also written to shared `num_teams`/`num_rounds`/`draft_pos` keys for the main draft.
- **Yahoo sync across pages** — Pages reuse `st.session_state.yahoo_client` for Yahoo sync instead of creating new clients. Token data stored in `st.session_state.yahoo_token_data` for re-authentication. Pages show "Sync League Data Now" button when Yahoo connected but DB empty.
- **`T = THEME` is now a plain dict** — `T` in `ui_shared.py` is just `THEME` (the single color palette dict). The old `_ThemeProxy` class was removed along with dark mode. Backward-compat aliases: `T["amber"]`→`"#e63946"` (primary red), `T["teal"]`→`"#457b9d"` (sky blue).
- **Sidebar nav rename via JS** — The sidebar "app" label is renamed to "Connect League" and "Mock Draft" to "Draft Simulator" using JS `textContent` replacement in `inject_custom_css()`. Same JS block also injects the HEATER logo + text into the sidebar header.
- **Streamlit CSS `!important` requirement** — Streamlit's React-based renderer applies high-specificity inline CSS. ALL custom CSS properties on injected HTML (`.page-title`, `.metric-card`, etc.) need `!important` to override. Without it, custom backgrounds, font-sizes, and display properties silently revert to Streamlit defaults.
- **Streamlit HTML sanitization** — `st.markdown('<div class="glass">')` immediately auto-closes the tag. Cannot use split open/close div patterns with Streamlit widgets between them. Content renders outside the wrapper. Use self-contained HTML blocks only.
- **Orange sidebar + white text** — Sidebar background is `linear-gradient(180deg, #e65c00, #cc5200)`. All sidebar text forced to `#ffffff !important`. Active nav item uses dark overlay `rgba(0,0,0,0.15)` instead of white highlight for contrast.
- **Title badge deep navy gradient** — Page title badges use `linear-gradient(135deg, #1a1a2e, #16213e)` background with gradient text overlay (red > orange > gold via `background-clip: text`).
- **All buttons are orange** — Secondary buttons globally styled with `linear-gradient(135deg, #e65c00, #cc5200)` + white bold text. This matches the sidebar branding and creates visual consistency.
- **Data table white background** — All `st.dataframe()` tables get `background: #ffffff !important` to contrast against the page's `#f4f5f0` chalk background. Column headers get `font-weight: 700 !important`.
- **Rate-stat aggregation** — AVG=sum(h)/sum(ab), ERA=sum(er)*9/sum(ip), WHIP=sum(bb+h)/sum(ip). Weighted averages, NOT simple averages. `_fix_rate_stats()` in `lineup_optimizer.py` recalculates these after LP solves.
- **Injury model scales rate stats** — `apply_injury_adjustment()` scales ER, BB_allowed, H_allowed by `_combined_factor` (health×age×workload), not just counting stats. Without this, injured pitchers show artificially low ERA/WHIP.
- **LP inverse stat weighting** — ERA/WHIP in lineup optimizer LP objective must be weighted by IP: `player_value -= val * ip * weight`. Without IP weighting, a 1-IP reliever with 0.00 ERA dominates a 200-IP starter.
- **`compare_players()` peer-group filtering** — Z-scores computed against `is_hitter`-filtered pool only. Hitter HR z-score uses hitter pool mean/std, not full pool (which includes pitchers with HR=0).
- **`check_staleness()` edge case** — `max_age_hours <= 0` returns `True` (always stale). Prevents division-by-zero and logical errors.
- **Trade engine graceful fallback** — `pages/3_Trade_Analyzer.py` wraps `from src.engine.output.trade_evaluator import evaluate_trade` in try/except. If the engine module is missing or broken, falls back to legacy `analyze_trade()` from `src/in_season.py`. Both code paths produce compatible output dicts.
- **Trade engine backward compat keys** — `evaluate_trade()` returns both new keys (`grade`, `surplus_sgp`, `category_analysis`) AND legacy keys (`total_sgp_change`, `mc_mean`, `mc_std`) so existing UI code doesn't break.
- **`compute_marginal_sgp()` uses `.get()` for missing categories** — Team totals may not have all 10 categories. Always use `team_totals.get(cat, 0.0)`, never `team_totals[cat]`.
- **Punt detection requires BOTH conditions** — A category is punt only when `gainable_positions == 0 AND rank >= 10`. Being rank 10 alone isn't enough if positions are still gainable; having 0 gainable alone isn't enough if you're already ranked high.
- **Bench option value sign convention** — `bench_cost` in the trade result is positive when you LOSE bench slots (2-for-1 trade receiving side), negative when you GAIN them (1-for-2). The surplus already accounts for this.
- **`_roster_category_totals()` lives in `src/in_season.py`** — The trade engine's `evaluate_trade()` imports this from the legacy module. Don't duplicate it.
- **`enable_mc=True` activates Phase 2** — By default, `evaluate_trade()` runs Phase 1 only (deterministic). Set `enable_mc=True` for MC overlay. MC gracefully falls back to Phase 1 on failure.
- **BMA sigma difference → unequal weights** — Even when all systems project the same value, posterior weights differ slightly because forecast sigmas differ by system (Steamer HR sigma=5.2 vs ZiPS sigma=5.5). Tighter sigma → higher likelihood density → more weight.
- **KDE needs ≥20 samples** — `PlayerMarginal` falls back to Normal when `len(historical_values) < MIN_KDE_SAMPLES`. KDE with too few points produces unreliable density estimates.
- **Gaussian copula nearest-PD** — The `_nearest_pd()` function clips negative eigenvalues to 1e-10 and re-normalizes the diagonal to 1.0. This ensures Cholesky decomposition always succeeds, even with user-provided non-PD correlation matrices.
- **Paired MC seed discipline** — `run_paired_monte_carlo` uses a master RNG to generate per-sim seeds. Each seed is used identically for both before/after rosters. NEVER use different seeds for before vs after — this breaks the variance reduction.
- **MC roster stats use string keys** — `build_roster_stats()` converts int player_ids to str keys (e.g., `"1"`, `"42"`). This avoids int/str key mismatches between the pool DataFrame and the MC simulator.
- **Copula inverse categories** — ERA/WHIP correlations in `DEFAULT_CORRELATION` are already negative against W/K (e.g., ERA↔K = -0.65). In `sample_correlated_stats()`, inverse categories use `ppf(1-u)` to flip the quantile direction.
- **`ruff` command on Windows** — Use `python -m ruff check .` and `python -m ruff format .` instead of bare `ruff` — the binary may not be on PATH.
- **Statcast requires pybaseball** — `PYBASEBALL_AVAILABLE` flag in `statcast.py`. Functions return empty results when pybaseball not installed. All rolling feature computation works without it (just no raw data to aggregate).
- **Signal decay λ=0 means no decay** — `decay_weight()` returns 1.0 when `lambda_param=0`. Used for counting stats (HR, RBI) that don't lose relevance over time.
- **Kalman observation variance is sample-size-aware** — `observation_variance("ba", 20)` >> `observation_variance("ba", 500)`. This is critical: early-season observations with 20 PA get high noise → filter trusts prior. Late-season with 500 PA → filter trusts data.
- **BOCPD cp_prob is bounded by 1/hazard_lambda** — The raw `P(r_t=0)` returned by `BOCPD.update()` can never exceed ~`1/hazard_lambda` (e.g., 0.005 for λ=200). The real detection signal is the run length distribution's **mode** dropping from a long run (mode ≈ 30) to near zero (mode ≤ 2). `detect_changepoints()` uses this mode-drop approach.
- **HMM graceful fallback** — `fit_player_hmm()` returns `(None, DEFAULT_STATE_PROBS)` when: (a) hmmlearn not installed, (b) fewer than 10 observations, or (c) fitting raises an exception. Always usable.
- **`classify_regime_simple()` is the fallback** — When no Statcast data exists for HMM, this rule-based function uses just `recent_xwoba` and `season_xwoba` to estimate regime probabilities. Good enough for non-Savant players.
- **`enable_context=True` activates Phase 4** — By default, `evaluate_trade()` runs with context analysis enabled. Set `enable_context=False` to skip concentration risk and enhanced bench value. Phase 4 gracefully wraps in try/except with fallback to Phase 1 behavior.
- **Log5 rate clamping** — `RATE_FLOOR=0.001`, `RATE_CEILING=0.999` prevent division-by-zero in the odds-ratio formula. Applied via `np.clip()` to all three inputs (batter, pitcher, league avg).
- **`matchup_adjustment_factor()` returns 1.0 gracefully** — When `player_xwoba is None`, returns 1.0 (neutral). When `opponent_xwoba_against is None`, returns just `park_factor`. Never crashes with missing schedule data.
- **Weibull frailty cap at 2.0** — `frailty_from_health_score()` uses `1/max(hs, 0.5)`. Health scores below 0.5 are floored, so frailty never exceeds 2.0× normal duration. This prevents unrealistically long injury predictions.
- **HHI concentration threshold is 0.15** — Penalty only applies when `HHI > 0.15`. A perfectly diversified 10-team roster has HHI=0.1. The `scale=3.0` parameter maps `HHI=0.25` to a 0.3 SGP penalty — meaningful but not dominant.
- **Team alias normalization** — `TEAM_ALIASES` maps 8 common variants (WSN→WSH, AZ→ARI, CHW→CWS, etc.). Applied in `roster_concentration_hhi()` before grouping. Without this, "WSN" and "WSH" are counted as separate teams.
- **`concentration_risk_penalty` applied to `total_surplus`** — The penalty is subtracted from the trade surplus AFTER all other SGP calculations. This means concentration risk can flip a marginal "Accept" to "Reject" but won't dominate large surplus trades.
- **Enhanced bench value backward compat** — When `enable_context=False`, trade evaluator uses the simple `bench_option_value()` from `src/engine/portfolio/lineup_optimizer.py`. Both produce compatible float results.
- **Injury process imports from `src/injury_model.py`** — `injury_process.py` reuses `age_risk_adjustment()` from the existing module rather than duplicating the age-risk curve logic.
- **`enable_game_theory=True` activates Phase 5** — By default, `evaluate_trade()` runs game theory analysis. Set `enable_game_theory=False` to skip adverse selection, market values, and sensitivity. Phase 5 wraps in try/except for graceful fallback.
- **Market clearing price = second-highest bid** — Vickrey auction principle: the fair trade price is what the second-most-interested team would pay, not the top bidder. `market_clearing_price()` sorts valuations descending and returns index [1].
- **Adverse selection MAX_DISCOUNT = 0.25** — The discount factor never goes below 0.75 (1.0 - 0.25). Even with the worst possible manager history, received player value is reduced by at most 25%. This prevents the model from overreacting to small sample sizes.
- **Adverse selection needs ≥3 trades** — `MIN_HISTORY_FOR_CALIBRATION = 3`. With fewer trades, falls back to default prior (P_flaw=0.15). This prevents volatile estimates from 1-2 data points.
- **Bellman rollout uses paired scenarios** — For each sim, BOTH "no trade" and "trade" scenarios are evaluated with the same RNG state, ensuring variance reduction identical to the paired MC technique in Phase 2.
- **`compute_roster_balance()` normalizes by median rank** — Deviation is `|rank - median| / median`, not raw difference. This makes the score invariant to league size.
- **Sensitivity `player_sensitivity()` calls `evaluate_fn` with `enable_mc=False, enable_context=False`** — To avoid expensive MC/context calls during sensitivity analysis (which may call evaluate_trade N times for N players). Only the deterministic Phase 1 surplus is used for marginal impact computation.
- **Counter-offer `MIN_SWAP_IMPROVEMENT = 0.2`** — Only suggests player swaps that improve trade surplus by at least 0.2 SGP. Prevents suggesting trivial lateral moves.
- **Running mean stability normalizes by sample std, not mean** — Using `std(running_means) / std(samples)` instead of coefficient of variation (`std/mean`) avoids division-by-near-zero for zero-mean distributions (which are common in trade surplus samples centered at 0).
- **ESS via FFT** — `effective_sample_size()` uses `np.fft.fft()` for fast autocorrelation computation. O(N log N) instead of O(N²) for the naive approach. Uses Geyer's initial positive sequence estimator: stops summing ACF when it drops below 0.05.
- **Cache TTL = 0 means immediately stale** — Setting `ttl=0` in `TradeEvalCache.set()` creates an entry that `is_stale` returns True for on the next `.get()`. Used in tests but should never be used in production.
- **`get_trade_cache()` is a module-level singleton** — All trade evaluations within a Streamlit session share the same cache. Call `reset_trade_cache()` to force a full refresh (e.g., after standings update).
- **Adaptive sim scaling caps at 100K** — `compute_adaptive_n_sims()` never returns more than 100K regardless of trade complexity. The minimum is 1K (quick mode). The `time_budget_s` parameter allows capping by estimated runtime.

## GitHub

- **Repo:** https://github.com/hicklax13/fantasy-baseball-draft-tool (public)
- **Current release:** v1.0.0
- **Release workflow:** Auto-creates GitHub releases on `v*.*.*` tags

## Testing Status

- **Unit tests:** 587 collected, 586 passed, 1 skipped (PyMC optional dep)
- **Test files:** 25 test files across draft engine, trade engine (Phase 1-6), in-season, analytics, data pipeline, bootstrap, integration, and math verification
- **Math verification suite:** 162 tests across 4 files (valuation, simulation, trade, trade engine math) — hand-calculated expected values verified against code formulas
- **Trade engine tests:** 207 tests total — Phase 1 (32): marginal SGP, punt detection, z-scores, grading, fuzzy match, integration. Phase 2 (33): BMA, KDE marginals, Gaussian copula, paired MC, correlated sampling, distributional metrics, integration. Phase 3 (32): Statcast aggregation, signal decay, Kalman filter, BOCPD changepoint detection, HMM regime classification, rolling features. Phase 4 (40): Log5 matchup math, Weibull injury duration, frailty, season availability, enhanced bench value, roster flexibility, HHI concentration, penalty thresholds, trade context integration. Phase 5 (38): opponent valuations, market clearing price, adverse selection Bayesian discount, Bellman rollout, roster balance, sensitivity ranking, counter-offers, game theory integration. Phase 6 (32): ESS convergence, split-R̂, running mean stability, cache TTL/invalidation/get_or_compute, adaptive sim scaling, time budget caps.
- **CI:** GitHub Actions runs ruff lint/format + pytest on Python 3.11, 3.12, 3.13
- **Coverage:** 64% (below 75% CI threshold; pre-existing, no regressions)
- **Systematic code reviews:** Three rounds of full codebase review completed (including parallel 7-agent sweep); all bugs fixed and pushed

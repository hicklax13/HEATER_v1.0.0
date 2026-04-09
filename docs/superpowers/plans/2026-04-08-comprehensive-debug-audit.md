# Comprehensive HEATER Codebase Debug & Audit Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:dispatching-parallel-agents to launch 8 parallel review agents, then superpowers:systematic-debugging for fixes, then superpowers:verification-before-completion for final validation.

**Goal:** Deep audit of the entire 120K-line HEATER codebase (129 source files, 11 pages, 140 test files) to find and fix all math errors, silent failures, dead code, integration gaps, and data pipeline bugs.

**Architecture:** 8 parallel review agents each audit a vertical slice (math, trade engine, optimizer, data pipeline, page wiring, silent failures, test coverage, intelligence models). Findings consolidated, triaged by severity, CRITICAL/HIGH fixed immediately with TDD, then full test suite + lint verification before commit.

**Tech Stack:** Python 3.14, Streamlit, SQLite, PuLP, SciPy, NumPy, pandas, PyMC5 (optional), XGBoost (optional), MLB Stats API, FanGraphs, Yahoo Fantasy API

**Known Bug (from memory):** Trade Analyzer standings bug — `league_standings` table stores W/L records not category totals. `team_stats` can be None early-season. Causes all trades to grade incorrectly. Must verify if still present and fix.

---

## Phase 1: Parallel Audit (8 Agents)

### Agent 1: Math Auditor

**Files to read:**
- `src/valuation.py` — SGP calculator, replacement levels, VORP, LeagueConfig
- `src/standings_engine.py` — Bayesian category probabilities, MC season sim
- `src/optimizer/sgp_theory.py` — Non-linear marginal SGP
- `src/standings_projection.py` — MC season simulation (legacy)
- `src/power_rankings.py` — 5-factor power rankings

**Check for these specific bugs:**
- [ ] **Rate-stat aggregation** — AVG must be `sum(h)/sum(ab)`, NOT `mean(avg)`. OBP must be `sum(h+bb+hbp)/sum(ab+bb+hbp+sf)`. ERA must be `sum(er)*9/sum(ip)`. WHIP must be `sum(bb+h)/sum(ip)`. Any simple average of rate stats is a CRITICAL bug.
- [ ] **Inverse stat sign-flip** — L, ERA, WHIP must have negative SGP coefficients. Check `compute_sgp_denominators()` handles these correctly.
- [ ] **VORP graduated scarcity** — C=1.20x, 2B=1.15x, SS=1.10x. Verify multipliers match CLAUDE.md spec.
- [ ] **pick_score composition** — Must be weighted SGP + positional scarcity + VORP bonus. Check weights and formula.
- [ ] **Bayesian SGP updating** — Priors must use league-specific denominators. Check prior sensibility.
- [ ] **Replacement level computation** — Verify `compute_replacement_levels()` uses correct position pool sizes.
- [ ] **Power rankings bootstrap CI** — Check 5 factors are correctly weighted.
- [ ] **MC season simulation** — Check simulation uses proper category projections, not just W/L extrapolation.

**Report format:** Each finding as `SEVERITY: file:line — description — expected vs actual`

---

### Agent 2: Trade Engine Auditor

**Files to read:**
- `src/engine/portfolio/` — All files (Z-scores, SGP, category analysis, lineup optimizer, copula)
- `src/engine/projections/` — ROS projections, BMA, KDE
- `src/engine/monte_carlo/` — Paired MC simulator
- `src/engine/signals/` — Statcast, decay, Kalman, regime
- `src/engine/context/` — Log5, injury, bench, concentration risk
- `src/engine/game_theory/` — Opponent valuation, adverse selection, Bellman DP
- `src/engine/production/` — Convergence, cache, adaptive sim
- `src/engine/output/` — Master trade orchestrator
- `src/trade_finder.py` — V4 trade finder
- `src/trade_intelligence.py` — Health/scarcity/FA gating
- `src/trade_value.py` — Universal value chart
- `src/trade_signals.py` — Kalman + regime
- `src/opponent_trade_analysis.py` — Opponent needs, acceptance

**Check for these specific bugs:**
- [ ] **LP-constrained lineup totals** — Only 18 starters feed SGP. Bench MUST be excluded. Check `build_lineup_totals()`.
- [ ] **Paired MC seed discipline** — Before/after rosters MUST use same random seed. Check `evaluate_trade()`.
- [ ] **Acceptance sigmoid** — Loss aversion parameter must be 1.8. Check `acceptance_probability()`.
- [ ] **Category fit scoring** — Must use user's actual team gaps, not league averages.
- [ ] **Punt detection** — Requires BOTH `gainable_positions == 0 AND rank >= 10`. Check both conditions.
- [ ] **Elite protection floor** — 75% value floor. Check implementation.
- [ ] **Closer scarcity premium** — Applied to saves ONLY, not entire SGP.
- [ ] **Roster cap enforcement** — 23 slots. Check add/drop validation.
- [ ] **Known bug: standings data** — Verify `load_league_standings()` handles missing category totals.

---

### Agent 3: Optimizer Auditor

**Files to read:**
- `src/optimizer/pipeline.py` — Master orchestrator
- `src/optimizer/projections.py` — Enhanced projections
- `src/optimizer/matchup_adjustments.py` — Park, platoon, weather, umpire, catcher, PvB
- `src/optimizer/h2h_engine.py` — H2H category weights
- `src/optimizer/sgp_theory.py` — Non-linear marginal SGP
- `src/optimizer/streaming.py` — Pitcher streaming
- `src/optimizer/scenario_generator.py` — Gaussian copula
- `src/optimizer/dual_objective.py` — H2H/Roto blending
- `src/optimizer/advanced_lp.py` — Maximin, epsilon-constraint, stochastic MIP
- `src/optimizer/category_urgency.py` — Sigmoid urgency
- `src/optimizer/daily_optimizer.py` — DCV scoring
- `src/optimizer/fa_recommender.py` — FA recommendation
- `src/optimizer/pivot_advisor.py` — Mid-week pivots
- `src/optimizer/shared_data_layer.py` — Unified data layer
- `src/lineup_optimizer.py` — PuLP LP solver
- `src/lineup_rl.py` — RL optimizer

**Check for these specific bugs:**
- [ ] **DCV daily scoring** — Verify per-player per-day scoring formula.
- [ ] **Category urgency sigmoid** — k=2 for counting stats, k=3 for rate stats. Check `sigmoid_urgency()`.
- [ ] **IP weighting in LP** — ERA/WHIP MUST be weighted by IP in objective. Without it, 1-IP relievers dominate.
- [ ] **DTD/IL exclusion** — `_il_statuses` must include both "dtd" AND "day-to-day".
- [ ] **Recent form blend** — 20% weight, requires >=7 games in L14. Check threshold.
- [ ] **Weekly scaling** — Counting stats divided by 26 weeks. Rate stats UNCHANGED.
- [ ] **Matchup data before state classification** — `yds.get_matchup()` must be called BEFORE `classify_matchup_state()`.
- [ ] **Streaming composite score** — Verify all factors included.
- [ ] **Punt mode zeroing** — Punted categories should get zero weight in LP.
- [ ] **Dual objective alpha** — Check blending formula.
- [ ] **Maximin LP** — Must EXCLUDE inverse cats from maximin objective.

---

### Agent 4: Data Pipeline Auditor

**Files to read:**
- `src/data_bootstrap.py` — 21-phase bootstrap
- `src/data_pipeline.py` — FanGraphs auto-fetch
- `src/live_stats.py` — MLB Stats API
- `src/yahoo_api.py` — Yahoo OAuth integration
- `src/yahoo_data_service.py` — 3-tier cache
- `src/espn_injuries.py` — ESPN injury feed
- `src/game_day.py` — Weather, opposing pitchers
- `src/adp_sources.py` — ADP aggregation
- `src/depth_charts.py` — FanGraphs depth charts
- `src/ecr.py` — 6-source ECR consensus
- `src/database.py` — SQLite schema

**Check for these specific bugs:**
- [ ] **21-phase bootstrap** — Verify all 21 phases exist and staleness thresholds match CLAUDE.md (1h live stats/news, 30min Yahoo, 2h game-day, 24h projections/ADP/ECR, 7d players/prospects).
- [ ] **429 backoff** — Check `_request_with_backoff()` retries 3x with exponential delay.
- [ ] **Python 3.14 SQLite bytes** — Integer columns may return bytes. Check for `CAST` in SQL and `pd.to_numeric()`.
- [ ] **Ghost player filtering** — Yahoo traded players with `position:None` must be filtered.
- [ ] **FanGraphs SYSTEM_MAP** — FG API uses `"fangraphsdc"`, DB stores `"depthcharts"`. Check mapping.
- [ ] **Connection leak** — Every `get_connection()` must be in `try/finally` with `conn.close()`.
- [ ] **Write-through cache** — Every Yahoo fetch must write to SQLite.
- [ ] **TTL enforcement** — Check all TTLs match spec.
- [ ] **Empty response handling** — API returns empty list/None. Check graceful handling.
- [ ] **Data sources completeness** — Verify every source in CLAUDE.md data table is fetched and populated.

---

### Agent 5: Page Wiring Auditor

**Files to read:**
- `app.py` — Main app
- `pages/1_My_Team.py`
- `pages/2_Draft_Simulator.py`
- `pages/3_Trade_Analyzer.py`
- `pages/4_Free_Agents.py`
- `pages/5_Line-up_Optimizer.py`
- `pages/6_Player_Compare.py`
- `pages/7_Closer_Monitor.py`
- `pages/8_League_Standings.py`
- `pages/9_Leaders.py`
- `pages/10_Trade_Finder.py`
- `pages/11_Matchup_Planner.py`

**Check for these specific bugs:**
- [ ] **Unified service usage** — Every page MUST use:
  - `MatchupContextService` for category weights (not local computation)
  - `SGPCalculator` for SGP (not local `_totals_sgp()`)
  - `standings_utils.get_team_totals()` for roster totals
  - `get_fa_pool()` for FA access
  - `format_stat()` for stat display
- [ ] **Local reimplementations** — Flag ANY local SGP, category weight, or roster total computation.
- [ ] **Session state key conflicts** — Check for duplicate keys across pages.
- [ ] **Connection leaks** — Check every `get_connection()` has matching `conn.close()`.
- [ ] **Empty state handling** — What happens when Yahoo is disconnected? When DB is empty?
- [ ] **Data flow per tab** — Verify correct data sources per UI tab.
- [ ] **Error states** — Check every page has proper error handling for missing data.

---

### Agent 6: Silent Failure Hunter

**Files to read:** All `src/*.py` and `pages/*.py`

**Check for these specific patterns:**
- [ ] **Empty except blocks** — `except: pass` or `except Exception: pass`
- [ ] **Bare except** — `except:` catching everything including SystemExit/KeyboardInterrupt
- [ ] **Fallback values masking errors** — Returning 0.0, empty dict, or empty DataFrame on failure
- [ ] **Swallowed HTTP errors** — API calls that catch and ignore HTTP status codes
- [ ] **Critical math in try/except** — SGP, VORP, trade value calculations that silently return wrong results
- [ ] **Optional imports disabling features** — `PYMC_AVAILABLE`, `PULP_AVAILABLE` etc. that fail silently without logging
- [ ] **Empty DataFrame returns** — DB queries returning empty results without warning
- [ ] **Default parameter gotchas** — Mutable default arguments (list/dict)

---

### Agent 7: Test Coverage Auditor

**Files to read:** `tests/` directory listing + sample test files for coverage patterns

**Check for these specific gaps:**
- [ ] **Zero-coverage modules** — Which `src/` modules have NO corresponding test file?
- [ ] **Mock-only tests** — Tests that mock so heavily they only test the mock, not real logic.
- [ ] **Hardcoded expected values** — Tests with magic numbers that may be wrong.
- [ ] **Overly loose tolerances** — `pytest.approx` with `rel=0.5` or similar.
- [ ] **Critical untested paths** — The 10 highest-risk untested code paths.
- [ ] **Dead test code** — Tests that are skipped but not for optional dependencies.
- [ ] **Rate-stat math coverage** — Are the weighted-average formulas tested?
- [ ] **Edge case coverage** — Empty rosters, single-player trades, zero-IP pitchers.

---

### Agent 8: Intelligence & Model Auditor

**Files to read:**
- `src/ecr.py` — 6-source ECR consensus
- `src/ml_ensemble.py` — XGBoost ensemble
- `src/bayesian.py` — PyMC hierarchical model
- `src/injury_model.py` — Health scores, aging curves
- `src/playing_time_model.py` — Ridge regression PT
- `src/projection_stacking.py` — Ridge regression projection weighting
- `src/prospect_engine.py` — Prospect call-ups
- `src/closer_monitor.py` — Job security scoring
- `src/war_room.py` — Mid-week pivot analysis
- `src/matchup_context.py` — MatchupContextService
- `src/game_day.py` — Weather, opposing pitchers

**Check for these specific issues:**
- [ ] **Statcast features** — Are EV, barrel%, xwOBA, Stuff+, sprint speed actually populated in DB AND used in scoring?
- [ ] **XGBoost ensemble** — Is it wired end-to-end or dead code?
- [ ] **Bayesian priors** — Are stabilization points per stat sensible?
- [ ] **ECR temporal weights** — 6-source consensus using correct temporal weighting?
- [ ] **Weather adjustments** — Wind direction, rain, temperature wired into DCV/optimizer?
- [ ] **Catcher framing** — Fetched and used?
- [ ] **Umpire tendencies** — Wired to scoring?
- [ ] **Injury position-specific aging** — Using correct curves per position?
- [ ] **Prospect call-up probabilities** — Realistic?
- [ ] **Closer job security gmLI** — Correct formula?

---

## Phase 2: Triage & Consolidation

- [ ] **Step 1:** Collect all 8 agent reports
- [ ] **Step 2:** Deduplicate findings (same bug found by multiple agents)
- [ ] **Step 3:** Sort by severity: CRITICAL > HIGH > MEDIUM > LOW
- [ ] **Step 4:** Create consolidated findings list with file:line references
- [ ] **Step 5:** Estimate fix effort per finding

---

## Phase 3: Fix CRITICAL & HIGH Issues

For each CRITICAL/HIGH finding:
- [ ] **Step 1:** Read the affected code
- [ ] **Step 2:** Write a failing test that exposes the bug
- [ ] **Step 3:** Run the test to confirm it fails
- [ ] **Step 4:** Fix the bug with minimal change
- [ ] **Step 5:** Run the test to confirm it passes
- [ ] **Step 6:** Run `python -m pytest -x -q` for regression check
- [ ] **Step 7:** Run `python -m ruff check . && python -m ruff format .`
- [ ] **Step 8:** Commit with conventional commit message

---

## Phase 4: Final Verification

- [ ] **Step 1:** Run full test suite: `python -m pytest -q`
- [ ] **Step 2:** Run linter: `python -m ruff check .`
- [ ] **Step 3:** Run formatter: `python -m ruff format --check .`
- [ ] **Step 4:** Run PR review toolkit as final pass
- [ ] **Step 5:** Push to master

---

## Severity Definitions

| Severity | Definition | Action |
|----------|-----------|--------|
| CRITICAL | Wrong math producing wrong results shown to user | Fix immediately |
| HIGH | Dead code path, data not integrated, silent failure masking errors | Fix immediately |
| MEDIUM | Suboptimal logic, missing edge case, loose tolerance | Document for future |
| LOW | Code quality, naming, minor improvements | Document for future |

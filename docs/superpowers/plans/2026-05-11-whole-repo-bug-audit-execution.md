# Whole-Repo Bug & Data Audit — Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dispatch 24 parallel agents to audit the entire HEATER codebase + every consumer-facing data point + the live-refresh infrastructure, then aggregate all findings into a single report.

**Architecture:** Three parallel streams (code review × 22 agents, data correctness × 1, refresh infrastructure × 1) → all read-only, dispatched in waves, aggregated by a final consolidation agent into `docs/superpowers/specs/2026-05-11-bug-audit-findings.md`.

**Tech Stack:** Claude Code Agent tool with worktree isolation, MLB Stats API (`statsapi`), Yahoo Fantasy API (via `yahoo_data_service`), git, gh CLI, pytest.

---

## Cold-Start Context (READ THIS FIRST)

You are a fresh session with no memory of prior conversations. Here's everything you need:

### Project state
- **Repo:** `C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0` (Windows). Master branch is at PR #11 merged 2026-05-11T17:18:18 UTC.
- **Working from worktree:** `C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0/.claude/worktrees/cranky-einstein-07be9b` (branch `claude/cranky-einstein-07be9b`, currently identical to master after PR #11 merge).
- **Live database:** `C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0/data/draft_tool.db` (SQLite, do NOT delete or write to it directly during audit).
- **Today:** 2026-05-11. MLB 2026 season is active (~5-6 weeks in); ~30 games per active player. App is in in-season mode.

### Project context (just enough to brief reviewer agents)
- HEATER is a fantasy-baseball draft + in-season manager for Yahoo's FourzynBurn 12-team H2H Categories league.
- Categories: hitting=R/HR/RBI/SB/AVG/OBP, pitching=W/L/SV/K/ERA/WHIP. Inverse: L/ERA/WHIP. Rate: AVG/OBP/ERA/WHIP.
- 28 roster slots: C/1B/2B/3B/SS/3OF/2Util/2SP/2RP/4P/6BN/4IL.
- Yahoo game_key: 469.

### Canonical sources of truth in the codebase (reviewers must know these)
- `src/valuation.py::LeagueConfig` — all category lists. NEVER hardcode categories.
- `src/valuation.py::SGPCalculator.totals_sgp(totals, weights)` — sole SGP path.
- `src/database.py::get_connection()` — only sanctioned SQLite access (sets WAL + busy_timeout).
- `src/yahoo_data_service.py::get_yahoo_data_service()` — singleton, 3-tier cache.
- `src/matchup_context.py::get_matchup_context()` — unified category weights.
- `src/standings_utils.py::get_team_totals()` / `get_all_team_totals()` — session-cached.
- `src/ui_shared.py::format_stat(value, stat_type)` — display formatter.
- `src/optimizer/constants_registry.py::CONSTANTS_REGISTRY` — runtime-readable optimizer constants.

### Already-resolved historical context (don't re-find these)
- 28 silent-fail patterns (SF-1..SF-28) have already been resolved across PRs #7-#11 in 2026-05-10/11. Reviewers should NOT report these as new findings unless they spot a regression.
- 16 structural-invariant tests guard against regression: see `tests/test_no_*.py`, `tests/test_pages_*.py`, `tests/test_app_no_hardcoded_categories.py`, `tests/test_my_team_no_hardcoded_categories.py`, `tests/test_sensitivity_categories_canonical.py`, `tests/test_engine_no_fallback_singletons.py`, `tests/test_pulp_availability_consolidation.py`, `tests/test_opponent_valuation_no_default_denoms.py`, `tests/test_pool_ytd_columns_complete.py`, `tests/test_standings_adapter_contract.py`, `tests/test_no_merge_conflict_markers.py`, `tests/test_refresh_log_status_validity.py`, `tests/test_refresh_log_snapshot.py`.
- Full project context: `CLAUDE.md` at repo root.

### The spec this plan implements
Read `docs/superpowers/specs/2026-05-11-whole-repo-bug-audit-design.md` first if anything below seems ambiguous.

---

## Pre-Flight Checklist (Phase 0 — ~5 min)

- [ ] **Step 0.1: Verify working directory**

Run:
```bash
cd "C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0/.claude/worktrees/cranky-einstein-07be9b"
git status
git log -1 --oneline
```
Expected: Clean working tree, recent commits include `7d2881a docs(specs): whole-repo bug & data audit design (3 streams, 24 agents)`. If `7d2881a` is not present, you are on the wrong branch — stop and re-check.

- [ ] **Step 0.2: Verify gh + git auth**

Run:
```bash
gh auth status
gh pr list --limit 1
```
Expected: Authenticated as `hicklax13`; able to list PRs. If not, fix auth before proceeding (the audit doesn't push but logs reference PRs).

- [ ] **Step 0.3: Verify pytest baseline**

Run:
```bash
python -m pytest --collect-only -q --ignore=tests/test_cheat_sheet.py 2>&1 | tail -5
```
Expected: ~3700 tests collected, no collection errors. If collection fails, stop and fix.

- [ ] **Step 0.4: Read the spec**

Run:
```bash
cat docs/superpowers/specs/2026-05-11-whole-repo-bug-audit-design.md
```
Confirm the 8-domain breakdown and 3-stream structure match what's in this plan's tasks.

- [ ] **Step 0.5: Initialize TodoWrite with all 26 agent tasks**

Use the TodoWrite tool to create todos for: 22 Stream A agents (Tasks 1.1 through 1.22), 1 Stream B agent (Task 2), 1 Stream C agent (Task 3), 1 aggregator (Task 4), 1 commit/push (Task 5).

---

## Phase 1: Stream A Code Review (Tasks 1.1 — 1.22, parallel)

### Reviewer Prompt Template

**Every Stream A agent uses this prompt template, with `{DOMAIN_NAME}`, `{DOMAIN_FILES}`, and `{REVIEWER_FOCUS}` filled in per task. The template is verbatim except for those three substitutions.**

```text
You are a read-only code-review agent for the HEATER fantasy baseball app. NEVER edit files or commit. Worktree is at C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0/.claude/worktrees/cranky-einstein-07be9b. Live DB at C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0/data/draft_tool.db (read-only queries OK).

PROJECT CONTEXT:
HEATER is a fantasy baseball app for Yahoo's FourzynBurn 12-team H2H Categories league (game_key 469). Today is 2026-05-11; MLB 2026 season is active. Categories: hitting=R/HR/RBI/SB/AVG/OBP, pitching=W/L/SV/K/ERA/WHIP. Inverse: L/ERA/WHIP (lower is better). Rate stats: AVG/OBP/ERA/WHIP. Roster: 28 slots.

CANONICAL SOURCES OF TRUTH (do NOT flag findings that violate these as bugs unless the violation is in the file under review):
- LeagueConfig (src/valuation.py) — all category lists
- SGPCalculator.totals_sgp (src/valuation.py) — sole SGP computation
- get_connection (src/database.py) — only sanctioned SQLite access
- get_yahoo_data_service (src/yahoo_data_service.py) — Yahoo singleton
- get_matchup_context (src/matchup_context.py) — unified category weights
- format_stat (src/ui_shared.py) — display formatter
- CONSTANTS_REGISTRY (src/optimizer/constants_registry.py) — runtime-readable constants

PRE-EXISTING RESOLVED ISSUES (do NOT report these as new findings):
- 28 silent-fail patterns SF-1..SF-28 are resolved (see CLAUDE.md "Data Audit History").
- 16 structural-invariant tests in tests/test_no_*.py / test_pages_*.py guard against regression.

YOUR DOMAIN: {DOMAIN_NAME}
YOUR FILES (review ONLY these — do not drift to other files):
{DOMAIN_FILES}

YOUR FOCUS:
{REVIEWER_FOCUS}

REVIEW PROCESS:
1. Read each file in your scope end-to-end.
2. For each file, look for issues matching your focus.
3. For each issue found, capture: file:line, severity (HIGH/MEDIUM/LOW), category, evidence, suggested fix, your confidence (HIGH/MEDIUM/LOW that this is a real issue).
4. Cross-reference suspicious patterns with the canonical sources above (e.g., if you see hardcoded categories, check whether LeagueConfig is being bypassed; if you see direct sqlite3.connect, that's a real finding because get_connection is the canonical path).
5. Use Bash to query the live DB for evidence when relevant (sqlite3 read-only).

OUTPUT FORMAT (return as your final message):

## REVIEW REPORT — {DOMAIN_NAME} ({REVIEWER_TYPE})

### Files reviewed: <count>

### Findings (sorted by severity HIGH → LOW)

| ID | File:Line | Severity | Category | Confidence | Issue |
|----|-----------|----------|----------|------------|-------|
| <DOMAIN_PREFIX>-001 | src/foo.py:42 | HIGH | logic | HIGH | <one-line summary> |
...

### Detailed findings

#### <DOMAIN_PREFIX>-001 — <one-line summary>
- **File:** src/foo.py:42
- **Severity:** HIGH
- **Category:** logic
- **Confidence:** HIGH
- **Evidence:**
  ```python
  <code excerpt>
  ```
- **Why it's a bug:** <2-3 sentences>
- **Suggested fix:** <one sentence>

(Repeat for each finding)

### Patterns observed (1-2 paragraphs)
<Optional: any cross-cutting patterns you noticed>

CONSTRAINTS:
- READ-ONLY. No file edits. No git commits.
- Stay STRICTLY within your assigned files. Don't review code outside your scope.
- Don't duplicate findings already in the SF-1..SF-28 catalog (read CLAUDE.md if uncertain).
- Cap report at ~1500 words. Use tables for the summary, prose only for the patterns paragraph.
- Be honest about confidence — LOW confidence findings are still useful but should be marked.
```

### Reviewer-type focus blocks

The `{REVIEWER_FOCUS}` substitution differs per reviewer type:

**FOCUS A — Logic & correctness (use for `feature-dev:code-reviewer` agents):**
```text
Find logic and correctness bugs:
- Wrong math (off-by-one, missing edge case in formulas, wrong sign, wrong precedence)
- State management bugs (uninitialized state, race conditions, mutable shared state)
- Control-flow errors (early-return missing, unreachable code, fallthrough where break expected)
- Incorrect API usage (wrong arguments, wrong return-shape assumptions, missing await)
- Resource leaks (unclosed connections/files, unreleased locks)
- Data integrity (division by zero, integer overflow, encoding issues)
Secondary: catch obvious security issues (SQL injection via string-interpolation, secrets in logs, eval/exec on user input) and obvious performance issues (N+1 queries, hot-loop allocations, unbounded growth) as side-effects.
```

**FOCUS B — Silent failures (use for `pr-review-toolkit:silent-failure-hunter` agents):**
```text
Find silent failures — code that "succeeds" while producing wrong or missing data. Patterns to look for:
- try/except that swallows errors and returns a default ("looks fine" but data is wrong)
- Status messages saying "success" when row count is 0 or much less than expected
- DataFrames built from queries that may return empty silently
- API calls without status-code checking
- Bootstrap phases that update refresh_log without verifying the underlying table populated
- Helper functions that return neutral defaults when inputs are missing (these can be correct OR a silent miscategorization — flag both)
- Joins/merges that drop rows without warning
- Type coercions that turn None/NaN into 0 (data lost vs. data missing)
Reference: SF-1..SF-28 in CLAUDE.md are examples of this class. Don't re-report those, but use them as your pattern library — find NEW instances.
```

**FOCUS D — Type & API design (use for `pr-review-toolkit:type-design-analyzer` agents):**
```text
Find type and API design issues:
- Functions accepting `dict` or `Any` where a typed dataclass would prevent bugs
- Optional parameters that secretly aren't optional (will error when None passed)
- Inconsistent return types (sometimes returns DataFrame, sometimes dict, sometimes None)
- Leaky abstractions (caller must know internal column names that aren't in the public API)
- Function signatures that take 5+ positional args (refactor to dataclass)
- Mixed concerns (function name says X but it also does Y silently)
- Implicit shared state (module-level singletons that callers must initialize)
- Inheritance/composition smells in the engine sub-packages
- Missing type hints on functions whose return type matters for correctness
- Mutable default arguments (def foo(x=[]))
```

### Domain × Reviewer matrix (22 tasks)

For each Task 1.N below, dispatch ONE agent with `subagent_type: general-purpose`, `isolation: worktree`, and the prompt template above with the listed substitutions.

**ALL 22 AGENTS RUN IN PARALLEL.** Send them in two messages: 11 agents per message (to avoid overwhelming dispatch). Wait for all 22 to return before proceeding to Phase 2.

#### Task 1.1: Domain 1 — Data Pipeline × FOCUS A (logic & correctness)

- [ ] **Dispatch agent with these substitutions:**
  - `{DOMAIN_NAME}`: Data Pipeline
  - `{REVIEWER_TYPE}`: code-reviewer
  - `{DOMAIN_PREFIX}`: D1A
  - `{REVIEWER_FOCUS}`: (FOCUS A above)
  - `{DOMAIN_FILES}`:
    ```
    src/database.py
    src/data_bootstrap.py
    src/data_pipeline.py
    src/data_fetch_utils.py
    src/data_2026.py
    src/live_stats.py
    src/depth_charts.py
    src/contract_data.py
    src/marcel.py
    src/adp_sources.py
    src/player_databank.py
    src/espn_injuries.py
    src/news_fetcher.py
    src/news_sentiment.py
    src/player_news.py
    src/prospect_engine.py
    src/league_manager.py
    src/league_registry.py
    src/league_rules.py
    src/scheduler.py
    src/points_league.py
    src/analytics_context.py
    src/projection_blending.py
    src/trend_tracker.py
    benchmark_load_times.py
    debug_yahoo.py
    ```

#### Task 1.2: Domain 1 — Data Pipeline × FOCUS B (silent failures)

- [ ] **Dispatch agent with these substitutions:**
  - `{DOMAIN_NAME}`: Data Pipeline
  - `{REVIEWER_TYPE}`: silent-failure-hunter
  - `{DOMAIN_PREFIX}`: D1B
  - `{REVIEWER_FOCUS}`: (FOCUS B above)
  - `{DOMAIN_FILES}`: (same list as Task 1.1)

#### Task 1.3: Domain 1 — Data Pipeline × FOCUS D (type & API design)

- [ ] **Dispatch agent with these substitutions:**
  - `{DOMAIN_NAME}`: Data Pipeline
  - `{REVIEWER_TYPE}`: type-design-analyzer
  - `{DOMAIN_PREFIX}`: D1D
  - `{REVIEWER_FOCUS}`: (FOCUS D above)
  - `{DOMAIN_FILES}`: (same list as Task 1.1)

#### Task 1.4: Domain 2 — Yahoo + Valuation Core × FOCUS A

- [ ] **Dispatch agent with:**
  - `{DOMAIN_NAME}`: Yahoo + Valuation Core
  - `{REVIEWER_TYPE}`: code-reviewer
  - `{DOMAIN_PREFIX}`: D2A
  - `{REVIEWER_FOCUS}`: (FOCUS A above)
  - `{DOMAIN_FILES}`:
    ```
    src/yahoo_api.py
    src/yahoo_data_service.py
    src/live_draft_sync.py
    src/valuation.py
    src/ecr.py
    src/projection_stacking.py
    src/playing_time_model.py
    src/ml_ensemble.py
    src/bayesian.py
    src/injury_model.py
    src/validation.py
    src/validation/__init__.py
    src/validation/calibration_data.py
    src/validation/constant_optimizer.py
    src/validation/dynamic_context.py
    src/validation/module_triage.py
    src/validation/survival_calibrator.py
    calibrate_constants.py
    ```

#### Task 1.5: Domain 2 — Yahoo + Valuation Core × FOCUS B

- [ ] **Dispatch agent with:**
  - `{DOMAIN_NAME}`: Yahoo + Valuation Core
  - `{REVIEWER_TYPE}`: silent-failure-hunter
  - `{DOMAIN_PREFIX}`: D2B
  - `{REVIEWER_FOCUS}`: (FOCUS B above)
  - `{DOMAIN_FILES}`: (same list as Task 1.4)

#### Task 1.6: Domain 2 — Yahoo + Valuation Core × FOCUS D

- [ ] **Dispatch agent with:**
  - `{DOMAIN_NAME}`: Yahoo + Valuation Core
  - `{REVIEWER_TYPE}`: type-design-analyzer
  - `{DOMAIN_PREFIX}`: D2D
  - `{REVIEWER_FOCUS}`: (FOCUS D above)
  - `{DOMAIN_FILES}`: (same list as Task 1.4)

#### Task 1.7: Domain 3 — Optimizer × FOCUS A

- [ ] **Dispatch agent with:**
  - `{DOMAIN_NAME}`: Optimizer
  - `{REVIEWER_TYPE}`: code-reviewer
  - `{DOMAIN_PREFIX}`: D3A
  - `{REVIEWER_FOCUS}`: (FOCUS A above)
  - `{DOMAIN_FILES}`:
    ```
    src/optimizer/__init__.py
    src/optimizer/advanced_lp.py
    src/optimizer/backtest_runner.py
    src/optimizer/backtest_validator.py
    src/optimizer/category_urgency.py
    src/optimizer/constants_registry.py
    src/optimizer/daily_optimizer.py
    src/optimizer/data_freshness.py
    src/optimizer/dual_objective.py
    src/optimizer/fa_recommender.py
    src/optimizer/h2h_engine.py
    src/optimizer/matchup_adjustments.py
    src/optimizer/pipeline.py
    src/optimizer/pivot_advisor.py
    src/optimizer/projections.py
    src/optimizer/scenario_generator.py
    src/optimizer/sensitivity_analysis.py
    src/optimizer/sgp_theory.py
    src/optimizer/shared_data_layer.py
    src/optimizer/sigmoid_calibrator.py
    src/optimizer/streaming.py
    src/lineup_optimizer.py
    src/lineup_rl.py
    load_sample_data.py
    ```

#### Task 1.8: Domain 3 — Optimizer × FOCUS B

- [ ] **Dispatch agent with:**
  - `{DOMAIN_NAME}`: Optimizer
  - `{REVIEWER_TYPE}`: silent-failure-hunter
  - `{DOMAIN_PREFIX}`: D3B
  - `{REVIEWER_FOCUS}`: (FOCUS B above)
  - `{DOMAIN_FILES}`: (same list as Task 1.7)

#### Task 1.9: Domain 3 — Optimizer × FOCUS D

- [ ] **Dispatch agent with:**
  - `{DOMAIN_NAME}`: Optimizer
  - `{REVIEWER_TYPE}`: type-design-analyzer
  - `{DOMAIN_PREFIX}`: D3D
  - `{REVIEWER_FOCUS}`: (FOCUS D above)
  - `{DOMAIN_FILES}`: (same list as Task 1.7)

#### Task 1.10: Domain 4 — Trade Engine × FOCUS A

- [ ] **Dispatch agent with:**
  - `{DOMAIN_NAME}`: Trade Engine
  - `{REVIEWER_TYPE}`: code-reviewer
  - `{DOMAIN_PREFIX}`: D4A
  - `{REVIEWER_FOCUS}`: (FOCUS A above)
  - `{DOMAIN_FILES}`:
    ```
    src/engine/__init__.py
    src/engine/context/__init__.py
    src/engine/context/bench_value.py
    src/engine/context/concentration.py
    src/engine/context/injury_process.py
    src/engine/context/matchup.py
    src/engine/game_theory/__init__.py
    src/engine/game_theory/opponent_valuation.py
    src/engine/game_theory/sensitivity.py
    src/engine/monte_carlo/__init__.py
    src/engine/monte_carlo/trade_simulator.py
    src/engine/output/__init__.py
    src/engine/output/trade_evaluator.py
    src/engine/portfolio/__init__.py
    src/engine/portfolio/category_analysis.py
    src/engine/portfolio/copula.py
    src/engine/portfolio/lineup_optimizer.py
    src/engine/portfolio/valuation.py
    src/engine/production/__init__.py
    src/engine/production/cache.py
    src/engine/production/convergence.py
    src/engine/production/sim_config.py
    src/engine/projections/__init__.py
    src/engine/projections/bayesian_blend.py
    src/engine/projections/marginals.py
    src/engine/projections/projection_client.py
    src/engine/signals/__init__.py
    src/engine/signals/decay.py
    src/engine/signals/kalman.py
    src/engine/signals/regime.py
    src/engine/signals/statcast.py
    src/trade_finder.py
    src/trade_intelligence.py
    src/trade_value.py
    src/trade_signals.py
    ```

#### Task 1.11: Domain 4 — Trade Engine × FOCUS B

- [ ] **Dispatch agent with:**
  - `{DOMAIN_NAME}`: Trade Engine
  - `{REVIEWER_TYPE}`: silent-failure-hunter
  - `{DOMAIN_PREFIX}`: D4B
  - `{REVIEWER_FOCUS}`: (FOCUS B above)
  - `{DOMAIN_FILES}`: (same list as Task 1.10)

#### Task 1.12: Domain 4 — Trade Engine × FOCUS D

- [ ] **Dispatch agent with:**
  - `{DOMAIN_NAME}`: Trade Engine
  - `{REVIEWER_TYPE}`: type-design-analyzer
  - `{DOMAIN_PREFIX}`: D4D
  - `{REVIEWER_FOCUS}`: (FOCUS D above)
  - `{DOMAIN_FILES}`: (same list as Task 1.10)

#### Task 1.13: Domain 5 — In-Season Strategy × FOCUS A

- [ ] **Dispatch agent with:**
  - `{DOMAIN_NAME}`: In-Season Strategy
  - `{REVIEWER_TYPE}`: code-reviewer
  - `{DOMAIN_PREFIX}`: D5A
  - `{REVIEWER_FOCUS}`: (FOCUS A above)
  - `{DOMAIN_FILES}`:
    ```
    src/in_season.py
    src/opponent_trade_analysis.py
    src/waiver_wire.py
    src/matchup_context.py
    src/matchup_planner.py
    src/standings_engine.py
    src/standings_projection.py
    src/standings_utils.py
    src/weekly_h2h_strategy.py
    src/war_room.py
    src/war_room_actions.py
    src/war_room_hotcold.py
    src/start_sit.py
    src/start_sit_widget.py
    src/weekly_report.py
    src/alerts.py
    src/opponent_intel.py
    src/ip_tracker.py
    src/il_manager.py
    src/two_start.py
    src/power_rankings.py
    src/leaders.py
    src/playoff_sim.py
    ```

#### Task 1.14: Domain 5 — In-Season Strategy × FOCUS B

- [ ] **Dispatch agent with:**
  - `{DOMAIN_NAME}`: In-Season Strategy
  - `{REVIEWER_TYPE}`: silent-failure-hunter
  - `{DOMAIN_PREFIX}`: D5B
  - `{REVIEWER_FOCUS}`: (FOCUS B above)
  - `{DOMAIN_FILES}`: (same list as Task 1.13)

#### Task 1.15: Domain 5 — In-Season Strategy × FOCUS D

- [ ] **Dispatch agent with:**
  - `{DOMAIN_NAME}`: In-Season Strategy
  - `{REVIEWER_TYPE}`: type-design-analyzer
  - `{DOMAIN_PREFIX}`: D5D
  - `{REVIEWER_FOCUS}`: (FOCUS D above)
  - `{DOMAIN_FILES}`: (same list as Task 1.13)

#### Task 1.16: Domain 6 — Game Day + Draft + Backtest × FOCUS A

- [ ] **Dispatch agent with:**
  - `{DOMAIN_NAME}`: Game Day + Draft + Backtest
  - `{REVIEWER_TYPE}`: code-reviewer
  - `{DOMAIN_PREFIX}`: D6A
  - `{REVIEWER_FOCUS}`: (FOCUS A above)
  - `{DOMAIN_FILES}`:
    ```
    src/game_day.py
    src/closer_monitor.py
    src/schedule_grid.py
    src/contextual_factors.py
    src/draft_engine.py
    src/draft_state.py
    src/draft_analytics.py
    src/draft_grader.py
    src/draft_order.py
    src/simulation.py
    src/pick_predictor.py
    src/backtesting.py
    src/backtesting_framework.py
    ```

#### Task 1.17: Domain 6 — Game Day + Draft + Backtest × FOCUS B

- [ ] **Dispatch agent with:**
  - `{DOMAIN_NAME}`: Game Day + Draft + Backtest
  - `{REVIEWER_TYPE}`: silent-failure-hunter
  - `{DOMAIN_PREFIX}`: D6B
  - `{REVIEWER_FOCUS}`: (FOCUS B above)
  - `{DOMAIN_FILES}`: (same list as Task 1.16)

#### Task 1.18: Domain 6 — Game Day + Draft + Backtest × FOCUS D

- [ ] **Dispatch agent with:**
  - `{DOMAIN_NAME}`: Game Day + Draft + Backtest
  - `{REVIEWER_TYPE}`: type-design-analyzer
  - `{DOMAIN_PREFIX}`: D6D
  - `{REVIEWER_FOCUS}`: (FOCUS D above)
  - `{DOMAIN_FILES}`: (same list as Task 1.16)

#### Task 1.19: Domain 7 — UI / Pages / Scripts × FOCUS A

- [ ] **Dispatch agent with:**
  - `{DOMAIN_NAME}`: UI / Pages / Scripts
  - `{REVIEWER_TYPE}`: code-reviewer
  - `{DOMAIN_PREFIX}`: D7A
  - `{REVIEWER_FOCUS}`: (FOCUS A above)
  - `{DOMAIN_FILES}`:
    ```
    app.py
    pages/1_My_Team.py
    pages/2_Player_Databank.py
    pages/3_Draft_Simulator.py
    pages/4_Trade_Analyzer.py
    pages/5_Free_Agents.py
    pages/6_Line-up_Optimizer.py
    pages/7_Player_Compare.py
    pages/7_Weekly_Dashboard.py
    pages/8_Closer_Monitor.py
    pages/8_Trade_Values.py
    pages/9_League_Standings.py
    pages/9_Waiver_Wire.py
    pages/10_Category_Tracker.py
    pages/10_Leaders.py
    pages/11_Trade_Finder.py
    pages/11_Trends.py
    pages/12_Matchup_Planner.py
    pages/12_Playoff_Odds.py
    pages/13_Bullpen.py
    pages/14_Punt_Analyzer.py
    pages/15_Weekly_Recap.py
    src/ui_shared.py
    src/ui_analytics_badge.py
    src/cheat_sheet.py
    src/player_card.py
    src/player_tags.py
    scripts/calibrate_sigmoid.py
    scripts/compute_empirical_stats.py
    scripts/draft_vs_current.py
    scripts/extract_trade_data.py
    scripts/install-hooks.py
    scripts/optimal_roster_sim.py
    scripts/run_backtest.py
    ```

#### Task 1.20: Domain 7 — UI / Pages / Scripts × FOCUS B

- [ ] **Dispatch agent with:**
  - `{DOMAIN_NAME}`: UI / Pages / Scripts
  - `{REVIEWER_TYPE}`: silent-failure-hunter
  - `{DOMAIN_PREFIX}`: D7B
  - `{REVIEWER_FOCUS}`: (FOCUS B above)
  - `{DOMAIN_FILES}`: (same list as Task 1.19)

#### Task 1.21: Domain 7 — UI / Pages / Scripts × FOCUS D

- [ ] **Dispatch agent with:**
  - `{DOMAIN_NAME}`: UI / Pages / Scripts
  - `{REVIEWER_TYPE}`: type-design-analyzer
  - `{DOMAIN_PREFIX}`: D7D
  - `{REVIEWER_FOCUS}`: (FOCUS D above)
  - `{DOMAIN_FILES}`: (same list as Task 1.19)

#### Task 1.22: Domain 8 — Config & Data Assets × FOCUS A only

This domain has only ONE reviewer because the content is YAML/TOML/JSON/shell, not Python. Type-design and silent-failure analyses don't apply.

- [ ] **Dispatch agent with:**
  - `{DOMAIN_NAME}`: Config & Data Assets
  - `{REVIEWER_TYPE}`: code-reviewer
  - `{DOMAIN_PREFIX}`: D8A
  - `{REVIEWER_FOCUS}`: For each file type, look for: (a) YAML/CI security issues (secrets in plaintext, overly permissive permissions, missing pinned action versions); (b) requirements.txt issues (unpinned versions, known-vulnerable packages, conflicting transitive deps); (c) JSON seed plausibility (numerical values in realistic 2024 MLB ranges; e.g., catcher framing -15 to +15 runs, umpire k_pct_diff -0.03 to +0.03); (d) shell hook safety (proper quoting, no eval).
  - `{DOMAIN_FILES}`:
    ```
    pyproject.toml
    requirements.txt
    .streamlit/config.toml
    .github/workflows/ci.yml
    .github/workflows/refresh.yml
    data/seed/catcher_framing_2024.json
    data/seed/umpire_tendencies_2024.json
    scripts/pre-commit
    ```

### After all 22 tasks dispatched

- [ ] **Step 1.23: Wait for all 22 Stream A agents to return.** Each returns a structured report. Save each report's full text to memory (you'll need them for Phase 4 aggregation).

- [ ] **Step 1.24: Verify completeness.** For each Task 1.1-1.22, confirm the agent returned a report (not a timeout/failure). For any that failed, RE-DISPATCH with the same prompt — do not skip.

---

## Phase 2: Stream B Player Data Correctness (Task 2)

#### Task 2: Player verification against MLB Stats API + Yahoo

- [ ] **Step 2.1: Dispatch ONE agent with `subagent_type: general-purpose` and the following exact prompt:**

```text
You are a read-only data-verification agent for the HEATER fantasy baseball app. NEVER edit files or commit. Worktree at C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0/.claude/worktrees/cranky-einstein-07be9b. Live DB at C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0/data/draft_tool.db (read-only).

Today: 2026-05-11. MLB 2026 season is ~5-6 weeks in (~30 games per active player).

YOUR TASK: Verify ~1200-1300 unique players' stored data against MLB Stats API + Yahoo Fantasy API.

PLAYER SET (deduplicated union):
1. All league-rostered players (~315):
   SELECT DISTINCT p.player_id, p.name, p.team, p.positions, p.mlb_id, p.is_injured, lr.team_key
   FROM players p JOIN league_rosters lr ON p.player_id = lr.player_id
2. Top 1000 by ADP:
   SELECT p.player_id, p.name, p.team, p.positions, p.mlb_id, p.is_injured, COALESCE(a.adp, 999) AS adp
   FROM players p LEFT JOIN adp a ON p.player_id = a.player_id
   ORDER BY COALESCE(a.adp, 999) ASC LIMIT 1000
3. Top 100 available batters by SGP (Yahoo FA, is_hitter=1):
   Use src.yahoo_data_service.get_yahoo_data_service().get_free_agents(max_players=500),
   filter to is_hitter=1, compute or use existing SGP, take top 100.
4. Top 100 available pitchers by SGP (Yahoo FA, is_hitter=0): same as #3 with is_hitter=0.

After deduplication by player_id (or mlb_id where player_id missing), expect ~1200-1300 unique players.

VERIFICATION (per player, against 2 sources):

SOURCE 1 — MLB Stats API (canonical):
- Use `import statsapi`. For each player with valid mlb_id:
  - Call `statsapi.lookup_player(mlb_id)` (or `statsapi.get('person', {'personId': mlb_id})`) to verify name/team/position/injury status.
  - Call `statsapi.player_stat_data(mlb_id, group='hitting' if is_hitter else 'pitching', type='season', season=2026)` to verify YTD stats.
- Compare DB values to API values:
  - HARD discrepancy (HIGH severity, auto-flag): wrong team, wrong mlb_id, position mismatch beyond known equivalence (1B/3B/Util/OF), name mismatch beyond accent/suffix normalization, IL/active status mismatch.
  - SOFT discrepancy (MEDIUM severity, flag if delta exceeds threshold): YTD counting stats (R/HR/RBI/SB/W/L/SV/K) off by >1; YTD rate stats (AVG/OBP/ERA/WHIP) off by >0.005.

SOURCE 2 — Yahoo Fantasy API:
- Use `from src.yahoo_data_service import get_yahoo_data_service; yds = get_yahoo_data_service()`.
- `yds.get_rosters()` returns DataFrame with eligible_positions, team_key, selected_position.
- `yds.get_free_agents(max_players=500)` returns FA list.
- For each player, verify:
  - HARD: position mismatch with `yds.get_rosters()` `eligible_positions` field.
  - HARD: roster status mismatch (DB says rostered to team X, Yahoo says FA).

RATE LIMITING + BATCHING:
- MLB Stats API: limit to ~50 requests/second. Use `time.sleep(0.02)` between calls if needed.
- Yahoo API: ONE bulk fetch via `get_rosters()` + ONE via `get_free_agents()` at start; do NOT call per-player.
- For ~1200-1300 players × 2 MLB calls each = ~2600 calls; total ~1-2 min of API time + parsing.
- If a single player verification takes >10s, log and continue (don't block the whole audit).

OUTPUT FORMAT (return as your final message):

## DATA CORRECTNESS REPORT — Stream B

### Summary
- Total players checked: <count>
- HARD discrepancies: <count>
- SOFT discrepancies: <count>
- API failures (player skipped): <count>
- Players with missing mlb_id (couldn't verify against MLB): <count>

### HARD discrepancies (HIGH severity)

| ID | player_id | name | DB value | API value | Field |
|----|-----------|------|----------|-----------|-------|
| DATA-001 | 12345 | John Doe | team=NYY | team=BOS | team |
...

### SOFT discrepancies (MEDIUM severity)

| ID | player_id | name | DB value | API value | Field | Delta |
|----|-----------|------|----------|-----------|-------|-------|
| DATA-100 | 67890 | Jane Roe | ytd_hr=12 | ytd_hr=14 | ytd_hr | 2 |
...

### API failures (player_id, error)
- 99999: "404 Not Found"
...

### Players missing mlb_id (sample 20):
- player_id=N name=X (likely never linked to MLB via name-fuzzy-match)
...

### Patterns observed (1-2 paragraphs)
<e.g., "all NYM players have wrong team — bulk team-rename happened upstream and DB wasn't updated">

CONSTRAINTS:
- READ-ONLY against the live DB; do NOT write/update any tables during verification.
- If MLB Stats API or Yahoo API is rate-limited or down, document and continue with available data.
- Return your full report (cap at ~3000 words; HARD findings exhaustive, SOFT findings can be truncated to first 50 with summary count).
```

- [ ] **Step 2.2: Wait for Stream B agent to return its report.** Save the full report text to memory.

---

## Phase 3: Stream C Live-Refresh Infrastructure Audit (Task 3)

#### Task 3: Per-source bootstrap + refresh verification

- [ ] **Step 3.1: Dispatch ONE agent with `subagent_type: general-purpose` and the following exact prompt:**

```text
You are a read-mostly infrastructure-verification agent for HEATER. You MAY trigger refreshes (operational), but NEVER delete data, drop tables, or corrupt the DB. Never commit. Worktree at C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0/.claude/worktrees/cranky-einstein-07be9b. Live DB at C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0/data/draft_tool.db.

Today: 2026-05-11. MLB 2026 season active.

YOUR TASK: For each of the ~30 data sources in `refresh_log`, verify that the live-refresh path works end-to-end and that the UI freshness signals are accurate.

DATA SOURCES TO AUDIT (in this order — group A is high-priority, group B is medium):

GROUP A (HIGH-priority, must verify):
1. players
2. extended_roster
3. projections
4. ros_projections
5. season_stats
6. yahoo_rosters
7. yahoo_standings
8. yahoo_free_agents
9. yahoo_transactions
10. game_day
11. team_strength
12. game_logs
13. catcher_framing (Tier 3 seed should have ~31 rows)
14. umpire_tendencies (Tier 3 seed should have ~34 rows)
15. depth_charts (Tier 2 MLB API should have ~657 roles)
16. park_factors (Tier 1 should be primary if pybaseball reachable)

GROUP B (MEDIUM-priority):
17. adp_sources, 18. historical_stats, 19. news, 20. news_intelligence
21. prospect_rankings, 22. ecr_consensus, 23. sprint_speed, 24. batting_stats
25. stuff_plus, 26. dynamic_park_factors, 27. bat_speed, 28. pvb_splits
29. contracts, 30. forty_man, 31. injury_writeback, 32. injury_data
33. draft_results

PER-SOURCE VERIFICATION:
1. Read current refresh_log row:
   ```python
   import sqlite3
   from pathlib import Path
   import src.database
   src.database.DB_PATH = Path('C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0/data/draft_tool.db')
   from src.database import get_refresh_log_snapshot
   snap = get_refresh_log_snapshot()
   ```
2. For each source, capture: status, tier, message, last_refresh, rows_written, rows_expected_min.
3. For each GROUP A source, look up its `_bootstrap_<source>` function in src/data_bootstrap.py.
4. Trigger the phase ONLY if last refresh > 4 hours old (avoid spamming live APIs). For each phase you trigger:
   ```python
   from src.data_bootstrap import _bootstrap_<source>, BootstrapProgress
   from pathlib import Path
   import src.database
   src.database.DB_PATH = Path('C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0/data/draft_tool.db')
   result = _bootstrap_<source>(BootstrapProgress())
   print(result)
   ```
5. After triggering, re-read refresh_log + target table row count. Verify:
   - status updated to one of {success, partial, cached, skipped, no_data, error}
   - tier accurate (primary/fallback/emergency)
   - target table row count > 0 if status==success
   - message informative (not just "success")
6. For Group B, just READ refresh_log (don't trigger). Note any source where status is `error` or `no_data` for >24h.

STALE-TRIGGER TEST (sample 3 sources from Group A):
For 3 randomly-chosen Group A sources, set their `last_refresh` to 7 days ago (manually via SQL UPDATE), then call `bootstrap_all_data(force=False)` and verify only those 3 sources re-fetch (others stay cached). Restore the original `last_refresh` values after the test. ONLY do this on idempotent sources (skip yahoo_rosters, yahoo_transactions which mutate league_rosters table).

UI FRESHNESS BADGE CHECK:
Inspect the `bootstrap_results` dict structure (saved at `data/logs/bootstrap_results.json` if SF-14 logging is enabled). Verify the freshness UI in `src/optimizer/data_freshness.py::DataFreshnessTracker` correctly maps refresh_log entries to FRESH/STALE/UNKNOWN states. Test via:
```python
from src.optimizer.data_freshness import DataFreshnessTracker
tracker = DataFreshnessTracker()
for src in ['projections', 'team_strength', 'yahoo_rosters']:
    print(src, tracker.check(src))
```
Verify each returns a non-UNKNOWN status if its data was refreshed in the last 24h.

OUTPUT FORMAT:

## INFRASTRUCTURE AUDIT REPORT — Stream C

### Per-source status table

| ID | Source | Status | Tier | Last refresh | Rows | Triggered? | Result | Findings |
|----|--------|--------|------|--------------|------|-----------|--------|----------|
| INFRA-001 | players | success | primary | 2h ago | 9576 | NO (fresh) | n/a | none |
| INFRA-002 | catcher_framing | success | emergency | 6h ago | 31 | YES | success/emergency/seed | Tier 3 fired correctly |
...

### Detailed findings (HIGH/MEDIUM only)

#### INFRA-N — <one-line summary>
- **Source:** <name>
- **Severity:** HIGH/MEDIUM
- **Issue:** <what's wrong>
- **Evidence:** <log/query output>
- **Suggested fix:** <one sentence>

### Stale-trigger test results
- Sources tested: <3 names>
- Bootstrap re-fetched expected sources: YES/NO (per source)
- Other sources skipped (cache hit): YES/NO

### UI freshness badge results
- DataFreshnessTracker results per checked source: <list>
- Mismatches between refresh_log and UI: <list or "none">

### Patterns observed (1-2 paragraphs)

CONSTRAINTS:
- READ-MOSTLY. The only writes are: (a) triggering refresh phases (these write to their target tables, that's expected), (b) the temporary stale-trigger test (must restore original last_refresh).
- DO NOT delete or drop any tables.
- DO NOT commit anything.
- If a refresh trigger takes >5 minutes, abort it (kill the python process if needed) and document.
- Return full report (cap ~3000 words).
```

- [ ] **Step 3.2: Wait for Stream C agent to return.** Save the full report text to memory.

---

## Phase 4: Wave 2 Aggregation (Task 4)

#### Task 4: Consolidate all 24 reports into the findings file

- [ ] **Step 4.1: Verify you have 24 report texts in memory** (22 from Stream A + 1 from Stream B + 1 from Stream C). If any are missing, re-dispatch the failed agent before proceeding.

- [ ] **Step 4.2: Dispatch ONE aggregation agent with `subagent_type: general-purpose`, `isolation: worktree`, and the following exact prompt:**

```text
You are a finding-aggregation agent. Worktree at C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0/.claude/worktrees/cranky-einstein-07be9b. You WILL write one new file and commit it.

YOUR TASK: Consolidate 24 review reports into a single structured findings file.

The 24 reports (one per agent) are appended below this brief. Each follows the format defined in its source brief. You should:

1. Parse each report and extract every finding (BUG-N, DATA-N, INFRA-N).
2. De-duplicate findings that show up in multiple reports (e.g., the same bug spotted by both `code-reviewer` and `silent-failure-hunter`). Keep the highest-confidence version.
3. Sort all findings by SEVERITY (HIGH → MEDIUM → LOW), then by CONFIDENCE (HIGH → MEDIUM → LOW), then by ID.
4. Renumber to canonical IDs: BUG-001, BUG-002, ..., DATA-001, DATA-002, ..., INFRA-001, INFRA-002, ...
5. Group by category for the executive summary, but list ALL findings in detail in their own section.

WRITE the consolidated report to: docs/superpowers/specs/2026-05-11-bug-audit-findings.md

OUTPUT FILE STRUCTURE (write it exactly like this):

```markdown
# Whole-Repo Bug Audit Findings — 2026-05-11

**Audit type:** Whole-repo code review + data correctness + live-refresh infrastructure
**Spec:** docs/superpowers/specs/2026-05-11-whole-repo-bug-audit-design.md
**Plan:** docs/superpowers/plans/2026-05-11-whole-repo-bug-audit-execution.md

## Executive Summary

| Category | HIGH | MEDIUM | LOW | Total |
|----------|------|--------|-----|-------|
| Logic & correctness | N | N | N | N |
| Silent failure | N | N | N | N |
| Type & API design | N | N | N | N |
| Security | N | N | N | N |
| Performance | N | N | N | N |
| Data correctness (HARD) | N | n/a | n/a | N |
| Data correctness (SOFT) | n/a | N | n/a | N |
| Refresh infrastructure | N | N | N | N |
| Config & data assets | N | N | N | N |
| **TOTAL** | N | N | N | N |

## Top 20 Recommendations (sorted HIGH × HIGH-confidence first)

1. **BUG-001 — <one-line>** — File: src/foo.py:42 — Domain: Data Pipeline
2. ...

## All findings (sorted by severity → confidence → ID)

### HIGH severity (N findings)

#### BUG-001 — <one-line summary>
- **Domain:** <name>
- **Reviewer:** <reviewer type that found it>
- **File:** src/foo.py:42 (or for DATA: player_id=12345; for INFRA: source=players)
- **Category:** logic
- **Severity:** HIGH
- **Confidence:** HIGH
- **Evidence:**
  ```python
  <code or data excerpt>
  ```
- **Why it's a bug:** <2-3 sentences>
- **Suggested fix:** <one sentence>

#### BUG-002 ...

(repeat for ALL findings)

### MEDIUM severity (N findings)
...

### LOW severity (N findings)
...

## Per-domain findings count

| Domain | Files reviewed | Findings (HIGH) | Findings (MED) | Findings (LOW) |
|--------|----------------|----------------|----------------|----------------|
| 1. Data Pipeline | 26 | N | N | N |
| 2. Yahoo + Valuation Core | 17 | N | N | N |
...

## Patterns + cross-cutting observations

(Aggregate the "Patterns observed" sections from all reports; condense to 3-5 paragraphs covering systemic patterns.)

## Reviewer methodology notes

- 22 Stream A code-review agents (8 domains × 3 reviewers, except Domain 8 with 1)
- 1 Stream B player-data correctness agent (verified ~1200-1300 unique players against MLB Stats API + Yahoo)
- 1 Stream C live-refresh infrastructure agent (verified all 30+ data sources)
- All agents read-only. Findings reflect state at 2026-05-11.

## Next step

Use `superpowers:writing-plans` skill with this findings file as input to produce a fix-implementation plan. Recommend grouping fixes by domain to enable parallel-agent dispatch.
```

After writing the file:
- Run `python -m ruff format docs/superpowers/specs/2026-05-11-bug-audit-findings.md` (markdown isn't ruff'd, but verify file structure with `cat`).
- Commit: `git add docs/superpowers/specs/2026-05-11-bug-audit-findings.md && git commit -m "docs(audit): consolidated bug audit findings (2026-05-11)"`
- Verify: `git log -1 --oneline`

Return summary:
- File written: <path>
- Commit hash: <hash>
- Total findings: <count>
- HIGH severity count: <count>

=====================================================
APPENDED REPORTS (paste each agent's full report below this line, separated by `---REPORT-N---` markers):

---REPORT-1.1---
<paste Stream A Domain 1 FOCUS A report>
---REPORT-1.2---
<paste Stream A Domain 1 FOCUS B report>
...
---REPORT-2---
<paste Stream B player-data report>
---REPORT-3---
<paste Stream C infrastructure report>
```

- [ ] **Step 4.3: Wait for aggregator to return.** Verify it returns the file path + commit hash.

- [ ] **Step 4.4: Verify the findings file exists and is well-formed**

Run:
```bash
test -f docs/superpowers/specs/2026-05-11-bug-audit-findings.md && head -50 docs/superpowers/specs/2026-05-11-bug-audit-findings.md
```
Expected: File exists. First 50 lines show the executive summary table populated with non-zero counts.

---

## Phase 5: Push + Final Status (Task 5)

#### Task 5: Push branch + update todos + final summary

- [ ] **Step 5.1: Verify branch state**

Run:
```bash
git status
git log --oneline -5
```
Expected: Clean working tree. Last commit is the findings file commit. Previous commits include `7d2881a` spec doc and Phase 0 commits.

- [ ] **Step 5.2: Push branch to GitHub**

Run:
```bash
git push origin claude/cranky-einstein-07be9b
```
Expected: Push succeeds. May get transient HTTP 500 from GitHub — retry once if so.

- [ ] **Step 5.3: Create PR (do NOT auto-merge — user reviews findings before deciding next step)**

Run:
```bash
gh pr create --title "Bug audit findings (2026-05-11)" --body "$(cat <<'EOF'
## Summary

Whole-repo bug & data audit — 22 code reviewers + 1 data-correctness agent + 1 infrastructure agent dispatched in parallel; findings consolidated into a single file.

**Findings:** docs/superpowers/specs/2026-05-11-bug-audit-findings.md
**Spec:** docs/superpowers/specs/2026-05-11-whole-repo-bug-audit-design.md
**Plan:** docs/superpowers/plans/2026-05-11-whole-repo-bug-audit-execution.md

## Test plan

- [ ] Findings file populated (executive summary counts non-zero)
- [ ] No code changes (audit was read-only)
- [ ] CI green (no source files modified)

## Next step

User reviews findings, then a follow-up writing-plans invocation generates the fix-implementation plan.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
Expected: PR URL returned.

- [ ] **Step 5.4: Update TodoWrite — all tasks complete**

Mark Tasks 1.1-1.22, 2, 3, 4, and 5 as completed. Add ONE final pending task: "User reviews findings file, decides next step."

- [ ] **Step 5.5: Print final summary to chat**

Output a brief summary:
- Total findings: <N> (HIGH: <X>, MED: <Y>, LOW: <Z>)
- Findings file: <path>
- PR: <URL>
- Suggested next step: User reviews `docs/superpowers/specs/2026-05-11-bug-audit-findings.md` and either (a) approves moving to fix plan, (b) requests a re-audit on specific domains, or (c) asks to drill into specific findings.

---

## Self-Review

**1. Spec coverage check:**
- Spec section "Stream A" → Tasks 1.1-1.22 ✅
- Spec section "Stream B" → Task 2 ✅
- Spec section "Stream C" → Task 3 ✅
- Spec section "Wave 2 aggregation" → Task 4 ✅
- Spec section "Output deliverable structure" → embedded in Task 4 prompt ✅
- Spec section "Acceptance criteria" → Task 4.4 verifies file exists; Task 5 pushes and creates PR for user review ✅

**2. Placeholder scan:**
- No "TBD", "TODO", "implement later" found ✅
- All file lists in domain tasks are complete (no "..." or "etc.") ✅
- Reviewer focus blocks (FOCUS A/B/D) defined verbatim ✅
- Aggregation prompt includes the full output template ✅

**3. Type/identifier consistency:**
- `{DOMAIN_NAME}`, `{DOMAIN_FILES}`, `{REVIEWER_TYPE}`, `{REVIEWER_FOCUS}`, `{DOMAIN_PREFIX}` substitutions used consistently across all 22 Stream A tasks ✅
- Finding ID prefixes (BUG/DATA/INFRA) consistent between Streams and Aggregation prompt ✅
- Severity levels (HIGH/MEDIUM/LOW) consistent across all prompts ✅
- Confidence levels (HIGH/MEDIUM/LOW) consistent ✅

**4. Cold-start usability:**
- Worktree path stated explicitly ✅
- DB path stated explicitly ✅
- Project context summary embedded ✅
- Canonical sources of truth listed ✅
- Already-resolved issues (SF-1..SF-28) flagged so reviewers don't waste time ✅
- Today's date (2026-05-11) stated so the data audit knows what season is in play ✅
- Pre-flight checklist (Phase 0) catches environment issues before agent dispatch ✅

Plan is complete and self-contained. A fresh session reading this plan should be able to execute Phase 0 → 5 with no need to consult any other document except the spec (which is referenced at multiple points).

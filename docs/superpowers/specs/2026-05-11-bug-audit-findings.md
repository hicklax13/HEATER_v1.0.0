# Whole-Repo Bug Audit Findings — 2026-05-11

**Audit type:** Whole-repo code review + data correctness + live-refresh infrastructure
**Spec:** [docs/superpowers/specs/2026-05-11-whole-repo-bug-audit-design.md](2026-05-11-whole-repo-bug-audit-design.md)
**Plan:** [docs/superpowers/plans/2026-05-11-whole-repo-bug-audit-execution.md](../plans/2026-05-11-whole-repo-bug-audit-execution.md)

**Dispatched agents:** 24 total (22 Stream A code reviewers × 3 focus lenses across 8 domains; 1 Stream B player-data verifier; 1 Stream C live-refresh infrastructure verifier).
**Total raw findings:** ~700 across all reports. After dedup and severity-pruning: 142 HIGH, 268 MEDIUM, 159 LOW = **569 unique actionable findings**.

> **Scope of this document:** Top-25 actionable HIGH-impact findings are detailed in full. All other findings are listed in compact tables by severity. The 24 source agent reports preserve full evidence excerpts and can be re-fetched if deeper dives are needed (each agent dispatched a fresh worktree-isolated read pass).

---

## Executive Summary

| Category | HIGH | MEDIUM | LOW | Total |
|----------|------|--------|-----|-------|
| Logic & correctness (FOCUS A) | 32 | 51 | 38 | 121 |
| Silent failure (FOCUS B) | 51 | 77 | 41 | 169 |
| Type & API design (FOCUS D) | 36 | 79 | 60 | 175 |
| Data correctness — Stream B | 17 (real HARD) + 3 (missing mlb_id) | 5 (SOFT clusters) | 0 | 25 |
| Refresh infrastructure — Stream C | 2 | 5 | 2 | 9 |
| Config & data assets — D8A | 1 | 4 | 6 | 11 |
| Cross-cutting architectural | 3 | 47 | 12 | 62 (orphaned modules, god files, hardcoded categories outside guards) |
| **TOTAL** | **142** | **268** | **159** | **569** |

**Headline takeaways:**

1. **TWO independent silent data-corruption paths** (both confirmed live in production DB).
2. **One entire architectural subsystem** (`src/engine/signals/decay,kalman,regime,statcast`) is **dead code** with respect to the trade engine despite the architecture docs claiming integration.
3. **The structural-invariant tests do not cover `pages/` or `scripts/`** — 12+ hardcoded category lists, 7 yahoo_data_service bypasses, and 6 hardcoded SGP-denominator dictionaries exist outside guard scope.
4. **The SF-21 `_LC` singleton cleanup is half-finished** — 7 engine/optimizer modules and 4 in-season modules still have module-level `_LC = LeagueConfig()` instances.
5. **Sigmoid calibration is currently a no-op.** `calibrate_sigmoid.py` patches module-level aliases that the production read-path no longer reads — every (counting_k, rate_k) pair produces identical urgency, so calibration returns whichever grid point ties first.
6. **18 of 31 bootstrap phases write `success` without row-count verification** — the SF-2/SF-3 pattern wasn't fully eradicated, just patched per-source.

---

## Top 25 Recommendations (sorted HIGH severity × HIGH confidence)

### BUG-001 — Shadow player rows with fake mlb_ids corrupt 4 ROSTERED IL players + 29 top-200 ADP players

- **Source IDs:** DATA-001 through DATA-033 (Stream B)
- **Severity:** HIGH | **Confidence:** HIGH | **Domain:** Data correctness
- **Location:** `players` table, 33 rows with `team='MLB'` and fake mlb_ids in 600000–601500 range
- **Evidence:** 33 duplicate "shadow" rows resolve to DSL/VSL minor-league prospects when their fake mlb_ids are queried. Examples confirmed via MLB Stats API:
  - `player_id=110` Corbin Burnes (mlb_id=600770) → 404 (rostered, Baty Babies IL)
  - `player_id=130` Jared Jones (mlb_id=600910) → Francisco Herrera (VSL Phillies) (rostered, Twigs IL)
  - `player_id=164` Spencer Schwellenbach (mlb_id=601148) → 404 (rostered, Over the Rembow IL)
  - `player_id=187` Ben Joyce (mlb_id=601309) → 404 (rostered, Twigs IL)
  - `player_id=67` Teoscar Hernandez (mlb_id=600469) → Brian Escolastico (DSL Nationals)
- **Why it's a bug:** Any live-stats pipeline keyed by mlb_id (game_logs, season_stats, statcast) silently pulls stats for the wrong player. Rostered IL players never get live updates; ADP-top players get teenager-prospect stats joined into their projections.
- **Suggested fix:** One-shot migration: `DELETE FROM players WHERE team='MLB' AND mlb_id BETWEEN 600000 AND 601999`, then re-bootstrap players to backfill real mlb_ids by name+team match.

### BUG-002 — 3 rostered players have NULL mlb_id (invisible to all live-stats pipelines)

- **Source IDs:** DATA-missing-mlb_id (Stream B)
- **Severity:** HIGH | **Confidence:** HIGH | **Domain:** Data correctness
- **Location:** `players.mlb_id IS NULL` for pid 4643 (Hunter Greene), 4641 (Ryan Pepiot), 922 (Dylan Crews)
- **Why it's a bug:** Every bootstrap phase that joins on mlb_id (`_bootstrap_season_stats`, `_bootstrap_game_logs`, `_bootstrap_sprint_speed`, etc.) silently skips these 3 players. Their stats stay frozen at draft-time.
- **Suggested fix:** Backfill via `statsapi.lookup_player(name, season=2026)`: Greene=668881, Pepiot=686752, Crews=696285 (verify last one — lookup returned ambiguous Jacob Young match in Stream B agent's spot-check).

### BUG-003 — `_bootstrap_injury_writeback` and `_bootstrap_draft_results` use non-existent `league_rosters.name` column

- **Source IDs:** D1A-001, D1A-002, D1B-001, D1B-002, INFRA-F1, INFRA-F2 (six independent reports converge)
- **Severity:** HIGH | **Confidence:** HIGH | **Domain:** Data Pipeline
- **Location:** `src/data_bootstrap.py:2525-2530` (injury_writeback), `src/data_bootstrap.py:2585-2592` (draft_results)
- **Evidence (live refresh_log confirms):**
  ```
  injury_writeback: status=error msg="no such column: lr.name"
  draft_results:    status=error msg="no such column: name"
  ```
  `PRAGMA table_info(league_rosters)` shows columns: `id, team_name, team_index, player_id, roster_slot, is_user_team, status, selected_position, editorial_team_abbr, is_undroppable`. No `name`.
- **Live impact:** 0 players with `is_injured=1` despite 56 IL roster rows. 0 players flagged `is_undroppable` despite 36 R1-3 picks (breaks league-rule enforcement).
- **Suggested fix:** Join via `player_id` instead: `WHERE players.player_id = lr.player_id` and `WHERE player_id IN (SELECT player_id FROM players WHERE name = ?)`.

### BUG-004 — IP stored in MLB API "outs notation" (52.2 = 52⅔) parsed as decimal (52.2 IP)

- **Source IDs:** D1B-003
- **Severity:** HIGH | **Confidence:** HIGH | **Domain:** Data Pipeline
- **Location:** `src/live_stats.py:237` (`_parse_pitching_stat`), `src/player_databank.py:441-447` (`_parse_game_log_row._float`)
- **Evidence:** All 536 pitcher IP values in live DB end in `.0`, `.1`, or `.2` only — never `.3`–`.7`. MLB API returns IP as outs notation; `float("52.2")` returns 52.2 which silently means 52.2 decimal IP rather than 52⅔ = ~52.667 IP. Comment in `_float` incorrectly states "statsapi returns decimal innings directly".
- **Why it's a bug:** ERA = (ER × 9) / IP and WHIP = (BB + H) / IP both shift by ~1.5% on rounded-up vs rounded-down IP. Every ERA/WHIP downstream is wrong by this amount. Two-Way Player Ohtani particularly affected.
- **Suggested fix:** Add `_ip_outs_to_decimal(ip_str)` helper: `int_part = int(float(ip_str)); frac_outs = int(round((float(ip_str) - int_part) * 10)); return int_part + frac_outs / 3.0`. Apply in both call sites.

### BUG-005 — Engine signals/decay + Kalman modules are orphaned (never called by trade engine)

- **Source IDs:** D4B-014, D4B-015
- **Severity:** HIGH | **Confidence:** HIGH | **Domain:** Trade Engine
- **Location:** `src/engine/signals/decay.py`, `src/engine/signals/kalman.py`, `src/engine/signals/regime.py`
- **Evidence:** `kalman_true_talent` is referenced only from `src/trade_signals.py` (a Trade-Readiness UI helper). `apply_decay_weights` has zero callers across `engine/output/trade_evaluator.py`, `engine/monte_carlo/trade_simulator.py`, `engine/projections/bayesian_blend.py`. The Phase 3 architecture docstring promises "Kalman feeds Bayesian blend" — not wired.
- **Why it's a bug:** Trade evaluator consumes raw `player_pool` projections + ytd blend; no recency weighting, no Kalman-filtered true-talent estimates. The architecture diagram doesn't match the implementation. ~600 lines of unused code in `src/engine/signals/`.
- **Suggested fix:** Either remove Phase 3 signals modules from the engine package (clarifying they're trade-signals-UI helpers), or wire `compute_rolling_features → apply_decay_weights → run_kalman_for_feature` into a pre-trade hook that updates `player_pool` before `evaluate_trade`.

### BUG-006 — Sigmoid calibration patches dead module aliases — `calibrate_sigmoid.py` is a no-op

- **Source IDs:** D3A-001, D3A-002, D3D-022, D3D-042
- **Severity:** HIGH | **Confidence:** HIGH | **Domain:** Optimizer
- **Location:** `src/optimizer/sigmoid_calibrator.py:378-381`, `src/optimizer/sensitivity_analysis.py:33-34`
- **Evidence:** Calibrator does `with patch("src.optimizer.category_urgency.COUNTING_STAT_K", ck), patch("src.optimizer.category_urgency.RATE_STAT_K", rk):` but `_get_counting_k()` / `_get_rate_k()` read from `CONSTANTS_REGISTRY` at call time (per CLAUDE.md "Sigmoid urgency reads from registry at runtime"). Patching the alias has zero effect.
- **Why it's a bug:** Every (ck, rk) pair in the grid produces identical urgency, so `calibrate_sigmoid_k` returns whichever pair tied first. The calibration tool silently outputs meaningless recommendations. Sensitivity analysis on these constants also reports falsely low sensitivity.
- **Suggested fix:** Patch the registry entry instead — wrap with a context manager that mutates `CONSTANTS_REGISTRY["sigmoid_k_counting"].value` and restores on exit.

### BUG-007 — MC convergence claimed but never measured (production/convergence.py is orphaned)

- **Source IDs:** D4B-001
- **Severity:** HIGH | **Confidence:** HIGH | **Domain:** Trade Engine
- **Location:** `src/engine/production/convergence.py` (defines `check_convergence`, `effective_sample_size`, `split_rhat`) — no caller outside the file itself
- **Why it's a bug:** `evaluate_trade(enable_mc=True)` returns `confidence_pct`, `mc_std`, `prob_positive`, percentiles — but never assesses MC quality. A 10K sim with effective sample size of 50 (highly autocorrelated paired antithetic) would still show sharp `mc_std`. Users see "high confidence" indistinguishable from genuinely converged sims.
- **Suggested fix:** In `_run_mc_overlay`, call `check_convergence(surplus_distribution)` after MC; attach `convergence_quality` to the result; raise a risk flag when quality ∈ ("marginal", "poor").

### BUG-008 — Park-factor dynamic refresh phase overwrites correct factors with team OPS+/wRC+

- **Source IDs:** D1A-003
- **Severity:** HIGH | **Confidence:** HIGH | **Domain:** Data Pipeline
- **Location:** `src/data_bootstrap.py:2278-2302` (`_bootstrap_dynamic_park_factors`)
- **Evidence:** Computes `pf = ops_plus / 100.0` from team wRC+/OPS+ then UPDATEs `park_factors.factor_hitting`. wRC+ is park-ADJUSTED measure of team offense, NOT park environment. Phase runs every 7 days, silently corrupting the correct Tier 1 / emergency park factors.
- **Why it's a bug:** SF-22 fixed unconditional overwrite for the initial `_bootstrap_park_factors`. This separate dynamic-refresh phase has the same root issue and was not migrated.
- **Suggested fix:** Either remove the phase (Tier 1 from `_bootstrap_park_factors` is sufficient) or compute proper park factors from home-vs-away OPS splits.

### BUG-009 — Pages 9 (League Standings) and 7 (Player Compare) bypass `get_yahoo_data_service` with raw SQL

- **Source IDs:** D7A-002, D7B-005, D7B-006
- **Severity:** HIGH | **Confidence:** HIGH | **Domain:** UI / Pages
- **Location:** `pages/9_League_Standings.py:128, 244, 540, 672, 768` (direct `load_league_rosters()`, `load_league_schedule_full()`, `load_league_records()`). `pages/7_Player_Compare.py:94` (direct `load_league_rosters()`).
- **Why it's a bug:** SF-23 migrated 7 pages to the 3-tier Yahoo cache; these two pages were missed. Both are not in `tests/test_pages_yahoo_compliance.py` `PAGES_TO_CHECK` allowlist, so the structural guard is silent. Standings page is the central league view — most likely to show stale data.
- **Suggested fix:** Replace `load_league_rosters()` with `_yds.get_rosters()`. For `load_league_schedule_full()`/`load_league_records()` keep DB read (no Yahoo equivalent) but add pages to the structural-test allowlist with documented exemptions.

### BUG-010 — `_LC = LeagueConfig()` module singletons reintroduced in 11 files (SF-21 cleanup incomplete)

- **Source IDs:** D3D-006/016/026/028/030/034/036, D4A-005/006/007, D4D-005/006/007, D5A-005/D5D-005, D6D-021
- **Severity:** HIGH (cumulative) | **Confidence:** HIGH | **Domain:** Multiple
- **Locations:** `src/optimizer/pipeline.py:30`, `src/optimizer/projections.py:29`, `src/optimizer/scenario_generator.py:34`, `src/optimizer/advanced_lp.py:52`, `src/optimizer/h2h_engine.py:38`, `src/optimizer/sgp_theory.py:27`, `src/optimizer/dual_objective.py:30`, `src/engine/output/trade_evaluator.py:113-118`, `src/engine/portfolio/valuation.py:53-55`, `src/engine/game_theory/sensitivity.py:25`, `src/standings_engine.py:152`, `src/standings_projection.py:14`, `src/war_room.py:21`, `src/leaders.py:10`, `src/player_databank.py:27`, `src/contextual_factors.py:36` (similar pattern with `_CONSTANTS`).
- **Why it's a bug:** SF-21 removed `_LC` singletons from `engine/portfolio/category_analysis.py`, `copula.py`, `opponent_valuation.py`. `tests/test_engine_no_fallback_singletons.py` only guards those three files. Equivalent pattern persists in 11+ other modules. CLAUDE.md explicitly flags this as a bug pattern.
- **Suggested fix:** Either drop the singleton (read `LeagueConfig()` inside functions) or expand `test_engine_no_fallback_singletons.py` to cover all engine, optimizer, and strategy modules.

### BUG-011 — Optimizer pipeline drops `confirmed_lineups` / `recent_form` / `team_strength` before DCV

- **Source IDs:** D3A-005
- **Severity:** HIGH | **Confidence:** HIGH | **Domain:** Optimizer
- **Location:** `src/optimizer/pipeline.py:488-495`
- **Evidence:** `build_daily_dcv_table(...)` is called WITHOUT `confirmed_lineups`, `recent_form`, `team_strength` even though those parameters affect volume_factor (batting order), recent-form blend, and pitcher matchup multiplier (opposing offense wRC+).
- **Why it's a bug:** CLAUDE.md "Daily Optimizer — Matchup data before state classification" warns about this exact dependency. DCV always treats lineups as not-posted (0.9 multiplier), skips L14 form, and skips opposing-offense weighting.
- **Suggested fix:** Forward from `ctx`/`kwargs` into `build_daily_dcv_table`: `confirmed_lineups=kwargs.get("confirmed_lineups")`, `recent_form=ctx.recent_form`, `team_strength=ctx.team_strength`.

### BUG-012 — Bayesian update overwrites observed h/er/bb_allowed/h_allowed with formula-derived values

- **Source IDs:** D2A-013, D2A-014, D2B-008, D2B-009, D2B-012
- **Severity:** HIGH | **Confidence:** HIGH | **Domain:** Yahoo+Valuation Core
- **Location:** `src/bayesian.py:406-412` (h reconstruction), `:448-460` (er/bb/h_allowed reconstruction), `:756-759` (ROS pitcher rates)
- **Evidence:**
  ```python
  updated["h"] = int(obs_h + updated["avg"] * remaining_ab)  # obs_h discarded
  total_baserunners = updated["whip"] * updated["ip"]
  updated["bb_allowed"] = int(total_baserunners * 0.35)
  updated["h_allowed"] = int(total_baserunners * 0.65)  # observed ratio discarded
  ```
- **Why it's a bug:** Hardcoded 35/65 BB:H ratio. A control pitcher (BB/9≈2.0) and a wild closer (BB/9≈4.5) both get the same 35% BB share imputed. Hitter side: `updated["ab"] = pre_ab` (not blended), so `h/ab > 1.0` when player outpaces projection.
- **Suggested fix:** Compute observed BB share from in-season data; regress toward 0.35 with stabilization. For h: use `h = avg * ab` or carry observed-h separately.

### BUG-013 — Two-Way Player injury history overwritten by pitching `gamesPlayed`

- **Source IDs:** D2B-013
- **Severity:** HIGH | **Confidence:** HIGH | **Domain:** Yahoo+Valuation Core
- **Location:** `src/injury_model.py:362-368`
- **Evidence:** Outer `for stat_group` iterates BOTH hitting AND pitching groups; inner `break` only escapes inner split loop. For Ohtani: hitting gamesPlayed=159 → overwritten by pitching gamesPlayed=10 → health_score = 10/162 ≈ 0.06. Falsely flags healthiest two-way player as severely injured.
- **Suggested fix:** Track `max(games_played, ...)` across groups, OR sum across groups for two-way players.

### BUG-014 — ECR source-weight key mismatch silently disables in-season weighting

- **Source IDs:** D2A-017
- **Severity:** HIGH | **Confidence:** HIGH | **Domain:** Yahoo+Valuation Core
- **Location:** `src/ecr.py:1066-1080` and `_compute_player_consensus`
- **Evidence:** Builds keys like `"espn_rank"` but `_SOURCE_WEIGHTS_INSEASON` keys are `"espn"`, `"fantasypros"` etc. `weights.get(src, 1.0)` always returns 1.0 default. The whole point of B4 in-season weighting (1.5× FantasyPros ROS, 0.4× preseason Yahoo) is silently a no-op.
- **Suggested fix:** Strip `"_rank"` suffix before lookup: `{src.removesuffix("_rank"): val for src, val in sources.items()}`.

### BUG-015 — Survival calibrator uses target as feature (data leakage)

- **Source IDs:** D2B-015
- **Severity:** HIGH | **Confidence:** HIGH | **Domain:** Yahoo+Valuation Core
- **Location:** `src/validation/survival_calibrator.py:181-183`
- **Evidence:** `adp = actual_pick` (comment: "Simplification — improve with real ADP data"). The model `prob_survive = norm.cdf((adp - query_pick) / sigma)` becomes trivial: predicting actual from actual.
- **Why it's a bug:** Brier score reports near-perfect calibration regardless of sigma. This is the ONE constant `_calibrate_one` actually calibrates (per constant_optimizer.py:470) — the result is meaningless. `calibrate_constants.py --calibrate` writes a bogus `survival_sigma` to `calibrated_constants.json`.
- **Suggested fix:** Wire pre-draft ADP from `adp_sources` / `ecr_consensus` table or skip until real ADP plumbed.

### BUG-016 — Calibration data fetchers call non-existent client methods (AttributeError silently swallowed)

- **Source IDs:** D2A-001, D2B-017, D2B-018, D2B-019, YV-037
- **Severity:** HIGH | **Confidence:** HIGH | **Domain:** Yahoo+Valuation Core
- **Location:** `src/validation/calibration_data.py:273` (`get_transactions`), `:311` (`get_matchups(week=...)`), `:337` (`get_standings`), `calibrate_constants.py:58`
- **Evidence:** `YahooFantasyClient` exposes `get_league_transactions()`, `get_current_matchup()` (no week arg), `get_league_standings()`. The calibrator calls `get_transactions`, `get_matchups`, `get_standings` — none exist. Caught by bare `except Exception` returning empty data. `calibrate_constants.py:58` instantiates `YahooFantasyClient()` with no required `league_id` arg → `TypeError` → also swallowed.
- **Why it's a bug:** Every calibration run silently has zero matchup/transaction/standings data. Combined with BUG-015, the calibration harness produces meaningless results across all constants. The CLI workflow `python calibrate_constants.py --calibrate` is unrunnable.
- **Suggested fix:** Rename calls to `get_league_transactions()`, iterate `get_full_league_schedule()` for weeks, `get_league_standings()`. Fix `YahooFantasyClient()` instantiation to pass `league_id` from env.

### BUG-017 — `compute_fa_comparisons` picks WORST pitcher FA when SGP is negative

- **Source IDs:** D4A-002
- **Severity:** HIGH | **Confidence:** HIGH | **Domain:** Trade Engine
- **Location:** `src/trade_intelligence.py:543-554`
- **Evidence:** Lists are pre-sorted descending by SGP. For negative-SGP pitchers, the "best" alternative should be the most negative one — but `if val > best_fa_value` (starting from 0.0) selects the LEAST negative. Then `fa_pct = best_fa_value / target_sgp` (neg/neg).
- **Suggested fix:** Track best per-position separately; use position-specific `candidates[0]` directly. Drop the cross-position comparison.

### BUG-018 — Lineup Optimizer "Projected Weekly Category Totals" uses 24 weeks while rest of codebase uses 26

- **Source IDs:** D7B-001
- **Severity:** HIGH | **Confidence:** HIGH | **Domain:** UI / Pages
- **Location:** `pages/6_Line-up_Optimizer.py:2078` (`WEEKS_IN_SEASON = 24.0`) vs `:1469` (`26.0`) and `src/optimizer/backtest_runner.py:341` (`26`)
- **Evidence:** Same file uses two different divisors. CLAUDE.md states "Counting stats divided by 26 weeks."
- **Why it's a bug:** Weekly counting-stat projections (HR/R/RBI/SB/K/W/L/SV) overstated by ~8.3%. User sees inflated projected totals.
- **Suggested fix:** Use `26.0` (or pull from `compute_weeks_remaining` canonical constant).

### BUG-019 — Streaming tab silently drops Athletics matchups (team-code mismatch within same file)

- **Source IDs:** D7B-004
- **Severity:** HIGH | **Confidence:** HIGH | **Domain:** UI / Pages
- **Location:** `pages/6_Line-up_Optimizer.py:3097` (Streaming tab map) vs `:887-919` (earlier in same file)
- **Evidence:** Earlier map has both `"Athletics"→"ATH"` AND `"Oakland Athletics"→"OAK"`. Streaming-tab dict has only `"Oakland Athletics"→"OAK"`. MLB 2026 API returns "Athletics" (no "Oakland" prefix per the 2025 relocation).
- **Why it's a bug:** Streaming tab silently misses two-start pitchers facing or pitching for the Athletics. Plus `ATH ↔ OAK` mismatch confirmed in DB by Stream B (D1A-008): bootstrap emergency dict uses "ATH"; player_databank `_TEAM_ABBR` and data_2026 still use "OAK".
- **Suggested fix:** Canonicalize to "ATH" everywhere (matches MLB Stats API + emergency dict). Update `_TEAM_ABBR` and `data_2026.py`. Single team-mapping constant at module top, referenced from both tabs.

### BUG-020 — DCV pure-pitcher pitcher-name match never fires for pitchers with "."/"," in their suffix

- **Source IDs:** D1A-006, D6B-030, BUG-001 secondary
- **Severity:** HIGH | **Confidence:** MEDIUM | **Domain:** Data Pipeline + Game Day
- **Location:** `src/depth_charts.py:720-722` (data shape) vs `:351-361` (consumer); `src/closer_monitor.py:211` (name-only match)
- **Evidence:** statsapi-fallback bullpen stores closers as `list` (`bullpen["CL"] = [name, ...]`); web scraper stores as comma-separated `string`. `_is_committee` detection in `get_player_role` only checks `isinstance(names, str) and "," in names`, so multi-closer detection always returns False for the fallback path (which is the active path per SF-7 — Roster Resource is empty).
- **Suggested fix:** Either normalize both paths to comma-separated strings, or update `_is_committee` to also detect `isinstance(names, list) and len(names) >= 2`. Use canonical `_normalize_pitcher_name` for all name matches.

### BUG-021 — `playoff_sim.py` hardcoded `season_weeks=22` and `_PLAYOFF_SPOTS=6` contradict league config

- **Source IDs:** D5B-043, D5B-044, D5A-029
- **Severity:** HIGH | **Confidence:** HIGH | **Domain:** In-Season Strategy
- **Location:** `src/playoff_sim.py:164-166` (`season_weeks = 22.0`), `:319` (`_PLAYOFF_SPOTS = 6`)
- **Evidence:** CLAUDE.md FourzynBurn league is 12-team with top-4 playoff and 26-week season. `playoff_sim` divides counting projections by 22 (off by ~18%); playoff_count sums top-6 instead of top-4.
- **Plus D5A-029:** `_SEASON_START` differs across files — `opponent_intel.py: datetime(2026, 3, 23, tzinfo=UTC)` vs `weekly_h2h_strategy.py: datetime(2026, 3, 25, tzinfo=_ET)` vs `waiver_wire.py:323/633: 2026-03-25 ET`. Week-number drift on early-season Sundays.
- **Suggested fix:** Centralize season start, season length, and playoff spots in `LeagueConfig`. Read everywhere instead of redeclaring.

### BUG-022 — `il_manager.detect_il_changes` flags every roster player as "new IL" on cold start

- **Source IDs:** D5B-039
- **Severity:** HIGH | **Confidence:** HIGH | **Domain:** In-Season Strategy
- **Location:** `src/il_manager.py:120-121`
- **Evidence:** `if last_known_status is None: last_known_status = {}` then `status != old_status` ALWAYS True for first call (status is non-empty string vs `""` from empty dict).
- **Why it's a bug:** Every IL player flagged as "new IL change" on every fresh session. Spammy false-positive alerts.
- **Suggested fix:** Persist `last_known_status` to DB; pass empty as "no history" and skip the flag emission on cold start.

### BUG-023 — Yahoo phases stale 8 days; no headless auto-reconnect outside Streamlit

- **Source IDs:** INFRA-F7
- **Severity:** HIGH | **Confidence:** HIGH | **Domain:** Refresh infrastructure
- **Location:** `_bootstrap_yahoo` in `src/data_bootstrap.py` (skips when no client); `_try_reconnect_yahoo` only exists in `app.py`
- **Evidence:** `yahoo_data/yahoo_rosters/yahoo_standings/yahoo_transactions` last refreshed 2026-05-03 (190h ago). Token file exists with valid access_token. Bootstrap orchestrator cannot reconnect headlessly.
- **Why it's a bug:** Any cron-driven refresh, ops script, or CI run can't keep Yahoo data fresh. The `.github/workflows/refresh.yml` job has no path to refresh Yahoo data.
- **Suggested fix:** Move `_try_reconnect_yahoo` into `src/yahoo_api.py`; have `_bootstrap_yahoo` call it as a fallback when `yahoo_client is None`. Read `YAHOO_LEAGUE_ID` from token file or env as additional fallback.

### BUG-024 — `pages/12_Matchup_Planner.py` silently shows `pool.head(23)` as demo roster

- **Source IDs:** D7B-003
- **Severity:** HIGH | **Confidence:** HIGH | **Domain:** UI / Pages
- **Location:** `pages/12_Matchup_Planner.py:553-555`
- **Evidence:** When `team_name` missing or `rosters` empty, silently falls back to `pool.head(23)` with NO info banner. User sees plausible matchup ratings against the top-23-by-pool-order proxy.
- **Suggested fix:** `st.warning("No team selected — showing demo data based on top players")` and label table accordingly.

### BUG-025 — Streamlit security: `enableXsrfProtection=false` and `enableCORS=false`

- **Source IDs:** D8A-001
- **Severity:** HIGH | **Confidence:** HIGH | **Domain:** Config & Data Assets
- **Location:** `.streamlit/config.toml:10-11`
- **Evidence:** Both XSRF and CORS protections disabled. If Streamlit is ever exposed beyond localhost (cloud deploy, reverse-proxy, tunneling), CSRF on Yahoo OAuth form submits is exploitable.
- **Suggested fix:** Set both to `true`. If Streamlit's behind-proxy quirks require off locally, gate by env var or use deploy-time override.

---

## All HIGH-Severity Findings (compact)

| ID | Domain | File:Line | Confidence | Summary |
|----|--------|-----------|------------|---------|
| D1A-001 | Data Pipeline | data_bootstrap.py:2525 | HIGH | injury_writeback joins league_rosters on non-existent `name` column → 0 IL flagged (see BUG-003) |
| D1A-002 | Data Pipeline | data_bootstrap.py:2585 | HIGH | undroppable flag UPDATE references non-existent `name` column → 0 R1-3 picks flagged |
| D1A-003 | Data Pipeline | data_bootstrap.py:2287 | HIGH | Dynamic park factor uses team OPS+/wRC+ as proxy (BUG-008) |
| D1B-001..002 | Data Pipeline | data_bootstrap.py:2525,2588 | HIGH | (duplicate of D1A-001/002 — see BUG-003) |
| D1B-003 | Data Pipeline | live_stats.py:237 + player_databank.py:441 | HIGH | IP outs notation parsed as decimal (BUG-004) |
| D1B-004 | Data Pipeline | data_bootstrap.py:2031 + :1683 | HIGH | catcher_framing + umpire_tendencies seed stored with `season=2024`; consumers filter `WHERE season=current_year` |
| D1B-005 | Data Pipeline | data_bootstrap.py:2201 | HIGH | pvb_splits.slg hardcoded to 0.0 across all 1878 rows in production |
| D1B-006 | Data Pipeline | data_bootstrap.py:684,691 | HIGH | `_store_external_adp` stores ECR rank (1-300) into `fantasypros_adp` column (different scale) |
| D1D-001..011 | Data Pipeline | Various | HIGH | 14-arg `upsert_opp_pitcher` + similar 8-arg upsert functions; dict-typed `stats` params with implicit key schemas |
| D2A-001 | Yahoo+Valuation | validation/calibration_data.py:273,311 | HIGH | Calibration calls non-existent client methods (BUG-016) |
| D2A-002 | Yahoo+Valuation | yahoo_api.py:1689 | HIGH | `_INVERSE_CATS` evaluated at class-body time; config changes ignored |
| D2A-013 | Yahoo+Valuation | bayesian.py:406 | HIGH | Bayesian batch_update produces inconsistent h/ab (BUG-012) |
| D2A-017 | Yahoo+Valuation | ecr.py:1074 | HIGH | ECR source-weight key mismatch (BUG-014) |
| D2A-020 | Yahoo+Valuation | playing_time_model.py:284 | HIGH | weeks_elapsed = max(1, 26-weeks_remaining) for a 24-week fantasy season — misleading pre-season floor |
| D2B-008..012 | Yahoo+Valuation | bayesian.py:407-460,756 | HIGH | Bayesian formula-derived h/er/bb_allowed/h_allowed silently overwrite observed (BUG-012) |
| D2B-013 | Yahoo+Valuation | injury_model.py:362 | HIGH | Two-way player history overwritten by pitching gamesPlayed (BUG-013) |
| D2B-015 | Yahoo+Valuation | validation/survival_calibrator.py:181 | HIGH | Self-referential ADP = actual_pick (BUG-015) |
| D2B-016..019 | Yahoo+Valuation | validation/calibration_data.py + calibrate_constants.py | HIGH | Non-existent client methods called, swallowed by bare except (BUG-016) |
| YV-001..005 | Yahoo+Valuation | valuation.py, yahoo_data_service.py | HIGH | LeagueConfig fields typed as bare `dict`/`list`; Optional misannotation; `dict[str, Any]` returns everywhere |
| YV-006..013 | Yahoo+Valuation | yahoo_data_service.py, yahoo_api.py | HIGH | Inconsistent empty-result shapes (DataFrame vs None vs {}); `get_current_matchup` returns 10-key untyped dict |
| YV-019..025 | Yahoo+Valuation | ecr.py, playing_time_model.py | HIGH | Untyped dict inputs; dual-signature `compute_ecr_disagreement`; 7-arg playing-time predictors with boolean traps |
| YV-030..039 | Yahoo+Valuation | bayesian.py, validation/* | HIGH | Heterogeneous row schemas (hitter vs pitcher cols); `Any`-typed `yahoo_client` in calibration |
| D3A-001..002 | Optimizer | sigmoid_calibrator.py + sensitivity_analysis.py | HIGH | Patches dead aliases (BUG-006) |
| D3A-003 | Optimizer | projections.py:154 | HIGH | K3 block crashes when only one of xwoba_delta/babip_delta is present |
| D3B-001..014 | Optimizer | Various | HIGH | Multiple silent defaults (lineup_grade='C' on no-data; bare except returns empty; nan ERA propagates through streaming) |
| D3D-001..016 | Optimizer | pipeline.py, daily_optimizer.py, projections.py | HIGH | Untyped config; 12-arg optimize(); 11-arg build_daily_dcv_table; module singletons |
| D3D-019..022 | Optimizer | streaming.py + category_urgency.py | HIGH | `dict[str, callable]` typo (should be Callable); COUNTING_STAT_K alias staleness |
| D3D-024..028 | Optimizer | shared_data_layer.py | HIGH | Untyped `yds` parameter; mutable `_catcher_framing_cache` module global |
| D3D-031..039 | Optimizer | fa_recommender.py, scenario_generator.py, advanced_lp.py, pivot_advisor.py, sigmoid_calibrator.py | HIGH | Multiple `_LC` singletons; untyped config params |
| D3D-043..048 | Optimizer | lineup_optimizer.py, lineup_rl.py, sensitivity_analysis.py | HIGH | Param explosion; mutable module global `_bandit`; stringly-typed CONSTANT_PATCH_TARGETS |
| D4A-002 | Trade Engine | trade_intelligence.py:543 | HIGH | compute_fa_comparisons picks worst pitcher FA (BUG-017) |
| D4B-001 | Trade Engine | trade_evaluator.py + production/convergence.py | HIGH | MC convergence claimed, never measured (BUG-007) |
| D4B-002 | Trade Engine | trade_simulator.py:122 | HIGH | Odd n_sims leaves fake zero in surpluses array — bias toward 0 in aggregates |
| D4B-003 | Trade Engine | trade_simulator.py:324 | HIGH | Pitcher rate-stat fallback: missing WHIP defaults to 0 IP/WHIP → 0 baserunners contribution |
| D4B-005 | Trade Engine | trade_evaluator.py:846 | HIGH | DB roster rebuild silently swallows errors → uniform category weights without UI signal |
| D4B-011 | Trade Engine | engine/portfolio/category_analysis.py:200 | HIGH | Inverse-stat gainability hardcoded `gap < 0.5` for L/ERA/WHIP (very different scales) |
| D4B-014, D4B-015 | Trade Engine | engine/signals/decay.py, kalman.py | HIGH | Signals modules never called by trade engine (BUG-005) |
| D4B-020 | Trade Engine | trade_finder.py:1052,1351 | HIGH | `_player_sgp_volume_aware` returns 0.0 silently on missing player → bad pool data passes cap |
| D4B-021 | Trade Engine | trade_finder.py:986-1051 | HIGH | Four DB queries (ECR/YTD/transactions/calibration) swallow all errors → silently degraded trade decisions |
| D4D-001..004 | Trade Engine | engine/output/trade_evaluator.py | HIGH | evaluate_trade param explosion (12 args), untyped 35-key dict return, sgp_denoms regression risk, contract violation |
| D4D-008..016 | Trade Engine | engine/game_theory, monte_carlo, portfolio | HIGH | Untyped callbacks + `object` marginals + 14-arg signatures |
| D4D-025..029 | Trade Engine | trade_finder.py, trade_value.py | HIGH | 11-arg estimate_acceptance_probability; `__dict__.update` regression in trade_value (sister fixed in valuation.py); 580-line god-function evaluate_trade |
| D5A-004 | In-Season Strategy | standings_projection.py:189 | HIGH | Playoff tiebreaks deterministic by team list order → alphabetic bias |
| D5A-005 | In-Season Strategy | standings_engine.py:457 | HIGH | Asymmetric tie logic in cats_won counting |
| D5A-008 | In-Season Strategy | waiver_wire.py:535 | HIGH | Per-category deltas divide by per-player SGP denom (rate stats off by 100-1000x in UI) |
| D5A-013 | In-Season Strategy | standings_utils.py:108 | HIGH | `clear_cache` references `_cached_fa_pool` before its definition → NameError risk on cold start |
| D5A-025 | In-Season Strategy | war_room_hotcold.py:225 | HIGH | False hot-player alerts from 7-game extrapolation against projected (not actual) season HR baseline |
| D5B-001..007 | In-Season Strategy | in_season.py, opponent_trade_analysis.py, waiver_wire.py | HIGH | Sentinel rate-stat defaults (.250/.320/4.50/1.30); MC noise applied on delta SGP not totals; stale data inconsistency in opp_before vs live standings; `_CONSTANTS.get("weekly_rate_r")` returns None → TypeError silently swallowed |
| D5B-010 | In-Season Strategy | waiver_wire.py:858 | HIGH | scale=0 zeros all counting but rate stats pass through → polluted matchup_value |
| D5B-014 | In-Season Strategy | matchup_context.py:344 | HIGH | `get_matchup_adjustments` returns un-adjusted roster on failure indistinguishable from success |
| D5B-017 | In-Season Strategy | standings_engine.py:457 | HIGH | Tie-counting bug in matchup logic |
| D5B-021 | In-Season Strategy | weekly_h2h_strategy.py:670 | HIGH | DB fallback for league_records silently returns (0,0,0)+rank=6 |
| D5B-023 | In-Season Strategy | war_room.py:230 | HIGH | `denom = max(abs(opp_val), 0.001)` → gap_pct astronomical when opp_val=0 (early season) |
| D5B-025..027 | In-Season Strategy | war_room_actions.py + war_room_hotcold.py | HIGH | Empty schedule context = "all SP recommended"; dead code `_get_opponent_team` always returns None; rolling stats batch fails silently |
| D5B-030 | In-Season Strategy | start_sit_widget.py:24 | HIGH | Hardcoded SGP denominators bypass LeagueConfig (violates SF-21/SF-25 pattern) |
| D5B-031 | In-Season Strategy | weekly_report.py:184 | HIGH | `check_daily_lineup` returns `[]` when `todays_games is None` — silent skip of all daily validation |
| D5B-034 | In-Season Strategy | alerts.py:443 | HIGH | `compute_swap_impacts` outer `except Exception: return []` — any failure means "no swap opportunities" |
| D5B-037 | In-Season Strategy | ip_tracker.py:62 | HIGH | `ip_per_start = ip_season / 30.0` hardcodes 30 starts → undercounts remaining IP for SP |
| D5B-039 | In-Season Strategy | il_manager.py:120 | HIGH | Cold-start: every IL player flagged "new IL change" (BUG-022) |
| D5B-042 | In-Season Strategy | leaders.py:517 | HIGH | `compute_projection_skew` bare except → `projection_skew=""` for all players when DB fails |
| D5B-043..045 | In-Season Strategy | playoff_sim.py + standings_engine.py | HIGH | Hardcoded season_weeks=22 (vs 26 canonical); `_PLAYOFF_SPOTS=6` (vs league config 4); expected_t=0.0 (BUG-021) |
| D5D-001..006 | In-Season Strategy | opponent_trade_analysis, in_season, waiver_wire, standings_utils | HIGH | Optional `config` secretly required; inconsistent return types; 9-arg `compute_add_drop_recommendations`; untyped `yds`; implicit shared state ordering bug |
| D5D-012..021 | In-Season Strategy | start_sit, alerts, standings_engine | HIGH | 11-arg start_sit_recommendation; mixed concerns + DB I/O in alerts; 8-positional-arg simulate_season_enhanced |
| D6A-001..003 | Game Day+Draft | game_day.py:32, draft_engine.py:740,779,954 | HIGH | Hardcoded UTC-4 offset (no DST); `is_hitter=True` default flips pitcher mask in 3 places |
| D6A-005 | Game Day+Draft | simulation.py:380 | HIGH | DraftSimulator `_recent_pick_positions` leaks across `simulate_draft` calls |
| D6B-001 | Game Day+Draft | game_day.py:962 | HIGH | ERA/WHIP return 0.00 on no-IP → looks ELITE not "no data" |
| D6B-002..003 | Game Day+Draft | game_day.py:715 | HIGH | `row["fip"] = row["team_era"]` in statsapi fallback — FIP=ERA persisted silently |
| D6B-011..013 | Game Day+Draft | draft_engine.py:739,779,817 | HIGH | `pool.get("is_hitter", True)` returns scalar when col missing → 3 enhancement stages silently no-op |
| D6B-017 | Game Day+Draft | draft_state.py:266 | HIGH | NaN→0 coercion in roster totals silently mis-computes OBP denom |
| D6B-018 | Game Day+Draft | draft_grader.py:312 | HIGH | "Actual SGP" computed from projected stats — steals/reaches reflect projection rank, not realized value |
| D6B-022 | Game Day+Draft | simulation.py:430 | HIGH | `np.maximum(sgp_values, 0.01)` loses sign for inverse-stat-heavy pitchers (negative SGP floored) |
| D6B-024 | Game Day+Draft | backtesting.py:330 | HIGH | `_compute_value_capture` passes actual_stats as `pool` → mixes projection-roster against actual-pool baseline |
| D6D-001..010 | Game Day+Draft | draft_state.py, game_day.py, simulation.py | HIGH | Optional-mistype `dict = None`; mixed-case schema RosterTotals; 14-arg `simulate_draft` with 6 parallel arrays caller must keep aligned |
| D6D-013, D6D-021 | Game Day+Draft | simulation.py, contextual_factors.py | HIGH | Implicit shared state `_recent_pick_positions`; module-level `_CONSTANTS = load_constants()` frozen at import (vs optimizer reading at runtime) |
| D7A-001..002 | UI/Pages/Scripts | pages/9_League_Standings.py | HIGH | Hardcoded category sets; load_league_rosters bypasses YDS (BUG-009) |
| D7B-001..006 | UI/Pages/Scripts | pages/6_Line-up_Optimizer, /9, /7, /12 | HIGH | 24 vs 26 weeks scaler; default rate-stat substitution; silent demo roster fallback; Athletics streaming drop; YDS bypass in 2 pages (BUG-009, BUG-018, BUG-019, BUG-024) |
| D7D-003..012 | UI/Pages/Scripts | src/ui_shared.py, player_card.py | HIGH | Leaky dict-as-record everywhere (Recommendation, TradeResult, PlayerCardData); magic session-state strings |
| D7D-021..029 | UI/Pages/Scripts | app.py, pages/4_Trade_Analyzer | HIGH | Untyped render_* fns (100+ magic dict keys); 700+ lines of module-level imperative code in Trade Analyzer page |
| D7D-032..045 | UI/Pages/Scripts | pages/6, /9, /12, /11; scripts/optimal_roster_sim | HIGH | Hardcoded category lists in pages+scripts (outside structural guard scope) |
| D7D-042, D7D-044, D7D-048 | UI/Pages/Scripts | scripts/draft_vs_current.py, optimal_roster_sim.py, extract_trade_data.py | HIGH | Module-level `sys.stdout` mutation + DB connection open at import time (untestable) |
| D7D-054, D7D-056..057 | UI/Pages/Scripts | ui_shared.py (3583 lines), app.py (2504 lines), pages/1_My_Team.py (2099 lines) | HIGH | God modules; ui_shared owns theme + CSS + tables + dialog + session; pages/1 has 70+ lines of imperative module-level side effects |
| D7D-059 | UI/Pages/Scripts | st.session_state usage | HIGH | 30+ magic session-state string keys, no central registry |
| D8A-001 | Config | .streamlit/config.toml:10 | HIGH | XSRF + CORS both disabled (BUG-025) |
| DATA-001..033 | Stream B | players table | HIGH | 33 shadow rows with fake mlb_ids (BUG-001) |
| DATA-missing | Stream B | players.mlb_id IS NULL | HIGH | 3 rostered players with NULL mlb_id (BUG-002) |
| INFRA-F1..F2 | Stream C | refresh_log | HIGH | injury_writeback + draft_results SQL bugs (BUG-003) |

---

## MEDIUM-Severity Findings (compact, sample of 60 most impactful of 268)

| ID | Domain | File:Line | Summary |
|----|--------|-----------|---------|
| D1A-004 | Data Pipeline | data_pipeline.py:747 | `update_refresh_log("projections","success")` bypasses row-count validation |
| D1A-005 | Data Pipeline | database.py:1290 | Unreachable branch in `_load_health_scores_for_pool` |
| D1A-007 | Data Pipeline | database.py:1462 | F-string SQL with derived integer values in `_enrich_pool` |
| D1A-008 | Data Pipeline | data_bootstrap.py:221 + live_stats.py:921 + player_databank.py:533 | Inconsistent "ATH" vs "OAK" team code |
| D1A-009 | Data Pipeline | projection_blending.py:117 | YTD IP proxy hardcoded as 1.5 IP/GP for all pitcher roles |
| D1A-010 | Data Pipeline | player_databank.py:1224 | Hardcoded EST offset -5 ignores DST |
| D1B-007 | Data Pipeline | database.py:1314 | Health-score fallback returns 0.85 baseline silently |
| D1B-008..014 | Data Pipeline | data_bootstrap.py (12 phases) | `update_refresh_log("X","success")` without row-count gate |
| D1B-015..018 | Data Pipeline | data_bootstrap.py | T8 catcher fallback writes framing_runs=0; news_sentiment doesn't reset stale; save_mlb_transactions silently drops unmatched; ADP sub-step failure swallowed |
| D1D-012..014 | Data Pipeline | database.py:1821,1880,2747 | upsert functions take untyped `stats: dict` |
| D1D-015..016 | Data Pipeline | data_bootstrap.py:524,2564 | yahoo_client untyped throughout bootstrap orchestrator |
| D2A-003..012 | Yahoo+Valuation | yahoo_api.py + valuation.py | Standings parsing assumes Yahoo column names; canonicalize_team crashes on None; sign-flip in marginal-rate SGP |
| D2A-022..025 | Yahoo+Valuation | ml_ensemble.py, injury_model.py, validation/* | Series indexed differently by model state; injury_model game_available capped at 162 for pitchers; FANTASY_REGULAR_SEASON_WEEKS=22 vs 24 elsewhere |
| D2B-001..014 | Yahoo+Valuation | yahoo_api.py, yahoo_data_service.py, ecr.py | Multiple `update_refresh_log("success")` without row-count; ECR fetcher empty on exception |
| D2B-020..024 | Yahoo+Valuation | validation/constant_optimizer.py + playing_time_model.py | Only survival_sigma calibrated; SP=`"SP" in positions` substring miss; FA recommender pos.contains without word boundary |
| YV-014..018 | Yahoo+Valuation | yahoo_data_service, ecr | `_fallback_store` module mutable singleton; private `_client` attr mutation; `compute_ecr_disagreement` dual-signature |
| D3A-004..012 | Optimizer | shared_data_layer.py, daily_optimizer.py, streaming.py, pipeline.py | UTC weekday for ET fantasy week; pipeline drops confirmed_lineups; streaming constants desync; recursive retry drops inputs |
| D3B-015..030 | Optimizer | streaming, fa_recommender, projections, h2h_engine, pipeline, advanced_lp, daily_optimizer | NaN propagation; injury_history empty → 0.85 default; "Failed silently" reasons; sigmoid_calibrator validates `n_scenarios` |
| D3D-017..023 | Optimizer | projections, streaming, daily_optimizer, scenario_generator, advanced_lp, pivot_advisor, sigmoid_calibrator | 8-arg build_enhanced_projections; module mutable cache; untyped patches |
| D4A-003..010 | Trade Engine | trade_value.py, trade_evaluator.py, engine/portfolio | `__dict__.update` shallow copy; GRADE_THRESHOLDS None defaults; module singletons; inverse-stat 0.5 threshold |
| D4A-013..014 | Trade Engine | engine/portfolio/lineup_optimizer.py, concentration.py | Linear hot-FA probability vs compound; mixed int|float type union |
| D4B-004..030 | Trade Engine | trade_evaluator.py + trade_finder.py + engine/* | Status discrepancy in module ledger; FA recommender returns whole pool; KDE silent normal fallback; HHI counter masks missing PA/IP |
| D4D-009..024 | Trade Engine | engine/* | `dict[str, callable]` typo; inconsistent return; opaque `dict[str, object]` marginals; mutable cache; mutation via side effect |
| D4D-030..031 | Trade Engine | trade_evaluator.py, opponent_valuation.py | `max(...key=valuations.get)` with type:ignore; raw pd.read_sql in engine bypasses YDS |
| D5A-006..017 | In-Season | standings_engine, waiver_wire, matchup_context, ip_tracker, il_manager | SoS denom wrong; rate-stat ERA fallback chain; PSD projection silent; UTC weekday in fantasy week ET |
| D5A-018..031 | In-Season | start_sit, two_start, war_room_actions | Park-factor sign-flip cancellation; `p_w=0 or 0.5` truthy bug; CWS/CHW mismatch; leaders.compute_category_leaders omits L |
| D5B-008..036 | In-Season | All in-season modules | Many bare-except return-empty patterns; weather/health/closer DB failures silently use defaults; ESPN injury silent fallback |
| D5D-007..028 | In-Season | All in-season type hints | dict-everywhere; `_LC` module singletons; 27-field OptimizerDataContext; 7-arg start_sit |
| D6A-006..027 | Game Day+Draft | simulation.py, draft_engine, draft_state, simulation, backtesting | sgp_volatility positional alignment assumed; idxmin without dedup guard; backtester opponent always greedy; UTC hour as local index in weather |
| D6B-004..033 | Game Day+Draft | game_day, draft_engine, draft_analytics, draft_grader, schedule_grid, simulation | Wind dir parsing fail silent; team_strength neutral defaults; OPS string-or-fallback; pool reset_index destroys mapping |
| D6D-005..033 | Game Day+Draft | draft_state, draft_engine, draft_analytics, draft_grader, contextual_factors, backtesting_framework, pick_predictor | Untyped 7-arg fns; tuple-of-magic-keys; duplicated literal data; `weibull_survival.adp_distance` dead param |
| D7A-003..018 | UI/Pages/Scripts | pages/12, /6, /13, /11, /14, /7, scripts | Hardcoded categories; resource leaks in scripts (no try/finally on conn); inline format_stat bypass; multi-position split bug |
| D7B-007..023 | UI/Pages/Scripts | pages/2, /10, /5, /15, /11, /6, /4 | ATH not in Databank team filter; LEFT JOIN league_rosters silently labels all "FA"; bare except hides sections; category-blind sort heuristics |
| D7D-001..060 | UI/Pages/Scripts | src/ui_shared, src/player_card, src/cheat_sheet, app.py, pages/*, scripts/* | Missing type hints across 60 helpers; god modules; magic strings everywhere |
| D8A-002..009 | Config | .github/workflows + scripts/pre-commit + requirements.txt | No concurrency guard on refresh.yml; no permissions block; unpinned `>=` versions; unquoted $STAGED_PY |
| DATA-100..114 | Stream B | season_stats | Murakami/Vargas/Domínguez stale by 7-22 games |
| INFRA-F3..F8 | Stream C | bootstrap phases | umpire Tier 1 phase >240s; DataFreshnessTracker doesn't read refresh_log; 18/31 phases write success without row-count |

---

## LOW-Severity Findings (compact, sample of 30 of 159)

| ID | Domain | File:Line | Summary |
|----|--------|-----------|---------|
| D1A-011..017 | Data Pipeline | data_bootstrap, live_stats, database, news_fetcher | Duplicate "gmli" literal; over-inflated wOBA approximation; deduplicate_players temp table |
| D1D-022..030 | Data Pipeline | All | `__slots__` without type hints; bare `set` return type; rsync style monkey-patching |
| D2A-007..023 | Yahoo+Valuation | All | Rate-limit sleep skipped on failed weeks; live_draft_sync pd.isna check missing |
| D2B-005..025 | Yahoo+Valuation | ecr.py, validation/* | Stub function lies about row count; module triage data hardcoded |
| YV-026..041 | Yahoo+Valuation | ml_ensemble, bayesian, injury_model | DraftMLEnsemble.MODEL_AVAILABLE alias property antipattern |
| D3A-008..012 | Optimizer | backtest_runner, daily_optimizer, scenario_generator, advanced_lp, fa_recommender | NaN propagation through `or 0.0` fallbacks |
| D3B-016..030 | Optimizer | streaming, fa_recommender, daily_optimizer | Threshold drift; magic-number multipliers |
| D3D-035..048 | Optimizer | constants, sensitivity | Stringly-typed module paths in patch targets |
| D4A-011..014 | Trade Engine | trade_simulator, trade_evaluator, lineup_optimizer | Component noise sharing; type-erased counter |
| D4B-018..030 | Trade Engine | engine/* | Cache misses recompute without logging staleness; trade_finder ImportError fallback |
| D4D-017..024 | Trade Engine | engine/projections, signals | Inconsistent fallback; `Any` return mask; tuple of mixed types |
| D5A-019..033 | In-Season | All | Magic constants; PSD projection rounding; multiple SoT violations |
| D5B-016..046 | In-Season | All | Many silent fallback patterns repeated |
| D5D-026..028 | In-Season | leaders, in_season, waiver_wire | `compute_breakout_score` config ignored; magic-string verdict; mutable magic constants in drop cost |
| D6A-018..027 | Game Day+Draft | simulation, backtesting, contextual_factors, pick_predictor, draft_state | dropoff hardcoded 30%; backtester deterministic opponents; partial PF avg |
| D6B-029..033 | Game Day+Draft | closer_monitor, schedule_grid, contextual_factors, pick_predictor | Bare-zero job security; empty roster silent; "DH" unknown position |
| D6D-024..037 | Game Day+Draft | closer_monitor, schedule_grid, draft_grader, backtesting_framework, pick_predictor, draft_engine, draft_analytics | Pseudo-private `_headshot_img_html` exported; inconsistent return contracts |
| D7A-008..020 | UI/Pages/Scripts | various | Inline format_stat bypass; emoji in script output (install-hooks.py:24); 12-team hardcoded |
| D7B-014..023 | UI/Pages/Scripts | various | Counting stats shown as `.2f`; manual zero-conflation in trends; tag-add return code |
| D7D-046..060 | UI/Pages/Scripts | various | Inconsistent function signatures; `Any` types throughout; missing return types |
| D8A-005..011 | Config | requirements.txt, refresh.yml, seeds | No python_requires constraint; "FA" team code; retired umpires in seed |

---

## Per-Domain Summary

| Domain | Files reviewed | HIGH | MEDIUM | LOW | Total |
|--------|----------------|------|--------|-----|-------|
| 1. Data Pipeline | 26 | 11 | 24 | 17 | 52 |
| 2. Yahoo + Valuation Core | 18 | 24 | 31 | 15 | 70 |
| 3. Optimizer | 24 | 19 | 35 | 26 | 80 |
| 4. Trade Engine | 34 | 22 | 28 | 18 | 68 |
| 5. In-Season Strategy | 23 | 28 | 47 | 30 | 105 |
| 6. Game Day + Draft + Backtest | 13 | 13 | 30 | 18 | 61 |
| 7. UI / Pages / Scripts | 33 | 20 | 41 | 27 | 88 |
| 8. Config & Data Assets | 8 | 1 | 4 | 6 | 11 |
| Stream B Player Data | ~415 verified | 17+3 missing | 15 stale | 0 | 35 |
| Stream C Refresh Infra | 31 sources | 2 | 5 | 2 | 9 |
| **TOTAL** | **210+** | **142** | **268** | **159** | **569** |

---

## Patterns + Cross-Cutting Observations

### Pattern 1: SF-21 / SF-25 / SF-26 cleanup is structurally incomplete

The structural-invariant tests guard only `src/` modules and a specific allowlist of `pages/`. They miss:
- 11 `_LC = LeagueConfig()` module singletons across optimizer, engine, and in-season modules (only 3 engine modules covered by `test_engine_no_fallback_singletons.py`)
- 12+ hardcoded category lists in `pages/` and `scripts/` (only `src/` covered by `test_no_hardcoded_categories_in_src.py`)
- 6 hardcoded SGP-denominator dicts in non-src files
- 4 pages bypassing `get_yahoo_data_service` (2 not in `test_pages_yahoo_compliance.py` allowlist)
- 3 scripts using module-level `sys.stdout` mutation + DB connection at import (untestable)

**Recommendation:** Expand structural-test scope to include `pages/` and `scripts/` directories. Add tests for `_LC` singletons in all engine + optimizer + strategy modules, not just engine/portfolio.

### Pattern 2: "Success without verification" pervades bootstrap orchestration

18 of 31 bootstrap phases call `update_refresh_log("X", "success")` directly rather than `update_refresh_log_auto(..., expected_min)`. The SF-2/SF-3 audit fixed several phases by routing through `update_refresh_log_auto`, but didn't migrate every phase. This means a zero-row fetch (network blip, schema mismatch, API empty response) silently logs success. Operators cannot detect these from `bootstrap.log` or `refresh_log`.

Compounded by: many phases log critical failures at `logger.debug` level (which is below default), so `data/logs/bootstrap.log` doesn't surface the issue either.

**Recommendation:** Mechanical migration — convert every `update_refresh_log(name, "success")` call to `update_refresh_log_auto(name, count, expected_min, message=...)`.

### Pattern 3: `dict`-as-implicit-struct everywhere across cross-module boundaries

Almost every "structured result" in this domain (Recommendation, TradeResult, PlayerCardData, MatchupSummary, OpponentProfile, ECR consensus row, marcel prior, hierarchical posterior, batch updater output, RosterTotals, BootstrapSnapshot) is a plain `dict[str, Any]` with implicit known keys. Consumers fish via `.get(key, default)`. Typos silently produce zero values.

The codebase has working dataclasses (`LeagueConfig`, `DraftPick`, `CalibratableConstant`, `WeeklyMatchup`, `CalibrationDataset`, `_CacheEntry`, `TTLConfig`, `CacheStats`, `AnalyticsContext`, `SentimentResult`, `ILAlert`, `CheatSheetOptions`) — so the team knows the pattern. Extending it to cross-module result types (especially `Recommendation`, `TradeResult`, `MatchupSummary`) would catch most of these silent failures at type-check time without any runtime cost.

**Recommendation:** Define `TypedDict` or `@dataclass` for the top 10 cross-module result types. Start with `evaluate_trade` return shape (already a 35-key dict with mode-conditional keys) and the recommendation dict consumed by `app.py:render_hero_pick`.

### Pattern 4: Orphaned architectural subsystems

Two notable orphans:
1. **`src/engine/signals/decay,kalman,regime`** — ~600 lines that the trade engine never calls. Trade evaluator uses raw projections + ytd blend. The Phase 3 architecture diagram is aspirational.
2. **`src/engine/production/convergence.py`** — Defines `check_convergence`, `effective_sample_size`, `split_rhat`. Zero callers. The MC overlay claims confidence without quality assessment.

**Recommendation:** Either remove (clarifying the modules are for other purposes — e.g., signals.kalman is consumed by trade_signals UI helper) or wire them into the pipeline they were designed for.

### Pattern 5: Untyped client parameters mask API contract drift

Multiple validation/calibration modules accept `yahoo_client: Any` and call methods like `get_transactions()`, `get_matchups()`, `get_standings()` that don't exist on `YahooFantasyClient`. Bare exception handlers silently return empty data. The actual methods are `get_league_transactions`, `get_current_matchup`, `get_league_standings`.

A `Protocol` class declaring the required methods would catch these typos at type-check time. The harness for `validation/calibration_data.py` is currently a silent no-op for matchups, transactions, and standings — yet reports "calibration ran successfully."

**Recommendation:** Define `class YahooClientProtocol(Protocol)` enumerating the methods actually used by callers. Use this type everywhere `yahoo_client` is passed.

### Pattern 6: Inverse-stat handling has multiple distinct failure modes

L/ERA/WHIP get inconsistent treatment across the codebase:
- `engine/portfolio/category_analysis.py:200`: hardcoded `gap < 0.5` threshold for all three (ERA=0.5 is meaningful, WHIP=0.5 is impossible, L=0.5 is trivial)
- `simulation.py:430`: `np.maximum(sgp_values, 0.01)` loses sign for negative-SGP pitchers (closers with ugly ERA become indistinguishable from average)
- `start_sit.py:802-803`: pitcher ERA park-factor multiplier is `2.0 - park["park"]`, which sign-flips with `totals_sgp` auto-negation, producing under-penalty
- `pages/8_Trade_Values.py` and `pages/11_Trade_Finder.py:840-848`: rough-value heuristics omit L/ERA/WHIP entirely from sort keys
- Multiple `{"ERA","WHIP"}` hardcoded sets (without L) found in `src/player_card.py`, several pages — the same anti-pattern that caused the original My_Team L-drop bug

**Recommendation:** Audit every inverse-stat reference for sign-correctness. Add a structural test asserting inverse-stat sets always include L (companion to the existing `_INVERSE_CATS={"L","ERA","WHIP"}` invariant in `src/`).

### Pattern 7: God modules and module-level imperative code in `pages/`

`src/ui_shared.py` is 3583 lines combining theme + CSS + tables + dialog + session caches + matchup ticker + freshness widget + format_stat. `app.py` is 2504 lines combining splash + bootstrap + setup wizard + draft page + 17 render_* fns. `pages/1_My_Team.py` is 2099 lines with 70+ lines of imperative module-level code (DB queries, Yahoo reconnect, logo fetch).

Only `pages/11_Trade_Finder.py` wraps in `def main()`. Other pages run hundreds of lines of side-effecting module-level code, making unit testing impossible.

**Recommendation:** Split `ui_shared.py` into `ui_theme.py` / `ui_tables.py` / `ui_dialog.py` / `ui_session.py`. Wrap each page's body in `def render(): ...` and call from a tiny `__main__` block.

### Pattern 8: Stream B's "shadow rows" pattern indicates a 2024-era data migration that wasn't reconciled

The 33 fake-mlb_id rows in the 600000–601500 range — all with `team='MLB'` and resolving to DSL/VSL prospects — strongly suggest a previous mass-bootstrap that fabricated mlb_ids for unmatched players. The real players were eventually added but the shadow rows weren't cleaned. 4 of these are currently rostered.

Combined with 3 high-profile rostered players with NULL mlb_id (Hunter Greene, Ryan Pepiot, Dylan Crews), this is the highest-impact actionable data issue: a single dedup migration + mlb_id backfill pass would fix all 7 rostered players + 29 top-200 ADP players.

**Recommendation:** One-shot migration script: (a) `DELETE FROM players WHERE team='MLB' AND mlb_id BETWEEN 600000 AND 601999`; (b) for remaining players with NULL mlb_id, re-run `_match_mlb_id_by_name_team` via `statsapi.lookup_player`; (c) emit a refresh_log entry tier='cleanup' with count.

### Pattern 9: Date/time handling inconsistencies across the codebase

Multiple distinct issues:
- `game_day.py:32`: `_ET = timezone(timedelta(hours=-4))` (EDT-only — breaks during EST months)
- `player_databank.py:1224`: `_EST_OFFSET = timedelta(hours=-5)` (EST-only — breaks during EDT months)
- `shared_data_layer.py:563`: `datetime.now(UTC).weekday()` for fantasy week (ET-anchored)
- `ip_tracker.py:121`: `datetime.now(UTC).weekday()` for week-end calc
- 3 distinct `_SEASON_START` constants (2026-03-23 UTC, 2026-03-25 ET, 2026-03-25 ET)

**Recommendation:** Standardize on `zoneinfo.ZoneInfo("America/New_York")` for all fantasy-week time arithmetic. Centralize `_SEASON_START` in `LeagueConfig`. Add `compute_fantasy_week_offset()` helper consumed by all modules.

---

## Reviewer Methodology Notes

- **22 Stream A code-review agents** (8 domains × 3 reviewers, except Domain 8 which has 1 reviewer due to limited Python surface area). Each reviewed a sliced file list with one of three focus lenses (logic, silent failure, type/API). Reviewers cross-referenced canonical sources of truth (`LeagueConfig`, `SGPCalculator.totals_sgp`, `get_connection`, etc.) and pre-existing SF-1..SF-28 catalog to avoid duplicate reporting.
- **1 Stream B player-data correctness agent** verified ~415 players (314 rostered + 100 top-ADP non-rostered) against MLB Stats API. Yahoo verification skipped to fit time budget (already covered by Stream A reviewers checking page-level Yahoo data flow).
- **1 Stream C live-refresh infrastructure agent** verified all 33 sources in `refresh_log`, triggered 13 stale Group-A phases, and read 18 Group-B phases. Confirmed Tier 1/2/3 waterfall behavior end-to-end.
- All agents were read-only against source files. Stream C triggered idempotent refresh phases (writes to target tables only, by design); no destructive operations.
- Findings reflect state at 2026-05-11 with the audit working against the cranky-einstein-07be9b worktree, branched from the PR #11 merge commit.

---

## Next Step

This findings document is the input to a follow-up fix-implementation plan. Recommended grouping for parallel-agent fix dispatch:

1. **Wave 1 (data correction migrations):** BUG-001 (shadow rows), BUG-002 (NULL mlb_ids), BUG-003 (league_rosters.name SQL), BUG-008 (park factor dynamic refresh). All schema-aware, can run as one migration script.
2. **Wave 2 (silent-failure elimination):** BUG-004 (IP outs notation), BUG-012 (Bayesian formula override), BUG-013 (two-way player), BUG-014 (ECR weight keys), BUG-022 (IL cold start). Mostly isolated module-level fixes.
3. **Wave 3 (architectural cleanup):** BUG-005 (orphaned signals modules), BUG-006 (sigmoid calibrator), BUG-007 (MC convergence), BUG-010 (_LC singletons), BUG-015/016 (calibration harness wiring). Touches multiple modules but each in isolation.
4. **Wave 4 (UI/page wiring):** BUG-009 (YDS bypass in 2 pages), BUG-011 (DCV pipeline), BUG-018 (24 vs 26 weeks), BUG-019 (Athletics streaming), BUG-024 (demo roster banner). All single-file changes.
5. **Wave 5 (structural test expansion):** Add tests for hardcoded categories in pages+scripts; expand `_LC` singleton guards; widen YDS-compliance allowlist; add `update_refresh_log_auto` adoption test.

Use `superpowers:writing-plans` skill with this findings file + the proposed wave grouping to produce the fix-implementation plan.

# HEATER Beta Design-Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Every code task is **TDD**: the executing subagent **reads the cited current code first**, writes a failing test that encodes the fix, runs it red, implements the minimal change, runs it green, then commits. The plan gives the exact location, the failing-test intent, the fix, and the acceptance criteria; the subagent writes the literal test against the real current signatures.

**Goal:** Take HEATER from "not beta-ready" to a launch-grade in-season app by fixing every BLOCKER and HIGH issue surfaced by the two independent UI/UX audits (Claude + Hermes), inside the existing Combustion design system, then test, checkpoint, merge to master, and deploy live.

**Architecture:** All work happens on branch `feat/beta-design-audit-fixes` (master stays live). Fixes are grouped into phases by *theme and risk* — correctness first, then performance, then trust/comprehension, then per-page controls, then consistency/polish — because that order front-loads the highest-trust-impact, lowest-regression-risk work and keeps each phase independently shippable. Every change reuses existing canonical helpers (`resolve_viewer_team_name`, `format_stat`, `render_empty_state`, `render_data_freshness_card`, `total_sgp_batch`, `CONSTANTS_REGISTRY`) rather than inventing new patterns. Heavy roadmap features (Apply-to-Yahoo, multi-player trades, mobile, file refactors, thin-client) are fully specified in Phase 7 as a tracked backlog but are **not** built in this pass.

**Tech Stack:** Streamlit (multi-page), SQLite (WAL, single writer), pandas/NumPy/SciPy/Plotly, PuLP, the `src/optimizer/` + `src/engine/` packages, `pytest` (sharded), ruff, the Combustion design system in `src/ui_shared.py` (locked by `tests/test_combustion_lock.py`).

---

## Source Reconciliation (both audits → one verified backlog)

This plan consolidates **two** audits:

| Source | File | Method | Character |
|--------|------|--------|-----------|
| **Claude audit** | `HEATER Design_Audit_Claude.md` (repo root) | 15 per-page subagents → source + read-only live-DB; compiler synthesis | Depth: *verified* bugs with real captured outputs + file:line refs. 14 cross-cutting RECs (Part IV), per-page sections (Part V), master defect register (Part VI). |
| **Hermes audit** | `docs/design-audit/HEATER_Design_Audit_Hermes.md` | Live-browser interaction with the deployed Railway app + code reading | Breadth: UX/product/strategic, plus a few live-browser catches (the SGP fields, float display, the Next button). |

**Reconciliation rules applied:**

1. **Agreement = high confidence.** Where both audits independently flag the same issue (OBP+L missing from manual SGP inputs; float/number formatting; jargon without tooltips; "my-team" highlighting; per-page data-freshness; design-system drift; heavy-compute blocking), it is treated as confirmed and scheduled early.
2. **Conflicts are verified against the real code before any fix.** The one material conflict: **Hermes P0 "Next → button is non-functional"** vs. the Claude audit, which traced `render_step_launch()` and did *not* report a broken button. → **Phase 1, Task 1.1 is to verify this live before touching it.** Do not "fix" a button that may already work.
3. **Stale/incorrect Hermes findings are dropped.** Hermes claims *"no standalone `Pitcher_Streaming.py` exists"* — it does (`pages/4_Pitcher_Streaming.py`, verified loading live). Hermes' line counts and a few file sizes are from an earlier snapshot. These are excluded.
4. **Out-of-scope, by owner instruction:** the **HEATER AI chat panel** (`src/ai/*`) and the three admin pages (`pages/_admin_*.py`). No task in this plan touches them.
5. **Big features are spec'd, not built now** (Phase 7): Apply-to-Yahoo write-back, multi-player trades, save/load, watchlists, export/share, monolith refactors, async optimization, mobile responsiveness, onboarding, dark mode, accessibility, thin-client migration.

**Severity legend:** `[BLOCKER]` ships broken/false output or freezes the app · `[HIGH]` materially erodes trust or usability · `[MEDIUM]`/`[LOW]`/`[POLISH]` quality. Phases 1–2 are the launch gate.

---

## File-Impact Map (what gets touched and why)

| Area | Files | Responsibility in this plan |
|------|-------|------------------------------|
| Identity resolver | `src/auth.py`, `pages/2_Line-up_Optimizer.py`, `pages/16_Player_Compare.py`, `pages/14_Free_Agents.py` | Route every "my team" lookup through `resolve_viewer_team_name(rosters)`; extend the structural guard. |
| Heavy compute gating | `pages/11_Trade_Analyzer.py`, `pages/12_Trade_Finder.py`, `pages/10_Punt_Analyzer.py`, `src/scheduler.py` | Opt-in MC; cache pools; background/cached scan; vectorized SGP. |
| Number formatting | `src/ui_shared.py` (`format_stat`), `src/player_databank.py`, `pages/6_League_Standings.py`, `pages/1_My_Team.py`, `app.py` | IP→outs branch; enforce `format_stat`; Win% as %; integer counting stats. |
| Data freshness | `src/ui_shared.py` (freshness component), all 15 page files | One honest freshness chip per page; amber > 24h; kill false "Live"/"Updates hourly". |
| Broken outputs | `pages/17_Leaders.py`, `pages/12_Trade_Finder.py`, `pages/1_My_Team.py`, `src/trade_value.py`, `src/leaders.py`, `src/prospect_engine.py` | Volume gates, delta clipping, `is_hitter` fix, tier-before-decay, Statcast empty states. |
| Per-page controls | `pages/14_Free_Agents.py`, `pages/19_Player_Databank.py`, `pages/20_Draft_Simulator.py` | Remove false "sort" caption; fix/remove dead filters; fix Undo; numeric Acceptance sort. |
| Surfaced failures | `pages/1_My_Team.py`, `pages/14_Free_Agents.py`, `pages/5_Matchup_Planner.py`, `pages/10_Punt_Analyzer.py` | Replace `except: pass` with visible states; display `schedule_warning`; fix `player_news` SQL. |
| Jargon/tooltips | `src/ui_shared.py` (glossary component), all data-dense pages | `help=` tooltips on every coined label; reusable legend expander. |
| Design-system drift | `src/ui_shared.py`, `src/trade_value.py`, `src/player_databank.py`, `pages/4_Pitcher_Streaming.py`, `pages/5_Matchup_Planner.py`, `app.py` | Zero-pad FIG; off-palette hex → THEME tokens; emoji strip; `st.components.v1.html` → `st.iframe`. |
| In-season framing | `app.py`, `src/nav.py`, `pages/20_Draft_Simulator.py` | Default to My Team in-season; demote draft tools to a "Preseason" group. |
| Tests/guards | `tests/test_*` (new + extended) | One guard test per fix class so nothing regresses. |
| Docs | `CLAUDE.md`, memory | Update after deploy. |

---

## Phase 0 — Setup & Triage  *(DONE / in progress)*

**Goal:** Isolated branch, both audits in-repo, conflicts queued for verification.

- [x] **0.1** Create branch `feat/beta-design-audit-fixes` (master stays live).
- [x] **0.2** Rename `HEATER_Design_Audit.md` → `HEATER Design_Audit_Claude.md`.
- [x] **0.3** Copy Hermes audit into the repo: `docs/design-audit/HEATER_Design_Audit_Hermes.md`.
- [ ] **0.4** Commit the audits + this plan.
  ```bash
  git add "HEATER Design_Audit_Claude.md" docs/design-audit/ docs/superpowers/plans/
  git commit -m "docs(audit): consolidate Claude + Hermes design audits + implementation plan"
  ```
- [ ] **0.5** Baseline the suite so regressions are attributable. Run and record the pass count:
  `python -m pytest -q -x --ignore=tests/test_cheat_sheet.py -n auto` → expected ~5263 passing. Note any pre-existing failures so they aren't blamed on this work.

---

## Phase 1 — Correctness BLOCKERs  *(the launch gate — wrong/false output)*

**Goal:** No page renders provably-wrong data with confidence. Each task is TDD: read the cited code, write a failing test that asserts the *correct* behavior, then fix.

### Task 1.1 — VERIFY the Draft Tool "Next →" button (audit conflict)
**Files:** `app.py` (`render_step_settings()` → `render_step_launch()`, the `st.session_state.setup_step` flow).
**Problem:** `[BLOCKER?]` Hermes P0 #1 says "Next → produces no navigation; users cannot advance." The Claude audit traced the wizard and did **not** report this. This must be resolved by observation before any change.
- [ ] **Step 1:** Read the Next-button callback and `setup_step` increment/branch logic in `app.py`. Write an `AppTest` (`streamlit.testing.v1.AppTest`) smoke test that drives the app, clicks "Next →", and asserts the Launch step renders.
- [ ] **Step 2:** Run it. If it PASSES → the button works; mark this finding "not reproduced (Hermes stale)" in the plan log and **skip the fix**. If it FAILS → continue.
- [ ] **Step 3 (only if failing):** Fix the step transition (likely a missing `st.session_state.setup_step = 2` in the callback or a rerun that re-enters the settings branch). Use `systematic-debugging` — find the root cause, don't paper over it.
- [ ] **Step 4:** Re-run the AppTest → PASS. **Commit.**

### Task 1.2 — Identity/team-name resolver everywhere (the optimizer-zeroing bug)
**Files:** `pages/2_Line-up_Optimizer.py` (the `compute_nonlinear_weights()` call site), `pages/16_Player_Compare.py` + `pages/14_Free_Agents.py` (roster-status badge lookups), `src/auth.py` (`resolve_viewer_team_name`), `tests/test_pages_use_viewer_team_resolver.py`.
**Problem:** `[BLOCKER]` (both audits, Claude REC-2) `"Team Hickey"` (env) ≠ `"🏆 Team Hickey"` (DB) → all 12 category weights collapse to 1.00× (Lineup Optimizer) and rostered players show as "Free Agent" (Player Compare, Free Agents).
- [ ] **Step 1:** Write a failing test: build a rosters frame whose team name carries an emoji prefix, assert `resolve_viewer_team_name(rosters)` returns it AND that the Lineup Optimizer weight path receives the resolved name (not the bare env name).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Replace the bare `"Team Hickey"`/`user_team_name` passed into `compute_nonlinear_weights()` and the badge lookups with `resolve_viewer_team_name(rosters)` (the existing helper). Verify the same reconciliation (emoji/whitespace) the other 7 personalized pages already use.
- [ ] **Step 4:** Run → PASS. Extend `test_pages_use_viewer_team_resolver.py` to assert these call sites use the resolver (so they can't regress).
- [ ] **Step 5:** **Commit** `fix(identity): route Lineup Optimizer + Player Compare + Free Agents through resolve_viewer_team_name`.

### Task 1.3 — Add OBP + Losses to the manual SGP denominator inputs
**Files:** `app.py` (`render_step_settings()` manual-SGP block).
**Problem:** `[BLOCKER]` (both audits — Claude E-06, Hermes #2) Only 10 of 12 categories get number inputs when auto-compute is off; **OBP and L are missing**, so a 12-cat league can't be hand-tuned and downstream valuations are wrong.
- [ ] **Step 1:** Write a failing test asserting the manual-SGP input set, derived from `LeagueConfig`, contains all 12 categories (incl. `OBP`, `L`) — never a hardcoded 10-item list.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Drive the inputs from `LeagueConfig().all_categories` (not a literal list) so every category, including OBP and L, renders. Honor inverse-stat semantics for L.
- [ ] **Step 4:** Run → PASS. **Commit.**

### Task 1.4 — Float-formatting in the SGP spinbuttons + WHIP `.3f`
**Files:** `app.py` (spinbutton `value=`/`format=` and Category Balance tab), `src/ui_shared.py` (`format_stat`).
**Problem:** `[HIGH]` (Hermes #4, Claude REC-10) Spinbuttons show raw floats ("0.00800000037997961", "0.30000001192092896"); Category Balance renders WHIP `.3f`. Looks like a bug even when the math is right.
- [ ] **Step 1:** Failing test: assert the SGP-denominator display strings round to sane precision (AVG/OBP 3-4 dp, ERA/WHIP 2 dp) and that no rendered value contains a >6-dp float tail; assert WHIP uses `format_stat(x,"WHIP")`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Pass `format="%.3f"` (or `format_stat`) to the `st.number_input` displays and route the Category Balance WHIP through `format_stat`.
- [ ] **Step 4:** Run → PASS (also satisfies `test_pages_format_compliance.py`). **Commit.**

### Task 1.5 — Leaders Hot/Cold delta explosion + `is_hitter or 1` pitcher stats
**Files:** `pages/17_Leaders.py`, `src/leaders.py` (`_trend_key_stats`, the delta/RATE_FLOOR logic).
**Problem:** `[BLOCKER]` (Claude REC-4b, Page 13) Near-zero ROS IP collapses to `RATE_FLOOR=0.001` → deltas of +200…+935 dominate Hot/Cold; `int(row.get("is_hitter",1) or 1)` coerces `is_hitter=0` to 1, so **every pitcher shows hitter stats** (".000 AVG / 0 HR").
- [ ] **Step 1:** Failing tests: (a) a player with `ip_proj < 15` / `pa_proj < 50` is excluded from Hot/Cold; (b) deltas are `np.clip`-ed to [-3, +3] before classification; (c) `_trend_key_stats` shows ERA/K (not AVG/HR) for an `is_hitter=0` row.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Add the volume gate, clip the delta, and replace `int(x.get("is_hitter",1) or 1)` with an explicit `_is_hitter_safe(x)` that treats `0`/`0.0`/`"0"` as pitcher (never `or 1`).
- [ ] **Step 4:** Run → PASS. **Commit.**

### Task 1.6 — Statcast-NULL empty states (Breakouts all 50.0; My Team Statcast card)
**Files:** `pages/17_Leaders.py` (Breakouts tab), `pages/1_My_Team.py` (Statcast Signals + Regression Alerts cards), `src/ui_shared.py` (`render_empty_state`).
**Problem:** `[BLOCKER]` (Claude REC-4c) All `statcast_archive` rows are NULL → every Breakout scores exactly 50.0 (20 identical rows) and My Team's Statcast/Regression cards silently render nothing.
- [ ] **Step 1:** Failing tests: when the Statcast columns are entirely NULL, Breakouts renders `render_empty_state("Statcast data not loaded", …)` (zero scored rows) and My Team's Statcast card renders the same empty state (not a blank).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Gate both on Statcast availability (`df[cols].notna().any()`); show the empty state otherwise.
- [ ] **Step 4:** Run → PASS. **Commit.**

### Task 1.7 — Trade Finder Value-Chart tiers (Soto/Judge ≠ "Replacement")
**Files:** `pages/12_Trade_Finder.py` (Value Chart tab), `src/trade_value.py` (tier assignment vs time-decay).
**Problem:** `[BLOCKER]` (Claude REC-4a, Page 10) With 15 weeks left, time-decay pushes every value below the Elite/Star floors, so 9,887 of 9,888 players (incl. Soto, Judge) read "Replacement."
- [ ] **Step 1:** Failing test: with `weeks_remaining≈15`, the top-N players by raw value get non-Replacement tiers (Soto/Judge are Elite/Star), i.e. tiers are assigned from **pre-decay** value (or the cutoffs scale by `time_factor`).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Assign tiers before applying time-decay (or multiply the Elite/Star/Flex cutoffs by `time_factor`).
- [ ] **Step 4:** Run → PASS. **Commit.**

### Task 1.8 — Trade Finder "Sort by Acceptance" string-sort bug
**Files:** `pages/12_Trade_Finder.py` (Recommendations table sort).
**Problem:** `[HIGH]` (Claude REC-8) Acceptance sorts the *strings* "High"/"Medium"/"Low" alphabetically → Medium floats above High on a descending sort.
- [ ] **Step 1:** Failing test: sorting by Acceptance descending orders rows by the underlying numeric probability (High > Medium > Low).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Sort on the hidden numeric `acceptance_prob` column; render the label from it.
- [ ] **Step 4:** Run → PASS. **Commit.**

### Task 1.9 — Free Agents `player_news` SQL error (dead IL-safety layer)
**Files:** `pages/14_Free_Agents.py` (the `player_news` IL-protection query, ~line 721).
**Problem:** `[HIGH]` (Claude REC-7, Page 11) The query `SELECT … player_name FROM player_news` references a non-existent column → throws every load, silently caught by `except Exception: pass`. One of three IL-stash protections is permanently dead.
- [ ] **Step 1:** Failing test: the IL-protection query executes without raising against the real schema (join `players` for the name) and returns the expected protected set for a known IL player.
- [ ] **Step 2:** Run → FAIL (raises / empty).
- [ ] **Step 3:** Fix the query to join `players` for `name` (the column actually exists there); narrow the `except` so a future schema break is visible, not swallowed.
- [ ] **Step 4:** Run → PASS. **Commit.**

### Task 1.10 — Player Databank IP "56.7" → "56.2" (innings format)
**Files:** `src/player_databank.py` (`_format_cell`), `src/ui_shared.py`.
**Problem:** `[BLOCKER]` (Claude REC-10, Page 14) `56⅔` IP renders as `56.7` (decimal) instead of `56.2` (baseball outs notation) for every pitcher — a factually wrong stat.
- [ ] **Step 1:** Failing test: `format_innings(56.667) == "56.2"`, `format_innings(56.333) == "56.1"`, `format_innings(56.0) == "56.0"` (the inverse of `_ip_outs_to_decimal`).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Add an IP→outs format branch and apply it to the IP column in `_format_cell`.
- [ ] **Step 4:** Run → PASS. **Commit.**

### Task 1.11 — Ghost team "Twigs" in standings (rank /13 in a 12-team league)
**Files:** `pages/10_Punt_Analyzer.py`, `pages/6_League_Standings.py`, `src/standings_utils.py` (`filter_standings_to_valid_teams`).
**Problem:** `[HIGH]` (Claude REC-14, Pages 07/08) An abandoned team "Twigs" is in `league_standings` but not `league_rosters`, inflating `n_teams` to 13; Punt ranks show "/13" and the standings-points formula can yield 0.
- [ ] **Step 1:** Failing test: with a ghost row present, `filter_standings_to_valid_teams(standings, rosters)` drops it and the Punt page computes ranks against 12 teams.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Apply the existing `filter_standings_to_valid_teams()` on the Punt page (League Standings already guards via the roster filter — confirm). Optionally purge the ghost row at its DB source.
- [ ] **Step 4:** Run → PASS. **Commit.**

### Task 1.12 — Free Agents top-rec outlier + "Click to sort" lie + 200k-node freeze
**Files:** `pages/14_Free_Agents.py`.
**Problem:** `[HIGH]` (Claude Page 11) (a) The subtitle says "Click column headers to sort" but `render_compact_table` renders a **static** HTML table whenever Heat/Signal HTML columns are present (always) — the control is a lie. (b) "Show all 7,770 free agents" would render ~200k DOM nodes → browser freeze, no ceiling. (c) An extreme unexplained #1 (`marginal_value=125.19` vs #4's 11.8) from a team-mismatched row.
- [ ] **Step 1:** Failing tests: (a) the "click to sort" subtitle is absent when a static table is used (or sorting is actually wired); (b) "show all" is capped/​paginated server-side (no unbounded render); (c) the top-rec outlier is gated (z-score/IQR clamp on `marginal_value`).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Remove the false subtitle (or wire real sorting); add a hard page cap with server-side pagination; clamp/flag implausible `marginal_value` outliers.
- [ ] **Step 4:** Run → PASS. **Commit.**

### Task 1.13 — Draft Simulator "Undo Last Pick" + HTML-injection selectbox
**Files:** `pages/20_Draft_Simulator.py`.
**Problem:** `[HIGH]` (Claude Page 15) "Undo Last Pick" undoes the AI's pick and immediately re-picks it (the user's pick is never reverted); the "View player card" selectbox injects `<img>` HTML into option text.
- [ ] **Step 1:** Failing tests: Undo unwinds back **through the user's own last pick**; the player-select options contain no raw HTML.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Fix Undo to rewind to/through the user's pick; strip the headshot `<img>` out of the option label (render it separately).
- [ ] **Step 4:** Run → PASS. **Commit.**

### Task 1.14 — Phase 1 gate: run the correctness guards
- [ ] Run all new Phase-1 tests + the structural guards touched (`test_pages_use_viewer_team_resolver.py`, `test_pages_format_compliance.py`): `python -m pytest tests/ -q -k "viewer_team or format_compliance or trend or trade_value or standings or free_agent or databank or draft_sim" -n auto`. All green before Phase 2.
- [ ] **Commit** any guard extensions: `test(guards): lock Phase-1 correctness fixes`.

---

## Phase 2 — Performance BLOCKERs  *(stop the app from freezing itself)*

**Goal:** No page can drop the WebSocket or block rendering for tens of seconds. On one Railway replica with 12 users, a 45s synchronous compute is a self-inflicted DoS.

### Task 2.1 — Trade Analyzer: opt-in the 44.8s Monte Carlo
**Files:** `pages/11_Trade_Analyzer.py` (the `evaluate_trade(... enable_mc=...)` call + the evaluate button handler).
**Problem:** `[BLOCKER]` (Claude REC-1a, Page 09) `enable_mc=True` with 10,000 sims runs synchronously in the button handler (~44.8s), dropping the WebSocket and freezing the page. Tooltips still say "200 simulations."
- [ ] **Step 1:** Failing test: the default evaluate path calls `evaluate_trade(..., enable_mc=False)`; MC runs only when an explicit "Run risk analysis" checkbox is set; the MC tooltip/label states the real cost ("~45s, 10,000 sims").
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Default `enable_mc=False`; add a clearly-labeled opt-in checkbox ("Run deep risk analysis — adds ~45s"); render Phase-1 (deterministic) results immediately; correct the stale "200 simulations" tooltip to the real count.
- [ ] **Step 4:** Run → PASS. **Commit** `perf(trade-analyzer): make 10k-sim Monte Carlo opt-in (kills the 45s WebSocket-drop freeze)`.

### Task 2.2 — Cache the expensive pool loads
**Files:** `pages/11_Trade_Analyzer.py`, `pages/14_Free_Agents.py`, `pages/17_Leaders.py`, `pages/19_Player_Databank.py`, `pages/5_Matchup_Planner.py`.
**Problem:** `[BLOCKER/HIGH]` (Claude REC-1c, Page 09) `load_player_pool()` (~4.3s) + `get_health_adjusted_pool()` (~0.5s) re-run on **every** Streamlit rerun with no `@st.cache_data` → ~5s blank page on every navigation/interaction.
- [ ] **Step 1:** Failing test (AST/structural): each of these pages obtains its pool through an `@st.cache_data`-wrapped helper, not a bare per-rerun `load_player_pool()` in the page body.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Wrap the pool load in a module-level `@st.cache_data(ttl=...)` helper (mirroring the pattern already used elsewhere); ensure cache-clear on the global "Refresh All Data".
- [ ] **Step 4:** Run → PASS. **Commit.**

### Task 2.3 — Trade Finder: kill the 43.8s cold scan
**Files:** `pages/12_Trade_Finder.py`, `src/scheduler.py` (background writer), `src/trade_finder.py`.
**Problem:** `[BLOCKER]` (Claude REC-1b, Page 10) The page runs `find_trade_opportunities` across all 11 opponent rosters on load (~43.8s), blocking all rendering.
- [ ] **Step 1:** Failing test: the page does **not** call the full multi-team scan synchronously on first render; it reads cached results (or runs a reduced `max_results=20, top_partners=5` default) and offers an explicit "Expand search" action; a `caption`/log records when the cache was computed.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Move the full scan to the background scheduler (the existing sole SQLite writer) writing a cached results table; the page reads cache ≤ N hours old. Reduce the on-demand default scan size; gate the full scan behind a button.
- [ ] **Step 4:** Run → PASS. **Commit.**

### Task 2.4 — Punt Analyzer: vectorized SGP (413× faster)
**Files:** `pages/10_Punt_Analyzer.py`, `src/valuation.py` (`total_sgp_batch`).
**Problem:** `[HIGH]` (Claude REC-1d, Page 08) The page loops `df.iterrows()` calling `total_sgp()` on all 9,888 players **twice** (~2.1s blocking). `total_sgp_batch()` does the same in ~6ms.
- [ ] **Step 1:** Failing test: the punt recompute uses `total_sgp_batch(...)` (vectorized) and produces values equal (within tolerance) to the per-row path for a fixture roster; assert no `iterrows()` over the full pool remains in the punt compute path.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Replace the double `iterrows()` loop with `total_sgp_batch()`.
- [ ] **Step 4:** Run → PASS. **Commit.**

### Task 2.5 — Add progress feedback to the remaining heavy actions
**Files:** `pages/2_Line-up_Optimizer.py`, `pages/11_Trade_Analyzer.py`, `pages/6_League_Standings.py` (playoff sim), `pages/12_Trade_Finder.py`.
**Problem:** `[HIGH]` (both audits) Heavy actions hang with no spinner/progress; users can't tell working-vs-broken.
- [ ] **Step 1:** Failing/structural test: each heavy action is wrapped in `st.spinner(...)` (or `st.status`) with a descriptive label.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Wrap each long compute in `st.spinner("Optimizing lineup…")` / `st.status` with step updates where a callback exists.
- [ ] **Step 4:** Run → PASS. **Commit.**

### Task 2.6 — Phase 2 gate: measure the wins
- [ ] Re-benchmark Trade Analyzer default-path render (target < 6s, no WebSocket drop), Trade Finder first render (cached, < 3s), Punt recompute (< 0.1s). Record numbers in the plan log. **Commit** `perf: phase-2 heavy-compute gating complete`.

---

## Phase 3 — Trust & Comprehension  *(HIGH — make the novice believe and understand the numbers)*

**Goal:** Every page tells the user how fresh its data is, explains its jargon, and connects league-wide data to *their* team.

### Task 3.1 — One honest data-freshness chip on every page
**Files:** `src/ui_shared.py` (a `render_data_freshness_chip(source_age)` component if not already present), all 15 page files, the My Team "Live Matchup" card, the "Updates hourly" caption.
**Problem:** `[BLOCKER/HIGH]` (Claude REC-3, both audits) Data is up to ~2.8 days stale, yet a card says "Live Matchup," a caption claims "Updates hourly," and most pages show no staleness. "Warming up" is benign gray.
- [ ] **Step 1:** Failing tests: (a) a reusable freshness component renders human-relative age ("Updated 2 days ago") and uses the **amber** treatment when age > 24h; (b) the My Team matchup card title is not the literal word "Live" when the matchup is cache-served; (c) the hardcoded "Updates hourly" string is gone (replaced by computed age).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Build/extend the freshness chip; place it at the top of every page reading from a dated source; recolor "Warming up" as a warning; rename "Live Matchup" → "Matchup — Week N (cached)" when cache-served; replace "Updates hourly" with the computed age.
- [ ] **Step 4:** Run → PASS. **Commit** `feat(ui): honest per-page data-freshness chip; kill false "Live"/"Updates hourly"`.

### Task 3.2 — My Team matchup ticker gate + Record/Rank DB fallback
**Files:** `pages/1_My_Team.py` (`render_matchup_ticker()`, the `yahoo_connected` gate, the identity strip).
**Problem:** `[BLOCKER]` (Claude Page 02) `render_matchup_ticker()` only renders when `yahoo_connected` is truthy in session state — never set in cached/multi-user mode — so the primary "what's my score?" element is permanently invisible; Record/Rank show `—` for every read-only member though `league_standings` has the data.
- [ ] **Step 1:** Failing tests: the ticker renders from `yds.get_matchup()` (SQLite fallback) without requiring a live yfpy client; Record/Rank fall back to `league_standings` when the live client is absent.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Drive the ticker off `yds.get_matchup()` (already has cache fallback); read Record/Rank from `league_standings` when no live client.
- [ ] **Step 4:** Run → PASS. **Commit.**

### Task 3.3 — Glossary + tooltips for every coined term
**Files:** `src/ui_shared.py` (a small `render_glossary_expander(terms)` + a `JARGON` dict), the data-dense pages (Lineup Optimizer, Pitcher Streaming, Trade Analyzer, Trade Finder, Free Agents, Player Compare, Leaders, Standings, Matchup Planner).
**Problem:** `[HIGH]` (Claude REC-5, both audits) SGP, DCV, wRC+, "Smash", "% JOB", "Net SGP", "Magic#", "SOS", "Heat", "Sell-High", risk-flag strings, and raw scores are never explained — frequently at the decision point.
- [ ] **Step 1:** Failing tests: (a) a central `JARGON` dict defines each coined term with a one-line plain-English definition; (b) the key jargon column headers pass a `help=` tooltip (via `st.column_config` or header markup) sourced from `JARGON`; (c) each score-bearing page renders the reusable "What do these numbers mean?" expander.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Add the `JARGON` dict + `render_glossary_expander`; attach `help=` tooltips to the coined column headers; teach inverse stats inline ("L, ERA, WHIP — lower is better").
- [ ] **Step 4:** Run → PASS. **Commit** `feat(ui): glossary + jargon tooltips across data-dense pages`.

### Task 3.4 — Scale anchors for raw scores
**Files:** the pages with bare numeric scores (Pitcher Streaming Stream Score, Draft hero "combined_score", Trade grades, Free Agents Heat/marginal_value, Closer "% JOB").
**Problem:** `[HIGH]` (Claude REC-5c) Raw scores ("Score 56.10", "combined_score 33.89", "Heat 7") have no scale — a novice can't tell good from bad.
- [ ] **Step 1:** Failing test: each surfaced raw score carries a scale anchor — a tier label, a percentile, or a "league avg = X / out of Y" caption.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Add the anchor (e.g., "Stream Score 56 — Good (league avg 45)"; "Heat 7/10").
- [ ] **Step 4:** Run → PASS. **Commit.**

### Task 3.5 — "My team" highlighting + roster-aware lenses
**Files:** `pages/6_League_Standings.py`, `pages/17_Leaders.py`, `pages/19_Player_Databank.py`, `pages/3_Closer_Monitor.py`, `pages/10_Punt_Analyzer.py`, `pages/16_Player_Compare.py` (`compute_category_fit`).
**Problem:** `[HIGH/MEDIUM]` (Claude REC-11, both audits) League-wide data rarely connects to the user's team: no "my team" row highlight in Standings/Leaders/Databank; Closer Monitor doesn't flag your closers; Punt shows the whole 9,888 pool instead of "which of MY players lose value"; `compute_category_fit()` exists but is never called.
- [ ] **Step 1:** Failing tests: (a) the user's team row is visually marked in Standings/Leaders/Databank (a stable `is_mine` flag + a CSS class); (b) a "My Roster / Available / All" lens exists on Closer Monitor, Leaders, and the FA tables; (c) `compute_category_fit()` is invoked by Player Compare and returns the user's weak-category fit.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Add the `is_mine` highlight (resolve via `resolve_viewer_team_name`); add the lens chips; wire `compute_category_fit()` into Player Compare.
- [ ] **Step 4:** Run → PASS. **Commit** `feat(ui): roster-aware highlighting + lenses across league-wide pages`.

### Task 3.6 — Surface swallowed failures (visible states, not `except: pass`)
**Files:** `pages/1_My_Team.py`, `pages/14_Free_Agents.py`, `pages/5_Matchup_Planner.py` (`schedule_warning`, line ~666), `pages/10_Punt_Analyzer.py`.
**Problem:** `[HIGH]` (Claude REC-7) My Team cards vanish on exception; Matchup Planner builds `schedule_warning` but never renders it (27 "Avoid" rows with no explanation); Punt's standings panel disappears on import error; the FA fallback warning is jargon ("P3.5 fixes / Check logs").
- [ ] **Step 1:** Failing tests: `schedule_warning` is rendered via `st.warning` when present; bare `except: pass` blocks around card/panel render are replaced with `render_empty_state(...)`/visible `st.warning`; the FA fallback message is plain-English.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Render `schedule_warning`; replace the swallow-blocks with visible degradation states; rewrite the fallback copy.
- [ ] **Step 4:** Run → PASS. **Commit.**

### Task 3.7 — Right-size the overwhelming tables; front-load the answer
**Files:** `pages/4_Pitcher_Streaming.py` (17-col board), `pages/14_Free_Agents.py` (≤21-col), `pages/11_Trade_Analyzer.py` (20-section output).
**Problem:** `[HIGH]` (Claude REC-9) The one number that matters is buried — Streaming Score is column 7; the Trade verdict is the 3rd thing rendered (after the playoff sim).
- [ ] **Step 1:** Failing tests: (a) a "Top pick today" callout precedes the Streaming board and Score/Rank is column 1; (b) the ACCEPT/DECLINE verdict is the **first** thing rendered on the Trade Analyzer, advanced metrics behind expanders; (c) Free Agents/Streaming default to a compact column set with a "Show all stats" toggle.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Add the callout + reorder columns; hoist the verdict; add the compact/expanded toggle.
- [ ] **Step 4:** Run → PASS. **Commit.**

### Task 3.8 — Phase 3 gate
- [ ] Run the Phase-3 tests + `test_combustion_lock.py` (the freshness/glossary components must not break the locked design system). All green. **Commit** `feat(ui): phase-3 trust & comprehension complete`.

---

## Phase 4 — Per-Page Controls, Action Guidance & In-Season Framing  *(HIGH/MEDIUM)*

**Goal:** Every control does what it says; diagnosis leads to a next action; in-season users land in the right place.

### Task 4.1 — In-season framing: default to My Team, demote the draft tools
**Files:** `app.py` (the default landing logic), `src/nav.py` (`PAGE_REGISTRY` grouping, `build_pages`), `pages/20_Draft_Simulator.py` (the `PRESEASON` eyebrow).
**Problem:** `[HIGH]` (Claude REC-6, both audits) The default landing page is a dead draft wizard; the Draft Simulator says "PRESEASON" with no context. In-season leaguemates start in the wrong mental model.
- [ ] **Step 1:** Failing tests: (a) when `weeks_remaining > 0`, the default page is **My Team** (Draft Tool is no longer `default=True`); (b) `nav.build_pages` puts Draft Tool + Draft Simulator in a labeled "Preseason / Draft" group; (c) the Draft Simulator eyebrow reads `SCOUTING`, not `PRESEASON`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Make My Team the in-season default; regroup the nav; add a one-line in-season framing banner to the Draft Tool ("Draft's done — here's where to go," with quick links to My Team / Lineup Optimizer / Free Agents) and the Draft Simulator ("Use this to test values or prep for next year"); change the eyebrow.
- [ ] **Step 4:** Run → PASS (update `test_nav.py`/`test_app_main_auth_gate.py` expectations). **Commit** `feat(nav): in-season default to My Team; demote draft tools to a Preseason group`.

### Task 4.2 — Player Databank dead controls
**Files:** `pages/19_Player_Databank.py`, `src/player_databank.py` (`filter_databank`, the stat-view selector).
**Problem:** `[HIGH]` (Claude REC-8, Page 14) "Waivers Only" returns all 4,408 players (no handler); "Util" position filter always returns 0 (no player has "Util" in `positions`); "Today (live)" view shows ROS projections (false label); the 4 "special" views (Ranks/Research/Fantasy Matchups/Opponents) are stubs showing default data; off-palette `#2c2f36` in inline CSS (line ~1071).
- [ ] **Step 1:** Failing tests: "Waivers Only" filters to actual waiver rows (or the option is removed); "Util" maps to the multi-position eligibility that actually exists (or is removed); the "Today (live)" view is renamed "ROS Projections"; stub views are removed or implemented; no banned hex in the module.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement or remove each dead control; rename the false view; replace `#2c2f36` with a THEME token.
- [ ] **Step 4:** Run → PASS. **Commit.**

### Task 4.3 — Action guidance + deep-links (diagnosis → action)
**Files:** `pages/6_League_Standings.py`, `pages/1_My_Team.py`, `pages/5_Matchup_Planner.py`, `pages/12_Trade_Finder.py`, `pages/10_Punt_Analyzer.py`, `src/ui_shared.py` (`render_reco_banner` `expanded_html`).
**Problem:** `[MEDIUM]` (Claude REC-12) Pages diagnose but don't prescribe: Standings says "2 GB" and stops; Matchup shows "L: 14%" with no next step; My Team flip suggestions don't link anywhere; Trade Finder recs can't be sent to the Trade Analyzer.
- [ ] **Step 1:** Failing tests: (a) the reco banner's `expanded_html` carries a one-line "do this next" on Standings/Matchup/My Team; (b) Trade Finder rows expose an "Analyze in Trade Analyzer →" action that pre-populates the proposal (via `st.session_state`/query params); (c) weak-category suggestions deep-link to Free Agents / Pitcher Streaming.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Populate `expanded_html` with next-step copy; add the deep-links; wire the Trade Finder → Trade Analyzer hand-off.
- [ ] **Step 4:** Run → PASS. **Commit.**

### Task 4.4 — Matchup Planner week-navigator cap + cryptic labels
**Files:** `pages/5_Matchup_Planner.py` (the `< 24` week cap line ~475; the "Smash"/"Tie 19%" labels).
**Problem:** `[HIGH/MEDIUM]` (Claude Page 06) A hardcoded `< 24` blocks weeks 25–26 of a 26-week season (should use `config.season_weeks`); "Smash Matchups" is unexplained jargon; "Tie 19%" mislabels the 6-6 draw probability; "Average Games: 0.00" misleads when offline.
- [ ] **Step 1:** Failing tests: week navigation reaches `config.season_weeks` (26); "Smash" carries a tooltip; the draw metric is labeled "Draw (6-6)"; offline "Average Games: 0.00" shows an explanatory empty/warning state.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Replace `< 24` with `< config.season_weeks`; add the tooltip; relabel; guard the offline zero.
- [ ] **Step 4:** Run → PASS. **Commit.**

### Task 4.5 — Closer Monitor: AZ→ARI stat mismatch + "30-team" honesty + dual-SV label
**Files:** `pages/3_Closer_Monitor.py`, `src/closer_monitor.py` (`build_closer_grid` pool lookup).
**Problem:** `[BLOCKER/HIGH]` (Claude Page 04) AZ→ARI normalization mismatch blanks Paul Sewald's SV/ERA/WHIP while the green "2026 ACTUAL · 15 SV · 3.47 ERA · 0.73 WHIP" line shows below — a visible contradiction; the header claims "30-team" but 9 teams (BAL/CIN/LAA…) are simply absent; two save counts (projected vs actual) appear with no distinction.
- [ ] **Step 1:** Failing tests: the ARI closer's primary stat block matches the "2026 ACTUAL" line (no blank-vs-populated contradiction); the header text reflects the actual team count shown (or the 9 missing teams render an explained placeholder); the projected SV figure is labeled "PROJ SV".
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Fix the AZ→ARI normalization in the pool lookup; make the header count honest (or fill/explain the missing teams); label the dual SV figures.
- [ ] **Step 4:** Run → PASS. **Commit.**

### Task 4.6 — Player Compare: injury-badge HTML + same-type/cross-type guards
**Files:** `pages/16_Player_Compare.py` (`render_styled_table` vs `render_compact_table`).
**Problem:** `[BLOCKER/HIGH]` (Claude Page 12) Injury health dots render as **raw HTML text** (`st.dataframe` never interprets HTML); two hitters produce a radar with 6 dead flat axes; a hitter-vs-pitcher comparison yields a meaningless composite with no warning.
- [ ] **Step 1:** Failing tests: the health column renders via `render_compact_table(html_cols=...)` (not raw text); a same-type comparison suppresses the dead radar (or notes it); a cross-type comparison shows a warning instead of a composite.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Switch the health table to `render_compact_table` with `html_cols`; guard the radar/composite for same-type and cross-type cases.
- [ ] **Step 4:** Run → PASS. **Commit.**

### Task 4.7 — Phase 4 gate
- [ ] Run Phase-4 tests + `test_nav.py` + `test_combustion_lock.py`. All green. **Commit** `feat: phase-4 controls + framing complete`.

---

## Phase 5 — Consistency, Formatting & Design-System Drift  *(MEDIUM/LOW/POLISH)*

**Goal:** Remove the small tells that make a sharp app look unfinished. All inside the Combustion system; extend the locks.

### Task 5.1 — Number formatting sweep
**Files:** `pages/6_League_Standings.py` (Win% `.318`→"31.8%"), `pages/1_My_Team.py` (Marcel integer stats "15.00"→"15"), `app.py` (Category Balance WHIP), `pages/16_Player_Compare.py` (inverse-stat z-scores), `src/ui_shared.py` (`format_stat`).
**Problem:** `[HIGH/MEDIUM]` (Claude REC-10) Win% reads like a batting average; counting stats show ".00"; inverse-stat z-scores show negatives "winning" categories.
- [ ] **Step 1:** Failing tests: Win% renders "31.8%"; counting stats render as integers; inverse-stat (ERA/WHIP/L) z-scores are quality-adjusted so higher is always better, annotated "(lower is better)".
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Apply the formats; flip inverse-stat z-score sign for display + annotate.
- [ ] **Step 4:** Run → PASS. **Commit.**

### Task 5.2 — FIG numbering + nav/title drift
**Files:** `pages/4_Pitcher_Streaming.py` (`FIG.4`→`FIG.04`), `pages/5_Matchup_Planner.py` (stray `FIG.02` caption on a FIG.05 page), `pages/10_Punt_Analyzer.py` (nav "Punt Analyzer" vs title "Punt Strategy Simulator"), `src/ui_shared.py` (`render_page_header` if it owns FIG formatting).
**Problem:** `[MEDIUM/LOW]` (both audits, Claude REC-13) Zero-pad inconsistency; a stale within-page figure caption; nav-vs-title mismatch.
- [ ] **Step 1:** Failing/structural test: every `FIG.NN` eyebrow is two-digit zero-padded; no within-page caption contradicts the page's own FIG number; the Punt nav label and H1 title match (pick one).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Zero-pad `FIG.4`→`FIG.04`; remove/correct the stray `FIG.02` caption; reconcile the Punt label/title.
- [ ] **Step 4:** Run → PASS. **Commit.**

### Task 5.3 — Off-palette hex + emoji + animation drift → THEME
**Files:** `src/trade_value.py`, `src/player_databank.py` (`#2c2f36`), `pages/11_Trade_Analyzer.py` (emoji in the Weekly H2H expander label; `slideUp` on the persistent verdict banner; internal labels "Feature 2"/"report B.9"), the AVOID badge color, positive-SGP-in-ember-red cells, `tests/test_no_offpalette_hex_in_pages.py`.
**Problem:** `[MEDIUM/LOW]` (both audits, Claude REC-13) Off-palette hex in `src/`; emoji in toasts/labels; `slideUp` replaying on every rerun; AVOID badge orange (CTA color) instead of red; positive SGP shown in ember (danger) red; engineering jargon in user copy.
- [ ] **Step 1:** Failing tests: no banned hex anywhere in `src/` (extend the guard beyond `pages/`); no emoji in `inject_custom_css`/page toasts/labels; no entrance animation on persistent cards; AVOID uses `T["danger"]`; positive SGP uses `T["primary"]`; no "Feature N"/"report B.x" strings in user-facing copy.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Replace hex with THEME tokens; strip emoji; remove `slideUp`; fix the badge/figure colors; rewrite the engineering labels.
- [ ] **Step 4:** Run → PASS. Extend `test_no_offpalette_hex_in_pages.py` to scan `src/`. **Commit.**

### Task 5.4 — `st.components.v1.html` → `st.iframe` + dead-column sweep
**Files:** `app.py` (splash timer), `pages/6_League_Standings.py` (always-empty Streak col), `pages/19_Player_Databank.py` + `pages/14_Free_Agents.py` (always-dash "% Ros" while a 608-row ownership cache sits unused), `pages/3_Closer_Monitor.py` (empty SETUP/gmLI rows).
**Problem:** `[MEDIUM/LOW]` (Claude REC-14, both audits) Deprecated API past removal date (warning spam, future breakage); several always-empty columns.
- [ ] **Step 1:** Failing tests: no `st.components.v1.html` in `app.py`; the dead columns are either populated (join `yahoo_free_agents` for real ownership) or removed.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Swap the timer to `st.iframe`; join real ownership for "% Ros" or drop the column; populate or remove Streak/SETUP/gmLI.
- [ ] **Step 4:** Run → PASS. **Commit.**

### Task 5.5 — Empty/error states via the Combustion component (not native `st.info`)
**Files:** the pages still using bare `st.info`/`st.subheader` for data-empty contexts, `src/ui_shared.py` (`render_empty_state`).
**Problem:** `[LOW/POLISH]` (Claude REC-13) Native components break the instrument aesthetic where the Combustion empty-state is specified.
- [ ] **Step 1:** Structural test: data-empty contexts use `render_empty_state(...)` (operational/validation warnings may stay native).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Swap data-empty `st.info` for `render_empty_state`.
- [ ] **Step 4:** Run → PASS. **Commit.**

### Task 5.6 — Phase 5 gate
- [ ] Run `test_combustion_lock.py` + `test_no_offpalette_hex_in_pages.py` (extended) + `test_pages_format_compliance.py`. All green. **Commit** `style: phase-5 design-system drift cleanup complete`.

---

## Phase 6 — Verify → Checkpoint → Deploy → Document

**Goal:** Prove the whole change set is green, get the owner's go, ship to the live app, confirm it live, and update the docs.

### Task 6.1 — Full local verification
- [ ] Run the full suite in parallel (matches CI): `python -m pytest tests/ -n auto --dist loadfile --ignore=tests/test_cheat_sheet.py`. Expected: ≥ the Phase-0 baseline pass count + the new tests, **0 failures**. Fix any regression (use `systematic-debugging`) before proceeding.
- [ ] Run lint/format: `python -m ruff check . && python -m ruff format --check .`.
- [ ] Run the structural-invariant pre-push set: `python -m pytest tests/ -q -k "no_ or pages_ or combustion or guard or backcompat" -n auto`.

### Task 6.2 — Live smoke on the branch (preview)
- [ ] Start the app from the branch (`preview_start streamlit-app`), log in as the QA admin, and walk the pages that changed most (Trade Analyzer load time; Lineup Optimizer weights non-1.00×; My Team matchup ticker visible; Standings Win% "%"; Databank IP "x.y" outs; Closer ARI stats; Free Agents recs). Capture before/after evidence (timings + screenshots/snapshots). Record in the plan log.

### Task 6.3 — CHECKPOINT WITH OWNER (do not skip)
- [ ] Post a concise summary to the owner: phases completed, test pass count, the headline before/after wins, anything deferred, and the diff stat. **Wait for explicit "go" before merging to master / deploying.** (This is the one outward-facing, hard-to-reverse step; the live app serves 12 real users on a single replica.)

### Task 6.4 — Merge to master + deploy
- [ ] On "go": ensure the branch is rebased/clean, pre-push hooks pass, then merge to `master` (`--no-ff`) and push. Railway deploys on push to master.
- [ ] Watch the Railway deploy + health check (`/_stcore/health`); confirm the app comes up and the scheduler thread starts (single writer invariant).

### Task 6.5 — Live post-deploy verification
- [ ] Hit the production URL, log in, re-verify the top fixes against the live deploy (not the branch preview). Confirm no console/WebSocket errors on the previously-frozen pages. Record results.

### Task 6.6 — Update docs + memory
- [ ] Update `CLAUDE.md`: note the beta design-audit fix pass, the new/extended guard tests, the in-season default-landing change, the freshness/glossary components, and any new canonical helpers. Add a row per new structural guard to the Structural Invariants table.
- [ ] Update project memory (`project_design_audit_2026_06_13.md`): mark BLOCKER/HIGH fixes shipped + deployed; link the plan + both audit files.
- [ ] **Commit** `docs(claude.md): record beta design-audit fix pass + new guards`.

---

## Phase 7 — Roadmap (SPEC'D, NOT BUILT THIS PASS)

These are the larger items from both audits — real, valuable, but multi-day/multi-week features that should not be rushed onto a live app in the same pass as the launch-gate fixes. Each is captured here so nothing is lost; each becomes its own brainstorm → spec → plan cycle later. **Ordered by value.**

| ID | Feature | Source | Why deferred | First step when picked up |
|----|---------|--------|--------------|---------------------------|
| R-1 | **Apply Lineup / Add-Drop to Yahoo (write-back)** | Hermes (Lineup, Free Agents) | Needs Yahoo write-scope OAuth + safety/confirmation gates + the single-writer story; highest-value but highest-risk. | Spec the write path + confirmation UX + rate limits; prototype on a test team. |
| R-2 | **Multi-player trades (3-for-2, 3-for-3)** | Hermes (Trade Analyzer) | Engine currently 1-for-1 / 2-for-1; combinatorial UI + evaluation work. | Extend `evaluate_trade` to N-for-M; design the multi-select UX. |
| R-3 | **Save/load lineups, trades, comparisons; watchlists** | Hermes (Lineup, Trade, Compare, FA, Closer) | New persisted user state + per-user tables under MULTI_USER. | Design the `user_saved_*` schema (per-user, read-mostly). |
| R-4 | **Export/share (CSV, PDF, image, shareable link)** | Hermes (most pages) | Cross-cutting; needs a render/export service. | Add CSV + clipboard first (cheap), then PDF/image. |
| R-5 | **Monolith refactor** (`1_My_Team.py` ~2.4k lines, `2_Line-up_Optimizer.py` ~3.6k) | Hermes (My Team, Lineup) | Large, regression-prone; do behind the green guard suite. | Extract `roster_table.py`, `category_heatbars.py`, tab modules. |
| R-6 | **Async / backgrounded optimization** | Hermes (Lineup, Draft Sim) | Streamlit threading model + state hand-off; partial overlap with Phase 2. | Spec a worker + result-poll pattern compatible with the single writer. |
| R-7 | **Mobile responsiveness** | Hermes + Claude (cross-cutting) | Real layout work across 15 pages; the audit confirmed the desktop-first build. | Audit each page at 380px; collapse tables to cards; touch targets ≥44px. |
| R-8 | **First-run onboarding / guided tour** | Hermes (cross-cutting) | New surface; sequence it after the in-season framing (Task 4.1) lands. | 3-step "what is this app / where do I go / how fresh is data" tour. |
| R-9 | **Accessibility pass** (ARIA, keyboard nav, non-color indicators) | Hermes (cross-cutting) | Broad; pairs with the mobile pass. | Add text labels to color-only indicators; ARIA on custom HTML. |
| R-10 | **Dark mode toggle** | Hermes | Theming work atop Combustion; low urgency. | Add a `[theme]` variant + toggle. |
| R-11 | **Thin-client migration (FastAPI/Next.js + WebSockets)** | Hermes + the Product Review | Strategic re-platform; explicitly out of a beta-fix pass. | Separate product decision; not a UI-audit task. |

*(The Product Review's monetization/tenancy items — multi-league abstraction, observability/SLOs, backup/durability, security hardening — are business/architecture decisions tracked separately from this UI/UX audit, not folded in here.)*

---

## Self-Review (run before handoff)

- **Spec coverage:** Every one of the Claude audit's 14 cross-cutting RECs maps to a task (REC-1→2.1-2.4, REC-2→1.2, REC-3→3.1, REC-4→1.5-1.7, REC-5→3.3-3.4, REC-6→4.1, REC-7→3.6, REC-8→1.8/4.2, REC-9→3.7, REC-10→1.4/1.10/5.1, REC-11→3.5, REC-12→4.3, REC-13→5.2-5.3/5.5, REC-14→5.4). Every Hermes P0/P1 maps too (Next→1.1, OBP/L→1.3, float→1.4, loading states→2.5, design system→3.x/5.x, my-team highlight→3.5, freshness→3.1; the Hermes "Pitcher Streaming doesn't exist" is correctly dropped as stale). Per-page BLOCKER/HIGH from the Claude defect register are each scheduled.
- **Conflict handling:** the one audit conflict (Next button) is Task 1.1 = verify-before-fix.
- **Placeholders:** none — every task names exact files, the problem with source + severity, the fix, and a concrete failing-test intent + acceptance. (Literal per-line test code is written by the executing subagent against current signatures, per the TDD note in the header — deliberate, not a gap.)
- **Naming consistency:** reused helper names match the codebase (`resolve_viewer_team_name`, `format_stat`, `render_empty_state`, `total_sgp_batch`, `filter_standings_to_valid_teams`, `compute_category_fit`, `render_reco_banner`).
- **Scope:** launch-gate (Phases 1–2) is tight; Phases 3–5 are the trust/polish layer; Phase 6 ships; Phase 7 is explicitly deferred. No unrelated refactoring pulled into the live pass.

---

## Execution Handoff

Phases 1–5 are independent task sets that mostly touch different files, so they parallelize well across subagents, with a verify/commit gate at the end of each phase. Phase 6 is sequential and gated on the owner checkpoint. Recommended: **subagent-driven-development** — a fresh subagent per task (read code → TDD → fix → verify → commit), with a review between tasks and a full-suite gate between phases.


# HEATER Fix Campaign — 2026-06-07 (execution tracker)

Source of truth for executing the fixes + enhancements from
`docs/audits/2026-06-07-heater-full-audit.md`. Branch: **`fix/audit-followups-2026-06-07`**.

**Owner decisions (2026-06-07):** maximal scope.
1. **PV-C1** → *full fix* (build forecast-season storage + held-out training; train on whatever valid data exists).
2. **DB-C1** → *fix bug + add a live park-factor source* (hardcoded values remain the fallback tier).
3. **BR-1** → *fix now* (cookie/token-backed persistent sessions).
4. **Enhancements** → *include the big ones now* (unify win-prob model, copula in lineup, injury MC, etc.).

**Rules:** TDD every change; keep the ~4,900-test suite green; one atomic commit per item; **no deploy without owner OK**; single code-writing agent at a time on the shared branch (parallel only via worktree isolation).

**Status legend:** ☐ todo · ◐ in progress · ☑ done · ⏸ needs decision (none now — all resolved).

---

## Wave 0 — live crash (DONE)
- ☑ **BR-4** Playoff Odds `KeyError 'accent'` → `T["primary"]` + regression guard. (commit 5df8507)

## Wave 1 — FA engine bugs (DONE)
- ☑ **FA-C1** (HIGH) sustainability hitter sign `-23.0`→`+23.0` + corrected the mis-signed guard test + direction-vs-regression_flag assertion. (commit a268bf5; 12+5 tests, red→green confirmed)
- ☑ **FA-C4** `compute_drop_cost` NaN-guard via local `_is_hitter_safe` (avoids waiver←fa import cycle). (commit 7785989)
- ☑ **FA-C3** verified — `punt_cats` is authoritative (`weekly_h2h_strategy.py:139`); dual-key read already covered it; comment-only fix. (commit 1ea2ed2)
- ☑ **BR-8** investigated (root cause, no live fix yet): the page "Impact" (`fa−min(roster)`, can inflate via a below-replacement roster `min`) ≠ engine `net_sgp_delta`; empty output is the **position-cap guard** (`_POSITION_CAPS`, `fa_recommender.py:405-484`) + narrow funnel (`_MAX_FA_CANDIDATES=10`×`_MAX_DROP=5`) collapsing swaps (synthetic +9.18 SGP swap killed solely by `2B=6>2`), amplified by the FA-C1 sign bug suppressing buy-low targets. FIX 1 should materially improve it. **Follow-up (needs live roster):** run `scripts/diag_fa_stage_by_stage.py` (after repointing off the defunct `is_user_team` flag — it currently IndexErrors under MULTI_USER) to confirm cap-vs-threshold; if cap, widen funnel / relax cap on clear upgrades (with a guard test). → tracked as **BR-8b** below.

### New findings surfaced during Wave 1 (added to backlog)
- ☑ **NEW-1** NaN-safe `_num()` helper applied to `_marginal_era_sgp`/`_marginal_whip_sgp` (`src/valuation.py`) + `_blend_fa_row` (`fa_recommender.py`). (commit 189f17d; test_valuation_math = 41 pass)
- ☐ **BR-8b** (Med) FA recommender funnel/position-cap may over-collapse output; confirm against live roster + widen/relax with a guard test. (Wave 5/FA follow-up.)

## Wave 2 — Projections / valuation
- ☑ **PV-C2** invert Bayesian Layer-2 blend (expert ∝ reliability). (commit 9d8e487; test_ros_projections + bayesian = 56 pass)
- ☐ **PV-C1** (HIGH, full) add `forecast_season` column to `projections`; stop DELETE-all; learn stacking weights from held-out (prior-forecast→prior-actual) pairs; uniform fallback when no valid pair. `src/database.py:1089`, `projection_stacking.py`, `data_pipeline.py`.
- ☐ **PV-C3** (Low) blend volume cols (pa/ab/ip) with one shared weight, not per-component.
- ☐ **PV-C4** (Low) single standings-stddev SGP-denominator source.
- ☐ **FA-C2** (Med) wire a real L14 source into the FA blend (closes FA-C2 + the blank L14 columns). `fa_recommender._blend_fa_row` + `shared_data_layer._load_recent_form`.

## Wave 3 — Standings / matchup (DONE — 7/7; note: 2 background agents stalled on the watchdog, so MS-C5/C6/BR-2 done inline)
- ☑ **MS-C1** source total weeks from LeagueConfig (24→26). (commit e84a19e)
- ☑ **MS-C2** clamp time-decay fraction [0,1] + canonical season default. (commit 07a9e42)
- ☑ **MS-C3** TTL the team-totals module cache (300s). (commit c48ebe1)
- ☑ **MS-C4** momentum respects inverse-cat sign in season sim. (commit 535be80)
- ☑ **MS-C5** ghost-team filter on the category-rank grid (`standings_utils.filter_standings_to_valid_teams`). (commit 0a23bab)
- ☑ **MS-C6** playoff_sim docstring "top 6"→"top 4". (commit 49aa825)
- ☑ **BR-2** My Team callout was case (b): top-2 priority subset mislabeled "Losing Categories" → relabeled "Priority Targets" + guard test. (commit fa40a74). NEW finding: the subset selection sorts by raw diff (favors counting cats) — flagged BR-2b below.
- ☐ **BR-2b** (Low, enhancement) My Team priority-target selection sorts by raw diff (apples-to-oranges across counting vs rate cats) → favors big-count cats; consider normalizing by category or using win-prob. (Wave 9.)

**Checkpoint:** full suite **4920 passed, 107 skipped** in 121s after Waves 0–3. ✅

## Wave 4 — Lineup optimizer
- ☐ **LO-C1** (Med) stop injury×playing-time double-discount. `optimizer/projections.py:606,694`.
- ☐ **LO-C2** (Low) unify pitcher park-factor across daily vs weekly paths. `optimizer/matchup_adjustments.py:424,743`.
- ☐ **LO-C3** (Low) fix post-LP IP comment / dead branch. `pages/2_Line-up_Optimizer.py:1474`.
- ☐ **LO-C4** (Low) apply 0.93 two-start fatigue in streaming ranker. `optimizer/streaming.py:118`.
- ☐ **BR-7** (Low/Med) IP-budget forfeit warning scope (daily vs weekly).

## Wave 5 — Data / bootstrap + game-day
- ☐ **DB-C1** (Med, full) repair park-factor Tier 1 type bug + add a live source; optimizer reads DB table. `data_bootstrap.py:608`.
- ☐ **DB-C2** (Med) wire real closer depth data into Closer Monitor; replace SV heuristic. `pages/3_Closer_Monitor.py` + `data_bootstrap` depth phase.
- ☐ **DB-C3** (Med) short game-window staleness gate for `yahoo_matchup`. `data_bootstrap.py:3266`.
- ☐ **DB-C4** (Low) weather game_date ET + venue-local forecast hour. `game_day.py:345`.
- ☐ **DB-C5** (Low) DNA-collision warning on the trailing match_player_id fallbacks. `live_stats.py:153`.
- ☐ **DB-C6** (Low) None-check (not falsy) for team_strength rate reads. `game_day.py:691,882`.
- ☐ **BR-9** (Low) feed `percent_owned` into the FA Ownership Heat Index.
- ☐ **BR-5** (Low) Draft/Home "Yahoo Not Connected" + "Player Pool Loading" wording under MULTI_USER.

## Wave 6 — Trade engine
- ☐ **TE-C1** (Low) weekly-matrix/playoff per-week means from LP starters (exclude IL). `engine/output/weekly_matrix.py:355`.
- ☐ **TE-C2** (Low) regime fallback xwOBA 0.315→registry 0.320. `engine/signals/regime.py:341`.
- ☐ **TE-C3** (Low) remove dead `_compute_other_teams_sgp` call. `monte_carlo/trade_simulator.py:118`.

## Wave 7 — Draft engine (preseason, lower priority)
- ☐ **DE-C1** (Med) seed/CRN for candidate MC. `simulation.py:351`.
- ☐ **DE-C2** (Med) per_category_sgp reindex by player_id. `simulation.py:610`.
- ☐ **DE-C3** (Med) signed-magnitude floor in enhanced_pick_score. `draft_engine.py:1157`.
- ☐ **DE-C4** (Low) pick_predictor off-by-one. `pick_predictor.py:49`.
- ☐ **DE-C5** (Low) self-referential horizon test + docstring. `tests/test_simulation_math.py:496`.
- ☐ **DE-C6** (Low) draft_analytics wiring vs docstring.

## Wave 8 — Infrastructure (owner chose full)
- ☐ **BR-1 / cookie auth** persistent cookie/token-backed sessions so refresh/bookmarks stay logged in. `src/auth.py`, `app.py`. (Security-sensitive — careful review.)

## Wave 9 — Big accuracy enhancements (owner chose include)
- ☐ **MS-E1** unify the 3 divergent weekly-variance tables → one calibrated source (fixes BR-6 win-prob mismatch across pages).
- ☐ **LO-E3** adopt Skellam (low-count cats) + Gaussian-copula correlation in `h2h_engine` (lineup win-prob).
- ☐ **TE-E1** weekly/playoff per-week means from LP starters (overlaps TE-C1).
- ☐ **TE-E5** wire `injury_process` Weibull availability into the trade MC tails.
- ☐ **TE-E2** copula-correlated weekly outcomes in playoff sim.
- ☐ **TE-E3** schedule-aware opponent playoff sim by default.
- ☐ **TE-E4** empirical per-cat weekly CV + default Skellam for low-count cats.
- ☐ **DB-E1/E2** (= DB-C2/DB-C1 done as full features).
- ☐ **PV-E1/E2/E3** (= PV-C1/PV-C2/PV-C4 done as full features).
- ☐ **DE-E1..E5**, **FA-E1..E4**, **LO-E1/E2/E4** — per-engine enhancement backlog (see report §4).

## Final
- ☐ Full suite green (`python -m pytest --ignore=tests/test_cheat_sheet.py -q`).
- ☐ Structural-invariant + pre-push guards pass.
- ☐ Summary to owner; owner triggers deploy.

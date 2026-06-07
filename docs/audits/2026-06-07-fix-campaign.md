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

## ✅ CAMPAIGN STATUS (2026-06-07)

**COMPLETE + full suite 5044 passed / 107 skipped / 0 failed.** All on branch
`fix/audit-followups-2026-06-07` — **NOT deployed; awaiting owner deploy decision.**
~45 fixes/features, each TDD'd red→green, atomically committed, suite green at every boundary.

- **All correctness findings (Waves 0–7 + NEW-1/2):** BR-4 crash · FA-C1 sign / FA-C3 / FA-C4 ·
  PV-C2 blend · MS-C1–C6 + BR-2 · LO-C1–C4 + BR-7 · DB-C3–C6 + BR-9 + BR-5 · TE-C1–C3 · DE-C1–C6 ·
  settings dict-contract.
- **4 big features:** FA-C2 (real L14) · DB-C2 (real closer depth) · DB-C1 (park-factor tier repaired +
  optimizer reads live table) · PV-C1 (projection-stacking foundation: forecast_season + matched-year training;
  invalid cross-year regression gone).
- **Cookie auth (BR-1):** opaque server-stored revocable tokens; flag-off byte-for-byte; security-reviewed.
- **Enhancements:** MS-E1 + LO-E3 (unified Skellam/copula win-prob → fixes the BR-6 cross-page mismatch) ·
  TE-E5 (injury MC risk tails) · TE-E2 (copula playoff odds) · BR-8b (FA no-recs fixed).

**REMAINING — lower-priority enhancement tail (for a follow-up / owner greenlight):**
TE-E1 (full LP-starter per-week means; IL already excluded by TE-C1) · TE-E3 (schedule-aware playoff default,
Path A exists) · TE-E4 (empirical CV — needs prior-season data) · PV-C4 (unify SGP denominators) ·
MS-E1b (4th tau in `trade_value.py`) · BR-2b (My Team priority-target raw-diff selection) · per-engine
E-items (FA-E3/E4, LO-E1/E2/E4, DE-E3/E4/E5) · UI: Trade Analyzer pass `enable_mc=True` to surface the injury tail.

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
- ☑ **BR-8b** FA "no recommendations" root cause was the DROP funnel (only the 5 globally-cheapest, all scrub pitchers on a keeper roster), NOT the cap (cap netting already correct). `_score_drop_candidates` now also surfaces the cheapest droppable player at each FA-candidate capped position → a stud FA gets a same-position upgrade partner. 0→1 rec on a synthetic keeper roster. (commit 88ccbb0)

## Wave 2 — Projections / valuation
- ☑ **PV-C2** invert Bayesian Layer-2 blend (expert ∝ reliability). (commit 9d8e487; test_ros_projections + bayesian = 56 pass)
- ☑ **PV-C1** (full) `forecast_season` column + idempotent migration; write retains prior seasons; `_load_stacking_weights` learns from matched (forecast=Y ∧ actual=Y) years, UNIFORM fallback when none — **invalid cross-year regression confirmed GONE on the live DB** (uniform until a season of forecasts accrues, then auto-skill-weights). (commits 646a422, b5e3dc4, e75d285)
- ☑ **PV-C3** volume cols (pa/ab/ip) blend with one shared playing-time weight. (commit 47df34e)
- ☐ **PV-C4** (Low) single standings-stddev SGP-denominator source. → Wave 9 (valuation enhancement).
- ☑ **FA-C2** wire real L14 into the FA blend. `_blend_fa_row(fa_data, l14_form=)` + new `_resolve_fa_l14` (prefers `ctx.recent_form[pid]['l14']`, else lazy `get_player_recent_form_cached`); row-column L14 kept for back-compat. (done inline after 3rd background-agent stall; 28 FA tests pass incl. 4 new dict-path tests.)

## Wave 3 — Standings / matchup (DONE — 7/7; note: 2 background agents stalled on the watchdog, so MS-C5/C6/BR-2 done inline)
- ☑ **MS-C1** source total weeks from LeagueConfig (24→26). (commit e84a19e)
- ☑ **MS-C2** clamp time-decay fraction [0,1] + canonical season default. (commit 07a9e42)
- ☑ **MS-C3** TTL the team-totals module cache (300s). (commit c48ebe1)
- ☑ **MS-C4** momentum respects inverse-cat sign in season sim. (commit 535be80)
- ☑ **MS-C5** ghost-team filter on the category-rank grid (`standings_utils.filter_standings_to_valid_teams`). (commit 0a23bab)
- ☑ **MS-C6** playoff_sim docstring "top 6"→"top 4". (commit 49aa825)
- ☑ **BR-2** My Team callout was case (b): top-2 priority subset mislabeled "Losing Categories" → relabeled "Priority Targets" + guard test. (commit fa40a74). NEW finding: the subset selection sorts by raw diff (favors counting cats) — flagged BR-2b below.
- ☑ **BR-2b** My Team "Priority Targets" now ranks losing cats by normalized z-gap (|diff|/canonical σ) via testable `_rank_priority_losing_cats`, so a near-flip rate cat surfaces over a wide-gap counting cat. (commit acf318d)

**Checkpoint:** full suite **4920 passed, 107 skipped** in 121s after Waves 0–3. ✅

## Wave 4 — Lineup optimizer (DONE — 5/5; 108 area tests pass)
- ☑ **LO-C1** avoid double-discounting counting-stat availability (injury factor recorded; playing-time divides it out → single factor). (commit 8ab2494)
- ☑ **LO-C2** pitcher park factor neutral in daily path (matches weekly). (commit c7aa6e0)
- ☑ **LO-C3** fix stale post-LP IP comment + collapse dead branch. (commit c02aa49)
- ☑ **LO-C4** 2-start fatigue (0.93) in compute_streaming_value. (commit 056c84a)
- ☑ **BR-7** suppress false weekly-IP forfeit warning on Today scope. (commit 76007b7)

## Wave 5a — Data / bootstrap + game-day (contained) (DONE — 6/6; 98 tests incl. backcompat guards)
- ☑ **DB-C3** independent 5-min matchup gate in the game window (Phase 7b). (commit 3034dca)
- ☑ **DB-C4** ET game_date + venue-local forecast hour for weather. (commit 616423d)
- ☑ **DB-C5** DNA-collision warning covers trailing match_player_id fallbacks. (commit 51ff4e3)
- ☑ **DB-C6** preserve real 0.0 in team_strength reads (`_safe_float_or`). (commit 6735988)
- ☑ **BR-9** FA Ownership Heat Index wired to `percent_owned`. (commit e899cf9)
- ☑ **BR-5** Home quick-stats reflects scheduler data under MULTI_USER (flag-gated; v1 byte-for-byte). (commit 6a453bb)

## Wave 5b — Big data features
- ☑ **DB-C1** repaired dead Tier-1 (`_tier1_fangraphs` returns a real `{team:factor}` dict; the `isinstance(dict)` guard now fires); optimizer reads the live `park_factors` DB table via `load_park_factors`/`_effective_park_factors` (5 call sites). (commits e7e4f25, f361d72) NOTE: no un-authenticated park-factor source works (FanGraphs guts→403, pybaseball/MLB/ESPN have none) → prod still uses the emergency dict but HONESTLY (tier='emergency'), not via dead code; self-heals when a working feed exists.
- ☑ **DB-C2** `build_depth_data_from_db()` surfaces real `depth_chart_role` (closer/setup/committee) → grid; SV heuristic kept as flagged ESTIMATE fallback; honest caption. (commit c346a6f) — live DB: 23 closer + 324 bullpen roles / 21 teams but 0 setup/committee (Roster Resource scrape empty, known limitation); richer hierarchy auto-activates when that scrape succeeds.

**CHECKPOINT (post FA-C2 + DB-C2 + PV-C1/C3): full suite 4978 passed, 107 skipped, 0 failed. ✅**

## Wave 6 — Trade engine (DONE — 3/3; 79 tests pass incl. grade-authority + playoff guards)
- ☑ **TE-C1** exclude IL players from weekly/playoff per-week means (one shared helper covers both). (commit 2621cca) — full LP-starter weighting remains TE-E1 (Wave 9).
- ☑ **TE-C2** regime fallback reads canonical league-avg xwOBA (0.320) from registry. (commit beb38f2)
- ☑ **TE-C3** remove dead `_compute_other_teams_sgp` call. (commit 900173d)

## Wave 7 — Draft engine (preseason) (DONE — 6/6; 203 area tests pass)
- ☑ **DE-C1** seed + common-random-numbers across MC candidate eval. (commit e82abb9)
- ☑ **DE-C2** reindex per_category_sgp by sim_available player_id. (commit 4d51911)
- ☑ **DE-C3** signed-magnitude clip preserves sign in enhanced_pick_score. (commit e947ae7)
- ☑ **DE-C4** exclude current on-the-clock pick from future picks. (commit b14c205)
- ☑ **DE-C5** horizon test exercises the real *4 horizon + docstring. (commit 683544a)
- ☑ **DE-C6** draft_analytics docstring corrected (imported only by tests; engine-wiring is a follow-up). (commit 1c30063)

## NEW-2 — settings dict-contract hardening (regression fix)
- ☑ **NEW-2** `save_league_settings` rejects non-dict; `load_league_settings` coerces non-dict→{} (honor `-> dict`). Fixed a deterministic xdist-ordering failure exposed by the campaign's added test files. (commit 738b1b3)

**CHECKPOINT (post contained-waves 0–7): full suite 4958 passed, 107 skipped, 0 failed. ✅**
All clear correctness bugs fixed. Remaining = big features + enhancements (below).

## Wave 8 — Infrastructure (owner chose full) (DONE)
- ☑ **BR-1 / cookie auth** opaque server-stored revocable tokens (`auth_tokens` table) + `heater_session` cookie (st.context.cookies read / document.cookie write; NO new dep; streamlit pin 1.40→1.42). `require_auth` re-hydrates from a server-validated token (checks revoked/expired/user-active → admin revoke kills sessions); logout revokes + clears; flag-off byte-for-byte; no secret in cookie. **Security-reviewed by controller** (auth diff read). (commit 151859b) [full suite 5027 green]

## Wave 9 — Big accuracy enhancements (owner chose include)
- ☑ **MS-E1** `default_weekly_sigmas()` canonical source; standings_engine + standings_projection + playoff_sim all read it; 1σ edge → ~0.76 (was saturating 0.99). (commit 0d5769f) [full suite 4996 green]
- ☑ **MS-E1b** investigated: `trade_value.WEEKLY_TAU` IS the same quantity (per-team weekly SD, same provenance) → unified onto `default_weekly_sigmas()`. All 4 variance tables now share one source (they'd drifted 2-5×). (commit 9a35277)
- ☑ **LO-E3** `h2h_engine` routes SB/SV/W/L → Skellam, rest → Normal, overall win-prob via Gaussian copula (reuses `weekly_matrix._category_win_prob_skellam` + `copula.py`, no import cycle); sample matchup 0.187→0.260 (de-saturated). **Completes BR-6 with MS-E1.** (commit f9eeadb)
- ◑ **TE-E1** partially covered by TE-C1 (IL excluded from per-week means); full LP-starter weighting deferred (low marginal). [remaining]
- ☑ **TE-E5** injury availability wired into the paired MC risk tails (antithetic-aware; `enable_injury_mc`, engages on `enable_mc=True`); fragile player cvar5 +17.95→−17.75. (commit 1da0317)
- ☑ **TE-E2** copula-correlated playoff weekly outcomes (reuses DEFAULT_CORRELATION); playoff_prob 0.798→0.758 de-saturated. (commit 35898d9)
- ☑ **TE-E3** playoff sim defaults to schedule-aware Path A (auto-loads `load_league_schedule_full`; Path B fallback). (commit c7b2b7e)
- ☑ **TE-E4** weekly matrix defaults low-count cats (SB/SV/W/L) to Skellam (`variance_model='auto'`); empirical CV left as needs-data. (commit dda05c3)
- ☑ **UI** Trade Analyzer passes `enable_mc=True` + renders the MC risk band incl. injury-aware CVaR₅ (also fixed: the Phase-1 engine path rendered NO MC output). (commit 864c337)
- ☐ **DB-E1/E2** (= DB-C2/DB-C1 done as full features).
- ☐ **PV-E1/E2/E3** (= PV-C1/PV-C2/PV-C4 done as full features).
- ☑ **LO-E1** roster-relative SLOT_FILL_BONUS + ratio-protect posture. (commit 54d4bac)
- ☑ **LO-E2** daily mode stops double-applying L14 recent form (DCV sole applier). (commit c153d57)
- ☑ **FA-E1** = FA-C2 (L14 wiring, done).
- ◐ **FA-E3/E4** — in progress.
- ☐ **Deferred (risk vs low-conf value):** **PV-C4** (unify SGP denominators — changes SGP everywhere) · **LO-E4** (LP normalize by SGP denom not roster stdev — documented-deliberate, structural tests lock it) · **TE-E1** (full LP-starter per-week means — IL already excluded by TE-C1, low marginal).
- ☐ **Deferred (preseason, no live use mid-season):** **DE-E3** (tier-cliff signal) · **DE-E4** (per-player ADP sigma / opponent need) · **DE-E5** (less self-referential grader baseline).

**CHECKPOINT (post TE-E3/E4/UI + BR-2b + MS-E1b + LO-E1/E2): full suite 5066 passed, 0 failed. ✅**

## Final
- ☐ Full suite green (`python -m pytest --ignore=tests/test_cheat_sheet.py -q`).
- ☐ Structural-invariant + pre-push guards pass.
- ☐ Summary to owner; owner triggers deploy.

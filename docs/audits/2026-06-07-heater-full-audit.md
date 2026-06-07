# HEATER Full Audit — 2026-06-07

> **STATUS: CODE-SIDE COMPLETE + ADVERSARIALLY VERIFIED. Browser sweep (Phase A) PENDING.**
> Read-only functional + accuracy audit per `docs/superpowers/specs/2026-06-07-heater-full-audit-design.md`
> + `docs/superpowers/plans/2026-06-07-heater-full-audit.md`. **No fixes in this effort** — this is a
> diagnosis report + ranked enhancement backlog; each fix/enhancement is a separate spec→plan→build.
>
> **What's done:** Phase B (7/7 engine code-audits, L4–L6) + Phase C adversarial re-verification of all
> HIGH findings (independent skeptics) + controller verification of key Mediums + Phase D report-conclusions
> review (verdict: publish-with-edits; applied — MS-C2 downgraded to dormant/Low). **What's pending:** Phase A
> (live read-only browser sweep, L1–L3 functional/sanity/cross-feature consistency) — needs the owner logged
> into the live app; will be folded into a v2 of this report. Per-engine raw notes: `docs/audits/raw/B*.md`;
> verification verdicts: `docs/audits/raw/_verdicts.md`.

---

## 1. Executive summary

**Overall health verdict: the core math is sound; the risk is silent degradation, not wrong formulas.**

Every load-bearing mathematical invariant the codebase advertises was verified to hold in code and is
test-guarded: inverse-category sign handling (L/ERA/WHIP) is correct everywhere via `SGPCalculator`; rate
stats are volume-weighted (never simple-averaged) across standings, trade, lineup, and projection paths;
the LP only counts the 18 starters; paired-MC seed discipline + true antithetic variates hold in the trade
simulator; the trade engine's 9 documented invariants (Bug A–F + playoff/championship) all verify. **The
audit found no Critical defects and no crashes in code review.**

The dominant failure mode is a single recurring archetype this codebase is already known to be prone to
(CLAUDE.md documents past instances): **"engine built, wires never connected — the unit test passes on
synthetic input the function never receives in production, so CI stays green while the live path silently
runs a simpler/wrong thing."** Four of the highest-value findings are this exact pattern.

**Top risks (in-season, user-facing), severities post-adversarial-verification:**

| Rank | ID | Sev | Risk |
|---|---|---|---|
| 1 | **FA-C1** | **High** | Free-agent **sustainability score is sign-inverted for hitters** → it actively *rewards sell-high* and *penalizes buy-low* hitters in FA ranking. The guard test masks it by encoding the wrong sign. |
| 2 | **PV-C1** | **High** | The foundational **7-system projection blend learns year-over-year persistence, not forecast skill** (2026 forecasts regressed onto 2025 actuals; no `forecast_season` column). **Confirmed active on the live DB.** Every downstream value (SGP/VORP/trade/FA/lineup/draft) sits on a weaker base than advertised. |
| 3 | **DB-C1** | Medium | **Park-factor live tier is dead code** — the fetcher returns the wrong type, so the frozen 2026-04-18 emergency dict feeds *every* lineup/matchup park adjustment year-round. |
| 4 | **DB-C2** | Medium | **Closer Monitor primary path is dead** — `closer_depth_data` is never written, so closers are a "most-saves-per-team" guess (no committee detection). *Confirms your open task #11.* |
| 5 | **MS-C3** | Medium | **Standings/team-totals cache never invalidates** on the long-lived Railway replica → the first snapshot can persist process-wide until redeploy. |
| 6 | **DB-C3** | Medium | **Live matchup scores effectively refresh every 30 min, not the advertised 5** during games. *Confirms your open task #12.* |

**Headline enhancement opportunities** (full ranked backlog in §4):
1. **Fix the projection foundation (PV-C1/PV-E1):** add a `forecast_season` key, learn stacking weights from a true held-out (prior-forecast → prior-actual) regression. Highest leverage — it lifts every engine.
2. **Wire real data into the two dead surfaces:** live park factors (DB-E2) and real closer depth data (DB-E1).
3. **Unify duplicated/divergent machinery:** one calibrated weekly-variance table (3 disagree 2–5×, MS-E1), single-source the season length (6+ sites still use 24 vs canonical 26, MS-E2), and adopt the **Skellam + Gaussian-copula** win-probability models *already built for the trade engine* in the lineup optimizer and standings (LO-E3).
4. **Risk realism:** wire the already-built `injury_process` Weibull availability into the trade MC tails (TE-E5); compute the weekly/playoff win-prob means from LP starters, not bench+IL (TE-E1).

---

## 2. Per-feature functional table (Phase A — PENDING)

> Requires the live read-only browser sweep (per-session auth; owner must be logged in). Not yet run.
> The matrix of surfaces × controls × edge-cases is specified in the plan (§ Phase A). On completion this
> section gets one row per surface/tab/control → ✅/⚠️/❌ + finding + evidence ref, plus
> `docs/audits/_browser-evidence.md`.

| Surface | Status | Note |
|---|---|---|
| _All 14 surfaces_ | ⏳ pending | Browser sweep not yet run — code-side audit below is complete and standalone. |

---

## 3. Correctness findings (severity-ranked, post-verification)

Severities below reflect **adversarial re-verification** where performed (✓V = independent skeptic confirmed;
✓C = controller-verified by reading cited code). Confidence is 0–100. No finding here matches a CLAUDE.md
"Known Design Choice" (each subagent checked). Full detail per finding in `docs/audits/raw/B*.md`.

### HIGH
| ID | Conf | V | Engine | `file:line` | Observed → Expected |
|---|---|---|---|---|---|
| **FA-C1** | 95 | ✓V | FA recommender | `waiver_wire.py:549` | Hitter sustainability `primary_logit = -23.0 * xwoba_delta`, but pool stores `xwoba_delta = xwOBA−wOBA` (`database.py:1503`; positive = buy-low per `:1505`). Sign inverted (weighted 2.0× so it dominates). Guard test encodes the opposite sign. → `+23.0 *`. |
| **PV-C1** | 90 | ✓V | Projections | `database.py:1089-1124` | Stacking weights regress **2026** system forecasts onto **2025** actuals (no `forecast_season` column; projections table holds only current year). **Live-DB confirmed active** (1,472 ’25 rows + 9 systems → gates pass). Weights measure persistence, not skill. → train held-out on matched (forecast-yr, actual-yr) pairs. |

### MEDIUM (in-season surfaces)
| ID | Conf | V | Engine | `file:line` | Observed → Expected |
|---|---|---|---|---|---|
| **DB-C1** | 92 | ✓V | Data/game-day | `data_bootstrap.py:608-652` | `_tier1_pybaseball()` returns `team_batting()` (a DataFrame, not a park-factor dict); `isinstance(data,dict)` never True → emergency dict always served; park factors frozen at 2026-04-18. Optimizer reads the frozen module alias; `matchup_planner.py:309` reads the DB table (also frozen). → real live park-factor source + DB read. |
| **DB-C2** | 92 | ✓V | Data/game-day | `pages/3_Closer_Monitor.py:101` | `closer_depth_data` read but never written (grep: 0 writes) → always falls to "top season SV per team = closer" heuristic; real `depth_chart_role` (DB) isn't even SELECTed into the pool. → wire real depth data; keep heuristic as flagged fallback. *(Task #11.)* |
| **LO-C1** | 82 | ✓C* | Lineup | `optimizer/projections.py:606-684 + 694-764` | Counting stats discounted **twice** for availability: injury Weibull (Step 5) × playing-time PA ratio (Step 5b) both scale the same cats, both on by default. → model availability once (min/geomean or gate one off). |
| **MS-C3** | 82 | ✓C | Standings | `standings_utils.py:74-76` | `_cached_team_totals` module-global short-circuits before the TTL'd YDS call; only `clear_cache()` resets it and nothing in prod calls it → stale standings/totals on the long-lived replica until redeploy. → tie to a refresh-log timestamp / TTL / call clear on refresh. |
| **MS-C1** | 95 | — | Standings | `pages/6_League_Standings.py:74` | `_TOTAL_WEEKS = 24` (canonical is 26) → under-counts remaining matchups + truncates the synthetic-schedule fallback. → source `LeagueConfig().season_weeks`. |
| **DB-C3** | 80 | — | Data/game-day | `data_bootstrap.py:3266` | Per-team matchups written only inside the 30-min-gated `_bootstrap_yahoo`; no independent short gate → live H2H scores up to 30 min stale during games despite the 5-min TTL. → game-window `check_staleness('yahoo_matchup', ~0.08)`. *(Task #12.)* |
| **PV-C2** | 72 | ✓C | Projections | `bayesian.py:979` | ROS Layer-2 blend `reliability*marcel + (1-reliability)*proj` with reliability↑ as PA/IP↑ → veterans pulled toward crude Marcel, away from the expert projection (backwards). → expert ∝ reliability. |
| **FA-C2** | 90 | — | FA recommender | `optimizer/fa_recommender.py:1090-1166` | The 0.10 **L14 blend term is dead** — `l14_*` columns are never populated on FAs (recent form is a dict, rostered-only) → canonical 0.70/0.20/0.10 silently runs ~0.78/0.22. → load L14 for FA candidates into `recent_form` and read it. |

### MEDIUM (preseason surface — Draft tool; lower live priority while in-season)
| ID | Conf | Engine | `file:line` | Observed → Expected |
|---|---|---|---|---|
| **DE-C1** | 95 | Draft | `simulation.py:351` | MC draft sim uses a fresh **unseeded** RNG per candidate (no common-random-numbers) → top pick non-reproducible + noise-dominated on close calls (auditor reproduced the flip). → shared seed / CRN + antithetic. |
| **DE-C2** | 88 | Draft | `simulation.py:610-621` | `per_category_sgp` re-aligned in pool order but consumed in ADP order → category-aware picks read the wrong player's per-cat SGP. **Latent** (no prod caller passes that path). → reindex by player_id. |
| **DE-C3** | 85 | Draft | `draft_engine.py:1157-1180` | `enhanced_pick_score` floors negative `base_sgp` to +0.01 → below-replacement ordering collapses; additive bonuses (closer +2.0) invert order. → signed-magnitude clip. |

### LOW / INFO
| ID | Conf | Engine | `file:line` | One-liner |
|---|---|---|---|---|
| TE-C1 | 80 | Trade | `weekly_matrix.py:355` | Weekly/playoff per-week means use the full 28-man roster (bench+IL), not the 18 LP starters → ~30-50% counting inflation (≈symmetric, so partly cancels). |
| TE-C2 | 70 | Trade | `engine/signals/regime.py:341` | `classify_regime_simple` hardcodes league-avg xwOBA 0.315 (canonical 0.320); residual of a drift already fixed in `matchup.py`. |
| TE-C3 | 90 | Trade | `monte_carlo/trade_simulator.py:118` | Dead call: `_compute_other_teams_sgp` result discarded; opponent standings context never enters the MC. |
| LO-C2 | 78 | Lineup | `matchup_adjustments.py:424,743` | Pitcher park factor reciprocal in the daily DCV path but omitted in the weekly path → Daily vs Optimize tabs rank pitchers differently. |
| LO-C3 | 88 | Lineup | `pages/2_Line-up_Optimizer.py:1474` | Post-LP IP header: comment claims RP IP ×4/wk but SP and RP branches compute identically (`/26`). Display-only; code is the correct interpretation. |
| LO-C4 | 70 | Lineup | `optimizer/streaming.py:118-219` | Streaming value for a 2-start week ignores the 0.93× 2nd-start fatigue the dedicated two-start fn applies. |
| LO-C5 | 55 | Lineup | `lineup_optimizer.py:171,629` | LP objective normalizes by within-roster stat stdev, not SGP denominators → category priority is roster-relative, not standings-relative. |
| MS-C4 | 90 | Standings | `standings_engine.py:417-423` | Momentum adjustment wrong-signed for inverse cats (hot team → higher ERA mean). **Dormant** (no prod caller passes momentum). |
| MS-C5 | 70 | Standings | `pages/6_League_Standings.py:680` | Category-rank grid ranks raw `get_standings()` with no `valid_teams` ghost filter (Bug-D parity gap); low risk on the live Yahoo path. |
| MS-C6 | 99 | Standings | `playoff_sim.py:307` | Docstring says "top 6"; code is `_PLAYOFF_SPOTS=4`. Doc-only. |
| MS-C2 | 88 | Standings | `trade_intelligence.py:87,107` | `apply_time_decay` total_weeks=24 (no clamp) → early-season `time_fraction>1.0` amplifies counting SGP. **Dormant — no production caller** (definition + tests only); Phase-D downgraded Medium→Low. Still resolved by MS-E2 (single-source 26 + clamp). |
| FA-C3 | 70 | FA | `fa_recommender.py:686` | Punt path keys on `h2h_strategy['punt']` but the producer emits `'punt_cats'`; dual-key read currently saves it but is fragile. |
| FA-C4 | 60 | FA | `waiver_wire.py:388` | `compute_drop_cost` `is_hitter` cast not NaN-guarded (add side uses `_is_hitter_safe`); NaN row would abort the drop loop. |
| DB-C4 | 78 | Data/game-day | `game_day.py:345` | Weather `game_date` stamped with UTC date while schedule is ET → night games get a next-day weather row; forecast hour indexed in UTC vs local. |
| DB-C5 | 60 | Data/game-day | `live_stats.py:153-172` | DNA-collision warning doesn't cover the two trailing last-name fallback branches (team-less branch has no team filter). |
| DB-C6 | 55 | Data/game-day | `game_day.py:691,882` | `_safe_float(x) or DEFAULT` coerces a legitimate 0.0; harmless at team-aggregate scale, latent footgun if copied. |
| DE-C4 | 80 | Draft | `pick_predictor.py:49-73` | `_future_user_picks` includes the current on-the-clock pick as "future" (off-by-one). Dead in prod. |
| DE-C5 | 92 | Draft | `tests/test_simulation_math.py:496` | `test_horizon_limit` asserts ×6, code uses ×4, and never calls `simulate_draft` → horizon contract unverified, docstring wrong. |
| DE-C6 | 90 | Draft | `draft_analytics.py:1-11` | Docstring claims it's wired into `draft_engine`, but `draft_engine` imports none of it (reimplements category balance + BUY/FAIR/AVOID) → 2 divergent implementations. |

\* LO-C1: the lineup-optimizer subagent confirmed both enable flags default on and both steps run; not handed to a separate skeptic.

**Tally:** 2 High · 10 Medium (7 in-season + 3 preseason-draft) · 19 Low/Info. **All 4 original HIGH findings survived adversarial verification (0 false positives); 2 were calibrated down to Medium for honest impact. The Phase-D review confirmed every spot-checked Medium/Low against current code and downgraded the one dormant Medium (MS-C2) to Low.**

---

## 4. Per-engine accuracy assessment + ranked enhancement backlog

Format per engine: soundness verdict, then ranked enhancements (impact × 1/effort). Full design sketches
in `docs/audits/raw/B*.md`.

### B1 — Trade engine — **sound**
The deterministic Phase-1 weighted-SGP grader (the sole grade authority) + stochastic overlays are all
correct; 9 documented invariants verified + test-guarded. No Critical/High defects.
1. **TE-E1** (High/Low) — Compute weekly-matrix & playoff-sim per-week means from the 18 LP starters (exclude IL), not the full roster. Direct fix for TE-C1; the playoff/champ delta is the engine's *primary* objective yet over-counts bench+IL.
2. **TE-E5** (Med/Med) — Wire the already-built `injury_process` Weibull availability into the MC arms (`run_paired_monte_carlo` never calls it) so CVaR tails reflect injury risk for fragile/IL players.
3. **TE-E2** (Med/Med) — Replace independent per-week category outcomes with copula-correlated draws (the engine already owns `DEFAULT_CORRELATION`) → better-calibrated, less over-confident playoff odds.
4. **TE-E3** (Med/Med) — Default the schedule-aware opponent sim (Path A exists + tested) instead of the Binomial league-average approximation.
5. **TE-E4** (Med/Med) — Empirically calibrate per-category weekly CV + default-on Skellam for low-count cats (SB/SV/W/L).

### B2 — Lineup optimizer — **sound core; adjustment-chain double-counts**
All LP/DCV invariants hold. Issues are in the projection-enhancement chain + cross-path consistency.
1. **LO-E1** (Med/Low) — Calibrate `SLOT_FILL_BONUS` so it can't mask negative-value picks (its dominance is roster-dependent given LO-C5) + a "leave-slot-empty" option in ratio-protect weeks.
2. **LO-E2** (Med/Med) — Unify the projection-adjustment chain into one availability factor + one recent-form blend across the pipeline and DCV paths (fixes LO-C1 + a latent double-L14-blend in 'daily' mode).
3. **LO-E3** (Med/Med) — Adopt the Skellam (low-count cats) + Gaussian-copula correlation machinery *already built for the trade engine* in `h2h_engine` (currently Normal + independent).
4. **LO-E4** (Low/Low) — Normalize the LP objective off SGP denominators (`slope_sgp_denominators`, already computed) instead of roster stdev (fixes LO-C5).

### B3 — FA recommender — **mostly sound; one High sign bug + one dead term**
The 22-PR overhaul's invariants largely hold (marginal_sgp inverse signs, symmetric scarcity, roster floors,
IL double-protection, punt down-weight, multiplicative composite). Two methodology bugs remain.
1. **FA-E1** (High/Med) — Wire a real L14 recent-form source into the blend (closes FA-C2; restores 0.70/0.20/0.10).
2. **FA-E2** (High/Low) — Fix FA-C1 (`+23.0×`) **and** add a production-convention guard test (derive `xwoba_delta` the way the pool does, assert agreement with `regression_flag`).
3. **FA-E3** (Med/Low) — Surface the stacked playing-time discount (`_scale_ros_by_playing_time` × `_playing_time_multiplier` → ~0.06× for phantoms) in the why-expander; consider relaxing to one gate for confirmed-everyday roles.
4. **FA-E4** (Low/Low) — Move the last inline FA tunables (ECR-stddev + regression nudges) into `CONSTANTS_REGISTRY`.

### B4 — Matchup & standings — **math sound; constants drift + a stale cache**
Rate-stat aggregation correctly volume-weighted everywhere; inverse cats correct; punt AND-logic correct;
single weight source; playoff spots=4/weeks=26.
1. **MS-E1** (High/Med) — Unify the **three divergent weekly-variance tables** (`CALIBRATED_WEEKLY_TAU` vs `WEEKLY_TAU` vs `_DEFAULT_WEEKLY_SIGMAS`, disagree 2–5×) into one empirically-calibrated source so win-probs agree across Matchup Planner / Season Proj / Playoff Odds.
2. **MS-E2** (Med/Low) — Single-source the season length (fixes MS-C1 + MS-C2 + the 6 `=24` defaults); clamp `time_fraction` to [0,1]; add a structural guard.
3. **MS-E3** (Med/Low) — Freshness-aware invalidation of the `standings_utils` derived caches (fixes MS-C3) + wire the standings-page ghost filter (fixes MS-C5).

### B5 — Projections / valuation — **SGP/Marcel sound; the stacking foundation is broken**
SGP signs, rate handling, Marcel, VORP all correct. The ridge-stacking foundation is the problem.
1. **PV-E1** (High/Med) — Add `forecast_season` key + learn weights from a true held-out (prior-forecast → prior-actual) regression (fixes PV-C1). **Highest-leverage fix in the whole audit** — lifts every downstream engine.
2. **PV-E2** (Med/Low) — Re-orient the Bayesian Layer-2 blend so experts dominate for well-sampled players (fixes PV-C2) + a direction test.
3. **PV-E3** (Med/Med) — Replace the 3 ad-hoc SGP-denominator estimators with one standings-stddev computation (Schechter), Bayesian-shrunk early-season; wire as the live denominator source (fixes PV-C4 + the unimplemented "Bayesian-updated denominators" claim).

### B6 — Draft engine — **snake/SGP correct; MC lacks seed discipline** (preseason)
Snake-order math verified empirically correct; routes through canonical SGP/VORP. Issues are MC reproducibility
+ a few dead/latent paths. Lower live priority (in-season now).
1. **DE-E1** (High/Low) — Common-random-numbers + optional seed for candidate MC (fixes DE-C1; matches the trade engine's discipline).
2. **DE-E2** (Med/Low) — Signed-magnitude floor in `enhanced_pick_score` (fixes DE-C3).
3. **DE-E3** (High/Med) — Tier-cliff / VOND value-drop signal in recommendations (modern dominant draft strategy; reuse `assign_tiers` + `compute_opportunity_cost`).
4. **DE-E4** (Med/Med) — Thread the already-built per-player ADP sigma + opponent need modeling into `recommend()`.
5. **DE-E5** (Med/Med) — Less self-referential draft-grader baseline (replacement/positional, not the pool's own sorted SGP).

### B7 — Data / bootstrap + game-day — **silent-failure coverage sound; two dead data paths**
Refresh-log error coverage, freshness honoring, connection-status honesty, roster partial-fetch guard,
TWP routing, IP parsing all hold. Two "dead wiring" data-quality bugs (DB-C1, DB-C2) + TTL/timezone issues.
1. **DB-E1** (High/Low) — Wire real closer depth data (Roster Resource CL + statsapi role) into the Closer Monitor; replace the SV heuristic (fixes DB-C2; the job-security/committee logic is already built, just unfed).
2. **DB-E2** (High/Med) — Replace the dead park-factor Tier 1 with a working live source + a real-output smoke test (fixes DB-C1); have the optimizer read the DB table, not the frozen alias.
3. **DB-E3** (Med/Low) — Give live-game matchup its own short staleness gate so the 5-min tick actually refreshes it (fixes DB-C3).
4. **DB-E4** (Low/Low) — Normalize weather/game-date to ET target date + venue-local forecast hour (fixes DB-C4).

---

## 5. Cross-feature consistency findings

Browser-driven cross-feature checks (same player/standings/matchup numbers across surfaces) are **pending the
Phase A sweep**. Code-level cross-engine inconsistencies already surfaced:

- **Three divergent weekly-variance tables** (MS-E1) → the *same* matchup yields different win-probabilities on
  Matchup Planner vs Season Projections vs Playoff Odds. *(User-visible "different number on every page" — the
  browser sweep should confirm the symptom.)*
- **Season-length constant drift** (MS-C1, live) → the Standings page uses 24 weeks while lineup/playoff use the
  canonical 26. (The related `apply_time_decay` 24-default, MS-C2, is dormant/no-caller — latent until wired.)
- **Pitcher park factor** applied in the Daily DCV path but not the weekly Optimize path (LO-C2) → the same
  pitcher can rank differently across the two lineup tabs.
- **Skellam + copula win-prob machinery exists in the trade engine but not in the lineup optimizer or standings**
  (LO-E3) → low-count categories (SB/SV) are modeled more accurately in one surface than another.
- **Park factors + closer data are frozen/heuristic** (DB-C1/DB-C2) → any surface that shows park-adjusted or
  closer-derived values is built on stale/guessed inputs; the browser sweep should sanity-check how visible this is.

---

## 6. Evidence appendix

- **Per-engine raw subagent notes:** `docs/audits/raw/B1-trade-engine.md` … `B7-data-bootstrap-gameday.md`.
- **Adversarial verification verdicts:** `docs/audits/raw/_verdicts.md` (4 HIGH findings, independent skeptics).
- **Browser evidence:** `docs/audits/_browser-evidence.md` — *pending Phase A.*
- **Method:** 7 parallel read-only engine code-audits (L4–L6) + independent skeptic re-verification of every HIGH
  finding (instructed to refute; default not-a-finding) + controller verification of key Mediums against cited
  code. Severities here are post-verification. Findings matching CLAUDE.md "Known Design Choices" were excluded
  by construction.
- **Confidence note:** "Accuracy" for projections means *methodology-sound + internally consistent + matching
  documented intent + free of logic bugs* — there is no future-outcome oracle for 2026 projections (spec §1).
```

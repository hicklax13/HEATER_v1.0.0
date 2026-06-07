# Design — HEATER Full Functional + Accuracy Audit

> Status: **approved design, pre-plan**
> Date: 2026-06-07
> Goal: produce a single authoritative audit of whether every HEATER feature **works**, whether
> its outputs are **correct/reliable**, and where each engine can be **enhanced** for more accurate
> results. Deliverable is a **diagnosis report + ranked enhancement backlog** — fixes and
> enhancements are separate follow-up plans (each its own spec → plan → build).

## 1. Why this exists (honesty framing)

A prior "review" only confirmed pages *render* with *plausible-looking* live data. That is a
load-and-look check, NOT an accuracy audit. "The numbers look real" ≠ "the engine is correct and
near-optimal." This audit closes that gap rigorously.

**Hard truths the methodology must respect:**
- **No ground-truth oracle.** We cannot verify a 2026 projection against the future. "Accurate"
  therefore means: *methodology-sound, internally consistent, sane-ranged, matching documented
  design intent, and free of logic bugs* — plus *measured against fantasy/statistical best
  practice* for the enhancement question.
- **Accuracy lives in the code, not the screen.** The browser proves "works + output is sane";
  the *math* proves "correct/optimal." Both are required.
- **Findings must be adversarially verified** before they enter the report, or the audit becomes
  noise. The bar that caught the real silent-failure earlier applies here.

## 2. Goals / non-goals

**Goals**
- Exercise every page → every tab → every button/input (default + meaningful edge cases) in the
  browser; record outputs; flag functional, sanity, and cross-feature-consistency issues.
- Code-audit every engine for methodology correctness + enhancement opportunities.
- Produce one report: per-feature functional status, severity-ranked correctness findings (with
  code refs + evidence), per-engine accuracy assessment, and a ranked enhancement backlog.

**Non-goals**
- Fixing bugs or implementing enhancements in this effort (separate follow-ups).
- Exhaustive input-combination sweeps (infinite). Representative + edge inputs only.
- Mutating the live league or DB in any way.

## 3. Verification layers (the definition of "correct")

Applied to every feature/engine. A finding cites the layer(s) it violates.

| Layer | What it checks | How |
|------|----------------|-----|
| **L1 Functional** | Button/input runs; no crash/error; produces output within a reasonable time | Browser |
| **L2 Sanity** | Plausible ranges; correct signs (inverse stats L/ERA/WHIP lower=better); no stuck "Loading…"/NaN/None; formatting (AVG/OBP `.3f`, ERA/WHIP `.2f`, SGP `+.2f`) | Browser + code |
| **L3 Consistency** | Same entity agrees across features: a player's value across Trade Analyzer ↔ Trade Value ↔ Free Agents ↔ Player Compare; standings identical across pages; matchup numbers match across My Team / Matchup Planner / Standings | Browser + code |
| **L4 Methodology** | Engine math matches documented design (CLAUDE.md invariants, specs) + sound principles: no sign errors, rate stats weighted not averaged, no stale reads, no double-counting/data-leakage, correct denominators | Code |
| **L5 Known-case** | Invariants encode reality: star ranks high; injured/IL de-weighted; punt cat down-weighted; SGP via `SGPCalculator`; positional scarcity; 26-week scaling; antithetic variance reduction; IP-floor forfeit logic; paired-MC seed discipline | Code + targeted observation; cross-check vs the ~4,893-test suite |
| **L6 Best-practice gap** | Current method vs fantasy/statistical state-of-the-art + `docs/Research.md` competitive gaps → ranked enhancement ideas with rationale + expected gain | Code + domain analysis |

## 4. Two work-streams

### 4.1 Browser sweep (controller / this session)
Only this session can drive the owner's logged-in browser (per-session auth; subagents would fight
over tabs). The controller serially visits every surface, exercises every tab + button + input
(default + edge cases enumerated in the plan), and captures outputs (screenshots + page text).
Produces raw L1–L3 findings + an evidence trail. **Read-only** (see §6).

### 4.2 Engine code-audits (parallel subagents)
One subagent per engine, dispatched in parallel. Each reads the engine's modules + tests, audits
L4–L6, and returns structured findings (correctness issues + ranked enhancements) grounded in the
code, `CLAUDE.md`, `docs/Research.md`, and the test suite. Engines:

1. **Trade engine** — `src/engine/` 6-phase pipeline + secondary diagnostics (G-score, VORP/PRP,
   DRL replacement chain, specialist cap, CARA λ-sweep, Kalman variances, Skellam win-prob, weekly
   H2H matrix, playoff/championship sim, IP-floor penalty). Grade authority = Phase-1 surplus.
2. **Lineup optimizer** — `src/optimizer/` 21 modules: enhanced projections, matchup adjustments
   (park/platoon/weather/umpire/framing/PvB), SGP theory, streaming, scenario generator
   (copula/CVaR), dual objective, advanced LP (PuLP), category urgency (sigmoid), daily DCV.
3. **FA recommender** — `src/optimizer/fa_recommender.py` + `src/waiver_wire.py` (PRs #89–#110):
   blend (0.70 ROS/0.20 YTD/0.10 L14), playing-time gate, positional scarcity, roster-construction
   guards, punt-awareness, IL weighting, sustainability, regression nudge.
4. **Matchup & standings** — `src/matchup_context.py`, `matchup_planner.py`, `standings_engine.py`,
   `standings_utils.py`, `weekly_h2h_strategy.py`, playoff/season projection sims.
5. **Projections / valuation** — `src/valuation.py` (LeagueConfig, SGPCalculator, VORP),
   `bayesian.py`, `marcel.py`, `projection_stacking.py`, `ml_ensemble.py`, `data_pipeline.py`
   (7-system ridge-stacked blend).
6. **Draft engine** — `src/draft_engine.py`, `draft_analytics.py`, `pick_predictor.py`,
   `simulation.py`, `draft_grader.py` (8-stage chain, MC recommendations).
7. **Data / bootstrap + game-day** — `src/data_bootstrap.py` (33-phase, 3-tier waterfall),
   `data_pipeline.py`, `live_stats.py`, `game_day.py` (weather/park/opposing-pitcher),
   `closer_monitor.py`, freshness/TTLs, Yahoo sync. (Where data-quality issues like the FanGraphs
   403 + closer assignments live.)

Cross-cutting (assigned to the nearest engine subagent): player intelligence (injury/news/prospect),
park/platoon/weather adjustments, ECR/ADP.

## 5. Execution architecture

```
[Browser sweep — controller]            [Engine code-audits — N parallel subagents]
 per surface: exercise + capture          per engine: L4–L6 findings + ranked enhancements
            \                                   /
             → SYNTHESIS (tie UI symptom ↔ code cause; dedup; map finding→engine)
             → ADVERSARIAL RE-VERIFICATION (each finding independently checked: real? severity?)
             → CODE-REVIEW pass on the audit's conclusions (requesting/receiving-code-review)
             → FINAL REPORT
```

- **Synthesis** correlates browser symptoms with code causes (e.g., "Closer Monitor shows odd
  assignments" ↔ "FanGraphs 403 → fallback heuristic in depth_charts").
- **Adversarial re-verification:** every candidate finding gets an independent skeptic pass
  (default to "not a finding" unless proven), killing false positives. Confirmed-design-choices
  (CLAUDE.md "Known Design Choices") are excluded by definition.
- **Code-review** the audit's conclusions before publishing, so the report itself is vetted.

## 6. Safety (non-negotiable)

- **LIVE app, READ-ONLY.** Exercise analysis controls only: Optimize, Analyze Trade, Search,
  filters, category/scope selectors, player pickers, tabs.
- **Never click write/mutation controls:** "Sync Yahoo", "Refresh Stats", "Refresh Data",
  "Refresh All Data", any add/drop/trade-execute, and **all admin write toggles** (page
  visibility, broadcast, maintenance, view-as, revoke, reassign, Yahoo-token paste). Inspect, do
  not actuate.
- Never submit forms that mutate state. Never trigger a deploy. Never touch the real league or DB.
- The browser sweep must not start a heavy mutation; the scheduler remains the sole writer.

## 7. Deliverable — the audit report

Path: `docs/audits/2026-06-07-heater-full-audit.md`. Sections:
1. **Executive summary** — overall health verdict, top risks, headline enhancement opportunities.
2. **Per-feature functional table** — each surface/tab/button: ✅ / ⚠️ / ❌ + one-line finding +
   evidence ref.
3. **Correctness findings** — severity-ranked (Critical / High / Medium / Low), each with: layer,
   feature(s), code ref (`file:line`), observed-vs-expected, and a confidence score.
4. **Per-engine accuracy assessment** — for each of the 7 engines: methodology summary, what's
   sound, what's questionable, and a **ranked enhancement backlog** (impact × effort, rationale,
   expected accuracy/reliability gain, references).
5. **Cross-feature consistency findings** — disagreements between surfaces for the same entity.
6. **Evidence appendix** — captured screenshots/page-text per surface; per-engine subagent notes.

Every finding is traceable to evidence; every enhancement is justified, not asserted.

## 8. Finding taxonomy + confidence

- **Severity:** Critical (wrong/misleading output a user would act on; crash) > High (materially
  inaccurate or inconsistent) > Medium (sane but suboptimal / minor inconsistency) > Low (cosmetic).
- **Confidence:** 0–100; findings < ~60 after adversarial verification are dropped or labeled
  "needs-data".
- **Enhancement rank:** impact (accuracy/reliability gain) × inverse effort; each with a one-paragraph
  design sketch (becomes a future spec seed).

## 9. Assumptions / constraints

- Auth is per-Streamlit-session; the owner is logged in in the controller's tab. The controller
  drives that tab only.
- Some engine outputs are slow on the live single-replica (e.g., Trade Finder, optimizer Full
  mode); the sweep allows generous waits and records latency as a finding where it harms UX.
- The ~4,893-test suite already encodes many invariants; the audit treats green tests as evidence
  of those invariants but does NOT assume tests cover accuracy (they encode design, not optimality).
- Findings that match `CLAUDE.md` "Known Design Choices & External Limitations" are NOT re-flagged
  (e.g., FanGraphs 403, bat_speed skipped) — but their *downstream effect on output accuracy* (e.g.,
  closer assignments) IS in scope as a data-quality finding + possible enhancement.

## 10. Out of scope (this effort)
- Implementing any fix or enhancement.
- Load/performance testing beyond noting egregious latency.
- Security/pentest (separate concern).
- The shallow-review findings already logged as tasks #10–#13 are folded in and re-verified by
  this audit, not treated as new.

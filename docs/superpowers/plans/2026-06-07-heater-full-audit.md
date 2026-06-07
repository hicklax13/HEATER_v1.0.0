# HEATER Full Audit — Implementation Plan

> **For agentic workers:** Execute with superpowers:subagent-driven-development. The browser sweep
> (Phase A) MUST be run by the controlling session (only it can drive the owner's logged-in
> browser); Phase B engine audits run as parallel subagents. Spec:
> `docs/superpowers/specs/2026-06-07-heater-full-audit-design.md` (read it first).

**Goal:** One audit report — does every feature work, are outputs correct/reliable, and how can each
engine be enhanced — via a read-only browser sweep + parallel per-engine code audits → synthesis →
adversarial re-verification → code review.

**Deliverable:** `docs/audits/2026-06-07-heater-full-audit.md` (create `docs/audits/`).

**Hard constraints:** READ-ONLY on the live app (never click Sync Yahoo / Refresh* / add-drop /
trade-execute / admin write toggles). No fixes in this effort. Every finding adversarially
re-verified before it lands. Exclude CLAUDE.md "Known Design Choices" from findings (but their
downstream output effects ARE in scope).

---

## Phase 0 — Setup
- [ ] Confirm logged into the live app in the controller's browser tab (per-session auth).
- [ ] `mkdir docs/audits`. Create the report skeleton (§ Report skeleton below).
- [ ] Create a findings ledger `docs/audits/_findings.jsonl` (one JSON object per candidate finding:
      `{id, layer, severity, confidence, surface|engine, observed, expected, evidence_ref, status}`).

## Phase A — Browser sweep (controller, serial, READ-ONLY)

For EACH surface below: open it, open every tab, exercise every button/input with the **default**
and the listed **edge cases**; capture a screenshot + `get_page_text`; record L1–L3 candidate
findings into the ledger with evidence refs. Allow generous waits (heavy engines). If a screenshot
freezes, use `get_page_text`.

**Surface × checks matrix** (edge cases are the minimum; add any obvious ones discovered):

| Surface (tabs) | Buttons/inputs to exercise | Edge cases | L2/L3 checks |
|---|---|---|---|
| **Home / Draft Tool** | Connect-Yahoo expander (inspect only), risk slider, engine mode, NEXT | — | "Yahoo Not Connected" vs live data wording |
| **My Team** | (read) roster, alerts, streaks, category standings; scroll all | — | roster count vs Yahoo; streak math sane; matchup matches Matchup Planner |
| **Lineup Optimizer** (Optimizer / Start-Sit / Category / Streaming) | scope=Today/Week/Season; mode=Quick/Std/Full; risk slider; **Optimize Lineup** | IL player in roster; 2-start pitcher; off-day hitter; pure-SP not probable | lineup respects slots; IP budget math; forfeit flag; DCV "—" for locked; no NaN |
| **Closer Monitor** | (read) 30-team grid | — | closer assignments plausibility (vs FanGraphs-403 fallback); SV/ERA/WHIP sane |
| **Matchup Planner** (Cat Prob / Player Matchups / Per-Game / Hitters / Pitchers) | week ±; days-ahead; player-type; team selector | opponent with missing data; punt cat | win% sums sane; inverse cats flipped; per-cat numbers match Standings |
| **League Standings** (Current / Season Proj / Playoff Odds) | run projections | — | standings identical to My Team/Matchup; playoff odds ≤ 1; ranks 1–12 (no ghost 13th) |
| **Punt Analyzer** | choose 1, 2, 3 punt cats | punt an inverse cat; punt a rate cat | values recompute; punted cat down-weighted |
| **Trade Analyzer** | pick give/receive players; **Analyze Trade** | lopsided (star for scrub); 2-for-1; equal trade; IL player; cross-position | grade ↔ surplus sign consistent (Bug A guard); reshuffle warning; playoff/IP deltas render |
| **Trade Finder** (tabs) | run recs; target a team; browse partners | — | latency (record); recs are real FAs/rosters; value chart sane |
| **Free Agents** | level filter; position filters; pagination | filter to C; an IL stash name; punt-cat context | streams are real FAs; no IL-ace-drop rec; ECR integer; "why" reconciliation |
| **Player Compare** | pick A vs B (star vs scrub; hitter vs pitcher; same player guard) | identical player; pitcher vs hitter | category fit signs; composite scores ordered sanely |
| **Leaders** (Leaders / Value / Breakouts / Prospects / Hot / Cold / Sell-High) | each tab; each category; position/org/ETA filters | inverse cat (ERA) leader order; prospects ETA filter | rostered-by matches Yahoo; inverse-cat ascending; counts plausible |
| **Player Databank** | search a name; position/status/team/stat/sort filters; **Search** | a TWP (Ohtani); a known dup-name (Muncy/Will Smith); minor leaguer | correct player (no DNA collision); stats present; rate formatting |
| **Draft Simulator** | start a sim; make a pick; view MC recs | — | recs sane; AI opponents pick plausibly; no crash |
| **Admin Console** (Users / Feedback) | (inspect only — DO NOT revoke/reassign) | — | users listed; feedback inbox renders |
| **Admin Analytics / Controls** | (inspect only — DO NOT toggle) | — | analytics render; flags shown |

- [ ] After the sweep, write `docs/audits/_browser-evidence.md` summarizing per-surface
      status + screenshots/page-text refs + the ledger entries created.

## Phase B — Engine code-audits (parallel subagents)

Dispatch ONE subagent per engine (general-purpose). Each prompt includes: the engine's modules, its
CLAUDE.md invariants, `docs/Research.md` gaps, and this instruction:

> "Audit <engine> for L4–L6 (spec §3). Read the modules + their tests. Report: (1) correctness
> findings — sign errors, simple-averaged rate stats, stale reads, double-counting/leakage, wrong
> denominators, mis-wired data — each with `file:line`, observed-vs-expected, severity, confidence;
> (2) a RANKED enhancement backlog — concrete, justified improvements toward fantasy/statistical
> best practice + Research.md gaps, each with impact×effort + expected gain + a one-paragraph design
> sketch. Exclude CLAUDE.md 'Known Design Choices'. Return strict JSON matching the finding schema."

- [ ] Task B1 — Trade engine (`src/engine/**`)
- [ ] Task B2 — Lineup optimizer (`src/optimizer/**` incl. advanced_lp, scenario_generator, daily_optimizer)
- [ ] Task B3 — FA recommender (`src/optimizer/fa_recommender.py`, `src/waiver_wire.py`)
- [ ] Task B4 — Matchup & standings (`src/matchup_context.py`, `matchup_planner.py`, `standings_engine.py`, `weekly_h2h_strategy.py`)
- [ ] Task B5 — Projections/valuation (`src/valuation.py`, `bayesian.py`, `marcel.py`, `projection_stacking.py`, `ml_ensemble.py`, `data_pipeline.py`)
- [ ] Task B6 — Draft engine (`src/draft_engine.py`, `draft_analytics.py`, `pick_predictor.py`, `simulation.py`, `draft_grader.py`)
- [ ] Task B7 — Data/bootstrap + game-day (`src/data_bootstrap.py`, `data_pipeline.py`, `live_stats.py`, `game_day.py`, `closer_monitor.py`, freshness/TTLs)
- [ ] Append all returned findings to the ledger.

## Phase C — Synthesis + adversarial re-verification
- [ ] Synthesis: correlate Phase-A UI symptoms with Phase-B code causes; dedup; map each finding to
      its engine + the violated layer(s).
- [ ] For EACH candidate finding (esp. Critical/High), dispatch an independent skeptic subagent:
      "Try to REFUTE this finding; default to NOT-a-finding unless proven; confirm severity." Drop
      anything that doesn't survive (confidence < ~60) or matches a Known Design Choice.

## Phase D — Code-review + report
- [ ] Run `pr-review-toolkit:code-reviewer` (or `feature-dev:code-reviewer`) over the *conclusions*
      (the report draft) to vet reasoning + catch overreach.
- [ ] Assemble `docs/audits/2026-06-07-heater-full-audit.md` (skeleton below). Commit on this branch.
- [ ] Present the executive summary to the owner; the ranked enhancement backlog seeds future
      spec→plan→build cycles.

## Report skeleton
```
# HEATER Full Audit — 2026-06-07
## Executive summary  (health verdict; top risks; headline enhancements)
## Per-feature functional table  (surface/tab/button → ✅/⚠️/❌ + finding + evidence)
## Correctness findings  (Critical→Low; layer; file:line; observed vs expected; confidence)
## Per-engine accuracy assessment + ranked enhancement backlog  (impact×effort; rationale; expected gain)
## Cross-feature consistency findings
## Evidence appendix
```

## Self-review (writing-plans)
- Spec coverage: L1–L6 (Phase A checks + Phase B briefs); browser/code split (A/B); synthesis+verify
  (C); code-review (D); deliverable (report skeleton); safety (Phase A header + matrix "inspect
  only"). All spec sections mapped.
- No placeholders: each surface has concrete checks/edge-cases; each engine task has exact modules.
- Consistency: finding schema used in Phase 0 ledger, Phase B prompt, Phase C, report. Engine list
  (B1–B7) matches spec §4.2.

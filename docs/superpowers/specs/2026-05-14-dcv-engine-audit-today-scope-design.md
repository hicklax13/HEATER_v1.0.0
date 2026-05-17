# DCV Engine Audit — "Today" Scope

**Date:** 2026-05-14
**Author:** Connor Hickey + Claude (Opus 4.7 1M)
**Status:** Spec — pending audit execution
**League:** FourzynBurn (Yahoo H2H Categories, 12-team, 28-slot)

## Purpose

Confirm that the lineup optimizer's **"Today" scope** is mathematically and logically correct against the FourzynBurn scoring settings. Produce a categorized findings catalog so subsequent fix waves (Wave 11+) can ship targeted patches with confidence.

The framing is "100% correct and as indisputable as mathematically and logically possible." Every HIGH-severity finding must carry one of: a math derivation, a direct citation in `CLAUDE.md` or `constants_registry`, or a FourzynBurn league rule the code contradicts. No "I think this is wrong."

## Audit Boundary

### In scope

| Code path | What's audited |
|---|---|
| `pages/2_Line-up_Optimizer.py` lines 939-1011 | The `_scope_key == "today"` branch — input wiring + display contract |
| `src/optimizer/daily_optimizer.py` (1302 LOC) | `build_daily_dcv_table` and every helper: `compute_health_factor`, `compute_volume_factor`, `compute_matchup_multiplier`, weighted DCV math, stud floor, SP gate, forced-start, locked-teams, `DailyDCVContext` |
| `src/optimizer/shared_data_layer.py` | `build_optimizer_context` — produces urgency_weights, confirmed_lineups, recent_form, team_strength |
| `src/matchup_context.py` | `MatchupContextService` (3 modes: matchup / standings / blended) — sole source of category weights |
| `src/optimizer/category_urgency.py` | Sigmoid urgency math + runtime reads from `CONSTANTS_REGISTRY` |
| `src/optimizer/constants_registry.py` | Every constant cited by the above (sigmoid k, IP target, league averages, platoon, fatigue, thresholds) |
| `src/game_day.py` | `get_target_game_date` — auto-advance today→tomorrow when all games final |

### Out of scope

- **Projection accuracy** — the projection blending was audited in SF-1..SF-83; we trust the blended values flowing into DCV
- **Park factor values** — audited in SF-22 / SF-30; we audit the *use* of park factors, not the values
- **statsapi schedule reliability** — covered by prior data-source audits
- **LP solver** (`src/lineup_optimizer.py`) — that's the Rest-of-Week / Rest-of-Season scope, not "Today"
- **Other optimizer tabs** — Streaming, Manual, Roster, Start/Sit; all separate code paths
- **Yahoo data freshness** — `yahoo_data_service` 3-tier cache already audited

## Methodology

Six parallel specialist audit agents, each owning a disjoint slice of the engine. Each agent:

- Receives a self-contained briefing (no shared session state required)
- Reads its slice of the code, plus `LeagueConfig`, `constants_registry`, and relevant `CLAUDE.md` sections
- Produces a markdown findings list per the format below
- Returns the report; no code is modified during audit

### Agent decomposition

| Agent | Slice | Findings target |
|---|---|---|
| **A1: Algorithms** | All formulas in `daily_optimizer.py` | Math errors, sign errors, missing factors, mis-weighted aggregations |
| **A2: Constants** | Every magic number → `constants_registry` citation | Wrong values, missing citations, drift from research baseline |
| **A3: League-config** | FourzynBurn alignment (cats, inverse, rate, roster, txn, undroppable) | Hardcoded assumptions wrong for FourzynBurn |
| **A4: Inputs/data** | Schemas of `urgency_weights`, `confirmed_lineups`, `recent_form`, `team_strength`, `schedule_today`, `matchup` | Shape mismatches, type drift, silent fallbacks |
| **A5: MatchupContext + sigmoid** | `MatchupContextService` (3 modes), `compute_urgency_weights`, sigmoid math, rate_modes (abandon/contest) | Wrong mode-switching, sigmoid k drift, rate-mode logic errors |
| **A6: Edge cases** | DTD/IL/SUSP exclusion, `locked_teams`, pure-P fallback, missing-projection fallback, zero-IP rate stats, sprint-speed boost, two-start fatigue | Silent dead branches, wrong fallback values, unreachable code |

### Aggregation flow

```
6 agents → 6 markdown reports (parallel execution)
         ↓
parse all → dedupe by symptom (multiple agents flagging same root cause = 1 finding)
         ↓
categorize by severity (HIGH / MED / LOW)
         ↓
group by code area → fix-wave plan (HIGH wave 1, MED wave 2, LOW backlog)
         ↓
final report: docs/superpowers/specs/2026-05-14-dcv-engine-audit-today-scope-findings.md
```

## Findings Format

Each finding follows the SF-1..SF-83 format:

```
ID:       DCV-A{agent}-{number}        Severity: HIGH | MED | LOW
File:     src/optimizer/daily_optimizer.py:642
Symptom:  One-sentence description of what's wrong
Expected: What the correct behavior should be (with citation)
Actual:   What the code does today
Proof:    Math derivation OR CLAUDE.md/research citation OR FourzynBurn rule
Repro:    Code/data scenario where the bug fires
Fix:      One-line suggestion (not implementation)
```

## Severity Rubric

| Tier | Definition | Examples |
|---|---|---|
| **HIGH** | Wrong math/data that changes lineup decisions for the user. Mis-ranks players by >5% DCV, or causes a wrong start/bench. | Wrong sign on inverse stat, wrong rate-stat aggregation (simple mean instead of weighted), sigmoid k drifted from calibrated value, missing IL status in exclusion list |
| **MED** | Bounded impact: rare scenario, edge-of-roster decision, or only fires when fallback path triggers. Won't change tonight's lineup but will eventually. | Wrong fallback when primary signal present, magic threshold without citation, silent default that masks data drift, inconsistent precision (`.3f` vs `.2f`) |
| **LOW** | Cosmetic, code-quality, missing guards. No behavior bug today; future regression risk. | Undocumented constants, type drift, dead-code branches, missing structural-invariant test, comment rot |

## Indisputability Statement

For every HIGH-severity finding, the report must include one of:

1. **Provable math derivation** — e.g., "Weighted AVG = Σh/Σab; the code uses Σ(h/ab)/N, which differs by Jensen's inequality. Real-roster diff: up to 8% on the user's roster (compute shown)."
2. **Direct citation** — e.g., "`CLAUDE.md` line 178 documents `sigmoid_k_counting=2.0`; this code reads `2.5` from a stale alias bypassing `CONSTANTS_REGISTRY` (SF-39 regression)."
3. **FourzynBurn rule contradiction** — e.g., "League settings (Yahoo `game_key=469`) include `L` (losses) as an inverse pitching cat; this code's `_INVERSE_CATS = {'ERA','WHIP'}` drops `L`, causing positive-good treatment of losses (SF-?? type bug)."

MED findings get a citation or repro but not all three. LOW findings get one-sentence justification.

**Demotion rule:** If an agent flags something as HIGH but cannot supply any of the three proof artifacts above, the aggregation step demotes it to MED with note "demoted: HIGH claim without indisputable proof." This prevents the audit from ranking suspicions above evidence.

## Deliverable

Single markdown spec at `docs/superpowers/specs/2026-05-14-dcv-engine-audit-today-scope-findings.md` with:

1. **Executive summary** — finding counts by severity, headline risks (1-2 sentences each), confidence statement
2. **Methodology** — which 6 agents ran, what they checked, what they didn't (links to this spec)
3. **Findings catalog** — every `DCV-AX-NNN` entry, sorted by severity then by code area
4. **Fix-wave proposal** — HIGH wave (must-fix), MED wave (should-fix), LOW backlog; each wave estimates LOC delta + test count
5. **Open questions** — anything ambiguous in FourzynBurn settings or league rules where the audit cannot conclude without user input
6. **Indisputability statements** — for each HIGH finding, the proof artifact

## Execution Plan

After spec approval:

1. **Brief 6 agents in parallel** (single message, 6 `Agent` tool calls). Each agent gets:
   - Audit boundary excerpt (in/out)
   - Slice-specific code paths to read
   - The findings format + severity rubric (so reports are comparable)
   - The indisputability requirement
   - Self-contained briefing (no reference to "this conversation")
2. **Receive 6 reports** as agents complete
3. **Aggregate**: parse, dedupe, categorize, group
4. **Write findings spec** to `2026-05-14-dcv-engine-audit-today-scope-findings.md`
5. **Present to user** for triage
6. **Hand off to writing-plans** for Wave 11 (HIGH fixes)

## Success Criteria

- ✅ All 6 agent reports returned with non-empty findings sections
- ✅ Every HIGH finding carries one of the three indisputability proofs
- ✅ Findings deduped (no symptom listed twice)
- ✅ Fix-wave proposal includes LOC delta + test count estimates per wave
- ✅ Open questions surfaced explicitly so the user can resolve before fixes ship
- ✅ Final confidence statement included: an explicit percentage estimate of "correctness against FourzynBurn settings," plus a numeric count of open questions and deferred LOW findings. The estimate is bounded by which audit slices returned high-coverage reports; agents that surface unknown-unknowns lower it.

## Out-of-Spec Notes

- This audit does NOT fix anything. Every fix is deferred to a separate wave.
- This audit does NOT touch the live DB.
- This audit does NOT add or modify production code; it only reads.
- The structural-invariant tests promised in the depth choice ("comprehensive trace + structural tests") will be authored as part of Wave 11+ fix waves, not this audit. The audit identifies *which* invariants need pinning; the fix waves write the tests.

## Risks

| Risk | Mitigation |
|---|---|
| Agents return overlapping findings | Aggregation step dedupes by symptom + file:line |
| An agent finds nothing because briefing was too narrow | Each agent gets a "search beyond your slice if you smell drift" instruction; cross-references caught at aggregation |
| User triage stalls | Fix-wave proposal makes triage a checkbox exercise (HIGH = approve all? approve some? skip?) |
| Audit reveals a fundamental design issue, not just bugs | Surface as "Open Question" with two paths: (a) targeted patch, (b) larger redesign in a follow-up brainstorm session |
| Token budget exhaustion across 6 agents | Each agent has bounded scope (max ~3 files of code + briefing); agents are dispatched in one parallel batch |

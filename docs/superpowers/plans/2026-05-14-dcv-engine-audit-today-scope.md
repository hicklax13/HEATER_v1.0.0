# DCV Engine Audit — "Today" Scope — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute a 6-agent parallel audit of the lineup optimizer's "Today" scope and produce a categorized findings catalog for Wave 11+ fix waves.

**Architecture:** Six specialist audit agents (each owning a disjoint slice of the DCV engine) are dispatched in a single parallel batch. Each returns a markdown findings report. The orchestrator (this plan's executor) aggregates, dedupes, categorizes, and writes a single findings spec for user triage.

**Tech Stack:** Python 3.11-3.14, pytest, ruff. Audit-only — no production code modified during this plan.

**Spec:** [`docs/superpowers/specs/2026-05-14-dcv-engine-audit-today-scope-design.md`](../specs/2026-05-14-dcv-engine-audit-today-scope-design.md)

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `docs/superpowers/specs/2026-05-14-dcv-engine-audit-today-scope-context.md` | Create | Shared briefing bundle every agent receives (boundary, format, severity, indisputability). One source of truth so all agents return comparable reports. |
| `docs/superpowers/specs/2026-05-14-dcv-engine-audit-today-scope-findings.md` | Create | Final aggregated findings catalog (the deliverable). Sorted by severity, deduped, with fix-wave proposal. |
| `docs/superpowers/specs/2026-05-14-dcv-engine-audit-agent-{a1..a6}-report.md` | Create (6 files) | Raw per-agent report files. Persisted so the aggregation step is reviewable and the audit is reproducible. |

**Note:** This is an audit, not a feature build. There are no production code changes, no tests, and no commits to `src/`. Every artifact lives in `docs/superpowers/specs/`.

---

### Task 1: Write the shared audit-context briefing bundle

**Why:** All 6 agents need identical context for the boundary, findings format, severity rubric, and indisputability rule. Centralizing it prevents drift between briefings and makes the audit reproducible.

**Files:**
- Create: `docs/superpowers/specs/2026-05-14-dcv-engine-audit-today-scope-context.md`

- [ ] **Step 1: Write the context bundle**

Write this file with the exact content below:

````markdown
# DCV Engine Audit — Shared Agent Briefing Bundle

This file is the shared context every audit agent receives. Every agent's
briefing references this file by path; the agent reads it once and applies
its rules consistently.

## Boundary

### In scope (the "Today" branch of the lineup optimizer)

- `pages/6_Line-up_Optimizer.py` lines 939-1011 (the `_scope_key == "today"` branch — input wiring + display contract)
- `src/optimizer/daily_optimizer.py` (1302 LOC) — `build_daily_dcv_table` and every helper: `compute_health_factor`, `compute_volume_factor`, `compute_matchup_multiplier`, weighted DCV math, stud floor, SP gate, forced-start, locked-teams, `DailyDCVContext`
- `src/optimizer/shared_data_layer.py` — `build_optimizer_context` (urgency_weights, confirmed_lineups, recent_form, team_strength)
- `src/matchup_context.py` — `MatchupContextService` (3 modes: matchup / standings / blended)
- `src/optimizer/category_urgency.py` — sigmoid urgency math + runtime reads from `CONSTANTS_REGISTRY`
- `src/optimizer/constants_registry.py` — every constant cited by the above
- `src/game_day.py` — `get_target_game_date` (auto-advance today→tomorrow when all games final)

### Out of scope

- Projection accuracy (audited in SF-1..SF-83)
- Park factor *values* (audited in SF-22 / SF-30)
- statsapi schedule reliability (data-source level — covered by prior audits)
- LP solver (`src/lineup_optimizer.py`) — that's Rest-of-Week / Rest-of-Season scope
- Other optimizer tabs (Streaming, Manual, Roster, Start/Sit)
- Yahoo data freshness (`yahoo_data_service` 3-tier cache audited)

## League Context (FourzynBurn)

Source: `CLAUDE.md` "League Context" section.

- Format: 12-team H2H Categories, snake draft, 23 rounds
- Hitting cats (6): R, HR, RBI, SB, AVG, OBP
- Pitching cats (6): W, L, SV, K, ERA, WHIP
- Inverse cats: L, ERA, WHIP (lower is better)
- Rate stats: AVG, OBP, ERA, WHIP
- Roster: C/1B/2B/3B/SS/3OF/2Util/2SP/2RP/4P/6BN/4IL = 28 slots
- Yahoo game_key: 469
- Transactions: 10 adds+trades combined per matchup week (FCFS)
- Can't-drop: Players drafted in rounds 1-3 per team (36 league-wide)

The single source of truth in code is `src.valuation.LeagueConfig`. Hardcoded
category lists in source files are structurally guarded against by
`tests/test_no_hardcoded_categories_in_src.py` and siblings.

## Findings Format

Every finding in your report MUST use this format:

```
ID:       DCV-A{your_agent_number}-{three_digit_serial}
Severity: HIGH | MED | LOW
File:     path/relative/to/repo:line_number
Symptom:  One-sentence description of what's wrong
Expected: What the correct behavior should be (with citation)
Actual:   What the code does today
Proof:    Math derivation OR CLAUDE.md/research citation OR FourzynBurn rule contradiction
Repro:    Code/data scenario where the bug fires (concrete inputs + expected output)
Fix:      One-line suggestion (NOT implementation — just the direction)
```

Number serially within your agent (DCV-A1-001, DCV-A1-002, ...). Do not
reuse numbers across agents.

## Severity Rubric

| Tier | Definition | Examples |
|---|---|---|
| HIGH | Wrong math/data that changes lineup decisions. Mis-ranks players by >5% DCV, or causes a wrong start/bench. | Wrong sign on inverse stat, wrong rate-stat aggregation (simple mean instead of weighted), sigmoid k drifted from calibrated value, missing IL status in exclusion list |
| MED | Bounded impact: rare scenario, edge-of-roster decision, or only fires when fallback path triggers. | Wrong fallback when primary signal present, magic threshold without citation, silent default that masks data drift, inconsistent precision (`.3f` vs `.2f`) |
| LOW | Cosmetic, code-quality, missing guards. No behavior bug today; future regression risk. | Undocumented constants, type drift, dead-code branches, missing structural-invariant test, comment rot |

## Indisputability Rule (HIGH findings)

Every HIGH finding MUST carry one of these proofs in the `Proof:` field:

1. **Provable math derivation** — show the math. Example: "Weighted AVG = Σh/Σab; the code uses Σ(h/ab)/N. By Jensen's inequality these are equal only when all (h/ab) are identical. Real-roster diff: 0.265 vs 0.273 = 3% relative error on user's 9-hitter starting lineup."
2. **Direct citation** — quote the source. Example: "`CLAUDE.md` line 178 documents `sigmoid_k_counting=2.0`; the code at `category_urgency.py:42` reads from a stale alias bypassing `CONSTANTS_REGISTRY` (SF-39 regression pattern)."
3. **FourzynBurn rule contradiction** — quote the rule. Example: "League settings (`CLAUDE.md` line 95: `Inverse cats: L, ERA, WHIP`); this code's `_INVERSE_CATS = {'ERA','WHIP'}` drops `L`, causing positive-good treatment of losses."

If you flag something HIGH but cannot supply any of these three proofs, the
aggregation step demotes it to MED with note "demoted: HIGH claim without
indisputable proof." So: prove it or rank it MED.

MED findings get a citation OR repro but not all three. LOW findings get a
one-sentence justification.

## Output Location

Write your report to the EXACT path specified in your individual briefing
(one of `docs/superpowers/specs/2026-05-14-dcv-engine-audit-agent-aN-report.md`).

Format:

```markdown
# Audit Agent A{N} — {slice name} Report

**Date:** 2026-05-14
**Slice:** {brief slice description}
**Files audited:** {list}
**Findings count:** HIGH={n} MED={n} LOW={n}

## Findings

### DCV-A{N}-001
ID: ...
Severity: ...
[full block per format above]

### DCV-A{N}-002
[...]

## Coverage gaps

[List anything you couldn't conclusively audit and why. This goes into
the "Open Questions" section of the final spec.]

## Notes

[Any context the orchestrator should know when aggregating. E.g.,
"DCV-A{N}-005 likely overlaps with another agent's finding on the SP gate
because that code has both math and constants concerns."]
```
````

- [ ] **Step 2: Verify file exists**

Run: `ls docs/superpowers/specs/2026-05-14-dcv-engine-audit-today-scope-context.md`
Expected: file path printed (no error)

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-05-14-dcv-engine-audit-today-scope-context.md
git commit -m "docs(audit): shared briefing bundle for DCV engine audit agents

Defines audit boundary (Today scope), findings format, severity rubric,
indisputability rule, and output template. Referenced by all 6 agent
briefings so reports return in comparable shape.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Dispatch 6 audit agents in parallel

**Why:** Parallel dispatch maximizes wall-clock efficiency and gives each agent fresh context (no drift). The 6 slices are disjoint so agents won't redundantly cover the same code.

**Files:**
- Each agent will create one file: `docs/superpowers/specs/2026-05-14-dcv-engine-audit-agent-a{1..6}-report.md`

This task uses ONE message with SIX `Agent` tool calls. Per Anthropic's docs:
"If you intend to call multiple tools and there are no dependencies between
them, make all of the independent calls in the same Agent tool calls block."

- [ ] **Step 1: Dispatch all 6 agents in a single message**

In a single response, call the `Agent` tool 6 times in parallel using
`subagent_type: general-purpose` for each. Use the briefings below verbatim
as the `prompt` parameter — they are already fully written; no substitution
needed. Set the `description` field to the value shown above each briefing.

#### Agent A1: Algorithms

```
description: DCV algorithm correctness audit
prompt:

You are audit Agent A1 for the HEATER fantasy baseball app's lineup
optimizer. Your slice: ALGORITHM CORRECTNESS in the daily DCV engine.

REQUIRED READING (in this order):
1. docs/superpowers/specs/2026-05-14-dcv-engine-audit-today-scope-context.md
   — read this entire file first. It defines the audit boundary, findings
   format, severity rubric, and indisputability rule. Apply it strictly.
2. src/optimizer/daily_optimizer.py — your primary slice. Trace every
   formula. Pay special attention to:
   - compute_health_factor
   - compute_volume_factor
   - compute_matchup_multiplier
   - weighted DCV math (per-category and total)
   - stud floor protection
   - SP gate (the "DCV gate" logic — see CLAUDE.md gotcha about pure-P)
   - forced-start logic (matchup_mult < 0.70 OR DCV < median × 0.5)
   - locked_teams (game already started zeroes DCV)
3. src/valuation.py — read SGPCalculator + LeagueConfig because the DCV
   math uses SGPCalculator.totals_sgp.
4. CLAUDE.md sections "Algorithms & Math" and "Lineup Optimizer" — these
   document the intended behavior. Compare code to docs.

YOUR JOB:
- Walk every formula in the in-scope code. For each, ask: is the math
  correct against the documented behavior in CLAUDE.md and against the
  FourzynBurn league rules?
- Check sign handling on inverse stats (L, ERA, WHIP).
- Check rate-stat aggregation (must be weighted: ERA = Σer*9/Σip; WHIP =
  Σ(bb+h)/Σip; AVG = Σh/Σab; OBP = Σ(h+bb+hbp)/Σ(ab+bb+hbp+sf)).
- Check that the SP gate handles all three cases: positions has SP token,
  positions has RP token, positions is pure-P (the third case is the
  one that was historically dead code; CLAUDE.md mentions the fix).
- Check stud floor: does it actually protect the highest-DCV players or
  is it a no-op?

WRITE YOUR REPORT TO:
docs/superpowers/specs/2026-05-14-dcv-engine-audit-agent-a1-report.md

Use the format specified in the context bundle. Number findings
DCV-A1-001 through DCV-A1-NNN.

DO NOT MODIFY ANY OTHER FILES. This is an audit — read-only.
DO NOT propose implementations — only describe what's wrong and the
direction of the fix in the Fix: line.

If you find nothing wrong in your slice, write the report with an empty
findings section and explain in "Coverage gaps" or "Notes" what you
checked and why nothing surfaced.

Return a brief 3-line summary in your final message: how many HIGH/MED/LOW
findings you wrote, and the path to your report file.
```

#### Agent A2: Constants & Thresholds

```
description: DCV constants/thresholds audit
prompt:

You are audit Agent A2 for the HEATER fantasy baseball app. Your slice:
CONSTANTS AND THRESHOLDS in the daily DCV engine.

REQUIRED READING (in this order):
1. docs/superpowers/specs/2026-05-14-dcv-engine-audit-today-scope-context.md
   — read this entire file first. It defines the audit boundary, findings
   format, severity rubric, and indisputability rule. Apply it strictly.
2. src/optimizer/constants_registry.py — your primary slice. Read every
   entry and its citation/bounds.
3. src/optimizer/daily_optimizer.py — every magic number in this file
   should either come from CONSTANTS_REGISTRY (preferred) or have a
   citation in CLAUDE.md.
4. src/optimizer/category_urgency.py — sigmoid k values must be read from
   CONSTANTS_REGISTRY at runtime (per SF-39 fix).
5. CLAUDE.md "Lineup Optimizer" section — documents constant values
   (e.g., COUNTING_STAT_K=2.0, RATE_STAT_K=3.0, _STREAM_IP_TARGET=54,
   two-start fatigue 0.93×, platoon 7.5%/5.8%, _PITCHER_EMPTY_THRESHOLD,
   _STREAM_NET_SGP_RELAXED, IP budget 54).

YOUR JOB:
- For every magic number in the in-scope code: find its citation in
  CONSTANTS_REGISTRY or CLAUDE.md. Findings if:
  - Number not cited anywhere (where did it come from?)
  - Number cited but value drifted from citation (the citation says 2.0,
    code uses 2.5)
  - Number duplicated in code instead of imported from registry
  - Sigmoid k values read from a module-level alias instead of
    CONSTANTS_REGISTRY at runtime (regression of SF-39)
- Check thresholds for boundary correctness: is "matchup_mult < 0.70"
  in forced-start exclusive or inclusive? Does CLAUDE.md say <= or <?
- Check that the IP budget (54) and weeks (26) match FourzynBurn (NOT
  24 weeks per SF-42).

WRITE YOUR REPORT TO:
docs/superpowers/specs/2026-05-14-dcv-engine-audit-agent-a2-report.md

Use the format in the context bundle. Number DCV-A2-001 onward.

DO NOT MODIFY ANY OTHER FILES. Audit only.

Return a brief 3-line summary in your final message: HIGH/MED/LOW counts
and report path.
```

#### Agent A3: League-Config Alignment

```
description: DCV league-config alignment audit
prompt:

You are audit Agent A3 for the HEATER fantasy baseball app. Your slice:
FOURZYNBURN LEAGUE-CONFIG ALIGNMENT in the daily DCV engine.

REQUIRED READING (in this order):
1. docs/superpowers/specs/2026-05-14-dcv-engine-audit-today-scope-context.md
   — read this entire file first.
2. src/valuation.py — LeagueConfig is the single source of truth.
3. src/optimizer/daily_optimizer.py — your primary check target.
4. src/optimizer/shared_data_layer.py — secondary check target.
5. src/matchup_context.py — secondary check target.
6. CLAUDE.md "League Context" section — the FourzynBurn settings.
7. tests/test_no_hardcoded_categories_in_src.py and siblings — the
   structural guards. Find anything they miss in your in-scope files.

YOUR JOB:
- For every place in the in-scope code where category lists, inverse
  sets, rate sets, or roster slot assumptions appear: verify they come
  from LeagueConfig (preferred) or that the hardcoded value matches
  FourzynBurn exactly.
- Specific FourzynBurn values to verify in code:
  - Hitting cats: {R, HR, RBI, SB, AVG, OBP}
  - Pitching cats: {W, L, SV, K, ERA, WHIP}
  - Inverse: {L, ERA, WHIP} — the L MUST be present (SF-29 type bug)
  - Rate: {AVG, OBP, ERA, WHIP}
  - 26-week season (SF-42 fix; older code had 24)
  - 28-slot roster: 2SP+2RP+4P+6BN+4IL
  - Athletics team code "ATH" (NOT "OAK") per Wave 1 / SF-57
- For every multiplier/blend that depends on a categorical assumption
  (e.g., "ERA weighted by IP, AVG weighted by AB"), verify the rule
  matches FourzynBurn rate-stat list.
- Check for assumptions that would break if FourzynBurn changed cats
  (e.g., adding HLD for pitching) — these are LOW (future-proofing) but
  worth flagging if structural guards are missing.

WRITE YOUR REPORT TO:
docs/superpowers/specs/2026-05-14-dcv-engine-audit-agent-a3-report.md

Use the format in the context bundle. Number DCV-A3-001 onward.

DO NOT MODIFY ANY OTHER FILES. Audit only.

Return a brief 3-line summary in your final message: HIGH/MED/LOW counts
and report path.
```

#### Agent A4: Inputs & Data Shapes

```
description: DCV input/data-shape audit
prompt:

You are audit Agent A4 for the HEATER fantasy baseball app. Your slice:
INPUT SCHEMAS AND DATA SHAPES feeding the daily DCV engine.

REQUIRED READING (in this order):
1. docs/superpowers/specs/2026-05-14-dcv-engine-audit-today-scope-context.md
   — read this entire file first.
2. src/optimizer/daily_optimizer.py — focus on build_daily_dcv_table
   signature: every parameter's expected type and shape.
3. src/optimizer/shared_data_layer.py — build_optimizer_context produces
   the inputs. Trace every output.
4. src/matchup_context.py — get_category_weights output schema.
5. src/optimizer/category_urgency.py — compute_urgency_weights output
   schema.
6. pages/6_Line-up_Optimizer.py lines 939-1011 — the call site that
   feeds inputs to build_daily_dcv_table.
7. CLAUDE.md "Key API Signatures" section.

YOUR JOB:
- For each input parameter to build_daily_dcv_table:
  - matchup
  - schedule_today
  - park_factors
  - urgency_weights
  - confirmed_lineups
  - recent_form
  - rate_modes
  - team_strength
- Verify:
  - Documented shape matches actual usage in the function
  - Producer (e.g., shared_data_layer for urgency_weights) emits the
    same shape the consumer expects
  - Type annotations match runtime behavior
  - Optional inputs (None) have sensible fallbacks (NOT silent zero)
  - Mutable defaults aren't shared across calls
- Check the call site (page) passes the correct types/shapes:
  - statsapi.schedule(date=...) returns list[dict] with what keys?
  - PARK_FACTORS dict has what shape?
  - yds.get_matchup() return type?
- Specific type bugs to look for (per SF-77 / Wave 8c findings):
  - dict[str, Any] where TypedDict would catch shape errors
  - dict[str, callable] (lowercase) which silently becomes Any
  - dict[int, dict] vs dict[str, dict] confusion (player_id keys)

WRITE YOUR REPORT TO:
docs/superpowers/specs/2026-05-14-dcv-engine-audit-agent-a4-report.md

Use the format in the context bundle. Number DCV-A4-001 onward.

DO NOT MODIFY ANY OTHER FILES. Audit only.

Return a brief 3-line summary in your final message: HIGH/MED/LOW counts
and report path.
```

#### Agent A5: MatchupContext + Sigmoid

```
description: MatchupContext and sigmoid urgency audit
prompt:

You are audit Agent A5 for the HEATER fantasy baseball app. Your slice:
MATCHUPCONTEXTSERVICE AND SIGMOID URGENCY MATH.

REQUIRED READING (in this order):
1. docs/superpowers/specs/2026-05-14-dcv-engine-audit-today-scope-context.md
   — read this entire file first.
2. src/matchup_context.py — your primary slice. The 3 modes:
   "matchup" (H2H urgency), "standings" (gap analysis), "blended"
   (alpha-weighted).
3. src/optimizer/category_urgency.py — sigmoid math. compute_urgency_weights
   reads sigmoid_k_counting and sigmoid_k_rate from CONSTANTS_REGISTRY at
   runtime (per SF-39).
4. src/optimizer/shared_data_layer.py — caller of MatchupContextService;
   how it picks which mode for "today" scope.
5. CLAUDE.md "Lineup Optimizer" section — sigmoid urgency entry.

YOUR JOB:
- Verify the 3-mode logic in MatchupContextService:
  - "matchup" — uses H2H opponent + my-team gap to weight categories
  - "standings" — uses league-wide gaps, ignores opponent
  - "blended" — alpha-weighted blend of the two
  - Are the alpha bounds correct? What does "today" scope use?
- Verify sigmoid math:
  - urgency = 1 / (1 + exp(-k * (gap)))
  - With k=2.0 for counting stats and k=3.0 for rate stats
  - Sign of gap: are we maximizing my-team gain or opponent's loss?
  - Per CLAUDE.md, sigmoid k MUST be read from CONSTANTS_REGISTRY at
    runtime so calibrate_sigmoid.py updates take effect without restart.
    Verify category_urgency.py does this (NOT module-level alias).
- Verify rate_modes ("abandon" / "contest") logic:
  - When ERA mode is "abandon", per CLAUDE.md pitcher DCV for ERA is
    zeroed. Trace this to make sure it actually happens.
- Check that get_matchup_context().get_category_weights returns ALL 12
  cats, not a subset (per LeagueConfig).

WRITE YOUR REPORT TO:
docs/superpowers/specs/2026-05-14-dcv-engine-audit-agent-a5-report.md

Use the format in the context bundle. Number DCV-A5-001 onward.

DO NOT MODIFY ANY OTHER FILES. Audit only.

Return a brief 3-line summary in your final message: HIGH/MED/LOW counts
and report path.
```

#### Agent A6: Edge Cases

```
description: DCV edge-case audit
prompt:

You are audit Agent A6 for the HEATER fantasy baseball app. Your slice:
EDGE CASES AND DEAD-CODE BRANCHES in the daily DCV engine.

REQUIRED READING (in this order):
1. docs/superpowers/specs/2026-05-14-dcv-engine-audit-today-scope-context.md
   — read this entire file first.
2. src/optimizer/daily_optimizer.py — focus on every fallback, every
   "if X is None" branch, every except clause, every `or 0.0` default.
3. src/optimizer/shared_data_layer.py — same scrutiny.
4. src/matchup_context.py — same scrutiny.
5. CLAUDE.md "Gotchas" section — many edge-case bugs are documented here.

YOUR JOB:
Find every branch where the engine silently degrades because input is
missing. For each:
- Is the fallback value reasonable? (e.g., volume_factor=0.0 for no
  game today is correct; matchup_mult=1.0 for missing park factor is
  reasonable; ERA=4.50 for missing pitcher is reasonable)
- Is the silent fallback masking a data bug? (e.g., if recent_form is
  None for a player who should have it, that's a silent data bug)
- Does the fallback log a warning? (Per Wave 8b, silent failures must
  log with exc_info=True)
- Specific edge cases to verify:
  - DTD/IL/SUSP exclusion (Wave 10 added SUSP — confirm no other Yahoo
    statuses leak through as healthy)
  - locked_teams (game in progress or final) zeroes volume — does this
    actually fire correctly?
  - pure-P pitcher fallback (Wave 10 added the SP gate fallback —
    confirm no other position string slips through the gate)
  - Missing projection (player not in projection table) — what's the
    fallback?
  - Zero-IP rate stats — does ERA/WHIP collapse to None or 0 or NaN?
  - sprint_speed boost — is the column always read defensively?
  - two-start fatigue — does it apply on the 2nd start only? what if
    a pitcher's second start is mid-week?
  - Pinch hitters / two-way players (Ohtani) — special handling
    correct?
  - Doubleheader handling — volume_factor=2.0 for confirmed
    doubleheader starters; is this checked?
- Look for dead-code branches: code that can never execute given the
  type system or surrounding logic. Flag with the demonstration.

WRITE YOUR REPORT TO:
docs/superpowers/specs/2026-05-14-dcv-engine-audit-agent-a6-report.md

Use the format in the context bundle. Number DCV-A6-001 onward.

DO NOT MODIFY ANY OTHER FILES. Audit only.

Return a brief 3-line summary in your final message: HIGH/MED/LOW counts
and report path.
```

- [ ] **Step 2: Wait for all 6 agent reports**

When the Agent tool calls complete, each agent's final message will include
a 3-line summary and confirm the report file path. Verify all 6 files exist:

```bash
ls docs/superpowers/specs/2026-05-14-dcv-engine-audit-agent-a*-report.md
```

Expected: 6 files listed.

- [ ] **Step 3: Spot-check report quality**

Open one report and verify it follows the format:

```bash
head -30 docs/superpowers/specs/2026-05-14-dcv-engine-audit-agent-a1-report.md
```

Expected: header with date/slice/files/findings count, then findings in
the prescribed format (ID/Severity/File/Symptom/Expected/Actual/Proof/
Repro/Fix).

If any report is missing, malformed, or empty (no findings AND no coverage
gaps explanation), re-dispatch that single agent with the same briefing.

- [ ] **Step 4: Commit raw reports**

```bash
git add docs/superpowers/specs/2026-05-14-dcv-engine-audit-agent-a*-report.md
git commit -m "docs(audit): raw findings reports from 6 audit agents

Per the DCV engine Today-scope audit. Reports are persisted as-is so
the aggregation step is reviewable and the audit is reproducible.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Aggregate, dedupe, and categorize findings

**Why:** Multiple agents may flag the same root cause from different angles. Without dedup the final spec is noisy. Categorization by severity (with the demotion rule applied) produces a triage-ready catalog.

**Files:**
- Read: all 6 `docs/superpowers/specs/2026-05-14-dcv-engine-audit-agent-a*-report.md`
- Stage for next task: in-memory aggregation (no intermediate file)

- [ ] **Step 1: Read all 6 reports**

Use the `Read` tool on each of the 6 report files. Maintain an in-memory
list of all findings as `(agent_id, finding_id, severity, file_line,
symptom, expected, actual, proof, repro, fix)` tuples.

- [ ] **Step 2: Apply the demotion rule**

For each HIGH finding, check the `Proof:` field. If it does NOT contain
one of:
- A math derivation (look for `=` signs, formula structure, numeric
  comparison)
- A direct citation of `CLAUDE.md`, `constants_registry`, a Wave/SF
  number, or a research paper
- A FourzynBurn rule contradiction (look for explicit reference to a
  league setting)

...demote it to MED and append to the `Symptom:` field: " (demoted from
HIGH: no indisputable proof supplied)".

- [ ] **Step 3: Dedupe by symptom + file**

Group findings where:
- Same `File:` value (same file:line ±5 lines), AND
- Symptoms describe the same root cause (paraphrase-tolerant; use
  judgment)

When duplicates found, merge into the highest-severity entry and append
the other agents' IDs to a new `Also flagged by:` field. Keep the proof
from the most rigorous source.

Example: A1 flags `daily_optimizer.py:642` as HIGH for math, A2 flags
the same line as HIGH for missing constant citation, A6 flags it as MED
for a dead branch — these may all describe the same root cause. Merge to
one entry, severity HIGH, with all three perspectives in Notes.

- [ ] **Step 4: Categorize and group**

Sort the deduped list by severity (HIGH → MED → LOW), then by file path
within each severity tier. Compute counts per severity.

- [ ] **Step 5: Plan fix waves**

Group HIGH findings by code area. Each group becomes a fix-wave candidate.
For each wave, estimate:
- LOC delta (count touched files × ~20 lines avg, plus tests)
- Test count (one regression test per finding minimum)
- Dependencies (does Wave A's fix change something Wave B depends on?)

Repeat for MED. Keep LOW as a backlog list, not a wave.

---

### Task 4: Write the findings spec

**Files:**
- Create: `docs/superpowers/specs/2026-05-14-dcv-engine-audit-today-scope-findings.md`

- [ ] **Step 1: Write the findings spec**

Write the file with this structure:

````markdown
# DCV Engine Audit — Today Scope — Findings

**Date:** 2026-05-14
**Auditors:** 6 parallel agents (A1-A6)
**Spec:** [design](2026-05-14-dcv-engine-audit-today-scope-design.md)
**Context:** [context bundle](2026-05-14-dcv-engine-audit-today-scope-context.md)
**Raw reports:** [a1](2026-05-14-dcv-engine-audit-agent-a1-report.md) | [a2](2026-05-14-dcv-engine-audit-agent-a2-report.md) | [a3](2026-05-14-dcv-engine-audit-agent-a3-report.md) | [a4](2026-05-14-dcv-engine-audit-agent-a4-report.md) | [a5](2026-05-14-dcv-engine-audit-agent-a5-report.md) | [a6](2026-05-14-dcv-engine-audit-agent-a6-report.md)

## Executive Summary

- HIGH: {count}
- MED: {count}
- LOW: {count}
- Demoted (HIGH → MED): {count}

**Headline risks:**
1. {1-sentence risk for top HIGH finding}
2. {1-sentence risk for second HIGH finding}
3. ...

**Confidence statement:** After this audit, the DCV engine for "Today"
scope is correct against FourzynBurn settings to {percentage}%
confidence, with {N} open questions and {M} deferred LOW findings.
{1-sentence justification — why this percentage and not higher.}

## Methodology

Six parallel specialist audit agents covered disjoint slices of the DCV
engine per the [audit design spec](2026-05-14-dcv-engine-audit-today-scope-design.md).
Each agent's raw report is linked above.

Aggregation applied:
- Demotion rule: HIGH findings without one of {math derivation, direct
  citation, FourzynBurn contradiction} were demoted to MED.
- Dedup: findings touching the same file:line within ±5 lines and
  describing the same root cause were merged.
- Categorization: severity tiers HIGH/MED/LOW per the rubric in the
  context bundle.

## Findings Catalog

### HIGH (must-fix before next H2H matchup)

#### {DCV-AX-NNN} — {short symptom}
[full block per format, with Also-flagged-by + Notes if merged]

[... repeat for each HIGH ...]

### MED (should-fix before season-end)

[... ]

### LOW (backlog — fix when touched)

[... ]

## Fix-Wave Proposal

### Wave 11A — HIGH fixes (recommended priority)
- Findings: {list of IDs}
- Estimated LOC delta: ~{N} lines + {M} new tests
- Estimated effort: {hours / sessions}
- Dependencies: {none | depends on Wave XYZ}
- Pattern: TDD per fix, structural-invariant test per fix, single PR

### Wave 11B — MED fixes
[same structure]

### Backlog — LOW
- Tracked in this spec; no wave commitment. Fix opportunistically when
  touching adjacent code.

## Open Questions

[Questions where the audit cannot conclude without user input. Each
labeled with the finding(s) that depend on the answer.]

1. **{Question}** — relevant to {DCV-AX-NNN, ...}. {Why it matters in
   1 sentence.}

## Indisputability Statements

For each HIGH finding, the proof artifact (math derivation, citation,
or FourzynBurn rule contradiction) is shown inline in the finding's
`Proof:` field. The aggregation step verified each HIGH carries a
proof; demotions are listed in the Executive Summary count.
````

Fill in the actual findings, counts, and content from Task 3.

- [ ] **Step 2: Verify spec is complete (no placeholder text)**

Search for `{` and `}` characters in the file:

```bash
grep -n "{" docs/superpowers/specs/2026-05-14-dcv-engine-audit-today-scope-findings.md
```

Expected: only template literals from intentional content (e.g., URLs,
code blocks). No `{count}`, `{N}`, `{question}`, etc. that should have
been filled in.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-05-14-dcv-engine-audit-today-scope-findings.md
git commit -m "docs(audit): aggregated DCV engine audit findings (Today scope)

{N} findings total: HIGH={x}, MED={y}, LOW={z}, demoted={w}.

Top HIGH risks summarized in Executive Summary. Fix-wave proposal
groups findings into Wave 11A (HIGH) and Wave 11B (MED). LOW
backlog deferred.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Present to user for triage

**Why:** The user owns the decision on which fix waves to execute. The audit deliverable is information; the user gates action.

- [ ] **Step 1: Push the audit branch and update PR description**

```bash
git push origin claude/great-shirley-ba6c4a
```

If PR #25 is still the active branch PR, comment on it with a link to
the findings spec. Otherwise no PR needed for audit-only work.

- [ ] **Step 2: Present summary to user**

In a single message:
- Top 3 HIGH risks (1 sentence each, with finding IDs)
- HIGH/MED/LOW counts
- Open questions (so the user knows what input is needed)
- Wave 11A/11B proposal (so the user can pick which to run)
- Link to the findings spec

End with an `AskUserQuestion` offering:
- Option 1: Approve Wave 11A — start fixing HIGH findings now
- Option 2: Triage first — user reviews findings spec, decides per-finding
- Option 3: Start with Wave 11B (MED) — if HIGH count is low or all are
  contested
- Option 4: Punt — close this audit and revisit later

---

## Self-Review Notes

Run after writing this plan:

1. **Spec coverage:** Every section of the design spec maps to a task here:
   - "Boundary" → Task 1 (context bundle includes boundary)
   - "Methodology" → Task 2 (6-agent dispatch)
   - "Findings format" → Task 1 (in context bundle) + Task 4 (used in spec)
   - "Severity rubric" → Task 1 + Task 3 (applied during aggregation)
   - "Indisputability rule" → Task 1 + Task 3 (demotion rule applied)
   - "Aggregation flow" → Task 3
   - "Deliverable" → Task 4 (writes the findings spec)
   - "Execution plan" → Tasks 1-5 (this whole plan)
   - "Success criteria" → Task 4 verifies all are met
   - "Risks" → mitigation embedded in tasks (e.g., Task 2 step 3 spot-
     checks reports; Task 3 dedupes; Task 4 step 2 placeholder scan)
2. **Placeholder scan:** Each agent briefing is fully written; severity
   rubric is fully written; demotion rule is explicit; spec template has
   `{count}` `{N}` etc. placeholders that ARE meant to be filled in by
   the executor (these are correctly flagged in Task 4 Step 2 with the
   grep check).
3. **Type consistency:** File paths use consistent format
   `docs/superpowers/specs/2026-05-14-dcv-engine-audit-...md`. Agent
   numbers consistent (A1-A6, lowercase a1-a6 in filenames).

No issues found.

---

## Notes on Adapting This Plan

This plan is structured as TDD-adjacent: each task has bite-sized steps
with explicit verification, but there is no failing-test-first cycle
because no production code is written. The "test" for each task is the
verification step (e.g., "ls returns 6 files," "grep finds no
placeholder text," "head shows correct format").

If during execution the agents return findings that suggest the audit
boundary should expand (e.g., a HIGH finding in `lineup_optimizer.py`
discovered while tracing daily_optimizer call sites), surface it as an
"Open Question" in the findings spec rather than auto-expanding scope.
The user decides whether to launch a follow-on audit.

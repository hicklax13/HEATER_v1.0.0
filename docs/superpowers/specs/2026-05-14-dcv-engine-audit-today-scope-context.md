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
2. **Direct citation** — quote the source. Example: "`CLAUDE.md` line 178 documents `sigmoid_k_counting=2.0`; the code at `category_urgency.py:42` reads `2.5` from a stale alias bypassing `CONSTANTS_REGISTRY` (SF-39 regression pattern)."
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

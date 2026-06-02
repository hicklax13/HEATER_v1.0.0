# Comprehensive Per-Team Pre-Launch QA — Design Spec

**Date:** 2026-06-02
**Status:** Design (awaiting owner review → writing-plans)
**Owner:** Team Hickey (CLI novice; Claude does all technical work, owner reviews + approves)

## Plain-language summary

Before inviting the 12 FourzynBurn leaguemates, prove that **every page, tab, and
feature of HEATER loads, works, and shows sensible data for all 12 teams** — not
just Team Hickey. This combines automated coverage (a multi-agent fleet builds a
per-team test suite), a real-browser walkthrough (Claude drives, captures
screenshots), and systematic per-team data sanity checks. Everything found is
fixed and re-verified before launch. Claude does all technical work; the owner
reviews findings and approves fixes/deploys.

This program exists because the session that preceded it shipped fixes verified
only for Team Hickey + one test account — which is **not** evidence the app is
correct for all 12 members. Two real per-team bugs (every member saw the admin's
team; mobile nav unreachable) were caught only by ad-hoc testing; this program
makes that coverage systematic.

## Acceptance criteria ("launch-ready")

For **each of the 12 teams**, across **every page (13) + every tab/feature**:

1. **Loads** — renders with no exception, no error banner, no
   blank-where-data-should-be.
2. **Works** — tabs, buttons, dropdowns, and forms respond; interactive flows
   complete.
3. **Sensible output** — no `NaN`/`None`/empty in shown values; no implausible
   numbers (negative counting stats, ERA 0 or ∞, lineup with wrong slot counts,
   standings rows that don't reconcile, SGP off by orders of magnitude).
4. **Role-safe** — a member sees only their own team's data and their own nav
   (no admin links, no other member's private data) and cannot trigger
   shared-DB writes. Admin sees admin surfaces.
5. **Mobile** — every page is reachable (nav opens) and usable on a phone width
   for each team.

**Done** = green per-team automated suite + clean browser walkthrough + all
findings triaged/fixed/re-verified + CLAUDE.md regression guards added.

**Explicitly out of scope** (owner's chosen bar): field-by-field reconciliation
of each team's data against the live Yahoo league. We verify *integrity and
plausibility* per team, not exact-match-vs-Yahoo. (Can be a later add-on.)

## Scope

- **Teams:** all 12, sourced from `league_teams` / `league_rosters`.
- **Surfaces:** the 13 season pages + their tabs (Lineup Optimizer 6, Leaders 7,
  Trade Finder 5, League Standings 3, Matchup Planner per-tab, etc.), the
  Home/Draft Tool landing page, and the 3 admin pages (tested under the admin
  role, and confirmed *inaccessible* under a member role).
- **Roles:** member (the 12) + admin (Team Hickey).

## Phases

### Phase 0 — Realistic local data
Automated tests run locally (fast, offline, repeatable; the `conftest` network
guard forces SQLite fallbacks). For coverage to be meaningful, the local
`data/draft_tool.db` must hold all 12 teams' rosters/stats/standings/projections.
Action: sync the league data from Yahoo into the local DB (the local token is
live) **or** verify it already has all 12 teams. Gate: ≥ 12 teams with non-empty
rosters present before Phase 1.

### Phase 1 — Automated per-team coverage (agent fleet)
- **Test template (Claude-authored, strict):** an `AppTest`-based harness that,
  given a page path + a team identity (a `MULTI_USER` session with
  `auth_user.team_name = TEAM`, `is_admin` per role), runs the page headlessly,
  asserts no exception, then inspects rendered elements for error/blank +
  output-sanity. One reusable parametrization over `(team, page, tab)`.
- **Fleet:** a `Workflow` fans out ~one agent per page; each fills the template
  for its page's tabs/features, parametrized over the 12 teams. Batched
  (≤ 4 concurrent) to stay under platform rate limits (learned 2026-06-01).
- **Quality gate:** Claude reviews every generated test; `pr-test-analyzer`
  audits whether the tests *actually assert* meaningful per-team behavior (not
  vacuous green).

### Phase 2 — Run + triage
- Run the full per-team suite (`pytest`). Collect each failure as
  `(team, page, tab, problem, severity)`.
- `silent-failure-hunter` sweep over the per-team render/data paths to catch
  *quiet* wrong/empty data that doesn't raise.
- Adversarially verify each finding is real (not a harness artifact). Output: a
  prioritized findings list (launch-blocker / high / medium / low).

### Phase 3 — Browser walkthrough (Claude drives)
- `Playwright` against the live app (and local for repro). A single test
  identity is pointed at each team in turn (reassign-team mechanism — avoids
  needing 12 real logins), then Claude walks each page + tab and captures
  screenshots, flagging visual/layout breakage, mobile issues (390px), and
  odd-looking values. Desktop + mobile viewports.
- Output: a screenshot report (per page, representative teams + any flagged
  team) for owner review.

### Phase 4 — Fix + re-verify
- Triage Phases 2 + 3 findings. **Default: catalog-then-fix in prioritized
  batches** (launch-blockers first) — cleaner than piecemeal across 12 teams;
  owner may switch to fix-as-we-go.
- Per fix: `systematic-debugging` (root cause first) → `test-driven-development`
  (failing test → fix → green) → code review (`pr-review-toolkit:code-reviewer`
  + `coderabbit:code-review`) → re-run the per-team suite + re-walk affected
  pages. Deploys to master are confirmed with the owner per-action.

### Phase 5 — Launch sign-off
- `verification-before-completion` gate: per-team suite green, browser pass
  clean, all findings closed, CLAUDE.md guards added. Produce a final
  launch-readiness report. Only then proceed to the league-invite onboarding
  (tracked in the `live-onboarding-state` memory).

## Per-team mechanism (how Claude "becomes" each team)
- **Programmatic:** `AppTest` simulates a `MULTI_USER` session per team by
  setting `st.session_state` (`auth_user` with the team + role). No login or
  network needed.
- **Browser:** one test identity (admin, or a throwaway member account)
  reassigned to each team via the Admin Console (or a direct local session set),
  then walked. Avoids standing up 12 real accounts.

## Tooling (skills / plugins / agents)

**Orchestration (breadth):**
- **`Workflow`** — runs the parallel fleet (Phase 1 build, Phase 2 run/triage).
- **`superpowers:dispatching-parallel-agents`** — discipline for the fan-out.

**Depth + assurance (the differentiators):**
- **`pr-review-toolkit:pr-test-analyzer`** — are the generated per-team tests
  meaningful (catching, not just passing)?
- **`pr-review-toolkit:silent-failure-hunter`** — per-team failures that don't
  crash but show wrong/empty data (the recurring bug class).
- **`pr-review-toolkit:code-reviewer`** + **`coderabbit:code-review`** —
  independent review of every fix.
- **`pr-review-toolkit:type-design-analyzer`** *(as-needed)* — for any new types
  introduced by fixes.

**Real-browser truth:**
- **Playwright plugin** — drive the live/local app per team; screenshots; DOM
  inspection (found + verified the mobile fix this session).
- **`Claude_Preview`** — local server for fast fix-iteration/repro.

**Systematic data audit:**
- **`data:validate-data` / `data:explore-data`** — per-team data
  anomaly/plausibility checks instead of eyeballing.

**Diagnosis + correct fixes:**
- **`superpowers:systematic-debugging`** — root cause before any fix.
- **`superpowers:test-driven-development`** — failing test → fix → green.
- **`feature-dev:code-explorer`** *(as-needed)* — trace per-team data flow for
  tricky failures.
- **`Explore` agent** *(as-needed)* — broad codebase searches during diagnosis.
- **`verify` skill** — confirm a fix actually works in the running app.

**Process + honesty gates:**
- **`superpowers:writing-plans`** — turns this spec into the step-by-step plan
  (immediate next step).
- **`superpowers:executing-plans`** / **`subagent-driven-development`** — execute
  the plan with review checkpoints.
- **`superpowers:requesting-code-review`** / **`receiving-code-review`** — review
  discipline on the fix branches.
- **`superpowers:verification-before-completion`** — the gate that blocks any
  "ready" claim without evidence (the direct fix for the earlier premature call).
- **`claude-md-management`** — lock each new guard into CLAUDE.md so fixes can't
  silently regress.
- **`claude-mem:mem-search`** *(as-needed)* — recall this session's findings in
  future sessions.

## Risks / notes
- **Local data representativeness** — Phase 0 is a hard gate; thin data → hollow
  tests.
- **AppTest limits** — some complex page flows may need direct-function tests as
  a fallback to full-page `AppTest`.
- **Workflow rate limits** — batch ≤ 4 concurrent agents (learned 2026-06-01:
  a 16-wide burst was throttled).
- **True-mobile vs viewport-resize** — resizing a desktop browser does NOT
  trigger Streamlit's real mobile collapse; reproduce by loading narrow +
  logging in, then inspect element ancestors (learned this session).
- **Deploys** — fixes pushed to master auto-deploy on Railway; confirm per-action.
  This spec is committed locally (not pushed) so it does not trigger a redeploy.

## Labor split
- **Claude:** all technical work — Phase 0 data, build/review the test fleet, run
  + triage, drive the browser walkthrough, diagnose + fix + review, prepare
  deploys, produce reports.
- **Owner:** review findings reports + screenshots, spot-check, approve
  fixes/deploys, give the final go for launch.

## Sequencing
Phase 0 → 1 → 2 run before any fixing. Phase 3 (browser) can overlap Phase 2.
Phase 4 fixes in prioritized batches, re-verifying after each. Phase 5 signs off.
Front-load coverage on the highest-use pages (My Team, Lineup Optimizer, Matchup
Planner, Trade Analyzer, Free Agents, League Standings) while still covering all.

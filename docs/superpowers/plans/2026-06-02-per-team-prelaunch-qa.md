# Per-Team Pre-Launch QA — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove every page + tab + feature of HEATER loads, works, and shows sensible data for all 12 teams, then fix everything found, before inviting the league.

**Architecture:** A reusable AppTest harness runs any page headlessly *as a chosen team/role* (seeded session against real local QA users). A single parametrized smoke suite gives broad `(team × page × role)` coverage; an agent fleet adds page-specific deep assertions. Failures are triaged (incl. a silent-failure sweep), a Playwright browser pass catches what code can't, then findings are fixed in prioritized batches and re-verified. Discovery phases (run/triage/browser/fix) are procedures — fix code is written per-finding during execution, not pre-specified.

**Tech Stack:** Python 3.12/3.14, pytest, `streamlit.testing.v1.AppTest`, Playwright (MCP), Workflow (multi-agent), the local SQLite DB. Spec: `docs/superpowers/specs/2026-06-02-per-team-prelaunch-qa-design.md`.

**Hard rules carried from the spec:** local-only QA users/data (never deployed); deploys to master confirmed per-action; batch fleet agents ≤ 4 concurrent; `verification-before-completion` before any "ready" claim.

---

## File Structure

- `tests/qa/_harness.py` — the per-team AppTest harness (`run_page_as_team`, `PAGES`, `harness sanity helpers`). One responsibility: run a page as a team and report what broke.
- `tests/qa/conftest.py` — fixtures: the 12 team names, the QA-users gate, env setup.
- `tests/qa/test_per_team_smoke.py` — broad parametrized suite `(team × page × role)`: no exception, no error element, nav role-safe.
- `tests/qa/test_page_<stem>.py` — page-specific deep assertions (one per page; fleet-authored from a template).
- `scripts/qa_seed_local.py` — idempotent: verify 12-team data + create 12 local QA users (one per team) + 1 QA admin. Local DB only.
- `docs/superpowers/plans/qa-findings.md` — living findings log (created in Phase 2; `(team,page,tab,problem,severity,status)`).

---

## Task 0: Local data + QA users gate (Phase 0)

**Files:**
- Create: `scripts/qa_seed_local.py`
- Test: `tests/qa/test_qa_seed.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/qa/test_qa_seed.py
def test_local_has_twelve_teams_and_qa_users():
    from src.database import load_league_rosters, get_connection
    from src.auth import get_user
    rosters = load_league_rosters()
    teams = sorted(t for t in rosters["team_name"].dropna().unique())
    assert len(teams) >= 12, f"local DB has only {len(teams)} teams: {teams}"
    # one active QA user per team
    for t in teams:
        u = get_user(f"qa_{_slug(t)}")
        assert u is not None and u["status"] == "active" and u["team_name"] == t

def _slug(name: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in name)
```

- [ ] **Step 2: Run it — expect FAIL** (`qa_seed_local` not run yet / users missing)

Run: `.venv\Scripts\python.exe -m pytest tests/qa/test_qa_seed.py -q`
Expected: FAIL (fewer than 12 teams, or QA users absent).

- [ ] **Step 3: Implement `scripts/qa_seed_local.py`** — verify 12-team data; if `< 12` teams, reconnect Yahoo (`try_reconnect_yahoo()`) and run `bootstrap_all_data(yahoo_client=..., force=False)` to populate league data; then create one active QA user per team (`create_user` + `approve_user(team_name=t)`), plus `qa_admin` (is_admin). Idempotent (skip existing). Print a summary. (QA users are LOCAL-ONLY — never on the Railway volume.)

- [ ] **Step 4: Run the seeder, then the test — expect PASS**

Run: `.venv\Scripts\python.exe scripts\qa_seed_local.py` then `.venv\Scripts\python.exe -m pytest tests/qa/test_qa_seed.py -q`
Expected: PASS (12 teams + QA users present).

- [ ] **Step 5: Commit** (local) — `git add scripts/qa_seed_local.py tests/qa/test_qa_seed.py && git commit -m "test(qa): local 12-team data + QA users gate"`

---

## Task 1: Per-team AppTest harness (Phase 1 foundation)

**Files:**
- Create: `tests/qa/_harness.py`, `tests/qa/conftest.py`
- Test: `tests/qa/test_harness_smoke.py`

> **Key risk this task resolves:** `require_auth()` re-validates the session user against the DB. The harness therefore seeds a session for a REAL local QA user (from Task 0), so `require_auth` passes and the page renders as that team. This step PROVES the harness works on one page before scaling.

- [ ] **Step 1: Write the failing test**

```python
# tests/qa/test_harness_smoke.py
from tests.qa._harness import run_page_as_team

def test_my_team_runs_for_a_member():
    r = run_page_as_team("pages/1_My_Team.py", team_name="Team Hickey", is_admin=True)
    assert r.ran, f"page raised before render: {r.exception}"
    assert r.exception is None, f"page exception: {r.exception}"
    assert not r.errors, f"st.error on page: {r.errors}"
```

- [ ] **Step 2: Run — expect FAIL** (`_harness` missing). `.venv\Scripts\python.exe -m pytest tests/qa/test_harness_smoke.py -q`

- [ ] **Step 3: Implement `tests/qa/_harness.py`**

```python
"""Run any HEATER page headlessly as a chosen team/role and report what broke."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from streamlit.testing.v1 import AppTest
from src.auth import get_user

@dataclass
class HarnessResult:
    page: str; team: str; is_admin: bool; ran: bool
    exception: str | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

def _slug(name: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in name)

def run_page_as_team(page_path: str, team_name: str, is_admin: bool = False,
                     timeout: float = 60.0) -> HarnessResult:
    os.environ["MULTI_USER"] = "1"
    user = get_user(f"qa_{_slug(team_name)}") if not is_admin else get_user("qa_admin")
    at = AppTest.from_file(page_path, default_timeout=timeout)
    # Seed a logged-in session for a REAL QA user so require_auth() passes.
    at.session_state["auth_user"] = dict(user)
    at.session_state["_auth_bootstrap_done"] = True
    try:
        at.run()
    except Exception as e:
        return HarnessResult(page_path, team_name, is_admin, ran=False,
                             exception=f"{type(e).__name__}: {e}")
    return HarnessResult(
        page=page_path, team=team_name, is_admin=is_admin, ran=True,
        exception=str(at.exception[0].value) if at.exception else None,
        errors=[e.value for e in at.error],
        warnings=[w.value for w in at.warning],
    )
```

(If `AppTest` interaction with a given page proves intractable, fall back to importing the page's data/logic functions directly for that page and assert on their return values — note it in `qa-findings.md`.)

- [ ] **Step 4: Run — expect PASS.** If it fails because the QA admin user isn't seeded, run Task 0's seeder first. Iterate the seeding/session details until My Team renders cleanly as a team.

- [ ] **Step 5: Add `tests/qa/conftest.py`** exposing a `team_names()` fixture (from `load_league_rosters`) + a session-scoped check that Task 0's gate passed.

- [ ] **Step 6: Commit** (local) — `git commit -m "test(qa): per-team AppTest harness + one-page validation"`

---

## Task 2: Broad per-team smoke suite (Phase 1)

**Files:** Create `tests/qa/test_per_team_smoke.py`

- [ ] **Step 1: Write the test** — parametrize over every page × every team (member role) + admin pages (admin role). Assert `r.ran`, `r.exception is None`, `not r.errors`. (Member role-safety — no admin nav links — is already guarded by `tests/test_nav.py` + verified visually in Phase 3, so this smoke suite focuses on load / exception / error / blank.)

```python
import pytest
from tests.qa._harness import run_page_as_team
from src.nav import PAGE_REGISTRY

MEMBER_PAGES = [f"pages/{e['key']}.py" for e in PAGE_REGISTRY]

@pytest.mark.parametrize("page", MEMBER_PAGES)
def test_page_loads_for_every_team(page, team_names):
    failures = []
    for t in team_names:
        r = run_page_as_team(page, t, is_admin=False)
        if not r.ran or r.exception or r.errors:
            failures.append((t, r.exception or r.errors))
    assert not failures, f"{page} broke for: {failures}"
```

- [ ] **Step 2: Run — collect failures** (this run IS the start of Phase 2). `.venv\Scripts\python.exe -m pytest tests/qa/test_per_team_smoke.py -q`. Record every failure in `qa-findings.md` (do NOT fix yet — catalog first).

- [ ] **Step 3: Commit** the suite (local) — `git commit -m "test(qa): broad per-team page smoke suite"`

---

## Task 3: Fleet — page-specific deep assertions (Phase 1)

**Files:** Create `tests/qa/test_page_<stem>.py` (one per page, fleet-authored); a Workflow script (inline).

- [ ] **Step 1: Author a `Workflow`** that fans out ≤4-concurrent, ~one agent per page. Each agent: reads its page, lists the key outputs it shows (e.g., roster size, category totals, lineup slot counts, win-prob %, SGP), and writes `tests/qa/test_page_<stem>.py` using `run_page_as_team` + assertions on plausible ranges for each team (e.g., 0 ≤ AVG ≤ 1; ERA ≥ 0; lineup fills exactly the league slot counts; no `NaN`/`None` in displayed metrics). Agents follow a STRICT template block (provided in the workflow prompt) so the modules are uniform. Schema-validated output: `{module_path, summary, assertions[]}`.

- [ ] **Step 2: Review every generated module** for quality; run `pr-review-toolkit:pr-test-analyzer` over `tests/qa/` to confirm the assertions are meaningful (not vacuous). Fix weak tests.

- [ ] **Step 3: Run the full `tests/qa/` suite — catalog failures** into `qa-findings.md`. `.venv\Scripts\python.exe -m pytest tests/qa/ -q`

- [ ] **Step 4: Commit** (local) — `git commit -m "test(qa): page-specific per-team deep assertions (fleet)"`

---

## Task 4: Run + triage (Phase 2) — procedure

- [ ] Run the complete `tests/qa/` suite; ensure every failure is logged in `qa-findings.md` as `(team, page, tab, problem, severity)`.
- [ ] Dispatch `pr-review-toolkit:silent-failure-hunter` over the per-team render/data paths the suite exercised, to catch wrong/empty data that does NOT raise. Add findings.
- [ ] For each finding, adversarially confirm it's real (re-run in isolation / read the code) — discard harness artifacts. Assign severity: launch_blocker / high / medium / low.
- [ ] Produce the prioritized findings list. **No fixes yet.**

---

## Task 5: Browser walkthrough (Phase 3) — procedure

- [ ] Bring up the app for browsing (live deploy for realism; local `streamlit-app` preview for repro). Per team, point a test identity at the team (reassign via Admin Console, or set the local session) — do NOT need 12 real logins.
- [ ] With Playwright at desktop (1280) and mobile (390) viewports, walk each page + tab. Capture a screenshot per page; flag layout/overflow/mobile-nav/odd-value issues. Front-load My Team, Lineup Optimizer, Matchup Planner, Trade Analyzer, Free Agents, League Standings.
- [ ] Append visual findings to `qa-findings.md`; deliver a screenshot report for owner review.

---

## Task 6: Fix + re-verify (Phase 4) — finding-driven loop

> Fix code is written per-finding here; it cannot be pre-specified. For EACH finding, in priority order (launch-blockers first), run this loop:

- [ ] **Diagnose** with `superpowers:systematic-debugging` — root cause before any change (use `feature-dev:code-explorer` / `Explore` for tricky data-flow traces).
- [ ] **Fix test-first** with `superpowers:test-driven-development` — add a failing test (prefer a `tests/qa/` per-team assertion or a structural guard), watch it fail, implement the minimal fix, watch it pass.
- [ ] **Review** the change with `pr-review-toolkit:code-reviewer` + `coderabbit:code-review`; run `silent-failure-hunter` if the fix touches error handling.
- [ ] **Re-verify:** re-run `tests/qa/` + the full suite (`pytest --ignore=tests/test_cheat_sheet.py -n auto`) + re-walk the affected page(s) in the browser.
- [ ] **Lock the guard** into CLAUDE.md's structural-invariant table (`claude-md-management`) so it can't regress.
- [ ] **Deploy** in prioritized batches: commit → **confirm push with owner** → push to master → Railway redeploys → re-verify on live. Update `qa-findings.md` status → fixed.

---

## Task 7: Launch-readiness sign-off (Phase 5)

- [ ] Run the full `tests/qa/` per-team suite — **all green for all 12 teams**. Capture the summary.
- [ ] Confirm the browser walkthrough is clean (no open visual/mobile findings).
- [ ] Confirm `qa-findings.md` has zero open launch-blocker/high items.
- [ ] Apply `superpowers:verification-before-completion`: paste the actual passing output + the closed-findings count BEFORE claiming ready. No assertion without evidence.
- [ ] Write the final launch-readiness report; update the `live-onboarding-state` memory: QA complete → proceed to league invite.
- [ ] Decide whether `tests/qa/` (incl. the QA-users seeder, local-only) is kept as a permanent regression suite or trimmed; note it in CLAUDE.md.

---

## Notes on the QA-users / data being local-only
The QA users and any synced data created by `scripts/qa_seed_local.py` live ONLY in the local `data/draft_tool.db`. They are never pushed and never reach the Railway volume. The deployed app's users remain exactly the real admin + whatever leaguemates register.

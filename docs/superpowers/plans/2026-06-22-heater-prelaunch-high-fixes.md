# Pre-launch HIGH-severity Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the three HIGH-severity findings from the 2026-06-22 audit (`Outstanding_June26.md`) so the M3 beta is safe to onboard the 12 leaguemates.

**Architecture:** Backend (`api/tenancy.py` + 8 thin routers + `matchup_service.py`) is TDD'd with pytest. The frontend slice (`web/`) stops swallowing live errors and adds a "team not linked" state; it is verified with `pnpm exec tsc --noEmit` + Claude_Preview (web has no test runner / CI). Every change keeps the Clerk-off / demo-build paths byte-for-byte identical.

**Tech Stack:** Python 3.12 / FastAPI / pytest (backend); Next.js 16 / React 19 / TypeScript / Tailwind (frontend).

**Spec:** `docs/superpowers/specs/2026-06-22-heater-prelaunch-high-fixes-design.md`

**Endpoint classification (used throughout):**
- **Team-REQUIRED** (meaningless without the viewer's team → 409 `team_not_linked` for an authed-unassigned user): `/api/me/team`, `/api/matchup`, `/api/lineup/optimize`, `/api/punt`, `/api/trade/evaluate`, `/api/trade-finder`, `/api/free-agents`, `/api/free-agents/pool`.
- **Team-OPTIONAL** (league-wide; the team only adds a "yours"/"you" marker → degrade gracefully, never 409): `/api/schedule/probables`, `/api/schedule/hitter-matchups`, `/api/playoff-odds`. (These keep calling `ctx.effective_team(...)`; a `None` team just drops the marker.)

**Conventions:**
- Run backend tests from the repo root: `C:\Users\conno\Code\HEATER_v1.0.1`.
- This is Windows; use the Bash tool (Git Bash) for `python -m pytest ...` and `cd web && pnpm ...`.
- We do NOT declare the 409 in the routers' `responses=` — that would force an `api/openapi.json` regen, and the local interpreter is on `fastapi 0.126.0` (not the pinned `0.137.1`), so a local regen would diverge from CI. The 409 is a runtime signal the frontend reads by status code. `api/openapi.json` stays untouched by this plan.

---

## Task 1: `ViewerContext.effective_team` — three-state resolution

Closes the core of HIGH-1: an authenticated-but-unassigned viewer must NEVER fall back to the client `team_name`.

**Files:**
- Modify: `api/tenancy.py:60-63`
- Test: `tests/api/test_api_tenancy_resolver.py` (modify existing assertions + add)

- [ ] **Step 1: Update the failing tests first (TDD).** In `tests/api/test_api_tenancy_resolver.py`, replace the body of `test_viewer_context_effective_team_falls_back_when_unresolved` (lines 50-52) with the new three-state expectation:

```python
def test_viewer_context_effective_team_dormant_falls_back():
    # Dormant (no identity, user_id None) → fall back to the query param.
    assert ViewerContext(user_id=None, league_id=None, team_name=None).effective_team("Team Hickey") == "Team Hickey"


def test_viewer_context_effective_team_authed_unassigned_returns_none():
    # Authenticated (user_id set) but no assignment → None, NEVER the fallback.
    assert ViewerContext(user_id=1, league_id=1, team_name=None).effective_team("Team Hickey") is None


def test_viewer_context_effective_team_prefers_resolved_over_fallback():
    assert ViewerContext(user_id=1, league_id=1, team_name="Mine").effective_team("Team Hickey") == "Mine"
```

Also update `test_resolver_clerk_user_without_assignment_resolves_none` (lines 95-106): change the final assertion + comment so the authed-unassigned probe gets `team` = None (the probe calls `ctx.effective_team` directly):

```python
def test_resolver_clerk_user_without_assignment_resolves_none():
    # Logged-in but unassigned → team_name None AND effective_team returns None
    # (never another user's team; the team-required routers turn this into 409).
    app = _resolver_app()
    users = InMemoryUserStore()
    app.dependency_overrides[get_auth_verifier] = lambda: _ClerkVerifier()
    app.dependency_overrides[get_user_store] = lambda: users
    app.dependency_overrides[get_league_store] = lambda: InMemoryLeagueStore()
    app.dependency_overrides[get_membership_store] = lambda: InMemoryMembershipStore()
    body = TestClient(app).get("/probe?team_name=Fallback", headers={"Authorization": "Bearer x"}).json()
    assert body["resolved"] is None
    assert body["team"] is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/api/test_api_tenancy_resolver.py -q`
Expected: FAIL — `test_viewer_context_effective_team_authed_unassigned_returns_none` and the updated resolver test fail (current `effective_team` returns the fallback, not None).

- [ ] **Step 3: Implement the three-state `effective_team`.** In `api/tenancy.py`, replace lines 60-63:

```python
    def effective_team(self, fallback: str | None) -> str | None:
        """The viewer's team for an endpoint, in three states:
        - assigned        → the resolved team (ignores the client fallback);
        - authed+unassigned (user_id set, no team) → None (NEVER the fallback —
          closes the cross-team exposure; the team-required routers map this to 409);
        - dormant (no identity) → the endpoint's query-param fallback (today's
          open behavior, byte-for-byte)."""
        if self.team_name:
            return self.team_name
        if self.user_id is not None:
            return None
        return fallback
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/api/test_api_tenancy_resolver.py -q`
Expected: PASS (all tests, including the unchanged assigned/dormant/activation-flip ones).

- [ ] **Step 5: Commit**

```bash
git add api/tenancy.py tests/api/test_api_tenancy_resolver.py
git commit -m "fix(api): effective_team returns None for authed-unassigned viewer (HIGH-1)"
```

---

## Task 2: `resolve_required_team` helper + `TEAM_NOT_LINKED`

Adds the centralized 409 signal that team-required routers use.

**Files:**
- Modify: `api/tenancy.py` (add the constant + function near the top, after imports)
- Test: `tests/api/test_api_tenancy_resolver.py` (add)

- [ ] **Step 1: Write the failing tests.** Append to `tests/api/test_api_tenancy_resolver.py`:

```python
import pytest
from fastapi import HTTPException
from api.tenancy import TEAM_NOT_LINKED, resolve_required_team


def test_resolve_required_team_raises_409_for_authed_unassigned():
    with pytest.raises(HTTPException) as ei:
        resolve_required_team(ViewerContext(user_id=1, league_id=1, team_name=None), "Team Hickey")
    assert ei.value.status_code == 409
    assert ei.value.detail == TEAM_NOT_LINKED


def test_resolve_required_team_returns_assigned_team():
    assert resolve_required_team(ViewerContext(user_id=1, team_name="Mine"), "x") == "Mine"


def test_resolve_required_team_dormant_returns_fallback():
    # Clerk off (no identity) → today's behavior preserved (incl. empty string).
    assert resolve_required_team(ViewerContext(), "Team Hickey") == "Team Hickey"
    assert resolve_required_team(ViewerContext(), "") == ""
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/api/test_api_tenancy_resolver.py -k resolve_required_team -q`
Expected: FAIL with `ImportError: cannot import name 'TEAM_NOT_LINKED'` / `resolve_required_team`.

- [ ] **Step 3: Implement.** In `api/tenancy.py`, after the imports (around line 23, before `def normalize_team_name`), add:

```python
TEAM_NOT_LINKED = "team_not_linked"
```

And after the `ViewerContext` class (after line 92), add:

```python
def resolve_required_team(ctx: "ViewerContext", fallback: str | None) -> str:
    """Resolve the viewer's team for a TEAM-REQUIRED endpoint.

    Raises HTTP 409 (detail=TEAM_NOT_LINKED) when an authenticated user has no
    team assignment — so the frontend shows a friendly 'not linked yet' state
    instead of another team's data. Dormant (Clerk off) and assigned paths return
    the team string unchanged (today's behavior). effective_team returns None ONLY
    for the authed-unassigned case, so `is None` is the precise trigger."""
    team = ctx.effective_team(fallback)
    if team is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=TEAM_NOT_LINKED)
    return team
```

(`HTTPException` and `status` are already imported at `api/tenancy.py:15`.)

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/api/test_api_tenancy_resolver.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add api/tenancy.py tests/api/test_api_tenancy_resolver.py
git commit -m "feat(api): resolve_required_team helper (409 team_not_linked) (HIGH-1)"
```

---

## Task 3: Wire the team-required routers to `resolve_required_team`

**Files (modify the single `return` line in each):**
- `api/routers/team.py:21`
- `api/routers/matchup.py:20`
- `api/routers/free_agents.py:21` and `:31`
- `api/routers/lineup.py:33`
- `api/routers/punt.py:20`
- `api/routers/trade.py:32`
- `api/routers/trade_finder.py:33`
- Test: `tests/api/test_api_team_not_linked.py` (create)

- [ ] **Step 1: Write the failing integration test.** Create `tests/api/test_api_team_not_linked.py`:

```python
"""A team-required router returns 409 team_not_linked for an authed-unassigned
viewer (HIGH-1), and 200 for an assigned one. Uses a tiny app whose route mirrors
the real routers' `resolve_required_team(ctx, team_name)` call."""

from fastapi import Depends, FastAPI
from starlette.testclient import TestClient

from api.auth import Principal, get_auth_verifier
from api.deps import get_league_store, get_membership_store, get_user_store
from api.stores.league_store import InMemoryLeagueStore
from api.stores.membership_store import InMemoryMembershipStore
from api.stores.user_store import InMemoryUserStore
from api.tenancy import ViewerContext, require_viewer_context, resolve_required_team


class _Clerk:
    def verify(self, authorization):
        return Principal(subject="user_42", clerk_user_id="user_42")


def _app():
    app = FastAPI()

    @app.get("/needs-team")
    def needs_team(team_name: str = "", ctx: ViewerContext = Depends(require_viewer_context)):
        return {"team": resolve_required_team(ctx, team_name)}

    return app


def test_authed_unassigned_gets_409():
    app = _app()
    app.dependency_overrides[get_auth_verifier] = lambda: _Clerk()
    app.dependency_overrides[get_user_store] = lambda: InMemoryUserStore()
    app.dependency_overrides[get_league_store] = lambda: InMemoryLeagueStore()
    app.dependency_overrides[get_membership_store] = lambda: InMemoryMembershipStore()
    r = TestClient(app).get("/needs-team?team_name=Team%20Hickey", headers={"Authorization": "Bearer x"})
    assert r.status_code == 409
    assert r.json()["detail"] == "team_not_linked"


def test_authed_assigned_gets_their_team():
    app = _app()
    users, leagues, members = InMemoryUserStore(), InMemoryLeagueStore(), InMemoryMembershipStore()
    u = users.get_or_create("user_42")
    lg = leagues.get_or_create_default()
    members.assign(u.id, lg.id, "Bronx Bombers", team_key=None, assigned_by=u.id)
    app.dependency_overrides[get_auth_verifier] = lambda: _Clerk()
    app.dependency_overrides[get_user_store] = lambda: users
    app.dependency_overrides[get_league_store] = lambda: leagues
    app.dependency_overrides[get_membership_store] = lambda: members
    r = TestClient(app).get("/needs-team?team_name=Team%20Hickey", headers={"Authorization": "Bearer x"})
    assert r.status_code == 200
    assert r.json()["team"] == "Bronx Bombers"


def test_dormant_passes_param_through():
    app = _app()
    app.dependency_overrides[get_user_store] = lambda: InMemoryUserStore()
    app.dependency_overrides[get_league_store] = lambda: InMemoryLeagueStore()
    app.dependency_overrides[get_membership_store] = lambda: InMemoryMembershipStore()
    r = TestClient(app).get("/needs-team?team_name=Team%20Hickey")
    assert r.status_code == 200
    assert r.json()["team"] == "Team Hickey"
```

- [ ] **Step 2: Run to verify it passes already** (this test exercises `resolve_required_team`, which exists after Task 2 — so it should PASS now; it guards the router wiring you do next).

Run: `python -m pytest tests/api/test_api_team_not_linked.py -q`
Expected: PASS (3 tests). This is the contract the routers must honor.

- [ ] **Step 3: Wire each team-required router.** Change `ctx.effective_team(...)` → `resolve_required_team(ctx, ...)` and add the import. For each file, update the import line `from api.tenancy import ViewerContext, require_viewer_context` → `from api.tenancy import ViewerContext, require_viewer_context, resolve_required_team`, then:

`api/routers/team.py:21`:
```python
    return service.get_my_team(resolve_required_team(ctx, team_name))
```
`api/routers/matchup.py:20`:
```python
    return service.get_matchup(resolve_required_team(ctx, team_name))
```
`api/routers/free_agents.py:21`:
```python
    return service.get_free_agents(resolve_required_team(ctx, team_name), limit)
```
`api/routers/free_agents.py:31`:
```python
    return service.get_free_agents_pool(resolve_required_team(ctx, team_name), limit)
```
`api/routers/lineup.py:33`:
```python
    return service.optimize(resolve_required_team(ctx, req.team_name), req.date, req.scope, req.mode)
```
`api/routers/punt.py:20`:
```python
    return service.get_punt(resolve_required_team(ctx, team_name))
```
`api/routers/trade.py:32`:
```python
    return service.evaluate(resolve_required_team(ctx, req.team_name), req.giving_ids, req.receiving_ids, req.enable_mc)
```
`api/routers/trade_finder.py:33`:
```python
    return service.get_suggestions(team_name=resolve_required_team(ctx, team_name), limit=limit)
```

Do NOT touch `api/routers/schedule.py` or `api/routers/playoff.py` (team-optional — they keep `ctx.effective_team(...)`).

- [ ] **Step 4: Verify routers stay logic-free + api suite still green**

Run: `python -m pytest tests/api/test_no_logic_in_routers.py tests/api/test_api_team_not_linked.py -q`
Expected: PASS (the helper call is not "logic" — no `src.` import, no arithmetic).

Then a broad api regression (the 1 known openapi failure is the unrelated local-pin artifact — see plan header):
Run: `python -m pytest tests/api/ -q`
Expected: same as baseline — `416 passed, 1 failed` (only `test_openapi_snapshot_is_current`, the env-pin artifact; nothing else newly red).

- [ ] **Step 5: Commit**

```bash
git add api/routers/ tests/api/test_api_team_not_linked.py
git commit -m "fix(api): team-required routers 409 on authed-unassigned viewer (HIGH-1)"
```

---

## Task 4: Matchup pitcher misclassification (HIGH-2)

Extract the classifier to a testable module-level function; on a pool miss, classify by **eligible positions** (not the assigned slot) and log the miss.

**Files:**
- Modify: `api/services/matchup_service.py` (add a module logger + module-level `_PITCHER_SLOTS` and `_is_pitcher`; rewire the nested call at `:550`; remove the nested `_PITCHER_SLOTS` at `:502` and `_is_pitcher_by_pool` at `:504-520`)
- Test: `tests/api/test_api_matchup_classify.py` (create)

- [ ] **Step 1: Write the failing test.** Create `tests/api/test_api_matchup_classify.py`:

```python
"""HIGH-2: pool is_hitter is authoritative; on a pool miss, classify by ELIGIBLE
positions (SP/RP/P → pitcher), NOT the assigned slot (a benched pitcher's slot is
'BN'/'IL'). Default unknown → hitter, with a debug log."""

import logging
import pandas as pd

from api.services.matchup_service import _is_pitcher


def _pool(rows):
    return pd.DataFrame(rows)


def test_pool_hit_pitcher_flag_wins():
    pool = _pool([{"player_id": 5, "is_hitter": False}])
    assert _is_pitcher(5, "BN", pool) is True  # pool says pitcher even though slot is BN


def test_pool_hit_hitter_flag_wins():
    pool = _pool([{"player_id": 6, "is_hitter": True}])
    assert _is_pitcher(6, "SP,RP", pool) is False  # pool says hitter; slot ignored


def test_pool_miss_eligible_sp_is_pitcher():
    pool = _pool([{"player_id": 6, "is_hitter": True}])  # 99 absent
    assert _is_pitcher(99, "SP,RP", pool) is True


def test_pool_miss_benched_pitcher_classified_by_eligibility():
    # selected_position would be 'BN' but eligible carries 'SP' → pitcher.
    assert _is_pitcher(99, "SP", None) is True


def test_pool_miss_eligible_hitter_is_hitter():
    assert _is_pitcher(99, "2B,SS", None) is False


def test_pool_miss_unknown_defaults_hitter_and_logs(caplog):
    with caplog.at_level(logging.DEBUG, logger="api.services.matchup_service"):
        assert _is_pitcher(99, "", None) is False
    assert any("unresolved" in r.message for r in caplog.records)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/api/test_api_matchup_classify.py -q`
Expected: FAIL with `ImportError: cannot import name '_is_pitcher'`.

- [ ] **Step 3: Implement.** In `api/services/matchup_service.py`:

(a) Ensure a module logger exists. Near the top of the file, after the imports, add (if not already present):

```python
import logging

logger = logging.getLogger(__name__)
```

(b) Add the module-level constant + classifier (above the `MatchupService` class):

```python
_PITCHER_SLOTS = frozenset({"SP", "RP", "P"})


def _is_pitcher(pid, eligible: str, side_pool) -> bool:
    """Classify hitter vs pitcher. The pool's ``is_hitter`` flag is authoritative
    when present; on a pool miss (or is_hitter None) fall back to the player's
    ELIGIBLE positions (SP/RP/P → pitcher). Use eligible positions, NOT the
    assigned slot — a benched/IL pitcher's slot is 'BN'/'IL'. Unknown → hitter
    (logged), the conservative default."""
    if side_pool is not None and not side_pool.empty:
        try:
            pmatch = side_pool[side_pool["player_id"] == pid]
            if not pmatch.empty:
                flag = pmatch.iloc[0].get("is_hitter")
                if flag is not None:
                    return not bool(flag)
        except Exception:
            pass
    toks = {t.strip().upper() for t in str(eligible or "").replace("/", ",").split(",") if t.strip()}
    if toks & _PITCHER_SLOTS:
        return True
    logger.debug("matchup: unresolved player_id=%s (eligible=%r) → defaulting to hitter", pid, eligible)
    return False
```

(c) In `_build_side` (the nested function), compute the eligible-positions string and call the new classifier. Replace the classify line. The loop currently does:
```python
                # Classify using pool is_hitter flag (handles BN/IL/SP,RP swingmen).
                is_pit = _is_pitcher_by_pool(pid, sel, side_pool)
```
with:
```python
                # Classify by pool is_hitter; on a pool miss, by ELIGIBLE positions
                # (roster_slot), never the assigned slot `sel` (which may be BN/IL).
                eligible = str(row.get("roster_slot") or "").strip() or sel
                is_pit = _is_pitcher(pid, eligible, side_pool)
```

(d) Delete the now-unused nested definitions: the `_PITCHER_SLOTS = frozenset(...)` at the old line 502 and the entire nested `def _is_pitcher_by_pool(...)` (old lines 504-520).

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/api/test_api_matchup_classify.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Verify no other matchup tests broke**

Run: `python -m pytest tests/api/ -k matchup -q`
Expected: PASS (existing matchup tests still green).

- [ ] **Step 6: Commit**

```bash
git add api/services/matchup_service.py tests/api/test_api_matchup_classify.py
git commit -m "fix(api): matchup classifies pool-missing players by eligibility + logs (HIGH-2)"
```

---

## Task 5: Frontend — `isTeamNotLinked` + the `liveOrMock`/`withTimeout` helpers

Foundation for HIGH-3 + the HIGH-1 frontend state. No UI yet.

**Files:**
- Modify: `web/src/lib/api/errors.ts`
- Create: `web/src/lib/api/live.ts`

- [ ] **Step 1: Locate the existing `isLive`.** `matchup-data.ts` already imports `isLive`, so it exists.

Run: `cd web && grep -rn "isLive" src/lib | grep -E "export|function"`
Expected: shows where `isLive` is defined (e.g. `src/lib/api/client.ts` or `src/lib/env.ts`). Note that path — Step 3 imports `liveOrMock` consumers from `live.ts`, and `live.ts` re-exports the existing `isLive` so call sites have one import.

- [ ] **Step 2: Add `isTeamNotLinked` to `web/src/lib/api/errors.ts`** (after `isAuthRequired`):

```typescript
/** 409 — authenticated viewer has no team assignment yet (HIGH-1). The frontend
 *  shows a friendly "team not linked" state. We only emit 409 for this case. */
export function isTeamNotLinked(e: unknown): e is ApiError {
  return e instanceof ApiError && e.status === 409;
}
```

- [ ] **Step 3: Create `web/src/lib/api/live.ts`.** (Adjust the `isLive` import to the path found in Step 1.)

```typescript
/** Live-vs-mock helpers. Under HEATER_LIVE we run the real API call and let any
 *  ApiError / network error PROPAGATE (so usePageData reaches error/locked/unlinked
 *  or routes to sign-in) — fetchers must NOT swallow to mock when live (HIGH-3).
 *  Off-live (demo/marketing build) the mock is the intended behavior. */
import { isLive } from "./client"; // ← use the path from Step 1

export { isLive };

export async function liveOrMock<T>(
  live: () => Promise<T>,
  mock: () => T | Promise<T>,
): Promise<T> {
  if (!isLive()) return mock();
  return live(); // no catch — errors propagate to usePageData
}

/** Reject (not resolve-to-mock) when a live call exceeds `ms`, so a hung API
 *  surfaces as the error state instead of an infinite spinner. */
export function withTimeout<T>(p: Promise<T>, ms: number): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const t = setTimeout(() => reject(new Error(`timeout after ${ms}ms`)), ms);
    p.then(
      (v) => {
        clearTimeout(t);
        resolve(v);
      },
      (e) => {
        clearTimeout(t);
        reject(e);
      },
    );
  });
}
```

> If Step 1 finds `isLive` is NOT exported from `client.ts`, instead import it in `live.ts` from its real module and drop the `export { isLive }` re-export if it would duplicate; the goal is one canonical `isLive`.

- [ ] **Step 4: Type-check**

Run: `cd web && pnpm exec tsc --noEmit`
Expected: clean (exit 0).

- [ ] **Step 5: Commit**

```bash
git add web/src/lib/api/errors.ts web/src/lib/api/live.ts
git commit -m "feat(web): isTeamNotLinked + liveOrMock/withTimeout helpers (HIGH-3)"
```

---

## Task 6: `usePageData` — `unlinked` state, 401 sign-in, 409 mapping

**Files:**
- Modify: `web/src/lib/use-page-data.ts`

- [ ] **Step 1: Add the `unlinked` state to the `PageState` union** (after `locked`):

```typescript
export type PageState<T> =
  | { status: "loading" }
  | { status: "error" }
  | { status: "empty" }
  | { status: "locked" }
  | { status: "unlinked" }
  | { status: "loaded"; data: T };
```

- [ ] **Step 2: Update the import + the `.catch` handler.** Change the import at the top:

```typescript
import { isPaywall, isAuthRequired, isTeamNotLinked } from "@/lib/api/errors";
```

Replace the `.catch((e) => { ... })` block (lines 73-76) with:

```typescript
      .catch((e) => {
        if (!alive) return;
        // 401 (signed out, gate live) → route to Clerk sign-in. retry() won't help.
        if (isAuthRequired(e)) {
          if (typeof window !== "undefined") window.location.assign("/sign-in");
          return;
        }
        // 409 → the viewer has no team assigned yet (HIGH-1).
        if (isTeamNotLinked(e)) {
          setState({ status: "unlinked" });
          return;
        }
        // 402 (Pro paywall) → locked; anything else → generic error.
        setState(isPaywall(e) ? { status: "locked" } : { status: "error" });
      });
```

- [ ] **Step 3: Type-check**

Run: `cd web && pnpm exec tsc --noEmit`
Expected: clean. (Pages that exhaustively switch on `state.status` may now warn about the new `unlinked` case — Task 7 handles those; if tsc errors on an exhaustive switch, proceed to Task 7 then re-run.)

- [ ] **Step 4: Commit**

```bash
git add web/src/lib/use-page-data.ts
git commit -m "feat(web): usePageData handles 401 (sign-in), 402 (locked), 409 (unlinked) (HIGH-1/3)"
```

---

## Task 7: `TeamNotLinked` component + wire into team-required pages

**Files:**
- Modify: `web/src/components/ui/PageStates.tsx` (add `PageNotLinked`)
- Modify the team-required pages to render the `unlinked` branch:
  - `web/src/app/page.tsx` (My Team)
  - `web/src/app/matchup/page.tsx`
  - `web/src/app/optimizer/page.tsx`
  - `web/src/app/punt/page.tsx`
  - `web/src/app/players/page.tsx`
  - `web/src/app/trades/page.tsx` (the finder tab that uses `usePageData`)

- [ ] **Step 1: Add `PageNotLinked` to `web/src/components/ui/PageStates.tsx`.** Match the existing `PageError`/`PageEmpty` style in that file (reuse its container + icon pattern; use the `Inbox`/`UserX`-style lucide icon already imported there, or import `UserX` from `lucide-react`):

```tsx
export function PageNotLinked() {
  return (
    <PageEmpty
      icon={UserX}
      title="Your team isn't linked yet"
      body="Your commissioner will assign your team shortly. League-wide views (Standings, Leaders, Players) work in the meantime."
    />
  );
}
```

(Add `UserX` to the `lucide-react` import in that file. If `PageEmpty`'s props differ, match its actual signature — open the file and follow the existing `PageError`/`PageEmpty` shape exactly.)

- [ ] **Step 2: Render the `unlinked` branch on each team-required page.** In each page listed above, find the block that renders `state.status === "error"` and add a sibling line. Example for `web/src/app/page.tsx` (after line 45):

```tsx
        {state.status === "unlinked" && <PageNotLinked />}
```

and add `PageNotLinked` to that page's import from `@/components/ui/PageStates`. Apply the same two edits (import + the `unlinked` line) to `matchup/page.tsx`, `optimizer/page.tsx`, `punt/page.tsx`, `players/page.tsx`, and the finder section of `trades/page.tsx`. (Pages that use a `switch` on `state.status` get a `case "unlinked": return <PageNotLinked />;` instead.)

- [ ] **Step 3: Type-check (the exhaustiveness gap from Task 6 should now be closed)**

Run: `cd web && pnpm exec tsc --noEmit`
Expected: clean (exit 0).

- [ ] **Step 4: Commit**

```bash
git add web/src/components/ui/PageStates.tsx web/src/app
git commit -m "feat(web): TeamNotLinked state on personalized pages (HIGH-1)"
```

---

## Task 8: Stop the page fetchers swallowing live errors

Convert each page `fetch*()` shim so that under live, errors PROPAGATE (no mock); off-live, mock stays. Use `liveOrMock` (+ `withTimeout` where a timeout existed).

**Files (one function each):**
- `web/src/lib/data.ts` → `fetchMyTeam` (remove the `try {} catch {}` swallow at `:114-116`)
- `web/src/lib/matchup-data.ts` → `fetchMatchup` (remove `.catch(() => null)` at `:237`; keep a timeout via `withTimeout`)
- `web/src/lib/standings-data.ts` → `fetchStandings` (outer `catch {}` at `:96-98`)
- `web/src/lib/players-data.ts` → `fetchPlayers` (`catch {}` at `:84-86`)
- `web/src/lib/streaming-data.ts` → `fetchStreaming` (`catch {}` at `:220-222`)
- `web/src/lib/closers-data.ts` → `fetchClosers` (`catch {}` at `:85-87`)
- `web/src/lib/punt-data.ts` → `fetchPunt` (`catch {}` at `:80-82`)
- `web/src/lib/probables-data.ts` → `fetchProbables` (`catch {}` at `:221-223`)
- `web/src/lib/hitter-matchups-data.ts` → `fetchHitterMatchups` (`catch {}` at `:208-210`)

- [ ] **Step 1: Refactor `fetchMyTeam` (the reference pattern).** In `web/src/lib/data.ts`, replace the body (lines ~106-119) with:

```typescript
export async function fetchMyTeam(delayMs = 700): Promise<MyTeamData> {
  return liveOrMock(
    async () => {
      const api = await apiGet<ApiMyTeamResponse>("/me/team", { team_name: VIEWER_TEAM });
      if (api.team_name) setViewerTeam(api.team_name);
      return apiMyTeamToData(api);
    },
    () => new Promise<MyTeamData>((resolve) => setTimeout(() => resolve(MY_TEAM), delayMs)),
  );
}
```

Add the import at the top of the file: `import { liveOrMock } from "@/lib/api/live";` (and drop the now-unused `isLive` import if `liveOrMock` fully replaces it in this file).

- [ ] **Step 2: Refactor `fetchMatchup` (the timeout pattern).** In `web/src/lib/matchup-data.ts`, replace the body (lines ~232-243) with:

```typescript
export async function fetchMatchup(delayMs = 600): Promise<MatchupData | null> {
  return liveOrMock(
    async () => {
      const d = await withTimeout(
        apiGet<ApiMatchupResponse>("/matchup", { team_name: VIEWER_TEAM }).then(apiMatchupToData),
        LIVE_TIMEOUT_MS,
      );
      // A useful-but-partial response renders as-is; a truly empty one → page `empty`.
      return d.cats.length > 0 || d.hitters.length > 0 || d.pitchers.length > 0 ? d : null;
    },
    () => new Promise<MatchupData>((resolve) => setTimeout(() => resolve(MATCHUP), delayMs)),
  );
}
```

Update the imports: `import { liveOrMock, withTimeout } from "@/lib/api/live";` (replace the lone `isLive` import). Keep `LIVE_TIMEOUT_MS` as defined in that file.

- [ ] **Step 3: Refactor the remaining 7 page fetchers** using the Step-1 pattern (no timeout) — wrap the existing live `apiGet(...)→adapt` body in `liveOrMock(live, mock)`, deleting the `catch {}`/`.catch(() => null)` so errors propagate. Each keeps its existing mock as the `mock()` arg. Files + functions: `standings-data.ts:fetchStandings`, `players-data.ts:fetchPlayers`, `streaming-data.ts:fetchStreaming`, `closers-data.ts:fetchClosers`, `punt-data.ts:fetchPunt`, `probables-data.ts:fetchProbables`, `hitter-matchups-data.ts:fetchHitterMatchups`. Add `import { liveOrMock } from "@/lib/api/live";` to each.

  Note on `standings-data.ts`: keep the INNER `/playoff-odds` handling (its 402→locked panel logic at `:88-94`) intact — only the OUTER `/standings` `catch {}` (`:96-98`) becomes propagate-under-live. `/playoff-odds` is team-optional, so a `team_name=null` viewer still gets odds without the `you` marker (no 409).

- [ ] **Step 4: Type-check**

Run: `cd web && pnpm exec tsc --noEmit`
Expected: clean. (Note: `fetchMatchup` now returns `MatchupData | null`; `usePageData<MatchupData>` already accepts `T | null` — no page change needed.)

- [ ] **Step 5: Commit**

```bash
git add web/src/lib
git commit -m "fix(web): page fetchers propagate live errors instead of swallowing to mock (HIGH-3)"
```

---

## Task 9: Stop imperative fetchers fabricating data under live

`compare`, `draft`, and `streaming analyze` substitute fabricated stat lines/advice for the user's real picks on error. Under live, surface the error instead.

**Files:**
- `web/src/lib/compare-data.ts` → the `catch {}` → `mockCompare(players)` at `:116-118`
- `web/src/lib/draft-data.ts` → the two non-402 fallbacks to `mockSimulate`/`mockRecommend` at `:113-118, 136-141`
- `web/src/lib/streaming-data.ts` → `analyzePitcher` → `mockScorecard` at `:269-271`

- [ ] **Step 1: `compare-data.ts`.** Wrap the live branch in `liveOrMock` so an `ApiError` propagates (the caller surfaces it); keep `mockCompare` only as the off-live `mock()`:

```typescript
export async function fetchCompare(players: PlayerLite[]): Promise<CompareData> {
  return liveOrMock(
    () => apiGet<ApiCompareResponse>("/compare", { ids: players.map((p) => p.id).join(",") }).then(apiCompareToData),
    () => mockCompare(players),
  );
}
```
(Match the real param/shape in the file; the point is: live → no `catch` to `mockCompare`.)

- [ ] **Step 2: `draft-data.ts`.** For both `simulatePicks` and `recommend`, keep the 402 rethrow (paywall) but, under live, let other errors propagate instead of switching to `mockSimulate`/`mockRecommend`. Pattern:

```typescript
  if (isLive()) {
    return apiPost<ApiX>("/draft/...", body).then(adapt); // 402 propagates → locked; other errors → caller/usePageData
  }
  return mockSimulate(/* ... */);
```
Remove the `.catch` that falls to the mock under live. (Keep `isLive` imported here.)

- [ ] **Step 3: `streaming-data.ts` `analyzePitcher`.** Under live, let the error propagate so the Analyze panel shows a failure instead of a fabricated "Synthesized scorecard". Off-live, keep `mockScorecard`:

```typescript
export async function analyzePitcher(pitcherId: number): Promise<Scorecard> {
  return liveOrMock(
    () => apiPost<ApiPitcherScorecard>("/streaming/analyze", { pitcher_id: pitcherId }).then(adapt),
    () => mockScorecard(pitcherId),
  );
}
```
(Use the file's real types/fn names; the rule is: no fabricated scorecard under live error. Ensure the Analyze UI catches the rejection and shows an inline error — if it currently assumes success, add a try/catch in the component that sets a local error message.)

- [ ] **Step 4: Type-check**

Run: `cd web && pnpm exec tsc --noEmit`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add web/src/lib
git commit -m "fix(web): imperative fetchers surface live errors, no fabricated data (HIGH-3)"
```

---

## Task 10: End-to-end verification (preview against live API) + final checks

`web/` has no CI/test runner, so the gate is tsc + Claude_Preview against the live API.

- [ ] **Step 1: Backend full api suite**

Run: `python -m pytest tests/api/ -q`
Expected: `419 passed, 1 failed` (the new Task 1–4 tests added; the 1 failure remains ONLY the `test_openapi_snapshot_is_current` env-pin artifact — confirm by name: `python -m pytest tests/api/test_openapi_contract.py -q` is the only red, and `git diff --stat api/openapi.json` shows NO change).

- [ ] **Step 2: Structural guards (pre-push set)**

Run: `python -m pytest tests/api/test_no_logic_in_routers.py -q`
Expected: PASS.

- [ ] **Step 3: Frontend type-check + production build**

Run: `cd web && pnpm exec tsc --noEmit && pnpm build`
Expected: tsc clean; build succeeds.

- [ ] **Step 4: Preview verification** (Claude_Preview, `NEXT_PUBLIC_HEATER_LIVE=1` against the live Railway API). Verify each:
  - **5xx/network → error state** (not mock): point a page at a bad path or stop the API; confirm `PageError` renders (not fabricated data). 
  - **Signed-out → sign-in**: with Clerk active and no session, a personalized page routes to `/sign-in`.
  - **Authed-unassigned → TeamNotLinked**: a signed-in user with no assignment sees the friendly "team isn't linked yet" state on My Team / Matchup / Optimizer / Punt / Players / Trades-finder; Standings / Leaders / Players-search / Closers / schedule grids still render.
  - **Demo build (flag off) → mock** still renders (no regression).

- [ ] **Step 5: Final commit (if any preview-driven tweaks) + push**

```bash
git add -A
git commit -m "test(web): preview-verify HIGH fixes (error/sign-in/unlinked states)"
git push origin master
```
Expected: pre-push structural suite passes (~20s); push succeeds.

---

## Self-Review (completed by plan author)

**Spec coverage:** Fix 1 → Tasks 1,2,3,6,7 (+ frontend 401/unlinked). Fix 2 → Task 4. Fix 3 → Tasks 5,8,9 (+ usePageData in 6). Verification → Task 10. ✓ All three fixes have tasks.

**Deviation from spec (intentional, better):** `/api/playoff-odds` is reclassified **team-optional** (degrade, no 409) so the Standings page keeps working for an unassigned viewer — the spec's two playoff-odds mentions will be aligned. ✓

**Placeholder scan:** The only soft spots are deliberate "match the file's real type/fn names" notes on the frontend imperative fetchers (compare/draft/streaming), because those files' exact symbol names weren't read; the transform (no mock under live error) is fully specified. Backend tasks have complete code. ✓

**Type consistency:** `TEAM_NOT_LINKED`/`resolve_required_team` (Task 2) used consistently in Tasks 3 + the test; `_is_pitcher`/`_PITCHER_SLOTS` (Task 4) consistent; `isTeamNotLinked`/`liveOrMock`/`withTimeout`/`PageNotLinked`/`unlinked` consistent across Tasks 5–9. ✓

**Dormancy invariant:** every backend change keeps `user_id is None` (Clerk-off) returning the fallback; every frontend change keeps `!isLive()` returning mock. ✓

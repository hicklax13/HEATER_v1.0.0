# HEATER — Pre-launch HIGH-severity Fixes (Design)

**Date:** 2026-06-22
**Lane:** CEO/backend (`api/`, `tests/`) + a frontend slice (`web/`)
**Source:** the 2026-06-22 pre-launch audit (`Outstanding_June26.md`); fixes the three HIGH findings.
**Scope decision (owner, 2026-06-22):** fix the **3 HIGH items only**. MED/LOW + buildable-now follow-ups are a fast-follow, out of scope here.
**Product decision (owner, 2026-06-22):** an authenticated user with no team assignment sees a friendly **"team not linked yet"** state on personalized pages; league-wide pages keep working.

## Context

The M3 single-league beta is deployed (React on Vercel + FastAPI on Railway over the unchanged `src/` engines; Streamlit fallback). The Clerk login-gate is **live in production** (`GET /api/me/team` → 401). The audit found three HIGH-severity defects that are live-reachable now — two of them specifically because Clerk is on and the 12 leaguemates are not yet assigned (assignment is sequenced as the last launch step). This spec closes all three so the beta is safe to onboard the 12.

These are bug fixes to existing surfaces — no new features. The `src/` analytics engines are NOT touched.

---

## Fix 1 — HIGH-1: Tenancy isolation hole

### Problem
`api/tenancy.py:60-63`:
```python
def effective_team(self, fallback: str | None) -> str | None:
    return self.team_name or fallback
```
For a logged-in user with **no** team assignment, `self.team_name is None`, so `effective_team(<client param>)` returns the client's `team_name` query param — which the frontend hardcodes to `"Team Hickey"`. Every personalized service then slices that team (e.g. `team_service.py` `lr[lr["team_name"] == team_name]`), so the user is served the **owner's** roster / lineup / matchup / trades / playoff-odds / free-agent recs. The `ViewerContext` docstring claims "never another user's team" — the code contradicts it.

This is live now (Clerk on + the 12 unassigned) and worse post-public-launch (a paying stranger with no league → sees Team Hickey). Bounded (read-only, one configured owner team, self-closes on assignment) but a real isolation break on a paid product.

### Design
Three viewer states must be distinguished, where today there are effectively two:

| State | Condition | `effective_team(fallback)` returns |
|---|---|---|
| **dormant** (Clerk off / no valid token) | `authenticated == False` | `fallback` (preserves today's open behavior, byte-for-byte) |
| **assigned** | authenticated + membership exists | the resolved team (ignores the client param — already correct) |
| **unassigned** | authenticated + no membership | **`None`** (NEVER the fallback) |

Changes:

1. **`api/tenancy.py`** — add an `authenticated: bool = False` field to `ViewerContext`; set it `True` in `require_viewer_context` whenever `app_user is not None`. Rewrite `effective_team`:
   - `if self.team_name: return self.team_name`
   - `if self.authenticated: return None`  *(authed-unassigned — no fallback)*
   - `return fallback`  *(dormant)*

2. **Team-required vs team-optional split.** Two classes of endpoint currently call `effective_team`:
   - **Team-required** (meaningless without a team): `/api/me/team`, `/api/matchup`, `/api/lineup/optimize`, `/api/punt`, `/api/trade/evaluate`, `/api/trade-finder`, `/api/free-agents`, `/api/free-agents/pool`.
   - **Team-optional** (league-wide; the team only adds a "yours"/"you" marker): `/api/schedule/probables`, `/api/schedule/hitter-matchups`, `/api/playoff-odds`. *(`/api/playoff-odds` is league-wide odds + a `you` marker, so an unassigned viewer still gets the Standings odds panel — just without the highlight; it must NOT 409 or it would break the Standings page.)*

   Add a small helper in `api/tenancy.py`:
   ```python
   TEAM_NOT_LINKED = "team_not_linked"
   def resolve_required_team(ctx: ViewerContext, fallback: str | None) -> str:
       team = ctx.effective_team(fallback)
       if team is None and ctx.authenticated:
           raise HTTPException(status_code=409, detail=TEAM_NOT_LINKED)
       # dormant with no fallback (shouldn't happen for these routes) → also 409-safe
       if team is None:
           raise HTTPException(status_code=409, detail=TEAM_NOT_LINKED)
       return team
   ```
   Team-required routers call `resolve_required_team(ctx, team_name)` instead of `ctx.effective_team(team_name)`. Team-optional routers (`schedule.py`) keep `ctx.effective_team(team_name)` and the service treats `None` as "no highlight" (build the grid normally).

   *Rationale for 409 (not 403):* it is a machine-readable signal the frontend maps to the friendly "not linked" state, not a generic error page. 403 is reserved for genuine authz failures. Chosen over a per-response `team_linked: bool` flag (which would touch 9 contracts) because it is one centralized change.

3. **Frontend** (`web/`):
   - The API client already throws `ApiError(status, path)` on non-2xx. Add `isTeamNotLinked(e)` to `web/src/lib/api/errors.ts` (409 + detail `team_not_linked`).
   - `web/src/lib/use-page-data.ts` gains an `"unlinked"` state, set when the fetcher rejects with `isTeamNotLinked`.
   - A shared `web/src/components/billing/PaywallGate.tsx`-style component `TeamNotLinked` renders the friendly message ("Your team isn't linked yet — your commissioner will assign it. League-wide views still work.").
   - Personalized pages render `TeamNotLinked` on the `unlinked` state. League-wide pages and the schedule grids are unaffected.

### Tests
- `tests/api/test_api_tenancy_resolver.py` — **invert** the assertion at lines 95-106 (an authed-unassigned viewer must NOT receive the client `team_name`); add a 3-state `effective_team` unit test (dormant→fallback, assigned→team, unassigned→None).
- New route test: an authed-unassigned request to a team-required endpoint returns **409 `team_not_linked`**, NOT Team Hickey's data; a team-optional endpoint (schedule) returns 200 with the grid and no highlight.
- Confirm the existing "assigned user overrides the client param" tests still pass (`test_api_personalized_team_resolution.py`).

### Non-goals
- No change to the assigned-user or dormant (Clerk-off) paths — both stay byte-for-byte.
- Does not implement multi-league `league_id` scoping (that's M4).

---

## Fix 2 — HIGH-2: Matchup pitcher misclassification

### Problem
`api/services/matchup_service.py:504-520` (`_is_pitcher_by_pool`): a `_PITCHER_SLOTS = {"SP","RP","P"}` set is defined (line 502) but the fallback ignores it — any player **absent from the pool** or with `is_hitter is None` hits `return False` (= hitter). The comment says "pitcher-leaning default"; the code does the opposite. A pool-missing rostered pitcher is silently routed into the hitter comparison table (blank/garbage stat line) and dropped from the pitcher table + `pitcher_totals`, with no log.

### Design
1. On pool-miss (or `is_hitter is None`), classify by the player's **eligible-positions string** (e.g. `"SP,RP"`, `"2B,SS"`): if any token ∈ `_PITCHER_SLOTS` → pitcher (`True`), else hitter (`False`). **Use eligible positions, NOT the assigned `selected_position`** — a benched/IL pitcher's `selected_position` is `"BN"`/`"IL"` and would misclassify; eligible positions carry the true SP/RP eligibility. The pool `is_hitter` flag remains authoritative when present (a pool hit still wins).
2. Add a module logger to `matchup_service.py` (it currently has none — also closes part of MED-2 for this file) and `logger.debug(...)` the unresolved pid so a systematic pool-coverage gap is observable.
3. The classifier needs the eligible-positions string. If `_is_pitcher_by_pool`'s current signature (`pid, roster_status, side_pool`) doesn't carry it, pass the roster row's eligible-positions value from the caller `_build_side` (fall back to `selected_position` only if eligible is absent). Keep the change local to these two helpers.

### Tests
- DB-free unit test for the classifier: pool-hit `is_hitter=False` → pitcher; pool-miss with slot `"SP"`/`"RP"`/`"P"` → pitcher; pool-miss with slot `"2B"` → hitter; pool-miss with empty slot → hitter (documented conservative default), and a log is emitted on the miss.

### Non-goals
- No change to the pool-hit classification path (unchanged behavior for known players).
- Does not add live in-game box scores (separate deferred item).

---

## Fix 3 — HIGH-3: Frontend swallows live errors → fabricated data

### Problem
`web/src/lib/use-page-data.ts` only reaches its `error` state if the fetcher rethrows — but ~12 `web/src/lib/*-data.ts` fetchers use a bare `catch {}` that returns the page **mock** (e.g. `data.ts:114`, `matchup-data.ts:237`, `standings-data.ts:96`, `players-data.ts:84`, `streaming-data.ts:220`, `closers-data.ts:85`, `punt-data.ts:80`, `probables-data.ts:221`, `hitter-matchups-data.ts:208`). Compare (`compare-data.ts:116`) and Draft (`draft-data.ts:113,136`) substitute fabricated stat lines / advice for the **real** players the user picked. **401 is swallowed everywhere** — `errors.ts:21 isAuthRequired` exists but no fetcher uses it — so with the login-gate live a signed-out visitor sees fabricated data instead of a login prompt.

### Design
One behavior rule, applied consistently via a small shared helper so the ~16 fetchers don't drift:

When `NEXT_PUBLIC_HEATER_LIVE === "1"` (live):
- **402** (paywall) → rethrow → `usePageData` `locked` (already works for Pro pages; unchanged).
- **401** (auth) → route to sign-in. Wire `isAuthRequired` → redirect to the Clerk `/sign-in` (or a "please sign in" state). No mock.
- **409 `team_not_linked`** → `unlinked` state (from Fix 1). No mock.
- **5xx / network / any other non-2xx** → rethrow → `usePageData` `error` state with retry. **No mock.**

When `NEXT_PUBLIC_HEATER_LIVE !== "1"` (demo/marketing build):
- Mock fallback is the intended behavior — keep it.

Implementation:
- Add a shared helper, e.g. `web/src/lib/api/live-or-mock.ts` exporting `liveOrMock<T>(live: () => Promise<T>, mock: () => T | Promise<T>): Promise<T>` that runs `live()` when the flag is on and **rethrows** `ApiError`/network errors (so `usePageData` classifies them), falling back to `mock()` only when the flag is off. Refactor the per-page `fetch*()` shims to use it, removing the bare `catch {}`.
- `use-page-data.ts` maps: `isPaywall`→`locked`, `isTeamNotLinked`→`unlinked`, `isAuthRequired`→sign-in redirect, else→`error`.
- The imperative fetchers (compare, draft simulate/recommend, streaming analyze) follow the same rule under live — surface the error instead of fabricating data.

### Tests / verification
- `web/` has no CI; the gate is `pnpm exec tsc --noEmit` (must stay clean) + Claude_Preview against the live API: verify (a) a 5xx shows the error state (not mock), (b) signed-out → sign-in routing, (c) an unassigned user → the `TeamNotLinked` state, (d) demo build (flag off) still renders mock.
- Where a lightweight unit test is feasible (the `liveOrMock` helper + the `usePageData` status→state mapping), add it; otherwise rely on tsc + preview.

### Non-goals
- No change to the demo/marketing (flag-off) behavior.
- Not a redesign of the error/empty components beyond adding `TeamNotLinked` and ensuring the `error` state is reachable.

---

## Cross-cutting

- **Ship order:** Fix 1 (backend) → Fix 3 (frontend, depends on Fix 1's 409 signal) are the launch-blockers; Fix 2 is independent and rides along. Backend (Fix 1 + Fix 2) can land first; the frontend slice (Fix 3) consumes Fix 1's contract.
- **Method:** TDD for the backend (test → red → green); the frontend slice is tsc + preview-verified (no web CI).
- **Dormancy preserved:** every change keeps the Clerk-off / demo-build paths byte-for-byte identical to today, so nothing regresses for the current open behavior.
- **Structural guards:** routers must stay logic-free (`tests/api/test_no_logic_in_routers.py`) — the `resolve_required_team` helper lives in `api/tenancy.py`, not in routers; routers just call it. OpenAPI snapshot regenerates only if a contract changes (Fix 1 adds 409 responses → regenerate `api/openapi.json` + frontend types).

## Out of scope (explicit)
- All MED/LOW findings (service logging sweep beyond matchup, write-endpoint team-targeting guard, OpsCards float, Punt/Standings real tests, type-design hardening, accessibility) — fast-follow.
- All owner-gated M4/M5 infra and M6 manual tasks.
- The ~11 buildable-now follow-ups behind shipped endpoints (champ odds, ip_pace, etc.).

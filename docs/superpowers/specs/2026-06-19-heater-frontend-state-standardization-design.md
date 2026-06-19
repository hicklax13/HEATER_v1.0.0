# HEATER Web — Page-State Standardization (design spec)

**Date:** 2026-06-19
**Track:** CMO / frontend (`web/`)
**Status:** approved direction; pending implementation plan
**Sub-project:** A (frontend foundation) — hardening pass before C (page migration)

## Goal

Give all six pages of the HEATER Next.js app the **same four-state machine**
(`loading → error → empty → loaded`) that the Team page already has, behind a
single shared hook and shared state components — so that when Sub-project C swaps
the mock shims for the live FastAPI endpoints, every page degrades gracefully and
identically instead of spinning forever on a skeleton.

This is "Task 1" of the current CMO handoff (CLAUDE.md ★ Web Frontend → NEXT).

## Current state (the gap)

- **Team** (`web/src/app/page.tsx`) is the reference: a discriminated-union
  `State` (`loading|error|empty|loaded`) with a bespoke `Skeleton` loader, an
  error card (`EmptyState` error tone + Retry), and an empty card. Its `error`/
  `empty` branches are currently unreachable because the mock shim never rejects
  or returns null — they are scaffolding for real data.
- **Optimizer, Matchup, Trades, Players, Research** each handle only
  `loading | loaded` via a `!data ? <LoadingView/> : <content/>` ternary, with
  **no `.catch()` and no empty branch**. A real API failure or empty payload
  would hang on the skeleton indefinitely.
- The shared primitives already exist: `<EmptyState>` (neutral + error tones,
  `web/src/components/ui/EmptyState.tsx`) and `<Skeleton>`.
- The in-table **"no results"** state (filtered list → zero rows) is **already
  handled** on Players (`players/page.tsx:180`) and Research
  (`research/page.tsx:147`) via `<EmptyState icon={SearchX}>`. It is **not** part
  of this work.
- All six fetch shims (`web/src/lib/*-data.ts`) return **non-nullable,
  never-rejecting** Promises (`setTimeout → resolve(DATA)`).

## Non-negotiables

- Mock shims (`web/src/lib/*-data.ts`) are **untouched** — their shapes are the
  API contract for Sub-project C.
- `pnpm build`, `pnpm exec tsc --noEmit`, and `pnpm run lint` stay green
  (`web/` has no CI; the local production build is the gate).
- Every visual state is verified in the **real Claude_Preview** and screenshotted,
  not mocked in isolation.
- `prefers-reduced-motion` continues to be honored (Skeleton pulse already is).
- Streamlit app and `src/` engine untouched (CMO writes stay inside `web/` and
  `docs/superpowers/`).
- Next 16 caveat (`web/AGENTS.md`): read `node_modules/next/dist/docs/` before
  using any Next-specific API.

---

## 1. The shared hook — `web/src/lib/use-page-data.ts`

```ts
export type PageState<T> =
  | { status: "loading" }
  | { status: "error" }
  | { status: "empty" }
  | { status: "loaded"; data: T };

export function usePageData<T>(
  fetcher: () => Promise<T | null>,
): { state: PageState<T>; retry: () => void };
```

Behavior:

- On mount and on `retry()`: set `loading`, call `fetcher()`.
- Resolve: `data == null` → `empty`; otherwise `{ status: "loaded", data }`.
- Reject: `error`.
- Alive-flag cleanup (mirrors the existing per-page pattern) prevents setState
  after unmount and tolerates React StrictMode's dev double-invoke.
- `retry` is a stable `useCallback` that re-runs the fetcher.
- The hook is the **single owner** of the dev `?state=` override (§4).

Emptiness is decided **only** at the `null` boundary. Page-specific "loaded but
zero items" cases (Trades with 0 recs, Optimizer's "lineup is optimal", the
filtered no-results tables) intentionally stay **inside** the `loaded` view —
they are a different situation from "no data at all," and must not be collapsed
into the page-level `empty` state.

## 2. Shared state components — `web/src/components/ui/PageStates.tsx`

Lift the Team page's `ErrorView` and `EmptyView` markup into two reusable
components so there is exactly one canonical implementation:

- `<PageError onRetry={() => void} />` — `Card` + `EmptyState` (error tone,
  `RefreshCw` icon) + the orange Retry button. Generic copy
  ("We couldn't load this / The data service didn't respond — try again.").
- `<PageEmpty icon title body />` — `Card` + `EmptyState` (neutral tone),
  copy supplied per page.

The **Team page is refactored to consume `usePageData` + these components too**,
so the reference is the shared code rather than a seventh hand-rolled variant.

## 3. Per-page wiring (Optimizer, Matchup, Trades, Players, Research, Team)

Each page:

- Replaces its `useState`/`useEffect` fetch block with
  `const { state, retry } = usePageData(fetchX);`.
- Renders the four branches **inside the page's existing `<main>` wrapper** (so
  error/empty inherit the page padding and center like Team's cards):
  - `loading` → the page's existing bespoke `<LoadingView/>` skeleton (unchanged;
    skeletons stay per-page so they mirror each layout).
  - `error` → `<PageError onRetry={retry} />`.
  - `empty` → `<PageEmpty … />` with the page's copy (§5).
  - `loaded` → existing content, sourced from `state.data`.
- `<Footer/>` moves to **loaded-only** (matches Team). Today the five lagging
  pages render `<Footer>` unconditionally; under error/empty it should not show.
  Each page keeps its current freshness literal for now (mock).

Pages keep their own framer-motion stagger wrappers inside the `loaded` branch
exactly as today.

## 4. Dev `?state=` override (verification aid)

- **Dev-only** (`process.env.NODE_ENV !== "production"`) and **client-only**.
- Reads the `state` query param from `window.location.search` (NOT
  `useSearchParams` — that triggers Next 16's static-generation CSR bailout /
  Suspense requirement; `window.location.search` in a client effect sidesteps it.
  Confirm against `node_modules/next/dist/docs/` before implementing).
- `?state=loading|error|empty|loaded` forces that branch:
  - `loading` → stay in loading, skip the fetch.
  - `error` / `empty` → set that state, skip the fetch.
  - `loaded` / absent → normal path.
- The whole override is behind the `NODE_ENV` guard so it is dead-code-eliminated
  from the production bundle. Confirm it is absent from `pnpm build` output.
- Purpose: lets us render and screenshot all four states this session
  (e.g. `/optimizer?state=error`) without altering the mock shims.

## 5. Per-page empty copy

Error copy is generic + Retry (same as Team) on every page. Empty copy is
page-specific (what "no data at all" means for that surface):

| Page      | Empty title                 | Empty body |
|-----------|-----------------------------|------------|
| Optimizer | No lineup to optimize       | We couldn't find your roster for today. |
| Matchup   | No active matchup           | You're between matchups — check back when the next week opens. |
| Trades    | No trade ideas yet          | We need a bit more league data to surface targets. |
| Players   | No free agents available    | The free-agent pool is empty right now. |
| Research  | No leaders to show          | Leaderboard data isn't available yet. |
| Team      | (unchanged) No team data yet | Connect your Yahoo league and we'll build your dashboard automatically. |

Copy is final-pass polish; may be tweaked during implementation without re-spec.

## 6. Risks & mitigations

- **Next 16 `useSearchParams` bailout** → use `window.location.search` in a client
  effect; verify against the bundled Next docs (`web/AGENTS.md`).
- **StrictMode double-invoke (dev)** → alive-flag in the hook; `retry` idempotent.
- **Override leaking to prod** → `NODE_ENV` guard + a check of the production build.
- **Footer regression** → moving Footer to loaded-only is the only change to the
  happy path; verify the loaded pages still show it.

## 7. Acceptance criteria / verification

- `pnpm build` + `pnpm exec tsc --noEmit` + `pnpm run lint` all green from `web/`.
- The discriminated union forces every page to handle all four `status` values
  (compiler-enforced consistency).
- Claude_Preview screenshots captured for **all four states** on each of the five
  migrated pages via `?state=…`, plus a Team regression check (loaded + one
  forced state).
- Mock shims unchanged (git diff shows no edits under `web/src/lib/*-data.ts`).

## 8. Out of scope (YAGNI)

- Wiring any page to the live FastAPI API — that is Task 3 / Sub-project C.
- Changing the in-table filtered "no results" state (already implemented).
- New freshness/data-age plumbing through the shims.
- Retry backoff, toasts, or error telemetry.

## Files touched

- **New:** `web/src/lib/use-page-data.ts`,
  `web/src/components/ui/PageStates.tsx`.
- **Edited:** `web/src/app/page.tsx` (Team — adopt hook + shared components),
  `web/src/app/optimizer/page.tsx`, `web/src/app/matchup/page.tsx`,
  `web/src/app/trades/page.tsx`, `web/src/app/players/page.tsx`,
  `web/src/app/research/page.tsx`.
- **Untouched:** all `web/src/lib/*-data.ts`, `src/`, Streamlit app.

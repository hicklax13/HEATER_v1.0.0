# Web Page-State Standardization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give all six `web/` pages the same `loading → error → empty → loaded` state machine the Team page has, behind one shared `usePageData` hook + shared `PageError`/`PageEmpty` components, with a dev `?state=` override for visual verification.

**Architecture:** A discriminated-union state hook (`usePageData`) owns fetch/catch/null-empty/retry and the dev override. Two shared state components render the error/empty cards. Every page (including Team, refactored onto the shared code) branches on `state.status`; each page's loaded content moves into a `Loaded` child so data-derived hooks only run when data exists. Mock shims are untouched.

**Tech Stack:** Next.js 16 (App Router, client components) · React 19 · TypeScript 5.9 · Tailwind v4 · framer-motion · lucide-react. Package manager **pnpm** (run all commands from `web/`).

**Spec:** `docs/superpowers/specs/2026-06-19-heater-frontend-state-standardization-design.md`

**Branch:** `feat/web-page-state-standardization` (already created; spec already committed).

---

## Verification model (read first)

`web/` has **no unit-test runner** by design. The per-task gate is:

- `pnpm exec tsc --noEmit` — type-check (the discriminated union is the consistency check).
- `pnpm run lint` — ESLint.
- Visual verification in **Claude_Preview** (`next dev`) using the dev `?state=` override.
- `pnpm build` — production gate, run at milestones (Task 3 and Task 9).

All commands run **from the `web/` directory**. Commit after each task. Do **not** push or merge (owner gates that). Do **not** edit any `web/src/lib/*-data.ts` shim.

### File structure (locked here)

- **Create** `web/src/lib/use-page-data.ts` — `PageState<T>` type, `usePageData` hook, dev `?state=` override. One responsibility: page-level async state.
- **Create** `web/src/components/ui/PageStates.tsx` — `PageError`, `PageEmpty`. One responsibility: the shared error/empty cards.
- **Modify** `web/src/app/page.tsx` (Team) — adopt hook + shared components; delete its local `State`/`ErrorView`/`EmptyView`.
- **Modify** `web/src/app/optimizer/page.tsx`, `matchup/page.tsx`, `trades/page.tsx`, `players/page.tsx`, `research/page.tsx` — adopt hook; extract loaded content into a `Loaded` child; Footer becomes loaded-only.

---

## Task 1: Create the `usePageData` hook

**Files:**
- Create: `web/src/lib/use-page-data.ts`

- [ ] **Step 1: Write the hook**

Create `web/src/lib/use-page-data.ts` with exactly this content:

```ts
"use client";

import { useCallback, useEffect, useRef, useState } from "react";

/** Page-level async state. `loaded` carries the data; the other three are terminal UI states. */
export type PageState<T> =
  | { status: "loading" }
  | { status: "error" }
  | { status: "empty" }
  | { status: "loaded"; data: T };

type Forced = "loading" | "error" | "empty";

/**
 * Dev-only verification aid: `?state=loading|error|empty` forces a branch so we
 * can screenshot it (mock shims never fail/return null). Returns undefined in
 * production (NODE_ENV guard → dead-code-eliminated) or when absent/`loaded`.
 * Reads `window.location.search` directly — NOT next/navigation's
 * `useSearchParams`, which triggers Next 16's static-generation CSR bailout.
 */
function forcedState(): Forced | undefined {
  if (process.env.NODE_ENV === "production") return undefined;
  if (typeof window === "undefined") return undefined;
  const v = new URLSearchParams(window.location.search).get("state");
  return v === "loading" || v === "error" || v === "empty" ? v : undefined;
}

/**
 * Drives the four-state machine for a page. Pass a STABLE fetcher (a module-level
 * `fetchX` reference, never an inline arrow — an inline arrow changes identity
 * every render and re-fires the effect). `null`/`undefined` resolves to `empty`;
 * a rejection resolves to `error`. `retry()` re-runs the fetcher.
 */
export function usePageData<T>(
  fetcher: () => Promise<T | null>,
): { state: PageState<T>; retry: () => void } {
  const [state, setState] = useState<PageState<T>>({ status: "loading" });
  const reqId = useRef(0);

  const load = useCallback(() => {
    const forced = forcedState();
    if (forced) {
      setState({ status: forced });
      return;
    }
    const id = ++reqId.current;
    setState({ status: "loading" });
    fetcher()
      .then((data) => {
        if (id !== reqId.current) return; // superseded by a newer load / unmount
        setState(data == null ? { status: "empty" } : { status: "loaded", data });
      })
      .catch(() => {
        if (id === reqId.current) setState({ status: "error" });
      });
  }, [fetcher]);

  useEffect(() => {
    load();
    return () => {
      reqId.current++; // invalidate any in-flight resolution on unmount / re-run
    };
  }, [load]);

  const retry = useCallback(() => load(), [load]);

  return { state, retry };
}
```

- [ ] **Step 2: Type-check**

Run (from `web/`): `pnpm exec tsc --noEmit`
Expected: no errors.

- [ ] **Step 3: Lint**

Run (from `web/`): `pnpm run lint`
Expected: no errors for `src/lib/use-page-data.ts`.

- [ ] **Step 4: Commit**

```bash
git add web/src/lib/use-page-data.ts
git commit -m "feat(web): add usePageData four-state hook + dev ?state= override

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Create the shared `PageError` / `PageEmpty` components

**Files:**
- Create: `web/src/components/ui/PageStates.tsx`

These lift the Team page's `ErrorView`/`EmptyView` markup verbatim (Card + `EmptyState`), with generic error copy.

- [ ] **Step 1: Write the components**

Create `web/src/components/ui/PageStates.tsx` with exactly this content:

```tsx
import type { LucideIcon } from "lucide-react";
import { RefreshCw } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";

/** Page-level load failure: error card + Retry. Generic copy works for every page. */
export function PageError({ onRetry }: { onRetry: () => void }) {
  return (
    <Card className="mx-auto mt-10 max-w-md">
      <EmptyState
        icon={RefreshCw}
        tone="error"
        title="We couldn't load this"
        body="The data service didn't respond. Your data is safe — try again."
        action={
          <button
            onClick={onRetry}
            className="inline-flex min-h-11 items-center gap-2 rounded-xl bg-gradient-to-b from-[#ff7a2e] to-heat px-5 text-sm font-semibold text-white shadow-[0_6px_16px_rgba(255,92,16,0.32)] transition-transform duration-[var(--dur-1)] hover:scale-[1.02] active:scale-95 motion-reduce:transform-none"
          >
            <RefreshCw className="size-4" aria-hidden />
            Retry
          </button>
        }
      />
    </Card>
  );
}

/** Page-level "no data at all" (distinct from in-table no-results). Copy is per page. */
export function PageEmpty({
  icon,
  title,
  body,
}: {
  icon: LucideIcon;
  title: string;
  body?: string;
}) {
  return (
    <Card className="mx-auto mt-10 max-w-md">
      <EmptyState icon={icon} title={title} body={body} />
    </Card>
  );
}
```

> Note: this file has no `"use client"` directive and needs none — the `onClick` button lives here but the file is only ever imported by client pages. If `pnpm exec tsc --noEmit` or the build complains that an event handler crossed a server boundary, add `"use client";` as the first line.

- [ ] **Step 2: Type-check**

Run (from `web/`): `pnpm exec tsc --noEmit`
Expected: no errors.

- [ ] **Step 3: Lint**

Run (from `web/`): `pnpm run lint`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add web/src/components/ui/PageStates.tsx
git commit -m "feat(web): add shared PageError/PageEmpty state components

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Refactor the Team page onto the shared code (reference adopter + first full verify)

**Files:**
- Modify: `web/src/app/page.tsx`

Team already has the four-state machine inline. This task replaces its bespoke `State` type, `ErrorView`, `EmptyView`, `load`, and `useEffect` with the shared hook + components — so the reference *is* the shared code.

- [ ] **Step 1: Edit imports**

In `web/src/app/page.tsx`, replace the React + local imports.

Old (lines 3–8):
```tsx
import { useCallback, useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Activity, Check, Bell, RefreshCw, Inbox, type LucideIcon } from "lucide-react";
import { fetchMyTeam } from "@/lib/data";
import type { MyTeamData } from "@/lib/types";
import { EASE_SNAP } from "@/lib/motion";
```

New:
```tsx
import { useState } from "react";
import { motion } from "framer-motion";
import { Activity, Check, Bell, Inbox, type LucideIcon } from "lucide-react";
import { fetchMyTeam } from "@/lib/data";
import type { MyTeamData } from "@/lib/types";
import { EASE_SNAP } from "@/lib/motion";
import { usePageData } from "@/lib/use-page-data";
import { PageError, PageEmpty } from "@/components/ui/PageStates";
```

(`useCallback`, `useEffect`, and `RefreshCw` are no longer used here. `useState` stays — it backs `leverHover`. The `Skeleton` import on line 20 stays — `LoadingView` still uses it.)

- [ ] **Step 2: Delete the local `State` type**

Remove the `type State = …` block (old lines 22–26).

- [ ] **Step 3: Replace the page component body**

Replace the whole `export default function MyTeamPage() { … }` (old lines 42–80) with:

```tsx
export default function MyTeamPage() {
  const { state, retry } = usePageData(fetchMyTeam);
  const [leverHover, setLeverHover] = useState(false);

  return (
    <>
      <main className="w-full flex-1 px-5 py-6">
        {state.status === "loading" && <LoadingView />}
        {state.status === "error" && <PageError onRetry={retry} />}
        {state.status === "empty" && (
          <PageEmpty
            icon={Inbox}
            title="No team data yet"
            body="Connect your Yahoo league and we'll build your dashboard automatically."
          />
        )}
        {state.status === "loaded" && (
          <Loaded data={state.data} leverHover={leverHover} setLeverHover={setLeverHover} />
        )}
      </main>
      {state.status === "loaded" && <Footer freshnessMinutes={state.data.freshnessMinutes} />}
    </>
  );
}
```

- [ ] **Step 4: Delete the now-unused local state views**

Remove the `function ErrorView(...) { … }` (old lines 179–199) and `function EmptyView() { … }` (old lines 201–211). Keep `function Loaded(...)` and `function LoadingView()` unchanged.

- [ ] **Step 5: Type-check + lint**

Run (from `web/`): `pnpm exec tsc --noEmit && pnpm run lint`
Expected: no errors. (If tsc reports `RefreshCw`/`useEffect`/`useCallback` unused, confirm Step 1 removed them.)

- [ ] **Step 6: Visually verify all four states (first full proof of the machine)**

Start the dev server and screenshot each state:

1. `preview_start` (cwd `web/`, command `pnpm dev`) → note the URL (e.g. `http://localhost:3000`).
2. Loaded: `preview_screenshot` of `/`.
3. Loading: navigate to `/?state=loading` (`preview_eval`: `window.location.href = '/?state=loading'`) → screenshot (skeleton stays).
4. Error: `/?state=error` → screenshot (error card + Retry).
5. Empty: `/?state=empty` → screenshot ("No team data yet").

Expected: all four render correctly; Retry button is present in error; Footer is absent in loading/error/empty.

- [ ] **Step 7: Production build (first milestone gate)**

Run (from `web/`): `pnpm build`
Expected: build succeeds. (Confirms the shared hook/components compile under production settings; the `?state=` override is inert in prod by construction via the `NODE_ENV` guard.)

- [ ] **Step 8: Commit**

```bash
git add web/src/app/page.tsx
git commit -m "refactor(web): Team page adopts shared usePageData + PageStates

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Optimizer page — adopt hook + four states

**Files:**
- Modify: `web/src/app/optimizer/page.tsx`

Pattern for Tasks 4–8: (a) rename the existing `export default function XPage()` to `function Loaded({ data })`, stripping its `data` fetch state and using the `data` prop; (b) add a new `export default function XPage()` shell that calls the hook and branches; (c) move `<Footer>` to loaded-only; (d) fix imports.

- [ ] **Step 1: Edit imports**

Old (lines 3, 15–19 region):
```tsx
import { useEffect, useState } from "react";
```
New:
```tsx
import { useState } from "react";
```
Then add, after the existing `import { LineupTable } …` line:
```tsx
import { usePageData } from "@/lib/use-page-data";
import { PageError, PageEmpty } from "@/components/ui/PageStates";
```
(`useState` stays — `Loaded` uses it for `optimized`. `Wand2` is already imported and will be the empty icon.)

- [ ] **Step 2: Convert the existing page component into `Loaded`**

Replace the existing `export default function OptimizerPage() { … }` (old lines 50–111) with the `Loaded` child below. The `<motion.div variants={staggerContainer} …>` block is the page's **existing** loaded JSX — keep it exactly as it was; only the wrapper, the removed fetch state, and the removed `<main>`/`<Footer>`/loading-ternary change:

```tsx
function Loaded({ data }: { data: OptimizerData }) {
  const [optimized, setOptimized] = useState(false);
  const view = applySwaps(data.starters, data.bench, data.swaps, optimized);

  return (
    <motion.div variants={staggerContainer} initial="hidden" animate="show" className="space-y-6">
      <motion.div variants={staggerItem}>
        <Header date={data.date} optimized={optimized} onOptimize={() => setOptimized(true)} />
      </motion.div>
      {optimized && (
        <motion.div variants={staggerItem}>
          <SuccessBanner swaps={data.swaps} />
        </motion.div>
      )}
      <motion.div variants={staggerItem} className="grid gap-6 lg:grid-cols-[1fr_300px]">
        <div className="space-y-6">
          <Card className="p-5">
            <SectionHead title="Starting Lineup" sub="Today" />
            <LineupTable slots={view.starters} />
          </Card>
          <Card className="p-5">
            <SectionHead title="Bench" sub="Available To Swap In" />
            <LineupTable slots={view.bench} />
          </Card>
        </div>
        <aside className="space-y-4">
          <SwapCard swaps={data.swaps} starters={data.starters} bench={data.bench} optimized={optimized} />
          <PaceCard ipPace={data.ipPace} movesLeft={data.movesLeft} />
          <ImpactCard impact={data.impact} />
        </aside>
      </motion.div>
    </motion.div>
  );
}
```

- [ ] **Step 3: Add the new page shell**

Immediately above `function Loaded`, add:

```tsx
export default function OptimizerPage() {
  const { state, retry } = usePageData(fetchOptimizer);

  return (
    <>
      <main className="w-full flex-1 px-5 py-6">
        {state.status === "loading" && <LoadingView />}
        {state.status === "error" && <PageError onRetry={retry} />}
        {state.status === "empty" && (
          <PageEmpty
            icon={Wand2}
            title="No lineup to optimize"
            body="We couldn't find your roster for today."
          />
        )}
        {state.status === "loaded" && <Loaded data={state.data} />}
      </main>
      {state.status === "loaded" && <Footer freshnessMinutes={9} />}
    </>
  );
}
```

- [ ] **Step 4: Type-check + lint**

Run (from `web/`): `pnpm exec tsc --noEmit && pnpm run lint`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add web/src/app/optimizer/page.tsx
git commit -m "refactor(web): Optimizer adopts usePageData four-state machine

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Matchup page — adopt hook + four states

**Files:**
- Modify: `web/src/app/matchup/page.tsx`

- [ ] **Step 1: Edit imports**

Old (line 3):
```tsx
import { useEffect, useState } from "react";
```
New (the page component no longer uses `useState`/`useEffect` directly; `Loaded` holds no UI state):
```tsx
```
Delete the React import line entirely **only if** nothing else in the file uses `useState`/`useEffect` (Matchup's `Loaded` does not). Then add, after the existing `import { cn } …` line:
```tsx
import { usePageData } from "@/lib/use-page-data";
import { PageError, PageEmpty } from "@/components/ui/PageStates";
```
(`Trophy` is already imported and will be the empty icon.)

> If lint flags the empty React import removal, just ensure line 3 is gone. If any later code in the file references `useState`/`useEffect`, keep the ones it needs.

- [ ] **Step 2: Convert the existing page component into `Loaded`**

Replace `export default function MatchupPage() { … }` (old lines 25–83) with the `Loaded` child (keep the existing `<motion.div …>` block exactly):

```tsx
function Loaded({ data }: { data: MatchupData }) {
  return (
    <motion.div variants={staggerContainer} initial="hidden" animate="show" className="space-y-6">
      <motion.div variants={staggerItem}>
        <ScoreHeader data={data} />
      </motion.div>
      <motion.div variants={staggerItem}>
        <CategoryBattle cats={data.cats} youScore={data.you.score} oppScore={data.opp.score} />
      </motion.div>
      <motion.div variants={staggerItem}>
        <CatTotals data={data} />
      </motion.div>
      <motion.div variants={staggerItem}>
        <RosterCompare
          title="Hitters"
          columns={data.hitterColumns}
          rows={data.hitters}
          totals={data.hitterTotals}
          you={data.you.name}
          opp={data.opp.name}
        />
      </motion.div>
      <motion.div variants={staggerItem}>
        <RosterCompare
          title="Pitchers"
          columns={data.pitcherColumns}
          rows={data.pitchers}
          totals={data.pitcherTotals}
          you={data.you.name}
          opp={data.opp.name}
        />
      </motion.div>
      <motion.div variants={staggerItem}>
        <LeagueMatchups data={data} />
      </motion.div>
    </motion.div>
  );
}
```

- [ ] **Step 3: Add the new page shell**

Immediately above `function Loaded`, add:

```tsx
export default function MatchupPage() {
  const { state, retry } = usePageData(fetchMatchup);

  return (
    <>
      <main className="w-full flex-1 px-5 py-6">
        {state.status === "loading" && <LoadingView />}
        {state.status === "error" && <PageError onRetry={retry} />}
        {state.status === "empty" && (
          <PageEmpty
            icon={Trophy}
            title="No active matchup"
            body="You're between matchups — check back when the next week opens."
          />
        )}
        {state.status === "loaded" && <Loaded data={state.data} />}
      </main>
      {state.status === "loaded" && <Footer freshnessMinutes={2} />}
    </>
  );
}
```

- [ ] **Step 4: Type-check + lint**

Run (from `web/`): `pnpm exec tsc --noEmit && pnpm run lint`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add web/src/app/matchup/page.tsx
git commit -m "refactor(web): Matchup adopts usePageData four-state machine

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Trades page — adopt hook + four states

**Files:**
- Modify: `web/src/app/trades/page.tsx`

- [ ] **Step 1: Edit imports**

Old (line 3):
```tsx
import { useEffect, useState } from "react";
```
Delete this line (Trades' `Loaded` holds no UI state). Then add, after the existing `import { cn } …` line:
```tsx
import { usePageData } from "@/lib/use-page-data";
import { PageError, PageEmpty } from "@/components/ui/PageStates";
```
(`TrendingUp` is already imported and will be the empty icon.)

> If `pnpm exec tsc --noEmit` later reports `useState` is needed by another component in this file, restore `import { useState } from "react";` (keep `useEffect` removed). The per-task tsc + lint gate makes a wrong guess obvious — unused import = lint error, missing import = tsc error.

- [ ] **Step 2: Convert the existing page component into `Loaded`**

Replace `export default function TradesPage() { … }` (old lines 17–53) with the `Loaded` child (keep the existing inner JSX exactly):

```tsx
function Loaded({ data }: { data: TradesData }) {
  return (
    <motion.div variants={staggerContainer} initial="hidden" animate="show" className="space-y-6">
      <motion.div variants={staggerItem}>
        <Header needs={data.needs} count={data.recs.length} />
      </motion.div>
      <motion.div variants={staggerItem}>
        <div className="space-y-4">
          {data.recs.map((rec) => (
            <TradeCard key={rec.id} rec={rec} />
          ))}
        </div>
      </motion.div>
    </motion.div>
  );
}
```

- [ ] **Step 3: Add the new page shell**

Immediately above `function Loaded`, add:

```tsx
export default function TradesPage() {
  const { state, retry } = usePageData(fetchTrades);

  return (
    <>
      <main className="w-full flex-1 px-5 py-6">
        {state.status === "loading" && <LoadingView />}
        {state.status === "error" && <PageError onRetry={retry} />}
        {state.status === "empty" && (
          <PageEmpty
            icon={TrendingUp}
            title="No trade ideas yet"
            body="We need a bit more league data to surface targets."
          />
        )}
        {state.status === "loaded" && <Loaded data={state.data} />}
      </main>
      {state.status === "loaded" && <Footer freshnessMinutes={9} />}
    </>
  );
}
```

- [ ] **Step 4: Type-check + lint**

Run (from `web/`): `pnpm exec tsc --noEmit && pnpm run lint`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add web/src/app/trades/page.tsx
git commit -m "refactor(web): Trades adopts usePageData four-state machine

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Players page — adopt hook + four states (filter state moves into `Loaded`)

**Files:**
- Modify: `web/src/app/players/page.tsx`

Players has `filter`/`query` UI state and a `rows` `useMemo` derived from `data`. These all move into `Loaded`, where `data` is always defined (so the `if (!data) return []` guard is dropped).

- [ ] **Step 1: Edit imports**

Old (line 3):
```tsx
import { useEffect, useMemo, useState } from "react";
```
New (keep `useMemo` + `useState` for `Loaded`; drop `useEffect`):
```tsx
import { useMemo, useState } from "react";
```
Then add, after the existing `import { cn } …` line:
```tsx
import { usePageData } from "@/lib/use-page-data";
import { PageError, PageEmpty } from "@/components/ui/PageStates";
```
(`Search` is already imported and will be the empty icon. `EmptyState` import stays — the in-table no-results `FATable` still uses it.)

- [ ] **Step 2: Convert the existing page component into `Loaded`**

Replace `export default function PlayersPage() { … }` (old lines 19–83) with the `Loaded` child below. Move the `filter`/`query`/`rows` logic in; `rows` now reads the `data` prop directly (no null guard). Keep the existing `<motion.div …>` JSX exactly:

```tsx
function Loaded({ data }: { data: PlayersData }) {
  const [filter, setFilter] = useState<Filter>("all");
  const [query, setQuery] = useState("");

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    return data.freeAgents.filter((p) => {
      if (filter === "hitters" && !p.hitter) return false;
      if (filter === "pitchers" && p.hitter) return false;
      if (filter === "need" && p.fit !== data.topNeed) return false;
      if (q && !p.name.toLowerCase().includes(q)) return false;
      return true;
    });
  }, [data, filter, query]);

  return (
    <motion.div variants={staggerContainer} initial="hidden" animate="show" className="space-y-6">
      <motion.div variants={staggerItem}>
        <Header />
      </motion.div>
      <motion.div variants={staggerItem}>
        <NeedCallout
          need={data.topNeed}
          count={data.freeAgents.filter((p) => p.fit === data.topNeed).length}
          onShow={() => setFilter("need")}
        />
      </motion.div>
      <motion.div variants={staggerItem}>
        <Card className="p-5">
          <Toolbar
            filter={filter}
            setFilter={setFilter}
            query={query}
            setQuery={setQuery}
            need={data.topNeed}
            shown={rows.length}
            total={data.freeAgents.length}
          />
          <FATable rows={rows} need={data.topNeed} />
        </Card>
      </motion.div>
    </motion.div>
  );
}
```

- [ ] **Step 3: Add the new page shell**

Immediately above `function Loaded`, add:

```tsx
export default function PlayersPage() {
  const { state, retry } = usePageData(fetchPlayers);

  return (
    <>
      <main className="w-full flex-1 px-5 py-6">
        {state.status === "loading" && <LoadingView />}
        {state.status === "error" && <PageError onRetry={retry} />}
        {state.status === "empty" && (
          <PageEmpty
            icon={Search}
            title="No free agents available"
            body="The free-agent pool is empty right now."
          />
        )}
        {state.status === "loaded" && <Loaded data={state.data} />}
      </main>
      {state.status === "loaded" && <Footer freshnessMinutes={9} />}
    </>
  );
}
```

- [ ] **Step 4: Type-check + lint**

Run (from `web/`): `pnpm exec tsc --noEmit && pnpm run lint`
Expected: no errors. (If tsc says `EmptyState` is unused, confirm `FATable` still imports/uses it — it should remain.)

- [ ] **Step 5: Commit**

```bash
git add web/src/app/players/page.tsx
git commit -m "refactor(web): Players adopts usePageData four-state machine

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: Research page — adopt hook + four states (filter state moves into `Loaded`)

**Files:**
- Modify: `web/src/app/research/page.tsx`

Research has `lens`/`query` UI state and a `rows` `useMemo` derived from `data`; all move into `Loaded`.

- [ ] **Step 1: Edit imports**

Old (line 3):
```tsx
import { useEffect, useMemo, useState } from "react";
```
New:
```tsx
import { useMemo, useState } from "react";
```
Then add, after the existing `import { cn } …` line:
```tsx
import { usePageData } from "@/lib/use-page-data";
import { PageError, PageEmpty } from "@/components/ui/PageStates";
```
(`Flame` is already imported and will be the empty icon. `EmptyState` import stays — the in-table no-results `LeaderTable` still uses it.)

- [ ] **Step 2: Convert the existing page component into `Loaded`**

Replace `export default function ResearchPage() { … }` (old lines 31–78) with:

```tsx
function Loaded({ data }: { data: ResearchData }) {
  const [lens, setLens] = useState<Lens>("overall");
  const [query, setQuery] = useState("");

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    return data.leaders.filter((p) => {
      if (lens !== "overall" && p.tag !== lens) return false;
      if (q && !p.name.toLowerCase().includes(q)) return false;
      return true;
    });
  }, [data, lens, query]);

  return (
    <motion.div variants={staggerContainer} initial="hidden" animate="show" className="space-y-6">
      <motion.div variants={staggerItem}>
        <Header />
      </motion.div>
      <motion.div variants={staggerItem}>
        <Card className="p-5">
          <Toolbar lens={lens} setLens={setLens} query={query} setQuery={setQuery} shown={rows.length} />
          <LeaderTable rows={rows} />
        </Card>
      </motion.div>
    </motion.div>
  );
}
```

- [ ] **Step 3: Add the new page shell**

Immediately above `function Loaded`, add:

```tsx
export default function ResearchPage() {
  const { state, retry } = usePageData(fetchResearch);

  return (
    <>
      <main className="w-full flex-1 px-5 py-6">
        {state.status === "loading" && <LoadingView />}
        {state.status === "error" && <PageError onRetry={retry} />}
        {state.status === "empty" && (
          <PageEmpty
            icon={Flame}
            title="No leaders to show"
            body="Leaderboard data isn't available yet."
          />
        )}
        {state.status === "loaded" && <Loaded data={state.data} />}
      </main>
      {state.status === "loaded" && <Footer freshnessMinutes={9} />}
    </>
  );
}
```

- [ ] **Step 4: Type-check + lint**

Run (from `web/`): `pnpm exec tsc --noEmit && pnpm run lint`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add web/src/app/research/page.tsx
git commit -m "refactor(web): Research adopts usePageData four-state machine

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 9: Final gate — production build + full visual sweep

**Files:** none (verification only).

- [ ] **Step 1: Clean type-check + lint across the app**

Run (from `web/`): `pnpm exec tsc --noEmit && pnpm run lint`
Expected: no errors.

- [ ] **Step 2: Production build**

Run (from `web/`): `pnpm build`
Expected: build succeeds with no type/lint errors. (The `?state=` override is inert in this bundle: Next inlines `process.env.NODE_ENV` to `"production"`, so `forcedState()` returns `undefined` and the branch is dead-code-eliminated.)

- [ ] **Step 3: Visual sweep of all states (the deliverable Connor wants to see)**

`preview_start` (cwd `web/`, `pnpm dev`), then for each of the 5 migrated pages (`/optimizer`, `/matchup`, `/trades`, `/players`, `/research`) screenshot:
- `loaded` (no query param)
- `?state=error`
- `?state=empty`

Navigate with `preview_eval`: `window.location.href = '/optimizer?state=error'` etc., and `preview_screenshot` each. Spot-check `?state=loading` on one page (skeleton shows).

Expected per page: loaded renders as before; error shows the card + Retry; empty shows the page-specific copy; Footer is hidden in error/empty/loading.

- [ ] **Step 4: Confirm mock shims untouched**

Run: `git diff --stat master -- web/src/lib`
Expected: **no** changes under `web/src/lib/*-data.ts` (only `use-page-data.ts` is new).

- [ ] **Step 5: Surface results to Connor**

Post the screenshots (via SendUserFile if helpful) and a one-line summary. Do **not** push or merge — the owner gates that.

---

## Self-review (completed by plan author)

- **Spec coverage:** §1 hook → Task 1. §2 PageStates → Task 2. §3 per-page wiring → Tasks 3–8. §4 dev override → Task 1 (`forcedState`). §5 empty copy → Tasks 3–8 (each `PageEmpty`). §6 risks → addressed in hook (`reqId` ref for StrictMode/retry races; `window.location.search` not `useSearchParams`; `NODE_ENV` guard). §7 verification → Task 3 Steps 6–7 + Task 9. §8 out-of-scope → no Task touches shims or in-table no-results. All covered.
- **Placeholder scan:** no TBD/TODO; every code step shows full code; move-refactors show the exact new shell + which existing hooks relocate.
- **Type consistency:** `PageState<T>`/`usePageData` signatures match across Task 1 and all consumers; each page passes its stable `fetchX` (`fetchMyTeam`/`fetchOptimizer`/`fetchMatchup`/`fetchTrades`/`fetchPlayers`/`fetchResearch`) and types its `Loaded` with the matching data type (`MyTeamData`/`OptimizerData`/`MatchupData`/`TradesData`/`PlayersData`/`ResearchData`). `PageError({onRetry})` and `PageEmpty({icon,title,body})` props match every call site.

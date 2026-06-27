# WS2 — Streaming: Date Picker + Matchup-Aware — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On the React Pitcher Streaming page, let the user pick any day in the next 7 (today → today+7) and re-query that day's streamable starts, and make the streaming board's scoring **matchup-aware** so the categories the user needs *this week* (e.g. K when behind in K, ERA/WHIP no longer protected once lost) bias the Stream Score. Additively surface the resolved per-category `urgency` map on the response for display + Bubba.

**Architecture:** The streaming surface is already a single thin router (`GET /api/streaming?date=`) → `StreamingService.get_streaming(date, limit)` → `build_optimizer_context(...)` + `build_stream_board(ctx, target_date)`. Investigation of the real code found the matchup wiring is **mostly already present**: `build_optimizer_context` (src/optimizer/shared_data_layer.py) resolves the live matchup itself at Step 4 (`ctx.live_matchup = yds.get_matchup()`), computes urgency at Step 6 (`compute_urgency_weights`), and sets `ctx.category_weights` at Step 8 — and `build_stream_board` reads `ctx.category_weights` (stream_analyzer.py:554) and threads it into `score_stream_candidate` (stream_analyzer.py:610). So the board is **already** matchup-weighted through the existing path; WS2 makes that explicit/robust, **exposes** the resolved urgency on the contract (additive `StreamingResponse.urgency`), and builds the **date strip** (the genuine user-facing gap — `GET /api/streaming?date=` works but the React page never sends a date and re-fetch isn't wired).

The service seam (`api/services/streaming_service.py`) stays the ONE place importing `src/`. Contract field is added in `api/contracts/streaming.py`; `api/openapi.json` is regenerated via `python scripts/export_openapi.py` (snapshot-guarded by `tests/api/test_openapi_contract.py`). No router change (the `date` query param already exists on `GET /api/streaming`).

**Tech Stack:** Python 3.12 (CI) / FastAPI `0.137.1` + httpx `0.28.1` (PINNED — openapi snapshot guard) + Pydantic v2; Next.js 16 + React 19 + TypeScript 5.9 + Tailwind v4 (`web/`), pnpm. Backend tests: `python -m pytest` (DB-free fake-service + monkeypatch — local DB may be EMPTY). Frontend gate: `pnpm build` + `pnpm exec tsc --noEmit` + `pnpm run lint` from `web/`. Windows PowerShell shell.

**Shared contracts (reuse, do NOT redefine):** `PlayerRef`, `StatItem` in `api/contracts/common.py`; `make_player_ref` in `api/services/player_ref.py`. The streaming candidate/budget/scorecard models in `api/contracts/streaming.py` are unchanged except the additive `urgency` field on `StreamingResponse`.

**Key reference paths (read before editing):**
- `api/services/streaming_service.py` — `StreamingService.get_streaming` builds the ctx at ~line 149 with NO explicit matchup arg (it doesn't need one — the builder resolves it). `_to_candidate` maps board rows.
- `api/routers/streaming.py` — `GET /api/streaming` already has `date: str | None = None`. No change.
- `api/contracts/streaming.py` — `StreamingResponse` (add `urgency`).
- `src/optimizer/category_urgency.py::compute_urgency_weights(matchup, config)` → `{"urgency": {CAT: 0-1}, "rate_modes": {...}, "summary": {...}}`. Returns equal `0.5` for every category when `matchup` is falsy (never raises).
- `src/optimizer/shared_data_layer.py` — `build_optimizer_context` Steps 4/6/8 (matchup → urgency → `category_weights`).
- `src/optimizer/stream_analyzer.py:554,610` — `build_stream_board` consumes `ctx.category_weights`.
- `web/src/app/probables/page.tsx` + `web/src/lib/probables-data.ts` — the EXISTING 7-day pattern (`days: string[]`, `fetchProbables` sends `{ days: 7 }`). Mirror its date-column style.
- `web/src/app/streaming/page.tsx`, `web/src/lib/streaming-data.ts` (`fetchStreaming` currently sends NO date), `web/src/lib/api/adapters.ts` (`apiStreamingToData`), `web/src/lib/use-page-data.ts` (the hook needs a STABLE fetcher — re-fetch on date change uses a `useMemo`'d fetcher).
- `web/src/lib/api/client.ts::apiGet(path, params?)` — already accepts a params object.

**Out of scope (do NOT do):** any `src/` engine edit (engines already do the weighting); the router; FA filtering (already `include_rostered=False` default); new Yahoo paths; the `urgency` is display-only (no scoring re-implementation in the service — the engine owns the math).

---

## Task 1 — Add `urgency` field to `StreamingResponse` contract

**Files:** `api/contracts/streaming.py` (class `StreamingResponse`, lines 88-93); test `tests/api/test_streaming.py` (extend).

- [ ] Write a failing test in `tests/api/test_streaming.py` (append after `test_get_streaming_returns_contract`):
  ```python
  def test_streaming_response_urgency_field_defaults_empty():
      """StreamingResponse carries an additive `urgency` map (per-category 0-1
      this-week need), defaulting to {} so the field is backward-compatible."""
      resp = StreamingResponse(date="2026-06-27")
      assert resp.urgency == {}
      resp2 = StreamingResponse(date="2026-06-27", urgency={"K": 0.82, "ERA": 0.18})
      dumped = resp2.model_dump()
      assert dumped["urgency"] == {"K": 0.82, "ERA": 0.18}
  ```
- [ ] Run: `python -m pytest tests/api/test_streaming.py::test_streaming_response_urgency_field_defaults_empty -q` — expect FAIL (`TypeError: unexpected keyword argument 'urgency'` / attribute missing).
- [ ] Add the field to `StreamingResponse` in `api/contracts/streaming.py`. Change:
  ```python
  class StreamingResponse(BaseModel):
      date: str
      candidates: list[StreamCandidate] = Field(default_factory=list)
      top_pick: StreamCandidate | None = None
      budget: BudgetStrip = Field(default_factory=BudgetStrip)
      probables: list[ProbableStarter] = Field(default_factory=list)
  ```
  to:
  ```python
  class StreamingResponse(BaseModel):
      date: str
      candidates: list[StreamCandidate] = Field(default_factory=list)
      top_pick: StreamCandidate | None = None
      budget: BudgetStrip = Field(default_factory=BudgetStrip)
      probables: list[ProbableStarter] = Field(default_factory=list)
      # Resolved this-week category urgency (CAT -> 0-1), from the live matchup
      # (compute_urgency_weights). Display + Bubba context only; the engine already
      # applied these to the scores. {} when no live matchup.
      urgency: dict[str, float] = Field(default_factory=dict)
  ```
- [ ] Run: `python -m pytest tests/api/test_streaming.py -q` — expect PASS.
- [ ] Commit: `git add api/contracts/streaming.py tests/api/test_streaming.py && git commit -m "feat(streaming): additive urgency map on StreamingResponse contract"`

---

## Task 2 — Resolve + attach `urgency` in the service (matchup-aware, never-raise)

**Files:** `api/services/streaming_service.py` (`StreamingService.get_streaming`, lines 128-178; reuse the `_f` helper at line 29); test `tests/api/test_streaming.py`.

Add a private helper that reads the urgency the builder already computed off `ctx` (`ctx.urgency_weights["urgency"]`), with a `compute_urgency_weights(ctx.live_matchup, config)` recompute as a fallback, finite-coerced via `_f`, and pass it into the `StreamingResponse`. This makes the matchup-awareness explicit and surfaces it for display — WITHOUT re-implementing scoring (the engine already weighted the board).

- [ ] Write a failing DB-free test in `tests/api/test_streaming.py` (append). It monkeypatches the THREE bound names inside `get_streaming` (`build_optimizer_context`, `build_stream_board`, `get_yahoo_data_service`) so no DB/Yahoo is touched, and asserts the resolved urgency flows onto the response:
  ```python
  def test_get_streaming_attaches_urgency_from_ctx(monkeypatch):
      """get_streaming surfaces ctx.urgency_weights['urgency'] (the matchup-derived
      per-category need) on StreamingResponse.urgency, finite-coerced. DB-free:
      the engine + Yahoo singletons are faked at their BOUND import sites."""
      import types
      import api.services.streaming_service as svc

      class _Ctx:
          # what build_optimizer_context would return; the service only reads
          # urgency_weights/category_gaps/adds_remaining_this_week off it.
          urgency_weights = {"urgency": {"K": 0.82, "ERA": 0.18, "BAD": float("nan")}}
          category_gaps = {}
          adds_remaining_this_week = 7
          live_matchup = {"categories": []}

      def _fake_ctx(*a, **k):
          return _Ctx()

      def _fake_board(ctx, date, include_rostered=False):
          import pandas as pd
          return pd.DataFrame()  # empty board → no candidates, urgency still attaches

      # Patch the names the service imports INSIDE get_streaming (function-local imports).
      monkeypatch.setattr(
          "src.optimizer.shared_data_layer.build_optimizer_context", _fake_ctx, raising=False
      )
      monkeypatch.setattr(
          "src.optimizer.stream_analyzer.build_stream_board", _fake_board, raising=False
      )
      monkeypatch.setattr(
          "src.yahoo_data_service.get_yahoo_data_service", lambda: object(), raising=False
      )
      # LeagueConfig import inside the try is real + cheap; leave it.

      out = svc.StreamingService().get_streaming(date="2026-06-27")
      assert out.date == "2026-06-27"
      # NaN dropped, finite kept:
      assert out.urgency == {"K": 0.82, "ERA": 0.18}
  ```
  > NOTE: `get_streaming` uses **function-local imports** (`from src.optimizer.shared_data_layer import build_optimizer_context` etc. inside the `try`), so patching the SOURCE module attribute (`src.optimizer.shared_data_layer.build_optimizer_context`) is correct here — the name is re-looked-up from the source module on each call, not bound at module top. (Contrast the monkeypatch-bound-name rule, which applies to `from x import y` at MODULE top.)
- [ ] Run: `python -m pytest tests/api/test_streaming.py::test_get_streaming_attaches_urgency_from_ctx -q` — expect FAIL (`out.urgency == {}`, field not populated yet).
- [ ] Add the helper + wire it. In `api/services/streaming_service.py`, add a module-level function near `_build_budget`:
  ```python
  def _resolve_urgency(ctx) -> dict[str, float]:
      """The matchup-derived per-category urgency the builder already computed
      (ctx.urgency_weights['urgency']); recompute from ctx.live_matchup as a
      fallback. Finite-coerced; {} on any failure (never raises)."""
      try:
          uw = getattr(ctx, "urgency_weights", None) or {}
          urgency = uw.get("urgency") if isinstance(uw, dict) else None
          if not urgency:
              from src.optimizer.category_urgency import compute_urgency_weights
              from src.valuation import LeagueConfig

              urgency = (compute_urgency_weights(getattr(ctx, "live_matchup", None), LeagueConfig()) or {}).get(
                  "urgency", {}
              )
          out: dict[str, float] = {}
          for cat, val in (urgency or {}).items():
              fv = _f(val, default=float("nan"))
              if not math.isnan(fv):
                  out[str(cat)] = fv
          return out
      except Exception as exc:  # never break the streaming response on urgency
          logger.warning("StreamingService urgency resolve failed: %s", exc)
          return {}
  ```
  Then inside `get_streaming`, initialise `urgency` before the `try` and populate it after `ctx` is built:
  - Add `urgency: dict[str, float] = {}` next to the other initialisers (after `probables: list[ProbableStarter] = []` near line 141).
  - Immediately after `budget = _build_budget(ctx)` (line 160), add: `urgency = _resolve_urgency(ctx)`.
  - Add `urgency=urgency,` to the `StreamingResponse(...)` return (after `probables=probables,`).
- [ ] Run: `python -m pytest tests/api/test_streaming.py -q` — expect PASS (all streaming tests).
- [ ] Run the broader DB-free streaming suite to confirm no regression: `python -m pytest tests/api/test_streaming.py tests/api/test_api_streaming_analyze.py tests/api/test_api_streaming_widen.py -q` — expect PASS.
- [ ] Commit: `git add api/services/streaming_service.py tests/api/test_streaming.py && git commit -m "feat(streaming): resolve + attach matchup urgency on the streaming response"`

---

## Task 3 — Regenerate `api/openapi.json` (snapshot guard)

**Files:** `api/openapi.json` (regenerated, not hand-edited); guard `tests/api/test_openapi_contract.py`.

- [ ] Confirm the guard currently fails (proves the new `urgency` field is unsynced): `python -m pytest tests/api/test_openapi_contract.py -q` — expect FAIL (`api/openapi.json is stale`).
- [ ] Regenerate: `python scripts/export_openapi.py`
- [ ] Run: `python -m pytest tests/api/test_openapi_contract.py -q` — expect PASS.
- [ ] Sanity-check the field landed: `python -c "import json; s=json.load(open('api/openapi.json')); print('urgency' in s['components']['schemas']['StreamingResponse']['properties'])"` — expect `True`.
- [ ] Commit: `git add api/openapi.json && git commit -m "chore(openapi): regenerate for StreamingResponse.urgency"`

---

## Task 4 — Regenerate frontend API types

**Files:** `web/src/lib/api/generated.ts` (regenerated via `pnpm gen:api`). `ApiStreamingResponse` aliases `components["schemas"]["StreamingResponse"]` in `web/src/lib/api/types.ts` (line 18) — no hand-edit; the new `urgency` property appears automatically.

- [ ] From `web/`: `pnpm gen:api` (runs `openapi-typescript ../api/openapi.json -o src/lib/api/generated.ts`).
- [ ] Verify the type carries `urgency`: from `web/`, `pnpm exec tsc --noEmit` — expect PASS (no errors; the new optional field is additive).
- [ ] Sanity-check: `grep -n "urgency" web/src/lib/api/generated.ts` (or Grep) — expect a `StreamingResponse` property hit.
- [ ] Commit: `git add web/src/lib/api/generated.ts && git commit -m "chore(web): regenerate API types for StreamingResponse.urgency"`

---

## Task 5 — `fetchStreaming(date?)` sends the date + exposes urgency

**Files:** `web/src/lib/streaming-data.ts` (`fetchStreaming` lines 231-239; `StreamingData` interface lines 82-88), `web/src/lib/api/adapters.ts` (`apiStreamingToData` lines 157-180).

`fetchStreaming` currently ignores the date and the adapter drops `urgency`. Thread an optional `date` into the live `apiGet` call and surface `urgency` on `StreamingData` for display/Bubba. `web/` has no test runner — the gate is `pnpm build` + `tsc` + `lint`.

- [ ] In `web/src/lib/streaming-data.ts`, add `urgency` to the `StreamingData` interface (after `probables`):
  ```ts
  export interface StreamingData {
    date: string; // canonical YYYY-MM-DD (also the /streaming/analyze POST body); format for DISPLAY only
    budget: BudgetStrip;
    topPick: StreamCandidate | null;
    board: StreamCandidate[];
    probables: ProbableStarter[];
    urgency: Record<string, number>; // this-week per-category need (0-1); {} when no live matchup
  }
  ```
- [ ] Add `urgency: {}` to the `STREAMING` mock object (after its `probables: [...]` array, before the closing `}` of the `export const STREAMING` literal) so the mock satisfies the interface.
- [ ] In `web/src/lib/api/adapters.ts`, add `urgency` to the object `apiStreamingToData` returns (after the `probables: (...).map(...)` block, before the closing `}`):
  ```ts
      urgency: api.urgency ?? {},
  ```
- [ ] Change `fetchStreaming` in `web/src/lib/streaming-data.ts` to accept an optional date and pass it to `apiGet` (params are omitted when no date, preserving today's default behavior). Replace lines 231-239 with:
  ```ts
  export async function fetchStreaming(date?: string, delayMs = 600): Promise<StreamingData | null> {
    return liveOrMock(
      async () => {
        const api = await apiGet<ApiStreamingResponse>("/streaming", date ? { date } : undefined);
        return (api.candidates?.length ?? 0) > 0 ? apiStreamingToData(api) : null;
      },
      () => new Promise<StreamingData>((resolve) => setTimeout(() => resolve(STREAMING), delayMs)),
    );
  }
  ```
  > NOTE: the signature change (date as first arg) means the page must no longer pass `fetchStreaming` bare to `usePageData` — Task 6 wires a date-keyed fetcher. The default-export call sites are only `web/src/app/streaming/page.tsx` (updated in Task 6); no other caller passes a delay positionally.
- [ ] Verify nothing else calls `fetchStreaming` with a positional delay: Grep `fetchStreaming(` across `web/src` — expect only the import in `page.tsx` (the call site is rewired in Task 6).
- [ ] From `web/`: `pnpm exec tsc --noEmit` — expect PASS.
- [ ] Commit: `git add web/src/lib/streaming-data.ts web/src/lib/api/adapters.ts && git commit -m "feat(web): fetchStreaming sends date + surfaces urgency"`

---

## Task 6 — 7-day date strip on the Streaming page (default today, refetch on select)

**Files:** `web/src/app/streaming/page.tsx` (whole file), new `web/src/components/streaming/DateStrip.tsx`. Mirror the Probables 7-day column convention (today … today+7, ISO `YYYY-MM-DD`, formatted via the existing `formatStreamDate`).

`usePageData` requires a STABLE fetcher (an inline arrow re-fires the effect every render). Re-fetch on date change uses a `useMemo`'d fetcher keyed on the selected date, so the effect re-runs exactly when the date changes (and only then). Default = today (ISO, local).

- [ ] Create `web/src/components/streaming/DateStrip.tsx` — a presentational 8-button strip (today + next 7). Build day ISO strings from a LOCAL date (no UTC offset, matching `formatStreamDate`'s local-parse). Highlight the selected day; label via `formatStreamDate`.
  ```tsx
  "use client";

  import { formatStreamDate } from "@/lib/streaming-data";

  /** today + the next 7 days as canonical YYYY-MM-DD (LOCAL date, no UTC shift). */
  export function next7Days(): string[] {
    const out: string[] = [];
    const base = new Date();
    base.setHours(0, 0, 0, 0);
    for (let i = 0; i <= 7; i++) {
      const d = new Date(base);
      d.setDate(base.getDate() + i);
      const yyyy = d.getFullYear();
      const mm = String(d.getMonth() + 1).padStart(2, "0");
      const dd = String(d.getDate()).padStart(2, "0");
      out.push(`${yyyy}-${mm}-${dd}`);
    }
    return out;
  }

  export function DateStrip({
    days,
    selected,
    onSelect,
  }: {
    days: string[];
    selected: string;
    onSelect: (d: string) => void;
  }) {
    return (
      <div className="flex flex-wrap gap-2" role="tablist" aria-label="Streaming date">
        {days.map((d, i) => {
          const active = d === selected;
          return (
            <button
              key={d}
              type="button"
              role="tab"
              aria-selected={active}
              onClick={() => onSelect(d)}
              className={
                "rounded-xl border px-3 py-2 text-[13px] font-semibold transition-colors " +
                (active
                  ? "border-orange bg-orange text-white"
                  : "border-line bg-white text-ink-2 hover:border-orange/50 hover:text-navy")
              }
            >
              {i === 0 ? "Today" : formatStreamDate(d)}
            </button>
          );
        })}
      </div>
    );
  }
  ```
  > NOTE: verify the Tailwind token class names (`border-orange`, `bg-orange`, `text-navy`, `text-ink-2`, `border-line`) against `web/src/app/globals.css` `@theme` / existing streaming components (e.g. `StreamBoard.tsx`, `BudgetStrip.tsx`). Swap to whatever the codebase already uses for the orange-accent selected pill + the neutral border (the Probables page / `PageStates` are reference). Do NOT introduce raw hex.
- [ ] Rewrite `web/src/app/streaming/page.tsx` to own the selected-date state, build a date-keyed stable fetcher with `useMemo`, render the `DateStrip` above the content, and feed the memoized fetcher to `usePageData`:
  ```tsx
  "use client";

  import { useMemo, useState } from "react";
  import { motion } from "framer-motion";
  import { CalendarDays } from "lucide-react";
  import { fetchStreaming, formatStreamDate, type StreamingData } from "@/lib/streaming-data";
  import { staggerContainer, staggerItem } from "@/lib/motion";
  import { usePageData } from "@/lib/use-page-data";
  import { Footer } from "@/components/chrome/Footer";
  import { Skeleton } from "@/components/ui/Skeleton";
  import { PageError, PageEmpty } from "@/components/ui/PageStates";
  import { BudgetStrip } from "@/components/streaming/BudgetStrip";
  import { TopPickCallout } from "@/components/streaming/TopPickCallout";
  import { StreamBoard } from "@/components/streaming/StreamBoard";
  import { AnalyzeStarter } from "@/components/streaming/AnalyzeStarter";
  import { DateStrip, next7Days } from "@/components/streaming/DateStrip";

  export default function StreamingPage() {
    const days = useMemo(() => next7Days(), []);
    const [selected, setSelected] = useState(days[0]);
    // Stable per-date fetcher: identity changes only when `selected` changes,
    // so usePageData's effect re-runs exactly on date selection (not every render).
    const fetcher = useMemo(() => () => fetchStreaming(selected), [selected]);
    const { state, retry } = usePageData(fetcher);
    return (
      <>
        <main className="w-full flex-1 px-5 py-6">
          <div className="mb-5 space-y-3">
            <div>
              <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-ink-3">
                Daily · {formatStreamDate(selected)}
              </div>
              <h1 className="font-display text-3xl font-extrabold text-navy">Pitcher Streaming</h1>
              <p className="mt-1 text-[13px] text-ink-2">
                Heat-ranked probable starters, scored on matchup, park, form, skill, and win odds — weighted by this
                week&apos;s category needs.
              </p>
            </div>
            <DateStrip days={days} selected={selected} onSelect={setSelected} />
          </div>
          {state.status === "loading" && <LoadingView />}
          {state.status === "error" && <PageError onRetry={retry} />}
          {state.status === "empty" && (
            <PageEmpty
              icon={CalendarDays}
              title="No streamable starts"
              body="No probable starters are posted for this date yet — probables typically appear 1–5 days out."
            />
          )}
          {state.status === "loaded" && <Loaded data={state.data} />}
        </main>
        {state.status === "loaded" && <Footer freshnessMinutes={12} />}
      </>
    );
  }

  function Loaded({ data }: { data: StreamingData }) {
    return (
      <motion.div variants={staggerContainer} initial="hidden" animate="show" className="space-y-6">
        <motion.div variants={staggerItem}>
          <BudgetStrip budget={data.budget} />
        </motion.div>
        {data.topPick && (
          <motion.div variants={staggerItem}>
            <TopPickCallout pick={data.topPick} />
          </motion.div>
        )}
        <motion.div variants={staggerItem}>
          <StreamBoard board={data.board} />
        </motion.div>
        <motion.div variants={staggerItem}>
          <AnalyzeStarter probables={data.probables} date={data.date} />
        </motion.div>
      </motion.div>
    );
  }

  function LoadingView() {
    return (
      <div className="space-y-6">
        <Skeleton className="h-14 w-72" />
        <div className="grid gap-3 sm:grid-cols-3">
          <Skeleton className="h-24 rounded-2xl" />
          <Skeleton className="h-24 rounded-2xl" />
          <Skeleton className="h-24 rounded-2xl" />
        </div>
        <Skeleton className="h-80 w-full rounded-2xl" />
      </div>
    );
  }
  ```
  > NOTE: the eyebrow/heading moved OUT of `Loaded` into the always-rendered header so the date strip and title stay visible across `loading`/`error`/`empty` (so the user can pick another day from an empty day without losing the control). `AnalyzeStarter` keeps receiving `data.date` (the canonical ISO the analyze POST needs) — unchanged.
- [ ] From `web/`: `pnpm exec tsc --noEmit` — expect PASS.
- [ ] From `web/`: `pnpm run lint` — expect PASS (watch the react-hooks rule: the `useMemo` fetcher pattern is intentional; if the linter flags the inline-arrow-in-useMemo, keep the dependency array `[selected]` — that is the stable-by-date contract).
- [ ] From `web/`: `pnpm build` — expect PASS (the production gate; `web/` has no CI).
- [ ] Commit: `git add web/src/app/streaming/page.tsx web/src/components/streaming/DateStrip.tsx && git commit -m "feat(web): 7-day date strip on the Streaming page (refetch per day)"`

---

## Task 7 — Final verification + integration sweep

**Files:** none (verification only).

- [ ] Backend suite (DB-free streaming + contract + openapi guards): `python -m pytest tests/api/test_streaming.py tests/api/test_api_streaming_analyze.py tests/api/test_api_streaming_widen.py tests/api/test_openapi_contract.py tests/api/test_no_logic_in_routers.py -q` — expect PASS.
- [ ] Confirm the router stayed logic-free (the no-logic-in-routers AST guard already in the line above covers `api/routers/streaming.py`; no router edit was made — re-confirm by reading `api/routers/streaming.py` is unchanged).
- [ ] Frontend gate from `web/`: `pnpm exec tsc --noEmit && pnpm run lint && pnpm build` — expect PASS.
- [ ] (Live smoke — optional, needs the main-checkout 26 MB DB or the live API, NOT the empty worktree DB) Start the API: `python -m uvicorn api.main:create_app --factory --port 8000`, then `curl "http://localhost:8000/api/streaming?date=<today>"` and `curl "http://localhost:8000/api/streaming?date=<today+3>"`; confirm (a) different dates return different boards, (b) `urgency` is populated when a live matchup exists (empty `{}` is the correct cold/no-matchup degradation), and (c) when behind in K the high-K streamers rank higher than when not. Document any live-only degradation (no Yahoo locally → `urgency: {}` and a thin board) rather than treating it as a failure.
- [ ] Finish per the standing rule: merge to local `master` and push `origin/master`. Run `superpowers:requesting-code-review` / the silent-failure-hunter on the service change (the `_resolve_urgency` never-raise + NaN-drop path is the one place worth a retrospective scan).

---

## Notes for the integrator

- `api/main.py` is **not** touched by WS2 (no new route — `GET /api/streaming` already exists). No mount change.
- `api/openapi.json` + `web/src/lib/api/generated.ts` are the only generated files WS2 changes; per the parallel-reconcile lesson, if another workstream also regenerated them, resolve by RE-running `python scripts/export_openapi.py` then `pnpm gen:api` (never hand-merge).
- `web/src/lib/api/adapters.ts` gets a one-line append (`urgency:` in `apiStreamingToData`) — disjoint from WS1/WS4 adapter appends; integrator reconciles by keeping all appends.
- The `urgency` field is display/Bubba context only — WS5's auto page-context will pick it up for free once Streaming publishes via `usePageData`. No coordination needed beyond shipping the field.

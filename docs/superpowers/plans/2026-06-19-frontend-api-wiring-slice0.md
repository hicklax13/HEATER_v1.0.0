# Frontend↔API Wiring Slice 0 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the frontend can consume the live FastAPI end-to-end by wiring the Research page to `/api/leaders` (Next proxy + typed client + adapter + mock fallback, env-flag-gated), and produce the per-page contract-gap spec that tells the backend track what to extend.

**Architecture:** A `next.config` rewrite proxies same-origin `/api/*` to FastAPI (no CORS, no backend change). A tiny typed client + adapter convert the minimal API leaders response into the richer frontend `ResearchData`. `fetchResearch()` gains a live path gated by `NEXT_PUBLIC_HEATER_LIVE`, falling back to the existing mock on error/empty/flag-off.

**Tech Stack:** Next.js 16 (App Router) · React 19 · TypeScript 5.9 · pnpm (frontend). FastAPI + uvicorn (backend, run only — never edited).

**Spec:** `docs/superpowers/specs/2026-06-19-frontend-api-wiring-slice0-design.md`
**Branch:** `feat/web-api-wiring-slice0` (already created; spec committed).

---

## Verification model (read first)

`web/` has **no unit-test runner** — the per-task gate is `pnpm exec tsc --noEmit`
+ `pnpm run lint` (from `web/`), plus Claude_Preview screenshots and live curls.
`pnpm build` runs at milestones. **Never edit `api/` or `src/`** — the API is run,
not modified. All pnpm commands run from `C:\Users\conno\Code\HEATER_v1.0.1\web`;
uvicorn/curl run from the repo root.

### File structure (locked here)

- **Create** `web/src/lib/api/types.ts` — TS mirrors of the API's leaders contract.
- **Create** `web/src/lib/api/client.ts` — `apiGet<T>` over the proxied `/api/*`.
- **Create** `web/src/lib/api/adapters.ts` — `apiLeadersToResearch()`.
- **Create** `docs/superpowers/specs/2026-06-19-frontend-api-contract-gaps.md` — the gap spec.
- **Modify** `web/next.config.ts` — add the `/api/*` rewrite.
- **Modify** `web/src/lib/research-data.ts` — flag-gated live path + fallback.
- **Modify** `.claude/launch.json` — add a `heater-web-live` preview config (env flag).
- **Untouched:** `api/`, `src/`, every other `web/src/lib/*-data.ts`, all page components.

---

## Task 1: Confirm the API boots and serves `/api/leaders` locally (gating)

This gates the live-data proof. If the API won't boot or returns empty, the proof
degrades to "client + proxy + fallback verified; API-up path unverified" — note it
and continue (the plumbing is still proven).

**Files:** none (recon).

- [ ] **Step 1: Boot the API in the background**

Run (from repo root):
```bash
.venv/Scripts/python -m uvicorn api.main:create_app --factory --host 127.0.0.1 --port 8000
```
Run it in the background (it's a long-lived server). Expected: a line like
`Uvicorn running on http://127.0.0.1:8000`. If it crashes on import, capture the
error and report BLOCKED (the engine import failed) — do not edit `api/`/`src/`.

- [ ] **Step 2: Confirm it serves leaders with real data**

Run (from repo root):
```bash
curl -s "http://127.0.0.1:8000/healthz"
curl -s "http://127.0.0.1:8000/api/leaders?category=HR&limit=10"
```
Expected: `{"status":"ok"}` then JSON shaped
`{"category":"HR","rows":[{"rank":1,"player":{"id":...,"name":"...","positions":"..."},"value":...}, ...]}`.

- [ ] **Step 3: Record the outcome**

If `rows` is non-empty → the live-data proof is achievable; pick this category
(`HR`) for the wire. If `rows` is empty (DB lacks 2026 `season_stats`) → the proof
will show the **fallback** path; note it and proceed (the wire still works).
Keep the API running for Task 6.

---

## Task 2: Add the Next.js `/api/*` proxy rewrite

**Files:**
- Modify: `web/next.config.ts`

- [ ] **Step 1: Confirm the Next 16 rewrites API**

Read `node_modules/next/dist/docs/` (or the `NextConfig` type) to confirm
`async rewrites()` returning `[{ source, destination }]` is current in Next 16.
(`web/AGENTS.md` warns this Next has breaking changes.)

- [ ] **Step 2: Write the rewrite**

Replace the entire contents of `web/next.config.ts` with:
```ts
import type { NextConfig } from "next";

const API_BASE = process.env.HEATER_API_BASE ?? "http://localhost:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${API_BASE}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
```

- [ ] **Step 3: Type-check + lint + build**

Run (from `web/`): `pnpm exec tsc --noEmit && pnpm run lint && pnpm build`
Expected: all green; build still prerenders the 6 routes (the rewrite doesn't add routes).

- [ ] **Step 4: Commit**

```bash
git add web/next.config.ts
git commit -m "feat(web): proxy /api/* to FastAPI via next.config rewrite

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Typed API client + leaders types

**Files:**
- Create: `web/src/lib/api/types.ts`
- Create: `web/src/lib/api/client.ts`

- [ ] **Step 1: Write the API types** (mirror `api/contracts/leaders.py` + `common.py`)

Create `web/src/lib/api/types.ts`:
```ts
/** TypeScript mirrors of the FastAPI contracts this slice consumes.
 *  Source of truth: api/contracts/leaders.py + api/contracts/common.py.
 *  (Full-surface generation from openapi.json is future work — see the gap spec.) */

export interface ApiPlayerRef {
  id: number;
  name: string;
  positions: string;
  yahoo_player_key?: string | null;
}

export interface ApiLeaderRow {
  rank: number;
  player: ApiPlayerRef;
  value: number;
}

export interface ApiLeadersResponse {
  category: string;
  rows: ApiLeaderRow[];
}
```

- [ ] **Step 2: Write the client**

Create `web/src/lib/api/client.ts`:
```ts
/** Minimal typed client for the HEATER API. Calls SAME-ORIGIN `/api/*`, which
 *  next.config rewrites to the FastAPI backend (so the browser sees no CORS). */

export async function apiGet<T>(
  path: string,
  params?: Record<string, string | number>,
): Promise<T> {
  const qs = params
    ? "?" + new URLSearchParams(Object.entries(params).map(([k, v]) => [k, String(v)])).toString()
    : "";
  const res = await fetch(`/api${path}${qs}`, { headers: { Accept: "application/json" } });
  if (!res.ok) throw new Error(`API ${path} -> ${res.status}`);
  return (await res.json()) as T;
}
```

- [ ] **Step 3: Type-check + lint**

Run (from `web/`): `pnpm exec tsc --noEmit && pnpm run lint`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add web/src/lib/api/types.ts web/src/lib/api/client.ts
git commit -m "feat(web): typed API client + leaders contract types

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Adapter — API leaders → frontend `ResearchData`

**Files:**
- Create: `web/src/lib/api/adapters.ts`

The frontend `LeaderRow` (`web/src/lib/research-data.ts`) needs fields the API
doesn't provide. Fill them with honest defaults; the lossiness is the gap evidence.
Frontend `LeaderRow` = `{ name, pos, teamAbbr, teamId, mlbId, rank, hitter,
stats: string[], value, trend, tag, note }` (NOTE: the field is `note`, not `why`,
and `stats` is `string[]`; `tag` is the required enum `"hot"|"cold"|"breakout"|"sell"`).

- [ ] **Step 1: Write the adapter**

Create `web/src/lib/api/adapters.ts`:
```ts
import type { ApiLeadersResponse } from "@/lib/api/types";
import type { LeaderRow, ResearchData } from "@/lib/research-data";

const PITCHER_POS = /\b(P|SP|RP)\b/;

function isHitter(positions: string): boolean {
  return positions ? !PITCHER_POS.test(positions) : true;
}

/** Format the single category figure the API returns (rate cats keep decimals). */
function fmt(value: number, category: string): string {
  const rate = ["AVG", "OBP"].includes(category)
    ? value.toFixed(3)
    : ["ERA", "WHIP"].includes(category)
      ? value.toFixed(2)
      : String(Math.round(value));
  return `${rate} ${category}`;
}

/** Map the minimal API leaders response → the richer frontend ResearchData,
 *  filling unavailable fields with honest placeholders (no headshot, one stat,
 *  forced tag). Surfaces the contract gap visibly. */
export function apiLeadersToResearch(api: ApiLeadersResponse): ResearchData {
  const leaders: LeaderRow[] = api.rows.map((r) => ({
    name: r.player.name,
    pos: r.player.positions,
    teamAbbr: "", // gap: API leaders has no team
    teamId: 0,
    mlbId: 0, // gap: no mlbId -> PlayerAvatar shows its initials fallback
    rank: r.rank,
    hitter: isHitter(r.player.positions),
    stats: [fmt(r.value, api.category)], // gap: API gives one figure, not 3
    value: Math.max(20, 100 - (r.rank - 1) * 3), // gap: no 0-100 score -> derive from rank
    trend: "flat", // gap: API has no trend
    tag: "hot", // gap: API has no tag; forced placeholder so the row renders
    note: `Live · ${api.category} leader`,
  }));
  return { leaders };
}
```

- [ ] **Step 2: Type-check + lint**

Run (from `web/`): `pnpm exec tsc --noEmit && pnpm run lint`
Expected: no errors. (Confirms the adapter output exactly satisfies the frontend
`LeaderRow` type — a wrong/missing field fails here.)

- [ ] **Step 3: Commit**

```bash
git add web/src/lib/api/adapters.ts
git commit -m "feat(web): adapter mapping API leaders -> frontend ResearchData

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Wire `fetchResearch()` with a flag-gated live path + fallback

**Files:**
- Modify: `web/src/lib/research-data.ts`

- [ ] **Step 1: Replace `fetchResearch` with the live-or-mock version**

In `web/src/lib/research-data.ts`, replace the existing `fetchResearch` function
(the `setTimeout`/`resolve(RESEARCH)` one at the bottom) with:
```ts
import { apiGet } from "@/lib/api/client";
import { apiLeadersToResearch } from "@/lib/api/adapters";
import type { ApiLeadersResponse } from "@/lib/api/types";

/** Mock by default. With NEXT_PUBLIC_HEATER_LIVE=1, fetch live leaders from the
 *  API and adapt them; fall back to the mock on any error or empty response. */
export async function fetchResearch(delayMs = 600): Promise<ResearchData> {
  if (process.env.NEXT_PUBLIC_HEATER_LIVE === "1") {
    try {
      const api = await apiGet<ApiLeadersResponse>("/leaders", { category: "HR", limit: 25 });
      if (api.rows.length > 0) return apiLeadersToResearch(api);
    } catch {
      // fall through to mock
    }
  }
  return new Promise((resolve) => setTimeout(() => resolve(RESEARCH), delayMs));
}
```
Put the three `import` lines at the TOP of the file with the other imports (not
inside the function). Keep everything else in the file (the `RESEARCH` mock, the
types, `lr`) unchanged.

- [ ] **Step 2: Type-check + lint**

Run (from `web/`): `pnpm exec tsc --noEmit && pnpm run lint`
Expected: no errors.

- [ ] **Step 3: Build (flag-off path is the default build)**

Run (from `web/`): `pnpm build`
Expected: green; `/research` still prerenders static (the live fetch is client-side
at runtime, behind the flag — it doesn't affect static generation).

- [ ] **Step 4: Commit**

```bash
git add web/src/lib/research-data.ts
git commit -m "feat(web): Research fetch — flag-gated live /api/leaders + mock fallback

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: End-to-end verification (the proof)

**Files:**
- Modify: `.claude/launch.json` (add a live-mode preview config)

- [ ] **Step 1: Add a live preview config**

In `.claude/launch.json`, add this object to the `configurations` array (leave the
existing entries untouched):
```json
{
  "name": "heater-web-live",
  "runtimeExecutable": "pnpm",
  "runtimeArgs": ["-C", "web", "dev"],
  "port": 3000,
  "env": { "NEXT_PUBLIC_HEATER_LIVE": "1", "HEATER_API_BASE": "http://localhost:8000" }
}
```

- [ ] **Step 2: Verify flag-OFF is unchanged**

`preview_start` the normal `heater-web` config → navigate to `/research` →
`preview_screenshot`. Expected: the full mock leaderboard (Judge, Witt… with
headshots/tags), identical to today. Then `preview_stop`.

- [ ] **Step 3: Verify flag-ON live data** (requires the API from Task 1 running)

`preview_start` the `heater-web-live` config → navigate to `/research` →
`preview_screenshot`. Expected (if Task 1 returned rows): live HR leaders from the
DB — real player names + rank-derived value bars, **initials avatars** (no
headshots), note "Live · HR leader". Confirm via `preview_network` that
`/api/leaders?category=HR&limit=25` returned 200 with rows.

- [ ] **Step 4: Verify API-down fallback**

Stop the uvicorn API (kill the Task 1 process). Reload `/research` in the
`heater-web-live` preview → `preview_screenshot`. Expected: the mock leaderboard
(the `catch` fell back). `preview_stop`.

- [ ] **Step 5: Commit**

```bash
git add .claude/launch.json
git commit -m "chore(preview): add heater-web-live config (NEXT_PUBLIC_HEATER_LIVE)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Write the per-page contract-gap spec (deliverable)

**Files:**
- Create: `docs/superpowers/specs/2026-06-19-frontend-api-contract-gaps.md`

- [ ] **Step 1: Gather each page's two shapes**

For each of the 6 pages, read the frontend mock module and the matching API
contract:
| Page | Frontend module | API contract | API endpoint |
|------|-----------------|--------------|--------------|
| Team | `web/src/lib/data.ts` (`MyTeamData`) | `api/contracts/my_team.py` | `/api/me/team` |
| Optimizer | `web/src/lib/optimizer-data.ts` | `api/contracts/lineup.py` | `/api/lineup/optimize` |
| Matchup | `web/src/lib/matchup-data.ts` | `api/contracts/matchup.py` | `/api/matchup` |
| Trades | `web/src/lib/trades-data.ts` | `api/contracts/trade_finder.py` | `/api/trade-finder` |
| Players | `web/src/lib/players-data.ts` | `api/contracts/free_agents.py` | `/api/free-agents` |
| Research | `web/src/lib/research-data.ts` | `api/contracts/leaders.py` | `/api/leaders` |

- [ ] **Step 2: Write the gap doc**

Create `docs/superpowers/specs/2026-06-19-frontend-api-contract-gaps.md` with:
- A header (Date, Track, Status, purpose: "input for the backend track / Sub-project B contract extensions").
- A **cross-cutting** section: every player needs `mlb_id` (headshots) + `team`
  (`teamAbbr`/`teamId`); the frontend uses richer per-player stat triples and
  display scores the contracts lack.
- A **per-page table**: columns `Frontend field` | `In API?` | `Gap / needed extension`.
- These three are pre-analyzed — include them verbatim, then complete the other
  three (Optimizer, Matchup, Trades) by reading their files from Step 1:

  **Research** (`/api/leaders`): API `LeaderRow{rank, player{id,name,positions}, value}`
  vs frontend `{rank, name, pos, teamAbbr, teamId, mlbId, hitter, value(0–100),
  trend, tag, note, stats[3]}`. Gaps: no `mlbId`/`team` (no headshot), no
  0–100 score (only a raw stat), no `trend`/`tag`/`note`/3-stat triple; per-category
  vs the frontend's "overall" + hot/cold/breakout/sell lenses.

  **Players** (`/api/free-agents`): API returns add/drop **recs**
  `{add, drop, marginal_value, categories_helped, rationale}` — a different concept
  from the frontend's ranked FA pool `{rank, value 0–100, 3 stats, fit, tag,
  hitter, ownPct}`. Gap: needs a ranked-pool endpoint (or the page consumes recs +
  a separate pool list); plus `mlb_id`/`team`/per-player stats.

  **Team** (`/api/me/team`): API `{team_name, record, rank, matchup(win/tie/loss),
  categories[{cat,you,opp,edge,win_prob,inverse}]}` vs frontend `MyTeamData`
  (12+ fields). Gaps: `movers`, `lever`, `ops`, `trajectory`, `winProbTrend`,
  `statusChips`, `eyebrow/subline/freshnessMinutes`, matchup logos/deltas.

- A short **prioritization** note: which extensions unblock the most pages
  (mlb_id + team are universal; they should come first).

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-06-19-frontend-api-contract-gaps.md
git commit -m "docs: per-page frontend<->API contract-gap spec (Sub-project C input)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: Final gate

**Files:** none.

- [ ] **Step 1: Clean type-check + lint + build**

Run (from `web/`): `pnpm exec tsc --noEmit && pnpm run lint && pnpm build`
Expected: all green; 6 routes static.

- [ ] **Step 2: Confirm no backend files changed**

Run (from repo root): `git diff --stat master -- api/ src/`
Expected: **empty** (no `api/`/`src/` changes on this branch).

- [ ] **Step 3: Surface the result**

Post the three screenshots (flag-off mock, flag-on live, API-down fallback) and a
one-line summary to Connor. Do **not** push/merge — owner gates that.

---

## Self-review (completed by plan author)

- **Spec coverage:** §A proxy → Task 2. §B client/types → Task 3. §C adapter →
  Task 4; live-path wire → Task 5. §D gap spec → Task 7. §E verification → Tasks 1
  + 6 + 8. Out-of-scope (no api/src edits) enforced by Task 8 Step 2. All covered.
- **Placeholder scan:** complete code for every new file; the gap-doc task gives 3
  pre-analyzed rows verbatim + the exact files to read for the other 3 (a research
  deliverable, not a hand-wave).
- **Type consistency:** `apiGet`, `ApiLeadersResponse`/`ApiLeaderRow`/`ApiPlayerRef`
  (Task 3) are consumed unchanged in the adapter (Task 4) and the wire (Task 5);
  the adapter returns exactly the frontend `LeaderRow`/`ResearchData` shape (Task 4
  Step 2's tsc gate proves it); `NEXT_PUBLIC_HEATER_LIVE` + `HEATER_API_BASE`
  match between next.config (Task 2), the shim (Task 5), and the launch config
  (Task 6).
- **Correction applied vs spec:** frontend field is `note` (not `why`) and `stats`
  is `string[]`; `tag` is a required enum so the adapter forces `"hot"` — all
  reflected in Task 4's code.

# HEATER Web — Frontend↔API Wiring, Slice 0 (design spec)

**Date:** 2026-06-19
**Track:** CMO / frontend (`web/`) — first step of Sub-project C (page migration)
**Status:** approved direction; pending implementation plan

## Goal

Two deliverables that together start Sub-project C honestly:

1. **Gap spec** — a per-page analysis of the frontend's mock shapes vs the
   built API contracts, documenting exactly what the backend (Sub-project B)
   must extend for full data parity. This is the artifact that unblocks the
   real migration.
2. **Plumbing proof** — the in-lane frontend plumbing (typed API client + a
   Next.js proxy for CORS) wiring **one** page (Research) end-to-end to the
   live FastAPI `/api/leaders`, with an adapter and a mock fallback, gated
   behind an env flag so the default stays 100% mock.

## Context — why this isn't a clean swap (the finding)

CLAUDE.md says "mock shape = API contract," and the B-slice contract docstrings
say "the frontend's types are generated from this OpenAPI." **Neither is true.**
The CMO hand-built richer mocks independently; the B-slice contracts are minimal.
They diverged:

- **Players:** frontend wants a ranked FA pool (`{rank, value 0–100, 3 stats,
  fit, tag, hitter, ownPct}`); `/api/free-agents` returns add/drop **recs**
  (`{add, drop, marginal_value, categories_helped, rationale}`) — a different
  concept.
- **Team:** frontend wants `{movers, lever, ops, trajectory, winProbTrend,
  statusChips, …}` (12+ fields); `/api/me/team` returns a subset
  (`{record, rank, matchup(win/tie/loss), categories}`).
- **Research:** frontend `LeaderRow` = `{rank, name, pos, teamAbbr, teamId,
  mlbId, value, trend, tag, why, stats[3]}`; `/api/leaders` `LeaderRow` =
  `{rank, player{id,name,positions}, value}` — no `mlbId` (no headshot), no
  team/tag/trend/why/stats, and it's **per-category** vs the frontend's overall
  leaderboard.

Operational realities:
- **No CORS** middleware on the FastAPI app — a browser cross-origin call is
  blocked.
- Yahoo-dependent services (`team`, `lineup`, `matchup`, `trade-finder`,
  `free-agents`) **degrade to empty** without a live Yahoo connection, so they
  return nothing locally. DB-backed services (`leaders`, verified) read the
  SQLite DB (which has data) and **return real data locally**.

This is why **Research → `/api/leaders`** is the proof page: it's the one
wireable page whose endpoint returns real data in a cold local environment.

## Non-negotiables

- **No edits to `api/` or `src/`** (backend / CEO-track lane). CORS is solved
  on the frontend via a proxy; no backend change.
- Default behavior unchanged: with the live flag **off**, every page (incl.
  Research) is byte-for-byte the current mock experience.
- The mock shims (`web/src/lib/*-data.ts`) remain the fallback — never deleted.
- `pnpm build` + `pnpm exec tsc --noEmit` + `pnpm run lint` stay green.
- Next 16 caveat (`web/AGENTS.md`): confirm the rewrites API against
  `node_modules/next/dist/docs/` before writing `next.config`.

---

## A. CORS — Next.js proxy

A rewrite in `web/next.config.ts` (or `.mjs` — match the existing file):

```
async rewrites() {
  return [{
    source: "/api/:path*",
    destination: `${process.env.HEATER_API_BASE ?? "http://localhost:8000"}/api/:path*`,
  }];
}
```

The browser calls **same-origin** `/api/...`; Next proxies to FastAPI. No CORS,
no backend change. In production the destination becomes the deployed API URL
via `HEATER_API_BASE`. The frontend has no `/api` routes of its own, so the
rewrite collides with nothing.

## B. Typed API client — `web/src/lib/api/`

- `web/src/lib/api/client.ts` — `apiGet<T>(path: string, params?: Record<string,
  string | number>): Promise<T>`. Builds a query string, `fetch`es the proxied
  `/api${path}`, throws on non-2xx, returns parsed JSON typed as `T`.
- `web/src/lib/api/types.ts` — TypeScript models mirroring the API contracts
  used this slice (`LeadersResponse`, `LeaderRow`, `PlayerRef`). Hand-written
  to match `api/contracts/leaders.py` + `common.py`; the gap spec (§D) tracks
  full-surface generation as future work.

## C. Wire Research — `web/src/lib/research-data.ts`

`fetchResearch()` gains a **live path**, gated by `NEXT_PUBLIC_HEATER_LIVE === "1"`:

- Flag **off** (default) → return the existing mock `RESEARCH` exactly as today.
- Flag **on** → `apiGet<LeadersResponse>("/leaders", { category, limit })`,
  run it through an **adapter** `apiLeaders → ResearchData`, and **fall back to
  the mock** on any error or empty `rows`.

Adapter (`web/src/lib/api/adapters.ts`) maps the minimal API row to the richer
frontend `LeaderRow`, filling unavailable fields with honest defaults:
- `rank`←rank, `name`←player.name, `pos`←player.positions
- `value`←a **rank-derived 0–100 display score** (the frontend `value` is a
  normalized score the API does not provide — surface it from rank so the heat
  ramp still works; the missing real value-score is a gap-spec item)
- `stats`←`[{ value: <the API stat value>, label: <category> }]` — the one
  category figure the API returned (e.g. `24` / `HR`)
- `mlbId`/`teamId`←0 (no headshot — `PlayerAvatar` shows its initials fallback),
  `teamAbbr`←"", `trend`←"flat", `tag`←none/omitted, `why`←""

The lossiness (no headshot, one stat, no tag/why) is the visible evidence of the
gap the backend must close.

Category choice: the frontend's "overall" lens has no single-category analog, so
live mode fetches a representative category (e.g. `HR`) for the proof and labels
it; the overall-vs-per-category semantics are a gap-spec item.

## D. The gap spec (deliverable) — `docs/superpowers/specs/2026-06-19-frontend-api-contract-gaps.md`

A table per frontend page: **frontend shape → API contract → missing fields →
recommended API extension.** Covers all 6 pages (Team, Optimizer, Matchup,
Trades, Players, Research) by reading each `api/contracts/*.py` against each
`web/src/lib/*-data.ts`. Explicitly calls out cross-cutting needs: every
player needs `mlbId` (for headshots) and `team`; per-page semantic mismatches
(free-agents recs vs FA pool; leaders per-category vs overall). Written as
guidance **for the backend track** — not a commitment to change `api/` here.

## E. Verification

- `pnpm build` + `tsc --noEmit` + `lint` green (flag off path).
- Start the API: `uvicorn api.main:create_app --factory --port 8000` (from repo
  root, `.venv`). Confirm `GET /api/leaders?category=HR&limit=25` returns
  non-empty JSON.
- `next dev` with `NEXT_PUBLIC_HEATER_LIVE=1 HEATER_API_BASE=http://localhost:8000`
  → Research renders **live DB leaders** (real names + values, initials avatars).
  Screenshot.
- Stop the API → reload → Research **falls back to the mock**. Screenshot.
- Confirm flag-off Research is unchanged.

## Out of scope (YAGNI / not this slice)

- Wiring any page other than Research (the rest are Yahoo-dependent / bigger
  gaps; they wait on the API extension the gap spec defines).
- Any change to `api/` or `src/` (CORS, contracts, auth, new fields) — that's
  CEO-track work the gap spec feeds.
- OpenAPI-driven type generation, auth/Clerk, write endpoints, production
  deploy config beyond the `HEATER_API_BASE` env seam.

## Files touched

- **New:** `web/src/lib/api/client.ts`, `web/src/lib/api/types.ts`,
  `web/src/lib/api/adapters.ts`, `docs/superpowers/specs/2026-06-19-frontend-api-contract-gaps.md`.
- **Edited:** `web/next.config.*` (add rewrite), `web/src/lib/research-data.ts`
  (live path + fallback).
- **Untouched:** `api/`, `src/`, all other `web/src/lib/*-data.ts`, all pages
  (Research's page component is unchanged — only its data shim gains a live path).

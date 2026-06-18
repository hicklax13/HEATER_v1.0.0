# HEATER re-platform — Sub-project B: Backend API + data foundation (FastAPI)

> **Status:** Draft for review · **Date:** 2026-06-16 · **Lane:** CEO / Product (technical migration)
> **Pairs with:** `2026-06-16-heater-frontend-foundation-design.md` (Sub-project A, CMO) and `2026-06-16-heater-brand-design-vision.md` (brand). This spec owns the **backend half** of the same re-platform; A owns the frontend, B (this) owns the API + data + infra.
> **North Star:** HEATER becomes a real, monetized, public consumer product (`CLAUDE.md` ★ NORTH STAR + memory `project_north_star_monetized_product`).

---

## 1. Why this exists

HEATER's analytics run inside Streamlit on a **single Railway replica backed by SQLite**, with an **in-process scheduler as the sole writer** and **one hardcoded league** (FourzynBurn, Yahoo `game_key 469`). Those are the three hard ceilings (`CLAUDE.md`) that make a public, monetized product impossible:

1. SQLite + single-writer + single-replica → cannot serve many concurrent users.
2. One hardcoded league → cannot be multi-tenant.
3. Heavy per-session Streamlit compute → expensive and slow per user.

This sub-project stands up the **production backend**: a FastAPI service that exposes the existing Python engines as a typed JSON API, backed by **Postgres + background workers + real auth + multi-tenancy**, consumed by the Sub-project A frontend. It is the "thin client" re-platform `CLAUDE.md` named as deferred and the North Star now green-lights.

### Load-bearing principle (shared with A): keep the brain, replace the plumbing
The value is the Python analytics — `src/engine/` (6-phase trade pipeline), `src/optimizer/` (21-module LP), `src/valuation.py` (SGP/VORP/SGPCalculator), Bayesian projections, Monte Carlo + playoff sims, and ~6,300 passing tests. **None of that logic is rewritten.** B wraps it in an API and swaps its *data access* (SQLite → Postgres) and *execution model* (in-process → workers). The engine modules stay the source of truth.

---

## 2. Program context (shared decomposition)

| # | Sub-project | Lane | This spec? |
|---|---|---|---|
| A | Design system + frontend foundation (Next.js, 3 pages, mock data) | CMO | no |
| **B** | **Backend API + data foundation (FastAPI, Postgres, auth, workers, multi-platform data)** | **CEO** | **this** |
| C | Page migration (×14 React pages wired to the API) | joint | no |
| D | Cutover, hosting, multi-tenancy rollout, billing-live, user migration, retire Streamlit | CEO | no |

**The convergence seam:** A's pages call typed functions in `web/lib/data/*` (e.g. `getMyTeam()`, `getFreeAgents(filters)`, `evaluateTrade(...)`) that return fixtures today. **B implements the real endpoints behind those exact function signatures.** B's first deliverable is therefore to *formalize that contract* (OpenAPI ↔ `lib/data/types.ts`) so frontend and backend cannot drift.

---

## 3. Goals & non-goals

### Goals
1. A **FastAPI** service exposing the existing engines as typed JSON endpoints matching A's `lib/data/types.ts` data seam.
2. **Postgres (Neon)** replacing SQLite, with a **multi-tenant schema** (per-user / per-league isolation) and Alembic migrations.
3. **Background workers (Redis + Arq)** replacing the in-process sole-writer scheduler: data refresh + heavy compute (MC, playoff sims) run as jobs.
4. **Real auth** (Clerk) — signup / login / JWT — replacing the MULTI_USER admin-approval mechanism.
5. **Multi-platform league connection** behind one interface — Yahoo, ESPN, Sleeper, CBS + manual import (your #1 product requirement), on the **commercial-safe-core** data posture.
6. **Strangler-fig coexistence:** the live Streamlit app keeps serving the 12 users throughout; it becomes the API's *first consumer* to prove the contract before A's frontend arrives.
7. **Feature-gating hooks** for the 2-tier Free→Pro model (gate compute-heavy edge endpoints).

### Non-goals (other sub-projects)
- The React UI, brand, design tokens, component kit (A + brand vision).
- Page-by-page migration of all 14 pages (C).
- Production cutover, retiring Streamlit, billing-live/Stripe checkout, full multi-tenant user onboarding (D).
- The marketing/landing site (CMO D3).
- Rewriting any analytics engine.

---

## 4. Architecture

```
                         ┌──────────────────────────────────────┐
  Sub-project A (CMO) →   │  Next.js frontend  (web/, Vercel)     │
                         └───────────────────┬──────────────────┘
                                             │  typed JSON  (contract == lib/data/types.ts)
  This spec (B) ───────────────────────────► │
                         ┌───────────────────▼──────────────────┐
                         │  FastAPI service  (api/, Railway)     │  ← thin: auth, routing,
                         │  routers · auth (Clerk JWT) · gating  │     serialization only
                         └───────────────────┬──────────────────┘
                                             │  in-process calls (no logic added)
                         ┌───────────────────▼──────────────────┐
                         │  PYTHON ENGINES — UNCHANGED           │  src/engine, src/optimizer,
                         │  trade · optimizer · valuation · sims │  src/valuation, bayesian, MC
                         └───────────────────┬──────────────────┘
                                             │  repository layer (NEW: replaces get_connection)
                  ┌──────────────────────────┼───────────────────────────┐
        ┌─────────▼─────────┐     ┌──────────▼──────────┐      ┌──────────▼──────────┐
        │ Postgres (Neon)   │     │ Redis + Arq workers │      │ League connectors   │
        │ multi-tenant      │     │ refresh + heavy MC  │      │ Yahoo/ESPN/Sleeper/ │
        │ Alembic           │     │ (sole-writer → jobs)│      │ CBS + manual import │
        └───────────────────┘     └─────────────────────┘      └─────────────────────┘

  Strangler-fig: the existing Streamlit app points at the FastAPI service as the
  FIRST consumer (proves the contract with a UI we already trust) before A's
  frontend cuts over. Streamlit retires in Sub-project D.
```

### 4.1 API layer (FastAPI)
- **Thin by rule:** routers do auth → validate → call an engine function → serialize. **No analytics logic in the API layer** (mirrors the structural discipline that kept logic out of `pages/`). A guard test enforces "no SGP/category math in `api/routers/`."
- Endpoints map 1:1 to A's data seam. Initial set (read-first):
  - `GET /api/me/team` → `getMyTeam()` (roster, record, matchup, category standings)
  - `GET /api/free-agents?filters` → `getFreeAgents()` (FA recommender output)
  - `POST /api/lineup/optimize` → optimizer pipeline (async job; returns job id → poll/stream)
  - `POST /api/trade/evaluate` → `evaluate_trade(...)` (async job for the MC/playoff path)
  - `GET /api/standings`, `GET /api/matchup`, `GET /api/leaders`, `GET /api/players/{id}` …
- **Contract source of truth:** OpenAPI schema generated from FastAPI/Pydantic models; A's `lib/data/types.ts` is generated from (or checked against) it. Contract tests fail CI on drift.

### 4.2 Data layer (Postgres + repository)
- Introduce a **repository abstraction** that the engines call instead of `get_connection()` directly. Today `get_connection()` (SQLite, WAL) is the single sanctioned access path; B generalizes it to a Postgres-backed implementation while keeping a SQLite implementation for local dev/tests. The engines depend on the repository interface, not the driver.
- **Schema migration:** port the SQLite schema to Postgres with **Alembic** migrations. Most SQL is portable; SQLite-specific bits (e.g. `INSERT OR REPLACE`, bytes-typing workarounds, `PRAGMA`) are translated. The `players`/`league_rosters`/`league_standings`/`statcast_archive`/`refresh_log` tables and the `_build_player_pool` query are the core to port.
- **Multi-tenancy:** add `tenant_id` (and/or `league_id`) scoping to every league-specific table; row-level isolation so user A never reads user B's league. The single-league hardcoding (game_key 469, FourzynBurn) becomes per-tenant config.

### 4.3 Execution model (workers)
- **Redis + Arq** job queue. The in-process scheduler (`src/scheduler.py`, today the sole SQLite writer) becomes a **worker** running `bootstrap_all_data` per tenant on a schedule. Postgres handles concurrent writes, so the single-replica/single-writer invariant is **lifted**.
- Heavy synchronous compute moves to jobs: `POST /lineup/optimize` and the MC/playoff `POST /trade/evaluate` enqueue a job, return a job id; the frontend polls or subscribes (SSE) — this is the production version of A's "async optimize" UX and kills the request-thread freezes (the same class as the 44.8s MC freeze and the live Railway renderer stalls).

### 4.4 Auth + gating
- **Clerk** (hosted auth): email/password + OAuth providers, JWT sessions. Replaces the MULTI_USER admin-approval flow. The FastAPI layer validates the JWT and resolves `tenant_id`.
- **Feature gating** for Free→Pro: a dependency on the compute-heavy endpoints (`/trade/evaluate` MC path, `/lineup/optimize`, playoff odds, FA recommender, AI chat) checks the user's tier. Free tier gets standings/scores/basic roster; Pro unlocks the edge (matches the CMO monetization-presentation spec §5). Stripe checkout itself is Sub-project D.

### 4.5 Multi-platform data + commercial-safe core
- A **`LeagueConnector` interface** with implementations: `SleeperConnector` (open API, lowest legal risk), `YahooConnector` (existing OAuth, user-authorized), `ESPNConnector`, `CBSConnector`, and `ManualImportConnector` (always-safe universal fallback). The current `yahoo_data_service` 3-tier cache generalizes into this interface.
- **Commercial-safe core** (decided 2026-06-16): the projection/data core runs on legally-clean sources (MLB Stats API + HEATER's own Marcel/proprietary projections); FanGraphs' 7-system blend + FantasyPros ECR are *optional enrichment* behind a switch. Lead with the safer connectors; always offer manual import.

---

## 5. Sequencing within Sub-project B

Strangler-fig, contract-first, read-before-write:

1. **B0 — Contract.** Formalize the API contract from A's `lib/data/types.ts` → OpenAPI/Pydantic models. Stand up an empty FastAPI app returning the A fixtures verbatim (so A can develop against a *real* server immediately).
2. **B1 — Read endpoints over the engines.** Wire `GET /me/team`, `/free-agents`, `/standings`, `/matchup`, `/leaders`, `/players/{id}` to the existing engines (still reading SQLite via the repository interface). No DB migration yet.
3. **B2 — Postgres + repository + Alembic.** Implement the Postgres repository, migrate the schema, port `_build_player_pool`. Dual-run (SQLite local/tests, Postgres staging).
4. **B3 — Workers.** Move `bootstrap_all_data` + heavy compute (`optimize`, `trade/evaluate`) onto Arq/Redis jobs; add job-status endpoints.
5. **B4 — Multi-tenancy + auth.** Add `tenant_id` scoping + Clerk JWT; the `LeagueConnector` interface; per-tenant league config.
6. **B5 — Strangler proof.** Point the existing Streamlit app at the FastAPI service as its first consumer for 1–2 pages; verify parity with live data. (De-risks the contract before A cuts over.)
7. **Handoff to C/D.** A consumes the live API (C); cutover/hosting/billing-live (D).

Each step is independently shippable and leaves the live Streamlit app running.

---

## 6. Decisions reconciled (CEO calls — flag any to change)

| Decision | Choice | Note |
|---|---|---|
| Frontend+backend stack | **Next.js + FastAPI + Postgres** | Confirmed 2026-06-16; matches A + business plan |
| Monetization tiers | **2-tier Free → Pro** (~$8.99/mo, ~$79/yr) | Adopts CMO recommendation over the April 3-tier; simpler for solo ops. **Owner-confirmed 2026-06-17.** |
| Data licensing posture | **Commercial-safe core** | Own projections core; FanGraphs droppable; multi-platform + manual import |
| League connection | **Multi-platform** (Sleeper/Yahoo/ESPN/CBS + manual) | Your #1 product requirement |
| Hosting | Vercel (frontend, A) · **Railway (FastAPI + workers)** · **Neon (Postgres)** · **Upstash (Redis)** | Matches business-plan §5.2 |
| Auth | **Clerk** | Hosted; replaces MULTI_USER |
| Launch target | **2027 MLB season** (paid ~Opening Day) | Confirmed 2026-06-16 |

---

## 7. Testing
- The ~6,300 engine tests **stay green** (engines unchanged; repository interface keeps a SQLite impl for tests).
- New: **API contract tests** (every endpoint's response validates against the OpenAPI/`types.ts` shape), **tenant-isolation tests** (a request scoped to tenant A never returns tenant B rows), **worker tests** (job enqueue → status → result), and a **"no analytics logic in `api/routers/`"** structural guard (mirrors the existing `pages/` discipline).

---

## 8. Risks & open questions
- **SQLite→Postgres SQL portability** — some raw SQL is SQLite-specific; the repository abstraction is the mitigation, but auditing every query is real work. (Medium.)
- **Engine ↔ `get_connection` coupling** — engines call `get_connection()` widely; introducing the repository interface touches many modules (mechanical, but broad). (Medium.)
- **Contract drift A↔B** — mitigated by generating one side from the other + CI contract tests. (Low if automated.)
- **ESPN/CBS connector legality** — undocumented APIs; lead with Sleeper/Yahoo/manual, treat ESPN/CBS as best-effort. (Tracked in the data posture.)
- **Two sessions, one working tree** — CMO writes live in `web/` + brand docs; B lives in `api/` + `src/` repository layer. Coordinate shared-file edits (e.g. this spec set, `CLAUDE.md`). (Process.)
- **Multi-tenant data volume** — per-tenant bootstrap × N users multiplies refresh load; the worker queue + per-tenant schedules must be rate-aware. (Revisit in D at scale.)

---

## 9. What happens after this spec
1. Owner reviews this spec (+ confirms the 2-tier monetization call in §6).
2. On approval → `writing-plans` produces the task-by-task implementation plan for **Sub-project B**, sequenced B0→B5.
3. Build `api/` + the repository layer alongside the untouched Streamlit app; the Streamlit app becomes the first API consumer (B5) before A cuts over.
4. Then Sub-projects C (page migration) and D (cutover, hosting, billing-live).

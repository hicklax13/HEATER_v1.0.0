# HEATER — Migration Roadmap to a Monetized Public Product

**Date:** 2026-06-19
**Lane:** CEO / Product strategy (sequences both the CEO/backend and CMO/frontend tracks)
**Status:** approved structure; the master sequencing index the ★ NORTH STAR named as "not yet written"

## Purpose

This is the **single sequenced plan** from where HEATER is today (a private single-league Streamlit app) to **first paying public users**, and then to scale. The North Star (`CLAUDE.md`, memory `project_north_star_monetized_product`) set the destination; the program was decomposed into 4 sub-projects (A frontend / B backend / C page-migration / D cutover); several phase plans exist. **What was missing was the order.** This doc supplies it.

It is an **index + sequence**, not a re-statement: each milestone points at the spec/plan that already owns its detail, and names the gaps that still need a plan.

## The three decisions that shape everything (owner-set 2026-06-19)

1. **Launch strategy = revenue-first, lean-infra.** Get to a proven, sellable product on the **current infrastructure** (SQLite, single Railway replica, Yahoo-only) before building the heavy infra (Postgres, workers, multi-tenancy, multi-platform). Validate the stack before over-building — matches the solo-operator / low-overhead North Star.
2. **Beta scope = full 14-page parity.** The beta ships a *complete* product — all 14 Streamlit surfaces ported to React — not a subset. This makes page-migration (M1) the largest item on the critical path; accepted deliberately for a credible product.
3. **First paid beta = single-league, not public.** The first beta puts the **existing FourzynBurn league (the 12 users)** on the new React product behind Clerk + Stripe, on current infra — **no multi-tenancy**. Its purpose is to prove the whole new stack (React parity + strangler-fig cutover + billing) end-to-end with real users *before* the multi-tenant infra bet; the dollars aren't the point (friends ≠ market). **Real market validation comes at the public launch (M5).**

> Why #3 (resolves a real conflict the CEO design correctly flags): multi-tenancy is a *functional* prerequisite for "public," not a scale nicety — the app is hardcoded to one league (game_key 469), so *every public user connecting their own league* needs per-tenant isolation, which realistically needs Postgres + workers (per-tenant data refresh would crush single-writer SQLite). So multi-tenancy belongs **before the public launch (M4 → M5)**, not bolted onto the single-league beta. Decisions #1/#2 are also two **different axes**, not a contradiction: the beta is **feature-complete** (14 pages) on **lean plumbing** (current infra) — *full features, thin plumbing.*

## Load-bearing principles (unchanged across all milestones)

- **Keep the brain, replace the plumbing.** The Python analytics (`src/engine`, `src/optimizer`, `src/valuation`, ~6,300 tests) are never rewritten — they're wrapped by the API and have their data access / execution model swapped underneath. (From the B foundation design.)
- **Strangler-fig.** The live Streamlit app keeps serving the 12 users through the entire migration; it is retired only at the very end (M5). Nothing goes dark.
- **Two tracks, one contract.** CEO/backend owns `api/`+`src/`; CMO/frontend owns `web/`. The **API contract is the shared artifact** both build to. (This session proved they drift when the contract isn't enforced — M0 fixes that.)
- **Owner-gated infra.** Anything touching the LIVE data path (Postgres/workers/auth) does not start without an agreed per-phase plan (the B2–B4 plans already honor this).

## Current state (2026-06-19)

| Sub-project | State |
|---|---|
| **A — frontend foundation** | Next.js app, 6 pages, design system, state-machine, uplift — **built** (`web/`, on master). |
| **B — backend API + data** | API **slices 1–9 built** (FastAPI over the engines, serves real data on SQLite). Postgres (B2), workers (B3), Clerk-auth/multi-tenancy (B4): **plans written, NOT executed** (owner-gated). |
| **C — page migration** | **Just started** — slice 0 (Research→`/api/leaders`) proven end-to-end; the contract gap is documented and is the blocker. |
| **D — cutover/launch** | Not started. |

---

## The milestones

**Critical path to a proven sellable product (stack + cutover + billing, on the existing league) = M0 → M1 → M2 → M3.** **Public market revenue requires M4 (multi-tenant backend) → M5 (public launch).** M4 is gated on M3's success + the owner's go.

### M0 — Contract lock *(foundation; unblocks everything)*
**Outcome:** the frontend and backend agree on one data contract, enforced automatically so they can't drift again.
- Extend the API contracts (`api/contracts/*.py`) to match what the frontend renders, per the gap spec `docs/superpowers/specs/2026-06-19-frontend-api-contract-gaps.md`. **Universal first:** add `mlb_id` + team to `PlayerRef` (unblocks headshots/logos on every page). Then the per-page semantic fixes (free-agents pool endpoint, leaders overall-vs-category, the Team dashboard sub-objects).
- **Generate the frontend types from the API's OpenAPI** (replace the hand-written `web/src/lib/api/types.ts` + the mock-shape drift) so the contract is machine-enforced going forward.
- **Same intent as the CEO design's B0** ("formalize the contract from the frontend's types") — M0 does the B0 step that got skipped (the cause of this session's divergence).
- **Lane:** backend-led (`api/`) + a frontend type-gen step. New plan needed.

### M1 — Page parity (Sub-project C) *(the big lift)*
**Outcome:** all **14** Streamlit surfaces exist as React pages wired to the live API, on **current infra**.
- The 6 built pages: swap mocks for the real client (the slice-0 pattern: proxy → typed client → adapter → fallback). The ~8 remaining surfaces (Closer Monitor, Pitcher Streaming, standalone Standings, Punt Analyzer, Draft Simulator, + any others) are built in React and wired.
- Per page: confirm/extend its endpoint (M0 contract) → build the React page → wire live → verify. Streamlit stays live throughout.
- **Lane:** joint. Largest milestone. Each page (or cluster) gets its own plan.

### M2 — Auth + billing *(launch enablers)*
**Outcome:** real accounts + the ability to charge — single-tenant.
- **Clerk** auth (sign-up / log-in / JWT) replacing the MULTI_USER admin-approval mechanism — the *authentication* half is planned (`docs/superpowers/plans/2026-06-19-heater-backend-clerk-auth-wiring-plan.md`); the tenant-resolution half is M4.
- **Stripe** Free→Pro billing + feature-gating the compute-heavy endpoints (the 2-tier model). Billing is per-user (Clerk user → Stripe customer) and needs no multi-tenancy. New plan needed.
- **Lane:** backend-led + frontend auth/billing UI.

### M3 — 🚀 Single-league paid beta (Sub-project D, beta cut)
**Outcome:** the new stack proven end-to-end with real users, billing live — on the *existing* league.
- Deploy React (Vercel) + the API (Railway) on **current infra** (SQLite / single replica / Yahoo, **single hardcoded league**). The **existing FourzynBurn 12** move onto the React product behind Clerk + Stripe — **no multi-tenancy needed**.
- **Purpose = stack + cutover + billing validation**, not revenue: prove React parity, the strangler-fig cutover, and the Stripe plumbing work for real users before the multi-tenant infra bet. (Real market validation is M5.)
- Streamlit still runs (fallback) until M5.
- **Lane:** CEO. New launch plan needed (deploy config, the `HEATER_API_BASE`/env seam, beta onboarding).

### M4 — Multi-tenant production backend *(the public prerequisite)*
**Outcome:** the backend can serve *many users, each with their own league* — the functional gate for a public product.
- **Postgres** (B2) → **workers** (B3, Redis/Arq) → **multi-tenancy + tenant-resolution auth** (B4: `tenant_id`/`league_id` scoping, per-tenant league config replacing the hardcoded game_key 469, per-user Yahoo OAuth), in that order. Plans written + owner-gated (`...b2-postgres-migration-plan.md`, `...b3-workers-plan.md`, `...clerk-auth-wiring-plan.md`).
- **Why here (after the single-league beta, before the public launch):** Postgres + workers lift the single-writer/single-replica ceiling that multi-tenancy's per-tenant data refresh would otherwise crush; multi-tenancy is what makes "public" possible at all. This is exactly the **CEO design's B2 → B3 → B4 order** — just sequenced *after* the stack is proven (M3) rather than before any of it.
- **Lane:** CEO. Plans exist; each phase owner-green-lit.

### M5 — Public launch (Sub-project D, full)
**Outcome:** publicly available to anyone (their own league), multi-platform, Streamlit retired — and the **first real market validation** (strangers paying).
- Open signup on the M4 multi-tenant backend; **multi-platform league connection** (ESPN / Sleeper / CBS + manual import — the B-design's "#1 product requirement" and the likely differentiator) behind the one interface.
- Full public launch + marketing/landing site (CMO D3); **retire Streamlit** entirely.
- **Lane:** CEO + CMO. New plans needed.

---

## Sequence at a glance

```
  PROVE THE STACK  ──────────────────────────►     GO PUBLIC  ──────────────►
  M0 ───► M1 ──────────► M2 ──────► M3 🚀          M4 ──────────► M5 🌐
 contract  14-page      auth +    single-league   Postgres +     public launch +
  lock     parity (C)   billing   beta (the 12,   workers +      multi-platform,
                                   current infra)  multi-tenancy  retire Streamlit
                                                   (public prereq)
```

## What this roadmap does NOT do

- It does not re-specify A or B (their design docs stand) or re-plan B2/B3/Clerk (their plans stand) — it **orders** them.
- It does not commit dates — it commits **sequence + gates**. The owner green-lights each milestone (and M4's sub-phases) when ready.
- It does not pick the Stripe tiers' exact pricing or the multi-platform launch set — those are decisions inside M2 / M5's own plans.

## Immediate next step

**M0 (contract lock)** is the unblock and the next thing to plan + execute. Its input is ready (this session's gap spec). Recommend: write the M0 plan next — extend `PlayerRef` (`mlb_id`+team) + the per-page contract fixes + OpenAPI→frontend type generation — then M1 page-migration can proceed page by page.

## Index of governing docs

- North Star: `CLAUDE.md` ★ + memory `project_north_star_monetized_product`.
- A (frontend): `docs/superpowers/specs/2026-06-16-heater-frontend-foundation-design.md` + brand `...-brand-design-vision.md`.
- B (backend): `docs/superpowers/specs/2026-06-16-heater-backend-api-foundation-design.md` (B0–B5 phasing).
- Contract gap (M0 input): `docs/superpowers/specs/2026-06-19-frontend-api-contract-gaps.md`.
- C slice 0 (the wiring pattern): `docs/superpowers/specs/2026-06-19-frontend-api-wiring-slice0-design.md`.
- M4 infra plans (owner-gated): `...-b2-postgres-migration-plan.md`, `...-b3-workers-plan.md`, `...-clerk-auth-wiring-plan.md`.

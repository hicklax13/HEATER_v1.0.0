# HEATER — Migration Roadmap to a Monetized Public Product

**Date:** 2026-06-19
**Lane:** CEO / Product strategy (sequences both the CEO/backend and CMO/frontend tracks)
**Status:** approved structure; the master sequencing index the ★ NORTH STAR named as "not yet written"

## Purpose

This is the **single sequenced plan** from where HEATER is today (a private single-league Streamlit app) to **first paying public users**, and then to scale. The North Star (`CLAUDE.md`, memory `project_north_star_monetized_product`) set the destination; the program was decomposed into 4 sub-projects (A frontend / B backend / C page-migration / D cutover); several phase plans exist. **What was missing was the order.** This doc supplies it.

It is an **index + sequence**, not a re-statement: each milestone points at the spec/plan that already owns its detail, and names the gaps that still need a plan.

## The two decisions that shape everything (owner-set 2026-06-19)

1. **Launch strategy = revenue-first thin beta.** Get to first paying users on the **current infrastructure** (SQLite, single Railway replica, Yahoo-only). Build the heavy infra (Postgres, workers, multi-tenancy, multi-platform) **after** the beta, gated by real user growth — so we validate willingness-to-pay before over-building. Matches the solo-operator / low-overhead North Star.
2. **Beta scope = full 14-page parity.** The paid beta ships a *complete* product — all 14 Streamlit surfaces ported to React — not a subset. This makes page-migration (M1) the largest item on the critical path; accepted deliberately for a credible paid product.

> These are two **different axes**, not a contradiction: the beta is **feature-complete** (all 14 pages) running on **lean infrastructure** (current SQLite/single-replica, no Postgres/workers yet) — *full features, thin plumbing.* The plumbing hardens in M4 once growth demands it.

## Load-bearing principles (unchanged across all milestones)

- **Keep the brain, replace the plumbing.** The Python analytics (`src/engine`, `src/optimizer`, `src/valuation`, ~6,300 tests) are never rewritten — they're wrapped by the API and have their data access / execution model swapped underneath. (From the B foundation design.)
- **Strangler-fig.** The live Streamlit app keeps serving the 12 users through the entire migration; it is retired only at the very end (M5). Nothing goes dark.
- **Two tracks, one contract.** CEO/backend owns `api/`+`src/`; CMO/frontend owns `web/`. The **API contract is the shared artifact** both build to. (This session proved they drift when the contract isn't enforced — M0 fixes that.)
- **Owner-gated infra.** Anything touching the LIVE data path (Postgres/workers/auth) does not start without an agreed per-phase plan (the B2–B4 plans already honor this).

## Current state (2026-06-19)

| Sub-project | State |
|---|---|
| **A — frontend foundation** | Next.js app, 6 pages, design system, state-machine, uplift — **built** (`web/`, on master). |
| **B — backend API + data** | API **slices 1–9 built** (FastAPI over the engines, serves real data on SQLite). Postgres (B2), workers (B3), Clerk-auth (B4): **plans written, NOT executed** (owner-gated). |
| **C — page migration** | **Just started** — slice 0 (Research→`/api/leaders`) proven end-to-end; the contract gap is documented and is the blocker. |
| **D — cutover/launch** | Not started. |

---

## The milestones

**Critical path to first revenue = M0 → M1 → M2 → M3.** M4–M5 are post-revenue scaling.

### M0 — Contract lock *(foundation; unblocks everything)*
**Outcome:** the frontend and backend agree on one data contract, enforced automatically so they can't drift again.
- Extend the API contracts (`api/contracts/*.py`) to match what the frontend renders, per the gap spec `docs/superpowers/specs/2026-06-19-frontend-api-contract-gaps.md`. **Universal first:** add `mlb_id` + team to `PlayerRef` (unblocks headshots/logos on every page). Then the per-page semantic fixes (free-agents pool endpoint, leaders overall-vs-category, the Team dashboard sub-objects).
- **Generate the frontend types from the API's OpenAPI** (replace the hand-written `web/src/lib/api/types.ts` + the mock-shape drift) so the contract is machine-enforced going forward.
- **Why first:** every page in M1 needs the contract it consumes. This session's divergence is the cost of skipping it.
- **Lane:** backend-led (`api/`) + a frontend type-gen step. New plan needed.

### M1 — Page parity (Sub-project C) *(the big lift)*
**Outcome:** all **14** Streamlit surfaces exist as React pages wired to the live API, on **current infra**.
- The 6 built pages: swap mocks for the real client (the slice-0 pattern: proxy → typed client → adapter → fallback). The ~8 remaining surfaces (Closer Monitor, Pitcher Streaming, standalone Standings, Punt Analyzer, Draft Simulator, + any others) are built in React and wired.
- Per page: confirm/extend its endpoint (M0 contract) → build the React page → wire live → verify. Streamlit stays live throughout.
- **Lane:** joint. Largest milestone. Each page (or cluster) gets its own plan.

### M2 — Auth + billing *(launch enablers)*
**Outcome:** real accounts + the ability to charge.
- **Clerk** auth (sign-up / log-in / JWT) replacing the MULTI_USER admin-approval mechanism — plan exists (`docs/superpowers/plans/2026-06-19-heater-backend-clerk-auth-wiring-plan.md`).
- **Stripe** Free→Pro billing + feature-gating the compute-heavy endpoints (the 2-tier model). New plan needed.
- **Lane:** backend-led + frontend auth/billing UI.

### M3 — 🚀 Paid beta launch (Sub-project D, beta cut)
**Outcome:** first paying users.
- Deploy React (Vercel) + the API (Railway) on **current infra** (SQLite / single replica / Yahoo). Invite a **small N** of paying beta users (N capped by the single-replica ceiling). Validate willingness-to-pay; gather feedback.
- Streamlit still runs (fallback / the original 12 users) until M5.
- **Lane:** CEO. New launch plan needed (deploy config, the `HEATER_API_BASE`/env seam, beta onboarding).

### M4 — Infra hardening *(post-beta; growth-gated)*
**Outcome:** the product scales past the single-replica ceiling — **only when user growth demands it.**
- **Postgres** (B2) → **workers** (B3, Redis/Arq) → **multi-tenancy** (B4), in that order. Plans written + owner-gated (`...b2-postgres-migration-plan.md`, `...b3-workers-plan.md`).
- **Why after the beta:** these are the three "hard ceilings," but they only bite at concurrency the beta won't have. Building them pre-revenue is the over-build the revenue-first choice avoids.
- **Lane:** CEO. Plans exist; each phase green-lit by the owner when growth requires it.

### M5 — Full public launch (Sub-project D, full)
**Outcome:** publicly available, multi-platform, Streamlit retired.
- **Multi-platform league connection** (ESPN / Sleeper / CBS + manual import — the B-design's "#1 product requirement" and the likely differentiator) behind the one interface.
- Full public launch + marketing/landing site (CMO D3); **retire Streamlit** entirely.
- **Lane:** CEO + CMO. New plans needed.

---

## Sequence at a glance

```
  FIRST REVENUE  ──────────────────────────►        SCALE  ──────────────►
  M0 ───► M1 ──────────► M2 ──────► M3 🚀          M4 ──────────► M5 🌐
 contract  14-page      auth +    paid beta       Postgres/      multi-platform
  lock     parity (C)   billing   (current        workers/       + full public,
                                   infra)          multi-tenant   retire Streamlit
                                                   (growth-gated)
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
- B (backend): `docs/superpowers/specs/2026-06-16-heater-backend-api-foundation-design.md`.
- Contract gap (M0 input): `docs/superpowers/specs/2026-06-19-frontend-api-contract-gaps.md`.
- C slice 0 (the wiring pattern): `docs/superpowers/specs/2026-06-19-frontend-api-wiring-slice0-design.md`.
- M4 infra plans (owner-gated): `...-b2-postgres-migration-plan.md`, `...-b3-workers-plan.md`, `...-clerk-auth-wiring-plan.md`.

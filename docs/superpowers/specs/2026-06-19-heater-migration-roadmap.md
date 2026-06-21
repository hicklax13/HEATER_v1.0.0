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
- **★ BACKEND COMPLETE (2026-06-20)** — Clerk JWT verification (built on `...-clerk-auth-wiring-plan.md`), Stripe Free→Pro billing (**hard gate, $7.99/mo, 7-day trial**), and a reusable `require_pro` gate on the 6 compute-heavy endpoints — all shipped, additive + **dormant until env-configured** (`docs/superpowers/{specs,plans}/2026-06-20-heater-m2-*`; memory `project_m2_auth_billing`). Tenant-resolution auth is still M4.
- **NEXT → M2 frontend (login / signup / pricing / upgrade / account UI)** in `web/` (CMO build) against the already-shipped M2 backend contract (Clerk React SDK `getToken()` → `Authorization: Bearer` → the API; `POST /api/billing/checkout-session`, `GET /api/billing/subscription`, the 402 Pro-gate → upgrade prompt). **Already spec'd by the design track:** `docs/superpowers/specs/2026-06-20-heater-m2-frontend-auth-billing-design.md` (env-gated dormancy, conditional `<ClerkProvider>`, Bearer-on-fetch, reactive 402→`PaywallGate`). Backend ✅; screens = CMO's next build.
- **Owner activation** (create the Clerk app + the Stripe product + set env) is consolidated into **M6** below.
- **Lane:** backend ✅ DONE; frontend = CMO (against the CEO contract spec).

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

### M6 — 🔧 Owner manual tasks *(consolidated — batched at the very end, owner-set 2026-06-20)*
**Outcome:** the hands-on setup only the owner (Connor) can do, gathered into ONE checklist so it isn't done piecemeal. Each task is technically needed at its milestone (noted), but bundled here per owner request — **keep building; Connor runs these together when prompted, with click-by-click steps provided per task.**
- **Clerk** (for the M2 frontend / M3 beta): create the Clerk application → set the backend `CLERK_ISSUER` (+ `CLERK_AUDIENCE` if using a JWT template) in Railway **AND** the frontend `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` + `NEXT_PUBLIC_HEATER_LIVE=1` in Vercel — the two frontend vars must be set TOGETHER or gated pages serve mock data instead of the paywall (CMO activation note). *(Backend verifier + frontend UI already built + dormant.)*
- **Stripe** (for M3 beta billing): create a Stripe account → a "HEATER Pro" product at **$7.99/mo with a 7-day trial** → a webhook endpoint for the 4 subscription events → set `STRIPE_SECRET_KEY` / `STRIPE_WEBHOOK_SECRET` / `STRIPE_PRO_PRICE_ID` in Railway. *(Backend already built + dormant.)*
- **Yahoo `fspt-w` write scope** (for live lineup / add-drop writes): re-register / re-authorize the Yahoo app with write scope (currently read-only → live writes 401).
- **Beta onboarding** (M3): invite the 12 FourzynBurn leaguemates onto the React product (revoke testuser → invite → approve/assign).
- **Domain / DNS / Vercel** (M3/M5): point the production domain at the Vercel frontend + the Railway API; set `HEATER_API_CORS_ORIGINS`.
- **Public-launch ops** (M5): landing/marketing site go-live, listings as applicable, multi-platform league-connection credentials.

---

## Planned features (carry-forward — each gets its OWN spec → plan → build)

Product features (not migration phases) to fold into the React app — captured here so they aren't lost. Each is a future spec→plan→build (TDD + review), integrated into the relevant page.

### Probable Pitcher schedule + Hitter Matchup schedule *(new — owner-requested 2026-06-20)*
A FantasyPros-style **7-day Probable Pitchers grid** (model: `https://www.fantasypros.com/mlb/probable-pitchers.php`), added as a **new tab on the Pitcher Streaming page/tool**. Requirements (from the owner + the FantasyPros reference screenshots):
- **Grid:** TEAM rows × the next-7-days columns; each cell = the probable SP (name, W-L record, SP rank), the opponent (`@SF` / `vs`), home/away.
- **League-connected:** per pitcher show **Rostered** (your team) / **Taken** (another league team) / **Available** (free agent), with toggle filters for each (the screenshot's Roster / Taken / Available checkboxes). Uses the league rosters + FA pool already in the backend.
- **Matchup difficulty:** easy vs tough, color-coded, with filters (Home / Away / Easy Matchups / Tough Matchups). Reuse the Stream Score / opponent-strength engine (`src/optimizer/stream_analyzer.py`).
- **Two-start pitchers** highlighted.
- **The inverse — "Hitter Matchup schedule"** (model: FantasyPros "Hitter Matchup Planner"): TEAM rows showing each team's **batting squad's** weekly matchup — the opposing probable SP per day (+ L/R handedness), a per-team totals strip (games, vs RHP, vs LHP) and a **matchups-rank**, color-coded easy/tough, same league-connection + filters.
- **Lane:** CEO (engine/contract — much already exists in `stream_analyzer` + `game_day` probables/team-strength) + CMO (the grid UI). **Status: captured only — future spec → plan → build → integrate. NOT built.**

### AI chat assistant — port to React *(carry-forward — owner-flagged 2026-06-20)*
The multi-provider **AI chat assistant is LIVE in the Streamlit app** on every page (`src/ai/`, spec `docs/superpowers/specs/2026-06-11-ai-chat-assistant-design.md` + `...-phase1.md`; memory `project_ai_chat_assistant`). It is **NOT yet in the React migration plan** — it must be **ported to the React app** (a cross-cutting embedded component against an API chat endpoint) so parity includes it. **Status: future — its own spec → plan → build** in the migration (fold into M1 page parity or a dedicated slice).

---

## Sequence at a glance

```
  PROVE THE STACK  ──────────────────────────►     GO PUBLIC  ──────────────►
  M0 ───► M1 ──────────► M2 ──────► M3 🚀          M4 ──────────► M5 🌐
 contract  14-page      auth +    single-league   Postgres +     public launch +
  lock     parity (C)   billing   beta (the 12,   workers +      multi-platform,
  ✅DONE   CMO active    ✅backend  current infra)  multi-tenancy  retire Streamlit
                        →login/    (current infra)  (public prereq)
                         pricing
                         (CMO)

  M6 🔧  = owner manual tasks (Clerk / Stripe / Yahoo / onboarding / DNS) — batched at the very end.
  Planned features (own spec→plan→build): Probable-Pitcher + Hitter-Matchup schedules (Pitcher Streaming tab); AI-chat React port.
```

## What this roadmap does NOT do

- It does not re-specify A or B (their design docs stand) or re-plan B2/B3/Clerk (their plans stand) — it **orders** them.
- It does not commit dates — it commits **sequence + gates**. The owner green-lights each milestone (and M4's sub-phases) when ready.
- It does not pick the Stripe tiers' exact pricing or the multi-platform launch set — those are decisions inside M2 / M5's own plans.

## Immediate next step *(updated 2026-06-20)*

**M0 ✅, M1 in progress (CMO), M2 backend ✅ (2026-06-20).** The next build is the **M2 frontend — login / signup / pricing / upgrade / account screens** (CMO, in `web/`), against the M2 backend contract that is **already shipped**. It is **already spec'd by the design track** (`docs/superpowers/specs/2026-06-20-heater-m2-frontend-auth-billing-design.md`) — so it is unblocked and queued in the CMO lane; the CEO/backend side is done. Owner manual setup (Clerk / Stripe) is deferred to **M6** (Connor does nothing manual until prompted, with steps provided per task).

## Index of governing docs

- North Star: `CLAUDE.md` ★ + memory `project_north_star_monetized_product`.
- A (frontend): `docs/superpowers/specs/2026-06-16-heater-frontend-foundation-design.md` + brand `...-brand-design-vision.md`.
- B (backend): `docs/superpowers/specs/2026-06-16-heater-backend-api-foundation-design.md` (B0–B5 phasing).
- Contract gap (M0 input): `docs/superpowers/specs/2026-06-19-frontend-api-contract-gaps.md`.
- C slice 0 (the wiring pattern): `docs/superpowers/specs/2026-06-19-frontend-api-wiring-slice0-design.md`.
- **M2 backend (auth + billing + gating): `docs/superpowers/{specs,plans}/2026-06-20-heater-m2-*` + memory `project_m2_auth_billing`.**
- **M2 frontend integration (login/pricing/upgrade): `docs/superpowers/specs/2026-06-20-heater-m2-frontend-auth-billing-design.md`.**
- **AI chat assistant (live in Streamlit; React port = carry-forward): `docs/superpowers/specs/2026-06-11-ai-chat-assistant-design.md`.**
- M4 infra plans (owner-gated): `...-b2-postgres-migration-plan.md`, `...-b3-workers-plan.md`, `...-clerk-auth-wiring-plan.md`.

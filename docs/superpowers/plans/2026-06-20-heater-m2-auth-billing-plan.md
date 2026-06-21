# HEATER M2 — Auth + Billing Plan (umbrella / slice index)

**Date:** 2026-06-20
**Lane:** CEO / backend (`api/` + `src/` + `tests/` + `docs/superpowers/`). NEVER touches `web/` (CMO) or the live Streamlit app (`src/auth.py` MULTI_USER).
**Milestone:** M2 of the migration roadmap (`docs/superpowers/specs/2026-06-19-heater-migration-roadmap.md` §M2).
**Status:** index approved structure; slice 1 (Clerk) has its own detailed TDD plan and executes now. Slices 2–3 sketched here; each spawns its own plan after the owner sets the product model (brainstorming).

> This is an **index + sequence**, like the roadmap. Each slice points at its own detailed plan and names the gaps. It does not restate slice detail.

---

## Outcome

Real user accounts (Clerk) + the ability to charge (Stripe Free→Pro) + feature-gating the compute-heavy endpoints — **single-tenant**. All ADDITIVE to the dormant `api/`. The live Streamlit app stays byte-for-byte unaffected; `src/` engines stay unchanged.

## What M2 is NOT (boundary, load-bearing)

- **Not multi-tenancy.** Clerk says *who the caller is* (verified). It does NOT map who → which league/team. Tenant resolution, per-user Yahoo OAuth, and the hardcoded `game_key 469` replacement are **M4/B4** (owner-gated, paused). Until then a verified Clerk user gets NO cross-tenant write authority — the single-configured-team write service is the guardrail. (Clerk plan §3/§5 boundary.)
- **Not the Yahoo `fspt-w` write scope.** Clerk auth does not grant Yahoo write scope; live writes still 401 at Yahoo until the app is re-registered. Owner's to enable; tracked independently.
- **Not the frontend.** Login/signup/pricing/upgrade UI is a later CMO task in `web/`. M2 backend only verifies tokens + serves billing endpoints the frontend will call.
- **Not Postgres/workers (B2/B3).** M2 runs on current SQLite infra. The api-owned state lives in its OWN sqlite file (see Architecture spine), swappable to Postgres at M4.

---

## Architecture spine (applies to all 3 slices)

**The seam idiom (already proven in `api/auth.py`): a `Protocol` + a DI provider + fake-override tests.** M2 extends it three ways:

1. **Auth verifier (exists).** `AuthVerifier` Protocol in `api/auth.py`; `EnvTokenVerifier` (interim, deny-by-default) is the default. M2 adds `ClerkVerifier` and makes `get_auth_verifier()` config-driven (Clerk when `CLERK_ISSUER` set, else the env verifier). `require_principal` + routers NEVER change.

2. **api-owned persistence (new).** The api/ layer becomes stateful for the first time. Introduce a small persistence layer:
   - `Store` Protocols (`UserStore`, `SubscriptionStore`) in `api/stores/`.
   - SQLite-backed default impls that create their OWN additive tables idempotently (`CREATE TABLE IF NOT EXISTS`), in a **separate** sqlite file `data/api_state.db` (env `HEATER_API_DB_PATH`) — NOT the live `draft_tool.db`. Zero contention with the Streamlit single-writer; zero risk to the live app; clean Postgres cutover at M4 (swap the impl behind the Protocol).
   - DI providers `get_user_store()` / `get_subscription_store()` in `api/deps.py`. Tests override them with in-memory fakes → DB-free (respects the worktree-empty-DB gotcha, memory `reference_worktree_empty_db`).
   - **Dormant when Clerk is unset:** nothing calls `get_or_create` on the env-token path, so no table is ever created or written until a Clerk user authenticates.

3. **Tier gate (new).** A single reusable dependency `require_pro` (like `require_principal`) that reads the caller's subscription tier from the `SubscriptionStore` and 402s (or returns a graceful tier-limited result — decided in slice 3) for Free callers on Pro-only endpoints. AST/structural-guarded so new heavy endpoints adopt it.

**Conventions inherited (non-negotiable):** contract (`api/contracts/`) → service/store seam (the ONE place importing `src/` or touching persistence) → thin logic-free router (AST-guarded by `tests/api/test_no_logic_in_routers.py`) → DI provider → fake-override test (`tests/api/test_api_*.py`) → mount in `api/main.py` → regen `api/openapi.json` (`python scripts/export_openapi.py`; snapshot-guarded; `fastapi==0.137.1` + `httpx==0.28.1` PINNED). Billing writes follow the existing write-route conventions: graceful, never-raise, HTTP-200-with-status-in-body (like `MutationResult`).

---

## Slice order + status

### Slice 1 — Clerk authentication  *(executes now)*
**Plan:** `docs/superpowers/plans/2026-06-20-heater-m2-slice1-clerk-auth.md`
**Outcome:** `ClerkVerifier` (PyJWT-JWKS local verification) drops into the `api/auth.py` seam; `get_auth_verifier()` is config-driven; `Principal` carries `clerk_user_id`; a local `AppUser` is provisioned (get-or-create) on the first authenticated Clerk call via the `UserStore`. Dormant by default (Clerk env unset → `EnvTokenVerifier`, no provisioning, no table). Reads stay open; the 2 write routes accept real Clerk JWTs.
**Owner prerequisite for ACTIVATION only (not for building):** create the Clerk app + provide `CLERK_ISSUER`/`CLERK_AUDIENCE` (Clerk plan §1). The code + tests land dormant without it (fake-verifier tests, RS256 keypair injected — no network).
**New dep:** `PyJWT[crypto]>=2.8`.

### Slice 2 — Stripe billing (Free→Pro)  *(brainstorm + own plan before building)*
**Plan:** TBD `docs/superpowers/plans/2026-06-20-heater-m2-slice2-stripe-billing.md` (after brainstorming).
**Outcome:** Clerk user → Stripe customer; `POST /api/billing/checkout-session` (start a Pro upgrade); `POST /api/billing/webhook` (Stripe events: subscription created/updated/canceled → `SubscriptionStore`); `GET /api/billing/subscription` (my tier). NaN/never-raise/HTTP-200-with-status conventions; the webhook verifies the Stripe signature (fail-closed).
**Owner decisions needed first (surface via brainstorming):** Free vs Pro feature split, the price point, trial?, monthly/annual. The roadmap explicitly defers exact pricing/tiers to "inside M2's own plans."
**New dep:** `stripe>=10` (pinned exact, like fastapi/httpx, so the SDK can't drift the webhook contract).

### Slice 3 — Feature-gating (the tier gate)  *(own plan; depends on slice 2's tier model)*
**Plan:** TBD `docs/superpowers/plans/2026-06-20-heater-m2-slice3-feature-gating.md`.
**Outcome:** a single reusable `require_pro` dependency on the compute-heavy endpoints (candidates: `POST /api/lineup/optimize` daily mode, `POST /api/trade/evaluate` with MC, `POST /api/playoff-odds`, `POST /api/trade-finder`, the draft sims). Free = limited, Pro = unlocked. A structural guard (`tests/api/test_*`) enforces that the agreed heavy endpoints carry the gate so a new heavy endpoint can't silently ship ungated.
**Owner decision needed:** the exact Free/Pro endpoint split (asked in slice 2's brainstorming alongside the feature split).

---

## Gates + sequencing

- Slice 1 lands additively + dormant; **mergeable now** (no owner prerequisite to BUILD; the Clerk-app setup is for ACTIVATION later).
- Slice 2 starts after the owner picks the Free/Pro model (brainstorming). Stripe test mode + a test secret key are enough to build + test; the live keys are an activation step.
- Slice 3 starts after slice 2 pins the tier model.
- Each slice = its own plan + its own commit(s) + a code-review pass (non-negotiable). `api/openapi.json` regenerated + green after every slice.

## Owner decisions to surface (batched at slice 2 brainstorming)

1. **Free vs Pro feature split** — which surfaces/endpoints are Pro-only vs free.
2. **Price** — monthly (and annual?) Pro price; free trial?
3. **Clerk `aud`** — JWT template with a fixed `aud` (recommended) vs validate `iss`/`azp` only (Clerk plan §7). Affects activation, not the build.

## Index of governing docs
- Roadmap: `docs/superpowers/specs/2026-06-19-heater-migration-roadmap.md` §M2.
- Clerk auth design (the authentication-half decisions + rejected alternatives): `docs/superpowers/plans/2026-06-19-heater-backend-clerk-auth-wiring-plan.md`.
- The auth seam: `api/auth.py`. The write-route conventions: `api/routers/roster_write.py` + `api/services/roster_write_service.py`.

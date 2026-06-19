# Clerk Auth Wiring Plan (B4 — auth half)

> **STATUS: PLAN ONLY — NOT APPROVED FOR EXECUTION.** This is the "agreed plan" the owner requires before B4 work. It also has a **hard external prerequisite**: a Clerk application the owner must create (no backend code can verify a real Clerk token without it — see §1). Nothing here executes until the owner sets up Clerk (§1) and ratifies §7. Each phase below spawns its own bite-sized TDD plan when green-lit.

**Goal:** Replace the interim env-bearer-token write gate with **real Clerk JWT verification**, by dropping a `ClerkVerifier` into the **existing** `api/auth.py` seam — `require_principal` and the routers never change. This is the *authentication* half of B4 ("who is the caller, verified"). It deliberately stops short of the *multi-tenancy* half ("which tenant/team does this caller own") — that is a separate B4 plan (§5 boundary).

**Why this is low-blast-radius despite being "B4":** the seam already exists and is proven (shipped 2026-06-19 — `api/auth.py`: `Principal`, `AuthVerifier` Protocol, `EnvTokenVerifier`, `get_auth_verifier`, `require_principal`; both write routes gated via `dependencies=[Depends(require_principal)]`). Clerk is literally "a second `AuthVerifier` implementation + a config switch in `get_auth_verifier`." The env-token verifier stays as the default/fallback (useful for server-to-server + CI). The write endpoints are still dormant (live Streamlit writes don't go through the API), so this touches no live data path until the frontend cutover (C/D).

---

## 1. Owner prerequisite — set up Clerk (no code can proceed without this)

Before Phase 1 can be *verified* end-to-end (and before activation), the owner creates the Clerk app and provides these values (they become backend env vars — secrets go in Railway, never in git):

| What | Where in Clerk | Backend env var |
|------|----------------|-----------------|
| Application | Clerk dashboard → Create application | — |
| **Issuer** (Frontend API URL) | API Keys / "Show JWT public key" → `iss` | `CLERK_ISSUER` (e.g. `https://<app>.clerk.accounts.dev`) |
| **JWKS URL** | `${issuer}/.well-known/jwks.json` (derivable from issuer) | `CLERK_JWKS_URL` (optional; default = issuer + `/.well-known/jwks.json`) |
| **Audience** | Create a **JWT template** with an `aud` claim (recommended) OR decide to validate `azp`/issuer only | `CLERK_AUDIENCE` (optional if validating `azp`) |
| Secret key (only if we adopt Clerk's SDK instead of pure JWKS) | API Keys → Secret key | `CLERK_SECRET_KEY` (NOT needed for the recommended PyJWT-JWKS path) |

**Decision the owner makes here (affects claim validation):** Clerk's *default* session token issuer is the Frontend API and may not carry a standard `aud`. The robust path is a **Clerk JWT template** that sets a fixed `aud` (e.g. `"heater-api"`). If the owner skips the template, the verifier validates `iss` + signature + expiry + `azp` (authorized party) instead. The plan supports both; the owner picks at §7.

**Frontend note (CMO track):** the Next.js app uses Clerk's React SDK; it obtains the session JWT (`getToken()`) and sends it as `Authorization: Bearer <jwt>` to the write endpoints. That wiring is the CMO/Sub-project-C job and is out of scope here — this plan only covers backend verification.

---

## 2. The architecture decision (the spine)

**Chosen: verify the Clerk JWT locally against Clerk's JWKS with `PyJWT[crypto]` + `PyJWKClient` (cached) — NOT the Clerk backend SDK.**

- Clerk session tokens are RS256 JWTs signed by Clerk; verification = fetch the JWKS (public keys) once, cache it, validate signature + `iss` + `exp`/`nbf` + (optionally) `aud`/`azp`, extract `sub` (the Clerk user id). No secret key needed, no per-request network call to Clerk, no SDK lock-in.
- **Rejected — `clerk-backend-api` SDK:** heavier dependency, ties us to Clerk's release cadence, and its networked `authenticate_request` adds a hop; the JWKS-verify path is the standard, lightest, most testable approach.
- **Rejected — opaque-token introspection:** Clerk tokens are JWTs; introspection would add a network round-trip per request for no benefit.

`ClerkVerifier` implements the existing `AuthVerifier` Protocol (`verify(authorization) -> Principal`), so it slots into the seam with zero router/`require_principal` change. `get_auth_verifier()` selects it when Clerk env is configured; otherwise the current `EnvTokenVerifier` (deny-by-default) remains.

**Dependency:** add `PyJWT[crypto]>=2.8` (brings `cryptography`). One new dep.

---

## 3. Phased plan (each phase = its own future TDD plan + worktree + 2-stage review + owner gate)

### Phase C-Auth.0 — Additive, dormant `ClerkVerifier` (default unchanged)
**What:** Add `PyJWT[crypto]`. Add `ClerkVerifier(authorization) -> Principal` to `api/auth.py`: parse the bearer token (reuse `_bearer`), validate via `PyJWKClient(CLERK_JWKS_URL).get_signing_key_from_jwt` + `jwt.decode(..., algorithms=["RS256"], issuer=CLERK_ISSUER, audience=CLERK_AUDIENCE or None, options=...)`, map `sub` → `Principal`. On any failure raise the same `HTTPException(401, WWW-Authenticate: Bearer)` shape `EnvTokenVerifier` uses (fail-closed). **`get_auth_verifier()` STILL returns `EnvTokenVerifier` by default** — `ClerkVerifier` is present but not wired as default.
**Tests** (no network — this is the key to testability): generate a local RS256 keypair in the test, build a tiny JWKS dict from the public key, **inject the JWKS** (constructor takes an optional `jwk_client`/keys for tests, prod builds its own from env). Cases: valid token → `Principal(subject=<sub>)`; expired → 401; bad signature (wrong key) → 401; wrong `iss` → 401; wrong `aud` (when configured) → 401; missing/garbage bearer → 401; non-ASCII header → 401 (byte-safe, same lesson as the env verifier).
**Risk:** low (additive, dormant). **Live path:** none.

### Phase C-Auth.1 — Config-driven activation + JWKS robustness
**What:** `get_auth_verifier()` returns `ClerkVerifier` when `CLERK_ISSUER` (and JWKS URL, derived if absent) is set, else `EnvTokenVerifier`. Add JWKS **caching + key rotation** handling (PyJWKClient caches; configure TTL + a refetch-on-unknown-kid path) and **clock-skew leeway** (`leeway=` on `jwt.decode`). Decide fail-mode when JWKS is unreachable: **fail closed (401)** — never fall open. Keep `EnvTokenVerifier` reachable for server-to-server/CI via a separate explicit config if needed.
**Tests:** the selector returns the right verifier per env (monkeypatched); unknown-`kid` triggers a refetch then 401 if still unknown; JWKS-unreachable → 401 (fail-closed), asserted with a stubbed client that raises.
**Risk:** medium (network dependency semantics). **Live path:** none (endpoints still dormant).

### Phase C-Auth.2 — Principal carries the Clerk identity + end-to-end
**What:** Add `clerk_user_id: str | None` (and keep `subject`) to `Principal`. Wire it through `require_principal` (already returns the Principal). Coordinate with the CMO track so the Next.js client sends `getToken()` as the bearer. **End-to-end test against a real Clerk dev instance** (needs the owner's Clerk app from §1) — a manual/integration check, owner-gated, not in the unit CI lane.
**Risk:** low-medium (cross-track coordination). **Live path:** still none until the frontend write-seam goes live (C/D).

> **NOT in this plan (the B4 multi-tenancy half — separate plan):** mapping `clerk_user_id` → a tenant/team (a `users`/tenancy table), per-user Yahoo OAuth, and Pro-tier gating. This plan ends at "the caller is a verified Clerk user." Authorizing *which data/team* that user may touch is multi-tenancy = the other half of B4 and must not be conflated — until it lands, a verified Clerk user must NOT be granted cross-tenant write authority. (The write service today targets a single configured team; that single-tenant assumption is the guardrail until multi-tenancy ships.)

---

## 4. Testing strategy
- **Unit lane (CI, no network):** self-signed RS256 keypair + injected JWKS; cover valid/expired/bad-sig/wrong-iss/wrong-aud/garbage/non-ASCII. The conftest network guard stays satisfied (JWKS client is injected, never dials out).
- **Selector tests:** env present → ClerkVerifier; absent → EnvTokenVerifier; fail-closed on JWKS-unreachable.
- **Integration (owner-gated, manual):** a real Clerk dev token verified end-to-end once §1 is done.
- The existing write-endpoint tests keep overriding `get_auth_verifier` (the seam already supports this) — they're verifier-agnostic, so they stay green unchanged.

## 5. Relationship to the existing env-token gate
The env-token `EnvTokenVerifier` (shipped 2026-06-19) **stays** as the default/fallback and the server-to-server/CI path. Clerk becomes the **user-facing** verifier, activated by config. Both coexist behind the one seam — no rip-and-replace. This means C-Auth can land additively and dormant without disturbing the current gate.

## 6. Risk register
1. **JWKS network dependency / fail mode** — auth must **fail closed** (401) if Clerk's JWKS is unreachable; never fall open. *Mitigation:* explicit fail-closed tests (C-Auth.1).
2. **Claim-validation correctness** (`iss`/`aud`/`azp`/`exp`, clock skew) — a loose check is an auth bypass. *Mitigation:* exhaustive negative tests; `aud` decision pinned in §1/§7.
3. **Identity ≠ authorization gap** — Clerk says *who*; until B4 multi-tenancy maps who→team, do not grant cross-tenant writes. *Mitigation:* the §3 boundary note + the single-tenant write-service guardrail; multi-tenancy is its own gated plan.
4. **Key rotation** — Clerk rotates signing keys; a cached JWKS with an old `kid` must refetch. *Mitigation:* refetch-on-unknown-kid (C-Auth.1).
5. **Yahoo `fspt-w` is separate** — Clerk auth does NOT grant Yahoo write scope; live writes still 401 at Yahoo until the app is re-registered with `fspt-w`. *Mitigation:* tracked independently; not solved here.

## 7. Decisions for owner ratification
- [ ] **Set up the Clerk app** and provide §1 values (issuer; and the `aud` decision: JWT template with `aud` vs validate `iss`/`azp` only).
- [ ] **Spine** (§2): PyJWT-JWKS local verification (not the Clerk SDK). One new dep `PyJWT[crypto]`.
- [ ] **Scope line**: this plan = authentication only. Multi-tenancy (clerk_user → team), per-user Yahoo OAuth, and Pro-tier gating are a SEPARATE B4 plan.
- [ ] **Fail-closed** on JWKS-unreachable (recommended).
- [ ] **Sequencing/gates**: C-Auth.0 (dormant) may merge early once approved; C-Auth.1/.2 verified behind config; nothing becomes the live default until you flip the env in prod (and the frontend sends Clerk tokens).
- [ ] **Order vs B2/B3**: decide whether Clerk-auth lands before or after the B2 Postgres work (they're independent — auth doesn't depend on Postgres — so this can proceed on its own timeline once §1 is done).

---

## 8. What this slice delivered (and did not)
**Delivered:** a ready-to-dispatch Clerk authentication plan that slots into the existing `api/auth.py` seam, the verification-approach decision (with rejected alternatives), the owner's Clerk-setup prerequisite list, a 3-phase additive/dormant sequence, a no-network test strategy, a risk register, and the explicit authentication-vs-multitenancy boundary.

**Did NOT (by design):** add any dependency, write any verifier code, touch the live data path, or start B4 multi-tenancy. Execution is deferred pending the owner's Clerk setup (§1) + §7 ratification. When green-lit, **Phase C-Auth.0** is the first to spawn a bite-sized TDD plan.

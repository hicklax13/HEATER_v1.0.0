# M2 Frontend — Auth (Clerk) + Billing (Stripe) UI — Design Spec

**Date:** 2026-06-20
**Track:** CMO / frontend (web/ only). M2, after M1 React parity.
**Backend:** COMPLETE + dormant on master (CEO track): Clerk verifier, Stripe billing, `require_pro` on 6 heavy endpoints. Activates when Connor sets Clerk/Stripe env.
**Contract (from api/openapi.json — read-only, regen TS via `pnpm gen:api`):**
- `POST /api/billing/checkout-session` body `{success_url?, cancel_url?}` → `{ok: boolean, url?: string, error?: string}`
- `GET /api/billing/subscription` → `{tier: "free"|"pro", status, current_period_end?: number, trial: boolean, error?}`
- 6 gated endpoints (`lineup/optimize`, `trade/evaluate`, `playoff-odds`, `trade-finder`, `draft/recommend`, `draft/simulate-picks`): return **402** (auth'd Free user) / **401** (unauthenticated) when billing is LIVE; **open** while dormant.

## Hard constraint — dormancy

When `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` is **unset** (today + the live app), the app must be **byte-identical to current M1**: no auth walls, no token attached, no 402s (backend keeps endpoints open), the existing mock TopBar account menu + "PRO" badge unchanged. Everything M2 is **env-gated** and activates automatically when Connor sets the key (his step, not mine). Pricing = **Pro $7.99/mo, 7-day free trial** (Free → Pro, 2-tier).

## Architecture

**Env flag (single source):** `web/src/lib/auth-config.ts` → `export const CLERK_KEY = process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY; export const authEnabled = !!CLERK_KEY;` Everything branches on `authEnabled`.

**No `clerkMiddleware`.** Auth is client-side sign-in + a Bearer JWT sent to the separate FastAPI (which verifies it). I do not protect Next routes, so I skip Clerk's middleware (its key requirement is the main dormant-build risk). This is the key simplification.

**Conditional provider:** `web/src/components/auth/AuthProvider.tsx` — `authEnabled ? <ClerkProvider publishableKey={CLERK_KEY}>{children}</ClerkProvider> : <>{children}</>`. Wrap the app in `layout.tsx` (or Providers). When dormant, zero Clerk code executes.

**Auth-aware client (`web/src/lib/api/client.ts`):** before each fetch, `const token = await window.Clerk?.session?.getToken?.()` (optional-chained; `undefined` when dormant) → attach `Authorization: Bearer <token>` when present. On `!res.ok`, throw a typed `ApiError { status }` (currently throws a plain Error — upgrade it) so callers can branch on 401/402. A minimal `declare global { interface Window { Clerk?: {...} } }`.

## Components & files

| File | Responsibility |
|------|----------------|
| `web/src/lib/auth-config.ts` | `CLERK_KEY` + `authEnabled` flag. |
| `web/src/components/auth/AuthProvider.tsx` | Conditional `<ClerkProvider>`. |
| `web/src/lib/api/client.ts` (modify) | Attach Clerk token; throw `ApiError {status}`. |
| `web/src/lib/api/errors.ts` | `ApiError` class + `isPaywall(e)`/`isAuthRequired(e)` guards. |
| `web/src/app/sign-in/[[...sign-in]]/page.tsx`, `sign-up/[[...sign-up]]/page.tsx` | Clerk `<SignIn>`/`<SignUp>` (path routing), centered in the Combustion shell. Redirect home when dormant. |
| `web/src/components/chrome/AccountArea.tsx` | TopBar auth zone: dormant → today's mock menu (unchanged); active+signed-out → "Log in" + "Get started"; active+signed-in → Clerk `<UserButton>` + a real Free/Pro chip. Replaces the inline mock menu in `TopBar.tsx`. |
| `web/src/lib/use-subscription.ts` | `SubscriptionProvider` + `useSubscription()` → `{tier, status, trial, currentPeriodEnd, loading}`. Fetches `GET /api/billing/subscription` only when `authEnabled` + signed-in; dormant → `{tier:"free", loading:false, active:false}` (no gating). |
| `web/src/app/pricing/page.tsx` | Free vs Pro comparison ($7.99/mo, 7-day trial). "Start free trial"/"Upgrade" → `POST /api/billing/checkout-session {success_url: <origin>/?upgraded=1, cancel_url: <origin>/pricing}` → `window.location.href = url`. States: dormant → "Coming soon" (billing returns `ok:false`); current Pro → "You're on Pro" + renewal/trial; Free → upgrade CTA. |
| `web/src/components/billing/PaywallGate.tsx` | Reusable "Upgrade to Pro" panel (used by the locked state). Brand-styled, lists the Pro value, CTA → `/pricing`. |
| `web/src/lib/use-page-data.ts` (modify) | Add a 5th state `"locked"`: when the fetcher throws an `ApiError` with status 402 → `{status:"locked"}`. 401 → also surface (pages route to login). Backward-compatible (dormant never 402s). |
| `web/src/components/ui/PageStates.tsx` (modify) | Add `PageLocked` (renders `PaywallGate`). |

**Paywall wiring (reactive, per-fetch — respects granular gating):** the gated fetchers already `try/catch → return null` (mock fallback). Change them to **rethrow `ApiError` 402/401** (not swallow) so `usePageData` maps 402→`locked`, 401→login-prompt. Pages with `usePageData` (Optimizer, Standings-odds, Trades-finder) render `PageLocked` on `locked`. The Draft Simulator (interactive, not `usePageData`) catches 402 in `useDraft` → a `locked` phase rendering `PaywallGate`. Dormant: no 402 ever → unchanged. Free Compare tab / Standings table (ungated) keep working.

## Verification

- **Dormant (the testable path):** with `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` unset → `pnpm build` green; every M1 page byte-identical (no auth UI, no token, no 402); `/pricing` renders in "coming soon" mode; `/sign-in` redirects home. This is the live-app safety guarantee and what I can fully preview-verify here.
- **Active (Connor's activation-time check):** can't fully test without his real Clerk/Stripe keys. I verify the wiring via `tsc` against the contract + structural correctness. Document the activation steps (set `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`, the FastAPI Clerk/Stripe env, create the Stripe product).
- `pnpm exec tsc --noEmit` + `pnpm run lint` + `pnpm build` green throughout.

## Out of scope (v1)

- Proactive tier-gating (hiding Pro pages before fetch) — reactive 402 is simpler + respects granularity; can add a teaser later.
- A full account/billing-management page beyond the pricing page + Clerk's `<UserButton>` (Stripe customer portal link is a later add).
- SSR auth / Clerk server helpers / route protection middleware (not needed for the client+Bearer model).

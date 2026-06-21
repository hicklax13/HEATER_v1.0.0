# HEATER M2 Slice 2 — Stripe Billing Design

**Date:** 2026-06-20
**Lane:** CEO / backend (`api/` + `tests/` + `docs/`). Additive to the dormant `api/`; does NOT touch `web/`, `src/`, or the live Streamlit app.
**Milestone:** M2 slice 2 (after slice 1 Clerk auth, which shipped — `2026-06-20-heater-m2-slice1-clerk-auth.md`).
**Status:** design APPROVED by owner 2026-06-20. Product decisions locked (below). Next: the slice-2 implementation plan (writing-plans) → TDD build → review.

## Goal

Free→Pro billing via Stripe, **single-tenant**, per-user (Clerk user → Stripe customer). A user can subscribe to **Pro ($7.99/mo, 7-day trial)**; the app records their tier; slice 3 gates the compute-heavy endpoints on it.

## Product decisions (owner-set 2026-06-20)

- **Tier model:** 2-tier Free→Pro, **hard gate** (NOT metered). Free = all read/data pages; Pro = the compute-heavy decision tools (gated in slice 3). No per-user usage counters needed.
- **Price:** **$7.99/mo** — configured in Stripe (a Price object), referenced by env `STRIPE_PRO_PRICE_ID`. The number is NOT in code, so it changes in the Stripe dashboard.
- **Trial:** **7-day** free trial, card-on-file via Stripe Checkout (auto-converts), `trial_period_days=7` (env `STRIPE_TRIAL_DAYS`, default 7).

## Scope

**In:** customer creation/reuse, a Checkout-session endpoint, a signature-verified webhook that records subscription state, a "my subscription" read endpoint, the `SubscriptionStore` persistence, and the `StripeGateway` testability seam.
**Out (YAGNI / later):** annual plan (a 2nd env price id when wanted), proration/upgrade-downgrade UI, usage metering (hard gate ⇒ none), invoices/billing-portal (Stripe hosts a portal — a future 1-endpoint add), multi-tenancy (M4). **The feature GATE itself is slice 3, not here.**

---

## Architecture (the seam idiom, same as slice 1)

`Protocol` + DI provider + fake-override tests. Three new units:

### 1. `StripeGateway` (testability seam) — `api/gateways/stripe_gateway.py`
Wraps the ~4 Stripe SDK calls so tests inject a fake (no network, no real charges).

```python
class StripeGateway(Protocol):
    def create_customer(self, clerk_user_id: str, email: str | None) -> str: ...          # -> customer_id
    def create_checkout_session(self, *, customer_id: str, price_id: str,
        success_url: str, cancel_url: str, trial_days: int, clerk_user_id: str) -> str: ... # -> hosted url
    def parse_webhook_event(self, payload: bytes, sig_header: str | None, secret: str) -> dict: ...
        # verifies the Stripe-Signature HMAC; raises BillingSignatureError on bad/missing sig
```

- **Real impl** `LiveStripeGateway`: configures `stripe.api_key` from `STRIPE_SECRET_KEY`; uses `stripe.Customer.create` (with `metadata.clerk_user_id`), `stripe.checkout.Session.create` (`mode="subscription"`, `client_reference_id=clerk_user_id`, `subscription_data.metadata.clerk_user_id`, `subscription_data.trial_period_days`), `stripe.Webhook.construct_event`.
- **`FakeStripeGateway`** (tests): records calls, returns canned ids/urls, lets a test feed a parsed event or raise `BillingSignatureError`.
- `BillingSignatureError` — a small module exception (the webhook's fail-closed signal).

### 2. `SubscriptionStore` (persistence) — `api/stores/subscription_store.py`
Owns `api_subscriptions` in the SAME separate `data/api_state.db` from slice 1 (env `HEATER_API_DB_PATH`) — never the live `draft_tool.db`. In-memory + SQLite impls behind the Protocol.

```python
class Subscription(BaseModel):
    clerk_user_id: str
    stripe_customer_id: str | None = None
    tier: str = "free"            # "free" | "pro"  (derived from status, persisted for fast reads)
    status: str = "none"          # Stripe status: none|trialing|active|canceled|past_due|unpaid|incomplete
    current_period_end: int | None = None  # unix ts (for "renews on X" display)
    updated_at: str

class SubscriptionStore(Protocol):
    def get(self, clerk_user_id: str) -> Subscription | None: ...
    def get_by_customer(self, stripe_customer_id: str) -> Subscription | None: ...  # webhook resolves customer->user
    def upsert(self, sub: Subscription) -> None: ...                                # idempotent by clerk_user_id (PK)
```

Table: `api_subscriptions(clerk_user_id TEXT PRIMARY KEY, stripe_customer_id TEXT, tier TEXT NOT NULL, status TEXT NOT NULL, current_period_end INTEGER, updated_at TEXT NOT NULL)` + `INDEX(stripe_customer_id)`. Created idempotently on first use (dormant until billing is used). Same self-cleaning `_connect` + WAL + per-call connect/close + threading.Lock pattern as `SqliteUserStore` (the slice-1 review hardened that pattern — reuse it).

### 3. `BillingService` — `api/services/billing_service.py`
The ONE billing-logic seam (imports the gateway + the store). Methods:
- `create_checkout(app_user, req) -> CheckoutSessionResponse`: get-or-create the user's Stripe customer (reuse `stripe_customer_id` from the store if present; else `gateway.create_customer` → write the clerk↔customer link to the store immediately so the webhook can resolve it), then `gateway.create_checkout_session(...)`. Graceful: not configured / no gateway → `ok=false, error="Billing not configured."`.
- `handle_webhook(payload, sig_header) -> WebhookResponse`: `gateway.parse_webhook_event(...)` (raises `BillingSignatureError` → the service raises `HTTPException(400)` so Stripe sees a rejected delivery + the operator sees it in the Stripe dashboard, logged WARNING — the slice-1 operator-visibility lesson). On a valid event, resolve `clerk_user_id` (event's `subscription.metadata.clerk_user_id` / `client_reference_id`, fallback `store.get_by_customer`), map `status → tier`, `upsert`. Unhandled event types → `handled=false`, 200.
- `read_subscription(app_user) -> SubscriptionResponse`: `store.get` → tier/status/period; missing → `tier="free", status="none"`. Never raises.

**Config** (read at call time, like `get_auth_verifier`): `STRIPE_SECRET_KEY`, `STRIPE_PRO_PRICE_ID`, `STRIPE_WEBHOOK_SECRET`, `STRIPE_TRIAL_DAYS` (default 7), `STRIPE_SUCCESS_URL` / `STRIPE_CANCEL_URL` (defaults from the first CORS origin). **Billing is "configured" iff `STRIPE_SECRET_KEY` + `STRIPE_PRO_PRICE_ID` are set.** Unset ⇒ every endpoint degrades gracefully (checkout `ok=false`; subscription `tier="free"`; webhook 200-ignore). Mirrors Clerk dormancy.

### 4. Routers — `api/routers/billing.py` (thin, mounted in `api/main.py`)
| Endpoint | Auth gate | Notes |
|---|---|---|
| `POST /api/billing/checkout-session` | `require_app_user` (slice 1) | Returns the hosted Checkout url. Needs a provisioned AppUser (Clerk). |
| `POST /api/billing/webhook` | NONE (Stripe signature instead) | Raw body read via `Request`; the service verifies the `Stripe-Signature` header. Documents 400 (bad signature). |
| `GET /api/billing/subscription` | `require_app_user` | "Am I Pro?" |

DI providers in `api/deps.py`: `get_stripe_gateway()` (→ `LiveStripeGateway` when configured, else a `NullStripeGateway` that reports "not configured"), `get_subscription_store()` (→ `SqliteSubscriptionStore`), `get_billing_service()`.

---

## Tier resolution (the contract slice 3 consumes)

`tier_for(status) = "pro" if status in {"trialing", "active"} else "free"`. Persisted on the subscription row by the webhook so the gate is a single store read (no per-request Stripe call). `past_due`/`canceled`/`unpaid`/`incomplete`/`none` ⇒ `free` (simplest safe default for the beta; a `past_due` grace window is a documented follow-up). Slice 3's `require_pro` reads `store.get(clerk_user_id)` → `tier`.

## Webhook events handled
- `customer.subscription.created` / `.updated` / `.deleted` — the lifecycle; each carries `customer`, `status`, `current_period_end`. Upsert tier from status (`.deleted` → `canceled`/free).
- `checkout.session.completed` — links `client_reference_id` (clerk_user_id) ↔ `customer`, sets an initial tier (the subscription events refine it).
- Any other type → 200, `handled=false` (ignored, not an error).

**Idempotency / ordering:** upsert is keyed by `clerk_user_id` (PK) ⇒ idempotent re-delivery. Stripe can deliver out of order; for the single-tenant friends beta last-write-wins is accepted. **Follow-up (documented, not built):** event-id dedup table + `event.created`-guarded writes to defend against a stale `.updated` re-activating a `.deleted`.

## Error handling (conventions)
- **Reads + checkout:** never raise → graceful `ok=false`/`tier="free"` + `error`. HTTP 200.
- **Webhook signature:** **fail closed** — bad/missing signature → `HTTPException(400)` + WARNING log (operator-visible in the Stripe dashboard). A valid event whose processing throws → propagates to 500 so Stripe RETRIES (delivery guarantee). Forged/garbage with no valid signature is never processed.
- **Not configured:** dormant + graceful everywhere; the live app is unaffected.

## Testing strategy (DB-free + network-free — the worktree-empty-DB gotcha)
- **`FakeStripeGateway`** injected via `app.dependency_overrides[get_stripe_gateway]` → no Stripe network, no charges.
- **`InMemorySubscriptionStore`** injected → no DB.
- **SQLite store** unit-tested against `tmp_path` (idempotent upsert, `get_by_customer`, separate file), like `SqliteUserStore`.
- Cases: checkout creates+reuses a customer and returns the url; checkout when unconfigured → `ok=false`; webhook with a good fake event upserts tier (`trialing`→pro, `deleted`→free); webhook bad signature → 400 + WARNING; `get_by_customer` resolution; subscription read default `free`; tier_for mapping table. Test files `tests/api/test_api_*.py`.
- **Stripe SDK pinned EXACT** in `requirements.txt` (resolved at install) so the webhook event schema the SDK parses can't drift — same rationale as the `fastapi`/`httpx` pins.

## OpenAPI
Three new routes + their contracts land in `api/openapi.json` (regen via `scripts/export_openapi.py`). The webhook documents its 400; the two authed routes document their 401 (the `_AUTH_401` pattern from `roster_write.py`). This IS a contract change (unlike slice 1) — the snapshot test will require the regen.

## Owner activation steps (LATER — the build + tests use Stripe TEST mode, no real account needed)
1. Create a Stripe account → a Product "HEATER Pro" with a **$7.99/mo recurring Price**.
2. Set a webhook endpoint (`https://<api-host>/api/billing/webhook`) for the 4 event types → copy the signing secret.
3. Put 3 values in Railway env: `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`, `STRIPE_PRO_PRICE_ID` (+ optionally `STRIPE_SUCCESS_URL`/`STRIPE_CANCEL_URL`/`STRIPE_TRIAL_DAYS`).
Until then everything is dormant.

## Follow-ups (documented, not in this slice)
- Stripe billing-portal endpoint (manage/cancel) — 1 endpoint over `stripe.billing_portal.Session.create`.
- Webhook event-id dedup + ordering guard.
- `past_due` grace window before dropping to free.
- Annual price (2nd `STRIPE_PRO_PRICE_ANNUAL_ID`).

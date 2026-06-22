# HEATER "Bubba" B3 — Tier-Aware Managed AI + Upgrade Nudge (Design Spec)

**Date:** 2026-06-22
**Lane:** CEO — full-stack (`api/` + `src/ai/` + `web/`).
**Parent spec:** `docs/superpowers/specs/2026-06-20-heater-bubba-ai-assistant-design.md` (Phase B3, monetization — a LEAN first cut).
**Builds on:** M2 billing ([[project_m2_auth_billing]]) + the B1 AI budget cap. **Phase B2 (B2.1–B2.4) is complete.**
**Status:** SHIPPED 2026-06-22 — built via plan `docs/superpowers/plans/2026-06-22-heater-bubba-b3-tier-aware-ai.md` (6 tasks / 2 slices; additive + DORMANT until billing is configured; both pr-review-toolkit reviewers ran — "no high-confidence issues" / "SOUND, zero CRITICAL/HIGH", applied LOW-1: documented the store-failure over-grant trade-off). Deferred (until Pro sells): Plus/Max tiers, a monthly free taste, admin-tunable per-tier caps.

## Scope

Make managed Bubba chat **metered by subscription tier**, and nudge a capped user to upgrade or bring their own key. Deliberately LEAN — it reuses M2's shipped billing (the `$7.99` Pro tier, `SubscriptionStore`, `/pricing`) and the B1 AI budget, adds NO new Stripe products, and is **dormant until billing is turned on**.

1. **Tier-aware daily managed cap.** Today one shared daily $-cap applies to every shared-key user; a user on their own key is unlimited. B3 makes the cap depend on tier: **Free** gets a small daily "taste" (≈$0.10), **Pro** gets a bigger managed allowance (≈$2.00), **own-key** stays unlimited. Values are tunable constants.
2. **Upgrade nudge.** When a user hits their managed cap, the chat's over-limit error carries a structured `code: "over_cap"` and the panel renders a nudge with two always-valid CTAs: **Upgrade** (→ M2's `/pricing`) and **Add your own key** (→ key settings).

**Dormant-safe (load-bearing):** until `billing_env_configured()` is true (Stripe env set), the resolver returns `None` and `budget.is_over_cap` falls back to today's single admin cap — so the live app + every current user are **byte-for-byte unchanged** until the owner turns billing on. Same additive-and-dormant pattern as M2.

**Out of scope (deferred until Pro sells):** Plus/Max tiers (new Stripe prices + multi-price tier resolution), a monthly (vs daily) free taste, admin-tunable per-tier caps. The lean design leaves clean seams for all three.

## Grounding (verified in source)

- `src/ai/budget.py::is_over_cap(user_id, on_own_key=False) -> bool`: `on_own_key` → `False` (own-key unlimited); else `spent_today(user_id) >= daily_cap_usd()` (one admin-set cap in `app_settings`, default `$1.00`). `spent_today` reads the `ai_usage_ledger` (draft_tool.db). **This module is ALSO used by the live Streamlit chat** — its existing call shape must stay behavior-identical.
- `api/services/chat_service.py`: both `send` and `send_stream` compute `on_own_key = any(k.get("provider") == provider for k in keys.list_keys(chat_user_id))`, then check `budget.is_over_cap(chat_user_id, on_own_key=on_own_key)`. `send`'s over-cap path returns `self._empty("You've reached today's usage limit. Add your own API key for unlimited use.", conversation_id)`; `send_stream` yields `_sse({"type": "error", "message": "You've reached today's usage limit. Add your own API key for unlimited use."})` then returns.
- `api/billing_config.py::billing_env_configured()` — the single dormancy predicate (`STRIPE_SECRET_KEY` AND `STRIPE_PRO_PRICE_ID`); the SAME one the Pro gate uses (so the AI cap can't tighten before checkout is possible).
- `api/stores/subscription_store.py::Subscription.tier` is `"free" | "pro"`; `SubscriptionStore.get(clerk_user_id) -> Subscription | None`. `api/deps.py::get_subscription_store()` returns the Sqlite store (api_state.db). `api/identity.py::require_app_user -> AppUser | None` (AppUser carries `clerk_user_id`); the chat routes already 401 on `None` via `_uid`.
- The streaming `error` event is a plain SSE dict (no pydantic contract) — `BubbaStreamEvent` (TS, `web/src/lib/api/bubba.ts`) is its only typed mirror; adding an optional `code` needs **no openapi change** (the streaming route has no response model).

## Architecture

### Backend — one cap override, tier logic stays in `api/`

- **`src/ai/budget.py`** (the ONLY `src/ai` change): `is_over_cap(user_id, on_own_key=False, cap_usd: float | None = None)`. The cap is `cap_usd if cap_usd is not None else daily_cap_usd()`. **`on_own_key` short-circuit and the no-`cap_usd` path are byte-identical to today** → the live Streamlit app (which never passes `cap_usd`) is unchanged. A guard test pins this.
- **`api/services/ai_allowance.py`** (new): `_TIER_DAILY_CAP_USD = {"free": 0.10, "pro": 2.00}` + `managed_cap_for_tier(tier: str) -> float` (unknown tier → the free cap). Documented as tunable; the seam where Plus/Max caps slot in later.
- **`api/deps.py::get_managed_ai_cap(app_user = Depends(require_app_user), store = Depends(get_subscription_store)) -> float | None`** — the dormancy-aware resolver:
  - `if not billing_env_configured(): return None` → `budget` uses the admin cap = today's behavior (dormant).
  - else resolve the tier (`store.get(app_user.clerk_user_id).tier` if a row exists, else `"free"`) and return `managed_cap_for_tier(tier)`. Never raises (a store read failure → `None`, the safe admin-cap fallback) + logged.
- **`ChatService.send` / `send_stream`** gain `cap_usd: float | None = None`, passed straight into `budget.is_over_cap(chat_user_id, on_own_key=on_own_key, cap_usd=cap_usd)`. On the over-cap branch: the message becomes upgrade-aware ("You've hit today's AI limit. Upgrade for more, or add your own key for unlimited use.") and **`send_stream` adds `"code": "over_cap"`** to the error event. (`send`'s response keeps a plain string error — the frontend uses streaming.)
- **`api/routers/chat.py`** `send` + `send_stream`: add `cap_usd: float | None = Depends(get_managed_ai_cap)` and pass `cap_usd=cap_usd`. The router stays logic-free (a declared dependency, not computed).

### Frontend (`web/`)

- `bubba.ts`: the `BubbaStreamEvent` error variant gains `code?: string`.
- `Bubba.tsx`: a `UiMessage` gains an optional `nudge?: "over_cap"`. On an `error` event with `code === "over_cap"`, the in-flight assistant turn is finalized as `{ isError: true, nudge: "over_cap", content: message }`. The transcript renders that turn as a nudge card with two buttons: **Upgrade** (an `<a href="/pricing">`, since Bubba is a global panel) and **Add your own key** (opens the existing key-settings panel via `setShowSettings(true)`). Every other error renders exactly as today.

### Data flow

`POST /send-stream` → router resolves `cap_usd` via `get_managed_ai_cap` (dormant → None) → `ChatService.send_stream(cap_usd=…)` → `budget.is_over_cap(uid, on_own_key, cap_usd)` → over cap → `error{code:"over_cap"}` SSE → the panel renders the upgrade nudge.

## Error handling

- **Billing dormant or a subscription-read failure** → `cap_usd = None` → the admin cap (today's behavior). Never harder than today by accident.
- **`get_managed_ai_cap` never raises** — a store error logs + returns `None` (admin-cap fallback). The chat is never 500'd by a billing read.
- **Own-key users** are unaffected (`on_own_key` short-circuits before the cap is even consulted) — they never see the nudge.
- **The nudge is informational** — both CTAs are always valid (Upgrade helps free users; Add-key helps everyone), so no tier knowledge is needed in the frontend.

## Testing

- **`budget.is_over_cap` (DB-free via monkeypatched `spent_today`/`daily_cap_usd`):** `cap_usd` override is honored (spend $0.50, cap $0.10 → over; cap $2.00 → under); **no-`cap_usd` call is byte-identical to today** (uses `daily_cap_usd()`) — the live-app guard; `on_own_key=True` → `False` regardless of `cap_usd`.
- **`ai_allowance.managed_cap_for_tier`:** free < pro; unknown tier → free cap; values > 0.
- **`get_managed_ai_cap` (fake store + monkeypatched `billing_env_configured`):** billing off → `None`; billing on + tier "pro" → the pro cap; billing on + no sub → the free cap; a raising store → `None` (never propagates).
- **`ChatService.send_stream` (DB-free):** a low `cap_usd` with simulated spend → emits an `error` frame with `code:"over_cap"`; `cap_usd` is threaded into `budget.is_over_cap` (assert via a fake `budget`).
- **Router:** `/send-stream` wires `get_managed_ai_cap` (override the dep, assert the service receives the cap); logic-free-router guard stays green; openapi unchanged (no contract model touched).
- **Read-only tool surface** unchanged — re-assert.
- **Frontend:** tsc + lint + build; preview smoke (panel mounts; the nudge renders on a simulated `over_cap` event). Live metering needs Clerk + Stripe → owner verifies post-activation.

## Risks

- **Accidentally tightening the live app's cap.** Mitigated three ways: the `src/ai` change is a no-op without `cap_usd`; the resolver returns `None` while billing is dormant; both are guard-tested. The live Streamlit app calls `is_over_cap(uid, on_own_key)` with no `cap_usd` → unchanged.
- **Per-send subscription read** adds one indexed SQLite SELECT on `api_state.db` per chat send when billing is live — negligible, and skipped entirely while dormant.
- **Cap values are guesses** (≈$0.10 / ≈$2.00) — constants, tunable pre-launch; the owner adjusts once real usage costs are known. Admin-settable per-tier caps are a documented follow-up.

## Slices (build order)

- **Slice A — backend:** `budget.is_over_cap(cap_usd=…)` + guard; `ai_allowance.py`; `get_managed_ai_cap` resolver; `ChatService` threads `cap_usd` + the `over_cap` code; router wires the dep; tests.
- **Slice B — frontend:** `BubbaStreamEvent.code`; the `over_cap` nudge card (Upgrade + Add-key).

## Index

- New: `api/services/ai_allowance.py`; `api/deps.py::get_managed_ai_cap`; `budget.is_over_cap` `cap_usd` param; `ChatService` `cap_usd` thread + `over_cap` code; `BubbaStreamEvent.code` + the `Bubba.tsx` nudge.
- Reuses unchanged: M2's `SubscriptionStore` / `billing_env_configured` / `/pricing`, the B1 budget ledger + own-key exemption, the B1 key-settings panel, the B2.1 streaming error path.
- Parent: the Bubba design spec (Phase B3). Memory: `project_bubba_ai_assistant`, `project_m2_auth_billing`.

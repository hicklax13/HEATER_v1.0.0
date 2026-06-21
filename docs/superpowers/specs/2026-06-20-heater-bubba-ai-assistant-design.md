# HEATER "Bubba" — AI Assistant (full-access, every-page) Design Spec

**Date:** 2026-06-20
**Lane:** CEO — **full-stack on this feature** (backend `api/` + `src/ai/` AND the frontend `web/`), owner-directed 2026-06-20. **One coordination seam:** the every-page button lives in the shared `web/` TopBar chrome the CMO owns — spec'd here, reconciled with the CMO at build time.
**Status:** design — strategic decisions LOCKED (below). A sub-project decomposed into 4 phases; each gets its own plan → build.

## Vision

**Bubba** is HEATER's AI assistant — a folksy, razor-sharp baseball-insider persona (named for Bubba Crosby) reachable from a button on **every page**, in a pop-up chat that **looks, operates, and feels as polished as the claude.ai main chat**. Bubba has **full access to — and control over — ANY and ALL of the app's data**: historical, live, and future. It is the single conversational surface over the entire HEATER brain.

## The name + persona

The assistant is **Bubba**. Voice: a knowledgeable, friendly, slightly folksy fantasy-baseball buddy who happens to have the whole league's data at his fingertips. The system prompt encodes this persona on top of the existing league/context injection (`src/ai/chat.py::build_system_prompt`). The button reads **"Ask Bubba"**; the ember-spark mark (concept shown 2026-06-20) is Bubba's avatar.

## Monetization model (LOCKED — CEO decision 2026-06-20)

**Principle: give away what costs us nothing (BYO-keys); meter what costs us money (managed AI) on a good-better-best ladder with a recurring free taste that builds the converting habit.**

| | Powered by | HEATER charge | Key settings visible? |
|---|---|---|---|
| **BYO-keys** | the user's own provider API key (any configured model); they pay the provider | **$0 — free for everyone** | YES (they enter/manage keys) |
| **Managed "Bubba"** | HEATER's keys + a metered monthly allowance of the **latest models only** (Gemini / GPT / Claude / Grok) | paid (markup baked in) | **NO — hidden** (we handle keys) |

**The tier ladder (the product's full good-better-best):**
- **Free** — read/data pages + BYO-key Bubba + a **small recurring monthly managed-Bubba allowance** (routed to a cheap model to bound COGS; this IS the trial/funnel).
- **Pro — $7.99/mo** — all compute-heavy analytics tools (the existing M2 gate) + BYO-key Bubba.
- **Plus — ~$14.99/mo** — Pro tools + a managed Bubba allowance (latest models).
- **Max — ~$29.99/mo** — Pro tools + a large managed Bubba allowance.

(Two managed-AI levels = a clean upgrade path; exact prices tune in Stripe.) **Account-view rule:** show the API-key settings when the user is on the BYO path; **hide them** once they're on a managed-Bubba tier (HEATER provides the keys). Managed usage is metered against the tier's monthly allowance, reset monthly, enforced by the existing `src/ai/budget.py` ledger extended with per-tier caps + Stripe-synced entitlements.

## Architecture

### Backend — wrap the existing (clean) `src/ai/` behind the API
The exploration confirmed `src/ai/` is already provider-agnostic with **zero Streamlit coupling** in the logic layer. The port is a thin FastAPI wrap (no `src/ai/` rewrite):
- **Reuse unchanged:** `src/ai/providers.py::chat(model, messages, api_key, user_id, …)` (the agentic tool loop), `keys.get_key`, `history.{create_conversation, append_message, load_messages, list_conversations}`, `budget.{record_usage, spent_today, is_over_cap}`, `tools.{tool_specs, dispatch_tool}`, `search.{web_search, deep_research}`, `router.{model catalog, price_per_token}`, `schema_card`.
- **New endpoints (`api/routers/chat.py` + `api/services/chat_service.py`):**
  - `POST /api/chat/send` — body `{message, model, conversation_id?, web_search?, deep_research?, reasoning_effort?, attached_text?, attachments?}` → `{content, tokens_in, tokens_out, conversation_id, cost_usd, tool_trace}`. Non-streamed Phase 1 (matches today); **SSE streaming Phase 2**.
  - `GET /api/chat/conversations` → the user's saved conversations.
  - `GET /api/chat/conversations/{id}/messages` → load a conversation.
  - `GET /api/chat/models` → the models available to this user (their own keys' providers ∪ shared/managed models for their tier).
  - `GET/PUT /api/chat/keys` — manage BYO provider keys (Fernet-encrypted, the existing `ai_provider_keys` path). Hidden in the UI for managed-tier users.
  - `POST /api/chat/saved-prompts` (+ list/delete) — the "save prompts" feature.
- **Identity seam (the real integration work):** `src/ai/` keys/history/budget are keyed by an integer `user_id` (the Streamlit `users.id`). The React caller is a Clerk `AppUser` (from M2, `api/stores/user_store.py`). **Decision: link `AppUser` → a stable chat `user_id`** — add a nullable `chat_user_id` to the AppUser record (or a mapping table) resolved on first chat call, so per-user keys/history/budget carry over. `require_app_user` provides the principal; the chat service resolves the chat user id.
- **Auth:** every `/api/chat/*` route is `require_app_user`-gated (chat is inherently per-user). Reads of the user's own data only.

### Full data ACCESS + CONTROL (the load-bearing requirement)
Bubba must reach **any and all** historical / live / future data, and be able to **act**.
- **READ (all data):** the existing `query_data` tool runs read-only `SELECT` over the **entire** SQLite schema (the schema card already advertises every table), plus the structured tools (`get_player`, `get_my_team`, `get_standings`, `get_free_agents`, `compare_players`). **Future-proof by design:** new tables/features automatically appear in the schema card → Bubba can query them with no per-table wiring. Live data flows through the same DB (the 3-tier Yahoo/stat caches) so "current/live" is covered.
- **CONTROL (actions):** Bubba gains **action tools** over the already-graceful write paths — set lineup, add/drop, queue/apply moves, trigger a refresh — wrapping the same `YahooFantasyClient.set_lineup`/`add_drop` + `request_refresh` the app uses. **Safety model (non-negotiable):** every mutation is **confirm-gated** — Bubba PROPOSES the action (structured: "set Soto to BN, start Judge"), the UI shows a confirm card, the user approves, THEN it executes. Actions are `viewer_can_write`-gated, audit-logged, and never auto-fire. (Mirrors the existing confirm-gated Yahoo write-back from the Beta roadmap.) Cross-tenant safety holds (single-tenant beta; the write service targets the authenticated user's team).
- This satisfies "full access AND control of ANY and ALL data" while keeping a human in the loop for every change.

### Frontend (my lane for this feature) — the button + the Claude-grade popup
- **The "Ask Bubba" button** on every page (the ember-spark mark): breathing pulse + rotating heat-ring + drifting constellation; intensifies on hover; a distinct **select-mode** state that puts a thin orange glow on the page edge. Lives in the shared TopBar (the CMO-chrome coordination seam).
- **The popup** — a floating, resizable window that feels like claude.ai: conversation transcript, a **history** rail (past conversations), a **model switcher** (BYO models from the user's keys for the free/BYO path; the preset latest models for managed tiers), composer with the feature controls below. Built in `web/` (React + the Combustion design language), Bubba-branded.
- **Features (both BYO + paid):**
  1. **Select-to-tag** — arm a mode where clicking on-screen text / a graphic / a player row / a stat **tags it to the chat**; the page visibly indicates the mode is on (the select-mode glow). Implemented via a React selection layer (`document.getSelection` + a click-capture overlay that reads a tagged element's data attributes), replacing the Streamlit `streamlit-js-eval` hack.
  2. **Web search** + **deep research** toggles (wrap the existing `src/ai/search.py`).
  3. **Attach local files** (file picker → multimodal message; the latest models are vision/doc-capable).
  4. **Screenshots** — capture the **HEATER page** (in-DOM, e.g. `html-to-image`) AND **any open window on the device** (`navigator.mediaDevices.getDisplayMedia()` — the user picks the window/screen to share; we grab a frame). Both → attached as image input. Privacy: the user explicitly chooses what to share.
  5. **Reasoning/thinking effort** toggle (low / medium / high / max) — mapped to each provider's supported control (e.g. the `reasoning_effort` / thinking-budget param); falls back to the provider's allowed set. Passed through `POST /api/chat/send`.
  6. **Message queue** — type-ahead multiple messages while Bubba is working; **edit/delete** queued items before they send.
  7. **Saved prompts** — save/name/reuse prompts (the `/api/chat/saved-prompts` store).

## Phasing (the sub-project — each phase = its own plan → build)
- **Phase B1 — Chat MVP (backend + popup):** the `/api/chat/*` endpoints over `src/ai/`, the identity seam (Clerk AppUser → chat user_id), BYO-keys, the popup with history + model switch + basic composer, the every-page button. Non-streamed. *Outcome: a working, branded Bubba chat on the new app, BYO-keys, no monetization gating yet.*
- **Phase B2 — The rich features:** select-to-tag (+ the page select-mode UI), web search, deep research, file attach, **screenshots** (page + device window), reasoning-effort toggle, message queue, saved prompts, and SSE streaming.
- **Phase B3 — Monetization:** the managed-Bubba tiers (Plus/Max) + the free recurring allowance, Stripe entitlements/metering synced to `budget.py` per-tier caps, the account-view key-visibility logic (show for BYO, hide for managed), upgrade prompts at the cap.
- **Phase B4 — Full control (actions) + future-data extensibility:** the confirm-gated action tools (lineup/add-drop/refresh), the audit + safety model, and the pattern for auto-exposing future data/features to Bubba (schema-card + tool-wrapper convention).

## Error handling
Backend never 500s on chat (graceful error in the response, like today's `st.warning` paths → HTTP body). Provider/key errors return a clear message. Over-cap (managed, free taste exhausted) → a structured "upgrade" response the UI renders as a prompt. Action tools fail closed (no mutation on any error) + require confirmation.

## Testing
- Backend: DB-free service tests with a fake provider (monkeypatch `providers.chat`) + in-memory stores; the identity-mapping + per-tier cap logic; the confirm-gate (an action tool never mutates without the confirm token). Reuse the M2 fake-service pattern.
- Frontend: the popup + button render; select-mode arms/disarms; the feature controls wire to the endpoints. Verified via the preview tools at build time.

## Open risks / decisions for build time
- **CMO chrome seam** — the every-page button + select-mode glow touch shared TopBar/page chrome; coordinate the exact integration with the CMO at B1 build.
- **Screen-capture "any window"** — `getDisplayMedia` is powerful + privacy-sensitive; the user explicitly picks what to share, nothing is captured silently. Document the permission UX.
- **Managed-AI COGS + abuse** — the free recurring taste must be a cheap model + a hard monthly cap; managed tiers need per-tier caps + Stripe entitlement sync; rate-limit to deter abuse.
- **Control-action safety** — confirm-gated + audited + `viewer_can_write`; never auto-execute; single-tenant guardrail until multi-tenancy (M4).

## Index
- AI chat (Streamlit, the engine being wrapped): `docs/superpowers/specs/2026-06-11-ai-chat-assistant-design.md`, `src/ai/`, memory `project_ai_chat_assistant`.
- M2 backend (auth/billing the monetization rides on): `docs/superpowers/specs/2026-06-20-heater-m2-stripe-billing-design.md`, memory `project_m2_auth_billing`.
- Roadmap entry: `docs/superpowers/specs/2026-06-19-heater-migration-roadmap.md` (Planned features).

# HEATER "Bubba" B2.1 — Streaming + Thinking-Effort + Research Toggles (Design Spec)

**Date:** 2026-06-21
**Lane:** CEO — full-stack (`api/` + `src/ai/` + `web/`).
**Parent spec:** `docs/superpowers/specs/2026-06-20-heater-bubba-ai-assistant-design.md` (Phase B2, first wave).
**Status:** design — approved by owner 2026-06-21 ("Premium feel" wave first).

## Scope

The first B2 wave — make Bubba *feel* like claude.ai:
1. **SSE streaming** — assistant text streams token-by-token; tool use shows live status chips (Bubba is tool-heavy, so "stream the final answer only" = dead silence — we stream **every turn** + emit tool-status events).
2. **Thinking-effort dial** — Off / Low / Med / High, mapped per-provider (Claude thinking-budget, OpenAI `reasoning_effort`; unsupported providers → no-op).
3. **Web-search + Deep-research toggles** — frontend only (the backend already honors the flags as of B1).

Out of scope (later B2 waves): select-to-tag, attachments/screenshots, message queue, saved prompts.

## Grounding (verified in source — the M0 lesson)

- `src/ai/providers.py::chat(model, messages, api_key, user_id, max_tool_rounds=6, web_search=False, deep_research=False) -> {"content","tokens_in","tokens_out","tool_trace"}` is a **sync agentic tool loop** (≤6 rounds of `litellm.completion`; tool calls dispatched between rounds via `tools.dispatch_tool`). The final text is the last turn's `msg.content`.
- LiteLLM supports `stream=True` (returns a **sync generator** of deltas); `litellm.stream_chunk_builder(chunks, messages=…)` reconstructs the full message (incl. `tool_calls`) from streamed chunks. So streaming needs **no async rewrite**.
- `api/services/chat_service.py::send(...)` already passes `web_search`/`deep_research` to `providers.chat`, records `budget.record_usage(...)` after the call, and appends history in a `try/except` (a persistence failure never discards the answer).
- **No** existing `reasoning_effort`/thinking param in `src/ai/`. **No** existing `StreamingResponse`/SSE anywhere — B2.1 establishes the pattern.
- The Next.js `web/next.config.ts` `rewrites()` proxy passes `text/event-stream` through transparently (no buffering) — no proxy work.
- `web/src/lib/api/client.ts::apiPost` buffers (`res.json()`) — a **new** streaming fetch path is required.

## Architecture

### Backend — one event core, two consumers (DRY + live-app-safe)

The load-bearing constraint: `providers.chat` is **also used by the live Streamlit app** — its return contract must stay byte-identical.

- **New private core** `src/ai/providers.py::_chat_events(...) -> Generator[dict]` — the existing loop, refactored to **yield events** instead of returning a dict:
  - `{"type":"text_delta","text": "..."}` — streamed content deltas (every turn, via `litellm.completion(stream=True)` + accumulating chunks).
  - `{"type":"tool_started","name": "...", "args": {...}}` — before each tool dispatch.
  - `{"type":"tool_result","name": "...","ok": bool}` — after dispatch (no payload — UI shows a chip, not the raw data).
  - `{"type":"done","content": str,"tokens_in": int,"tokens_out": int,"tool_trace": [...]}` — terminal, carries the same totals `chat()` returns today.
  Each round: stream the completion, yield text deltas, `stream_chunk_builder` → rebuilt message; if it has `tool_calls` → yield `tool_started`, dispatch, append, loop; else → `done`. Tokens summed across rounds (via `stream_options={"include_usage": True}` or the rebuilt response).
- **`chat(...)` becomes a thin DRAIN of `_chat_events`** — consume the generator, concatenate `text_delta`s, return the `done` event's fields in **today's exact dict shape**. **Guard test:** `chat()` output equals a frozen reference for a fixed fake-provider script (the proven slice-9 / `auto_pick_opponents` equivalence pattern). This is what makes the refactor safe for the live app.
- **`reasoning_effort` threading:** new `src/ai/router.py::thinking_params_for_model(model, effort) -> dict` (per-provider: Anthropic `thinking={type:enabled,budget_tokens:…}`, OpenAI `reasoning_effort`; others `{}`). `_chat_events`/`chat` accept `reasoning_effort` and pass `**thinking_params` into `_completion`. `effort=None`/`"low"` → `{}` (fastest).

### API — the SSE endpoint

- **New** `POST /api/chat/send-stream` (router `api/routers/chat.py`) → `StreamingResponse(svc.send_stream(...), media_type="text/event-stream")`. The existing `POST /api/chat/send` is **unchanged** (fallback + back-compat).
- **`ChatService.send_stream(chat_user_id, message, model, conversation_id?, web_search?, deep_research?, reasoning_effort?, page?, viewer_team?) -> Generator[str]`** — mirrors `send`'s pre-call validation (key check, over-cap → a single `error` SSE event, history load, system prompt), then iterates `_chat_events`, formatting each as an SSE frame `f"data: {json}\n\n"`. **On the `done` event (in a `try/finally`): record `budget.record_usage` + append history server-side** — so usage is metered even if the client disconnects mid-stream. The `done` SSE frame carries `conversation_id` + `cost_usd` for the UI.
- **Request contract** (`api/contracts/chat.py`): add `reasoning_effort: Literal["off","low","medium","high"] | None = None` to `ChatSendRequest` (used by both `/send` and `/send-stream`; `/send` ignores it harmlessly or threads it too).
- **Auth:** `require_app_user`-gated like all `/api/chat/*` (401 when no Clerk identity). Never 500 — provider/key/over-cap errors become an `error` SSE event (HTTP already 200).

### Frontend (`web/`)

- **New** `web/src/lib/api/bubba.ts::sendStream(body, onEvent) : Promise<void>` — `fetch("/api/chat/send-stream", {…Bearer…})` → `res.body.getReader()` + `TextDecoder`, buffering on `\n\n`, parsing `data: …` JSON frames, calling `onEvent(event)` per frame. (Mirrors `apiPost`'s auth-header logic; throws `ApiError` on non-OK.)
- **`Bubba.tsx`:** the send handler calls `sendStream` instead of `bubba.send`; on `text_delta` it appends to the in-flight assistant message (incremental render); on `tool_started`/`tool_result` it shows a transient **status chip** ("querying the database…"); on `done` it finalizes (conversation_id, cost, `sending=false`); on `error` it renders the graceful message. Add to the composer: the **effort dial** (Off/Low/Med/High, default Off) + **Web-search** / **Deep-research** toggles, threaded into the request body. `ChatSendBody` gains `reasoning_effort?`.

## Error handling

Backend never 500s on chat: the generator wraps the engine call; any provider/key/over-cap/exception becomes `{"type":"error","message": "..."}` then closes. Over-cap (managed free-taste exhausted) → an `error` event the UI can render as an upgrade prompt (B3 styles it). Usage recording is in `try/finally` so a mid-stream disconnect still meters.

## Testing

- **`_chat_events` / `chat` equivalence (the safety guard):** with a monkeypatched fake `providers._completion` (scripted deltas incl. a tool round), assert (a) `_chat_events` yields the expected `text_delta`/`tool_started`/`done` sequence, and (b) `chat()` (the drain) returns the SAME dict it returns today — a frozen-reference test so the live Streamlit path is provably unchanged.
- **`thinking_params_for_model`:** Anthropic → `thinking` budget; OpenAI → `reasoning_effort`; unknown provider / `off` / `low` → `{}`. Table-driven.
- **`ChatService.send_stream`:** DB-free with a fake provider + in-memory budget/history stores — assert SSE frames are well-formed (`data: …\n\n`), the `error` path on a missing key / over-cap, and that **usage + history are recorded even when the consumer stops early** (disconnect simulation: drain only the first N frames, assert `record_usage` still ran via the `finally`).
- **Router:** `/send-stream` returns `text/event-stream` + delegates (fake-service override); `/send` unchanged. Logic-free router guard. openapi regen (the streaming route has no response model schema — `StreamingResponse`; confirm the snapshot updates cleanly).
- **Tool surface stays read-only** (the parent spec's invariant) — unaffected, but re-assert.
- **Frontend:** preview — tokens render incrementally, status chips appear during a tool round, the effort dial + toggles thread into the request (verify via the network panel / a fake-streamed response).

## Risks

- **Touching `providers.chat` (live app).** Mitigated by the drain-keeps-shape design + the equivalence guard test. If the refactor proves risky, fall back to a **separate `chat_stream` generator** that reuses the tool-dispatch helpers without touching `chat()` (some loop duplication, zero live-app risk) — decide at build time based on the guard test.
- **LiteLLM streaming + tool_calls + usage** depends on `stream_chunk_builder` + `include_usage` coverage per provider. Pin LiteLLM; verify on Anthropic + OpenAI + DeepSeek before merge; degrade gracefully (if a provider can't stream, `send_stream` falls back to a single `text_delta` of the full `chat()` result + `done`).
- **Per-provider effort fragmentation** — centralized in `thinking_params_for_model`; unsupported → no-op (the dial still shows but is a silent no-op on those models; documented).
- **Client disconnect** — usage recorded server-side in `finally`; the UI may miss the trailing `done` metadata (acceptable).

## Index

- New: `_chat_events` + `thinking_params_for_model` (`src/ai/`), `ChatService.send_stream` + `POST /api/chat/send-stream`, `bubba.sendStream` + the streaming composer (`web/`).
- Reuses unchanged: `tools.dispatch_tool`, `search.{web_search,deep_research}`, `budget`, `history`, `keys`, the B1 popup shell.
- Parent: the Bubba design spec (Phase B2). Engine being wrapped: `src/ai/` (memory `project_ai_chat_assistant`).

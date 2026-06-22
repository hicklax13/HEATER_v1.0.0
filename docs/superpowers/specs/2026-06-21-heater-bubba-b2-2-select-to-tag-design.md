# HEATER "Bubba" B2.2 — Tag Page Context to Bubba (Design Spec)

**Date:** 2026-06-21
**Lane:** CEO — full-stack (`api/` + `src/ai/` + `web/`).
**Parent spec:** `docs/superpowers/specs/2026-06-20-heater-bubba-ai-assistant-design.md` (Phase B2, second wave).
**Sibling:** `docs/superpowers/specs/2026-06-21-heater-bubba-b2-1-streaming-design.md` (B2.1, shipped).
**Status:** SHIPPED 2026-06-21 — built via plan `docs/superpowers/plans/2026-06-21-heater-bubba-b2-2-select-to-tag.md` (6 tasks / 3 slices; both pr-review-toolkit reviewers ran, findings applied: snapshot-failure logging + cross-origin fidelity caveat + defensive tag parse). Deferred (LOW, documented): the transcript "+context note" was skipped to keep `send`/`send_stream` persistence consistent (bare message).

## Scope

The second B2 wave — let the user hand Bubba context **from the page they're looking at**, three ways:

1. **Highlight text** anywhere on a page → it becomes a removable chip in the composer.
2. **Click a player row / card** (in select-mode) → grabs that player's structured data (name + id), not just the visible text.
3. **Snapshot the page** → attaches an image of the current HEATER page so a vision model can look at it.

A **select-mode** toggle in the composer arms #1 and #2 and shows a thin orange **page-edge glow** so the mode is visibly on. Tagged items render as removable chips above the input; on send they travel with the question.

**Out of scope (later waves):** device-window capture via `getDisplayMedia` (B2.3 proper), message queue + saved prompts (B2.4). This wave does **page** screenshots only (in-DOM `html-to-image`), not arbitrary device windows.

## Grounding (verified in source)

- The live Streamlit chat already supports "attach page text as context": `src/ai/chat.py` wraps the message as `f"[Context the user selected on the page]\n{attached}\n\n[Question]\n{prompt}"` (state key `_STATE_ATTACHED`). **B2.2 mirrors this exact wrapping** so the React + Streamlit paths behave identically.
- `api/contracts/chat.py::ChatSendRequest` currently has `message, model, conversation_id?, web_search, deep_research, reasoning_effort?` — **no** `attached_text`/`attachments` yet. The parent spec's send body already anticipates both (`{… attached_text?, attachments?}`).
- `api/services/chat_service.py::ChatService.send` + `send_stream` build the user turn as `convo.append({"role": "user", "content": message})`. That `content` is where text-wrapping and multimodal construction happen.
- `src/ai/providers.py::_chat_events`/`chat` pass `messages` straight through to litellm (`_completion`). **litellm accepts OpenAI-style multimodal content** (`content` as a list of `{"type":"text",…}` / `{"type":"image_url","image_url":{"url":"data:image/png;base64,…"}}` parts) and routes to the model's vision API. So **providers needs NO change** — chat_service constructs the multimodal message; the loop is content-agnostic.
- Frontend: `web/src/components/player/PlayerLink.tsx` is the **canonical** player element (every player name/row renders through it) — the one focused place to add a `data-bubba-tag`. `web/src/components/bubba/Bubba.tsx` is the panel (post-B2.1 it streams via `bubba.sendStream`). `web/package.json` has **no** image library yet — Slice C adds one (`html-to-image`).
- `web/src/lib/api/bubba.ts::ChatSendBody` + `sendStream` are the request shape the panel sends; both gain the new fields.

## Architecture

### Backend — additive context fields, live-app-safe

- **Contract** (`api/contracts/chat.py`): `ChatSendRequest` gains
  - `attached_text: str | None = None` — the concatenated text/row tags.
  - `attachments: list[ChatAttachment] | None = None` where `ChatAttachment = {kind: Literal["image"], data_url: str}` (a `data:image/png;base64,…` URL). Kept a typed list so non-image kinds can be added later without a contract break.
- **`ChatService`** (`send` + `send_stream`): a shared helper `_build_user_content(message, attached_text, attachments) -> str | list` —
  - text-only (no attachments): returns a **string**. If `attached_text` is set, wrap as `[Context the user selected on the page]\n{attached_text}\n\n[Question]\n{message}` (mirrors Streamlit); else the bare `message`. **So with no attachments the user turn is byte-identical to today** (a guard test pins this).
  - with image attachments: returns a **list** of parts — the (optionally context-wrapped) text part + one `{"type":"image_url","image_url":{"url": data_url}}` per attachment.
  - Both `send` and `send_stream` call this for the user turn. Everything downstream (metering, history, the tool loop) is unchanged. History persists the *original* `message` (+ a short note that context/an image was attached) — not the wrapped/multimodal blob.
- **`providers.py`: unchanged.** The multimodal message flows through `_chat_events`→`_completion`→litellm. `chat()`'s dict contract is untouched; the B2.1 equivalence guard still holds. A model that isn't vision-capable → litellm raises → caught → graceful `error` (frame or body), never a 500.

### Frontend (`web/`) — a self-contained select layer in Bubba

- **State (in `BubbaPanel`):** `selectMode: boolean`, `tags: Tag[]` where `Tag = {id: string, kind: "text"|"player", label: string, text: string}` (the `text` is what goes into `attached_text`; `label` is the chip caption), and `shots: {id, dataUrl}[]` for screenshots.
- **Select-mode toggle:** a crosshair icon button in the composer controls row (next to the B2.1 effort dial / toggles). Arming it:
  - renders a **page-edge glow** — a `fixed inset-0 pointer-events-none` ring overlay (orange, low-opacity, `z` below the panel) so the page visibly shows the mode is on. Rendered by Bubba itself (no TopBar change).
  - **text tagging (Slice A):** a `mouseup` listener reads `window.getSelection()`; a non-empty selection whose anchor is **outside** the Bubba panel becomes a `text` tag (truncated label, full text stored). Works on every page, no per-page wiring.
  - **row tagging (Slice B):** a capture-phase `click` listener finds `event.target.closest("[data-bubba-tag]")`; if found, it `preventDefault`s the normal click and reads the element's `data-bubba-tag` (a pipe-delimited `name|mlbId` string) into a `player` tag. `PlayerLink` gains `data-bubba-tag={`${name}|${mlbId}`}`. Only that one shared component is touched; rows/cards built from it inherit it.
- **Screenshot (Slice C):** a camera button calls `html-to-image`'s `toPng` on the main app content node, **excluding** the Bubba panel (a `filter` that drops the panel subtree), → a `data:image/png` URL added to `shots`. Downscaled (cap width, e.g. 1280px) to keep payload + tokens sane.
- **Chips:** tags + shots render as removable chips above the composer input. Sending clears them.
- **Send:** `handleSend` passes `attached_text` (joined tag `text`s) + `attachments` (shots as `{kind:"image", data_url}`) into `bubba.sendStream`. `ChatSendBody` gains both fields.

### Data flow

`arm select-mode → glow on → user highlights text / clicks a player row / snaps the page → chips appear → user types a question → sendStream({message, attached_text, attachments, …}) → ChatService._build_user_content wraps text + builds multimodal content → _chat_events streams as usual → answer streams back`.

## Error handling

- No attachments + no `attached_text` → byte-identical to today's send (guarded).
- A vision-incapable model with an image attachment → litellm error → existing graceful path (`error` frame / body), never 500.
- `html-to-image` failure (tainted canvas, etc.) → the camera button surfaces a small inline "couldn't snapshot the page" and adds nothing; the chat still works.
- An over-large image → frontend downscales before send; backend doesn't trust size (the daily cap + provider limits apply).
- A `data-bubba-tag` with malformed payload → fall back to the element's text content; never throw.

## Testing

- **Backend (DB-free):** `_build_user_content` — (a) no attachments/no text → returns the bare `message` string (equivalence to today); (b) `attached_text` only → the exact `[Context…][Question…]` wrapped string; (c) image attachment → a list with a text part + an `image_url` part carrying the data URL; (d) text + image → wrapped text part + image part. Contract tests for the new fields' defaults. `send`/`send_stream` thread the fields (fake-provider, assert the constructed content reaches `providers`). Re-assert the B2.1 `chat()` equivalence guard is untouched.
- **Frontend:** tsc + lint + production build per slice. Preview: arm select-mode (glow shows), highlight text → chip appears, click a player row → player chip, snapshot → image chip; verify the request body carries `attached_text`/`attachments` (network panel). (Live model round-trip is Clerk-gated — verify the request shape + graceful states, per the B2.1 preview limitation.)
- **Read-only invariant:** unchanged — re-assert `tests/api/test_chat_tool_surface_readonly.py` (tagging adds context, never a write tool).

## Risks

- **Multimodal token cost** — a full-page PNG is large. Mitigate: downscale client-side; the existing daily cap still meters. Documented; tune later.
- **`html-to-image` fidelity** — it rasterizes the DOM, not a true browser screenshot; some canvas/cross-origin images may render blank. Acceptable for "show Bubba roughly what I see"; the device-window true-capture is the later B2.3 path.
- **Re-touching the user-turn construction** (used by the live Streamlit app's sibling flow) — mitigated: the change is in `chat_service` (the React seam), not `src/ai`; `providers.chat`/`_chat_events` are untouched and stay equivalence-guarded.
- **Click-capture interfering with normal clicks** — only active while `selectMode` is on; disarming removes the listener. Selections inside the panel are ignored.

## Slices (build order)

- **Slice A — highlight-text tagging:** backend `attached_text` + `_build_user_content` text path + wiring into `send`/`send_stream`; frontend select-mode toggle + page glow + text-selection tagging + chips + send. Ship + verify.
- **Slice B — click rows/cards:** `data-bubba-tag` on `PlayerLink`; the capture-phase click→`player` tag. (Backend already done in A — a player tag is just more `attached_text`.)
- **Slice C — page screenshot:** `attachments` contract + `_build_user_content` multimodal path; add `html-to-image`; camera button + downscale + image chips + send.

## Index

- New: `ChatAttachment` contract + `attached_text`/`attachments` on `ChatSendRequest`; `ChatService._build_user_content`; `bubba` `sendStream` body fields + a Bubba select layer (toggle, glow, text/row tagging, screenshot, chips); `data-bubba-tag` on `PlayerLink`; `html-to-image` dep.
- Reuses unchanged: `providers.chat`/`_chat_events` (multimodal passes through), B2.1 streaming + metering, the B1 popup shell.
- Parent: the Bubba design spec (Phase B2). Engine wrapped: `src/ai/` (memory `project_ai_chat_assistant`, `project_bubba_ai_assistant`).

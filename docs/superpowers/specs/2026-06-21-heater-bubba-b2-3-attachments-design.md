# HEATER "Bubba" B2.3 — Attachments: Files, Screen Capture, Documents (Design Spec)

**Date:** 2026-06-21
**Lane:** CEO — full-stack (mostly `web/`; `api/` essentially unchanged).
**Parent spec:** `docs/superpowers/specs/2026-06-20-heater-bubba-ai-assistant-design.md` (Phase B2, third wave).
**Siblings:** B2.1 streaming (`...-b2-1-streaming-design.md`, shipped) + B2.2 tag-page-context (`...-b2-2-select-to-tag-design.md`, shipped).
**Status:** SHIPPED 2026-06-21 — built via plan `docs/superpowers/plans/2026-06-21-heater-bubba-b2-3-attachments.md` (4 tasks / 3 slices; frontend-only, no backend change; both pr-review-toolkit reviewers ran — "sound and ready" / "zero silent failures", applied LOW-1: PDF truncation now shown on the doc chip). `pdfjs-dist` 6.0.227 added; the `new URL(worker, import.meta.url)` pattern bundled cleanly in Next 16 (no `/public` fallback needed).

## Scope

Three new ways to hand Bubba context, all feeding the **two request fields B2.2 already built** (`attachments` for images, `attached_text` for text):

1. **Image files** — a paperclip button → file picker (PNG/JPG/etc.) → data URL → `attachments` (the existing image path).
2. **Screen capture** — a "share screen" button → `navigator.mediaDevices.getDisplayMedia()` (the user picks any open window/screen) → grab one frame → data URL → `attachments`.
3. **PDFs / documents** — file picker (PDF) → **extract the text in the browser** (`pdfjs-dist`) → folded into `attached_text` as `[Document: <name>]\n<text>`.

**The load-bearing decision — documents go as extracted TEXT, not raw-PDF passthrough.** Native PDF input only works on a couple of providers (Anthropic/Gemini) and breaks on others incl. the default DeepSeek shared model — fragile + provider-fragmented. Text extraction is **universal** (works on EVERY model, since it's just text), needs **zero backend change**, and captures what matters for fantasy content (articles, rankings — mostly text). Tradeoff: it loses charts/images inside the PDF. So the clean split is: **images need a vision model** (same as B2.2's page snapshot); **documents work on any model**.

**Out of scope (later):** native vision-PDF passthrough (raw PDF to a doc-capable model), message queue + saved prompts (B2.4).

## Grounding (verified in source)

- **B2.2 already shipped the backend this wave needs:** `api/contracts/chat.py::ChatSendRequest` has `attached_text: str | None` + `attachments: list[ChatAttachment]` (`ChatAttachment = {kind:"image", data_url}`); `api/services/chat_service.py::_build_user_content` wraps `attached_text` as `[Context the user selected on the page]…[Question]…` and builds the multimodal parts list for images. **No `api/` change is required** for B2.3 — images are more `attachments`, document text is more `attached_text`.
- `web/src/components/bubba/Bubba.tsx` (post-B2.2): `BubbaPanel` holds `tags` (text/player → `attached_text`) + `shots` (page snapshots → `attachments`) + `snapError`, renders removable `TagChip`s, and `handleSend` captures+clears then sends `attached_text` (joined `tags[].text`) + `attachments` (mapped `shots`). The composer controls row has the B2.1 effort dial / toggles + the B2.2 Tag / Snap buttons.
- `getDisplayMedia` returns a `MediaStream`; a frame is grabbed by drawing the stream's video track to a `<canvas>` then `canvas.toDataURL("image/png")`, then stopping the track. Browser-only, user-gesture + permission gated.
- `pdfjs-dist` (v4+, ESM) extracts text via `getDocument({data}).promise` → per-page `getTextContent()`. It needs `GlobalWorkerOptions.workerSrc` set to the bundled worker URL (`new URL("pdfjs-dist/build/pdf.worker.min.mjs", import.meta.url)`). **`web/AGENTS.md` warns this Next.js has breaking changes — verify the worker-URL pattern against `node_modules/next/dist/docs/` at build time.**
- `web/src/lib/api/bubba.ts::ChatSendBody` already carries `attached_text?` + `attachments?` — no client-type change needed.

## Architecture

### Backend — unchanged

No `api/` or `src/ai/` change. Images ride `attachments` (B2.2's multimodal `_build_user_content`); document text rides `attached_text` (B2.2's context-wrap). The B2.1 `chat()` equivalence guard and the live Streamlit app stay untouched. (One optional safety tweak considered + rejected as YAGNI: a server-side `attached_text` length cap — the daily usage cap already meters oversized text.)

### Frontend (`web/src/components/bubba/Bubba.tsx`) — the whole wave

- **Unify image sources.** Rename the B2.2 `shots` state → `images` (semantics: "image attachments"), since it now holds page snapshots **and** picked image files **and** screen captures — all `{id, label, dataUrl}` images that map to `attachments`. (A focused rename across the state, chips, `snapPage`, send-mapping, and clear sites.)
- **Add `docs` state** — `{id, label, text}[]` extracted PDF text, folded into `attached_text` on send as `[Document: <label>]\n<text>`.
- **Paperclip button + hidden file input** (`accept="image/*,application/pdf" multiple`): on change, route each file by type —
  - image → `FileReader.readAsDataURL` → push to `images`.
  - PDF → a `extractPdfText(file)` helper (dynamic `import("pdfjs-dist")`, capped to ~12k chars with a "(truncated)" note) → push to `docs`.
  - anything else → ignored (with a brief inline note).
- **Screen-capture button** → `captureScreen()` helper: `getDisplayMedia({video:true})` → draw a frame to canvas → `toDataURL` → push to `images` → stop the track. Hidden/disabled when `navigator.mediaDevices?.getDisplayMedia` is absent.
- **Chips:** `images` (image chip, with the B2.2 Camera-style icon) + `docs` (document chip, file icon) + the existing `tags` chips. All removable.
- **Send:** `attached_text` = `[...tags.map(t=>t.text), ...docs.map(d=>`[Document: ${d.label}]\n${d.text}`)].join("\n")` (undefined if empty); `attachments` = `images.map(i=>({kind:"image", data_url:i.dataUrl}))` (undefined if empty). Capture + clear all three (`tags`/`images`/`docs`) before the await, as B2.2 does for tags/shots.

### Data flow

`paperclip → pick file(s)` OR `share-screen → pick window` OR (B2.2) `Tag/Snap` → chips accumulate → type a question → `sendStream({message, attached_text, attachments, …})` → B2.2's `_build_user_content` wraps text + builds multimodal content → streams back.

## Error handling

- **Image read failure** (`FileReader.onerror`) → inline "Couldn't read that file." note; skip it.
- **PDF parse failure** (corrupt/encrypted PDF, worker load failure) → `console.error(cause)` + inline "Couldn't read that PDF." (mirror the B2.2 `snapPage` discipline — log the cause, don't swallow silently); skip it.
- **`getDisplayMedia` denied / unsupported** → inline "Screen capture was cancelled or isn't available."; the stream's tracks are always stopped (no dangling capture).
- **Oversized inputs** → image files capped (e.g. reject > ~8 MB with a note); extracted doc text truncated to ~12k chars + a visible "(truncated)" marker so the user knows it's partial.
- **Vision-model requirement** — an image attachment to a non-vision model degrades to B2.2's existing graceful `error` (never a 500). Documents (text) work on any model.
- Every failure is **per-item**: a bad file never blocks the others or the chat.

## Testing

- **Backend:** no code change → no new backend tests; re-run the chat suite to confirm B2.1/B2.2 invariants (incl. the `chat()` equivalence guard) still pass.
- **Frontend extraction helpers are factored for testability where practical:** `extractPdfText` and `captureScreen` live in a small `web/src/components/bubba/attachments.ts` so the component stays focused. `pdfjs`/`getDisplayMedia` are browser-only, so the gate is tsc + lint + production build per slice, plus a preview smoke (the panel mounts; the paperclip / share-screen / send wiring is present). A jsdom unit test of `extractPdfText` is impractical (worker + binary fixture) — documented; the build + a real-session manual check cover it.
- **Read-only invariant** unchanged — re-assert `tests/api/test_chat_tool_surface_readonly.py`.
- **Preview limitation** (same as B2.1/B2.2): the composer needs a Clerk session to reach "ready", so live attachment round-trips are verified by the owner post-deploy; locally verify the component mounts + builds clean.

## Risks

- **`pdfjs-dist` worker wiring in Next.js 16** — the trickiest bit; the `new URL(worker, import.meta.url)` pattern is the target, verified against the bundled Next docs at build time. Fallback if the worker won't bundle: pin a known-good `pdfjs-dist` major and/or set `workerSrc` to a versioned static asset. The plan nails the exact wiring.
- **`getDisplayMedia` is untestable locally** (needs a real session + a user picking a window) — pure frontend reusing the proven image path; graceful when absent.
- **Large extracted text** inflates tokens — capped client-side + the daily cap meters it; documented.
- **`shots`→`images` rename** touches B2.2 code — a focused mechanical rename, covered by tsc + build; the behavior (page snapshot → image attachment) is unchanged.
- **Bundle size** — `pdfjs-dist` is sizable; mitigated by **dynamic `import()`** so it loads only when a PDF is actually attached (never in the main bundle).

## Slices (build order)

- **Slice A — image-file attach:** paperclip + hidden file input (images only first) + `images` rename/consolidation + image chips + send. (Establishes the file-picker + unified image state.)
- **Slice B — screen capture:** `captureScreen()` + the share-screen button → `images`. (Reuses A's image chips + send.)
- **Slice C — PDF documents:** add `pdfjs-dist`; `extractPdfText()` + route PDFs from the paperclip → `docs` + doc chips + fold into `attached_text`.

## Index

- New: `web/src/components/bubba/attachments.ts` (`extractPdfText`, `captureScreen`); Bubba paperclip + share-screen buttons; `images` (renamed from `shots`) + `docs` state + chips; `pdfjs-dist` dep (dynamic import).
- Reuses unchanged: the B2.2 `attached_text`/`attachments` contract + `_build_user_content`; B2.1 streaming + metering; `src/ai/` (untouched).
- Parent: the Bubba design spec (Phase B2). Memory: `project_bubba_ai_assistant`.

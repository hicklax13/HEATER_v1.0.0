# Bubba B2.2 — Tag Page Context to Bubba Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user attach page context to a Bubba question three ways — highlight text, click a player row/card, or snapshot the page — via a composer select-mode, with the tagged context traveling to the model.

**Architecture:** Additive + live-app-safe. The backend gains `attached_text` + `attachments` on `ChatSendRequest`; a shared `ChatService._build_user_content` wraps text exactly like the live Streamlit app (`[Context…][Question…]`) and builds an OpenAI-style **multimodal** message for images. `src/ai/providers.py` is **untouched** — litellm consumes list-content messages natively, so `chat()`'s dict + the B2.1 equivalence guard are unaffected. The frontend adds a self-contained select layer to `Bubba.tsx` (text selection + a capture-phase click reading `data-bubba-tag` off the canonical `PlayerLink`, plus an `html-to-image` page snapshot).

**Tech Stack:** FastAPI 0.137.1 (pinned), Pydantic v2, `src/ai/` (litellm, multimodal passthrough), pytest. Frontend: Next.js 16 + React 19 + TS, `window.getSelection`, capture-phase listeners, `html-to-image` (new dep). **No backend deps.**

**Coordination:** Touches `api/` + `web/` only; `src/ai/` is NOT touched (the B2.1 streaming core stays frozen). Build in three slices (A text → B rows → C screenshot); each ships + verifies independently.

---

## Grounding (verified in source — do NOT re-derive)

- `src/ai/chat.py` (live Streamlit) wraps tagged context as `f"[Context the user selected on the page]\n{attached}\n\n[Question]\n{prompt}"`. **Mirror this exact string.**
- `api/contracts/chat.py::ChatSendRequest` = `message, model, conversation_id?, web_search, deep_research, reasoning_effort?` (B2.1). No `attached_text`/`attachments` yet.
- `api/services/chat_service.py`: both `send` (line ~68) and `send_stream` (line ~190) build the user turn as `convo.append({"role": "user", "content": message})`. History persistence stores the raw `message` and stays UNCHANGED (the attached context is ephemeral, not persisted — a deliberate simplification).
- `src/ai/providers.py::_chat_events`/`chat` pass `messages` straight to `_completion` (litellm) and never introspect user content → a list-content (multimodal) user message flows through untouched. **No providers change.**
- `web/src/components/player/PlayerLink.tsx` renders a `<button>` inside `<PlayerDialog>`; `DialogPlayer` has `name` + `mlbId`. Add `data-bubba-tag` to that button.
- `web/src/components/bubba/Bubba.tsx` (post-B2.1): `BubbaPanel` has `effort`/`webSearch`/`deepResearch`/`toolStatus` state, a composer controls row (effort dial + `Toggle`s), and `handleSend` calls `bubba.sendStream({...}, onEvent)`. The panel is a `motion.div role="dialog"` at `z-[60]`.
- `web/src/lib/api/bubba.ts::ChatSendBody` = the request body type (gains the two fields); `sendStream` already posts it.
- `web/package.json` has no image lib.

**Venv python:** `.venv/Scripts/python.exe`. Frontend from `web/` with `pnpm`. Run git from the repo root `C:/Users/conno/Code/HEATER_v1.0.1`.

---

## File Structure

- **Modify** `api/contracts/chat.py` — add `ChatAttachment` + `attached_text`/`attachments` on `ChatSendRequest`.
- **Modify** `api/services/chat_service.py` — add module-level `_build_user_content(...)`; thread `attached_text`/`attachments` into `send` + `send_stream`'s user turn.
- **Modify** `api/routers/chat.py` — pass `attached_text`/`attachments` from the body into `send` + `send_stream`.
- **Modify** `api/openapi.json` — regenerate.
- **Modify** `web/src/lib/api/bubba.ts` — `ChatSendBody` gains `attached_text?`/`attachments?`.
- **Modify** `web/src/components/bubba/Bubba.tsx` — select layer (toggle, glow, text + row tagging, screenshot, chips), send wiring, `data-bubba-panel` marker.
- **Modify** `web/src/components/player/PlayerLink.tsx` — `data-bubba-tag`.
- **Modify** `web/package.json` / lockfile — add `html-to-image`.
- **Create** `tests/api/test_chat_user_content.py` — `_build_user_content` unit tests.
- **Modify** `tests/api/test_chat_contracts.py`, `tests/api/test_chat_service.py`, `tests/api/test_api_chat.py` — new-field coverage.

---

## SLICE A — Highlight-text tagging

### Task 1: Backend — `attached_text` contract + `_build_user_content` (text path) + wiring

**Files:**
- Modify: `api/contracts/chat.py`, `api/services/chat_service.py`, `api/routers/chat.py`
- Create: `tests/api/test_chat_user_content.py`
- Modify: `tests/api/test_chat_contracts.py`, `tests/api/test_chat_service.py`

- [ ] **Step 1: Write the failing `_build_user_content` tests**

```python
# tests/api/test_chat_user_content.py
"""_build_user_content wraps tagged text like the live Streamlit app and builds
multimodal content for image attachments. DB-free."""

from types import SimpleNamespace

from api.services.chat_service import _build_user_content


def test_no_attached_text_no_attachments_returns_bare_message():
    # byte-identical to today's user turn (the live-app-safe invariant)
    assert _build_user_content("who is hot?", None, None) == "who is hot?"


def test_attached_text_wraps_like_streamlit():
    out = _build_user_content("is this good?", "Mike Trout .312 AVG", None)
    assert out == "[Context the user selected on the page]\nMike Trout .312 AVG\n\n[Question]\nis this good?"


def test_image_attachment_builds_multimodal_list():
    att = SimpleNamespace(kind="image", data_url="data:image/png;base64,AAA")
    out = _build_user_content("what's wrong here?", None, [att])
    assert isinstance(out, list)
    assert out[0] == {"type": "text", "text": "what's wrong here?"}
    assert out[1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}


def test_text_plus_image_wraps_text_then_image():
    att = SimpleNamespace(kind="image", data_url="data:image/png;base64,BBB")
    out = _build_user_content("explain", "ERA 5.40", [att])
    assert out[0]["text"].startswith("[Context the user selected on the page]\nERA 5.40")
    assert out[1]["image_url"]["url"] == "data:image/png;base64,BBB"


def test_non_image_attachment_is_skipped():
    bad = SimpleNamespace(kind="audio", data_url="data:audio/x")
    # no usable image -> falls back to the plain text string
    assert _build_user_content("hi", None, [bad]) == "hi"
```

- [ ] **Step 2: Run — expect FAIL** (`cannot import name '_build_user_content'`)

Run: `.venv/Scripts/python.exe -m pytest tests/api/test_chat_user_content.py -q`

- [ ] **Step 3: Implement `_build_user_content`** — add to `api/services/chat_service.py` (after the `_sse` helper):

```python
def _build_user_content(message: str, attached_text: str | None, attachments: list | None):
    """Build the user-turn content. Returns a STRING (today's shape) unless image
    attachments are present, then an OpenAI-style multimodal parts list. Text tags
    are wrapped exactly like the live Streamlit app so both paths behave identically."""
    if attached_text:
        text = f"[Context the user selected on the page]\n{attached_text}\n\n[Question]\n{message}"
    else:
        text = message

    image_urls = [
        getattr(a, "data_url", None)
        for a in (attachments or [])
        if getattr(a, "kind", None) == "image" and getattr(a, "data_url", None)
    ]
    if not image_urls:
        return text  # byte-identical to today when no text tag + no image
    return [{"type": "text", "text": text}] + [
        {"type": "image_url", "image_url": {"url": url}} for url in image_urls
    ]
```

- [ ] **Step 4: Run — expect PASS**

Run: `.venv/Scripts/python.exe -m pytest tests/api/test_chat_user_content.py -q`

- [ ] **Step 5: Add the contract fields** — `api/contracts/chat.py`. Add `ChatAttachment` and the two fields:

```python
class ChatAttachment(BaseModel):
    kind: Literal["image"] = "image"
    data_url: str  # a data:image/...;base64,... URL


class ChatSendRequest(BaseModel):
    message: str
    model: str
    conversation_id: int | None = None
    web_search: bool = False
    deep_research: bool = False
    reasoning_effort: Literal["off", "low", "medium", "high"] | None = None
    attached_text: str | None = None
    attachments: list[ChatAttachment] | None = None
```

- [ ] **Step 6: Add contract defaults test** — append to `tests/api/test_chat_contracts.py`:

```python
def test_send_request_attach_fields_default_none():
    from api.contracts.chat import ChatAttachment, ChatSendRequest

    req = ChatSendRequest(message="hi", model="m")
    assert req.attached_text is None and req.attachments is None
    req2 = ChatSendRequest(
        message="hi", model="m", attached_text="ctx",
        attachments=[ChatAttachment(kind="image", data_url="data:image/png;base64,A")],
    )
    assert req2.attachments[0].data_url.startswith("data:image/png")
```

- [ ] **Step 7: Thread into `send` + `send_stream`** — `api/services/chat_service.py`.

In `send`, add params (after `deep_research`):
```python
        deep_research: bool = False,
        reasoning_effort: str | None = None,
        attached_text: str | None = None,
        attachments: list | None = None,
        page: str | None = None,
        viewer_team: str | None = None,
    ) -> dict:
```
and change the user-turn line:
```python
            convo.append({"role": "user", "content": _build_user_content(message, attached_text, attachments)})
```

In `send_stream`, add the SAME two params (after `reasoning_effort`):
```python
        reasoning_effort: str | None = None,
        attached_text: str | None = None,
        attachments: list | None = None,
        page: str | None = None,
        viewer_team: str | None = None,
    ) -> Generator[str, None, None]:
```
and change its user-turn line the same way:
```python
            convo.append({"role": "user", "content": _build_user_content(message, attached_text, attachments)})
```

- [ ] **Step 8: Add the service threading test** — append to `tests/api/test_chat_service.py`:

```python
def test_send_threads_attached_text_into_user_turn(monkeypatch):
    captured = {}
    monkeypatch.setattr(cs, "_provider_of", lambda m: "openai")
    monkeypatch.setattr(cs.keys, "get_key", lambda uid, prov: "sk-user")
    monkeypatch.setattr(cs.keys, "list_keys", lambda uid: [{"provider": "openai"}])
    monkeypatch.setattr(cs.budget, "is_over_cap", lambda uid, on_own_key=False: False)
    monkeypatch.setattr(cs, "_build_system_prompt", lambda page, team: "SYS")
    monkeypatch.setattr(cs.history, "load_messages", lambda *a, **k: [])
    monkeypatch.setattr(cs.history, "create_conversation", lambda *a, **k: 1)
    monkeypatch.setattr(cs.history, "append_message", lambda *a, **k: 1)
    monkeypatch.setattr(cs, "_price_per_token", lambda m: (0.0, 0.0))
    monkeypatch.setattr(cs.budget, "record_usage", lambda *a, **k: None)

    def _chat(**k):
        captured.update(k)
        return {"content": "ok", "tokens_in": 1, "tokens_out": 1, "tool_trace": []}

    monkeypatch.setattr(cs.providers, "chat", _chat)
    ChatService().send(
        chat_user_id=1_000_000_001, message="good?", model="gpt-5", attached_text="Trout .312"
    )
    user_turn = captured["messages"][-1]
    assert user_turn["role"] == "user"
    assert user_turn["content"] == "[Context the user selected on the page]\nTrout .312\n\n[Question]\ngood?"
```

- [ ] **Step 9: Thread into the router** — `api/routers/chat.py`. In `send(...)`'s `svc.send(...)` kwargs add (after `reasoning_effort=body.reasoning_effort,`):
```python
            reasoning_effort=body.reasoning_effort,
            attached_text=body.attached_text,
            attachments=body.attachments,
        )
    )
```
and in `send_stream(...)`'s `svc.send_stream(...)` kwargs add the same two lines after `reasoning_effort=body.reasoning_effort,`.

- [ ] **Step 10: Run the backend slice + guards**

Run: `.venv/Scripts/python.exe -m pytest tests/api/test_chat_user_content.py tests/api/test_chat_contracts.py tests/api/test_chat_service.py tests/api/test_api_chat.py tests/api/test_no_logic_in_routers.py tests/test_ai_providers_streaming.py -q`
Expected: PASS (incl. the B2.1 equivalence guard — providers untouched).

- [ ] **Step 11: Regenerate OpenAPI + verify**

Run: `.venv/Scripts/python.exe scripts/export_openapi.py && .venv/Scripts/python.exe -m pytest tests/api/test_openapi_contract.py -q`
Expected: PASS.

- [ ] **Step 12: Commit**

```bash
git add api/contracts/chat.py api/services/chat_service.py api/routers/chat.py api/openapi.json tests/api/test_chat_user_content.py tests/api/test_chat_contracts.py tests/api/test_chat_service.py
git commit -m "feat(api): attached_text + multimodal _build_user_content for Bubba tagging (B2.2 slice A)"
```

---

### Task 2: Frontend — select-mode + page glow + text tagging + chips + send

**Files:**
- Modify: `web/src/lib/api/bubba.ts`, `web/src/components/bubba/Bubba.tsx`

- [ ] **Step 1: Add the body fields to `bubba.ts`** — extend `ChatSendBody`:

```typescript
export interface ChatSendBody {
  message: string;
  model: string;
  conversation_id?: number | null;
  web_search?: boolean;
  deep_research?: boolean;
  reasoning_effort?: "off" | "low" | "medium" | "high";
  attached_text?: string;
  attachments?: { kind: "image"; data_url: string }[];
}
```

- [ ] **Step 2: Add select-state + a panel ref to `BubbaPanel`** — `web/src/components/bubba/Bubba.tsx`. Add next to the other `useState`s + the `scrollRef`:

```typescript
  const [selectMode, setSelectMode] = useState(false);
  const [tags, setTags] = useState<{ id: string; kind: "text" | "player"; label: string; text: string }[]>([]);
  const [shots, setShots] = useState<{ id: string; dataUrl: string }[]>([]);
  const panelRef = useRef<HTMLDivElement>(null);
```

- [ ] **Step 3: Attach the ref + a panel marker to the panel root.** Find the `return ( <motion.div role="dialog" aria-label="Bubba AI assistant"` and add `ref={panelRef} data-bubba-panel="1"`:

```tsx
    <motion.div
      ref={panelRef}
      data-bubba-panel="1"
      role="dialog"
      aria-label="Bubba AI assistant"
```

- [ ] **Step 4: Add the text-selection effect.** Add after the existing effects (e.g. after the scroll effect):

```typescript
  // Select-mode: highlighting page text (outside the panel) tags it.
  useEffect(() => {
    if (!selectMode) return undefined;
    const onMouseUp = () => {
      const sel = window.getSelection();
      const text = sel?.toString().trim();
      if (!text) return;
      const anchor = sel?.anchorNode ?? null;
      if (anchor && panelRef.current?.contains(anchor)) return; // ignore in-panel selections
      const label = text.length > 48 ? text.slice(0, 48) + "…" : text;
      setTags((t) => [...t, { id: crypto.randomUUID(), kind: "text", label, text }]);
      sel?.removeAllRanges();
    };
    document.addEventListener("mouseup", onMouseUp);
    return () => document.removeEventListener("mouseup", onMouseUp);
  }, [selectMode]);
```

- [ ] **Step 5: Render the page-edge glow** when armed. Add as the FIRST child inside the returned `<motion.div>` fragment is awkward (it's a single root). Instead render it as a sibling before the panel — wrap the existing `return (<motion.div …>)` is one node; add the glow via a fragment. Change the `return (` to return a fragment:

```tsx
  return (
    <>
      {selectMode && (
        <div
          aria-hidden
          className="pointer-events-none fixed inset-0 z-[55] ring-2 ring-inset ring-heat/60"
        />
      )}
      <motion.div
        ref={panelRef}
        data-bubba-panel="1"
        role="dialog"
        aria-label="Bubba AI assistant"
        /* …existing props… */
      >
        {/* …existing panel body… */}
      </motion.div>
    </>
  );
```
(Keep all existing `motion.div` props + children; only wrap in the fragment and add the glow + `ref`/`data-bubba-panel`.)

- [ ] **Step 6: Render tag chips above the input.** In the composer block, immediately inside `<div className="shrink-0 border-t border-line bg-surface p-2">` and BEFORE the controls row, add:

```tsx
            {(tags.length > 0 || shots.length > 0) && (
              <div className="mb-2 flex flex-wrap gap-1.5">
                {tags.map((t) => (
                  <TagChip key={t.id} label={t.label} onRemove={() => setTags((x) => x.filter((y) => y.id !== t.id))} />
                ))}
                {shots.map((s) => (
                  <TagChip
                    key={s.id}
                    label="page snapshot"
                    icon
                    onRemove={() => setShots((x) => x.filter((y) => y.id !== s.id))}
                  />
                ))}
              </div>
            )}
```

- [ ] **Step 7: Add the select-mode toggle button** to the controls row (next to the B2.1 effort dial / `Toggle`s). Inside the `<div className="mb-2 flex items-center gap-2 text-[11px]">` row, after the Web/Research `Toggle`s, add:

```tsx
              <button
                type="button"
                onClick={() => setSelectMode((v) => !v)}
                aria-pressed={selectMode}
                title="Tag page content — highlight text or click a player"
                className={cn(
                  "ml-auto flex items-center gap-1 rounded-full border px-2.5 py-1 font-semibold transition-colors",
                  selectMode ? "border-heat bg-heat/10 text-heat" : "border-line bg-canvas text-ink-3 hover:text-ink",
                )}
              >
                <MousePointerClick className="size-3.5" aria-hidden /> Tag
              </button>
```
and add `MousePointerClick` to the lucide import line at the top:
```typescript
import { Flame, KeyRound, Loader2, MousePointerClick, Plus, Send, Settings, Trash2, X } from "lucide-react";
```

- [ ] **Step 8: Send the tags + clear them.** In `handleSend`, capture + clear before sending, and pass `attached_text`. Replace the `bubba.sendStream({...})` body object's first lines:

```typescript
    const sentTags = tags;
    const sentShots = shots;
    setTags([]);
    setShots([]);
    try {
      await bubba.sendStream(
        {
          message: text,
          model,
          conversation_id: conversationId,
          web_search: webSearch,
          deep_research: deepResearch,
          reasoning_effort: effort,
          attached_text: sentTags.length ? sentTags.map((t) => t.text).join("\n") : undefined,
          attachments: sentShots.length
            ? sentShots.map((s) => ({ kind: "image" as const, data_url: s.dataUrl }))
            : undefined,
        },
        (e) => {
```
and add `tags`, `shots` to the `handleSend` `useCallback` dependency array:
```typescript
  }, [input, sending, model, conversationId, webSearch, deepResearch, effort, tags, shots]);
```

- [ ] **Step 9: Add the `TagChip` component** (bottom of file, next to `Toggle`):

```tsx
function TagChip({ label, onRemove, icon }: { label: string; onRemove: () => void; icon?: boolean }) {
  return (
    <span className="inline-flex max-w-[200px] items-center gap-1 rounded-full border border-heat/40 bg-heat/5 px-2 py-0.5 text-[11px] text-ink">
      {icon && <Camera className="size-3 shrink-0 text-heat" aria-hidden />}
      <span className="truncate">{label}</span>
      <button type="button" onClick={onRemove} aria-label={`Remove ${label}`} className="text-ink-3 hover:text-ember">
        <X className="size-3" aria-hidden />
      </button>
    </span>
  );
}
```
and add `Camera` to the lucide import:
```typescript
import { Camera, Flame, KeyRound, Loader2, MousePointerClick, Plus, Send, Settings, Trash2, X } from "lucide-react";
```

- [ ] **Step 10: Typecheck + lint + build**

Run (from `web/`): `pnpm exec tsc --noEmit && pnpm run lint && pnpm build`
Expected: all green.

- [ ] **Step 11: Commit**

```bash
git add web/src/lib/api/bubba.ts web/src/components/bubba/Bubba.tsx
git commit -m "feat(web): Bubba select-mode + highlight-text tagging + chips (B2.2 slice A)"
```

---

## SLICE B — Click player rows/cards

### Task 3: `data-bubba-tag` on PlayerLink + capture-phase click tagging

**Files:**
- Modify: `web/src/components/player/PlayerLink.tsx`, `web/src/components/bubba/Bubba.tsx`

- [ ] **Step 1: Tag the canonical player element** — `web/src/components/player/PlayerLink.tsx`. Add `data-bubba-tag` to the `<button>`:

```tsx
      <button
        type="button"
        data-bubba-tag={`${player.name}|${player.mlbId ?? ""}`}
        className={cn(
```

- [ ] **Step 2: Add the capture-phase click effect** to `Bubba.tsx` (after the text-selection effect):

```typescript
  // Select-mode: clicking a player row/card (any element carrying data-bubba-tag)
  // tags that player INSTEAD of opening its dialog. Capture phase + stopPropagation
  // intercept the click before the PlayerDialog trigger sees it.
  useEffect(() => {
    if (!selectMode) return undefined;
    const onClick = (e: MouseEvent) => {
      const el = (e.target as HTMLElement | null)?.closest?.("[data-bubba-tag]");
      if (!el || panelRef.current?.contains(el)) return;
      e.preventDefault();
      e.stopPropagation();
      const raw = el.getAttribute("data-bubba-tag") ?? "";
      const [name, mlbId] = raw.split("|");
      const text = mlbId ? `Player: ${name} (mlbId ${mlbId})` : `Player: ${name}`;
      setTags((t) => [...t, { id: crypto.randomUUID(), kind: "player", label: name || "player", text }]);
    };
    document.addEventListener("click", onClick, true);
    return () => document.removeEventListener("click", onClick, true);
  }, [selectMode]);
```

- [ ] **Step 3: Typecheck + lint + build**

Run (from `web/`): `pnpm exec tsc --noEmit && pnpm run lint && pnpm build`
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add web/src/components/player/PlayerLink.tsx web/src/components/bubba/Bubba.tsx
git commit -m "feat(web): click-to-tag player rows in Bubba select-mode (B2.2 slice B)"
```

---

## SLICE C — Page screenshot

### Task 4: Backend — `attachments` already wired; confirm multimodal end-to-end

**Files:** Test only — `tests/api/test_chat_service.py` (the contract + `_build_user_content` multimodal path landed in Task 1; this locks the service threading for images).

- [ ] **Step 1: Write the multimodal threading test** — append to `tests/api/test_chat_service.py`:

```python
def test_send_threads_image_attachment_as_multimodal(monkeypatch):
    from types import SimpleNamespace

    captured = {}
    monkeypatch.setattr(cs, "_provider_of", lambda m: "openai")
    monkeypatch.setattr(cs.keys, "get_key", lambda uid, prov: "sk-user")
    monkeypatch.setattr(cs.keys, "list_keys", lambda uid: [{"provider": "openai"}])
    monkeypatch.setattr(cs.budget, "is_over_cap", lambda uid, on_own_key=False: False)
    monkeypatch.setattr(cs, "_build_system_prompt", lambda page, team: "SYS")
    monkeypatch.setattr(cs.history, "load_messages", lambda *a, **k: [])
    monkeypatch.setattr(cs.history, "create_conversation", lambda *a, **k: 1)
    monkeypatch.setattr(cs.history, "append_message", lambda *a, **k: 1)
    monkeypatch.setattr(cs, "_price_per_token", lambda m: (0.0, 0.0))
    monkeypatch.setattr(cs.budget, "record_usage", lambda *a, **k: None)

    def _chat(**k):
        captured.update(k)
        return {"content": "ok", "tokens_in": 1, "tokens_out": 1, "tool_trace": []}

    monkeypatch.setattr(cs.providers, "chat", _chat)
    ChatService().send(
        chat_user_id=1_000_000_001, message="what's wrong?", model="gpt-5",
        attachments=[SimpleNamespace(kind="image", data_url="data:image/png;base64,ZZZ")],
    )
    content = captured["messages"][-1]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "what's wrong?"}
    assert content[1]["image_url"]["url"] == "data:image/png;base64,ZZZ"
```

- [ ] **Step 2: Run — expect PASS** (the impl already exists from Task 1)

Run: `.venv/Scripts/python.exe -m pytest tests/api/test_chat_service.py::test_send_threads_image_attachment_as_multimodal -q`

- [ ] **Step 3: Commit**

```bash
git add tests/api/test_chat_service.py
git commit -m "test(api): lock multimodal image threading in ChatService.send (B2.2 slice C)"
```

---

### Task 5: Frontend — `html-to-image` page snapshot + camera button + send

**Files:**
- Modify: `web/package.json` (+ lockfile), `web/src/components/bubba/Bubba.tsx`

- [ ] **Step 1: Add the dependency**

Run (from `web/`): `pnpm add html-to-image`
Expected: `html-to-image` appears in `web/package.json` dependencies; lockfile updates.

- [ ] **Step 2: Add a snapshot-error state** — `Bubba.tsx`, next to the other `BubbaPanel` state:

```typescript
  const [snapError, setSnapError] = useState<string | null>(null);
```

- [ ] **Step 3: Add the `snapPage` handler** — inside `BubbaPanel`, near `handleSend`:

```typescript
  const snapPage = useCallback(async () => {
    setSnapError(null);
    try {
      const { toPng } = await import("html-to-image");
      const node = (document.querySelector("main") as HTMLElement | null) ?? document.body;
      const width = node.offsetWidth || 1280;
      const pixelRatio = Math.min(1, 1280 / width); // cap output width ~1280px
      const dataUrl = await toPng(node, {
        pixelRatio,
        cacheBust: true,
        // exclude the Bubba panel + the select-mode glow from the snapshot
        filter: (n) => !(n instanceof HTMLElement && n.dataset?.bubbaPanel === "1"),
      });
      setShots((s) => [...s, { id: crypto.randomUUID(), dataUrl }]);
    } catch {
      setSnapError("Couldn't snapshot the page.");
    }
  }, []);
```

- [ ] **Step 4: Add the camera button** to the controls row, right after the Tag button:

```tsx
              <button
                type="button"
                onClick={snapPage}
                title="Attach a snapshot of this page"
                className="flex items-center gap-1 rounded-full border border-line bg-canvas px-2.5 py-1 font-semibold text-ink-3 transition-colors hover:text-ink"
              >
                <Camera className="size-3.5" aria-hidden /> Snap
              </button>
```
(`Camera` is already imported from Task 2 Step 9.)

- [ ] **Step 5: Surface the snapshot error** — under the chips row (or near the composer), add:

```tsx
            {snapError && <p className="mb-1 text-[11px] text-ember">{snapError}</p>}
```

- [ ] **Step 6: Confirm send already includes `attachments`.** Task 2 Step 8 already maps `sentShots` → `attachments`. No change needed — verify the mapping is present.

- [ ] **Step 7: Typecheck + lint + build**

Run (from `web/`): `pnpm exec tsc --noEmit && pnpm run lint && pnpm build`
Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add web/package.json web/pnpm-lock.yaml web/src/components/bubba/Bubba.tsx
git commit -m "feat(web): page-snapshot attachment via html-to-image in Bubba (B2.2 slice C)"
```

---

### Task 6: Verify, preview, review, ship

- [ ] **Step 1: Full backend chat/AI suites + lint**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/api/ tests/test_ai_providers.py tests/test_ai_providers_streaming.py tests/test_ai_thinking_params.py -q
python -m ruff check api/ tests/api/ && python -m ruff format --check api/
```
Expected: PASS / no findings.

- [ ] **Step 2: Re-assert read-only tool surface**

Run: `.venv/Scripts/python.exe -m pytest tests/api/test_chat_tool_surface_readonly.py -q`
Expected: PASS (tagging adds context, never a write tool).

- [ ] **Step 3: Preview.** Chat is Clerk-gated locally, so do the verifiable checks: start the web preview, confirm tsc/build are green, and (panel reachable only with a Clerk session) verify what's observable without auth — the composer renders the Tag + Snap controls; arming Tag shows the page glow. If a Clerk dev session is available, also verify: highlight text → chip; click a player → player chip; Snap → page-snapshot chip; and that the `send-stream` request body carries `attached_text`/`attachments` (network panel). Capture a `preview_snapshot` of the composer with the new controls. (Per the B2.1 preview limitation: live model round-trip needs Clerk + a provider key.)

- [ ] **Step 4: Code review.** Dispatch `pr-review-toolkit:code-reviewer` + `pr-review-toolkit:silent-failure-hunter` on `git diff master...HEAD`. Verify: `_build_user_content` returns the bare string when no tag/image (live-app-safe); `providers.py` untouched (B2.1 equivalence guard still green); the capture-phase listener can't leak/Persist beyond select-mode (cleanup on disarm); `html-to-image` failure is graceful; no write tool added. Apply findings.

- [ ] **Step 5: Reconcile + merge + push.** `git checkout master && git pull --no-rebase origin master`, then `git merge --no-ff feat/bubba-b2-2-select-to-tag` (or `--ff-only` if master hasn't moved), confirm the api suite green, `git push origin master`. Pre-push runs the structural suite.

- [ ] **Step 6: Docs.** Update the `project_bubba_ai_assistant` memory + CLAUDE.md (Bubba line) to mark B2.2 shipped; flip the spec `Status:` to shipped.

---

## Self-Review

**1. Spec coverage**
- Highlight-text tagging → Task 1 (backend `attached_text`) + Task 2 (frontend select-mode/glow/selection/chips/send). ✅
- Click player rows/cards → Task 3 (`data-bubba-tag` on `PlayerLink` + capture-phase click). ✅
- Page screenshot → Task 4 (multimodal threading lock) + Task 5 (`html-to-image` + camera + send `attachments`). ✅
- `[Context…][Question…]` wrapping mirrors Streamlit → Task 1 `_build_user_content` + test. ✅
- Multimodal message; `providers.py` untouched; `chat()` equivalence intact → Task 1 (impl) + Task 10/Step-1 run of the B2.1 guard. ✅
- `attached_text` + `attachments` on both `/send` and `/send-stream` → Task 1 Step 7 + Step 9. ✅
- Page-edge glow self-rendered (no TopBar change) → Task 2 Step 5. ✅
- Read-only invariant re-asserted → Task 6 Step 2. ✅
- Out-of-scope (device-window capture, queue, saved prompts) → not in any task. ✅

**2. Placeholder scan** — every code step is complete. Judgement calls flagged inline: the `1280px` snapshot width cap (tunable) and the Clerk-gated preview (documented fallback, same as B2.1). No TBD/TODO.

**3. Type consistency** — `_build_user_content(message, attached_text, attachments)` signature matches its callers in `send`/`send_stream` and the router (`attached_text=body.attached_text, attachments=body.attachments`). The TS `tags`/`shots` shapes (`{id,kind,label,text}` / `{id,dataUrl}`) are consistent across state, chips, and the send mapping (`attached_text` from `tags[].text`, `attachments` from `shots[].dataUrl`). `ChatAttachment{kind,data_url}` (py) ↔ `{kind:"image", data_url}` (ts) ↔ `SimpleNamespace(kind,data_url)` (tests). `data-bubba-tag` = `name|mlbId` written in PlayerLink and split the same way in the click effect.

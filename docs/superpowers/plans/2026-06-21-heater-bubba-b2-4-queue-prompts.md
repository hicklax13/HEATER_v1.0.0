# Bubba B2.4 — Message Queue + Saved Prompts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete Phase B2 — let the user queue follow-up messages while Bubba is answering (auto-sent in turn) and save/pick/delete reusable named prompts (backed by a new per-user store in the api-owned `api_state.db`).

**Architecture:** Message queue is local `Bubba.tsx` state (enqueue-while-sending + an auto-drain effect). Saved prompts follow the proven contract→store→service→thin-router→DI→openapi pattern: a new `api/stores/prompt_store.py` (mirrors `user_store.py`, owns its table in the SEPARATE `api_state.db` — the live `draft_tool.db` + `src/` are untouched), wrapped by `ChatService` (which gains an injected store), exposed at `/api/chat/saved-prompts`, with a typed `bubba` client + a composer prompts menu.

**Tech Stack:** FastAPI 0.137.1 (pinned), Pydantic v2, SQLite (`api_state.db`), pytest; Next.js 16 + React 19 + TS. No new deps.

**Coordination:** `api/` + `web/` only; `src/` untouched. Three slices (A queue → B saved-prompts backend → C saved-prompts frontend); each ships + verifies independently.

---

## Grounding (verified in source — do NOT re-derive)

- **Store pattern = `api/stores/user_store.py`:** `Protocol` + `InMemory*` (DB-free fake) + `Sqlite*` (owns its table in `api_state.db` via `HEATER_API_DB_PATH`, default `data/api_state.db`; `_connect()` sets WAL + `busy_timeout=60000` + `synchronous=NORMAL`, `CREATE TABLE IF NOT EXISTS`, and `conn.close()` then `raise` if PRAGMA/CREATE fails; a `threading.Lock` serializes writes). Domain models are `pydantic.BaseModel`.
- `api/services/chat_service.py::ChatService` has NO `__init__` today (plain methods); `get_chat_service()` in `api/deps.py` returns `ChatService()`. Read methods degrade to `[]` + `_log.warning(..., exc_info=True)`; write methods (`store_key`/`delete_key`) return `(ok: bool, msg: str)`. `_log = logging.getLogger(__name__)`.
- `api/routers/chat.py`: thin, AST-guarded (`tests/api/test_no_logic_in_routers.py` — no `src.*` import, no arithmetic). `_uid(app_user)` raises 401 on `None`, else `chat_user_id_for(app_user.id)`. `MutationResponse{ok, message}` already exists in `api/contracts/chat.py`. Endpoint tests: `_FakeChatService` via `dependency_overrides` in `tests/api/test_api_chat.py`.
- OpenAPI: `.venv/Scripts/python.exe scripts/export_openapi.py`; guard `tests/api/test_openapi_contract.py`.
- `web/src/lib/api/bubba.ts` imports `apiDelete, apiGet, apiPost, apiPut` already; `apiDelete<T>(path, params?)` puts params in the query string (so a path-id DELETE passes the id IN the path, no params).
- `web/src/components/bubba/Bubba.tsx` (post-B2.3): `handleSend` is the send path; `sending` gates the Send button (`disabled={sending || !input.trim() || !model}`); the textarea `onKeyDown` Enter calls `handleSend()`; lucide imports include `X`, `Send`, `Paperclip`, etc.

**Venv python:** `.venv/Scripts/python.exe`. Frontend from `web/` with `pnpm`. Git from repo root `C:/Users/conno/Code/HEATER_v1.0.1`.

---

## File Structure

- **Modify** `web/src/components/bubba/Bubba.tsx` — Slice A: `queue` state + `onSubmit` (enqueue-while-sending) + auto-drain effect + queued list UI + `handleSend(textArg?)`. Slice C: a `PromptsMenu` + a Prompts button.
- **Create** `api/stores/prompt_store.py` — `SavedPrompt`/`PromptStore`/`InMemoryPromptStore`/`SqlitePromptStore`.
- **Modify** `api/services/chat_service.py` — `__init__(prompt_store=None)` + `saved_prompts`/`save_prompt`/`delete_prompt`.
- **Modify** `api/contracts/chat.py` — `SavedPrompt`/`SavedPromptsResponse`/`CreatePromptRequest`.
- **Modify** `api/routers/chat.py` — 3 `/saved-prompts` routes.
- **Modify** `api/openapi.json` — regenerate.
- **Modify** `web/src/lib/api/bubba.ts` — `savedPrompts`/`savePrompt`/`deletePrompt` + types.
- **Create** `tests/api/test_prompt_store.py`; **Modify** `tests/api/test_chat_service.py`, `tests/api/test_api_chat.py`.

---

## SLICE A — Message queue (frontend-only)

### Task 1: queue state + enqueue-while-sending + auto-drain + queued list

**Files:** Modify `web/src/components/bubba/Bubba.tsx`

- [ ] **Step 1: Add `Pencil` to the lucide import** (alphabetical, after `Paperclip`):
```typescript
  Paperclip,
  Pencil,
  Plus,
```

- [ ] **Step 2: Add the `queue` state** next to the other `BubbaPanel` state (after `snapError`/`attachError`):
```typescript
  const [queue, setQueue] = useState<{ id: string; text: string }[]>([]);
```

- [ ] **Step 3: Make `handleSend` accept an optional text + skip attachments for queued sends.** Replace the start of `handleSend`:
```typescript
  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || sending || !model) return;
    setInput("");
    // Append the user turn + an empty assistant turn we stream INTO.
    setMessages((prev) => [...prev, { role: "user", content: text }, { role: "assistant", content: "" }]);
    setSending(true);
    setToolStatus(null);
```
with:
```typescript
  const handleSend = useCallback(async (textArg?: string) => {
    const isQueued = textArg !== undefined;
    const text = (textArg ?? input).trim();
    if (!text || sending || !model) return;
    if (!isQueued) setInput("");
    // Append the user turn + an empty assistant turn we stream INTO.
    setMessages((prev) => [...prev, { role: "user", content: text }, { role: "assistant", content: "" }]);
    setSending(true);
    setToolStatus(null);
```
and change the attachment capture (a few lines down) — replace:
```typescript
    const sentTags = tags;
    const sentImages = images;
    const sentDocs = docs;
    setTags([]);
    setImages([]);
    setDocs([]);
```
with (queued messages are text-only — they never grab the composer's chips):
```typescript
    const sentTags = isQueued ? [] : tags;
    const sentImages = isQueued ? [] : images;
    const sentDocs = isQueued ? [] : docs;
    if (!isQueued) {
      setTags([]);
      setImages([]);
      setDocs([]);
    }
```

- [ ] **Step 4: Add an `onSubmit` wrapper** (enqueue while sending, else send) right after `handleSend`:
```typescript
  const onSubmit = useCallback(() => {
    const text = input.trim();
    if (!text || !model) return;
    if (sending) {
      setQueue((q) => [...q, { id: crypto.randomUUID(), text }]);
      setInput("");
    } else {
      handleSend();
    }
  }, [input, model, sending, handleSend]);
```

- [ ] **Step 5: Add the auto-drain effect** (after `onSubmit`): when a turn finishes cleanly and the queue isn't empty, send the next item.
```typescript
  // Drain the queue: when Bubba finishes a turn (sending→false) cleanly, auto-send
  // the next queued message. Pause on an errored turn so it doesn't blast the rest.
  useEffect(() => {
    if (sending || queue.length === 0) return;
    const last = messages[messages.length - 1];
    if (last?.isError) return;
    const [next, ...rest] = queue;
    setQueue(rest);
    handleSend(next.text);
  }, [sending, queue, messages, handleSend]);
```

- [ ] **Step 6: Let the Send button + Enter enqueue while sending.** In the composer, change the textarea `onKeyDown` to call `onSubmit`:
```tsx
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    onSubmit();
                  }
                }}
```
and change the Send button to call `onSubmit`, stay enabled while sending, and reflect the mode:
```tsx
              <button
                onClick={onSubmit}
                disabled={!input.trim() || !model}
                aria-label={sending ? "Queue message" : "Send"}
                title={sending ? "Queue message (sends after the current answer)" : "Send"}
                className="flex size-[38px] shrink-0 items-center justify-center rounded-lg bg-heat text-white transition-colors hover:bg-heat-bright disabled:cursor-not-allowed disabled:opacity-40"
              >
                {sending ? <Plus className="size-4" aria-hidden /> : <Send className="size-4" aria-hidden />}
              </button>
```

- [ ] **Step 7: Render the queued list** above the attachment chips (first child inside the composer `<div className="shrink-0 border-t border-line bg-surface p-2">`):
```tsx
            {queue.length > 0 && (
              <div className="mb-2 space-y-1">
                {queue.map((q) => (
                  <div
                    key={q.id}
                    className="flex items-center gap-1 rounded-lg border border-line bg-canvas px-2 py-1 text-[11px] text-ink"
                  >
                    <span className="flex-1 truncate">{q.text}</span>
                    <button
                      type="button"
                      onClick={() => {
                        setInput(q.text);
                        setQueue((x) => x.filter((y) => y.id !== q.id));
                      }}
                      aria-label="Edit queued message"
                      className="text-ink-3 hover:text-heat"
                    >
                      <Pencil className="size-3" aria-hidden />
                    </button>
                    <button
                      type="button"
                      onClick={() => setQueue((x) => x.filter((y) => y.id !== q.id))}
                      aria-label="Delete queued message"
                      className="text-ink-3 hover:text-ember"
                    >
                      <X className="size-3" aria-hidden />
                    </button>
                  </div>
                ))}
              </div>
            )}
```

- [ ] **Step 8: Typecheck + lint + build**

Run (from `web/`): `pnpm exec tsc --noEmit && pnpm run lint && pnpm build`
Expected: all green. (eslint's exhaustive-deps is satisfied: `onSubmit` deps `[input, model, sending, handleSend]`; the drain effect deps `[sending, queue, messages, handleSend]`.)

- [ ] **Step 9: Commit**

```bash
git add web/src/components/bubba/Bubba.tsx
git commit -m "feat(web): Bubba message queue — type-ahead while busy, auto-drain, edit/delete (B2.4 slice A)"
```

---

## SLICE B — Saved-prompts backend

### Task 2: `prompt_store.py` + store tests

**Files:** Create `api/stores/prompt_store.py`; Test `tests/api/test_prompt_store.py`

- [ ] **Step 1: Write the failing store tests**

```python
# tests/api/test_prompt_store.py
"""PromptStore: in-memory fake + the SQLite impl over a tmp api_state.db."""

import pytest

from api.stores.prompt_store import InMemoryPromptStore, SqlitePromptStore


@pytest.fixture(params=["mem", "sqlite"])
def store(request, tmp_path):
    if request.param == "mem":
        return InMemoryPromptStore()
    return SqlitePromptStore(db_path=str(tmp_path / "api_state.db"))


def test_create_then_list_newest_first(store):
    store.create(7, "first", "text one")
    store.create(7, "second", "text two")
    names = [p.name for p in store.list(7)]
    assert names == ["second", "first"]  # newest first


def test_list_is_owner_scoped(store):
    store.create(1, "mine", "a")
    store.create(2, "theirs", "b")
    assert [p.name for p in store.list(1)] == ["mine"]


def test_delete_returns_true_then_false(store):
    p = store.create(5, "n", "t")
    assert store.delete(5, p.id) is True
    assert store.delete(5, p.id) is False  # already gone


def test_delete_is_owner_scoped(store):
    p = store.create(1, "n", "t")
    assert store.delete(2, p.id) is False  # owner 2 can't delete owner 1's prompt
    assert [x.id for x in store.list(1)] == [p.id]  # still there
```

- [ ] **Step 2: Run — expect FAIL** (`No module named 'api.stores.prompt_store'`)

Run: `.venv/Scripts/python.exe -m pytest tests/api/test_prompt_store.py -q`

- [ ] **Step 3: Implement the store** (mirror `user_store.py`)

```python
# api/stores/prompt_store.py
"""api-owned saved-prompts persistence for Bubba (B2.4). Mirrors user_store.py:
a Protocol + an in-memory fake + a SQLite impl owning its OWN table in the
SEPARATE api_state.db (HEATER_API_DB_PATH) — never the live draft_tool.db.
Dormant until a user saves a prompt (no table created/written otherwise)."""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from datetime import UTC, datetime
from typing import Protocol

from pydantic import BaseModel

_DEFAULT_API_DB = os.path.join("data", "api_state.db")

logger = logging.getLogger(__name__)


class SavedPrompt(BaseModel):
    id: int
    name: str
    text: str
    created_at: str


class PromptStore(Protocol):
    def list(self, owner_id: int) -> list[SavedPrompt]: ...
    def create(self, owner_id: int, name: str, text: str) -> SavedPrompt: ...
    def delete(self, owner_id: int, prompt_id: int) -> bool: ...


class InMemoryPromptStore:
    """Test/fake impl. Thread-safe, autoincrement id, newest-first list."""

    def __init__(self) -> None:
        self._rows: list[tuple[int, SavedPrompt]] = []  # (owner_id, prompt)
        self._next_id = 1
        self._lock = threading.Lock()

    def list(self, owner_id: int) -> list[SavedPrompt]:
        with self._lock:
            return [p for (o, p) in reversed(self._rows) if o == owner_id]

    def create(self, owner_id: int, name: str, text: str) -> SavedPrompt:
        with self._lock:
            p = SavedPrompt(id=self._next_id, name=name, text=text, created_at=datetime.now(UTC).isoformat())
            self._rows.append((owner_id, p))
            self._next_id += 1
            return p

    def delete(self, owner_id: int, prompt_id: int) -> bool:
        with self._lock:
            before = len(self._rows)
            self._rows = [(o, p) for (o, p) in self._rows if not (o == owner_id and p.id == prompt_id)]
            return len(self._rows) < before


class SqlitePromptStore:
    """Default prod impl. Owns api_saved_prompts in a SEPARATE sqlite file (never
    the live draft_tool.db). Creates the table idempotently on first use."""

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or os.environ.get("HEATER_API_DB_PATH", _DEFAULT_API_DB)
        self._lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        parent = os.path.dirname(self._db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        conn = sqlite3.connect(self._db_path, timeout=60.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=60000")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS api_saved_prompts ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "owner_id INTEGER NOT NULL, "
                "name TEXT NOT NULL, "
                "text TEXT NOT NULL, "
                "created_at TEXT NOT NULL)"
            )
        except Exception:
            conn.close()  # don't leak the handle if PRAGMA/CREATE fails
            raise
        return conn

    def list(self, owner_id: int) -> list[SavedPrompt]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT id, name, text, created_at FROM api_saved_prompts WHERE owner_id = ? ORDER BY id DESC",
                    (owner_id,),
                ).fetchall()
                return [SavedPrompt(id=int(r[0]), name=r[1], text=r[2], created_at=r[3]) for r in rows]
            finally:
                conn.close()

    def create(self, owner_id: int, name: str, text: str) -> SavedPrompt:
        with self._lock:
            conn = self._connect()
            try:
                created_at = datetime.now(UTC).isoformat()
                cur = conn.execute(
                    "INSERT INTO api_saved_prompts (owner_id, name, text, created_at) VALUES (?, ?, ?, ?)",
                    (owner_id, name, text, created_at),
                )
                conn.commit()
                return SavedPrompt(id=int(cur.lastrowid), name=name, text=text, created_at=created_at)
            finally:
                conn.close()

    def delete(self, owner_id: int, prompt_id: int) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "DELETE FROM api_saved_prompts WHERE owner_id = ? AND id = ?",
                    (owner_id, prompt_id),
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()
```

- [ ] **Step 4: Run — expect PASS.** Run: `.venv/Scripts/python.exe -m pytest tests/api/test_prompt_store.py -q`

- [ ] **Step 5: Commit**

```bash
git add api/stores/prompt_store.py tests/api/test_prompt_store.py
git commit -m "feat(api): PromptStore (api_state.db) for Bubba saved prompts (B2.4 slice B)"
```

---

### Task 3: `ChatService` saved-prompts methods

**Files:** Modify `api/services/chat_service.py`; Test `tests/api/test_chat_service.py`

- [ ] **Step 1: Write the failing service tests** — append to `tests/api/test_chat_service.py`:

```python
def test_saved_prompts_roundtrip_with_injected_store():
    from api.stores.prompt_store import InMemoryPromptStore

    svc = ChatService(prompt_store=InMemoryPromptStore())
    ok, _ = svc.save_prompt(1_000_000_001, "Start SS?", "Who should I start at SS tonight?")
    assert ok
    prompts = svc.saved_prompts(1_000_000_001)
    assert len(prompts) == 1 and prompts[0]["name"] == "Start SS?"
    pid = prompts[0]["id"]
    assert svc.delete_prompt(1_000_000_001, pid)[0] is True
    assert svc.saved_prompts(1_000_000_001) == []


def test_save_prompt_rejects_empty():
    from api.stores.prompt_store import InMemoryPromptStore

    svc = ChatService(prompt_store=InMemoryPromptStore())
    ok, msg = svc.save_prompt(1, "", "")
    assert ok is False and "required" in msg.lower()


def test_saved_prompts_read_degrades_on_store_error():
    class _Boom:
        def list(self, owner_id):
            raise RuntimeError("db down")

    svc = ChatService(prompt_store=_Boom())
    assert svc.saved_prompts(1) == []  # never raises
```

- [ ] **Step 2: Run — expect FAIL** (`ChatService() got an unexpected keyword argument 'prompt_store'`)

Run: `.venv/Scripts/python.exe -m pytest tests/api/test_chat_service.py -q`

- [ ] **Step 3: Implement** — `api/services/chat_service.py`. Add the import near the top (with the other `from src.ai` / api imports):
```python
from api.stores.prompt_store import PromptStore, SqlitePromptStore
```
Add an `__init__` at the top of the `ChatService` class (before `def send`):
```python
class ChatService:
    def __init__(self, prompt_store: PromptStore | None = None) -> None:
        # Saved prompts live in the api-owned api_state.db (NOT the live draft_tool.db).
        self._prompts = prompt_store or SqlitePromptStore()

    def send(
```
Add the three methods after `delete_key` (the last method):
```python
    def saved_prompts(self, chat_user_id: int) -> list[dict]:
        try:
            return [p.model_dump() for p in self._prompts.list(chat_user_id)]
        except Exception:
            _log.warning("saved_prompts read failed", exc_info=True)
            return []

    def save_prompt(self, chat_user_id: int, name: str, text: str) -> tuple[bool, str]:
        name, text = (name or "").strip(), (text or "").strip()
        if not name or not text:
            return False, "Name and prompt text are required."
        try:
            self._prompts.create(chat_user_id, name, text)
            return True, "Prompt saved."
        except Exception as e:  # noqa: BLE001
            _log.warning("save_prompt failed", exc_info=True)
            return False, f"{type(e).__name__}: {str(e)[:200]}"

    def delete_prompt(self, chat_user_id: int, prompt_id: int) -> tuple[bool, str]:
        try:
            removed = self._prompts.delete(chat_user_id, prompt_id)
            return (True, "Prompt removed.") if removed else (False, "Prompt not found.")
        except Exception as e:  # noqa: BLE001
            _log.warning("delete_prompt failed", exc_info=True)
            return False, f"{type(e).__name__}: {str(e)[:200]}"
```

- [ ] **Step 4: Run — expect PASS.** Run: `.venv/Scripts/python.exe -m pytest tests/api/test_chat_service.py -q`

- [ ] **Step 5: Commit**

```bash
git add api/services/chat_service.py tests/api/test_chat_service.py
git commit -m "feat(api): ChatService saved-prompts methods over an injected store (B2.4 slice B)"
```

---

### Task 4: Contracts + 3 routes + OpenAPI

**Files:** Modify `api/contracts/chat.py`, `api/routers/chat.py`, `api/openapi.json`; Test `tests/api/test_api_chat.py`

- [ ] **Step 1: Write the failing endpoint tests** — extend `_FakeChatService` in `tests/api/test_api_chat.py` (add three methods) and add tests:

```python
    def saved_prompts(self, uid):  # noqa: ARG002
        return [{"id": 3, "name": "Start SS?", "text": "Who at SS?", "created_at": "2026"}]

    def save_prompt(self, uid, name, text):  # noqa: ARG002
        return True, "Prompt saved."

    def delete_prompt(self, uid, prompt_id):  # noqa: ARG002
        return True, "Prompt removed."
```

```python
def test_saved_prompts_crud():
    c = _client()
    assert c.get("/api/chat/saved-prompts").json()["prompts"][0]["name"] == "Start SS?"
    assert c.post("/api/chat/saved-prompts", json={"name": "n", "text": "t"}).json()["ok"] is True
    assert c.request("DELETE", "/api/chat/saved-prompts/3").json()["ok"] is True


def test_saved_prompts_unauthenticated_is_401():
    assert _client(user=None).get("/api/chat/saved-prompts").status_code == 401
```

- [ ] **Step 2: Run — expect FAIL** (routes 404)

Run: `.venv/Scripts/python.exe -m pytest tests/api/test_api_chat.py -q`

- [ ] **Step 3: Add the contracts** — `api/contracts/chat.py`, after `MutationResponse`:
```python
class SavedPrompt(BaseModel):
    id: int
    name: str
    text: str
    created_at: str | None = None


class SavedPromptsResponse(BaseModel):
    prompts: list[SavedPrompt] = []


class CreatePromptRequest(BaseModel):
    name: str
    text: str
```

- [ ] **Step 4: Add the routes** — `api/routers/chat.py`. Extend the contracts import:
```python
from api.contracts.chat import (
    ChatModelsResponse,
    ChatSendRequest,
    ChatSendResponse,
    ConversationListResponse,
    ConversationMessagesResponse,
    CreatePromptRequest,
    KeysResponse,
    MutationResponse,
    SavedPromptsResponse,
    StoreKeyRequest,
)
```
and add the three routes (after `delete_key`, the last route):
```python
@router.get("/saved-prompts", response_model=SavedPromptsResponse)
def saved_prompts(
    app_user: AppUser | None = Depends(require_app_user),
    svc: ChatService = Depends(get_chat_service),
) -> SavedPromptsResponse:
    return SavedPromptsResponse(prompts=svc.saved_prompts(_uid(app_user)))


@router.post("/saved-prompts", response_model=MutationResponse)
def create_saved_prompt(
    body: CreatePromptRequest,
    app_user: AppUser | None = Depends(require_app_user),
    svc: ChatService = Depends(get_chat_service),
) -> MutationResponse:
    ok, msg = svc.save_prompt(_uid(app_user), body.name, body.text)
    return MutationResponse(ok=ok, message=msg)


@router.delete("/saved-prompts/{prompt_id}", response_model=MutationResponse)
def delete_saved_prompt(
    prompt_id: int,
    app_user: AppUser | None = Depends(require_app_user),
    svc: ChatService = Depends(get_chat_service),
) -> MutationResponse:
    ok, msg = svc.delete_prompt(_uid(app_user), prompt_id)
    return MutationResponse(ok=ok, message=msg)
```

- [ ] **Step 5: Run — expect PASS** (endpoints + logic-free-router guard)

Run: `.venv/Scripts/python.exe -m pytest tests/api/test_api_chat.py tests/api/test_no_logic_in_routers.py -q`

- [ ] **Step 6: Regenerate OpenAPI + verify**

Run: `.venv/Scripts/python.exe scripts/export_openapi.py && .venv/Scripts/python.exe -m pytest tests/api/test_openapi_contract.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add api/contracts/chat.py api/routers/chat.py api/openapi.json tests/api/test_api_chat.py
git commit -m "feat(api): /api/chat/saved-prompts routes + contracts + openapi (B2.4 slice B)"
```

---

## SLICE C — Saved-prompts frontend

### Task 5: `bubba` client + composer prompts menu

**Files:** Modify `web/src/lib/api/bubba.ts`, `web/src/components/bubba/Bubba.tsx`

- [ ] **Step 1: Add the client methods + type** to `bubba.ts`. Add the interface (near `ChatModelOption`):
```typescript
export interface SavedPrompt {
  id: number;
  name: string;
  text: string;
  created_at?: string | null;
}
```
and add to the exported `bubba` object (after `deleteKey`):
```typescript
  savedPrompts: () => apiGet<{ prompts: SavedPrompt[] }>("/chat/saved-prompts").then((r) => r.prompts),

  savePrompt: (body: { name: string; text: string }) =>
    apiPost<{ ok: boolean; message: string }>("/chat/saved-prompts", body),

  deletePrompt: (id: number) =>
    apiDelete<{ ok: boolean; message: string }>(`/chat/saved-prompts/${id}`),
```

- [ ] **Step 2: Add a `PromptsMenu` component** at the bottom of `Bubba.tsx` (next to `TagChip`). It loads prompts on mount, lets the user pick (→ `onPick(text)`), save the current input, and delete:
```tsx
function PromptsMenu({ currentInput, onPick }: { currentInput: string; onPick: (text: string) => void }) {
  const [prompts, setPrompts] = useState<SavedPrompt[]>([]);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [epoch, setEpoch] = useState(0);

  useEffect(() => {
    let alive = true;
    Promise.resolve()
      .then(() => bubba.savedPrompts())
      .then((ps) => {
        if (alive) setPrompts(ps);
      })
      .catch(() => {
        if (alive) setPrompts([]);
      });
    return () => {
      alive = false;
    };
  }, [epoch]);

  const save = async () => {
    const n = name.trim();
    if (!n || !currentInput.trim() || busy) return;
    setBusy(true);
    try {
      await bubba.savePrompt({ name: n, text: currentInput.trim() });
      setName("");
      setEpoch((e) => e + 1);
    } catch {
      // graceful: leave the list as-is
    } finally {
      setBusy(false);
    }
  };

  const remove = async (id: number) => {
    await bubba.deletePrompt(id).catch(() => undefined);
    setEpoch((e) => e + 1);
  };

  return (
    <div className="mb-2 space-y-2 rounded-lg border border-line bg-surface-2 p-2">
      <div className="flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-wide text-ink-3">
        <BookMarked className="size-3.5" aria-hidden /> Saved prompts
      </div>
      {prompts.length === 0 && <p className="text-[11px] text-ink-3">No saved prompts yet.</p>}
      {prompts.map((p) => (
        <div key={p.id} className="flex items-center gap-1 text-[11px] text-ink">
          <button
            type="button"
            onClick={() => onPick(p.text)}
            className="flex-1 truncate text-left font-semibold hover:text-heat"
            title={p.text}
          >
            {p.name}
          </button>
          <button
            type="button"
            onClick={() => remove(p.id)}
            aria-label={`Delete ${p.name}`}
            className="text-ink-3 hover:text-ember"
          >
            <Trash2 className="size-3" aria-hidden />
          </button>
        </div>
      ))}
      <div className="flex items-center gap-1.5 pt-1">
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Name this prompt"
          className="min-w-0 flex-1 rounded-lg border border-line bg-canvas px-2 py-1 text-[11px] text-ink outline-none focus:border-heat"
        />
        <button
          type="button"
          onClick={save}
          disabled={busy || !name.trim() || !currentInput.trim()}
          className="rounded-lg bg-heat px-2 py-1 text-[11px] font-semibold text-white hover:bg-heat-bright disabled:opacity-40"
        >
          Save current
        </button>
      </div>
    </div>
  );
}
```
and add `BookMarked` to the lucide import (alphabetical, before `Camera`) + `SavedPrompt` to the bubba import:
```typescript
import { bubba, type ChatModelOption, type ConversationSummary, type KeyMeta, type SavedPrompt } from "@/lib/api/bubba";
```

- [ ] **Step 3: Add a Prompts toggle + render the menu.** Add state near the other `BubbaPanel` state:
```typescript
  const [showPrompts, setShowPrompts] = useState(false);
```
Add a Prompts button in the controls row (after the Screen button / the `</input>`, inside the controls `<div>`):
```tsx
              <button
                type="button"
                onClick={() => setShowPrompts((v) => !v)}
                aria-pressed={showPrompts}
                title="Saved prompts"
                className={cn(
                  "flex items-center gap-1 rounded-full border px-2.5 py-1 font-semibold transition-colors",
                  showPrompts ? "border-heat bg-heat/10 text-heat" : "border-line bg-canvas text-ink-3 hover:text-ink",
                )}
              >
                <BookMarked className="size-3.5" aria-hidden /> Prompts
              </button>
```
Render the menu above the queued list (first child inside the composer `<div className="shrink-0 border-t border-line bg-surface p-2">`, before the `{queue.length > 0 && …}` block):
```tsx
            {showPrompts && (
              <PromptsMenu
                currentInput={input}
                onPick={(t) => {
                  setInput(t);
                  setShowPrompts(false);
                }}
              />
            )}
```

- [ ] **Step 4: Typecheck + lint + build**

Run (from `web/`): `pnpm exec tsc --noEmit && pnpm run lint && pnpm build`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add web/src/lib/api/bubba.ts web/src/components/bubba/Bubba.tsx
git commit -m "feat(web): Bubba saved-prompts menu — pick/save/delete (B2.4 slice C)"
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

- [ ] **Step 2: Re-assert the read-only tool surface**

Run: `.venv/Scripts/python.exe -m pytest tests/api/test_chat_tool_surface_readonly.py -q`
Expected: PASS.

- [ ] **Step 3: Preview smoke.** `preview_start` the `heater-web` config; confirm the page loads with no console errors and the Bubba panel mounts cleanly (the composer is Clerk-gated locally → the queue/prompts UI live behind the ready phase; the smoke confirms no runtime mount regression). Stop the preview. (Live queue + saved-prompts round-trips need a Clerk session → owner verifies post-deploy.)

- [ ] **Step 4: Code review.** Dispatch `pr-review-toolkit:code-reviewer` + `pr-review-toolkit:silent-failure-hunter` on `git diff master...HEAD`. Verify: the auto-drain effect sends exactly one queued item per completion (no double-send / re-entrancy) and pauses on an errored turn; queued items are text-only (no attachment leakage); `PromptStore` deletes are `owner_id`-scoped (no cross-user delete) and the table lives in `api_state.db` (never `draft_tool.db`); `ChatService` saved-prompts reads degrade + writes return `(ok,msg)`, never 500; the router stays logic-free + 401-gates; no write tool added. Apply findings.

- [ ] **Step 5: Reconcile + merge + push.** `git checkout master && git pull --no-rebase origin master`, then `git merge --no-ff feat/bubba-b2-4-queue-prompts` (master may have moved — the platform track has been active), confirm the api suite green, `git push origin master`. Pre-push runs the structural suite.

- [ ] **Step 6: Docs.** Update `CLAUDE.md` (Bubba line) + the `project_bubba_ai_assistant` memory to mark B2.4 shipped → **Phase B2 COMPLETE**; flip the spec `Status:` to shipped.

---

## Self-Review

**1. Spec coverage**
- Message queue (enqueue-while-sending, edit/delete, auto-drain, text-only) → Task 1 (`queue` + `onSubmit` + drain effect + `handleSend(textArg?)` text-only + queued list). ✅
- Auto-drain pauses on an errored turn → Task 1 Step 5 (`if (last?.isError) return`). ✅
- Saved-prompts store in `api_state.db` (not `draft_tool.db`) → Task 2 `SqlitePromptStore` (mirrors `user_store.py`). ✅
- `owner_id`-scoped list + delete → Task 2 (queries filter by `owner_id`; tests lock it). ✅
- `ChatService` methods (degrade reads / `(ok,msg)` writes / injected store) → Task 3. ✅
- 3 routes `GET/POST /saved-prompts` + `DELETE /saved-prompts/{id}`, 401-gated, logic-free → Task 4. ✅
- Contracts `SavedPrompt`/`SavedPromptsResponse`/`CreatePromptRequest` (reuse `MutationResponse`) → Task 4 Step 3. ✅
- OpenAPI regen → Task 4 Step 6. ✅
- Frontend client + composer prompts menu (pick/save/delete, graceful empty) → Task 5. ✅
- Read-only invariant re-assert → Task 6 Step 2. ✅
- `src/` untouched → no `src/` task. ✅

**2. Placeholder scan** — every code step is complete. Concrete choices flagged inline (queue items text-only; owner_id = chat_user_id). No TBD/TODO.

**3. Type consistency** — store `SavedPrompt(BaseModel){id,name,text,created_at}` ↔ service returns `.model_dump()` dicts ↔ contract `SavedPrompt{id,name,text,created_at?}` ↔ TS `SavedPrompt{id,name,text,created_at?}`. `PromptStore.list/create/delete(owner_id,…)` signatures match `ChatService` calls (`self._prompts.list(chat_user_id)` etc.). `ChatService.saved_prompts/save_prompt/delete_prompt(chat_user_id,…)` match the router (`svc.saved_prompts(_uid(app_user))` …) and the `_FakeChatService` test methods. `bubba.savedPrompts/savePrompt/deletePrompt` match the routes (`GET`/`POST /chat/saved-prompts`, `DELETE /chat/saved-prompts/${id}`). `handleSend(textArg?)` + `onSubmit` + the drain effect use consistent names; `queue` items `{id,text}` consistent across state/enqueue/drain/list.

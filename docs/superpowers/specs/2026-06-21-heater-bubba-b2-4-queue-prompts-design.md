# HEATER "Bubba" B2.4 — Message Queue + Saved Prompts (Design Spec)

**Date:** 2026-06-21
**Lane:** CEO — full-stack (`api/` + `web/`; `src/` untouched).
**Parent spec:** `docs/superpowers/specs/2026-06-20-heater-bubba-ai-assistant-design.md` (Phase B2, fourth + final wave).
**Siblings:** B2.1 streaming, B2.2 tag-page-context, B2.3 attachments (all shipped).
**Status:** design — approved by owner 2026-06-21. Completes Phase B2.

## Scope

The last B2 wave — two productivity features the parent spec lists (lines 62-63):

1. **Message queue** (frontend-only) — while Bubba is answering, typing + send **queues** the next message instead of being blocked. Queued messages render above the composer; each can be **edited** (loads back into the input) or **deleted**. When the current answer finishes, the next queued message **auto-sends**, draining the queue. Queued items are **text-only** (attachments/tags stay on the active message — YAGNI).
2. **Saved prompts** (backend + frontend) — save a reusable prompt with a name; a composer menu lets the user **pick** one (drops its text into the input), **save** the current input, or **delete** one. Backed by a new per-user store the parent spec already specifies: `GET/POST /api/chat/saved-prompts` + `DELETE /api/chat/saved-prompts/{id}`.

**Store location decision:** saved prompts live in the **api-owned `api_state.db`** (env `HEATER_API_DB_PATH`), NOT the live `draft_tool.db` — so the live Streamlit app + its single-writer are completely untouched, matching where the M2 auth/billing data already lives. This is a different choice than B1's chat history/keys (which reuse `src/ai` in `draft_tool.db`); saved prompts are new api-owned data with no `src/ai` engine to reuse, so the M2 api-state pattern is the right home.

**Out of scope:** B3 monetization (managed tiers/metering) — separate phase.

## Grounding (verified in source)

- **The store pattern to mirror is `api/stores/user_store.py`:** a `Protocol` + an `InMemory*` fake (DB-free tests) + a `Sqlite*` prod impl that owns its OWN table in `api_state.db` (`HEATER_API_DB_PATH`, default `data/api_state.db`), creates the table idempotently in `_connect()` (WAL + `busy_timeout=60000`, closes the handle if PRAGMA/CREATE fails), and serializes writes with a `threading.Lock` (TOCTOU). `AppUser` is a `pydantic.BaseModel`.
- `api/services/chat_service.py::ChatService` is constructed argless (`ChatService()` in `api/deps.py::get_chat_service`). Its read methods (`conversations`/`messages`/`models`/`list_keys`) degrade to `[]` on error + `_log.warning`; its write methods (`store_key`/`delete_key`) return `(ok: bool, msg: str)`. Saved-prompts methods follow the SAME shapes.
- `api/routers/chat.py` is thin + AST-guarded (`tests/api/test_no_logic_in_routers.py`: no `src.*` import, no arithmetic assignment); `_uid(app_user)` raises 401 on `None` and returns `chat_user_id_for(app_user.id)`. Endpoint tests use a `_FakeChatService` via `dependency_overrides` (`tests/api/test_api_chat.py`).
- OpenAPI regen: `python scripts/export_openapi.py`; snapshot guard `tests/api/test_openapi_contract.py`.
- `web/src/components/bubba/Bubba.tsx`: `handleSend` is the send path; `sending` gates the composer (Send disabled while sending). `web/src/lib/api/bubba.ts` is the typed client (apiGet/apiPost/apiDelete already imported).

## Architecture

### Message queue (frontend-only, `Bubba.tsx`)

- **State:** `queue: { id: string; text: string }[]`.
- **Enqueue:** while `sending`, the send action (Enter / Send button) pushes `{id, text}` to `queue` and clears the input instead of calling `handleSend`. While idle, it sends immediately (today's behavior). The Send button reflects the mode (a queue/＋ affordance while sending).
- **Auto-drain:** a `useEffect` keyed on `sending` — when `sending` flips false AND `queue` is non-empty AND the just-finished assistant turn did NOT error, shift the front item and send it. On an errored turn the queue **pauses** (items stay; the user edits/resumes) — "doesn't blast the rest."
- **Queued list:** rendered above the composer; each row shows the text (truncated) + **edit** (loads the text back into the input + removes from queue) + **delete** (✕).
- No backend, no attachments on queued items.

### Saved prompts — backend

Follows the proven per-slice pattern (contract → store → service → thin router → DI → fake-override test → openapi):

- **Store** `api/stores/prompt_store.py` (mirrors `user_store.py`):
  - `SavedPrompt(BaseModel){ id: int, name: str, text: str, created_at: str }`.
  - `PromptStore(Protocol)`: `list(owner_id: int) -> list[SavedPrompt]`; `create(owner_id, name, text) -> SavedPrompt`; `delete(owner_id, prompt_id) -> bool`.
  - `InMemoryPromptStore` (fake) + `SqlitePromptStore` — table `api_saved_prompts(id INTEGER PK AUTOINCREMENT, owner_id INTEGER NOT NULL, name TEXT NOT NULL, text TEXT NOT NULL, created_at TEXT NOT NULL)` in `api_state.db`; `list` ordered newest-first; `delete` scoped to `owner_id` (a user can't delete another's prompt) returns whether a row was removed.
  - `owner_id` is the **namespaced `chat_user_id`** (`chat_user_id_for(app_user.id)`) so `ChatService`'s interface stays uniform (every method takes `chat_user_id`); it's an opaque per-user int key.
- **Service** — `ChatService` gains an injected store: `__init__(self, prompt_store: PromptStore | None = None)` → defaults to `SqlitePromptStore()`. Methods: `saved_prompts(chat_user_id) -> list[dict]` (degrade `[]` + log), `save_prompt(chat_user_id, name, text) -> tuple[bool, str]`, `delete_prompt(chat_user_id, prompt_id) -> tuple[bool, str]` (mirror the key methods). Docstring updated to note it now also owns the api-state saved-prompts store.
- **Contracts** (`api/contracts/chat.py`): `SavedPrompt{id, name, text, created_at}`, `SavedPromptsResponse{prompts: list[SavedPrompt]}`, `CreatePromptRequest{name, text}` (reuse `MutationResponse` for create/delete).
- **Router** (`api/routers/chat.py`): `GET /api/chat/saved-prompts` → `SavedPromptsResponse`; `POST /api/chat/saved-prompts` (`CreatePromptRequest`) → `MutationResponse`; `DELETE /api/chat/saved-prompts/{prompt_id}` → `MutationResponse`. All `require_app_user`-gated via `_uid`.
- **DI:** `get_chat_service()` stays `ChatService()` (defaults the real store). Service tests inject `InMemoryPromptStore`.
- **OpenAPI** regenerated.

### Saved prompts — frontend (`web/`)

- `bubba.ts`: `savedPrompts()`, `savePrompt({name, text})`, `deletePrompt(id)` + types.
- `Bubba.tsx`: a small **prompts menu** in the composer — pick a saved prompt (insert its `text` into the input), **save** the current input (a name prompt), and **delete** from the list. Loaded lazily when the menu opens; gracefully empty when none.

## Error handling

- **Queue auto-drain** pauses on an errored turn (won't auto-fire the rest); a send failure surfaces in the transcript exactly as today.
- **Saved-prompts reads** degrade to `[]` + `_log.warning` (never 500); **writes** return `(ok, msg)` — a failure shows a message, never a crash (the B1 key-store pattern).
- **`delete` of a missing/foreign prompt** → `(False, "not found")` (scoped by `owner_id`); never raises.
- **Empty name/text** on save → rejected client-side + server returns `(False, …)`.

## Testing

- **Store** (`tests/api/test_prompt_store.py`, DB-free via `InMemoryPromptStore` + a tmp-path `SqlitePromptStore`): create→list (newest-first), delete returns True/False, delete is `owner_id`-scoped (can't delete another owner's row), list isolates by owner.
- **Service** (`tests/api/test_chat_service.py`, inject `InMemoryPromptStore`): `saved_prompts`/`save_prompt`/`delete_prompt` happy paths + a degrade test (a raising store → `[]` / `(False, …)`, never propagates).
- **Endpoint** (`tests/api/test_api_chat.py`, `_FakeChatService` + `dependency_overrides`): GET list, POST create, DELETE by id, + the 401-on-unauthenticated gate; logic-free-router guard stays green; openapi snapshot regenerated.
- **Read-only invariant** unchanged — re-assert `tests/api/test_chat_tool_surface_readonly.py`.
- **Frontend:** tsc + lint + production build per slice; preview smoke (panel mounts; the queue + prompts UI render). Live round-trips need a Clerk session → owner verifies post-deploy (the B2.1-3 preview limitation).

## Risks

- **Auto-drain re-entrancy** — the drain effect must not double-send (guard on `sending` flips + a non-empty queue, shifting exactly one item per completion). Covered by keeping the send path single-entry and the effect idempotent per `sending` transition.
- **`api_state.db` first-write** — the store creates its table idempotently on first use (dormant until a Clerk user saves a prompt), exactly like `user_store.py`; no migration, no live-DB touch.
- **Queue + attachments interplay** — explicitly excluded (queued items are text-only) to avoid snapshotting chip state per item.

## Slices (build order)

- **Slice A — message queue** (frontend-only): `queue` state + enqueue-while-sending + auto-drain effect + queued-list edit/delete UI.
- **Slice B — saved-prompts backend:** `prompt_store.py` + `ChatService` methods + contracts + 3 routes + openapi + store/service/endpoint tests.
- **Slice C — saved-prompts frontend:** `bubba` client methods + the composer prompts menu.

## Index

- New: `api/stores/prompt_store.py` (`SavedPrompt`/`PromptStore`/`InMemoryPromptStore`/`SqlitePromptStore`); `ChatService` saved-prompts methods (+ injected store); `api/contracts/chat.py` prompt models; 3 `/api/chat/saved-prompts` routes; `bubba` prompt client + composer prompts menu; `Bubba.tsx` message queue.
- Reuses unchanged: the B1 contract→service→router→DI→openapi pattern, `api_state.db` + the `user_store.py` store shape, `require_app_user`/`_uid`, `src/ai/` (untouched), the B2.1-3 chat surface.
- Parent: the Bubba design spec (Phase B2). Memory: `project_bubba_ai_assistant`.

# Bubba — Phase B1 (Chat MVP) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Ship the `/api/chat/*` backend that wraps the existing `src/ai/` engine so the React app can run "Bubba" — per-user, BYO-keys, non-streamed chat with saved history — gated to Clerk users.

**Architecture:** A thin FastAPI wrap over the **unchanged** `src/ai/` (providers/keys/history/budget/router/tools). One service seam (`api/services/chat_service.py`) is the ONLY place that imports `src/ai`. The router resolves the caller's Clerk `AppUser` (via `require_app_user`), derives a **namespaced chat `user_id`** (`1_000_000_000 + AppUser.id` — avoids colliding with the live Streamlit chat's `users.user_id` rows in the shared `draft_tool.db` ai_* tables), and passes that int to the service. Follows the proven per-slice pattern: contract → service → thin logic-free router → DI provider in `api/deps.py` → fake-service `tests/api/test_api_*.py` override → mount in `api/main.py` → regen `api/openapi.json`.

**Tech Stack:** FastAPI 0.137.1 (pinned), Pydantic v2, `src/ai/` (litellm-based), pytest. No new deps.

**Coordination:** M4 track has FROZEN `api/main.py` / `api/openapi.json` / `api/deps.py` for this work (I own them + the openapi snapshot during B1). `src/ai/` is confirmed clear. The identity link is `AppUser.id` namespaced by offset — NO change to `api/stores/user_store.py` or `src/ai/`.

**Scope (B1 only):** `POST /api/chat/send`, `GET /api/chat/conversations`, `GET /api/chat/conversations/{id}/messages`, `GET /api/chat/models`, `GET/PUT/DELETE /api/chat/keys`. **Deferred:** saved-prompts, SSE streaming, select-to-tag/search/screenshots/queue (B2); managed tiers/metering (B3); recommendation cards (B4). Frontend popup/button = CMO-seam, separate.

---

## Pre-flight (read before coding — do NOT skip)
- `src/ai/chat_shell.py` — the existing Streamlit orchestrator. **`ChatService.send` is a direct port of its non-Streamlit send sequence** (system prompt → history → `providers.chat` → cost → `budget.record_usage` → `history.append_message`), with `st.*` replaced by return values. Read it for the EXACT sequence + `build_system_prompt` call shape.
- `src/ai/chat.py::build_system_prompt(...)` — confirm its signature; call it the same way `chat_shell.py` does.
- `src/ai/router.py` — confirm `model_catalog() -> list[(label, model)]`, `provider_of(model) -> str`, `price_per_token(model) -> tuple[float, float]` (used already in `src/ai/keys.py::_probe_model_for_provider`).
- Exemplar trio to mirror exactly: `api/services/leaders_overall_service.py`, `api/routers/*.py`, `tests/api/test_api_*.py`, and a `get_*_service` in `api/deps.py`.
- Confirmed engine signatures (already verified):
  - `providers.chat(model, messages, api_key, user_id, max_tool_rounds=6, web_search=False, deep_research=False) -> {content, tokens_in, tokens_out, tool_trace}`
  - `keys.get_key(user_id, provider) -> str|None`, `keys.store_key(user_id, provider, api_key, label=None)`, `keys.list_keys(user_id) -> list[dict]`, `keys.delete_key(user_id, provider, label=None)`, `keys.get_admin_shared_key(provider) -> str|None`
  - `history.create_conversation(user_id, title, model=None) -> int`, `history.list_conversations(user_id, limit=50)`, `history.append_message(conversation_id, role, content, model=None, tokens_in=0, tokens_out=0, cost_usd=0.0)`, `history.load_messages(conversation_id, user_id=None)`
  - `budget.is_over_cap(user_id, on_own_key=False) -> bool`, `budget.record_usage(user_id, tokens_in, tokens_out, cost_usd)`
  - `api/identity.py::require_app_user -> AppUser | None` (None on env-token/no-Clerk path → endpoint must 401)

---

## File Structure
- **Create** `api/contracts/chat.py` — request/response models.
- **Create** `api/services/chat_service.py` — `ChatService` (wraps `src/ai`) + `chat_user_id_for()` + `_CHAT_USER_ID_OFFSET`. ONLY file importing `src/ai`.
- **Create** `api/routers/chat.py` — thin `/api/chat/*` router; resolves AppUser→chat_user_id; logic-free.
- **Modify** `api/deps.py` — add `get_chat_service()`.
- **Modify** `api/main.py` — `include_router(chat.router)`.
- **Modify** `api/openapi.json` — regenerate.
- **Create** `tests/api/test_chat_service.py` — service tests (monkeypatch `src/ai` funcs; DB-free).
- **Create** `tests/api/test_api_chat.py` — endpoint tests via `dependency_overrides` (fake service) + the 401-on-None-AppUser gate.
- **Create** `tests/api/test_chat_tool_surface_readonly.py` — guard: no league-write tool is ever registered.

---

### Task 1: Chat contracts

**Files:** Create `api/contracts/chat.py`; Test `tests/api/test_chat_contracts.py`

- [ ] **Step 1: Write the failing test**
```python
# tests/api/test_chat_contracts.py
from api.contracts.chat import ChatSendRequest, ChatSendResponse


def test_send_request_defaults():
    req = ChatSendRequest(message="hi", model="gpt-5")
    assert req.conversation_id is None and req.web_search is False and req.deep_research is False


def test_send_response_carries_error_optional():
    r = ChatSendResponse(content="ok", conversation_id=1, tokens_in=1, tokens_out=2, cost_usd=0.0)
    assert r.error is None
```

- [ ] **Step 2: Run — expect ImportError**
Run: `.venv/Scripts/python.exe -m pytest tests/api/test_chat_contracts.py -q`
Expected: FAIL (`No module named 'api.contracts.chat'`).

- [ ] **Step 3: Implement the contracts**
```python
# api/contracts/chat.py
from __future__ import annotations

from pydantic import BaseModel


class ChatSendRequest(BaseModel):
    message: str
    model: str
    conversation_id: int | None = None
    web_search: bool = False
    deep_research: bool = False


class ChatSendResponse(BaseModel):
    content: str
    conversation_id: int | None
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    tool_trace: list[dict] = []
    error: str | None = None  # graceful provider/key/over-cap message; HTTP stays 200


class ConversationSummary(BaseModel):
    id: int
    title: str
    model: str | None = None
    updated_at: str | None = None


class ConversationListResponse(BaseModel):
    conversations: list[ConversationSummary] = []


class ChatMessage(BaseModel):
    role: str
    content: str
    model: str | None = None
    created_at: str | None = None


class ConversationMessagesResponse(BaseModel):
    conversation_id: int
    messages: list[ChatMessage] = []


class ChatModel(BaseModel):
    id: str            # the litellm model id
    label: str
    provider: str


class ChatModelsResponse(BaseModel):
    models: list[ChatModel] = []


class KeyMeta(BaseModel):
    provider: str
    label: str | None = None
    created_at: str | None = None


class KeysResponse(BaseModel):
    keys: list[KeyMeta] = []


class StoreKeyRequest(BaseModel):
    provider: str
    api_key: str
    label: str | None = None


class MutationResponse(BaseModel):
    ok: bool
    message: str = ""
```

- [ ] **Step 4: Run — expect PASS.** Run: `.venv/Scripts/python.exe -m pytest tests/api/test_chat_contracts.py -q`
- [ ] **Step 5: Commit** `git add api/contracts/chat.py tests/api/test_chat_contracts.py && git commit -m "feat(api): Bubba chat contracts (B1)"`

---

### Task 2: `ChatService.send` + the namespaced chat user_id

**Files:** Create `api/services/chat_service.py`; Test `tests/api/test_chat_service.py`

- [ ] **Step 1: Write the failing test** (monkeypatch every `src/ai` call — DB-free, no network)
```python
# tests/api/test_chat_service.py
import api.services.chat_service as cs
from api.services.chat_service import ChatService, chat_user_id_for


def test_chat_user_id_is_namespaced_above_streamlit_ids():
    # must not collide with draft_tool.db users.user_id (1..N) written by Streamlit chat
    assert chat_user_id_for(1) >= 1_000_000_000
    assert chat_user_id_for(2) == chat_user_id_for(1) + 1


def test_send_records_usage_and_appends_history(monkeypatch):
    calls = {}
    monkeypatch.setattr(cs, "_provider_of", lambda m: "openai")
    monkeypatch.setattr(cs.keys, "get_key", lambda uid, prov: "sk-user")
    monkeypatch.setattr(cs.keys, "get_admin_shared_key", lambda prov: None)
    monkeypatch.setattr(cs.budget, "is_over_cap", lambda uid, on_own_key=False: False)
    monkeypatch.setattr(cs.history, "create_conversation", lambda uid, title, model=None: 42)
    monkeypatch.setattr(cs.history, "load_messages", lambda cid, user_id=None: [])
    monkeypatch.setattr(cs, "_build_system_prompt", lambda uid: "SYS")
    monkeypatch.setattr(cs, "_price_per_token", lambda m: (0.0, 0.0))
    monkeypatch.setattr(cs.budget, "record_usage", lambda *a, **k: calls.setdefault("usage", True))
    monkeypatch.setattr(cs.history, "append_message", lambda *a, **k: calls.setdefault("append", (calls.get("append") or 0) + 1) or 1)
    monkeypatch.setattr(cs.providers, "chat", lambda **k: {"content": "hi back", "tokens_in": 3, "tokens_out": 5, "tool_trace": []})

    out = ChatService().send(chat_user_id=1_000_000_001, message="hi", model="gpt-5")
    assert out["content"] == "hi back" and out["conversation_id"] == 42 and out["error"] is None
    assert calls["usage"] and calls["append"] == 2  # user + assistant


def test_send_over_cap_returns_structured_error_not_raise(monkeypatch):
    monkeypatch.setattr(cs, "_provider_of", lambda m: "openai")
    monkeypatch.setattr(cs.keys, "get_key", lambda uid, prov: None)
    monkeypatch.setattr(cs.keys, "get_admin_shared_key", lambda prov: "sk-shared")
    monkeypatch.setattr(cs.budget, "is_over_cap", lambda uid, on_own_key=False: True)
    out = ChatService().send(chat_user_id=1_000_000_001, message="hi", model="gpt-5")
    assert out["error"] and "limit" in out["error"].lower() and out["content"] == ""


def test_send_provider_error_is_graceful(monkeypatch):
    monkeypatch.setattr(cs, "_provider_of", lambda m: "openai")
    monkeypatch.setattr(cs.keys, "get_key", lambda uid, prov: "sk-user")
    monkeypatch.setattr(cs.keys, "get_admin_shared_key", lambda prov: None)
    monkeypatch.setattr(cs.budget, "is_over_cap", lambda uid, on_own_key=False: False)
    monkeypatch.setattr(cs.history, "create_conversation", lambda *a, **k: 7)
    monkeypatch.setattr(cs.history, "load_messages", lambda *a, **k: [])
    monkeypatch.setattr(cs, "_build_system_prompt", lambda uid: "SYS")
    def boom(**k):
        raise RuntimeError("provider 500")
    monkeypatch.setattr(cs.providers, "chat", boom)
    out = ChatService().send(chat_user_id=1_000_000_001, message="hi", model="gpt-5")
    assert out["error"] and out["content"] == ""  # never raises
```

- [ ] **Step 2: Run — expect ImportError / FAIL.** Run: `.venv/Scripts/python.exe -m pytest tests/api/test_chat_service.py -q`

- [ ] **Step 3: Implement the service** (port `src/ai/chat_shell.py`'s send sequence; read it to mirror `_build_system_prompt`'s real call)
```python
# api/services/chat_service.py
"""The ONE api/ seam over src/ai. Wraps the unchanged chat engine; never raises
on a provider/key/cap error (returns a structured error in the response body so
the route stays HTTP 200, matching today's st.warning behavior)."""
from __future__ import annotations

from src.ai import budget, history, keys, providers
from src.ai.router import model_catalog as _model_catalog
from src.ai.router import price_per_token as _price_per_token
from src.ai.router import provider_of as _provider_of

# Namespacing offset: src/ai writes ai_* in the SHARED draft_tool.db keyed by an
# int user_id; the live Streamlit chat already uses users.user_id (1..N). Offset
# the React/Clerk AppUser.id well above that so the two never collide. The ai_*
# FK->users is declared but UNENFORCED (get_connection sets no PRAGMA
# foreign_keys=ON), so a non-existent users row does not raise. (B-phase follow-up:
# move chat persistence into api_state.db when B2 enforces FKs.)
_CHAT_USER_ID_OFFSET = 1_000_000_000


def chat_user_id_for(app_user_id: int) -> int:
    return _CHAT_USER_ID_OFFSET + int(app_user_id)


def _build_system_prompt(chat_user_id: int) -> str:
    # Mirror src/ai/chat_shell.py's build_system_prompt call (persona + league
    # context injection). Read chat_shell.py for the exact arguments it passes.
    from src.ai.chat import build_system_prompt

    return build_system_prompt()


class ChatService:
    def send(
        self,
        chat_user_id: int,
        message: str,
        model: str,
        conversation_id: int | None = None,
        web_search: bool = False,
        deep_research: bool = False,
    ) -> dict:
        try:
            provider = _provider_of(model)
            api_key = keys.get_key(chat_user_id, provider)
            on_own_key = api_key is not None and keys.get_admin_shared_key(provider) != api_key
            if budget.is_over_cap(chat_user_id, on_own_key=on_own_key):
                return self._empty("You've reached today's usage limit. Add your own API key for unlimited use.")
            if not api_key:
                return self._empty(f"No API key for {provider}. Add one in settings to chat.")

            prior = history.load_messages(conversation_id, user_id=chat_user_id) if conversation_id else []
            convo = [{"role": "system", "content": _build_system_prompt(chat_user_id)}]
            convo += [{"role": m["role"], "content": m["content"]} for m in prior]
            convo.append({"role": "user", "content": message})

            result = providers.chat(
                model=model, messages=convo, api_key=api_key, user_id=chat_user_id,
                web_search=web_search, deep_research=deep_research,
            )
            t_in, t_out = int(result.get("tokens_in", 0)), int(result.get("tokens_out", 0))
            p_in, p_out = _price_per_token(model)
            cost = t_in * p_in + t_out * p_out

            if conversation_id is None:
                conversation_id = history.create_conversation(chat_user_id, message[:120], model=model)
            history.append_message(conversation_id, "user", message, model=model)
            history.append_message(
                conversation_id, "assistant", result.get("content", ""), model=model,
                tokens_in=t_in, tokens_out=t_out, cost_usd=cost,
            )
            budget.record_usage(chat_user_id, t_in, t_out, cost)
            return {
                "content": result.get("content", ""), "conversation_id": conversation_id,
                "tokens_in": t_in, "tokens_out": t_out, "cost_usd": cost,
                "tool_trace": result.get("tool_trace", []), "error": None,
            }
        except Exception as e:  # noqa: BLE001 - chat must never 500; surface as body
            return self._empty(f"Chat error: {type(e).__name__}: {str(e)[:200]}", conversation_id)

    def _empty(self, error: str, conversation_id: int | None = None) -> dict:
        return {"content": "", "conversation_id": conversation_id, "tokens_in": 0,
                "tokens_out": 0, "cost_usd": 0.0, "tool_trace": [], "error": error}

    def conversations(self, chat_user_id: int) -> list[dict]:
        try:
            return history.list_conversations(chat_user_id)
        except Exception:
            return []

    def messages(self, chat_user_id: int, conversation_id: int) -> list[dict]:
        try:
            return history.load_messages(conversation_id, user_id=chat_user_id)
        except Exception:
            return []

    def models(self, chat_user_id: int) -> list[dict]:
        try:
            owned = {k["provider"] for k in keys.list_keys(chat_user_id)}
            out = []
            for label, model in _model_catalog():
                prov = _provider_of(model)
                if prov in owned or keys.get_admin_shared_key(prov):
                    out.append({"id": model, "label": label, "provider": prov})
            return out
        except Exception:
            return []

    def list_keys(self, chat_user_id: int) -> list[dict]:
        try:
            return keys.list_keys(chat_user_id)
        except Exception:
            return []

    def store_key(self, chat_user_id: int, provider: str, api_key: str, label: str | None) -> tuple[bool, str]:
        try:
            keys.store_key(chat_user_id, provider, api_key, label=label)
            return True, "Key saved."
        except Exception as e:  # noqa: BLE001
            return False, f"{type(e).__name__}: {str(e)[:200]}"

    def delete_key(self, chat_user_id: int, provider: str, label: str | None) -> tuple[bool, str]:
        try:
            keys.delete_key(chat_user_id, provider, label=label)
            return True, "Key removed."
        except Exception as e:  # noqa: BLE001
            return False, f"{type(e).__name__}: {str(e)[:200]}"
```

- [ ] **Step 4: Run — expect PASS.** Run: `.venv/Scripts/python.exe -m pytest tests/api/test_chat_service.py -q`
- [ ] **Step 5: Commit** `git add api/services/chat_service.py tests/api/test_chat_service.py && git commit -m "feat(api): Bubba ChatService over src/ai (B1)"`

---

### Task 3: Read-only tool-surface guard (load-bearing product boundary)

**Files:** Create `tests/api/test_chat_tool_surface_readonly.py`

- [ ] **Step 1: Write the test** (Bubba READS + ANALYZES; it must never register a league-write/action tool)
```python
# tests/api/test_chat_tool_surface_readonly.py
from src.ai.tools import tool_specs

_BANNED = ("set_lineup", "add_drop", "add_player", "drop_player", "make_trade",
           "execute", "submit", "place", "transaction", "write", "mutate")


def test_no_league_write_tool_is_registered():
    names = [s.get("function", {}).get("name", "").lower()
             for s in tool_specs(web_search_enabled=True, deep_research_enabled=True)]
    offenders = [n for n in names if any(b in n for b in _BANNED)]
    assert not offenders, f"Bubba must stay read-only; banned tools registered: {offenders}"
```

- [ ] **Step 2: Run — expect PASS** (no write tools exist today; this LOCKS it).
Run: `.venv/Scripts/python.exe -m pytest tests/api/test_chat_tool_surface_readonly.py -q`
If it FAILS, a write tool already exists — STOP and reconcile with the spec's read-only boundary before continuing.

- [ ] **Step 3: Commit** `git add tests/api/test_chat_tool_surface_readonly.py && git commit -m "test(api): guard Bubba tool surface stays read-only (B1)"`

---

### Task 4: Router (new file) + DI provider

**Files:** Create `api/routers/chat.py`; Modify `api/deps.py`; Test `tests/api/test_api_chat.py`

- [ ] **Step 1: Write the failing endpoint tests** (fake service via `dependency_overrides`; covers the 401-on-None gate)
```python
# tests/api/test_api_chat.py
import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from api.deps import get_chat_service
from api.identity import require_app_user
from api.routers import chat as chat_router


class _FakeUser:
    id = 7


class _FakeChatService:
    def send(self, **k):
        return {"content": "hi", "conversation_id": 1, "tokens_in": 1, "tokens_out": 2,
                "cost_usd": 0.0, "tool_trace": [], "error": None}
    def conversations(self, uid):  # noqa: ARG002
        return [{"id": 1, "title": "t", "model": "m", "updated_at": "2026"}]
    def messages(self, uid, cid):  # noqa: ARG002
        return [{"role": "user", "content": "hi", "model": "m", "created_at": "2026"}]
    def models(self, uid):  # noqa: ARG002
        return [{"id": "gpt-5", "label": "GPT-5", "provider": "openai"}]
    def list_keys(self, uid):  # noqa: ARG002
        return [{"provider": "openai", "label": None, "created_at": "2026"}]
    def store_key(self, *a):
        return True, "Key saved."
    def delete_key(self, *a):
        return True, "Key removed."


def _client(user=_FakeUser()):
    app = FastAPI()
    app.include_router(chat_router.router)
    app.dependency_overrides[get_chat_service] = lambda: _FakeChatService()
    app.dependency_overrides[require_app_user] = lambda: user
    return TestClient(app)


def test_send_ok():
    r = _client().post("/api/chat/send", json={"message": "hi", "model": "gpt-5"})
    assert r.status_code == 200 and r.json()["content"] == "hi"


def test_unauthenticated_is_401():
    r = _client(user=None).post("/api/chat/send", json={"message": "hi", "model": "gpt-5"})
    assert r.status_code == 401


def test_conversations_models_keys():
    c = _client()
    assert c.get("/api/chat/conversations").json()["conversations"][0]["id"] == 1
    assert c.get("/api/chat/conversations/1/messages").json()["conversation_id"] == 1
    assert c.get("/api/chat/models").json()["models"][0]["id"] == "gpt-5"
    assert c.get("/api/chat/keys").json()["keys"][0]["provider"] == "openai"
    assert c.put("/api/chat/keys", json={"provider": "openai", "api_key": "sk"}).json()["ok"] is True
    assert c.delete("/api/chat/keys", params={"provider": "openai"}).json()["ok"] is True
```

- [ ] **Step 2: Run — expect FAIL** (`get_chat_service` / router missing).
Run: `.venv/Scripts/python.exe -m pytest tests/api/test_api_chat.py -q`

- [ ] **Step 3a: Add the DI provider** — append to `api/deps.py`:
```python
from api.services.chat_service import ChatService  # add with the other service imports


def get_chat_service() -> ChatService:
    return ChatService()
```

- [ ] **Step 3b: Implement the router** (logic-free — passes the AST `test_no_logic_in_routers` guard; the only logic is the 401 gate + the offset call)
```python
# api/routers/chat.py
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.contracts.chat import (
    ChatModelsResponse, ChatSendRequest, ChatSendResponse, ConversationListResponse,
    ConversationMessagesResponse, KeysResponse, MutationResponse, StoreKeyRequest,
)
from api.deps import get_chat_service
from api.identity import require_app_user
from api.services.chat_service import ChatService, chat_user_id_for
from api.stores.user_store import AppUser

router = APIRouter(prefix="/api/chat", tags=["chat"])


def _uid(app_user: AppUser | None) -> int:
    if app_user is None:
        raise HTTPException(status_code=401, detail="Sign in to use Bubba.")
    return chat_user_id_for(app_user.id)


@router.post("/send", response_model=ChatSendResponse)
def send(body: ChatSendRequest, app_user: AppUser | None = Depends(require_app_user),
         svc: ChatService = Depends(get_chat_service)) -> ChatSendResponse:
    return ChatSendResponse(**svc.send(
        chat_user_id=_uid(app_user), message=body.message, model=body.model,
        conversation_id=body.conversation_id, web_search=body.web_search,
        deep_research=body.deep_research,
    ))


@router.get("/conversations", response_model=ConversationListResponse)
def conversations(app_user: AppUser | None = Depends(require_app_user),
                  svc: ChatService = Depends(get_chat_service)) -> ConversationListResponse:
    return ConversationListResponse(conversations=svc.conversations(_uid(app_user)))


@router.get("/conversations/{conversation_id}/messages", response_model=ConversationMessagesResponse)
def messages(conversation_id: int, app_user: AppUser | None = Depends(require_app_user),
             svc: ChatService = Depends(get_chat_service)) -> ConversationMessagesResponse:
    return ConversationMessagesResponse(
        conversation_id=conversation_id, messages=svc.messages(_uid(app_user), conversation_id))


@router.get("/models", response_model=ChatModelsResponse)
def models(app_user: AppUser | None = Depends(require_app_user),
           svc: ChatService = Depends(get_chat_service)) -> ChatModelsResponse:
    return ChatModelsResponse(models=svc.models(_uid(app_user)))


@router.get("/keys", response_model=KeysResponse)
def list_keys(app_user: AppUser | None = Depends(require_app_user),
              svc: ChatService = Depends(get_chat_service)) -> KeysResponse:
    return KeysResponse(keys=svc.list_keys(_uid(app_user)))


@router.put("/keys", response_model=MutationResponse)
def put_key(body: StoreKeyRequest, app_user: AppUser | None = Depends(require_app_user),
            svc: ChatService = Depends(get_chat_service)) -> MutationResponse:
    ok, msg = svc.store_key(_uid(app_user), body.provider, body.api_key, body.label)
    return MutationResponse(ok=ok, message=msg)


@router.delete("/keys", response_model=MutationResponse)
def delete_key(provider: str, label: str | None = None,
               app_user: AppUser | None = Depends(require_app_user),
               svc: ChatService = Depends(get_chat_service)) -> MutationResponse:
    ok, msg = svc.delete_key(_uid(app_user), provider, label)
    return MutationResponse(ok=ok, message=msg)
```

- [ ] **Step 4: Run — expect PASS.** Run: `.venv/Scripts/python.exe -m pytest tests/api/test_api_chat.py tests/api/test_no_logic_in_routers.py -q`
- [ ] **Step 5: Commit** `git add api/routers/chat.py api/deps.py tests/api/test_api_chat.py && git commit -m "feat(api): Bubba chat router + DI provider (B1)"`

---

### Task 5: Mount + regenerate OpenAPI

**Files:** Modify `api/main.py`; Modify `api/openapi.json`

- [ ] **Step 1:** In `api/main.py`, import the chat router and `app.include_router(chat.router)` alongside the other `include_router` calls (mirror the existing pattern).
- [ ] **Step 2: Regenerate the snapshot** — run the repo's openapi-gen step (the command/script `api/openapi.json` is produced by; check `tests/api/test_openapi_contract.py` for how it loads/compares, and regenerate the same way).
- [ ] **Step 3: Run the snapshot + full api guard**
Run: `.venv/Scripts/python.exe -m pytest tests/api/test_openapi_contract.py tests/api/ -q`
Expected: PASS (chat routes now in the snapshot).
- [ ] **Step 4: Commit** `git add api/main.py api/openapi.json && git commit -m "feat(api): mount Bubba chat router + regen openapi (B1)"`

---

### Task 6: Verify, review, ship

- [ ] **Step 1: Full api suite + ruff**
Run: `.venv/Scripts/python.exe -m pytest tests/api/ -q` then `python -m ruff check api/ tests/api/ && python -m ruff format --check api/ tests/api/`
- [ ] **Step 2: Code review** — dispatch `pr-review-toolkit:code-reviewer` + `pr-review-toolkit:silent-failure-hunter` on the diff. Verify: chat never 500s; per-user scoping (history/keys/budget keyed by the namespaced chat_user_id, never a raw param); the 401-on-None gate; read-only tool guard intact; no logic in the router. Apply findings.
- [ ] **Step 3: Reconcile + push** — `git pull --no-rebase` (M4 track may have advanced), confirm api suite still green, `git push origin master`. Pre-push runs the structural suite.

---

## Self-Review
- **Spec coverage:** B1 endpoints (send/conversations/messages/models/keys) ✅; identity seam (AppUser→namespaced chat_user_id) ✅; BYO-keys ✅; auth-gated (401 on None) ✅; read-only boundary guard ✅; graceful no-500 ✅. Deferred items (saved-prompts, streaming, rich features, tiers, cards) explicitly out of B1 ✅.
- **Placeholder scan:** the only "read the exemplar" pointers are `chat_shell.py` (exact `build_system_prompt` call) + the openapi-regen command — both are existing-codebase facts the executor confirms at the file, not hidden logic.
- **Type consistency:** `chat_user_id_for`, `ChatService` method names, and the dict keys returned by the service match what the router/contracts consume (`content/conversation_id/tokens_in/tokens_out/cost_usd/tool_trace/error`). `require_app_user -> AppUser | None` handled in `_uid`.
- **Collision/FK note** carried in `chat_service.py` docstring + the B-phase follow-up (move chat persistence to `api_state.db` when FKs are enforced).

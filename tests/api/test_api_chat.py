"""Endpoint tests for /api/chat/* via dependency_overrides (fake service + fake
identity). DB-free; covers the 401-on-unauthenticated gate."""

from fastapi import FastAPI
from starlette.testclient import TestClient

from api.deps import get_chat_service
from api.identity import require_app_user
from api.routers import chat as chat_router


class _FakeUser:
    id = 7


class _FakeChatService:
    def send(self, **k):
        return {
            "content": "hi",
            "conversation_id": 1,
            "tokens_in": 1,
            "tokens_out": 2,
            "cost_usd": 0.0,
            "tool_trace": [],
            "error": None,
        }

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

    def send_stream(self, **k):  # noqa: ARG002
        yield 'data: {"type": "text_delta", "text": "hi"}\n\n'
        yield 'data: {"type": "done", "content": "hi", "conversation_id": 1, "cost_usd": 0.0, "tokens_in": 1, "tokens_out": 1, "tool_trace": []}\n\n'


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


def test_send_stream_returns_event_stream():
    r = _client().post("/api/chat/send-stream", json={"message": "hi", "model": "gpt-5"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert '"type": "text_delta"' in r.text and '"type": "done"' in r.text


def test_send_stream_unauthenticated_is_401():
    r = _client(user=None).post("/api/chat/send-stream", json={"message": "hi", "model": "gpt-5"})
    assert r.status_code == 401


def test_conversations_models_keys():
    c = _client()
    assert c.get("/api/chat/conversations").json()["conversations"][0]["id"] == 1
    assert c.get("/api/chat/conversations/1/messages").json()["conversation_id"] == 1
    assert c.get("/api/chat/models").json()["models"][0]["id"] == "gpt-5"
    assert c.get("/api/chat/keys").json()["keys"][0]["provider"] == "openai"
    assert c.put("/api/chat/keys", json={"provider": "openai", "api_key": "sk"}).json()["ok"] is True
    assert c.request("DELETE", "/api/chat/keys", params={"provider": "openai"}).json()["ok"] is True

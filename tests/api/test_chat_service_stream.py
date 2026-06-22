"""ChatService.send_stream — SSE frames + server-side metering on `done`.
Every src/ai call is monkeypatched (DB-free, no network)."""

import json

import api.services.chat_service as cs
from api.services.chat_service import ChatService


def _frames(gen):
    """Parse the SSE text frames a send_stream generator yields into dicts."""
    out = []
    for raw in gen:
        assert raw.endswith("\n\n"), f"frame not SSE-terminated: {raw!r}"
        assert raw.startswith("data: "), f"frame missing data: prefix: {raw!r}"
        out.append(json.loads(raw[len("data: ") : -2]))
    return out


def _happy_path(monkeypatch, on_record=None):
    monkeypatch.setattr(cs, "_provider_of", lambda m: "openai")
    monkeypatch.setattr(cs.keys, "get_key", lambda uid, prov: "sk-user")
    monkeypatch.setattr(cs.keys, "list_keys", lambda uid: [{"provider": "openai"}])
    monkeypatch.setattr(cs.budget, "is_over_cap", lambda uid, on_own_key=False, cap_usd=None: False)
    monkeypatch.setattr(cs, "_build_system_prompt", lambda page, team: "SYS")
    monkeypatch.setattr(cs.history, "load_messages", lambda *a, **k: [])
    monkeypatch.setattr(cs.history, "create_conversation", lambda *a, **k: 77)
    monkeypatch.setattr(cs.history, "append_message", lambda *a, **k: 1)
    monkeypatch.setattr(cs, "_price_per_token", lambda m: (0.0, 0.0))
    monkeypatch.setattr(cs.budget, "record_usage", on_record or (lambda *a, **k: None))

    def _events(**k):
        yield {"type": "text_delta", "text": "Hi "}
        yield {"type": "text_delta", "text": "there"}
        yield {"type": "done", "content": "Hi there", "tokens_in": 3, "tokens_out": 5, "tool_trace": []}

    monkeypatch.setattr(cs.providers, "_chat_events", _events)


def test_send_stream_emits_deltas_then_enriched_done(monkeypatch):
    _happy_path(monkeypatch)
    frames = _frames(ChatService().send_stream(chat_user_id=1_000_000_001, message="hi", model="gpt-5"))
    kinds = [f["type"] for f in frames]
    assert kinds == ["text_delta", "text_delta", "done"]
    done = frames[-1]
    assert done["content"] == "Hi there" and done["conversation_id"] == 77 and done["cost_usd"] == 0.0


def test_send_stream_meters_usage_on_done(monkeypatch):
    metered = {}
    _happy_path(monkeypatch, on_record=lambda *a, **k: metered.__setitem__("ok", True))
    list(ChatService().send_stream(chat_user_id=1_000_000_001, message="hi", model="gpt-5"))
    assert metered.get("ok")  # billed exactly when the answer completed


def test_send_stream_no_key_yields_single_error_frame(monkeypatch):
    monkeypatch.setattr(cs, "_provider_of", lambda m: "openai")
    monkeypatch.setattr(cs.keys, "get_key", lambda uid, prov: None)
    frames = _frames(ChatService().send_stream(chat_user_id=1_000_000_001, message="hi", model="gpt-5"))
    assert len(frames) == 1 and frames[0]["type"] == "error" and "no api key" in frames[0]["message"].lower()


def test_send_stream_over_cap_yields_error_frame(monkeypatch):
    monkeypatch.setattr(cs, "_provider_of", lambda m: "openai")
    monkeypatch.setattr(cs.keys, "get_key", lambda uid, prov: "sk-shared")
    monkeypatch.setattr(cs.keys, "list_keys", lambda uid: [])  # no own key -> shared -> capped
    monkeypatch.setattr(cs.budget, "is_over_cap", lambda uid, on_own_key=False, cap_usd=None: True)
    frames = _frames(ChatService().send_stream(chat_user_id=1_000_000_001, message="hi", model="gpt-5"))
    assert frames[0]["type"] == "error" and "limit" in frames[0]["message"].lower()


def test_send_stream_engine_exception_becomes_error_frame(monkeypatch):
    _happy_path(monkeypatch)

    def _boom(**k):
        raise RuntimeError("provider 500")
        yield  # pragma: no cover - generator marker

    monkeypatch.setattr(cs.providers, "_chat_events", _boom)
    frames = _frames(ChatService().send_stream(chat_user_id=1_000_000_001, message="hi", model="gpt-5"))
    assert any(f["type"] == "error" for f in frames)  # never raises


def test_send_stream_early_stop_before_done_does_not_meter(monkeypatch):
    """If the consumer disconnects mid-stream (no `done` pulled), nothing is
    billed — no complete answer was produced."""
    metered = {}
    _happy_path(monkeypatch, on_record=lambda *a, **k: metered.__setitem__("ok", True))
    gen = ChatService().send_stream(chat_user_id=1_000_000_001, message="hi", model="gpt-5")
    next(gen)  # pull only the first text_delta
    gen.close()
    assert not metered.get("ok")

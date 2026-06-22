"""ChatService tests — every src/ai call is monkeypatched (DB-free, no network).
Mirrors the canonical send orchestration in src/ai/chat.py::_handle_send."""

import api.services.chat_service as cs
from api.services.chat_service import ChatService, chat_user_id_for


def test_chat_user_id_is_namespaced_above_streamlit_ids():
    # must not collide with draft_tool.db users.user_id (1..N) the Streamlit chat writes
    assert chat_user_id_for(1) >= 1_000_000_000
    assert chat_user_id_for(2) == chat_user_id_for(1) + 1


def test_send_records_usage_and_appends_history(monkeypatch):
    calls = {"append": 0}
    monkeypatch.setattr(cs, "_provider_of", lambda m: "openai")
    monkeypatch.setattr(cs.keys, "get_key", lambda uid, prov: "sk-user")
    monkeypatch.setattr(cs.keys, "list_keys", lambda uid: [{"provider": "openai"}])
    monkeypatch.setattr(cs.budget, "is_over_cap", lambda uid, on_own_key=False, cap_usd=None: False)
    monkeypatch.setattr(cs, "_build_system_prompt", lambda page, team: "SYS")
    monkeypatch.setattr(cs.history, "load_messages", lambda cid, user_id=None: [])
    monkeypatch.setattr(cs.history, "create_conversation", lambda uid, title, model=None: 42)
    monkeypatch.setattr(cs, "_price_per_token", lambda m: (0.0, 0.0))
    monkeypatch.setattr(cs.budget, "record_usage", lambda *a, **k: calls.__setitem__("usage", True))

    def _append(*a, **k):
        calls["append"] += 1
        return 1

    monkeypatch.setattr(cs.history, "append_message", _append)
    monkeypatch.setattr(
        cs.providers,
        "chat",
        lambda **k: {"content": "hi back", "tokens_in": 3, "tokens_out": 5, "tool_trace": []},
    )

    out = ChatService().send(chat_user_id=1_000_000_001, message="hi", model="gpt-5")
    assert out["content"] == "hi back" and out["conversation_id"] == 42 and out["error"] is None
    assert calls.get("usage") and calls["append"] == 2  # user + assistant


def test_send_no_key_returns_error(monkeypatch):
    monkeypatch.setattr(cs, "_provider_of", lambda m: "openai")
    monkeypatch.setattr(cs.keys, "get_key", lambda uid, prov: None)
    out = ChatService().send(chat_user_id=1_000_000_001, message="hi", model="gpt-5")
    assert out["error"] and "no api key" in out["error"].lower() and out["content"] == ""


def test_send_over_cap_returns_structured_error(monkeypatch):
    monkeypatch.setattr(cs, "_provider_of", lambda m: "openai")
    monkeypatch.setattr(cs.keys, "get_key", lambda uid, prov: "sk-shared")
    monkeypatch.setattr(cs.keys, "list_keys", lambda uid: [])  # no own key -> shared -> capped
    monkeypatch.setattr(cs.budget, "is_over_cap", lambda uid, on_own_key=False, cap_usd=None: True)
    out = ChatService().send(chat_user_id=1_000_000_001, message="hi", model="gpt-5")
    assert out["error"] and "limit" in out["error"].lower() and out["content"] == ""


def test_send_provider_error_is_graceful(monkeypatch):
    monkeypatch.setattr(cs, "_provider_of", lambda m: "openai")
    monkeypatch.setattr(cs.keys, "get_key", lambda uid, prov: "sk-user")
    monkeypatch.setattr(cs.keys, "list_keys", lambda uid: [{"provider": "openai"}])
    monkeypatch.setattr(cs.budget, "is_over_cap", lambda uid, on_own_key=False, cap_usd=None: False)
    monkeypatch.setattr(cs, "_build_system_prompt", lambda page, team: "SYS")
    monkeypatch.setattr(cs.history, "load_messages", lambda *a, **k: [])

    def boom(**k):
        raise RuntimeError("provider 500")

    monkeypatch.setattr(cs.providers, "chat", boom)
    out = ChatService().send(chat_user_id=1_000_000_001, message="hi", model="gpt-5")
    assert out["error"] and out["content"] == ""  # never raises


def test_send_persist_failure_returns_answer_and_still_meters(monkeypatch):
    """A successful, billed call whose history-write fails must NOT be discarded,
    and usage must still be recorded (no cap bypass) — the silent-failure fix."""
    metered = {}
    monkeypatch.setattr(cs, "_provider_of", lambda m: "openai")
    monkeypatch.setattr(cs.keys, "get_key", lambda uid, prov: "sk-user")
    monkeypatch.setattr(cs.keys, "list_keys", lambda uid: [{"provider": "openai"}])
    monkeypatch.setattr(cs.budget, "is_over_cap", lambda uid, on_own_key=False, cap_usd=None: False)
    monkeypatch.setattr(cs, "_build_system_prompt", lambda page, team: "SYS")
    monkeypatch.setattr(cs.history, "load_messages", lambda *a, **k: [])
    monkeypatch.setattr(cs, "_price_per_token", lambda m: (0.0, 0.0))
    monkeypatch.setattr(cs.budget, "record_usage", lambda *a, **k: metered.__setitem__("ok", True))
    monkeypatch.setattr(cs.history, "create_conversation", lambda *a, **k: 9)

    def _boom_append(*a, **k):
        raise RuntimeError("db locked")

    monkeypatch.setattr(cs.history, "append_message", _boom_append)
    monkeypatch.setattr(
        cs.providers,
        "chat",
        lambda **k: {"content": "answer", "tokens_in": 3, "tokens_out": 5, "tool_trace": []},
    )

    out = ChatService().send(chat_user_id=1_000_000_001, message="hi", model="gpt-5")
    assert out["content"] == "answer" and out["error"] is None  # answer NOT discarded
    assert metered.get("ok")  # usage recorded despite the persist failure (no cap bypass)


def test_send_threads_reasoning_effort_into_providers(monkeypatch):
    captured = {}
    monkeypatch.setattr(cs, "_provider_of", lambda m: "openai")
    monkeypatch.setattr(cs.keys, "get_key", lambda uid, prov: "sk-user")
    monkeypatch.setattr(cs.keys, "list_keys", lambda uid: [{"provider": "openai"}])
    monkeypatch.setattr(cs.budget, "is_over_cap", lambda uid, on_own_key=False, cap_usd=None: False)
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
    ChatService().send(chat_user_id=1_000_000_001, message="hi", model="gpt-5", reasoning_effort="high")
    assert captured.get("reasoning_effort") == "high"


def test_send_threads_attached_text_into_user_turn(monkeypatch):
    captured = {}
    monkeypatch.setattr(cs, "_provider_of", lambda m: "openai")
    monkeypatch.setattr(cs.keys, "get_key", lambda uid, prov: "sk-user")
    monkeypatch.setattr(cs.keys, "list_keys", lambda uid: [{"provider": "openai"}])
    monkeypatch.setattr(cs.budget, "is_over_cap", lambda uid, on_own_key=False, cap_usd=None: False)
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
    ChatService().send(chat_user_id=1_000_000_001, message="good?", model="gpt-5", attached_text="Trout .312")
    user_turn = captured["messages"][-1]
    assert user_turn["role"] == "user"
    assert user_turn["content"] == "[Context the user selected on the page]\nTrout .312\n\n[Question]\ngood?"


def test_send_threads_image_attachment_as_multimodal(monkeypatch):
    from types import SimpleNamespace

    captured = {}
    monkeypatch.setattr(cs, "_provider_of", lambda m: "openai")
    monkeypatch.setattr(cs.keys, "get_key", lambda uid, prov: "sk-user")
    monkeypatch.setattr(cs.keys, "list_keys", lambda uid: [{"provider": "openai"}])
    monkeypatch.setattr(cs.budget, "is_over_cap", lambda uid, on_own_key=False, cap_usd=None: False)
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
        chat_user_id=1_000_000_001,
        message="what's wrong?",
        model="gpt-5",
        attachments=[SimpleNamespace(kind="image", data_url="data:image/png;base64,ZZZ")],
    )
    content = captured["messages"][-1]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "what's wrong?"}
    assert content[1]["image_url"]["url"] == "data:image/png;base64,ZZZ"


def test_models_filters_to_available_providers(monkeypatch):
    monkeypatch.setattr(cs.keys, "list_keys", lambda uid: [{"provider": "openai"}])
    monkeypatch.setattr(cs, "_list_admin_shared_providers", lambda: [])
    monkeypatch.setattr(cs, "_model_catalog", lambda: [("GPT-5", "gpt-5"), ("Claude", "claude-x")])
    monkeypatch.setattr(cs, "_provider_of", lambda m: "openai" if m == "gpt-5" else "anthropic")
    out = ChatService().models(chat_user_id=1)
    assert [m["id"] for m in out] == ["gpt-5"]  # only the provider with a key


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


def test_send_stream_over_cap_emits_over_cap_code(monkeypatch):
    import json

    monkeypatch.setattr(cs, "_provider_of", lambda m: "openai")
    monkeypatch.setattr(cs.keys, "get_key", lambda uid, prov: "sk-shared")
    monkeypatch.setattr(cs.keys, "list_keys", lambda uid: [])  # no own key -> managed
    monkeypatch.setattr(cs.budget, "is_over_cap", lambda uid, on_own_key=False, cap_usd=None: True)
    frames = list(ChatService().send_stream(chat_user_id=1_000_000_001, message="hi", model="gpt-5", cap_usd=0.10))
    first = json.loads(frames[0][len("data: ") : -2])
    assert first["type"] == "error" and first["code"] == "over_cap"


def test_send_stream_threads_cap_usd_into_budget(monkeypatch):
    captured = {}
    monkeypatch.setattr(cs, "_provider_of", lambda m: "openai")
    monkeypatch.setattr(cs.keys, "get_key", lambda uid, prov: "sk-shared")
    monkeypatch.setattr(cs.keys, "list_keys", lambda uid: [])

    def _over(uid, on_own_key=False, cap_usd=None):
        captured["cap_usd"] = cap_usd
        return True  # short-circuit (over cap) so we don't need a fake provider

    monkeypatch.setattr(cs.budget, "is_over_cap", _over)
    list(ChatService().send_stream(chat_user_id=1, message="hi", model="gpt-5", cap_usd=0.42))
    assert captured["cap_usd"] == 0.42

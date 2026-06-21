"""_chat_events streams events; chat() drains them into TODAY's exact dict.
Both _completion (streamed chunks) and _rebuild (chunk reassembly) are
monkeypatched so the test is DB-free and litellm-internals-free."""

import json
from types import SimpleNamespace

from src.ai import providers


def _delta_chunk(content=None):
    # A streamed chunk: choices[0].delta.content carries the token piece.
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content, tool_calls=None))])


def _rebuilt(content=None, tool_calls=None, in_tok=10, out_tok=5):
    # What _rebuild returns: a full response with .choices[0].message + .usage.
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))],
        usage=SimpleNamespace(prompt_tokens=in_tok, completion_tokens=out_tok),
    )


def _script(monkeypatch, completion_returns, rebuild_returns):
    """completion_returns / rebuild_returns are lists popped per round."""
    monkeypatch.setattr(providers, "_completion", lambda **kw: iter(completion_returns.pop(0)))
    monkeypatch.setattr(providers, "_rebuild", lambda chunks, messages: rebuild_returns.pop(0))


def test_chat_events_simple_answer_yields_deltas_then_done(monkeypatch):
    _script(
        monkeypatch,
        completion_returns=[[_delta_chunk("Hi "), _delta_chunk("there")]],
        rebuild_returns=[_rebuilt(content="Hi there")],
    )
    events = list(
        providers._chat_events(
            model="anthropic/claude-haiku-4-5",
            messages=[{"role": "user", "content": "hi"}],
            api_key="sk-test",
            user_id=99,
        )
    )
    kinds = [e["type"] for e in events]
    assert kinds == ["text_delta", "text_delta", "done"]
    assert "".join(e["text"] for e in events if e["type"] == "text_delta") == "Hi there"
    done = events[-1]
    assert done["content"] == "Hi there" and done["tokens_out"] == 5 and done["tool_trace"] == []


def test_chat_events_tool_round_emits_tool_chips(monkeypatch):
    tc = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="query_data", arguments=json.dumps({"sql": "SELECT 1 AS one"})),
    )
    monkeypatch.setattr(providers, "dispatch_tool", lambda name, args, user_id: json.dumps([{"one": 1}]))
    _script(
        monkeypatch,
        completion_returns=[[_delta_chunk(None)], [_delta_chunk("The answer is 1")]],
        rebuild_returns=[_rebuilt(content=None, tool_calls=[tc]), _rebuilt(content="The answer is 1")],
    )
    events = list(
        providers._chat_events(
            model="anthropic/claude-haiku-4-5",
            messages=[{"role": "user", "content": "what is one"}],
            api_key="sk-test",
            user_id=99,
        )
    )
    kinds = [e["type"] for e in events]
    assert "tool_started" in kinds and "tool_result" in kinds and kinds[-1] == "done"
    started = next(e for e in events if e["type"] == "tool_started")
    assert started["name"] == "query_data" and started["args"] == {"sql": "SELECT 1 AS one"}
    assert next(e for e in events if e["type"] == "tool_result")["ok"] is True
    assert events[-1]["content"] == "The answer is 1"
    assert any(t["name"] == "query_data" for t in events[-1]["tool_trace"])


def test_chat_is_a_drain_returning_todays_exact_dict(monkeypatch):
    """FROZEN REFERENCE: chat() must return byte-identical to the pre-B2.1 dict
    shape so the live Streamlit app (src/ai/chat.py) is provably unchanged."""
    tc = SimpleNamespace(
        id="c1", function=SimpleNamespace(name="query_data", arguments=json.dumps({"sql": "SELECT 1"}))
    )
    monkeypatch.setattr(providers, "dispatch_tool", lambda name, args, user_id: json.dumps([{"one": 1}]))
    _script(
        monkeypatch,
        completion_returns=[[_delta_chunk(None)], [_delta_chunk("The answer is 1")]],
        rebuild_returns=[
            _rebuilt(content=None, tool_calls=[tc], in_tok=10, out_tok=5),
            _rebuilt(content="The answer is 1", in_tok=7, out_tok=3),
        ],
    )
    out = providers.chat(
        model="anthropic/claude-haiku-4-5",
        messages=[{"role": "user", "content": "what is one"}],
        api_key="sk-test",
        user_id=99,
    )
    assert out == {
        "content": "The answer is 1",
        "tokens_in": 17,
        "tokens_out": 8,
        "tool_trace": [{"name": "query_data", "args": {"sql": "SELECT 1"}}],
    }


def test_loop_cap_still_returns_graceful_message(monkeypatch):
    tc = SimpleNamespace(id="c", function=SimpleNamespace(name="query_data", arguments=json.dumps({"sql": "SELECT 1"})))
    monkeypatch.setattr(providers, "dispatch_tool", lambda name, args, user_id: json.dumps([{"one": 1}]))
    # Every round asks for a tool -> cap hits. Provide enough scripted rounds.
    monkeypatch.setattr(providers, "_completion", lambda **kw: iter([_delta_chunk(None)]))
    monkeypatch.setattr(providers, "_rebuild", lambda chunks, messages: _rebuilt(content=None, tool_calls=[tc]))
    out = providers.chat(
        model="anthropic/claude-haiku-4-5",
        messages=[{"role": "user", "content": "loop"}],
        api_key="sk-test",
        user_id=99,
        max_tool_rounds=3,
    )
    assert out["content"] and out["tool_trace"]  # graceful, not a hang


def test_empty_stream_rebuild_none_yields_graceful_done_not_crash(monkeypatch):
    """litellm.stream_chunk_builder returns None for an empty stream (zero chunks).
    _chat_events must terminate gracefully (exactly one `done`) and chat() must NOT
    crash on resp.choices — it is the live Streamlit app's hot path."""
    monkeypatch.setattr(providers, "_completion", lambda **kw: iter([]))  # zero chunks
    monkeypatch.setattr(providers, "_rebuild", lambda chunks, messages: None)
    events = list(
        providers._chat_events(
            model="anthropic/claude-haiku-4-5",
            messages=[{"role": "user", "content": "hi"}],
            api_key="sk-test",
            user_id=99,
        )
    )
    assert [e["type"] for e in events] == ["done"]
    out = providers.chat(
        model="anthropic/claude-haiku-4-5",
        messages=[{"role": "user", "content": "hi"}],
        api_key="sk-test",
        user_id=99,
    )
    assert out["content"] and out["tokens_in"] == 0 and out["tool_trace"] == []

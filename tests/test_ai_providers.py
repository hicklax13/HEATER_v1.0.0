"""Agentic tool loop: model asks for a tool, we dispatch, model answers.
Streaming-mocked (B2.1): _completion yields chunks; _rebuild reassembles."""

import json
from types import SimpleNamespace

import pytest

from src.database import init_db


def _delta(content=None):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content, tool_calls=None))])


def _rebuilt(content=None, tool_calls=None, in_tok=10, out_tok=5):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))],
        usage=SimpleNamespace(prompt_tokens=in_tok, completion_tokens=out_tok),
    )


@pytest.fixture(autouse=True)
def _db(monkeypatch):
    monkeypatch.setenv("MULTI_USER", "1")
    init_db()


def test_simple_answer_no_tools(monkeypatch):
    from src.ai import providers

    monkeypatch.setattr(providers, "_completion", lambda **kw: iter([_delta("Hi "), _delta("there")]))
    monkeypatch.setattr(providers, "_rebuild", lambda chunks, messages: _rebuilt(content="Hi there"))
    out = providers.chat(
        model="anthropic/claude-haiku-4-5",
        messages=[{"role": "user", "content": "hi"}],
        api_key="sk-test",
        user_id=99,
    )
    assert out["content"] == "Hi there"
    assert out["tokens_out"] == 5


def test_tool_then_answer(monkeypatch):
    from src.ai import providers

    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="query_data", arguments=json.dumps({"sql": "SELECT 1 AS one"})),
    )
    completions = [[_delta(None)], [_delta("The answer is 1")]]
    rebuilds = [_rebuilt(content=None, tool_calls=[tool_call]), _rebuilt(content="The answer is 1")]
    monkeypatch.setattr(providers, "_completion", lambda **kw: iter(completions.pop(0)))
    monkeypatch.setattr(providers, "_rebuild", lambda chunks, messages: rebuilds.pop(0))
    out = providers.chat(
        model="anthropic/claude-haiku-4-5",
        messages=[{"role": "user", "content": "what is one"}],
        api_key="sk-test",
        user_id=99,
    )
    assert out["content"] == "The answer is 1"
    assert any(t["name"] == "query_data" for t in out["tool_trace"])


def test_loop_cap(monkeypatch):
    from src.ai import providers

    tool_call = SimpleNamespace(
        id="c", function=SimpleNamespace(name="query_data", arguments=json.dumps({"sql": "SELECT 1"}))
    )
    # model loops forever asking for tools; the cap must stop it
    monkeypatch.setattr(providers, "_completion", lambda **kw: iter([_delta(None)]))
    monkeypatch.setattr(providers, "_rebuild", lambda chunks, messages: _rebuilt(content=None, tool_calls=[tool_call]))
    out = providers.chat(
        model="anthropic/claude-haiku-4-5",
        messages=[{"role": "user", "content": "loop"}],
        api_key="sk-test",
        user_id=99,
        max_tool_rounds=3,
    )
    assert out["content"]  # returns a graceful message rather than hanging

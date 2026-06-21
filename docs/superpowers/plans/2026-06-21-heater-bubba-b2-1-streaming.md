# Bubba B2.1 — Streaming + Thinking-Effort + Research Toggles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Bubba *feel* like claude.ai — assistant text streams token-by-token over SSE, tool use shows live status chips, a thinking-effort dial (Off/Low/Med/High) maps per-provider, and Web-search / Deep-research toggles surface in the composer.

**Architecture:** ONE event core, two consumers. A new private generator `src/ai/providers.py::_chat_events(...)` yields `text_delta`/`tool_started`/`tool_result`/`done` events; the existing `chat()` becomes a thin **drain** of that generator returning **today's exact dict** (a frozen-reference equivalence test proves the live Streamlit app's contract is unchanged). A new `ChatService.send_stream(...)` iterates `_chat_events`, formats SSE frames, and meters/persists server-side on `done`. A new `POST /api/chat/send-stream` returns a `StreamingResponse`. The frontend gains a streaming `fetch` reader and a streaming composer.

**Tech Stack:** FastAPI 0.137.1 (pinned), Pydantic v2, `src/ai/` (litellm `>=1.80.0`, sync streaming via `litellm.completion(stream=True)` + `litellm.stream_chunk_builder`), pytest. Frontend: Next.js 16 + React 19 + TypeScript, `fetch` + `ReadableStream`. **No new deps.**

**Coordination:** This touches `src/ai/providers.py` (also used by the LIVE Streamlit app) + `api/` + `web/`. The drain-keeps-shape design + the equivalence guard make the `providers.py` change contract-safe. `api/main.py` already mounts the chat router (B1), so the new route lands by adding it to the existing `api/routers/chat.py` — no main.py change, but `api/openapi.json` must be regenerated.

---

## Grounding (verified in source — do NOT re-derive)

- `src/ai/providers.py::chat(model, messages, api_key, user_id, max_tool_rounds=6, web_search=False, deep_research=False) -> {"content","tokens_in","tokens_out","tool_trace"}` is a sync ≤6-round tool loop. `_completion(**kwargs)` wraps `litellm.completion`. The final text is the last round's `msg.content`.
- `tests/test_ai_providers.py` monkeypatches `providers._completion` with **non-streaming** `SimpleNamespace` responses (3 tests: `test_simple_answer_no_tools`, `test_tool_then_answer`, `test_loop_cap`). These MUST be migrated to streaming chunk mocks in Task 2.
- The LIVE Streamlit app calls `providers.chat` at `src/ai/chat.py:203` (`from src.ai.providers import chat as provider_chat`) and reads the dict — so `chat()`'s return shape is load-bearing.
- `src/ai/router.py` holds `provider_of(model) -> str`, `model_catalog()`, `price_per_token(model)`. `thinking_params_for_model` lands here.
- `api/services/chat_service.py::ChatService.send(...)` already: validates key, checks `budget.is_over_cap(uid, on_own_key=...)`, loads history, builds the system prompt via `from src.ai.chat import build_system_prompt as _build_system_prompt` called as `_build_system_prompt(page or _DEFAULT_PAGE, viewer_team)`, calls `providers.chat`, then meters (`budget.record_usage`) + persists (guarded). `send_stream` mirrors this.
- `src/ai/chat.py::build_system_prompt(page: str, viewer_team: str | None) -> str`.
- `api/routers/chat.py` is thin + AST-guarded by `tests/api/test_no_logic_in_routers.py` (no `from src.*`, no `BinOp` assignment). `_uid(app_user)` raises 401 on `None`.
- `api/contracts/chat.py::ChatSendRequest` is shared by `/send` (and will be by `/send-stream`).
- OpenAPI regen: `python scripts/export_openapi.py`. Snapshot guard: `tests/api/test_openapi_contract.py` compares `create_app().openapi()` to `api/openapi.json` (sorted-keys).
- Frontend: `web/src/lib/api/client.ts` has a module-private `authToken()` (Clerk bearer) used by `apiPost`; `web/next.config.ts` rewrites `/api/:path*` to the FastAPI origin (passes `text/event-stream` through). `web/src/components/bubba/Bubba.tsx` is the B1 popup; `web/src/lib/api/bubba.ts` is the typed client; `web/src/lib/api/errors.ts` has `ApiError`/`isAuthRequired`.

**Use the venv python:** `.venv/Scripts/python.exe`. Frontend commands run from `web/` with `pnpm`.

---

## File Structure

- **Modify** `src/ai/router.py` — add `thinking_params_for_model(model, effort) -> dict`.
- **Modify** `src/ai/providers.py` — add `_rebuild(...)` wrapper + `_chat_events(...)` generator + `_tool_ok(...)`; rewrite `chat()` as a drain; thread `reasoning_effort`.
- **Modify** `api/contracts/chat.py` — add `reasoning_effort` to `ChatSendRequest`.
- **Modify** `api/services/chat_service.py` — thread `reasoning_effort` into `send`; add `send_stream(...)` + the `_meter_and_persist` + `_sse` helpers.
- **Modify** `api/routers/chat.py` — thread `reasoning_effort` into `/send`; add `POST /api/chat/send-stream` (`StreamingResponse`).
- **Modify** `api/openapi.json` — regenerate.
- **Modify** `web/src/lib/api/client.ts` — export `authToken`.
- **Modify** `web/src/lib/api/bubba.ts` — add `sendStream(...)`, event types, `reasoning_effort` on `ChatSendBody`.
- **Modify** `web/src/components/bubba/Bubba.tsx` — streaming send + status chips + effort dial + toggles.
- **Create** `tests/test_ai_thinking_params.py`, `tests/test_ai_providers_streaming.py`.
- **Modify** `tests/test_ai_providers.py` (migrate 3 tests), `tests/api/test_chat_service.py` (extend), `tests/api/test_api_chat.py` (extend).

---

### Task 1: `thinking_params_for_model` (per-provider effort mapping)

**Files:**
- Modify: `src/ai/router.py` (append a function)
- Test: `tests/test_ai_thinking_params.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ai_thinking_params.py
"""thinking_params_for_model maps the Off/Low/Med/High dial to per-provider
reasoning params. off/low -> {} (fastest); unsupported provider -> {} (no-op)."""

import pytest

from src.ai.router import thinking_params_for_model


@pytest.mark.parametrize("effort", [None, "off", "low"])
def test_off_and_low_are_noop_for_every_provider(effort):
    assert thinking_params_for_model("anthropic/claude-opus-4-8", effort) == {}
    assert thinking_params_for_model("openai/gpt-5.5", effort) == {}
    assert thinking_params_for_model("deepseek/deepseek-v4-pro", effort) == {}


def test_anthropic_medium_and_high_emit_thinking_budget():
    med = thinking_params_for_model("anthropic/claude-opus-4-8", "medium")
    high = thinking_params_for_model("anthropic/claude-opus-4-8", "high")
    assert med["thinking"]["type"] == "enabled" and med["thinking"]["budget_tokens"] > 0
    assert high["thinking"]["budget_tokens"] > med["thinking"]["budget_tokens"]


def test_openai_uses_reasoning_effort_passthrough():
    assert thinking_params_for_model("openai/gpt-5.5", "medium") == {"reasoning_effort": "medium"}
    assert thinking_params_for_model("openai/gpt-5.5", "high") == {"reasoning_effort": "high"}


def test_unsupported_provider_is_noop_even_on_high():
    assert thinking_params_for_model("deepseek/deepseek-v4-pro", "high") == {}
    assert thinking_params_for_model("gemini/gemini-3.5-flash", "high") == {}
```

- [ ] **Step 2: Run — expect FAIL** (`cannot import name 'thinking_params_for_model'`)

Run: `.venv/Scripts/python.exe -m pytest tests/test_ai_thinking_params.py -q`

- [ ] **Step 3: Implement** — append to `src/ai/router.py` (after `price_per_token`):

```python
# Thinking-effort dial (Bubba B2.1). The UI offers Off/Low/Med/High; off/low map
# to {} (fastest, no reasoning) so the dial defaults to zero overhead. Only
# medium/high spend reasoning. Per-provider: Anthropic takes a thinking-budget
# (token count); OpenAI takes reasoning_effort ("medium"/"high"). Providers
# without a reasoning param (DeepSeek, Gemini, xAI, OpenRouter, ollama) -> {}:
# the dial still renders but is a silent no-op there (documented).
_ANTHROPIC_THINKING_BUDGET = {"medium": 4096, "high": 12288}


def thinking_params_for_model(model: str, effort: str | None) -> dict:
    """Map the effort dial to litellm completion kwargs for `model`. {} = no-op."""
    if effort in (None, "off", "low"):
        return {}
    provider = provider_of(model)
    if provider == "anthropic":
        budget = _ANTHROPIC_THINKING_BUDGET.get(effort)
        return {"thinking": {"type": "enabled", "budget_tokens": budget}} if budget else {}
    if provider == "openai":
        return {"reasoning_effort": effort}
    return {}
```

- [ ] **Step 4: Run — expect PASS**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ai_thinking_params.py -q`

- [ ] **Step 5: Commit**

```bash
git add src/ai/router.py tests/test_ai_thinking_params.py
git commit -m "feat(ai): thinking_params_for_model per-provider effort dial (Bubba B2.1)"
```

---

### Task 2: `_chat_events` generator + `chat()` becomes its drain (the live-app-safe refactor)

**Files:**
- Modify: `src/ai/providers.py` (rewrite)
- Test: `tests/test_ai_providers_streaming.py` (new), `tests/test_ai_providers.py` (migrate the 3 existing tests)

- [ ] **Step 1: Write the new failing test** — event sequence + the frozen-reference drain equivalence

```python
# tests/test_ai_providers_streaming.py
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
    tc = SimpleNamespace(
        id="c", function=SimpleNamespace(name="query_data", arguments=json.dumps({"sql": "SELECT 1"}))
    )
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
```

- [ ] **Step 2: Run — expect FAIL** (`_chat_events` / `_rebuild` not defined)

Run: `.venv/Scripts/python.exe -m pytest tests/test_ai_providers_streaming.py -q`

- [ ] **Step 3: Rewrite `src/ai/providers.py`** (full file)

```python
"""LiteLLM wrapper + the agentic tool loop (Bubba B2.1 — streaming core).

_chat_events() runs the loop and YIELDS events (text deltas, tool chips, a
terminal `done`). chat() is a thin DRAIN of _chat_events returning the same
dict it always has — the live Streamlit app (src/ai/chat.py) depends on that
shape, so a frozen-reference equivalence test guards it. _completion and
_rebuild are split out so tests can mock LiteLLM streaming with no network.
"""

from __future__ import annotations

import json
from collections.abc import Generator

from src.ai.router import thinking_params_for_model
from src.ai.tools import dispatch_tool, tool_specs

_MAX_TOOL_ROUNDS = 6


def _completion(**kwargs):
    import litellm

    return litellm.completion(**kwargs)


def _rebuild(chunks: list, messages: list[dict]):
    """Reassemble streamed chunks into a full response (message + usage)."""
    import litellm

    return litellm.stream_chunk_builder(chunks, messages=messages)


def _tool_ok(result: str) -> bool:
    """A tool result is a JSON string; dispatch_tool emits {"error": ...} on
    failure. The chip is cosmetic, so be lenient: non-JSON counts as ok."""
    try:
        parsed = json.loads(result)
    except (ValueError, TypeError):
        return True
    return not (isinstance(parsed, dict) and "error" in parsed)


def _chat_events(
    model: str,
    messages: list[dict],
    api_key: str | None,
    user_id: int,
    max_tool_rounds: int = _MAX_TOOL_ROUNDS,
    web_search: bool = False,
    deep_research: bool = False,
    reasoning_effort: str | None = None,
) -> Generator[dict, None, None]:
    """Yield streaming events for one chat turn:
      {"type":"text_delta","text": ...}        - streamed content (every round)
      {"type":"tool_started","name","args"}    - before each tool dispatch
      {"type":"tool_result","name","ok"}       - after dispatch (no payload)
      {"type":"done","content","tokens_in","tokens_out","tool_trace"} - terminal
    The `done` event carries the SAME totals chat() returns today.
    """
    convo = list(messages)
    tool_trace: list[dict] = []
    tokens_in = tokens_out = 0
    specs = tool_specs(web_search_enabled=web_search, deep_research_enabled=deep_research)
    thinking = thinking_params_for_model(model, reasoning_effort)

    for _ in range(max_tool_rounds):
        chunks: list = []
        for chunk in _completion(
            model=model,
            messages=convo,
            tools=specs,
            api_key=api_key,
            stream=True,
            stream_options={"include_usage": True},
            **thinking,
        ):
            chunks.append(chunk)
            choices = getattr(chunk, "choices", None)
            delta = getattr(choices[0], "delta", None) if choices else None
            piece = getattr(delta, "content", None) if delta is not None else None
            if piece:
                yield {"type": "text_delta", "text": piece}

        resp = _rebuild(chunks, convo)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            tokens_in += int(getattr(usage, "prompt_tokens", 0) or 0)
            tokens_out += int(getattr(usage, "completion_tokens", 0) or 0)
        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        if not tool_calls:
            yield {
                "type": "done",
                "content": msg.content or "",
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "tool_trace": tool_trace,
            }
            return

        # record the assistant turn that requested tools
        convo.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            }
        )
        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except (ValueError, TypeError):
                args = {}
            yield {"type": "tool_started", "name": tc.function.name, "args": args}
            result = dispatch_tool(tc.function.name, args, user_id=user_id)
            tool_trace.append({"name": tc.function.name, "args": args})
            # `name` is optional on OpenAI/Anthropic tool messages but some
            # OpenAI-compatible + ollama gateways rely on it — echo it for safety.
            convo.append({"role": "tool", "tool_call_id": tc.id, "name": tc.function.name, "content": result})
            yield {"type": "tool_result", "name": tc.function.name, "ok": _tool_ok(result)}

    # loop cap hit
    yield {
        "type": "done",
        "content": "I wasn't able to finish that within the tool-call limit. Try narrowing the question.",
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "tool_trace": tool_trace,
    }


def chat(
    model: str,
    messages: list[dict],
    api_key: str | None,
    user_id: int,
    max_tool_rounds: int = _MAX_TOOL_ROUNDS,
    web_search: bool = False,
    deep_research: bool = False,
    reasoning_effort: str | None = None,
) -> dict:
    """Drain _chat_events into {content, tokens_in, tokens_out, tool_trace} —
    the unchanged contract the live Streamlit app consumes (src/ai/chat.py)."""
    done: dict | None = None
    for event in _chat_events(
        model=model,
        messages=messages,
        api_key=api_key,
        user_id=user_id,
        max_tool_rounds=max_tool_rounds,
        web_search=web_search,
        deep_research=deep_research,
        reasoning_effort=reasoning_effort,
    ):
        if event["type"] == "done":
            done = event
    # _chat_events ALWAYS yields exactly one terminal done.
    return {
        "content": done["content"],
        "tokens_in": done["tokens_in"],
        "tokens_out": done["tokens_out"],
        "tool_trace": done["tool_trace"],
    }
```

- [ ] **Step 4: Migrate the 3 existing tests** in `tests/test_ai_providers.py` to streaming mocks. Replace the `_resp` helper and the 3 test bodies so they script `_completion` (streamed chunks) + `_rebuild` (reassembly). Full replacement file:

```python
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
```

- [ ] **Step 5: Run — expect PASS** (new streaming tests + migrated behavioral tests)

Run: `.venv/Scripts/python.exe -m pytest tests/test_ai_providers_streaming.py tests/test_ai_providers.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/ai/providers.py tests/test_ai_providers_streaming.py tests/test_ai_providers.py
git commit -m "refactor(ai): _chat_events streaming core; chat() drains it (Bubba B2.1)"
```

---

### Task 3: Thread `reasoning_effort` through the contract + `/send` (back-compat)

**Files:**
- Modify: `api/contracts/chat.py`, `api/services/chat_service.py`, `api/routers/chat.py`
- Test: `tests/api/test_chat_service.py` (extend), `tests/api/test_chat_contracts.py` (extend)

- [ ] **Step 1: Write the failing tests**

Append to `tests/api/test_chat_contracts.py`:

```python
def test_send_request_reasoning_effort_defaults_none_and_accepts_levels():
    from api.contracts.chat import ChatSendRequest

    assert ChatSendRequest(message="hi", model="m").reasoning_effort is None
    assert ChatSendRequest(message="hi", model="m", reasoning_effort="high").reasoning_effort == "high"
```

Append to `tests/api/test_chat_service.py`:

```python
def test_send_threads_reasoning_effort_into_providers(monkeypatch):
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
    ChatService().send(chat_user_id=1_000_000_001, message="hi", model="gpt-5", reasoning_effort="high")
    assert captured.get("reasoning_effort") == "high"
```

- [ ] **Step 2: Run — expect FAIL**

Run: `.venv/Scripts/python.exe -m pytest tests/api/test_chat_contracts.py tests/api/test_chat_service.py -q`

- [ ] **Step 3a: Add `reasoning_effort` to the contract** — `api/contracts/chat.py`, change the import line and `ChatSendRequest`:

```python
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ChatSendRequest(BaseModel):
    message: str
    model: str
    conversation_id: int | None = None
    web_search: bool = False
    deep_research: bool = False
    reasoning_effort: Literal["off", "low", "medium", "high"] | None = None
```

- [ ] **Step 3b: Thread it through `ChatService.send`** — `api/services/chat_service.py`. Add the param to `send`'s signature (after `deep_research`):

```python
        deep_research: bool = False,
        reasoning_effort: str | None = None,
        page: str | None = None,
        viewer_team: str | None = None,
    ) -> dict:
```

And pass it into the `providers.chat(...)` call:

```python
            result = providers.chat(
                model=model,
                messages=convo,
                api_key=api_key,
                user_id=chat_user_id,
                web_search=web_search,
                deep_research=deep_research,
                reasoning_effort=reasoning_effort,
            )
```

- [ ] **Step 3c: Thread it through the `/send` router** — `api/routers/chat.py`, in `send(...)` add to the `svc.send(...)` kwargs:

```python
            web_search=body.web_search,
            deep_research=body.deep_research,
            reasoning_effort=body.reasoning_effort,
        )
    )
```

- [ ] **Step 4: Run — expect PASS** (contract + service + the existing endpoint + router-logic guards)

Run: `.venv/Scripts/python.exe -m pytest tests/api/test_chat_contracts.py tests/api/test_chat_service.py tests/api/test_api_chat.py tests/api/test_no_logic_in_routers.py -q`

- [ ] **Step 5: Commit**

```bash
git add api/contracts/chat.py api/services/chat_service.py api/routers/chat.py tests/api/test_chat_contracts.py tests/api/test_chat_service.py
git commit -m "feat(api): thread reasoning_effort through chat contract + /send (Bubba B2.1)"
```

---

### Task 4: `ChatService.send_stream` (SSE generator + server-side metering)

**Files:**
- Modify: `api/services/chat_service.py` (add `send_stream` + helpers)
- Test: `tests/api/test_chat_service_stream.py` (new)

- [ ] **Step 1: Write the failing tests** (DB-free: fake `providers._chat_events`, in-memory budget/history)

```python
# tests/api/test_chat_service_stream.py
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
    monkeypatch.setattr(cs.budget, "is_over_cap", lambda uid, on_own_key=False: False)
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
    monkeypatch.setattr(cs.budget, "is_over_cap", lambda uid, on_own_key=False: True)
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
```

- [ ] **Step 2: Run — expect FAIL** (`send_stream` not defined)

Run: `.venv/Scripts/python.exe -m pytest tests/api/test_chat_service_stream.py -q`

- [ ] **Step 3: Implement** — add to `api/services/chat_service.py`. First add the import + a module helper near the top (after `_log = logging.getLogger(__name__)`):

```python
import json as _json
from collections.abc import Generator


def _sse(payload: dict) -> str:
    """Format one Server-Sent-Events frame."""
    return f"data: {_json.dumps(payload)}\n\n"
```

Then add these methods to `ChatService` (after `send`, before `_empty`):

```python
    def send_stream(
        self,
        chat_user_id: int,
        message: str,
        model: str,
        conversation_id: int | None = None,
        web_search: bool = False,
        deep_research: bool = False,
        reasoning_effort: str | None = None,
        page: str | None = None,
        viewer_team: str | None = None,
    ) -> Generator[str, None, None]:
        """Stream the chat as SSE frames. Mirrors send()'s pre-call validation,
        then iterates providers._chat_events. Meters + persists on the `done`
        event BEFORE yielding the terminal frame, so a client that drops at the
        final frame is still billed. Never raises — failures become an `error`
        frame (the HTTP response is already 200)."""
        try:
            provider = _provider_of(model)
            api_key = keys.get_key(chat_user_id, provider)
            if not api_key:
                yield _sse({"type": "error", "message": f"No API key for {provider}. Add one in settings to chat."})
                return
            on_own_key = any(k.get("provider") == provider for k in keys.list_keys(chat_user_id))
            if budget.is_over_cap(chat_user_id, on_own_key=on_own_key):
                yield _sse(
                    {
                        "type": "error",
                        "message": "You've reached today's usage limit. Add your own API key for unlimited use.",
                    }
                )
                return

            prior = history.load_messages(conversation_id, user_id=chat_user_id) if conversation_id else []
            convo = [{"role": "system", "content": _build_system_prompt(page or _DEFAULT_PAGE, viewer_team)}]
            convo += [{"role": m["role"], "content": m["content"]} for m in prior]
            convo.append({"role": "user", "content": message})
        except Exception as e:  # noqa: BLE001 - pre-call failure -> error frame, never 500
            _log.warning("chat send_stream failed before streaming", exc_info=True)
            yield _sse({"type": "error", "message": f"Chat error: {type(e).__name__}: {str(e)[:200]}"})
            return

        try:
            for event in providers._chat_events(
                model=model,
                messages=convo,
                api_key=api_key,
                user_id=chat_user_id,
                web_search=web_search,
                deep_research=deep_research,
                reasoning_effort=reasoning_effort,
            ):
                if event["type"] != "done":
                    yield _sse(event)
                    continue
                # Terminal: meter + persist FIRST (so a disconnect at the done
                # frame is still billed), then emit the enriched done frame.
                cost, conversation_id = self._meter_and_persist(
                    chat_user_id, model, message, event, conversation_id
                )
                yield _sse(
                    {
                        "type": "done",
                        "conversation_id": conversation_id,
                        "cost_usd": cost,
                        "content": event["content"],
                        "tokens_in": event["tokens_in"],
                        "tokens_out": event["tokens_out"],
                        "tool_trace": event["tool_trace"],
                    }
                )
                return
        except Exception as e:  # noqa: BLE001 - engine failure mid-stream -> error frame
            _log.warning("chat send_stream failed mid-stream", exc_info=True)
            yield _sse({"type": "error", "message": f"Chat error: {type(e).__name__}: {str(e)[:200]}"})

    def _meter_and_persist(
        self, chat_user_id: int, model: str, message: str, done: dict, conversation_id: int | None
    ) -> tuple[float, int | None]:
        """Record usage + append history for a completed answer. Each side is
        isolated + logged + never fatal (mirrors send()). Returns (cost, conversation_id)."""
        t_in = int(done.get("tokens_in", 0) or 0)
        t_out = int(done.get("tokens_out", 0) or 0)
        p_in, p_out = _price_per_token(model)
        cost = t_in * p_in + t_out * p_out

        try:
            budget.record_usage(chat_user_id, t_in, t_out, cost)
        except Exception:
            _log.warning("record_usage failed (spend uncounted)", exc_info=True)

        try:
            if conversation_id is None:
                conversation_id = history.create_conversation(chat_user_id, message[:60], model=model)
            history.append_message(conversation_id, "user", message, model=model)
            history.append_message(
                conversation_id,
                "assistant",
                done.get("content", ""),
                model=model,
                tokens_in=t_in,
                tokens_out=t_out,
                cost_usd=cost,
            )
        except Exception:
            _log.warning("chat history persist failed (answer streamed, not saved)", exc_info=True)

        return cost, conversation_id
```

- [ ] **Step 4: Run — expect PASS**

Run: `.venv/Scripts/python.exe -m pytest tests/api/test_chat_service_stream.py tests/api/test_chat_service.py -q`

- [ ] **Step 5: Commit**

```bash
git add api/services/chat_service.py tests/api/test_chat_service_stream.py
git commit -m "feat(api): ChatService.send_stream SSE generator + server-side metering (Bubba B2.1)"
```

---

### Task 5: `POST /api/chat/send-stream` router + OpenAPI regen

**Files:**
- Modify: `api/routers/chat.py` (add the streaming route), `api/openapi.json`
- Test: `tests/api/test_api_chat.py` (extend)

- [ ] **Step 1: Write the failing endpoint tests** — append to `tests/api/test_api_chat.py`. First add a streaming method to `_FakeChatService`:

```python
    def send_stream(self, **k):  # noqa: ARG002
        yield 'data: {"type": "text_delta", "text": "hi"}\n\n'
        yield 'data: {"type": "done", "content": "hi", "conversation_id": 1, "cost_usd": 0.0, "tokens_in": 1, "tokens_out": 1, "tool_trace": []}\n\n'
```

Then add the tests:

```python
def test_send_stream_returns_event_stream():
    r = _client().post("/api/chat/send-stream", json={"message": "hi", "model": "gpt-5"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert '"type": "text_delta"' in r.text and '"type": "done"' in r.text


def test_send_stream_unauthenticated_is_401():
    r = _client(user=None).post("/api/chat/send-stream", json={"message": "hi", "model": "gpt-5"})
    assert r.status_code == 401
```

- [ ] **Step 2: Run — expect FAIL** (route 404)

Run: `.venv/Scripts/python.exe -m pytest tests/api/test_api_chat.py -q`

- [ ] **Step 3: Add the route** — `api/routers/chat.py`. Add the import (top, with the FastAPI imports):

```python
from fastapi.responses import StreamingResponse
```

Then add the route right after `send` (the AST guard allows this — no `src.*` import, no `BinOp` assignment):

```python
@router.post("/send-stream")
def send_stream(
    body: ChatSendRequest,
    app_user: AppUser | None = Depends(require_app_user),
    svc: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    return StreamingResponse(
        svc.send_stream(
            chat_user_id=_uid(app_user),
            message=body.message,
            model=body.model,
            conversation_id=body.conversation_id,
            web_search=body.web_search,
            deep_research=body.deep_research,
            reasoning_effort=body.reasoning_effort,
        ),
        media_type="text/event-stream",
    )
```

- [ ] **Step 4: Run — expect PASS** (endpoint + router-logic guard)

Run: `.venv/Scripts/python.exe -m pytest tests/api/test_api_chat.py tests/api/test_no_logic_in_routers.py -q`

- [ ] **Step 5: Regenerate the OpenAPI snapshot**

Run: `.venv/Scripts/python.exe scripts/export_openapi.py`
Then verify the snapshot guard passes (the new route has no response model — `StreamingResponse` — so it gets a default 200 with no schema; confirm the snapshot updates cleanly):
Run: `.venv/Scripts/python.exe -m pytest tests/api/test_openapi_contract.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add api/routers/chat.py api/openapi.json tests/api/test_api_chat.py
git commit -m "feat(api): POST /api/chat/send-stream SSE route + regen openapi (Bubba B2.1)"
```

---

### Task 6: Frontend streaming client (`sendStream` + event types)

**Files:**
- Modify: `web/src/lib/api/client.ts` (export `authToken`), `web/src/lib/api/bubba.ts`

- [ ] **Step 1: Export `authToken` from `client.ts`** — change the function declaration so `bubba.ts` can reuse the exact Clerk-bearer logic (DRY). In `web/src/lib/api/client.ts`:

```typescript
export async function authToken(): Promise<string | null> {
```

(Only the `async function authToken` line gains the `export` keyword — body unchanged.)

- [ ] **Step 2: Add the streaming client to `bubba.ts`** — add the event types + `sendStream`, and extend `ChatSendBody`. In `web/src/lib/api/bubba.ts`:

Change the import line:

```typescript
import { apiDelete, apiGet, apiPost, apiPut, authToken } from "./client";
import { ApiError } from "./errors";
```

Extend `ChatSendBody`:

```typescript
export interface ChatSendBody {
  message: string;
  model: string;
  conversation_id?: number | null;
  web_search?: boolean;
  deep_research?: boolean;
  reasoning_effort?: "off" | "low" | "medium" | "high";
}
```

Add the streaming event union + `sendStream` (place above the `export const bubba`):

```typescript
/** SSE events from POST /api/chat/send-stream (mirrors src/ai/providers._chat_events). */
export type BubbaStreamEvent =
  | { type: "text_delta"; text: string }
  | { type: "tool_started"; name: string; args: Record<string, unknown> }
  | { type: "tool_result"; name: string; ok: boolean }
  | {
      type: "done";
      content: string;
      conversation_id: number | null;
      cost_usd: number;
      tokens_in: number;
      tokens_out: number;
      tool_trace: unknown[];
    }
  | { type: "error"; message: string };

/** POST the chat and invoke onEvent() per SSE frame. Mirrors apiPost's auth
 *  header (Clerk bearer). Throws ApiError on a non-OK status (401 -> sign in). */
async function sendStream(body: ChatSendBody, onEvent: (e: BubbaStreamEvent) => void): Promise<void> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Accept: "text/event-stream",
  };
  const token = await authToken();
  if (token) headers.Authorization = `Bearer ${token}`;

  const res = await fetch("/api/chat/send-stream", {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });
  if (!res.ok || !res.body) throw new ApiError(res.status, "/chat/send-stream");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    // SSE frames are separated by a blank line.
    let sep: number;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      const line = frame.split("\n").find((l) => l.startsWith("data: "));
      if (!line) continue;
      try {
        onEvent(JSON.parse(line.slice(6)) as BubbaStreamEvent);
      } catch {
        // ignore a malformed frame rather than killing the stream
      }
    }
  }
}
```

Add `sendStream` to the exported `bubba` object (add the property):

```typescript
export const bubba = {
  send: (body: ChatSendBody) => apiPost<ChatSendResult>("/chat/send", body),

  sendStream,
```

- [ ] **Step 3: Typecheck**

Run (from `web/`): `pnpm exec tsc --noEmit`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add web/src/lib/api/client.ts web/src/lib/api/bubba.ts
git commit -m "feat(web): bubba.sendStream SSE client + reasoning_effort body (Bubba B2.1)"
```

---

### Task 7: Frontend composer — streaming render + status chips + effort dial + toggles

**Files:**
- Modify: `web/src/components/bubba/Bubba.tsx`

- [ ] **Step 1: Add streaming state + a tool-status chip type.** In `BubbaPanel`, replace the `UiMessage` interface (top of file) and add the new composer state. Change `UiMessage`:

```typescript
type Phase = "loading" | "ready" | "auth" | "error";
type Effort = "off" | "low" | "medium" | "high";
interface UiMessage {
  role: "user" | "assistant";
  content: string;
  isError?: boolean;
}
```

Add state inside `BubbaPanel` (next to the other `useState` calls):

```typescript
  const [effort, setEffort] = useState<Effort>("off");
  const [webSearch, setWebSearch] = useState(false);
  const [deepResearch, setDeepResearch] = useState(false);
  const [toolStatus, setToolStatus] = useState<string | null>(null);
```

- [ ] **Step 2: Replace `handleSend` with the streaming version.** Replace the whole `handleSend` callback:

```typescript
  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || sending || !model) return;
    setInput("");
    // Append the user turn + an empty assistant turn we stream INTO.
    setMessages((prev) => [...prev, { role: "user", content: text }, { role: "assistant", content: "" }]);
    setSending(true);
    setToolStatus(null);

    const appendToLast = (chunk: string) =>
      setMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        if (last && last.role === "assistant") next[next.length - 1] = { ...last, content: last.content + chunk };
        return next;
      });

    try {
      await bubba.sendStream(
        {
          message: text,
          model,
          conversation_id: conversationId,
          web_search: webSearch,
          deep_research: deepResearch,
          reasoning_effort: effort,
        },
        (e) => {
          if (e.type === "text_delta") {
            appendToLast(e.text);
          } else if (e.type === "tool_started") {
            setToolStatus(toolLabel(e.name));
          } else if (e.type === "tool_result") {
            setToolStatus(null);
          } else if (e.type === "done") {
            if (e.conversation_id) setConversationId(e.conversation_id);
            setToolStatus(null);
          } else if (e.type === "error") {
            setMessages((prev) => {
              const next = [...prev];
              next[next.length - 1] = { role: "assistant", content: e.message, isError: true };
              return next;
            });
          }
        },
      );
    } catch (err) {
      if (isAuthRequired(err)) {
        setPhase("auth");
      } else {
        setMessages((prev) => {
          const next = [...prev];
          next[next.length - 1] = {
            role: "assistant",
            content: "Bubba couldn't reach the server. Try again.",
            isError: true,
          };
          return next;
        });
      }
    } finally {
      setSending(false);
      setToolStatus(null);
    }
  }, [input, sending, model, conversationId, webSearch, deepResearch, effort]);
```

- [ ] **Step 3: Add the `toolLabel` helper** (bottom of file, next to `Centered`):

```typescript
function toolLabel(name: string): string {
  const map: Record<string, string> = {
    query_data: "querying the database…",
    get_player: "looking up the player…",
    compare_players: "comparing players…",
    get_my_team: "reading your roster…",
    get_standings: "checking the standings…",
    get_free_agents: "scanning free agents…",
    web_search: "searching the web…",
    deep_research: "researching…",
    request_refresh: "queuing a data refresh…",
  };
  return map[name] ?? `running ${name}…`;
}
```

- [ ] **Step 4: Swap the "thinking…" indicator to show the tool-status chip.** Replace the `{sending && (...)}` block in the transcript:

```tsx
            {sending && (
              <div className="flex items-center gap-2 text-sm text-ink-3">
                <Loader2 className="size-4 animate-spin text-heat" aria-hidden />
                {toolStatus ?? "Bubba is thinking…"}
              </div>
            )}
```

- [ ] **Step 5: Add the effort dial + toggles to the composer.** In the `{/* composer */}` block, insert a controls row above the `<div className="flex items-end gap-2">`:

```tsx
          <div className="shrink-0 border-t border-line bg-surface p-2">
            <div className="mb-2 flex items-center gap-2 text-[11px]">
              <select
                aria-label="Thinking effort"
                value={effort}
                onChange={(e) => setEffort(e.target.value as Effort)}
                className="rounded-lg border border-line bg-canvas px-2 py-1 font-semibold text-ink outline-none focus:border-heat"
              >
                <option value="off">Effort: Off</option>
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
              </select>
              <Toggle label="Web" on={webSearch} onClick={() => setWebSearch((v) => !v)} />
              <Toggle label="Research" on={deepResearch} onClick={() => setDeepResearch((v) => !v)} />
            </div>
            <div className="flex items-end gap-2">
```

- [ ] **Step 6: Add the `Toggle` component** (bottom of file, next to `IconBtn`):

```tsx
function Toggle({ label, on, onClick }: { label: string; on: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={on}
      className={cn(
        "rounded-full border px-2.5 py-1 font-semibold transition-colors",
        on ? "border-heat bg-heat/10 text-heat" : "border-line bg-canvas text-ink-3 hover:text-ink",
      )}
    >
      {label}
    </button>
  );
}
```

- [ ] **Step 7: Typecheck + build (the production gate — `web/` has no CI)**

Run (from `web/`): `pnpm exec tsc --noEmit && pnpm run lint && pnpm build`
Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add web/src/components/bubba/Bubba.tsx
git commit -m "feat(web): Bubba streaming composer — live tokens, tool chips, effort dial, toggles (Bubba B2.1)"
```

---

### Task 8: Verify, preview, review, ship

- [ ] **Step 1: Full backend suites + lint**

Run:
```bash
.venv/Scripts/python.exe -m pytest tests/api/ tests/test_ai_providers.py tests/test_ai_providers_streaming.py tests/test_ai_thinking_params.py -q
python -m ruff check api/ src/ai/ tests/api/ tests/test_ai_providers.py tests/test_ai_providers_streaming.py tests/test_ai_thinking_params.py
python -m ruff format --check api/ src/ai/
```
Expected: PASS / no findings.

- [ ] **Step 2: Re-assert the read-only tool surface guard** (parent-spec invariant — Bubba never gains a write tool)

Run: `.venv/Scripts/python.exe -m pytest tests/api/test_chat_tool_surface_readonly.py -q`
Expected: PASS.

- [ ] **Step 3: Preview the streaming UI.** Start the FastAPI backend + the web preview, sign in (Clerk dormant locally → the panel shows the `auth` state; if a Clerk dev key is set, sign in), add a provider key, and verify:
  - tokens render incrementally (not all-at-once),
  - a tool-status chip appears during a query (ask "what's my team?"),
  - the effort dial + Web/Research toggles thread into the request (check the Network panel: the `send-stream` request body carries `reasoning_effort`/`web_search`/`deep_research`).

Use the preview tools (`preview_start`, `preview_snapshot`, `preview_network`) per the project's preview workflow. If Clerk is dormant and you cannot exercise a live provider, verify the request body shape + that `Accept: text/event-stream` is sent, and capture a `preview_snapshot` of the composer (effort dial + toggles present).

- [ ] **Step 4: Code review.** Dispatch `pr-review-toolkit:code-reviewer` + `pr-review-toolkit:silent-failure-hunter` on the diff. Verify:
  - `chat()` returns the byte-identical dict (equivalence test present + green) — the live Streamlit app is provably unchanged.
  - `send_stream` never raises (pre-call + mid-stream both → `error` frame); metering precedes the `done` frame so a tail disconnect is still billed; persistence stays best-effort + logged.
  - Router stays logic-free (no `src.*` import, no arithmetic) and 401s on no Clerk identity.
  - Read-only tool surface intact.
  Apply findings.

- [ ] **Step 5: Reconcile + push.** `git pull --no-rebase` (the M4/platform track may have advanced — disjoint files, should be clean), confirm the api suite + `tests/test_ai_*` still green, then `git push origin master`. Pre-push runs the structural-invariant suite.

- [ ] **Step 6: Update docs.** Flip the B2.1 status in `CLAUDE.md` (Bubba line: "B2.1 streaming DESIGNED, build pending" → "SHIPPED") and the `project_bubba_ai_assistant` memory. Mark the spec `Status: shipped`.

---

## Self-Review

**1. Spec coverage**
- SSE streaming, every turn + tool-status events → `_chat_events` (Task 2) + `send_stream` (Task 4) + `/send-stream` (Task 5) + composer render/chips (Task 7). ✅
- Thinking-effort dial Off/Low/Med/High, per-provider, unsupported → no-op → `thinking_params_for_model` (Task 1), threaded (Tasks 2/3), UI dial (Task 7). ✅
- Web-search + Deep-research toggles (frontend; backend already honors flags) → composer toggles (Task 7) threaded into the body (Task 6). ✅
- One event core, two consumers (DRY + live-app-safe) + drain-keeps-shape + equivalence guard → Task 2 (`test_chat_is_a_drain_returning_todays_exact_dict`). ✅
- `chat()` byte-identical for the live Streamlit app → frozen-reference test + the 3 migrated behavioral tests. ✅
- `reasoning_effort` on `ChatSendRequest`, used by `/send` + `/send-stream` → Task 3 + Task 5. ✅
- Auth: `require_app_user`-gated, 401 on no Clerk, never 500 (errors → `error` frame) → Task 4/5 (`test_send_stream_unauthenticated_is_401`, error-frame tests). ✅
- Server-side metering on `done` even on disconnect → `_meter_and_persist` before the `done` frame (Task 4); `test_send_stream_meters_usage_on_done` + `test_send_stream_early_stop_before_done_does_not_meter`. ✅
- Frontend streaming `fetch` reader (apiPost buffers; new path required) → `sendStream` (Task 6). ✅
- Tool surface stays read-only (re-assert) → Task 8 Step 2. ✅
- OpenAPI regen (StreamingResponse, no response model) → Task 5 Step 5. ✅

**2. Placeholder scan** — every code step shows complete code. The only judgement calls flagged inline are the Anthropic thinking-budget numbers (4096/12288, documented as tunable) and the preview being Clerk-gated (handled with a documented fallback verification). No TBD/TODO/"add error handling".

**3. Type consistency** — `BubbaStreamEvent` (TS) mirrors `_chat_events`'s dict shapes (`text_delta.text`, `tool_started.{name,args}`, `tool_result.{name,ok}`, `done.{content,conversation_id,cost_usd,tokens_in,tokens_out,tool_trace}`, `error.message`). `reasoning_effort: Literal["off","low","medium","high"] | None` (py) ↔ `"off"|"low"|"medium"|"high"` (ts `ChatSendBody`) ↔ `Effort` (Bubba.tsx). `send_stream`/`_meter_and_persist`/`_sse`/`_rebuild`/`_tool_ok`/`thinking_params_for_model` names are used consistently across producer + tests.

**4. Deviation from spec (documented):** the spec's optional "degrade to a single `text_delta` of the full `chat()` result" fallback for a provider that can't stream is **intentionally dropped** — since `chat()` now also drains the streaming core, a true non-streaming path no longer exists, and keeping a second unverifiable path violates YAGNI. Instead a provider that cannot stream surfaces a graceful `error` frame (Task 4 `test_send_stream_engine_exception_becomes_error_frame`); the owner verifies Anthropic/OpenAI/DeepSeek streaming post-deploy (a fast-follow if one fails). This matches the spec's "never 500; errors become an error SSE event" guarantee.

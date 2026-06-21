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

"""LiteLLM wrapper + the agentic tool loop.

chat() runs: call model -> if it asks for tools, dispatch + append results + call
again -> else return the final text. Non-streamed (Phase 1); the UI fakes the
typewriter. _completion is split out so tests can mock LiteLLM with no network.
"""

from __future__ import annotations

import json

from src.ai.tools import dispatch_tool, tool_specs

_MAX_TOOL_ROUNDS = 6


def _completion(**kwargs):
    import litellm

    return litellm.completion(**kwargs)


def chat(
    model: str,
    messages: list[dict],
    api_key: str | None,
    user_id: int,
    max_tool_rounds: int = _MAX_TOOL_ROUNDS,
) -> dict:
    """Run the tool loop. Returns {content, tokens_in, tokens_out, tool_trace}."""
    convo = list(messages)
    tool_trace: list[dict] = []
    tokens_in = tokens_out = 0
    specs = tool_specs()

    for _ in range(max_tool_rounds):
        resp = _completion(model=model, messages=convo, tools=specs, api_key=api_key)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            tokens_in += int(getattr(usage, "prompt_tokens", 0) or 0)
            tokens_out += int(getattr(usage, "completion_tokens", 0) or 0)
        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        if not tool_calls:
            return {
                "content": msg.content or "",
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "tool_trace": tool_trace,
            }

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
            result = dispatch_tool(tc.function.name, args, user_id=user_id)
            tool_trace.append({"name": tc.function.name, "args": args})
            # `name` is optional on OpenAI/Anthropic tool messages but some
            # OpenAI-compatible + ollama gateways rely on it — echo it for safety.
            convo.append({"role": "tool", "tool_call_id": tc.id, "name": tc.function.name, "content": result})

    # loop cap hit
    return {
        "content": "I wasn't able to finish that within the tool-call limit. Try narrowing the question.",
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "tool_trace": tool_trace,
    }

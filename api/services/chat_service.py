"""The ONE api/ seam over src/ai (Bubba B1).

Wraps the UNCHANGED chat engine (src/ai). Mirrors the canonical send
orchestration in src/ai/chat.py::_handle_send. Never raises on a provider/key/cap
error — returns a structured error in the response body so the route stays HTTP
200, matching the Streamlit st.warning behavior.
"""

from __future__ import annotations

import json as _json
import logging
from collections.abc import Generator

from src.ai import budget, history, keys, providers
from src.ai.chat import build_system_prompt as _build_system_prompt
from src.ai.keys import list_admin_shared_providers as _list_admin_shared_providers
from src.ai.router import model_catalog as _model_catalog
from src.ai.router import price_per_token as _price_per_token
from src.ai.router import provider_of as _provider_of

# Namespacing offset: src/ai writes its ai_* tables in the SHARED draft_tool.db
# keyed by an int user_id; the live Streamlit chat already uses users.user_id
# (1..N). Offset the React/Clerk AppUser.id well above that so the two never
# collide in those shared tables. The ai_* FK->users is declared but UNENFORCED
# (get_connection sets no PRAGMA foreign_keys=ON), so a non-existent users row
# never raises. B-phase follow-up: move chat persistence into api_state.db when
# B2 enforces FKs.
_CHAT_USER_ID_OFFSET = 1_000_000_000

_DEFAULT_PAGE = "the app"

_log = logging.getLogger(__name__)


def _sse(payload: dict) -> str:
    """Format one Server-Sent-Events frame."""
    return f"data: {_json.dumps(payload)}\n\n"


def chat_user_id_for(app_user_id: int) -> int:
    """Map a Clerk AppUser.id to the namespaced src/ai chat user_id."""
    return _CHAT_USER_ID_OFFSET + int(app_user_id)


class ChatService:
    def send(
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
    ) -> dict:
        # Pre-call: key/cap/prompt + the paid provider call. A failure HERE (no
        # spend yet, or the provider itself erroring) becomes a structured error.
        try:
            provider = _provider_of(model)
            api_key = keys.get_key(chat_user_id, provider)
            if not api_key:
                return self._empty(f"No API key for {provider}. Add one in settings to chat.", conversation_id)
            on_own_key = any(k.get("provider") == provider for k in keys.list_keys(chat_user_id))
            if budget.is_over_cap(chat_user_id, on_own_key=on_own_key):
                return self._empty(
                    "You've reached today's usage limit. Add your own API key for unlimited use.",
                    conversation_id,
                )

            prior = history.load_messages(conversation_id, user_id=chat_user_id) if conversation_id else []
            convo = [{"role": "system", "content": _build_system_prompt(page or _DEFAULT_PAGE, viewer_team)}]
            convo += [{"role": m["role"], "content": m["content"]} for m in prior]
            convo.append({"role": "user", "content": message})

            result = providers.chat(
                model=model,
                messages=convo,
                api_key=api_key,
                user_id=chat_user_id,
                web_search=web_search,
                deep_research=deep_research,
                reasoning_effort=reasoning_effort,
            )
        except Exception as e:  # noqa: BLE001 - provider/key/cap failure -> body, never 500
            _log.warning("chat send failed before completion", exc_info=True)
            return self._empty(f"Chat error: {type(e).__name__}: {str(e)[:200]}", conversation_id)

        # The paid call SUCCEEDED. From here we must never discard the answer the
        # user paid for. Meter FIRST (so a persist failure can't bypass the cap),
        # then best-effort persist — each isolated + logged, never fatal.
        content = result.get("content", "")
        t_in = int(result.get("tokens_in", 0) or 0)
        t_out = int(result.get("tokens_out", 0) or 0)
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
                content,
                model=model,
                tokens_in=t_in,
                tokens_out=t_out,
                cost_usd=cost,
            )
        except Exception:
            # Persistence is best-effort: a save failure must not discard the answer.
            # (B-phase follow-up: an atomic history.append_exchange to avoid an
            # orphaned user turn if the assistant insert is the one that fails.)
            _log.warning("chat history persist failed (answer returned, not saved)", exc_info=True)

        return {
            "content": content,
            "conversation_id": conversation_id,
            "tokens_in": t_in,
            "tokens_out": t_out,
            "cost_usd": cost,
            "tool_trace": result.get("tool_trace", []),
            "error": None,
        }

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
        frame (the HTTP response is already 200).

        Billing caveat: only a COMPLETED answer (one that reaches `done`) is
        metered. If the engine raises mid-stream AFTER some tokens were produced,
        those tokens are not billed (a documented under-count — surfacing partial
        usage through the generator is a follow-up; see B2.1 plan Finding 2)."""
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
                cost, conversation_id = self._meter_and_persist(chat_user_id, model, message, event, conversation_id)
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
            # Partial-answer tokens already produced this turn are NOT metered
            # (documented under-count, see docstring). Log loudly so an operator
            # sees it rather than mistaking it for a no-spend failure.
            _log.warning("chat send_stream failed mid-stream (partial usage uncounted)", exc_info=True)
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

    def _empty(self, error: str, conversation_id: int | None = None) -> dict:
        return {
            "content": "",
            "conversation_id": conversation_id,
            "tokens_in": 0,
            "tokens_out": 0,
            "cost_usd": 0.0,
            "tool_trace": [],
            "error": error,
        }

    # Reads degrade to empty (never 500 a read), but LOG the error so a real DB
    # outage is visible in the operator log instead of looking like "no data".
    def conversations(self, chat_user_id: int) -> list[dict]:
        try:
            return history.list_conversations(chat_user_id)
        except Exception:
            _log.warning("conversations read failed", exc_info=True)
            return []

    def messages(self, chat_user_id: int, conversation_id: int) -> list[dict]:
        try:
            return history.load_messages(conversation_id, user_id=chat_user_id)
        except Exception:
            _log.warning("messages read failed", exc_info=True)
            return []

    def models(self, chat_user_id: int) -> list[dict]:
        try:
            available = {k.get("provider") for k in keys.list_keys(chat_user_id)} | set(_list_admin_shared_providers())
            return [
                {"id": m, "label": label, "provider": _provider_of(m)}
                for label, m in _model_catalog()
                if _provider_of(m) in available
            ]
        except Exception:
            _log.warning("models read failed", exc_info=True)
            return []

    def list_keys(self, chat_user_id: int) -> list[dict]:
        try:
            return keys.list_keys(chat_user_id)
        except Exception:
            _log.warning("list_keys read failed", exc_info=True)
            return []

    def store_key(self, chat_user_id: int, provider: str, api_key: str, label: str | None) -> tuple[bool, str]:
        try:
            keys.store_key(chat_user_id, provider, api_key, label=label)
            return True, "Key saved."
        except Exception as e:  # noqa: BLE001
            _log.warning("store_key failed", exc_info=True)
            return False, f"{type(e).__name__}: {str(e)[:200]}"

    def delete_key(self, chat_user_id: int, provider: str, label: str | None) -> tuple[bool, str]:
        try:
            keys.delete_key(chat_user_id, provider, label=label)
            return True, "Key removed."
        except Exception as e:  # noqa: BLE001
            _log.warning("delete_key failed", exc_info=True)
            return False, f"{type(e).__name__}: {str(e)[:200]}"

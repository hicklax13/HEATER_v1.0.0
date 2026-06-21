"""The ONE api/ seam over src/ai (Bubba B1).

Wraps the UNCHANGED chat engine (src/ai). Mirrors the canonical send
orchestration in src/ai/chat.py::_handle_send. Never raises on a provider/key/cap
error — returns a structured error in the response body so the route stays HTTP
200, matching the Streamlit st.warning behavior.
"""

from __future__ import annotations

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
        page: str | None = None,
        viewer_team: str | None = None,
    ) -> dict:
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
            )
            content = result.get("content", "")
            t_in = int(result.get("tokens_in", 0) or 0)
            t_out = int(result.get("tokens_out", 0) or 0)
            p_in, p_out = _price_per_token(model)
            cost = t_in * p_in + t_out * p_out

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
            budget.record_usage(chat_user_id, t_in, t_out, cost)
            return {
                "content": content,
                "conversation_id": conversation_id,
                "tokens_in": t_in,
                "tokens_out": t_out,
                "cost_usd": cost,
                "tool_trace": result.get("tool_trace", []),
                "error": None,
            }
        except Exception as e:  # noqa: BLE001 - chat must never 500; surface as body
            return self._empty(f"Chat error: {type(e).__name__}: {str(e)[:200]}", conversation_id)

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
            available = {k.get("provider") for k in keys.list_keys(chat_user_id)} | set(_list_admin_shared_providers())
            return [
                {"id": m, "label": label, "provider": _provider_of(m)}
                for label, m in _model_catalog()
                if _provider_of(m) in available
            ]
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

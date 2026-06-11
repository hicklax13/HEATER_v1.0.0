"""The chat widget: floating window contents + the send/stream orchestration.

MULTI_USER-gated and inert when off (v1 byte-for-byte). The conversation lives in
session_state (survives page navigation) and is written through to SQLite.
"""

from __future__ import annotations

import time

import streamlit as st

from src.ai import budget, history
from src.ai.chat_shell import CONTAINER_ID, render_launcher_and_shell
from src.ai.keys import delete_key, get_key, list_keys, store_key
from src.ai.router import model_for_tier, price_per_token, provider_of
from src.ai.schema_card import build_schema_card
from src.auth import current_user, multi_user_enabled, resolve_viewer_team_name

_STATE_MSGS = "ai_chat_messages"
_STATE_CONV = "ai_chat_conversation_id"
_TIER_LABEL = {"simple": "Simple", "moderate": "Moderate", "complex": "Complex"}


def build_system_prompt(page: str, viewer_team: str | None) -> str:
    schema = build_schema_card()
    team = viewer_team or "the user's team"
    return (
        "You are HEATER's in-app fantasy-baseball assistant for a 12-team Yahoo H2H "
        "categories league (R,HR,RBI,SB,AVG,OBP / W,L,SV,K,ERA,WHIP; L/ERA/WHIP are "
        "inverse). Present data and analysis; do not give unsolicited personal trade "
        f"opinions — it is the user's team. The viewer's team is '{team}'. The user is "
        f"currently on the '{page}' page.\n\n"
        "You can call tools to read any app data. Prefer the specific tools; use "
        "query_data (read-only SELECT) for anything else. Only request_refresh when the "
        "user explicitly asks for fresh data.\n\n" + schema
    )


def _ensure_state() -> None:
    if _STATE_MSGS not in st.session_state:
        st.session_state[_STATE_MSGS] = []
    if _STATE_CONV not in st.session_state:
        st.session_state[_STATE_CONV] = None


def render_chat_widget(page: str) -> None:
    """Mount the floating AI chat. No-op unless MULTI_USER + logged in."""
    if not multi_user_enabled():
        return
    user = current_user()
    if user is None:
        return

    _ensure_state()
    from streamlit_float import float_init

    float_init()
    render_launcher_and_shell()

    from streamlit_float import float_parent

    window = st.container()
    with window:
        st.markdown(f'<div id="{CONTAINER_ID}-anchor"></div>', unsafe_allow_html=True)
        _render_window_body(page, user)
        float_parent()


def _render_window_body(page: str, user: dict) -> None:
    st.markdown(
        f'<div id="{CONTAINER_ID}-header" style="display:flex;justify-content:space-between;'
        'align-items:center;padding:8px 10px;border-bottom:1px solid #eee;">'
        "<strong>HEATER AI</strong><span>"
        '<button data-ai-act="minimize" style="border:none;background:none;cursor:pointer;">_</button>'
        '<button data-ai-act="close" style="border:none;background:none;cursor:pointer;">x</button>'
        "</span></div>",
        unsafe_allow_html=True,
    )
    _render_chat_fragment(page, user)


@st.fragment
def _render_chat_fragment(page: str, user: dict) -> None:
    cols = st.columns([3, 1])
    convos = history.list_conversations(user["user_id"])
    options = ["New conversation"] + [c["title"] for c in convos]
    pick = cols[0].selectbox("Conversations", options, label_visibility="collapsed", key="ai_conv_pick")
    if cols[1].button("+", key="ai_new_chat"):
        st.session_state[_STATE_MSGS] = []
        st.session_state[_STATE_CONV] = None
    if pick != "New conversation":
        chosen = convos[options.index(pick) - 1]
        if st.session_state[_STATE_CONV] != chosen["id"]:
            st.session_state[_STATE_CONV] = chosen["id"]
            st.session_state[_STATE_MSGS] = [
                {"role": m["role"], "content": m["content"]}
                for m in history.load_messages(chosen["id"], user_id=user["user_id"])
            ]

    _render_ai_settings(user)

    for m in st.session_state[_STATE_MSGS]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    tier = st.selectbox(
        "Model",
        ["simple", "moderate", "complex"],
        index=1,
        key="ai_tier",
        format_func=lambda t: _TIER_LABEL[t],
        label_visibility="collapsed",
    )
    prompt = st.chat_input("Ask anything about your league...", key="ai_prompt")
    if prompt:
        _handle_send(prompt, tier, page, user)


def _handle_send(prompt: str, tier: str, page: str, user: dict) -> None:
    from src.ai.providers import chat as provider_chat

    uid = user["user_id"]
    model = model_for_tier(tier)
    provider = provider_of(model)
    api_key = get_key(uid, provider)
    on_own_key = api_key is not None and _is_user_own_key(uid, provider)

    if api_key is None:
        st.warning(f"No API key for {provider}. Add one in AI Settings, or ask the admin to set a shared key.")
        return
    if budget.is_over_cap(uid, on_own_key=on_own_key):
        st.warning("You've hit your AI usage limit for today. Try again tomorrow or add your own key.")
        return

    st.session_state[_STATE_MSGS].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # No rosters frame is passed: under MULTI_USER the resolver returns the
    # session user's admin-assigned team from identity (not a frame match), and
    # here the team name is only an informational label in the system prompt —
    # it is never used as a DataFrame match key, so the 2026-06-05 frame-guard
    # (a standings-matching concern) does not apply.
    viewer_team = resolve_viewer_team_name()
    system = build_system_prompt(page, viewer_team)
    messages = [{"role": "system", "content": system}] + st.session_state[_STATE_MSGS]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = provider_chat(model=model, messages=messages, api_key=api_key, user_id=uid)
        st.write_stream(_typewriter(result["content"]))

    st.session_state[_STATE_MSGS].append({"role": "assistant", "content": result["content"]})

    pin, pout = price_per_token(model)
    cost = result["tokens_in"] * pin + result["tokens_out"] * pout
    budget.record_usage(uid, result["tokens_in"], result["tokens_out"], cost)

    cid = st.session_state[_STATE_CONV]
    if cid is None:
        cid = history.create_conversation(uid, title=prompt[:60], model=model)
        st.session_state[_STATE_CONV] = cid
    history.append_message(cid, "user", prompt)
    history.append_message(
        cid,
        "assistant",
        result["content"],
        model=model,
        tokens_in=result["tokens_in"],
        tokens_out=result["tokens_out"],
        cost_usd=cost,
    )


def _typewriter(text: str):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.01)


def _is_user_own_key(user_id: int, provider: str) -> bool:
    return any(k["provider"] == provider for k in list_keys(user_id))


def _render_ai_settings(user: dict) -> None:
    """Gear popover: each user adds / labels / removes their OWN provider keys.

    Keys are Fernet-encrypted by store_key; this UI never shows a stored key back.
    """
    uid = user["user_id"]
    with st.popover("AI Settings", help="Your API keys"):
        st.caption("Add your own provider key to use your own model + skip shared-key caps.")
        with st.form("ai_key_form", clear_on_submit=True):
            provider = st.selectbox("Provider", ["anthropic", "openai", "gemini", "openrouter", "ollama"])
            label = st.text_input("Label (optional)", value="")
            key_text = st.text_input("API key", value="", type="password")
            if st.form_submit_button("Save key") and key_text.strip():
                store_key(uid, provider, key_text.strip(), label=(label.strip() or None))
                st.success(f"Saved your {provider} key.")
        existing = list_keys(uid)
        if existing:
            st.caption("Your keys:")
            for i, k in enumerate(existing):
                row = st.columns([3, 1])
                row[0].write(f"{k['provider']}" + (f" - {k['label']}" if k["label"] else ""))
                # row index in the key keeps it unique even for two same-(provider,label) rows
                if row[1].button("Remove", key=f"del_{i}_{k['provider']}_{k['label']}"):
                    delete_key(uid, k["provider"], k["label"])
                    st.rerun()

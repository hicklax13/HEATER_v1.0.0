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
from src.ai.router import model_catalog, model_for_tier, price_per_token, provider_of
from src.ai.schema_card import build_schema_card
from src.auth import current_user, multi_user_enabled, resolve_viewer_team_name

_STATE_MSGS = "ai_chat_messages"
_STATE_CONV = "ai_chat_conversation_id"
_STATE_ATTACHED = "ai_chat_attached"
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

    options, resolver = _model_picker_options()
    pick = st.selectbox("Model", options, index=1, key="ai_model_pick", label_visibility="collapsed")
    model = _resolve_model(resolver[pick])

    # Tool toggles: web search + deep research are off by default (cost/latency).
    tcols = st.columns(3)
    web_search = tcols[0].toggle("Web", value=False, key="ai_web_search", help="Let the AI search the web")
    deep_research = tcols[1].toggle(
        "Research", value=False, key="ai_deep_research", help="Deeper web research (search + read pages)"
    )
    attach_mode = tcols[2].toggle(
        "Attach", value=False, key="ai_attach_mode", help="Attach text you highlight on the page"
    )

    _render_attach_controls(attach_mode)

    prompt = st.chat_input("Ask anything about your league...", key="ai_prompt")
    if prompt:
        _handle_send(prompt, model, page, user, web_search=web_search, deep_research=deep_research)


def _render_attach_controls(attach_mode: bool) -> None:
    """When attach mode is on: a button that captures the user's on-page text
    selection (via streamlit-js-eval reading window.parent.getSelection) and stores
    it as context for the next message; plus a chip to review/clear it."""
    if not attach_mode:
        return
    if st.button("Attach highlighted text", key="ai_attach_btn"):
        st.session_state["_ai_attach_n"] = st.session_state.get("_ai_attach_n", 0) + 1
    n = st.session_state.get("_ai_attach_n", 0)
    if n:
        try:
            from streamlit_js_eval import streamlit_js_eval

            sel = streamlit_js_eval(
                js_expressions="window.parent.getSelection ? window.parent.getSelection().toString() : ''",
                key=f"ai_getsel_{n}",
                want_output=True,
            )
        except Exception:
            sel = None
        if sel and str(sel).strip():
            st.session_state[_STATE_ATTACHED] = str(sel).strip()[:4000]
    attached = st.session_state.get(_STATE_ATTACHED)
    if attached:
        st.caption(f"Attached ({len(attached)} chars): {attached[:80]}...")
        if st.button("Clear attachment", key="ai_attach_clear"):
            st.session_state[_STATE_ATTACHED] = None


def _model_picker_options() -> tuple[list[str], dict]:
    """Picker labels + a resolver: label -> ('tier', name) or ('model', model_string).

    Offers the 3 tier autos (which route via the admin tier map) plus every
    specific model in the catalog (DeepSeek V4 Flash/Pro, Claude tiers).
    """
    options: list[str] = []
    resolver: dict[str, tuple[str, str]] = {}
    for t in ("simple", "moderate", "complex"):
        label = f"Auto - {_TIER_LABEL[t]}"
        options.append(label)
        resolver[label] = ("tier", t)
    for label, m in model_catalog():
        options.append(label)
        resolver[label] = ("model", m)
    return options, resolver


def _resolve_model(selection: tuple[str, str]) -> str:
    kind, value = selection
    return model_for_tier(value) if kind == "tier" else value


def _handle_send(
    prompt: str, model: str, page: str, user: dict, web_search: bool = False, deep_research: bool = False
) -> None:
    from src.ai.providers import chat as provider_chat

    uid = user["user_id"]
    provider = provider_of(model)
    api_key = get_key(uid, provider)
    on_own_key = api_key is not None and _is_user_own_key(uid, provider)

    if api_key is None:
        st.warning(f"No API key for {provider}. Add one in AI Settings, or ask the admin to set a shared key.")
        return
    if budget.is_over_cap(uid, on_own_key=on_own_key):
        st.warning("You've hit your AI usage limit for today. Try again tomorrow or add your own key.")
        return

    # Prepend any page text the user attached as explicit context for THIS message.
    attached = st.session_state.get(_STATE_ATTACHED)
    sent = prompt if not attached else f"[Context the user selected on the page]\n{attached}\n\n[Question]\n{prompt}"

    st.session_state[_STATE_MSGS].append({"role": "user", "content": sent})
    with st.chat_message("user"):
        st.markdown(prompt if not attached else f"_(with attached selection)_\n\n{prompt}")

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
            result = provider_chat(
                model=model,
                messages=messages,
                api_key=api_key,
                user_id=uid,
                web_search=web_search,
                deep_research=deep_research,
            )
        st.write_stream(_typewriter(result["content"]))

    st.session_state[_STATE_ATTACHED] = None  # consume the attachment

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
            provider = st.selectbox("Provider", ["deepseek", "anthropic", "openai", "gemini", "openrouter", "ollama"])
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

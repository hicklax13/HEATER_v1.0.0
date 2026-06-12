"""The chat widget: floating window contents + the send/stream orchestration.

MULTI_USER-gated and inert when off (v1 byte-for-byte). The conversation lives in
session_state (survives page navigation) and is written through to SQLite.
"""

from __future__ import annotations

import time

import streamlit as st

from src.ai import budget, history
from src.ai.chat_shell import CONTAINER_ID, render_launcher_and_shell, window_frame_css
from src.ai.keys import delete_key, get_key, list_admin_shared_providers, list_keys, store_key
from src.ai.router import model_catalog, price_per_token, provider_of
from src.ai.schema_card import build_schema_card
from src.auth import current_user, multi_user_enabled, resolve_viewer_team_name

_STATE_MSGS = "ai_chat_messages"
_STATE_CONV = "ai_chat_conversation_id"
_STATE_ATTACHED = "ai_chat_attached"


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
        # Scope the frame to THIS container only (float_parent tags it with a unique
        # class). A global :has() frame rule would over-match ancestors — see
        # window_frame_css() — and pin the whole page, blanking the main column.
        float_parent(css=window_frame_css())


def _render_window_body(page: str, user: dict) -> None:
    # Navy/orange title bar — consistent with the app's panel headers (Combustion).
    st.markdown(
        f'<div id="{CONTAINER_ID}-header" style="display:flex;justify-content:space-between;'
        "align-items:center;padding:9px 12px;margin-bottom:6px;background:#112744;color:#fff;"
        "border-radius:9px;cursor:move;user-select:none;"
        'font-family:Archivo,system-ui,sans-serif;font-weight:700;letter-spacing:.3px;">'
        '<span>HEATER <span style="color:#ff6d00;">AI</span></span>'
        '<span style="display:flex;gap:2px;">'
        '<button data-ai-act="minimize" title="Minimize" style="border:none;background:none;'
        'color:#cfd6e2;cursor:pointer;font-size:18px;line-height:1;padding:0 6px;">_</button>'
        '<button data-ai-act="close" title="Close" style="border:none;background:none;'
        'color:#cfd6e2;cursor:pointer;font-size:15px;line-height:1;padding:0 6px;">x</button>'
        "</span></div>",
        unsafe_allow_html=True,
    )
    _render_chat_fragment(page, user)


@st.fragment
def _render_chat_fragment(page: str, user: dict) -> None:
    # ---- top controls: conversation history + new ----
    cols = st.columns([3, 1])
    convos = history.list_conversations(user["user_id"])
    conv_options = ["New conversation"] + [c["title"] for c in convos]
    conv_pick = cols[0].selectbox("Conversations", conv_options, label_visibility="collapsed", key="ai_conv_pick")
    if cols[1].button("+", key="ai_new_chat", help="Start a new conversation"):
        st.session_state[_STATE_MSGS] = []
        st.session_state[_STATE_CONV] = None
    if conv_pick != "New conversation":
        chosen = convos[conv_options.index(conv_pick) - 1]
        if st.session_state[_STATE_CONV] != chosen["id"]:
            st.session_state[_STATE_CONV] = chosen["id"]
            st.session_state[_STATE_MSGS] = [
                {"role": m["role"], "content": m["content"]}
                for m in history.load_messages(chosen["id"], user_id=user["user_id"])
            ]

    # ---- model picker: concrete models for providers that have a key ----
    available = _available_providers(user["user_id"]) or {provider_of(m) for _, m in model_catalog()}
    model_options, resolver = _model_picker_options(available)
    model_pick = st.selectbox("Model", model_options, index=0, key="ai_model_pick2", label_visibility="collapsed")
    model = resolver[model_pick]
    _render_ai_settings(user)

    # ---- tool toggles (off by default: cost/latency) ----
    tcols = st.columns(3)
    web_search = tcols[0].toggle("Web", value=False, key="ai_web_search", help="Let the AI search the web")
    deep_research = tcols[1].toggle(
        "Research", value=False, key="ai_deep_research", help="Deeper web research (search + read pages)"
    )
    attach_mode = tcols[2].toggle(
        "Attach", value=False, key="ai_attach_mode", help="Attach text you highlight on the page"
    )
    _render_attach_controls(attach_mode)

    # ---- transcript (scrolls); the input stays pinned at the window bottom ----
    messages = st.session_state[_STATE_MSGS]
    transcript = st.container(height=240)
    with transcript:
        for m in messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

    prompt = st.chat_input("Ask anything about your league...", key="ai_prompt")
    if prompt:
        _handle_send(prompt, model, page, user, transcript, web_search=web_search, deep_research=deep_research)


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


def _available_providers(user_id: int) -> set[str]:
    """Providers the viewer can actually use: their own keys + the admin shared keys."""
    own = {k["provider"] for k in list_keys(user_id)}
    shared = set(list_admin_shared_providers())
    return own | shared


def _model_picker_options(available_providers: set[str]) -> tuple[list[str], dict]:
    """Picker labels + a resolver mapping label -> model_string.

    Lists every catalog model whose provider has a usable key (the viewer's own
    key or the admin shared key). The Simple/Moderate/Complex tier autos are NOT
    offered — members pick a concrete model directly.
    """
    options: list[str] = []
    resolver: dict[str, str] = {}
    for label, m in model_catalog():
        if provider_of(m) in available_providers:
            options.append(label)
            resolver[label] = m
    return options, resolver


def _handle_send(
    prompt: str,
    model: str,
    page: str,
    user: dict,
    transcript,
    web_search: bool = False,
    deep_research: bool = False,
) -> None:
    from src.ai.providers import chat as provider_chat

    uid = user["user_id"]
    provider = provider_of(model)
    api_key = get_key(uid, provider)
    on_own_key = api_key is not None and _is_user_own_key(uid, provider)

    if api_key is None:
        with transcript:
            st.warning(f"No API key for {provider}. Add one in AI Settings, or ask the admin to set a shared key.")
        return
    if budget.is_over_cap(uid, on_own_key=on_own_key):
        with transcript:
            st.warning("You've hit your AI usage limit for today. Try again tomorrow or add your own key.")
        return

    # Prepend any page text the user attached as explicit context for THIS message.
    attached = st.session_state.get(_STATE_ATTACHED)
    sent = prompt if not attached else f"[Context the user selected on the page]\n{attached}\n\n[Question]\n{prompt}"

    st.session_state[_STATE_MSGS].append({"role": "user", "content": sent})

    # No rosters frame is passed: under MULTI_USER the resolver returns the
    # session user's admin-assigned team from identity (not a frame match), and
    # here the team name is only an informational label in the system prompt —
    # it is never used as a DataFrame match key, so the 2026-06-05 frame-guard
    # (a standings-matching concern) does not apply.
    viewer_team = resolve_viewer_team_name()
    system = build_system_prompt(page, viewer_team)
    messages = [{"role": "system", "content": system}] + st.session_state[_STATE_MSGS]

    # Render the new exchange INSIDE the transcript so it lands in the scroll area
    # (not below the input). Re-entering the container appends to it.
    with transcript:
        with st.chat_message("user"):
            st.markdown(prompt if not attached else f"_(with attached selection)_\n\n{prompt}")
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
            provider = st.selectbox(
                "Provider", ["deepseek", "anthropic", "openai", "gemini", "xai", "openrouter", "ollama"]
            )
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

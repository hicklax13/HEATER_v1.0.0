"""Admin-only operational controls (nav-routed; MULTI_USER only).

The leading underscore hides this file from v1 auto-discovery; src/nav.py routes
here via st.Page for admins when MULTI_USER is on. No set_page_config — app.py
owns the single call under st.navigation().
"""

import streamlit as st

from src.ai.budget import daily_cap_usd, set_daily_cap
from src.ai.keys import set_admin_shared_key
from src.ai.router import model_catalog, model_for_tier, set_tier_models
from src.app_settings import get_broadcast, get_maintenance, set_broadcast, set_maintenance
from src.audit import list_audit, log_action
from src.auth import current_user, enter_view_as, require_admin
from src.feature_flags import list_page_flags, set_page_flag
from src.feedback import feedback_csv
from src.nav import PAGE_REGISTRY
from src.ui_shared import inject_custom_css
from src.usage import per_user_activity, usage_csv
from src.yahoo_api import save_yahoo_token_json

inject_custom_css()
require_admin()

_admin_id = (current_user() or {}).get("user_id", 0)

st.title("Admin Controls")

# --- Page visibility -------------------------------------------------------
st.subheader("Page visibility")
st.caption("Disabled pages vanish from non-admin navigation. Admins always see every page.")
_flags = list_page_flags()
for _entry in PAGE_REGISTRY:
    _flag_key = "page:" + _entry["key"]
    _current = _flags.get(_flag_key, True)
    _new = st.toggle(_entry["title"], value=_current, key="flag_" + _entry["key"])
    if _new != _current:
        set_page_flag(_flag_key, _new, admin_id=_admin_id)
        st.rerun()

# --- Broadcast banner ------------------------------------------------------
st.subheader("Broadcast banner")
_bc = get_broadcast()
_bc_msg = st.text_input("Broadcast message", value=_bc["message"], key="bc_msg")
_bc_on = st.checkbox("Show broadcast to all users", value=_bc["enabled"], key="bc_on")
if st.button("Save broadcast"):
    set_broadcast(_bc_on, _bc_msg, admin_id=_admin_id)
    st.success("Broadcast saved.")

# --- Maintenance mode ------------------------------------------------------
st.subheader("Maintenance mode")
_mt = get_maintenance()
_mt_msg = st.text_input("Maintenance message", value=_mt["message"], key="mt_msg")
_mt_on = st.toggle("Enable maintenance mode", value=_mt["enabled"], key="mt_on")
if st.button("Save maintenance"):
    set_maintenance(_mt_on, _mt_msg, admin_id=_admin_id)
    st.success("Maintenance setting saved.")

# --- View as user ----------------------------------------------------------
st.subheader("View as user")
_usernames = [r["username"] for r in per_user_activity()]
if _usernames:
    _target = st.selectbox("Impersonate user", _usernames, key="view_as_target")
    if st.button("Enter view-as"):
        enter_view_as(_target, admin_id=_admin_id)
        st.rerun()
else:
    st.caption("No users to impersonate yet.")

# --- Exports ---------------------------------------------------------------
st.subheader("Exports")
st.download_button(
    "Download usage CSV",
    data=usage_csv(),
    file_name="heater_usage.csv",
    mime="text/csv",
    on_click=lambda: log_action(_admin_id, "export_csv", target="usage"),
)
st.download_button(
    "Download feedback CSV",
    data=feedback_csv(),
    file_name="heater_feedback.csv",
    mime="text/csv",
    on_click=lambda: log_action(_admin_id, "export_csv", target="feedback"),
)

# --- Yahoo token -----------------------------------------------------------
st.subheader("Yahoo token")
st.caption(
    "Paste the contents of a locally-generated data/yahoo_token.json to seed "
    "headless Yahoo reconnect on this server. Stored on the persistent volume; "
    "never displayed back."
)
_yahoo_token_text = st.text_area("yahoo_token.json contents", value="", key="yahoo_token_paste", height=160)
if st.button("Save Yahoo token"):
    _ok, _msg = save_yahoo_token_json(_yahoo_token_text)
    if _ok:
        log_action(_admin_id, "yahoo_token_update")
        st.success(_msg)
    else:
        st.error(_msg)

# --- AI chat assistant -----------------------------------------------------
st.subheader("AI chat assistant")
st.caption(
    "Set a shared provider key so every member can chat without their own key, "
    "and cap per-user daily spend. The key is encrypted at rest and never shown back."
)

_ai_provider = st.selectbox(
    "Shared key provider",
    ["deepseek", "anthropic", "openai", "gemini", "openrouter"],
    key="ai_shared_provider",
)
ai_shared_key_text = st.text_input("Shared API key", value="", type="password", key="ai_shared_key_input")
if st.button("Save shared key"):
    if ai_shared_key_text.strip():
        set_admin_shared_key(_ai_provider, ai_shared_key_text.strip(), admin_id=_admin_id)
        log_action(_admin_id, "ai_shared_key_update", target=_ai_provider)
        st.success(f"Shared {_ai_provider} key saved.")
    else:
        st.error("Enter a key first.")

_cap = st.number_input("Per-user daily cap (USD)", min_value=0.0, value=float(daily_cap_usd()), step=0.25, key="ai_cap")
if st.button("Save daily cap"):
    set_daily_cap(_cap, admin_id=_admin_id)
    log_action(_admin_id, "ai_daily_cap_update", detail={"usd": _cap})
    st.success(f"Daily cap set to ${_cap:.2f}.")

# Tier -> model mapping (which model the Simple/Moderate/Complex autos use).
st.caption("Default model per task tier (members can still pick a specific model in the chat window).")
_catalog = model_catalog()
_model_labels = [label for label, _ in _catalog]
_model_by_label = dict(_catalog)
_label_by_model = {m: label for label, m in _catalog}
_tier_cols = st.columns(3)
_tier_choice = {}
for _i, _tier in enumerate(("simple", "moderate", "complex")):
    _cur_model = model_for_tier(_tier)
    _cur_label = _label_by_model.get(_cur_model, _model_labels[0])
    _tier_choice[_tier] = _tier_cols[_i].selectbox(
        _tier.capitalize(),
        _model_labels,
        index=_model_labels.index(_cur_label) if _cur_label in _model_labels else 0,
        key=f"ai_tier_{_tier}",
    )
if st.button("Save tier models"):
    set_tier_models({t: _model_by_label[lbl] for t, lbl in _tier_choice.items()}, admin_id=_admin_id)
    log_action(_admin_id, "ai_tier_models_update")
    st.success("Tier models saved.")

# --- Audit log -------------------------------------------------------------
st.subheader("Audit log")
st.dataframe(list_audit(limit=200), width="stretch")

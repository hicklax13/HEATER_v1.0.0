"""Admin-only operational controls (nav-routed; MULTI_USER only).

The leading underscore hides this file from v1 auto-discovery; src/nav.py routes
here via st.Page for admins when MULTI_USER is on. No set_page_config — app.py
owns the single call under st.navigation().
"""

import streamlit as st

from src.app_settings import get_broadcast, get_maintenance, set_broadcast, set_maintenance
from src.audit import list_audit, log_action
from src.auth import current_user, enter_view_as, require_admin
from src.feature_flags import list_page_flags, set_page_flag
from src.feedback import feedback_csv
from src.nav import PAGE_REGISTRY
from src.ui_shared import inject_custom_css
from src.usage import per_user_activity, usage_csv

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

# --- Audit log -------------------------------------------------------------
st.subheader("Audit log")
st.dataframe(list_audit(limit=200), width="stretch")

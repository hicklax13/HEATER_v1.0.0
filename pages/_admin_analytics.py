"""Admin-only usage-analytics surfaces (nav-routed; MULTI_USER only).

The leading underscore keeps this file out of Streamlit's automatic pages/
discovery; src/nav.py routes here explicitly via st.Page when MULTI_USER is on
and the viewer is an admin. No set_page_config — app.py owns the single call
under st.navigation().
"""

import pandas as pd
import streamlit as st

from src.auth import require_admin
from src.ui_shared import inject_custom_css
from src.usage import (
    dau_series,
    last_seen_summary,
    most_used_pages,
    page_dwell_summary,
    per_user_activity,
    session_timeline,
    usage_csv,
)

inject_custom_css()
require_admin()

st.title("Usage Analytics")

st.subheader("Daily active users (last 30 days)")
_dau = dau_series()
if _dau:
    _df = pd.DataFrame(_dau).set_index("day")
    st.line_chart(_df["users"])
else:
    st.caption("No usage events recorded yet.")

st.subheader("Most-used pages (last 30 days)")
st.dataframe(most_used_pages(), width="stretch")

st.subheader("Per-user activity")
st.dataframe(per_user_activity(), width="stretch")

st.subheader("Session timeline")
st.dataframe(session_timeline(), width="stretch")

st.subheader("Page dwell")
st.dataframe(page_dwell_summary(), width="stretch")

st.subheader("Last seen")
st.dataframe(last_seen_summary(), width="stretch")

st.download_button(
    "Download usage CSV",
    data=usage_csv(),
    file_name="heater_usage.csv",
    mime="text/csv",
)

"""Admin Console — account lifecycle (v2 multi-user foundation).

Approve pending registrations and assign each user a Yahoo team, reassign
teams, and revoke access. Gated by require_admin(): non-admins are hard-stopped.
The richer admin dashboard (feature flags, usage analytics, feedback inbox)
arrives in a later plan.
"""

import streamlit as st

from src.auth import (
    approve_user,
    get_league_team_names,
    list_users,
    require_admin,
    revoke_user,
    set_user_team,
)
from src.ui_shared import inject_custom_css

st.set_page_config(
    page_title="Heater | Admin Console",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_custom_css()
require_admin()

st.title("Admin Console")

team_names = get_league_team_names()

# ── Pending approvals ────────────────────────────────────────────────
st.header("Pending approvals")
pending = list_users(status="pending")
if not pending:
    st.info("No pending registrations.")
else:
    for user in pending:
        cols = st.columns([3, 3, 2])
        with cols[0]:
            label = user["display_name"] or user["username"]
            st.write(f"**{label}**  \n`{user['username']}` · registered {user['created_at'][:10]}")
        with cols[1]:
            if team_names:
                team = st.selectbox(
                    "Assign team",
                    options=team_names,
                    key=f"team_{user['username']}",
                )
            else:
                team = st.text_input(
                    "Assign team (no league_teams found)",
                    key=f"team_{user['username']}",
                )
        with cols[2]:
            if st.button("Approve", key=f"approve_{user['username']}", width="stretch"):
                approve_user(user["username"], team_name=team, approved_by="admin")
                st.rerun()

# ── Active users ─────────────────────────────────────────────────────
st.header("Active users")
active = list_users(status="active")
if not active:
    st.info("No active users yet.")
else:
    for user in active:
        cols = st.columns([3, 3, 2])
        with cols[0]:
            label = user["display_name"] or user["username"]
            admin_tag = " · admin" if user["is_admin"] else ""
            st.write(f"**{label}**  \n`{user['username']}`{admin_tag} · {user['team_name'] or '—'}")
        with cols[1]:
            if team_names:
                idx = team_names.index(user["team_name"]) if user["team_name"] in team_names else 0
                new_team = st.selectbox(
                    "Reassign team",
                    options=team_names,
                    index=idx,
                    key=f"reassign_{user['username']}",
                )
                if new_team != user["team_name"] and st.button("Save team", key=f"save_{user['username']}"):
                    set_user_team(user["username"], new_team)
                    st.rerun()
        with cols[2]:
            if not user["is_admin"] and st.button("Revoke", key=f"revoke_{user['username']}", width="stretch"):
                revoke_user(user["username"])
                st.rerun()

# ── Revoked users ────────────────────────────────────────────────────
revoked = list_users(status="revoked")
if revoked:
    st.header("Revoked users")
    for user in revoked:
        cols = st.columns([6, 2])
        with cols[0]:
            st.write(f"`{user['username']}` · {user['display_name'] or '—'}")
        with cols[1]:
            if team_names and st.button("Re-approve", key=f"reapprove_{user['username']}"):
                approve_user(user["username"], team_name=user["team_name"] or team_names[0], approved_by="admin")
                st.rerun()

"""Admin Console — account lifecycle + feedback inbox (v2 multi-user).

Two tabs:
  - Users: approve pending registrations + assign Yahoo teams, reassign, revoke.
  - Feedback: triage per-feature feedback (status + admin notes + data-state).

Gated by require_admin(): non-admins are hard-stopped. The richer admin
dashboard (feature flags, usage analytics) arrives in a later plan.
"""

import json

import streamlit as st

from src.auth import (
    approve_user,
    get_league_team_names,
    list_users,
    require_admin,
    revoke_user,
    set_user_team,
)
from src.feedback import list_feedback, set_feedback_notes, set_feedback_status
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

users_tab, feedback_tab = st.tabs(["Users", "Feedback"])

with users_tab:
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
                    approve_user(
                        user["username"],
                        team_name=user["team_name"] or team_names[0],
                        approved_by="admin",
                    )
                    st.rerun()

with feedback_tab:
    st.header("Feedback inbox")
    status_filter = st.selectbox(
        "Filter by status",
        options=["all", "new", "triaged", "resolved"],
        key="feedback_status_filter",
    )
    rows = list_feedback(status=None if status_filter == "all" else status_filter)
    if not rows:
        st.info("No feedback yet.")
    else:
        _statuses = ["new", "triaged", "resolved"]
        for fb in rows:
            who = fb.get("username") or f"user #{fb['user_id']}"
            team = fb.get("team_name") or "—"
            tag = f" · `{fb['feature_tag']}`" if fb.get("feature_tag") else ""
            st.markdown(f"**{fb['page']}**{tag}  \n{who} · {team} · {fb['created_at'][:16]} · v{fb['app_version']}")
            st.write(fb["message"])
            cols = st.columns([2, 4, 2])
            with cols[0]:
                current = fb["status"] if fb["status"] in _statuses else "new"
                new_status = st.selectbox(
                    "Status",
                    options=_statuses,
                    index=_statuses.index(current),
                    key=f"fbstatus_{fb['id']}",
                )
                if new_status != fb["status"] and st.button("Update", key=f"fbupd_{fb['id']}"):
                    set_feedback_status(fb["id"], new_status)
                    st.rerun()
            with cols[1]:
                notes = st.text_input(
                    "Admin notes",
                    value=fb.get("admin_notes") or "",
                    key=f"fbnotes_{fb['id']}",
                )
                if st.button("Save notes", key=f"fbsave_{fb['id']}"):
                    set_feedback_notes(fb["id"], notes)
                    st.rerun()
            with cols[2]:
                if fb.get("data_state"):
                    with st.expander("Data state"):
                        try:
                            st.json(json.loads(fb["data_state"]))
                        except (ValueError, TypeError):
                            st.write(fb["data_state"])
            st.divider()

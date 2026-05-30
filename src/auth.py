"""Multi-user authentication and account lifecycle for HEATER v2.

Everything here is gated behind the MULTI_USER env flag. When MULTI_USER is
unset or falsey, require_auth()/require_admin() are no-ops and HEATER behaves
exactly as the single-user v1 app. When MULTI_USER is on, users self-register
(status='pending'), an admin approves them and assigns a Yahoo team
(status='active'), and per-session identity replaces the global
league_teams.is_user_team flag for personalized surfaces.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime

import bcrypt
import streamlit as st

logger = logging.getLogger(__name__)

# ── Password hashing ─────────────────────────────────────────────────


def hash_password(password: str) -> str:
    """Return a bcrypt hash (utf-8 str) of the given plaintext password."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Return True iff password matches the bcrypt hash. Never raises."""
    if not password_hash:
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        # Malformed/corrupt hash → treat as non-match, don't crash the page.
        return False


# ── Feature flag ─────────────────────────────────────────────────────

_TRUTHY = {"1", "true", "yes", "on"}


def multi_user_enabled() -> bool:
    """True iff the MULTI_USER env flag is set to a truthy value."""
    return os.environ.get("MULTI_USER", "").strip().lower() in _TRUTHY


# ── DB row helpers ───────────────────────────────────────────────────


def _row_to_dict(row) -> dict | None:
    """Convert a sqlite3.Row to a plain dict (or None)."""
    return dict(row) if row is not None else None


def get_user(username: str) -> dict | None:
    """Fetch a user by username (case-insensitive). Returns None if absent."""
    from src.database import get_connection

    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ? COLLATE NOCASE",
            (username,),
        ).fetchone()
        return _row_to_dict(row)
    finally:
        conn.close()


def create_user(username: str, password: str, display_name: str | None = None) -> dict:
    """Create a self-registered user with status='pending'.

    Raises ValueError if the username is already taken (case-insensitive).
    """
    username = username.strip()
    if not username:
        raise ValueError("Username cannot be empty.")
    if not password:
        raise ValueError("Password cannot be empty.")
    if get_user(username) is not None:
        raise ValueError(f"Username '{username}' is already taken.")

    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, display_name, status, "
            "is_admin, created_at) VALUES (?, ?, ?, 'pending', 0, ?)",
            (
                username,
                hash_password(password),
                (display_name or "").strip() or None,
                datetime.now(UTC).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return get_user(username)


# ── Login decision (pure) ────────────────────────────────────────────


def classify_login(user: dict | None, password: str) -> str:
    """Pure decision: what is the result of this login attempt?

    Returns one of: 'bad_credentials', 'pending', 'revoked', 'ok'.
    Password is always checked first so a wrong password never reveals
    whether an account exists or what state it's in.
    """
    if user is None:
        return "bad_credentials"
    if not verify_password(password, user.get("password_hash", "")):
        return "bad_credentials"
    status = user.get("status")
    if status == "active":
        return "ok"
    if status == "revoked":
        return "revoked"
    return "pending"


# ── Admin lifecycle ──────────────────────────────────────────────────


def list_users(status: str | None = None) -> list[dict]:
    """Return all users, optionally filtered by status, newest first."""
    from src.database import get_connection

    conn = get_connection()
    try:
        if status is None:
            rows = conn.execute("SELECT * FROM users ORDER BY created_at DESC, user_id DESC").fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM users WHERE status = ? ORDER BY created_at DESC, user_id DESC",
                (status,),
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _require_existing(username: str) -> dict:
    user = get_user(username)
    if user is None:
        raise ValueError(f"No such user: '{username}'.")
    return user


def approve_user(username: str, team_name: str, approved_by: str | None = None) -> None:
    """Activate a pending user and assign their Yahoo team."""
    _require_existing(username)
    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute(
            "UPDATE users SET status='active', team_name=?, approved_at=?, approved_by=? "
            "WHERE username = ? COLLATE NOCASE",
            (team_name, datetime.now(UTC).isoformat(), approved_by, username),
        )
        conn.commit()
    finally:
        conn.close()


def revoke_user(username: str) -> None:
    """Revoke a user's access (reversible — admin can re-approve)."""
    _require_existing(username)
    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute(
            "UPDATE users SET status='revoked' WHERE username = ? COLLATE NOCASE",
            (username,),
        )
        conn.commit()
    finally:
        conn.close()


def set_user_team(username: str, team_name: str) -> None:
    """Reassign a user's Yahoo team without changing their status."""
    _require_existing(username)
    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute(
            "UPDATE users SET team_name=? WHERE username = ? COLLATE NOCASE",
            (team_name, username),
        )
        conn.commit()
    finally:
        conn.close()


# ── Bootstrap admin ──────────────────────────────────────────────────


def ensure_bootstrap_admin() -> None:
    """Seed the admin account from ADMIN_USERNAME / ADMIN_PASSWORD env vars.

    Idempotent: if the admin already exists, this does nothing (it never
    resets the password). No-op when the env vars are unset. ADMIN_TEAM_NAME
    is optional — set it so the admin's personalized surfaces pin to their
    own league team.
    """
    username = os.environ.get("ADMIN_USERNAME", "").strip()
    password = os.environ.get("ADMIN_PASSWORD", "")
    if not username or not password:
        return
    if get_user(username) is not None:
        return

    team_name = os.environ.get("ADMIN_TEAM_NAME", "").strip() or None
    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, display_name, team_name, "
            "status, is_admin, created_at, approved_at, approved_by) "
            "VALUES (?, ?, ?, ?, 'active', 1, ?, ?, 'bootstrap')",
            (
                username,
                hash_password(password),
                username,
                team_name,
                datetime.now(UTC).isoformat(),
                datetime.now(UTC).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    logger.info("ensure_bootstrap_admin: seeded admin account '%s'", username)


# ── Session identity ─────────────────────────────────────────────────

_SESSION_KEY = "auth_user"


def _session_state():
    """Return st.session_state (indirection seam for unit tests)."""
    return st.session_state


def current_user() -> dict | None:
    """Return the logged-in user dict for this session, or None."""
    return _session_state().get(_SESSION_KEY)


def _set_session_user(user: dict) -> None:
    _session_state()[_SESSION_KEY] = user


def logout() -> None:
    """Clear the session's identity."""
    _session_state().pop(_SESSION_KEY, None)


# ── Auth guards ──────────────────────────────────────────────────────


def _ensure_session_bootstrap() -> None:
    """Idempotently init the DB schema + seed the admin, once per session.

    Pages can be deep-linked without app.py's main() ever running this
    session, so the guard self-bootstraps rather than assuming setup ran.
    """
    state = _session_state()
    if state.get("_auth_bootstrap_done"):
        return
    from src.database import init_db

    init_db()
    ensure_bootstrap_admin()
    state["_auth_bootstrap_done"] = True


def require_auth() -> None:
    """Gate the current page. No-op when MULTI_USER is off.

    When on: render login/register and stop if there's no valid session user;
    otherwise re-validate against the DB so admin revoke/reassign takes effect
    on the next navigation.
    """
    if not multi_user_enabled():
        return
    _ensure_session_bootstrap()

    sess_user = current_user()
    if sess_user is None:
        _render_login_and_register()
        st.stop()
        return  # unreachable in real Streamlit; kept for test stubs

    fresh = get_user(sess_user.get("username", ""))
    if fresh is None or fresh.get("status") != "active":
        logout()
        _render_login_and_register()
        st.stop()
        return
    _set_session_user(fresh)


def require_admin() -> None:
    """Gate an admin-only page. Hard-stops non-admins."""
    require_auth()
    user = current_user()
    if not user or not user.get("is_admin"):
        st.error("You don't have access to this page.")
        st.stop()


_VIEW_AS_KEY = "auth_view_as_real"


def enter_view_as(target_username: str, admin_id: int) -> None:
    if not multi_user_enabled():
        return
    real = _session_state().get(_SESSION_KEY)
    if not (real and real.get("is_admin")):
        return
    target = get_user(target_username)
    if target is None:
        return
    _session_state()[_VIEW_AS_KEY] = real
    _set_session_user(target)
    from src.audit import log_action

    log_action(admin_id, "view_as", target=target_username)


def exit_view_as() -> None:
    real = _session_state().pop(_VIEW_AS_KEY, None)
    if real is None:
        return
    _set_session_user(real)
    from src.audit import log_action

    log_action(real.get("user_id", 0), "exit_view_as")


def is_viewing_as() -> dict | None:
    return _session_state().get(_VIEW_AS_KEY)


# ── Login / register UI ──────────────────────────────────────────────


def get_league_team_names() -> list[str]:
    """Yahoo team names from league_teams, for the admin approval dropdown."""
    from src.database import get_connection

    conn = get_connection()
    try:
        rows = conn.execute("SELECT team_name FROM league_teams ORDER BY team_name").fetchall()
        return [r[0] for r in rows if r[0]]
    except Exception:
        return []
    finally:
        conn.close()


def _render_login_and_register() -> None:
    """Render the login + self-registration tabs and stop the page."""
    st.title("HEATER — League Sign In")
    login_tab, register_tab = st.tabs(["Sign in", "Create account"])

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in")
        if submitted:
            result = classify_login(get_user(username), password)
            if result == "ok":
                _set_session_user(get_user(username))
                st.rerun()
            elif result == "pending":
                st.warning("Your account is awaiting admin approval.")
            elif result == "revoked":
                st.error("Your access has been revoked. Contact the league admin.")
            else:
                st.error("Invalid username or password.")

    with register_tab:
        with st.form("register_form"):
            new_username = st.text_input("Choose a username")
            display_name = st.text_input("Display name (optional)")
            new_password = st.text_input("Choose a password", type="password")
            confirm = st.text_input("Confirm password", type="password")
            registered = st.form_submit_button("Create account")
        if registered:
            if new_password != confirm:
                st.error("Passwords do not match.")
            else:
                try:
                    create_user(new_username, new_password, display_name=display_name)
                    st.success(
                        "Account created. The league admin will approve you and assign your team — check back shortly."
                    )
                except ValueError as exc:
                    st.error(str(exc))

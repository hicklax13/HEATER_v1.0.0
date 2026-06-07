"""Multi-user authentication and account lifecycle for HEATER v2.

Everything here is gated behind the MULTI_USER env flag. When MULTI_USER is
unset or falsey, require_auth()/require_admin() are no-ops and HEATER behaves
exactly as the single-user v1 app. When MULTI_USER is on, users self-register
(status='pending'), an admin approves them and assigns a Yahoo team
(status='active'), and per-session identity replaces the global
league_teams.is_user_team flag for personalized surfaces.
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
from datetime import UTC, datetime, timedelta

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


# ── Persistent session tokens (BR-1) ─────────────────────────────────
#
# A page refresh / bookmark / deep-link starts a fresh st.session_state (it is
# per-websocket), which would otherwise drop a logged-in member back to the
# login wall. At login we mint an opaque token, store it server-side, and set it
# in a browser cookie; on app load with an empty session we re-hydrate auth_user
# from that token. The cookie cannot be HttpOnly (Streamlit can only set cookies
# from JS), so the token is opaque (secrets.token_urlsafe), validated server-side
# on every restore, expires, and is revoked on logout — never identity-bearing.

_AUTH_COOKIE_NAME = "heater_session"
_TOKEN_TTL_DAYS = 30
_SESSION_TOKEN_KEY = "_auth_session_token"  # session_state cache of this session's token


def issue_session_token(user_id: int, days: int = _TOKEN_TTL_DAYS) -> str:
    """Mint + persist an opaque session token for ``user_id``; return the token.

    ``days`` may be negative in tests to mint an already-expired token. The token
    itself is never logged.
    """
    token = secrets.token_urlsafe(32)
    now = datetime.now(UTC)
    expires = now + timedelta(days=days)
    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO auth_tokens (token, user_id, created_at, expires_at, revoked) VALUES (?, ?, ?, ?, 0)",
            (token, int(user_id), now.isoformat(), expires.isoformat()),
        )
        conn.commit()
    finally:
        conn.close()
    return token


def validate_session_token(token: str | None) -> dict | None:
    """Return the user dict for a valid token, else None.

    Valid means: token exists, not revoked, not expired, AND the backing user is
    still status='active' (so an admin revoke/suspend of the ACCOUNT also kills
    the persistent session, not just an explicit logout).
    """
    if not token:
        return None
    from src.database import get_connection

    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT user_id, expires_at, revoked FROM auth_tokens WHERE token = ?",
            (token,),
        ).fetchone()
    finally:
        conn.close()
    if row is None or row["revoked"]:
        return None
    try:
        if datetime.fromisoformat(row["expires_at"]) <= datetime.now(UTC):
            return None
    except (ValueError, TypeError):
        return None

    from src.database import get_connection as _gc

    conn = _gc()
    try:
        urow = conn.execute("SELECT * FROM users WHERE user_id = ?", (row["user_id"],)).fetchone()
    finally:
        conn.close()
    user = _row_to_dict(urow)
    if user is None or user.get("status") != "active":
        return None
    return user


def revoke_session_token(token: str | None) -> None:
    """Mark a token revoked (idempotent; unknown token is a no-op)."""
    if not token:
        return
    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute("UPDATE auth_tokens SET revoked = 1 WHERE token = ?", (token,))
        conn.commit()
    finally:
        conn.close()


# ── Cookie layer (thin, mockable seams) ──────────────────────────────
#
# Streamlit has no native cookie-WRITE API, so reads use st.context.cookies
# (Streamlit >= 1.42) and writes/clears inject a `document.cookie` assignment via
# st.components.v1.html. The cookie is set Secure + SameSite=Lax. Both helpers
# are no-ops on any failure (missing st.context, headless test runner) so a
# cookie hiccup can never crash a page.


def _read_auth_cookie() -> str | None:
    """Read the persistent-session token from the browser cookie, or None."""
    try:
        cookies = getattr(st.context, "cookies", None)
        if not cookies:
            return None
        val = cookies.get(_AUTH_COOKIE_NAME)
        return val or None
    except Exception:
        return None


def _write_auth_cookie(token: str, max_age_days: int = _TOKEN_TTL_DAYS) -> None:
    """Set the session-token cookie in the browser (Secure; SameSite=Lax)."""
    try:
        import streamlit.components.v1 as components

        max_age = int(max_age_days * 24 * 3600)
        # json.dumps gives a safely-quoted JS string literal for the opaque token.
        js = (
            "<script>document.cookie = "
            + json.dumps(f"{_AUTH_COOKIE_NAME}=")
            + " + "
            + json.dumps(token)
            + " + "
            + json.dumps(f"; Max-Age={max_age}; Path=/; SameSite=Lax; Secure")
            + ";</script>"
        )
        components.html(js, height=0, width=0)
    except Exception:
        # Cookie persistence is best-effort: if the component can't render, the
        # user simply re-logs in on the next fresh session (v1-equivalent UX).
        logger.debug("auth: _write_auth_cookie failed", exc_info=True)


def _clear_auth_cookie() -> None:
    """Expire the session-token cookie in the browser."""
    try:
        import streamlit.components.v1 as components

        js = (
            "<script>document.cookie = "
            + json.dumps(f"{_AUTH_COOKIE_NAME}=; Max-Age=0; Path=/; SameSite=Lax; Secure")
            + ";</script>"
        )
        components.html(js, height=0, width=0)
    except Exception:
        logger.debug("auth: _clear_auth_cookie failed", exc_info=True)


def _establish_persistent_session(user: dict | None) -> None:
    """Mint a token for ``user`` + set the browser cookie + cache the token.

    Called after a successful login so a later refresh re-hydrates. No-op when
    the flag is off (v1 never sets a cookie) or the user has no id.
    """
    if not multi_user_enabled() or not user or not user.get("user_id"):
        return
    token = issue_session_token(user["user_id"])
    _session_state()[_SESSION_TOKEN_KEY] = token
    _write_auth_cookie(token)


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
    """Clear the session's identity.

    Under MULTI_USER this also revokes the persistent-session token server-side
    and clears the browser cookie so a refresh after logout does NOT re-hydrate.
    When the flag is off this is exactly the v1 behavior: just drop the session
    key (no cookie/token machinery is ever touched).
    """
    state = _session_state()
    if multi_user_enabled():
        token = state.pop(_SESSION_TOKEN_KEY, None)
        if token:
            revoke_session_token(token)
        _clear_auth_cookie()
    state.pop(_SESSION_KEY, None)


def _normalize_team_name(name) -> str:
    """Lowercase + strip all non-alphanumerics (emoji, whitespace, punctuation)
    for tolerant team-name matching, so an env-seeded admin assignment missing the
    Yahoo team's leading emoji ("Team Hickey") still matches the roster name
    ("🏆 Team Hickey"). 2026-06-02 (surfaced on the owner's own team at launch)."""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def resolve_viewer_team_name(rosters=None) -> str | None:
    """The team that belongs to the current viewer — the SOLE source of truth.

    MULTI_USER on and the logged-in session user has an admin-assigned team:
    return that team (per-session identity; set by the admin, not self-asserted).
    Otherwise fall back to the legacy ``league_teams.is_user_team`` flag so
    single-user v1 behavior is byte-for-byte unchanged — preferring the supplied
    ``rosters`` DataFrame (the same source the pages filtered pre-fix) and finally
    a ``league_teams`` DB lookup.

    Every personalized page MUST call this instead of filtering
    ``rosters["is_user_team"] == 1`` directly — otherwise every logged-in user
    sees the single is_user_team team (the 2026-06-01 launch-blocker). Guarded by
    ``tests/test_pages_use_viewer_team_resolver.py``.
    """
    if multi_user_enabled():
        # Per-session identity. A logged-in member with no assigned team resolves
        # to None (NOT the legacy is_user_team flag, which points at the admin's
        # team) so pages show a 'connect / no team' state instead of the wrong
        # team. Under MULTI_USER we never fall through to the global flag.
        user = current_user()
        team = (user or {}).get("team_name")
        team = team.strip() if isinstance(team, str) else team
        if not team:
            return None
        # Reconcile the admin-assigned name against the ACTUAL roster names: a
        # team whose Yahoo name carries a leading emoji/whitespace (e.g.
        # "🏆 Team Hickey") won't exact-match an env-seeded assignment
        # ("Team Hickey"), leaving My Team empty. If a normalized match exists,
        # return the EXACT roster name so downstream get_team_roster() filters
        # hit. Exact matches (the common case — members assigned via the admin
        # dropdown) short-circuit unchanged.
        if (
            rosters is not None
            and not getattr(rosters, "empty", True)
            and "team_name" in getattr(rosters, "columns", [])
        ):
            names = [str(n) for n in rosters["team_name"].dropna().unique()]
            if team not in names:
                tnorm = _normalize_team_name(team)
                if tnorm:
                    for n in names:
                        if _normalize_team_name(n) == tnorm:
                            return n
        return team

    # v1 / fallback — the legacy is_user_team flag, from the rosters frame if given.
    if (
        rosters is not None
        and not getattr(rosters, "empty", True)
        and "is_user_team" in getattr(rosters, "columns", [])
    ):
        user_rows = rosters[rosters["is_user_team"] == 1]
        if not user_rows.empty:
            name = user_rows.iloc[0]["team_name"]
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            return name

    try:
        from src.database import get_connection

        conn = get_connection()
        try:
            row = conn.execute("SELECT team_name FROM league_teams WHERE is_user_team = 1").fetchone()
            return row[0] if row else None
        finally:
            conn.close()
    except Exception:
        logger.warning("resolve_viewer_team_name: DB fallback failed.", exc_info=True)
        return None


def viewer_can_write() -> bool:
    """True if the current viewer may trigger data writes/refreshes.

    Single-user v1 (MULTI_USER off): always True — the lone user owns the data.
    MULTI_USER on: only admins, because the in-process scheduler is the SOLE
    SQLite writer and member sessions are read-only consumers. Gates the
    member-facing 'Refresh'/'Sync' buttons so a dozen members can't fire
    concurrent writes against the single-writer DB ("database is locked").
    """
    if not multi_user_enabled():
        return True
    return bool((current_user() or {}).get("is_admin"))


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
        # BR-1: a fresh per-websocket session (refresh / bookmark / deep-link)
        # has no auth_user. Try to re-hydrate from a valid persistent-session
        # cookie before falling back to the login wall.
        cookie_token = _read_auth_cookie()
        restored = validate_session_token(cookie_token)
        if restored is not None:
            _set_session_user(restored)
            _session_state()[_SESSION_TOKEN_KEY] = cookie_token
            return
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
    if target is None or target.get("is_admin"):
        # View-as previews a MEMBER's experience; never impersonate another admin
        # (it would mis-attribute that admin's actions in the audit log).
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
                user = get_user(username)
                _set_session_user(user)
                _establish_persistent_session(user)
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

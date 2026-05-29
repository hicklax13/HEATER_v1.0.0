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

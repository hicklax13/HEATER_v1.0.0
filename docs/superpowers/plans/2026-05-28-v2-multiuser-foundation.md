# HEATER v2 Multi-User Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add self-registration + admin-approved team assignment + per-session identity to HEATER, fully gated behind a `MULTI_USER` env flag so the single-user v1 app is byte-for-byte unchanged when the flag is off.

**Architecture:** A new `src/auth.py` module owns the entire account lifecycle (register → pending → admin-approve+assign-team → active; revoke/reassign anytime). Identity is per-session (`st.session_state["auth_user"]`), replacing the single global `league_teams.is_user_team=1` flag as the source of "who is the user" for personalized surfaces. The `users` table is created additively by `init_db()`. Every page and `app.py` call a one-line `require_auth()` guard that is a no-op when `MULTI_USER` is off. Pure decision functions (`classify_login`, password hashing) carry the logic so it unit-tests without a Streamlit runtime; a `DB_PATH` monkeypatch gives each test an isolated SQLite file with the real schema.

**Tech Stack:** Python 3.12/3.14, Streamlit (file-based multipage), SQLite (WAL via `get_connection()`), `bcrypt>=4.1` for password hashing, pytest (+ `streamlit.testing.v1.AppTest` for UI smoke).

---

## File Structure

**Create:**
- `src/auth.py` — multi-user auth + account lifecycle (the whole feature lives here). One responsibility: "who is this session's user, and are they allowed in."
- `pages/00_Admin_Console.py` — minimal admin page: pending-approval queue (approve + assign team), active-user list (revoke / reassign team). Hard-walled by `require_admin()`.
- `tests/test_auth_password.py` — password hash/verify unit tests.
- `tests/test_auth_flag.py` — `multi_user_enabled()` flag behavior.
- `tests/test_auth_users_table.py` — `users` table schema after `init_db()`.
- `tests/test_auth_lifecycle.py` — create/get/approve/revoke/reassign/list + `classify_login` + `ensure_bootstrap_admin`.
- `tests/test_auth_session.py` — session helpers + `require_auth`/`require_admin` decision branches.
- `tests/test_auth_backcompat.py` — MULTI_USER-off = v1 behavior (consolidated regression guard).
- `tests/test_yahoo_user_team_identity.py` — `_get_user_team_name` reroute (per-session identity vs. `league_teams` fallback).
- `tests/test_pages_have_auth_guard.py` — structural invariant: every page calling `inject_custom_css()` also calls `require_auth()`.
- `tests/test_app_main_auth_gate.py` — structural invariant: `app.py main()` calls `require_auth()` between `init_db()` and `render_splash_screen()`.
- `tests/test_admin_console_guarded.py` — `pages/00_Admin_Console.py` calls `require_admin()`; AppTest smoke.

**Modify:**
- `requirements.txt` — add `bcrypt>=4.1`.
- `src/database.py:204` — call new `_init_multiuser_tables(conn)` from `init_db()`; define it after `_init_db_tables_and_columns`.
- `app.py:2454-2455` — insert auth gate between `init_db()` and `render_splash_screen()`.
- `src/yahoo_data_service.py:1103-1123` — reroute `_get_user_team_name` to per-session identity when multi-user is on.
- All 13 `pages/*.py` — add `from src.auth import require_auth` + `require_auth()` immediately after the page's `inject_custom_css()` call.
- `CLAUDE.md` — document env flags, `src/auth.py`, new structural invariants.

---

## Task 0: Add bcrypt dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add the dependency**

In `requirements.txt`, add this line in the analytics/util section (anywhere among the top-level pins, e.g. right after the `scipy` line):

```
bcrypt>=4.1
```

- [ ] **Step 2: Install it**

Run: `python -m pip install "bcrypt>=4.1"`
Expected: `Successfully installed bcrypt-<version>` (or "Requirement already satisfied").

- [ ] **Step 3: Verify import**

Run: `python -c "import bcrypt; print(bcrypt.__version__)"`
Expected: prints a version `>= 4.1` with no traceback.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "build(deps): add bcrypt for multi-user password hashing"
```

---

## Task 1: Create the `users` table via `_init_multiuser_tables`

**Files:**
- Modify: `src/database.py` (call site at line 204 inside `init_db`; new function after `_init_db_tables_and_columns`)
- Test: `tests/test_auth_users_table.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_auth_users_table.py`:

```python
"""The users table must be created additively by init_db()."""

import pytest


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Point src.database at a throwaway SQLite file with the real schema.

    get_connection() reads the module-global DB_PATH at call time, so
    monkeypatching it redirects every connection for the duration of the test.
    """
    db_file = tmp_path / "auth_test.db"
    monkeypatch.setattr("src.database.DB_PATH", db_file)
    from src.database import init_db

    init_db()
    return db_file


def _columns(conn, table):
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def test_users_table_exists(temp_db):
    from src.database import get_connection

    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
        ).fetchone()
        assert row is not None, "init_db() must create the users table"
    finally:
        conn.close()


def test_users_table_columns(temp_db):
    from src.database import get_connection

    conn = get_connection()
    try:
        cols = _columns(conn, "users")
    finally:
        conn.close()
    expected = {
        "user_id",
        "username",
        "password_hash",
        "display_name",
        "team_name",
        "status",
        "is_admin",
        "created_at",
        "approved_at",
        "approved_by",
        "last_seen_at",
    }
    assert expected.issubset(cols), f"missing columns: {expected - cols}"


def test_username_unique_case_insensitive(temp_db):
    import sqlite3

    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, status, is_admin, created_at) "
            "VALUES ('Sam', 'x', 'pending', 0, '2026-05-28')"
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO users (username, password_hash, status, is_admin, created_at) "
                "VALUES ('sam', 'y', 'pending', 0, '2026-05-28')"
            )
            conn.commit()
    finally:
        conn.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_auth_users_table.py -v`
Expected: FAIL — `test_users_table_exists` asserts `row is not None` but `users` table does not exist yet (`assert None is not None`).

- [ ] **Step 3: Write minimal implementation**

In `src/database.py`, change the `init_db()` body so the second connection block also calls the new function. The current code (lines 202-206) is:

```python
    conn = get_connection()
    try:
        _init_db_tables_and_columns(conn)
    finally:
        conn.close()
```

Replace it with:

```python
    conn = get_connection()
    try:
        _init_db_tables_and_columns(conn)
        _init_multiuser_tables(conn)
    finally:
        conn.close()
```

Then add this new function immediately after the end of `_init_db_tables_and_columns` (before the next top-level `def`):

```python
def _init_multiuser_tables(conn):
    """Create v2 multi-user tables (additive; no-op once created).

    Gated at the app layer by the MULTI_USER env flag, but the table itself is
    always created so a flag flip needs no migration. username is UNIQUE +
    COLLATE NOCASE so 'Sam' and 'sam' are the same account (Muncy-DNA-class
    case-collision guard, consistent with players-table lookups).
    """
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            display_name TEXT,
            team_name TEXT,                          -- assigned Yahoo team; NULL until approved
            status TEXT NOT NULL DEFAULT 'pending',  -- pending | active | revoked
            is_admin INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            approved_at TEXT,
            approved_by TEXT,
            last_seen_at TEXT                        -- updated by usage logging (Plan 2); nullable now
        );
        CREATE INDEX IF NOT EXISTS idx_users_status ON users(status);
    """)
    conn.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_auth_users_table.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/database.py tests/test_auth_users_table.py
git commit -m "feat(db): add users table for v2 multi-user (additive via init_db)"
```

---

## Task 2: Password hashing (`hash_password` / `verify_password`)

**Files:**
- Create: `src/auth.py`
- Test: `tests/test_auth_password.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_auth_password.py`:

```python
"""bcrypt password hashing round-trips and rejects bad input safely."""

from src.auth import hash_password, verify_password


def test_hash_is_not_plaintext():
    h = hash_password("correct horse")
    assert h != "correct horse"
    assert isinstance(h, str)
    assert len(h) > 20


def test_verify_correct_password():
    h = hash_password("s3cret!")
    assert verify_password("s3cret!", h) is True


def test_verify_wrong_password():
    h = hash_password("s3cret!")
    assert verify_password("nope", h) is False


def test_verify_malformed_hash_is_false_not_error():
    # A corrupted/empty hash must return False, never raise.
    assert verify_password("anything", "") is False
    assert verify_password("anything", "not-a-bcrypt-hash") is False


def test_distinct_salts_per_hash():
    assert hash_password("same") != hash_password("same")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_auth_password.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.auth'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/auth.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_auth_password.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/auth.py tests/test_auth_password.py
git commit -m "feat(auth): bcrypt password hash/verify helpers"
```

---

## Task 3: `multi_user_enabled()` env flag

**Files:**
- Modify: `src/auth.py`
- Test: `tests/test_auth_flag.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_auth_flag.py`:

```python
"""multi_user_enabled() reads the MULTI_USER env var (default off)."""

import pytest

from src.auth import multi_user_enabled


@pytest.mark.parametrize("value", ["1", "true", "True", "yes", "on", "  1  "])
def test_enabled_truthy(monkeypatch, value):
    monkeypatch.setenv("MULTI_USER", value)
    assert multi_user_enabled() is True


@pytest.mark.parametrize("value", ["", "0", "false", "False", "no", "off"])
def test_disabled_falsey(monkeypatch, value):
    monkeypatch.setenv("MULTI_USER", value)
    assert multi_user_enabled() is False


def test_default_is_off(monkeypatch):
    monkeypatch.delenv("MULTI_USER", raising=False)
    assert multi_user_enabled() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_auth_flag.py -v`
Expected: FAIL — `ImportError: cannot import name 'multi_user_enabled' from 'src.auth'`.

- [ ] **Step 3: Write minimal implementation**

In `src/auth.py`, add `import os` to the imports (alphabetical, before `import bcrypt`'s block — put `import os` under `import logging`), then add after the password section:

```python
# ── Feature flag ─────────────────────────────────────────────────────

_TRUTHY = {"1", "true", "yes", "on"}


def multi_user_enabled() -> bool:
    """True iff the MULTI_USER env flag is set to a truthy value."""
    return os.environ.get("MULTI_USER", "").strip().lower() in _TRUTHY
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_auth_flag.py -v`
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
git add src/auth.py tests/test_auth_flag.py
git commit -m "feat(auth): MULTI_USER feature flag"
```

---

## Task 4: `create_user` + `get_user` (self-registration → pending)

**Files:**
- Modify: `src/auth.py`
- Test: `tests/test_auth_lifecycle.py` (created here; extended in Task 6)

- [ ] **Step 1: Write the failing test**

Create `tests/test_auth_lifecycle.py`:

```python
"""Account lifecycle: register (pending) → get → (Task 6 adds approve/revoke)."""

import pytest


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_file = tmp_path / "auth_lifecycle.db"
    monkeypatch.setattr("src.database.DB_PATH", db_file)
    from src.database import init_db

    init_db()
    return db_file


def test_create_user_starts_pending(temp_db):
    from src.auth import create_user

    user = create_user("alice", "pw-alice", display_name="Alice A.")
    assert user["username"] == "alice"
    assert user["status"] == "pending"
    assert user["is_admin"] == 0
    assert user["team_name"] is None
    assert user["display_name"] == "Alice A."


def test_get_user_roundtrip(temp_db):
    from src.auth import create_user, get_user

    create_user("bob", "pw-bob")
    fetched = get_user("bob")
    assert fetched is not None
    assert fetched["username"] == "bob"


def test_get_user_is_case_insensitive(temp_db):
    from src.auth import create_user, get_user

    create_user("Carol", "pw")
    assert get_user("carol") is not None
    assert get_user("CAROL") is not None


def test_get_unknown_user_returns_none(temp_db):
    from src.auth import get_user

    assert get_user("nobody") is None


def test_duplicate_username_raises(temp_db):
    from src.auth import create_user

    create_user("dave", "pw1")
    with pytest.raises(ValueError):
        create_user("dave", "pw2")


def test_duplicate_username_case_insensitive_raises(temp_db):
    from src.auth import create_user

    create_user("Eve", "pw1")
    with pytest.raises(ValueError):
        create_user("eve", "pw2")


def test_stored_password_is_hashed(temp_db):
    from src.auth import create_user, verify_password

    create_user("frank", "frank-secret")
    from src.auth import get_user

    row = get_user("frank")
    assert row["password_hash"] != "frank-secret"
    assert verify_password("frank-secret", row["password_hash"]) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_auth_lifecycle.py -v`
Expected: FAIL — `ImportError: cannot import name 'create_user' from 'src.auth'`.

- [ ] **Step 3: Write minimal implementation**

In `src/auth.py`, add `from datetime import UTC, datetime` to the imports, then add this DB section after the feature-flag section:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_auth_lifecycle.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/auth.py tests/test_auth_lifecycle.py
git commit -m "feat(auth): self-registration (create_user/get_user, pending status)"
```

---

## Task 5: `classify_login` (pure login-decision function)

**Files:**
- Modify: `src/auth.py`
- Test: `tests/test_auth_lifecycle.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_auth_lifecycle.py`:

```python
# ── classify_login (pure; no DB) ─────────────────────────────────────


def _make_user(status="active", password="pw"):
    from src.auth import hash_password

    return {
        "username": "u",
        "password_hash": hash_password(password),
        "status": status,
        "is_admin": 0,
        "team_name": "Team Hickey",
    }


def test_classify_login_no_such_user():
    from src.auth import classify_login

    assert classify_login(None, "anything") == "bad_credentials"


def test_classify_login_wrong_password():
    from src.auth import classify_login

    assert classify_login(_make_user(password="right"), "wrong") == "bad_credentials"


def test_classify_login_pending():
    from src.auth import classify_login

    assert classify_login(_make_user(status="pending"), "pw") == "pending"


def test_classify_login_revoked():
    from src.auth import classify_login

    assert classify_login(_make_user(status="revoked"), "pw") == "revoked"


def test_classify_login_active_ok():
    from src.auth import classify_login

    assert classify_login(_make_user(status="active"), "pw") == "ok"


def test_classify_login_checks_password_before_status():
    # A pending user with the WRONG password is bad_credentials, not "pending"
    # (don't leak account-existence/status to someone who can't authenticate).
    from src.auth import classify_login

    assert classify_login(_make_user(status="pending", password="right"), "wrong") == "bad_credentials"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_auth_lifecycle.py -k classify_login -v`
Expected: FAIL — `ImportError: cannot import name 'classify_login' from 'src.auth'`.

- [ ] **Step 3: Write minimal implementation**

In `src/auth.py`, add after `create_user`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_auth_lifecycle.py -k classify_login -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/auth.py tests/test_auth_lifecycle.py
git commit -m "feat(auth): classify_login pure decision (creds before status)"
```

---

## Task 6: Admin lifecycle (`list_users`, `approve_user`, `revoke_user`, `set_user_team`)

**Files:**
- Modify: `src/auth.py`
- Test: `tests/test_auth_lifecycle.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_auth_lifecycle.py`:

```python
# ── Admin lifecycle (needs temp_db) ──────────────────────────────────


def test_approve_user_activates_and_assigns_team(temp_db):
    from src.auth import approve_user, create_user, get_user

    create_user("grace", "pw")
    approve_user("grace", team_name="Team Hickey", approved_by="admin")
    u = get_user("grace")
    assert u["status"] == "active"
    assert u["team_name"] == "Team Hickey"
    assert u["approved_by"] == "admin"
    assert u["approved_at"] is not None


def test_revoke_user(temp_db):
    from src.auth import approve_user, create_user, get_user, revoke_user

    create_user("heidi", "pw")
    approve_user("heidi", team_name="Team A")
    revoke_user("heidi")
    assert get_user("heidi")["status"] == "revoked"


def test_set_user_team_reassigns(temp_db):
    from src.auth import approve_user, create_user, get_user, set_user_team

    create_user("ivan", "pw")
    approve_user("ivan", team_name="Team A")
    set_user_team("ivan", "Team B")
    assert get_user("ivan")["team_name"] == "Team B"


def test_list_users_filters_by_status(temp_db):
    from src.auth import approve_user, create_user, list_users

    create_user("p1", "pw")
    create_user("p2", "pw")
    create_user("a1", "pw")
    approve_user("a1", team_name="Team A")

    pending = {u["username"] for u in list_users(status="pending")}
    active = {u["username"] for u in list_users(status="active")}
    everyone = {u["username"] for u in list_users()}

    assert pending == {"p1", "p2"}
    assert active == {"a1"}
    assert {"p1", "p2", "a1"}.issubset(everyone)


def test_approve_unknown_user_raises(temp_db):
    from src.auth import approve_user

    with pytest.raises(ValueError):
        approve_user("ghost", team_name="Team A")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_auth_lifecycle.py -k "approve or revoke or set_user_team or list_users" -v`
Expected: FAIL — `ImportError: cannot import name 'approve_user' from 'src.auth'`.

- [ ] **Step 3: Write minimal implementation**

In `src/auth.py`, add after `classify_login`:

```python
# ── Admin lifecycle ──────────────────────────────────────────────────


def list_users(status: str | None = None) -> list[dict]:
    """Return all users, optionally filtered by status, newest first."""
    from src.database import get_connection

    conn = get_connection()
    try:
        if status is None:
            rows = conn.execute(
                "SELECT * FROM users ORDER BY created_at DESC, user_id DESC"
            ).fetchall()
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_auth_lifecycle.py -v`
Expected: PASS (all lifecycle tests, including Tasks 4-5).

- [ ] **Step 5: Commit**

```bash
git add src/auth.py tests/test_auth_lifecycle.py
git commit -m "feat(auth): admin lifecycle (list/approve/revoke/reassign team)"
```

---

## Task 7: `ensure_bootstrap_admin` (seed the admin from env)

**Files:**
- Modify: `src/auth.py`
- Test: `tests/test_auth_lifecycle.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_auth_lifecycle.py`:

```python
# ── Bootstrap admin from env ─────────────────────────────────────────


def test_ensure_bootstrap_admin_creates_active_admin(temp_db, monkeypatch):
    from src.auth import ensure_bootstrap_admin, get_user

    monkeypatch.setenv("ADMIN_USERNAME", "connor")
    monkeypatch.setenv("ADMIN_PASSWORD", "admin-pw")
    monkeypatch.setenv("ADMIN_TEAM_NAME", "Team Hickey")

    ensure_bootstrap_admin()
    u = get_user("connor")
    assert u is not None
    assert u["is_admin"] == 1
    assert u["status"] == "active"
    assert u["team_name"] == "Team Hickey"


def test_ensure_bootstrap_admin_is_idempotent(temp_db, monkeypatch):
    from src.auth import ensure_bootstrap_admin, get_user, verify_password

    monkeypatch.setenv("ADMIN_USERNAME", "connor")
    monkeypatch.setenv("ADMIN_PASSWORD", "first-pw")
    ensure_bootstrap_admin()
    # Second call with a DIFFERENT password must NOT reset the existing admin.
    monkeypatch.setenv("ADMIN_PASSWORD", "second-pw")
    ensure_bootstrap_admin()
    u = get_user("connor")
    assert verify_password("first-pw", u["password_hash"]) is True
    assert verify_password("second-pw", u["password_hash"]) is False


def test_ensure_bootstrap_admin_noop_without_env(temp_db, monkeypatch):
    from src.auth import ensure_bootstrap_admin, list_users

    monkeypatch.delenv("ADMIN_USERNAME", raising=False)
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    ensure_bootstrap_admin()
    assert list_users() == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_auth_lifecycle.py -k bootstrap_admin -v`
Expected: FAIL — `ImportError: cannot import name 'ensure_bootstrap_admin' from 'src.auth'`.

- [ ] **Step 3: Write minimal implementation**

In `src/auth.py`, add after the admin-lifecycle section:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_auth_lifecycle.py -k bootstrap_admin -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/auth.py tests/test_auth_lifecycle.py
git commit -m "feat(auth): ensure_bootstrap_admin seeds admin from env (idempotent)"
```

---

## Task 8: Session identity helpers (`current_user`, `_set_session_user`, `logout`)

**Files:**
- Modify: `src/auth.py`
- Test: `tests/test_auth_session.py`

**Design note:** `current_user()`/`_set_session_user()`/`logout()` operate on Streamlit's
`st.session_state`, which can't be read outside a script run. The testable seam is a tiny
`_session_state()` indirection that returns `st.session_state`; tests monkeypatch it to a plain
dict. This keeps the session helpers thin and fully unit-testable.

- [ ] **Step 1: Write the failing test**

Create `tests/test_auth_session.py`:

```python
"""Session identity helpers, tested via a monkeypatched _session_state()."""

import pytest


@pytest.fixture
def fake_session(monkeypatch):
    state: dict = {}
    monkeypatch.setattr("src.auth._session_state", lambda: state)
    return state


def test_current_user_none_when_empty(fake_session):
    from src.auth import current_user

    assert current_user() is None


def test_set_and_get_session_user(fake_session):
    from src.auth import _set_session_user, current_user

    _set_session_user({"username": "alice", "team_name": "Team Hickey", "is_admin": 0})
    assert current_user()["username"] == "alice"


def test_logout_clears_user(fake_session):
    from src.auth import _set_session_user, current_user, logout

    _set_session_user({"username": "alice"})
    logout()
    assert current_user() is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_auth_session.py -v`
Expected: FAIL — `AttributeError: <module 'src.auth'> does not have the attribute '_session_state'` (monkeypatch target missing).

- [ ] **Step 3: Write minimal implementation**

In `src/auth.py`, add `import streamlit as st` to the imports (after `import os`), then add this session section after the bootstrap-admin section:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_auth_session.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/auth.py tests/test_auth_session.py
git commit -m "feat(auth): per-session identity helpers (current_user/logout)"
```

---

## Task 9: `require_auth` + `require_admin` guards + login/register UI

**Files:**
- Modify: `src/auth.py`
- Test: `tests/test_auth_session.py` (append)

**Behavior contract:**
- `require_auth()` — no-op when `MULTI_USER` off. When on: ensure DB schema + bootstrap admin
  once per session; if no session user, render the login/register UI and `st.stop()`. If a session
  user exists, **re-fetch from the DB** and re-validate status (so admin revoke/reassign takes
  effect on the next page nav), refreshing the session copy.
- `require_admin()` — calls `require_auth()` first; then if the user isn't an admin, `st.error()`
  + `st.stop()`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_auth_session.py`:

```python
# ── require_auth / require_admin ─────────────────────────────────────


class _Stop(Exception):
    """Sentinel standing in for st.stop()."""


@pytest.fixture
def stub_st(monkeypatch):
    """Stub the st.* calls require_auth/require_admin touch."""

    def _stop(*a, **k):
        raise _Stop()

    monkeypatch.setattr("src.auth.st.stop", _stop, raising=False)
    monkeypatch.setattr("src.auth.st.error", lambda *a, **k: None, raising=False)


def test_require_auth_noop_when_disabled(monkeypatch):
    # The load-bearing back-compat guarantee: flag off → never touches session/DB/st.
    from src.auth import require_auth

    monkeypatch.setattr("src.auth.multi_user_enabled", lambda: False)
    assert require_auth() is None


def test_require_auth_stops_when_no_user(monkeypatch, fake_session, stub_st):
    import src.auth as auth

    monkeypatch.setattr(auth, "multi_user_enabled", lambda: True)
    monkeypatch.setattr(auth, "_ensure_session_bootstrap", lambda: None)
    rendered = {"called": False}
    monkeypatch.setattr(auth, "_render_login_and_register", lambda: rendered.__setitem__("called", True))

    with pytest.raises(_Stop):
        auth.require_auth()
    assert rendered["called"] is True


def test_require_auth_ok_for_active_user(monkeypatch, fake_session):
    import src.auth as auth

    monkeypatch.setattr(auth, "multi_user_enabled", lambda: True)
    monkeypatch.setattr(auth, "_ensure_session_bootstrap", lambda: None)
    auth._set_session_user({"username": "alice", "status": "active"})
    # DB re-validation returns a fresh active row.
    monkeypatch.setattr(
        auth, "get_user", lambda u: {"username": "alice", "status": "active", "team_name": "Team Hickey", "is_admin": 0}
    )
    assert auth.require_auth() is None
    # session copy refreshed with the fresh DB row (team_name now present)
    assert auth.current_user()["team_name"] == "Team Hickey"


def test_require_auth_logs_out_revoked_user(monkeypatch, fake_session, stub_st):
    import src.auth as auth

    monkeypatch.setattr(auth, "multi_user_enabled", lambda: True)
    monkeypatch.setattr(auth, "_ensure_session_bootstrap", lambda: None)
    monkeypatch.setattr(auth, "_render_login_and_register", lambda: None)
    auth._set_session_user({"username": "bob", "status": "active"})
    # Admin revoked bob since login.
    monkeypatch.setattr(auth, "get_user", lambda u: {"username": "bob", "status": "revoked"})

    with pytest.raises(_Stop):
        auth.require_auth()
    assert auth.current_user() is None  # logged out


def test_require_admin_blocks_non_admin(monkeypatch, fake_session, stub_st):
    import src.auth as auth

    monkeypatch.setattr(auth, "require_auth", lambda: None)
    auth._set_session_user({"username": "alice", "status": "active", "is_admin": 0})
    with pytest.raises(_Stop):
        auth.require_admin()


def test_require_admin_allows_admin(monkeypatch, fake_session):
    import src.auth as auth

    monkeypatch.setattr(auth, "require_auth", lambda: None)
    auth._set_session_user({"username": "connor", "status": "active", "is_admin": 1})
    assert auth.require_admin() is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_auth_session.py -k "require_" -v`
Expected: FAIL — `ImportError: cannot import name 'require_auth' from 'src.auth'`.

- [ ] **Step 3: Write minimal implementation**

In `src/auth.py`, add this guards section after the session-identity section:

```python
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
```

Then add the login/register UI at the end of `src/auth.py`:

```python
# ── Login / register UI ──────────────────────────────────────────────


def get_league_team_names() -> list[str]:
    """Yahoo team names from league_teams, for the admin approval dropdown."""
    from src.database import get_connection

    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT team_name FROM league_teams ORDER BY team_name"
        ).fetchall()
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
                        "Account created. The league admin will approve you and "
                        "assign your team — check back shortly."
                    )
                except ValueError as exc:
                    st.error(str(exc))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_auth_session.py -v`
Expected: PASS (all session + guard tests).

- [ ] **Step 5: Commit**

```bash
git add src/auth.py tests/test_auth_session.py
git commit -m "feat(auth): require_auth/require_admin guards + login/register UI"
```

---

## Task 10: Wire the auth gate into `app.py main()`

**Files:**
- Modify: `app.py` (import block ~line 24; `main()` line 2454)
- Test: `tests/test_app_main_auth_gate.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_app_main_auth_gate.py`:

```python
"""Structural guard: app.py main() gates auth between init_db and splash."""

import ast
from pathlib import Path

_APP = Path(__file__).resolve().parent.parent / "app.py"


def _main_body_calls():
    """Ordered list of top-level call-expression names inside main()."""
    tree = ast.parse(_APP.read_text(encoding="utf-8"))
    main = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    calls = []
    for node in ast.walk(main):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                calls.append(func.id)
            elif isinstance(func, ast.Attribute):
                calls.append(func.attr)
    return calls


def test_require_auth_called_in_main():
    assert "require_auth" in _main_body_calls()


def test_require_auth_after_init_db_before_splash():
    src = _APP.read_text(encoding="utf-8")
    i_init = src.index("init_db()")
    i_auth = src.index("require_auth()")
    i_splash = src.index("render_splash_screen()")
    assert i_init < i_auth < i_splash, "require_auth() must sit between init_db() and splash"


def test_require_auth_imported():
    src = _APP.read_text(encoding="utf-8")
    assert "from src.auth import" in src and "require_auth" in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_app_main_auth_gate.py -v`
Expected: FAIL — `test_require_auth_called_in_main` fails (`require_auth` not in main's calls); `ValueError: substring not found` on `require_auth()`.

- [ ] **Step 3: Write minimal implementation**

In `app.py`, add the import after the `from src.data_bootstrap import ...` line (line 24):

```python
from src.auth import multi_user_enabled, require_auth
```

Then in `main()`, change the bootstrap block. Current (lines 2453-2455):

```python
    # Bootstrap all data on every session start (splash screen with progress)
    init_db()
    render_splash_screen()
```

Replace with:

```python
    # Bootstrap all data on every session start (splash screen with progress)
    init_db()

    # v2 multi-user gate: no-op when MULTI_USER is off (v1 behavior preserved).
    # When on, this renders login/register and stops for unauthenticated users
    # before any data or draft UI is shown.
    if multi_user_enabled():
        require_auth()

    render_splash_screen()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_app_main_auth_gate.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_app_main_auth_gate.py
git commit -m "feat(app): wire multi-user auth gate into main() (flag-gated)"
```

---

## Task 11: Reroute `_get_user_team_name` to per-session identity

**Files:**
- Modify: `src/yahoo_data_service.py:1103-1123`
- Test: `tests/test_yahoo_user_team_identity.py`

**Why this is load-bearing:** `_get_user_team_name` is the single point that answers "which team is
*the user*" for all personalized surfaces (My Team, Lineup Optimizer, FA recs, alerts). In v1 it
reads the global `league_teams.is_user_team=1` flag. In v2 it must instead reflect the
logged-in session's assigned team — otherwise every user would see Connor's team.

- [ ] **Step 1: Write the failing test**

Create `tests/test_yahoo_user_team_identity.py`:

```python
"""_get_user_team_name reflects session identity in multi-user mode."""


def _service():
    """A bare YahooDataService instance without running __init__ side effects."""
    from src.yahoo_data_service import YahooDataService

    return YahooDataService.__new__(YahooDataService)


def test_multiuser_returns_session_team(monkeypatch):
    import src.yahoo_data_service as yds_mod

    monkeypatch.setattr(yds_mod, "multi_user_enabled", lambda: True, raising=False)
    monkeypatch.setattr(
        yds_mod, "current_user", lambda: {"username": "alice", "team_name": "Team Alice"}, raising=False
    )
    svc = _service()
    assert svc._get_user_team_name() == "Team Alice"


def test_multiuser_no_session_falls_back_to_db(monkeypatch):
    import src.yahoo_data_service as yds_mod

    monkeypatch.setattr(yds_mod, "multi_user_enabled", lambda: True, raising=False)
    monkeypatch.setattr(yds_mod, "current_user", lambda: None, raising=False)
    # current_user None → fall through to the legacy league_teams query.
    called = {"db": False}

    def _fake_conn():
        called["db"] = True
        raise RuntimeError("stop here — we only assert the DB path was taken")

    monkeypatch.setattr("src.database.get_connection", _fake_conn)
    svc = _service()
    assert svc._get_user_team_name() is None
    assert called["db"] is True


def test_single_user_mode_uses_db(monkeypatch):
    import src.yahoo_data_service as yds_mod

    monkeypatch.setattr(yds_mod, "multi_user_enabled", lambda: False, raising=False)
    captured = {"sql": None}

    class _Cur:
        def execute(self, sql, *a):
            captured["sql"] = sql
            return self

        def fetchone(self):
            return ("Team Hickey",)

    class _Conn:
        def cursor(self):
            return _Cur()

        def close(self):
            pass

    monkeypatch.setattr("src.database.get_connection", lambda: _Conn())
    svc = _service()
    assert svc._get_user_team_name() == "Team Hickey"
    assert "is_user_team = 1" in captured["sql"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_yahoo_user_team_identity.py -v`
Expected: FAIL — `test_multiuser_returns_session_team` fails because the current implementation ignores session identity and queries the DB (returns None / wrong value). `multi_user_enabled`/`current_user` are also not yet imported into the module (the `raising=False` lets the monkeypatch attach, but the function won't call them).

- [ ] **Step 3: Write minimal implementation**

In `src/yahoo_data_service.py`, add this import near the other `from src.*` imports at the top of the file:

```python
from src.auth import current_user, multi_user_enabled
```

Then replace the `_get_user_team_name` method (lines 1103-1123) with:

```python
    def _get_user_team_name(self) -> str | None:
        """Get the current user's team name.

        v2 multi-user: when the flag is on and a logged-in session user has an
        assigned team, that team is the identity (non-impersonable — it was set
        by the admin, not self-asserted). Otherwise fall back to the legacy
        league_teams.is_user_team flag so single-user v1 behavior is unchanged.
        """
        if multi_user_enabled():
            user = current_user()
            if user and user.get("team_name"):
                return user["team_name"]

        try:
            from src.database import get_connection

            conn = get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT team_name FROM league_teams WHERE is_user_team = 1")
                result = cursor.fetchone()
                return result[0] if result else None
            finally:
                conn.close()
        except Exception as exc:
            logger.warning(
                "yahoo_data_service._get_user_team_name: DB lookup for user team failed; "
                "downstream personalization will be disabled: %s",
                exc,
                exc_info=True,
            )
            return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_yahoo_user_team_identity.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/yahoo_data_service.py tests/test_yahoo_user_team_identity.py
git commit -m "feat(yahoo): per-session team identity in multi-user mode"
```

---

## Task 12: Add the auth guard to all 13 pages

**Files:**
- Modify (all 13): `pages/1_My_Team.py`, `pages/2_Line-up_Optimizer.py`, `pages/3_Closer_Monitor.py`,
  `pages/5_Matchup_Planner.py`, `pages/6_League_Standings.py`, `pages/10_Punt_Analyzer.py`,
  `pages/11_Trade_Analyzer.py`, `pages/12_Trade_Finder.py`, `pages/14_Free_Agents.py`,
  `pages/16_Player_Compare.py`, `pages/17_Leaders.py`, `pages/19_Player_Databank.py`,
  `pages/20_Draft_Simulator.py`
- Test: `tests/test_pages_have_auth_guard.py`

**Why every page:** Streamlit runs each `pages/*.py` as its own top-level script. A guard only in
`app.py` does nothing when a user deep-links to `/My_Team`. The guard must live on every page. The
uniform anchor (verified across all 13) is the `inject_custom_css()` call, which always follows
`st.set_page_config()` — so `require_auth()` goes on the line right after it (page config has run,
so `st.*` calls are legal; CSS is loaded, so the login form is themed).

- [ ] **Step 1: Write the failing test**

Create `tests/test_pages_have_auth_guard.py`:

```python
"""Structural invariant: every interactive page calls require_auth().

Streamlit executes each pages/*.py independently, so the auth gate from app.py
does not protect deep-linked pages. Any page that renders UI (detected via its
inject_custom_css() call) MUST import and call require_auth().

The admin console (00_Admin_Console.py) is exempt: it uses the stricter
require_admin() guard instead.
"""

from pathlib import Path

import pytest

_PAGES_DIR = Path(__file__).resolve().parent.parent / "pages"

_INTERACTIVE_PAGES = sorted(
    p
    for p in _PAGES_DIR.glob("*.py")
    if "inject_custom_css()" in p.read_text(encoding="utf-8")
    and p.name != "00_Admin_Console.py"
)


def test_found_the_pages():
    # Sanity: we should be guarding all 13 in-season pages.
    assert len(_INTERACTIVE_PAGES) == 13, [p.name for p in _INTERACTIVE_PAGES]


@pytest.mark.parametrize("page", _INTERACTIVE_PAGES, ids=lambda p: p.name)
def test_page_imports_require_auth(page):
    src = page.read_text(encoding="utf-8")
    assert "from src.auth import" in src and "require_auth" in src, (
        f"{page.name} must import require_auth from src.auth"
    )


@pytest.mark.parametrize("page", _INTERACTIVE_PAGES, ids=lambda p: p.name)
def test_page_calls_require_auth_after_css(page):
    src = page.read_text(encoding="utf-8")
    i_css = src.index("inject_custom_css()")
    i_auth = src.index("require_auth()")
    assert i_auth > i_css, f"{page.name}: require_auth() must follow inject_custom_css()"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pages_have_auth_guard.py -v`
Expected: FAIL — every `test_page_imports_require_auth[...]` fails (no page imports it yet).

- [ ] **Step 3: Write minimal implementation**

For **each** of the 13 pages, make two edits:

**(a)** Add the import alongside the page's existing `from src.ui_shared import ...` block (or any `src.*` import). The exact import line to add:

```python
from src.auth import require_auth
```

**(b)** Add the guard call on the line immediately after that page's `inject_custom_css()` call. For example, in `pages/1_My_Team.py` the current sequence (lines 435-436) is:

```python
inject_custom_css()
page_timer_start()
```

becomes:

```python
inject_custom_css()
require_auth()
page_timer_start()
```

For `pages/10_Punt_Analyzer.py` (which has no `page_timer_start()` — `inject_custom_css()` at line 25 is followed by page body), insert `require_auth()` on its own line directly after `inject_custom_css()`:

```python
inject_custom_css()
require_auth()
```

Apply the same two edits to all 13 files listed above. The guard is a no-op when `MULTI_USER` is off, so this is safe and inert for single-user runs.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pages_have_auth_guard.py -v`
Expected: PASS (1 sanity + 13 import + 13 ordering = 27 tests).

- [ ] **Step 5: Commit**

```bash
git add pages/ tests/test_pages_have_auth_guard.py
git commit -m "feat(pages): add flag-gated require_auth() guard to all 13 pages"
```

---

## Task 13: Admin Console page (pending approvals + team assignment + revoke)

**Files:**
- Create: `pages/00_Admin_Console.py`
- Test: `tests/test_admin_console_guarded.py`

**Scope note:** This is the *foundation* admin surface — just account lifecycle (approve pending
users + assign their Yahoo team, reassign teams, revoke). The richer admin dashboard (feature
hiding, usage analytics, feedback inbox, broadcast banner, kill switch) is **Plan 3**. Hiding this
page from non-admins in the sidebar nav requires converting HEATER's file-based nav to
`st.navigation()` — also deferred to Plan 3. For now the security boundary is `require_admin()`'s
hard-stop: a non-admin who clicks the page is bounced with an error.

- [ ] **Step 1: Write the failing test**

Create `tests/test_admin_console_guarded.py`:

```python
"""The admin console must be guarded by require_admin and render its queue."""

from pathlib import Path

_PAGE = Path(__file__).resolve().parent.parent / "pages" / "00_Admin_Console.py"


def test_admin_console_exists():
    assert _PAGE.exists(), "pages/00_Admin_Console.py must exist"


def test_admin_console_calls_require_admin():
    src = _PAGE.read_text(encoding="utf-8")
    assert "from src.auth import" in src
    assert "require_admin" in src
    # The guard must run before any lifecycle action is offered.
    i_guard = src.index("require_admin()")
    i_approve = src.index("approve_user")
    assert i_guard < i_approve, "require_admin() must gate the page before approve actions"


def test_admin_console_smoke_blocks_non_admin(monkeypatch):
    """AppTest: a logged-in non-admin hits the require_admin hard-stop."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("MULTI_USER", "1")
    at = AppTest.from_file(str(_PAGE))
    at.session_state["auth_user"] = {
        "username": "alice",
        "status": "active",
        "is_admin": 0,
        "team_name": "Team Alice",
    }
    at.session_state["_auth_bootstrap_done"] = True
    at.run()
    # require_admin() calls st.error + st.stop → an error is shown, page halts.
    assert any("access" in e.value.lower() for e in at.error), [e.value for e in at.error]


def test_admin_console_smoke_renders_for_admin(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("MULTI_USER", "1")
    at = AppTest.from_file(str(_PAGE))
    at.session_state["auth_user"] = {
        "username": "connor",
        "status": "active",
        "is_admin": 1,
        "team_name": "Team Hickey",
    }
    at.session_state["_auth_bootstrap_done"] = True
    at.run()
    assert not at.exception
    # The page title should render for an admin.
    assert any("admin" in m.value.lower() for m in at.title), [m.value for m in at.title]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_admin_console_guarded.py -v`
Expected: FAIL — `test_admin_console_exists` fails (`assert False` — file does not exist).

- [ ] **Step 3: Write minimal implementation**

Create `pages/00_Admin_Console.py`:

```python
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
                if new_team != user["team_name"] and st.button(
                    "Save team", key=f"save_{user['username']}"
                ):
                    set_user_team(user["username"], new_team)
                    st.rerun()
        with cols[2]:
            if not user["is_admin"] and st.button(
                "Revoke", key=f"revoke_{user['username']}", width="stretch"
            ):
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_admin_console_guarded.py -v`
Expected: PASS (4 tests). If the AppTest smoke tests are environment-flaky on Windows, the two structural tests (`exists`, `calls_require_admin`) are the load-bearing guards.

- [ ] **Step 5: Commit**

```bash
git add pages/00_Admin_Console.py tests/test_admin_console_guarded.py
git commit -m "feat(admin): account-lifecycle admin console (approve/assign/revoke)"
```

---

## Task 14: Consolidated back-compat regression guard

**Files:**
- Test: `tests/test_auth_backcompat.py`

**Why a dedicated guard:** the entire safety story for shipping this to a live in-season app is
"flag off ⇒ nothing changes." This test pins that contract in one place so a future refactor can't
erode it silently.

- [ ] **Step 1: Write the failing test**

Create `tests/test_auth_backcompat.py`:

```python
"""MULTI_USER off ⇒ exact v1 behavior (the load-bearing rollout guarantee)."""

import pytest


@pytest.fixture(autouse=True)
def _flag_off(monkeypatch):
    monkeypatch.delenv("MULTI_USER", raising=False)


def test_require_auth_is_noop_when_off():
    from src.auth import require_auth

    # Must not touch the DB, session, or st.* — just return.
    assert require_auth() is None


def test_require_admin_is_noop_path_when_off(monkeypatch):
    # With the flag off require_auth() returns immediately; require_admin then
    # inspects current_user(). Simulate "no session" → it should hard-stop only
    # via st.stop, which we capture. (Admin pages are still admin-gated even in
    # single-user mode — there's simply no logged-in user, so access is denied.)
    import src.auth as auth

    class _Stop(Exception):
        pass

    monkeypatch.setattr("src.auth.st.stop", lambda *a, **k: (_ for _ in ()).throw(_Stop()), raising=False)
    monkeypatch.setattr("src.auth.st.error", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(auth, "current_user", lambda: None)
    with pytest.raises(_Stop):
        auth.require_admin()


def test_user_team_name_uses_league_teams_when_off(monkeypatch):
    import src.yahoo_data_service as yds_mod

    captured = {"sql": None}

    class _Cur:
        def execute(self, sql, *a):
            captured["sql"] = sql
            return self

        def fetchone(self):
            return ("Team Hickey",)

    class _Conn:
        def cursor(self):
            return _Cur()

        def close(self):
            pass

    monkeypatch.setattr("src.database.get_connection", lambda: _Conn())
    svc = yds_mod.YahooDataService.__new__(yds_mod.YahooDataService)
    assert svc._get_user_team_name() == "Team Hickey"
    assert "is_user_team = 1" in captured["sql"]
```

- [ ] **Step 2: Run test to verify it fails OR passes**

Run: `python -m pytest tests/test_auth_backcompat.py -v`
Expected: PASS (all 3) — Tasks 9 and 11 already implement the flag-off behavior. This task adds the *guard* that locks it. (If any fails, the corresponding earlier task's flag-off path has a bug — fix it before committing.)

- [ ] **Step 3: (No new implementation)**

This task is a pure regression guard over behavior built in Tasks 9 + 11. If Step 2 passed, proceed to commit.

- [ ] **Step 4: Run the full auth suite together**

Run: `python -m pytest tests/test_auth_*.py tests/test_yahoo_user_team_identity.py tests/test_app_main_auth_gate.py tests/test_pages_have_auth_guard.py tests/test_admin_console_guarded.py -v`
Expected: PASS (all auth-foundation tests green together).

- [ ] **Step 5: Commit**

```bash
git add tests/test_auth_backcompat.py
git commit -m "test(auth): lock MULTI_USER-off = v1 behavior (back-compat guard)"
```

---

## Task 15: Document the feature in CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add the env-flag + module documentation**

In `CLAUDE.md`, under the **Local Environment** section (after the "Hooks" bullet), add:

```markdown
- **Multi-user mode (v2, additive):** Set `MULTI_USER=1` to enable league-mate self-registration + admin-approved team assignment. Off (unset) = single-user v1 behavior, byte-for-byte. Admin is seeded from `ADMIN_USERNAME` / `ADMIN_PASSWORD` / `ADMIN_TEAM_NAME` env vars (idempotent — set once). Auth lives entirely in `src/auth.py`; identity is per-session (`st.session_state["auth_user"]`) and replaces the global `league_teams.is_user_team` flag for personalized surfaces only.
```

- [ ] **Step 2: Add `src/auth.py` to the File Structure src/ listing**

In the `src/` file-structure block, under the `# In-season management` group (or a new `# Multi-user (v2)` group), add:

```
  auth.py                   — v2 multi-user: register→pending→admin-approve→active; MULTI_USER-gated; per-session identity
```

- [ ] **Step 3: Add the new structural invariants to the Structural Invariants table**

Append these rows to the **Structural Invariants (machine-checked)** table:

```markdown
| `test_app_main_auth_gate.py` | `app.py main()` calls `require_auth()` between `init_db()` and `render_splash_screen()`; imports it from `src.auth` |
| `test_pages_have_auth_guard.py` | Every interactive `pages/*.py` (those calling `inject_custom_css()`, excluding `00_Admin_Console.py`) imports + calls `require_auth()` after `inject_custom_css()`. Streamlit runs each page independently, so the gate must be per-page |
| `test_admin_console_guarded.py` | `pages/00_Admin_Console.py` calls `require_admin()` before any `approve_user` action; AppTest smoke confirms non-admins are hard-stopped |
| `test_auth_backcompat.py` | `MULTI_USER` off ⇒ `require_auth()` is a no-op and `_get_user_team_name` uses the `league_teams.is_user_team=1` query (v1 behavior preserved) |
```

- [ ] **Step 4: Verify the doc edits land cleanly**

Run: `python -m pytest tests/test_no_merge_conflict_markers.py -v`
Expected: PASS (no stray markers introduced).

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude.md): document MULTI_USER mode + auth module + invariants"
```

---

## Final verification

- [ ] **Run the full multi-user foundation suite**

Run: `python -m pytest tests/test_auth_password.py tests/test_auth_flag.py tests/test_auth_users_table.py tests/test_auth_lifecycle.py tests/test_auth_session.py tests/test_auth_backcompat.py tests/test_yahoo_user_team_identity.py tests/test_app_main_auth_gate.py tests/test_pages_have_auth_guard.py tests/test_admin_console_guarded.py -v`
Expected: ALL PASS.

- [ ] **Run lint + format**

Run: `python -m ruff check . && python -m ruff format --check .`
Expected: clean (or auto-fixable with `python -m ruff format .`).

- [ ] **Smoke-test single-user mode is unchanged (flag off)**

Run: `streamlit run app.py` with `MULTI_USER` unset.
Expected: app boots straight to the splash/setup/draft flow — NO login screen.

- [ ] **Smoke-test multi-user mode (flag on)**

Run: set `MULTI_USER=1`, `ADMIN_USERNAME=connor`, `ADMIN_PASSWORD=<pick>`, `ADMIN_TEAM_NAME="Team Hickey"`, then `streamlit run app.py`.
Expected: login/register screen appears; registering a second user lands them in `pending`; signing in as `connor` → Admin Console shows the pending user → approve + assign team → that user can now sign in and their personalized pages pin to the assigned team.

---

## Notes for the next plans (roadmap, not part of this plan)

- **Plan 2 — Feedback + usage:** per-feature feedback boxes → `feedback` table → admin inbox; lightweight `usage_events` logging. Depends on `current_user()` from this plan.
- **Plan 3 — Admin dashboard:** `feature_flags` + `audit_log` tables; feature/page/tab hide controls; usage analytics; feedback inbox UI; broadcast banner; maintenance/kill switch; view-as-user; convert file-based nav to `st.navigation()` for role-based nav (hides Admin Console from non-admins). Depends on Plans 1 + 2.
- **Plan 4 — Railway hosting + data-freshness inversion:** Railway service + persistent disk; persist `yahoo_token.json` to the volume; replace per-session `force=True` bootstrap with ONE scheduled server-side refresh (Railway cron); user sessions become read-only consumers.

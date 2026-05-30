# HEATER v2 Plan 3 — Admin Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a role-based admin dashboard (usage analytics, operational controls, CSV export) and convert HEATER's page navigation to role-aware `st.navigation()`, entirely additive behind the `MULTI_USER` flag (off = byte-for-byte v1).

**Architecture:** Five new SQLite tables (`feature_flags`, `audit_log`, `app_settings`, `sessions`, `page_visits`) created idempotently by `init_db()`. Five new `src/` modules (`audit.py`, `app_settings.py`, `feature_flags.py`, `nav.py`, plus a rewritten `usage.py`) hold the business logic; each is MULTI_USER-gated and returns inert/empty when the flag is off. When the flag is ON, `app.py` stops using Streamlit's automatic `pages/` discovery and instead calls `st.navigation()` with a role-filtered page set; when OFF it runs the existing single-user app unchanged. Three admin surfaces (`pages/_admin_console.py` renamed from `00_Admin_Console.py`, plus new `pages/_admin_analytics.py` and `pages/_admin_controls.py`) are hidden from v1 auto-discovery by their `_` prefix and routed only via `st.navigation()`.

**Tech Stack:** Python 3.12 (CI) / 3.14 (local), Streamlit ≥1.40 (already pinned; `st.navigation` needs ≥1.36, `st.fragment(run_every=...)` needs ≥1.37), SQLite via `get_connection()` (WAL + 60s busy_timeout + `sqlite3.Row` factory), pytest + `streamlit.testing.v1.AppTest`.

---

## File Structure

**New source modules:**
- `src/audit.py` — append-only admin action log. `log_action()` (write) + `list_audit()` (read with optional action filter + username join). MULTI_USER-gated on write.
- `src/app_settings.py` — key/value settings store for broadcast banner + maintenance mode. Typed getters return dicts; setters are flag-gated and each writes exactly one audit row.
- `src/feature_flags.py` — per-page enable/disable. `is_page_enabled()` / `set_page_flag()` / `list_page_flags()` / `require_page_enabled()`. "Absence = enabled" semantics.
- `src/nav.py` — `PAGE_REGISTRY` (the 13 season pages), `filter_enabled_pages()` (pure), `build_pages(user, draft_page)` (assembles `st.Page` groups for `st.navigation`). Admin group added only for admins.

**Rewritten source module:**
- `src/usage.py` — keeps existing `log_page_view()` contract (flag-off no-op, per-session dedup, `last_seen_at` bump) and ADDS session rows, page-visit dwell tracking, `bump_activity()` heartbeat, lazy idle-close, and analytics readers (`dau_series`, `most_used_pages`, `per_user_activity`, `last_seen_summary`, `session_timeline`, `page_dwell_summary`, `usage_csv`).

**Modified source modules:**
- `src/database.py` — `_init_multiuser_tables()` gains the 5 new `CREATE TABLE IF NOT EXISTS` statements (additive, idempotent).
- `src/feedback.py` — add `feedback_csv()` export helper.
- `src/auth.py` — add view-as-user impersonation (`enter_view_as` / `exit_view_as` / `is_viewing_as`).
- `app.py` — `main()` branches on the flag: single-user fast path (extracted to `render_single_user_app()`) vs. multi-user `st.navigation()` path with banners + heartbeat. Module-level `st.set_page_config` stays unconditional (app.py owns it under nav).

**Renamed page:**
- `pages/00_Admin_Console.py` → `pages/_admin_console.py` (the `_` prefix hides it from v1 auto-discovery; routed via nav under the flag). Its `st.set_page_config` becomes guarded by `if not multi_user_enabled():`.

**New pages (hidden from auto-discovery by `_` prefix):**
- `pages/_admin_analytics.py` — usage analytics surfaces + CSV download.
- `pages/_admin_controls.py` — page-visibility toggles, broadcast, maintenance, view-as, exports, audit log.

**Modified pages (×13 season pages):**
- Each gains a `multi_user_enabled`-guarded `st.set_page_config` and a `require_page_enabled("page:<stem>")` call after `require_auth()`.

**New test files:**
- `tests/test_admin_tables.py`, `tests/test_streamlit_min_version.py`, `tests/test_audit.py`, `tests/test_app_settings.py`, `tests/test_feature_flags.py`, `tests/test_nav.py`, `tests/test_view_as.py`, `tests/test_usage_sessions_dwell.py`, `tests/test_usage_analytics.py`, `tests/test_feedback_csv.py`, `tests/test_admin_analytics_page.py`, `tests/test_admin_controls_page.py`, `tests/test_admin_pages_flag_enforced.py`, `tests/test_pages_guard_set_page_config.py`, `tests/test_admin_backcompat.py` (consolidated `MULTI_USER`-off back-compat guard).

**Modified test files:**
- `tests/test_admin_console_guarded.py`, `tests/test_admin_console_feedback_tab.py`, `tests/test_pages_have_auth_guard.py`, `tests/test_pages_have_feedback_and_usage.py`, `tests/test_app_main_auth_gate.py`.

**Task order rationale (two hard constraints):**
1. The Admin Console rename (Task 3) MUST precede the nav registry (Task 7): `test_nav.py::test_registry_matches_disk` requires disk to hold exactly the 13 non-underscore season pages.
2. Task 3's discriminator flip (`not p.name.startswith("_")` in the auth/feedback structural tests) MUST precede Tasks 11–12 (which create `_admin_analytics.py` / `_admin_controls.py`), so those new admin pages are correctly exempted from the 13-page interactive-page guards.

---

### Task 1: Admin dashboard SQLite tables

**Files:**
- Modify: `src/database.py` (inside `_init_multiuser_tables`, before the closing `"""` of the `executescript` block ending at the `idx_usage_user_created` index)
- Test: `tests/test_admin_tables.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_admin_tables.py`:

```python
"""init_db() creates the v2 admin-dashboard tables (additive, idempotent)."""

import pytest


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db = tmp_path / "admin_tables.db"
    monkeypatch.setattr("src.database.DB_PATH", db)
    from src.database import init_db

    init_db()
    return db


def _table_columns(table: str) -> set[str]:
    from src.database import get_connection

    conn = get_connection()
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {r[1] for r in rows}
    finally:
        conn.close()


def test_feature_flags_table(temp_db):
    assert {"key", "enabled", "updated_by", "updated_at"} <= _table_columns("feature_flags")


def test_audit_log_table(temp_db):
    assert {"id", "admin_id", "action", "target", "detail", "created_at"} <= _table_columns("audit_log")


def test_app_settings_table(temp_db):
    assert {"key", "value", "updated_by", "updated_at"} <= _table_columns("app_settings")


def test_sessions_table(temp_db):
    assert {"session_id", "user_id", "login_at", "last_activity_at"} <= _table_columns("sessions")


def test_page_visits_table(temp_db):
    assert {
        "id",
        "session_id",
        "user_id",
        "page",
        "enter_at",
        "exit_at",
        "dwell_seconds",
    } <= _table_columns("page_visits")


def test_init_db_idempotent_for_admin_tables(temp_db):
    # A second init_db() on the same DB must not raise.
    from src.database import init_db

    init_db()
    assert "enabled" in _table_columns("feature_flags")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_admin_tables.py -v`
Expected: FAIL — `sqlite3.OperationalError: no such table: feature_flags` (PRAGMA returns no columns).

- [ ] **Step 3: Write minimal implementation**

In `src/database.py`, locate the end of the `executescript` block in `_init_multiuser_tables`. It currently ends:

```python
        CREATE INDEX IF NOT EXISTS idx_usage_user_created ON usage_events(user_id, created_at);
    """)
```

Replace that exact two-line fragment with (keep the 8-space SQL indentation):

```python
        CREATE INDEX IF NOT EXISTS idx_usage_user_created ON usage_events(user_id, created_at);

        CREATE TABLE IF NOT EXISTS feature_flags (
            key TEXT PRIMARY KEY,
            enabled INTEGER NOT NULL DEFAULT 1,
            updated_by INTEGER,
            updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            admin_id INTEGER NOT NULL,
            action TEXT NOT NULL,
            target TEXT,
            detail TEXT,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_log(created_at);

        CREATE TABLE IF NOT EXISTS app_settings (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_by INTEGER,
            updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            login_at TEXT NOT NULL,
            last_activity_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);

        CREATE TABLE IF NOT EXISTS page_visits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            page TEXT NOT NULL,
            enter_at TEXT NOT NULL,
            exit_at TEXT,
            dwell_seconds REAL
        );
        CREATE INDEX IF NOT EXISTS idx_page_visits_session ON page_visits(session_id);
    """)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_admin_tables.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/database.py tests/test_admin_tables.py
git commit -m "feat(db): add admin-dashboard tables (feature_flags/audit_log/app_settings/sessions/page_visits)"
```

---

### Task 2: Streamlit minimum-version regression lock

**Files:**
- Test: `tests/test_streamlit_min_version.py`

This task is a guard-test only. `requirements.txt` line 1 already pins `streamlit>=1.40.0`, which satisfies the `st.navigation` (≥1.36) and `st.fragment(run_every=...)` (≥1.37) requirements. The test passes immediately and locks the floor against an accidental downgrade.

- [ ] **Step 1: Write the test**

Create `tests/test_streamlit_min_version.py`:

```python
"""Admin dashboard needs Streamlit >= 1.37 (st.navigation + st.fragment run_every)."""

from importlib.metadata import version

from packaging.version import Version


def test_streamlit_at_least_1_37():
    assert Version(version("streamlit")) >= Version("1.37"), (
        "Admin dashboard nav (st.navigation) + heartbeat (st.fragment run_every) need Streamlit >= 1.37"
    )
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python -m pytest tests/test_streamlit_min_version.py -v`
Expected: PASS (regression lock, not red-green — the floor is already satisfied).

- [ ] **Step 3: Commit**

```bash
git add tests/test_streamlit_min_version.py
git commit -m "test(deps): lock streamlit floor at >=1.37 for admin dashboard nav + heartbeat"
```

---

### Task 3: Rename Admin Console to hide it from v1 auto-discovery

**Files:**
- Rename: `pages/00_Admin_Console.py` → `pages/_admin_console.py`
- Modify: `pages/_admin_console.py` (import `multi_user_enabled`; guard `st.set_page_config`)
- Modify: `tests/test_admin_console_guarded.py` (path constants)
- Modify: `tests/test_admin_console_feedback_tab.py` (AppTest path)
- Modify: `tests/test_pages_have_auth_guard.py` (discriminator + docstring)
- Modify: `tests/test_pages_have_feedback_and_usage.py` (discriminator + docstring)

Streamlit's automatic `pages/` discovery ignores files whose names start with `_`. Renaming the console hides it from the v1 sidebar; under the flag it is routed explicitly by `st.navigation()`. The discriminator flip in the two structural guards switches their "interactive page" definition from "not `00_Admin_Console.py`" to "not starting with `_`", which exempts the console and the soon-to-exist `_admin_analytics.py` / `_admin_controls.py` without changing the count of 13 season pages.

- [ ] **Step 1: Rename the file (preserves git history)**

```bash
git mv pages/00_Admin_Console.py pages/_admin_console.py
```

- [ ] **Step 2: Update the auth import in `pages/_admin_console.py`**

The import block is:

```python
from src.auth import (
    approve_user,
    get_league_team_names,
    list_users,
    require_admin,
    revoke_user,
    set_user_team,
)
```

Add `multi_user_enabled,` after `list_users,`:

```python
from src.auth import (
    approve_user,
    get_league_team_names,
    list_users,
    multi_user_enabled,
    require_admin,
    revoke_user,
    set_user_team,
)
```

- [ ] **Step 3: Guard `st.set_page_config` in `pages/_admin_console.py`**

The current block is:

```python
st.set_page_config(
    page_title="Heater | Admin Console",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)
```

Wrap it so it only runs in single-user mode (under nav, app.py owns the one allowed call):

```python
if not multi_user_enabled():
    st.set_page_config(
        page_title="Heater | Admin Console",
        page_icon="",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
```

- [ ] **Step 4: Update `tests/test_admin_console_guarded.py`**

Change the path constant (line 14) from `"00_Admin_Console.py"` to `"_admin_console.py"`:

```python
_PAGE = Path(__file__).resolve().parent.parent / "pages" / "_admin_console.py"
```

Change the existence assertion (line 29) message:

```python
    assert _PAGE.exists(), "pages/_admin_console.py must exist"
```

(The two AppTest smokes already build from `str(_PAGE)`, so they follow the constant automatically.)

- [ ] **Step 5: Update `tests/test_admin_console_feedback_tab.py`**

Read the file first to confirm the exact `AppTest.from_file(...)` call, then change the page path:

```python
at = AppTest.from_file("pages/_admin_console.py")
```

- [ ] **Step 6: Update the two structural-guard discriminators**

In `tests/test_pages_have_auth_guard.py`, the comprehension (line 20) is:

```python
    if "inject_custom_css()" in p.read_text(encoding="utf-8") and p.name != "00_Admin_Console.py"
```

Change it to:

```python
    if "inject_custom_css()" in p.read_text(encoding="utf-8") and not p.name.startswith("_")
```

Update the docstring reference (it currently calls out `00_Admin_Console.py` as exempt) to say "underscore-prefixed pages (admin surfaces) are exempt."

Apply the identical comprehension change (line 20) in `tests/test_pages_have_feedback_and_usage.py`, and update its docstring line that names `00_Admin_Console.py` to "underscore-prefixed admin pages are exempt." Leave both `assert len(_INTERACTIVE_PAGES) == 13` assertions at 13.

- [ ] **Step 7: Run the affected tests**

Run: `python -m pytest tests/test_admin_console_guarded.py tests/test_admin_console_feedback_tab.py tests/test_pages_have_auth_guard.py tests/test_pages_have_feedback_and_usage.py -v`
Expected: PASS (the console is found at its new path; both interactive-page guards still count 13 and no longer require the console).

- [ ] **Step 8: Commit**

```bash
git add pages/_admin_console.py tests/test_admin_console_guarded.py tests/test_admin_console_feedback_tab.py tests/test_pages_have_auth_guard.py tests/test_pages_have_feedback_and_usage.py
git commit -m "refactor(admin): rename Admin Console to _admin_console.py (hide from v1 auto-nav)"
```

---

### Task 4: Audit log module

**Files:**
- Create: `src/audit.py`
- Test: `tests/test_audit.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_audit.py`:

```python
"""Admin audit log: write is MULTI_USER-gated; read joins username and is newest-first."""

import pytest


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db = tmp_path / "audit.db"
    monkeypatch.setattr("src.database.DB_PATH", db)
    from src.database import init_db

    init_db()
    return db


@pytest.fixture
def _flag_on(monkeypatch):
    monkeypatch.setenv("MULTI_USER", "1")


def _seed_admin(username: str = "admin_amy") -> int:
    from src.auth import approve_user, create_user, get_user
    from src.database import get_connection

    create_user(username, "pw")
    approve_user(username, team_name="Team " + username, approved_by="test")
    uid = get_user(username)["user_id"]
    conn = get_connection()
    try:
        conn.execute("UPDATE users SET is_admin = 1 WHERE user_id = ?", (uid,))
        conn.commit()
    finally:
        conn.close()
    return uid


def _rows() -> list[dict]:
    from src.database import get_connection

    conn = get_connection()
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM audit_log ORDER BY id").fetchall()]
    finally:
        conn.close()


def test_log_action_writes_row(temp_db, _flag_on):
    from src.audit import log_action

    uid = _seed_admin()
    log_action(uid, "approve_user", target="bob", detail={"team": "Team Bob"})
    rows = _rows()
    assert len(rows) == 1
    assert rows[0]["action"] == "approve_user"
    assert rows[0]["target"] == "bob"
    assert rows[0]["detail"] == '{"team": "Team Bob"}'


def test_log_action_noop_when_flag_off(temp_db, monkeypatch):
    monkeypatch.delenv("MULTI_USER", raising=False)
    from src.audit import log_action

    log_action(1, "approve_user", target="bob")
    assert _rows() == []


def test_log_action_none_detail_stores_null(temp_db, _flag_on):
    from src.audit import log_action

    uid = _seed_admin()
    log_action(uid, "exit_view_as")
    assert _rows()[0]["detail"] is None


def test_list_audit_returns_newest_first(temp_db, _flag_on):
    from src.audit import list_audit, log_action

    uid = _seed_admin()
    log_action(uid, "first")
    log_action(uid, "second")
    actions = [r["action"] for r in list_audit()]
    assert actions[:2] == ["second", "first"]
    assert list_audit()[0]["admin_username"] == "admin_amy"


def test_list_audit_filters_by_action(temp_db, _flag_on):
    from src.audit import list_audit, log_action

    uid = _seed_admin()
    log_action(uid, "toggle_flag", target="page:1_My_Team")
    log_action(uid, "export_csv", target="usage")
    only = list_audit(action="export_csv")
    assert len(only) == 1
    assert only[0]["target"] == "usage"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_audit.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.audit'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/audit.py`:

```python
"""Admin audit log — records privileged actions. MULTI_USER-gated."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from src.auth import multi_user_enabled


def log_action(admin_id: int, action: str, target: str | None = None, detail: dict | None = None) -> None:
    if not multi_user_enabled():
        return
    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO audit_log (admin_id, action, target, detail, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                admin_id,
                action,
                target,
                json.dumps(detail) if detail is not None else None,
                datetime.now(UTC).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def list_audit(limit: int = 200, action: str | None = None) -> list[dict]:
    from src.database import get_connection

    conn = get_connection()
    try:
        sql = "SELECT a.*, u.username AS admin_username FROM audit_log a LEFT JOIN users u ON u.user_id = a.admin_id"
        params: list = []
        if action is not None:
            sql += " WHERE a.action = ?"
            params.append(action)
        sql += " ORDER BY a.created_at DESC, a.id DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_audit.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/audit.py tests/test_audit.py
git commit -m "feat(audit): MULTI_USER-gated admin action log (log_action + list_audit)"
```

---

### Task 5: App settings (broadcast + maintenance)

**Files:**
- Create: `src/app_settings.py`
- Test: `tests/test_app_settings.py`

`app_settings` stores small JSON toggles. The `value` column reads unconditionally (so banners can render even when deciding flag state), but every setter is MULTI_USER-gated and writes exactly one audit row via `log_action`. A private `_put_setting` does the upsert with no audit; the public setters add the single semantic audit entry.

- [ ] **Step 1: Write the failing test**

Create `tests/test_app_settings.py`:

```python
"""app_settings: typed broadcast/maintenance toggles; setters flag-gated + one audit row each."""

import pytest


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db = tmp_path / "settings.db"
    monkeypatch.setattr("src.database.DB_PATH", db)
    from src.database import init_db

    init_db()
    return db


@pytest.fixture
def _flag_on(monkeypatch):
    monkeypatch.setenv("MULTI_USER", "1")


def _audit_rows() -> list[dict]:
    from src.database import get_connection

    conn = get_connection()
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM audit_log ORDER BY id").fetchall()]
    finally:
        conn.close()


def test_default_disabled(temp_db):
    from src.app_settings import get_broadcast, get_maintenance

    assert get_broadcast() == {"enabled": False, "message": ""}
    assert get_maintenance() == {"enabled": False, "message": ""}


def test_set_broadcast_roundtrips(temp_db, _flag_on):
    from src.app_settings import get_broadcast, set_broadcast

    set_broadcast(True, "Heads up: trade deadline Friday", admin_id=1)
    assert get_broadcast() == {"enabled": True, "message": "Heads up: trade deadline Friday"}


def test_set_broadcast_writes_one_audit_row(temp_db, _flag_on):
    from src.app_settings import set_broadcast

    set_broadcast(True, "hi", admin_id=7)
    rows = _audit_rows()
    assert len(rows) == 1
    assert rows[0]["action"] == "set_broadcast"
    assert rows[0]["admin_id"] == 7


def test_maintenance_roundtrips(temp_db, _flag_on):
    from src.app_settings import get_maintenance, set_maintenance

    set_maintenance(True, "Back at 5pm ET", admin_id=1)
    assert get_maintenance() == {"enabled": True, "message": "Back at 5pm ET"}


def test_maintenance_audit_action(temp_db, _flag_on):
    from src.app_settings import set_maintenance

    set_maintenance(True, "down", admin_id=2)
    assert _audit_rows()[0]["action"] == "toggle_maintenance"


def test_setters_noop_when_flag_off(temp_db, monkeypatch):
    monkeypatch.delenv("MULTI_USER", raising=False)
    from src.app_settings import get_broadcast, set_broadcast

    set_broadcast(True, "should not persist", admin_id=1)
    assert get_broadcast() == {"enabled": False, "message": ""}
    assert _audit_rows() == []


def test_corrupt_json_falls_back_to_default(temp_db):
    from src.app_settings import _put_setting, get_broadcast

    _put_setting("broadcast", "not-json", admin_id=1)
    assert get_broadcast() == {"enabled": False, "message": ""}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_app_settings.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.app_settings'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/app_settings.py`:

```python
"""Application settings (broadcast banner + maintenance mode). MULTI_USER-gated writes."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from src.audit import log_action
from src.auth import multi_user_enabled

_BROADCAST_KEY = "broadcast"
_MAINTENANCE_KEY = "maintenance"
_DEFAULT_TOGGLE = {"enabled": False, "message": ""}


def _put_setting(key: str, value: str, admin_id: int) -> None:
    """Upsert a raw setting value. No audit (callers add the semantic audit row)."""
    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO app_settings (key, value, updated_by, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                           updated_by = excluded.updated_by,
                                           updated_at = excluded.updated_at
            """,
            (key, value, admin_id, datetime.now(UTC).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def get_setting(key: str, default: str | None = None) -> str | None:
    from src.database import get_connection

    conn = get_connection()
    try:
        row = conn.execute("SELECT value FROM app_settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row is not None else default
    finally:
        conn.close()


def set_setting(key: str, value: str, admin_id: int) -> None:
    if not multi_user_enabled():
        return
    _put_setting(key, value, admin_id)
    log_action(admin_id, "set_setting", target=key, detail={"value": value})


def _get_toggle(key: str) -> dict:
    raw = get_setting(key)
    if raw is None:
        return dict(_DEFAULT_TOGGLE)
    try:
        data = json.loads(raw)
        return {"enabled": bool(data.get("enabled", False)), "message": str(data.get("message", ""))}
    except (ValueError, TypeError):
        return dict(_DEFAULT_TOGGLE)


def get_broadcast() -> dict:
    return _get_toggle(_BROADCAST_KEY)


def get_maintenance() -> dict:
    return _get_toggle(_MAINTENANCE_KEY)


def set_broadcast(enabled: bool, message: str, admin_id: int) -> None:
    if not multi_user_enabled():
        return
    _put_setting(_BROADCAST_KEY, json.dumps({"enabled": bool(enabled), "message": message}), admin_id)
    log_action(admin_id, "set_broadcast", detail={"enabled": bool(enabled), "message": message})


def set_maintenance(enabled: bool, message: str, admin_id: int) -> None:
    if not multi_user_enabled():
        return
    _put_setting(_MAINTENANCE_KEY, json.dumps({"enabled": bool(enabled), "message": message}), admin_id)
    log_action(admin_id, "toggle_maintenance", detail={"enabled": bool(enabled), "message": message})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_app_settings.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/app_settings.py tests/test_app_settings.py
git commit -m "feat(app_settings): broadcast + maintenance toggles with single-audit setters"
```

---

### Task 6: Per-page feature flags

**Files:**
- Create: `src/feature_flags.py`
- Test: `tests/test_feature_flags.py`

"Absence = enabled": a page with no row is enabled. `require_page_enabled` is a hard gate for non-admins on a disabled page; admins always pass (so they can still reach a page they've disabled for others).

- [ ] **Step 1: Write the failing test**

Create `tests/test_feature_flags.py`:

```python
"""Per-page feature flags: absence = enabled; non-admins blocked on disabled pages."""

import pytest


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db = tmp_path / "flags.db"
    monkeypatch.setattr("src.database.DB_PATH", db)
    from src.database import init_db

    init_db()
    return db


@pytest.fixture
def _flag_on(monkeypatch):
    monkeypatch.setenv("MULTI_USER", "1")


def _audit_actions() -> list[str]:
    from src.database import get_connection

    conn = get_connection()
    try:
        return [r["action"] for r in conn.execute("SELECT action FROM audit_log ORDER BY id").fetchall()]
    finally:
        conn.close()


def test_absent_flag_is_enabled(temp_db, _flag_on):
    from src.feature_flags import is_page_enabled

    assert is_page_enabled("page:1_My_Team") is True


def test_set_disabled_then_enabled(temp_db, _flag_on):
    from src.feature_flags import is_page_enabled, set_page_flag

    set_page_flag("page:17_Leaders", False, admin_id=1)
    assert is_page_enabled("page:17_Leaders") is False
    set_page_flag("page:17_Leaders", True, admin_id=1)
    assert is_page_enabled("page:17_Leaders") is True


def test_list_page_flags(temp_db, _flag_on):
    from src.feature_flags import list_page_flags, set_page_flag

    set_page_flag("page:11_Trade_Analyzer", False, admin_id=1)
    flags = list_page_flags()
    assert flags["page:11_Trade_Analyzer"] is False


def test_toggle_writes_audit(temp_db, _flag_on):
    from src.feature_flags import set_page_flag

    set_page_flag("page:14_Free_Agents", False, admin_id=3)
    assert _audit_actions() == ["toggle_flag"]


def test_is_page_enabled_true_when_flag_off(temp_db, monkeypatch):
    monkeypatch.delenv("MULTI_USER", raising=False)
    from src.feature_flags import is_page_enabled

    assert is_page_enabled("page:anything") is True


def test_require_page_enabled_admin_bypass(temp_db, _flag_on, monkeypatch):
    from src import feature_flags
    from src.feature_flags import require_page_enabled, set_page_flag

    set_page_flag("page:1_My_Team", False, admin_id=1)
    monkeypatch.setattr("src.auth.current_user", lambda: {"is_admin": 1})
    # Admin on a disabled page must NOT stop.
    require_page_enabled("page:1_My_Team")


def test_require_page_enabled_noop_when_enabled(temp_db, _flag_on, monkeypatch):
    from src.feature_flags import require_page_enabled

    monkeypatch.setattr("src.auth.current_user", lambda: {"is_admin": 0})
    require_page_enabled("page:1_My_Team")  # enabled by absence → no stop


def test_require_page_enabled_stops_disabled_for_non_admin(temp_db, _flag_on, monkeypatch):
    import sys
    import types

    from src.feature_flags import require_page_enabled, set_page_flag

    set_page_flag("page:1_My_Team", False, admin_id=1)
    monkeypatch.setattr("src.auth.current_user", lambda: {"is_admin": 0})

    calls = {"error": None}

    def _stop():
        raise RuntimeError("stopped")

    fake_st = types.SimpleNamespace(error=lambda m: calls.update(error=m), stop=_stop)
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)

    with pytest.raises(RuntimeError, match="stopped"):
        require_page_enabled("page:1_My_Team")
    assert "disabled" in calls["error"].lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_feature_flags.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.feature_flags'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/feature_flags.py`:

```python
"""Per-page feature flags. Absence = enabled. MULTI_USER-gated writes; admins bypass gate."""

from __future__ import annotations

from datetime import UTC, datetime

from src.audit import log_action
from src.auth import multi_user_enabled


def is_page_enabled(key: str) -> bool:
    if not multi_user_enabled():
        return True
    from src.database import get_connection

    conn = get_connection()
    try:
        row = conn.execute("SELECT enabled FROM feature_flags WHERE key = ?", (key,)).fetchone()
        return row is None or row["enabled"] == 1
    finally:
        conn.close()


def set_page_flag(key: str, enabled: bool, admin_id: int) -> None:
    if not multi_user_enabled():
        return
    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO feature_flags (key, enabled, updated_by, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET enabled = excluded.enabled,
                                           updated_by = excluded.updated_by,
                                           updated_at = excluded.updated_at
            """,
            (key, 1 if enabled else 0, admin_id, datetime.now(UTC).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()
    log_action(admin_id, "toggle_flag", target=key, detail={"enabled": bool(enabled)})


def list_page_flags() -> dict[str, bool]:
    if not multi_user_enabled():
        return {}
    from src.database import get_connection

    conn = get_connection()
    try:
        rows = conn.execute("SELECT key, enabled FROM feature_flags").fetchall()
        return {r["key"]: r["enabled"] == 1 for r in rows}
    finally:
        conn.close()


def require_page_enabled(key: str) -> None:
    if not multi_user_enabled():
        return
    from src.auth import current_user

    user = current_user()
    if user and user.get("is_admin"):
        return
    if is_page_enabled(key):
        return
    import streamlit as st

    st.error("This page is currently disabled by the administrator.")
    st.stop()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_feature_flags.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add src/feature_flags.py tests/test_feature_flags.py
git commit -m "feat(feature_flags): per-page enable/disable with admin bypass + audit"
```

---

### Task 7: Navigation registry + page builder

**Files:**
- Create: `src/nav.py`
- Test: `tests/test_nav.py`

`build_pages` takes the draft-page callable as a parameter (rather than importing it from `app.py`) to avoid a `nav` → `app` import cycle. The registry "key" is the bare file stem; the feature-flag key is `"page:" + stem`. `filter_enabled_pages` is pure and namespace-agnostic.

- [ ] **Step 1: Write the failing test**

Create `tests/test_nav.py`:

```python
"""nav registry stays in sync with disk; build_pages assembles role-aware st.Page groups."""

from pathlib import Path

import pytest

_PAGES_DIR = Path(__file__).resolve().parent.parent / "pages"


def test_filter_enabled_pages_drops_disabled():
    from src.nav import filter_enabled_pages

    keys = ["1_My_Team", "17_Leaders"]
    assert filter_enabled_pages(keys, {"1_My_Team": False}) == ["17_Leaders"]


def test_filter_enabled_pages_absence_is_enabled():
    from src.nav import filter_enabled_pages

    assert filter_enabled_pages(["a", "b"], {}) == ["a", "b"]


def test_registry_matches_disk():
    from src.nav import PAGE_REGISTRY

    disk_stems = {p.stem for p in _PAGES_DIR.glob("*.py") if not p.name.startswith("_")}
    registry_keys = {e["key"] for e in PAGE_REGISTRY}
    assert registry_keys == disk_stems


class _FakePage:
    def __init__(self, page, title=None, default=False, icon=None):
        self.page = page
        self.title = title
        self.default = default


def _patch_pages(monkeypatch, flags):
    monkeypatch.setattr("streamlit.Page", _FakePage)
    monkeypatch.setattr("src.feature_flags.list_page_flags", lambda: flags)


def test_build_pages_groups(monkeypatch):
    _patch_pages(monkeypatch, {})
    from src.nav import build_pages

    groups = build_pages({"is_admin": 1}, draft_page=lambda: None)
    assert set(groups) == {"Home", "Season", "Admin"}
    assert len(groups["Season"]) == 13
    assert len(groups["Admin"]) == 3


def test_build_pages_no_admin_for_non_admin(monkeypatch):
    _patch_pages(monkeypatch, {})
    from src.nav import build_pages

    groups = build_pages({"is_admin": 0}, draft_page=lambda: None)
    assert "Admin" not in groups


def test_build_pages_respects_disabled_flag(monkeypatch):
    _patch_pages(monkeypatch, {"page:1_My_Team": False})
    from src.nav import build_pages

    groups = build_pages({"is_admin": 0}, draft_page=lambda: None)
    season_paths = [p.page for p in groups["Season"]]
    assert "pages/1_My_Team.py" not in season_paths
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_nav.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.nav'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/nav.py`:

```python
"""Role-aware navigation for st.navigation(). Used only when MULTI_USER is on."""

from __future__ import annotations

PAGE_REGISTRY = [
    {"key": "1_My_Team", "title": "My Team", "path": "pages/1_My_Team.py"},
    {"key": "2_Line-up_Optimizer", "title": "Lineup Optimizer", "path": "pages/2_Line-up_Optimizer.py"},
    {"key": "3_Closer_Monitor", "title": "Closer Monitor", "path": "pages/3_Closer_Monitor.py"},
    {"key": "5_Matchup_Planner", "title": "Matchup Planner", "path": "pages/5_Matchup_Planner.py"},
    {"key": "6_League_Standings", "title": "League Standings", "path": "pages/6_League_Standings.py"},
    {"key": "10_Punt_Analyzer", "title": "Punt Analyzer", "path": "pages/10_Punt_Analyzer.py"},
    {"key": "11_Trade_Analyzer", "title": "Trade Analyzer", "path": "pages/11_Trade_Analyzer.py"},
    {"key": "12_Trade_Finder", "title": "Trade Finder", "path": "pages/12_Trade_Finder.py"},
    {"key": "14_Free_Agents", "title": "Free Agents", "path": "pages/14_Free_Agents.py"},
    {"key": "16_Player_Compare", "title": "Player Compare", "path": "pages/16_Player_Compare.py"},
    {"key": "17_Leaders", "title": "Leaders", "path": "pages/17_Leaders.py"},
    {"key": "19_Player_Databank", "title": "Player Databank", "path": "pages/19_Player_Databank.py"},
    {"key": "20_Draft_Simulator", "title": "Draft Simulator", "path": "pages/20_Draft_Simulator.py"},
]

_ADMIN_PAGES = [
    {"title": "Admin Console", "path": "pages/_admin_console.py"},
    {"title": "Usage Analytics", "path": "pages/_admin_analytics.py"},
    {"title": "Admin Controls", "path": "pages/_admin_controls.py"},
]


def filter_enabled_pages(keys: list[str], flags: dict[str, bool]) -> list[str]:
    """Pure: keep keys whose flag is truthy or absent (absence = enabled)."""
    return [k for k in keys if flags.get(k, True)]


def build_pages(user: dict, draft_page) -> dict:
    import streamlit as st

    from src.feature_flags import list_page_flags

    raw = list_page_flags()
    flags = {e["key"]: raw.get("page:" + e["key"], True) for e in PAGE_REGISTRY}
    enabled_keys = filter_enabled_pages([e["key"] for e in PAGE_REGISTRY], flags)
    by_key = {e["key"]: e for e in PAGE_REGISTRY}

    home = st.Page(draft_page, title="Draft Tool", default=True)
    season = [st.Page(by_key[k]["path"], title=by_key[k]["title"]) for k in enabled_keys]
    groups = {"Home": [home], "Season": season}

    if user and user.get("is_admin"):
        groups["Admin"] = [st.Page(p["path"], title=p["title"]) for p in _ADMIN_PAGES]

    return groups
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_nav.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/nav.py tests/test_nav.py
git commit -m "feat(nav): role-aware st.navigation page registry + builder"
```

---

### Task 8: View-as-user impersonation

**Files:**
- Modify: `src/auth.py` (insert after `require_admin`, which ends at line 324)
- Test: `tests/test_view_as.py`

View-as lets an admin temporarily adopt another user's identity to debug their view. The real admin identity is stashed under a separate session key and restored on exit. Both transitions are audited.

- [ ] **Step 1: Write the failing test**

Create `tests/test_view_as.py`:

```python
"""View-as-user: admin swaps identity, restores on exit; non-admins blocked; flag-gated."""

import pytest


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db = tmp_path / "viewas.db"
    monkeypatch.setattr("src.database.DB_PATH", db)
    from src.database import init_db

    init_db()
    return db


@pytest.fixture
def state(monkeypatch):
    import src.auth as auth

    s: dict = {}
    monkeypatch.setattr(auth, "_session_state", lambda: s)
    return s


@pytest.fixture
def _flag_on(monkeypatch):
    monkeypatch.setenv("MULTI_USER", "1")


def _seed(username: str, is_admin: int = 0) -> dict:
    from src.auth import approve_user, create_user, get_user
    from src.database import get_connection

    create_user(username, "pw")
    approve_user(username, team_name="Team " + username, approved_by="test")
    if is_admin:
        conn = get_connection()
        try:
            conn.execute("UPDATE users SET is_admin = 1 WHERE username = ?", (username,))
            conn.commit()
        finally:
            conn.close()
    return get_user(username)


def _audit_actions() -> list[str]:
    from src.database import get_connection

    conn = get_connection()
    try:
        return [r["action"] for r in conn.execute("SELECT action FROM audit_log ORDER BY id").fetchall()]
    finally:
        conn.close()


def test_enter_view_as_swaps_identity(temp_db, state, _flag_on):
    from src.auth import current_user, enter_view_as

    admin = _seed("admin_amy", is_admin=1)
    _seed("bob")
    state["auth_user"] = admin
    enter_view_as("bob", admin_id=admin["user_id"])
    assert current_user()["username"] == "bob"


def test_exit_view_as_restores_admin(temp_db, state, _flag_on):
    from src.auth import current_user, enter_view_as, exit_view_as

    admin = _seed("admin_amy", is_admin=1)
    _seed("bob")
    state["auth_user"] = admin
    enter_view_as("bob", admin_id=admin["user_id"])
    exit_view_as()
    assert current_user()["username"] == "admin_amy"


def test_non_admin_cannot_view_as(temp_db, state, _flag_on):
    from src.auth import current_user, enter_view_as

    carol = _seed("carol")
    _seed("bob")
    state["auth_user"] = carol
    enter_view_as("bob", admin_id=carol["user_id"])
    assert current_user()["username"] == "carol"


def test_view_as_writes_audit(temp_db, state, _flag_on):
    from src.auth import enter_view_as, exit_view_as

    admin = _seed("admin_amy", is_admin=1)
    _seed("bob")
    state["auth_user"] = admin
    enter_view_as("bob", admin_id=admin["user_id"])
    exit_view_as()
    assert _audit_actions() == ["view_as", "exit_view_as"]


def test_view_as_noop_when_flag_off(temp_db, state, monkeypatch):
    monkeypatch.delenv("MULTI_USER", raising=False)
    from src.auth import current_user, enter_view_as

    admin = _seed("admin_amy", is_admin=1)
    _seed("bob")
    state["auth_user"] = admin
    enter_view_as("bob", admin_id=admin["user_id"])
    assert current_user()["username"] == "admin_amy"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_view_as.py -v`
Expected: FAIL — `ImportError: cannot import name 'enter_view_as' from 'src.auth'`.

- [ ] **Step 3: Write minimal implementation**

In `src/auth.py`, immediately after the end of `require_admin` (line 324, the `st.stop()` line), insert:

```python


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_view_as.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/auth.py tests/test_view_as.py
git commit -m "feat(auth): admin view-as-user impersonation (enter/exit/is_viewing_as)"
```

---

### Task 9: Usage module rewrite — sessions, dwell, analytics readers

**Files:**
- Modify (full rewrite): `src/usage.py`
- Test: `tests/test_usage_sessions_dwell.py`, `tests/test_usage_analytics.py`
- Must stay green: `tests/test_usage_logging.py`, `tests/test_feedback_usage_backcompat.py`, `tests/test_pages_have_feedback_and_usage.py`

The existing `log_page_view` contract is preserved exactly (flag-off no-op, one deduped event per `(page, action)` per session, `last_seen_at` bump). The rewrite ADDS, on top of that: a `sessions` row per browser session, `page_visits` open/close dwell tracking, a `bump_activity()` heartbeat, lazy idle-close of dangling visits at read time, and the analytics readers the admin pages consume.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_usage_sessions_dwell.py`:

```python
"""usage.py session-row creation + page-visit dwell tracking."""

import pytest

import src.usage as usage


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db = tmp_path / "usage_dwell.db"
    monkeypatch.setattr("src.database.DB_PATH", db)
    from src.database import init_db

    init_db()
    return db


@pytest.fixture
def state(monkeypatch):
    s: dict = {}
    monkeypatch.setattr(usage, "_session_state", lambda: s)
    monkeypatch.setattr(usage, "multi_user_enabled", lambda: True)
    return s


@pytest.fixture
def clock(monkeypatch):
    c = {"now": "2026-05-29T12:00:00+00:00"}
    monkeypatch.setattr(usage, "_now_iso", lambda: c["now"])
    return c


def _seed_user(username: str) -> int:
    from src.auth import approve_user, create_user, get_user

    create_user(username, "pw")
    approve_user(username, team_name="Team " + username, approved_by="test")
    return get_user(username)["user_id"]


def _sessions() -> list[dict]:
    from src.database import get_connection

    conn = get_connection()
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM sessions").fetchall()]
    finally:
        conn.close()


def _visits() -> list[dict]:
    from src.database import get_connection

    conn = get_connection()
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM page_visits ORDER BY id").fetchall()]
    finally:
        conn.close()


def test_first_view_creates_session_row(temp_db, state, clock, monkeypatch):
    uid = _seed_user("dwell_amy")
    monkeypatch.setattr(usage, "current_user", lambda: {"user_id": uid})
    usage.log_page_view("My Team")
    rows = _sessions()
    assert len(rows) == 1
    assert rows[0]["user_id"] == uid


def test_page_visit_opened_on_first_view(temp_db, state, clock, monkeypatch):
    uid = _seed_user("dwell_ben")
    monkeypatch.setattr(usage, "current_user", lambda: {"user_id": uid})
    usage.log_page_view("My Team")
    visits = _visits()
    assert len(visits) == 1
    assert visits[0]["page"] == "My Team"
    assert visits[0]["exit_at"] is None


def test_navigating_closes_prior_visit_with_dwell(temp_db, state, clock, monkeypatch):
    uid = _seed_user("dwell_cat")
    monkeypatch.setattr(usage, "current_user", lambda: {"user_id": uid})
    usage.log_page_view("My Team")
    clock["now"] = "2026-05-29T12:00:30+00:00"
    usage.log_page_view("Leaders")
    visits = _visits()
    assert len(visits) == 2
    closed = visits[0]
    assert closed["page"] == "My Team"
    assert closed["exit_at"] == "2026-05-29T12:00:30+00:00"
    assert closed["dwell_seconds"] == 30.0


def test_repeat_same_page_does_not_open_new_visit(temp_db, state, clock, monkeypatch):
    uid = _seed_user("dwell_dan")
    monkeypatch.setattr(usage, "current_user", lambda: {"user_id": uid})
    usage.log_page_view("My Team")
    usage.log_page_view("My Team")
    assert len(_visits()) == 1


def test_bump_activity_updates_last_activity(temp_db, state, clock, monkeypatch):
    uid = _seed_user("dwell_eve")
    monkeypatch.setattr(usage, "current_user", lambda: {"user_id": uid})
    usage.log_page_view("My Team")
    clock["now"] = "2026-05-29T12:05:00+00:00"
    usage.bump_activity()
    assert _sessions()[0]["last_activity_at"] == "2026-05-29T12:05:00+00:00"


def test_session_row_noop_when_flag_off(temp_db, state, clock, monkeypatch):
    monkeypatch.setattr(usage, "multi_user_enabled", lambda: False)
    usage.log_page_view("My Team")
    assert _sessions() == []
```

Create `tests/test_usage_analytics.py`:

```python
"""usage.py analytics readers: DAU, top pages, per-user, sessions/dwell, CSV, lazy-close."""

import pytest

import src.usage as usage


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db = tmp_path / "usage_analytics.db"
    monkeypatch.setattr("src.database.DB_PATH", db)
    from src.database import init_db

    init_db()
    return db


def _seed_user(username: str) -> int:
    from src.auth import approve_user, create_user, get_user

    create_user(username, "pw")
    approve_user(username, team_name="Team " + username, approved_by="test")
    return get_user(username)["user_id"]


def _event(uid, page, created_at, action="view", session_id="s1"):
    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO usage_events (user_id, page, action, session_id, created_at) VALUES (?, ?, ?, ?, ?)",
            (uid, page, action, session_id, created_at),
        )
        conn.commit()
    finally:
        conn.close()


def test_dau_series_counts_distinct_users_per_day(temp_db):
    amy = _seed_user("an_amy")
    ben = _seed_user("an_ben")
    _event(amy, "My Team", "2026-05-28T10:00:00+00:00")
    _event(ben, "Leaders", "2026-05-28T11:00:00+00:00")
    _event(amy, "Leaders", "2026-05-29T09:00:00+00:00")
    series = usage.dau_series(days=30)
    by_day = {r["day"]: r["users"] for r in series}
    assert by_day["2026-05-28"] == 2
    assert by_day["2026-05-29"] == 1


def test_most_used_pages_orders_by_views(temp_db):
    amy = _seed_user("an_cat")
    _event(amy, "Leaders", "2026-05-29T09:00:00+00:00")
    _event(amy, "Leaders", "2026-05-29T09:05:00+00:00", session_id="s2")
    _event(amy, "My Team", "2026-05-29T09:10:00+00:00")
    rows = usage.most_used_pages(days=30)
    assert rows[0]["page"] == "Leaders"
    assert rows[0]["views"] == 2


def test_per_user_activity_keeps_zero_event_users(temp_db):
    _seed_user("an_dan")  # no events
    rows = usage.per_user_activity()
    dan = [r for r in rows if r["username"] == "an_dan"][0]
    assert dan["events"] == 0


def test_session_timeline_has_duration(temp_db):
    uid = _seed_user("an_eve")
    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO sessions (session_id, user_id, login_at, last_activity_at) VALUES (?, ?, ?, ?)",
            ("sess1", uid, "2026-05-29T12:00:00+00:00", "2026-05-29T12:10:00+00:00"),
        )
        conn.commit()
    finally:
        conn.close()
    rows = usage.session_timeline()
    assert rows[0]["duration_seconds"] == 600.0


def test_page_dwell_summary_aggregates(temp_db):
    uid = _seed_user("an_fay")
    from src.database import get_connection

    conn = get_connection()
    try:
        conn.executemany(
            "INSERT INTO page_visits (session_id, user_id, page, enter_at, exit_at, dwell_seconds) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("s1", uid, "My Team", "2026-05-29T12:00:00+00:00", "2026-05-29T12:00:10+00:00", 10.0),
                ("s1", uid, "My Team", "2026-05-29T12:01:00+00:00", "2026-05-29T12:01:30+00:00", 30.0),
            ],
        )
        conn.commit()
    finally:
        conn.close()
    rows = usage.page_dwell_summary()
    mt = [r for r in rows if r["page"] == "My Team"][0]
    assert mt["total_seconds"] == 40.0
    assert mt["visits"] == 2
    assert mt["avg_seconds"] == 20.0


def test_last_seen_summary(temp_db):
    uid = _seed_user("an_gus")
    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute("UPDATE users SET last_seen_at = ? WHERE user_id = ?", ("2026-05-29T12:00:00+00:00", uid))
        conn.commit()
    finally:
        conn.close()
    rows = usage.last_seen_summary()
    gus = [r for r in rows if r["username"] == "an_gus"][0]
    assert gus["last_seen"] == "2026-05-29T12:00:00+00:00"


def test_usage_csv_roundtrip(temp_db):
    amy = _seed_user("an_hal")
    _event(amy, "My Team", "2026-05-29T09:00:00+00:00")
    csv_text = usage.usage_csv()
    lines = csv_text.strip().splitlines()
    assert lines[0].split(",")[:4] == ["id", "created_at", "user_id", "username"]
    assert "My Team" in csv_text
    assert "an_hal" in csv_text


def test_lazy_close_resolves_open_visits(temp_db):
    uid = _seed_user("an_ivy")
    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO sessions (session_id, user_id, login_at, last_activity_at) VALUES (?, ?, ?, ?)",
            ("os1", uid, "2026-05-29T12:00:00+00:00", "2026-05-29T12:05:00+00:00"),
        )
        conn.execute(
            "INSERT INTO page_visits (session_id, user_id, page, enter_at, exit_at, dwell_seconds) "
            "VALUES (?, ?, ?, ?, NULL, NULL)",
            ("os1", uid, "My Team", "2026-05-29T12:00:00+00:00"),
        )
        conn.commit()
    finally:
        conn.close()
    rows = usage.page_dwell_summary()
    mt = [r for r in rows if r["page"] == "My Team"][0]
    assert mt["total_seconds"] == 300.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_usage_sessions_dwell.py tests/test_usage_analytics.py -v`
Expected: FAIL — `AttributeError: module 'src.usage' has no attribute '_now_iso'` (and missing readers).

- [ ] **Step 3: Write the full rewrite**

Replace the entire contents of `src/usage.py` with:

```python
"""Per-session page-view logging + session/dwell tracking + analytics readers.

MULTI_USER-gated. When the flag is off, log_page_view() and bump_activity() are
no-ops (v1 byte-for-byte). When on: one deduped usage_event per (page, action)
per session, a sessions row per browser session, page_visits dwell tracking, and
a last_seen_at bump on the user. Readers power the admin analytics surfaces.
"""

from __future__ import annotations

import csv
import io
import uuid
from datetime import UTC, datetime, timedelta

from src.auth import current_user, multi_user_enabled


def _session_state():
    import streamlit as st

    return st.session_state


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _seconds_between(a_iso: str, b_iso: str) -> float:
    try:
        a = datetime.fromisoformat(a_iso)
        b = datetime.fromisoformat(b_iso)
    except (ValueError, TypeError):
        return 0.0
    return max(0.0, (b - a).total_seconds())


def _ensure_session_row(conn, session_id: str, user_id: int, now: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO sessions (session_id, user_id, login_at, last_activity_at) VALUES (?, ?, ?, ?)",
        (session_id, user_id, now, now),
    )


def _close_open_visit(conn, session_id: str, exit_at: str) -> None:
    row = conn.execute(
        "SELECT id, enter_at FROM page_visits WHERE session_id = ? AND exit_at IS NULL ORDER BY id DESC LIMIT 1",
        (session_id,),
    ).fetchone()
    if row is None:
        return
    dwell = _seconds_between(row["enter_at"], exit_at)
    conn.execute(
        "UPDATE page_visits SET exit_at = ?, dwell_seconds = ? WHERE id = ?",
        (exit_at, dwell, row["id"]),
    )


def _track_page_visit(conn, state, session_id: str, user_id: int, page: str, now: str) -> None:
    current = state.get("_current_page")
    if current == page:
        return
    if current is not None:
        _close_open_visit(conn, session_id, now)
    conn.execute(
        "INSERT INTO page_visits (session_id, user_id, page, enter_at, exit_at, dwell_seconds) "
        "VALUES (?, ?, ?, ?, NULL, NULL)",
        (session_id, user_id, page, now),
    )
    state["_current_page"] = page


def log_page_view(page: str, action: str = "view") -> None:
    if not multi_user_enabled():
        return
    user = current_user()
    if not user:
        return
    state = _session_state()
    session_id = state.get("_usage_session_id")
    if not session_id:
        session_id = uuid.uuid4().hex
        state["_usage_session_id"] = session_id
    logged = state.get("_usage_logged")
    if logged is None:
        logged = set()
        state["_usage_logged"] = logged
    dedup_key = (page, action)

    from src.database import get_connection

    user_id = user["user_id"]
    now = _now_iso()
    conn = get_connection()
    try:
        _ensure_session_row(conn, session_id, user_id, now)
        _track_page_visit(conn, state, session_id, user_id, page, now)
        conn.execute("UPDATE sessions SET last_activity_at = ? WHERE session_id = ?", (now, session_id))
        inserted = False
        if dedup_key not in logged:
            conn.execute(
                "INSERT INTO usage_events (user_id, page, action, session_id, created_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, page, action, session_id, now),
            )
            conn.execute("UPDATE users SET last_seen_at = ? WHERE user_id = ?", (now, user_id))
            inserted = True
        conn.commit()
    finally:
        conn.close()
    if inserted:
        logged.add(dedup_key)


def bump_activity() -> None:
    if not multi_user_enabled():
        return
    user = current_user()
    if not user:
        return
    state = _session_state()
    session_id = state.get("_usage_session_id")
    if not session_id:
        return
    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute("UPDATE sessions SET last_activity_at = ? WHERE session_id = ?", (_now_iso(), session_id))
        conn.commit()
    finally:
        conn.close()


def _lazy_close_open_visits(conn) -> None:
    """Resolve dangling open visits using their session's last_activity_at."""
    rows = conn.execute(
        """
        SELECT pv.id AS id, pv.enter_at AS enter_at, s.last_activity_at AS last_activity_at
        FROM page_visits pv
        JOIN sessions s ON s.session_id = pv.session_id
        WHERE pv.exit_at IS NULL
        """
    ).fetchall()
    if not rows:
        return
    for r in rows:
        dwell = _seconds_between(r["enter_at"], r["last_activity_at"])
        conn.execute(
            "UPDATE page_visits SET exit_at = ?, dwell_seconds = ? WHERE id = ?",
            (r["last_activity_at"], dwell, r["id"]),
        )
    conn.commit()


def dau_series(days: int = 30) -> list[dict]:
    from src.database import get_connection

    cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT substr(created_at, 1, 10) AS day, COUNT(DISTINCT user_id) AS users
            FROM usage_events
            WHERE created_at >= ?
            GROUP BY day
            ORDER BY day
            """,
            (cutoff,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def most_used_pages(days: int = 30) -> list[dict]:
    from src.database import get_connection

    cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT page, COUNT(*) AS views
            FROM usage_events
            WHERE created_at >= ?
            GROUP BY page
            ORDER BY views DESC, page
            """,
            (cutoff,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def per_user_activity() -> list[dict]:
    from src.database import get_connection

    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT u.user_id AS user_id,
                   u.username AS username,
                   COUNT(e.id) AS events,
                   u.last_seen_at AS last_seen
            FROM users u
            LEFT JOIN usage_events e ON e.user_id = u.user_id
            GROUP BY u.user_id
            ORDER BY events DESC, u.username
            """
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def last_seen_summary() -> list[dict]:
    from src.database import get_connection

    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT username, last_seen_at AS last_seen FROM users ORDER BY last_seen_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def session_timeline(user_id: int | None = None) -> list[dict]:
    from src.database import get_connection

    conn = get_connection()
    try:
        _lazy_close_open_visits(conn)
        sql = (
            "SELECT s.session_id AS session_id, s.user_id AS user_id, u.username AS username, "
            "s.login_at AS login_at, s.last_activity_at AS last_activity_at "
            "FROM sessions s LEFT JOIN users u ON u.user_id = s.user_id"
        )
        params: list = []
        if user_id is not None:
            sql += " WHERE s.user_id = ?"
            params.append(user_id)
        sql += " ORDER BY s.login_at DESC"
        out = []
        for r in conn.execute(sql, params).fetchall():
            d = dict(r)
            d["duration_seconds"] = _seconds_between(d["login_at"], d["last_activity_at"])
            out.append(d)
        return out
    finally:
        conn.close()


def page_dwell_summary(user_id: int | None = None) -> list[dict]:
    from src.database import get_connection

    conn = get_connection()
    try:
        _lazy_close_open_visits(conn)
        sql = (
            "SELECT page, "
            "COALESCE(SUM(dwell_seconds), 0) AS total_seconds, "
            "COALESCE(AVG(dwell_seconds), 0) AS avg_seconds, "
            "COUNT(*) AS visits "
            "FROM page_visits WHERE dwell_seconds IS NOT NULL"
        )
        params: list = []
        if user_id is not None:
            sql += " AND user_id = ?"
            params.append(user_id)
        sql += " GROUP BY page ORDER BY total_seconds DESC"
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def usage_csv() -> str:
    from src.database import get_connection

    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT e.id AS id, e.created_at AS created_at, e.user_id AS user_id,
                   u.username AS username, e.page AS page, e.action AS action,
                   e.session_id AS session_id
            FROM usage_events e
            LEFT JOIN users u ON u.user_id = e.user_id
            ORDER BY e.created_at DESC
            """
        ).fetchall()
    finally:
        conn.close()
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "created_at", "user_id", "username", "page", "action", "session_id"])
    for r in rows:
        writer.writerow([r["id"], r["created_at"], r["user_id"], r["username"], r["page"], r["action"], r["session_id"]])
    return buf.getvalue()
```

- [ ] **Step 4: Run the new + existing usage tests to verify all pass**

Run: `python -m pytest tests/test_usage_sessions_dwell.py tests/test_usage_analytics.py tests/test_usage_logging.py tests/test_feedback_usage_backcompat.py -v`
Expected: PASS (new dwell + analytics tests, plus the preserved `log_page_view` contract tests).

- [ ] **Step 5: Commit**

```bash
git add src/usage.py tests/test_usage_sessions_dwell.py tests/test_usage_analytics.py
git commit -m "feat(usage): add session rows, dwell tracking, heartbeat, analytics readers + CSV"
```

---

### Task 10: Feedback CSV export

**Files:**
- Modify: `src/feedback.py` (add `feedback_csv`)
- Test: `tests/test_feedback_csv.py`

`feedback_csv` reuses the existing `list_feedback()` reader and drops the bulky `data_state` JSON snapshot and raw `user_id` from the export, keeping the human-readable columns. `DictWriter(extrasaction="ignore")` lets us pass each full row dict and emit only the chosen fields.

- [ ] **Step 1: Write the failing test**

Create `tests/test_feedback_csv.py`:

```python
"""feedback_csv exports the inbox as human-readable CSV (header always present)."""

import pytest


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db = tmp_path / "feedback_csv.db"
    monkeypatch.setattr("src.database.DB_PATH", db)
    from src.database import init_db

    init_db()
    return db


def _seed_feedback_row():
    from datetime import UTC, datetime

    from src.auth import approve_user, create_user, get_user
    from src.database import get_connection

    create_user("fb_amy", "pw")
    approve_user("fb_amy", team_name="Team Amy", approved_by="test")
    uid = get_user("fb_amy")["user_id"]
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO feedback (user_id, page, feature_tag, message, app_version, data_state, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (uid, "Trade Analyzer", "trade", "great tool", "v2.0", "{}", "new", datetime.now(UTC).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def test_feedback_csv_has_header_and_row(temp_db):
    from src.feedback import feedback_csv

    _seed_feedback_row()
    csv_text = feedback_csv()
    lines = csv_text.strip().splitlines()
    assert lines[0].startswith("id,created_at,username,team_name,page")
    assert "great tool" in csv_text
    assert "fb_amy" in csv_text
    # data_state must NOT be exported.
    assert "data_state" not in lines[0]


def test_feedback_csv_empty_inbox_is_header_only(temp_db):
    from src.feedback import feedback_csv

    lines = feedback_csv().strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("id,created_at,username,team_name,page")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_feedback_csv.py -v`
Expected: FAIL — `ImportError: cannot import name 'feedback_csv' from 'src.feedback'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/feedback.py`:

```python
def feedback_csv() -> str:
    """Export the feedback inbox as CSV (drops the data_state snapshot + raw user_id)."""
    import csv
    import io

    fieldnames = [
        "id",
        "created_at",
        "username",
        "team_name",
        "page",
        "feature_tag",
        "status",
        "message",
        "app_version",
        "admin_notes",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in list_feedback():
        writer.writerow(row)
    return buf.getvalue()
```

(If `list_feedback` is defined below this point in the file, place `feedback_csv` after it. Confirm `list_feedback` is module-level and returns dict rows including `username`, `team_name`, and `message`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_feedback_csv.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/feedback.py tests/test_feedback_csv.py
git commit -m "feat(feedback): feedback_csv export helper for admin download"
```

---

### Task 11: Admin usage-analytics page

**Files:**
- Create: `pages/_admin_analytics.py`
- Test: `tests/test_admin_analytics_page.py`

This page consumes the Task 9 readers. It is **nav-routed only**: the leading
`_` keeps it out of v1 auto-discovery, and it deliberately omits
`st.set_page_config` (app.py owns the single call under `st.navigation()`).

- [ ] **Step 1: Write the failing smoke + guard test**

Create `tests/test_admin_analytics_page.py`:

```python
"""Smoke + guard tests for the admin usage-analytics page (nav-routed).

Mirrors test_admin_console_guarded.py: a real temp-DB user is seeded because
require_auth()/require_admin() re-validate the session identity against the DB
on every page load. The page omits set_page_config (nav owns it), so AppTest
can drive it directly.
"""

from pathlib import Path

import pytest

_PAGE = Path(__file__).resolve().parent.parent / "pages" / "_admin_analytics.py"


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db = tmp_path / "admin_analytics.db"
    monkeypatch.setattr("src.database.DB_PATH", db)
    from src.database import init_db

    init_db()
    return db


def test_admin_analytics_page_exists():
    assert _PAGE.exists(), "pages/_admin_analytics.py must exist"


def test_admin_analytics_smoke_renders_for_admin(temp_db, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from src.auth import ensure_bootstrap_admin

    monkeypatch.setenv("MULTI_USER", "1")
    monkeypatch.setenv("ADMIN_USERNAME", "connor")
    monkeypatch.setenv("ADMIN_PASSWORD", "pw")
    monkeypatch.setenv("ADMIN_TEAM_NAME", "Team Hickey")
    ensure_bootstrap_admin()

    at = AppTest.from_file(str(_PAGE))
    at.session_state["auth_user"] = {
        "username": "connor",
        "status": "active",
        "is_admin": 1,
        "team_name": "Team Hickey",
    }
    at.session_state["_auth_bootstrap_done"] = True
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert any("analytics" in m.value.lower() for m in at.title), [m.value for m in at.title]


def test_admin_analytics_smoke_blocks_non_admin(temp_db, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from src.auth import approve_user, create_user

    monkeypatch.setenv("MULTI_USER", "1")
    create_user("alice", "pw", display_name="Alice")
    approve_user("alice", team_name="Team Alice", approved_by="test")

    at = AppTest.from_file(str(_PAGE))
    at.session_state["auth_user"] = {
        "username": "alice",
        "status": "active",
        "is_admin": 0,
        "team_name": "Team Alice",
    }
    at.session_state["_auth_bootstrap_done"] = True
    at.run()
    assert any("access" in e.value.lower() for e in at.error), [e.value for e in at.error]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_admin_analytics_page.py -v`
Expected: FAIL — `AssertionError: pages/_admin_analytics.py must exist` (file not created yet).

- [ ] **Step 3: Write the page**

Create `pages/_admin_analytics.py`:

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_admin_analytics_page.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add pages/_admin_analytics.py tests/test_admin_analytics_page.py
git commit -m "feat(admin): usage-analytics page (nav-routed, admin-gated)"
```

---

### Task 12: Admin operational-controls page

**Files:**
- Create: `pages/_admin_controls.py`
- Test: `tests/test_admin_controls_page.py`

This page wires together every operational module built in Tasks 4-10: per-page
feature flags, broadcast banner, maintenance mode, view-as, CSV exports (each
export logs an audit row via `on_click`), and the audit-log table. Nav-routed
only; no `st.set_page_config`.

- [ ] **Step 1: Write the failing smoke + guard test**

Create `tests/test_admin_controls_page.py`:

```python
"""Smoke + guard tests for the admin operational-controls page (nav-routed)."""

from pathlib import Path

import pytest

_PAGE = Path(__file__).resolve().parent.parent / "pages" / "_admin_controls.py"


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db = tmp_path / "admin_controls.db"
    monkeypatch.setattr("src.database.DB_PATH", db)
    from src.database import init_db

    init_db()
    return db


def test_admin_controls_page_exists():
    assert _PAGE.exists(), "pages/_admin_controls.py must exist"


def test_admin_controls_smoke_renders_for_admin(temp_db, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from src.auth import ensure_bootstrap_admin

    monkeypatch.setenv("MULTI_USER", "1")
    monkeypatch.setenv("ADMIN_USERNAME", "connor")
    monkeypatch.setenv("ADMIN_PASSWORD", "pw")
    monkeypatch.setenv("ADMIN_TEAM_NAME", "Team Hickey")
    ensure_bootstrap_admin()

    at = AppTest.from_file(str(_PAGE))
    at.session_state["auth_user"] = {
        "username": "connor",
        "status": "active",
        "is_admin": 1,
        "team_name": "Team Hickey",
    }
    at.session_state["_auth_bootstrap_done"] = True
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert any("controls" in m.value.lower() for m in at.title), [m.value for m in at.title]


def test_admin_controls_smoke_blocks_non_admin(temp_db, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from src.auth import approve_user, create_user

    monkeypatch.setenv("MULTI_USER", "1")
    create_user("alice", "pw", display_name="Alice")
    approve_user("alice", team_name="Team Alice", approved_by="test")

    at = AppTest.from_file(str(_PAGE))
    at.session_state["auth_user"] = {
        "username": "alice",
        "status": "active",
        "is_admin": 0,
        "team_name": "Team Alice",
    }
    at.session_state["_auth_bootstrap_done"] = True
    at.run()
    assert any("access" in e.value.lower() for e in at.error), [e.value for e in at.error]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_admin_controls_page.py -v`
Expected: FAIL — `AssertionError: pages/_admin_controls.py must exist`.

- [ ] **Step 3: Write the page**

Create `pages/_admin_controls.py`:

```python
"""Admin-only operational controls (nav-routed; MULTI_USER only).

The leading underscore hides this file from v1 auto-discovery; src/nav.py routes
here via st.Page for admins when MULTI_USER is on. No set_page_config — app.py
owns the single call under st.navigation().
"""

import streamlit as st

from src.app_settings import get_broadcast, get_maintenance, set_broadcast, set_maintenance
from src.audit import list_audit, log_action
from src.auth import current_user, enter_view_as, require_admin
from src.feedback import feedback_csv
from src.feature_flags import list_page_flags, set_page_flag
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_admin_controls_page.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add pages/_admin_controls.py tests/test_admin_controls_page.py
git commit -m "feat(admin): operational-controls page (flags, broadcast, maintenance, view-as, exports, audit)"
```

---

### Task 13: Per-page flag enforcement + set_page_config guard (13 season pages)

**Files:**
- Modify: all 13 interactive season pages (3 edits each — see recipe in Step 3)
- Create: `tests/test_pages_guard_set_page_config.py`
- Create: `tests/test_admin_pages_flag_enforced.py`

Two structural concerns land together here because they touch the same prologue lines in the same 13 files:

1. **`set_page_config` guard** — under `st.navigation()` (flag ON) `app.py` owns the single `st.set_page_config`. Each page must therefore call its own only when the flag is OFF (v1 auto-discovery mode).
2. **Flag enforcement** — a page an admin has disabled must hard-stop non-admins even via direct URL, so each page calls `require_page_enabled("page:<stem>")` between `require_auth()` and `log_page_view()`.

Discovery rules match `test_pages_have_auth_guard.py`: a page is "interactive" if it calls `inject_custom_css()` and its filename does not start with `_` (so the renamed `_admin_console.py` from Task 3 and the new `_admin_*.py` pages from Tasks 11-12 are excluded; the count stays exactly 13).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pages_guard_set_page_config.py`:

```python
"""Structural guard: interactive pages call st.set_page_config only when
MULTI_USER is OFF, so st.navigation() (flag ON) owns page config via app.py.

Under st.navigation() Streamlit expects exactly one set_page_config (app.py's).
Each page therefore guards its own behind `if not multi_user_enabled():`.
Mirrors test_pages_have_auth_guard.py discovery rules.
"""

from pathlib import Path

import pytest

_PAGES_DIR = Path(__file__).resolve().parent.parent / "pages"

_INTERACTIVE_PAGES = sorted(
    p
    for p in _PAGES_DIR.glob("*.py")
    if "inject_custom_css()" in p.read_text(encoding="utf-8") and not p.name.startswith("_")
)


def test_found_the_pages():
    assert len(_INTERACTIVE_PAGES) == 13, [p.name for p in _INTERACTIVE_PAGES]


@pytest.mark.parametrize("page", _INTERACTIVE_PAGES, ids=lambda p: p.name)
def test_page_imports_multi_user_enabled(page):
    src = page.read_text(encoding="utf-8")
    assert "multi_user_enabled" in src, f"{page.name} must import multi_user_enabled"


@pytest.mark.parametrize("page", _INTERACTIVE_PAGES, ids=lambda p: p.name)
def test_set_page_config_guarded(page):
    lines = page.read_text(encoding="utf-8").splitlines()
    cfg_idx = next((i for i, ln in enumerate(lines) if "st.set_page_config(" in ln), None)
    assert cfg_idx is not None, f"{page.name}: no st.set_page_config( call found"
    # Walk back to the previous non-blank line; it must be the flag guard.
    j = cfg_idx - 1
    while j >= 0 and not lines[j].strip():
        j -= 1
    assert lines[j].strip() == "if not multi_user_enabled():", (
        f"{page.name}: st.set_page_config must be guarded by "
        f"'if not multi_user_enabled():' (found {lines[j].strip()!r})"
    )
```

Create `tests/test_admin_pages_flag_enforced.py`:

```python
"""Structural guard: each interactive season page enforces its feature flag
via require_page_enabled("page:<stem>") AFTER require_auth() and BEFORE
log_page_view(). A disabled page must hard-stop non-admins even on direct nav.

Mirrors test_pages_have_auth_guard.py / test_pages_have_feedback_and_usage.py
discovery rules (calls inject_custom_css(); name does not start with "_").
"""

from pathlib import Path

import pytest

_PAGES_DIR = Path(__file__).resolve().parent.parent / "pages"

_INTERACTIVE_PAGES = sorted(
    p
    for p in _PAGES_DIR.glob("*.py")
    if "inject_custom_css()" in p.read_text(encoding="utf-8") and not p.name.startswith("_")
)


def test_found_the_pages():
    assert len(_INTERACTIVE_PAGES) == 13, [p.name for p in _INTERACTIVE_PAGES]


@pytest.mark.parametrize("page", _INTERACTIVE_PAGES, ids=lambda p: p.name)
def test_page_imports_require_page_enabled(page):
    src = page.read_text(encoding="utf-8")
    assert "from src.feature_flags import require_page_enabled" in src, (
        f"{page.name} must import require_page_enabled"
    )


@pytest.mark.parametrize("page", _INTERACTIVE_PAGES, ids=lambda p: p.name)
def test_page_calls_require_page_enabled_with_correct_key(page):
    src = page.read_text(encoding="utf-8")
    stem = page.stem
    assert f'require_page_enabled("page:{stem}")' in src, (
        f'{page.name} must call require_page_enabled("page:{stem}")'
    )


@pytest.mark.parametrize("page", _INTERACTIVE_PAGES, ids=lambda p: p.name)
def test_require_page_enabled_between_auth_and_log(page):
    src = page.read_text(encoding="utf-8")
    i_auth = src.index("require_auth()")
    i_gate = src.index("require_page_enabled(")
    i_log = src.index("log_page_view(")
    assert i_auth < i_gate < i_log, (
        f"{page.name}: require_page_enabled() must sit after require_auth() and before log_page_view()"
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_pages_guard_set_page_config.py tests/test_admin_pages_flag_enforced.py -v`
Expected: FAIL — `test_set_page_config_guarded` and the `require_page_enabled` checks fail for all 13 pages (the guard/import/call are not present yet). `test_found_the_pages` passes.

- [ ] **Step 3: Apply the 3-edit recipe to all 13 pages**

Apply these three edits to every page in the table below. `<stem>` is the filename without `.py` (e.g. `3_Closer_Monitor`).

**Edit A — import `multi_user_enabled`.** Extend the page's existing `from src.auth import require_auth` line:

```python
from src.auth import multi_user_enabled, require_auth
```

**Edit B — guard `set_page_config`.** Wrap the existing `st.set_page_config(...)` call (single- or multi-line) in `if not multi_user_enabled():`, indenting every line of the call by 4 spaces. Do not change the call's arguments.

**Edit C — enforce the flag.** Add the import (place it next to the existing `from src.usage import log_page_view`):

```python
from src.feature_flags import require_page_enabled
```

then insert a `require_page_enabled("page:<stem>")` call on the line immediately after `require_auth()`, matching the surrounding indentation:

```python
require_auth()
require_page_enabled("page:<stem>")
log_page_view("...")
```

**Worked example 1 — `pages/3_Closer_Monitor.py` (single-line config):**

Edit A:
```python
# old
from src.auth import require_auth
# new
from src.auth import multi_user_enabled, require_auth
```

Edit B:
```python
# old
st.set_page_config(page_title="Heater | Closer Monitor", page_icon="", layout="wide", initial_sidebar_state="collapsed")
# new
if not multi_user_enabled():
    st.set_page_config(page_title="Heater | Closer Monitor", page_icon="", layout="wide", initial_sidebar_state="collapsed")
```

Edit C — add the import beside `from src.usage import log_page_view`:
```python
from src.feature_flags import require_page_enabled
```
then:
```python
# old
require_auth()
log_page_view("Closer Monitor")
# new
require_auth()
require_page_enabled("page:3_Closer_Monitor")
log_page_view("Closer Monitor")
```

**Worked example 2 — `pages/2_Line-up_Optimizer.py` (multi-line config):**

Edit B (indent the whole block under the guard):
```python
# old
st.set_page_config(
    page_title="Heater | Line-up Optimizer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)
# new
if not multi_user_enabled():
    st.set_page_config(
        page_title="Heater | Line-up Optimizer",
        page_icon="",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
```

Edit C:
```python
# old
require_auth()
log_page_view("Lineup")
# new
require_auth()
require_page_enabled("page:2_Line-up_Optimizer")
log_page_view("Lineup")
```

**Per-page reference table** (pre-edit anchors; line numbers shift as you edit — match on the text anchors, not the numbers):

| Page file | Edit C call | set_page_config shape | require_auth indent |
|-----------|-------------|------------------------|----------------------|
| `1_My_Team.py` | `require_page_enabled("page:1_My_Team")` | single-line (L434) | module (col 0) |
| `2_Line-up_Optimizer.py` | `require_page_enabled("page:2_Line-up_Optimizer")` | multi-line (L129-134) | module (col 0) |
| `3_Closer_Monitor.py` | `require_page_enabled("page:3_Closer_Monitor")` | single-line (L25) | module (col 0) |
| `5_Matchup_Planner.py` | `require_page_enabled("page:5_Matchup_Planner")` | multi-line (L91) | module (col 0) |
| `6_League_Standings.py` | `require_page_enabled("page:6_League_Standings")` | multi-line (L237) | module (col 0) |
| `10_Punt_Analyzer.py` | `require_page_enabled("page:10_Punt_Analyzer")` | single-line (L25) | module (col 0) |
| `11_Trade_Analyzer.py` | `require_page_enabled("page:11_Trade_Analyzer")` | single-line (L46) | module (col 0) |
| `12_Trade_Finder.py` | `require_page_enabled("page:12_Trade_Finder")` | multi-line (L44) | **indented (4 spaces)** |
| `14_Free_Agents.py` | `require_page_enabled("page:14_Free_Agents")` | multi-line (L173) | module (col 0) |
| `16_Player_Compare.py` | `require_page_enabled("page:16_Player_Compare")` | single-line (L48) | module (col 0) |
| `17_Leaders.py` | `require_page_enabled("page:17_Leaders")` | single-line (L65) | module (col 0) |
| `19_Player_Databank.py` | `require_page_enabled("page:19_Player_Databank")` | multi-line (L39) | module (col 0) |
| `20_Draft_Simulator.py` | `require_page_enabled("page:20_Draft_Simulator")` | multi-line (L50) | module (col 0) |

**Special case — `pages/12_Trade_Finder.py`:** its `require_auth()` / `log_page_view()` sit inside an indented block (4 spaces, lines 137-138). Insert `require_page_enabled("page:12_Trade_Finder")` at that same 4-space indent. Its `st.set_page_config(` is at module level (column 0), so the Edit B guard wrap there is un-indented exactly like the other pages.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_pages_guard_set_page_config.py tests/test_admin_pages_flag_enforced.py -v`
Expected: PASS — 27 + 40 = 67 tests (1 + 13 + 13, and 1 + 13 + 13 + 13).

Re-run the existing page-prologue guards to confirm no regression:

Run: `python -m pytest tests/test_pages_have_auth_guard.py tests/test_pages_have_feedback_and_usage.py -v`
Expected: PASS (unchanged — the recipe inserts between existing anchors).

- [ ] **Step 5: Commit**

```bash
git add pages/ tests/test_pages_guard_set_page_config.py tests/test_admin_pages_flag_enforced.py
git commit -m "feat(pages): guard set_page_config + enforce per-page flags under MULTI_USER nav"
```

---

### Task 14: app.py `st.navigation()` entry point under MULTI_USER

**Files:**
- Modify: `app.py` (imports near line 24; `main()` at lines 2450-2492)
- Test: `tests/test_app_main_auth_gate.py` (rewrite)

This is the keystone task: it turns the flag into the switch between Streamlit's automatic `pages/` sidebar (v1) and a role-aware `st.navigation()` (v2). The existing `main()` body — splash, refresh button, setup/draft state machine — is extracted verbatim into `render_single_user_app()`. That function does double duty: when the flag is OFF, `main()` calls it directly and returns (byte-for-byte v1); when ON, it is passed to `build_pages()` as the default **"Draft Tool"** Home page (Task 7 takes the draft page as a callable precisely so `src/nav.py` never imports `app.py`).

`app.py`'s module-level `st.set_page_config` (line 90) is left **unchanged** — under `st.navigation()` Streamlit expects exactly one config call, and this is it. The 13 season pages already guard their own behind `if not multi_user_enabled():` (Task 13), so under nav only app.py's runs.

The four flag-on helpers are thin: a view-as banner (Task 8 contract — `is_viewing_as()` returns the stashed admin dict, `current_user()` returns the impersonated target), a broadcast banner and maintenance gate (Task 5 readers), and a `@st.fragment(run_every="60s")` heartbeat that calls `bump_activity()` (Task 9) so `sessions.last_activity_at` stays fresh for the lazy idle-close.

- [ ] **Step 1: Rewrite the test to lock the new entry-point shape**

Replace the entire contents of `tests/test_app_main_auth_gate.py` with:

```python
"""Structural guard: app.py main() is the MULTI_USER entry point.

Flag OFF -> main() runs the single-user app directly and returns (v1 path).
Flag ON  -> main() gates auth, then hands off to st.navigation().run().

The AST tests lock the *placement* of require_auth() and the single-user
fast-path in main(); the AppTest smoke at the bottom locks the *runtime
effect*: with MULTI_USER on and no session, app.py must render the login
screen and stop BEFORE st.navigation() / the splash bootstrap ever runs.
"""

import ast
from pathlib import Path

import pytest

_APP = Path(__file__).resolve().parent.parent / "app.py"


def _main_node(src: str):
    """The ast.FunctionDef node for main()."""
    tree = ast.parse(src)
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")


def _main_body_calls():
    """Ordered list of call-expression names inside main()."""
    main = _main_node(_APP.read_text(encoding="utf-8"))
    calls = []
    for node in ast.walk(main):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                calls.append(func.id)
            elif isinstance(func, ast.Attribute):
                calls.append(func.attr)
    return calls


def _main_src():
    """Source text of main() only, so substring checks ignore defs elsewhere."""
    src = _APP.read_text(encoding="utf-8")
    return ast.get_source_segment(src, _main_node(src))


def test_require_auth_called_in_main():
    assert "require_auth" in _main_body_calls()


def test_require_auth_after_init_db_before_nav():
    body = _main_src()
    i_init = body.index("init_db()")
    i_auth = body.index("require_auth()")
    i_nav = body.index("st.navigation(")
    assert i_init < i_auth < i_nav, "require_auth() must sit between init_db() and st.navigation()"


def test_main_uses_st_navigation():
    calls = _main_body_calls()
    assert "navigation" in calls, "main() must call st.navigation()"
    assert "run" in calls, "main() must call .run() on the navigation object"


def test_main_has_single_user_fast_path():
    body = _main_src()
    assert "if not multi_user_enabled():" in body
    assert "render_single_user_app()" in body
    i_branch = body.index("if not multi_user_enabled():")
    i_return = body.index("return")
    i_nav = body.index("st.navigation(")
    assert i_branch < i_return < i_nav, "flag-off fast path must return before st.navigation()"


def test_require_auth_imported():
    src = _APP.read_text(encoding="utf-8")
    assert "from src.auth import" in src and "require_auth" in src


@pytest.fixture
def _temp_db(tmp_path, monkeypatch):
    """Isolate app.py's init_db() onto a throwaway SQLite file."""
    monkeypatch.setattr("src.database.DB_PATH", tmp_path / "app_auth_gate.db")


def test_flag_on_no_session_renders_login_and_stops(_temp_db, monkeypatch):
    """MULTI_USER on + no session => login screen shown, nav/splash never reached."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("MULTI_USER", "1")
    monkeypatch.delenv("ADMIN_USERNAME", raising=False)  # no bootstrap admin
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)

    at = AppTest.from_file(str(_APP))
    # Generous timeout: app.py's first import pulls heavy ML/data deps
    # (pybaseball, statsapi, PyMC). The gate itself halts in milliseconds.
    at.run(timeout=60)

    assert not at.exception, [str(e) for e in at.exception]
    # require_auth() rendered the sign-in title and st.stop()'d the script.
    assert any("sign in" in t.value.lower() for t in at.title), [t.value for t in at.title]
    # The splash/bootstrap path was never entered — proof the gate halted early.
    assert "bootstrap_complete" not in at.session_state
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_app_main_auth_gate.py -v`
Expected: FAIL/ERROR — `test_main_uses_st_navigation`, `test_main_has_single_user_fast_path`, and `test_require_auth_after_init_db_before_nav` all fail (current `main()` has no `st.navigation(` and no `render_single_user_app()` fast path; `.index("st.navigation(")` raises `ValueError`). `test_require_auth_called_in_main`, `test_require_auth_imported`, and the AppTest smoke still pass.

- [ ] **Step 3a: Add the imports**

Apply these three edits to the `from src.* import` block near the top of `app.py`. Names within `src.auth` and the new lines must stay in isort order (`I` is enabled in ruff lint — see `pyproject.toml`).

Edit 1 — replace the single-line auth import with the expanded form, prefixed by the new `app_settings` import:
```python
# old
from src.auth import multi_user_enabled, require_auth
# new
from src.app_settings import get_broadcast, get_maintenance
from src.auth import (
    current_user,
    exit_view_as,
    is_viewing_as,
    multi_user_enabled,
    require_auth,
)
```

Edit 2 — add the `nav` import (alphabetically before `src.simulation`):
```python
# old
from src.simulation import DraftSimulator, compute_team_preferences, detect_position_run
# new
from src.nav import build_pages
from src.simulation import DraftSimulator, compute_team_preferences, detect_position_run
```

Edit 3 — add the `usage` import (alphabetically before `src.valuation`):
```python
# old
from src.valuation import (
    LeagueConfig,
# new
from src.usage import bump_activity
from src.valuation import (
    LeagueConfig,
```

- [ ] **Step 3b: Extract `render_single_user_app`, add helpers, rewrite `main()`**

Replace the entire current `main()` definition (lines 2450-2492, from `def main():` down to the end of the `elif st.session_state.page == "draft":` block — but NOT the `if __name__ == "__main__":` block below it) with:

```python
def render_single_user_app():
    """The v1 single-user experience: splash/bootstrap, refresh, draft flow.

    Called directly when MULTI_USER is off, and registered as the default
    "Draft Tool" Home page under st.navigation() when MULTI_USER is on.
    """
    render_splash_screen()

    # Force Refresh button in sidebar (only after bootstrap is done)
    # Refreshes ALL data sources (force=True), clears Streamlit cache_data so
    # stale in-memory player pools don't survive the refresh, preserves the
    # Data Status panel by assigning the new results (previous bug: set to
    # None, which blanked the panel), and records the new elapsed load time.
    if st.session_state.get("bootstrap_complete"):
        with st.sidebar:
            if st.button("Refresh All Data", key="force_refresh_btn", width="stretch"):
                import time as _time

                with st.spinner("Refreshing all data sources..."):
                    yahoo_client = st.session_state.get("yahoo_client")
                    try:
                        st.cache_data.clear()
                    except Exception:
                        pass
                    _rf_start = _time.monotonic()
                    results = bootstrap_all_data(yahoo_client=yahoo_client, force=True)
                    _rf_elapsed = _time.monotonic() - _rf_start
                    st.session_state["bootstrap_results"] = results
                    st.session_state["bootstrap_elapsed_secs"] = float(_rf_elapsed)
                    st.session_state["bootstrap_elapsed_hms"] = _format_elapsed_hms(_rf_elapsed)
                st.rerun()

    if st.session_state.page == "setup":
        render_setup_page()
    elif st.session_state.page == "draft":
        render_draft_page()


def _render_view_as_banner():
    """Show an exit-able banner whenever an admin is impersonating a user."""
    if is_viewing_as() is None:
        return
    target = (current_user() or {}).get("username", "?")
    col1, col2 = st.columns([5, 1])
    with col1:
        st.warning(f"Viewing as **{target}** — you are seeing this user's view.")
    with col2:
        if st.button("Exit view-as", key="_exit_view_as_btn", width="stretch"):
            exit_view_as()
            st.rerun()


def _render_broadcast_banner():
    """Render the admin broadcast message at the top of the page, if enabled."""
    bc = get_broadcast()
    if bc.get("enabled") and bc.get("message"):
        st.info(bc["message"])


def _enforce_maintenance_gate():
    """Hard-stop non-admins when maintenance mode is on; admins pass through."""
    mt = get_maintenance()
    if not mt.get("enabled"):
        return
    if (current_user() or {}).get("is_admin"):
        return
    st.warning(mt.get("message") or "HEATER is temporarily down for maintenance. Check back soon.")
    st.stop()


@st.fragment(run_every="60s")
def _heartbeat_fragment():
    """Bump the session's last_activity_at every 60s (powers idle-close)."""
    bump_activity()


def main():
    init_session()
    inject_custom_css()
    init_db()

    # v2 multi-user gate. Flag OFF -> byte-for-byte v1: run the single-user app
    # directly (Streamlit's automatic pages/ sidebar nav, no auth). Flag ON ->
    # role-aware st.navigation() with auth, banners, maintenance gate, heartbeat.
    if not multi_user_enabled():
        render_single_user_app()
        return

    require_auth()
    _render_view_as_banner()
    _render_broadcast_banner()
    _enforce_maintenance_gate()
    _heartbeat_fragment()
    st.navigation(build_pages(current_user(), render_single_user_app)).run()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_app_main_auth_gate.py -v`
Expected: PASS (6 tests — the 4 AST tests, the import test, and the AppTest smoke).

Confirm the flag-off path is unbroken with a quick import + lint check:

Run: `python -m ruff check app.py && python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('ok')"`
Expected: ruff reports no errors (imports in isort order); prints `ok`.

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_app_main_auth_gate.py
git commit -m "feat(app): st.navigation() entry point under MULTI_USER"
```

---

### Task 15: Consolidated MULTI_USER-off back-compat guard

**Files:**
- Test: `tests/test_admin_backcompat.py` (create)

This is the consolidated flag-off back-compat guard the spec's File Manifest (§11) and §9.2 call for — the Plan 3 sibling of `tests/test_feedback_usage_backcompat.py` (Plan 2) and `tests/test_auth_backcompat.py` (Plan 1). It proves, in ONE place, that the entire Plan 3 admin surface is inert when `MULTI_USER` is off: every admin helper returns before opening a DB connection (verified by a connection spy), the five new tables stay empty, and `app.py`'s `main()` early-returns on the flag-off branch before it can reach `st.navigation()`.

It is a **regression guard, not new behavior** — Tasks 4, 5, 6, 9, and 14 already built the flag-off early-returns this test asserts. Because those tasks are merged before this one runs, the test PASSES the moment it is written. If any assertion FAILS, an earlier task regressed back-compat: fix that task, do not weaken this test. (The codebase has many such guard tests with no implementation step, e.g. `test_drop_candidate_diversity.py`.)

- [ ] **Step 1: Write the back-compat guard test**

Create `tests/test_admin_backcompat.py`:

```python
"""With MULTI_USER off, the whole Plan 3 admin surface is inert (v1 byte-for-byte).

Consolidated back-compat guard. A connection spy proves the flag-off admin
helpers never open a DB connection, the five Plan 3 tables stay empty, and
app.py keeps the automatic-nav (v1) entry path instead of st.navigation().
Mirrors tests/test_feedback_usage_backcompat.py and tests/test_auth_backcompat.py.
"""

import ast
import pathlib
import sqlite3

import pytest

import src.app_settings as app_settings
import src.audit as audit
import src.feature_flags as feature_flags
import src.usage as usage

_TABLES = ("feature_flags", "audit_log", "app_settings", "sessions", "page_visits")


@pytest.fixture(autouse=True)
def _flag_off(monkeypatch):
    monkeypatch.delenv("MULTI_USER", raising=False)


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db = tmp_path / "backcompat.db"
    monkeypatch.setattr("src.database.DB_PATH", db)
    from src.database import init_db

    init_db()
    return db


class _ConnSpy:
    """Wraps get_connection and counts how many times it is opened."""

    def __init__(self, real):
        self._real = real
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self._real()


@pytest.fixture
def conn_spy(temp_db, monkeypatch):
    # All admin helpers do a function-local `from src.database import get_connection`,
    # so patching the source symbol catches every open at call time.
    import src.database as database

    spy = _ConnSpy(database.get_connection)
    monkeypatch.setattr(database, "get_connection", spy)
    return spy


def _row_counts(db) -> dict:
    conn = sqlite3.connect(db)
    try:
        return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in _TABLES}
    finally:
        conn.close()


def test_is_page_enabled_true_without_db(conn_spy):
    assert feature_flags.is_page_enabled("page:1_My_Team") is True
    feature_flags.require_page_enabled("page:1_My_Team")  # must not raise, must not touch DB
    assert conn_spy.calls == 0


def test_log_action_noop(conn_spy):
    audit.log_action(admin_id=1, action="toggle_flag", target="page:1_My_Team", detail={"enabled": False})
    assert conn_spy.calls == 0


def test_set_setting_noop(conn_spy):
    app_settings.set_setting("broadcast", "hello league", admin_id=1)
    assert conn_spy.calls == 0


def test_session_dwell_logger_noop(conn_spy, monkeypatch):
    state: dict = {}
    monkeypatch.setattr(usage, "_session_state", lambda: state)
    monkeypatch.setattr(usage, "current_user", lambda: {"user_id": 1})
    usage.log_page_view("My Team")
    usage.bump_activity()
    assert conn_spy.calls == 0
    assert "_usage_session_id" not in state


def test_five_tables_stay_empty(conn_spy, temp_db, monkeypatch):
    state: dict = {}
    monkeypatch.setattr(usage, "_session_state", lambda: state)
    monkeypatch.setattr(usage, "current_user", lambda: {"user_id": 1})
    feature_flags.is_page_enabled("page:1_My_Team")
    feature_flags.set_page_flag("page:1_My_Team", enabled=False, admin_id=1)
    audit.log_action(admin_id=1, action="toggle_flag", target="page:1_My_Team", detail={"enabled": False})
    app_settings.set_setting("broadcast", "hi", admin_id=1)
    usage.log_page_view("My Team")
    usage.bump_activity()
    assert conn_spy.calls == 0
    assert _row_counts(temp_db) == dict.fromkeys(_TABLES, 0)


def test_app_main_returns_before_navigation_when_flag_off():
    """main()'s flag-off branch early-returns before it can reach st.navigation()."""
    src = pathlib.Path("app.py").read_text(encoding="utf-8")
    main = next(
        n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    guard_lines = [
        n.lineno
        for n in ast.walk(main)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "multi_user_enabled"
    ]
    nav_lines = [n.lineno for n in ast.walk(main) if isinstance(n, ast.Attribute) and n.attr == "navigation"]
    return_lines = [n.lineno for n in ast.walk(main) if isinstance(n, ast.Return)]
    assert guard_lines, "main() must branch on multi_user_enabled()"
    assert nav_lines, "main() must call st.navigation() on the flag-on path"
    assert return_lines, "main() must early-return on the flag-off path"
    assert min(guard_lines) < min(nav_lines), "the flag guard must precede st.navigation()"
    assert any(min(guard_lines) <= r < min(nav_lines) for r in return_lines), (
        "an early return must sit between the flag guard and st.navigation()"
    )
```

- [ ] **Step 2: Run the guard and confirm it passes**

Run: `python -m pytest tests/test_admin_backcompat.py -v`
Expected: PASS (6 tests). This guard codifies the flag-off early-returns already built in Tasks 4/5/6/9/14, so it goes green on first run. If any test FAILS, an earlier task regressed back-compat — fix that task (do not relax this guard).

- [ ] **Step 3: Commit**

```bash
git add tests/test_admin_backcompat.py
git commit -m "test(admin): consolidated MULTI_USER-off back-compat guard"
```

---

### Task 16: CLAUDE.md documentation

**Files:**
- Modify: `CLAUDE.md:24` (Multi-user mode paragraph)
- Modify: `CLAUDE.md:600` (structural-invariant table — add Plan 3 guard rows)
- Modify: `CLAUDE.md:594` (update the `test_app_main_auth_gate.py` row)
- Modify: `CLAUDE.md:812` (milestone tags line)

This task has no pytest test — it documents Plan 3 in the project's source-of-truth `CLAUDE.md` so future sessions and audits understand the admin dashboard. Verification is grep-based (the new strings must land). Do this LAST, after Tasks 1–14 are merged, so every module/table/page it references actually exists.

- [ ] **Step 1: Extend the Multi-user mode paragraph with Plan 3**

The current paragraph ends with the Plan 2 sentence. Append a Plan 3 sentence. Edit `CLAUDE.md`:

Replace this exact text:

```
Plan 2 adds two additive, MULTI_USER-gated surfaces: a per-feature feedback inbox (`src/feedback.py` → Admin Console "Feedback" tab) and per-page usage logging (`src/usage.py`, one deduped view per session + `last_seen_at`). Both are inert when the flag is off.
```

with:

```
Plan 2 adds two additive, MULTI_USER-gated surfaces: a per-feature feedback inbox (`src/feedback.py` → Admin Console "Feedback" tab) and per-page usage logging (`src/usage.py`, one deduped view per session + `last_seen_at`). Both are inert when the flag is off. Plan 3 adds the admin dashboard: role-aware `st.navigation()` replaces Streamlit's automatic `pages/` discovery when the flag is ON (OFF still uses auto-discovery = v1), plus five additive tables (`feature_flags`, `audit_log`, `app_settings`, `sessions`, `page_visits`), four new modules (`src/audit.py`, `src/app_settings.py`, `src/feature_flags.py`, `src/nav.py`) + a rewritten `src/usage.py` (session/dwell timing, `bump_activity()` heartbeat, lazy idle-close, analytics readers), per-page feature flags (`require_page_enabled`, "absence = enabled"), view-as-user impersonation in `src/auth.py`, broadcast + maintenance banners, CSV exports (usage + feedback), and an append-only admin audit log. The Admin Console moved to `pages/_admin_console.py` (joined by `pages/_admin_analytics.py` + `pages/_admin_controls.py`); the leading `_` hides all three from v1 auto-discovery — they route only via `st.navigation()`. Every new surface is inert when the flag is off.
```

- [ ] **Step 2: Add Plan 3 rows to the structural-invariant table**

The table's last Plan 2 row is `test_feedback_usage_backcompat.py`. Insert the 15 Plan 3 guard rows immediately after it. Edit `CLAUDE.md`:

Replace this exact text:

```
| `test_feedback_usage_backcompat.py` | `MULTI_USER` off ⇒ `log_page_view()` and `render_feedback_widget()` are no-ops (zero DB writes, no popover) — v1 byte-for-byte (v2 Plan 2) |
```

with:

```
| `test_feedback_usage_backcompat.py` | `MULTI_USER` off ⇒ `log_page_view()` and `render_feedback_widget()` are no-ops (zero DB writes, no popover) — v1 byte-for-byte (v2 Plan 2) |
| `test_admin_tables.py` | `init_db()` creates the 5 additive Plan 3 tables (`feature_flags`, `audit_log`, `app_settings`, `sessions`, `page_visits`) idempotently; all start empty/inert until written (v2 Plan 3) |
| `test_streamlit_min_version.py` | `requirements.txt` pins `streamlit>=1.40` — covers `st.navigation` (≥1.36) + `st.fragment(run_every=...)` (≥1.37) (v2 Plan 3) |
| `test_audit.py` | `src/audit.log_action()` is MULTI_USER-gated (off ⇒ no write); `list_audit()` joins usernames + filters by action; append-only (v2 Plan 3) |
| `test_app_settings.py` | `get_broadcast()`/`get_maintenance()` return `{"enabled","message"}` (default disabled); setters are flag-gated and each writes exactly one audit row (v2 Plan 3) |
| `test_feature_flags.py` | `is_page_enabled()` honors "absence = enabled"; `set_page_flag`/`require_page_enabled` flag-gated; flag off ⇒ every page always enabled (v2 Plan 3) |
| `test_nav.py` | `PAGE_REGISTRY` matches the 13 non-underscore season pages on disk; `build_pages(user, draft_page)` adds the Admin group only for `user["is_admin"]`; `filter_enabled_pages` is pure (v2 Plan 3) |
| `test_view_as.py` | `enter_view_as`/`exit_view_as`/`is_viewing_as` stash the real admin and swap `current_user()` to the target; flag-gated (off ⇒ no-op) (v2 Plan 3) |
| `test_usage_sessions_dwell.py` | `usage.log_page_view` opens a `sessions` row + `page_visits` dwell row; `bump_activity()` refreshes `last_activity_at`; idle visits lazily closed at query time (v2 Plan 3) |
| `test_usage_analytics.py` | analytics readers (`dau_series`, `most_used_pages`, `per_user_activity`, `last_seen_summary`, `session_timeline`, `page_dwell_summary`, `usage_csv`) return empty/inert when the flag is off (v2 Plan 3) |
| `test_feedback_csv.py` | `feedback.feedback_csv()` serializes the feedback inbox to CSV; header row always present (flag-off ⇒ header-only) (v2 Plan 3) |
| `test_admin_analytics_page.py` | `pages/_admin_analytics.py` calls `require_admin()` before rendering; AppTest smoke confirms non-admins are hard-stopped (v2 Plan 3) |
| `test_admin_controls_page.py` | `pages/_admin_controls.py` calls `require_admin()` before any flag/broadcast/maintenance/view-as mutation; each mutation writes an audit row (v2 Plan 3) |
| `test_pages_guard_set_page_config.py` | Every season `pages/*.py` guards its `st.set_page_config` with `if not multi_user_enabled():` so `app.py` owns the single config call under `st.navigation()` (v2 Plan 3) |
| `test_admin_pages_flag_enforced.py` | Each season page calls `require_page_enabled("page:<stem>")` after `require_auth()`; the 3 `_`-prefixed admin pages are exempt (v2 Plan 3) |
| `test_admin_backcompat.py` | `MULTI_USER` off ⇒ the whole Plan 3 admin surface is inert: a connection spy proves `is_page_enabled`/`log_action`/`set_setting`/`set_page_flag`/`log_page_view`/`bump_activity` never open a DB connection, the 5 tables stay empty, and `app.py`'s `main()` early-returns before `st.navigation()` — v1 byte-for-byte (v2 Plan 3) |
```

- [ ] **Step 3: Update the `test_app_main_auth_gate.py` row (splash moved out of main)**

Task 14 moved the splash/draft flow into `render_single_user_app()` and made `main()` branch on the flag, so the existing row's description is stale. Edit `CLAUDE.md`:

Replace this exact text:

```
| `test_app_main_auth_gate.py` | `app.py main()` calls `require_auth()` between `init_db()` and `render_splash_screen()`; imports it from `src.auth` |
```

with:

```
| `test_app_main_auth_gate.py` | `app.py main()`: flag-off fast path calls `render_single_user_app()` and returns (v1 unchanged); flag-on path runs `require_auth()` (after `init_db()`, before `st.navigation(`) then `st.navigation(build_pages(...)).run()`. AppTest smoke: flag-on + no session ⇒ login renders and `bootstrap_complete` never set (v2 Plan 3) |
```

- [ ] **Step 4: Record the Plan 3 milestone tag**

Edit `CLAUDE.md`:

Replace this exact text:

```
Milestone tags: `milestone/2026-05-21-fa-engine-overhaul-complete` (PR #110); `milestone/t1.21-oauth-decoupling-complete` (PR #118, current master).
```

with:

```
Milestone tags: `milestone/2026-05-21-fa-engine-overhaul-complete` (PR #110); `milestone/t1.21-oauth-decoupling-complete` (PR #118); `milestone/2026-05-29-v2-admin-dashboard-complete` (Plan 3, current master).
```

- [ ] **Step 5: Verify all four edits landed**

Run (PowerShell-safe — every probe substring is free of backticks and `$`, which PowerShell would otherwise interpret inside the double-quoted `-c` string):

```
python -c "import pathlib; t = pathlib.Path('CLAUDE.md').read_text(encoding='utf-8'); assert 'Plan 3 adds the admin dashboard' in t; assert 'test_admin_tables.py' in t; assert 'test_pages_guard_set_page_config.py' in t; assert 'test_admin_backcompat.py' in t; assert 'flag-off fast path calls' in t; assert '2026-05-29-v2-admin-dashboard-complete' in t; print('docs ok')"
```

Expected: prints `docs ok` (raises `AssertionError` if any edit is missing).

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude.md): document v2 Plan 3 admin dashboard"
```

The `milestone/2026-05-29-v2-admin-dashboard-complete` git tag itself is created during `superpowers:finishing-a-development-branch` (after the branch merges to master), not here — this step only records its name in the docs.

---

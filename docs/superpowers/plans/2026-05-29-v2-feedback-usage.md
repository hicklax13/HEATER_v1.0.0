# HEATER v2 Plan 2 — Per-Feature Feedback Inbox + Usage Logging — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-feature feedback inbox and lightweight per-page usage logging to HEATER, fully additive behind the existing `MULTI_USER` flag (off ⇒ byte-for-byte v1).

**Architecture:** Two new SQLite tables (`feedback`, `usage_events`) appended to the existing `_init_multiuser_tables()` block so a flag flip needs no migration. Two thin service modules — `src/feedback.py` (capture + admin inbox helpers + a Streamlit popover widget) and `src/usage.py` (one deduped "view" event per session + `last_seen_at` bump). A tiny `src/version.py` stamps each feedback row with the running app version. The Admin Console gains a "Feedback" tab; all 13 interactive pages get two uniform wiring lines. Every cross-cutting hook is a no-op unless `MULTI_USER` is on AND a user is logged in — matching the per-page auth-gate pattern Streamlit forces (each page re-runs top-to-bottom; there is no global middleware).

**Tech Stack:** Python 3.11+ (CI 3.12, local 3.14), Streamlit, SQLite via `src.database.get_connection()`, pytest (+ `streamlit.testing.v1.AppTest` for page smokes). No new third-party dependencies — stdlib `uuid`, `json`, `datetime` only.

**Branch:** Do this work on a feature branch (e.g. `feat/v2-feedback-usage`), not `master`. The execution skill (subagent-driven-development or executing-plans) + using-git-worktrees handles branch/worktree creation before Task 1.

**Spec:** `docs/superpowers/specs/2026-05-29-v2-feedback-usage-design.md`

---

## Conventions used throughout this plan

- **Run tests with `python -m pytest ...`** (bare `pytest`/`ruff` are unreliable on the user's Windows shell).
- **DB isolation in tests:** a local `temp_db` fixture monkeypatches `src.database.DB_PATH` to a fresh `tmp_path` file and calls `init_db()`. `get_connection()` reads the module global `DB_PATH` on every call (database.py:132-133), so this isolates connections in both single-process and pytest-xdist runs. This mirrors the existing `tests/test_admin_console_guarded.py` pattern.
- **No `from __future__ import annotations`** in the new `src/` modules — `src/auth.py` omits it and py311 evaluates `str | None` natively. Stay consistent with `auth.py`.
- **Lint:** ruff lint selects `E, F, I, UP` (line-length 120, `E501` ignored). `tests/*` ignore `F401` (unused imports OK in tests). `src/` does NOT — no unused imports in `src/feedback.py` / `src/usage.py`.
- **Network guard:** `tests/conftest.py` blocks non-loopback sockets. Every test that touches `_capture_data_state` MUST mock `src.database.get_refresh_log_snapshot` so nothing reaches the network.

---

## File Structure

**Create (source):**
- `src/version.py` — single source of the running app version string (`APP_VERSION`), env-overridable.
- `src/feedback.py` — `submit_feedback`, `_capture_data_state`, inbox helpers (`list_feedback`, `set_feedback_status`, `set_feedback_notes`), and the `render_feedback_widget` Streamlit popover.
- `src/usage.py` — `log_page_view` (one deduped event per `(page, action)` per session) + `last_seen_at` bump, with a `_session_state()` test seam.

**Modify (source):**
- `src/database.py` — append `feedback` + `usage_events` `CREATE TABLE`/`CREATE INDEX` to `_init_multiuser_tables()` (database.py:798-813).
- `pages/00_Admin_Console.py` — wrap the existing body under a "Users" tab; add a "Feedback" inbox tab.
- `pages/{1_My_Team, 2_Line-up_Optimizer, 3_Closer_Monitor, 5_Matchup_Planner, 6_League_Standings, 10_Punt_Analyzer, 11_Trade_Analyzer, 12_Trade_Finder, 14_Free_Agents, 16_Player_Compare, 17_Leaders, 19_Player_Databank, 20_Draft_Simulator}.py` — two import lines + `log_page_view(...)` after `require_auth()` + `render_feedback_widget(...)` at EOF.
- `CLAUDE.md` — document the three new modules and the new structural-invariant guard.

**Create (tests):**
- `tests/test_version.py`
- `tests/test_feedback_usage_tables.py`
- `tests/test_feedback_capture.py`
- `tests/test_feedback_inbox.py`
- `tests/test_feedback_widget.py`
- `tests/test_usage_logging.py`
- `tests/test_admin_console_feedback_tab.py`
- `tests/test_pages_have_feedback_and_usage.py`
- `tests/test_feedback_usage_backcompat.py`

---

## Task 1: App version stamp (`src/version.py`)

**Files:**
- Create: `src/version.py`
- Test: `tests/test_version.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_version.py`:

```python
"""src.version exposes a stable, env-overridable app version string."""

import importlib

import src.version


def test_app_version_is_nonempty_string():
    assert isinstance(src.version.APP_VERSION, str)
    assert src.version.APP_VERSION.strip()


def test_app_version_env_override(monkeypatch):
    monkeypatch.setenv("HEATER_APP_VERSION", "9.9.9-test")
    try:
        importlib.reload(src.version)
        assert src.version.APP_VERSION == "9.9.9-test"
    finally:
        # Restore the module's default so later tests see the real version.
        monkeypatch.delenv("HEATER_APP_VERSION", raising=False)
        importlib.reload(src.version)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_version.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.version'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/version.py`:

```python
"""Running app version, stamped onto feedback so the admin can reproduce context.

Defaults to the current local/CLAUDE.md folder version ("1.0.1"). Override at
deploy time with the HEATER_APP_VERSION env var (e.g. a Railway build tag) so
feedback rows record exactly which build the user was on.
"""

import os

APP_VERSION = os.environ.get("HEATER_APP_VERSION") or "1.0.1"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_version.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/version.py tests/test_version.py
git commit -m "feat(version): app version stamp for feedback (v2 Plan 2)"
```

---

## Task 2: Feedback + usage_events tables

**Files:**
- Modify: `src/database.py` (inside `_init_multiuser_tables`, database.py:798-813)
- Test: `tests/test_feedback_usage_tables.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_feedback_usage_tables.py`:

```python
"""init_db() creates the v2 feedback + usage_events tables (additive, idempotent)."""

import pytest


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db = tmp_path / "tables.db"
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


def test_feedback_table_created(temp_db):
    cols = _table_columns("feedback")
    assert {
        "id",
        "user_id",
        "page",
        "feature_tag",
        "message",
        "app_version",
        "data_state",
        "status",
        "admin_notes",
        "created_at",
    } <= cols


def test_usage_events_table_created(temp_db):
    cols = _table_columns("usage_events")
    assert {"id", "user_id", "page", "action", "session_id", "created_at"} <= cols


def test_init_db_idempotent_for_new_tables(temp_db, monkeypatch):
    # A second init_db() on the same DB must not raise.
    from src.database import init_db

    init_db()
    assert "status" in _table_columns("feedback")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_feedback_usage_tables.py -v`
Expected: FAIL — `no such table: feedback` (the `PRAGMA table_info` returns no rows → empty column set → assertion fails).

- [ ] **Step 3: Write minimal implementation**

In `src/database.py`, edit `_init_multiuser_tables` (database.py:798-813). Append the two tables INSIDE the existing `conn.executescript("""...""")` string, immediately after the `idx_users_status` index line and before the closing `"""`:

Find:

```python
        CREATE INDEX IF NOT EXISTS idx_users_status ON users(status);
    """)
    conn.commit()
```

Replace with:

```python
        CREATE INDEX IF NOT EXISTS idx_users_status ON users(status);

        -- v2 Plan 2: per-feature feedback inbox. user_id/page/message are the
        -- core; app_version + data_state snapshot the build + data-freshness
        -- state at submit time so an admin can reproduce what the user saw.
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(user_id),
            page TEXT NOT NULL,
            feature_tag TEXT,
            message TEXT NOT NULL,
            app_version TEXT NOT NULL,
            data_state TEXT,                         -- JSON refresh_log snapshot, nullable
            status TEXT NOT NULL DEFAULT 'new',       -- new | triaged | resolved
            admin_notes TEXT,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_feedback_status ON feedback(status);

        -- v2 Plan 2: lightweight usage logging. One 'view' row per (page, action)
        -- per session; session_id groups a single visit's events.
        CREATE TABLE IF NOT EXISTS usage_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(user_id),
            page TEXT NOT NULL,
            action TEXT NOT NULL DEFAULT 'view',
            session_id TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_usage_user_created ON usage_events(user_id, created_at);
    """)
    conn.commit()
```

> Note: the `REFERENCES users(user_id)` clauses are documentary — `get_connection()` does not set `PRAGMA foreign_keys=ON`, so they are not enforced at runtime. They record intent and survive a future PRAGMA flip.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_feedback_usage_tables.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Run ruff on the changed source**

Run: `python -m ruff check src/database.py && python -m ruff format src/database.py`
Expected: no errors; formatter reports the file unchanged or reformatted cleanly.

- [ ] **Step 6: Commit**

```bash
git add src/database.py tests/test_feedback_usage_tables.py
git commit -m "feat(db): feedback + usage_events tables in _init_multiuser_tables (v2 Plan 2)"
```

---

## Task 3: Feedback capture (`submit_feedback` + `_capture_data_state`)

**Files:**
- Create: `src/feedback.py`
- Test: `tests/test_feedback_capture.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_feedback_capture.py`:

```python
"""submit_feedback writes a row; _capture_data_state is best-effort JSON."""

import json

import pytest


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db = tmp_path / "feedback.db"
    monkeypatch.setattr("src.database.DB_PATH", db)
    from src.database import init_db

    init_db()
    return db


def _row(feedback_id):
    from src.database import get_connection

    conn = get_connection()
    try:
        return dict(conn.execute("SELECT * FROM feedback WHERE id = ?", (feedback_id,)).fetchone())
    finally:
        conn.close()


def test_submit_feedback_inserts_and_returns_id(temp_db):
    from src.feedback import submit_feedback
    from src.version import APP_VERSION

    fid = submit_feedback(user_id=1, page="My Team", message="Looks great", feature_tag="roster")
    assert isinstance(fid, int) and fid > 0
    row = _row(fid)
    assert row["user_id"] == 1
    assert row["page"] == "My Team"
    assert row["feature_tag"] == "roster"
    assert row["message"] == "Looks great"
    assert row["app_version"] == APP_VERSION
    assert row["status"] == "new"  # DB default
    assert row["created_at"]


def test_submit_feedback_allows_null_feature_tag(temp_db):
    from src.feedback import submit_feedback

    fid = submit_feedback(user_id=2, page="Leaders", message="No tag here")
    assert _row(fid)["feature_tag"] is None


def test_capture_data_state_returns_json(monkeypatch):
    monkeypatch.setattr(
        "src.database.get_refresh_log_snapshot",
        lambda: [{"source": "players", "status": "success"}],
    )
    from src.feedback import _capture_data_state

    captured = _capture_data_state()
    assert json.loads(captured) == [{"source": "players", "status": "success"}]


def test_capture_data_state_none_when_empty(monkeypatch):
    monkeypatch.setattr("src.database.get_refresh_log_snapshot", lambda: [])
    from src.feedback import _capture_data_state

    assert _capture_data_state() is None


def test_capture_data_state_swallows_errors(monkeypatch):
    def boom():
        raise RuntimeError("db down")

    monkeypatch.setattr("src.database.get_refresh_log_snapshot", boom)
    from src.feedback import _capture_data_state

    assert _capture_data_state() is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_feedback_capture.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.feedback'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/feedback.py`:

```python
"""Per-feature feedback capture + admin inbox (v2 Plan 2, MULTI_USER-gated).

Feedback is recorded against the logged-in user, tagged with the page and an
optional feature, stamped with the running app version, and snapshotted with
the current data-freshness state so an admin can reproduce what the user saw.
"""

import json
from datetime import UTC, datetime

from src.version import APP_VERSION

_VALID_STATUSES = ("new", "triaged", "resolved")


def _capture_data_state() -> str | None:
    """JSON snapshot of the refresh_log, or None if unavailable.

    Best-effort: a feedback submission must never fail because the snapshot did.
    Imported lazily so importing this module never drags in the database layer.
    """
    try:
        from src.database import get_refresh_log_snapshot

        snapshot = get_refresh_log_snapshot()
        if not snapshot:
            return None
        return json.dumps(snapshot)
    except Exception:
        return None


def submit_feedback(user_id: int, page: str, message: str, feature_tag: str | None = None) -> int:
    """Insert a feedback row; return the new row id.

    status defaults to 'new' at the DB layer. app_version + data_state are
    captured here so the admin inbox can reproduce the user's context.
    """
    from src.database import get_connection

    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO feedback (user_id, page, feature_tag, message, app_version, "
            "data_state, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                user_id,
                page,
                feature_tag,
                message,
                APP_VERSION,
                _capture_data_state(),
                datetime.now(UTC).isoformat(),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_feedback_capture.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/feedback.py tests/test_feedback_capture.py
git commit -m "feat(feedback): submit_feedback + data-state capture (v2 Plan 2)"
```

---

## Task 4: Feedback inbox helpers (`list_feedback`, `set_feedback_status`, `set_feedback_notes`)

**Files:**
- Modify: `src/feedback.py`
- Test: `tests/test_feedback_inbox.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_feedback_inbox.py`:

```python
"""Admin inbox helpers: list (joined with submitter), status, notes."""

import pytest


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db = tmp_path / "inbox.db"
    monkeypatch.setattr("src.database.DB_PATH", db)
    from src.database import init_db

    init_db()
    return db


def _seed_user(username: str, team_name: str) -> int:
    from src.auth import approve_user, create_user, get_user

    create_user(username, "pw")
    approve_user(username, team_name=team_name, approved_by="test")
    return get_user(username)["user_id"]


def test_list_feedback_joins_username_and_team(temp_db):
    from src.feedback import list_feedback, submit_feedback

    uid = _seed_user("inbox_alice", "Team Alice")
    fid = submit_feedback(uid, "Trade Analyzer", "trade math looks off")
    match = [r for r in list_feedback() if r["id"] == fid]
    assert match, "submitted feedback must appear in the inbox"
    assert match[0]["username"] == "inbox_alice"
    assert match[0]["team_name"] == "Team Alice"


def test_list_feedback_filters_by_status(temp_db):
    from src.feedback import list_feedback, set_feedback_status, submit_feedback

    uid = _seed_user("inbox_bob", "Team Bob")
    fid = submit_feedback(uid, "Leaders", "needs a sort toggle")
    set_feedback_status(fid, "triaged")
    assert fid not in {r["id"] for r in list_feedback(status="new")}
    assert fid in {r["id"] for r in list_feedback(status="triaged")}


def test_set_feedback_status_rejects_invalid(temp_db):
    from src.feedback import set_feedback_status, submit_feedback

    uid = _seed_user("inbox_carol", "Team Carol")
    fid = submit_feedback(uid, "My Team", "x")
    with pytest.raises(ValueError):
        set_feedback_status(fid, "bogus")


def test_set_feedback_notes_persists(temp_db):
    from src.feedback import list_feedback, set_feedback_notes, submit_feedback

    uid = _seed_user("inbox_dave", "Team Dave")
    fid = submit_feedback(uid, "My Team", "y")
    set_feedback_notes(fid, "Investigated — projection cache was stale.")
    match = [r for r in list_feedback() if r["id"] == fid][0]
    assert match["admin_notes"] == "Investigated — projection cache was stale."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_feedback_inbox.py -v`
Expected: FAIL — `ImportError: cannot import name 'list_feedback' from 'src.feedback'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/feedback.py` (after `submit_feedback`):

```python
def list_feedback(status: str | None = None) -> list[dict]:
    """Feedback rows joined with submitter username/team, newest first."""
    from src.database import get_connection

    conn = get_connection()
    try:
        sql = (
            "SELECT f.*, u.username AS username, u.team_name AS team_name "
            "FROM feedback f LEFT JOIN users u ON u.user_id = f.user_id"
        )
        params: tuple = ()
        if status is not None:
            sql += " WHERE f.status = ?"
            params = (status,)
        sql += " ORDER BY f.created_at DESC, f.id DESC"
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def set_feedback_status(feedback_id: int, status: str) -> None:
    """Update a feedback row's triage status. Validates against _VALID_STATUSES."""
    if status not in _VALID_STATUSES:
        raise ValueError(f"Invalid feedback status: {status!r}. Expected one of {_VALID_STATUSES}.")
    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute("UPDATE feedback SET status = ? WHERE id = ?", (status, feedback_id))
        conn.commit()
    finally:
        conn.close()


def set_feedback_notes(feedback_id: int, notes: str) -> None:
    """Save an admin's triage notes onto a feedback row."""
    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute("UPDATE feedback SET admin_notes = ? WHERE id = ?", (notes, feedback_id))
        conn.commit()
    finally:
        conn.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_feedback_inbox.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/feedback.py tests/test_feedback_inbox.py
git commit -m "feat(feedback): admin inbox helpers list/status/notes (v2 Plan 2)"
```

---

## Task 5: Feedback widget (`render_feedback_widget`)

**Files:**
- Modify: `src/feedback.py` (add module-level `streamlit` + `auth` imports; add `render_feedback_widget`)
- Test: `tests/test_feedback_widget.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_feedback_widget.py`:

```python
"""render_feedback_widget is a no-op unless MULTI_USER is on AND a user exists."""

import pytest

import src.feedback as feedback


def test_widget_noop_when_multiuser_off(monkeypatch):
    monkeypatch.setattr(feedback, "multi_user_enabled", lambda: False)
    monkeypatch.setattr(feedback, "current_user", lambda: {"user_id": 1})
    # If the flag is off, the popover must never be reached.
    monkeypatch.setattr(feedback.st, "popover", lambda *a, **k: pytest.fail("popover called"))
    feedback.render_feedback_widget("My Team")  # must simply return


def test_widget_noop_when_no_user(monkeypatch):
    monkeypatch.setattr(feedback, "multi_user_enabled", lambda: True)
    monkeypatch.setattr(feedback, "current_user", lambda: None)
    monkeypatch.setattr(feedback.st, "popover", lambda *a, **k: pytest.fail("popover called"))
    feedback.render_feedback_widget("My Team")  # must simply return
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_feedback_widget.py -v`
Expected: FAIL — `AttributeError: module 'src.feedback' has no attribute 'st'` (and no `render_feedback_widget`).

- [ ] **Step 3: Write minimal implementation**

In `src/feedback.py`, add the two imports to the top-of-file import block. Find:

```python
import json
from datetime import UTC, datetime

from src.version import APP_VERSION
```

Replace with:

```python
import json
from datetime import UTC, datetime

import streamlit as st

from src.auth import current_user, multi_user_enabled
from src.version import APP_VERSION
```

Then append `render_feedback_widget` to the end of `src/feedback.py`:

```python
def render_feedback_widget(page: str, feature_tag: str | None = None) -> None:
    """Render a 'Send feedback' popover. No-op unless multi-user + logged in.

    Streamlit re-runs each page top-to-bottom, so this is called explicitly per
    page rather than via global middleware. When the flag is off (v1) or there
    is no session user, it returns immediately and renders nothing.
    """
    if not multi_user_enabled():
        return
    user = current_user()
    if user is None:
        return

    suffix = feature_tag or page
    with st.popover("Send feedback on this"):
        with st.form(f"feedback_form_{suffix}", clear_on_submit=True):
            message = st.text_area(
                "What's working, broken, or confusing?",
                key=f"feedback_msg_{suffix}",
            )
            submitted = st.form_submit_button("Send feedback")
        if submitted:
            text = (message or "").strip()
            if not text:
                st.warning("Please enter a message before sending.")
            else:
                submit_feedback(user["user_id"], page, text, feature_tag=feature_tag)
                st.success("Thanks — your feedback was sent to the admin.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_feedback_widget.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Run ruff (import order matters — `I` is enabled)**

Run: `python -m ruff check src/feedback.py && python -m ruff format src/feedback.py`
Expected: no errors. (The import block above is already isort-ordered: stdlib → third-party `streamlit` → first-party `src.*`.)

- [ ] **Step 6: Commit**

```bash
git add src/feedback.py tests/test_feedback_widget.py
git commit -m "feat(feedback): MULTI_USER-gated feedback popover widget (v2 Plan 2)"
```

---

## Task 6: Usage logging (`src/usage.py`)

**Files:**
- Create: `src/usage.py`
- Test: `tests/test_usage_logging.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_usage_logging.py`:

```python
"""log_page_view: one deduped event per (page, action) per session + last_seen bump."""

import pytest

import src.usage as usage


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db = tmp_path / "usage.db"
    monkeypatch.setattr("src.database.DB_PATH", db)
    from src.database import init_db

    init_db()
    return db


@pytest.fixture
def state(monkeypatch):
    """A plain-dict stand-in for st.session_state + flag forced on."""
    s: dict = {}
    monkeypatch.setattr(usage, "_session_state", lambda: s)
    monkeypatch.setattr(usage, "multi_user_enabled", lambda: True)
    return s


def _seed_user(username: str) -> int:
    from src.auth import approve_user, create_user, get_user

    create_user(username, "pw")
    approve_user(username, team_name="Team " + username, approved_by="test")
    return get_user(username)["user_id"]


def _events(user_id: int) -> list[dict]:
    from src.database import get_connection

    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM usage_events WHERE user_id = ? ORDER BY id", (user_id,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def test_log_inserts_one_event(temp_db, state, monkeypatch):
    uid = _seed_user("usage_amy")
    monkeypatch.setattr(usage, "current_user", lambda: {"user_id": uid})
    usage.log_page_view("My Team")
    events = _events(uid)
    assert len(events) == 1
    assert events[0]["page"] == "My Team"
    assert events[0]["action"] == "view"
    assert events[0]["session_id"]


def test_log_dedups_same_page(temp_db, state, monkeypatch):
    uid = _seed_user("usage_ben")
    monkeypatch.setattr(usage, "current_user", lambda: {"user_id": uid})
    usage.log_page_view("Leaders")
    usage.log_page_view("Leaders")
    usage.log_page_view("Leaders")
    assert len(_events(uid)) == 1


def test_log_distinct_pages(temp_db, state, monkeypatch):
    uid = _seed_user("usage_cat")
    monkeypatch.setattr(usage, "current_user", lambda: {"user_id": uid})
    usage.log_page_view("My Team")
    usage.log_page_view("Leaders")
    assert {e["page"] for e in _events(uid)} == {"My Team", "Leaders"}


def test_session_id_stable_across_views(temp_db, state, monkeypatch):
    uid = _seed_user("usage_dan")
    monkeypatch.setattr(usage, "current_user", lambda: {"user_id": uid})
    usage.log_page_view("My Team")
    usage.log_page_view("Leaders")
    assert len({e["session_id"] for e in _events(uid)}) == 1


def test_last_seen_at_bumped(temp_db, state, monkeypatch):
    uid = _seed_user("usage_eve")
    monkeypatch.setattr(usage, "current_user", lambda: {"user_id": uid})
    usage.log_page_view("My Team")
    from src.database import get_connection

    conn = get_connection()
    try:
        row = dict(conn.execute("SELECT last_seen_at FROM users WHERE user_id = ?", (uid,)).fetchone())
    finally:
        conn.close()
    assert row["last_seen_at"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_usage_logging.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.usage'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/usage.py`:

```python
"""Lightweight per-page usage logging (v2 Plan 2, MULTI_USER-gated).

Records one 'view' event per (page, action) per session so the admin can see
which features get used, and bumps users.last_seen_at. No-op in single-user v1
(flag off) or when there is no logged-in user.
"""

import uuid
from datetime import UTC, datetime

from src.auth import current_user, multi_user_enabled


def _session_state():
    """Seam over st.session_state so unit tests can inject a plain dict."""
    import streamlit as st

    return st.session_state


def log_page_view(page: str, action: str = "view") -> None:
    """Record a usage event once per (page, action) per session; bump last_seen_at.

    No-op when MULTI_USER is off or no user is logged in. The dedup key is added
    only after a successful commit, so a failed write is retried on the next run.
    """
    if not multi_user_enabled():
        return
    user = current_user()
    if user is None:
        return

    state = _session_state()
    logged = state.get("_usage_logged")
    if logged is None:
        logged = set()
        state["_usage_logged"] = logged
    dedup_key = (page, action)
    if dedup_key in logged:
        return

    session_id = state.get("_usage_session_id")
    if not session_id:
        session_id = uuid.uuid4().hex
        state["_usage_session_id"] = session_id

    from src.database import get_connection

    conn = get_connection()
    try:
        now = datetime.now(UTC).isoformat()
        conn.execute(
            "INSERT INTO usage_events (user_id, page, action, session_id, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (user["user_id"], page, action, session_id, now),
        )
        conn.execute(
            "UPDATE users SET last_seen_at = ? WHERE user_id = ?",
            (now, user["user_id"]),
        )
        conn.commit()
    finally:
        conn.close()

    logged.add(dedup_key)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_usage_logging.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Run ruff**

Run: `python -m ruff check src/usage.py && python -m ruff format src/usage.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/usage.py tests/test_usage_logging.py
git commit -m "feat(usage): per-session deduped page-view logging + last_seen (v2 Plan 2)"
```

---

## Task 7: Admin Console "Feedback" tab

**Files:**
- Modify: `pages/00_Admin_Console.py` (full rewrite below — wrap existing body under a "Users" tab, add a "Feedback" tab)
- Test: `tests/test_admin_console_feedback_tab.py`

> The existing `tests/test_admin_console_guarded.py` invariants MUST still hold: `require_admin()` runs before any `approve_user(` call, and the page title renders for an admin. The rewrite keeps `require_admin()` at the top (before the tabs) and keeps `approve_user(...)` calls inside the "Users" tab (which is after the guard). Do not reorder those.

- [ ] **Step 1: Write the failing test**

Create `tests/test_admin_console_feedback_tab.py`:

```python
"""AppTest smoke: the admin Feedback tab renders submitted feedback for an admin.

Uses a temp DB (DB_PATH monkeypatch) + a real bootstrap admin because
require_auth() re-validates the session identity against the DB on every page
load — a session-only stub would be logged straight back out.
"""

import pytest


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db = tmp_path / "admin_feedback.db"
    monkeypatch.setattr("src.database.DB_PATH", db)
    from src.database import init_db

    init_db()
    return db


def test_feedback_tab_renders_message_for_admin(temp_db, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from src.auth import approve_user, create_user, ensure_bootstrap_admin, get_user
    from src.feedback import submit_feedback

    monkeypatch.setenv("MULTI_USER", "1")
    monkeypatch.setenv("ADMIN_USERNAME", "connor")
    monkeypatch.setenv("ADMIN_PASSWORD", "pw")
    monkeypatch.setenv("ADMIN_TEAM_NAME", "Team Hickey")
    ensure_bootstrap_admin()

    # A non-admin submits feedback that the admin should see in the inbox.
    create_user("submitter", "pw", display_name="Sam")
    approve_user("submitter", team_name="Team Sam", approved_by="test")
    submit_feedback(get_user("submitter")["user_id"], "Trade Analyzer", "ZNVECTOR sentinel message")

    at = AppTest.from_file("pages/00_Admin_Console.py")
    at.session_state["auth_user"] = {
        "username": "connor",
        "status": "active",
        "is_admin": 1,
        "team_name": "Team Hickey",
    }
    at.session_state["_auth_bootstrap_done"] = True
    at.run()

    assert not at.exception, [str(e) for e in at.exception]
    # The submitted message must appear somewhere in the rendered markdown/text.
    blobs = [m.value for m in at.markdown] + [t.value for t in at.text]
    assert any("ZNVECTOR sentinel message" in b for b in blobs), blobs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_admin_console_feedback_tab.py -v`
Expected: FAIL — the feedback message is not rendered (the current Admin Console has no Feedback tab), so the final assertion fails.

- [ ] **Step 3: Write the implementation (full file rewrite)**

Overwrite `pages/00_Admin_Console.py` with:

```python
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
            st.markdown(
                f"**{fb['page']}**{tag}  \n{who} · {team} · {fb['created_at'][:16]} · v{fb['app_version']}"
            )
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
```

- [ ] **Step 4: Run the new smoke + the existing guard test**

Run: `python -m pytest tests/test_admin_console_feedback_tab.py tests/test_admin_console_guarded.py -v`
Expected: PASS (1 new + 4 existing = 5 passed). The existing guard confirms `require_admin()` still precedes `approve_user(` and the title still renders.

- [ ] **Step 5: Run ruff**

Run: `python -m ruff check pages/00_Admin_Console.py && python -m ruff format pages/00_Admin_Console.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add pages/00_Admin_Console.py tests/test_admin_console_feedback_tab.py
git commit -m "feat(admin): Feedback inbox tab on Admin Console (v2 Plan 2)"
```

---

## Task 8: Structural guard for page wiring (write the failing test FIRST)

**Files:**
- Test: `tests/test_pages_have_feedback_and_usage.py`

> This is the TDD anchor for Task 9. It is a static-source test (mirrors `tests/test_pages_have_auth_guard.py`): it reads each page's text and asserts the wiring is present and correctly placed. It MUST fail now (pages aren't wired yet) and pass after Task 9.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pages_have_feedback_and_usage.py`:

```python
"""Structural invariant: every interactive page logs usage + offers feedback.

Mirrors test_pages_have_auth_guard.py. Streamlit runs each pages/*.py top to
bottom on every interaction, so the usage/feedback wiring is per-page, not
global. log_page_view() must sit after the auth gate; the feedback widget is
rendered on the page (appended at EOF).

The admin console (00_Admin_Console.py) is exempt — it has its own surfaces.
"""

from pathlib import Path

import pytest

_PAGES_DIR = Path(__file__).resolve().parent.parent / "pages"

_INTERACTIVE_PAGES = sorted(
    p
    for p in _PAGES_DIR.glob("*.py")
    if "inject_custom_css()" in p.read_text(encoding="utf-8") and p.name != "00_Admin_Console.py"
)


def test_found_the_pages():
    assert len(_INTERACTIVE_PAGES) == 13, [p.name for p in _INTERACTIVE_PAGES]


@pytest.mark.parametrize("page", _INTERACTIVE_PAGES, ids=lambda p: p.name)
def test_page_imports_usage_and_feedback(page):
    src = page.read_text(encoding="utf-8")
    assert "from src.usage import log_page_view" in src, f"{page.name} must import log_page_view"
    assert "from src.feedback import render_feedback_widget" in src, (
        f"{page.name} must import render_feedback_widget"
    )


@pytest.mark.parametrize("page", _INTERACTIVE_PAGES, ids=lambda p: p.name)
def test_log_page_view_after_require_auth(page):
    src = page.read_text(encoding="utf-8")
    i_auth = src.index("require_auth()")
    i_log = src.index("log_page_view(")
    assert i_log > i_auth, f"{page.name}: log_page_view() must follow require_auth()"


@pytest.mark.parametrize("page", _INTERACTIVE_PAGES, ids=lambda p: p.name)
def test_feedback_widget_called(page):
    src = page.read_text(encoding="utf-8")
    assert "render_feedback_widget(" in src, f"{page.name} must call render_feedback_widget()"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pages_have_feedback_and_usage.py -v`
Expected: FAIL — `test_found_the_pages` passes (13 pages), but `test_page_imports_usage_and_feedback`, `test_log_page_view_after_require_auth`, and `test_feedback_widget_called` fail for all 13 pages (imports/calls absent → `ValueError: substring not found` on `.index(...)` or assertion failure).

- [ ] **Step 3: Commit the failing guard**

```bash
git add tests/test_pages_have_feedback_and_usage.py
git commit -m "test(pages): structural guard for feedback+usage wiring (v2 Plan 2, failing)"
```

> Committing a known-failing guard is intentional here: Task 9 makes it green in the very next commit. If you prefer a single green commit, defer this `git add`/`commit` and stage it together with Task 9's changes.

---

## Task 9: Wire all 13 interactive pages

**Files (modify):**
- `pages/1_My_Team.py`
- `pages/2_Line-up_Optimizer.py`
- `pages/3_Closer_Monitor.py`
- `pages/5_Matchup_Planner.py`
- `pages/6_League_Standings.py`
- `pages/10_Punt_Analyzer.py`
- `pages/11_Trade_Analyzer.py`
- `pages/12_Trade_Finder.py`
- `pages/14_Free_Agents.py`
- `pages/16_Player_Compare.py`
- `pages/17_Leaders.py`
- `pages/19_Player_Databank.py`
- `pages/20_Draft_Simulator.py`

**The uniform edit (apply to every page, three changes each):**

1. **Imports** — directly after the existing `from src.auth import require_auth` line, add:

```python
from src.feedback import render_feedback_widget
from src.usage import log_page_view
```

2. **Usage log** — on the line immediately after the existing `require_auth()` call, add (with the page's canonical label):

```python
log_page_view("<LABEL>")
```

3. **Feedback widget** — append as the final line of the file (column 0, top-level):

```python
render_feedback_widget("<LABEL>")
```

**Canonical `<LABEL>` per page** (reuse the exact strings already passed to `page_timer_footer(...)`; the two footer-less pages — Punt Analyzer, Player Databank — use the names below):

| Page file | `<LABEL>` |
|-----------|-----------|
| `1_My_Team.py` | `My Team` |
| `2_Line-up_Optimizer.py` | `Lineup` |
| `3_Closer_Monitor.py` | `Closer Monitor` |
| `5_Matchup_Planner.py` | `Matchup Planner` |
| `6_League_Standings.py` | `League Standings` |
| `10_Punt_Analyzer.py` | `Punt Analyzer` |
| `11_Trade_Analyzer.py` | `Trade Analyzer` |
| `12_Trade_Finder.py` | `Trade Finder` |
| `14_Free_Agents.py` | `Free Agents` |
| `16_Player_Compare.py` | `Player Compare` |
| `17_Leaders.py` | `Leaders` |
| `19_Player_Databank.py` | `Player Databank` |
| `20_Draft_Simulator.py` | `Draft Simulator` |

- [ ] **Step 1: Worked example — `pages/1_My_Team.py`**

Find (My_Team.py:8):

```python
from src.auth import require_auth
```

Replace with:

```python
from src.auth import require_auth
from src.feedback import render_feedback_widget
from src.usage import log_page_view
```

Find the `require_auth()` call (it sits just after `inject_custom_css()`), e.g.:

```python
inject_custom_css()
require_auth()
```

Replace with:

```python
inject_custom_css()
require_auth()
log_page_view("My Team")
```

Append to the very end of the file (after the existing `page_timer_footer("My Team")` line):

```python
render_feedback_widget("My Team")
```

- [ ] **Step 2: Apply the same three edits to the remaining 12 pages**

Use the exact labels from the table above. The explicit per-page calls are:

```text
2_Line-up_Optimizer.py   → log_page_view("Lineup")            ; render_feedback_widget("Lineup")
3_Closer_Monitor.py      → log_page_view("Closer Monitor")    ; render_feedback_widget("Closer Monitor")
5_Matchup_Planner.py     → log_page_view("Matchup Planner")   ; render_feedback_widget("Matchup Planner")
6_League_Standings.py    → log_page_view("League Standings")  ; render_feedback_widget("League Standings")
10_Punt_Analyzer.py      → log_page_view("Punt Analyzer")     ; render_feedback_widget("Punt Analyzer")
11_Trade_Analyzer.py     → log_page_view("Trade Analyzer")    ; render_feedback_widget("Trade Analyzer")
12_Trade_Finder.py       → log_page_view("Trade Finder")      ; render_feedback_widget("Trade Finder")
14_Free_Agents.py        → log_page_view("Free Agents")       ; render_feedback_widget("Free Agents")
16_Player_Compare.py     → log_page_view("Player Compare")    ; render_feedback_widget("Player Compare")
17_Leaders.py            → log_page_view("Leaders")           ; render_feedback_widget("Leaders")
19_Player_Databank.py    → log_page_view("Player Databank")   ; render_feedback_widget("Player Databank")
20_Draft_Simulator.py    → log_page_view("Draft Simulator")   ; render_feedback_widget("Draft Simulator")
```

For every page: add the two imports after its `from src.auth import require_auth` line; add the `log_page_view("<LABEL>")` line right after that page's `require_auth()` call; append `render_feedback_widget("<LABEL>")` as the last line.

> Notes:
> - `12_Trade_Finder.py` has an early `st.stop()` path (Trade_Finder.py:347). `log_page_view` runs before it (right after `require_auth()`), so the visit is always logged; the EOF `render_feedback_widget` simply won't render on that early-return path — acceptable.
> - `2_Line-up_Optimizer.py` imports `require_auth` at line 22; its `require_auth()` call is just after `inject_custom_css()` like the others. The structural test checks call-order, not line numbers, so exact placement is robust to drift.

- [ ] **Step 3: Normalize import order + format (ruff `I` is enabled)**

Run: `python -m ruff check --fix pages/ && python -m ruff format pages/`
Expected: ruff sorts the new first-party imports into place and reports success. Re-run `python -m ruff check pages/` → no remaining errors.

- [ ] **Step 4: Run the structural guard from Task 8 — now PASSES**

Run: `python -m pytest tests/test_pages_have_feedback_and_usage.py -v`
Expected: PASS (1 + 13 + 13 + 13 = 40 passed).

- [ ] **Step 5: Smoke that pages still import cleanly (no syntax/indent breakage)**

Run: `python -m pytest tests/test_pages_have_auth_guard.py -v`
Expected: PASS — the existing auth-guard structural test still finds 13 pages and the auth wiring is intact (proves the edits didn't disturb the `inject_custom_css()` / `require_auth()` ordering).

- [ ] **Step 6: Commit**

```bash
git add pages/1_My_Team.py pages/2_Line-up_Optimizer.py pages/3_Closer_Monitor.py \
        pages/5_Matchup_Planner.py pages/6_League_Standings.py pages/10_Punt_Analyzer.py \
        pages/11_Trade_Analyzer.py pages/12_Trade_Finder.py pages/14_Free_Agents.py \
        pages/16_Player_Compare.py pages/17_Leaders.py pages/19_Player_Databank.py \
        pages/20_Draft_Simulator.py
git commit -m "feat(pages): wire usage logging + feedback widget into 13 pages (v2 Plan 2)"
```

---

## Task 10: Back-compat guard (flag OFF ⇒ both helpers are no-ops, zero DB writes)

**Files:**
- Test: `tests/test_feedback_usage_backcompat.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_feedback_usage_backcompat.py`:

```python
"""With MULTI_USER off, usage + feedback hooks are inert (v1 byte-for-byte)."""

import pytest

import src.feedback as feedback
import src.usage as usage


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


def _usage_count() -> int:
    from src.database import get_connection

    conn = get_connection()
    try:
        return conn.execute("SELECT COUNT(*) FROM usage_events").fetchone()[0]
    finally:
        conn.close()


def test_log_page_view_noop_when_flag_off(temp_db, monkeypatch):
    # Even with a user present, flag-off must write nothing.
    state: dict = {}
    monkeypatch.setattr(usage, "_session_state", lambda: state)
    monkeypatch.setattr(usage, "current_user", lambda: {"user_id": 1})
    usage.log_page_view("My Team")
    assert _usage_count() == 0
    assert "_usage_session_id" not in state


def test_render_feedback_widget_noop_when_flag_off(monkeypatch):
    monkeypatch.setattr(feedback, "current_user", lambda: {"user_id": 1})
    monkeypatch.setattr(feedback.st, "popover", lambda *a, **k: pytest.fail("popover called"))
    feedback.render_feedback_widget("My Team")  # must return without rendering
```

- [ ] **Step 2: Run test to verify it passes immediately**

Run: `python -m pytest tests/test_feedback_usage_backcompat.py -v`
Expected: PASS (2 passed).

> This is a regression *lock*, not new behavior — the no-op guards were already written in Tasks 5 & 6. Confirming green here proves flag-off makes both hooks inert. (`multi_user_enabled()` reads the real env, which `_flag_off` clears, so no monkeypatch of the flag is needed.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_feedback_usage_backcompat.py
git commit -m "test(v2): back-compat guard — feedback+usage inert when MULTI_USER off"
```

---

## Task 11: Documentation (CLAUDE.md)

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add the three modules to the File Structure "Multi-user (v2)" block**

Find:

```text
  # Multi-user (v2)
  auth.py                   — register→pending→admin-approve→active; MULTI_USER-gated; per-session identity
```

Replace with:

```text
  # Multi-user (v2)
  auth.py                   — register→pending→admin-approve→active; MULTI_USER-gated; per-session identity
  version.py                — APP_VERSION string (env-overridable via HEATER_APP_VERSION); stamped onto feedback
  feedback.py               — per-feature feedback capture + admin inbox helpers + render_feedback_widget popover (MULTI_USER-gated)
  usage.py                  — per-session deduped page-view logging + last_seen_at bump (MULTI_USER-gated)
```

- [ ] **Step 2: Add a structural-invariant table row**

Find the row (in the Structural Invariants table):

```text
| `test_auth_backcompat.py` | `MULTI_USER` off ⇒ `require_auth()` is a no-op and `_get_user_team_name` uses the `league_teams.is_user_team=1` query (v1 behavior preserved) |
```

Add immediately after it:

```text
| `test_pages_have_feedback_and_usage.py` | Every interactive `pages/*.py` (those calling `inject_custom_css()`, excluding `00_Admin_Console.py`) imports `log_page_view` + `render_feedback_widget`; `log_page_view()` is called after `require_auth()`; `render_feedback_widget()` is called on the page. Per-page because Streamlit re-runs each page independently (v2 Plan 2) |
| `test_feedback_usage_tables.py` | `init_db()` creates the `feedback` + `usage_events` tables (additive, idempotent). feedback carries `app_version` + `data_state` (JSON refresh_log snapshot); usage_events dedupes per `(page, action)` per `session_id` (v2 Plan 2) |
| `test_feedback_usage_backcompat.py` | `MULTI_USER` off ⇒ `log_page_view()` and `render_feedback_widget()` are no-ops (zero DB writes, no popover) — v1 byte-for-byte (v2 Plan 2) |
```

- [ ] **Step 3: Note the feature in the Multi-user-mode overview paragraph**

Find (in the "Local Environment" section, the multi-user bullet):

```text
- **Multi-user mode (v2, additive):** Set `MULTI_USER=1` to enable league-mate self-registration + admin-approved team assignment. Off (unset) = single-user v1 behavior, byte-for-byte. Admin is seeded from `ADMIN_USERNAME` / `ADMIN_PASSWORD` / `ADMIN_TEAM_NAME` env vars (idempotent — set once). Auth lives entirely in `src/auth.py`; identity is per-session (`st.session_state["auth_user"]`) and replaces the global `league_teams.is_user_team` flag for personalized surfaces only.
```

Append to the end of that same bullet (one sentence):

```text
 Plan 2 adds two additive, MULTI_USER-gated surfaces: a per-feature feedback inbox (`src/feedback.py` → Admin Console "Feedback" tab) and per-page usage logging (`src/usage.py`, one deduped view per session + `last_seen_at`). Both are inert when the flag is off.
```

- [ ] **Step 4: Verify the docs still describe reality (no test to run — read-back check)**

Re-read the three edited regions in `CLAUDE.md` and confirm the module names + test names match what was actually created in Tasks 1-10.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude.md): document feedback/usage modules + guards (v2 Plan 2)"
```

---

## Task 12: Final verification (lint + full suite)

**Files:** none (verification only)

- [ ] **Step 1: Lint + format the whole repo**

Run: `python -m ruff check . && python -m ruff format --check .`
Expected: `All checks passed!` and the format check reports no files would be reformatted. If `format --check` flags a file, run `python -m ruff format .`, re-stage, and amend the relevant task's commit (do NOT create a stray formatting commit mid-history — prefer fixing within the task).

- [ ] **Step 2: Run the full Plan 2 test set together**

Run:

```bash
python -m pytest tests/test_version.py tests/test_feedback_usage_tables.py \
  tests/test_feedback_capture.py tests/test_feedback_inbox.py tests/test_feedback_widget.py \
  tests/test_usage_logging.py tests/test_admin_console_feedback_tab.py \
  tests/test_pages_have_feedback_and_usage.py tests/test_feedback_usage_backcompat.py \
  tests/test_admin_console_guarded.py tests/test_pages_have_auth_guard.py \
  tests/test_auth_backcompat.py -v
```

Expected: all pass — the 9 new files plus the 3 adjacent v2 guards.

- [ ] **Step 3: Run the full suite (parallel, matches CI layout)**

Run: `python -m pytest --ignore=tests/test_cheat_sheet.py -n auto --dist loadfile -q`
Expected: green (~4534 + the ~26 new Plan 2 tests). No new failures, no network-guard `NetworkBlockedError` (all data-state captures are mocked).

- [ ] **Step 4: Manual UI smoke (optional but recommended)**

```bash
# Flag ON: feedback popover appears on pages; Admin Console shows the Feedback tab.
# PowerShell:  $env:MULTI_USER = "1";  streamlit run app.py
# Flag OFF (default): no popover anywhere, no behavioral change vs v1.
```

Verify: with `MULTI_USER=1`, log in as the bootstrap admin, open any page → a "Send feedback on this" popover renders; submit a message; open Admin Console → "Feedback" tab → the message appears with the submitter, page, version, and a "Data state" expander. With the flag unset, no popover appears and the Admin Console has no behavioral change beyond the (empty) Feedback tab being inert.

- [ ] **Step 5: Hand off to finishing-a-development-branch**

Announce: "I'm using the finishing-a-development-branch skill to complete this work." Then follow that skill (verify tests → present merge/PR options → execute choice).

---

## Self-Review (completed at plan-authoring time)

**1. Spec coverage:** Every spec section maps to a task — Data Model §3 → Task 2; capture/version §4 → Tasks 1, 3; inbox §4 → Tasks 4, 7; widget §4 → Task 5; usage §4 → Task 6; page integration §5 → Tasks 8, 9; edge cases §6 (dedup, best-effort data-state, status enum, flag-off no-op) → Tasks 3, 4, 6, 10; testing §7 → all `test_*` steps; back-compat §8 → Task 10; file manifest §9 → File Structure section above. No spec requirement is unaddressed. (Deferred-to-Plan-3 items — usage analytics dashboard, feature flags, audit log — are intentionally absent.)

**2. Placeholder scan:** No "TBD"/"implement later"/"add error handling"/"similar to Task N". Every code step contains complete, runnable code. The only `<LABEL>` placeholders (Task 9) are immediately resolved by the explicit per-page table and call list in the same task.

**3. Type consistency:** `submit_feedback(user_id, page, message, feature_tag=None) -> int` is produced in Task 3 and consumed identically in Task 5 and the tests. `list_feedback(status=None) -> list[dict]`, `set_feedback_status(id, status)`, `set_feedback_notes(id, notes)` (Task 4) are consumed identically in Task 7 and the inbox test. `log_page_view(page, action="view")` (Task 6) matches all call sites (Task 9 pages + tests). `_capture_data_state() -> str | None` and `_VALID_STATUSES` are defined once and referenced consistently. Column names (`feedback.id/user_id/page/feature_tag/message/app_version/data_state/status/admin_notes/created_at`; `usage_events.id/user_id/page/action/session_id/created_at`) in Task 2 match every SQL string in Tasks 3, 4, 6, 7 and the tests.

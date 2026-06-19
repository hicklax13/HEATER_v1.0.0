# B2.0 — Engine Seam (backward-compatible) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax. This is **Phase B2.0** of the owner-ratified B2 Postgres migration (`docs/superpowers/plans/2026-06-19-heater-backend-b2-postgres-migration-plan.md`). The owner has green-lit B2.0 specifically (the backward-compatible first step).

**Goal:** Introduce a SQLAlchemy **engine seam** (`src/db/engine.py`) selected by `DATABASE_URL` (absent → today's SQLite file), as the single future swap point for Postgres — while leaving the live app **byte-for-byte unchanged on SQLite**.

**Architecture:** Add `SQLAlchemy` (only — `alembic` lands in B2.1, `psycopg` in B2.2). New `src/db/engine.py` exposes `engine_url()`, `get_engine()` (cached), `reset_engine_cache()`. `get_connection()` keeps its **exact** SQLite behavior (raw `sqlite3.connect` + `row_factory=sqlite3.Row` + WAL/busy_timeout PRAGMAs); it only gains a guard branch that raises a clear `NotImplementedError` if a non-SQLite `DATABASE_URL` is set (real Postgres connection wiring is B2.2). B2.0 does **not** migrate any `read_sql`/query (B2.2) and does **not** wire Postgres connections — it establishes and tests the seam, which B2.1 (Alembic) consumes next.

**Tech Stack:** SQLAlchemy 2.0, sqlite3, pytest + monkeypatch.

**Conventions:** Run from the worktree root `C:\Users\conno\Code\HEATER_v1.0.1\.claude\worktrees\b2-slice0-engine-seam`. `PY` = `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe`. Lint touched `.py` with `$PY -m ruff format` + `$PY -m ruff check`.

**⚠️ THE BAR (this touches the live data path):** the **full test suite** must be green BEFORE (baseline) and AFTER — not just `tests/api/`. Run `$PY -m pytest --ignore=tests/test_cheat_sheet.py -n auto --dist loadfile -q` (the CLAUDE.md parallel command, ~155-160s). SQLite `get_connection()` behavior must be **identical** — if any existing DB test changes behavior, STOP and report.

## File Structure
- **Modify** `requirements.txt` — add `SQLAlchemy>=2.0` (one line; NOT alembic/psycopg yet).
- **Create** `src/db/__init__.py` (empty) + `src/db/engine.py` — the engine seam.
- **Modify** `src/database.py` — `get_connection()` gains the `DATABASE_URL` guard branch (SQLite path unchanged); add the `import` of the engine helpers lazily inside the function to avoid import-time cycles.
- **Create** `tests/test_db_engine_seam.py` — engine + get_connection-branch tests.

No change to the 482 `get_connection` call sites, no `read_sql` migration, no `api/` change, no `pages/` change.

---

### Task 1: Add SQLAlchemy dependency

**Files:** Modify `requirements.txt`; Test: `tests/test_db_engine_seam.py`

- [ ] **Step 1: Baseline — full suite green BEFORE any change**

Run: `$PY -m pytest --ignore=tests/test_cheat_sheet.py -n auto --dist loadfile -q`
Expected: all green (record the count). If anything fails pre-change, STOP and report — do not proceed.

- [ ] **Step 2: Write the failing import test**

Create `tests/test_db_engine_seam.py`:

```python
"""B2.0 engine-seam guards. SQLite stays the default + byte-identical; the
SQLAlchemy engine is the future Postgres swap point (selected by DATABASE_URL)."""

import sqlite3

import pytest


def test_sqlalchemy_is_available():
    import sqlalchemy  # noqa: F401
```

- [ ] **Step 3: Run to verify it fails**

Run: `$PY -m pytest tests/test_db_engine_seam.py::test_sqlalchemy_is_available -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'sqlalchemy'`.

- [ ] **Step 4: Add the dep + install**

Add to `requirements.txt` (near the other core libs, with a brief comment):
```
SQLAlchemy>=2.0          # B2.0 engine seam (DB backend abstraction; alembic/psycopg land in B2.1/B2.2)
```
Install into the shared venv:
Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pip install "SQLAlchemy>=2.0"`

- [ ] **Step 5: Run to verify it passes**

Run: `$PY -m pytest tests/test_db_engine_seam.py::test_sqlalchemy_is_available -q`
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt tests/test_db_engine_seam.py
git commit -m "build(db): add SQLAlchemy dep for the B2.0 engine seam"
```

---

### Task 2: The engine module (`src/db/engine.py`)

**Files:** Create `src/db/__init__.py`, `src/db/engine.py`; Test: `tests/test_db_engine_seam.py`

- [ ] **Step 1: Write the failing engine tests**

Append to `tests/test_db_engine_seam.py`:

```python
def test_engine_url_defaults_to_sqlite(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from src.db.engine import engine_url, reset_engine_cache

    reset_engine_cache()
    url = engine_url()
    assert url.startswith("sqlite:///")  # local file backend by default


def test_engine_url_honors_database_url(monkeypatch):
    # engine_url() returns the string WITHOUT importing a driver, so this is
    # safe even though psycopg is not installed until B2.2.
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/heater")
    from src.db.engine import engine_url, reset_engine_cache

    reset_engine_cache()
    assert engine_url() == "postgresql+psycopg://u:p@localhost:5432/heater"


def test_engine_connects_and_queries_sqlite(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from sqlalchemy import text

    from src.db.engine import get_engine, reset_engine_cache

    reset_engine_cache()
    with get_engine().connect() as conn:
        assert conn.execute(text("SELECT 1")).scalar() == 1


def test_engine_is_cached(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from src.db.engine import get_engine, reset_engine_cache

    reset_engine_cache()
    assert get_engine() is get_engine()  # one engine per process (pool reuse)
```

- [ ] **Step 2: Run to verify it fails**

Run: `$PY -m pytest tests/test_db_engine_seam.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.db'` (or `engine`).

- [ ] **Step 3: Implement the engine module**

Create empty `src/db/__init__.py`.

Create `src/db/engine.py`:

```python
"""SQLAlchemy engine seam (B2.0) — the single place the DB backend is chosen.

Today the default is the existing SQLite file (zero behavior change). Set
DATABASE_URL to point at Postgres later (the connection wiring for non-SQLite
backends lands in B2.2). `get_connection()` in src/database.py still owns direct
sqlite3 cursor access on SQLite; this module is the engine the rest of the
migration (Alembic in B2.1, the read/query port in B2.2) builds on."""

from __future__ import annotations

import os

from sqlalchemy import Engine, create_engine

_engine: Engine | None = None


def engine_url() -> str:
    """The SQLAlchemy URL for the active backend. DATABASE_URL if set, else the
    local SQLite file (resolved via the same path logic as direct connections)."""
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    from src.database import _resolve_db_path

    return f"sqlite:///{_resolve_db_path().as_posix()}"


def get_engine() -> Engine:
    """Process-wide cached engine. Lazy so importing this module never connects
    and never imports a driver that isn't installed (e.g. psycopg pre-B2.2)."""
    global _engine
    if _engine is None:
        _engine = create_engine(engine_url(), future=True)
    return _engine


def reset_engine_cache() -> None:
    """Dispose + drop the cached engine (tests that switch DATABASE_URL)."""
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = None


def is_sqlite_backend() -> bool:
    """True when the active backend is the local SQLite file (the default)."""
    return engine_url().startswith("sqlite")
```

- [ ] **Step 4: Run to verify it passes**

Run: `$PY -m pytest tests/test_db_engine_seam.py -q`
Expected: 5 passed.

- [ ] **Step 5: Lint + commit**

```bash
$PY -m ruff format src/db/__init__.py src/db/engine.py tests/test_db_engine_seam.py
$PY -m ruff check src/db/engine.py tests/test_db_engine_seam.py
git add src/db/__init__.py src/db/engine.py tests/test_db_engine_seam.py
git commit -m "feat(db): add SQLAlchemy engine seam (DATABASE_URL → backend; SQLite default) (B2.0)"
```

---

### Task 3: `get_connection()` backend guard (SQLite unchanged)

**Files:** Modify `src/database.py` (`get_connection`); Test: `tests/test_db_engine_seam.py`

- [ ] **Step 1: Write the failing guard tests**

Append to `tests/test_db_engine_seam.py`:

```python
def test_get_connection_sqlite_unchanged(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from src.database import get_connection

    conn = get_connection()
    try:
        assert isinstance(conn, sqlite3.Connection)
        assert conn.row_factory is sqlite3.Row  # dict-row access preserved
        assert conn.execute("SELECT 1").fetchone()[0] == 1
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    finally:
        conn.close()


def test_get_connection_rejects_unwired_postgres(monkeypatch):
    # B2.0 establishes the seam but does NOT wire Postgres connections — that is
    # B2.2. A non-SQLite DATABASE_URL must fail LOUD, never silently mis-connect.
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/heater")
    from src.database import get_connection
    from src.db.engine import reset_engine_cache

    reset_engine_cache()
    with pytest.raises(NotImplementedError):
        get_connection()
    reset_engine_cache()  # leave cache clean for other tests
```

- [ ] **Step 2: Run to verify it fails**

Run: `$PY -m pytest tests/test_db_engine_seam.py::test_get_connection_rejects_unwired_postgres -q`
Expected: FAIL — no `NotImplementedError` raised (get_connection ignores DATABASE_URL today).

- [ ] **Step 3: Add the guard branch to `get_connection`**

In `src/database.py`, at the TOP of `get_connection()` (before the existing `sqlite3.connect` logic), add the backend guard. The existing SQLite body stays exactly as-is below it:

```python
def get_connection(timeout: float = 60.0) -> sqlite3.Connection:
    # B2.0 seam: SQLite (the default) keeps the exact behavior below. A non-SQLite
    # DATABASE_URL means Postgres — whose connection wiring lands in B2.2 — so fail
    # loud rather than silently fall through to a SQLite file.
    from src.db.engine import is_sqlite_backend

    if not is_sqlite_backend():
        raise NotImplementedError(
            "Non-SQLite DATABASE_URL set, but Postgres connection wiring is not "
            "implemented until B2.2. Unset DATABASE_URL to use the SQLite backend."
        )
    # --- existing SQLite implementation unchanged from here ---
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=timeout)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=60000")
        conn.execute("PRAGMA synchronous=NORMAL")
    except sqlite3.Error:
        pass
    return conn
```

(Keep the rest of the existing body verbatim — only the guard branch + import are new.)

- [ ] **Step 4: Run to verify the guard tests pass**

Run: `$PY -m pytest tests/test_db_engine_seam.py -q`
Expected: 7 passed.

- [ ] **Step 5: Confirm the seam structural guard still passes**

Run: `$PY -m pytest tests/test_no_direct_sqlite_connect_in_scripts.py -q`
Expected: green — `get_connection` is still the sanctioned `sqlite3.connect` site; nothing new connects directly.

- [ ] **Step 6: Lint + commit**

```bash
$PY -m ruff format src/database.py tests/test_db_engine_seam.py
$PY -m ruff check src/database.py tests/test_db_engine_seam.py
git add src/database.py tests/test_db_engine_seam.py
git commit -m "feat(db): get_connection guards non-SQLite DATABASE_URL until B2.2 (B2.0)"
```

---

### Task 4: Full-suite verification (the live-path bar)

**Files:** none (verification only)

- [ ] **Step 1: Run the FULL suite (parallel)**

Run: `$PY -m pytest --ignore=tests/test_cheat_sheet.py -n auto --dist loadfile -q`
Expected: same pass count as the Task-1 baseline + the 7 new engine-seam tests, **0 failures**. If ANY previously-passing test now fails, STOP and report (the SQLite path must be byte-identical).

- [ ] **Step 2: Lint the changeset**

Run: `$PY -m ruff format --check src/db/engine.py src/database.py tests/test_db_engine_seam.py && $PY -m ruff check src/db/ src/database.py tests/test_db_engine_seam.py`
Expected: clean.

- [ ] **Step 3: Confirm clean tree + commits**

Run: `git status --short && git log --oneline <baseline>..HEAD`
Expected: clean; 3 B2.0 commits (dep, engine module, get_connection guard).

Hand back to the orchestrator for the 2-stage review.

---

## Self-Review
**Spec coverage (vs the B2 plan's Phase B2.0):** engine seam selected by DATABASE_URL (default SQLite) ✅ (Task 2); SQLite `get_connection` byte-identical ✅ (Task 3 keeps the body verbatim + `test_get_connection_sqlite_unchanged`); backward-compatible / suite green ✅ (Task 4 full-suite bar); `psycopg`/`alembic` NOT added (deferred to B2.2/B2.1) ✅; no `read_sql` migration (B2.2) ✅. **Placeholders:** none. **Type consistency:** `engine_url() -> str`, `get_engine() -> Engine`, `reset_engine_cache() -> None`, `is_sqlite_backend() -> bool` used identically in tests + `get_connection`. The `_resolve_db_path()` import inside `engine_url()` matches its definition in `src/database.py` (survey §1).

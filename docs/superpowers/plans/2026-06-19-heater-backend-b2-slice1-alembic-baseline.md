# B2.1 — Alembic Baseline + Schema Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax. Phase B2.1 of the owner-ratified B2 Postgres migration. Builds on B2.0 (the engine seam — already on master). PG-only scaffolding: it does **NOT** touch the live SQLite path (`init_db()` stays raw-SQL and unchanged).

**Goal:** Stand up Alembic and author a **baseline migration** that builds the full HEATER schema (53 tables + the 62 boot-time `_safe_add_column` columns + indexes/constraints) in a **dialect-aware** SQLAlchemy `MetaData` (`src/db/schema.py`), so `alembic upgrade head` builds the schema on Postgres (and SQLite). `init_db()` keeps owning the live SQLite path unchanged; a **schema-parity test** guarantees the Alembic schema and `init_db()` agree.

**Architecture:** New `src/db/schema.py` defines every table as a SQLAlchemy `Table` on a shared `MetaData`, dialect-aware (autoincrement PKs render `SERIAL` on PG / `INTEGER PK` on SQLite automatically; the 2 `COLLATE NOCASE` columns use `String(...).with_variant(postgresql.CITEXT(), "postgresql")`; partial unique indexes use `postgresql_where=` + `sqlite_where=`; single-row `CHECK(id=1)` and `CHECK(... IN ...)` as `CheckConstraint`; TEXT timestamps stay `Text`; 0/1 boolean flags stay `Integer`). Alembic's baseline migration creates this MetaData (`metadata.create_all(bind)`), preceded on PG only by `CREATE EXTENSION IF NOT EXISTS citext`. `alembic/env.py` resolves its URL from the B2.0 engine seam (`engine_url()`). **The 53-table completeness is enforced by a parity test, not by listing tables here** — the implementer translates every `CREATE TABLE` in `init_db()` into `schema.py` and iterates until parity passes.

**Tech Stack:** Alembic, SQLAlchemy 2.0, sqlite3, pytest. (SQLAlchemy + the engine seam are already in place from B2.0.)

**Conventions:** Worktree root `C:\Users\conno\Code\HEATER_v1.0.1\.claude\worktrees\b2-slice1-alembic-baseline`. `PY` = `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe`. Lint touched `.py` with ruff.

**⚠️ VERIFICATION BAR (no Postgres server in this env):**
- `alembic upgrade head` against a temp **SQLite** DB must create all tables (structure runs).
- **PG-dialect compilation** tests assert PG-correctness (SERIAL/citext/partial-index) WITHOUT a server.
- **Schema-parity** test: `schema.py` tables+columns == `init_db()` SQLite tables+columns.
- `init_db()` SQLite path **unchanged** + the **full suite green** (this is still program infra).
- DEFERRED (note in code + report): the real `alembic upgrade head` against a live Postgres — runs when a PG instance exists (owner starts Docker or provides a dev PG URL). Do NOT claim real-PG verification.

## File Structure
- **Modify** `requirements.txt` — add `alembic>=1.13` (NOT psycopg — that's B2.2).
- **Create** `src/db/schema.py` — the dialect-aware `MetaData` with all 53 tables (the bulk of the work).
- **Create** `alembic.ini`, `alembic/env.py`, `alembic/script.py.mako`, `alembic/versions/0001_baseline.py` — Alembic scaffolding + baseline migration.
- **Create** `tests/test_db_alembic_baseline.py` — parity + SQLite-upgrade + PG-compilation tests.

No change to `src/database.py` `init_db()` (the live SQLite path), no `pages/`, no `api/`.

---

### Task 1: Alembic dependency + scaffolding wired to the engine seam

**Files:** Modify `requirements.txt`; Create `alembic.ini`, `alembic/env.py`, `alembic/script.py.mako`; Test: `tests/test_db_alembic_baseline.py`

- [ ] **Step 1: Baseline — full suite green BEFORE**

Run: `$PY -m pytest --ignore=tests/test_cheat_sheet.py -n auto --dist loadfile -q` — record the count. If anything fails pre-change, STOP.

- [ ] **Step 2: Write the failing scaffolding test**

Create `tests/test_db_alembic_baseline.py`:

```python
"""B2.1 Alembic baseline guards. Alembic builds the full schema (dialect-aware)
on SQLite (and renders PG-correct DDL); it must match init_db()'s schema. The
real `alembic upgrade head` on Postgres is deferred until a PG instance exists."""

import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_alembic_is_available():
    import alembic  # noqa: F401


def test_alembic_scaffolding_exists():
    assert (_ROOT / "alembic.ini").is_file()
    assert (_ROOT / "alembic" / "env.py").is_file()
    assert (_ROOT / "alembic" / "versions").is_dir()
```

- [ ] **Step 3: Run to verify it fails**

Run: `$PY -m pytest tests/test_db_alembic_baseline.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'alembic'`.

- [ ] **Step 4: Add dep + init Alembic**

Add to `requirements.txt`: `alembic>=1.13              # B2.1 schema migrations (Postgres-owned schema)`
Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pip install "alembic>=1.13"`
Run: `$PY -m alembic init alembic`  (creates `alembic.ini`, `alembic/env.py`, `alembic/script.py.mako`, `alembic/versions/`)

- [ ] **Step 5: Wire `alembic/env.py` to the engine seam + schema MetaData**

Replace the generated `alembic/env.py` body so it (a) resolves the URL from `src.db.engine.engine_url()` (NOT a hardcoded alembic.ini URL), and (b) sets `target_metadata` to the schema MetaData (created in Task 2 — import lazily/guarded so Task-1 tests pass before schema.py exists). Use:

```python
from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from src.db.engine import engine_url

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Schema is the single source for autogenerate/metadata; imported here so
# `alembic` commands see it. (Created in B2.1 Task 2.)
try:
    from src.db.schema import metadata as target_metadata
except Exception:  # during Task 1, schema.py may not exist yet
    target_metadata = None


def _url() -> str:
    return engine_url()


def run_migrations_offline() -> None:
    context.configure(url=_url(), target_metadata=target_metadata, literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _url()
    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

(Leave `alembic.ini`'s `sqlalchemy.url` as the harmless default — `env.py` overrides it. Optionally set it to a comment noting the URL comes from `engine_url()`.)

- [ ] **Step 6: Run to verify the scaffolding tests pass**

Run: `$PY -m pytest tests/test_db_alembic_baseline.py -q`
Expected: 2 passed (`test_alembic_is_available`, `test_alembic_scaffolding_exists`).

- [ ] **Step 7: Lint + commit**

```bash
$PY -m ruff check alembic/env.py tests/test_db_alembic_baseline.py
git add requirements.txt alembic.ini alembic/ tests/test_db_alembic_baseline.py
git commit -m "build(db): add Alembic + scaffolding wired to the engine seam (B2.1)"
```

---

### Task 2: The dialect-aware schema MetaData (`src/db/schema.py`)

**Files:** Create `src/db/schema.py`; Test: `tests/test_db_alembic_baseline.py`

> This is the bulk of B2.1: translate EVERY `CREATE TABLE` in `src/database.py`'s `init_db()` (the three `conn.executescript(...)` blocks + `_init_db_tables_and_columns` + `_init_multiuser_tables`) AND every `_safe_add_column(conn, table, col, type)` column into a SQLAlchemy `Table` on a shared `MetaData`. The parity test (below) FAILS until all 53 tables + all columns are present, so iterate against it.

**Translation rules (apply per column/table):**
- Autoincrement integer PK (`INTEGER PRIMARY KEY AUTOINCREMENT` / `INTEGER PRIMARY KEY`) → `Column("id", Integer, primary_key=True)` (SQLAlchemy renders `SERIAL`/identity on PG, `INTEGER PRIMARY KEY` on SQLite). Do NOT write `SERIAL` literally.
- `TEXT` → `Text`; `INTEGER` → `Integer`; `REAL` → `Float`. 0/1 boolean flag columns (`is_hitter`, `is_injured`, `is_user_team`, `is_admin`, `enabled`, `revoked`, `is_active`, `is_undroppable`, etc.) STAY `Integer` (do not convert to Boolean).
- Timestamp-as-text columns (`created_at`, `updated_at`, `fetched_at`, `last_updated`, `last_seen_at`, …) STAY `Text` (native timestamptz is a later, separate cleanup — out of B2.1).
- The 2 case-insensitive-unique columns (`users.username` and the player-name dedup column — find them via `COLLATE NOCASE` in `init_db()`) → `Column(..., String().with_variant(postgresql.CITEXT(), "postgresql"))`. Import `from sqlalchemy.dialects import postgresql`.
- Unique/PK/FK/`CHECK` constraints → `UniqueConstraint`, `PrimaryKeyConstraint`, `ForeignKey`, `CheckConstraint` (e.g. `league_settings` single-row `CHECK(id = 1)`; `player_tags.tag IN (...)`).
- Partial unique indexes (the 4 on `player_id_map ... WHERE col IS NOT NULL`) → `Index("ix_name", "col", unique=True, postgresql_where=text("col IS NOT NULL"), sqlite_where=text("col IS NOT NULL"))`.
- Plain indexes → `Index(...)`.
- Default values (`DEFAULT 0`, `DEFAULT ''`, `datetime('now')`, `CURRENT_TIMESTAMP`) → `server_default=text("...")` where present (use `func.now()`/`text("CURRENT_TIMESTAMP")` for timestamps; keep `0`/`''` literals). Match init_db's defaults.

**Worked example (pattern to follow for all tables):**

```python
"""Dialect-aware SQLAlchemy schema (B2.1) — the Postgres-side source of truth
the Alembic baseline builds. Mirrors init_db()'s SQLite schema exactly (enforced
by tests/test_db_alembic_baseline.py::test_schema_matches_init_db). init_db()
still owns the live SQLite path; this is additive."""

from __future__ import annotations

from sqlalchemy import (
    CheckConstraint, Column, Float, Index, Integer, MetaData, String, Table, Text, UniqueConstraint, text,
)
from sqlalchemy.dialects import postgresql

metadata = MetaData()

players = Table(
    "players",
    metadata,
    Column("player_id", Integer, primary_key=True),  # SERIAL on PG, INTEGER PK on SQLite
    Column("mlb_id", Integer),
    Column("name", Text),
    Column("is_hitter", Integer),  # 0/1 flag stays Integer
    # ... every other players column, INCLUDING those added via _safe_add_column ...
)

users = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("username", String().with_variant(postgresql.CITEXT(), "postgresql")),  # was COLLATE NOCASE
    # ...
    UniqueConstraint("username", name="uq_users_username"),
)

Index(
    "ix_player_id_map_mlb",
    player_id_map.c.mlb_id,  # define the player_id_map Table above this
    unique=True,
    postgresql_where=text("mlb_id IS NOT NULL"),
    sqlite_where=text("mlb_id IS NOT NULL"),
)
# ... all 53 tables ...
```

- [ ] **Step 1: Write the failing parity test**

Append to `tests/test_db_alembic_baseline.py`:

```python
import os
import sqlite3


def _init_db_schema(tmp_path) -> dict[str, set[str]]:
    """Run the real init_db() on a fresh temp SQLite DB; return {table: {cols}}."""
    db = tmp_path / "parity.db"
    os.environ["HEATER_DB_PATH"] = str(db)
    # Re-resolve the module-level DB_PATH against the temp path.
    import importlib

    import src.database as database

    importlib.reload(database)
    database.init_db()
    conn = sqlite3.connect(str(db))
    try:
        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            if not r[0].startswith("sqlite_") and r[0] != "alembic_version"
        }
        return {t: {c[1] for c in conn.execute(f"PRAGMA table_info({t})")} for t in tables}
    finally:
        conn.close()


def test_schema_matches_init_db(tmp_path):
    from src.db.schema import metadata

    init_schema = _init_db_schema(tmp_path)
    meta_tables = {name: {c.name for c in tbl.columns} for name, tbl in metadata.tables.items()}

    # Every init_db table exists in the MetaData with (at least) the same columns.
    missing_tables = set(init_schema) - set(meta_tables)
    assert not missing_tables, f"schema.py is missing tables: {sorted(missing_tables)}"
    for t, cols in init_schema.items():
        missing_cols = cols - meta_tables[t]
        assert not missing_cols, f"schema.py table {t} missing columns: {sorted(missing_cols)}"
    # No stray extra tables (alembic_version excluded above).
    extra = set(meta_tables) - set(init_schema)
    assert not extra, f"schema.py has tables not in init_db: {sorted(extra)}"
```

(Note: reloading `src.database` mutates global state for the process; run this test in isolation if needed. If the reload approach is brittle under xdist, the implementer may instead spawn a subprocess that sets `HEATER_DB_PATH`, runs `init_db()`, and prints the schema as JSON — choose whichever is reliable, but the assertion semantics above are the contract.)

- [ ] **Step 2: Run to verify it fails**

Run: `$PY -m pytest tests/test_db_alembic_baseline.py::test_schema_matches_init_db -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.db.schema'`.

- [ ] **Step 3: Author `src/db/schema.py`** (translate ALL 53 tables per the rules + example above). Read `init_db()` in `src/database.py` as the source of truth; fold in every `_safe_add_column` column.

- [ ] **Step 4: Iterate until parity passes**

Run: `$PY -m pytest tests/test_db_alembic_baseline.py::test_schema_matches_init_db -q`
Expected: eventually PASS. The failure messages name exactly which tables/columns are missing — add them until green. Do NOT edit the test to pass; fix `schema.py`.

- [ ] **Step 5: Lint + commit**

```bash
$PY -m ruff format src/db/schema.py tests/test_db_alembic_baseline.py
$PY -m ruff check src/db/schema.py tests/test_db_alembic_baseline.py
git add src/db/schema.py tests/test_db_alembic_baseline.py
git commit -m "feat(db): dialect-aware SQLAlchemy schema MetaData mirroring init_db (B2.1)"
```

---

### Task 3: The baseline migration + SQLite-upgrade verification

**Files:** Create `alembic/versions/0001_baseline.py`; Test: `tests/test_db_alembic_baseline.py`

- [ ] **Step 1: Write the failing upgrade test**

Append to `tests/test_db_alembic_baseline.py`:

```python
def test_alembic_upgrade_head_builds_schema_on_sqlite(tmp_path, monkeypatch):
    from alembic import command
    from alembic.config import Config

    db = tmp_path / "alembic.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db.as_posix()}")
    from src.db.engine import reset_engine_cache

    reset_engine_cache()
    cfg = Config(str(_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(_ROOT / "alembic"))
    command.upgrade(cfg, "head")  # must not raise

    import sqlite3

    conn = sqlite3.connect(str(db))
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        from src.db.schema import metadata

        for t in metadata.tables:
            assert t in tables, f"alembic upgrade did not create {t}"
    finally:
        conn.close()
        reset_engine_cache()
```

- [ ] **Step 2: Run to verify it fails**

Run: `$PY -m pytest tests/test_db_alembic_baseline.py::test_alembic_upgrade_head_builds_schema_on_sqlite -q`
Expected: FAIL — no baseline revision yet (nothing created).

- [ ] **Step 3: Author the baseline migration** `alembic/versions/0001_baseline.py`:

```python
"""baseline schema

Revision ID: 0001_baseline
Revises:
Create Date: (static — Date.now is unavailable in this env)
"""

from __future__ import annotations

from alembic import op

from src.db.schema import metadata

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS citext")
    metadata.create_all(bind)


def downgrade() -> None:
    metadata.drop_all(op.get_bind())
```

- [ ] **Step 4: Run to verify the upgrade test passes**

Run: `$PY -m pytest tests/test_db_alembic_baseline.py::test_alembic_upgrade_head_builds_schema_on_sqlite -q`
Expected: PASS (all MetaData tables created on the temp SQLite DB).

- [ ] **Step 5: Lint + commit**

```bash
$PY -m ruff check alembic/versions/0001_baseline.py tests/test_db_alembic_baseline.py
git add alembic/versions/0001_baseline.py tests/test_db_alembic_baseline.py
git commit -m "feat(db): Alembic baseline migration builds the schema (citext ext on PG) (B2.1)"
```

---

### Task 4: PG-dialect correctness tests (no server)

**Files:** Test: `tests/test_db_alembic_baseline.py`

- [ ] **Step 1: Write the PG-compilation tests**

Append:

```python
def _pg_ddl(table) -> str:
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.schema import CreateTable

    return str(CreateTable(table).compile(dialect=postgresql.dialect()))


def test_autoincrement_pk_renders_serial_on_pg():
    from src.db.schema import metadata

    players = metadata.tables["players"]
    ddl = _pg_ddl(players).upper()
    assert "SERIAL" in ddl  # autoincrement PK → SERIAL on Postgres


def test_username_is_citext_on_pg():
    from src.db.schema import metadata

    ddl = _pg_ddl(metadata.tables["users"]).upper()
    assert "CITEXT" in ddl  # case-insensitive unique on Postgres


def test_partial_unique_index_renders_where_on_pg():
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.schema import CreateIndex

    from src.db.schema import metadata

    pim = metadata.tables["player_id_map"]
    partials = [ix for ix in pim.indexes if ix.dialect_options.get("postgresql", {}).get("where") is not None]
    assert partials, "expected partial unique index(es) on player_id_map"
    ddl = str(CreateIndex(partials[0]).compile(dialect=postgresql.dialect())).upper()
    assert "WHERE" in ddl
```

(If `users`/`player_id_map` column names differ, adjust to the actual schema — the intent is: SERIAL, CITEXT, and a partial WHERE all render for Postgres.)

- [ ] **Step 2: Run to verify they pass**

Run: `$PY -m pytest tests/test_db_alembic_baseline.py -q`
Expected: all pass (scaffolding + parity + upgrade + 3 PG-compilation).

- [ ] **Step 3: Lint + commit**

```bash
$PY -m ruff check tests/test_db_alembic_baseline.py
git add tests/test_db_alembic_baseline.py
git commit -m "test(db): assert Alembic baseline renders PG-correct DDL (SERIAL/citext/partial index) (B2.1)"
```

---

### Task 5: Full verification + init_db-unchanged

- [ ] **Step 1: Confirm `init_db()` (live SQLite path) is unchanged** — `git diff <baseline>..HEAD -- src/database.py` should be EMPTY (B2.1 must not touch it). If it's non-empty, STOP and report.

- [ ] **Step 2: Full suite (parallel)**

Run: `$PY -m pytest --ignore=tests/test_cheat_sheet.py -n auto --dist loadfile -q`
Expected: Task-1 baseline count + the new B2.1 tests, **0 failures**. If a previously-green test fails, STOP and report.

- [ ] **Step 3: Lint the changeset**

Run: `$PY -m ruff check src/db/ alembic/ tests/test_db_alembic_baseline.py && $PY -m ruff format --check src/db/schema.py alembic/env.py tests/test_db_alembic_baseline.py`
Expected: clean.

- [ ] **Step 4: Clean tree + commits**

Run: `git status --short && git log --oneline <baseline>..HEAD` — clean; ~4 B2.1 commits.

Hand back for 2-stage review. **In your report, explicitly state that the real `alembic upgrade head` against a live Postgres is DEFERRED (no PG instance in this env) — B2.1 is verified via SQLite-upgrade + PG-dialect compilation + schema parity only.**

---

## Self-Review
**Spec coverage (vs B2 plan Phase B2.1):** Alembic stood up + env wired to the seam ✅ (Task 1); baseline builds 53 tables + 62 added columns dialect-aware ✅ (Tasks 2-3, parity-enforced); citext for NOCASE cols + SERIAL + partial indexes + CHECK ✅ (rules + Task 4); `init_db()` SQLite unchanged ✅ (Task 5 Step 1 + no edit); schema-parity guarantees Alembic ≡ init_db ✅ (Task 2). **Placeholders:** none — rules + worked example + enforcing tests provided; the 53-table enumeration is intentionally delegated to the implementer + guarded by the parity test (listing them in prose would be error-prone duplication). **Type consistency:** `metadata` is the single MetaData exported by `src/db/schema.py`, imported identically by `env.py`, the baseline migration, and every test. The real-PG `upgrade head` is explicitly deferred (no PG here) — the SQLite-upgrade + PG-compilation + parity trio is the B2.1 bar.

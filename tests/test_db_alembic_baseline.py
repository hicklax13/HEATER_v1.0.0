"""B2.1 Alembic baseline guards. Alembic builds the full schema (dialect-aware)
on SQLite (and renders PG-correct DDL); it must match init_db()'s schema. The
real `alembic upgrade head` on Postgres is deferred until a PG instance exists."""

import os
import pathlib
import sqlite3

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_alembic_is_available():
    import alembic  # noqa: F401


def test_alembic_scaffolding_exists():
    assert (_ROOT / "alembic.ini").is_file()
    assert (_ROOT / "alembic" / "env.py").is_file()
    assert (_ROOT / "alembic" / "versions").is_dir()


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


def test_alembic_upgrade_head_builds_schema_on_sqlite(tmp_path, monkeypatch):
    from alembic.config import Config

    from alembic import command

    db = tmp_path / "alembic.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db.as_posix()}")
    from src.db.engine import reset_engine_cache

    reset_engine_cache()
    cfg = Config(str(_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(_ROOT / "alembic"))
    command.upgrade(cfg, "head")  # must not raise

    conn = sqlite3.connect(str(db))
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        from src.db.schema import metadata

        for t in metadata.tables:
            assert t in tables, f"alembic upgrade did not create {t}"
    finally:
        conn.close()
        reset_engine_cache()


def _pg_ddl(table) -> str:
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.schema import CreateTable

    return str(CreateTable(table).compile(dialect=postgresql.dialect()))


def test_autoincrement_pk_renders_serial_on_pg():
    from src.db.schema import metadata

    players = metadata.tables["players"]
    ddl = _pg_ddl(players).upper()
    assert "SERIAL" in ddl  # autoincrement PK -> SERIAL on Postgres


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

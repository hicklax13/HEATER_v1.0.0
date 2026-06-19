"""B2.0 engine-seam guards. SQLite stays the default + byte-identical; the
SQLAlchemy engine is the future Postgres swap point (selected by DATABASE_URL)."""

import sqlite3

import pytest


def test_sqlalchemy_is_available():
    import sqlalchemy  # noqa: F401


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

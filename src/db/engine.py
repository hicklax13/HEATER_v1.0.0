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
    local SQLite file. Reuses the SAME `DB_PATH` constant `get_connection()` uses
    (not a fresh re-resolve), so the engine and direct connections point at one
    file by construction — the seam B2.2 wires the read path onto."""
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    from src.database import DB_PATH

    return f"sqlite:///{DB_PATH.as_posix()}"


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
    """True when the active backend is the local SQLite file (the default).
    Case-insensitive on the URL scheme (RFC 3986 schemes are case-insensitive)."""
    return engine_url().lower().startswith("sqlite")

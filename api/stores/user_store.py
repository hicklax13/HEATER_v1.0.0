"""api-owned user persistence — maps a Clerk user id to a local AppUser.

Single-tenant (no team/tenant yet — that's M4). The store is the seam: an
in-memory fake for DB-free tests, and a SQLite default that owns its OWN table
in a SEPARATE file (data/api_state.db, env HEATER_API_DB_PATH) so it never
contends with the live Streamlit single-writer on draft_tool.db. At M4 a Postgres
impl drops in behind the same Protocol. Dormant until a Clerk user authenticates
— no table is created or written otherwise."""

from __future__ import annotations

import os
import sqlite3
import threading
from datetime import UTC, datetime
from typing import Protocol

from pydantic import BaseModel

_DEFAULT_API_DB = os.path.join("data", "api_state.db")


class AppUser(BaseModel):
    id: int
    clerk_user_id: str
    created_at: str


class UserStore(Protocol):
    def get_or_create(self, clerk_user_id: str) -> AppUser: ...


class InMemoryUserStore:
    """Test/fake impl. Thread-safe, autoincrement id, idempotent by clerk id."""

    def __init__(self) -> None:
        self._by_clerk: dict[str, AppUser] = {}
        self._next_id = 1
        self._lock = threading.Lock()

    def get_or_create(self, clerk_user_id: str) -> AppUser:
        with self._lock:
            existing = self._by_clerk.get(clerk_user_id)
            if existing is not None:
                return existing
            user = AppUser(id=self._next_id, clerk_user_id=clerk_user_id, created_at=datetime.now(UTC).isoformat())
            self._by_clerk[clerk_user_id] = user
            self._next_id += 1
            return user


class SqliteUserStore:
    """Default prod impl. Owns api_users in a SEPARATE sqlite file (never the live
    draft_tool.db). Creates the table idempotently on first use. WAL +
    busy_timeout mirror get_connection()'s protections for the api process."""

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or os.environ.get("HEATER_API_DB_PATH", _DEFAULT_API_DB)
        # Serialize get-or-create within the process so two concurrent first-calls
        # for the same clerk id can't both SELECT-miss then INSERT (TOCTOU). The
        # UNIQUE constraint is the cross-process backstop.
        self._lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        parent = os.path.dirname(self._db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        conn = sqlite3.connect(self._db_path, timeout=60.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=60000")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS api_users ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "clerk_user_id TEXT UNIQUE NOT NULL, "
            "created_at TEXT NOT NULL)"
        )
        return conn

    def get_or_create(self, clerk_user_id: str) -> AppUser:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT id, clerk_user_id, created_at FROM api_users WHERE clerk_user_id = ?",
                    (clerk_user_id,),
                ).fetchone()
                if row is not None:
                    return AppUser(id=int(row[0]), clerk_user_id=row[1], created_at=row[2])
                created_at = datetime.now(UTC).isoformat()
                cur = conn.execute(
                    "INSERT INTO api_users (clerk_user_id, created_at) VALUES (?, ?)",
                    (clerk_user_id, created_at),
                )
                conn.commit()
                return AppUser(id=int(cur.lastrowid), clerk_user_id=clerk_user_id, created_at=created_at)
            finally:
                conn.close()

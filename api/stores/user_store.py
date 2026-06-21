"""api-owned user persistence — maps a Clerk user id to a local AppUser.

Single-tenant (no team/tenant yet — that's M4). The store is the seam: an
in-memory fake for DB-free tests, and a SQLite default that owns its OWN table
in a SEPARATE file (data/api_state.db, env HEATER_API_DB_PATH) so it never
contends with the live Streamlit single-writer on draft_tool.db. At M4 a Postgres
impl drops in behind the same Protocol. Dormant until a Clerk user authenticates
— no table is created or written otherwise."""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from datetime import UTC, datetime
from typing import Protocol

from pydantic import BaseModel

_DEFAULT_API_DB = os.path.join("data", "api_state.db")

# src/ai chat tables (ai_conversations/keys/usage_ledger) live in draft_tool.db and
# key on `user_id` in the Streamlit `users(user_id)` namespace (small ints, ~1-20).
# The API's AppUser.id lives in a SEPARATE db (api_state.db) and would COLLIDE with
# real Streamlit chat users if passed raw. This offset maps AppUser.id into a
# disjoint range so a Clerk user's chat never mixes with a Streamlit user's.
# M4 follow-up (Platform track): move the chat tables into the api-owned store/Postgres
# keyed by AppUser.id and retire this offset (also removes the API→draft_tool.db write).
_CHAT_USER_ID_OFFSET = 1_000_000

logger = logging.getLogger(__name__)


class AppUser(BaseModel):
    id: int
    clerk_user_id: str
    created_at: str

    @property
    def chat_user_id(self) -> int:
        """Collision-free user_id for src/ai chat (history/keys/budget). Disjoint
        from the live Streamlit users.user_id range by construction. A plain property
        (NOT a pydantic field) so it never enters the serialized schema/openapi."""
        return _CHAT_USER_ID_OFFSET + self.id


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
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=60000")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS api_users ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "clerk_user_id TEXT UNIQUE NOT NULL, "
                "created_at TEXT NOT NULL)"
            )
        except Exception:
            # A PRAGMA / CREATE TABLE failure (locked DB, full disk, corrupt file)
            # after a successful connect() would otherwise leak this handle — the
            # caller's finally:close only runs once _connect RETURNS. Close our own.
            conn.close()
            raise
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
            except Exception as exc:
                # Surface (never swallow), but leave a house-style breadcrumb — a bare
                # 500 from a provisioning failure is otherwise causeless. clerk_user_id
                # is an opaque Clerk subject (not a secret); never log tokens.
                # NOTE (cross-process race, M4): a 2nd API replica first-calling the same
                # brand-new clerk_user_id loses the UNIQUE bet → IntegrityError → 500 (a
                # retry succeeds). Acceptable at numReplicas=1; when multi-replica lands,
                # catch IntegrityError here and re-SELECT the row the winner wrote.
                logger.warning("SqliteUserStore.get_or_create failed for clerk_user_id=%r: %s", clerk_user_id, exc)
                raise
            finally:
                conn.close()

"""api-owned saved-prompts persistence for Bubba (B2.4). Mirrors user_store.py:
a Protocol + an in-memory fake + a SQLite impl owning its OWN table in the
SEPARATE api_state.db (HEATER_API_DB_PATH) — never the live draft_tool.db.
Dormant until a user saves a prompt (no table created/written otherwise)."""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from datetime import UTC, datetime
from typing import Protocol

from pydantic import BaseModel

_DEFAULT_API_DB = os.path.join("data", "api_state.db")

logger = logging.getLogger(__name__)


class SavedPrompt(BaseModel):
    id: int
    name: str
    text: str
    created_at: str


class PromptStore(Protocol):
    def list(self, owner_id: int) -> list[SavedPrompt]: ...
    def create(self, owner_id: int, name: str, text: str) -> SavedPrompt: ...
    def delete(self, owner_id: int, prompt_id: int) -> bool: ...


class InMemoryPromptStore:
    """Test/fake impl. Thread-safe, autoincrement id, newest-first list."""

    def __init__(self) -> None:
        self._rows: list[tuple[int, SavedPrompt]] = []  # (owner_id, prompt)
        self._next_id = 1
        self._lock = threading.Lock()

    def list(self, owner_id: int) -> list[SavedPrompt]:
        with self._lock:
            return [p for (o, p) in reversed(self._rows) if o == owner_id]

    def create(self, owner_id: int, name: str, text: str) -> SavedPrompt:
        with self._lock:
            p = SavedPrompt(id=self._next_id, name=name, text=text, created_at=datetime.now(UTC).isoformat())
            self._rows.append((owner_id, p))
            self._next_id += 1
            return p

    def delete(self, owner_id: int, prompt_id: int) -> bool:
        with self._lock:
            before = len(self._rows)
            self._rows = [(o, p) for (o, p) in self._rows if not (o == owner_id and p.id == prompt_id)]
            return len(self._rows) < before


class SqlitePromptStore:
    """Default prod impl. Owns api_saved_prompts in a SEPARATE sqlite file (never
    the live draft_tool.db). Creates the table idempotently on first use."""

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or os.environ.get("HEATER_API_DB_PATH", _DEFAULT_API_DB)
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
                "CREATE TABLE IF NOT EXISTS api_saved_prompts ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "owner_id INTEGER NOT NULL, "
                "name TEXT NOT NULL, "
                "text TEXT NOT NULL, "
                "created_at TEXT NOT NULL)"
            )
        except Exception:
            conn.close()  # don't leak the handle if PRAGMA/CREATE fails
            raise
        return conn

    def list(self, owner_id: int) -> list[SavedPrompt]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT id, name, text, created_at FROM api_saved_prompts WHERE owner_id = ? ORDER BY id DESC",
                    (owner_id,),
                ).fetchall()
                return [SavedPrompt(id=int(r[0]), name=r[1], text=r[2], created_at=r[3]) for r in rows]
            finally:
                conn.close()

    def create(self, owner_id: int, name: str, text: str) -> SavedPrompt:
        with self._lock:
            conn = self._connect()
            try:
                created_at = datetime.now(UTC).isoformat()
                cur = conn.execute(
                    "INSERT INTO api_saved_prompts (owner_id, name, text, created_at) VALUES (?, ?, ?, ?)",
                    (owner_id, name, text, created_at),
                )
                conn.commit()
                return SavedPrompt(id=int(cur.lastrowid), name=name, text=text, created_at=created_at)
            finally:
                conn.close()

    def delete(self, owner_id: int, prompt_id: int) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                cur = conn.execute(
                    "DELETE FROM api_saved_prompts WHERE owner_id = ? AND id = ?",
                    (owner_id, prompt_id),
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

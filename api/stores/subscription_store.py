"""api-owned subscription persistence — the Stripe tier source of truth.

Owns api_subscriptions in the SAME separate file as the user store (api_state.db,
env HEATER_API_DB_PATH) — never the live draft_tool.db. In-memory fake + SQLite
default behind the Protocol; Postgres at M4. Dormant until billing is used."""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from typing import Protocol

from pydantic import BaseModel

_DEFAULT_API_DB = os.path.join("data", "api_state.db")

logger = logging.getLogger(__name__)


class Subscription(BaseModel):
    clerk_user_id: str
    stripe_customer_id: str | None = None
    tier: str = "free"  # "free" | "pro"
    status: str = "none"
    current_period_end: int | None = None
    updated_at: str


class SubscriptionStore(Protocol):
    def get(self, clerk_user_id: str) -> Subscription | None: ...
    def get_by_customer(self, stripe_customer_id: str) -> Subscription | None: ...
    def upsert(self, sub: Subscription) -> None: ...


class InMemorySubscriptionStore:
    def __init__(self) -> None:
        self._by_clerk: dict[str, Subscription] = {}
        self._lock = threading.Lock()

    def get(self, clerk_user_id: str) -> Subscription | None:
        with self._lock:
            return self._by_clerk.get(clerk_user_id)

    def get_by_customer(self, stripe_customer_id: str) -> Subscription | None:
        with self._lock:
            for sub in self._by_clerk.values():
                if sub.stripe_customer_id == stripe_customer_id:
                    return sub
            return None

    def upsert(self, sub: Subscription) -> None:
        with self._lock:
            self._by_clerk[sub.clerk_user_id] = sub


class SqliteSubscriptionStore:
    _COLS = "clerk_user_id, stripe_customer_id, tier, status, current_period_end, updated_at"

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
                "CREATE TABLE IF NOT EXISTS api_subscriptions ("
                "clerk_user_id TEXT PRIMARY KEY, "
                "stripe_customer_id TEXT, "
                "tier TEXT NOT NULL, "
                "status TEXT NOT NULL, "
                "current_period_end INTEGER, "
                "updated_at TEXT NOT NULL)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS ix_api_subs_customer ON api_subscriptions(stripe_customer_id)")
        except Exception:
            # Close our own handle if setup fails before we return it (the caller's
            # finally:close only runs once _connect RETURNS) — no leaked connection.
            conn.close()
            raise
        return conn

    @staticmethod
    def _row_to_sub(row) -> Subscription:
        return Subscription(
            clerk_user_id=row[0],
            stripe_customer_id=row[1],
            tier=row[2],
            status=row[3],
            current_period_end=row[4],
            updated_at=row[5],
        )

    def get(self, clerk_user_id: str) -> Subscription | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    f"SELECT {self._COLS} FROM api_subscriptions WHERE clerk_user_id = ?", (clerk_user_id,)
                ).fetchone()
                return self._row_to_sub(row) if row else None
            finally:
                conn.close()

    def get_by_customer(self, stripe_customer_id: str) -> Subscription | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    f"SELECT {self._COLS} FROM api_subscriptions WHERE stripe_customer_id = ?", (stripe_customer_id,)
                ).fetchone()
                return self._row_to_sub(row) if row else None
            finally:
                conn.close()

    def upsert(self, sub: Subscription) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "INSERT INTO api_subscriptions "
                    "(clerk_user_id, stripe_customer_id, tier, status, current_period_end, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(clerk_user_id) DO UPDATE SET "
                    "stripe_customer_id=excluded.stripe_customer_id, tier=excluded.tier, "
                    "status=excluded.status, current_period_end=excluded.current_period_end, "
                    "updated_at=excluded.updated_at",
                    (
                        sub.clerk_user_id,
                        sub.stripe_customer_id,
                        sub.tier,
                        sub.status,
                        sub.current_period_end,
                        sub.updated_at,
                    ),
                )
                conn.commit()
            except Exception as exc:
                logger.warning("SqliteSubscriptionStore.upsert failed for clerk_user_id=%r: %s", sub.clerk_user_id, exc)
                raise
            finally:
                conn.close()

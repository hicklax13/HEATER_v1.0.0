"""api-owned processed-Stripe-event ledger for webhook idempotency + ordering.

Stripe delivers webhooks at-least-once (redelivery) and can reorder them. This
store records each processed `event.id` (+ its `event.created` and the
customer/subscription it applied to) so the billing webhook handler can:
  - skip a redelivered event.id (idempotent), and
  - refuse to let an OLDER event.created overwrite newer subscription state.

Mirrors subscription_store.py / user_store.py: a Protocol + an in-memory fake +
a SQLite impl owning its OWN table in the SEPARATE api_state.db
(HEATER_API_DB_PATH) — never the live draft_tool.db. Dormant until billing
processes a webhook (no table created/written otherwise)."""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from typing import Protocol

_DEFAULT_API_DB = os.path.join("data", "api_state.db")

logger = logging.getLogger(__name__)


class ProcessedEventStore(Protocol):
    def was_processed(self, event_id: str) -> bool: ...
    def last_applied_created(self, customer_id: str) -> int | None: ...
    def record(
        self,
        event_id: str,
        *,
        event_created: int | None,
        customer_id: str | None,
        subscription_id: str | None,
    ) -> None: ...


class InMemoryProcessedEventStore:
    """Test/fake impl. Thread-safe; first-write-wins per event id."""

    def __init__(self) -> None:
        # event_id -> (event_created, customer_id, subscription_id)
        self._rows: dict[str, tuple[int | None, str | None, str | None]] = {}
        self._lock = threading.Lock()

    def was_processed(self, event_id: str) -> bool:
        with self._lock:
            return event_id in self._rows

    def last_applied_created(self, customer_id: str) -> int | None:
        with self._lock:
            createds = [
                created for (created, cust, _sub) in self._rows.values() if cust == customer_id and created is not None
            ]
            return max(createds) if createds else None

    def record(self, event_id, *, event_created, customer_id, subscription_id) -> None:
        with self._lock:
            # First-write-wins: a processed event is immutable.
            self._rows.setdefault(event_id, (event_created, customer_id, subscription_id))


class SqliteProcessedEventStore:
    """Default prod impl. Owns api_processed_events in a SEPARATE sqlite file
    (never the live draft_tool.db). Creates the table idempotently on first use.
    WAL + busy_timeout mirror get_connection()'s protections for the api process."""

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
                "CREATE TABLE IF NOT EXISTS api_processed_events ("
                "event_id TEXT PRIMARY KEY, "
                "event_created INTEGER, "
                "customer_id TEXT, "
                "subscription_id TEXT, "
                "applied_at TEXT NOT NULL DEFAULT (datetime('now')))"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_api_proc_events_customer "
                "ON api_processed_events(customer_id, event_created)"
            )
        except Exception:
            # Close our own handle if setup fails before we return it (the caller's
            # finally:close only runs once _connect RETURNS) — no leaked connection.
            conn.close()
            raise
        return conn

    def was_processed(self, event_id: str) -> bool:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute("SELECT 1 FROM api_processed_events WHERE event_id = ?", (event_id,)).fetchone()
                return row is not None
            finally:
                conn.close()

    def last_applied_created(self, customer_id: str) -> int | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT MAX(event_created) FROM api_processed_events WHERE customer_id = ?",
                    (customer_id,),
                ).fetchone()
                return row[0] if row and row[0] is not None else None
            finally:
                conn.close()

    def record(self, event_id, *, event_created, customer_id, subscription_id) -> None:
        with self._lock:
            conn = self._connect()
            try:
                # First-write-wins (a processed event is immutable) — ON CONFLICT
                # DO NOTHING so a redelivery race can't rewrite the created/customer.
                conn.execute(
                    "INSERT INTO api_processed_events (event_id, event_created, customer_id, subscription_id) "
                    "VALUES (?, ?, ?, ?) ON CONFLICT(event_id) DO NOTHING",
                    (event_id, event_created, customer_id, subscription_id),
                )
                conn.commit()
            except Exception as exc:
                logger.warning("SqliteProcessedEventStore.record failed for event_id=%r: %s", event_id, exc)
                raise
            finally:
                conn.close()

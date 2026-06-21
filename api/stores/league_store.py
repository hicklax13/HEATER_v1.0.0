"""api-owned league registry — a connected-league dataset (provider +
external_league_id). Single-tenant beta: exactly one default league. Mirrors the
M2 user_store seam: in-memory fake for tests, sqlite default in a SEPARATE file
(data/api_state.db, env HEATER_API_DB_PATH) so it never contends with the live
draft_tool.db. Dormant until first use. A Postgres impl drops in at M4."""

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


def _beta_provider() -> str:
    return os.environ.get("HEATER_BETA_LEAGUE_PROVIDER", "yahoo").strip() or "yahoo"


def _beta_external_id() -> str:
    return os.environ.get("HEATER_BETA_LEAGUE_EXTERNAL_ID", "469").strip() or "469"


def _beta_name() -> str:
    return os.environ.get("HEATER_BETA_LEAGUE_NAME", "FourzynBurn").strip() or "FourzynBurn"


class League(BaseModel):
    id: int
    provider: str
    external_league_id: str
    name: str
    owner_user_id: int | None = None
    created_at: str


class LeagueStore(Protocol):
    def get_or_create_default(self) -> League: ...
    def get(self, league_id: int) -> League | None: ...


class InMemoryLeagueStore:
    """Test/fake impl. Holds a single default league; idempotent."""

    def __init__(
        self,
        provider: str | None = None,
        external_league_id: str | None = None,
        name: str | None = None,
    ) -> None:
        self._provider = provider or _beta_provider()
        self._external = external_league_id or _beta_external_id()
        self._name = name or _beta_name()
        self._by_id: dict[int, League] = {}
        self._default_id: int | None = None
        self._next_id = 1
        self._lock = threading.Lock()

    def get_or_create_default(self) -> League:
        with self._lock:
            if self._default_id is not None:
                return self._by_id[self._default_id]
            lg = League(
                id=self._next_id,
                provider=self._provider,
                external_league_id=self._external,
                name=self._name,
                owner_user_id=None,
                created_at=datetime.now(UTC).isoformat(),
            )
            self._by_id[lg.id] = lg
            self._default_id = lg.id
            self._next_id += 1
            return lg

    def get(self, league_id: int) -> League | None:
        return self._by_id.get(league_id)


class SqliteLeagueStore:
    """Default prod impl. Owns api_leagues in a SEPARATE sqlite file (never the live
    draft_tool.db). WAL + busy_timeout mirror get_connection()'s protections."""

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
                "CREATE TABLE IF NOT EXISTS api_leagues ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "provider TEXT NOT NULL, "
                "external_league_id TEXT NOT NULL, "
                "name TEXT NOT NULL, "
                "owner_user_id INTEGER, "
                "created_at TEXT NOT NULL, "
                "UNIQUE(provider, external_league_id))"
            )
        except Exception:
            conn.close()
            raise
        return conn

    def get_or_create_default(self) -> League:
        provider, external, name = _beta_provider(), _beta_external_id(), _beta_name()
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT id, provider, external_league_id, name, owner_user_id, created_at "
                    "FROM api_leagues WHERE provider = ? AND external_league_id = ?",
                    (provider, external),
                ).fetchone()
                if row is not None:
                    return self._row_to_league(row)
                created_at = datetime.now(UTC).isoformat()
                cur = conn.execute(
                    "INSERT INTO api_leagues (provider, external_league_id, name, owner_user_id, created_at) "
                    "VALUES (?, ?, ?, NULL, ?)",
                    (provider, external, name, created_at),
                )
                conn.commit()
                return League(
                    id=int(cur.lastrowid),
                    provider=provider,
                    external_league_id=external,
                    name=name,
                    owner_user_id=None,
                    created_at=created_at,
                )
            except Exception as exc:
                logger.warning("SqliteLeagueStore.get_or_create_default failed: %s", exc)
                raise
            finally:
                conn.close()

    def get(self, league_id: int) -> League | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT id, provider, external_league_id, name, owner_user_id, created_at "
                    "FROM api_leagues WHERE id = ?",
                    (league_id,),
                ).fetchone()
                return self._row_to_league(row) if row is not None else None
            finally:
                conn.close()

    @staticmethod
    def _row_to_league(row) -> League:
        return League(
            id=int(row[0]),
            provider=row[1],
            external_league_id=row[2],
            name=row[3],
            owner_user_id=(int(row[4]) if row[4] is not None else None),
            created_at=row[5],
        )

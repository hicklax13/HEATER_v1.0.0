"""api-owned membership — maps a user to their team within a league (the
replacement for the trusted team_name query param). UNIQUE(user_id, league_id):
assign upserts. Mirrors the M2 user_store seam (in-memory fake + sqlite in a
SEPARATE file, never the live draft_tool.db). Postgres impl drops in at M4."""

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


class UserTeam(BaseModel):
    id: int
    user_id: int
    league_id: int
    team_name: str
    team_key: str | None = None
    assigned_by: int | None = None
    created_at: str


class MembershipStore(Protocol):
    def assign(
        self, user_id: int, league_id: int, team_name: str, team_key: str | None, assigned_by: int | None
    ) -> UserTeam: ...
    def get_for_user(self, user_id: int, league_id: int) -> UserTeam | None: ...
    def list_for_league(self, league_id: int) -> list[UserTeam]: ...


class InMemoryMembershipStore:
    def __init__(self) -> None:
        self._rows: dict[tuple[int, int], UserTeam] = {}
        self._next_id = 1
        self._lock = threading.Lock()

    def assign(self, user_id, league_id, team_name, team_key, assigned_by) -> UserTeam:
        with self._lock:
            key = (user_id, league_id)
            existing = self._rows.get(key)
            row_id = existing.id if existing else self._next_id
            if existing is None:
                self._next_id += 1
            m = UserTeam(
                id=row_id,
                user_id=user_id,
                league_id=league_id,
                team_name=team_name,
                team_key=team_key,
                assigned_by=assigned_by,
                created_at=(existing.created_at if existing else datetime.now(UTC).isoformat()),
            )
            self._rows[key] = m
            return m

    def get_for_user(self, user_id, league_id) -> UserTeam | None:
        return self._rows.get((user_id, league_id))

    def list_for_league(self, league_id) -> list[UserTeam]:
        return [m for (_, lid), m in self._rows.items() if lid == league_id]


class SqliteMembershipStore:
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
                "CREATE TABLE IF NOT EXISTS api_user_teams ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "user_id INTEGER NOT NULL, "
                "league_id INTEGER NOT NULL, "
                "team_name TEXT NOT NULL, "
                "team_key TEXT, "
                "assigned_by INTEGER, "
                "created_at TEXT NOT NULL, "
                "UNIQUE(user_id, league_id))"
            )
        except Exception:
            conn.close()
            raise
        return conn

    def assign(self, user_id, league_id, team_name, team_key, assigned_by) -> UserTeam:
        with self._lock:
            conn = self._connect()
            try:
                now = datetime.now(UTC).isoformat()
                # Upsert on (user_id, league_id); keep the original created_at.
                conn.execute(
                    "INSERT INTO api_user_teams (user_id, league_id, team_name, team_key, assigned_by, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(user_id, league_id) DO UPDATE SET "
                    "team_name=excluded.team_name, team_key=excluded.team_key, assigned_by=excluded.assigned_by",
                    (user_id, league_id, team_name, team_key, assigned_by, now),
                )
                conn.commit()
                row = conn.execute(
                    "SELECT id, user_id, league_id, team_name, team_key, assigned_by, created_at "
                    "FROM api_user_teams WHERE user_id = ? AND league_id = ?",
                    (user_id, league_id),
                ).fetchone()
                return self._row(row)
            except Exception as exc:
                logger.warning("SqliteMembershipStore.assign failed for user=%s league=%s: %s", user_id, league_id, exc)
                raise
            finally:
                conn.close()

    def get_for_user(self, user_id, league_id) -> UserTeam | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT id, user_id, league_id, team_name, team_key, assigned_by, created_at "
                    "FROM api_user_teams WHERE user_id = ? AND league_id = ?",
                    (user_id, league_id),
                ).fetchone()
                return self._row(row) if row is not None else None
            finally:
                conn.close()

    def list_for_league(self, league_id) -> list[UserTeam]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT id, user_id, league_id, team_name, team_key, assigned_by, created_at "
                    "FROM api_user_teams WHERE league_id = ? ORDER BY id",
                    (league_id,),
                ).fetchall()
                return [self._row(r) for r in rows]
            finally:
                conn.close()

    @staticmethod
    def _row(row) -> UserTeam:
        return UserTeam(
            id=int(row[0]),
            user_id=int(row[1]),
            league_id=int(row[2]),
            team_name=row[3],
            team_key=row[4],
            assigned_by=(int(row[5]) if row[5] is not None else None),
            created_at=row[6],
        )

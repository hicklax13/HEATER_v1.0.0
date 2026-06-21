# B4 / M3-1 — Per-user team resolution (single-league) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let each authenticated (Clerk) user see THEIR OWN team — resolving the viewer's team from their login instead of trusting a client-supplied `team_name` query param — built additively + dormant on the CURRENT SQLite infra (no Postgres, no `src/` change, no live-data-path touch).

**Architecture:** New api-owned identity/membership stores in the SEPARATE `api_state.db` (never the live `draft_tool.db`), mirroring the M2 `user_store` seam. A `ViewerContext` resolver composes the existing (dormant) Clerk auth seam **optionally** (reads stay open when Clerk is off → fall back to today's query param). An admin-only endpoint maps each user → team (the beta mechanism). The 7 personalized routers gain one additive dependency; the analytics engine is untouched.

**Tech Stack:** FastAPI + Pydantic, sqlite3 (api-owned file), pytest (DB-free in-memory fakes + tmp_path sqlite). Spec: `docs/superpowers/specs/2026-06-21-heater-b4-multi-tenancy-tenant-resolution-design.md` (slice M3-1).

**Scope guardrails (read before every task):**
- Lane = `api/` + `tests/api/`. NEVER `web/` (CMO). NEVER the CEO's active files (`src/ai/*`, the chat router/contract/service, `src/stream_analyzer.py`, `src/game_day.py`, any schedule endpoint/contract/service).
- `git pull --no-rebase origin master` immediately BEFORE and AFTER any task that edits a SHARED hot file (`api/deps.py`, `api/main.py`, the 7 personalized routers). Keep those commits tiny.
- No `src/` edits. Reuse existing api services for any `src/`-backed read.
- Every store/dependency is DORMANT until Clerk is configured AND memberships exist; Clerk-off behavior is byte-for-byte today's.
- Finish each task by committing; push at the natural checkpoints (after Task 4, after Task 6, after Task 7) with `git pull --no-rebase` first.

---

## File structure

| File | Responsibility | Action |
|---|---|---|
| `api/stores/league_store.py` | `League` model + `LeagueStore` Protocol + `InMemoryLeagueStore` + `SqliteLeagueStore` (`api_leagues` table; default-beta-league get-or-create) | Create |
| `api/stores/membership_store.py` | `UserTeam` model + `MembershipStore` Protocol + `InMemoryMembershipStore` + `SqliteMembershipStore` (`api_user_teams` table; assign/upsert + lookups) | Create |
| `api/tenancy.py` | `normalize_team_name` + `reconcile_team_name` (pure) + `ViewerContext` (+ `effective_team`) + `require_viewer_context` dependency | Create |
| `api/identity.py` | add `optional_app_user` (non-raising identity for OPEN reads) | Modify (additive) |
| `api/admin.py` | `require_admin` (env allowlist, deny-by-default) + `assign_team` logic helper | Create |
| `api/contracts/admin.py` | `AssignmentRequest` / `Assignment` / `AssignmentsResponse` | Create |
| `api/routers/admin.py` | thin `POST` + `GET /api/admin/assignments` | Create |
| `api/deps.py` | `get_league_store` + `get_membership_store` providers | Modify (additive) |
| `api/main.py` | mount the admin router | Modify (additive) |
| `api/routers/{team,matchup,lineup,punt,free_agents,playoff,trade_finder}.py` | add `ctx=Depends(require_viewer_context)`; pass `ctx.effective_team(team_name)` | Modify (additive) |
| `api/openapi.json` | regenerate snapshot | Modify (generated) |
| `tests/api/test_api_league_store.py` | LeagueStore tests | Create |
| `tests/api/test_api_membership_store.py` | MembershipStore tests | Create |
| `tests/api/test_api_tenancy_resolver.py` | helpers + resolver + dormancy tests | Create |
| `tests/api/test_api_admin_assignments.py` | admin gate + assignment endpoint tests | Create |
| `tests/api/test_api_personalized_team_resolution.py` | router dormant-fallback + resolved-override tests | Create |

---

## Task 1: LeagueStore

**Files:**
- Create: `api/stores/league_store.py`
- Test: `tests/api/test_api_league_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_api_league_store.py
"""LeagueStore tests — DB-free in-memory fake + tmp-file sqlite (NEVER the live
draft_tool.db). Proves get_or_create_default is idempotent and api owns its own file."""

from api.stores.league_store import InMemoryLeagueStore, League, SqliteLeagueStore


def test_inmemory_default_league_is_idempotent():
    store = InMemoryLeagueStore()
    a = store.get_or_create_default()
    b = store.get_or_create_default()
    assert isinstance(a, League)
    assert a.id == b.id
    assert a.provider == "yahoo"


def test_inmemory_default_league_uses_overrides():
    store = InMemoryLeagueStore(provider="sleeper", external_league_id="abc", name="My League")
    lg = store.get_or_create_default()
    assert lg.provider == "sleeper"
    assert lg.external_league_id == "abc"
    assert lg.name == "My League"


def test_sqlite_default_league_idempotent_in_separate_file(tmp_path):
    db = tmp_path / "api_state.db"
    a = SqliteLeagueStore(db_path=str(db)).get_or_create_default()
    b = SqliteLeagueStore(db_path=str(db)).get_or_create_default()
    assert a.id == b.id
    assert db.exists()  # api owns its OWN file


def test_sqlite_get_by_id(tmp_path):
    store = SqliteLeagueStore(db_path=str(tmp_path / "api_state.db"))
    lg = store.get_or_create_default()
    assert store.get(lg.id).id == lg.id
    assert store.get(999) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/api/test_api_league_store.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'api.stores.league_store'`

- [ ] **Step 3: Write minimal implementation**

```python
# api/stores/league_store.py
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

    def __init__(self, provider: str | None = None, external_league_id: str | None = None, name: str | None = None) -> None:
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
                    id=int(cur.lastrowid), provider=provider, external_league_id=external,
                    name=name, owner_user_id=None, created_at=created_at,
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
            id=int(row[0]), provider=row[1], external_league_id=row[2], name=row[3],
            owner_user_id=(int(row[4]) if row[4] is not None else None), created_at=row[5],
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/api/test_api_league_store.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add api/stores/league_store.py tests/api/test_api_league_store.py
git commit -m "feat(api): LeagueStore (api-owned league registry, default-beta league) — B4/M3-1

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: MembershipStore

**Files:**
- Create: `api/stores/membership_store.py`
- Test: `tests/api/test_api_membership_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_api_membership_store.py
"""MembershipStore tests — DB-free in-memory fake + tmp-file sqlite. Proves
assign upserts (one row per user+league), lookups, and api-owned separate file."""

from api.stores.membership_store import InMemoryMembershipStore, SqliteMembershipStore, UserTeam


def test_inmemory_assign_and_get():
    store = InMemoryMembershipStore()
    m = store.assign(user_id=1, league_id=1, team_name="Team Hickey", team_key="t1", assigned_by=9)
    assert isinstance(m, UserTeam)
    got = store.get_for_user(user_id=1, league_id=1)
    assert got is not None and got.team_name == "Team Hickey" and got.assigned_by == 9


def test_inmemory_assign_is_upsert_per_user_league():
    store = InMemoryMembershipStore()
    store.assign(user_id=1, league_id=1, team_name="Old", team_key=None, assigned_by=9)
    store.assign(user_id=1, league_id=1, team_name="New", team_key=None, assigned_by=9)
    assert store.get_for_user(1, 1).team_name == "New"
    assert len(store.list_for_league(1)) == 1  # upsert, not duplicate


def test_inmemory_get_missing_returns_none():
    assert InMemoryMembershipStore().get_for_user(1, 1) is None


def test_sqlite_assign_upsert_separate_file(tmp_path):
    db = tmp_path / "api_state.db"
    store = SqliteMembershipStore(db_path=str(db))
    store.assign(user_id=1, league_id=1, team_name="A", team_key=None, assigned_by=9)
    store.assign(user_id=1, league_id=1, team_name="B", team_key=None, assigned_by=9)
    assert SqliteMembershipStore(db_path=str(db)).get_for_user(1, 1).team_name == "B"
    assert len(SqliteMembershipStore(db_path=str(db)).list_for_league(1)) == 1
    assert db.exists()


def test_sqlite_list_for_league(tmp_path):
    store = SqliteMembershipStore(db_path=str(tmp_path / "api_state.db"))
    store.assign(user_id=1, league_id=1, team_name="A", team_key=None, assigned_by=9)
    store.assign(user_id=2, league_id=1, team_name="B", team_key=None, assigned_by=9)
    assert {m.team_name for m in store.list_for_league(1)} == {"A", "B"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/api/test_api_membership_store.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'api.stores.membership_store'`

- [ ] **Step 3: Write minimal implementation**

```python
# api/stores/membership_store.py
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
    def assign(self, user_id: int, league_id: int, team_name: str, team_key: str | None, assigned_by: int | None) -> UserTeam: ...
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
                id=row_id, user_id=user_id, league_id=league_id, team_name=team_name,
                team_key=team_key, assigned_by=assigned_by,
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
            id=int(row[0]), user_id=int(row[1]), league_id=int(row[2]), team_name=row[3],
            team_key=row[4], assigned_by=(int(row[5]) if row[5] is not None else None), created_at=row[6],
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/api/test_api_membership_store.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add api/stores/membership_store.py tests/api/test_api_membership_store.py
git commit -m "feat(api): MembershipStore (user->team upsert, api-owned) — B4/M3-1

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Tenancy helpers (pure name normalization + reconciliation)

**Files:**
- Create: `api/tenancy.py` (helpers only this task; the resolver is added in Task 4)
- Test: `tests/api/test_api_tenancy_resolver.py` (helper tests this task)

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_api_tenancy_resolver.py
"""Tenancy helpers + resolver tests. This task covers the pure helpers; Task 4
appends resolver tests to this same file."""

from api.tenancy import normalize_team_name, reconcile_team_name


def test_normalize_strips_emoji_whitespace_punctuation():
    # Same semantics as src.auth._normalize_team_name (replicated, not imported).
    assert normalize_team_name("🏆 Team Hickey") == normalize_team_name("Team Hickey")
    assert normalize_team_name("Team Hickey") == "teamhickey"
    assert normalize_team_name("  A.B-C  ") == "abc"


def test_reconcile_exact_match_returns_canonical_roster_name():
    names = ["🏆 Team Hickey", "Bronx Bombers"]
    assert reconcile_team_name("Team Hickey", names) == "🏆 Team Hickey"


def test_reconcile_exact_string_short_circuits():
    names = ["Team Hickey", "Other"]
    assert reconcile_team_name("Team Hickey", names) == "Team Hickey"


def test_reconcile_no_match_with_known_names_returns_none():
    assert reconcile_team_name("Nonexistent", ["A", "B"]) is None


def test_reconcile_empty_names_returns_assigned_as_is():
    # Cold/empty roster source must NOT block assignment (graceful).
    assert reconcile_team_name("Team Hickey", []) == "Team Hickey"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/api/test_api_tenancy_resolver.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'api.tenancy'`

- [ ] **Step 3: Write minimal implementation**

```python
# api/tenancy.py
"""Viewer tenancy resolution — maps a verified identity to their team within a
league (the replacement for the trusted team_name query param).

normalize_team_name REPLICATES the behavior of src.auth._normalize_team_name (a
non-service api module importing src/ would break the 'services are the one place
importing src/' discipline; re-homing it would be a src/ edit). The resolver
(Task 4) composes the OPTIONAL identity path so currently-open reads stay open
when Clerk is off."""

from __future__ import annotations

import re
from collections.abc import Iterable

from pydantic import BaseModel


def normalize_team_name(name: object) -> str:
    """Lowercase + strip all non-alphanumerics (emoji/whitespace/punctuation) so a
    name missing the Yahoo team's leading emoji ('Team Hickey') still matches the
    roster name ('🏆 Team Hickey'). Mirrors src.auth._normalize_team_name."""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def reconcile_team_name(assigned: str, roster_names: Iterable[str]) -> str | None:
    """Map an admin-typed team name to the EXACT roster name.

    - exact string match → that name (short-circuit);
    - tolerant (normalized) match → the exact roster name;
    - no match but roster_names is non-empty → None (caller signals 422);
    - roster_names empty (cold source) → the assigned name as-is (never block)."""
    names = [str(n) for n in roster_names if str(n).strip()]
    if not names:
        return assigned
    if assigned in names:
        return assigned
    target = normalize_team_name(assigned)
    for n in names:
        if normalize_team_name(n) == target:
            return n
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/api/test_api_tenancy_resolver.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add api/tenancy.py tests/api/test_api_tenancy_resolver.py
git commit -m "feat(api): tenancy name normalize+reconcile helpers (replicate Streamlit) — B4/M3-1

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: ViewerContext resolver + optional identity + DI providers

**Files:**
- Modify: `api/identity.py` (add `optional_app_user`)
- Modify: `api/tenancy.py` (add `ViewerContext` + `require_viewer_context`)
- Modify: `api/deps.py` (add `get_league_store` + `get_membership_store`)
- Test: `tests/api/test_api_tenancy_resolver.py` (append)

- [ ] **Step 1: Write the failing test (append to the existing file)**

```python
# tests/api/test_api_tenancy_resolver.py  (append)
from fastapi import Depends, FastAPI
from starlette.testclient import TestClient

from api.auth import Principal, get_auth_verifier
from api.deps import get_league_store, get_membership_store, get_user_store
from api.stores.league_store import InMemoryLeagueStore
from api.stores.membership_store import InMemoryMembershipStore
from api.stores.user_store import InMemoryUserStore
from api.tenancy import ViewerContext, require_viewer_context


def test_viewer_context_effective_team_prefers_resolved():
    assert ViewerContext(user_id=1, league_id=1, team_name="Mine").effective_team("fallback") == "Mine"


def test_viewer_context_effective_team_falls_back_when_unresolved():
    assert ViewerContext(user_id=None, league_id=None, team_name=None).effective_team("Team Hickey") == "Team Hickey"
    assert ViewerContext(user_id=1, league_id=1, team_name=None).effective_team("Team Hickey") == "Team Hickey"


class _ClerkVerifier:
    def verify(self, authorization):
        return Principal(subject="user_42", clerk_user_id="user_42")


def _app():
    app = FastAPI()

    @app.get("/probe")
    def probe(team_name: str = "", ctx: ViewerContext = Depends(require_viewer_context)):
        return {"team": ctx.effective_team(team_name), "resolved": ctx.team_name}

    return app


def test_resolver_dormant_when_no_token_uses_query_param():
    # Clerk off / no Authorization header → reads stay OPEN, fall back to the param.
    app = _app()
    app.dependency_overrides[get_user_store] = lambda: InMemoryUserStore()
    app.dependency_overrides[get_league_store] = lambda: InMemoryLeagueStore()
    app.dependency_overrides[get_membership_store] = lambda: InMemoryMembershipStore()
    body = TestClient(app).get("/probe?team_name=Team%20Hickey").json()
    assert body == {"team": "Team Hickey", "resolved": None}


def test_resolver_overrides_with_assigned_team_for_clerk_user():
    app = _app()
    users, leagues, members = InMemoryUserStore(), InMemoryLeagueStore(), InMemoryMembershipStore()
    u = users.get_or_create("user_42")
    lg = leagues.get_or_create_default()
    members.assign(user_id=u.id, league_id=lg.id, team_name="Bronx Bombers", team_key=None, assigned_by=u.id)
    app.dependency_overrides[get_auth_verifier] = lambda: _ClerkVerifier()
    app.dependency_overrides[get_user_store] = lambda: users
    app.dependency_overrides[get_league_store] = lambda: leagues
    app.dependency_overrides[get_membership_store] = lambda: members
    body = TestClient(app).get("/probe?team_name=Team%20Hickey", headers={"Authorization": "Bearer x"}).json()
    assert body["resolved"] == "Bronx Bombers"
    assert body["team"] == "Bronx Bombers"  # resolved wins over the query param


def test_resolver_clerk_user_without_assignment_resolves_none():
    # Logged-in but unassigned → team_name None (never another user's team); the
    # endpoint falls back to the param (open read) in M3-1.
    app = _app()
    users = InMemoryUserStore()
    app.dependency_overrides[get_auth_verifier] = lambda: _ClerkVerifier()
    app.dependency_overrides[get_user_store] = lambda: users
    app.dependency_overrides[get_league_store] = lambda: InMemoryLeagueStore()
    app.dependency_overrides[get_membership_store] = lambda: InMemoryMembershipStore()
    body = TestClient(app).get("/probe?team_name=Fallback", headers={"Authorization": "Bearer x"}).json()
    assert body["resolved"] is None
    assert body["team"] == "Fallback"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/api/test_api_tenancy_resolver.py -q`
Expected: FAIL — `ImportError: cannot import name 'get_league_store'` (and `ViewerContext`/`require_viewer_context`)

- [ ] **Step 3a: Add `optional_app_user` to `api/identity.py`**

Append to `api/identity.py`:

```python
from fastapi import Header, HTTPException

from api.auth import AuthVerifier, get_auth_verifier


def optional_app_user(
    authorization: str | None = Header(default=None),
    verifier: AuthVerifier = Depends(get_auth_verifier),
    store: UserStore = Depends(get_user_store),
) -> AppUser | None:
    """Non-raising identity for OPEN (currently-unauthenticated) read endpoints.

    No Authorization header → None (reads stay open; the endpoint falls back to its
    team_name query param = today's behavior). A present-but-invalid token also →
    None (a bad token must not 401 a read that is open today). A valid Clerk token →
    the provisioned AppUser. require_app_user (mandatory, fail-closed) still exists
    for write/admin endpoints; this is the read-side complement."""
    if not authorization:
        return None
    try:
        principal = verifier.verify(authorization)
    except HTTPException:
        return None
    return provision_app_user(principal, store)
```

> The existing `from fastapi import Depends` import stays; add `Header, HTTPException` to it (or a second import line as shown). `AppUser`/`UserStore`/`provision_app_user` are already in this module.

- [ ] **Step 3b: Add `ViewerContext` + `require_viewer_context` to `api/tenancy.py`**

Append to `api/tenancy.py`:

```python
from fastapi import Depends

from api.deps import get_league_store, get_membership_store
from api.identity import optional_app_user
from api.stores.league_store import LeagueStore
from api.stores.membership_store import MembershipStore
from api.stores.user_store import AppUser


class ViewerContext(BaseModel):
    """The resolved viewer. team_name is None when there is no authenticated user
    or no assignment (→ effective_team falls back to the endpoint's query param)."""

    user_id: int | None = None
    league_id: int | None = None
    team_name: str | None = None

    def effective_team(self, fallback: str | None) -> str | None:
        """The resolved team if present, else the endpoint's current query-param
        fallback (preserves today's behavior when dormant)."""
        return self.team_name or fallback


def require_viewer_context(
    app_user: AppUser | None = Depends(optional_app_user),
    league_store: LeagueStore = Depends(get_league_store),
    membership_store: MembershipStore = Depends(get_membership_store),
) -> ViewerContext:
    if app_user is None:
        return ViewerContext()
    league = league_store.get_or_create_default()
    membership = membership_store.get_for_user(app_user.id, league.id)
    return ViewerContext(
        user_id=app_user.id,
        league_id=league.id,
        team_name=(membership.team_name if membership else None),
    )
```

- [ ] **Step 3c: Add the store providers to `api/deps.py`**

Append to `api/deps.py` (after `get_user_store`):

```python
from api.stores.league_store import LeagueStore, SqliteLeagueStore
from api.stores.membership_store import MembershipStore, SqliteMembershipStore


def get_league_store() -> LeagueStore:
    return SqliteLeagueStore()


def get_membership_store() -> MembershipStore:
    return SqliteMembershipStore()
```

> Place the two `from api.stores...` imports with the other top-of-file imports (alphabetical block) per ruff; the provider functions go in the body. Run `python -m ruff check api/deps.py --fix && python -m ruff format api/deps.py` after.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/api/test_api_tenancy_resolver.py -q`
Expected: PASS (10 passed total)

- [ ] **Step 5: Commit + checkpoint push**

```bash
python -m ruff check api/ --fix && python -m ruff format api/deps.py api/identity.py api/tenancy.py
git add api/identity.py api/tenancy.py api/deps.py tests/api/test_api_tenancy_resolver.py
git commit -m "feat(api): ViewerContext resolver + optional_app_user (open-read identity) — B4/M3-1

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git pull --no-rebase origin master && git push origin master
```

---

## Task 5: Admin gate + assignment endpoint

**Files:**
- Create: `api/admin.py`, `api/contracts/admin.py`, `api/routers/admin.py`
- Modify: `api/main.py` (mount the router)
- Test: `tests/api/test_api_admin_assignments.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_api_admin_assignments.py
"""Admin assignment endpoint — gate (env allowlist, deny-by-default) + assign/list,
all DB-free via in-memory stores + a faked team-names provider."""

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from api.auth import Principal, get_auth_verifier
from api.deps import get_league_store, get_membership_store, get_user_store
from api.routers.admin import router as admin_router
from api.admin import get_team_names_provider
from api.stores.league_store import InMemoryLeagueStore
from api.stores.membership_store import InMemoryMembershipStore
from api.stores.user_store import InMemoryUserStore


class _Verifier:
    def __init__(self, clerk_id):
        self._id = clerk_id

    def verify(self, authorization):
        return Principal(subject=self._id, clerk_user_id=self._id)


def _client(clerk_id, monkeypatch, names=("🏆 Team Hickey", "Bronx Bombers")):
    monkeypatch.setenv("HEATER_ADMIN_CLERK_IDS", "admin_1,admin_2")
    app = FastAPI()
    app.include_router(admin_router)
    app.dependency_overrides[get_auth_verifier] = lambda: _Verifier(clerk_id)
    app.dependency_overrides[get_user_store] = lambda: InMemoryUserStore()
    app.dependency_overrides[get_league_store] = lambda: InMemoryLeagueStore()
    app.dependency_overrides[get_membership_store] = lambda: InMemoryMembershipStore()
    app.dependency_overrides[get_team_names_provider] = lambda: (lambda: list(names))
    return TestClient(app)


def test_non_admin_clerk_user_forbidden(monkeypatch):
    c = _client("not_admin", monkeypatch)
    r = c.post("/api/admin/assignments", json={"clerk_user_id": "u1", "team_name": "Team Hickey"},
               headers={"Authorization": "Bearer x"})
    assert r.status_code == 403


def test_env_token_path_forbidden(monkeypatch):
    # A caller with no clerk_user_id (env-token/server path) is never an admin.
    monkeypatch.setenv("HEATER_ADMIN_CLERK_IDS", "admin_1")
    app = FastAPI()
    app.include_router(admin_router)
    app.dependency_overrides[get_auth_verifier] = lambda: type("V", (), {"verify": lambda self, a: Principal(subject="api-token")})()
    app.dependency_overrides[get_user_store] = lambda: InMemoryUserStore()
    app.dependency_overrides[get_league_store] = lambda: InMemoryLeagueStore()
    app.dependency_overrides[get_membership_store] = lambda: InMemoryMembershipStore()
    app.dependency_overrides[get_team_names_provider] = lambda: (lambda: [])
    r = TestClient(app).post("/api/admin/assignments", json={"clerk_user_id": "u1", "team_name": "X"},
                             headers={"Authorization": "Bearer x"})
    assert r.status_code == 403


def test_admin_assigns_and_canonicalizes_team_name(monkeypatch):
    c = _client("admin_1", monkeypatch)
    r = c.post("/api/admin/assignments", json={"clerk_user_id": "member_a", "team_name": "Team Hickey"},
               headers={"Authorization": "Bearer x"})
    assert r.status_code == 200
    assert r.json()["team_name"] == "🏆 Team Hickey"  # reconciled to the exact roster name


def test_admin_assign_unknown_team_is_422(monkeypatch):
    c = _client("admin_1", monkeypatch)
    r = c.post("/api/admin/assignments", json={"clerk_user_id": "member_a", "team_name": "Ghost Team"},
               headers={"Authorization": "Bearer x"})
    assert r.status_code == 422
    assert "Bronx Bombers" in r.json()["detail"]  # candidates surfaced


def test_admin_list_assignments(monkeypatch):
    c = _client("admin_1", monkeypatch)
    c.post("/api/admin/assignments", json={"clerk_user_id": "member_a", "team_name": "Bronx Bombers"},
           headers={"Authorization": "Bearer x"})
    r = c.get("/api/admin/assignments", headers={"Authorization": "Bearer x"})
    assert r.status_code == 200
    body = r.json()
    assert any(a["team_name"] == "Bronx Bombers" for a in body["assignments"])
    assert "Bronx Bombers" in body["available_teams"]


def test_empty_allowlist_denies_everyone(monkeypatch):
    monkeypatch.delenv("HEATER_ADMIN_CLERK_IDS", raising=False)
    app = FastAPI()
    app.include_router(admin_router)
    app.dependency_overrides[get_auth_verifier] = lambda: _Verifier("anyone")
    app.dependency_overrides[get_user_store] = lambda: InMemoryUserStore()
    app.dependency_overrides[get_league_store] = lambda: InMemoryLeagueStore()
    app.dependency_overrides[get_membership_store] = lambda: InMemoryMembershipStore()
    app.dependency_overrides[get_team_names_provider] = lambda: (lambda: [])
    r = TestClient(app).get("/api/admin/assignments", headers={"Authorization": "Bearer x"})
    assert r.status_code == 403
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/api/test_api_admin_assignments.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'api.admin'`

- [ ] **Step 3a: Contracts — `api/contracts/admin.py`**

```python
# api/contracts/admin.py
"""Admin assignment contracts (the beta user->team mapping surface)."""

from __future__ import annotations

from pydantic import BaseModel


class AssignmentRequest(BaseModel):
    clerk_user_id: str
    team_name: str
    league_id: int | None = None  # default = the single beta league


class Assignment(BaseModel):
    clerk_user_id: str
    user_id: int
    league_id: int
    team_name: str


class AssignmentsResponse(BaseModel):
    assignments: list[Assignment]
    available_teams: list[str]
```

- [ ] **Step 3b: Gate + logic — `api/admin.py`**

```python
# api/admin.py
"""Admin authorization + assignment logic for the beta user->team mapping.

require_admin: deny-by-default env allowlist (HEATER_ADMIN_CLERK_IDS, comma-sep of
Clerk user ids) over a MANDATORY app user — the env-token/no-clerk path is never an
admin. Team-name reconciliation reuses the EXISTING RosterQueryService (no new src
import here) and is graceful when the roster source is cold."""

from __future__ import annotations

import os
from collections.abc import Callable

from fastapi import Depends, HTTPException, status

from api.identity import require_app_user
from api.stores.user_store import AppUser

TeamNamesProvider = Callable[[], list[str]]


def _admin_ids() -> set[str]:
    raw = os.environ.get("HEATER_ADMIN_CLERK_IDS", "")
    return {p.strip() for p in raw.split(",") if p.strip()}


def require_admin(app_user: AppUser | None = Depends(require_app_user)) -> AppUser:
    """Allow only configured Clerk admins. Deny-by-default: empty allowlist → no
    admins; env-token path (app_user None) → never admin."""
    ids = _admin_ids()
    if app_user is None or app_user.clerk_user_id not in ids:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required.")
    return app_user


def _live_team_names() -> list[str]:
    """The league's canonical team names from the existing roster service (empty on
    any failure — keeps assignment graceful when the DB is cold)."""
    from api.services.roster_query_service import RosterQueryService

    return [t.team_name for t in RosterQueryService().league_rosters().teams]


def get_team_names_provider() -> TeamNamesProvider:
    """DI seam so tests inject a fake names list (no live DB)."""
    return _live_team_names
```

- [ ] **Step 3c: Router — `api/routers/admin.py`**

```python
# api/routers/admin.py
"""Admin router (beta user->team assignment). THIN: gate + store calls only.
Reconciliation/auth logic lives in api/admin.py + api/tenancy.py."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from api.admin import TeamNamesProvider, get_team_names_provider, require_admin
from api.contracts.admin import Assignment, AssignmentRequest, AssignmentsResponse
from api.deps import get_league_store, get_membership_store, get_user_store
from api.stores.league_store import LeagueStore
from api.stores.membership_store import MembershipStore
from api.stores.user_store import AppUser, UserStore
from api.tenancy import reconcile_team_name

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.post("/assignments", response_model=Assignment)
def assign_team(
    req: AssignmentRequest,
    admin: AppUser = Depends(require_admin),
    users: UserStore = Depends(get_user_store),
    leagues: LeagueStore = Depends(get_league_store),
    members: MembershipStore = Depends(get_membership_store),
    names_provider: TeamNamesProvider = Depends(get_team_names_provider),
) -> Assignment:
    league = leagues.get_or_create_default() if req.league_id is None else leagues.get(req.league_id)
    if league is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="League not found.")
    canonical = reconcile_team_name(req.team_name, names_provider())
    if canonical is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown team '{req.team_name}'. Choose one of: {', '.join(names_provider())}",
        )
    target = users.get_or_create(req.clerk_user_id)  # pre-provision (may not have logged in yet)
    m = members.assign(target.id, league.id, canonical, team_key=None, assigned_by=admin.id)
    return Assignment(clerk_user_id=req.clerk_user_id, user_id=m.user_id, league_id=m.league_id, team_name=m.team_name)


@router.get("/assignments", response_model=AssignmentsResponse)
def list_assignments(
    admin: AppUser = Depends(require_admin),
    leagues: LeagueStore = Depends(get_league_store),
    members: MembershipStore = Depends(get_membership_store),
    names_provider: TeamNamesProvider = Depends(get_team_names_provider),
) -> AssignmentsResponse:
    league = leagues.get_or_create_default()
    rows = members.list_for_league(league.id)
    return AssignmentsResponse(
        assignments=[
            Assignment(clerk_user_id="", user_id=m.user_id, league_id=m.league_id, team_name=m.team_name)
            for m in rows
        ],
        available_teams=names_provider(),
    )
```

> Note: `clerk_user_id` is empty in the list response because the membership row stores `user_id`, not the Clerk id (the store is keyed by local id). Surfacing the Clerk id back is a future nicety (needs a `UserStore.get(id)` reverse lookup); out of M3-1 scope. The list is for verifying team assignments, which it does.

- [ ] **Step 3d: Mount in `api/main.py`** (run `git pull --no-rebase origin master` first)

Add with the other router imports:

```python
    from api.routers.admin import router as admin_router
```

Add with the other `include_router` calls:

```python
    app.include_router(admin_router)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/api/test_api_admin_assignments.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
python -m ruff check api/ --fix && python -m ruff format api/admin.py api/contracts/admin.py api/routers/admin.py api/main.py
git add api/admin.py api/contracts/admin.py api/routers/admin.py api/main.py tests/api/test_api_admin_assignments.py
git commit -m "feat(api): admin user->team assignment endpoints (deny-by-default gate) — B4/M3-1

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Wire the 7 personalized routers (additive, dormant)

**Files (each: add `ctx=Depends(require_viewer_context)`, pass `ctx.effective_team(team_name)`):**
- Modify: `api/routers/team.py`, `matchup.py`, `lineup.py`, `punt.py`, `free_agents.py`, `playoff.py`, `trade_finder.py`
- Test: `tests/api/test_api_personalized_team_resolution.py`

> `git pull --no-rebase origin master` BEFORE starting (these are shared hot files).

- [ ] **Step 1: Write the failing test**

```python
# tests/api/test_api_personalized_team_resolution.py
"""The personalized endpoints resolve the viewer's team from identity, falling back
to the query param when dormant. Targets the /free-agents/pool route with a spy
service (FreeAgentPoolResponse is no-arg constructible) to assert which team name
the router forwarded — DB-free. (The resolver itself is proven in
test_api_tenancy_resolver.py; this proves the real routers wire it.)"""

from fastapi import FastAPI
from starlette.testclient import TestClient

from api.auth import Principal, get_auth_verifier
from api.contracts.free_agents import FreeAgentPoolResponse
from api.deps import get_fa_pool_service, get_league_store, get_membership_store, get_user_store
from api.routers.free_agents import router as fa_router
from api.stores.league_store import InMemoryLeagueStore
from api.stores.membership_store import InMemoryMembershipStore
from api.stores.user_store import InMemoryUserStore


class _SpyPoolService:
    def __init__(self):
        self.seen = None

    def get_free_agents_pool(self, team_name, limit=100):
        self.seen = team_name  # capture what the router forwarded
        return FreeAgentPoolResponse()


class _Clerk:
    def verify(self, authorization):
        return Principal(subject="user_42", clerk_user_id="user_42")


def _app(spy):
    app = FastAPI()
    app.include_router(fa_router)
    app.dependency_overrides[get_fa_pool_service] = lambda: spy
    return app


def test_pool_endpoint_dormant_uses_query_param():
    spy = _SpyPoolService()
    app = _app(spy)
    app.dependency_overrides[get_user_store] = lambda: InMemoryUserStore()
    app.dependency_overrides[get_league_store] = lambda: InMemoryLeagueStore()
    app.dependency_overrides[get_membership_store] = lambda: InMemoryMembershipStore()
    r = TestClient(app).get("/api/free-agents/pool?team_name=Team%20Hickey")
    assert r.status_code == 200
    assert spy.seen == "Team Hickey"  # unchanged behavior when off


def test_pool_endpoint_resolves_assigned_team_for_clerk_user():
    spy = _SpyPoolService()
    app = _app(spy)
    users, leagues, members = InMemoryUserStore(), InMemoryLeagueStore(), InMemoryMembershipStore()
    u = users.get_or_create("user_42")
    lg = leagues.get_or_create_default()
    members.assign(u.id, lg.id, "Bronx Bombers", team_key=None, assigned_by=u.id)
    app.dependency_overrides[get_auth_verifier] = lambda: _Clerk()
    app.dependency_overrides[get_user_store] = lambda: users
    app.dependency_overrides[get_league_store] = lambda: leagues
    app.dependency_overrides[get_membership_store] = lambda: members
    r = TestClient(app).get(
        "/api/free-agents/pool?team_name=Team%20Hickey", headers={"Authorization": "Bearer x"}
    )
    assert r.status_code == 200
    assert spy.seen == "Bronx Bombers"  # resolved overrides the param
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/api/test_api_personalized_team_resolution.py -q`
Expected: FAIL — `test_pool_endpoint_resolves_assigned_team_for_clerk_user` fails (`spy.seen == "Team Hickey"`, not "Bronx Bombers") because the router doesn't yet resolve identity. (The dormant test passes already — that's today's behavior.)

- [ ] **Step 3: Edit each router (additive — keep the existing `team_name` param + service dep)**

`api/routers/team.py`:
```python
from api.tenancy import ViewerContext, require_viewer_context

@router.get("/me/team", response_model=MyTeamResponse)
def get_my_team(
    team_name: str,
    ctx: ViewerContext = Depends(require_viewer_context),
    service=Depends(get_team_service),
) -> MyTeamResponse:
    return service.get_my_team(ctx.effective_team(team_name))
```

`api/routers/matchup.py`:
```python
from api.tenancy import ViewerContext, require_viewer_context

@router.get("/matchup", response_model=MatchupResponse)
def get_matchup(
    team_name: str = "",
    ctx: ViewerContext = Depends(require_viewer_context),
    service=Depends(get_matchup_service),
) -> MatchupResponse:
    return service.get_matchup(ctx.effective_team(team_name))
```

`api/routers/punt.py`:
```python
from api.tenancy import ViewerContext, require_viewer_context

@router.get("/punt", response_model=PuntResponse)
def get_punt(
    team_name: str = "",
    ctx: ViewerContext = Depends(require_viewer_context),
    service=Depends(get_punt_service),
) -> PuntResponse:
    return service.get_punt(ctx.effective_team(team_name))
```

`api/routers/playoff.py`:
```python
from api.tenancy import ViewerContext, require_viewer_context

@router.get("/playoff-odds", response_model=PlayoffOddsResponse)
def get_playoff_odds(
    team_name: str,
    ctx: ViewerContext = Depends(require_viewer_context),
    service=Depends(get_playoff_service),
) -> PlayoffOddsResponse:
    return service.get_playoff_odds(ctx.effective_team(team_name))
```

`api/routers/trade_finder.py` (preserve the existing `limit` param + call shape):
```python
from api.tenancy import ViewerContext, require_viewer_context

# in the route signature add the ctx dep; change the service call's team_name arg:
#   team_name: str = "",
#   ctx: ViewerContext = Depends(require_viewer_context),
#   ...
#   return service.get_suggestions(team_name=ctx.effective_team(team_name), limit=limit)
```

`api/routers/free_agents.py` (it has TWO routes — wire BOTH; keep `limit`):
```python
from api.tenancy import ViewerContext, require_viewer_context

@router.get("/free-agents", response_model=FreeAgentsResponse)
def get_free_agents(
    team_name: str,
    limit: int = 5,
    ctx: ViewerContext = Depends(require_viewer_context),
    service=Depends(get_fa_service),
) -> FreeAgentsResponse:
    return service.get_free_agents(ctx.effective_team(team_name), limit)

@router.get("/free-agents/pool", response_model=FreeAgentPoolResponse)
def get_free_agents_pool(
    team_name: str,
    limit: int = 100,
    ctx: ViewerContext = Depends(require_viewer_context),
    service=Depends(get_fa_pool_service),
) -> FreeAgentPoolResponse:
    return service.get_free_agents_pool(ctx.effective_team(team_name), limit)
```

`api/routers/lineup.py` is a `POST` whose team is in the request body (`req.team_name`) and is **Pro-gated** (`dependencies=[Depends(require_pro)]`). The `ctx` dep coexists with the Pro gate — keep the gate, resolve over the body field:
```python
from api.tenancy import ViewerContext, require_viewer_context

@router.post(
    "/lineup/optimize",
    response_model=LineupOptimizeResponse,
    dependencies=[Depends(require_pro)],
    responses=_PRO_GATE,
)
def optimize_lineup(
    req: LineupOptimizeRequest,
    ctx: ViewerContext = Depends(require_viewer_context),
    service=Depends(get_lineup_service),
) -> LineupOptimizeResponse:
    return service.optimize(ctx.effective_team(req.team_name), req.date, req.scope, req.mode)
```

- [ ] **Step 4: Run to verify it passes + no regressions**

Run: `python -m pytest tests/api/test_api_personalized_team_resolution.py tests/api/test_no_logic_in_routers.py -q`
Expected: PASS (both the new tests and the router-guard — `ctx.effective_team(...)` is a call, not a BinOp assignment, and `require_viewer_context` is imported from `api.tenancy`, not `src.*`).

Run the full api suite to confirm the existing endpoint tests still pass (they call with `team_name=` and no token → dormant fallback = unchanged):
Run: `python -m pytest tests/api/ -q`
Expected: PASS (all existing + new).

- [ ] **Step 5: Commit + checkpoint push**

```bash
git pull --no-rebase origin master
python -m ruff check api/ --fix && python -m ruff format api/routers/
git add api/routers/ tests/api/test_api_personalized_team_resolution.py
git commit -m "feat(api): personalized endpoints resolve viewer team from identity (dormant fallback) — B4/M3-1

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git pull --no-rebase origin master && git push origin master
```

---

## Task 7: Regenerate OpenAPI + full verification

**Files:**
- Modify: `api/openapi.json` (generated)
- Test: existing `tests/api/test_openapi_contract.py` (snapshot guard)

- [ ] **Step 1: Run the snapshot test to confirm it now fails (admin routes added)**

Run: `python -m pytest tests/api/test_openapi_contract.py -q`
Expected: FAIL — the live schema now includes `/api/admin/assignments`; the snapshot is stale.

- [ ] **Step 2: Regenerate the snapshot**

The repo's regen command (named in the test's own failure message) is:
```bash
python scripts/export_openapi.py
```

- [ ] **Step 3: Verify the snapshot test passes**

Run: `python -m pytest tests/api/test_openapi_contract.py -q`
Expected: PASS

- [ ] **Step 4: Run the API live + smoke the new surface**

Start the API (factory):
```bash
.venv/Scripts/python.exe -m uvicorn api.main:create_app --factory --host 127.0.0.1 --port 8000
```
In another shell, confirm a personalized read still works WITHOUT a token (dormant):
```bash
curl -s "http://127.0.0.1:8000/api/me/team?team_name=Team%20Hickey" -o /dev/null -w "%{http_code}\n"
```
Expected: `200` (byte-for-byte today's behavior). Confirm `/api/admin/assignments` requires auth:
```bash
curl -s "http://127.0.0.1:8000/api/admin/assignments" -o /dev/null -w "%{http_code}\n"
```
Expected: `401` (no token → require_app_user fail-closed). Stop the server.

- [ ] **Step 5: Full api suite + structural guards + commit + push**

```bash
python -m pytest tests/api/ -q
python -m pytest tests/api/test_no_logic_in_routers.py tests/api/test_openapi_contract.py -q
git pull --no-rebase origin master
git add api/openapi.json
git commit -m "chore(api): regen openapi for admin assignment routes — B4/M3-1

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push origin master
```

---

## Definition of done (M3-1)

- New api-owned `league` + `user_team` stores in `api_state.db` (NEVER `draft_tool.db`); idempotent, dormant until used.
- `require_viewer_context` resolves the viewer's team from a valid Clerk identity; falls back to the existing `team_name` query param when no/invalid token (reads stay open — byte-for-byte today's behavior).
- Admin-only assignment endpoints (deny-by-default env allowlist) with team-name reconciliation against the live roster names (graceful when cold).
- The 7 personalized routers wired additively; engine + `src/` untouched; router guard + openapi snapshot green; full api suite green.
- Pre-push structural-invariant suite green; pushed to origin/master.

## What is explicitly NOT in M3-1 (owner-gated / later slices)

- `league_credential` table + per-user OAuth (M4-2).
- `league_id` tagging of the engine's baseball tables + request-scoped repository filtering + tenant-isolation tests (M4-1; 🔒 needs real Postgres).
- Requiring auth on reads / flipping the frontend off its hardcoded team (the M3 activation step — frontend lane + owner env).
- Reverse Clerk-id surfacing in the assignments list (nicety).

## Code review before merge

Run a `code-reviewer` + `silent-failure-hunter` pass over the diff (focus: the sqlite stores' error handling/close-on-failure, the `optional_app_user` open-read semantics, and the admin deny-by-default gate). Apply findings before the final push.

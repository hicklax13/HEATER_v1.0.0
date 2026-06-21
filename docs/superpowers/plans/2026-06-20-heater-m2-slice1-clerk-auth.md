# M2 Slice 1 — Clerk Authentication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drop a `ClerkVerifier` (PyJWT-JWKS local verification) into the existing `api/auth.py` seam, make `get_auth_verifier()` config-driven, carry the Clerk user id on `Principal`, and provision a local `AppUser` on the first authenticated Clerk call — all ADDITIVE and dormant by default (Clerk env unset → the interim `EnvTokenVerifier`, no provisioning, no table).

**Architecture:** `ClerkVerifier` implements the existing `AuthVerifier` Protocol, so `require_principal` + routers never change. A new `api/stores/` persistence layer (`UserStore` Protocol + in-memory fake + SQLite default in a SEPARATE `data/api_state.db`) holds the local user; a `require_app_user` dependency in `api/identity.py` composes auth + the store. Tests inject a local RS256 keypair (no network) and in-memory stores (no DB) — fully DB-free.

**Tech Stack:** FastAPI, PyJWT[crypto] (new dep), pydantic, sqlite3 (api-owned file).

---

## Context an executor needs

- The seam to extend: `api/auth.py` — `Principal` (pydantic), `AuthVerifier` (Protocol), `EnvTokenVerifier` (deny-by-default, KEEP byte-identical), `get_auth_verifier()` (DI provider), `require_principal` (the dependency). `_bearer()` parses `Authorization: Bearer <t>`.
- `api/auth.py` deliberately imports nothing from `api.deps` (no import cycle). KEEP that — put anything needing `api.deps` in the new `api/identity.py`.
- Run tests (shared venv): `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/ -q`
- Lint: `python -m ruff check api/ tests/api/`
- Regen openapi: `python scripts/export_openapi.py` (expect NO diff this slice — nothing here is in the schema).
- Gotcha: worktree DBs are empty; all tests must be DB-free (injected keypair / in-memory store / tmp_path).
- The `test_no_direct_sqlite_connect_*` guard scans only 3 named scripts + `app.py` — NOT `api/`, so `SqliteUserStore`'s own `sqlite3.connect` is allowed.

---

## Commit A — `ClerkVerifier` + config-driven selector + `Principal.clerk_user_id`

### Task A1: Add the PyJWT dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add the dep** (after the `httpx==0.28.1` line, keep the api-stack pins grouped)

```
# Clerk JWT verification (M2 slice 1): local RS256 verification against Clerk's
# JWKS. Brings `cryptography`. Verifier lives in api/auth.py (ClerkVerifier).
PyJWT[crypto]>=2.8
```

- [ ] **Step 2: Install into the shared venv**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pip install "PyJWT[crypto]>=2.8"`
Expected: installs PyJWT + cryptography (cryptography already present from the Yahoo stack — only PyJWT is new).

- [ ] **Step 3: Verify import**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -c "import jwt; from jwt import PyJWKClient; print(jwt.__version__)"`
Expected: prints a version `>= 2.8`.

### Task A2: Write the failing ClerkVerifier + Principal tests

**Files:**
- Create: `tests/api/test_api_clerk_auth.py`

- [ ] **Step 1: Write the tests**

```python
import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException

from api.auth import ClerkVerifier, EnvTokenVerifier, Principal, get_auth_verifier

_ISS = "https://heater.clerk.accounts.dev"


def _keypair():
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return priv, priv.public_key()


def _token(priv, *, iss=_ISS, sub="user_abc", aud=None, exp_delta=3600, kid="kid1"):
    now = int(time.time())
    payload = {"iss": iss, "sub": sub, "iat": now, "exp": now + exp_delta}
    if aud is not None:
        payload["aud"] = aud
    return jwt.encode(payload, priv, algorithm="RS256", headers={"kid": kid})


def _verifier(pub, *, issuer=_ISS, audience=None):
    # Inject the signing key so no JWKS network call happens (conftest guard).
    return ClerkVerifier(issuer=issuer, audience=audience, signing_key_resolver=lambda _t: pub)


def test_principal_carries_clerk_user_id():
    p = Principal(subject="user_abc", clerk_user_id="user_abc")
    assert p.clerk_user_id == "user_abc"
    assert Principal(subject="api-token").clerk_user_id is None  # optional, back-compat


def test_valid_token_returns_principal():
    priv, pub = _keypair()
    p = _verifier(pub).verify(f"Bearer {_token(priv)}")
    assert isinstance(p, Principal)
    assert p.subject == "user_abc"
    assert p.clerk_user_id == "user_abc"


def test_expired_token_denied():
    priv, pub = _keypair()
    with pytest.raises(HTTPException) as ei:
        _verifier(pub).verify(f"Bearer {_token(priv, exp_delta=-10)}")
    assert ei.value.status_code == 401


def test_bad_signature_denied():
    priv, _ = _keypair()
    _, other_pub = _keypair()
    with pytest.raises(HTTPException) as ei:
        _verifier(other_pub).verify(f"Bearer {_token(priv)}")
    assert ei.value.status_code == 401


def test_wrong_issuer_denied():
    priv, pub = _keypair()
    with pytest.raises(HTTPException) as ei:
        _verifier(pub, issuer="https://evil.example").verify(f"Bearer {_token(priv)}")
    assert ei.value.status_code == 401


def test_wrong_audience_denied():
    priv, pub = _keypair()
    with pytest.raises(HTTPException) as ei:
        _verifier(pub, audience="heater-api").verify(f"Bearer {_token(priv, aud='other')}")
    assert ei.value.status_code == 401


def test_correct_audience_accepted():
    priv, pub = _keypair()
    p = _verifier(pub, audience="heater-api").verify(f"Bearer {_token(priv, aud='heater-api')}")
    assert p.clerk_user_id == "user_abc"


def test_missing_or_garbage_bearer_denied():
    priv, pub = _keypair()
    v = _verifier(pub)
    for header in (None, "", "Bearer", "Bearer   ", "Token x"):
        with pytest.raises(HTTPException) as ei:
            v.verify(header)
        assert ei.value.status_code == 401


def test_non_ascii_bearer_denies_not_crashes():
    priv, pub = _keypair()
    with pytest.raises(HTTPException) as ei:
        _verifier(pub).verify("Bearer café")  # non-ASCII → clean 401, not 500
    assert ei.value.status_code == 401


def test_missing_subject_denied():
    priv, pub = _keypair()
    with pytest.raises(HTTPException) as ei:
        _verifier(pub).verify(f"Bearer {_token(priv, sub='')}")
    assert ei.value.status_code == 401


def test_selector_returns_clerk_when_issuer_set(monkeypatch):
    monkeypatch.setenv("CLERK_ISSUER", _ISS)
    assert isinstance(get_auth_verifier(), ClerkVerifier)


def test_selector_returns_env_verifier_when_issuer_unset(monkeypatch):
    monkeypatch.delenv("CLERK_ISSUER", raising=False)
    assert isinstance(get_auth_verifier(), EnvTokenVerifier)
```

- [ ] **Step 2: Run to verify it fails**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_clerk_auth.py -q`
Expected: FAIL with `ImportError: cannot import name 'ClerkVerifier'`.

### Task A3: Implement ClerkVerifier + Principal.clerk_user_id + selector

**Files:**
- Modify: `api/auth.py`

- [ ] **Step 1: Add imports** (top of file, with the existing imports)

```python
import logging
from typing import Callable, Protocol

import jwt
from jwt import PyJWKClient
```
(Keep the existing `import hmac`, `import os`, the fastapi + pydantic imports. `Protocol` is already imported — merge, don't duplicate.)

- [ ] **Step 2: Add a module logger + the shared 401 helper** (after the `_TOKEN_ENV` constant)

```python
logger = logging.getLogger(__name__)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )
```

- [ ] **Step 3: Extend `Principal`** (add the optional field; KEEP `subject`)

```python
class Principal(BaseModel):
    """The authenticated caller. `subject` is opaque (the Clerk `sub` for Clerk
    callers, "api-token" for the env path). `clerk_user_id` is set only for Clerk
    callers — it ties the request to a local AppUser at provisioning time."""

    subject: str
    clerk_user_id: str | None = None
```

- [ ] **Step 4: Add `ClerkVerifier`** (after `EnvTokenVerifier`)

```python
SigningKeyResolver = Callable[[str], object]


class ClerkVerifier:
    """Verifies a Clerk RS256 session JWT against Clerk's JWKS, locally — no
    per-request network call, no Clerk SDK. Fail-closed: any error → 401. Tests
    inject `signing_key_resolver` so no network is touched."""

    def __init__(
        self,
        issuer: str,
        audience: str | None = None,
        jwks_url: str | None = None,
        *,
        signing_key_resolver: SigningKeyResolver | None = None,
        leeway: int = 30,
    ) -> None:
        self._issuer = issuer
        self._audience = audience or None
        self._jwks_url = jwks_url or (issuer.rstrip("/") + "/.well-known/jwks.json")
        self._leeway = leeway
        self._resolver = signing_key_resolver
        self._client: PyJWKClient | None = None

    def _signing_key(self, token: str) -> object:
        if self._resolver is not None:
            return self._resolver(token)
        if self._client is None:
            # PyJWKClient caches keys and refetches on an unknown kid (rotation).
            self._client = PyJWKClient(self._jwks_url)
        return self._client.get_signing_key_from_jwt(token).key

    def verify(self, authorization: str | None) -> Principal:
        token = _bearer(authorization)
        if token is None:
            raise _unauthorized("Invalid or missing bearer token.")
        try:
            key = self._signing_key(token)
            claims = jwt.decode(
                token,
                key,
                algorithms=["RS256"],
                issuer=self._issuer,
                audience=self._audience,
                leeway=self._leeway,
                options={"require": ["exp", "iss", "sub"], "verify_aud": self._audience is not None},
            )
        except Exception as exc:
            # Fail closed on ANY error (bad sig, expired, wrong iss/aud, unreachable
            # JWKS, malformed). Log the failure TYPE (never the token/claims).
            logger.debug("Clerk token verification failed: %s", type(exc).__name__)
            raise _unauthorized("Invalid or expired token.")
        subject = claims.get("sub")
        if not subject:
            raise _unauthorized("Token missing subject.")
        return Principal(subject=subject, clerk_user_id=subject)
```

- [ ] **Step 5: Make `get_auth_verifier()` config-driven** (replace the existing body)

```python
def get_auth_verifier() -> AuthVerifier:
    """DI provider for the write-auth verifier. Returns ClerkVerifier when Clerk
    is configured (CLERK_ISSUER set), else the interim EnvTokenVerifier (the
    server-to-server/CI + dormant default). Read at call time so env changes take
    effect without reload. Tests override this."""
    issuer = os.environ.get("CLERK_ISSUER", "").strip()
    if issuer:
        return ClerkVerifier(
            issuer=issuer,
            audience=os.environ.get("CLERK_AUDIENCE", "").strip() or None,
            jwks_url=os.environ.get("CLERK_JWKS_URL", "").strip() or None,
        )
    return EnvTokenVerifier()
```

- [ ] **Step 6: Run the tests**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_clerk_auth.py -q`
Expected: PASS (all 13).

### Task A4: Regression + openapi + commit A

- [ ] **Step 1: Existing auth tests stay green** (verifier-agnostic via override)

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_roster_write_auth.py tests/api/test_api_write_auth_openapi.py -q`
Expected: PASS.

- [ ] **Step 2: Regen openapi (expect no change)**

Run: `python scripts/export_openapi.py && git diff --stat api/openapi.json`
Expected: prints "wrote ..."; NO diff on `api/openapi.json` (ClerkVerifier/Principal aren't in the schema).

- [ ] **Step 3: Full api suite + lint**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/ -q && python -m ruff check api/ tests/api/`
Expected: PASS, no lint errors.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt api/auth.py tests/api/test_api_clerk_auth.py
git commit -m "feat(api): ClerkVerifier (PyJWT-JWKS) + config-driven auth selector — M2 slice 1A"
```

---

## Commit B — `UserStore` + local AppUser provisioning

### Task B1: Write the failing store tests

**Files:**
- Create: `tests/api/test_api_user_store.py`

- [ ] **Step 1: Write the tests**

```python
from api.stores.user_store import AppUser, InMemoryUserStore, SqliteUserStore


def test_inmemory_get_or_create_is_idempotent():
    store = InMemoryUserStore()
    a = store.get_or_create("user_1")
    b = store.get_or_create("user_1")
    assert isinstance(a, AppUser)
    assert a.id == b.id
    assert a.clerk_user_id == "user_1"


def test_inmemory_distinct_users_get_distinct_ids():
    store = InMemoryUserStore()
    assert store.get_or_create("user_1").id != store.get_or_create("user_2").id


def test_sqlite_store_idempotent_in_separate_file(tmp_path):
    db = tmp_path / "api_state.db"
    store = SqliteUserStore(db_path=str(db))
    a = store.get_or_create("user_x")
    b = store.get_or_create("user_x")
    assert a.id == b.id
    assert db.exists()  # api owns its OWN file


def test_sqlite_store_persists_across_instances(tmp_path):
    db = tmp_path / "api_state.db"
    a = SqliteUserStore(db_path=str(db)).get_or_create("user_y")
    b = SqliteUserStore(db_path=str(db)).get_or_create("user_y")
    assert a.id == b.id


def test_sqlite_store_distinct_users(tmp_path):
    db = tmp_path / "api_state.db"
    store = SqliteUserStore(db_path=str(db))
    assert store.get_or_create("u1").id != store.get_or_create("u2").id
```

- [ ] **Step 2: Run to verify it fails**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_user_store.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.stores'`.

### Task B2: Implement the store

**Files:**
- Create: `api/stores/__init__.py` (empty)
- Create: `api/stores/user_store.py`
- Modify: `api/deps.py`

- [ ] **Step 1: Create `api/stores/__init__.py`** (empty file)

- [ ] **Step 2: Create `api/stores/user_store.py`**

```python
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
        self._lock = threading.Lock()  # serialize get-or-create within the process (avoids SELECT/INSERT TOCTOU)

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
```

- [ ] **Step 3: Add the DI provider** to `api/deps.py` (with the other providers; add the import at top with the other `from api.services...` imports block — note this one is `from api.stores...`)

```python
from api.stores.user_store import SqliteUserStore, UserStore


def get_user_store() -> UserStore:
    return SqliteUserStore()
```

- [ ] **Step 4: Run the store tests**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_user_store.py -q`
Expected: PASS (5).

### Task B3: Write the failing provisioning tests

**Files:**
- Create: `tests/api/test_api_identity_provisioning.py`

- [ ] **Step 1: Write the tests**

```python
from fastapi import Depends, FastAPI
from starlette.testclient import TestClient

from api.auth import Principal, get_auth_verifier
from api.deps import get_user_store
from api.identity import provision_app_user, require_app_user
from api.stores.user_store import InMemoryUserStore


def test_provision_returns_none_on_env_token_path():
    # A Principal without clerk_user_id (env-token/server path) provisions nothing.
    assert provision_app_user(Principal(subject="api-token"), InMemoryUserStore()) is None


def test_provision_get_or_creates_for_clerk_principal():
    store = InMemoryUserStore()
    p = Principal(subject="user_9", clerk_user_id="user_9")
    u1 = provision_app_user(p, store)
    u2 = provision_app_user(p, store)
    assert u1 is not None
    assert u1.id == u2.id  # idempotent
    assert u1.clerk_user_id == "user_9"


class _ClerkPrincipalVerifier:
    def verify(self, authorization):
        return Principal(subject="user_42", clerk_user_id="user_42")


class _EnvStyleVerifier:
    def verify(self, authorization):
        return Principal(subject="api-token")


def _app_with_protected_route():
    app = FastAPI()

    @app.get("/whoami")
    def whoami(user=Depends(require_app_user)):
        return {"id": user.id, "clerk_user_id": user.clerk_user_id} if user else {"id": None}

    return app


def test_require_app_user_provisions_via_dependency():
    app = _app_with_protected_route()
    store = InMemoryUserStore()
    app.dependency_overrides[get_auth_verifier] = lambda: _ClerkPrincipalVerifier()
    app.dependency_overrides[get_user_store] = lambda: store
    body = TestClient(app).get("/whoami", headers={"Authorization": "Bearer anything"}).json()
    assert body["clerk_user_id"] == "user_42"
    assert body["id"] == 1


def test_require_app_user_none_on_env_path():
    app = _app_with_protected_route()
    app.dependency_overrides[get_auth_verifier] = lambda: _EnvStyleVerifier()
    app.dependency_overrides[get_user_store] = lambda: InMemoryUserStore()
    body = TestClient(app).get("/whoami", headers={"Authorization": "Bearer x"}).json()
    assert body["id"] is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_identity_provisioning.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.identity'`.

### Task B4: Implement provisioning + final checks + commit B

**Files:**
- Create: `api/identity.py`

- [ ] **Step 1: Create `api/identity.py`**

```python
"""Request-identity dependencies that need api-owned persistence.

Kept separate from api/auth.py (which imports nothing from api.deps, to stay
import-cycle-free). This module composes auth (require_principal) + the UserStore
to provision a local AppUser on the first authenticated Clerk call. Dormant on the
env-token path (Principal.clerk_user_id is None → no provisioning, no table)."""

from __future__ import annotations

from fastapi import Depends

from api.auth import Principal, require_principal
from api.deps import get_user_store
from api.stores.user_store import AppUser, UserStore


def provision_app_user(principal: Principal, store: UserStore) -> AppUser | None:
    """Get-or-create the local AppUser for a verified Clerk caller. Returns None
    for the env-token path (no clerk_user_id) so reads/server-to-server stay
    dormant. Pure + DB-free-testable (the store is injected)."""
    if not principal.clerk_user_id:
        return None
    return store.get_or_create(principal.clerk_user_id)


def require_app_user(
    principal: Principal = Depends(require_principal),
    store: UserStore = Depends(get_user_store),
) -> AppUser | None:
    """FastAPI dependency: a verified caller + their provisioned local AppUser
    (None on the env-token path). Slice 2's billing routes depend on this so a
    Stripe customer can be tied to a stable local user id."""
    return provision_app_user(principal, store)
```

- [ ] **Step 2: Run the provisioning tests**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_identity_provisioning.py -q`
Expected: PASS (4).

- [ ] **Step 3: Full api suite + lint + openapi (no change)**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/ -q && python -m ruff check api/ tests/api/ && python scripts/export_openapi.py && git diff --stat api/openapi.json`
Expected: PASS; no lint errors; NO diff on `api/openapi.json`.

- [ ] **Step 4: Commit**

```bash
git add api/stores/ api/identity.py api/deps.py tests/api/test_api_user_store.py tests/api/test_api_identity_provisioning.py
git commit -m "feat(api): local AppUser provisioning (UserStore + require_app_user) — M2 slice 1B"
```

---

## Self-review checklist (run after both commits)

1. **Spec coverage:** ClerkVerifier ✓ (A3), config-driven selector ✓ (A3 step 5), Principal.clerk_user_id ✓ (A3 step 3), local user provisioned on first authenticated call ✓ (B4), dormant when Clerk unset ✓ (selector returns EnvTokenVerifier + provision returns None), DB-free tests ✓ (injected keypair / in-memory store / tmp_path), openapi regenerated ✓.
2. **Boundary respected:** no multi-tenancy (no team/tenant field on AppUser), no Yahoo scope change, `require_principal` + routers unchanged, `EnvTokenVerifier` byte-identical, `src/` untouched.
3. **Type consistency:** `get_or_create(clerk_user_id) -> AppUser` used identically in both stores + `provision_app_user`; `Principal.clerk_user_id` read in `provision_app_user` and set in `ClerkVerifier.verify`.

## Review gate
After commit B: dispatch a `pr-review-toolkit:code-reviewer` (full engine context) + a `pr-review-toolkit:silent-failure-hunter` over the diff (the fail-closed except block + the SQLite store are the things to scrutinize). Apply findings before merge.

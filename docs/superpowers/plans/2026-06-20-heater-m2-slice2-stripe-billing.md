# M2 Slice 2 — Stripe Billing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Free→Pro Stripe billing (single-tenant): a checkout-session endpoint, a signature-verified webhook that records subscription tier, and a "my subscription" read endpoint — additive + dormant until the Stripe env is set.

**Architecture:** Per the spec `docs/superpowers/specs/2026-06-20-heater-m2-stripe-billing-design.md`. Three units behind the seam idiom: `StripeGateway` (Protocol; `LiveStripeGateway` / `NullStripeGateway` / a `FakeStripeGateway` in tests), `SubscriptionStore` (Protocol; in-memory + SQLite in the slice-1 `api_state.db`), `BillingService` (the one billing-logic seam). Tests inject the fake gateway + in-memory store → DB-free + network-free.

**Tech Stack:** FastAPI, `stripe==15.2.1` (new, pinned exact), pydantic, sqlite3 (the separate `api_state.db`).

---

## Context an executor needs

- Reuse the slice-1 SQLite pattern verbatim (`api/stores/user_store.py::SqliteUserStore`): self-cleaning `_connect` (close on a setup failure before return), WAL + busy_timeout, per-call connect/close, a `threading.Lock`, idempotent `CREATE TABLE IF NOT EXISTS`, a WARNING breadcrumb on the failure path.
- `require_app_user` (from `api/identity.py`, slice 1) returns `AppUser | None` — None on the env-token path. Billing's authed routes depend on it; a None app_user → graceful "sign in required".
- Run tests: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/ -q`. Lint: `python -m ruff check api/ tests/api/`. Regen openapi: `python scripts/export_openapi.py` (this slice DOES change it — 3 new routes).
- Stripe v15: `stripe.SignatureVerificationError` (top-level), `stripe.Webhook.construct_event(payload, sig_header, secret)`, `stripe.Customer.create`, `stripe.checkout.Session.create`. A `stripe.Event`/`StripeObject` is a dict subclass — `event["type"]`, `event["data"]["object"]`, `.get(...)` all work, so the same service code handles a real event and a fake plain-dict event.

---

## Commit A — dependency + contracts + StripeGateway

### Task A1: Add the Stripe dependency

**Files:** Modify `requirements.txt`

- [ ] **Step 1: Add (after the `PyJWT[crypto]` line)**

```
# Stripe billing (M2 slice 2). PINNED EXACT (like fastapi/httpx) so the webhook
# event schema the SDK parses can't drift under us. Bump deliberately + re-test.
stripe==15.2.1
```

- [ ] **Step 2: Install + verify**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pip install "stripe==15.2.1" && C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -c "import stripe; print(stripe.VERSION)"`
Expected: prints `15.2.1`.

### Task A2: Contracts

**Files:** Create `api/contracts/billing.py`

- [ ] **Step 1: Write the models** (no test — pydantic models; exercised by the service/route tests)

```python
"""Billing (Stripe) contracts. ok/error + graceful defaults follow the write-route
convention (never raise; the frontend reads ok + error)."""

from __future__ import annotations

from pydantic import BaseModel


class CheckoutSessionRequest(BaseModel):
    success_url: str | None = None  # frontend supplies; falls back to STRIPE_SUCCESS_URL / CORS origin
    cancel_url: str | None = None


class CheckoutSessionResponse(BaseModel):
    ok: bool
    url: str | None = None
    error: str | None = None


class SubscriptionResponse(BaseModel):
    tier: str = "free"  # "free" | "pro"
    status: str = "none"  # none|trialing|active|canceled|past_due|unpaid|incomplete
    current_period_end: int | None = None
    trial: bool = False
    error: str | None = None


class WebhookResponse(BaseModel):
    ok: bool
    event_type: str | None = None
    handled: bool = False
```

### Task A3: StripeGateway — write the failing test

**Files:** Create `tests/api/test_api_stripe_gateway.py`

- [ ] **Step 1: Write the test**

```python
import pytest

from api.gateways.stripe_gateway import BillingSignatureError, NullStripeGateway


def test_null_gateway_is_unconfigured_and_refuses():
    g = NullStripeGateway()
    assert g.configured is False
    with pytest.raises(RuntimeError):
        g.create_customer("user_1", None)
    with pytest.raises(RuntimeError):
        g.create_checkout_session(
            customer_id="c", price_id="p", success_url="s", cancel_url="x", trial_days=7, clerk_user_id="user_1"
        )
    with pytest.raises(BillingSignatureError):
        g.parse_webhook_event(b"{}", "sig", "secret")


def test_billing_signature_error_is_exception():
    assert issubclass(BillingSignatureError, Exception)
```

- [ ] **Step 2: Run → FAIL** (`ModuleNotFoundError: api.gateways`)

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_stripe_gateway.py -q`

### Task A4: StripeGateway — implement

**Files:** Create `api/gateways/__init__.py` (empty), `api/gateways/stripe_gateway.py`

- [ ] **Step 1: Create `api/gateways/__init__.py`** (empty)

- [ ] **Step 2: Create `api/gateways/stripe_gateway.py`**

```python
"""Stripe SDK seam — the ONE place the api/ layer imports `stripe`. A Protocol so
tests inject a fake (no network, no real charges). LiveStripeGateway wraps the SDK;
NullStripeGateway is the dormant default when Stripe is unconfigured."""

from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class BillingSignatureError(Exception):
    """Webhook signature missing/invalid, or the event could not be parsed
    (fail-closed signal — the service maps this to HTTP 400)."""


class StripeGateway(Protocol):
    configured: bool

    def create_customer(self, clerk_user_id: str, email: str | None) -> str: ...

    def create_checkout_session(
        self,
        *,
        customer_id: str,
        price_id: str,
        success_url: str,
        cancel_url: str,
        trial_days: int,
        clerk_user_id: str,
    ) -> str: ...

    def parse_webhook_event(self, payload: bytes, sig_header: str | None, secret: str) -> dict: ...


class NullStripeGateway:
    """Dormant default when STRIPE_SECRET_KEY is unset. Refuses every call so the
    service's not-configured guard short-circuits before it (the parse path
    fail-closes to a signature error → 400)."""

    configured = False

    def create_customer(self, clerk_user_id: str, email: str | None) -> str:
        raise RuntimeError("Stripe not configured")

    def create_checkout_session(self, **kwargs) -> str:
        raise RuntimeError("Stripe not configured")

    def parse_webhook_event(self, payload: bytes, sig_header: str | None, secret: str) -> dict:
        raise BillingSignatureError("Stripe not configured")


class LiveStripeGateway:
    """Real Stripe. Constructed only when STRIPE_SECRET_KEY is set."""

    configured = True

    def __init__(self, api_key: str) -> None:
        import stripe

        self._stripe = stripe
        stripe.api_key = api_key

    def create_customer(self, clerk_user_id: str, email: str | None) -> str:
        customer = self._stripe.Customer.create(email=email, metadata={"clerk_user_id": clerk_user_id})
        return customer.id

    def create_checkout_session(
        self, *, customer_id, price_id, success_url, cancel_url, trial_days, clerk_user_id
    ) -> str:
        session = self._stripe.checkout.Session.create(
            mode="subscription",
            customer=customer_id,
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=success_url,
            cancel_url=cancel_url,
            client_reference_id=clerk_user_id,
            subscription_data={"trial_period_days": trial_days, "metadata": {"clerk_user_id": clerk_user_id}},
        )
        return session.url

    def parse_webhook_event(self, payload: bytes, sig_header: str | None, secret: str) -> dict:
        try:
            return self._stripe.Webhook.construct_event(payload, sig_header, secret)
        except self._stripe.SignatureVerificationError as exc:
            raise BillingSignatureError(str(exc)) from exc
        except Exception as exc:
            # Malformed payload, wrong secret type, etc. — fail closed.
            raise BillingSignatureError(f"Webhook parse failed: {type(exc).__name__}") from exc
```

- [ ] **Step 3: Run → PASS**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_stripe_gateway.py -q`

- [ ] **Step 4: Commit A**

```bash
git add requirements.txt api/contracts/billing.py api/gateways/ tests/api/test_api_stripe_gateway.py
git commit -m "feat(api): Stripe billing contracts + StripeGateway seam — M2 slice 2A"
```

---

## Commit B — SubscriptionStore

### Task B1: Write the failing store test

**Files:** Create `tests/api/test_api_subscription_store.py`

- [ ] **Step 1: Write the test**

```python
import logging
import sqlite3

import pytest

from api.stores.subscription_store import InMemorySubscriptionStore, Subscription, SqliteSubscriptionStore


def _sub(clerk="u1", customer="cus_1", tier="pro", status="active"):
    return Subscription(
        clerk_user_id=clerk, stripe_customer_id=customer, tier=tier, status=status, updated_at="2026-06-20T00:00:00Z"
    )


def test_inmemory_upsert_get_idempotent():
    store = InMemorySubscriptionStore()
    store.upsert(_sub())
    store.upsert(_sub(status="canceled", tier="free"))  # same clerk id overwrites
    got = store.get("u1")
    assert got is not None
    assert got.status == "canceled"
    assert got.tier == "free"


def test_inmemory_get_by_customer():
    store = InMemorySubscriptionStore()
    store.upsert(_sub(clerk="u1", customer="cus_X"))
    assert store.get_by_customer("cus_X").clerk_user_id == "u1"
    assert store.get_by_customer("nope") is None


def test_inmemory_missing_get_is_none():
    assert InMemorySubscriptionStore().get("ghost") is None


def test_sqlite_upsert_get_and_by_customer(tmp_path):
    store = SqliteSubscriptionStore(db_path=str(tmp_path / "api_state.db"))
    store.upsert(_sub(clerk="u9", customer="cus_9"))
    assert store.get("u9").tier == "pro"
    assert store.get_by_customer("cus_9").clerk_user_id == "u9"
    store.upsert(_sub(clerk="u9", customer="cus_9", status="canceled", tier="free"))
    assert store.get("u9").status == "canceled"  # upsert overwrites, not duplicates


def test_sqlite_persists_across_instances(tmp_path):
    db = str(tmp_path / "api_state.db")
    SqliteSubscriptionStore(db_path=db).upsert(_sub(clerk="u2"))
    assert SqliteSubscriptionStore(db_path=db).get("u2") is not None


def test_sqlite_logs_and_propagates_on_failure(tmp_path, monkeypatch, caplog):
    store = SqliteSubscriptionStore(db_path=str(tmp_path / "x.db"))

    class _BoomConn:
        closed = False

        def execute(self, *a, **k):
            raise sqlite3.OperationalError("disk I/O error")

        def commit(self):  # pragma: no cover
            pass

        def close(self):
            self.closed = True

    boom = _BoomConn()
    monkeypatch.setattr(store, "_connect", lambda: boom)
    with caplog.at_level(logging.WARNING):
        with pytest.raises(sqlite3.OperationalError):
            store.upsert(_sub())
    assert "upsert failed" in caplog.text
    assert boom.closed is True
```

- [ ] **Step 2: Run → FAIL** (`ModuleNotFoundError: subscription_store`)

### Task B2: Implement the store

**Files:** Create `api/stores/subscription_store.py`

- [ ] **Step 1: Write it** (mirrors `SqliteUserStore` — same hardened pattern)

```python
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

    _COLS = "clerk_user_id, stripe_customer_id, tier, status, current_period_end, updated_at"

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
```

- [ ] **Step 2: Run → PASS**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_subscription_store.py -q`

- [ ] **Step 3: Commit B**

```bash
git add api/stores/subscription_store.py tests/api/test_api_subscription_store.py
git commit -m "feat(api): SubscriptionStore (api_state.db) — M2 slice 2B"
```

---

## Commit C — BillingService

### Task C1: Write the failing service test

**Files:** Create `tests/api/test_api_billing_service.py`

- [ ] **Step 1: Write the test**

```python
import logging

import pytest
from fastapi import HTTPException

from api.contracts.billing import CheckoutSessionRequest
from api.gateways.stripe_gateway import BillingSignatureError
from api.services.billing_service import BillingService, tier_for
from api.stores.subscription_store import InMemorySubscriptionStore
from api.stores.user_store import AppUser

_USER = AppUser(id=1, clerk_user_id="user_abc", created_at="2026-06-20T00:00:00Z")


class _FakeGateway:
    configured = True

    def __init__(self, event=None, raise_sig=False):
        self.created_customers = 0
        self.sessions = 0
        self._event = event
        self._raise_sig = raise_sig

    def create_customer(self, clerk_user_id, email=None):
        self.created_customers += 1
        return f"cus_{clerk_user_id}"

    def create_checkout_session(self, **kwargs):
        self.sessions += 1
        return "https://checkout.stripe.test/session"

    def parse_webhook_event(self, payload, sig_header, secret):
        if self._raise_sig:
            raise BillingSignatureError("bad sig")
        return self._event


def _sub_event(etype, *, customer="cus_user_abc", clerk="user_abc", status="trialing", period=1800000000):
    obj = {"customer": customer, "status": status, "current_period_end": period, "metadata": {"clerk_user_id": clerk}}
    return {"type": etype, "data": {"object": obj}}


def test_tier_for_mapping():
    assert tier_for("trialing") == "pro"
    assert tier_for("active") == "pro"
    for s in ("canceled", "past_due", "unpaid", "incomplete", "none"):
        assert tier_for(s) == "free"


def test_checkout_unconfigured_is_graceful(monkeypatch):
    monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
    monkeypatch.delenv("STRIPE_PRO_PRICE_ID", raising=False)
    svc = BillingService(_FakeGateway(), InMemorySubscriptionStore())
    resp = svc.create_checkout(_USER, CheckoutSessionRequest())
    assert resp.ok is False
    assert "not configured" in resp.error.lower()


def test_checkout_requires_signed_in_user(monkeypatch):
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test")
    monkeypatch.setenv("STRIPE_PRO_PRICE_ID", "price_x")
    svc = BillingService(_FakeGateway(), InMemorySubscriptionStore())
    resp = svc.create_checkout(None, CheckoutSessionRequest())
    assert resp.ok is False


def test_checkout_creates_then_reuses_customer(monkeypatch):
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test")
    monkeypatch.setenv("STRIPE_PRO_PRICE_ID", "price_x")
    gw = _FakeGateway()
    store = InMemorySubscriptionStore()
    svc = BillingService(gw, store)
    r1 = svc.create_checkout(_USER, CheckoutSessionRequest())
    assert r1.ok and r1.url
    assert gw.created_customers == 1
    # the clerk<->customer link is written immediately (so the webhook can resolve it)
    assert store.get("user_abc").stripe_customer_id == "cus_user_abc"
    r2 = svc.create_checkout(_USER, CheckoutSessionRequest())  # reuses the stored customer
    assert r2.ok
    assert gw.created_customers == 1  # NOT incremented


def test_webhook_subscription_event_sets_tier(monkeypatch):
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_x")
    store = InMemorySubscriptionStore()
    gw = _FakeGateway(event=_sub_event("customer.subscription.created", status="trialing"))
    svc = BillingService(gw, store)
    resp = svc.handle_webhook(b"{}", "sig")
    assert resp.ok and resp.handled
    assert store.get("user_abc").tier == "pro"
    assert store.get("user_abc").status == "trialing"


def test_webhook_deleted_event_downgrades(monkeypatch):
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_x")
    store = InMemorySubscriptionStore()
    gw = _FakeGateway(event=_sub_event("customer.subscription.deleted", status="active"))
    BillingService(gw, store).handle_webhook(b"{}", "sig")
    assert store.get("user_abc").tier == "free"
    assert store.get("user_abc").status == "canceled"


def test_webhook_resolves_clerk_by_customer_when_metadata_absent(monkeypatch):
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_x")
    store = InMemorySubscriptionStore()
    from api.stores.subscription_store import Subscription

    store.upsert(Subscription(clerk_user_id="user_abc", stripe_customer_id="cus_user_abc", updated_at="t"))
    obj = {"customer": "cus_user_abc", "status": "active", "current_period_end": 1, "metadata": {}}
    gw = _FakeGateway(event={"type": "customer.subscription.updated", "data": {"object": obj}})
    BillingService(gw, store).handle_webhook(b"{}", "sig")
    assert store.get("user_abc").tier == "pro"


def test_webhook_bad_signature_400_and_warns(monkeypatch, caplog):
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_x")
    svc = BillingService(_FakeGateway(raise_sig=True), InMemorySubscriptionStore())
    with caplog.at_level(logging.WARNING):
        with pytest.raises(HTTPException) as ei:
            svc.handle_webhook(b"{}", "badsig")
    assert ei.value.status_code == 400
    assert "signature" in caplog.text.lower()


def test_webhook_unconfigured_ignores(monkeypatch):
    monkeypatch.delenv("STRIPE_WEBHOOK_SECRET", raising=False)
    resp = BillingService(_FakeGateway(), InMemorySubscriptionStore()).handle_webhook(b"{}", "sig")
    assert resp.ok is False
    assert resp.handled is False


def test_webhook_unknown_event_type_is_ignored(monkeypatch):
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_x")
    gw = _FakeGateway(event={"type": "invoice.paid", "data": {"object": {}}})
    resp = BillingService(gw, InMemorySubscriptionStore()).handle_webhook(b"{}", "sig")
    assert resp.ok is True
    assert resp.handled is False


def test_read_subscription_defaults_free():
    resp = BillingService(_FakeGateway(), InMemorySubscriptionStore()).read_subscription(_USER)
    assert resp.tier == "free"
    assert resp.status == "none"


def test_read_subscription_reflects_store(monkeypatch):
    from api.stores.subscription_store import Subscription

    store = InMemorySubscriptionStore()
    store.upsert(Subscription(clerk_user_id="user_abc", tier="pro", status="trialing", updated_at="t"))
    resp = BillingService(_FakeGateway(), store).read_subscription(_USER)
    assert resp.tier == "pro"
    assert resp.trial is True
```

- [ ] **Step 2: Run → FAIL** (`ModuleNotFoundError: billing_service`)

### Task C2: Implement the service

**Files:** Create `api/services/billing_service.py`

- [ ] **Step 1: Write it**

```python
"""Billing logic seam — the one place that orchestrates the StripeGateway + the
SubscriptionStore. Never raises on the read/checkout paths (graceful ok/error);
the webhook fails CLOSED on a bad signature (HTTP 400) so Stripe + the operator
see a rejected delivery."""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime

from fastapi import HTTPException

from api.contracts.billing import CheckoutSessionRequest, CheckoutSessionResponse, SubscriptionResponse, WebhookResponse
from api.gateways.stripe_gateway import BillingSignatureError, StripeGateway
from api.stores.subscription_store import Subscription, SubscriptionStore

logger = logging.getLogger(__name__)

_SUB_EVENTS = {"customer.subscription.created", "customer.subscription.updated", "customer.subscription.deleted"}


def tier_for(status: str) -> str:
    """The contract slice 3 consumes: only an active or trialing subscription is Pro."""
    return "pro" if status in {"trialing", "active"} else "free"


def _now() -> str:
    return datetime.now(UTC).isoformat()


class BillingService:
    def __init__(self, gateway: StripeGateway, store: SubscriptionStore) -> None:
        self._gateway = gateway
        self._store = store

    # --- config (read at call time, like get_auth_verifier) -----------------
    @staticmethod
    def _env(name: str) -> str:
        return os.environ.get(name, "").strip()

    def _configured(self) -> bool:
        return bool(self._env("STRIPE_SECRET_KEY") and self._env("STRIPE_PRO_PRICE_ID")) and getattr(
            self._gateway, "configured", True
        )

    def _trial_days(self) -> int:
        try:
            return int(self._env("STRIPE_TRIAL_DAYS") or "7")
        except ValueError:
            return 7

    def _success_url(self, req: CheckoutSessionRequest) -> str:
        return req.success_url or self._env("STRIPE_SUCCESS_URL") or "http://localhost:3000/billing/success"

    def _cancel_url(self, req: CheckoutSessionRequest) -> str:
        return req.cancel_url or self._env("STRIPE_CANCEL_URL") or "http://localhost:3000/billing/cancel"

    # --- checkout -----------------------------------------------------------
    def create_checkout(self, app_user, req: CheckoutSessionRequest) -> CheckoutSessionResponse:
        if app_user is None:
            return CheckoutSessionResponse(ok=False, error="Sign in required.")
        if not self._configured():
            return CheckoutSessionResponse(ok=False, error="Billing not configured.")
        try:
            existing = self._store.get(app_user.clerk_user_id)
            customer_id = existing.stripe_customer_id if existing and existing.stripe_customer_id else None
            if not customer_id:
                customer_id = self._gateway.create_customer(app_user.clerk_user_id, None)
                # Write the clerk<->customer link NOW so a webhook arriving before the
                # subscription event can still resolve customer -> clerk_user_id.
                self._store.upsert(
                    Subscription(
                        clerk_user_id=app_user.clerk_user_id,
                        stripe_customer_id=customer_id,
                        tier="free",
                        status="none",
                        updated_at=_now(),
                    )
                )
            url = self._gateway.create_checkout_session(
                customer_id=customer_id,
                price_id=self._env("STRIPE_PRO_PRICE_ID"),
                success_url=self._success_url(req),
                cancel_url=self._cancel_url(req),
                trial_days=self._trial_days(),
                clerk_user_id=app_user.clerk_user_id,
            )
            return CheckoutSessionResponse(ok=True, url=url)
        except Exception as exc:
            logger.warning("create_checkout failed: %s", type(exc).__name__)
            return CheckoutSessionResponse(ok=False, error="Checkout could not be started.")

    # --- webhook ------------------------------------------------------------
    def handle_webhook(self, payload: bytes, sig_header: str | None) -> WebhookResponse:
        secret = self._env("STRIPE_WEBHOOK_SECRET")
        if not secret:
            return WebhookResponse(ok=False, handled=False)  # 200-ignore: not configured
        try:
            event = self._gateway.parse_webhook_event(payload, sig_header, secret)
        except BillingSignatureError as exc:
            # Fail closed + operator-visible (shows as a failed delivery in Stripe).
            logger.warning("Stripe webhook signature verification failed: %s", exc)
            raise HTTPException(status_code=400, detail="Invalid webhook signature.")
        etype = event.get("type")
        obj = ((event.get("data") or {}).get("object")) or {}
        handled = self._apply_event(etype, obj)
        return WebhookResponse(ok=True, event_type=etype, handled=handled)

    def _apply_event(self, etype, obj) -> bool:
        if etype in _SUB_EVENTS:
            status = "canceled" if etype.endswith("deleted") else (obj.get("status") or "none")
            self._record(
                clerk=(obj.get("metadata") or {}).get("clerk_user_id"),
                customer_id=obj.get("customer"),
                status=status,
                period_end=obj.get("current_period_end"),
            )
            return True
        if etype == "checkout.session.completed":
            self._record(
                clerk=obj.get("client_reference_id") or (obj.get("metadata") or {}).get("clerk_user_id"),
                customer_id=obj.get("customer"),
                status="trialing",
                period_end=None,
            )
            return True
        return False

    def _record(self, *, clerk, customer_id, status, period_end) -> None:
        if not clerk and customer_id:
            found = self._store.get_by_customer(customer_id)
            clerk = found.clerk_user_id if found else None
        if not clerk:
            logger.warning("Stripe event without resolvable clerk_user_id (customer=%s)", customer_id)
            return
        self._store.upsert(
            Subscription(
                clerk_user_id=clerk,
                stripe_customer_id=customer_id,
                tier=tier_for(status),
                status=status,
                current_period_end=period_end,
                updated_at=_now(),
            )
        )

    # --- read ---------------------------------------------------------------
    def read_subscription(self, app_user) -> SubscriptionResponse:
        if app_user is None:
            return SubscriptionResponse(tier="free", status="none")
        sub = self._store.get(app_user.clerk_user_id)
        if sub is None:
            return SubscriptionResponse(tier="free", status="none")
        return SubscriptionResponse(
            tier=sub.tier, status=sub.status, current_period_end=sub.current_period_end, trial=(sub.status == "trialing")
        )
```

- [ ] **Step 2: Run → PASS**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_billing_service.py -q`

- [ ] **Step 3: Commit C**

```bash
git add api/services/billing_service.py tests/api/test_api_billing_service.py
git commit -m "feat(api): BillingService (checkout/webhook/read) — M2 slice 2C"
```

---

## Commit D — routers + DI + mount + openapi

### Task D1: Write the failing route test

**Files:** Create `tests/api/test_api_billing_routes.py`

- [ ] **Step 1: Write the test**

```python
from starlette.testclient import TestClient

from api.auth import Principal, get_auth_verifier
from api.deps import get_stripe_gateway, get_subscription_store
from api.main import create_app
from api.stores.subscription_store import InMemorySubscriptionStore, Subscription


class _ClerkVerifier:
    def verify(self, authorization):
        return Principal(subject="user_abc", clerk_user_id="user_abc")


class _FakeGateway:
    configured = True

    def create_customer(self, clerk_user_id, email=None):
        return f"cus_{clerk_user_id}"

    def create_checkout_session(self, **kwargs):
        return "https://checkout.stripe.test/s"

    def parse_webhook_event(self, payload, sig_header, secret):
        return {
            "type": "customer.subscription.created",
            "data": {"object": {"customer": "cus_user_abc", "status": "trialing", "metadata": {"clerk_user_id": "user_abc"}}},
        }


def _app(store=None, gateway=None):
    app = create_app()
    app.dependency_overrides[get_auth_verifier] = lambda: _ClerkVerifier()
    app.dependency_overrides[get_subscription_store] = lambda: store or InMemorySubscriptionStore()
    app.dependency_overrides[get_stripe_gateway] = lambda: gateway or _FakeGateway()
    return app


def test_checkout_requires_auth(monkeypatch):
    # No verifier override + no env token → require_principal denies.
    monkeypatch.delenv("HEATER_API_WRITE_TOKEN", raising=False)
    monkeypatch.delenv("CLERK_ISSUER", raising=False)
    app = create_app()
    app.dependency_overrides[get_subscription_store] = lambda: InMemorySubscriptionStore()
    app.dependency_overrides[get_stripe_gateway] = lambda: _FakeGateway()
    resp = TestClient(app).post("/api/billing/checkout-session", json={})
    assert resp.status_code == 401


def test_checkout_returns_url(monkeypatch):
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test")
    monkeypatch.setenv("STRIPE_PRO_PRICE_ID", "price_x")
    resp = TestClient(_app()).post("/api/billing/checkout-session", json={}, headers={"Authorization": "Bearer x"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["url"]


def test_webhook_processes_event(monkeypatch):
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_x")
    store = InMemorySubscriptionStore()
    resp = TestClient(_app(store=store)).post(
        "/api/billing/webhook", content=b"{}", headers={"Stripe-Signature": "sig"}
    )
    assert resp.status_code == 200
    assert resp.json()["handled"] is True
    assert store.get("user_abc").tier == "pro"


def test_subscription_read_default_free():
    resp = TestClient(_app()).get("/api/billing/subscription", headers={"Authorization": "Bearer x"})
    assert resp.status_code == 200
    assert resp.json()["tier"] == "free"


def test_subscription_read_reflects_store():
    store = InMemorySubscriptionStore()
    store.upsert(Subscription(clerk_user_id="user_abc", tier="pro", status="active", updated_at="t"))
    resp = TestClient(_app(store=store)).get("/api/billing/subscription", headers={"Authorization": "Bearer x"})
    assert resp.json()["tier"] == "pro"


def test_openapi_documents_billing_routes():
    schema = create_app().openapi()
    assert "/api/billing/checkout-session" in schema["paths"]
    assert "/api/billing/webhook" in schema["paths"]
    assert "/api/billing/subscription" in schema["paths"]
    # the two authed routes advertise 401; the webhook advertises 400
    assert "401" in schema["paths"]["/api/billing/checkout-session"]["post"]["responses"]
    assert "400" in schema["paths"]["/api/billing/webhook"]["post"]["responses"]
```

- [ ] **Step 2: Run → FAIL** (no billing routes / `get_billing_service` missing)

### Task D2: Implement router + DI + mount

**Files:** Create `api/routers/billing.py`; Modify `api/deps.py`, `api/main.py`

- [ ] **Step 1: Add DI providers to `api/deps.py`** (imports at top with the others; providers with the rest)

```python
# top, with the other imports:
import os

from fastapi import Depends

from api.gateways.stripe_gateway import LiveStripeGateway, NullStripeGateway, StripeGateway
from api.services.billing_service import BillingService
from api.stores.subscription_store import SqliteSubscriptionStore, SubscriptionStore

# providers (append):
def get_stripe_gateway() -> StripeGateway:
    key = os.environ.get("STRIPE_SECRET_KEY", "").strip()
    return LiveStripeGateway(key) if key else NullStripeGateway()


def get_subscription_store() -> SubscriptionStore:
    return SqliteSubscriptionStore()


def get_billing_service(
    gateway: StripeGateway = Depends(get_stripe_gateway),
    store: SubscriptionStore = Depends(get_subscription_store),
) -> BillingService:
    return BillingService(gateway, store)
```

- [ ] **Step 2: Create `api/routers/billing.py`**

```python
"""Billing (Stripe) router. THIN — delegates to BillingService. The two authed
routes gate on require_app_user (a verified Clerk caller + provisioned AppUser);
the webhook is signature-verified inside the service (no Clerk JWT from Stripe)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from api.contracts.billing import (
    CheckoutSessionRequest,
    CheckoutSessionResponse,
    SubscriptionResponse,
    WebhookResponse,
)
from api.deps import get_billing_service
from api.identity import require_app_user

router = APIRouter(prefix="/api/billing", tags=["billing"])

_AUTH_401 = {401: {"description": "Authentication required: missing or invalid bearer token."}}
_SIG_400 = {400: {"description": "Invalid or missing Stripe webhook signature."}}


@router.post("/checkout-session", response_model=CheckoutSessionResponse, responses=_AUTH_401)
def checkout_session(
    req: CheckoutSessionRequest, user=Depends(require_app_user), service=Depends(get_billing_service)
) -> CheckoutSessionResponse:
    return service.create_checkout(user, req)


@router.post("/webhook", response_model=WebhookResponse, responses=_SIG_400)
async def webhook(request: Request, service=Depends(get_billing_service)) -> WebhookResponse:
    payload = await request.body()
    signature = request.headers.get("Stripe-Signature")
    return service.handle_webhook(payload, signature)


@router.get("/subscription", response_model=SubscriptionResponse, responses=_AUTH_401)
def subscription(user=Depends(require_app_user), service=Depends(get_billing_service)) -> SubscriptionResponse:
    return service.read_subscription(user)
```

- [ ] **Step 3: Mount in `api/main.py`** (import with the other routers + include_router)

```python
    from api.routers.billing import router as billing_router
    ...
    app.include_router(billing_router)
```

- [ ] **Step 4: Run → PASS**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_billing_routes.py -q`

### Task D3: Full suite + openapi + lint + commit D

- [ ] **Step 1: Full api suite + lint**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/ -q && python -m ruff check api/ tests/api/`
Expected: PASS, no lint errors.

- [ ] **Step 2: Regen openapi (DOES change — 3 new routes)**

Run: `python scripts/export_openapi.py && C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_openapi_contract.py -q`
Expected: openapi.json updated; the snapshot test passes against the regenerated file.

- [ ] **Step 3: Commit D**

```bash
git add api/routers/billing.py api/deps.py api/main.py api/openapi.json tests/api/test_api_billing_routes.py
git commit -m "feat(api): billing routes (checkout/webhook/subscription) + mount — M2 slice 2D"
```

---

## Self-review checklist (after all 4 commits)

1. **Spec coverage:** checkout-session ✓ (C/D), webhook + signature fail-closed-400 ✓ (C), subscription read ✓ (C/D), SubscriptionStore in api_state.db ✓ (B), StripeGateway seam ✓ (A), tier_for ✓ (C), dormant when unconfigured ✓ (graceful guards), DB-free + network-free tests ✓ (fake gateway + in-memory store), openapi regenerated ✓ (D).
2. **Conventions:** routers thin (delegate only — guarded), services the only src/stripe importers, `api/openapi.json` regenerated, never-raise on read/checkout, fail-closed-400 on webhook signature, separate api_state.db (no live-DB contention).
3. **Type consistency:** `Subscription` fields identical across store + service; `tier_for` used in `_record` + tests; `StripeGateway` methods identical across Live/Null/Fake.

## Review gate
After commit D: dispatch `pr-review-toolkit:code-reviewer` (full context) + `pr-review-toolkit:silent-failure-hunter` over the slice-2 diff — scrutinize the webhook fail-closed path, the customer→clerk resolution, the `create_checkout` broad except, and the SQLite store. Apply findings, then push.

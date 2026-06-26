"""Stripe webhook event-id dedup + ordering guard (pre-billing hardening).

Stripe delivers webhooks at-least-once and can reorder them. The handler must:
  - apply a given event.id exactly once (a redelivery is a no-op),
  - not let an OLDER event.created overwrite newer subscription state,
  - still apply a normal new event,
  - stay byte-for-byte inert when billing env is unset (dormant).

DB-free: uses the InMemory fakes + the existing _FakeGateway style.
"""

import logging

from api.contracts.billing import CheckoutSessionRequest
from api.services.billing_service import BillingService
from api.stores.processed_event_store import InMemoryProcessedEventStore
from api.stores.subscription_store import InMemorySubscriptionStore, Subscription
from api.stores.user_store import AppUser

_USER = AppUser(id=1, clerk_user_id="user_abc", created_at="2026-06-20T00:00:00Z")


class _FakeGateway:
    configured = True

    def __init__(self, event=None):
        self.created_customers = 0
        self.sessions = 0
        self._event = event

    def create_customer(self, clerk_user_id, email=None):
        self.created_customers += 1
        return f"cus_{clerk_user_id}"

    def create_checkout_session(self, **kwargs):
        self.sessions += 1
        return "https://checkout.stripe.test/session"

    def create_portal_session(self, customer_id, return_url):
        return "https://billing.stripe.test/portal"

    def parse_webhook_event(self, payload, sig_header, secret):
        return self._event


def _sub_event(
    etype,
    *,
    event_id,
    created,
    customer="cus_user_abc",
    clerk="user_abc",
    status="trialing",
    period=1800000000,
):
    obj = {"customer": customer, "status": status, "current_period_end": period, "metadata": {"clerk_user_id": clerk}}
    return {"id": event_id, "created": created, "type": etype, "data": {"object": obj}}


def _svc(gw, *, sub_store=None, events=None):
    return BillingService(gw, sub_store or InMemorySubscriptionStore(), events=events)


# --- 1. duplicate event.id applied once ------------------------------------
def test_duplicate_event_id_is_applied_once(monkeypatch):
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_x")
    sub = InMemorySubscriptionStore()
    events = InMemoryProcessedEventStore()
    gw = _FakeGateway(
        event=_sub_event("customer.subscription.updated", event_id="evt_1", created=1000, status="active")
    )
    svc = _svc(gw, sub_store=sub, events=events)

    r1 = svc.handle_webhook(b"{}", "sig")
    assert r1.ok and r1.handled
    assert sub.get("user_abc").status == "active"
    assert events.was_processed("evt_1") is True

    # Mutate the stored sub to a sentinel, then redeliver the SAME event id.
    sub.upsert(
        Subscription(
            clerk_user_id="user_abc", stripe_customer_id="cus_user_abc", tier="free", status="SENTINEL", updated_at="t"
        )
    )
    r2 = svc.handle_webhook(b"{}", "sig")
    assert r2.ok is True
    assert r2.handled is False  # redelivery wrote no NEW state
    # State must be untouched by the redelivery (still the sentinel).
    assert sub.get("user_abc").status == "SENTINEL"


# --- 2. out-of-order OLDER event must not overwrite newer state ------------
def test_older_event_created_does_not_overwrite_newer_state(monkeypatch):
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_x")
    sub = InMemorySubscriptionStore()
    events = InMemoryProcessedEventStore()

    # Newest event first: subscription becomes canceled at created=2000.
    gw_new = _FakeGateway(
        event=_sub_event("customer.subscription.deleted", event_id="evt_new", created=2000, status="active")
    )
    _svc(gw_new, sub_store=sub, events=events).handle_webhook(b"{}", "sig")
    assert sub.get("user_abc").status == "canceled"
    assert sub.get("user_abc").tier == "free"

    # Now an OLDER reactivation arrives (created=1000) — must NOT re-activate.
    gw_old = _FakeGateway(
        event=_sub_event("customer.subscription.updated", event_id="evt_old", created=1000, status="active")
    )
    resp = _svc(gw_old, sub_store=sub, events=events).handle_webhook(b"{}", "sig")
    assert resp.handled is False  # stale event applied no state
    assert sub.get("user_abc").status == "canceled"  # still canceled
    assert sub.get("user_abc").tier == "free"
    # The stale event is recorded so a redelivery of it is also a no-op.
    assert events.was_processed("evt_old") is True


def test_checkout_completed_does_not_stale_block_later_subscription_event(monkeypatch):
    """A link-only checkout.session.completed with a HIGH created must NOT advance the
    ordering watermark — otherwise a later subscription event carrying an earlier
    event.created would be wrongly judged stale and dropped (it records created=None)."""
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_x")
    sub = InMemorySubscriptionStore()
    events = InMemoryProcessedEventStore()

    # 1) checkout.session.completed at created=5000 (link-only, no subscription status).
    checkout = {
        "id": "evt_checkout",
        "created": 5000,
        "type": "checkout.session.completed",
        "data": {"object": {"customer": "cus_user_abc", "client_reference_id": "user_abc"}},
    }
    _svc(_FakeGateway(event=checkout), sub_store=sub, events=events).handle_webhook(b"{}", "sig")

    # 2) a subscription.updated at created=2000 (EARLIER than the checkout) must STILL apply.
    sub_evt = _sub_event("customer.subscription.updated", event_id="evt_sub", created=2000, status="active")
    resp = _svc(_FakeGateway(event=sub_evt), sub_store=sub, events=events).handle_webhook(b"{}", "sig")
    assert resp.handled is True  # not stale-blocked by the checkout's higher created
    assert sub.get("user_abc").status == "active"


# --- 3. a normal new event applies correctly -------------------------------
def test_normal_new_event_applies(monkeypatch):
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_x")
    sub = InMemorySubscriptionStore()
    events = InMemoryProcessedEventStore()
    gw = _FakeGateway(
        event=_sub_event("customer.subscription.created", event_id="evt_a", created=1000, status="trialing")
    )
    resp = _svc(gw, sub_store=sub, events=events).handle_webhook(b"{}", "sig")
    assert resp.ok and resp.handled
    assert sub.get("user_abc").tier == "pro"
    assert sub.get("user_abc").status == "trialing"
    assert events.was_processed("evt_a") is True


def test_newer_event_after_older_still_applies(monkeypatch):
    # Ordering guard must NOT block a genuinely-newer event.
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_x")
    sub = InMemorySubscriptionStore()
    events = InMemoryProcessedEventStore()
    gw1 = _FakeGateway(
        event=_sub_event("customer.subscription.created", event_id="evt_1", created=1000, status="trialing")
    )
    _svc(gw1, sub_store=sub, events=events).handle_webhook(b"{}", "sig")
    gw2 = _FakeGateway(
        event=_sub_event("customer.subscription.updated", event_id="evt_2", created=2000, status="active")
    )
    resp = _svc(gw2, sub_store=sub, events=events).handle_webhook(b"{}", "sig")
    assert resp.handled is True
    assert sub.get("user_abc").status == "active"


# --- 4. dormant: billing env unset → handler stays inert -------------------
def test_dormant_handler_does_not_touch_stores_when_unconfigured(monkeypatch):
    monkeypatch.delenv("STRIPE_WEBHOOK_SECRET", raising=False)
    monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
    sub = InMemorySubscriptionStore()
    events = InMemoryProcessedEventStore()
    gw = _FakeGateway(event=_sub_event("customer.subscription.created", event_id="evt_x", created=1, status="active"))
    resp = _svc(gw, sub_store=sub, events=events).handle_webhook(b"{}", "sig")
    assert resp.ok is False
    assert resp.handled is False
    # No event recorded, no subscription written.
    assert events.was_processed("evt_x") is False
    assert sub.get("user_abc") is None


# --- backward-compat: events store is optional -----------------------------
def test_handler_works_without_event_store(monkeypatch):
    # Constructing BillingService with the legacy 2-arg signature must still work
    # (events=None → dedup/ordering simply not enforced, today's behavior).
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_x")
    sub = InMemorySubscriptionStore()
    gw = _FakeGateway(
        event=_sub_event("customer.subscription.created", event_id="evt_a", created=1000, status="active")
    )
    resp = BillingService(gw, sub).handle_webhook(b"{}", "sig")
    assert resp.ok and resp.handled
    assert sub.get("user_abc").status == "active"


# --- a non-state event (checkout.session.completed) is NOT order-guarded ----
def test_checkout_completed_recorded_but_not_order_guarded(monkeypatch):
    # checkout.session.completed is link-only; it must still be deduped by id,
    # but it carries no subscription status to order-guard.
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_x")
    sub = InMemorySubscriptionStore()
    events = InMemoryProcessedEventStore()
    obj = {"client_reference_id": "user_abc", "customer": "cus_user_abc"}
    evt = {"id": "evt_co", "created": 500, "type": "checkout.session.completed", "data": {"object": obj}}
    gw = _FakeGateway(event=evt)
    r1 = _svc(gw, sub_store=sub, events=events).handle_webhook(b"{}", "sig")
    assert r1.handled is True
    assert events.was_processed("evt_co") is True
    # Redelivery is a no-op.
    r2 = _svc(gw, sub_store=sub, events=events).handle_webhook(b"{}", "sig")
    assert r2.handled is False


class TestProcessedEventStoreSqlite:
    """The Sqlite impl owns its table in api_state.db (never draft_tool.db) and
    is created idempotently — mirrors the other api_state stores. DB-free via a
    tmp_path db file."""

    def _store(self, tmp_path, monkeypatch):
        from api.stores.processed_event_store import SqliteProcessedEventStore

        db = tmp_path / "api_state.db"
        monkeypatch.setenv("HEATER_API_DB_PATH", str(db))
        return SqliteProcessedEventStore(str(db))

    def test_record_then_was_processed(self, tmp_path, monkeypatch):
        store = self._store(tmp_path, monkeypatch)
        assert store.was_processed("evt_1") is False
        store.record("evt_1", event_created=1000, customer_id="cus_1", subscription_id="sub_1")
        assert store.was_processed("evt_1") is True

    def test_last_applied_created_returns_max(self, tmp_path, monkeypatch):
        store = self._store(tmp_path, monkeypatch)
        assert store.last_applied_created("cus_1") is None
        store.record("evt_1", event_created=1000, customer_id="cus_1", subscription_id=None)
        store.record("evt_2", event_created=3000, customer_id="cus_1", subscription_id=None)
        store.record("evt_3", event_created=2000, customer_id="cus_1", subscription_id=None)
        assert store.last_applied_created("cus_1") == 3000
        # Scoped per-customer.
        assert store.last_applied_created("cus_other") is None

    def test_record_is_idempotent(self, tmp_path, monkeypatch):
        store = self._store(tmp_path, monkeypatch)
        store.record("evt_1", event_created=1000, customer_id="cus_1", subscription_id=None)
        # Re-recording the same id must not raise (ON CONFLICT no-op).
        store.record("evt_1", event_created=9999, customer_id="cus_1", subscription_id=None)
        assert store.was_processed("evt_1") is True
        # First write wins (event is immutable once processed).
        assert store.last_applied_created("cus_1") == 1000

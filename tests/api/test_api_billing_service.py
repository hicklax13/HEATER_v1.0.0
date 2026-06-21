import logging

import pytest
from fastapi import HTTPException

from api.contracts.billing import CheckoutSessionRequest
from api.gateways.stripe_gateway import BillingSignatureError
from api.services.billing_service import BillingService, tier_for
from api.stores.subscription_store import InMemorySubscriptionStore, Subscription
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


def test_read_subscription_reflects_store():
    store = InMemorySubscriptionStore()
    store.upsert(Subscription(clerk_user_id="user_abc", tier="pro", status="trialing", updated_at="t"))
    resp = BillingService(_FakeGateway(), store).read_subscription(_USER)
    assert resp.tier == "pro"
    assert resp.trial is True

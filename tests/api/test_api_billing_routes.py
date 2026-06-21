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

    def create_portal_session(self, customer_id, return_url):
        return "https://billing.stripe.test/p"

    def parse_webhook_event(self, payload, sig_header, secret):
        return {
            "type": "customer.subscription.created",
            "data": {
                "object": {"customer": "cus_user_abc", "status": "trialing", "metadata": {"clerk_user_id": "user_abc"}}
            },
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


def test_portal_session_requires_auth(monkeypatch):
    monkeypatch.delenv("HEATER_API_WRITE_TOKEN", raising=False)
    monkeypatch.delenv("CLERK_ISSUER", raising=False)
    app = create_app()
    app.dependency_overrides[get_subscription_store] = lambda: InMemorySubscriptionStore()
    app.dependency_overrides[get_stripe_gateway] = lambda: _FakeGateway()
    resp = TestClient(app).post("/api/billing/portal-session", json={})
    assert resp.status_code == 401


def test_portal_session_returns_url(monkeypatch):
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test")
    monkeypatch.setenv("STRIPE_PRO_PRICE_ID", "price_x")
    store = InMemorySubscriptionStore()
    store.upsert(
        Subscription(
            clerk_user_id="user_abc", stripe_customer_id="cus_user_abc", tier="pro", status="active", updated_at="t"
        )
    )
    resp = TestClient(_app(store=store)).post(
        "/api/billing/portal-session", json={}, headers={"Authorization": "Bearer x"}
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert resp.json()["url"]


def test_openapi_documents_billing_routes():
    schema = create_app().openapi()
    assert "/api/billing/checkout-session" in schema["paths"]
    assert "/api/billing/webhook" in schema["paths"]
    assert "/api/billing/subscription" in schema["paths"]
    assert "/api/billing/portal-session" in schema["paths"]
    # the authed routes advertise 401; the webhook advertises 400
    assert "401" in schema["paths"]["/api/billing/checkout-session"]["post"]["responses"]
    assert "401" in schema["paths"]["/api/billing/portal-session"]["post"]["responses"]
    assert "400" in schema["paths"]["/api/billing/webhook"]["post"]["responses"]

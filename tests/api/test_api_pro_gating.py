import pytest
from fastapi import HTTPException

from api.auth import Principal
from api.gating import require_pro, stripe_enabled
from api.stores.subscription_store import InMemorySubscriptionStore, Subscription


class _Verifier:
    def __init__(self, principal=None, raise_401=False):
        self._principal = principal
        self._raise_401 = raise_401

    def verify(self, authorization):
        if self._raise_401:
            raise HTTPException(status_code=401, detail="no")
        return self._principal


def _pro_store(clerk="user_abc"):
    s = InMemorySubscriptionStore()
    s.upsert(Subscription(clerk_user_id=clerk, tier="pro", status="active", updated_at="t"))
    return s


def test_dormant_when_stripe_unset(monkeypatch):
    # No Stripe → no-op: returns without touching the verifier or store (endpoint open).
    monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
    assert stripe_enabled() is False
    # verifier set to raise — proving require_pro never calls it when dormant
    require_pro(authorization=None, verifier=_Verifier(raise_401=True), store=InMemorySubscriptionStore())


def test_live_allows_pro(monkeypatch):
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test")
    v = _Verifier(principal=Principal(subject="user_abc", clerk_user_id="user_abc"))
    require_pro(authorization="Bearer x", verifier=v, store=_pro_store())  # no raise


def test_live_blocks_free_with_402(monkeypatch):
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test")
    v = _Verifier(principal=Principal(subject="user_free", clerk_user_id="user_free"))
    with pytest.raises(HTTPException) as ei:
        require_pro(authorization="Bearer x", verifier=v, store=InMemorySubscriptionStore())
    assert ei.value.status_code == 402


def test_live_blocks_non_clerk_principal_with_402(monkeypatch):
    # billing live but caller is the env-token/server path (no clerk id) → not Pro.
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test")
    v = _Verifier(principal=Principal(subject="api-token"))
    with pytest.raises(HTTPException) as ei:
        require_pro(authorization="Bearer x", verifier=v, store=InMemorySubscriptionStore())
    assert ei.value.status_code == 402


def test_live_propagates_401_when_unauthenticated(monkeypatch):
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test")
    with pytest.raises(HTTPException) as ei:
        require_pro(authorization=None, verifier=_Verifier(raise_401=True), store=InMemorySubscriptionStore())
    assert ei.value.status_code == 401

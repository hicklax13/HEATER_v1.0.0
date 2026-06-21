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


# --- structural guard: the compute-heavy endpoints must carry the gate ----------
# Introspect the ROUTER objects directly (their routes are materialized with the full
# prefixed path + .dependant). app.routes can't be walked here: FastAPI 0.137's lazy
# include_router stores _IncludedRouter wrappers whose nested APIRoutes aren't expanded.

from api.routers.draft import router as _draft_router  # noqa: E402
from api.routers.lineup import router as _lineup_router  # noqa: E402
from api.routers.playoff import router as _playoff_router  # noqa: E402
from api.routers.standings import router as _standings_router  # noqa: E402
from api.routers.trade import router as _trade_router  # noqa: E402
from api.routers.trade_finder import router as _trade_finder_router  # noqa: E402

# The canonical compute-heavy endpoints. Adding a new heavy endpoint? Add it here AND
# its router below AND gate it with dependencies=[Depends(require_pro)] — this fails until you do.
_HEAVY = {
    ("POST", "/api/lineup/optimize"),
    ("POST", "/api/trade/evaluate"),
    ("GET", "/api/playoff-odds"),
    ("GET", "/api/trade-finder"),
    ("POST", "/api/draft/recommend"),
    ("POST", "/api/draft/simulate-picks"),
}

_GATED_ROUTERS = (_lineup_router, _trade_router, _playoff_router, _trade_finder_router, _draft_router)


def _route_map(*routers):
    out = {}
    for rtr in routers:
        for r in rtr.routes:
            path = getattr(r, "path", None)
            for m in getattr(r, "methods", None) or set():
                out[(m, path)] = r
    return out


def _dep_calls(route):
    return [d.call for d in route.dependant.dependencies]


def test_heavy_endpoints_carry_require_pro():
    routes = _route_map(*_GATED_ROUTERS)
    for key in _HEAVY:
        route = routes.get(key)
        assert route is not None, f"{key} not found on its router"
        assert require_pro in _dep_calls(route), f"{key} is missing the require_pro gate"


def test_read_endpoint_is_not_gated():
    # A free read endpoint must NOT carry the Pro gate.
    routes = _route_map(_standings_router)
    route = routes.get(("GET", "/api/standings"))
    assert route is not None
    assert require_pro not in _dep_calls(route)

# M2 Slice 3 — Feature-Gating (the Pro tier gate) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A single reusable `require_pro` dependency that paywalls the compute-heavy endpoints (hard gate — Connor's choice), DORMANT until Stripe is configured, structurally guarded so the gate can't silently fall off a heavy route.

**Architecture:** `require_pro` (in new `api/gating.py`) resolves the caller's tier from the `SubscriptionStore` and 402s non-Pro callers — but is a **no-op when Stripe is unconfigured** (no way to be Pro → gating would lock everyone out, and it must not break the open read endpoints the frontend wires on current infra). Applied route-level via `dependencies=[Depends(require_pro)]` to the 6 heavy endpoints. A structural test asserts each carries the gate.

**Tech Stack:** FastAPI dependency, the slice-1 auth verifier + slice-2 SubscriptionStore.

---

## Context an executor needs

- The gate MUST be dormant when `STRIPE_SECRET_KEY` is unset: the live CMO frontend (M1) calls these endpoints with NO auth on current infra. If `require_pro` enforced auth unconditionally it would 401 every M1 call and break the beta. So: Stripe unset → `require_pro` returns immediately (endpoint stays exactly as today).
- Do NOT build on `require_app_user` (it hard-requires a principal via `require_principal` → would 401 the open deployment). `require_pro` resolves the principal itself, ONLY after confirming billing is live.
- The 6 heavy routes + methods: `POST /api/lineup/optimize`, `POST /api/trade/evaluate`, `GET /api/playoff-odds`, `GET /api/trade-finder`, `POST /api/draft/recommend`, `POST /api/draft/simulate-picks`.
- Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/ -q`. Lint: `python -m ruff check api/ tests/api/`. Regen openapi (DOES change — the 6 routes gain a 402): `python scripts/export_openapi.py`.
- Product note: hard gate = the WHOLE heavy endpoint is Pro (Connor chose "heavy decision tools are Pro-only"; free = the data/read pages). Per-mode freemium (free standard optimize / Pro daily mode) is a documented FUTURE refinement, not this slice.

---

## Commit A — the `require_pro` dependency + unit tests

### Task A1: Write the failing unit test

**Files:** Create `tests/api/test_api_pro_gating.py`

- [ ] **Step 1: Write the dependency unit tests** (the structural-guard test comes in Commit B)

```python
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
```

- [ ] **Step 2: Run → FAIL** (`ModuleNotFoundError: api.gating`)

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_pro_gating.py -q`

### Task A2: Implement the gate

**Files:** Create `api/gating.py`

- [ ] **Step 1: Write it**

```python
"""Pro-tier gate — the ONE reusable dependency that paywalls the compute-heavy
endpoints. DORMANT when Stripe is unconfigured (no way to be Pro → gating would
lock everyone out), so it is a no-op on the current pre-billing deployment and
never breaks the open read endpoints the frontend wires today. Once
STRIPE_SECRET_KEY is set it enforces a verified Clerk caller (401) who is Pro (402)."""

from __future__ import annotations

import os

from fastapi import Depends, Header, HTTPException, status

from api.auth import AuthVerifier, get_auth_verifier
from api.deps import get_subscription_store
from api.stores.subscription_store import SubscriptionStore


def stripe_enabled() -> bool:
    """Billing is live (users CAN be Pro) when the Stripe secret key is set."""
    return bool(os.environ.get("STRIPE_SECRET_KEY", "").strip())


def _payment_required() -> HTTPException:
    return HTTPException(status_code=status.HTTP_402_PAYMENT_REQUIRED, detail="Pro subscription required.")


def require_pro(
    authorization: str | None = Header(default=None),
    verifier: AuthVerifier = Depends(get_auth_verifier),
    store: SubscriptionStore = Depends(get_subscription_store),
) -> None:
    """Route dependency: allow only Pro subscribers. No-op until Stripe is configured
    (so the current open deployment + the frontend's read wiring are unaffected). When
    billing is live it requires a verified Clerk caller (401) who is Pro (else 402)."""
    if not stripe_enabled():
        return  # dormant: no billing → the endpoint stays open exactly as today
    principal = verifier.verify(authorization)  # raises 401 if missing/invalid (billing live)
    if not principal.clerk_user_id:
        raise _payment_required()  # env-token/server path is not a paying user
    sub = store.get(principal.clerk_user_id)
    if (sub.tier if sub else "free") != "pro":
        raise _payment_required()
```

- [ ] **Step 2: Run → PASS**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_pro_gating.py -q`

- [ ] **Step 3: Commit A**

```bash
git add api/gating.py tests/api/test_api_pro_gating.py
git commit -m "feat(api): require_pro tier gate (dormant until Stripe configured) — M2 slice 3A"
```

---

## Commit B — apply the gate to the 6 heavy routes + structural guard

### Task B1: Add the structural guard (failing — routes not gated yet)

**Files:** Modify `tests/api/test_api_pro_gating.py` (append)

- [ ] **Step 1: Append the guard test**

```python
from api.gating import require_pro as _require_pro  # noqa: E402  (re-import for the guard section)
from api.main import create_app  # noqa: E402

# The canonical compute-heavy endpoints. Adding a new heavy endpoint? Add it here AND
# gate it with dependencies=[Depends(require_pro)] — this test fails until you do.
_HEAVY = {
    ("POST", "/api/lineup/optimize"),
    ("POST", "/api/trade/evaluate"),
    ("GET", "/api/playoff-odds"),
    ("GET", "/api/trade-finder"),
    ("POST", "/api/draft/recommend"),
    ("POST", "/api/draft/simulate-picks"),
}


def _flatten_calls(dependant):
    calls = []
    for dep in dependant.dependencies:
        calls.append(dep.call)
        calls.extend(_flatten_calls(dep))
    return calls


def test_heavy_endpoints_carry_require_pro():
    app = create_app()
    routes = {}
    for route in app.routes:
        methods = getattr(route, "methods", None) or set()
        path = getattr(route, "path", None)
        for m in methods:
            routes[(m, path)] = route
    for key in _HEAVY:
        route = routes.get(key)
        assert route is not None, f"{key} not mounted"
        assert _require_pro in _flatten_calls(route.dependant), f"{key} is missing the require_pro gate"


def test_read_endpoint_is_not_gated():
    # A free read endpoint must NOT carry the Pro gate.
    app = create_app()
    for route in app.routes:
        if getattr(route, "path", None) == "/api/standings":
            assert _require_pro not in _flatten_calls(route.dependant)
            return
    raise AssertionError("/api/standings not found")
```

- [ ] **Step 2: Run → FAIL** (`test_heavy_endpoints_carry_require_pro` — routes not yet gated)

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_pro_gating.py::test_heavy_endpoints_carry_require_pro -q`

### Task B2: Apply `require_pro` to the 6 routes

For EACH router below: import `require_pro`, add `dependencies=[Depends(require_pro)]` + `responses=_PRO_GATE` to the heavy decorator. `_PRO_GATE` documents the 402 (and the 401 billing-live auth).

- [ ] **Step 1: `api/routers/lineup.py`** — replace the decorator + add the import

```python
from api.gating import require_pro
```
and change the route to:
```python
_PRO_GATE = {
    402: {"description": "Pro subscription required (when billing is enabled)."},
    401: {"description": "Authentication required (when billing is enabled)."},
}


@router.post(
    "/lineup/optimize",
    response_model=LineupOptimizeResponse,
    dependencies=[Depends(require_pro)],
    responses=_PRO_GATE,
)
def optimize_lineup(req: LineupOptimizeRequest, service=Depends(get_lineup_service)) -> LineupOptimizeResponse:
    return service.optimize(req.team_name, req.date, req.scope, req.mode)
```

- [ ] **Step 2: `api/routers/trade.py`** — add `from api.gating import require_pro`, the same `_PRO_GATE` dict, and:
```python
@router.post(
    "/trade/evaluate",
    response_model=TradeEvaluationResponse,
    dependencies=[Depends(require_pro)],
    responses=_PRO_GATE,
)
def evaluate_trade_endpoint(req: TradeEvaluateRequest, service=Depends(get_trade_service)) -> TradeEvaluationResponse:
    return service.evaluate(req.team_name, req.giving_ids, req.receiving_ids, req.enable_mc)
```

- [ ] **Step 3: `api/routers/playoff.py`** — add `from api.gating import require_pro`, `_PRO_GATE`, and:
```python
@router.get(
    "/playoff-odds",
    response_model=PlayoffOddsResponse,
    dependencies=[Depends(require_pro)],
    responses=_PRO_GATE,
)
def get_playoff_odds(team_name: str, service=Depends(get_playoff_service)) -> PlayoffOddsResponse:
    return service.get_playoff_odds(team_name)
```

- [ ] **Step 4: `api/routers/trade_finder.py`** — add `from api.gating import require_pro`, `_PRO_GATE`, and:
```python
@router.get(
    "/trade-finder",
    response_model=TradeFinderResponse,
    dependencies=[Depends(require_pro)],
    responses=_PRO_GATE,
)
def get_trade_finder(
    team_name: str = "",
    limit: int = 10,
    service=Depends(get_trade_finder_service),
) -> TradeFinderResponse:
    return service.get_suggestions(team_name=team_name, limit=limit)
```

- [ ] **Step 5: `api/routers/draft.py`** — add `from api.gating import require_pro`, `_PRO_GATE`, and add `dependencies=[Depends(require_pro)], responses=_PRO_GATE` to BOTH the `/draft/recommend` and `/draft/simulate-picks` decorators:
```python
@router.post(
    "/draft/recommend",
    response_model=DraftRecommendResponse,
    dependencies=[Depends(require_pro)],
    responses=_PRO_GATE,
)
def draft_recommend(req: DraftRecommendRequest, service=Depends(get_draft_service)) -> DraftRecommendResponse:
    return service.recommend(req)


@router.post(
    "/draft/simulate-picks",
    response_model=DraftSimulatePicksResponse,
    dependencies=[Depends(require_pro)],
    responses=_PRO_GATE,
)
def draft_simulate_picks(
    req: DraftSimulatePicksRequest, service=Depends(get_draft_service)
) -> DraftSimulatePicksResponse:
    return service.simulate_picks(req)
```
(`Depends` is already imported in each router. `_PRO_GATE` is defined once per router file near the top.)

- [ ] **Step 6: Run the guard + full api suite**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_pro_gating.py tests/api/ -q`
Expected: PASS. (The existing heavy-endpoint tests stay green because the gate is dormant — STRIPE_SECRET_KEY is unset in the test env.)

### Task B3: openapi + lint + commit B

- [ ] **Step 1: Regen openapi (DOES change — the 6 routes gain 402/401)**

Run: `python scripts/export_openapi.py && C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_openapi_contract.py -q`

- [ ] **Step 2: Lint**

Run: `python -m ruff check api/ tests/api/`

- [ ] **Step 3: Commit B**

```bash
git add api/routers/lineup.py api/routers/trade.py api/routers/playoff.py api/routers/trade_finder.py api/routers/draft.py api/openapi.json tests/api/test_api_pro_gating.py
git commit -m "feat(api): gate the 6 compute-heavy endpoints behind require_pro + structural guard — M2 slice 3B"
```

---

## Self-review checklist

1. **Spec coverage:** single reusable `require_pro` ✓ (A), applied to the 6 heavy endpoints ✓ (B), structural guard so a new heavy endpoint can't ship ungated ✓ (B1), dormant until Stripe configured ✓ (A — `stripe_enabled` no-op), 402 for Free / 401 for unauth (billing live) ✓.
2. **Conventions:** routers stay thin (a dependency, not logic), openapi regenerated, no `src/` change, no `require_app_user` coupling (dormancy preserved), the live read endpoints untouched (`test_read_endpoint_is_not_gated`).
3. **Doesn't break M1:** STRIPE_SECRET_KEY unset on current infra → gate is a no-op → the CMO's frontend calls to these endpoints still work.

## Review gate
After commit B: dispatch `pr-review-toolkit:code-reviewer` over the slice-3 diff — confirm the dormancy logic can't fall open/closed wrong, the 402/401 split, and that no read endpoint got gated. Apply findings, then push. This COMPLETES M2 (auth + billing + gating).

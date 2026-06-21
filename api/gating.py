"""Pro-tier gate — the ONE reusable dependency that paywalls the compute-heavy
endpoints. DORMANT when Stripe is unconfigured (no way to be Pro → gating would
lock everyone out), so it is a no-op on the current pre-billing deployment and
never breaks the open read endpoints the frontend wires today. Once
STRIPE_SECRET_KEY is set it enforces a verified Clerk caller (401) who is Pro (402)."""

from __future__ import annotations

from fastapi import Depends, Header, HTTPException, status

from api.auth import AuthVerifier, get_auth_verifier
from api.billing_config import billing_env_configured
from api.deps import get_subscription_store
from api.stores.subscription_store import SubscriptionStore


def stripe_enabled() -> bool:
    """Billing is live (users CAN be Pro) ONLY when checkout is actually operable —
    the SAME predicate the billing service uses (secret key AND price id). Gating on
    just the secret key would paywall users during a one-var-at-a-time deploy while
    checkout still returns 'not configured' — locking them out with no upgrade path."""
    return billing_env_configured()


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

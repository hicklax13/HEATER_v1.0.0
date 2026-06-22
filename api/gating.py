"""Pro-tier gate — the ONE reusable dependency that paywalls the compute-heavy
endpoints. DORMANT when Stripe is unconfigured (no way to be Pro → gating would
lock everyone out), so it is a no-op on the current pre-billing deployment and
never breaks the open read endpoints the frontend wires today. Once
STRIPE_SECRET_KEY is set it enforces a verified Clerk caller (401) who is Pro (402)."""

from __future__ import annotations

import logging

from fastapi import Depends, Header, HTTPException, status

from api.auth import AuthVerifier, get_auth_verifier
from api.billing_config import billing_env_configured
from api.deps import get_subscription_store
from api.services.ai_allowance import managed_cap_for_tier
from api.stores.subscription_store import SubscriptionStore

logger = logging.getLogger(__name__)


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


def get_managed_ai_cap(
    authorization: str | None = Header(default=None),
    verifier: AuthVerifier = Depends(get_auth_verifier),
    store: SubscriptionStore = Depends(get_subscription_store),
) -> float | None:
    """The caller's DAILY managed-AI cap (shared-key spend), or None while billing
    is dormant -> budget.is_over_cap falls back to the admin cap (today's behavior).
    Never raises: an unverifiable caller / store error -> None (safe admin-cap
    fallback; the chat route's own require_app_user owns the 401).

    Degraded-path note: a subscription-read failure returns None, so the caller
    gets the ADMIN cap, which may EXCEED a free user's tier cap (≈$1.00 vs $0.10) —
    a bounded, WARNING-logged over-grant during an api_state.db outage. This favors
    availability (keep chatting) over hard-blocking, and the blast radius is at most
    (admin_cap − tier_cap) per user per day."""
    if not stripe_enabled():
        return None
    try:
        clerk = verifier.verify(authorization).clerk_user_id
    except Exception:
        return None  # unauthenticated/invalid — the route 401s independently
    if not clerk:
        return managed_cap_for_tier("free")
    try:
        sub = store.get(clerk)
    except Exception:
        logger.warning("get_managed_ai_cap subscription read failed", exc_info=True)
        return None
    return managed_cap_for_tier(sub.tier if sub else "free")

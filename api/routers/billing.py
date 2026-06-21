"""Billing (Stripe) router. THIN — delegates to BillingService. The two authed
routes gate on require_app_user (a verified Clerk caller + provisioned AppUser);
the webhook is signature-verified inside the service (no Clerk JWT from Stripe)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from api.contracts.billing import (
    CheckoutSessionRequest,
    CheckoutSessionResponse,
    PortalSessionRequest,
    PortalSessionResponse,
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


@router.post("/portal-session", response_model=PortalSessionResponse, responses=_AUTH_401)
def portal_session(
    req: PortalSessionRequest, user=Depends(require_app_user), service=Depends(get_billing_service)
) -> PortalSessionResponse:
    return service.create_portal_session(user, req)


@router.get("/subscription", response_model=SubscriptionResponse, responses=_AUTH_401)
def subscription(user=Depends(require_app_user), service=Depends(get_billing_service)) -> SubscriptionResponse:
    return service.read_subscription(user)

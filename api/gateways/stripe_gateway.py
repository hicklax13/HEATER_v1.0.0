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

"""Billing logic seam — the one place that orchestrates the StripeGateway + the
SubscriptionStore. Never raises on the read/checkout paths (graceful ok/error);
the webhook fails CLOSED on a bad signature (HTTP 400) so Stripe + the operator
see a rejected delivery."""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime

from fastapi import HTTPException

from api.contracts.billing import CheckoutSessionRequest, CheckoutSessionResponse, SubscriptionResponse, WebhookResponse
from api.gateways.stripe_gateway import BillingSignatureError, StripeGateway
from api.stores.subscription_store import Subscription, SubscriptionStore

logger = logging.getLogger(__name__)

_SUB_EVENTS = {"customer.subscription.created", "customer.subscription.updated", "customer.subscription.deleted"}


def tier_for(status: str) -> str:
    """The contract slice 3 consumes: only an active or trialing subscription is Pro."""
    return "pro" if status in {"trialing", "active"} else "free"


def _now() -> str:
    return datetime.now(UTC).isoformat()


class BillingService:
    def __init__(self, gateway: StripeGateway, store: SubscriptionStore) -> None:
        self._gateway = gateway
        self._store = store

    # --- config (read at call time, like get_auth_verifier) -----------------
    @staticmethod
    def _env(name: str) -> str:
        return os.environ.get(name, "").strip()

    def _configured(self) -> bool:
        return bool(self._env("STRIPE_SECRET_KEY") and self._env("STRIPE_PRO_PRICE_ID")) and getattr(
            self._gateway, "configured", True
        )

    def _trial_days(self) -> int:
        try:
            return int(self._env("STRIPE_TRIAL_DAYS") or "7")
        except ValueError:
            return 7

    def _success_url(self, req: CheckoutSessionRequest) -> str:
        return req.success_url or self._env("STRIPE_SUCCESS_URL") or "http://localhost:3000/billing/success"

    def _cancel_url(self, req: CheckoutSessionRequest) -> str:
        return req.cancel_url or self._env("STRIPE_CANCEL_URL") or "http://localhost:3000/billing/cancel"

    # --- checkout -----------------------------------------------------------
    def create_checkout(self, app_user, req: CheckoutSessionRequest) -> CheckoutSessionResponse:
        if app_user is None:
            return CheckoutSessionResponse(ok=False, error="Sign in required.")
        if not self._configured():
            return CheckoutSessionResponse(ok=False, error="Billing not configured.")
        try:
            existing = self._store.get(app_user.clerk_user_id)
            customer_id = existing.stripe_customer_id if existing and existing.stripe_customer_id else None
            if not customer_id:
                customer_id = self._gateway.create_customer(app_user.clerk_user_id, None)
                # Write the clerk<->customer link NOW so a webhook arriving before the
                # subscription event can still resolve customer -> clerk_user_id.
                self._store.upsert(
                    Subscription(
                        clerk_user_id=app_user.clerk_user_id,
                        stripe_customer_id=customer_id,
                        tier="free",
                        status="none",
                        updated_at=_now(),
                    )
                )
            url = self._gateway.create_checkout_session(
                customer_id=customer_id,
                price_id=self._env("STRIPE_PRO_PRICE_ID"),
                success_url=self._success_url(req),
                cancel_url=self._cancel_url(req),
                trial_days=self._trial_days(),
                clerk_user_id=app_user.clerk_user_id,
            )
            return CheckoutSessionResponse(ok=True, url=url)
        except Exception as exc:
            logger.warning("create_checkout failed: %s", type(exc).__name__)
            return CheckoutSessionResponse(ok=False, error="Checkout could not be started.")

    # --- webhook ------------------------------------------------------------
    def handle_webhook(self, payload: bytes, sig_header: str | None) -> WebhookResponse:
        secret = self._env("STRIPE_WEBHOOK_SECRET")
        if not secret:
            return WebhookResponse(ok=False, handled=False)  # 200-ignore: not configured
        try:
            event = self._gateway.parse_webhook_event(payload, sig_header, secret)
        except BillingSignatureError as exc:
            # Fail closed + operator-visible (shows as a failed delivery in Stripe).
            logger.warning("Stripe webhook signature verification failed: %s", exc)
            raise HTTPException(status_code=400, detail="Invalid webhook signature.")
        etype = event.get("type")
        obj = ((event.get("data") or {}).get("object")) or {}
        handled = self._apply_event(etype, obj)
        return WebhookResponse(ok=True, event_type=etype, handled=handled)

    def _apply_event(self, etype, obj) -> bool:
        if etype in _SUB_EVENTS:
            status = "canceled" if etype.endswith("deleted") else (obj.get("status") or "none")
            self._record(
                clerk=(obj.get("metadata") or {}).get("clerk_user_id"),
                customer_id=obj.get("customer"),
                status=status,
                period_end=obj.get("current_period_end"),
            )
            return True
        if etype == "checkout.session.completed":
            self._record(
                clerk=obj.get("client_reference_id") or (obj.get("metadata") or {}).get("clerk_user_id"),
                customer_id=obj.get("customer"),
                status="trialing",
                period_end=None,
            )
            return True
        return False

    def _record(self, *, clerk, customer_id, status, period_end) -> None:
        if not clerk and customer_id:
            found = self._store.get_by_customer(customer_id)
            clerk = found.clerk_user_id if found else None
        if not clerk:
            logger.warning("Stripe event without resolvable clerk_user_id (customer=%s)", customer_id)
            return
        self._store.upsert(
            Subscription(
                clerk_user_id=clerk,
                stripe_customer_id=customer_id,
                tier=tier_for(status),
                status=status,
                current_period_end=period_end,
                updated_at=_now(),
            )
        )

    # --- read ---------------------------------------------------------------
    def read_subscription(self, app_user) -> SubscriptionResponse:
        if app_user is None:
            return SubscriptionResponse(tier="free", status="none")
        sub = self._store.get(app_user.clerk_user_id)
        if sub is None:
            return SubscriptionResponse(tier="free", status="none")
        return SubscriptionResponse(
            tier=sub.tier,
            status=sub.status,
            current_period_end=sub.current_period_end,
            trial=(sub.status == "trialing"),
        )

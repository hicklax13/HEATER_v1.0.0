"""Billing logic seam — the one place that orchestrates the StripeGateway + the
SubscriptionStore. Never raises on the read/checkout paths (graceful ok/error);
the webhook fails CLOSED on a bad signature (HTTP 400) so Stripe + the operator
see a rejected delivery."""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime

from fastapi import HTTPException

from api.billing_config import billing_env_configured
from api.contracts.billing import (
    CheckoutSessionRequest,
    CheckoutSessionResponse,
    PortalSessionRequest,
    PortalSessionResponse,
    SubscriptionResponse,
    WebhookResponse,
)
from api.gateways.stripe_gateway import BillingSignatureError, StripeGateway
from api.stores.processed_event_store import ProcessedEventStore
from api.stores.subscription_store import Subscription, SubscriptionStore

logger = logging.getLogger(__name__)

_SUB_EVENTS = {"customer.subscription.created", "customer.subscription.updated", "customer.subscription.deleted"}
# Event types whose processing we ledger (for dedup). The subscription events
# are also ordering-guarded; checkout.session.completed is link-only but still
# deduped so a redelivery is a no-op.
_PROCESSED_EVENT_TYPES = _SUB_EVENTS | {"checkout.session.completed"}


def tier_for(status: str) -> str:
    """The contract slice 3 consumes: only an active or trialing subscription is Pro."""
    return "pro" if status in {"trialing", "active"} else "free"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _period_end(obj) -> int | None:
    """current_period_end moved from the subscription top-level to the item level in
    recent Stripe API versions; read whichever the webhook payload carries (display-only)."""
    val = obj.get("current_period_end")
    if val is not None:
        return val
    items = (obj.get("items") or {}).get("data") or []
    return items[0].get("current_period_end") if items else None


class BillingService:
    def __init__(
        self,
        gateway: StripeGateway,
        store: SubscriptionStore,
        events: ProcessedEventStore | None = None,
    ) -> None:
        self._gateway = gateway
        self._store = store
        # Optional processed-event ledger for webhook idempotency + ordering.
        # None (the legacy 2-arg signature) => dedup/ordering not enforced =
        # byte-for-byte today's behavior. DI wires the real store in prod.
        self._events = events

    # --- config (read at call time, like get_auth_verifier) -----------------
    @staticmethod
    def _env(name: str) -> str:
        return os.environ.get(name, "").strip()

    def _configured(self) -> bool:
        # Shared env predicate (same one the Pro gate uses) AND a live gateway.
        return billing_env_configured() and getattr(self._gateway, "configured", True)

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
            # Log the full Stripe message server-side (NOT a secret — Stripe errors
            # don't contain the API key) so the operator can see WHICH config is wrong
            # (bad price id, revoked key, livemode mismatch). User-facing error stays generic.
            logger.warning("create_checkout failed (%s): %s", type(exc).__name__, exc)
            return CheckoutSessionResponse(ok=False, error="Checkout could not be started.")

    # --- billing portal (manage / cancel) -----------------------------------
    def create_portal_session(self, app_user, req: PortalSessionRequest) -> PortalSessionResponse:
        if app_user is None:
            return PortalSessionResponse(ok=False, error="Sign in required.")
        if not self._configured():
            return PortalSessionResponse(ok=False, error="Billing not configured.")
        sub = self._store.get(app_user.clerk_user_id)
        customer_id = sub.stripe_customer_id if sub and sub.stripe_customer_id else None
        if not customer_id:
            return PortalSessionResponse(ok=False, error="No active subscription.")
        return_url = (
            (req.return_url or "").strip() or self._env("STRIPE_PORTAL_RETURN_URL") or "http://localhost:3000/account"
        )
        try:
            url = self._gateway.create_portal_session(customer_id, return_url)
            return PortalSessionResponse(ok=True, url=url)
        except Exception as exc:
            logger.warning("create_portal_session failed (%s): %s", type(exc).__name__, exc)
            return PortalSessionResponse(ok=False, error="Could not open the billing portal.")

    # --- webhook ------------------------------------------------------------
    def handle_webhook(self, payload: bytes, sig_header: str | None) -> WebhookResponse:
        secret = self._env("STRIPE_WEBHOOK_SECRET")
        if not secret:
            # If Stripe is otherwise live (secret key set) but the webhook secret is
            # unset/typo'd, we'd silently drop EVERY real event (paying users stay free)
            # and Stripe sees 200 = healthy → no retry, no alert. Make that operator-
            # visible. A fully-dormant install (no secret key) stays quiet.
            if self._env("STRIPE_SECRET_KEY"):
                logger.warning(
                    "Stripe webhook received but STRIPE_WEBHOOK_SECRET is unset — DROPPING "
                    "event (subscription state will NOT update). Set the webhook secret."
                )
            return WebhookResponse(ok=False, handled=False)
        try:
            event = self._gateway.parse_webhook_event(payload, sig_header, secret)
        except BillingSignatureError as exc:
            # Fail closed + operator-visible (shows as a failed delivery in Stripe).
            logger.warning("Stripe webhook signature verification failed: %s", exc)
            raise HTTPException(status_code=400, detail="Invalid webhook signature.")
        etype = event.get("type")
        event_id = event.get("id")
        event_created = event.get("created")
        obj = ((event.get("data") or {}).get("object")) or {}

        # Idempotency: Stripe redelivers at-least-once. If we've already processed
        # this event.id, it's a no-op (handled=False — no NEW state written). The
        # event store is optional (legacy 2-arg construction) → skip the guard.
        if self._events is not None and event_id and self._events.was_processed(event_id):
            logger.info("Stripe webhook duplicate event %s ignored (already processed).", event_id)
            return WebhookResponse(ok=True, event_type=etype, handled=False)

        handled = self._apply_event(etype, obj, event_created=event_created)

        # Record the event id AFTER applying so a redelivery (or a stale reorder)
        # of the same id is a no-op. Recorded for every event we recognized — incl.
        # ones we deliberately did not write (stale/unattributable) — so they are
        # not reprocessed.
        if self._events is not None and event_id and etype in _PROCESSED_EVENT_TYPES:
            try:
                self._events.record(
                    event_id,
                    event_created=event_created,
                    customer_id=obj.get("customer"),
                    subscription_id=obj.get("id") if etype in _SUB_EVENTS else None,
                )
            except Exception as exc:
                # Recording is hardening, not correctness-critical for THIS delivery —
                # never let a ledger write turn a handled event into a 500.
                logger.warning("Stripe webhook event-id record failed for %s: %s", event_id, exc)

        return WebhookResponse(ok=True, event_type=etype, handled=handled)

    def _apply_event(self, etype, obj, *, event_created=None) -> bool:
        # Returns whether subscription state was actually written. An event we can't
        # attribute to a user returns False so handle_webhook reports handled=False
        # (honest — not a false success that hides a dropped event).
        if etype in _SUB_EVENTS:
            customer_id = obj.get("customer")
            # Ordering guard: Stripe can deliver out of order. Do not let an OLDER
            # event.created overwrite subscription state a NEWER event already
            # applied for the same customer.
            if self._is_stale(customer_id, event_created):
                logger.info(
                    "Stripe webhook stale subscription event (created=%s) for customer=%s ignored — "
                    "newer state already applied.",
                    event_created,
                    customer_id,
                )
                return False
            status = "canceled" if etype.endswith("deleted") else (obj.get("status") or "none")
            return self._record(
                clerk=(obj.get("metadata") or {}).get("clerk_user_id"),
                customer_id=customer_id,
                status=status,
                period_end=_period_end(obj),
            )
        if etype == "checkout.session.completed":
            # LINK-ONLY: ensure the clerk<->customer link without forcing a status. The
            # customer.subscription.* events own tier/status, so an out-of-order completed
            # event can't re-activate a canceled user or null current_period_end.
            return self._ensure_link(
                clerk=obj.get("client_reference_id") or (obj.get("metadata") or {}).get("clerk_user_id"),
                customer_id=obj.get("customer"),
            )
        return False

    def _is_stale(self, customer_id, event_created) -> bool:
        # True when a NEWER subscription event has already been applied for this
        # customer (so the incoming OLDER event must not overwrite state). Needs
        # the event store + both timestamps; missing any → not stale (today's
        # behavior, no false skips).
        if self._events is None or not customer_id or event_created is None:
            return False
        last = self._events.last_applied_created(customer_id)
        return last is not None and event_created < last

    def _resolve_clerk(self, clerk, customer_id):
        if not clerk and customer_id:
            found = self._store.get_by_customer(customer_id)
            return found.clerk_user_id if found else None
        return clerk

    def _record(self, *, clerk, customer_id, status, period_end) -> bool:
        clerk = self._resolve_clerk(clerk, customer_id)
        if not clerk:
            logger.warning("Stripe event without resolvable clerk_user_id (customer=%s) — not handled", customer_id)
            return False
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
        return True

    def _ensure_link(self, *, clerk, customer_id) -> bool:
        clerk = self._resolve_clerk(clerk, customer_id)
        if not clerk:
            logger.warning(
                "checkout.session.completed without resolvable clerk_user_id (customer=%s) — not handled", customer_id
            )
            return False
        existing = self._store.get(clerk)
        if existing is not None:
            if existing.stripe_customer_id == customer_id:
                return True  # link already correct — preserve the authoritative status
            self._store.upsert(
                Subscription(
                    clerk_user_id=clerk,
                    stripe_customer_id=customer_id,
                    tier=existing.tier,
                    status=existing.status,
                    current_period_end=existing.current_period_end,
                    updated_at=_now(),
                )
            )
            return True
        # No row yet — create a free link row; the subscription.* events set the real status.
        self._store.upsert(
            Subscription(
                clerk_user_id=clerk, stripe_customer_id=customer_id, tier="free", status="none", updated_at=_now()
            )
        )
        return True

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

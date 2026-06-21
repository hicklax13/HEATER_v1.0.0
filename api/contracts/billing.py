"""Billing (Stripe) contracts. ok/error + graceful defaults follow the write-route
convention (never raise; the frontend reads ok + error)."""

from __future__ import annotations

from pydantic import BaseModel


class CheckoutSessionRequest(BaseModel):
    success_url: str | None = None  # frontend supplies; falls back to STRIPE_SUCCESS_URL / CORS origin
    cancel_url: str | None = None


class CheckoutSessionResponse(BaseModel):
    ok: bool
    url: str | None = None
    error: str | None = None


class PortalSessionRequest(BaseModel):
    return_url: str | None = None  # where Stripe sends the user back; falls back to STRIPE_PORTAL_RETURN_URL


class PortalSessionResponse(BaseModel):
    ok: bool
    url: str | None = None  # the Stripe-hosted billing-portal URL to redirect to
    error: str | None = None


class SubscriptionResponse(BaseModel):
    tier: str = "free"  # "free" | "pro"
    status: str = "none"  # none|trialing|active|canceled|past_due|unpaid|incomplete
    current_period_end: int | None = None
    trial: bool = False
    error: str | None = None


class WebhookResponse(BaseModel):
    ok: bool
    event_type: str | None = None
    handled: bool = False

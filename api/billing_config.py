"""Single source of truth for 'is Stripe billing operable from env?'.

Used by BOTH the billing service (checkout) and the Pro gate (gating) so they can't
drift — the gate must NEVER paywall users before checkout is possible (else a
half-configured deploy locks free users out with no upgrade path). Kept dependency-
free (only stdlib) so neither importer creates a cycle."""

from __future__ import annotations

import os


def billing_env_configured() -> bool:
    """True when checkout can actually run: BOTH the Stripe secret key AND the Pro
    price id are set. (The webhook secret is separately required to PROCESS events,
    not to START checkout, so it is not part of this 'can users upgrade?' predicate.)"""
    env = os.environ.get
    return bool(env("STRIPE_SECRET_KEY", "").strip() and env("STRIPE_PRO_PRICE_ID", "").strip())

import { apiPost } from "@/lib/api/client";
import { isAuthRequired } from "@/lib/api/errors";
import type { ApiCheckoutSessionResponse, ApiPortalSessionResponse } from "@/lib/api/types";

export interface CheckoutResult {
  ok: boolean;
  error?: string;
  needAuth?: boolean;
}

/**
 * Start a Stripe Checkout session and redirect to the hosted checkout. On success
 * the browser navigates to Stripe (this function does not return). `{ok:false}`
 * (a 200 from a dormant/misconfigured backend) and thrown errors (401 / network)
 * are surfaced for the caller to render.
 */
export async function startCheckout(): Promise<CheckoutResult> {
  const origin = window.location.origin;
  try {
    const r = await apiPost<ApiCheckoutSessionResponse>("/billing/checkout-session", {
      success_url: `${origin}/?upgraded=1`,
      cancel_url: `${origin}/pricing`,
    });
    if (r.ok && r.url) {
      window.location.href = r.url; // Stripe-hosted checkout
      return { ok: true };
    }
    return { ok: false, error: r.error ?? "Checkout isn't available yet." };
  } catch (e) {
    if (isAuthRequired(e)) return { ok: false, error: "Please sign in to start your trial.", needAuth: true };
    return { ok: false, error: "Something went wrong starting checkout. Try again." };
  }
}

/**
 * Open the Stripe customer portal (manage payment method / cancel subscription)
 * and redirect to it. On success the browser navigates to Stripe (no return).
 * Mirrors startCheckout's error surfacing.
 */
export async function openBillingPortal(): Promise<CheckoutResult> {
  const origin = window.location.origin;
  try {
    const r = await apiPost<ApiPortalSessionResponse>("/billing/portal-session", {
      return_url: `${origin}/account`,
    });
    if (r.ok && r.url) {
      window.location.href = r.url; // Stripe-hosted customer portal
      return { ok: true };
    }
    return { ok: false, error: r.error ?? "Subscription management isn't available yet." };
  } catch (e) {
    if (isAuthRequired(e)) return { ok: false, error: "Please sign in to manage your plan.", needAuth: true };
    return { ok: false, error: "Something went wrong opening the billing portal. Try again." };
  }
}

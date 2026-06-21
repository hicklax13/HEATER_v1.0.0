"use client";

import { useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { Sparkles, CreditCard, LogIn } from "lucide-react";
import { useSubscription, type SubscriptionState } from "@/lib/use-subscription";
import { openBillingPortal } from "@/lib/billing";
import { authEnabled } from "@/lib/auth-config";
import { staggerContainer, staggerItem } from "@/lib/motion";
import { Card } from "@/components/ui/Card";

/** Stripe current_period_end is a Unix timestamp in seconds. */
function formatDate(ts: number | null): string | null {
  if (!ts) return null;
  return new Date(ts * 1000).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

export default function AccountPage() {
  const sub = useSubscription();
  return (
    <main className="w-full flex-1 px-5 py-10">
      <motion.div
        variants={staggerContainer}
        initial="hidden"
        animate="show"
        className="mx-auto max-w-lg"
      >
        <motion.div variants={staggerItem}>
          <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-ink-3">Account</div>
          <h1 className="font-display text-3xl font-extrabold text-navy">Billing &amp; plan</h1>
        </motion.div>
        <motion.div variants={staggerItem} className="mt-6">
          {sub.signedIn ? <PlanCard sub={sub} /> : <SignedOutCard />}
        </motion.div>
      </motion.div>
    </main>
  );
}

function SignedOutCard() {
  // Covers dormant (authEnabled false → never signed in) and active-but-signed-out.
  return (
    <Card className="p-6 text-center">
      <p className="text-[14px] font-semibold text-navy">
        {authEnabled ? "Sign in to manage your plan" : "Account management launches with subscriptions"}
      </p>
      <p className="mx-auto mt-1.5 max-w-sm text-[13px] text-ink-2">
        {authEnabled
          ? "View your plan, update your payment method, or cancel anytime."
          : "Subscriptions aren't live yet — check the plans to see what Pro unlocks."}
      </p>
      <Link
        href={authEnabled ? "/sign-in" : "/pricing"}
        className="mt-5 inline-flex min-h-11 items-center justify-center gap-2 rounded-xl bg-gradient-to-b from-heat-bright to-heat px-6 text-sm font-bold text-white shadow-[0_6px_16px_rgba(255,92,16,0.32)] transition-transform duration-[var(--dur-1)] hover:scale-[1.02] active:scale-95 motion-reduce:transform-none"
      >
        {authEnabled ? <LogIn className="size-4" aria-hidden /> : <Sparkles className="size-4" aria-hidden />}
        {authEnabled ? "Log in" : "See plans"}
      </Link>
    </Card>
  );
}

function PlanCard({ sub }: { sub: SubscriptionState }) {
  const isPro = sub.tier === "pro";
  const renews = formatDate(sub.currentPeriodEnd);
  return (
    <Card className="p-6">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-[12px] font-bold uppercase tracking-wider text-ink-3">Current plan</div>
          <div className="mt-0.5 font-display text-2xl font-extrabold text-navy">
            {isPro ? "Pro" : "Free"}
          </div>
        </div>
        <span
          className={
            isPro
              ? "rounded-md bg-gradient-to-b from-heat-bright to-heat px-2.5 py-1 text-[11px] font-bold uppercase tracking-wider text-white shadow-[0_2px_8px_rgba(255,92,16,0.35)]"
              : "rounded-md border border-line px-2.5 py-1 text-[11px] font-bold uppercase tracking-wider text-ink-2"
          }
        >
          {isPro ? (sub.trial ? "Trial" : "Active") : "Free"}
        </span>
      </div>

      {isPro && renews && (
        <p className="mt-3 text-[13px] text-ink-2">
          {sub.trial ? "Your free trial ends " : "Renews "}
          <span className="font-semibold text-navy">{renews}</span>.
        </p>
      )}

      <div className="mt-5 border-t border-line pt-5">
        {isPro ? <ManageButton /> : <UpgradeLink />}
      </div>
    </Card>
  );
}

function UpgradeLink() {
  return (
    <Link
      href="/pricing"
      className="inline-flex min-h-11 w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-b from-heat-bright to-heat px-5 text-sm font-bold text-white shadow-[0_6px_16px_rgba(255,92,16,0.32)] transition-transform duration-[var(--dur-1)] hover:scale-[1.01] active:scale-95 motion-reduce:transform-none"
    >
      <Sparkles className="size-4" aria-hidden /> Upgrade to Pro
    </Link>
  );
}

function ManageButton() {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const onManage = async () => {
    setBusy(true);
    setErr(null);
    const r = await openBillingPortal();
    if (!r.ok) {
      setErr(r.error ?? "Try again.");
      setBusy(false);
    }
    // success → browser redirects to the Stripe portal
  };
  return (
    <>
      <button
        onClick={onManage}
        disabled={busy}
        className="inline-flex min-h-11 w-full items-center justify-center gap-2 rounded-xl border border-line bg-surface px-5 text-sm font-bold text-navy transition-colors hover:bg-surface-2 disabled:opacity-60"
      >
        <CreditCard className="size-4 text-heat" aria-hidden />
        {busy ? "Opening…" : "Manage subscription"}
      </button>
      <p className="mt-2 text-center text-[12px] text-ink-3">Update payment, change, or cancel — anytime.</p>
      {err && <p className="mt-1.5 text-center text-[12px] font-semibold text-ember">{err}</p>}
    </>
  );
}

"use client";

import { useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { Check, Sparkles } from "lucide-react";
import { useSubscription } from "@/lib/use-subscription";
import { startCheckout } from "@/lib/billing";
import { authEnabled } from "@/lib/auth-config";
import { staggerContainer, staggerItem } from "@/lib/motion";
import { Card } from "@/components/ui/Card";
import { cn } from "@/lib/utils";

const FREE_FEATURES = [
  "My Team dashboard + alerts",
  "Matchup planner & live standings",
  "Free agents & player databank",
  "Pitcher streaming & closer monitor",
  "Leaders & research",
];

const PRO_FEATURES = [
  "Daily Lineup Optimizer — start/sit by win probability",
  "Trade Analyzer — 6-phase Monte-Carlo evaluation",
  "Trade Finder — league-wide deal discovery",
  "Playoff Odds — full-season simulation",
  "Draft Simulator — recommendations vs AI opponents",
];

export default function PricingPage() {
  const sub = useSubscription();
  const isPro = sub.tier === "pro";

  return (
    <main className="w-full flex-1 px-5 py-10">
      <motion.div
        variants={staggerContainer}
        initial="hidden"
        animate="show"
        className="mx-auto max-w-4xl"
      >
        <motion.div variants={staggerItem} className="text-center">
          <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-ink-3">Plans</div>
          <h1 className="font-display text-4xl font-extrabold text-navy">Unlock the analyst&apos;s edge</h1>
          <p className="mx-auto mt-2 max-w-xl text-[14px] text-ink-2">
            Every decision tool, backed by Monte-Carlo simulation. Free to start — go Pro when
            you&apos;re ready to win the week.
          </p>
        </motion.div>

        <motion.div variants={staggerItem} className="mt-8 grid gap-5 md:grid-cols-2">
          {/* Free */}
          <Card className="flex flex-col p-6">
            <div className="text-[12px] font-bold uppercase tracking-wider text-ink-3">Free</div>
            <div className="mt-1 flex items-baseline gap-1">
              <span className="font-display text-4xl font-extrabold text-navy">$0</span>
              <span className="text-[13px] text-ink-3">/ forever</span>
            </div>
            <p className="mt-1 text-[13px] text-ink-2">The everyday manager&apos;s toolkit.</p>
            <ul className="mt-4 flex-1 space-y-2.5">
              {FREE_FEATURES.map((f) => (
                <FeatureRow key={f} label={f} tone="ink" />
              ))}
            </ul>
            <div className="mt-5 inline-flex min-h-11 items-center justify-center rounded-xl border border-line px-5 text-sm font-bold text-ink-2">
              {isPro ? "Included with Pro" : "Your current plan"}
            </div>
          </Card>

          {/* Pro */}
          <Card className="relative flex flex-col p-6 ring-2 ring-heat">
            <span className="absolute -top-3 left-6 rounded-full bg-gradient-to-b from-heat-bright to-heat px-3 py-1 text-[10px] font-bold uppercase tracking-wider text-white shadow-[0_2px_8px_rgba(255,92,16,0.4)]">
              7-day free trial
            </span>
            <div className="flex items-center gap-1.5 text-[12px] font-bold uppercase tracking-wider text-heat">
              <Sparkles className="size-4" aria-hidden /> Pro
            </div>
            <div className="mt-1 flex items-baseline gap-1">
              <span className="font-display text-4xl font-extrabold text-navy">$7.99</span>
              <span className="text-[13px] text-ink-3">/ month</span>
            </div>
            <p className="mt-1 text-[13px] text-ink-2">Everything in Free, plus the simulation engine.</p>
            <ul className="mt-4 flex-1 space-y-2.5">
              {PRO_FEATURES.map((f) => (
                <FeatureRow key={f} label={f} tone="heat" />
              ))}
            </ul>
            <div className="mt-5">
              <ProCta isPro={isPro} signedIn={sub.signedIn} trial={sub.trial} />
            </div>
          </Card>
        </motion.div>

        <motion.p variants={staggerItem} className="mt-6 text-center text-[12px] text-ink-3">
          Cancel anytime. The 7-day trial is free — you won&apos;t be charged until it ends.
        </motion.p>
      </motion.div>
    </main>
  );
}

function FeatureRow({ label, tone }: { label: string; tone: "ink" | "heat" }) {
  return (
    <li className="flex items-start gap-2.5 text-[13px] text-ink">
      <Check className={cn("mt-0.5 size-4 shrink-0", tone === "heat" ? "text-heat" : "text-ok")} aria-hidden />
      {label}
    </li>
  );
}

function ProCta({ isPro, signedIn, trial }: { isPro: boolean; signedIn: boolean; trial: boolean }) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  if (isPro) {
    return (
      <div className="inline-flex min-h-11 w-full items-center justify-center gap-1.5 rounded-xl bg-ok/12 px-5 text-sm font-bold text-ok">
        <Check className="size-4" aria-hidden /> {trial ? "Pro trial active" : "You're on Pro"}
      </div>
    );
  }

  // Dormant: billing not live yet.
  if (!authEnabled) {
    return (
      <div className="inline-flex min-h-11 w-full items-center justify-center rounded-xl border border-line px-5 text-sm font-bold text-ink-3">
        Subscriptions launching soon
      </div>
    );
  }

  // Active but signed out → sign up first.
  if (!signedIn) {
    return (
      <Link
        href="/sign-up"
        className="inline-flex min-h-11 w-full items-center justify-center rounded-xl bg-gradient-to-b from-heat-bright to-heat px-5 text-sm font-bold text-white shadow-[0_6px_16px_rgba(255,92,16,0.32)] transition-transform duration-[var(--dur-1)] hover:scale-[1.01] active:scale-95 motion-reduce:transform-none"
      >
        Start 7-day free trial
      </Link>
    );
  }

  const onUpgrade = async () => {
    setBusy(true);
    setErr(null);
    const r = await startCheckout();
    if (!r.ok) {
      setErr(r.error ?? "Try again.");
      setBusy(false);
    }
    // success → browser redirects to Stripe
  };

  return (
    <>
      <button
        onClick={onUpgrade}
        disabled={busy}
        className="inline-flex min-h-11 w-full items-center justify-center rounded-xl bg-gradient-to-b from-heat-bright to-heat px-5 text-sm font-bold text-white shadow-[0_6px_16px_rgba(255,92,16,0.32)] transition-transform duration-[var(--dur-1)] hover:scale-[1.01] active:scale-95 disabled:opacity-60 motion-reduce:transform-none"
      >
        {busy ? "Starting checkout…" : "Start 7-day free trial"}
      </button>
      {err && <p className="mt-2 text-center text-[12px] font-semibold text-ember">{err}</p>}
    </>
  );
}

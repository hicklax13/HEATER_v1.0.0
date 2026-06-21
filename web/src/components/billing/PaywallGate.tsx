"use client";

import Link from "next/link";
import { Lock, Sparkles } from "lucide-react";
import { Card } from "@/components/ui/Card";

/** Shown in place of a Pro feature when the API returns 402 (signed-in Free user).
 *  Only reachable once billing is live; dormant never 402s. */
export function PaywallGate({ feature }: { feature?: string }) {
  return (
    <Card className="mx-auto max-w-md p-8 text-center">
      <div className="mx-auto flex size-12 items-center justify-center rounded-full bg-heat/12 text-heat">
        <Lock className="size-6" aria-hidden />
      </div>
      <h2 className="mt-4 font-display text-xl font-bold text-navy">
        {feature ? `${feature} is a Pro feature` : "This is a Pro feature"}
      </h2>
      <p className="mx-auto mt-1.5 max-w-sm text-[13px] text-ink-2">
        Unlock the full simulation engine — optimizer, trades, playoff odds, and the draft
        simulator — with a 7-day free trial.
      </p>
      <Link
        href="/pricing"
        className="mt-5 inline-flex min-h-11 items-center justify-center gap-2 rounded-xl bg-gradient-to-b from-heat-bright to-heat px-6 text-sm font-bold text-white shadow-[0_6px_16px_rgba(255,92,16,0.32)] transition-transform duration-[var(--dur-1)] hover:scale-[1.02] active:scale-95 motion-reduce:transform-none"
      >
        <Sparkles className="size-4" aria-hidden /> See Pro plans
      </Link>
    </Card>
  );
}

"use client";

import type { LucideIcon } from "lucide-react";
import { CloudOff, RefreshCw, UserX } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";
import { PaywallGate } from "@/components/billing/PaywallGate";

/** Page-level Pro paywall (the `locked` PageState — a 402). */
export function PageLocked({ feature }: { feature?: string }) {
  return (
    <div className="mt-10">
      <PaywallGate feature={feature} />
    </div>
  );
}

/** Page-level load failure: error card + Retry. Generic copy works for every page. */
export function PageError({ onRetry }: { onRetry: () => void }) {
  return (
    <Card className="mx-auto mt-10 max-w-md">
      <EmptyState
        icon={CloudOff}
        tone="error"
        title="We couldn't load this"
        body="The data service didn't respond. Your data is safe — try again."
        action={
          <button
            onClick={onRetry}
            className="inline-flex min-h-11 items-center gap-2 rounded-xl bg-gradient-to-b from-heat-bright to-heat px-5 text-sm font-semibold text-white shadow-[0_6px_16px_rgba(255,92,16,0.32)] transition-transform duration-[var(--dur-1)] hover:scale-[1.02] active:scale-95 motion-reduce:transform-none"
          >
            <RefreshCw className="size-4" aria-hidden />
            Retry
          </button>
        }
      />
    </Card>
  );
}

/** Page-level "no data at all" (distinct from in-table no-results). Copy is per page. */
export function PageEmpty({
  icon,
  title,
  body,
}: {
  icon: LucideIcon;
  title: string;
  body?: string;
}) {
  return (
    <Card className="mx-auto mt-10 max-w-md">
      <EmptyState icon={icon} title={title} body={body} />
    </Card>
  );
}

/** Page-level "team not linked yet" (the `unlinked` PageState — a 409). An authed
 *  viewer has no team assignment; personalized pages show this instead of another
 *  team's data, while league-wide views keep working (HIGH-1). */
export function PageNotLinked() {
  return (
    <PageEmpty
      icon={UserX}
      title="Your team isn't linked yet"
      body="Your commissioner will assign your team shortly. League-wide views (Standings, Leaders, Players) work in the meantime."
    />
  );
}

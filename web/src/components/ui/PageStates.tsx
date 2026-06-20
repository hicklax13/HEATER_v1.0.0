"use client";

import type { LucideIcon } from "lucide-react";
import { CloudOff, RefreshCw } from "lucide-react";
import { Card } from "@/components/ui/Card";
import { EmptyState } from "@/components/ui/EmptyState";

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

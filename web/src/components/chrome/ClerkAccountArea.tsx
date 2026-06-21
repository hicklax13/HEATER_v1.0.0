"use client";

import Link from "next/link";
import { CreditCard } from "lucide-react";
import { UserButton } from "@clerk/nextjs";
import { useSubscription } from "@/lib/use-subscription";

/**
 * The TopBar account zone when auth is ACTIVE (Clerk key set). Signed-out shows
 * Log in / Get started; signed-in shows a Pro/Upgrade chip (→ /pricing) + Clerk's
 * UserButton. Branches on useSubscription().signedIn (sourced from Clerk's
 * useAuth). Only rendered inside ClerkProvider — the dormant TopBar branch keeps
 * the original mock menu, so today's app is unchanged. */
export function ClerkAccountArea() {
  const sub = useSubscription();
  const isPro = sub.tier === "pro";

  if (!sub.signedIn) {
    return (
      <>
        <Link
          href="/sign-in"
          className="hidden h-9 items-center rounded-lg px-3 text-sm font-semibold text-white/85 transition-colors hover:bg-white/10 sm:inline-flex"
        >
          Log in
        </Link>
        <Link
          href="/sign-up"
          className="inline-flex h-9 min-h-9 items-center rounded-lg bg-gradient-to-b from-heat-bright to-heat px-3.5 text-sm font-bold text-white shadow-[0_2px_8px_rgba(255,92,16,0.35)] transition-transform duration-[var(--dur-1)] hover:scale-[1.03] active:scale-95 motion-reduce:transform-none"
        >
          Get started
        </Link>
      </>
    );
  }

  return (
    <>
      <Link
        href="/pricing"
        aria-label={isPro ? "Pro plan" : "Upgrade to Pro"}
        className={
          isPro
            ? "tnum rounded-md bg-gradient-to-b from-heat-bright to-heat px-2.5 py-1 text-[11px] font-bold tracking-wider text-white shadow-[0_2px_8px_rgba(255,92,16,0.35)]"
            : "rounded-md border border-white/25 px-2.5 py-1 text-[11px] font-bold tracking-wider text-white/85 transition-colors hover:bg-white/10"
        }
      >
        {isPro ? "PRO" : "UPGRADE"}
      </Link>
      <UserButton>
        <UserButton.MenuItems>
          <UserButton.Link
            label="Billing & plan"
            labelIcon={<CreditCard className="size-4" />}
            href="/account"
          />
        </UserButton.MenuItems>
      </UserButton>
    </>
  );
}

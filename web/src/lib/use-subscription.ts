"use client";

import { createContext, createElement, useCallback, useContext, useEffect, useState } from "react";
import { useAuth } from "@clerk/nextjs";
import { apiGet } from "@/lib/api/client";
import type { ApiSubscriptionResponse } from "@/lib/api/types";
import { authEnabled } from "@/lib/auth-config";

export type Tier = "free" | "pro";

export interface SubscriptionState {
  active: boolean; // billing/auth env is ON (Clerk key set)
  signedIn: boolean;
  tier: Tier;
  status: string;
  trial: boolean;
  currentPeriodEnd: number | null;
  loading: boolean;
  refresh: () => void;
}

const DORMANT: SubscriptionState = {
  active: false,
  signedIn: false,
  tier: "free",
  status: "none",
  trial: false,
  currentPeriodEnd: null,
  loading: false,
  refresh: () => {},
};

const Ctx = createContext<SubscriptionState>(DORMANT);
export const useSubscription = () => useContext(Ctx);

/**
 * Provides subscription state app-wide. When dormant (no Clerk key) it serves a
 * frozen Free/inactive value and NEVER calls a Clerk hook (so it's valid outside
 * ClerkProvider). When active it delegates to <ActiveProvider>, which is only
 * rendered inside ClerkProvider — so its useAuth() is always legal. `authEnabled`
 * is a build-time constant, so the branch is stable (no conditional-hook hazard).
 */
export function SubscriptionProvider({ children }: { children: React.ReactNode }) {
  if (!authEnabled) return createElement(Ctx.Provider, { value: DORMANT }, children);
  return createElement(ActiveProvider, null, children);
}

function ActiveProvider({ children }: { children: React.ReactNode }) {
  const { isLoaded, isSignedIn } = useAuth();
  const [sub, setSub] = useState({
    tier: "free" as Tier,
    status: "none",
    trial: false,
    currentPeriodEnd: null as number | null,
    loading: true,
  });
  const [epoch, setEpoch] = useState(0);
  const refresh = useCallback(() => setEpoch((n) => n + 1), []);

  useEffect(() => {
    if (!isLoaded) return;
    let alive = true;
    // Defer all setState into async callbacks so the synchronous effect body
    // never calls setState (satisfies react-hooks/set-state-in-effect; mirrors
    // usePageData).
    Promise.resolve()
      .then(() => {
        if (!alive) return undefined;
        if (!isSignedIn) {
          setSub((s) => ({ ...s, tier: "free", loading: false }));
          return undefined;
        }
        setSub((s) => ({ ...s, loading: true }));
        return apiGet<ApiSubscriptionResponse>("/billing/subscription");
      })
      .then((r) => {
        if (!alive || !r) return; // r is undefined on the signed-out branch
        setSub({
          tier: r.tier === "pro" ? "pro" : "free",
          status: r.status,
          trial: r.trial,
          currentPeriodEnd: r.current_period_end ?? null,
          loading: false,
        });
      })
      .catch(() => {
        // 401 / not-configured / network → treat as Free (no Pro unlock)
        if (alive) setSub((s) => ({ ...s, tier: "free", loading: false }));
      });
    return () => {
      alive = false;
    };
  }, [isLoaded, isSignedIn, epoch]);

  return createElement(
    Ctx.Provider,
    { value: { active: true, signedIn: !!isSignedIn, ...sub, refresh } },
    children,
  );
}

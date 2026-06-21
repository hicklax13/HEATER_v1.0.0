"use client";

import { useCallback, useEffect, useState } from "react";
import { isPaywall } from "@/lib/api/errors";

/** Page-level async state. `loaded` carries the data; the others are terminal UI
 *  states. `locked` is the M2 paywall (fetcher threw an ApiError 402 — signed-in
 *  Free user); it never occurs while billing is dormant. */
export type PageState<T> =
  | { status: "loading" }
  | { status: "error" }
  | { status: "empty" }
  | { status: "locked" }
  | { status: "loaded"; data: T };

type Forced = "loading" | "error" | "empty";

/**
 * Dev-only verification aid: `?state=loading|error|empty` forces a branch so we
 * can screenshot it (mock shims never fail/return null). Returns undefined in
 * production (NODE_ENV guard → dead-code-eliminated) or when absent/`loaded`.
 * Reads `window.location.search` directly — NOT next/navigation's
 * `useSearchParams`, which triggers Next 16's static-generation CSR bailout.
 */
function forcedState(): Forced | undefined {
  if (process.env.NODE_ENV === "production") return undefined;
  if (typeof window === "undefined") return undefined;
  const v = new URLSearchParams(window.location.search).get("state");
  return v === "loading" || v === "error" || v === "empty" ? v : undefined;
}

/**
 * Drives the four-state machine for a page. Pass a STABLE fetcher (a module-level
 * `fetchX` reference, never an inline arrow — an inline arrow changes identity
 * every render and re-fires the effect).
 *
 * Empty contract: a fetcher signals "no data at all" by resolving `null` (or
 * `undefined`) → `empty`. ANY non-null value — including an empty array or a
 * zero-item object — resolves to `loaded`; pages render "loaded but zero items"
 * cases (e.g. 0 trade recs) inside their own loaded view, not as page-level empty.
 * A rejection resolves to `error`. `retry()` re-runs the fetcher.
 */
export function usePageData<T>(
  fetcher: () => Promise<T | null>,
): { state: PageState<T>; retry: () => void } {
  const [state, setState] = useState<PageState<T>>({ status: "loading" });
  // Incrementing this counter triggers the effect to re-run (retry path).
  const [epoch, setEpoch] = useState(0);

  useEffect(() => {
    const forced = forcedState();
    let alive = true;

    // All setState calls happen inside async callbacks so the synchronous
    // effect body never calls setState (satisfies react-hooks/set-state-in-effect).
    Promise.resolve()
      .then(() => {
        if (!alive) return;
        if (forced) {
          setState({ status: forced });
          return;
        }
        // Reset to loading — necessary on the retry path, where the prior state
        // may be `loaded`/`error` and must return to `loading` before the refetch.
        setState({ status: "loading" });
        return fetcher();
      })
      .then((data) => {
        // data is undefined when the forced branch early-returned above.
        if (!alive || data === undefined) return;
        setState(data == null ? { status: "empty" } : { status: "loaded", data });
      })
      .catch((e) => {
        // A 402 (Pro paywall) maps to `locked`; everything else is generic `error`.
        if (alive) setState(isPaywall(e) ? { status: "locked" } : { status: "error" });
      });

    return () => {
      alive = false;
    };
  }, [fetcher, epoch]);

  const retry = useCallback(() => setEpoch((n) => n + 1), []);

  return { state, retry };
}

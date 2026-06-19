"use client";

import { useCallback, useEffect, useState } from "react";

/** Page-level async state. `loaded` carries the data; the other three are terminal UI states. */
export type PageState<T> =
  | { status: "loading" }
  | { status: "error" }
  | { status: "empty" }
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
 * every render and re-fires the effect). `null`/`undefined` resolves to `empty`;
 * a rejection resolves to `error`. `retry()` re-runs the fetcher.
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
        setState({ status: "loading" });
        return fetcher();
      })
      .then((data) => {
        // data is undefined when the forced branch early-returned above.
        if (!alive || data === undefined) return;
        setState(data == null ? { status: "empty" } : { status: "loaded", data });
      })
      .catch(() => {
        if (alive) setState({ status: "error" });
      });

    return () => {
      alive = false;
    };
  }, [fetcher, epoch]);

  const retry = useCallback(() => setEpoch((n) => n + 1), []);

  return { state, retry };
}

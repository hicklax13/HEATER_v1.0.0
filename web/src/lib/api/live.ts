/** Live-vs-mock helpers. Under HEATER_LIVE we run the real API call and let any
 *  ApiError / network error PROPAGATE (so usePageData reaches error/locked/unlinked
 *  or routes to sign-in) — fetchers must NOT swallow to mock when live (HIGH-3).
 *  Off-live (demo/marketing build) the mock is the intended behavior. */

/** Canonical live-mode predicate. `isLive` was not exported from a shared module
 *  (each `*-data.ts` inlined `process.env.NEXT_PUBLIC_HEATER_LIVE === "1"`); this
 *  is now the one canonical source the fetcher refactors (Task 8/9) migrate to. */
export function isLive(): boolean {
  return process.env.NEXT_PUBLIC_HEATER_LIVE === "1";
}

export async function liveOrMock<T>(
  live: () => Promise<T>,
  mock: () => T | Promise<T>,
): Promise<T> {
  if (!isLive()) return mock();
  return live(); // no catch — errors propagate to usePageData
}

/** Reject (not resolve-to-mock) when a live call exceeds `ms`, so a hung API
 *  surfaces as the error state instead of an infinite spinner. */
export function withTimeout<T>(p: Promise<T>, ms: number): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const t = setTimeout(() => reject(new Error(`timeout after ${ms}ms`)), ms);
    p.then(
      (v) => {
        clearTimeout(t);
        resolve(v);
      },
      (e) => {
        clearTimeout(t);
        reject(e);
      },
    );
  });
}

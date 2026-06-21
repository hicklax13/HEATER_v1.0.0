/** Typed API error so callers can branch on the HTTP status — specifically the
 *  M2 paywall (402 = authenticated Free user hitting a Pro endpoint) and auth
 *  (401 = unauthenticated). While billing is dormant the gated endpoints stay
 *  open, so neither status ever occurs and existing catch-and-fallback is intact. */
export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly path: string,
  ) {
    super(`API ${path} -> ${status}`);
    this.name = "ApiError";
  }
}

/** 402 — the user is signed in but on the Free tier; show the upgrade gate. */
export function isPaywall(e: unknown): e is ApiError {
  return e instanceof ApiError && e.status === 402;
}

/** 401 — no/invalid session; route to login. */
export function isAuthRequired(e: unknown): e is ApiError {
  return e instanceof ApiError && e.status === 401;
}

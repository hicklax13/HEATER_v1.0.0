/**
 * Single env gate for ALL M2 auth/billing UI. When the Clerk publishable key is
 * unset (today + the live app), the entire auth/billing layer is dormant and the
 * app is byte-identical to M1: no auth walls, no Bearer token attached, no 402s
 * (the backend keeps the gated endpoints open while dormant). Connor flips this
 * by setting NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY (his activation step). Because Next
 * inlines NEXT_PUBLIC_* at build time, `authEnabled` is a compile-time constant,
 * so the dormant branches are dead-code-eliminated.
 */
export const CLERK_KEY = process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY ?? "";
export const authEnabled = CLERK_KEY.length > 0;

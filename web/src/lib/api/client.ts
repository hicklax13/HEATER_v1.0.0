/** Minimal typed client for the HEATER API. Calls SAME-ORIGIN `/api/*`, which
 *  next.config rewrites to the FastAPI backend (so the browser sees no CORS).
 *
 *  Auth (M2): when Clerk is active, the browser exposes `window.Clerk`; we attach
 *  the session JWT as `Authorization: Bearer <token>` to every call (required by
 *  the billing + 6 gated endpoints, harmless on the open ones). When dormant
 *  (no Clerk), there's no session → no header → unauthenticated requests, exactly
 *  as in M1. Non-OK responses throw a typed `ApiError` so callers can branch on
 *  402 (paywall) / 401 (login). */

import { ApiError } from "./errors";
import { authEnabled } from "@/lib/auth-config";

declare global {
  interface Window {
    Clerk?: { loaded?: boolean; session?: { getToken?: () => Promise<string | null> } | null };
  }
}

const CLERK_READY_TIMEOUT_MS = 5000;
const CLERK_POLL_MS = 50;

/** Resolve once Clerk has finished loading. Clerk's script attaches `window.Clerk`
 *  ASYNCHRONOUSLY after the page loads, so on a hard load / refresh the first API
 *  call can fire before the session exists → no token → a spurious 401 that the
 *  page handler turns into a /sign-in redirect (which, for an already-signed-in
 *  user, bounces home). Waiting for `window.Clerk.loaded` lets us attach the real
 *  token on hard loads. No-op (returns immediately) when auth is dormant — there
 *  is no Clerk script to wait for, so the loop would otherwise stall every call. */
async function clerkReady(): Promise<void> {
  if (!authEnabled || typeof window === "undefined") return;
  const start = Date.now();
  while (Date.now() - start < CLERK_READY_TIMEOUT_MS) {
    if (window.Clerk?.loaded) return;
    await new Promise((r) => setTimeout(r, CLERK_POLL_MS));
  }
}

/** Bearer token from the active Clerk session, or null (dormant / signed out / SSR).
 *  Waits for Clerk to finish loading first (hard-load race — see clerkReady), then
 *  calls getToken AS A METHOD on the session — destructuring it loses the `this`
 *  receiver Clerk needs and would silently yield no token (→ spurious 401s). */
export async function authToken(): Promise<string | null> {
  if (typeof window === "undefined") return null;
  if (!authEnabled) return null; // dormant: byte-identical to today (no Clerk script)
  await clerkReady();
  const session = window.Clerk?.session;
  if (!session?.getToken) return null; // Clerk loaded but genuinely signed out
  try {
    return await session.getToken();
  } catch {
    return null;
  }
}

export async function apiGet<T>(
  path: string,
  params?: Record<string, string | number>,
): Promise<T> {
  const qs = params
    ? "?" + new URLSearchParams(Object.entries(params).map(([k, v]) => [k, String(v)])).toString()
    : "";
  const headers: Record<string, string> = { Accept: "application/json" };
  const token = await authToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  const res = await fetch(`/api${path}${qs}`, { headers });
  if (!res.ok) throw new ApiError(res.status, path);
  return (await res.json()) as T;
}

export async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Accept: "application/json",
  };
  const token = await authToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  const res = await fetch(`/api${path}`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new ApiError(res.status, path);
  return (await res.json()) as T;
}

export async function apiPut<T>(path: string, body: unknown): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    Accept: "application/json",
  };
  const token = await authToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  const res = await fetch(`/api${path}`, {
    method: "PUT",
    headers,
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new ApiError(res.status, path);
  return (await res.json()) as T;
}

export async function apiDelete<T>(
  path: string,
  params?: Record<string, string | number>,
): Promise<T> {
  const qs = params
    ? "?" + new URLSearchParams(Object.entries(params).map(([k, v]) => [k, String(v)])).toString()
    : "";
  const headers: Record<string, string> = { Accept: "application/json" };
  const token = await authToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  const res = await fetch(`/api${path}${qs}`, { method: "DELETE", headers });
  if (!res.ok) throw new ApiError(res.status, path);
  return (await res.json()) as T;
}

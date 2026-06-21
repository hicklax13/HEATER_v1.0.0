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

declare global {
  interface Window {
    Clerk?: { session?: { getToken?: () => Promise<string | null> } };
  }
}

/** Bearer token from the active Clerk session, or null (dormant / signed out / SSR).
 *  Call getToken AS A METHOD on the session — destructuring it loses the `this`
 *  receiver Clerk needs and would silently yield no token (→ spurious 401s). */
async function authToken(): Promise<string | null> {
  if (typeof window === "undefined") return null;
  const session = window.Clerk?.session;
  if (!session?.getToken) return null;
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

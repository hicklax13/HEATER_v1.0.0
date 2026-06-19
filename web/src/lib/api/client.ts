/** Minimal typed client for the HEATER API. Calls SAME-ORIGIN `/api/*`, which
 *  next.config rewrites to the FastAPI backend (so the browser sees no CORS). */

export async function apiGet<T>(
  path: string,
  params?: Record<string, string | number>,
): Promise<T> {
  const qs = params
    ? "?" + new URLSearchParams(Object.entries(params).map(([k, v]) => [k, String(v)])).toString()
    : "";
  const res = await fetch(`/api${path}${qs}`, { headers: { Accept: "application/json" } });
  if (!res.ok) throw new Error(`API ${path} -> ${res.status}`);
  return (await res.json()) as T;
}

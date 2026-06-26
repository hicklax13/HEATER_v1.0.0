"use client";

import { useEffect, useState } from "react";
import { apiGet } from "@/lib/api/client";
import { getViewerTeam } from "@/lib/viewer-team";
import type { ApiMatchupResponse } from "@/lib/api/types";

/**
 * Live current league week, sourced from the matchup endpoint (the same week the
 * rest of the app shows), cached for the module lifetime so every consumer shares
 * ONE request. `null` off-live or on error → callers omit the week label rather
 * than showing a stale/hardcoded value (L-8: the Trades eyebrow was pinned to
 * "Week 13" while the live app was on Week 14).
 */
let _weekPromise: Promise<number | null> | null = null;

export function fetchCurrentWeek(): Promise<number | null> {
  if (process.env.NEXT_PUBLIC_HEATER_LIVE !== "1") return Promise.resolve(null);
  if (!_weekPromise) {
    _weekPromise = (async () => {
      try {
        const m = await apiGet<ApiMatchupResponse>("/matchup", { team_name: getViewerTeam() });
        return typeof m.week === "number" && m.week > 0 ? m.week : null;
      } catch {
        _weekPromise = null; // transient failure → let a later mount retry
        return null;
      }
    })();
  }
  return _weekPromise;
}

/** The live current week, or null until it resolves / off-live. */
export function useCurrentWeek(): number | null {
  const [week, setWeek] = useState<number | null>(null);
  useEffect(() => {
    let alive = true;
    fetchCurrentWeek().then((w) => {
      if (alive) setWeek(w);
    });
    return () => {
      alive = false;
    };
  }, []);
  return week;
}

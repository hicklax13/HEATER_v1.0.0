"use client";

import { usePathname } from "next/navigation";
import { useEffect } from "react";

/** Route → browser-tab title. Every page is a client component, so none can
 *  export per-route `metadata`; the root layout therefore ships one static
 *  title ("HEATER — My Team") that would otherwise label every tab "My Team".
 *  This sets `document.title` per page instead. Unknown routes fall back to the
 *  bare brand. */
const TITLES: Record<string, string> = {
  "/": "My Team",
  "/optimizer": "Lineup Optimizer",
  "/streaming": "Pitcher Streaming",
  "/probables": "Probable Pitchers",
  "/hitter-matchups": "Hitter Matchups",
  "/closers": "Closer Monitor",
  "/matchup": "Matchup",
  "/standings": "Standings",
  "/punt": "Punt Analyzer",
  "/trades": "Trades",
  "/players": "Players",
  "/research": "Research",
  "/databank": "Databank",
  "/draft": "Draft",
  "/pricing": "Pricing",
  "/account": "Account",
};

export function DocumentTitle() {
  const pathname = usePathname();
  useEffect(() => {
    const page = TITLES[pathname];
    document.title = page ? `HEATER — ${page}` : "HEATER";
  }, [pathname]);
  return null;
}

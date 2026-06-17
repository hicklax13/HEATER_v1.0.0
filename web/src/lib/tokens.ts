/**
 * HEATER palette in TS — for SVG / canvas / recharts where CSS vars are awkward.
 * Cloned from the Gemini brand assets (V2 wordmark + baseball icon).
 * Keep in sync with globals.css @theme.
 */
export const COLORS = {
  canvas: "#ffffff",
  surface: "#f5f6f8",
  surface2: "#eef0f3",
  ink: "#1b1c20",
  ink2: "#646a78",
  ink3: "#9aa0ac",
  line: "#e3e2de",

  navy: "#0a1f3a",
  navyDeep: "#041731",
  navy700: "#1b1f3a",
  navyElevated: "#122844",

  heat: "#ff5c10",
  flame: "#ff8a1e",
  gold: "#ffe519",
  redhot: "#ff4401",
  ember: "#e63946",
  stitch: "#7f292e",

  ok: "#1f9d6b",
  warn: "#eab308",
  coolBright: "#4fb0e6",
  steel: "#5f7d9c",
  chrome: "#eef0f2",
  cream: "#e4d9c0",
} as const;

export const TIERS = [
  "#ff5c10",
  "#ff8a1e",
  "#ffae42",
  "#1f9d6b",
  "#5f7d9c",
  "#8aa6c0",
  "#b0b5be",
  "#cdd1d8",
] as const;

/** MLB public CDN helpers (configured in next.config remotePatterns). */
export const MLB = {
  headshot: (mlbId: number) =>
    `https://midfield.mlbstatic.com/v1/people/${mlbId}/spots/120`,
  teamLogo: (teamId: number) =>
    `https://www.mlbstatic.com/team-logos/${teamId}.svg`,
} as const;

/**
 * Heat-gauge ramp: map a win probability (0..100) to a temperature color.
 * Low = ember/navy (cold), mid = neutral steel, >60 = orange→white-hot.
 */
export function heatColor(pct: number): string {
  // Brighter stops so the gauge contrasts on the navy hero.
  if (pct < 20) return "#ff5a3c"; // bright ember (cold-bad)
  if (pct < 38) return "#ff7a4d"; // warm coral
  if (pct < 52) return COLORS.coolBright; // bright cool blue — reads "cooling", high contrast
  if (pct < 62) return COLORS.flame;
  if (pct < 75) return COLORS.heat;
  if (pct < 88) return "#ff8a1e";
  return COLORS.gold; // white-hot
}

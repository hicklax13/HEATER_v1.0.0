/** Confidence label → 0–100 job-security score (drives the closer card's heat bar).
 *  Leaf module (imports nothing) so BOTH the live adapter and the mock derive
 *  security identically without creating an adapters ⇄ closers-data import cycle. */
export const CONFIDENCE_SECURITY: Record<string, number> = {
  Locked: 95,
  High: 78,
  Committee: 50,
  Shaky: 30,
};

export function securityFor(confidence: string): number {
  return CONFIDENCE_SECURITY[confidence] ?? 45;
}

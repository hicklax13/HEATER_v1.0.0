// Module-level cache of the logged-in viewer's API-resolved team name, so non-
// component code (data shims + adapters) can mark "you" correctly without re-fetching
// the heavy /api/me/team. Populated by fetchMyTeam (the Team landing page) from the
// API's identity-resolved team_name; defaults to the single-league owner until then.
let _viewerTeam = "Team Hickey";

export function getViewerTeam(): string {
  return _viewerTeam;
}

export function setViewerTeam(name: string | null | undefined): void {
  const n = (name ?? "").trim();
  if (n) _viewerTeam = n;
}

/** Thin aliases over the OpenAPI-generated schemas (./generated.ts).
 *
 *  Source of truth: api/openapi.json (committed, snapshot-guarded by the backend).
 *  Regenerate after any contract change with `pnpm gen:api`.
 *
 *  These names are kept stable so client.ts / adapters.ts / research-data.ts keep
 *  importing the same symbols — only their underlying definition is now generated,
 *  not hand-maintained, so the frontend types can no longer drift from the API. */

import type { components } from "./generated";

export type ApiPlayerRef = components["schemas"]["PlayerRef"];
export type ApiLeaderRow = components["schemas"]["LeaderRow"];
export type ApiLeadersResponse = components["schemas"]["LeadersResponse"];
export type ApiLeadersOverallResponse = components["schemas"]["LeadersOverallResponse"];
export type ApiFreeAgentPoolResponse = components["schemas"]["FreeAgentPoolResponse"];
export type ApiFreeAgentPoolItem = components["schemas"]["FreeAgentPoolItem"];
export type ApiStreamingResponse = components["schemas"]["StreamingResponse"];
export type ApiStreamAnalyzeResponse = components["schemas"]["StreamAnalyzeResponse"];
export type ApiStandingsResponse = components["schemas"]["StandingsResponse"];
export type ApiPuntResponse = components["schemas"]["PuntResponse"];
export type ApiMatchupResponse = components["schemas"]["MatchupResponse"];
export type ApiCompareResponse = components["schemas"]["CompareResponse"];

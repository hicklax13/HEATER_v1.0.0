/** TypeScript mirrors of the FastAPI contracts this slice consumes.
 *  Source of truth: api/contracts/leaders.py + api/contracts/common.py.
 *  (Full-surface generation from openapi.json is future work — see the gap spec.) */

export interface ApiPlayerRef {
  id: number;
  name: string;
  positions: string;
  yahoo_player_key?: string | null;
}

export interface ApiLeaderRow {
  rank: number;
  player: ApiPlayerRef;
  value: number;
}

export interface ApiLeadersResponse {
  category: string;
  rows: ApiLeaderRow[];
}

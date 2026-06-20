"""Roster-query service — the ONE place importing src/ for the player-id endpoints.

Two read methods: search() (pool-wide name search) + league_rosters() (all teams'
rosters with ids). Both reuse PlayerRef enrichment. Resilient: missing data → empty."""

from __future__ import annotations

import logging

from api.contracts.players import LeagueRostersResponse, LeagueRosterTeam, PlayerSearchResponse
from api.services.player_ref import make_player_ref, player_ref_from_pool

logger = logging.getLogger(__name__)


class RosterQueryService:
    def search(self, q: str, limit: int = 25) -> PlayerSearchResponse:
        query = (q or "").strip()
        if not query:
            return PlayerSearchResponse(query=query, results=[])
        try:
            limit = max(1, min(int(limit or 25), 200))  # clamp: never scan/return the whole pool

            from src.database import load_player_pool

            pool = load_player_pool()
            if pool is None or pool.empty or "name" not in pool.columns:
                return PlayerSearchResponse(query=query, results=[])

            names = pool["name"].fillna("").astype(str)
            matches = pool[names.str.contains(query, case=False, regex=False)]
            if matches.empty:
                return PlayerSearchResponse(query=query, results=[])
            if "consensus_rank" in matches.columns:
                matches = matches.sort_values("consensus_rank", na_position="last")
            results = [
                make_player_ref(
                    id=int(r.get("player_id", 0) or 0),
                    name=str(r.get("name", "") or ""),
                    positions=str(r.get("positions", "") or ""),
                    mlb_id=r.get("mlb_id"),
                    team_abbr=r.get("team"),
                )
                for r in matches.head(limit).to_dict("records")
            ]
            return PlayerSearchResponse(query=query, results=results)
        except Exception as exc:
            logger.warning("RosterQueryService.search(%r) failed: %s", query, exc)
            return PlayerSearchResponse(query=query, results=[])

    def league_rosters(self) -> LeagueRostersResponse:
        try:
            from src.database import load_league_rosters, load_player_pool

            lr = load_league_rosters()
            if lr is None or lr.empty or "team_name" not in lr.columns or "player_id" not in lr.columns:
                return LeagueRostersResponse(teams=[])
            # league_rosters is SELECT * (no name/positions cols) — name/positions/mlb_id/team
            # all come from the pool via player_ref_from_pool (keyed by player_id).
            pool = load_player_pool()
            managers = self._managers()

            teams: list[LeagueRosterTeam] = []
            for team_name, group in lr.groupby("team_name"):
                players = []
                for r in group.to_dict("records"):
                    try:
                        pid = int(r.get("player_id", 0) or 0)
                    except (TypeError, ValueError):
                        continue
                    if pid == 0:
                        continue
                    players.append(player_ref_from_pool(pid, pool))
                teams.append(
                    LeagueRosterTeam(
                        team_name=str(team_name), manager=managers.get(str(team_name), ""), players=players
                    )
                )
            return LeagueRostersResponse(teams=teams)
        except Exception as exc:
            logger.warning("RosterQueryService.league_rosters failed: %s", exc)
            return LeagueRostersResponse(teams=[])

    @staticmethod
    def _managers() -> dict[str, str]:
        """{team_name: manager_name} from league_teams; empty on failure."""
        try:
            from src.database import get_connection

            conn = get_connection()
            try:
                rows = conn.execute("SELECT team_name, manager_name FROM league_teams").fetchall()
            finally:
                conn.close()
            return {str(t): str(m) for t, m in rows if t and m}
        except Exception:
            return {}

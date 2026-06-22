"""Player-id router (search + league rosters). THIN — delegates to the service."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.auth import require_login
from api.contracts.players import LeagueRostersResponse, PlayerSearchResponse
from api.deps import get_roster_query_service

router = APIRouter(prefix="/api", tags=["players"], dependencies=[Depends(require_login)])


@router.get("/players/search", response_model=PlayerSearchResponse)
def search_players(q: str, limit: int = 25, service=Depends(get_roster_query_service)) -> PlayerSearchResponse:
    return service.search(q, limit)


@router.get("/league/rosters", response_model=LeagueRostersResponse)
def league_rosters(service=Depends(get_roster_query_service)) -> LeagueRostersResponse:
    return service.league_rosters()

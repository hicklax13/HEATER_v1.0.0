"""Standings router. THIN — delegates to the service."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.auth import require_login
from api.contracts.standings import StandingsResponse
from api.deps import get_standings_service

router = APIRouter(prefix="/api", tags=["standings"], dependencies=[Depends(require_login)])


@router.get("/standings", response_model=StandingsResponse)
def get_standings(service=Depends(get_standings_service)) -> StandingsResponse:
    return service.get_standings()

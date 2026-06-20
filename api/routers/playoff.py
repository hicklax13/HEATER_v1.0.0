"""Playoff-odds router. THIN — delegates to the service."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.contracts.playoff import PlayoffOddsResponse
from api.deps import get_playoff_service

router = APIRouter(prefix="/api", tags=["playoff"])


@router.get("/playoff-odds", response_model=PlayoffOddsResponse)
def get_playoff_odds(team_name: str, service=Depends(get_playoff_service)) -> PlayoffOddsResponse:
    return service.get_playoff_odds(team_name)

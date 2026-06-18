"""Matchup router. THIN — delegates to the service."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.contracts.matchup import MatchupResponse
from api.deps import get_matchup_service

router = APIRouter(prefix="/api", tags=["matchup"])


@router.get("/matchup", response_model=MatchupResponse)
def get_matchup(team_name: str = "", service=Depends(get_matchup_service)) -> MatchupResponse:
    return service.get_matchup(team_name)

"""Matchup router. THIN — delegates to the service."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.contracts.matchup import MatchupResponse
from api.deps import get_matchup_service
from api.tenancy import ViewerContext, require_viewer_context

router = APIRouter(prefix="/api", tags=["matchup"])


@router.get("/matchup", response_model=MatchupResponse)
def get_matchup(
    team_name: str = "",
    ctx: ViewerContext = Depends(require_viewer_context),
    service=Depends(get_matchup_service),
) -> MatchupResponse:
    return service.get_matchup(ctx.effective_team(team_name))

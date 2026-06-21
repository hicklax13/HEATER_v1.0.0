"""Punt router. THIN — delegates to the service."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.contracts.punt import PuntResponse
from api.deps import get_punt_service
from api.tenancy import ViewerContext, require_viewer_context

router = APIRouter(prefix="/api", tags=["punt"])


@router.get("/punt", response_model=PuntResponse)
def get_punt(
    team_name: str = "",
    ctx: ViewerContext = Depends(require_viewer_context),
    service=Depends(get_punt_service),
) -> PuntResponse:
    return service.get_punt(ctx.effective_team(team_name))

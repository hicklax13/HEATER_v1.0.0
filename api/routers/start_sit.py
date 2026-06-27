"""Start/Sit router. THIN — delegates to the service.

Personalized reads: gated by require_login (matches the other personalized
routers; dormant until Clerk is configured). Team resolution via the viewer
context so the client cannot spoof another team's data."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.auth import require_login
from api.contracts.start_sit import (
    StartSitCompareRequest,
    StartSitCompareResponse,
    StartSitOptimizeRequest,
    StartSitOptimizeResponse,
)
from api.deps import get_start_sit_service
from api.tenancy import ViewerContext, require_viewer_context, resolve_required_team

router = APIRouter(prefix="/api", tags=["start-sit"], dependencies=[Depends(require_login)])


@router.post("/start-sit/compare", response_model=StartSitCompareResponse)
def compare_start_sit(
    req: StartSitCompareRequest,
    ctx: ViewerContext = Depends(require_viewer_context),
    service=Depends(get_start_sit_service),
) -> StartSitCompareResponse:
    req.team_name = resolve_required_team(ctx, req.team_name)
    return service.compare(req)


@router.post("/start-sit/optimize", response_model=StartSitOptimizeResponse)
def optimize_start_sit(
    req: StartSitOptimizeRequest,
    ctx: ViewerContext = Depends(require_viewer_context),
    service=Depends(get_start_sit_service),
) -> StartSitOptimizeResponse:
    req.team_name = resolve_required_team(ctx, req.team_name)
    return service.optimize(req)

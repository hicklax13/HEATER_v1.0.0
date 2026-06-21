"""Playoff-odds router. THIN — delegates to the service."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.contracts.playoff import PlayoffOddsResponse
from api.deps import get_playoff_service
from api.gating import require_pro
from api.tenancy import ViewerContext, require_viewer_context

router = APIRouter(prefix="/api", tags=["playoff"])

# Pro-gated (dormant until Stripe is configured).
_PRO_GATE = {
    402: {"description": "Pro subscription required (when billing is enabled)."},
    401: {"description": "Authentication required (when billing is enabled)."},
}


@router.get(
    "/playoff-odds",
    response_model=PlayoffOddsResponse,
    dependencies=[Depends(require_pro)],
    responses=_PRO_GATE,
)
def get_playoff_odds(
    team_name: str,
    ctx: ViewerContext = Depends(require_viewer_context),
    service=Depends(get_playoff_service),
) -> PlayoffOddsResponse:
    return service.get_playoff_odds(ctx.effective_team(team_name))

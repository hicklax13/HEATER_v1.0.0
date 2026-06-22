"""Lineup router. THIN — delegates to the service."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.contracts.lineup import LineupOptimizeRequest, LineupOptimizeResponse
from api.deps import get_lineup_service
from api.gating import require_pro
from api.tenancy import ViewerContext, require_viewer_context, resolve_required_team

router = APIRouter(prefix="/api", tags=["lineup"])

# Pro-gated (dormant until Stripe is configured). 402 = not Pro; 401 = unauthenticated
# once billing is live. Both only fire when STRIPE_SECRET_KEY is set.
_PRO_GATE = {
    402: {"description": "Pro subscription required (when billing is enabled)."},
    401: {"description": "Authentication required (when billing is enabled)."},
}


@router.post(
    "/lineup/optimize",
    response_model=LineupOptimizeResponse,
    dependencies=[Depends(require_pro)],
    responses=_PRO_GATE,
)
def optimize_lineup(
    req: LineupOptimizeRequest,
    ctx: ViewerContext = Depends(require_viewer_context),
    service=Depends(get_lineup_service),
) -> LineupOptimizeResponse:
    return service.optimize(resolve_required_team(ctx, req.team_name), req.date, req.scope, req.mode)

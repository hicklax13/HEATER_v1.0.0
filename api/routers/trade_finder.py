"""Trade Finder router. THIN — delegates to the service."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.contracts.trade_finder import TradeFinderResponse
from api.deps import get_trade_finder_service
from api.gating import require_pro

router = APIRouter(prefix="/api", tags=["trade-finder"])

# Pro-gated (dormant until Stripe is configured).
_PRO_GATE = {
    402: {"description": "Pro subscription required (when billing is enabled)."},
    401: {"description": "Authentication required (when billing is enabled)."},
}


@router.get(
    "/trade-finder",
    response_model=TradeFinderResponse,
    dependencies=[Depends(require_pro)],
    responses=_PRO_GATE,
)
def get_trade_finder(
    team_name: str = "",
    limit: int = 10,
    service=Depends(get_trade_finder_service),
) -> TradeFinderResponse:
    return service.get_suggestions(team_name=team_name, limit=limit)

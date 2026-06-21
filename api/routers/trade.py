"""Trade Analyzer router. THIN — delegates to the service."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.contracts.trade import TradeEvaluateRequest, TradeEvaluationResponse
from api.deps import get_trade_service
from api.gating import require_pro
from api.tenancy import ViewerContext, require_viewer_context

router = APIRouter(prefix="/api", tags=["trade"])

# Pro-gated (dormant until Stripe is configured).
_PRO_GATE = {
    402: {"description": "Pro subscription required (when billing is enabled)."},
    401: {"description": "Authentication required (when billing is enabled)."},
}


@router.post(
    "/trade/evaluate",
    response_model=TradeEvaluationResponse,
    dependencies=[Depends(require_pro)],
    responses=_PRO_GATE,
)
def evaluate_trade_endpoint(
    req: TradeEvaluateRequest,
    ctx: ViewerContext = Depends(require_viewer_context),
    service=Depends(get_trade_service),
) -> TradeEvaluationResponse:
    return service.evaluate(ctx.effective_team(req.team_name), req.giving_ids, req.receiving_ids, req.enable_mc)

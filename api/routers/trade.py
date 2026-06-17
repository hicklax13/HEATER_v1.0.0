"""Trade Analyzer router. THIN — delegates to the service."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.contracts.trade import TradeEvaluateRequest, TradeEvaluationResponse
from api.deps import get_trade_service

router = APIRouter(prefix="/api", tags=["trade"])


@router.post("/trade/evaluate", response_model=TradeEvaluationResponse)
def evaluate_trade_endpoint(req: TradeEvaluateRequest, service=Depends(get_trade_service)) -> TradeEvaluationResponse:
    return service.evaluate(req.team_name, req.giving_ids, req.receiving_ids, req.enable_mc)

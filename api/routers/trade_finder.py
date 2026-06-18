"""Trade Finder router. THIN — delegates to the service."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.contracts.trade_finder import TradeFinderResponse
from api.deps import get_trade_finder_service

router = APIRouter(prefix="/api", tags=["trade-finder"])


@router.get("/trade-finder", response_model=TradeFinderResponse)
def get_trade_finder(
    team_name: str = "",
    limit: int = 10,
    service=Depends(get_trade_finder_service),
) -> TradeFinderResponse:
    return service.get_suggestions(team_name=team_name, limit=limit)

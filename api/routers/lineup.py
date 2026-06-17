"""Lineup Optimize router. THIN: depends on the service, returns its contract output.
No analytics/category math here (guarded by test_no_logic_in_routers)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.contracts.lineup import LineupOptimizeRequest, LineupOptimizeResponse
from api.deps import get_lineup_service

router = APIRouter(prefix="/api", tags=["lineup"])


@router.post("/lineup/optimize", response_model=LineupOptimizeResponse)
def optimize_lineup(
    body: LineupOptimizeRequest,
    service=Depends(get_lineup_service),
) -> LineupOptimizeResponse:
    return service.optimize(body)

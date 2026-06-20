"""Lineup router. THIN — delegates to the service."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.contracts.lineup import LineupOptimizeRequest, LineupOptimizeResponse
from api.deps import get_lineup_service

router = APIRouter(prefix="/api", tags=["lineup"])


@router.post("/lineup/optimize", response_model=LineupOptimizeResponse)
def optimize_lineup(req: LineupOptimizeRequest, service=Depends(get_lineup_service)) -> LineupOptimizeResponse:
    return service.optimize(req.team_name, req.date, req.scope, req.mode)

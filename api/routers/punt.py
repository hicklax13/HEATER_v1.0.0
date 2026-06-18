"""Punt router. THIN — delegates to the service."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.contracts.punt import PuntResponse
from api.deps import get_punt_service

router = APIRouter(prefix="/api", tags=["punt"])


@router.get("/punt", response_model=PuntResponse)
def get_punt(team_name: str = "", service=Depends(get_punt_service)) -> PuntResponse:
    return service.get_punt(team_name)

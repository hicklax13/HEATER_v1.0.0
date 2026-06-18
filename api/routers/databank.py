"""Databank router. THIN — delegates to the service."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.contracts.databank import DatabankResponse
from api.deps import get_databank_service

router = APIRouter(prefix="/api", tags=["databank"])


@router.get("/databank", response_model=DatabankResponse)
def get_databank(
    player_id: int = 0,
    service=Depends(get_databank_service),
) -> DatabankResponse:
    return service.get_player(player_id=player_id)

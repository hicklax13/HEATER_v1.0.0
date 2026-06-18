"""Free Agents router. THIN — delegates to the service."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.contracts.free_agents import FreeAgentsResponse
from api.deps import get_fa_service

router = APIRouter(prefix="/api", tags=["free-agents"])


@router.get("/free-agents", response_model=FreeAgentsResponse)
def get_free_agents(team_name: str, limit: int = 5, service=Depends(get_fa_service)) -> FreeAgentsResponse:
    return service.get_free_agents(team_name, limit)

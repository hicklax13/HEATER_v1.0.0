"""Free Agents router. THIN: depends on the service, returns its contract output.
No analytics/category math here (guarded by test_no_logic_in_routers)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.contracts.free_agents import FreeAgentsResponse
from api.deps import get_fa_service

router = APIRouter(prefix="/api", tags=["free-agents"])


@router.get("/free-agents", response_model=FreeAgentsResponse)
def get_free_agents(
    team_name: str,
    max_moves: int = 5,
    service=Depends(get_fa_service),
) -> FreeAgentsResponse:
    return service.get_free_agents(team_name, max_moves)

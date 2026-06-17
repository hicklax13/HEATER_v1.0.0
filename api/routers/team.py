"""My Team router. THIN: depends on the service, returns its contract output.
No analytics/category math here (guarded by test_no_logic_in_routers)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.contracts.my_team import MyTeamResponse
from api.deps import get_team_service

router = APIRouter(prefix="/api", tags=["team"])


@router.get("/me/team", response_model=MyTeamResponse)
def get_my_team(team_name: str, service=Depends(get_team_service)) -> MyTeamResponse:
    return service.get_my_team(team_name)

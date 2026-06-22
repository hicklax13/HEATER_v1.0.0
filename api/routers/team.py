"""My Team router. THIN: depends on the service, returns its contract output.
No analytics/category math here (guarded by test_no_logic_in_routers)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.contracts.my_team import MyTeamResponse
from api.deps import get_team_service
from api.tenancy import ViewerContext, require_viewer_context, resolve_required_team

router = APIRouter(prefix="/api", tags=["team"])


@router.get("/me/team", response_model=MyTeamResponse)
def get_my_team(
    team_name: str,
    ctx: ViewerContext = Depends(require_viewer_context),
    service=Depends(get_team_service),
) -> MyTeamResponse:
    return service.get_my_team(resolve_required_team(ctx, team_name))

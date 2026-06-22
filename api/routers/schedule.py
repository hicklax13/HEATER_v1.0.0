"""Schedule router (Probable Pitcher P1 + Hitter Matchup P2). Thin + logic-free:
delegates the 7-day grids to ScheduleService. The 'yours' tag is PERSONALIZED, so
the viewer's team is resolved from their (optional) Clerk identity via
require_viewer_context — NOT the client-supplied team_name (which a user could
spoof) — consistent with the other 7 personalized routers. Dormant when Clerk is
off (the query param is used = today's behavior)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from api.contracts.schedule import HitterMatchupGridResponse, ProbableGridResponse
from api.deps import get_schedule_service
from api.services.schedule_service import ScheduleService
from api.tenancy import ViewerContext, require_viewer_context

router = APIRouter(prefix="/api/schedule", tags=["schedule"])


@router.get("/probables", response_model=ProbableGridResponse)
def probables(
    days: int = Query(7, ge=1, le=14),
    team_name: str | None = Query(None, description="Fallback viewer team (used only when Clerk is off)"),
    svc: ScheduleService = Depends(get_schedule_service),
    ctx: ViewerContext = Depends(require_viewer_context),
) -> ProbableGridResponse:
    return svc.probables(days=days, team_name=ctx.effective_team(team_name))


@router.get("/hitter-matchups", response_model=HitterMatchupGridResponse)
def hitter_matchups(
    days: int = Query(7, ge=1, le=14),
    team_name: str | None = Query(None, description="Fallback viewer team (used only when Clerk is off)"),
    svc: ScheduleService = Depends(get_schedule_service),
    ctx: ViewerContext = Depends(require_viewer_context),
) -> HitterMatchupGridResponse:
    return svc.hitter_matchups(days=days, team_name=ctx.effective_team(team_name))

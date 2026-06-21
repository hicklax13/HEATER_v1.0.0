"""Schedule router (Probable Pitcher P1). Thin + logic-free: delegates the 7-day
probable-pitcher grid to ScheduleService. Read-only, free/ungated (consistent
with /api/streaming)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from api.contracts.schedule import HitterMatchupGridResponse, ProbableGridResponse
from api.deps import get_schedule_service
from api.services.schedule_service import ScheduleService

router = APIRouter(prefix="/api/schedule", tags=["schedule"])


@router.get("/probables", response_model=ProbableGridResponse)
def probables(
    days: int = Query(7, ge=1, le=14),
    team_name: str | None = Query(None, description="Viewer's team — enables the 'yours' tag"),
    svc: ScheduleService = Depends(get_schedule_service),
) -> ProbableGridResponse:
    return svc.probables(days=days, team_name=team_name)


@router.get("/hitter-matchups", response_model=HitterMatchupGridResponse)
def hitter_matchups(
    days: int = Query(7, ge=1, le=14),
    team_name: str | None = Query(None, description="Viewer's team — enables the 'yours' tag"),
    svc: ScheduleService = Depends(get_schedule_service),
) -> HitterMatchupGridResponse:
    return svc.hitter_matchups(days=days, team_name=team_name)

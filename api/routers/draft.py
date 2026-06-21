"""Draft Simulator router. THIN — delegates to the service. No engine imports,
no logic (guarded by tests/api/test_no_logic_in_routers.py)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.contracts.draft import (
    DraftGradeRequest,
    DraftGradeResponse,
    DraftRecommendRequest,
    DraftRecommendResponse,
    DraftSimulatePicksRequest,
    DraftSimulatePicksResponse,
)
from api.deps import get_draft_service
from api.gating import require_pro

router = APIRouter(prefix="/api", tags=["draft"])

# Pro-gated (dormant until Stripe is configured).
_PRO_GATE = {
    402: {"description": "Pro subscription required (when billing is enabled)."},
    401: {"description": "Authentication required (when billing is enabled)."},
}


@router.post(
    "/draft/recommend",
    response_model=DraftRecommendResponse,
    dependencies=[Depends(require_pro)],
    responses=_PRO_GATE,
)
def draft_recommend(req: DraftRecommendRequest, service=Depends(get_draft_service)) -> DraftRecommendResponse:
    return service.recommend(req)


@router.post(
    "/draft/simulate-picks",
    response_model=DraftSimulatePicksResponse,
    dependencies=[Depends(require_pro)],
    responses=_PRO_GATE,
)
def draft_simulate_picks(
    req: DraftSimulatePicksRequest, service=Depends(get_draft_service)
) -> DraftSimulatePicksResponse:
    return service.simulate_picks(req)


@router.post(
    "/draft/grade",
    response_model=DraftGradeResponse,
    dependencies=[Depends(require_pro)],
    responses=_PRO_GATE,
)
def draft_grade(req: DraftGradeRequest, service=Depends(get_draft_service)) -> DraftGradeResponse:
    return service.grade(req)

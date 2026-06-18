"""Draft Simulator router. THIN — delegates to the service. No engine imports,
no logic (guarded by tests/api/test_no_logic_in_routers.py)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.contracts.draft import DraftRecommendRequest, DraftRecommendResponse
from api.deps import get_draft_service

router = APIRouter(prefix="/api", tags=["draft"])


@router.post("/draft/recommend", response_model=DraftRecommendResponse)
def draft_recommend(req: DraftRecommendRequest, service=Depends(get_draft_service)) -> DraftRecommendResponse:
    return service.recommend(req)

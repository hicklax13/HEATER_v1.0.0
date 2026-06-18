"""Roster write-back router — the SINGLE mutation entry point.

THIN: validate → delegate to the service → return its MutationResult. ALL
mutation endpoints live here so auth + Pro-tier gating + audit can attach in one
place at B4. No engine imports, no logic (guarded by
tests/api/test_no_logic_in_routers.py)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.contracts.roster_write import AddDropRequest, LineupSetRequest, MutationResult
from api.deps import get_roster_write_service

router = APIRouter(prefix="/api", tags=["roster-write"])


@router.post("/lineup/set", response_model=MutationResult)
def set_lineup(req: LineupSetRequest, service=Depends(get_roster_write_service)) -> MutationResult:
    return service.set_lineup(req)


@router.post("/transactions/add-drop", response_model=MutationResult)
def add_drop(req: AddDropRequest, service=Depends(get_roster_write_service)) -> MutationResult:
    return service.add_drop(req)

"""Roster write-back router — the SINGLE mutation entry point.

THIN: validate → delegate to the service → return its MutationResult. ALL
mutation endpoints live here so auth + Pro-tier gating + audit attach in one
place. Slice 2 attaches auth (require_principal); cross-team guard lives in the
SERVICE so test_no_logic_in_routers.py stays green. No engine imports, no logic
(guarded by tests/api/test_no_logic_in_routers.py)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.auth import require_principal
from api.contracts.roster_write import AddDropRequest, LineupSetRequest, MutationResult
from api.deps import get_roster_write_service
from api.tenancy import ViewerContext, require_viewer_context

router = APIRouter(prefix="/api", tags=["roster-write"])

# The `dependencies=[require_principal]` gate does not auto-document its 401, so
# we declare it explicitly — the frontend generates its client from openapi.json
# and needs to know these endpoints require (and can reject) a bearer token.
_AUTH_401 = {401: {"description": "Authentication required: missing or invalid bearer token."}}


@router.post(
    "/lineup/set", response_model=MutationResult, dependencies=[Depends(require_principal)], responses=_AUTH_401
)
def set_lineup(
    req: LineupSetRequest,
    ctx: ViewerContext = Depends(require_viewer_context),
    service=Depends(get_roster_write_service),
) -> MutationResult:
    return service.set_lineup(req, caller_team=ctx.team_name)


@router.post(
    "/transactions/add-drop",
    response_model=MutationResult,
    dependencies=[Depends(require_principal)],
    responses=_AUTH_401,
)
def add_drop(
    req: AddDropRequest,
    ctx: ViewerContext = Depends(require_viewer_context),
    service=Depends(get_roster_write_service),
) -> MutationResult:
    return service.add_drop(req, caller_team=ctx.team_name)

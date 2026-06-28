"""Lineup router. THIN — delegates to the service / job store."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from api.contracts.lineup import (
    LineupOptimizeRequest,
    LineupOptimizeResponse,
    OptimizeJobRef,
    OptimizeJobResult,
)
from api.deps import get_lineup_service
from api.gating import require_pro
from api.services import optimize_jobs
from api.tenancy import ViewerContext, require_viewer_context, resolve_required_team

router = APIRouter(prefix="/api", tags=["lineup"])

# Pro-gated (dormant until Stripe is configured). 402 = not Pro; 401 = unauthenticated
# once billing is live. Both only fire when STRIPE_SECRET_KEY is set.
_PRO_GATE = {
    402: {"description": "Pro subscription required (when billing is enabled)."},
    401: {"description": "Authentication required (when billing is enabled)."},
}


@router.post(
    "/lineup/optimize",
    response_model=LineupOptimizeResponse,
    dependencies=[Depends(require_pro)],
    responses=_PRO_GATE,
)
def optimize_lineup(
    req: LineupOptimizeRequest,
    ctx: ViewerContext = Depends(require_viewer_context),
    service=Depends(get_lineup_service),
) -> LineupOptimizeResponse:
    return service.optimize(resolve_required_team(ctx, req.team_name), req.date, req.scope, req.mode, req.depth)


@router.post(
    "/lineup/optimize/start",
    response_model=OptimizeJobRef,
    dependencies=[Depends(require_pro)],
    responses=_PRO_GATE,
)
def start_optimize(
    req: LineupOptimizeRequest,
    background_tasks: BackgroundTasks,
    ctx: ViewerContext = Depends(require_viewer_context),
    service=Depends(get_lineup_service),
) -> OptimizeJobRef:
    """Kick off an optimize in the background (so Enhanced/FA can run ~3min past the
    Vercel gateway timeout) and return a job handle immediately. The client polls
    /lineup/optimize/result/{job_id}. Team resolution mirrors the sync route."""
    team = resolve_required_team(ctx, req.team_name)
    job_id = optimize_jobs.create()
    background_tasks.add_task(optimize_jobs.run_optimize_job, service, job_id, team, req.date, req.scope, req.depth)
    return OptimizeJobRef(job_id=job_id, status="running")


@router.get(
    "/lineup/optimize/result/{job_id}",
    response_model=OptimizeJobResult,
    dependencies=[Depends(require_pro)],
    responses=_PRO_GATE,
)
def optimize_result(job_id: str) -> OptimizeJobResult:
    """Poll a background optimize job. 404 if the job id is unknown/expired."""
    job = optimize_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown optimize job.")
    return OptimizeJobResult(status=job["status"], result=job["result"], error=job["error"])

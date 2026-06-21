"""Admin router (beta user->team assignment). THIN: gate + store calls only.
Reconciliation/auth logic lives in api/admin.py + api/tenancy.py."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from api.admin import TeamNamesProvider, get_team_names_provider, require_admin
from api.contracts.admin import Assignment, AssignmentRequest, AssignmentsResponse
from api.deps import get_league_store, get_membership_store, get_user_store
from api.stores.league_store import LeagueStore
from api.stores.membership_store import MembershipStore
from api.stores.user_store import AppUser, UserStore
from api.tenancy import reconcile_team_name

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.post("/assignments", response_model=Assignment)
def assign_team(
    req: AssignmentRequest,
    admin: AppUser = Depends(require_admin),
    users: UserStore = Depends(get_user_store),
    leagues: LeagueStore = Depends(get_league_store),
    members: MembershipStore = Depends(get_membership_store),
    names_provider: TeamNamesProvider = Depends(get_team_names_provider),
) -> Assignment:
    league = leagues.get_or_create_default() if req.league_id is None else leagues.get(req.league_id)
    if league is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="League not found.")
    names = names_provider()
    canonical = reconcile_team_name(req.team_name, names)
    if canonical is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Unknown team '{req.team_name}'. Choose one of: {', '.join(names)}",
        )
    # When the roster source is cold/empty, reconcile passes the name through
    # UNVALIDATED so cold-start seeding isn't blocked. Make that observable (flag +
    # operator log) so a typo doesn't silently mis-personalize a user later.
    validated = bool(names)
    if not validated:
        logger.warning(
            "Admin assigned team %r UNVALIDATED (roster source cold/empty); "
            "re-check it matches the live team name once data warms.",
            req.team_name,
        )
    target = users.get_or_create(req.clerk_user_id)  # pre-provision (may not have logged in yet)
    m = members.assign(target.id, league.id, canonical, team_key=None, assigned_by=admin.id)
    return Assignment(
        clerk_user_id=req.clerk_user_id,
        user_id=m.user_id,
        league_id=m.league_id,
        team_name=m.team_name,
        validated=validated,
    )


@router.get("/assignments", response_model=AssignmentsResponse)
def list_assignments(
    admin: AppUser = Depends(require_admin),
    leagues: LeagueStore = Depends(get_league_store),
    members: MembershipStore = Depends(get_membership_store),
    names_provider: TeamNamesProvider = Depends(get_team_names_provider),
) -> AssignmentsResponse:
    league = leagues.get_or_create_default()
    rows = members.list_for_league(league.id)
    return AssignmentsResponse(
        assignments=[
            Assignment(clerk_user_id="", user_id=m.user_id, league_id=m.league_id, team_name=m.team_name) for m in rows
        ],
        available_teams=names_provider(),
    )

"""Admin assignment contracts (the beta user->team mapping surface)."""

from __future__ import annotations

from pydantic import BaseModel


class AssignmentRequest(BaseModel):
    clerk_user_id: str
    team_name: str
    league_id: int | None = None  # default = the single beta league


class Assignment(BaseModel):
    clerk_user_id: str
    user_id: int
    league_id: int
    team_name: str
    # False when the team name was accepted UNVALIDATED because the roster source
    # was cold/empty (cold-start seeding). The admin UI can flag it for re-check.
    validated: bool = True


class AssignmentsResponse(BaseModel):
    assignments: list[Assignment]
    available_teams: list[str]

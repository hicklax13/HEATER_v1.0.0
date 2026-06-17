"""Dependency-injection providers. Tests override these via
app.dependency_overrides so they never touch the live data layer."""

from __future__ import annotations

from api.services.team_service import TeamService


def get_team_service() -> TeamService:
    return TeamService()

"""Dependency-injection providers. Tests override these via
app.dependency_overrides so they never touch the live data layer."""

from __future__ import annotations

from api.services.fa_service import FaService
from api.services.lineup_service import LineupService
from api.services.team_service import TeamService


def get_team_service() -> TeamService:
    return TeamService()


def get_fa_service() -> FaService:
    return FaService()


def get_lineup_service() -> LineupService:
    return LineupService()

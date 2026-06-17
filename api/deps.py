"""Dependency-injection providers. Tests override these via
app.dependency_overrides so they never touch the live data layer."""

from __future__ import annotations

from api.services.closers_service import CloserService
from api.services.fa_service import FreeAgentService
from api.services.leaders_service import LeadersService
from api.services.lineup_service import LineupService
from api.services.standings_service import StandingsService
from api.services.team_service import TeamService


def get_team_service() -> TeamService:
    return TeamService()


def get_fa_service() -> FreeAgentService:
    return FreeAgentService()


def get_lineup_service() -> LineupService:
    return LineupService()


def get_standings_service() -> StandingsService:
    return StandingsService()


def get_closer_service() -> CloserService:
    return CloserService()


def get_leaders_service() -> LeadersService:
    return LeadersService()

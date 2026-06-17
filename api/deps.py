"""Dependency-injection providers. Tests override these via
app.dependency_overrides so they never touch the live data layer."""

from __future__ import annotations

from api.services.closers_service import CloserService
from api.services.fa_service import FreeAgentService
from api.services.leaders_service import LeadersService
from api.services.lineup_service import LineupService
from api.services.matchup_service import MatchupService
from api.services.punt_service import PuntService
from api.services.standings_service import StandingsService
from api.services.streaming_service import StreamingService
from api.services.team_service import TeamService
from api.services.trade_finder_service import TradeFinderService
from api.services.trade_service import TradeService


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


def get_matchup_service() -> MatchupService:
    return MatchupService()


def get_streaming_service() -> StreamingService:
    return StreamingService()


def get_punt_service() -> PuntService:
    return PuntService()


def get_trade_service() -> TradeService:
    return TradeService()


def get_trade_finder_service() -> TradeFinderService:
    return TradeFinderService()

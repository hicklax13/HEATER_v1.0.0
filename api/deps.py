"""Dependency-injection providers. Tests override these via
app.dependency_overrides so they never touch the live data layer."""

from __future__ import annotations

import os

from fastapi import Depends

from api.gateways.stripe_gateway import LiveStripeGateway, NullStripeGateway, StripeGateway
from api.services.billing_service import BillingService
from api.services.chat_service import ChatService
from api.services.closers_service import CloserService
from api.services.compare_service import CompareService
from api.services.databank_service import DatabankService
from api.services.draft_service import DraftService
from api.services.fa_pool_service import FreeAgentPoolService
from api.services.fa_service import FreeAgentService
from api.services.leaders_overall_service import LeadersOverallService
from api.services.leaders_service import LeadersService
from api.services.lineup_service import LineupService
from api.services.matchup_service import MatchupService
from api.services.player_detail_service import PlayerDetailService
from api.services.playoff_service import PlayoffService
from api.services.punt_service import PuntService
from api.services.roster_query_service import RosterQueryService
from api.services.roster_write_service import RosterWriteService
from api.services.schedule_service import ScheduleService
from api.services.standings_service import StandingsService
from api.services.streaming_service import StreamingService
from api.services.team_service import TeamService
from api.services.trade_finder_service import TradeFinderService
from api.services.trade_service import TradeService
from api.stores.league_store import LeagueStore, SqliteLeagueStore
from api.stores.membership_store import MembershipStore, SqliteMembershipStore
from api.stores.processed_event_store import ProcessedEventStore, SqliteProcessedEventStore
from api.stores.subscription_store import SqliteSubscriptionStore, SubscriptionStore
from api.stores.user_store import SqliteUserStore, UserStore


def get_team_service() -> TeamService:
    return TeamService()


def get_chat_service() -> ChatService:
    return ChatService()


def get_schedule_service() -> ScheduleService:
    return ScheduleService()


def get_fa_service() -> FreeAgentService:
    return FreeAgentService()


def get_fa_pool_service() -> FreeAgentPoolService:
    return FreeAgentPoolService()


def get_lineup_service() -> LineupService:
    return LineupService()


def get_standings_service() -> StandingsService:
    return StandingsService()


def get_closer_service() -> CloserService:
    return CloserService()


def get_leaders_service() -> LeadersService:
    return LeadersService()


def get_leaders_overall_service() -> LeadersOverallService:
    return LeadersOverallService()


def get_playoff_service() -> PlayoffService:
    return PlayoffService()


def get_roster_query_service() -> RosterQueryService:
    return RosterQueryService()


def get_player_detail_service() -> PlayerDetailService:
    return PlayerDetailService()


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


def get_compare_service() -> CompareService:
    return CompareService()


def get_databank_service() -> DatabankService:
    return DatabankService()


def get_roster_write_service() -> RosterWriteService:
    return RosterWriteService()


def get_draft_service() -> DraftService:
    return DraftService()


def get_user_store() -> UserStore:
    return SqliteUserStore()


def get_league_store() -> LeagueStore:
    return SqliteLeagueStore()


def get_membership_store() -> MembershipStore:
    return SqliteMembershipStore()


def get_stripe_gateway() -> StripeGateway:
    key = os.environ.get("STRIPE_SECRET_KEY", "").strip()
    return LiveStripeGateway(key) if key else NullStripeGateway()


def get_subscription_store() -> SubscriptionStore:
    return SqliteSubscriptionStore()


def get_processed_event_store() -> ProcessedEventStore:
    return SqliteProcessedEventStore()


def get_billing_service(
    gateway: StripeGateway = Depends(get_stripe_gateway),
    store: SubscriptionStore = Depends(get_subscription_store),
    events: ProcessedEventStore = Depends(get_processed_event_store),
) -> BillingService:
    return BillingService(gateway, store, events)

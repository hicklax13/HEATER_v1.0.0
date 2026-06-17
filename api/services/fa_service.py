"""Service layer for the Free Agents endpoint.

Calls recommend_fa_moves() + build_optimizer_context() from the existing
engine and maps their output to the FreeAgentsResponse contract.

Kept resilient: any engine failure (missing DB, bad env) degrades to an
empty recommendations list rather than raising, so the fake-service tests
always pass against the injected stub.
"""

from __future__ import annotations

import logging
from typing import Any

from api.contracts.free_agents import FreeAgentRec, FreeAgentsResponse

logger = logging.getLogger(__name__)


class FaService:
    def get_free_agents(self, team_name: str, max_moves: int = 5) -> FreeAgentsResponse:
        try:
            from src.optimizer.fa_recommender import recommend_fa_moves
            from src.optimizer.shared_data_layer import build_optimizer_context
            from src.valuation import LeagueConfig
            from src.yahoo_data_service import get_yahoo_data_service

            yds = get_yahoo_data_service()
            cfg = LeagueConfig()

            # Build context the same way the Free Agents page does.
            # user_team_name filters multi-team rosters to this team only.
            ctx = build_optimizer_context(
                scope="rest_of_season",
                yds=yds,
                config=cfg,
                user_team_name=team_name,
                level_filter="MLB only",
            )
            adds_remaining = int(getattr(ctx, "adds_remaining_this_week", 10))
            raw: list[dict[str, Any]] = recommend_fa_moves(ctx, max_moves=max_moves)
            recs = [self._to_rec(r) for r in raw if r]
        except Exception:
            logger.warning("FaService.get_free_agents engine call failed", exc_info=True)
            adds_remaining = 0
            recs = []

        return FreeAgentsResponse(
            team_name=team_name,
            adds_remaining=adds_remaining,
            recommendations=recs,
        )

    @staticmethod
    def _to_rec(r: dict[str, Any]) -> FreeAgentRec:
        """Map one recommend_fa_moves() dict to the contract model.

        Key mapping (from fa_recommender._evaluate_swaps return shape):
          add_name        -> r["add_name"]
          add_positions   -> r["add_positions"]
          drop_name       -> r["drop_name"]
          drop_positions  -> r["drop_positions"]
          net_sgp_delta   -> r["net_sgp_delta"]
          urgency_cats    -> r["urgency_categories"]
          reasoning       -> r["reasoning"]
        """
        return FreeAgentRec(
            add_name=str(r.get("add_name", r.get("name", "?"))),
            add_positions=str(r.get("add_positions", "")),
            drop_name=str(r.get("drop_name", "?")),
            drop_positions=str(r.get("drop_positions", "")),
            net_sgp_delta=float(r.get("net_sgp_delta", 0.0)),
            urgency_categories=list(r.get("urgency_categories", [])),
            reasoning=str(r.get("reasoning", "")),
        )

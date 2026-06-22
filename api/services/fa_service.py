"""Free Agents service — the ONE place that calls the FA recommender engine.
Maps engine output → the Free Agents contract. Resilient: missing live data
degrades to an empty recommendation list rather than raising."""

from __future__ import annotations

import logging

from api.contracts.free_agents import FreeAgentRec, FreeAgentsResponse
from api.services.player_ref import player_ref_from_pool

logger = logging.getLogger(__name__)


class FreeAgentService:
    def get_free_agents(self, team_name: str, limit: int = 5) -> FreeAgentsResponse:
        from src.optimizer.fa_recommender import recommend_fa_moves
        from src.optimizer.shared_data_layer import build_optimizer_context
        from src.valuation import LeagueConfig
        from src.yahoo_data_service import get_yahoo_data_service

        recs: list[FreeAgentRec] = []
        try:
            yds = get_yahoo_data_service()
            ctx = build_optimizer_context(
                scope="rest_of_season",
                yds=yds,
                config=LeagueConfig(),
                user_team_name=team_name,
                level_filter="MLB only",
            )
            for move in recommend_fa_moves(ctx, max_moves=limit) or []:
                recs.append(self._to_rec(move, getattr(ctx, "player_pool", None)))
        except Exception as exc:
            logger.warning("FreeAgentService.get_free_agents failed: %s", exc)
            recs = []  # cold env / no data → empty list (page shows EmptyState)
        return FreeAgentsResponse(team_name=team_name, recommendations=recs)

    @staticmethod
    def _to_rec(move, pool=None) -> FreeAgentRec:
        # `move` is the recommender's per-swap dict; map defensively.
        # Engine keys: add_id, add_name, add_positions, drop_id, drop_name,
        #              drop_positions, net_sgp_delta, urgency_categories, reasoning
        g = move.get if isinstance(move, dict) else lambda k, d=None: getattr(move, k, d)
        add_ref = player_ref_from_pool(
            g("add_id", 0),
            pool,
            name=g("add_name", g("name", "")),
            positions=g("add_positions", ""),
        )
        drop_name = g("drop_name", None)
        drop_ref = None
        if drop_name:
            drop_ref = player_ref_from_pool(
                g("drop_id", 0),
                pool,
                name=drop_name,
                positions=g("drop_positions", ""),
            )
        return FreeAgentRec(
            add=add_ref,
            drop=drop_ref,
            marginal_value=float(g("net_sgp_delta", 0.0) or 0.0),
            categories_helped=list(g("urgency_categories", []) or []),
            ownership_pct=g("ownership_pct"),
            rationale=str(g("reasoning", "") or ""),
        )

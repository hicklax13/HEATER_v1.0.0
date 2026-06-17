"""Service layer for the Lineup Optimize endpoint.

Calls LineupOptimizerPipeline from the existing engine and maps its output
to the LineupOptimizeResponse contract.

Kept resilient: any engine failure (missing DB, no PuLP, cold env) degrades
to an empty slots list rather than raising, so fake-service tests always pass.

Engine output mapping:
  result["lineup"]["assignments"]  -> list of {slot, player_name, player_id}
  result["recommendations"]        -> list[str] human-readable tips
"""

from __future__ import annotations

import logging
from typing import Any

from api.contracts.lineup import LineupOptimizeRequest, LineupOptimizeResponse, LineupSlot

logger = logging.getLogger(__name__)


class LineupService:
    def optimize(self, req: LineupOptimizeRequest) -> LineupOptimizeResponse:
        try:
            from src.database import load_player_pool
            from src.optimizer.pipeline import LineupOptimizerPipeline
            from src.valuation import LeagueConfig

            pool = load_player_pool()
            if pool.empty or not req.roster_ids:
                return LineupOptimizeResponse(
                    team_name=req.team_name,
                    mode=req.mode,
                    slots=[],
                    recommendations=["No roster data available."],
                )

            roster = pool[pool["player_id"].isin(req.roster_ids)].copy()
            cfg = LeagueConfig()
            pipeline = LineupOptimizerPipeline(
                roster=roster,
                mode=req.mode,
                weeks_remaining=req.weeks_remaining,
                config=cfg,
            )
            result = pipeline.optimize()
            slots = self._to_slots(result)
            recs: list[str] = list(result.get("recommendations", []))
        except Exception:
            logger.warning("LineupService.optimize engine call failed", exc_info=True)
            slots = []
            recs = []

        return LineupOptimizeResponse(
            team_name=req.team_name,
            mode=req.mode,
            slots=slots,
            recommendations=recs,
        )

    @staticmethod
    def _to_slots(result: dict[str, Any]) -> list[LineupSlot]:
        """Extract slot assignments from the pipeline result.

        The pipeline returns:
          result["lineup"]["assignments"] = [
            {"slot": str, "player_name": str, "player_id": int}, ...
          ]
        Falls back to [] on any missing/malformed structure.
        """
        try:
            lineup = result.get("lineup", {})
            assignments = lineup.get("assignments", [])
            out = []
            for a in assignments:
                out.append(
                    LineupSlot(
                        slot=str(a.get("slot", "")),
                        player_name=str(a.get("player_name", a.get("name", "?"))),
                        player_id=int(a.get("player_id", 0)),
                    )
                )
            return out
        except Exception:
            logger.debug("LineupService._to_slots mapping failed", exc_info=True)
            return []

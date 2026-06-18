"""Lineup service — the ONE place that calls the optimizer pipeline. Maps
its output → the Lineup contract. Resilient: missing data → empty slots.
NOTE: synchronous for B1; becomes an Arq background job in B3."""

from __future__ import annotations

from api.contracts.common import PlayerRef
from api.contracts.lineup import LineupOptimizeResponse, LineupSlot


class LineupService:
    def optimize(self, team_name: str, date=None, scope: str = "rest_of_season") -> LineupOptimizeResponse:
        slots: list[LineupSlot] = []
        summary = ""
        resolved_date = date or ""
        try:
            from src.game_day import get_target_game_date
            from src.optimizer.pipeline import LineupOptimizerPipeline
            from src.valuation import LeagueConfig
            from src.yahoo_data_service import get_yahoo_data_service

            resolved_date = date or str(get_target_game_date())
            yds = get_yahoo_data_service()
            roster = yds.get_rosters()
            pipeline = LineupOptimizerPipeline(roster, mode="standard", config=LeagueConfig())
            result = pipeline.run() if hasattr(pipeline, "run") else None
            slots = self._to_slots(result)
            summary = f"{sum(1 for s in slots if s.action == 'START')} starters set."
        except Exception:
            summary = "Lineup unavailable (no live data in this environment)."
        return LineupOptimizeResponse(team_name=team_name, date=resolved_date, slots=slots, summary=summary)

    @staticmethod
    def _to_slots(result) -> list[LineupSlot]:
        # `result` shape is the integration seam; map defensively. Return [] if absent.
        rows = []
        if result is None:
            return rows
        lineup = getattr(result, "lineup", None) or (result.get("lineup") if isinstance(result, dict) else None) or []
        for r in lineup:
            g = r.get if isinstance(r, dict) else lambda k, d=None: getattr(r, k, d)
            rows.append(
                LineupSlot(
                    slot=str(g("slot", "") or ""),
                    player=PlayerRef(
                        id=int(g("player_id", 0) or 0),
                        name=str(g("player_name", "") or ""),
                        positions=str(g("positions", "") or ""),
                    ),
                    action="START" if g("action", "START") in ("START", "start", True) else "SIT",
                    projected=float(g("projected", 0.0) or 0.0),
                    forced_start=bool(g("forced_start", False)),
                    reason=g("reason"),
                )
            )
        return rows

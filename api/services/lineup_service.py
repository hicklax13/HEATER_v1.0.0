"""Lineup service — the ONE place that calls the optimizer pipeline. Maps
its output → the Lineup contract. Resilient: missing data → empty slots.
NOTE: synchronous for B1; becomes an Arq background job in B3."""

from __future__ import annotations

import logging
import math

from api.contracts.lineup import CatImpact, LineupOptimizeResponse, LineupSlot
from api.services.player_ref import player_ref_from_pool

logger = logging.getLogger(__name__)

# Yahoo slots that are NOT in the active lineup (a player here is benched/stashed).
_NON_LINEUP_SLOTS = {"BN", "IL", "IL10", "IL15", "IL60", "NA", "DTD", "", "BENCH"}


class LineupService:
    def optimize(self, team_name: str, date=None, scope: str = "rest_of_season") -> LineupOptimizeResponse:
        slots: list[LineupSlot] = []
        bench: list[LineupSlot] = []
        impact: list[CatImpact] = []
        optimal = False
        summary = "Lineup unavailable (no live data in this environment)."
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
            # The pipeline method is optimize() (NOT run()); it returns an OptimizerResult
            # dict whose "lineup" is {assignments, bench, projected_stats, status}.
            result = pipeline.optimize() if hasattr(pipeline, "optimize") else None
            lineup = (result or {}).get("lineup") if isinstance(result, dict) else None

            pool = None
            try:
                from src.database import load_player_pool

                pool = load_player_pool()
            except Exception:
                pool = None

            slots, bench = self._to_slots(lineup, pool, roster)
            impact = self._impact((lineup or {}).get("projected_stats") if isinstance(lineup, dict) else None)
            optimal = self._optimal(roster, {s.player.id for s in slots if s.player.id})
            if slots:
                summary = f"{len(slots)} starters set."
        except Exception as exc:
            logger.warning("LineupService.optimize failed: %s", exc)
        return LineupOptimizeResponse(
            team_name=team_name,
            date=resolved_date,
            slots=slots,
            summary=summary,
            bench=bench,
            optimal=optimal,
            impact=impact,
        )

    @staticmethod
    def _to_slots(lineup, pool=None, roster=None) -> tuple[list[LineupSlot], list[LineupSlot]]:
        """(starters, bench). Starters from lineup['assignments']; bench = roster players
        not in the optimal lineup (player_id-based, so no fragile name lookup)."""
        import pandas as pd

        if not isinstance(lineup, dict):
            return [], []
        assignments = lineup.get("assignments") or []
        starters: list[LineupSlot] = []
        starter_ids: set[int] = set()
        for a in assignments:
            g = a.get if isinstance(a, dict) else (lambda k, d=None, _a=a: getattr(_a, k, d))
            try:
                pid = int(g("player_id", 0) or 0)
            except (TypeError, ValueError):
                pid = 0
            starter_ids.add(pid)
            starters.append(
                LineupSlot(
                    slot=str(g("slot", "") or ""),
                    player=player_ref_from_pool(pid, pool, name=g("player_name", ""), positions=g("positions", "")),
                    action="START",
                    projected=0.0,  # per-player value is daily-mode (DCV) → slice 2
                    status="start",
                )
            )

        bench: list[LineupSlot] = []
        if isinstance(roster, pd.DataFrame) and not roster.empty and "player_id" in roster.columns:
            for r in roster.to_dict("records"):
                try:
                    pid = int(r.get("player_id", 0) or 0)
                except (TypeError, ValueError):
                    continue
                if pid == 0 or pid in starter_ids:
                    continue
                slot = str(r.get("selected_position", "") or r.get("roster_slot", "") or "BN")
                bench.append(
                    LineupSlot(
                        slot=slot or "BN",
                        player=player_ref_from_pool(
                            pid, pool, name=r.get("name") or r.get("player_name"), positions=r.get("positions")
                        ),
                        action="SIT",
                        projected=0.0,
                        status="bench",
                    )
                )
        return starters, bench

    @staticmethod
    def _impact(projected_stats) -> list[CatImpact]:
        """Projected category totals for the optimal lineup → display rows. trend flat
        (no current-lineup baseline to diff against — a slice-2 enhancement)."""
        if not isinstance(projected_stats, dict) or not projected_stats:
            return []
        out: list[CatImpact] = []
        for cat, val in projected_stats.items():
            key = str(cat).upper()
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(fval):  # never emit "nan"/"inf" to JSON
                continue
            if key in ("AVG", "OBP"):
                proj = f"{fval:.3f}".lstrip("0") if 0.0 <= fval < 1.0 else f"{fval:.3f}"
            elif key in ("ERA", "WHIP"):
                proj = f"{fval:.2f}"
            else:
                proj = str(int(round(fval)))
            out.append(CatImpact(key=key, proj=proj, trend="flat"))
        return out

    @staticmethod
    def _optimal(roster, starter_ids: set[int]) -> bool:
        """True iff the user's CURRENT Yahoo starters (selected_position in a lineup slot)
        are exactly the optimizer's chosen starters. False if no starters / no selected_position."""
        import pandas as pd

        if not starter_ids or not isinstance(roster, pd.DataFrame) or roster.empty:
            return False
        if "selected_position" not in roster.columns or "player_id" not in roster.columns:
            return False
        current: set[int] = set()
        for r in roster.to_dict("records"):
            sp = str(r.get("selected_position", "") or "").upper().strip()
            if sp in _NON_LINEUP_SLOTS:
                continue
            try:
                current.add(int(r.get("player_id", 0) or 0))
            except (TypeError, ValueError):
                continue
        current.discard(0)
        return bool(current) and current == starter_ids

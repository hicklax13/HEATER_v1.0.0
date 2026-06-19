"""Compare service — the ONE place that loads the player pool and slices it
to the requested player ids. Maps pool rows → the CompareResponse contract.
Resilient: unknown ids are skipped; cold env → empty response."""

from __future__ import annotations

import logging

from api.contracts.compare import ComparePlayer, CompareResponse
from api.services.player_ref import make_player_ref

logger = logging.getLogger(__name__)

# Canonical mapping from display category to pool column (mirrors LeagueConfig.STAT_MAP)
_STAT_MAP: dict[str, str] = {
    "R": "r",
    "HR": "hr",
    "RBI": "rbi",
    "SB": "sb",
    "AVG": "avg",
    "OBP": "obp",
    "W": "w",
    "L": "l",
    "SV": "sv",
    "K": "k",
    "ERA": "era",
    "WHIP": "whip",
}


class CompareService:
    def compare(self, player_ids: list[int]) -> CompareResponse:
        if not player_ids:
            return CompareResponse()
        try:
            from src.database import load_player_pool
            from src.valuation import LeagueConfig

            pool = load_player_pool()
            if pool is None or pool.empty:
                return CompareResponse()

            config = LeagueConfig()
            categories: list[str] = list(config.all_categories)

            players: list[ComparePlayer] = []
            name_col = "player_name" if "player_name" in pool.columns else "name"

            for pid in player_ids:
                row = pool[pool["player_id"] == pid]
                if row.empty:
                    continue
                r = row.iloc[0]
                ref = make_player_ref(
                    id=pid,
                    name=str(r.get(name_col, f"Player {pid}") or f"Player {pid}"),
                    positions=str(r.get("positions", "") or ""),
                    mlb_id=r.get("mlb_id"),
                    team_abbr=r.get("team"),
                )
                stats: dict[str, float] = {}
                for cat in categories:
                    col = _STAT_MAP.get(cat, cat.lower())
                    val = r.get(col)
                    if val is not None:
                        try:
                            import math

                            fv = float(val)
                            stats[cat] = 0.0 if math.isnan(fv) else fv
                        except (TypeError, ValueError):
                            pass
                players.append(ComparePlayer(player=ref, stats=stats))

            return CompareResponse(categories=categories, players=players)

        except Exception as exc:
            logger.warning("CompareService.compare failed: %s", exc)
            return CompareResponse()

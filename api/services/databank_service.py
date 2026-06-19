"""Databank service — the ONE place that calls the player_databank engine.
Fetches multi-season stats for a single player and maps to DatabankResponse.
Resilient: unknown id / cold env → empty seasons, never a 500."""

from __future__ import annotations

import logging
import math

from api.contracts.common import PlayerRef
from api.contracts.databank import DatabankResponse, SeasonStat
from api.services.player_ref import make_player_ref

logger = logging.getLogger(__name__)

# Seasons to surface, newest first
_SEASONS = [2026, 2025, 2024]

# Stat columns to pull from rolling stats output (total view, full season)
_STAT_COLS: dict[str, str] = {
    "R": "r",
    "HR": "hr",
    "RBI": "rbi",
    "SB": "sb",
    "AVG": "avg_calc",
    "OBP": "obp_calc",
    "W": "w",
    "L": "l",
    "SV": "sv",
    "K": "k",
    "ERA": "era_calc",
    "WHIP": "whip_calc",
}


class DatabankService:
    def get_player(self, player_id: int) -> DatabankResponse:
        try:
            from src.database import load_player_pool
            from src.player_databank import compute_rolling_stats

            pool = load_player_pool()
            ref = self._build_ref(player_id, pool)

            seasons: list[SeasonStat] = []
            for year in _SEASONS:
                try:
                    rolled = compute_rolling_stats(
                        player_ids=[player_id],
                        days=None,
                        stat_type="total",
                        season=year,
                    )
                    if rolled.empty:
                        continue
                    row_df = rolled[rolled["player_id"] == player_id]
                    if row_df.empty:
                        continue
                    row = row_df.iloc[0]
                    stats = _extract_stats(row)
                    if stats:
                        seasons.append(SeasonStat(year=year, stats=stats))
                except Exception as exc:
                    logger.debug("DatabankService: season %d failed for player %d: %s", year, player_id, exc)

            return DatabankResponse(player=ref, seasons=seasons)

        except Exception as exc:
            logger.warning("DatabankService.get_player failed for player_id=%d: %s", player_id, exc)
            return DatabankResponse(player=PlayerRef(id=player_id, name=f"Player {player_id}", positions=""))

    # ── private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _build_ref(player_id: int, pool) -> PlayerRef:
        try:
            import pandas as pd

            if isinstance(pool, pd.DataFrame) and not pool.empty:
                row_df = pool[pool["player_id"] == player_id]
                if not row_df.empty:
                    r = row_df.iloc[0]
                    name_col = "player_name" if "player_name" in pool.columns else "name"
                    return make_player_ref(
                        id=player_id,
                        name=str(r.get(name_col, f"Player {player_id}") or f"Player {player_id}"),
                        positions=str(r.get("positions", "") or ""),
                        mlb_id=r.get("mlb_id"),
                        team_abbr=r.get("team"),
                    )
        except Exception:
            pass
        return PlayerRef(id=player_id, name=f"Player {player_id}", positions="")


def _extract_stats(row) -> dict[str, float]:
    """Pull stat values from a rolling-stats row, skipping NaN/None."""
    stats: dict[str, float] = {}
    for display_name, col in _STAT_COLS.items():
        val = row.get(col)
        if val is None:
            continue
        try:
            fv = float(val)
            if not math.isnan(fv):
                stats[display_name] = fv
        except (TypeError, ValueError):
            pass
    return stats

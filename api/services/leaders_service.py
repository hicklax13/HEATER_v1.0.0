"""Leaders service — the ONE place that calls the category leaders engine.
Maps engine output → the Leaders contract. Resilient: missing live data
or unknown category degrades to empty rows rather than raising."""

from __future__ import annotations

import logging

from api.contracts.leaders import LeaderRow, LeadersResponse
from api.services.player_ref import make_player_ref

logger = logging.getLogger(__name__)

# Map display category names to the lowercase stat column in the DataFrame
_STAT_COL_MAP: dict[str, str] = {
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


class LeadersService:
    def get_leaders(self, category: str, limit: int = 25) -> LeadersResponse:
        from src.database import coerce_numeric_df, get_connection
        from src.leaders import compute_category_leaders

        rows: list[LeaderRow] = []
        cat_upper = category.upper()
        try:
            conn = get_connection()
            try:
                import pandas as pd

                df = pd.read_sql_query(
                    """
                    SELECT
                        s.player_id,
                        p.name,
                        p.positions,
                        p.mlb_id,
                        p.team,
                        p.is_hitter,
                        s.pa, s.ip,
                        s.r, s.hr, s.rbi, s.sb,
                        s.avg, s.obp,
                        s.w, s.l, s.sv, s.k,
                        s.era, s.whip
                    FROM season_stats s
                    JOIN players p ON p.player_id = s.player_id
                    WHERE s.season = 2026
                    GROUP BY s.player_id
                    """,
                    conn,
                )
            finally:
                conn.close()

            df = coerce_numeric_df(df)
            if df.empty:
                return LeadersResponse(category=category, rows=[])

            leaders_dict = compute_category_leaders(
                df,
                categories=[cat_upper],
                top_n=limit,
                min_pa=1,
                min_ip=1.0,
            )
            if cat_upper not in leaders_dict or leaders_dict[cat_upper].empty:
                return LeadersResponse(category=category, rows=[])

            ldf = leaders_dict[cat_upper].reset_index(drop=True)
            stat_col = _STAT_COL_MAP.get(cat_upper, cat_upper.lower())

            for rank, (_, row) in enumerate(ldf.iterrows(), start=1):
                rows.append(self._to_leader_row(rank, row, stat_col))
        except Exception as exc:
            logger.warning("LeadersService.get_leaders failed: %s", exc)
            rows = []  # cold env / no data / unknown category → empty rows
        return LeadersResponse(category=category, rows=rows)

    @staticmethod
    def _to_leader_row(rank: int, row, stat_col: str) -> LeaderRow:
        g = row.get if hasattr(row, "get") else lambda k, d=None: row[k] if k in row else d
        raw_val = g(stat_col)
        try:
            value = float(raw_val) if raw_val is not None else 0.0
        except (TypeError, ValueError):
            value = 0.0
        return LeaderRow(
            rank=rank,
            player=make_player_ref(
                id=int(g("player_id", 0) or 0),
                name=str(g("name", "") or ""),
                positions=str(g("positions", "") or ""),
                mlb_id=g("mlb_id"),
                team_abbr=g("team"),
            ),
            value=value,
        )

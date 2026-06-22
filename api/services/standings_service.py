"""Standings service — the ONE place that calls the Yahoo standings engine.
Maps engine output → the Standings contract. Resilient: missing live data
degrades to an empty team list rather than raising."""

from __future__ import annotations

import logging

from api.contracts.standings import StandingsResponse, TeamStanding

logger = logging.getLogger(__name__)


class StandingsService:
    def get_standings(self) -> StandingsResponse:
        from src.yahoo_data_service import get_yahoo_data_service

        teams: list[TeamStanding] = []
        try:
            yds = get_yahoo_data_service()
            df = yds.get_standings()
            if df is None or df.empty:
                return StandingsResponse(teams=[])
            teams = self._build_teams(df)
        except Exception as exc:
            logger.warning("StandingsService.get_standings failed: %s", exc)
            teams = []  # cold env / no data → empty list
        return StandingsResponse(teams=teams)

    @staticmethod
    def _build_teams(df) -> list[TeamStanding]:
        """Aggregate per-(team_name, category) rows into one TeamStanding per team.

        The league_standings table has rows keyed (team_name, category).
        Special categories: "overall" (or the record row) carries wins/losses/ties/points.
        All other categories carry rank.
        """
        import pandas as pd

        result: list[TeamStanding] = []
        # Group by team_name
        team_names = df["team_name"].unique() if "team_name" in df.columns else []
        for team_name in sorted(team_names):
            team_rows = df[df["team_name"] == team_name]

            wins = 0
            losses = 0
            ties = 0
            points = 0.0
            rank = 0
            category_ranks: dict[str, int] = {}

            for _, row in team_rows.iterrows():
                cat = str(row.get("category", "") or "").strip().upper()
                # "OVERALL" row carries the overall rank and record
                if cat in ("OVERALL", "RECORD", ""):
                    rk = row.get("rank")
                    if rk is not None and not (isinstance(rk, float) and pd.isna(rk)):
                        rank = int(rk)
                    pts = row.get("points")
                    if pts is not None and not (isinstance(pts, float) and pd.isna(pts)):
                        points = float(pts)
                    # total may encode wins-losses-ties as "W-L-T" or separate fields
                    total = row.get("total")
                    if total is not None and not (isinstance(total, float) and pd.isna(total)):
                        total_str = str(total)
                        if "-" in total_str:
                            parts = total_str.split("-")
                            if len(parts) >= 2:
                                try:
                                    wins = int(float(parts[0]))
                                    losses = int(float(parts[1]))
                                    ties = int(float(parts[2])) if len(parts) > 2 else 0
                                except (ValueError, IndexError):
                                    pass
                        else:
                            try:
                                wins = int(float(total_str))
                            except ValueError:
                                pass
                else:
                    # Per-category rank
                    rk = row.get("rank")
                    if rk is not None and not (isinstance(rk, float) and pd.isna(rk)):
                        try:
                            category_ranks[cat] = int(rk)
                        except (ValueError, TypeError):
                            pass

            result.append(
                TeamStanding(
                    rank=rank,
                    team_name=str(team_name),
                    wins=wins,
                    losses=losses,
                    ties=ties,
                    points=points,
                    category_ranks=category_ranks,
                )
            )

        # Sort by rank (ascending), then by team_name for ties / rank=0
        result.sort(key=lambda t: (t.rank if t.rank > 0 else 999, t.team_name))
        return result

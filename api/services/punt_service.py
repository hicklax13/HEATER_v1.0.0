"""Punt service — the ONE place that calls the category gap analysis engine.
Maps engine output → the Punt contract. Resilient: missing live data
degrades to an empty categories list rather than raising.

Punt detection logic (per CLAUDE.md):
  A category is a punt if gainable_positions == 0 AND rank >= 10.
  Source: src/engine/portfolio/category_analysis.py::category_gap_analysis()
"""

from __future__ import annotations

from api.contracts.punt import PuntCategory, PuntResponse


class PuntService:
    def get_punt(self, team_name: str) -> PuntResponse:
        categories: list[PuntCategory] = []
        punt_candidates: list[str] = []

        try:
            from src.engine.portfolio.category_analysis import (
                category_gap_analysis,
            )
            from src.standings_utils import get_all_team_totals
            from src.valuation import LeagueConfig

            config = LeagueConfig()
            all_team_totals = get_all_team_totals()

            if not all_team_totals or team_name not in all_team_totals:
                return PuntResponse(team_name=team_name)

            your_totals = all_team_totals[team_name]
            analysis = category_gap_analysis(
                your_totals=your_totals,
                all_team_totals=all_team_totals,
                your_team_id=team_name,
                config=config,
            )
            for cat, info in analysis.items():
                rank = int(info.get("rank", 0) or 0)
                gainable_positions = int(info.get("gainable_positions", 0) or 0)
                gainable = gainable_positions > 0
                is_punt = bool(info.get("is_punt", False))
                if is_punt:
                    recommendation = "Punt"
                elif gainable:
                    recommendation = "Contend"
                else:
                    recommendation = "Hold"
                categories.append(
                    PuntCategory(
                        cat=cat,
                        current_rank=rank,
                        gainable=gainable,
                        recommendation=recommendation,
                    )
                )
                if is_punt:
                    punt_candidates.append(cat)
        except Exception:
            categories = []  # cold env / no data → empty list
            punt_candidates = []

        return PuntResponse(
            team_name=team_name,
            punt_candidates=punt_candidates,
            categories=categories,
        )

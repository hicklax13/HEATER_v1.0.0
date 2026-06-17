"""Matchup service — the ONE place that calls the Yahoo matchup + matchup context engine.
Maps engine output → the Matchup contract. Resilient: missing live data
degrades to an empty categories list rather than raising."""

from __future__ import annotations

from api.contracts.matchup import MatchupCategory, MatchupResponse


class MatchupService:
    def get_matchup(self, team_name: str) -> MatchupResponse:
        from src.yahoo_data_service import get_yahoo_data_service

        categories: list[MatchupCategory] = []
        opponent = ""
        week = 0
        projected_cat_wins = 0.0
        win_prob = 0.0

        try:
            yds = get_yahoo_data_service()
            matchup = yds.get_matchup()
            if matchup is None:
                return MatchupResponse(team_name=team_name)
            week = int(matchup.get("week", 0) or 0)
            opponent = str(matchup.get("opp_name", "") or "")
            raw_cats = matchup.get("categories") or []
            categories = self._build_categories(raw_cats, team_name)
            wins_count = sum(1 for c in categories if self._is_win(c))
            projected_cat_wins = float(wins_count)
            n = len(categories)
            win_prob = round(wins_count / n, 4) if n > 0 else 0.0
        except Exception:
            categories = []  # cold env / no data → empty list

        return MatchupResponse(
            team_name=team_name,
            opponent=opponent,
            week=week,
            projected_cat_wins=projected_cat_wins,
            win_prob=win_prob,
            categories=categories,
        )

    @staticmethod
    def _is_win(cat: MatchupCategory) -> bool:
        """Determine if user is winning this category (you > opp, accounting for inverse)."""
        if cat.inverse:
            return cat.you < cat.opp
        return cat.you > cat.opp

    @staticmethod
    def _build_categories(raw_cats: list, team_name: str) -> list[MatchupCategory]:
        """Map MatchupCategoryEntry dicts → MatchupCategory contract objects."""
        from src.valuation import LeagueConfig

        try:
            cfg = LeagueConfig()
            inverse_stats = set(cfg.inverse_stats)
        except Exception:
            inverse_stats = {"ERA", "WHIP", "L"}

        result: list[MatchupCategory] = []
        for entry in raw_cats:
            if not isinstance(entry, dict):
                continue
            cat = str(entry.get("cat", "") or "").strip()
            if not cat:
                continue
            you_raw = entry.get("you", "0")
            opp_raw = entry.get("opp", "0")
            try:
                you = float(str(you_raw).replace("-", "0") or 0)
            except (TypeError, ValueError):
                you = 0.0
            try:
                opp = float(str(opp_raw).replace("-", "0") or 0)
            except (TypeError, ValueError):
                opp = 0.0
            inverse = cat.upper() in inverse_stats
            # Simple win prob: 1.0 if winning, 0.0 if losing, 0.5 if tied
            if inverse:
                if you < opp:
                    cat_win_prob = 1.0
                elif you > opp:
                    cat_win_prob = 0.0
                else:
                    cat_win_prob = 0.5
            else:
                if you > opp:
                    cat_win_prob = 1.0
                elif you < opp:
                    cat_win_prob = 0.0
                else:
                    cat_win_prob = 0.5
            result.append(
                MatchupCategory(
                    cat=cat,
                    you=you,
                    opp=opp,
                    win_prob=cat_win_prob,
                    inverse=inverse,
                )
            )
        return result

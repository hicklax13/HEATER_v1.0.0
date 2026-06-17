"""The ONE module in the API package allowed to call the engines.

Maps existing engine output -> the My Team contract. Engine calls mirror the
canonical signatures in CLAUDE.md (get_yahoo_data_service, load_player_pool,
resolve_viewer_team_name, MatchupContextService). Kept resilient: any missing
live data degrades to an empty/None field rather than raising."""

from __future__ import annotations

from api.contracts.my_team import CategoryLine, MatchupHero, MyTeamResponse


class TeamService:
    def get_my_team(self, team_name: str) -> MyTeamResponse:
        from src.valuation import LeagueConfig
        from src.yahoo_data_service import get_yahoo_data_service

        yds = get_yahoo_data_service()
        standings = yds.get_standings()  # pd.DataFrame
        rank, record = self._rank_and_record(standings, team_name)
        matchup = self._matchup(yds.get_matchup(), LeagueConfig())
        categories = self._categories(yds.get_matchup(), LeagueConfig())
        return MyTeamResponse(
            team_name=team_name,
            record=record,
            rank=rank,
            matchup=matchup,
            categories=categories,
        )

    @staticmethod
    def _rank_and_record(standings, team_name: str) -> tuple[int, str]:
        # standings carries per-category rows; the WINS category holds the W-L.
        try:
            wins = standings[(standings["team_name"] == team_name) & (standings["category"] == "WINS")]
            rank = int(wins["rank"].iloc[0]) if not wins.empty else 0
            record = str(wins["total"].iloc[0]) if not wins.empty else "0-0-0"
        except Exception:
            rank, record = 0, "0-0-0"
        return rank, record

    @staticmethod
    def _matchup(raw_matchup: dict | None, cfg) -> MatchupHero | None:
        if not raw_matchup:
            return None
        return MatchupHero(
            opponent=str(raw_matchup.get("opponent", "")),
            week=int(raw_matchup.get("week", 0)),
            win_prob=float(raw_matchup.get("win_prob", 0.0)),
            tie_prob=float(raw_matchup.get("tie_prob", 0.0)),
            loss_prob=float(raw_matchup.get("loss_prob", 0.0)),
        )

    @staticmethod
    def _categories(raw_matchup: dict | None, cfg) -> list[CategoryLine]:
        if not raw_matchup or "categories" not in raw_matchup:
            return []
        inverse = set(cfg.inverse_stats)
        out: list[CategoryLine] = []
        for c in raw_matchup["categories"]:
            cat = str(c.get("cat", ""))
            you = float(c.get("you", 0.0))
            opp = float(c.get("opp", 0.0))
            out.append(
                CategoryLine(
                    cat=cat,
                    you=you,
                    opp=opp,
                    edge=you - opp,
                    win_prob=float(c.get("win_prob", 0.0)),
                    inverse=cat in inverse,
                )
            )
        return out

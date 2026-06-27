"""Trade Finder service — the ONE place that calls find_trade_opportunities.
Maps engine output → the TradeFinderResponse contract.
Resilient: cold env / heavy-scan failure → empty suggestions, never a 500."""

from __future__ import annotations

import logging

from api.contracts.common import PlayerRef
from api.contracts.trade_finder import TradeFinderResponse, TradeSuggestion
from api.services.player_ref import make_player_ref

logger = logging.getLogger(__name__)


class TradeFinderService:
    def get_suggestions(self, team_name: str, limit: int = 10) -> TradeFinderResponse:
        try:
            from src.database import load_player_pool
            from src.standings_utils import get_all_team_totals
            from src.trade_finder import find_trade_opportunities
            from src.valuation import LeagueConfig
            from src.yahoo_data_service import get_yahoo_data_service

            pool = load_player_pool()
            if pool is None or pool.empty:
                return TradeFinderResponse(team_name=team_name)

            yds = get_yahoo_data_service()
            rosters_df = yds.get_rosters()
            league_rosters = self._build_league_rosters(rosters_df)
            if not league_rosters:
                return TradeFinderResponse(team_name=team_name)

            # Resolve the user's team against the ACTUAL roster keys (Yahoo keys
            # carry emoji/whitespace, e.g. "🏆 Team Hickey"); a raw .get(team_name)
            # missed them → empty roster → zero suggestions. Use the resolved key
            # name downstream so all_team_totals / find_trade_opportunities agree.
            resolved_team, user_roster_ids = self._resolve_user_roster(team_name, league_rosters)
            if not user_roster_ids:
                return TradeFinderResponse(team_name=team_name)

            # all_team_totals is REQUIRED: find_trade_opportunities early-returns []
            # when it's None (src/trade_finder.py:1895). Compute it (Yahoo-standings-
            # first, projection fallback) and pass it.
            all_team_totals = get_all_team_totals(league_rosters=league_rosters, player_pool=pool)

            raw = find_trade_opportunities(
                user_roster_ids=user_roster_ids,
                player_pool=pool,
                config=LeagueConfig(),
                all_team_totals=all_team_totals or None,
                user_team_name=resolved_team,
                league_rosters=league_rosters,
                max_results=limit,
            )

            suggestions = self._build_suggestions(raw, pool, user_roster_ids)
            return TradeFinderResponse(team_name=team_name, suggestions=suggestions)

        except Exception as exc:
            logger.warning("TradeFinderService.get_suggestions failed: %s", exc)
            return TradeFinderResponse(team_name=team_name)

    # ── private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _resolve_user_roster(team_name: str, league_rosters: dict[str, list[int]]) -> tuple[str, list[int]]:
        """Map the requested team_name to the EXACT roster key (emoji/whitespace
        tolerant), returning (resolved_key, roster_ids). Exact match wins; else a
        normalized match; else ("", []) so the caller returns empty (never a crash,
        never another team's roster)."""
        # Deferred import: api.tenancy → api.deps → this service would cycle at
        # module load. normalize_team_name is a tiny pure fn, cheap to import lazily.
        from api.tenancy import normalize_team_name

        if team_name in league_rosters:
            return team_name, league_rosters[team_name]
        target = normalize_team_name(team_name)
        for key, ids in league_rosters.items():
            if normalize_team_name(key) == target:
                return key, ids
        return "", []

    @staticmethod
    def _build_league_rosters(rosters_df) -> dict[str, list[int]]:
        """Convert rosters DataFrame to {team_name: [player_ids]}."""
        result: dict[str, list[int]] = {}
        if rosters_df is None:
            return result
        try:
            import pandas as pd

            if not isinstance(rosters_df, pd.DataFrame) or rosters_df.empty:
                return result
            team_col = next((c for c in ("team_name", "team") if c in rosters_df.columns), None)
            id_col = next((c for c in ("player_id", "id") if c in rosters_df.columns), None)
            if team_col is None or id_col is None:
                return result
            for team, grp in rosters_df.groupby(team_col):
                result[str(team)] = [int(v) for v in grp[id_col].dropna().tolist()]
        except Exception:
            pass
        return result

    @staticmethod
    def _build_suggestions(raw: list[dict], pool, user_roster_ids: list[int] | None = None) -> list[TradeSuggestion]:
        """Map engine output dicts → TradeSuggestion list."""
        suggestions: list[TradeSuggestion] = []
        for opp in raw:
            giving_ids: list[int] = opp.get("giving_ids", [])
            receiving_ids: list[int] = opp.get("receiving_ids", [])
            partner_team: str = str(opp.get("opponent_team", ""))
            net_sgp: float = float(opp.get("user_sgp_gain", 0.0) or 0.0)
            rationale: str = str(opp.get("rationale", "") or "")

            giving = _build_player_refs(giving_ids, pool)
            receiving = _build_player_refs(receiving_ids, pool)

            suggestions.append(
                TradeSuggestion(
                    partner_team=partner_team,
                    giving=giving,
                    receiving=receiving,
                    net_sgp=net_sgp,
                    rationale=rationale,
                )
            )
        return suggestions


def _build_player_refs(player_ids: list[int], pool) -> list[PlayerRef]:
    refs: list[PlayerRef] = []
    if pool is None:
        return refs
    try:
        import pandas as pd

        if not isinstance(pool, pd.DataFrame) or pool.empty:
            return refs
        name_col = "player_name" if "player_name" in pool.columns else "name"
        for pid in player_ids:
            row = pool[pool["player_id"] == pid]
            if row.empty:
                refs.append(PlayerRef(id=pid, name=f"Player {pid}", positions=""))
            else:
                r = row.iloc[0]
                refs.append(
                    make_player_ref(
                        id=pid,
                        name=str(r.get(name_col, f"Player {pid}") or f"Player {pid}"),
                        positions=str(r.get("positions", "") or ""),
                        mlb_id=r.get("mlb_id"),
                        team_abbr=r.get("team"),
                    )
                )
    except Exception:
        for pid in player_ids:
            refs.append(PlayerRef(id=pid, name=f"Player {pid}", positions=""))
    return refs

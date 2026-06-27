"""Trade Finder service — the ONE place that calls find_trade_opportunities.
Maps engine output → the TradeFinderResponse contract.
Resilient: cold env / heavy-scan failure → empty suggestions, never a 500."""

from __future__ import annotations

import logging

from api.contracts.common import PlayerRef
from api.contracts.trade import CategoryImpact
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
    def _ordinal(n: int) -> str:
        if 10 <= n % 100 <= 20:
            suf = "th"
        else:
            suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suf}"

    @staticmethod
    def _partner_records() -> dict[str, str]:
        """{normalized_team_name: 'W-L-T · Nth'} from load_league_records; {} on any
        failure (records degrade to None, never crash)."""
        from api.tenancy import normalize_team_name

        try:
            from src.database import load_league_records

            df = load_league_records()
            if df is None or df.empty:
                return {}
            out: dict[str, str] = {}
            for _, r in df.iterrows():
                name = str(r.get("team_name", "") or "")
                if not name.strip():
                    continue
                w = int(r.get("wins", 0) or 0)
                loss = int(r.get("losses", 0) or 0)
                t = int(r.get("ties", 0) or 0)
                rank = int(r.get("rank", 0) or 0)
                rec = f"{w}-{loss}-{t}"
                if rank > 0:
                    rec = f"{rec} · {TradeFinderService._ordinal(rank)}"
                out[normalize_team_name(name)] = rec
            return out
        except Exception:
            logger.warning("TradeFinderService._partner_records failed", exc_info=True)
            return {}

    @staticmethod
    def _category_impacts(
        user_roster_ids: list[int],
        giving_ids: list[int],
        receiving_ids: list[int],
        pool,
        config=None,
    ) -> list[CategoryImpact]:
        """Per-category SGP delta of the post-trade roster vs the current roster.
        Cheap totals diff (no MC/LP) — fine for ≤10 cards. Mirrors src.in_season.
        analyze_trade: each cat's delta = totals_sgp({cat: after - before}) via the
        SOLE SGP path (handles inverse-stat signs + denominators correctly). Returns
        [] on any failure or empty roster."""
        if not user_roster_ids:
            return []
        try:
            from src.in_season import _roster_category_totals
            from src.valuation import LeagueConfig, SGPCalculator

            cfg = config or LeagueConfig()
            calc = SGPCalculator(cfg)
            before_ids = list(user_roster_ids)
            give = set(giving_ids)
            after_ids = [pid for pid in before_ids if pid not in give] + list(receiving_ids)
            before = _roster_category_totals(before_ids, pool)
            after = _roster_category_totals(after_ids, pool)
            out: list[CategoryImpact] = []
            for cat in cfg.all_categories:
                b = float(before.get(cat, 0.0) or 0.0)
                a = float(after.get(cat, 0.0) or 0.0)
                # totals_sgp applies the inverse-stat sign (L/ERA/WHIP) + denom, so a
                # raw (after - before) delta becomes a correctly-signed SGP delta.
                delta = calc.totals_sgp({cat: a - b})
                if delta != delta:  # NaN guard
                    continue
                out.append(CategoryImpact(cat=cat, delta=round(delta, 3)))
            return out
        except Exception:
            logger.warning("TradeFinderService._category_impacts failed", exc_info=True)
            return []

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
        from api.tenancy import normalize_team_name

        records = TradeFinderService._partner_records()
        suggestions: list[TradeSuggestion] = []
        for opp in raw:
            giving_ids: list[int] = opp.get("giving_ids", [])
            receiving_ids: list[int] = opp.get("receiving_ids", [])
            partner_team: str = str(opp.get("opponent_team", ""))
            net_sgp: float = float(opp.get("user_sgp_gain", 0.0) or 0.0)
            rationale: str = str(opp.get("rationale", "") or "")
            grade: str = str(opp.get("grade", "") or "")
            partner_record = records.get(normalize_team_name(partner_team))

            giving = _build_player_refs(giving_ids, pool)
            receiving = _build_player_refs(receiving_ids, pool)
            category_impacts = TradeFinderService._category_impacts(
                user_roster_ids or [], giving_ids, receiving_ids, pool
            )

            suggestions.append(
                TradeSuggestion(
                    partner_team=partner_team,
                    partner_record=partner_record,
                    grade=grade,
                    giving=giving,
                    receiving=receiving,
                    net_sgp=net_sgp,
                    category_impacts=category_impacts,
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

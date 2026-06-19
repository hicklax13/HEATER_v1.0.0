"""Free-agent POOL service — ALL available FAs ranked by marginal value
(distinct from the add/drop recommendations in fa_service). Powers the
Players page. Resilient: cold env / no data → empty response.

Composes the existing engine: build_optimizer_context → rank_free_agents
(src.in_season) → map. Player identity reuses player_ref_from_pool; value is
the marginal SGP normalized to 0-100 (top FA = 100)."""

from __future__ import annotations

import logging

from api.contracts.free_agents import FreeAgentPoolItem, FreeAgentPoolResponse, StatItem
from api.services.player_ref import player_ref_from_pool

logger = logging.getLogger(__name__)

# (display label, source column, format kind). Presentation choice; rate stats
# go through format_stat, counting stats render as integers.
_HITTER_STATS = (("HR", "ytd_hr", "int"), ("SB", "ytd_sb", "int"), ("AVG", "ytd_avg", "AVG"))
_PITCHER_STATS = (("K", "ytd_k", "int"), ("ERA", "ytd_era", "ERA"), ("WHIP", "ytd_whip", "WHIP"))


def _tag_from(regression_flag) -> str | None:
    flag = str(regression_flag or "").upper()
    if flag == "BUY_LOW":
        return "Buy Low"
    if flag == "SELL_HIGH":
        return "Sell High"
    return None


def _top_need(category_gaps: dict) -> str:
    """User's biggest need = the most-negative gap (gap < 0 ⇒ behind). '' if none."""
    if not category_gaps:
        return ""
    return min(category_gaps, key=category_gaps.get)


def _key_stats(row, hitter: bool) -> list[StatItem]:
    from src.ui_shared import format_stat

    g = row.get if hasattr(row, "get") else lambda k, d=None: row[k] if k in row else d
    spec = _HITTER_STATS if hitter else _PITCHER_STATS
    out: list[StatItem] = []
    for label, col, kind in spec:
        raw = g(col)
        try:
            fval = float(raw) if raw is not None else 0.0
        except (TypeError, ValueError):
            fval = 0.0
        value = format_stat(fval, kind) if kind != "int" else str(int(round(fval)))
        out.append(StatItem(label=label, value=value))
    return out


def _to_pool_item(rank: int, row, full_pool, max_value: float) -> FreeAgentPoolItem:
    g = row.get if hasattr(row, "get") else lambda k, d=None: row[k] if k in row else d
    pid = int(g("player_id", 0) or 0)
    hitter = bool(g("is_hitter", True))
    mval = float(g("marginal_value", 0.0) or 0.0)
    value = round(max(0.0, min(100.0, mval / max_value * 100.0)), 1) if max_value and max_value > 0 else 0.0
    return FreeAgentPoolItem(
        player=player_ref_from_pool(pid, full_pool, name=g("player_name"), positions=g("positions")),
        rank=rank,
        value=value,
        own_pct=float(g("percent_owned", 0.0) or 0.0),
        own_delta=0.0,  # ownership-trend delta deferred (gap spec: not in API)
        hitter=hitter,
        stats=_key_stats(row, hitter),
        fit=str(g("best_category", "") or ""),
        tag=_tag_from(g("regression_flag")),
    )


class FreeAgentPoolService:
    def get_free_agents_pool(self, team_name: str, limit: int = 100) -> FreeAgentPoolResponse:
        try:
            from src.in_season import rank_free_agents
            from src.optimizer.shared_data_layer import build_optimizer_context
            from src.valuation import LeagueConfig
            from src.yahoo_data_service import get_yahoo_data_service

            config = LeagueConfig()
            ctx = build_optimizer_context(
                scope="rest_of_season",
                yds=get_yahoo_data_service(),
                config=config,
                user_team_name=team_name,
                level_filter="MLB only",
            )
            top_need = _top_need(ctx.category_gaps)
            if ctx.free_agents is None or ctx.free_agents.empty or ctx.player_pool.empty:
                return FreeAgentPoolResponse(top_need=top_need, free_agents=[])

            ranked = rank_free_agents(ctx.user_roster_ids, ctx.free_agents, ctx.player_pool, config)
            if ranked is None or ranked.empty:
                return FreeAgentPoolResponse(top_need=top_need, free_agents=[])

            # Bring Yahoo ownership % onto the ranked rows (rank_free_agents doesn't carry it).
            if "percent_owned" in ctx.free_agents.columns and "percent_owned" not in ranked.columns:
                ranked = ranked.merge(
                    ctx.free_agents[["player_id", "percent_owned"]].drop_duplicates("player_id"),
                    on="player_id",
                    how="left",
                )
            max_value = float(ranked["marginal_value"].max() or 0.0)
            items = [
                _to_pool_item(i + 1, row, ctx.player_pool, max_value)
                for i, row in enumerate(ranked.head(limit).to_dict("records"))
            ]
            return FreeAgentPoolResponse(top_need=top_need, free_agents=items)
        except Exception as exc:
            logger.warning("FreeAgentPoolService.get_free_agents_pool failed: %s", exc)
            return FreeAgentPoolResponse(top_need="", free_agents=[])

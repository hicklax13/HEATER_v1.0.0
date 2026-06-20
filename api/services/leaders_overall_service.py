"""leaders/overall service — a cross-category, 5-lens overall-value leaderboard
for the Research page (distinct from the per-category /api/leaders). Resilient:
cold env / unknown lens → empty rows. Composes existing src engines; src untouched."""

from __future__ import annotations

import logging
import math

from api.contracts.leaders import LeadersOverallResponse, OverallLeaderRow
from api.services.player_ref import player_ref_from_pool

logger = logging.getLogger(__name__)

# lens -> (tag, trend, note). trend is fixed per lens (momentum implied by the lens).
_LENS_META: dict[str, tuple[str, str, str]] = {
    "overall": ("", "flat", "Overall value across all categories"),
    "hot": ("hot", "up", "Trending hot vs projection"),
    "cold": ("cold", "down", "Cooling off vs projection"),
    "breakout": ("breakout", "up", "Process metrics signal a breakout"),
    "sell": ("sell", "up", "Hot but low sustainability — regression risk"),
}

_HIT_STATS = (("HR", "ytd_hr", "int"), ("R", "ytd_r", "int"), ("AVG", "ytd_avg", "avg"))
_PIT_STATS = (("K", "ytd_k", "int"), ("ERA", "ytd_era", "f2"), ("WHIP", "ytd_whip", "f2"))


def _sf(value, default: float = 0.0) -> float:
    """Finite-float coercion (None/NaN/inf/junk → default)."""
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return default
    return default if (math.isnan(fval) or math.isinf(fval)) else fval


def _norm_value(z) -> float:
    """Overall category_value → 0-100. NaN/inf → 0.0.

    NOTE: `category_value` is a SUM of up to 6 per-category z-scores (one per
    scoring cat), so it ranges to ~±10 — NOT a single z (~±4). Mapping [-10,10]
    keeps the top of the board differentiated instead of saturating at 100.
    """
    try:
        fval = float(z)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(fval) or math.isinf(fval):
        return 0.0
    return round(max(0.0, min(100.0, (fval + 10.0) / 20.0 * 100.0)), 1)


def _norm_delta(d) -> float:
    """trend_delta (~−3..+3) → 0-100. NaN/inf → 0.0."""
    try:
        fval = float(d)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(fval) or math.isinf(fval):
        return 0.0
    return round(max(0.0, min(100.0, (fval + 3.0) / 6.0 * 100.0)), 1)


def _fmt(value, kind: str) -> str:
    fval = _sf(value)
    if kind == "int":
        return str(int(round(fval)))
    if kind == "avg":
        return f"{fval:.3f}"[1:] if 0.0 <= fval < 1.0 else f"{fval:.3f}"  # ".322"
    return f"{fval:.2f}"  # f2


def _overall_stats(row, hitter: bool) -> list[str]:
    g = row.get if hasattr(row, "get") else lambda k, d=None: row[k] if k in row else d
    spec = _HIT_STATS if hitter else _PIT_STATS
    return [f"{_fmt(g(col), kind)} {label}" for label, col, kind in spec]


def _to_overall_row(rank: int, row, pool, lens: str) -> OverallLeaderRow:
    g = row.get if hasattr(row, "get") else lambda k, d=None: getattr(row, k, d)
    try:
        pid = int(g("player_id", 0) or 0)
    except (TypeError, ValueError):
        pid = 0
    tag, trend, note = _LENS_META.get(lens, ("", "flat", ""))
    # Pull is_hitter + stats from the pool row (season actuals); graceful if absent.
    prow = None
    hitter = True
    try:
        import pandas as pd

        if isinstance(pool, pd.DataFrame) and not pool.empty:
            match = pool[pool["player_id"] == pid]
            if not match.empty:
                prow = match.iloc[0]
                ih = prow.get("is_hitter", True)
                hitter = True if (isinstance(ih, float) and math.isnan(ih)) else bool(ih)
    except Exception:
        prow = None
    return OverallLeaderRow(
        rank=rank,
        player=player_ref_from_pool(pid, pool, name=g("name") or g("player_name"), positions=g("positions")),
        value=round(_sf(g("_value")), 1),
        stats=_overall_stats(prow, hitter) if prow is not None else [],
        trend=trend,
        tag=tag,
        note=note,
        hitter=hitter,
    )


class LeadersOverallService:
    _VALID = ("overall", "hot", "cold", "breakout", "sell")

    def get_leaders_overall(self, lens: str = "overall", limit: int = 25) -> LeadersOverallResponse:
        lens = lens if lens in self._VALID else "overall"
        rows: list[OverallLeaderRow] = []
        try:
            ranked, pool = self._ranked(lens, limit)
            if ranked is not None and not ranked.empty:
                rows = [
                    _to_overall_row(i + 1, r, pool, lens) for i, r in enumerate(ranked.head(limit).to_dict("records"))
                ]
        except Exception as exc:
            logger.warning("LeadersOverallService(%s) failed: %s", lens, exc)
            rows = []
        return LeadersOverallResponse(lens=lens, rows=rows)

    def _ranked(self, lens: str, limit: int):
        """Return (ranked_df_with_player_id_and__value, pool). Each branch normalizes
        its raw score into a uniform `_value` ∈ [0,100]."""
        import pandas as pd

        from src.database import load_player_pool

        pool = load_player_pool()
        if pool is None or pool.empty:
            return pd.DataFrame(), pd.DataFrame()

        if lens == "breakout":
            from src.leaders import compute_breakout_scores_batch

            bdf = compute_breakout_scores_batch(pool)
            if bdf is None or bdf.empty:
                return pd.DataFrame(), pool
            bdf = bdf.sort_values("breakout_score", ascending=False).reset_index(drop=True)
            bdf["_value"] = [round(max(0.0, min(100.0, _sf(v))), 1) for v in bdf["breakout_score"]]
            return bdf, pool

        season_stats = self._load_season_stats()
        if season_stats is None or season_stats.empty:
            return pd.DataFrame(), pool

        if lens == "overall":
            from src.leaders import compute_category_value_leaders

            vdf = compute_category_value_leaders(season_stats, top_n=max(limit, 25))
            if vdf is None or vdf.empty:
                return pd.DataFrame(), pool
            vdf = vdf.reset_index(drop=True)
            vdf["_value"] = [_norm_value(v) for v in vdf["category_value"]]
            return vdf, pool

        if lens in ("hot", "cold"):
            from src.trend_tracker import compute_player_trends

            tdf = compute_player_trends(pool, season_stats)
            if tdf is None or tdf.empty:
                return pd.DataFrame(), pool
            want = "HOT" if lens == "hot" else "COLD"
            tdf = tdf[tdf["trend_label"] == want].copy()
            tdf = tdf.sort_values("trend_delta", ascending=(lens == "cold")).reset_index(drop=True)
            tdf["_value"] = [_norm_delta(v) for v in tdf["trend_delta"]]
            return tdf, pool

        # sell
        from src.trend_tracker import detect_sell_high_candidates

        sdf = detect_sell_high_candidates(pool, season_stats)
        if sdf is None or sdf.empty:
            return pd.DataFrame(), pool
        sdf = sdf.sort_values("trend_delta", ascending=False).reset_index(drop=True)
        sdf["_value"] = [_norm_delta(v) for v in sdf["trend_delta"]]
        return sdf, pool

    @staticmethod
    def _load_season_stats():
        """Season-stats frame feeding the value/trend engines — the SAME shape the
        existing leaders_service uses."""
        import pandas as pd

        from src.database import get_connection

        conn = get_connection()
        try:
            return pd.read_sql_query(
                """
                SELECT
                    s.player_id, p.name, p.positions, p.is_hitter,
                    s.pa, s.ip, s.r, s.hr, s.rbi, s.sb, s.avg, s.obp,
                    s.w, s.l, s.sv, s.k, s.era, s.whip
                FROM season_stats s
                JOIN players p ON p.player_id = s.player_id
                WHERE s.season = 2026
                GROUP BY s.player_id
                """,
                conn,
            )
        finally:
            conn.close()

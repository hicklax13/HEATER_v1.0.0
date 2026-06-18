"""Trade service — the ONE place that calls evaluate_trade. Maps its output
→ the Trade contract. Resilient: missing data / cold env → default response.

NOTE: synchronous for B1; MC path becomes an Arq background job in B3."""

from __future__ import annotations

import logging

from api.contracts.common import PlayerRef
from api.contracts.trade import (
    CategoryImpact,
    GradeRange,
    TradeEvaluationResponse,
)

logger = logging.getLogger(__name__)


class TradeService:
    def evaluate(
        self,
        team_name: str,
        giving_ids: list[int],
        receiving_ids: list[int],
        enable_mc: bool = False,
    ) -> TradeEvaluationResponse:
        try:
            from src.database import load_player_pool
            from src.engine.output.trade_evaluator import evaluate_trade
            from src.valuation import LeagueConfig
            from src.yahoo_data_service import get_yahoo_data_service

            pool = load_player_pool()

            # Resolve user roster IDs for the requested team
            yds = get_yahoo_data_service()
            rosters = yds.get_rosters()
            user_roster_ids = self._resolve_roster_ids(rosters, team_name)

            result = evaluate_trade(
                giving_ids=giving_ids,
                receiving_ids=receiving_ids,
                user_roster_ids=user_roster_ids,
                player_pool=pool,
                config=LeagueConfig(),
                enable_mc=enable_mc,
                enable_context=True,
                enable_game_theory=True,
            )

            return self._to_response(result, pool, giving_ids, receiving_ids, enable_mc)

        except Exception as exc:
            logger.warning("TradeService.evaluate failed: %s", exc)
            return TradeEvaluationResponse(
                verdict="Could not evaluate",
                summary=f"Trade evaluation unavailable ({type(exc).__name__}).",
            )

    # ── private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _resolve_roster_ids(rosters, team_name: str) -> list[int]:
        """Return player_id list for the named team from the rosters frame."""
        if rosters is None or (hasattr(rosters, "empty") and rosters.empty):
            return []
        try:
            import pandas as pd

            if isinstance(rosters, pd.DataFrame):
                team_rows = rosters[rosters.get("team_name", rosters.get("team", rosters.iloc[:, 0])) == team_name]
                # Try common column names for team filtering
                for col in ("team_name", "team"):
                    if col in rosters.columns:
                        team_rows = rosters[rosters[col] == team_name]
                        break
                else:
                    return []
                for id_col in ("player_id", "id"):
                    if id_col in team_rows.columns:
                        return [int(v) for v in team_rows[id_col].dropna().tolist()]
        except Exception:
            pass
        return []

    @staticmethod
    def _to_response(
        result: dict,
        pool,
        giving_ids: list[int],
        receiving_ids: list[int],
        enable_mc: bool,
    ) -> TradeEvaluationResponse:
        """Map evaluate_trade output dict → TradeEvaluationResponse."""
        g = result.get if isinstance(result, dict) else lambda k, d=None: getattr(result, k, d)

        # --- grade_range: engine returns {grade, grade_low, grade_high, confidence}
        # contract wants {grade, low, center, high} as floats (SGP values).
        # The engine's grade_range uses letter grades, not SGP floats.
        # Map: center = surplus_sgp, and derive low/high via a simple uncertainty
        # band mirroring _compute_grade_range's default SD=0.8.
        grade_range: GradeRange | None = None
        raw_gr = g("grade_range", None)
        surplus = float(g("surplus_sgp", 0.0) or 0.0)
        if raw_gr is not None:
            _uncertainty_sd = 0.8
            grade_range = GradeRange(
                grade=str(
                    raw_gr.get("grade", g("grade", "")) if isinstance(raw_gr, dict) else getattr(raw_gr, "grade", "")
                ),
                low=round(surplus - _uncertainty_sd, 3),
                center=round(surplus, 3),
                high=round(surplus + _uncertainty_sd, 3),
            )

        # --- category_impacts: engine returns {cat: {delta, before, after, weighted_sgp}}
        cat_impacts: list[CategoryImpact] = []
        raw_ci = g("category_impact", None) or {}
        if isinstance(raw_ci, dict):
            for cat, entry in raw_ci.items():
                if isinstance(entry, dict):
                    delta = float(entry.get("delta", 0.0) or 0.0)
                else:
                    delta = float(entry or 0.0)
                cat_impacts.append(CategoryImpact(cat=str(cat), delta=delta))

        # --- giving / receiving PlayerRefs from pool
        giving_refs = _build_player_refs(giving_ids, pool)
        receiving_refs = _build_player_refs(receiving_ids, pool)

        # --- warnings: reshuffle info + risk_flags + IP-floor text
        warnings: list[str] = []
        risk_flags = g("risk_flags", []) or []
        for flag in risk_flags:
            if flag:
                warnings.append(str(flag))
        reshuffle = g("reshuffle", None)
        if isinstance(reshuffle, dict) and reshuffle.get("reshuffle_pct"):
            pct = reshuffle["reshuffle_pct"]
            if isinstance(pct, (int, float)) and abs(pct) > 1.0:
                warnings.append(f"Lineup reshuffle accounts for {pct:.0f}% of surplus.")
        ip_detail = g("ip_floor_detail", None)
        if isinstance(ip_detail, dict) and ip_detail.get("risk"):
            warnings.append(str(ip_detail["risk"]))

        return TradeEvaluationResponse(
            grade=str(g("grade", "") or ""),
            verdict=str(g("verdict", "") or ""),
            confidence_pct=float(g("confidence_pct", 0.0) or 0.0),
            surplus_sgp=surplus,
            grade_range=grade_range,
            giving=giving_refs,
            receiving=receiving_refs,
            category_impacts=cat_impacts,
            delta_playoff_prob=_opt_float(g("delta_playoff_prob", None)),
            delta_champ_prob=_opt_float(g("delta_champ_prob", None)),
            mc_enabled=enable_mc,
            summary=_build_summary(g),
            warnings=warnings,
        )


# ── module-level helpers ─────────────────────────────────────────────────────


def _build_player_refs(player_ids: list[int], pool) -> list[PlayerRef]:
    """Look up each player id in the pool and return a PlayerRef list."""
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
                    PlayerRef(
                        id=pid,
                        name=str(r.get(name_col, f"Player {pid}") or f"Player {pid}"),
                        positions=str(r.get("positions", "") or ""),
                    )
                )
    except Exception:
        for pid in player_ids:
            refs.append(PlayerRef(id=pid, name=f"Player {pid}", positions=""))
    return refs


def _opt_float(val) -> float | None:
    """Return float or None, never raise."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _build_summary(g) -> str:
    """Build a short summary string from engine output keys."""
    grade = g("grade", "")
    verdict = g("verdict", "")
    surplus = g("surplus_sgp", None)
    if not grade and not verdict:
        return ""
    parts = []
    if verdict:
        parts.append(verdict.capitalize())
    if grade:
        parts.append(f"(Grade {grade})")
    if surplus is not None:
        try:
            parts.append(f"| SGP surplus: {float(surplus):+.2f}")
        except (TypeError, ValueError):
            pass
    return " ".join(parts)

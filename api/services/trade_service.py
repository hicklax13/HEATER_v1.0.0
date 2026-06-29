"""Trade service — evaluates a USER-PROPOSED trade for the Build tab.

The HEADLINE (grade / verdict / surplus / category_impacts / confidence) comes
from an HONEST YTD + need-weighted computation — the SAME "actual 2026 season
stats" valuation the Trade FINDER uses (reusing its shared helpers) — NOT the
projection-based 6-phase engine, whose preseason projections compress an elite
player and a replacement-level one (so it would grade "give Olson+Reynolds for
Moniak" favorably). UNLIKE the Finder, the Build tab evaluates an arbitrary
proposed trade that CAN be a loss, so the grade ladder here is SYMMETRIC.

The projection-based ``evaluate_trade`` is still called for SUPPLEMENTARY context
only (risk flags / reshuffle / IP-floor / playoff deltas / MC confidence), wrapped
so a failure there can't sink the honest headline. Resilient: total failure →
the default 'Could not evaluate' response. Never raises.

NOTE: synchronous for B1; MC path becomes an Arq background job in B3."""

from __future__ import annotations

import logging
import math

from api.contracts.common import PlayerRef
from api.contracts.trade import (
    CategoryImpact,
    GradeRange,
    TradeEvaluationResponse,
)
from api.services.player_ref import make_player_ref
from api.services.trade_finder_service import (
    _REPLACEMENT_SLOT_SGP,
    TradeFinderService,
    _category_need_weights,
)

logger = logging.getLogger(__name__)

# Uncertainty band (SGP) around the headline surplus for grade_range, mirroring the
# engine's _compute_grade_range default SD and the prior _to_response band.
_GRADE_RANGE_SD = 0.8


def _f(value, default: float = 0.0) -> float:
    """Finite-float coercion (None/NaN/inf/junk → default). ``float(v or 0.0)``
    does NOT guard NaN (NaN is truthy), so a NaN engine value (surplus_sgp, a
    category delta, confidence_pct) would serialize as NaN into the response."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return default if (math.isnan(f) or math.isinf(f)) else f


def _grade_symmetric(gain: float) -> str:
    """SYMMETRIC letter grade for a proposed trade's need-weighted SGP gain. Unlike
    the Finder's gain-only ladder (floored at C — it only surfaces gains), a proposed
    trade can LOSE, so this descends through D/F for value giveaways."""
    if gain >= 3.0:
        return "A+"
    if gain >= 2.0:
        return "A"
    if gain >= 1.2:
        return "B+"
    if gain >= 0.5:
        return "B"
    if gain >= -0.5:
        return "C"
    if gain >= -1.5:
        return "D"
    return "F"


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
            from src.standings_utils import get_all_team_totals
            from src.valuation import LeagueConfig
            from src.yahoo_data_service import get_yahoo_data_service

            pool = load_player_pool()

            # Resolve user roster IDs + league rosters for the requested team.
            yds = get_yahoo_data_service()
            rosters = yds.get_rosters()
            user_roster_ids = self._resolve_roster_ids(rosters, team_name)

            # Category-need weights from the user's standing (best-effort: {} → all 1.0).
            all_team_totals: dict = {}
            try:
                league_rosters = self._build_league_rosters(rosters)
                if league_rosters:
                    all_team_totals = get_all_team_totals(league_rosters=league_rosters, player_pool=pool) or {}
            except Exception:
                logger.warning("TradeService.evaluate: all_team_totals unavailable", exc_info=True)
                all_team_totals = {}

            # ── HONEST headline: YTD + need-weighted marginal SGP (NOT the engine) ──
            # Reuse the Finder's cached per-cat resolver (values by ACTUAL 2026 YTD with
            # the small-sample rate guard) and need-weight fn — do not re-implement.
            per_cat = TradeFinderService._player_sgp_lookup(pool)
            cat_net: dict[str, float] = {}
            for pid in receiving_ids:
                for cat, val in per_cat(pid).items():
                    if math.isfinite(val):
                        cat_net[cat] = cat_net.get(cat, 0.0) + val
            for pid in giving_ids:
                for cat, val in per_cat(pid).items():
                    if math.isfinite(val):
                        cat_net[cat] = cat_net.get(cat, 0.0) - val

            slot_credit = max(0, len(giving_ids) - len(receiving_ids)) * _REPLACEMENT_SLOT_SGP
            need_weights = _category_need_weights(all_team_totals, team_name, LeagueConfig())
            weighted_gain = sum(cat_net[c] * need_weights.get(c, 1.0) for c in cat_net) + slot_credit
            weighted_gain = _f(weighted_gain)  # finite-guard the headline

            # ── SUPPLEMENTARY engine context (own try/except: never sinks headline) ──
            engine_extra = self._engine_context(giving_ids, receiving_ids, user_roster_ids, pool, enable_mc)

            return self._build_response(
                cat_net=cat_net,
                weighted_gain=weighted_gain,
                pool=pool,
                giving_ids=giving_ids,
                receiving_ids=receiving_ids,
                enable_mc=enable_mc,
                engine_extra=engine_extra,
            )

        except Exception as exc:
            logger.warning("TradeService.evaluate failed: %s", exc)
            return TradeEvaluationResponse(
                verdict="Could not evaluate",
                summary=f"Trade evaluation unavailable ({type(exc).__name__}).",
            )

    # ── private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _engine_context(
        giving_ids: list[int],
        receiving_ids: list[int],
        user_roster_ids: list[int],
        pool,
        enable_mc: bool,
    ) -> dict:
        """Run the projection-based 6-phase engine for SUPPLEMENTARY context ONLY
        (risk flags / reshuffle / IP-floor / playoff deltas / MC confidence). Its
        grade / verdict / surplus / category_impact are IGNORED — the honest YTD
        headline owns those. Wrapped so any engine failure → {} (headline survives)."""
        try:
            from src.engine.output.trade_evaluator import evaluate_trade
            from src.valuation import LeagueConfig

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
            return result if isinstance(result, dict) else {}
        except Exception:
            logger.warning("TradeService._engine_context failed (headline unaffected)", exc_info=True)
            return {}

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
    def _build_league_rosters(rosters_df) -> dict[str, list[int]]:
        """Convert a rosters DataFrame → {team_name: [player_ids]} for get_all_team_totals
        (need weighting). Empty/missing → {} so need weights default to 1.0. Never raises."""
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
    def _build_response(
        *,
        cat_net: dict[str, float],
        weighted_gain: float,
        pool,
        giving_ids: list[int],
        receiving_ids: list[int],
        enable_mc: bool,
        engine_extra: dict,
    ) -> TradeEvaluationResponse:
        """Assemble the response from the HONEST headline (cat_net + weighted_gain),
        layering the supplementary engine context (warnings / playoff deltas / MC
        confidence) on top. Grade / verdict / surplus / category_impacts are NEVER
        taken from the engine."""
        surplus = round(_f(weighted_gain), 3)
        grade = _grade_symmetric(weighted_gain)

        # verdict from the honest gain.
        if weighted_gain > 0.5:
            verdict = "You win"
        elif weighted_gain < -0.5:
            verdict = "You lose"
        else:
            verdict = "Even / fair value"

        # confidence from magnitude: a clear verdict → high confidence. Optionally
        # overridden by the engine's MC confidence when enable_mc is set.
        confidence_pct = min(95.0, 50.0 + abs(weighted_gain) * 15.0)
        if enable_mc:
            mc_conf = _f(engine_extra.get("confidence_pct"), default=0.0)
            if mc_conf > 0.0:
                confidence_pct = mc_conf

        grade_range = GradeRange(
            grade=grade,
            low=round(surplus - _GRADE_RANGE_SD, 3),
            center=surplus,
            high=round(surplus + _GRADE_RANGE_SD, 3),
        )

        # category_impacts = the honest per-cat marginal deltas (correct signs,
        # realistic YTD magnitudes), finite + non-trivial only.
        cat_impacts: list[CategoryImpact] = [
            CategoryImpact(cat=str(c), delta=round(v, 3))
            for c, v in cat_net.items()
            if math.isfinite(v) and abs(v) >= 0.01
        ]

        giving_refs = _build_player_refs(giving_ids, pool)
        receiving_refs = _build_player_refs(receiving_ids, pool)

        # warnings + playoff deltas: SUPPLEMENTARY engine context (pass-through).
        warnings = _engine_warnings(engine_extra)

        summary = f"{verdict} (Grade {grade}) | SGP surplus: {surplus:+.2f}"

        return TradeEvaluationResponse(
            grade=grade,
            verdict=verdict,
            confidence_pct=round(_f(confidence_pct), 3),
            surplus_sgp=surplus,
            grade_range=grade_range,
            giving=giving_refs,
            receiving=receiving_refs,
            category_impacts=cat_impacts,
            delta_playoff_prob=_opt_float(engine_extra.get("delta_playoff_prob")),
            delta_champ_prob=_opt_float(engine_extra.get("delta_champ_prob")),
            mc_enabled=enable_mc,
            summary=summary,
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


def _opt_float(val) -> float | None:
    """Return float or None, never raise."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _engine_warnings(engine_extra: dict) -> list[str]:
    """Pull SUPPLEMENTARY context warnings (risk flags / reshuffle / IP-floor) out of
    the projection engine's result dict. Best-effort: any odd shape → fewer/no warnings,
    never an exception. These DO NOT affect the honest grade — they're advisory text."""
    warnings: list[str] = []
    if not isinstance(engine_extra, dict):
        return warnings
    try:
        for flag in engine_extra.get("risk_flags", []) or []:
            if flag:
                warnings.append(str(flag))
        reshuffle = engine_extra.get("reshuffle")
        if isinstance(reshuffle, dict) and reshuffle.get("reshuffle_pct"):
            pct = reshuffle["reshuffle_pct"]
            if isinstance(pct, (int, float)) and abs(pct) > 1.0:
                warnings.append(f"Lineup reshuffle accounts for {pct:.0f}% of surplus.")
        ip_detail = engine_extra.get("ip_floor_detail")
        if isinstance(ip_detail, dict) and ip_detail.get("risk"):
            warnings.append(str(ip_detail["risk"]))
    except Exception:
        logger.warning("TradeService._engine_warnings: malformed engine context", exc_info=True)
    return warnings

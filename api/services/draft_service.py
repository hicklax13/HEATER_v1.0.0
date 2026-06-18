"""Draft Simulator service — the ONE place that calls the draft engine.

Stateless: rebuilds DraftState from (config + pick_log) by replaying picks
(mirrors DraftState.load), loads the player pool, runs the recommendation
engine, and maps the result → the Draft contract. Resilient: the snake-draft
clock is always computed from the rebuilt state (pure Python, no DB); only the
pool+MC compute degrades to an empty list. The endpoint never 500s.

NOTE: synchronous for now; becomes an Arq background job in B3 (like
POST /lineup/optimize). `engine`/`pool` params exist for tests to inject fakes;
production passes neither (the real engine + pool are loaded lazily)."""

from __future__ import annotations

from api.contracts.common import PlayerRef
from api.contracts.draft import (
    DraftClock,
    DraftRecommendation,
    DraftRecommendRequest,
    DraftRecommendResponse,
)

_TOP_N_CAP = 50
_SIM_CAP = 1000
_ZERO_CLOCK = DraftClock(current_pick=0, round=0, pick_in_round=0, picking_team_index=0, is_user_turn=False)


class DraftService:
    def recommend(self, req: DraftRecommendRequest, engine=None, pool=None) -> DraftRecommendResponse:
        try:
            ds = self._rebuild_state(req)
        except Exception:
            return DraftRecommendResponse(clock=_ZERO_CLOCK, recommendations=[], summary="Invalid draft state.")
        clock = self._clock(ds)
        try:
            results = self._run_engine(req, ds, engine=engine, pool=pool)
            recs = self._to_recs(results)
            return DraftRecommendResponse(
                clock=clock,
                recommendations=recs,
                summary=f"{len(recs)} recommendation{'s' if len(recs) != 1 else ''} for pick {ds.current_pick + 1}.",
            )
        except Exception:
            return DraftRecommendResponse(
                clock=clock,
                recommendations=[],
                summary="Draft recommendations unavailable (no pool data in this environment).",
            )

    @staticmethod
    def _rebuild_state(req: DraftRecommendRequest):
        from src.draft_state import DraftState

        cfg = req.config
        ds = DraftState(
            num_teams=cfg.num_teams,
            num_rounds=cfg.num_rounds,
            user_team_index=cfg.user_team_index,
            roster_config=cfg.roster_config,
        )
        for p in req.pick_log:
            ds.make_pick(p.player_id, p.player_name, p.positions, team_index=p.team_index)
        return ds

    @staticmethod
    def _clock(ds) -> DraftClock:
        return DraftClock(
            current_pick=ds.current_pick,
            round=ds.current_round,
            pick_in_round=ds.pick_in_round,
            picking_team_index=ds.picking_team_index(),
            is_user_turn=ds.is_user_turn,
        )

    @staticmethod
    def _run_engine(req: DraftRecommendRequest, ds, engine=None, pool=None):
        top_n = max(1, min(req.top_n, _TOP_N_CAP))
        n_sims = max(1, min(req.n_simulations, _SIM_CAP))
        if pool is None:
            from src.database import load_player_pool

            pool = load_player_pool()
        if engine is None:
            from src.draft_engine import DraftRecommendationEngine
            from src.valuation import LeagueConfig

            engine = DraftRecommendationEngine(LeagueConfig(), mode="standard")
        return engine.recommend(pool, ds, top_n=top_n, n_simulations=n_sims)

    @staticmethod
    def _to_recs(results) -> list[DraftRecommendation]:
        out: list[DraftRecommendation] = []
        if results is None or getattr(results, "empty", True):
            return out
        for row in results.to_dict("records"):
            g = row.get
            pid = _i(g("player_id"))
            if pid == 0:
                continue
            rank = g("overall_rank")
            score = g("composite_value")
            psgp = g("mean_sgp")
            out.append(
                DraftRecommendation(
                    player=PlayerRef(
                        id=pid,
                        name=str(g("player_name") or g("name") or ""),
                        positions=str(g("positions") or ""),
                    ),
                    rank=_i(rank if rank is not None else g("rank")),
                    score=_f(score if score is not None else g("combined_score")),
                    projected_sgp=_f(psgp if psgp is not None else g("risk_adjusted_sgp")),
                    confidence=_opt_f(g("confidence")),
                    tag=_opt_s(g("buy_fair_avoid")),
                    reason="",
                )
            )
        return out


def _f(v, d: float = 0.0) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return d
    return d if x != x else x  # x != x is True only for NaN


def _i(v, d: int = 0) -> int:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return d
    return d if x != x else int(x)


def _opt_f(v):
    if v is None:
        return None
    x = _f(v, d=float("nan"))
    return None if x != x else x


def _opt_s(v):
    if v is None:
        return None
    # pandas None in an object column becomes float NaN — treat it as absent
    try:
        if v != v:  # NaN check: NaN != NaN is True
            return None
    except TypeError:
        pass
    s = str(v).strip()
    return s or None

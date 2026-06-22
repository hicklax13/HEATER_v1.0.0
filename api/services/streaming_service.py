"""Streaming service — the ONE place that calls the stream_analyzer engine.
Maps engine output → the Streaming contract. Resilient: missing live data
degrades to an empty candidates list rather than raising."""

from __future__ import annotations

import logging
import math

from api.contracts.streaming import (
    BudgetStrip,
    FactorDetail,
    PitcherScorecard,
    ProbableStarter,
    StreamAnalyzeRequest,
    StreamAnalyzeResponse,
    StreamCandidate,
    StreamComponents,
    StreamingResponse,
)
from api.services.player_ref import make_player_ref

logger = logging.getLogger(__name__)


def _f(value, default: float = 0.0) -> float:
    """Finite-float coercion (None/NaN/inf/junk → default) — keeps NaN AND inf
    out of JSON (both serialize to RFC-8259-invalid tokens)."""
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return default
    return default if (math.isnan(fval) or math.isinf(fval)) else fval


def _pick_top(candidates: list[StreamCandidate]) -> StreamCandidate | None:
    """The #1 actionable stream (the board is score-sorted, so the first actionable wins)."""
    for c in candidates:
        if c.actionable:
            return c
    return None


def _build_budget(ctx) -> BudgetStrip:
    from src.league_rules import WEEKLY_TRANSACTION_LIMIT

    adds_total = int(WEEKLY_TRANSACTION_LIMIT)
    try:
        adds_left = int(getattr(ctx, "adds_remaining_this_week", adds_total))
    except (TypeError, ValueError):
        adds_left = adds_total
    try:
        from src.ip_tracker import WEEKLY_TARGET

        ip_target = float(WEEKLY_TARGET)
    except Exception:
        ip_target = 54.0
    gaps = getattr(ctx, "category_gaps", {}) or {}
    pitching = {"W", "L", "SV", "K", "ERA", "WHIP"}
    cats_in_play = [c for c, gap in gaps.items() if c in pitching and _f(gap, 1.0) <= 0]
    return BudgetStrip(
        adds_left=adds_left,
        adds_total=adds_total,
        ip_pace=0.0,
        ip_target=ip_target,
        cats_in_play=cats_in_play,
    )


def _likelihood_from(confidence) -> str:
    """Map the date-proximity confidence tier → a start-likelihood label (proxy)."""
    return {"HIGH": "confirmed", "MEDIUM": "likely", "LOW": "projected"}.get(str(confidence or "").upper(), "projected")


def _to_probable(row) -> ProbableStarter:
    g = row.get if hasattr(row, "get") else lambda k, d=None: getattr(row, k, d)
    try:
        pid = int(g("player_id", 0) or 0)
    except (TypeError, ValueError):
        pid = 0
    return ProbableStarter(
        player=make_player_ref(
            id=pid,
            name=str(g("player_name", "") or ""),
            positions="SP",
            mlb_id=g("mlb_id"),
            team_abbr=g("team"),
        ),
        team=str(g("team", "") or ""),
        opponent=str(g("opponent", "") or ""),
        is_home=bool(g("is_home", False)),
        pos_group="SP",
        start_likelihood=_likelihood_from(g("confidence")),
    )


def _factors(row) -> list[FactorDetail]:
    """The 6 stream-score factors with registry weights + composed detail strings."""
    from src.optimizer.constants_registry import CONSTANTS_REGISTRY as _CR

    g = row.get if hasattr(row, "get") else lambda k, d=None: getattr(row, k, d)
    comp = g("components", {}) or {}
    opp = str(g("opponent", "") or "")
    wrc, kpct, park = _f(g("opp_wrc_plus")), _f(g("opp_k_pct")), _f(g("park_factor"), 1.0)
    nsgp, wp = _f(g("net_sgp")), _f(g("win_probability"))

    def _w(key: str) -> float:
        try:
            return float(_CR[f"stream_score_w_{key}"].value)
        except Exception:
            return 0.0

    specs = [
        ("matchup", "Matchup", f"vs {opp}: {wrc:.0f} wRC+, {kpct:.0f}% K"),
        ("sgp", "Streaming value", f"{nsgp:+.2f} SGP"),
        ("form", "Recent form", "L14 form vs baseline"),
        ("lineup", "Lineup", "Opposing lineup exposure"),
        ("env", "Environment", f"Park factor {park:.2f}"),
        ("winprob", "Win probability", f"{wp * 100:.0f}% team win prob"),
    ]
    return [FactorDetail(key=k, label=lbl, value=_f(comp.get(k)), weight=_w(k), detail=d) for k, lbl, d in specs]


class StreamingService:
    def get_streaming(self, date: str | None = None, limit: int = 25) -> StreamingResponse:
        from src.game_day import get_target_game_date

        try:
            target_date = date or get_target_game_date()
        except Exception:
            from datetime import UTC, datetime

            target_date = datetime.now(UTC).strftime("%Y-%m-%d")

        candidates: list[StreamCandidate] = []
        top_pick: StreamCandidate | None = None
        budget = BudgetStrip()
        probables: list[ProbableStarter] = []
        try:
            from src.optimizer.shared_data_layer import build_optimizer_context
            from src.optimizer.stream_analyzer import build_stream_board
            from src.valuation import LeagueConfig
            from src.yahoo_data_service import get_yahoo_data_service

            yds = get_yahoo_data_service()
            ctx = build_optimizer_context(
                scope="rest_of_season",
                yds=yds,
                config=LeagueConfig(),
                level_filter="MLB only",
            )
            board = build_stream_board(ctx, target_date)
            if board is not None and not board.empty:
                for rank, (_, row) in enumerate(board.head(limit).iterrows(), start=1):
                    candidates.append(self._to_candidate(row, rank))
            top_pick = _pick_top(candidates)
            budget = _build_budget(ctx)
            try:
                full_board = build_stream_board(ctx, target_date, include_rostered=True)
                if full_board is not None and not full_board.empty:
                    probables = [_to_probable(row) for _, row in full_board.iterrows()]
            except Exception:
                probables = []
        except Exception as exc:
            logger.warning("StreamingService.get_streaming failed: %s", exc)
            candidates = []  # cold env / no data → empty list

        return StreamingResponse(
            date=target_date,
            candidates=candidates,
            top_pick=top_pick,
            budget=budget,
            probables=probables,
        )

    @staticmethod
    def _to_candidate(row, rank: int = 0) -> StreamCandidate:
        g = row.get if hasattr(row, "get") else lambda k, d=None: getattr(row, k, d)
        try:
            pid_int = int(g("player_id", 0) or 0)
        except (TypeError, ValueError):
            pid_int = 0
        comp = g("components", {}) or {}
        ip, k, er = _f(g("expected_ip")), _f(g("expected_k")), _f(g("expected_er"))
        flags = g("risk_flags", []) or []
        try:
            num_starts = int(g("num_starts", 1) or 1)
        except (TypeError, ValueError):
            num_starts = 1  # NaN/junk → default (int(nan) would raise)
        return StreamCandidate(
            player=make_player_ref(
                id=pid_int,
                name=str(g("player_name", "") or ""),
                positions="SP",
                mlb_id=g("mlb_id"),
                team_abbr=g("team"),
            ),
            team=str(g("team", "") or ""),
            opponent=str(g("opponent", "") or ""),
            is_home=bool(g("is_home", False)),
            score=_f(g("stream_score")),
            status=str(g("status", "") or ""),
            confidence=str(g("confidence", "") or ""),
            actionable=bool(g("actionable", True)),
            num_starts=num_starts,
            net_sgp=_f(g("net_sgp")),
            opp_wrc_plus=_f(g("opp_wrc_plus")),
            opp_k_pct=_f(g("opp_k_pct")),
            park=_f(g("park_factor"), 1.0),
            expected_ip=ip,
            expected_k=k,
            expected_er=er,
            win_pct=_f(g("win_probability")),
            own_pct=_f(g("percent_owned")),
            risk_flags=[str(x) for x in flags],
            components=StreamComponents(
                matchup=_f(comp.get("matchup")),
                env=_f(comp.get("env")),
                form=_f(comp.get("form")),
                lineup=_f(comp.get("lineup")),
                sgp=_f(comp.get("sgp")),
                winprob=_f(comp.get("winprob")),
            ),
            expected_line=f"{ip:.1f} IP · {k:.0f} K · {er:.0f} ER",
            rank=rank,
            reason="",
        )

    def analyze_pitcher(self, req: StreamAnalyzeRequest) -> StreamAnalyzeResponse:
        from src.game_day import get_target_game_date

        try:
            date = req.date or str(get_target_game_date())
        except Exception:
            from datetime import UTC, datetime

            date = req.date or datetime.now(UTC).strftime("%Y-%m-%d")
        try:
            from src.optimizer.shared_data_layer import build_optimizer_context
            from src.optimizer.stream_analyzer import build_stream_board
            from src.valuation import LeagueConfig
            from src.yahoo_data_service import get_yahoo_data_service

            ctx = build_optimizer_context(
                scope="rest_of_season",
                yds=get_yahoo_data_service(),
                config=LeagueConfig(),
                level_filter="MLB only",
            )
            board = build_stream_board(ctx, date, include_rostered=True)
            if board is not None and not board.empty:
                match = board[board["player_id"] == req.pitcher_id]
                if not match.empty:
                    row = match.iloc[0]
                    rank = int(match.index[0]) + 1  # board is reset_index'd, so index == 0-based rank
                    cand = self._to_candidate(row, rank)
                    scorecard = PitcherScorecard(**cand.model_dump(), factors=_factors(row))
                    return StreamAnalyzeResponse(found=True, scorecard=scorecard)
        except Exception as exc:
            logger.warning("StreamingService.analyze_pitcher failed: %s", exc)
        return StreamAnalyzeResponse(found=False, scorecard=None)

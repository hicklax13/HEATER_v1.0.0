"""Streaming service — the ONE place that calls the stream_analyzer engine.
Maps engine output → the Streaming contract. Resilient: missing live data
degrades to an empty candidates list rather than raising."""

from __future__ import annotations

from api.contracts.common import PlayerRef
from api.contracts.streaming import StreamCandidate, StreamingResponse


class StreamingService:
    def get_streaming(self, date: str | None = None, limit: int = 25) -> StreamingResponse:
        from src.game_day import get_target_game_date

        try:
            target_date = date or get_target_game_date()
        except Exception:
            from datetime import UTC, datetime

            target_date = datetime.now(UTC).strftime("%Y-%m-%d")

        candidates: list[StreamCandidate] = []
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
                for _, row in board.head(limit).iterrows():
                    candidates.append(self._to_candidate(row))
        except Exception:
            candidates = []  # cold env / no data → empty list

        return StreamingResponse(date=target_date, candidates=candidates)

    @staticmethod
    def _to_candidate(row) -> StreamCandidate:
        g = row.get if hasattr(row, "get") else lambda k, d=None: row.get(k, d) if hasattr(row, "get") else d
        pid = g("player_id", None)
        try:
            pid_int = int(pid) if pid is not None else 0
        except (TypeError, ValueError):
            pid_int = 0
        name = str(g("player_name", "") or "")
        positions = "SP"  # stream board is SP-focused by design
        player = PlayerRef(id=pid_int, name=name, positions=positions)
        return StreamCandidate(
            player=player,
            team=str(g("team", "") or ""),
            opponent=str(g("opponent", "") or ""),
            score=float(g("stream_score", 0.0) or 0.0),
            actionable=bool(g("actionable", True)),
            status=str(g("status", "") or ""),
            reason="",
        )

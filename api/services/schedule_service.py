"""Schedule service — the ONE place that composes the streaming engine into the
7-day probable-pitcher grid, plus the league-tag layer (yours/taken/available).

Resilient: missing live data degrades to an empty grid rather than raising. The
cell/band/availability logic is factored into pure functions so it is unit-tested
DB-free (synthetic board rows + a fake roster map); the engine orchestration is
verified live.
"""

from __future__ import annotations

import math

from api.contracts.schedule import ProbableCell, ProbableGridResponse, ProbableTeamRow
from api.services.player_ref import make_player_ref

# Difficulty bands over the 0-100 streamability score (higher = easier matchup).
_BAND_EASY = 60.0
_BAND_TOUGH = 40.0
_MAX_DAYS = 14


def _f(value, default: float = 0.0) -> float:
    """Finite-float coercion (None/NaN/inf/junk -> default)."""
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return default
    return default if (math.isnan(fval) or math.isinf(fval)) else fval


def _g(row, key, default=None):
    """Accessor that works for a dict OR a pandas Series row."""
    if hasattr(row, "get"):
        return row.get(key, default)
    return getattr(row, key, default)


def _band(difficulty: float) -> str:
    if difficulty >= _BAND_EASY:
        return "easy"
    if difficulty <= _BAND_TOUGH:
        return "tough"
    return "medium"


def _availability(pid: int, roster_map: dict[int, str], user_ids: set[int]) -> tuple[str, str | None]:
    """(availability, rostered_by) for a probable's HEATER player_id."""
    if pid in user_ids:
        return "yours", roster_map.get(pid)
    if pid in roster_map:
        return "taken", roster_map.get(pid)
    return "available", None


def _to_cell(row, roster_map: dict[int, str], user_ids: set[int]) -> ProbableCell:
    try:
        pid = int(_g(row, "player_id", 0) or 0)
    except (TypeError, ValueError):
        pid = 0
    try:
        num_starts = int(_g(row, "num_starts", 1) or 1)
    except (TypeError, ValueError):
        num_starts = 1
    difficulty = _f(_g(row, "stream_score"))
    availability, rostered_by = _availability(pid, roster_map, user_ids)
    return ProbableCell(
        pitcher=make_player_ref(
            id=pid,
            name=str(_g(row, "player_name", "") or ""),
            positions="SP",
            mlb_id=_g(row, "mlb_id"),
            team_abbr=_g(row, "team"),
        ),
        opponent=str(_g(row, "opponent", "") or ""),
        is_home=bool(_g(row, "is_home", False)),
        difficulty=difficulty,
        band=_band(difficulty),
        two_start=num_starts >= 2,
        availability=availability,
        rostered_by=rostered_by,
        status=str(_g(row, "status", "") or ""),
        confidence=str(_g(row, "confidence", "") or ""),
    )


def _assemble_grid(
    boards_by_day: list,
    date_list: list[str],
    roster_map: dict[int, str],
    user_ids: set[int],
) -> ProbableGridResponse:
    """Pure grid assembly. boards_by_day[i] = the iterable of probable rows for
    date_list[i]. Each team gets a cell per day (None on off-days)."""
    cell_map: dict[tuple[str, int], ProbableCell] = {}
    teams: set[str] = set()
    for day_index, rows in enumerate(boards_by_day):
        for row in rows or []:
            team = str(_g(row, "team", "") or "")
            if not team:
                continue
            teams.add(team)
            cell_map[(team, day_index)] = _to_cell(row, roster_map, user_ids)
    team_rows = [
        ProbableTeamRow(team=team, cells=[cell_map.get((team, i)) for i in range(len(date_list))])
        for team in sorted(teams)
    ]
    return ProbableGridResponse(days=date_list, teams=team_rows)


def _date_range(days: int) -> list[str]:
    """[today, ..., today+days-1] as YYYY-MM-DD. today = the engine's target date."""
    from datetime import UTC, datetime, timedelta

    try:
        from src.game_day import get_target_game_date

        start = str(get_target_game_date())[:10]
        base = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC)
    except Exception:
        base = datetime.now(UTC)
    return [(base + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]


class ScheduleService:
    def probables(self, days: int = 7, team_name: str | None = None) -> ProbableGridResponse:
        """team_name (the viewer's team) is REQUIRED for the "yours" tag: with all
        12 teams in the roster frame, build_optimizer_context only populates
        ctx.user_roster_ids when user_team_name is set (the F5 multi-team guard).
        Without it the grid still renders — owned players just show "taken"."""
        days = max(1, min(int(days or 7), _MAX_DAYS))
        date_list = _date_range(days)
        try:
            from src.database import load_league_rosters
            from src.optimizer.shared_data_layer import build_optimizer_context
            from src.optimizer.stream_analyzer import build_stream_board
            from src.valuation import LeagueConfig
            from src.yahoo_data_service import get_yahoo_data_service

            ctx = build_optimizer_context(
                scope="rest_of_season",
                yds=get_yahoo_data_service(),
                config=LeagueConfig(),
                user_team_name=team_name or None,
                level_filter="MLB only",
            )
            roster_map = self._roster_map(load_league_rosters)
            user_ids = {int(p) for p in getattr(ctx, "user_roster_ids", []) or []}

            boards_by_day = []
            for date in date_list:
                try:
                    board = build_stream_board(ctx, date, include_rostered=True)
                    rows = [] if board is None or board.empty else [r for _, r in board.iterrows()]
                except Exception:
                    rows = []
                boards_by_day.append(rows)

            return _assemble_grid(boards_by_day, date_list, roster_map, user_ids)
        except Exception:
            return ProbableGridResponse(days=date_list, teams=[])

    @staticmethod
    def _roster_map(load_league_rosters) -> dict[int, str]:
        """player_id -> team_name across all league teams (never-raise -> {})."""
        try:
            df = load_league_rosters()
            if df is None or df.empty or "player_id" not in df.columns or "team_name" not in df.columns:
                return {}
            out: dict[int, str] = {}
            for _, r in df.iterrows():
                try:
                    out[int(r["player_id"])] = str(r["team_name"])
                except (TypeError, ValueError):
                    continue
            return out
        except Exception:
            return {}

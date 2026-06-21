"""Live in-game box-score lines for the Matchup page (api-only, additive).

For each in-progress/final game on a date, pulls statsapi.boxscore_data and builds
a per-mlb_id map of today's ACTUAL stat line + a status string, so the Matchup page
shows live lines instead of projections while games are on. Short in-process TTL
cache (live data). Never raises — any failure → that game contributes nothing (the
projected line stays as the fallback)."""

from __future__ import annotations

import logging
import math
import time

logger = logging.getLogger(__name__)

_HITTER_KEY = "hitter"
_PITCHER_KEY = "pitcher"
_TTL_SECS = 90
_CACHE: dict[str, tuple[float, dict]] = {}


def reset_cache() -> None:
    """Test hook — clear the TTL cache between tests."""
    _CACHE.clear()


def _f(value, default: float = 0.0) -> float:
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return default
    return default if (math.isnan(fv) or math.isinf(fv)) else fv


def _avg(fval: float) -> str:
    return f"{fval:.3f}"[1:] if 0.0 <= fval < 1.0 else f"{fval:.3f}"


def _hitter_line(bat: dict) -> list[str]:
    h, ab = int(_f(bat.get("hits"))), int(_f(bat.get("atBats")))
    bb = _f(bat.get("baseOnBalls"))
    avg = (h / ab) if ab > 0 else 0.0
    obp = ((h + bb) / (ab + bb)) if (ab + bb) > 0 else 0.0  # boxscore lacks HBP/SF
    return [
        f"{h}/{ab}",
        str(int(_f(bat.get("runs")))),
        str(int(_f(bat.get("homeRuns")))),
        str(int(_f(bat.get("rbi")))),
        str(int(_f(bat.get("stolenBases")))),
        _avg(avg),
        _avg(obp),
    ]


def _pitcher_line(pit: dict) -> list[str]:
    from src.live_stats import _ip_outs_to_decimal

    ip = _ip_outs_to_decimal(pit.get("inningsPitched"))
    note = str(pit.get("note", "") or "").strip()
    # The per-GAME decision is in `note` ("(W, 5)"/"(L, 3)"/"(S, 12)"), not the
    # season wins/losses fields on the player line.
    w = 1 if note.startswith("(W") else 0
    losses = 1 if note.startswith("(L") else 0
    sv = 1 if note.startswith("(S") else 0
    er, bb, h = _f(pit.get("earnedRuns")), _f(pit.get("baseOnBalls")), _f(pit.get("hits"))
    era = (er * 9.0 / ip) if ip > 0 else 0.0
    whip = ((bb + h) / ip) if ip > 0 else 0.0
    return [
        f"{ip:.1f}",
        str(w),
        str(losses),
        str(sv),
        str(int(_f(pit.get("strikeOuts")))),
        f"{era:.2f}",
        f"{whip:.2f}",
    ]


def _state_of(status: str) -> str:
    # Use the SAME canonical sets matchup_service._game_state uses, so a row labeled
    # final/live (and thus override-eligible) is exactly what we fetch a boxscore for.
    from src.game_day import FINAL_GAME_STATUSES, LOCKED_GAME_STATUSES

    s = (status or "").strip().lower()
    if s in FINAL_GAME_STATUSES:
        return "final"
    if s in LOCKED_GAME_STATUSES:
        return "live"
    return "sched"


def _status_str(game: dict, state: str) -> str:
    away, home = _f(game.get("away_score")), _f(game.get("home_score"))
    score = f"{int(away)}-{int(home)}"
    if state == "final":
        return f"Final · {score}"
    inning = f"{game.get('inning_state', '')} {game.get('current_inning', '')}".strip()
    return f"{inning} · {score}" if inning else score


def _has_batting(bat: dict) -> bool:
    return int(_f(bat.get("atBats"))) > 0 or any(_f(bat.get(k)) for k in ("hits", "runs", "baseOnBalls"))


def fetch_live_player_lines(schedule, date_key: str = "", boxscore_fn=None) -> dict[int, dict]:
    """Map mlb_id → {"hitter": [...], "pitcher": [...], "status": str} for every
    live/final game in `schedule`. Cached per date_key for _TTL_SECS. boxscore_fn
    defaults to statsapi.boxscore_data; tests inject a fake (no network)."""
    now = time.monotonic()
    cached = _CACHE.get(date_key)
    if cached is not None and (now - cached[0]) < _TTL_SECS:
        return cached[1]

    if boxscore_fn is None:
        import statsapi

        boxscore_fn = statsapi.boxscore_data

    out: dict[int, dict] = {}
    for game in schedule or []:
        state = _state_of(str(game.get("status", "")))
        if state == "sched":
            continue
        gid = game.get("game_id")
        if not gid:
            continue
        try:
            box = boxscore_fn(gid)
        except Exception as exc:
            logger.debug("live boxscore fetch failed for game %s: %s", gid, type(exc).__name__)
            continue
        status = _status_str(game, state)
        for side in ("home", "away"):
            players = (box.get(side, {}) or {}).get("players", {}) or {}
            for pdata in players.values():
                pid = (pdata.get("person", {}) or {}).get("id")
                if not pid:
                    continue
                stats = pdata.get("stats", {}) or {}
                bat, pit = stats.get("batting", {}) or {}, stats.get("pitching", {}) or {}
                entry = out.setdefault(int(pid), {_HITTER_KEY: [], _PITCHER_KEY: [], "status": status})
                if _has_batting(bat):
                    entry[_HITTER_KEY] = _hitter_line(bat)
                if pit.get("inningsPitched") is not None:
                    entry[_PITCHER_KEY] = _pitcher_line(pit)
    _CACHE[date_key] = (now, out)
    return out

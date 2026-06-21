# Matchup Live In-Game Stats — Build Plan (M1 enrichment)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans / subagent-driven-development. Checkbox steps.

**Goal:** Show today's ACTUAL per-player stat line + a live status string on the Matchup page while games are live/final, replacing the projected line. Builds the deferred follow-up `docs/superpowers/plans/2026-06-19-heater-matchup-live-stats-followup.md`.

**Architecture:** A new api-side `api/services/live_boxscore.py` pulls `statsapi.boxscore_data(game_id)` for each live/final game in the schedule the matchup service already fetches, building a per-`mlb_id` map of today's line + status (short in-process TTL cache, injectable fetcher for tests). The matchup service post-processes its already-built RosterRows: when a player's `mlb_id` is in the live map and the game is live/final, it overrides `MatchPlayer.stats` + `status`. Additive; `src/` untouched (reuses `src.live_stats._ip_outs_to_decimal` for ⅓-inning math); never raises (projected line is the fallback).

**Tech Stack:** statsapi (already imported by the matchup service), pydantic.

---

## Verified statsapi shapes (probed against a real final game, do NOT guess)

- `statsapi.boxscore_data(game_id)["home"|"away"]["players"]["ID{mlb_id}"]` → `{"person": {"id", "fullName"}, "stats": {"batting": {...}, "pitching": {...}}}`.
- **batting** fields: `hits, atBats, runs, homeRuns, rbi, stolenBases, strikeOuts, baseOnBalls, doubles, triples, leftOnBase` (NO hitByPitch / sacFly — per-game OBP approximated as (H+BB)/(AB+BB)).
- **pitching** fields: `inningsPitched` (string "6.1" = 6⅓), `strikeOuts, earnedRuns, hits, baseOnBalls, wins, losses, holds, blownSaves, note`. **Per-game W/L/SV decision is in `note`** (`"(W, 5)"` / `"(L, 3)"` / `"(S, 12)"` / `"(H, 18)"`), NOT the season `wins`/`losses` fields.
- schedule entry already carries `game_id, status, current_inning, inning_state, home_score, away_score`.

---

## Task 1: `live_boxscore.py` — formatters (write the failing test first)

**Files:** Create `tests/api/test_api_live_boxscore.py`, then `api/services/live_boxscore.py`

- [ ] **Step 1: failing test** — formatters + the fetch with an injected fake boxscore_fn (real field names)

```python
from api.services.live_boxscore import fetch_live_player_lines


def _fake_box(_gid):
    # Real boxscore_data shape (verified against statsapi).
    return {
        "home": {
            "players": {
                "ID100": {  # a hitter: 2-for-4, HR, 2 RBI, SB
                    "person": {"id": 100, "fullName": "Bat Man"},
                    "stats": {
                        "batting": {"hits": 2, "atBats": 4, "runs": 1, "homeRuns": 1, "rbi": 2,
                                    "stolenBases": 1, "baseOnBalls": 1, "strikeOuts": 0},
                        "pitching": {},
                    },
                },
                "ID200": {  # a pitcher: 6.1 IP, won, 7 K, 2 ER
                    "person": {"id": 200, "fullName": "Ace Pitcher"},
                    "stats": {
                        "batting": {},
                        "pitching": {"inningsPitched": "6.1", "strikeOuts": 7, "earnedRuns": 2,
                                     "hits": 5, "baseOnBalls": 1, "note": "(W, 5)"},
                    },
                },
            }
        },
        "away": {"players": {}},
    }


_SCHED = [{"game_id": 1, "status": "Final", "inning_state": "", "current_inning": 9,
           "home_score": 5, "away_score": 3, "home_name": "A", "away_name": "B"}]


def test_hitter_line_from_boxscore():
    m = fetch_live_player_lines(_SCHED, date_key="t1", boxscore_fn=_fake_box)
    assert 100 in m
    # [H/AB, R, HR, RBI, SB, AVG, OBP]
    line = m[100]["hitter"]
    assert line[0] == "2/4"
    assert line[1] == "1" and line[2] == "1" and line[3] == "2" and line[4] == "1"
    assert line[5] == ".500"  # 2/4
    assert m[100]["pitcher"] == []


def test_pitcher_line_from_boxscore():
    m = fetch_live_player_lines(_SCHED, date_key="t2", boxscore_fn=_fake_box)
    assert 200 in m
    # [IP, W, L, SV, K, ERA, WHIP] — IP 6.1 = 6.333, W from note "(W,...)"
    line = m[200]["pitcher"]
    assert line[0] == "6.3"  # 6.1 outs -> 6.333 -> "6.3"
    assert line[1] == "1" and line[2] == "0" and line[3] == "0"  # W=1 from note
    assert line[4] == "7"  # K
    # ERA = 2*9/6.333 = 2.84; WHIP = (1+5)/6.333 = 0.95
    assert line[5] == "2.84"
    assert line[6] == "0.95"
    assert m[200]["hitter"] == []


def test_status_string_present():
    m = fetch_live_player_lines(_SCHED, date_key="t3", boxscore_fn=_fake_box)
    assert "Final" in m[100]["status"]


def test_scheduled_games_skipped():
    sched = [{"game_id": 9, "status": "Scheduled", "home_name": "A", "away_name": "B"}]
    assert fetch_live_player_lines(sched, date_key="t4", boxscore_fn=_fake_box) == {}


def test_fetch_failure_is_graceful():
    def _boom(_gid):
        raise RuntimeError("api down")
    m = fetch_live_player_lines(_SCHED, date_key="t5", boxscore_fn=_boom)
    assert m == {}  # never raises; game contributes nothing
```

- [ ] **Step 2: run → FAIL** (`ModuleNotFoundError`)

- [ ] **Step 3: implement `api/services/live_boxscore.py`**

```python
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

_FINAL = {"final", "game over", "completed early"}
_LIVE = {"in progress", "live", "warmup", "delayed", "manager challenge"}


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
    note = str(pit.get("note", "") or "")
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
    s = (status or "").strip().lower()
    if s in _FINAL:
        return "final"
    if s in _LIVE or "in progress" in s:
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
```

- [ ] **Step 4: run → PASS**, then commit:

```bash
git add api/services/live_boxscore.py tests/api/test_api_live_boxscore.py
git commit -m "feat(api): live in-game boxscore line fetcher for Matchup (additive) — M1 enrich"
```

## Task 2: wire the override into the matchup service

**Files:** Modify `api/services/matchup_service.py`, `tests/api/test_api_matchup.py` (add an override test)

- [ ] **Step 1: failing test** — append to `tests/api/test_api_matchup.py` a test that builds two RosterRows with a hitter MatchPlayer (mlb_id 100, state "live") and calls the new `_apply_live_lines([row], [], schedule, "k")` with a fake live map, asserting `mp.stats` is overridden to the live line. (Use the module's `fetch_live_player_lines` monkeypatched, or call `_apply_live_lines` with an injected map.) Run → FAIL.

```python
def test_apply_live_lines_overrides_stats(monkeypatch):
    from api.contracts.common import PlayerRef
    from api.contracts.matchup import MatchPlayer, RosterRow
    from api.services import matchup_service as ms

    mp = MatchPlayer(player=PlayerRef(id=1, mlb_id=100, name="Bat Man", positions="OF"),
                     pos="OF", status="vs SF · In Progress", state="live", stats=["0/0", "0", "0", "0", "0", ".000", ".000"])
    row = RosterRow(slot="OF", you=mp, opp=None)

    monkeypatch.setattr(ms, "fetch_live_player_lines",
                        lambda *a, **k: {100: {"hitter": ["2/4", "1", "1", "2", "1", ".500", ".600"], "pitcher": [], "status": "Top 5 · 3-2"}})
    ms._apply_live_lines([row], [], schedule=[{"game_id": 1, "status": "In Progress"}], date_key="k")
    assert row.you.stats[0] == "2/4"
    assert "Top 5" in row.you.status
```

- [ ] **Step 2: implement** in `api/services/matchup_service.py`:
  - Add near the top: `from api.services.live_boxscore import fetch_live_player_lines`.
  - Add a module function:

```python
def _apply_live_lines(hitters, pitchers, schedule, date_key: str) -> None:
    """Override MatchPlayer.stats + status with today's ACTUAL line for any player
    whose game is live/final. No-op (never raises) on any failure — the projected
    line stays as the fallback."""
    try:
        live = fetch_live_player_lines(schedule, date_key=date_key)
    except Exception:
        return
    if not live:
        return

    def _override(rows, key: str) -> None:
        for row in rows:
            for mp in (row.you, row.opp):
                if mp is None or mp.state not in ("live", "final"):
                    continue
                mid = getattr(mp.player, "mlb_id", None)
                entry = live.get(int(mid)) if mid else None
                if not entry:
                    continue
                line = entry.get(key) or []
                if line:
                    mp.stats = line
                    mp.status = entry.get("status") or mp.status

    _override(hitters, "hitter")
    _override(pitchers, "pitcher")
```

  - In `_build_roster_tables`, AFTER `hitters = _pair_rows(...)` and `pitchers = _pair_rows(...)` are built (before computing totals/returning), insert:

```python
        # Overlay today's live in-game lines (no-op when no games are live/final).
        _apply_live_lines(hitters, pitchers, schedule, date_key=str(game_date))
```

- [ ] **Step 3: run the matchup tests → PASS**, full api suite + lint + openapi (no contract change — only field VALUES change, not the schema; expect no openapi diff).

- [ ] **Step 4: commit**

```bash
git add api/services/matchup_service.py tests/api/test_api_matchup.py
git commit -m "feat(api): Matchup overlays live in-game stat lines when games are live/final — M1 enrich"
```

## Review gate
Dispatch `pr-review-toolkit:code-reviewer` (full context) over the diff — verify the boxscore field mapping against the probed real shape, the note→W/L/SV parse, the IP ⅓-inning math, the cache TTL, and that the override never raises / falls back to projected. Apply findings, then push.

## Out of scope (the follow-up plan's secondary items)
Totals recompute from live lines (mixed projected/live), per-batter opposing pitcher, play-by-play micro-status. Documented; not built here.

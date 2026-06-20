"""Matchup service — the ONE place that calls the Yahoo matchup + matchup context engine.
Maps engine output → the Matchup contract. Resilient: missing live data
degrades to an empty categories list rather than raising."""

from __future__ import annotations

import math

from api.contracts.matchup import MatchPlayer, MatchupCategory, MatchupResponse, RosterRow
from api.services.player_ref import player_ref_from_pool

_HITTER_COLUMNS = ["H/AB", "R", "HR", "RBI", "SB", "AVG", "OBP"]
_PITCHER_COLUMNS = ["IP", "W", "L", "SV", "K", "ERA", "WHIP"]


def _f(value, default: float = 0.0) -> float:
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return default
    return default if (math.isnan(fval) or math.isinf(fval)) else fval


def _avg(value) -> str:
    fval = _f(value)
    return f"{fval:.3f}"[1:] if 0.0 <= fval < 1.0 else f"{fval:.3f}"


def _fmt_hitter_stats(row) -> list[str]:
    g = row.get if hasattr(row, "get") else lambda k, d=None: row[k] if k in row else d
    return [
        f"{int(_f(g('h')))}/{int(_f(g('ab')))}",
        str(int(_f(g("r")))),
        str(int(_f(g("hr")))),
        str(int(_f(g("rbi")))),
        str(int(_f(g("sb")))),
        _avg(g("avg")),
        _avg(g("obp")),
    ]


def _fmt_pitcher_stats(row) -> list[str]:
    g = row.get if hasattr(row, "get") else lambda k, d=None: row[k] if k in row else d
    return [
        f"{_f(g('ip')):.1f}",
        str(int(_f(g("w")))),
        str(int(_f(g("l")))),
        str(int(_f(g("sv")))),
        str(int(_f(g("k")))),
        f"{_f(g('era')):.2f}",
        f"{_f(g('whip')):.2f}",
    ]


def _cat_win(you, opp, inverse: bool) -> str:
    y, o = _f(you), _f(opp)
    if y == o:
        return ""
    higher_you = y > o
    return ("you" if higher_you else "opp") if not inverse else ("opp" if higher_you else "you")


def _date_tabs(week: int) -> list[str]:
    # Live + Totals + the 7 weekday labels (Mon..Sun). Day dates are presentation;
    # the frontend may relabel — keep simple + deterministic.
    return ["Live", "Totals", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _game_state(team_abbr: str, schedule: list, abbr_to_name: dict) -> tuple[str, str]:
    """Map a team abbr → today's game state (sched/live/final/none) + basic status."""
    from src.game_day import FINAL_GAME_STATUSES, LOCKED_GAME_STATUSES

    name = abbr_to_name.get(str(team_abbr).upper(), "")
    for g in schedule or []:
        home, away = str(g.get("home_name", "")), str(g.get("away_name", ""))
        if name and (name == home or name == away):
            status = str(g.get("status", "")).strip()
            low = status.lower()
            if low in FINAL_GAME_STATUSES:
                state = "final"
            elif low in LOCKED_GAME_STATUSES:
                state = "live"
            else:
                state = "sched"
            opp = away if name == home else home
            vs = "vs" if name == home else "@"
            return state, f"{vs} {opp} · {status}" if status else f"{vs} {opp}"
    return "none", ""


def _to_match_player(player_id, slot: str, pool, hitter: bool, state: str, status: str) -> MatchPlayer:
    import pandas as pd

    prow = None
    try:
        if isinstance(pool, pd.DataFrame) and not pool.empty:
            m = pool[pool["player_id"] == player_id]
            if not m.empty:
                prow = m.iloc[0]
    except Exception:
        prow = None
    name = str(prow.get("name", "")) if prow is not None else ""
    stats = (_fmt_hitter_stats(prow) if hitter else _fmt_pitcher_stats(prow)) if prow is not None else []
    return MatchPlayer(
        player=player_ref_from_pool(player_id, pool, name=name, positions=slot),
        pos=slot,
        status=status,
        state=state,
        stats=stats,
        badge=None,
    )


def _pair_rows(you: list, opp: list, slots: list) -> list[RosterRow]:
    rows = []
    n = max(len(you), len(opp), len(slots))
    for i in range(n):
        rows.append(
            RosterRow(
                slot=slots[i] if i < len(slots) else "",
                you=you[i] if i < len(you) else None,
                opp=opp[i] if i < len(opp) else None,
            )
        )
    return rows


class MatchupService:
    def get_matchup(self, team_name: str) -> MatchupResponse:
        from src.yahoo_data_service import get_yahoo_data_service

        categories: list[MatchupCategory] = []
        opponent = ""
        week = 0
        projected_cat_wins = 0.0
        win_prob = 0.0

        try:
            yds = get_yahoo_data_service()
            matchup = yds.get_matchup()
            if matchup is None:
                return MatchupResponse(team_name=team_name)
            week = int(matchup.get("week", 0) or 0)
            opponent = str(matchup.get("opp_name", "") or "")
            raw_cats = matchup.get("categories") or []
            categories = self._build_categories(raw_cats, team_name)
            wins_count = sum(1 for c in categories if self._is_win(c))
            projected_cat_wins = float(wins_count)
            n = len(categories)
            win_prob = round(wins_count / n, 4) if n > 0 else 0.0
        except Exception:
            categories = []  # cold env / no data → empty list

        return MatchupResponse(
            team_name=team_name,
            opponent=opponent,
            week=week,
            projected_cat_wins=projected_cat_wins,
            win_prob=win_prob,
            categories=categories,
        )

    @staticmethod
    def _is_win(cat: MatchupCategory) -> bool:
        """Determine if user is winning this category (you > opp, accounting for inverse)."""
        if cat.inverse:
            return cat.you < cat.opp
        return cat.you > cat.opp

    @staticmethod
    def _build_categories(raw_cats: list, team_name: str) -> list[MatchupCategory]:
        """Map MatchupCategoryEntry dicts → MatchupCategory contract objects."""
        from src.valuation import LeagueConfig

        try:
            cfg = LeagueConfig()
            inverse_stats = set(cfg.inverse_stats)
        except Exception:
            inverse_stats = {"ERA", "WHIP", "L"}

        result: list[MatchupCategory] = []
        for entry in raw_cats:
            if not isinstance(entry, dict):
                continue
            cat = str(entry.get("cat", "") or "").strip()
            if not cat:
                continue
            you_raw = entry.get("you", "0")
            opp_raw = entry.get("opp", "0")
            try:
                you = float(str(you_raw).replace("-", "0") or 0)
            except (TypeError, ValueError):
                you = 0.0
            try:
                opp = float(str(opp_raw).replace("-", "0") or 0)
            except (TypeError, ValueError):
                opp = 0.0
            inverse = cat.upper() in inverse_stats
            # Simple win prob: 1.0 if winning, 0.0 if losing, 0.5 if tied
            if inverse:
                if you < opp:
                    cat_win_prob = 1.0
                elif you > opp:
                    cat_win_prob = 0.0
                else:
                    cat_win_prob = 0.5
            else:
                if you > opp:
                    cat_win_prob = 1.0
                elif you < opp:
                    cat_win_prob = 0.0
                else:
                    cat_win_prob = 0.5
            result.append(
                MatchupCategory(
                    cat=cat,
                    you=you,
                    opp=opp,
                    win_prob=cat_win_prob,
                    inverse=inverse,
                )
            )
        return result

"""Matchup service — the ONE place that calls the Yahoo matchup + matchup context engine.
Maps engine output → the Matchup contract. Resilient: missing live data
degrades to an empty categories list rather than raising."""

from __future__ import annotations

import math

from api.contracts.matchup import MatchPlayer, MatchupCategory, MatchupResponse, RosterRow, SideTotals, TeamSide
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


def _format_record(wins, losses, ties, rank) -> str:
    w, l, t = int(_f(wins)), int(_f(losses)), int(_f(ties))
    r = int(_f(rank))
    if r <= 0:
        return f"{w}-{l}-{t}"
    suffix = "th" if 11 <= (r % 100) <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(r % 10, "th")
    return f"{w}-{l}-{t} · {r}{suffix}"


def _aggregate_totals(rows, hitter: bool) -> list[str]:
    """Aggregate a side's roster stat line: counting stats summed, rate stats
    weighted (AVG=Σh/Σab, OBP=Σ(h+bb+hbp)/Σ(ab+bb+hbp+sf), ERA=Σer·9/Σip,
    WHIP=Σ(bb_allowed+h_allowed)/Σip). NaN/zero-safe."""
    import pandas as pd

    def _sum(col: str) -> float:
        try:
            if isinstance(rows, pd.DataFrame) and col in rows.columns:
                return float(sum(_f(v) for v in rows[col]))
        except Exception:
            pass
        return 0.0

    if hitter:
        h, ab = _sum("h"), _sum("ab")
        bb, hbp, sf = _sum("bb"), _sum("hbp"), _sum("sf")
        obp_den = ab + bb + hbp + sf
        avg = (h / ab) if ab > 0 else 0.0
        obp = ((h + bb + hbp) / obp_den) if obp_den > 0 else 0.0
        return [
            f"{int(h)}/{int(ab)}",
            str(int(_sum("r"))),
            str(int(_sum("hr"))),
            str(int(_sum("rbi"))),
            str(int(_sum("sb"))),
            _avg(avg),
            _avg(obp),
        ]
    ip = _sum("ip")
    er, bba, ha = _sum("er"), _sum("bb_allowed"), _sum("h_allowed")
    era = (er * 9.0 / ip) if ip > 0 else 0.0
    whip = ((bba + ha) / ip) if ip > 0 else 0.0
    return [
        f"{ip:.1f}",
        str(int(_sum("w"))),
        str(int(_sum("l"))),
        str(int(_sum("sv"))),
        str(int(_sum("k"))),
        f"{era:.2f}",
        f"{whip:.2f}",
    ]


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


def _badge_from_status(status: str) -> str | None:
    """Derive the display badge from a Yahoo roster status string.

    Returns ``"IL"`` for any IL/NA status, ``"DTD"`` for day-to-day, else ``None``.
    """
    s = (status or "").strip()
    sl = s.lower()
    if sl.startswith("il") or sl == "na":
        return "IL"
    if "dtd" in sl or "day" in sl:
        return "DTD"
    return None


def _to_match_player(
    player_id,
    slot: str,
    pool,
    hitter: bool,
    state: str,
    status: str,
    roster_status: str = "",
) -> MatchPlayer:
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
        badge=_badge_from_status(roster_status),
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

        # Set per-category win field now that categories are built.
        for cat in categories:
            cat.win = _cat_win(cat.you, cat.opp, cat.inverse)

        # Compute scores from per-category win fields (set above).
        you_score = sum(1 for c in categories if c.win == "you")
        opp_score = sum(1 for c in categories if c.win == "opp")

        # Build roster-comparison tables (wrapped so any failure leaves fields empty).
        hitters: list[RosterRow] = []
        pitchers: list[RosterRow] = []
        date_tabs: list[str] = []
        hitter_columns: list[str] = []
        pitcher_columns: list[str] = []
        hitter_totals: SideTotals = SideTotals()
        pitcher_totals: SideTotals = SideTotals()
        try:
            from src.yahoo_data_service import get_yahoo_data_service as _yds_fn

            _yds = _yds_fn()
            (
                hitters,
                pitchers,
                date_tabs,
                hitter_columns,
                pitcher_columns,
                hitter_totals,
                pitcher_totals,
            ) = self._build_roster_tables(team_name, opponent, _yds, week)
        except Exception:
            pass  # cold env — leave roster fields at defaults

        you = self._team_side(team_name, you_score)
        opp = self._team_side(opponent, opp_score)

        return MatchupResponse(
            team_name=team_name,
            opponent=opponent,
            week=week,
            projected_cat_wins=projected_cat_wins,
            win_prob=win_prob,
            categories=categories,
            hitters=hitters,
            pitchers=pitchers,
            date_tabs=date_tabs,
            hitter_columns=hitter_columns,
            pitcher_columns=pitcher_columns,
            you=you,
            opp=opp,
            hitter_totals=hitter_totals,
            pitcher_totals=pitcher_totals,
        )

    @staticmethod
    def _team_side(team_name: str, score: int) -> TeamSide:
        manager, record = "", ""
        try:
            from src.database import get_connection, load_league_records

            conn = get_connection()
            try:
                r = conn.execute("SELECT manager_name FROM league_teams WHERE team_name = ?", (team_name,)).fetchone()
                if r and r[0]:
                    manager = str(r[0])
            finally:
                conn.close()
            recs = load_league_records()
            if recs is not None and not recs.empty:
                m = recs[recs["team_name"] == team_name]
                if not m.empty:
                    row = m.iloc[0]
                    record = _format_record(row.get("wins"), row.get("losses"), row.get("ties"), row.get("rank"))
        except Exception:
            pass
        return TeamSide(name=team_name, manager=manager, record=record, score=int(score))

    @staticmethod
    def _build_roster_tables(
        team_name: str,
        opponent: str,
        yds,
        week: int,
    ) -> tuple[list[RosterRow], list[RosterRow], list[str], list[str], list[str], SideTotals, SideTotals]:
        """Build hitter/pitcher RosterRow lists for the matchup comparison.

        Returns (hitters, pitchers, date_tabs, hitter_columns, pitcher_columns,
                 hitter_totals, pitcher_totals).
        Never raises — on any error returns empty lists.
        """
        import pandas as pd
        import statsapi

        from src.database import load_player_pool
        from src.game_day import get_target_game_date
        from src.valuation import TEAM_NAME_TO_ABBR

        # Build abbr → full_name map by inverting the canonical map.
        abbr_to_name: dict[str, str] = {}
        for full, abbr in TEAM_NAME_TO_ABBR.items():
            if abbr not in abbr_to_name:
                abbr_to_name[abbr] = full.title()

        # Load rosters + player pool.
        try:
            rosters = yds.get_rosters()
        except Exception:
            rosters = pd.DataFrame()

        if rosters is None or rosters.empty:
            return [], [], [], [], [], SideTotals(), SideTotals()

        try:
            pool = load_player_pool()
        except Exception:
            pool = pd.DataFrame()

        # Fetch today's schedule for game-state resolution (graceful on failure).
        schedule: list[dict] = []
        try:
            game_date = get_target_game_date()
            schedule = statsapi.schedule(date=game_date) or []
        except Exception:
            schedule = []

        # Filter to the two matchup teams.
        you_rosters = rosters[rosters["team_name"] == team_name].copy()
        opp_rosters = rosters[rosters["team_name"] == opponent].copy()

        # Determine team abbr for game-state from editorial_team_abbr column (if present).
        def _abbr_for_roster(df: pd.DataFrame) -> str:
            """Return the most common editorial_team_abbr in this roster slice."""
            col = "editorial_team_abbr"
            if col not in df.columns or df.empty:
                return ""
            abbrs = df[col].dropna().astype(str).str.strip().str.upper()
            abbrs = abbrs[abbrs != ""]
            return abbrs.mode().iloc[0] if not abbrs.empty else ""

        # Stable slot ordering for the display grid.
        # BN and IL* sort last within their respective table.
        _HITTER_SLOT_ORDER = ["C", "1B", "2B", "3B", "SS", "OF", "Util", "BN"]
        _PITCHER_SLOT_ORDER = ["SP", "RP", "P", "BN"]

        def _slot_key(slot: str) -> int:
            order = _HITTER_SLOT_ORDER + _PITCHER_SLOT_ORDER
            s = str(slot)
            try:
                return order.index(s)
            except ValueError:
                # IL10/IL15/IL60 etc. sort after named slots
                if s.upper().startswith("IL"):
                    return len(order) + 1
                return len(order)

        _PITCHER_SLOTS = frozenset({"SP", "RP", "P"})

        def _is_pitcher_by_pool(pid, roster_status: str, side_pool: pd.DataFrame) -> bool:
            """Classify hitter vs pitcher using the pool's is_hitter flag.

            Falls back to slot-string heuristic only when the player is absent
            from the pool (never raises).
            """
            if side_pool is not None and not side_pool.empty:
                try:
                    pmatch = side_pool[side_pool["player_id"] == pid]
                    if not pmatch.empty:
                        flag = pmatch.iloc[0].get("is_hitter")
                        if flag is not None:
                            return not bool(flag)
                except Exception:
                    pass
            # Fallback: use roster status/IL as pitcher-leaning default
            return False

        def _build_side(
            side_rosters: pd.DataFrame,
            side_pool: pd.DataFrame,
            side_team_abbr: str,
            hitter: bool,
        ) -> tuple[list, list]:
            """Build (players, slots) for one side (you or opp), hitters or pitchers.

            Uses ``selected_position`` as the assigned slot (falls back to
            ``roster_slot`` only when missing). Classifies hitter/pitcher via
            the pool's ``is_hitter`` flag. Includes BN and IL rows.
            """
            players: list[MatchPlayer] = []
            slots: list[str] = []
            if side_rosters.empty:
                return players, slots
            for _, row in side_rosters.iterrows():
                # Use the manager-assigned slot, not the eligible-positions list.
                sel = str(row.get("selected_position") or "").strip()
                if not sel:
                    sel = str(row.get("roster_slot") or "").strip()
                if not sel:
                    continue

                pid = row.get("player_id")
                roster_status = str(row.get("status") or "").strip()

                # Classify using pool is_hitter flag (handles BN/IL/SP,RP swingmen).
                is_pit = _is_pitcher_by_pool(pid, sel, side_pool)

                if hitter and is_pit:
                    continue
                if not hitter and not is_pit:
                    continue

                # Look up team_abbr for game-state from pool or editorial_team_abbr.
                team_abbr = side_team_abbr
                if side_pool is not None and not side_pool.empty:
                    try:
                        pmatch = side_pool[side_pool["player_id"] == pid]
                        if not pmatch.empty:
                            t = str(pmatch.iloc[0].get("team", "") or "").strip().upper()
                            if t:
                                team_abbr = t
                    except Exception:
                        pass
                state, status = _game_state(team_abbr, schedule, abbr_to_name)
                mp = _to_match_player(pid, sel, side_pool, hitter, state, status, roster_status=roster_status)
                players.append(mp)
                slots.append(sel)
            # Sort by slot order for a stable grid.
            paired = sorted(zip(slots, players), key=lambda x: _slot_key(x[0]))
            if paired:
                slots, players = zip(*paired)  # type: ignore[assignment]
                return list(players), list(slots)
            return [], []

        # Merge pool into rosters for abbr lookup (need per-player team abbr).
        you_team_abbr = _abbr_for_roster(you_rosters)
        opp_team_abbr = _abbr_for_roster(opp_rosters)

        you_hitters, you_h_slots = _build_side(you_rosters, pool, you_team_abbr, hitter=True)
        opp_hitters, opp_h_slots = _build_side(opp_rosters, pool, opp_team_abbr, hitter=True)
        you_pitchers, you_p_slots = _build_side(you_rosters, pool, you_team_abbr, hitter=False)
        opp_pitchers, opp_p_slots = _build_side(opp_rosters, pool, opp_team_abbr, hitter=False)

        hitters = _pair_rows(you_hitters, opp_hitters, you_h_slots)
        pitchers = _pair_rows(you_pitchers, opp_pitchers, you_p_slots)

        # Aggregate per-side totals from the pool rows for each side's players.
        def _side_totals(side_rosters: pd.DataFrame, is_hit: bool) -> list[str]:
            if pool is None or pool.empty or side_rosters.empty:
                return []
            ids = [int(p) for p in side_rosters["player_id"].dropna().astype(int).tolist()]
            sub = pool[pool["player_id"].isin(ids)]
            if "is_hitter" in sub.columns:
                sub = sub[sub["is_hitter"].astype(bool) == is_hit]
            return _aggregate_totals(sub, hitter=is_hit)

        hitter_totals = SideTotals(you=_side_totals(you_rosters, True), opp=_side_totals(opp_rosters, True))
        pitcher_totals = SideTotals(you=_side_totals(you_rosters, False), opp=_side_totals(opp_rosters, False))

        return (
            hitters,
            pitchers,
            _date_tabs(week),
            _HITTER_COLUMNS,
            _PITCHER_COLUMNS,
            hitter_totals,
            pitcher_totals,
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

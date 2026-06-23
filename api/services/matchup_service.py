"""Matchup service — the ONE place that calls the Yahoo matchup + matchup context engine.
Maps engine output → the Matchup contract. Resilient: missing live data
degrades to an empty categories list rather than raising."""

from __future__ import annotations

import logging
import math

from api.contracts.common import Record, StatItem
from api.contracts.matchup import (
    LeagueMatchup,
    MatchPlayer,
    MatchupCategory,
    MatchupResponse,
    RosterRow,
    SideTotals,
    TeamSide,
)
from api.services.live_boxscore import fetch_live_player_lines
from api.services.player_ref import player_ref_from_pool

logger = logging.getLogger(__name__)

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


def _weekly_divisor(week) -> int:
    """ROS projections → weekly: the rest-of-season line ÷ the weeks remaining gives a
    per-week estimate for a weekly H2H matchup. max(1, season_weeks - week); 1 = raw."""
    try:
        from src.valuation import LeagueConfig

        sw = int(LeagueConfig().season_weeks)
    except Exception:
        sw = 26
    try:
        w = int(week)
    except (TypeError, ValueError):
        w = 0
    return max(1, sw - w)


def _scale(value, weeks) -> float:
    """Scale a counting stat to weekly (no-op when weeks is falsy/None — raw projection)."""
    v = _f(value)
    return (v / weeks) if (weeks and weeks > 0) else v


def _format_record(wins, losses, ties, rank) -> str:
    w, l, t = int(_f(wins)), int(_f(losses)), int(_f(ties))
    r = int(_f(rank))
    if r <= 0:
        return f"{w}-{l}-{t}"
    suffix = "th" if 11 <= (r % 100) <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(r % 10, "th")
    return f"{w}-{l}-{t} · {r}{suffix}"


def _aggregate_totals(rows, hitter: bool, weeks=None) -> list[StatItem]:
    """Aggregate a side's roster stat line: counting stats summed (scaled to weekly when
    `weeks` given), rate stats weighted (AVG=Σh/Σab, OBP=Σ(h+bb+hbp)/Σ(ab+bb+hbp+sf),
    ERA=Σer·9/Σip, WHIP=Σ(bb_allowed+h_allowed)/Σip — scale-invariant). NaN/zero-safe."""
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
        avg = (h / ab) if ab > 0 else 0.0  # rates from UNSCALED sums (ratio is scale-invariant)
        obp = ((h + bb + hbp) / obp_den) if obp_den > 0 else 0.0
        vals = [
            f"{round(_scale(h, weeks))}/{round(_scale(ab, weeks))}",
            str(round(_scale(_sum("r"), weeks))),
            str(round(_scale(_sum("hr"), weeks))),
            str(round(_scale(_sum("rbi"), weeks))),
            str(round(_scale(_sum("sb"), weeks))),
            _avg(avg),
            _avg(obp),
        ]
        return [StatItem(label=c, value=v) for c, v in zip(_HITTER_COLUMNS, vals)]
    ip = _sum("ip")
    er, bba, ha = _sum("er"), _sum("bb_allowed"), _sum("h_allowed")
    era = (er * 9.0 / ip) if ip > 0 else 0.0
    whip = ((bba + ha) / ip) if ip > 0 else 0.0
    vals = [
        f"{_scale(ip, weeks):.1f}",
        str(round(_scale(_sum("w"), weeks))),
        str(round(_scale(_sum("l"), weeks))),
        str(round(_scale(_sum("sv"), weeks))),
        str(round(_scale(_sum("k"), weeks))),
        f"{era:.2f}",
        f"{whip:.2f}",
    ]
    return [StatItem(label=c, value=v) for c, v in zip(_PITCHER_COLUMNS, vals)]


def _fmt_hitter_stats(row, weeks=None) -> list[StatItem]:
    g = row.get if hasattr(row, "get") else lambda k, d=None: row[k] if k in row else d
    vals = [
        f"{round(_scale(g('h'), weeks))}/{round(_scale(g('ab'), weeks))}",
        str(round(_scale(g("r"), weeks))),
        str(round(_scale(g("hr"), weeks))),
        str(round(_scale(g("rbi"), weeks))),
        str(round(_scale(g("sb"), weeks))),
        _avg(g("avg")),
        _avg(g("obp")),
    ]
    return [StatItem(label=c, value=v) for c, v in zip(_HITTER_COLUMNS, vals)]


def _fmt_pitcher_stats(row, weeks=None) -> list[StatItem]:
    g = row.get if hasattr(row, "get") else lambda k, d=None: row[k] if k in row else d
    vals = [
        f"{_scale(g('ip'), weeks):.1f}",
        str(round(_scale(g("w"), weeks))),
        str(round(_scale(g("l"), weeks))),
        str(round(_scale(g("sv"), weeks))),
        str(round(_scale(g("k"), weeks))),
        f"{_f(g('era')):.2f}",
        f"{_f(g('whip')):.2f}",
    ]
    return [StatItem(label=c, value=v) for c, v in zip(_PITCHER_COLUMNS, vals)]


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
    weeks=None,
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
    stats = (_fmt_hitter_stats(prow, weeks) if hitter else _fmt_pitcher_stats(prow, weeks)) if prow is not None else []
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

    def _override(rows, key: str, columns: list[str]) -> None:
        for row in rows:
            for mp in (row.you, row.opp):
                if mp is None or mp.state not in ("live", "final"):
                    continue
                mid = getattr(mp.player, "mlb_id", None)
                entry = live.get(int(mid)) if mid else None
                if not entry:
                    continue
                line: list[str] = entry.get(key) or []
                if line:
                    # live_boxscore returns list[str]; wrap into list[StatItem] for the contract
                    mp.stats = [StatItem(label=col, value=v) for col, v in zip(columns, line)]
                    mp.status = entry.get("status") or mp.status

    _override(hitters, "hitter", _HITTER_COLUMNS)
    _override(pitchers, "pitcher", _PITCHER_COLUMNS)


_PITCHER_SLOTS = frozenset({"SP", "RP", "P"})


def _is_pitcher(pid, eligible: str, side_pool) -> bool:
    """Classify hitter vs pitcher. The pool's ``is_hitter`` flag is authoritative
    when present; on a pool miss (or is_hitter None) fall back to the player's
    ELIGIBLE positions (SP/RP/P → pitcher). Use eligible positions, NOT the
    assigned slot — a benched/IL pitcher's slot is 'BN'/'IL'. Unknown → hitter
    (logged), the conservative default."""
    if side_pool is not None and not side_pool.empty:
        try:
            pmatch = side_pool[side_pool["player_id"] == pid]
            if not pmatch.empty:
                flag = pmatch.iloc[0].get("is_hitter")
                if flag is not None:
                    return not bool(flag)
        except Exception:
            pass
    toks = {t.strip().upper() for t in str(eligible or "").replace("/", ",").split(",") if t.strip()}
    if toks & _PITCHER_SLOTS:
        return True
    logger.debug("matchup: unresolved player_id=%s (eligible=%r) → defaulting to hitter", pid, eligible)
    return False


def _parse_cat_value(raw) -> float:
    """Parse a Yahoo category value. The bare ``-`` (and empty) is Yahoo's
    not-yet-played placeholder → 0.0. A real value KEEPS its sign — unlike the old
    blanket ``str(x).replace("-", "0")`` which turned ``"-5"`` into ``"05"`` = 5.0
    (sign flip; masked only because H2H category totals are normally non-negative)."""
    s = str(raw).strip()
    if s in ("", "-"):
        return 0.0
    try:
        return float(s)
    except (TypeError, ValueError):
        return 0.0


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
        except Exception as exc:
            logger.warning("MatchupService.get_matchup failed: %s", exc)
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
        except Exception as exc:
            logger.warning("MatchupService._build_roster_tables failed: %s", exc)
            # cold env — leave roster fields at defaults

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
            league=self._league(team_name, opponent, week, you_score, opp_score),
        )

    def _league(self, team_name: str, opponent: str, week: int, you_score: int, opp_score: int) -> list[LeagueMatchup]:
        """The week's full scoreboard (all matchups). Pairings + records always
        derivable; per-team weekly scores are best-effort from each pairing's cached
        matchup (the user's own pairing reuses the already-computed you/opp scores).
        Never raises → [] on any failure / cold env."""
        if not week:
            return []
        try:
            from src.auth import _normalize_team_name
            from src.database import load_league_schedule_full

            pairings = load_league_schedule_full().get(int(week), [])
            user = {_normalize_team_name(team_name), _normalize_team_name(opponent)}
            out: list[LeagueMatchup] = []
            for a, b in pairings:
                if {_normalize_team_name(a), _normalize_team_name(b)} == user and "" not in user:
                    # User's own pairing — reuse the header scores, aligned to a/b.
                    a_score = you_score if _normalize_team_name(a) == _normalize_team_name(team_name) else opp_score
                    b_score = opp_score if a_score == you_score else you_score
                else:
                    a_score, b_score = self._pairing_scores(a, b, week)
                out.append(LeagueMatchup(a=self._team_side(a, a_score), b=self._team_side(b, b_score)))
            return out
        except Exception as exc:
            logger.warning("MatchupService._league failed: %s", exc)
            return []

    def _pairing_scores(self, a: str, b: str, week: int) -> tuple[int, int]:
        """Best-effort (a_wins, b_wins) for a non-user pairing from cached matchups.
        Tries a's cache, then b's (swapped); (0, 0) when neither is cached."""
        from_a = self._score_from_cache(a, week)
        if from_a is not None:
            return from_a
        from_b = self._score_from_cache(b, week)
        if from_b is not None:
            return from_b[1], from_b[0]  # b's cache is (b_wins, a_wins) → swap to (a, b)
        return 0, 0

    def _score_from_cache(self, name: str, week: int) -> tuple[int, int] | None:
        """(name_wins, opp_wins) from name's cached weekly matchup, or None if uncached."""
        try:
            from src.database import load_matchup_cache

            cached = load_matchup_cache(name, int(week))
            raw = (cached or {}).get("categories") if isinstance(cached, dict) else None
            if not raw:
                return None
            cats = self._build_categories(raw, name)
            wins = sum(1 for c in cats if _cat_win(c.you, c.opp, c.inverse) == "you")
            opp_wins = sum(1 for c in cats if _cat_win(c.you, c.opp, c.inverse) == "opp")
            return wins, opp_wins  # (name's wins, opponent's wins)
        except Exception:
            return None

    @staticmethod
    def _team_side(team_name: str, score: int) -> TeamSide:
        manager, record = "", ""
        record_wlt: Record | None = None
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
                    # Parse W-L-T once; both the structured field and the display
                    # string derive from the SAME ints so they can't drift.
                    w = int(_f(row.get("wins")))
                    l = int(_f(row.get("losses")))
                    t = int(_f(row.get("ties")))
                    record_wlt = Record(wins=w, losses=l, ties=t)
                    record = _format_record(w, l, t, row.get("rank"))
        except Exception as exc:
            logger.warning("MatchupService._team_side failed: %s", exc)
        return TeamSide(name=team_name, manager=manager, record=record, record_wlt=record_wlt, score=int(score))

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
        game_date = ""  # always bound (used as the live-lines cache key below)
        try:
            game_date = get_target_game_date()
            schedule = statsapi.schedule(date=game_date) or []
        except Exception:
            schedule = []

        # Projected lines are rest-of-season — scale to weekly for this H2H matchup.
        weeks = _weekly_divisor(week)

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

        def _build_side(
            side_rosters: pd.DataFrame,
            side_pool: pd.DataFrame,
            side_team_abbr: str,
            hitter: bool,
        ) -> tuple[list, list]:
            """Build (players, slots) for one side (you or opp), hitters or pitchers.

            Uses ``selected_position`` as the assigned slot (falls back to
            ``roster_slot`` only when missing). Classifies hitter/pitcher via
            the pool's ``is_hitter`` flag, falling back to eligible positions
            (``roster_slot``) on a pool miss. Includes BN and IL rows.
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

                # Classify by pool is_hitter; on a pool miss, by ELIGIBLE positions
                # (roster_slot), never the assigned slot `sel` (which may be BN/IL).
                eligible = str(row.get("roster_slot") or "").strip() or sel
                is_pit = _is_pitcher(pid, eligible, side_pool)

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
                mp = _to_match_player(
                    pid, sel, side_pool, hitter, state, status, roster_status=roster_status, weeks=weeks
                )
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

        # Overlay today's live in-game lines (no-op when no games are live/final).
        _apply_live_lines(hitters, pitchers, schedule, date_key=str(game_date))

        # Aggregate per-side totals from the pool rows for each side's players.
        def _side_totals(side_rosters: pd.DataFrame, is_hit: bool) -> list[str]:
            if pool is None or pool.empty or side_rosters.empty:
                return []
            ids = [int(p) for p in side_rosters["player_id"].dropna().astype(int).tolist()]
            sub = pool[pool["player_id"].isin(ids)]
            if "is_hitter" in sub.columns:
                sub = sub[sub["is_hitter"].astype(bool) == is_hit]
            return _aggregate_totals(sub, hitter=is_hit, weeks=weeks)

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
            you = _parse_cat_value(entry.get("you", "0"))
            opp = _parse_cat_value(entry.get("opp", "0"))
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

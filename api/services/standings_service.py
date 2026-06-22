"""Standings service — the ONE place that calls the Yahoo standings engine.
Maps engine output → the Standings contract. Resilient: missing live data
degrades to an empty team list rather than raising."""

from __future__ import annotations

import logging
import math

from api.contracts.standings import StandingsResponse, TeamStanding

logger = logging.getLogger(__name__)

# W-L-T can arrive two ways depending on how the standings were synced:
#   • the dedicated ``league_records`` table (current sync / Railway), or
#   • WINS/LOSSES/TIES rows inside ``league_standings`` (older sync / local DB).
# These category names are league-structure metadata, NOT scoring categories,
# so they must never receive a per-category rank.
_WLT_CATEGORIES = {"WINS": "wins", "LOSSES": "losses", "TIES": "ties"}


def _num(value) -> float | None:
    """Best-effort finite float, else None (NaN/garbage → None)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


class StandingsService:
    def get_standings(self) -> StandingsResponse:
        from src.yahoo_data_service import get_yahoo_data_service

        teams: list[TeamStanding] = []
        try:
            import pandas as pd

            from src.database import load_league_records

            yds = get_yahoo_data_service()
            df = yds.get_standings()
            if df is None or df.empty:
                return StandingsResponse(teams=[])
            try:
                records = load_league_records()
            except Exception as exc:
                # A genuine DB fault (locked / missing table) here is otherwise
                # indistinguishable from an empty records table → W-L-T shows 0-0.
                logger.warning("StandingsService: load_league_records failed; W-L-T will be 0-0: %s", exc)
                records = pd.DataFrame()
            teams = self._build_teams(df, records)
        except Exception as exc:
            logger.warning("StandingsService.get_standings failed; returning empty standings: %s", exc)
            teams = []  # cold env / no data → empty list
        return StandingsResponse(teams=teams)

    @staticmethod
    def _build_teams(standings_df, records_df) -> list[TeamStanding]:
        """Aggregate the long-format ``league_standings`` frame into one
        ``TeamStanding`` per team.

        Two correctness rules the old version got wrong:
          1. **Per-category ranks** are computed by ranking teams *within each
             scoring category* by their ``total`` (inverse-aware: ERA/WHIP/L
             lower-is-better). The synced ``rank`` column is the team's OVERALL
             rank cloned into every category row, so it must NOT be used as the
             per-category rank.
          2. **W-L-T + overall rank** come from ``records_df``
             (``load_league_records()``) when present, else from the
             WINS/LOSSES/TIES category rows in ``standings_df`` (older local
             data), else default to 0.
        """
        from src.valuation import LeagueConfig

        if standings_df is None or "team_name" not in getattr(standings_df, "columns", []):
            return []

        cfg = LeagueConfig()
        stat_cats = {str(c).upper() for c in cfg.all_categories}
        inverse = {str(c).upper() for c in cfg.inverse_stats}

        # ── 1. Per-category ranks from `total` (the real fix) ──────────────
        cat_totals: dict[str, list[tuple[str, float]]] = {}
        std_wlt: dict[str, dict[str, int]] = {}
        std_overall_rank: dict[str, int] = {}
        for _, row in standings_df.iterrows():
            tname = str(row.get("team_name", "") or "")
            cat = str(row.get("category", "") or "").strip().upper()
            if cat in stat_cats:
                tot = _num(row.get("total"))
                if tot is not None:
                    cat_totals.setdefault(cat, []).append((tname, tot))
            elif cat in _WLT_CATEGORIES:
                tot = _num(row.get("total"))
                if tot is not None:
                    std_wlt.setdefault(tname, {})[_WLT_CATEGORIES[cat]] = int(round(tot))
            # The cloned overall rank rides on every row — capture once per team.
            if tname not in std_overall_rank:
                rk = _num(row.get("rank"))
                if rk is not None:
                    std_overall_rank[tname] = int(rk)

        cat_ranks: dict[str, dict[str, int]] = {}
        for cat, pairs in cat_totals.items():
            # Inverse cats: lower total is better → ascending. Else descending.
            ordered = sorted(pairs, key=lambda p: p[1], reverse=cat not in inverse)
            prev_val: float | None = None
            prev_rank = 0
            for position, (tname, val) in enumerate(ordered, start=1):
                if prev_val is not None and val == prev_val:
                    rk = prev_rank  # tie → share the better (min) rank
                else:
                    rk = position
                    prev_val, prev_rank = val, rk
                cat_ranks.setdefault(tname, {})[cat] = rk

        # ── 2. Records lookup (authoritative for W-L-T + overall rank) ─────
        rec_by_team: dict[str, dict] = {}
        if records_df is not None and not getattr(records_df, "empty", True):
            for _, r in records_df.iterrows():
                rec_by_team[str(r.get("team_name", "") or "")] = r

        def _rec_int(rec, key: str) -> int | None:
            if rec is None:
                return None
            v = _num(rec.get(key))
            return int(v) if v is not None else None

        # ── 3. Assemble one TeamStanding per team ─────────────────────────
        result: list[TeamStanding] = []
        for team_name in standings_df["team_name"].dropna().unique():
            tname = str(team_name)
            rec = rec_by_team.get(tname)
            std = std_wlt.get(tname, {})

            wins = _rec_int(rec, "wins")
            wins = std.get("wins", 0) if wins is None else wins
            losses = _rec_int(rec, "losses")
            losses = std.get("losses", 0) if losses is None else losses
            ties = _rec_int(rec, "ties")
            ties = std.get("ties", 0) if ties is None else ties

            rank = _rec_int(rec, "rank") or 0
            if rank == 0:
                rank = std_overall_rank.get(tname, 0)

            points = 0.0
            if rec is not None:
                pf = _num(rec.get("points_for"))
                if pf is not None:
                    points = pf

            result.append(
                TeamStanding(
                    rank=rank,
                    team_name=tname,
                    wins=int(wins),
                    losses=int(losses),
                    ties=int(ties),
                    points=points,
                    category_ranks=cat_ranks.get(tname, {}),
                )
            )

        # Sort by overall rank (rank 0 / unknown sinks to the bottom), then name.
        result.sort(key=lambda t: (t.rank if t.rank > 0 else 999, t.team_name))
        return result

"""Playoff-odds service — the ONE place importing src/ for this endpoint.

Composes the existing, structurally-guarded standings simulator
(standings_engine.simulate_season_enhanced) over per-team weekly totals
(standings_utils.get_all_team_totals) → per-team playoff odds + projected
standings. Resilient: any missing live data degrades to an empty response."""

from __future__ import annotations

import logging

from api.contracts.common import Record
from api.contracts.playoff import PlayoffOddsResponse, PlayoffTeam

logger = logging.getLogger(__name__)

_PLAYOFF_SPOTS = 4
_WEEKS_IN_SEASON = 26.0
# API uses fewer sims than the Streamlit 10k default for read responsiveness.
# simulate_season_enhanced uses a FIXED seed → deterministic (no run-to-run wobble);
# 2000 sims ≈ ~1% SE + ~2.4s, plenty for a dashboard odds bar (vs ~4.8s at 4000).
_API_N_SIMS = 2000


class PlayoffService:
    def get_playoff_odds(self, team_name: str, n_sims: int = _API_N_SIMS) -> PlayoffOddsResponse:
        empty = PlayoffOddsResponse(team_name=team_name, playoff_spots=_PLAYOFF_SPOTS)
        try:
            from src.standings_engine import simulate_season_enhanced

            weekly_totals = self._team_weekly_totals()
            current_standings, current_wins = self._current_standings()
            if not weekly_totals or not current_standings:
                return empty  # cold env (league_rosters / records unpopulated locally)

            full_schedule = self._full_schedule()
            current_week = self._current_week(current_standings)

            sim = simulate_season_enhanced(
                current_standings=current_standings,
                team_weekly_totals=weekly_totals,
                full_schedule=full_schedule,
                current_week=current_week,
                n_sims=n_sims,
                playoff_spots=_PLAYOFF_SPOTS,
            )
            rows = self._to_rows(sim, current_wins, team_name)
            you = next((r for r in rows if r.is_user), None)
            return PlayoffOddsResponse(
                team_name=team_name,
                playoff_spots=_PLAYOFF_SPOTS,
                you=you,
                league=rows,
                n_sims=n_sims,  # simulate_season_enhanced doesn't echo n_sims; report what we requested
            )
        except Exception as exc:
            logger.warning("PlayoffService.get_playoff_odds failed: %s", exc)
            return empty

    # ── input assembly (mirrors the Streamlit Season Projections tab) ──────
    @staticmethod
    def _team_weekly_totals() -> dict[str, dict[str, float]]:
        """{team: {cat: per-week mean}} — counting stats /26, rate stats passthrough.

        NOTE: in the API context there's no Streamlit session, so get_all_team_totals
        falls to its projection-based tier (not Yahoo season-to-date actuals). The sim
        still STARTS from real current_standings (W/L/T); only the go-forward weekly
        strength is projection-based. A future standings_utils `yds=` param could use
        YTD actuals — out of scope here.
        """
        try:
            from src.database import load_league_rosters, load_player_pool
            from src.standings_utils import get_all_team_totals
            from src.valuation import LeagueConfig

            lr = load_league_rosters()
            if lr is None or lr.empty or "team_name" not in lr.columns or "player_id" not in lr.columns:
                return {}
            rosters: dict[str, list[int]] = {}
            for team, group in lr.groupby("team_name"):
                rosters[str(team)] = [int(p) for p in group["player_id"].dropna().tolist()]

            pool = load_player_pool()
            if pool is not None and not pool.empty and "name" in pool.columns:
                pool = pool.rename(columns={"name": "player_name"})

            season_totals = get_all_team_totals(
                league_rosters=rosters or None,
                player_pool=pool if pool is not None and not pool.empty else None,
            )
            if not season_totals:
                return {}

            cfg = LeagueConfig()
            rate = set(cfg.rate_stats)
            counting = {c for c in cfg.all_categories if c not in rate}
            weekly: dict[str, dict[str, float]] = {}
            for team, cat_map in season_totals.items():
                per_week: dict[str, float] = {}
                for cat, val in cat_map.items():
                    try:
                        v = float(val)
                    except (TypeError, ValueError):
                        continue
                    if cat in counting:
                        per_week[cat] = v / _WEEKS_IN_SEASON
                    elif cat in rate:
                        per_week[cat] = v
                weekly[str(team)] = per_week
            return weekly
        except Exception as exc:
            logger.warning("PlayoffService._team_weekly_totals failed: %s", exc)
            return {}

    @staticmethod
    def _current_standings() -> tuple[dict[str, dict[str, int]], dict[str, int]]:
        """({team: {W,L,T}}, {team: wins}) from league_records; empty on failure."""
        try:
            from src.database import load_league_records

            recs = load_league_records()
            if recs is None or recs.empty:
                return {}, {}
            standings: dict[str, dict[str, int]] = {}
            wins: dict[str, int] = {}
            for _, row in recs.iterrows():
                # Per-row guard: one malformed/NaN record row must not zero the whole
                # panel (load_league_records already coerces, but be defensive).
                try:
                    tn = str(row.get("team_name", "") or "")
                    if not tn:
                        continue
                    w, lo, t = int(row.get("wins", 0)), int(row.get("losses", 0)), int(row.get("ties", 0))
                except (TypeError, ValueError):
                    continue
                standings[tn] = {"W": w, "L": lo, "T": t}
                wins[tn] = w
            return standings, wins
        except Exception as exc:
            logger.warning("PlayoffService._current_standings failed: %s", exc)
            return {}, {}

    @staticmethod
    def _full_schedule():
        try:
            from src.database import load_league_schedule_full

            return load_league_schedule_full() or {}
        except Exception:
            return {}

    @staticmethod
    def _current_week(current_standings: dict[str, dict[str, int]]) -> int:
        """Next unplayed week = max games-played across teams + 1 (≥1).

        Derived from the records DB (no calendar/_SEASON_START dependency). Caveat:
        if the records table is stale vs the calendar, the sim may re-play one already-
        played week (slightly pessimistic) — acceptable degradation, refreshes with the data.
        """
        try:
            played = max((rec["W"] + rec["L"] + rec["T"]) for rec in current_standings.values())
            return max(1, int(played) + 1)
        except Exception:
            return 1

    @staticmethod
    def _to_rows(sim, current_wins: dict[str, int], team_name: str) -> list[PlayoffTeam]:
        from src.auth import _normalize_team_name

        probs = (sim.get("playoff_probability") or {}) if isinstance(sim, dict) else {}
        recs = (sim.get("projected_records") or {}) if isinstance(sim, dict) else {}
        wanted = _normalize_team_name(team_name)

        # Sort (team, odds) by odds desc FIRST, then construct rows with their final
        # rank/in_cut — no post-construction mutation (frozen-model safe).
        def _odds(prob) -> float:
            try:
                return round(float(prob) * 100.0, 1)
            except (TypeError, ValueError):
                return 0.0

        ordered = sorted(probs.items(), key=lambda kv: _odds(kv[1]), reverse=True)
        rows: list[PlayoffTeam] = []
        for i, (team, prob) in enumerate(ordered):
            rec = recs.get(team, {}) if isinstance(recs, dict) else {}
            try:
                w, lo, t = float(rec.get("W", 0)), float(rec.get("L", 0)), float(rec.get("T", 0))
            except (TypeError, ValueError):
                w, lo, t = 0.0, 0.0, 0.0
            rank = i + 1
            rows.append(
                PlayoffTeam(
                    team=str(team),
                    playoff_odds=_odds(prob),
                    projected_wins=round(w, 1),
                    projected_record=f"{w:.0f}-{lo:.0f}-{t:.0f}",
                    projected_record_wlt=Record(wins=int(w), losses=int(lo), ties=int(t)),
                    current_wins=int(current_wins.get(str(team), 0)),
                    rank=rank,
                    in_cut=rank <= _PLAYOFF_SPOTS,
                    is_user=_normalize_team_name(str(team)) == wanted and bool(wanted),
                )
            )
        return rows

"""Lineup service — the ONE place that calls the optimizer pipeline. Maps
its output → the Lineup contract. Resilient: missing data → empty slots.
NOTE: synchronous for B1; becomes an Arq background job in B3."""

from __future__ import annotations

import logging
import math

from api.contracts.lineup import (
    CatImpact,
    DailyMeta,
    IpPace,
    LineupOptimizeResponse,
    LineupSlot,
    Swap,
)
from api.services.player_ref import player_ref_from_pool

logger = logging.getLogger(__name__)

# Yahoo slots that are NOT in the active lineup (a player here is benched/stashed).
_NON_LINEUP_SLOTS = {"BN", "IL", "IL10", "IL15", "IL60", "NA", "DTD", "", "BENCH"}


class LineupService:
    def optimize(
        self, team_name: str, date=None, scope: str = "rest_of_season", mode: str = "standard"
    ) -> LineupOptimizeResponse:
        if str(mode or "standard").lower() == "daily":
            return self._optimize_daily(team_name, date)
        return self._optimize_standard(team_name, date, scope)

    # ------------------------------------------------------------------ standard (ROS/weekly LP)
    def _optimize_standard(self, team_name: str, date, scope: str) -> LineupOptimizeResponse:
        slots: list[LineupSlot] = []
        bench: list[LineupSlot] = []
        impact: list[CatImpact] = []
        optimal = False
        summary = "Lineup unavailable (no live data in this environment)."
        resolved_date = date or ""
        try:
            from src.game_day import get_target_game_date
            from src.optimizer.pipeline import LineupOptimizerPipeline
            from src.valuation import LeagueConfig
            from src.yahoo_data_service import get_yahoo_data_service

            resolved_date = date or str(get_target_game_date())
            yds = get_yahoo_data_service()
            roster = yds.get_rosters()
            pipeline = LineupOptimizerPipeline(roster, mode="standard", config=LeagueConfig())
            # The pipeline method is optimize() (NOT run()); it returns an OptimizerResult
            # dict whose "lineup" is {assignments, bench, projected_stats, status}.
            result = pipeline.optimize() if hasattr(pipeline, "optimize") else None
            lineup = (result or {}).get("lineup") if isinstance(result, dict) else None

            pool = self._load_pool()
            slots, bench = self._to_slots(lineup, pool, roster)
            impact = self._impact((lineup or {}).get("projected_stats") if isinstance(lineup, dict) else None)
            optimal = self._optimal(roster, {s.player.id for s in slots if s.player.id})
            if slots:
                summary = f"{len(slots)} starters set."
        except Exception as exc:
            logger.warning("LineupService.optimize failed: %s", exc)
        return LineupOptimizeResponse(
            team_name=team_name,
            date=resolved_date,
            slots=slots,
            summary=summary,
            bench=bench,
            optimal=optimal,
            impact=impact,
            mode="standard",
        )

    # ------------------------------------------------------------------ daily (today's DCV start/sit)
    def _optimize_daily(self, team_name: str, date) -> LineupOptimizeResponse:
        slots: list[LineupSlot] = []
        bench: list[LineupSlot] = []
        daily: DailyMeta | None = None
        optimal = False
        summary = "Daily lineup unavailable (no live data in this environment)."
        resolved_date = date or ""
        try:
            from src.game_day import get_target_game_date
            from src.optimizer.pipeline import LineupOptimizerPipeline
            from src.valuation import LeagueConfig
            from src.yahoo_data_service import get_yahoo_data_service

            resolved_date = date or str(get_target_game_date())
            yds = get_yahoo_data_service()
            roster = yds.get_rosters()
            matchup = None
            try:
                matchup = yds.get_matchup()
            except Exception:
                matchup = None
            schedule = self._schedule_today(resolved_date)

            pipeline = LineupOptimizerPipeline(roster, mode="daily", config=LeagueConfig())
            result = pipeline.optimize(matchup=matchup, schedule_today=schedule)
            result = result if isinstance(result, dict) else {}

            pool = self._load_pool()
            slots, bench = self._daily_slots(
                result.get("daily_dcv"), result.get("daily_lineup"), roster, pool, schedule
            )
            daily = self._daily_meta(result, slots, roster, pool, resolved_date)
            optimal = self._optimal(roster, {s.player.id for s in slots if s.player.id})
            if slots:
                summary = f"{len(slots)} starters set for {resolved_date}."
        except Exception as exc:
            logger.warning("LineupService daily failed: %s", exc)
        return LineupOptimizeResponse(
            team_name=team_name,
            date=resolved_date,
            slots=slots,
            summary=summary,
            bench=bench,
            optimal=optimal,
            mode="daily",
            daily=daily,
        )

    def _daily_slots(self, dcv, daily_lineup, roster, pool, schedule) -> tuple[list[LineupSlot], list[LineupSlot]]:
        """Build starter/bench LineupSlots from the daily DCV table.

        PlayerRef is keyed off `daily_dcv` (has player_id → robust). The LP start/slot
        decision comes from `daily_lineup` (name-keyed only), joined by (name, total_dcv);
        total_dcv is a per-player near-unique float, so it disambiguates same-name players."""
        import pandas as pd

        if not isinstance(dcv, pd.DataFrame) or dcv.empty or "player_id" not in dcv.columns:
            return [], []
        name_col = "name" if "name" in dcv.columns else "player_name"
        daily_lineup = daily_lineup if isinstance(daily_lineup, dict) else {}

        started: dict[tuple[str, float | None], str] = {}
        for s in daily_lineup.get("starters") or []:
            started[(self._norm(s.get("name") or s.get("player_name")), self._round_dcv(s.get("total_dcv")))] = str(
                s.get("slot") or ""
            )

        current = self._current_slots(roster)
        records = dcv.to_dict("records")
        max_dcv = max((self._f(r.get("total_dcv")) for r in records), default=0.0)

        starters: list[LineupSlot] = []
        bench: list[LineupSlot] = []
        for r in records:
            try:
                pid = int(r.get("player_id", 0) or 0)
            except (TypeError, ValueError):
                pid = 0
            tdcv = self._f(r.get("total_dcv"))
            is_start = (self._norm(r.get(name_col)), self._round_dcv(tdcv)) in started
            slot = (
                started.get((self._norm(r.get(name_col)), self._round_dcv(tdcv)))
                or str(r.get("positions", "") or "BN").split(",")[0]
            )
            matchup_mult = self._f(r.get("matchup_mult"), 1.0)
            reason = str(r.get("reason") or "").strip() or None
            value = round(100.0 * tdcv / max_dcv, 1) if max_dcv > 0 and tdcv > 0 else 0.0
            slotobj = LineupSlot(
                slot=slot or "BN",
                player=player_ref_from_pool(pid, pool, name=r.get(name_col), positions=r.get("positions")),
                action="START" if is_start else "SIT",
                status="start" if is_start else "bench",
                value=value,
                matchup=self._matchup_str(r.get("team"), schedule),
                current_slot=current.get(pid, ""),
                forced_start=bool(is_start and (matchup_mult < 0.70 or tdcv <= 0)),
                reason=reason,
            )
            (starters if is_start else bench).append(slotobj)
        return starters, bench

    def _daily_meta(self, result, slots, roster, pool, resolved_date) -> DailyMeta:
        result = result if isinstance(result, dict) else {}
        # The pipeline flattens these onto the top-level result (pipeline.py:514-516):
        #   result["urgency_weights"]  = the FLAT {cat: weight} dict (NOT nested under "urgency")
        #   result["rate_stat_modes"]  = {ERA/WHIP: protect|compete|abandon}
        #   result["matchup_summary"]  = {winning/losing/tied: [...]}
        urgency: dict[str, float] = {}
        for k, v in (result.get("urgency_weights") or {}).items():
            fv = self._f(v, float("nan"))  # nan sentinel so non-finite raws are dropped, not zeroed
            if math.isfinite(fv):
                urgency[str(k)] = fv
        rate_modes = result.get("rate_stat_modes") or {}
        summ = result.get("matchup_summary") or {}
        recs = [str(x) for x in (result.get("recommendations") or [])] if isinstance(result, dict) else []
        swaps = [
            Swap(player=s.player, slot=s.slot, value=s.value)
            for s in slots
            if s.action == "START" and s.current_slot.upper().strip() in _NON_LINEUP_SLOTS
        ]
        return DailyMeta(
            urgency=urgency,
            rate_modes={str(k): str(v) for k, v in dict(rate_modes).items()},
            winning=[str(x) for x in (summ.get("winning") or [])],
            losing=[str(x) for x in (summ.get("losing") or [])],
            tied=[str(x) for x in (summ.get("tied") or [])],
            ip_pace=self._ip_pace(roster, pool, resolved_date),
            recommendations=recs,
            swaps=swaps,
        )

    def _ip_pace(self, roster, pool, resolved_date) -> IpPace | None:
        """Weekly IP pacing from the roster's pitchers (season IP merged from the pool)."""
        import pandas as pd

        try:
            from src.ip_tracker import compute_weekly_ip_projection

            if not isinstance(roster, pd.DataFrame) or roster.empty or "player_id" not in roster.columns:
                return None
            pool_ip: dict[int, float] = {}
            if (
                isinstance(pool, pd.DataFrame)
                and not pool.empty
                and "player_id" in pool.columns
                and "ip" in pool.columns
            ):
                for pr in pool.to_dict("records"):
                    try:
                        pool_ip[int(pr.get("player_id", 0) or 0)] = self._f(pr.get("ip"))
                    except (TypeError, ValueError):
                        continue
            pitchers: list[dict] = []
            for r in roster.to_dict("records"):
                positions = str(r.get("positions", "") or r.get("roster_slot", "") or "").upper()
                if not any(p in positions for p in ("SP", "RP", "P")):
                    continue
                try:
                    pid = int(r.get("player_id", 0) or 0)
                except (TypeError, ValueError):
                    pid = 0
                pitchers.append(
                    {
                        "name": str(r.get("name") or r.get("player_name") or ""),
                        "positions": positions,
                        "ip": pool_ip.get(pid, 0.0),
                        "is_starter": "SP" in positions,
                        "status": str(r.get("status", "") or ""),
                    }
                )
            if not pitchers:
                return None
            proj = compute_weekly_ip_projection(pitchers, days_remaining=self._days_remaining(resolved_date))
            if not isinstance(proj, dict):
                return None
            return IpPace(
                projected=self._f(proj.get("projected_ip")),
                target=self._f(proj.get("ip_target")),
                pace_pct=int(self._f(proj.get("ip_pace"))),
                status=str(proj.get("status", "") or ""),
                message=str(proj.get("message", "") or ""),
            )
        except Exception as exc:
            logger.warning("LineupService._ip_pace failed: %s", exc)
            return None

    # ------------------------------------------------------------------ shared helpers
    @staticmethod
    def _load_pool():
        try:
            from src.database import load_player_pool

            return load_player_pool()
        except Exception:
            return None

    @staticmethod
    def _schedule_today(resolved_date) -> list:
        try:
            import statsapi

            sched = statsapi.schedule(date=resolved_date) if resolved_date else statsapi.schedule()
            return sched if isinstance(sched, list) else []
        except Exception:
            return []

    @staticmethod
    def _matchup_str(team, schedule) -> str:
        """'vs BOS' (home) / '@ NYY' (away) for the player's team; '' if not found.

        statsapi.schedule emits FULL team names (home_name/away_name) while the DCV `team`
        is an abbr — so canonicalize BOTH sides via team_name_to_abbr+canonicalize_team
        (the engine's own convention, daily_optimizer.py:770-773) or they never match."""
        if not isinstance(schedule, list) or not schedule:
            return ""
        try:
            from src.valuation import canonicalize_team, team_name_to_abbr
        except Exception:
            return ""

        def _canon(x) -> str:
            return canonicalize_team(team_name_to_abbr(str(x or ""))).upper().strip()

        me = _canon(team)
        if not me:
            return ""
        for g in schedule:
            if not isinstance(g, dict):
                continue
            home = _canon(g.get("home_name") or g.get("home_team"))
            away = _canon(g.get("away_name") or g.get("away_team"))
            if me == home and away:
                return f"vs {away}"
            if me == away and home:
                return f"@ {home}"
        return ""

    @staticmethod
    def _current_slots(roster) -> dict[int, str]:
        import pandas as pd

        out: dict[int, str] = {}
        if not isinstance(roster, pd.DataFrame) or roster.empty or "player_id" not in roster.columns:
            return out
        for r in roster.to_dict("records"):
            try:
                pid = int(r.get("player_id", 0) or 0)
            except (TypeError, ValueError):
                continue
            if pid:
                out[pid] = str(r.get("selected_position", "") or r.get("roster_slot", "") or "")
        return out

    @staticmethod
    def _days_remaining(resolved_date) -> int:
        """Days left in the Mon–Sun fantasy week (Mon=7 … Sun=1); 7 on parse failure."""
        from datetime import datetime

        try:
            d = datetime.strptime(str(resolved_date)[:10], "%Y-%m-%d")
            return max(1, 7 - d.weekday())
        except Exception:
            return 7

    @staticmethod
    def _norm(name) -> str:
        return str(name or "").strip().lower()

    @staticmethod
    def _round_dcv(v) -> float | None:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return round(f, 4) if math.isfinite(f) else None

    @staticmethod
    def _f(v, default: float = 0.0) -> float:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return default
        return f if math.isfinite(f) else default

    @staticmethod
    def _to_slots(lineup, pool=None, roster=None) -> tuple[list[LineupSlot], list[LineupSlot]]:
        """(starters, bench). Starters from lineup['assignments']; bench = roster players
        not in the optimal lineup (player_id-based, so no fragile name lookup)."""
        import pandas as pd

        if not isinstance(lineup, dict):
            return [], []
        assignments = lineup.get("assignments") or []
        starters: list[LineupSlot] = []
        starter_ids: set[int] = set()
        for a in assignments:
            g = a.get if isinstance(a, dict) else (lambda k, d=None, _a=a: getattr(_a, k, d))
            try:
                pid = int(g("player_id", 0) or 0)
            except (TypeError, ValueError):
                pid = 0
            starter_ids.add(pid)
            starters.append(
                LineupSlot(
                    slot=str(g("slot", "") or ""),
                    player=player_ref_from_pool(pid, pool, name=g("player_name", ""), positions=g("positions", "")),
                    action="START",
                    projected=0.0,  # per-player value is daily-mode (DCV) → slice 2
                    status="start",
                )
            )

        bench: list[LineupSlot] = []
        if isinstance(roster, pd.DataFrame) and not roster.empty and "player_id" in roster.columns:
            for r in roster.to_dict("records"):
                try:
                    pid = int(r.get("player_id", 0) or 0)
                except (TypeError, ValueError):
                    continue
                if pid == 0 or pid in starter_ids:
                    continue
                slot = str(r.get("selected_position", "") or r.get("roster_slot", "") or "BN")
                bench.append(
                    LineupSlot(
                        slot=slot or "BN",
                        player=player_ref_from_pool(
                            pid, pool, name=r.get("name") or r.get("player_name"), positions=r.get("positions")
                        ),
                        action="SIT",
                        projected=0.0,
                        status="bench",
                    )
                )
        return starters, bench

    @staticmethod
    def _impact(projected_stats) -> list[CatImpact]:
        """Projected category totals for the optimal lineup → display rows. trend flat
        (no current-lineup baseline to diff against — a slice-2 enhancement)."""
        if not isinstance(projected_stats, dict) or not projected_stats:
            return []
        out: list[CatImpact] = []
        for cat, val in projected_stats.items():
            key = str(cat).upper()
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(fval):  # never emit "nan"/"inf" to JSON
                continue
            if key in ("AVG", "OBP"):
                proj = f"{fval:.3f}".lstrip("0") if 0.0 <= fval < 1.0 else f"{fval:.3f}"
            elif key in ("ERA", "WHIP"):
                proj = f"{fval:.2f}"
            else:
                proj = str(int(round(fval)))
            out.append(CatImpact(key=key, proj=proj, trend="flat"))
        return out

    @staticmethod
    def _optimal(roster, starter_ids: set[int]) -> bool:
        """True iff the user's CURRENT Yahoo starters (selected_position in a lineup slot)
        are exactly the optimizer's chosen starters. False if no starters / no selected_position."""
        import pandas as pd

        if not starter_ids or not isinstance(roster, pd.DataFrame) or roster.empty:
            return False
        if "selected_position" not in roster.columns or "player_id" not in roster.columns:
            return False
        current: set[int] = set()
        for r in roster.to_dict("records"):
            sp = str(r.get("selected_position", "") or "").upper().strip()
            if sp in _NON_LINEUP_SLOTS:
                continue
            try:
                current.add(int(r.get("player_id", 0) or 0))
            except (TypeError, ValueError):
                continue
        current.discard(0)
        return bool(current) and current == starter_ids

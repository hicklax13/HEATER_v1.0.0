"""Start/Sit service — the ONE place importing src.start_sit + the optimizer
context/daily-DCV engines for this feature. Maps engine output -> the Start/Sit
contracts. Resilient: missing live data degrades to empty candidates / full-open
slots rather than raising.

The /compare verdict is a BOUNDED greedy slot-assignment heuristic; /optimize is
the authoritative LP. They are intentionally different (documented in the spec)."""

from __future__ import annotations

import logging
import math
from collections import Counter

logger = logging.getLogger(__name__)

# FourzynBurn STARTING template (the "open lineup slots" universe — BN/IL excluded;
# those are not slots a start/sit decision fills). Order = Yahoo display order.
STARTING_SLOTS: list[str] = [
    "C",
    "1B",
    "2B",
    "3B",
    "SS",
    "OF",
    "OF",
    "OF",
    "Util",
    "Util",
    "SP",
    "SP",
    "RP",
    "RP",
    "P",
    "P",
    "P",
    "P",
]

# Slots that are NOT in the active lineup (a player here is benched/stashed).
_NON_LINEUP_SLOTS = {"BN", "IL", "IL10", "IL15", "IL60", "NA", "DTD", "", "BENCH"}

_SCOPES = ("today", "rest_of_week", "rest_of_season")

# Statuses that make a player un-startable for the scope.
_INACTIVE_STATUSES = {
    "il10",
    "il15",
    "il60",
    "il",
    "na",
    "not active",
    "dl",
    "dtd",
    "day-to-day",
    "minors",
    "out",
    "suspended",
}


def _f(value, default: float = 0.0) -> float:
    """Finite-float coercion (None/NaN/inf/junk -> default) — keeps NaN/inf out of JSON."""
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return default
    return default if (math.isnan(fval) or math.isinf(fval)) else fval


class StartSitService:
    # ----------------------------------------------------------------- slot helpers
    @staticmethod
    def _eligible_slots(positions, is_hitter: bool) -> list[str]:
        """Which STARTING_SLOTS this player can fill, from its comma-separated
        eligible positions. Hitters also fill Util; pitchers also fill the generic
        P slot. Returns DISTINCT slot labels (template multiplicity handled by the
        assignment, not here)."""
        toks = {t.strip().upper() for t in str(positions or "").split(",") if t.strip()}
        out: set[str] = set()
        # Direct position-name matches against the template's distinct labels.
        template_labels = {"C", "1B", "2B", "3B", "SS", "OF", "SP", "RP"}
        for t in toks:
            if t in template_labels:
                out.add(t)
        if is_hitter:
            out.add("Util")
            # OF aliases (LF/CF/RF) -> OF slot.
            if toks & {"LF", "CF", "RF", "OF"}:
                out.add("OF")
        else:
            # Any pitcher (SP/RP/P) fills the generic P slot.
            if toks & {"SP", "RP", "P"}:
                out.add("P")
            if "P" in toks and not (toks & {"SP", "RP"}):
                # A pure-"P" pitcher can fill SP or RP too (eligible either way).
                out.update({"SP", "RP"})
        return [s for s in dict.fromkeys(STARTING_SLOTS) if s in out]

    @classmethod
    def _open_slots(cls, roster) -> dict[str, int]:
        """Open lineup slots by position = STARTING_SLOTS minus the user's CURRENT
        starters (rows whose selected_position is a real lineup slot). Empty/missing
        roster -> the full template (everything open)."""
        import pandas as pd

        template = Counter(STARTING_SLOTS)
        if not isinstance(roster, pd.DataFrame) or roster.empty or "selected_position" not in roster.columns:
            return dict(template)
        taken: Counter = Counter()
        for r in roster.to_dict("records"):
            sp = str(r.get("selected_position", "") or "").upper().strip()
            if sp in _NON_LINEUP_SLOTS:
                continue
            # Normalize Yahoo SP/RP/Util casing to the template label.
            label = {"UTIL": "Util", "SP": "SP", "RP": "RP", "P": "P"}.get(sp, sp)
            if label in template:
                taken[label] += 1
        return {slot: max(0, template[slot] - taken.get(slot, 0)) for slot in template}

    # Finite-float coercion exposed on the class (the module-level `_f` is reused so
    # both StartSitService._f(...) and self._f(...) resolve identically).
    _f = staticmethod(_f)

    # ----------------------------------------------------------------- compare
    @staticmethod
    def _scope(scope: str) -> str:
        s = str(scope or "").lower().strip()
        return s if s in _SCOPES else "rest_of_season"

    def compare(self, req) -> StartSitCompareResponse:  # noqa: F821 (forward ref via lazy import)
        from api.contracts.start_sit import StartSitCompareResponse, StartSitVerdict

        scope = self._scope(req.scope)
        # Clamp 2..6 (the engine cap is 6; <2 just yields a thin verdict, never raises).
        pids = [int(p) for p in (req.player_ids or [])][:6]

        candidates: list = []
        open_slots: dict[str, int] = {}
        verdict = StartSitVerdict()
        confidence = 0.0
        confidence_label = "Toss-up"
        try:
            from src.optimizer.shared_data_layer import build_optimizer_context
            from src.start_sit import start_sit_recommendation
            from src.valuation import LeagueConfig
            from src.yahoo_data_service import get_yahoo_data_service

            config = LeagueConfig()
            yds = get_yahoo_data_service()
            # Matchup-aware by construction: build_optimizer_context calls
            # yds.get_matchup() and sets ctx.category_weights from compute_urgency_weights.
            # weeks_remaining is intentionally left at the builder default to match every
            # other api/services/* caller (fa/streaming/schedule/team) — keeps a player's
            # projected line consistent across pages.
            ctx = build_optimizer_context(
                scope=scope,
                yds=yds,
                config=config,
                user_team_name=req.team_name,
                level_filter="All",
            )
            pool = getattr(ctx, "player_pool", None)
            roster = getattr(ctx, "roster", None)
            open_slots = self._open_slots(roster)

            # start_sit_recommendation already weighs categories internally; it falls
            # back to uniform weights when standings/totals are absent (never raises).
            rec = start_sit_recommendation(pids, pool, config)
            players = rec.get("players") if isinstance(rec, dict) else None
            if players:
                candidates = self._score_candidates(players, pool, scope, ctx)
                confidence = round(float(rec.get("confidence", 0.0) or 0.0), 4)
                confidence_label = self._confidence_label(confidence)
                verdict = self._greedy_verdict(candidates, open_slots)
        except Exception as exc:
            logger.warning("StartSitService.compare failed: %s", exc)

        return StartSitCompareResponse(
            scope=scope,
            candidates=candidates,
            verdict=verdict,
            open_slots=open_slots,
            confidence=confidence,
            confidence_label=confidence_label,
        )

    @staticmethod
    def _confidence_label(conf: float) -> str:
        if conf > 0.30:
            return "Clear"
        if conf > 0.15:
            return "Lean"
        return "Toss-up"

    def _score_candidates(self, players, pool, scope, ctx) -> list:
        """Engine player dicts -> ranked StartSitCandidate list (start_score normalized
        0-100 across the set, best = 100). players is already score-sorted desc by the
        engine; preserve that order for rank."""
        from api.contracts.start_sit import StartSitCandidate
        from api.services.player_ref import player_ref_from_pool

        raw = [self._f(p.get("start_score")) for p in players]
        # Normalize against the REAL max (not max-abs): best player -> 100, worse ->
        # lower. An all-negative slate (every player a bad matchup) has top <= 0, so
        # every score floors at 0.0 — the 0-100 contract holds (no negative heat bar).
        top = max(raw, default=0.0)
        pool_rows = self._pool_index(pool)

        out: list = []
        for rank, p in enumerate(players, start=1):
            try:
                pid = int(p.get("player_id", 0) or 0)
            except (TypeError, ValueError):
                pid = 0
            row = pool_rows.get(pid, {})
            is_hitter = bool(row.get("is_hitter", 1))
            score = self._f(p.get("start_score"))
            norm = round(min(100.0, max(0.0, 100.0 * score / top)), 1) if top > 0 else 0.0
            status = str(row.get("status", "") or "").strip().lower()
            out.append(
                StartSitCandidate(
                    player=player_ref_from_pool(pid, pool, name=p.get("name"), positions=row.get("positions")),
                    start_score=norm,
                    rank=rank,
                    eligible_slots=self._eligible_slots(row.get("positions"), is_hitter),
                    projected=self._projected_line(row, scope),
                    category_impact=self._impact_items(p.get("category_impact")),
                    matchup="",  # filled by /optimize's daily path; compare leaves blank (no schedule fetch)
                    reason="; ".join(str(r) for r in (p.get("reasoning") or [])[:2]),
                    playable=status not in _INACTIVE_STATUSES,
                )
            )
        return out

    @staticmethod
    def _pool_index(pool) -> dict[int, dict]:
        import pandas as pd

        if not isinstance(pool, pd.DataFrame) or pool.empty or "player_id" not in pool.columns:
            return {}
        out: dict[int, dict] = {}
        for r in pool.to_dict("records"):
            try:
                out[int(r.get("player_id", 0) or 0)] = r
            except (TypeError, ValueError):
                continue
        return out

    @staticmethod
    def _impact_items(category_impact) -> list:
        """Engine {cat: sgp} -> StatItem[] sorted by |impact| desc (top 4)."""
        from api.contracts.common import StatItem

        if not isinstance(category_impact, dict):
            return []
        items = sorted(category_impact.items(), key=lambda kv: abs(StartSitService._f(kv[1])), reverse=True)
        return [StatItem(label=str(k).upper(), value=f"{StartSitService._f(v):+.2f}") for k, v in items[:4]]

    def _projected_line(self, row, scope) -> list:
        """Scope-scaled projected stat line (StatItem[]). Uses compute_weekly_projection
        for the weekly/today shape; degrades to the row's season rates on any failure."""
        from api.contracts.common import StatItem

        try:
            import pandas as pd

            from src.start_sit import compute_weekly_projection

            wk = compute_weekly_projection(pd.Series(row)) if row else {}
        except Exception:
            wk = {}
        is_hitter = bool(row.get("is_hitter", 1))
        keys = ["r", "hr", "rbi", "sb", "avg", "obp"] if is_hitter else ["w", "k", "sv", "era", "whip"]
        out: list = []
        for k in keys:
            v = self._f(wk.get(k, row.get(k)))
            if k in ("avg", "obp"):
                disp = f"{v:.3f}".lstrip("0") if 0.0 <= v < 1.0 else f"{v:.3f}"
            elif k in ("era", "whip"):
                disp = f"{v:.2f}"
            else:
                disp = str(int(round(v)))
            out.append(StatItem(label=k.upper(), value=disp))
        return out

    def _greedy_verdict(self, candidates, open_slots) -> StartSitVerdict:  # noqa: F821
        """Bounded greedy slot assignment: walk candidates best-first, assign each to
        ANY open eligible slot (decrementing the slot count). Assigned -> start; the
        rest -> sit. This is a heuristic; /optimize is the authoritative LP."""
        from api.contracts.start_sit import StartSitVerdict

        remaining = dict(open_slots)
        start_ids: list[int] = []
        sit_ids: list[int] = []
        for c in candidates:
            placed = False
            if c.playable:
                for slot in c.eligible_slots:
                    if remaining.get(slot, 0) > 0:
                        remaining[slot] -= 1
                        start_ids.append(c.player.id)
                        placed = True
                        break
            if not placed:
                sit_ids.append(c.player.id)
        n = len(start_ids)
        reasoning = (
            f"Start the top {n} by projected value that fit your open slots; sit the rest."
            if n
            else "No open lineup slots for these players (or none are playable today)."
        )
        return StartSitVerdict(start_ids=start_ids, sit_ids=sit_ids, reasoning=reasoning)

    # ----------------------------------------------------------------- optimize (PATH-B)
    def optimize(self, req) -> StartSitOptimizeResponse:  # noqa: F821 (forward ref via lazy import)
        """Authoritative LP fill of the user's open slots with the SELECTED candidates.

        PATH-B (self-contained): WS1's LineupService.optimize() has no extra_ids hook,
        so we compose the pipeline directly — the selected candidates' pool rows are
        appended to the user's enriched roster (the roster the LP fills), so the LP can
        slot them into open lineup spots. The mapping reuses LineupService._to_slots /
        _daily_slots (single source of slot-shape truth). Matchup-aware via yds.get_matchup()
        (same as the Optimizer's standard/daily paths). Never raises → empty lineup."""
        from api.contracts.start_sit import StartSitOptimizeResponse
        from api.services.lineup_service import LineupService

        scope = self._scope(req.scope)
        pids = [int(p) for p in (req.player_ids or [])]
        slots: list = []
        bench: list = []
        summary = "Lineup unavailable (no live data in this environment)."
        daily = None
        try:
            from src.game_day import get_target_game_date
            from src.optimizer.pipeline import LineupOptimizerPipeline
            from src.valuation import LeagueConfig
            from src.yahoo_data_service import get_yahoo_data_service

            yds = get_yahoo_data_service()
            resolved_date = str(get_target_game_date())
            # Same enriched, team-filtered roster the Optimizer LP uses (projections +
            # positions); a bare league roster zeroes the LP objective.
            roster = yds.get_team_roster(req.team_name)
            matchup = None
            try:
                matchup = yds.get_matchup()
            except Exception:
                matchup = None

            pool = LineupService._load_pool()
            roster = self._augment_roster_with_candidates(roster, pool, pids)
            config = LeagueConfig()
            mode = "daily" if scope == "today" else "standard"
            pipeline = LineupOptimizerPipeline(roster, mode=mode, config=config)

            if mode == "daily":
                schedule = LineupService._schedule_today(resolved_date)
                lsvc = LineupService()
                inputs = lsvc._daily_inputs(roster, pool, schedule)
                result = pipeline.optimize(
                    matchup=matchup,
                    schedule_today=schedule,
                    park_factors=inputs["park_factors"],
                    confirmed_lineups=inputs["confirmed_lineups"],
                    recent_form=inputs["recent_form"],
                    team_strength=inputs["team_strength"],
                )
                result = result if isinstance(result, dict) else {}
                slots, bench = lsvc._daily_slots(
                    result.get("daily_dcv"), result.get("daily_lineup"), roster, pool, schedule
                )
                daily = lsvc._daily_meta(result, slots, roster, pool, resolved_date)
            else:
                result = pipeline.optimize(matchup=matchup) if hasattr(pipeline, "optimize") else None
                lineup = (result or {}).get("lineup") if isinstance(result, dict) else None
                slots, bench = LineupService._to_slots(lineup, pool, roster)

            if slots:
                summary = f"{len(slots)} starters set."
        except Exception as exc:
            logger.warning("StartSitService.optimize failed: %s", exc)
        return StartSitOptimizeResponse(scope=scope, slots=slots, bench=bench, summary=summary, daily=daily)

    @staticmethod
    def _augment_roster_with_candidates(roster, pool, pids):
        """Return the user's roster with the SELECTED candidates' pool rows appended
        (so the LP can fill open slots with them). Dedupes by player_id (roster row
        wins for an already-rostered candidate). Missing roster/pool degrade
        gracefully — a candidate the pool can't supply is simply skipped."""
        import pandas as pd

        if not isinstance(roster, pd.DataFrame):
            roster = pd.DataFrame()
        have = set()
        if not roster.empty and "player_id" in roster.columns:
            for v in roster["player_id"].tolist():
                try:
                    have.add(int(v))
                except (TypeError, ValueError):
                    continue
        if not isinstance(pool, pd.DataFrame) or pool.empty or "player_id" not in pool.columns:
            return roster
        want = [p for p in pids if p not in have]
        if not want:
            return roster
        extra = pool[pool["player_id"].isin(want)].copy()
        if extra.empty:
            return roster
        # Candidate rows enter as benched (no current lineup slot) so the LP decides
        # whether to start them; a missing column on either frame is fine (concat fills NaN).
        if "selected_position" not in extra.columns:
            extra["selected_position"] = "BN"
        if roster.empty:
            return extra.reset_index(drop=True)
        return pd.concat([roster, extra], ignore_index=True)

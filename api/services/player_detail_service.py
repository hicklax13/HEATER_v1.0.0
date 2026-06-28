"""Player-detail service — the ONE place that composes the player-card engines
into the PlayerDetail contract. Resilient: every live-source-dependent field
degrades to "—"/[] rather than raising. The only hard error is an unknown mlb_id
(404), since that means the player isn't tracked by HEATER at all.

Slice 1 fills the DB-backed core (identity, season line, prior years, ownership,
ranks, rostered-by, ROS projections, history). The l7/l14/l30 windows, game logs,
and near-term projection horizons are emitted as "—"/[] for a later statsapi slice.
"""

from __future__ import annotations

import logging
import math

from fastapi import HTTPException

from api.contracts.player_detail import (
    HistoryEvent,
    LabelValue,
    PlayerDetailResponse,
    PriorBlock,
    PriorRow,
    ProjRow,
    StatRow,
)
from api.services.player_ref import team_id_for

logger = logging.getLogger(__name__)

_HIT_CATS = ["R", "HR", "RBI", "SB", "AVG", "OBP"]
_PIT_CATS = ["W", "L", "SV", "K", "ERA", "WHIP"]
_RATE = {"AVG", "OBP", "ERA", "WHIP"}
_BATS = {"R": "Right", "L": "Left", "S": "Switch", "B": "Switch"}
# Transaction type → the frontend's history "kind".
_TXN_KIND = {"add": "added", "drop": "dropped", "trade": "traded", "draft": "drafted"}


def _f(value, default: float = 0.0) -> float:
    """Finite-float coercion (None/NaN/inf/junk → default)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return default if (math.isnan(f) or math.isinf(f)) else f


def _fmt(cat: str, value) -> str:
    """Display a category value: AVG/OBP → .3f, ERA/WHIP → .2f, counting → int."""
    f = _f(value)
    if cat in _RATE:
        return f"{f:.2f}" if cat in ("ERA", "WHIP") else f"{f:.3f}"
    return str(int(round(f)))


def _g(row, key, default=None):
    """Safe accessor for a pandas Series / dict row."""
    try:
        v = row.get(key, default) if hasattr(row, "get") else row[key]
    except (KeyError, TypeError):
        return default
    return default if v is None else v


class PlayerDetailService:
    def get(self, mlb_id: int) -> PlayerDetailResponse:
        from src.database import load_player_pool

        pool = self._load(load_player_pool)
        row = self._find(pool, mlb_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Player not found")

        is_pitcher = not bool(int(_f(_g(row, "is_hitter"), 1.0)))
        cats = _PIT_CATS if is_pitcher else _HIT_CATS
        player_id = int(_f(_g(row, "player_id"), 0.0))
        team_abbr = str(_g(row, "team", "") or "")
        name = str(_g(row, "name", "") or "")

        return PlayerDetailResponse(
            mlb_id=int(mlb_id),
            team_id=team_id_for(team_abbr),
            name=name,
            pos=str(_g(row, "positions", "") or ""),
            bats=self._bats(row, is_pitcher),
            jersey="",  # not in the pool; slice-2 enrichment
            team_name=team_abbr,
            is_pitcher=is_pitcher,
            own_pct=round(_f(_g(row, "percent_owned")), 1),
            own_delta=0.0,  # ownership-trend delta deferred (not in the pool)
            rostered_by=self._rostered_by(player_id),
            headline=self._headline(row, is_pitcher),
            ranks=self._ranks(row),
            game_columns=self._game_columns(is_pitcher),
            game_log=[],  # statsapi slice 2
            stats=[StatRow(cat=c, season=_fmt(c, _g(row, f"ytd_{c.lower()}"))) for c in cats],
            prior=self._prior(player_id, cats),
            # In-season, the pool's bare cat columns ARE the ROS projections
            # (_build_player_pool fills them from ros_projections when ros_count>0,
            # the live mode). Pre-season they'd be full-season blended — acceptable.
            projections=[ProjRow(cat=c, ros=_fmt(c, _g(row, c.lower()))) for c in cats],
            history=self._history(name),
        )

    # ── helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _load(loader):
        try:
            return loader()
        except Exception as exc:
            logger.warning("player_detail: %s failed: %s", getattr(loader, "__name__", "load"), exc)
            return None

    @staticmethod
    def _find(pool, mlb_id):
        import pandas as pd

        if pool is None or getattr(pool, "empty", True) or "mlb_id" not in getattr(pool, "columns", []):
            return None
        match = pool[pd.to_numeric(pool["mlb_id"], errors="coerce") == mlb_id]
        if match.empty:
            return None
        if len(match) > 1:
            logger.warning("player_detail: %d pool rows share mlb_id=%s; using the first", len(match), mlb_id)
        return match.iloc[0]

    @staticmethod
    def _bats(row, is_pitcher: bool) -> str:
        code = str(_g(row, "throws" if is_pitcher else "bats", "") or "").upper()[:1]
        side = _BATS.get(code, "")
        verb = "Throws" if is_pitcher else "Bats"
        return f"{verb} {side}" if side else ""

    def _rostered_by(self, player_id: int) -> str:
        if not player_id:
            return "Free Agent"
        from src.database import load_league_rosters

        rosters = self._load(load_league_rosters)
        try:
            if rosters is not None and not rosters.empty and "player_id" in rosters.columns:
                hit = rosters[rosters["player_id"] == player_id]
                if not hit.empty:
                    return str(_g(hit.iloc[0], "team_name", "") or "") or "Free Agent"
        except Exception as exc:
            logger.warning("player_detail._rostered_by failed for %s: %s", player_id, exc)
        return "Free Agent"

    @staticmethod
    def _headline(row, is_pitcher: bool) -> list[LabelValue]:
        rank = int(_f(_g(row, "consensus_rank")))
        vol_label, vol = ("IP", "ytd_ip") if is_pitcher else ("GP", "ytd_gp")
        cats = _PIT_CATS if is_pitcher else _HIT_CATS
        out: list[LabelValue] = []
        if rank > 0:
            out.append(LabelValue(label="Rank", value=f"#{rank}"))
        gp = _g(row, vol)
        if gp is not None:
            out.append(LabelValue(label=vol_label, value=_fmt(vol_label, gp)))
        # Per-category season highlights (the same ytd_* the Season Stats tab uses) so the
        # header shows R/HR/RBI… (or W/K/ERA…), not just Rank + GP.
        for c in cats:
            v = _g(row, f"ytd_{c.lower()}")
            if v is not None:
                out.append(LabelValue(label=c, value=_fmt(c, v)))
        return out

    @staticmethod
    def _ranks(row) -> list[LabelValue]:
        rank = int(_f(_g(row, "consensus_rank")))
        return [LabelValue(label="Season Rank", value=f"#{rank}")] if rank > 0 else []

    @staticmethod
    def _game_columns(is_pitcher: bool) -> list[str]:
        return ["IP", "K", "ER", "W/L"] if is_pitcher else ["AB", "H", "HR", "RBI"]

    def _prior(self, player_id: int, cats: list[str]) -> PriorBlock:
        y2025 = self._year_stats(player_id, 2025)
        y2024 = self._year_stats(player_id, 2024)
        if not y2025 and not y2024:
            return PriorBlock()
        rows = [
            PriorRow(
                cat=c,
                y2025=_fmt(c, y2025.get(c.lower())) if c.lower() in y2025 else "—",
                y2024=_fmt(c, y2024.get(c.lower())) if c.lower() in y2024 else "—",
            )
            for c in cats
        ]
        return PriorBlock(rows=rows)

    def _year_stats(self, player_id: int, year: int) -> dict:
        if not player_id:
            return {}
        from src.database import load_season_stats

        try:
            df = load_season_stats(year)
            if df is None or df.empty or "player_id" not in df.columns:
                return {}
            hit = df[df["player_id"] == player_id]
            if hit.empty:
                return {}
            return {k.lower(): v for k, v in hit.iloc[0].to_dict().items()}
        except Exception as exc:
            logger.warning("player_detail._year_stats(%s, %s) failed: %s", player_id, year, exc)
            return {}

    def _history(self, name: str) -> list[HistoryEvent]:
        """Transaction history for the player. ``load_transactions()`` returns
        ``player_name``/``type``/``team_from``/``team_to``/``timestamp`` (it drops
        player_id — joins to get the name), so we match by name."""
        if not name:
            return []
        from src.database import load_transactions

        try:
            df = load_transactions()
            if df is None or df.empty or "player_name" not in df.columns:
                return []
            hits = df[df["player_name"].astype(str).str.casefold() == name.casefold()]
            out: list[HistoryEvent] = []
            for _, t in hits.iterrows():
                txn = str(_g(t, "type", "") or "").lower()
                team_to = str(_g(t, "team_to", "") or "")
                team_from = str(_g(t, "team_from", "") or "")
                if txn == "add/drop":
                    # Yahoo's combined waiver move; this player is the add side
                    # (team_to set) or the drop side (team_from set).
                    kind = "added" if team_to else "dropped"
                else:
                    kind = _TXN_KIND.get(txn)
                if kind is None:
                    continue
                member = team_from if kind == "dropped" else team_to
                out.append(
                    HistoryEvent(
                        kind=kind,
                        date=str(_g(t, "timestamp", "") or ""),
                        text=self._txn_text(kind, team_from, team_to),
                        member=member,
                    )
                )
            return out
        except Exception as exc:
            logger.warning("player_detail._history failed for %r: %s", name, exc)
            return []

    @staticmethod
    def _txn_text(kind: str, team_from: str, team_to: str) -> str:
        if kind == "traded" and team_from and team_to:
            return f"Traded {team_from} → {team_to}"
        if kind == "added" and team_to:
            return f"Added by {team_to}"
        if kind == "dropped" and team_from:
            return f"Dropped by {team_from}"
        if kind == "drafted" and team_to:
            return f"Drafted by {team_to}"
        return kind.title()

"""Closer Monitor service — the ONE place that calls the closer engine.
Maps engine output → the Closers contract. Resilient: missing live data
degrades to an empty entry list rather than raising."""

from __future__ import annotations

from api.contracts.closers import CloserEntry, ClosersResponse
from api.contracts.common import PlayerRef
from api.services.player_ref import make_player_ref


class CloserService:
    def get_closers(self) -> ClosersResponse:
        from src.closer_monitor import build_closer_grid, build_depth_data_from_db
        from src.database import load_player_pool

        entries: list[CloserEntry] = []
        try:
            depth_data = build_depth_data_from_db()
            player_pool = None
            try:
                player_pool = load_player_pool()
            except Exception:
                pass  # pool unavailable — grid still works with defaults
            grid = build_closer_grid(depth_data, player_pool=player_pool)
            for row in grid or []:
                entries.append(self._to_entry(row, player_pool))
        except Exception:
            entries = []  # cold env / no data → empty list
        return ClosersResponse(entries=entries)

    @staticmethod
    def _f(value, default: float = 0.0) -> float:
        try:
            fv = float(value)
        except (TypeError, ValueError):
            return default
        return default if (fv != fv or fv in (float("inf"), float("-inf"))) else fv

    @staticmethod
    def _handcuff_ref(name, pool) -> PlayerRef | None:
        """Resolve a setup-man name to a rich PlayerRef via the pool ONLY when the
        name matches exactly one row (Muncy-DNA guard: ambiguous → name-only, never
        a guessed id). Falls back to a name-only PlayerRef."""
        import pandas as pd

        nm = str(name or "").strip()
        if not nm:
            return None
        if isinstance(pool, pd.DataFrame) and not pool.empty and "name" in pool.columns:
            try:
                m = pool[pool["name"].astype(str).str.strip().str.lower() == nm.lower()]
                if len(m) == 1:
                    r = m.iloc[0]
                    return make_player_ref(
                        id=int(r.get("player_id", 0) or 0),
                        name=nm,
                        positions=str(r.get("positions", "RP") or "RP"),
                        mlb_id=r.get("mlb_id"),
                        team_abbr=str(r.get("team", "") or ""),
                    )
            except Exception:
                pass
        return PlayerRef(id=0, name=nm, positions="RP")

    @staticmethod
    def _to_entry(row: dict, pool=None) -> CloserEntry:
        """Map a build_closer_grid row dict to a CloserEntry.

        Grid row keys: team, closer_name, setup_names, job_security,
        security_color, projected_sv, era, whip, mlb_id.
        """
        g = row.get if isinstance(row, dict) else lambda k, d=None: getattr(row, k, d)

        team = str(g("team", "") or "")
        closer_name = str(g("closer_name", "") or "").strip()
        mlb_id = g("mlb_id")
        security = CloserService._f(g("job_security", 0.5), 0.5)

        # Build the closer PlayerRef (None when no closer identified)
        closer_ref: PlayerRef | None = None
        if closer_name and closer_name.lower() not in ("unknown", ""):
            closer_ref = make_player_ref(
                id=int(mlb_id) if mlb_id is not None else 0,
                name=closer_name,
                positions="RP",
                mlb_id=mlb_id,
                team_abbr=team,
            )

        # Confidence label derived from job_security float
        if security >= 0.7:
            confidence = "Firm"
        elif security >= 0.4:
            confidence = "Shaky"
        else:
            confidence = "Uncertain"

        # Role label
        # build_closer_grid doesn't output committee flag directly on the row dict
        # — infer from closer_name == "Committee" (set by build_depth_data_from_db)
        role = "Committee" if closer_name == "Committee" else "Closer"

        # Handcuffs from setup_names — pool-resolved (mlb_id) when the name is unambiguous.
        setup_names = g("setup_names", []) or []
        handcuffs: list[PlayerRef] = []
        for name in setup_names or []:
            ref = CloserService._handcuff_ref(name, pool)
            if ref is not None:
                handcuffs.append(ref)

        return CloserEntry(
            team=team,
            closer=closer_ref,
            role=role,
            confidence=confidence,
            handcuffs=handcuffs,
            job_security=round(security, 3),
            security_color=str(g("security_color", "") or ""),
            projected_sv=CloserService._f(g("projected_sv", 0.0)),
            era=CloserService._f(g("era", 0.0)),
            whip=CloserService._f(g("whip", 0.0)),
        )

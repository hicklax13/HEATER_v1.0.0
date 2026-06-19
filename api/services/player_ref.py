"""Shared PlayerRef construction with display enrichment (mlb_id + team).

The frontend renders headshots (via mlb_id) and team logos (via team_abbr /
team_id) from PlayerRef. This module centralizes:
  - team_abbr -> MLB Stats API numeric team_id, reusing the single-source
    30-team map in src.depth_charts (no duplicate map).
  - normalization of mlb_id (0 / negative / NaN -> None) and team_abbr
    (blank -> None).

It lives in api/services because that is the one layer permitted to import
from src/. The src import is lazy (inside team_id_for) to keep this module's
import graph light and avoid any startup-import surprise.
"""

from __future__ import annotations

from api.contracts.common import PlayerRef


def team_id_for(team_abbr: str | None) -> int | None:
    """Map a canonical MLB team abbreviation to its MLB Stats API numeric id.

    Returns None for unknown/blank abbreviations. Reuses the single-source
    `_STATSAPI_TEAM_IDS` map in src.depth_charts so the abbr->id mapping is
    not duplicated.
    """
    if not team_abbr:
        return None
    from src.depth_charts import _STATSAPI_TEAM_IDS

    return _STATSAPI_TEAM_IDS.get(team_abbr.strip().upper())


def _coerce_mlb_id(value) -> int | None:
    """Coerce a raw mlb_id (int/float/str/None/NaN) to a positive int or None."""
    if value is None:
        return None
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return None
    return ivalue if ivalue > 0 else None


def _coerce_team_abbr(value) -> str | None:
    """Coerce a raw team value to a clean abbreviation, or None.

    Handles None, float NaN, and pandas NA/NaT (which the player pool can
    yield when the `team` column is float-typed) — mirroring _coerce_mlb_id's
    NaN discipline so a missing team never leaks as the string "nan"/"<NA>".
    """
    if value is None:
        return None
    try:
        import pandas as pd

        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass  # non-scalar / unrecognized -> fall through to string coercion
    text = str(value).strip()
    return text or None


def make_player_ref(
    *,
    id: int,
    name: str,
    positions: str,
    mlb_id=None,
    team_abbr=None,
    yahoo_player_key: str | None = None,
) -> PlayerRef:
    """Build a PlayerRef with display enrichment populated where available.

    mlb_id is normalized (0 / negative / NaN -> None); team_abbr is stripped
    (blank -> None) and team_id is derived from it. All enrichment fields
    default to None so callers that lack the data still emit a valid ref.
    """
    abbr = _coerce_team_abbr(team_abbr)
    return PlayerRef(
        id=int(id) if id is not None else 0,
        mlb_id=_coerce_mlb_id(mlb_id),
        name=str(name) if name is not None else "",
        positions=str(positions) if positions is not None else "",
        team_abbr=abbr,
        team_id=team_id_for(abbr),
        yahoo_player_key=yahoo_player_key,
    )


def player_ref_from_pool(
    player_id,
    pool,
    *,
    name=None,
    positions=None,
    yahoo_player_key: str | None = None,
) -> PlayerRef:
    """Build an enriched PlayerRef by looking up player_id in the player pool.

    Pulls mlb_id + team (-> team_id) from the matching pool row. PREFERS the
    caller-provided name/positions (the engine's view) and only falls back to
    the pool's when they're absent, then to a "Player {id}" placeholder. A
    missing/None/empty pool or a not-found id degrades to a valid un-enriched
    ref (never raises).
    """
    try:
        pid = int(player_id) if player_id is not None else 0
    except (TypeError, ValueError):
        pid = 0
    mlb_id = None
    team_abbr = None
    final_name = name
    final_pos = positions
    try:
        import pandas as pd

        if isinstance(pool, pd.DataFrame) and not pool.empty:
            match = pool[pool["player_id"] == pid]
            if not match.empty:
                r = match.iloc[0]
                mlb_id = r.get("mlb_id")
                team_abbr = r.get("team")
                if not final_name:
                    name_col = "player_name" if "player_name" in pool.columns else "name"
                    final_name = r.get(name_col)
                if not final_pos:
                    final_pos = r.get("positions")
    except Exception:
        pass
    return make_player_ref(
        id=pid,
        name=str(final_name) if final_name else f"Player {pid}",
        positions=str(final_pos) if final_pos else "",
        mlb_id=mlb_id,
        team_abbr=team_abbr,
        yahoo_player_key=yahoo_player_key,
    )

"""Contract models for roster write-back (set-lineup + add/drop).

All mutation outcomes return HTTP 200; success/failure is carried in the `ok`
field of MutationResult — the underlying YahooFantasyClient methods never raise,
they return {"ok": ...}. The frontend owns the confirm dialog."""

from __future__ import annotations

from pydantic import BaseModel


class LineupAssignment(BaseModel):
    yahoo_player_key: str  # authoritative Yahoo id, e.g. "469.p.11111"
    slot: str  # Yahoo slot token: "SS","OF","BN","SP","Util","IL"…
    player_id: int | None = None  # optional HEATER id, echoed for display/audit


class LineupSetRequest(BaseModel):
    team_name: str  # echoed; today the write targets the authenticated team
    date: str  # "YYYY-MM-DD"
    assignments: list[LineupAssignment]


class AddDropRequest(BaseModel):
    add_player_key: str | None = None  # Yahoo player_key to add, or None
    drop_player_key: str | None = None  # Yahoo player_key to drop, or None
    # ≥1 expected; the client returns ok=False if both are None (no validator — keep it thin)


class MutationResult(BaseModel):
    ok: bool
    applied: int | None = None  # set-lineup: number of assignments applied
    error: str | None = None  # graceful Yahoo message (incl. write-scope re-auth)
    status: int | None = None  # Yahoo HTTP status on failure (e.g. 401/403)

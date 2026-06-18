"""Roster write service — the ONE place in the write path that touches src/.

Thin pass-through to YahooFantasyClient.set_lineup / add_drop (both already
graceful: they return {"ok": bool, "error"?, "status"?, "applied"?} and never
raise). Maps that dict → MutationResult. Reaches the write-capable client via
the YahooDataService singleton's ._client; in a cold/dormant environment that
client is None → returns a graceful "Not connected" result.

NOTE: today the write always targets the authenticated user's team (the client
discovers the team key itself). At B4, tenant/auth resolution selects the team —
this single service is the seam where that lands."""

from __future__ import annotations

from api.contracts.roster_write import AddDropRequest, LineupSetRequest, MutationResult

_NOT_CONNECTED = "Not connected to Yahoo."


class RosterWriteService:
    def set_lineup(self, req: LineupSetRequest, client=None) -> MutationResult:
        c = client if client is not None else self._client()
        if c is None:
            return MutationResult(ok=False, error=_NOT_CONNECTED, status=None)
        assignments = [{"player_key": a.yahoo_player_key, "position": a.slot} for a in req.assignments]
        return self._to_result(c.set_lineup(assignments, req.date))

    def add_drop(self, req: AddDropRequest, client=None) -> MutationResult:
        c = client if client is not None else self._client()
        if c is None:
            return MutationResult(ok=False, error=_NOT_CONNECTED, status=None)
        return self._to_result(c.add_drop(req.add_player_key, req.drop_player_key))

    @staticmethod
    def _client():
        # The write-capable client is the YahooDataService singleton's wrapped
        # YahooFantasyClient. None in a cold/dormant env (no live session).
        try:
            from src.yahoo_data_service import get_yahoo_data_service

            return get_yahoo_data_service()._client
        except Exception:
            return None

    @staticmethod
    def _to_result(raw) -> MutationResult:
        if not isinstance(raw, dict):
            return MutationResult(ok=False, error="Unexpected response from Yahoo client.", status=None)
        return MutationResult(
            ok=bool(raw.get("ok", False)),
            applied=raw.get("applied"),
            error=raw.get("error"),
            status=raw.get("status"),
        )

"""Roster write service — the ONE place in the write path that touches src/.

Thin pass-through to YahooFantasyClient.set_lineup / add_drop (both already
graceful: they return {"ok": bool, "error"?, "status"?, "applied"?} and never
raise). Maps that dict → MutationResult. Reaches the write-capable client via
the YahooDataService singleton's ._client; in a cold/dormant environment that
client is None → returns a graceful "Not connected" result.

Cross-team write guard (MED-1):
  When Clerk is configured, only the token-owner's team may mutate.  The guard
  lives HERE (not in the router) so tests/api/test_no_logic_in_routers.py stays
  green. Dormant (Clerk off) = guard is a strict no-op so the single-owner path
  is byte-for-byte unchanged.

Traceback logging (MED-3):
  Exception paths now log with exc_info=True so operators see the full traceback,
  not just the exception class name."""

from __future__ import annotations

import logging

from api.auth import clerk_configured
from api.contracts.roster_write import AddDropRequest, LineupSetRequest, MutationResult

logger = logging.getLogger(__name__)

_NOT_CONNECTED = "Not connected to Yahoo."
_FORBIDDEN = "Not authorized to modify another team's roster."


class RosterWriteService:
    # ------------------------------------------------------------------
    # Cross-team authorization guard
    # ------------------------------------------------------------------

    def _owner_team(self) -> str | None:
        """Resolve the token-owner's team name from the Yahoo data service.
        Returns None if unavailable (cold env / exception)."""
        try:
            from src.yahoo_data_service import get_yahoo_data_service

            return get_yahoo_data_service()._get_user_team_name()
        except Exception as exc:
            logger.warning("RosterWriteService._owner_team failed: %s", exc)
            return None

    def _authorized(self, caller_team: str | None) -> bool:
        """True when the caller is allowed to write.

        Dormant path (Clerk off): always True — no multi-tenant risk yet.
        Live path (Clerk on):     caller_team must equal the token-owner's team."""
        if not clerk_configured():
            return True
        owner = self._owner_team()
        return bool(caller_team) and bool(owner) and caller_team == owner

    # ------------------------------------------------------------------
    # Public write methods
    # ------------------------------------------------------------------

    def set_lineup(self, req: LineupSetRequest, caller_team: str | None = None, client=None) -> MutationResult:
        if not self._authorized(caller_team):
            logger.warning("RosterWriteService.set_lineup refused: caller_team=%r", caller_team)
            return MutationResult(ok=False, error=_FORBIDDEN, status=None)

        c = client if client is not None else self._client()
        if c is None:
            return MutationResult(ok=False, error=_NOT_CONNECTED, status=None)
        assignments = [{"player_key": a.yahoo_player_key, "position": a.slot} for a in req.assignments]
        return self._safe_call("set_lineup", lambda: c.set_lineup(assignments, req.date))

    def add_drop(self, req: AddDropRequest, caller_team: str | None = None, client=None) -> MutationResult:
        if not self._authorized(caller_team):
            logger.warning("RosterWriteService.add_drop refused: caller_team=%r", caller_team)
            return MutationResult(ok=False, error=_FORBIDDEN, status=None)

        c = client if client is not None else self._client()
        if c is None:
            return MutationResult(ok=False, error=_NOT_CONNECTED, status=None)
        return self._safe_call("add_drop", lambda: c.add_drop(req.add_player_key, req.drop_player_key))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _client():
        # The write-capable client is the YahooDataService singleton's wrapped
        # YahooFantasyClient. None in a cold/dormant env (no live session).
        try:
            from src.yahoo_data_service import get_yahoo_data_service

            return get_yahoo_data_service()._client
        except Exception:
            return None

    def _safe_call(self, method_name: str, fn) -> MutationResult:
        # The client documents "never raises", but its auth/XML helpers run
        # outside its own try/except — wrap defensively so a write endpoint
        # can never 500 (matches the sibling read services' resilience idiom).
        try:
            raw = fn()
        except Exception as exc:
            logger.warning("RosterWriteService.%s failed: %s", method_name, exc, exc_info=True)
            return MutationResult(ok=False, error=f"Write failed: {type(exc).__name__}", status=None)
        return self._to_result(raw)

    @staticmethod
    def _to_result(raw) -> MutationResult:
        if not isinstance(raw, dict):
            logger.warning(
                "RosterWriteService write returned ok=False: unexpected response type %s",
                type(raw).__name__,
            )
            return MutationResult(ok=False, error="Unexpected response from Yahoo client.", status=None)
        result = MutationResult(
            ok=bool(raw.get("ok", False)),
            applied=raw.get("applied"),
            error=raw.get("error"),
            status=raw.get("status"),
        )
        if not result.ok:
            logger.warning("RosterWriteService write returned ok=False: %s", result.error)
        return result

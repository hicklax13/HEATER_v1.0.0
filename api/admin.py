"""Admin authorization + assignment logic for the beta user->team mapping.

require_admin: deny-by-default env allowlist (HEATER_ADMIN_CLERK_IDS, comma-sep of
Clerk user ids) over a MANDATORY app user — the env-token/no-clerk path is never an
admin. Team-name reconciliation reuses the EXISTING RosterQueryService (no new src
import here) and is graceful when the roster source is cold."""

from __future__ import annotations

import os
from collections.abc import Callable

from fastapi import Depends, HTTPException, status

from api.identity import require_app_user
from api.stores.user_store import AppUser

TeamNamesProvider = Callable[[], list[str]]


def _admin_ids() -> set[str]:
    raw = os.environ.get("HEATER_ADMIN_CLERK_IDS", "")
    return {p.strip() for p in raw.split(",") if p.strip()}


def require_admin(app_user: AppUser | None = Depends(require_app_user)) -> AppUser:
    """Allow only configured Clerk admins. Deny-by-default: empty allowlist → no
    admins; env-token path (app_user None) → never admin."""
    ids = _admin_ids()
    if app_user is None or app_user.clerk_user_id not in ids:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required.")
    return app_user


def _live_team_names() -> list[str]:
    """The league's canonical team names from the existing roster service (empty on
    any failure — keeps assignment graceful when the DB is cold)."""
    from api.services.roster_query_service import RosterQueryService

    return [t.team_name for t in RosterQueryService().league_rosters().teams]


def get_team_names_provider() -> TeamNamesProvider:
    """DI seam so tests inject a fake names list (no live DB)."""
    return _live_team_names

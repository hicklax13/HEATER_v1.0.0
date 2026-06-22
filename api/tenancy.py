"""Viewer tenancy resolution — maps a verified identity to their team within a
league (the replacement for the trusted team_name query param).

normalize_team_name REPLICATES the behavior of src.auth._normalize_team_name (a
non-service api module importing src/ would break the 'services are the one place
importing src/' discipline; re-homing it would be a src/ edit). The resolver
(require_viewer_context) composes the OPTIONAL identity path so currently-open
reads stay open when Clerk is off."""

from __future__ import annotations

import re
from collections.abc import Iterable

from fastapi import Depends, HTTPException, status
from pydantic import BaseModel

from api.auth import clerk_configured
from api.deps import get_league_store, get_membership_store
from api.identity import optional_app_user
from api.stores.league_store import LeagueStore
from api.stores.membership_store import MembershipStore
from api.stores.user_store import AppUser


def normalize_team_name(name: object) -> str:
    """Lowercase + strip all non-alphanumerics (emoji/whitespace/punctuation) so a
    name missing the Yahoo team's leading emoji ('Team Hickey') still matches the
    roster name ('🏆 Team Hickey'). Mirrors src.auth._normalize_team_name."""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def reconcile_team_name(assigned: str, roster_names: Iterable[str]) -> str | None:
    """Map an admin-typed team name to the EXACT roster name.

    - exact string match → that name (short-circuit);
    - tolerant (normalized) match → the exact roster name;
    - no match but roster_names is non-empty → None (caller signals 422);
    - roster_names empty (cold source) → the assigned name as-is (never block)."""
    names = [str(n) for n in roster_names if str(n).strip()]
    if not names:
        return assigned
    if assigned in names:
        return assigned
    target = normalize_team_name(assigned)
    for n in names:
        if normalize_team_name(n) == target:
            return n
    return None


class ViewerContext(BaseModel):
    """The resolved viewer. team_name is None when there is no authenticated user
    or no assignment (→ effective_team falls back to the endpoint's query param)."""

    user_id: int | None = None
    league_id: int | None = None
    team_name: str | None = None

    def effective_team(self, fallback: str | None) -> str | None:
        """The viewer's team for an endpoint, in three states:
        - assigned        → the resolved team (ignores the client fallback);
        - authed+unassigned (user_id set, no team) → None (NEVER the fallback —
          closes the cross-team exposure; the team-required routers map this to 409);
        - dormant (no identity) → the endpoint's query-param fallback (today's
          open behavior, byte-for-byte)."""
        if self.team_name:
            return self.team_name
        if self.user_id is not None:
            return None
        return fallback


def require_viewer_context(
    app_user: AppUser | None = Depends(optional_app_user),
    league_store: LeagueStore = Depends(get_league_store),
    membership_store: MembershipStore = Depends(get_membership_store),
) -> ViewerContext:
    """Resolve the viewer's team from their (optional) Clerk identity. Dormant when
    no/invalid token → empty ViewerContext (the endpoint falls back to its query
    param = today's behavior). A logged-in user with no assignment → team_name None
    (never another user's team)."""
    if app_user is None:
        # Activation flip: once Clerk is live, an open read with no valid login is
        # rejected (each user must log in → only sees their own team). Dormant until
        # CLERK_ISSUER is set, so today's open reads + existing tests are unchanged.
        if clerk_configured():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Login required.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return ViewerContext()
    league = league_store.get_or_create_default()
    membership = membership_store.get_for_user(app_user.id, league.id)
    return ViewerContext(
        user_id=app_user.id,
        league_id=league.id,
        team_name=(membership.team_name if membership else None),
    )

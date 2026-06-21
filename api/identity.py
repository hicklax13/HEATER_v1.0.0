"""Request-identity dependencies that need api-owned persistence.

Kept separate from api/auth.py (which imports nothing from api.deps, to stay
import-cycle-free). This module composes auth (require_principal) + the UserStore
to provision a local AppUser on the first authenticated Clerk call. Dormant on the
env-token path (Principal.clerk_user_id is None → no provisioning, no table)."""

from __future__ import annotations

from fastapi import Depends

from api.auth import Principal, require_principal
from api.deps import get_user_store
from api.stores.user_store import AppUser, UserStore


def provision_app_user(principal: Principal, store: UserStore) -> AppUser | None:
    """Get-or-create the local AppUser for a verified Clerk caller. Returns None
    for the env-token path (no clerk_user_id) so reads/server-to-server stay
    dormant. Pure + DB-free-testable (the store is injected)."""
    if not principal.clerk_user_id:
        return None
    return store.get_or_create(principal.clerk_user_id)


def require_app_user(
    principal: Principal = Depends(require_principal),
    store: UserStore = Depends(get_user_store),
) -> AppUser | None:
    """FastAPI dependency: a verified caller + their provisioned local AppUser
    (None on the env-token path). Slice 2's billing routes depend on this so a
    Stripe customer can be tied to a stable local user id."""
    return provision_app_user(principal, store)

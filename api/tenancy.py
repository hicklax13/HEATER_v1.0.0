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

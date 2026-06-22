"""Per-tier DAILY managed-AI cap (Bubba B3, lean). The shared-key ('managed')
daily $-allowance per subscription tier. BYO-key users are exempt upstream
(budget.is_over_cap's on_own_key short-circuit). Values are tunable; Plus/Max
slot in here when those tiers ship."""

from __future__ import annotations

_FREE_DAILY_CAP_USD = 0.10  # a small daily "taste" of managed AI on a cheap model
_TIER_DAILY_CAP_USD: dict[str, float] = {
    "free": _FREE_DAILY_CAP_USD,
    "pro": 2.00,
}


def managed_cap_for_tier(tier: str) -> float:
    """Daily managed-AI $ cap for a subscription tier. Unknown tier -> the free cap."""
    return _TIER_DAILY_CAP_USD.get(tier, _FREE_DAILY_CAP_USD)

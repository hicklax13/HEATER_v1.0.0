"""Start/Sit service — the ONE place importing src.start_sit + the optimizer
context/daily-DCV engines for this feature. Maps engine output -> the Start/Sit
contracts. Resilient: missing live data degrades to empty candidates / full-open
slots rather than raising.

The /compare verdict is a BOUNDED greedy slot-assignment heuristic; /optimize is
the authoritative LP. They are intentionally different (documented in the spec)."""

from __future__ import annotations

import logging
import math
from collections import Counter

logger = logging.getLogger(__name__)

# FourzynBurn STARTING template (the "open lineup slots" universe — BN/IL excluded;
# those are not slots a start/sit decision fills). Order = Yahoo display order.
STARTING_SLOTS: list[str] = [
    "C",
    "1B",
    "2B",
    "3B",
    "SS",
    "OF",
    "OF",
    "OF",
    "Util",
    "Util",
    "SP",
    "SP",
    "RP",
    "RP",
    "P",
    "P",
    "P",
    "P",
]

# Slots that are NOT in the active lineup (a player here is benched/stashed).
_NON_LINEUP_SLOTS = {"BN", "IL", "IL10", "IL15", "IL60", "NA", "DTD", "", "BENCH"}

_SCOPES = ("today", "rest_of_week", "rest_of_season")

# Statuses that make a player un-startable for the scope.
_INACTIVE_STATUSES = {
    "il10",
    "il15",
    "il60",
    "il",
    "na",
    "not active",
    "dl",
    "dtd",
    "day-to-day",
    "minors",
    "out",
    "suspended",
}


def _f(value, default: float = 0.0) -> float:
    """Finite-float coercion (None/NaN/inf/junk -> default) — keeps NaN/inf out of JSON."""
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return default
    return default if (math.isnan(fval) or math.isinf(fval)) else fval


class StartSitService:
    # ----------------------------------------------------------------- slot helpers
    @staticmethod
    def _eligible_slots(positions, is_hitter: bool) -> list[str]:
        """Which STARTING_SLOTS this player can fill, from its comma-separated
        eligible positions. Hitters also fill Util; pitchers also fill the generic
        P slot. Returns DISTINCT slot labels (template multiplicity handled by the
        assignment, not here)."""
        toks = {t.strip().upper() for t in str(positions or "").split(",") if t.strip()}
        out: set[str] = set()
        # Direct position-name matches against the template's distinct labels.
        template_labels = {"C", "1B", "2B", "3B", "SS", "OF", "SP", "RP"}
        for t in toks:
            if t in template_labels:
                out.add(t)
        if is_hitter:
            out.add("Util")
            # OF aliases (LF/CF/RF) -> OF slot.
            if toks & {"LF", "CF", "RF", "OF"}:
                out.add("OF")
        else:
            # Any pitcher (SP/RP/P) fills the generic P slot.
            if toks & {"SP", "RP", "P"}:
                out.add("P")
            if "P" in toks and not (toks & {"SP", "RP"}):
                # A pure-"P" pitcher can fill SP or RP too (eligible either way).
                out.update({"SP", "RP"})
        return [s for s in dict.fromkeys(STARTING_SLOTS) if s in out]

    @classmethod
    def _open_slots(cls, roster) -> dict[str, int]:
        """Open lineup slots by position = STARTING_SLOTS minus the user's CURRENT
        starters (rows whose selected_position is a real lineup slot). Empty/missing
        roster -> the full template (everything open)."""
        import pandas as pd

        template = Counter(STARTING_SLOTS)
        if not isinstance(roster, pd.DataFrame) or roster.empty or "selected_position" not in roster.columns:
            return dict(template)
        taken: Counter = Counter()
        for r in roster.to_dict("records"):
            sp = str(r.get("selected_position", "") or "").upper().strip()
            if sp in _NON_LINEUP_SLOTS:
                continue
            # Normalize Yahoo SP/RP/Util casing to the template label.
            label = {"UTIL": "Util", "SP": "SP", "RP": "RP", "P": "P"}.get(sp, sp)
            if label in template:
                taken[label] += 1
        return {slot: max(0, template[slot] - taken.get(slot, 0)) for slot in template}

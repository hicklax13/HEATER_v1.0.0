"""Pure, self-contained helper functions extracted from pages/2_Line-up_Optimizer.py.

All helpers here are:
- Stateless (no Streamlit calls, no THEME references, no session-state access).
- Fully covered by tests/test_lineup_optimizer_helpers.py.
- Safe to import from background threads (no st.* usage).

Do NOT add helpers that reference THEME, st.*, or page-closure variables here.
"""

from __future__ import annotations

import pandas as pd


def fmt_deltas(d: dict) -> str:
    """Format a category-delta dict into a human-readable string.

    Each entry is rendered as ``CAT +1.23`` or ``CAT -0.45``, sorted by
    absolute magnitude descending.  Returns an em-dash when *d* is empty.

    Args:
        d: Mapping of category-key (str) to numeric delta.

    Returns:
        Comma-joined string, e.g. ``"HR +2.00, AVG -0.01"``.
    """
    if not d:
        return "—"
    return ", ".join(f"{k.upper()} {v:+.2f}" for k, v in sorted(d.items(), key=lambda kv: -abs(kv[1])))


def decision_row_classes(df_in: pd.DataFrame) -> dict[int, str]:
    """Map row indices to CSS class names based on the ``Decision`` column.

    Used with ``render_compact_table(row_classes=...)`` to colour-code
    start/bench/IL/empty rows.

    Decision values → CSS class:
        ``"START ⚠"`` (forced start) → ``"row-start-forced"``
        ``"START"`` or starts-with-"START" → ``"row-start"``
        ``"IL"``                        → ``"row-il"``
        ``"LEAVE EMPTY"``               → ``"row-empty"``
        everything else                 → ``"row-bench"``

    Args:
        df_in: DataFrame that must contain a ``"Decision"`` column.

    Returns:
        ``{row_index: css_class_name}`` for every row in *df_in*.
    """
    classes: dict[int, str] = {}
    for i in range(len(df_in)):
        decision = str(df_in.iloc[i].get("Decision", "BENCH"))
        if decision.startswith("START") and "⚠" in decision:
            classes[i] = "row-start-forced"
        elif decision.startswith("START"):
            classes[i] = "row-start"
        elif decision == "IL":
            classes[i] = "row-il"
        elif decision == "LEAVE EMPTY":
            classes[i] = "row-empty"
        else:
            classes[i] = "row-bench"
    return classes


def slot_sort_key(df_in: pd.DataFrame, order_map: dict[str, int]) -> pd.Series:
    """Return a sort-key Series mapping each row's slot to its display order.

    Rows whose ``selected_position`` is not in *order_map* receive a value
    one larger than the maximum in the map (sorts to the bottom).

    Args:
        df_in:      DataFrame with a ``"selected_position"`` column.
        order_map:  Slot-name → integer-priority mapping (lower = earlier).

    Returns:
        A pandas Series of integers, aligned to *df_in*'s index.
    """
    default = max(order_map.values()) + 1
    return df_in["selected_position"].map(lambda s: order_map.get(s, default))


def expand_skeleton(slots_dict: dict) -> list[str]:
    """Expand a slot-spec dict into an ordered flat list of slot names.

    Converts ``{"C": (1, [...]), "OF": (3, [...])}`` into
    ``["C", "OF", "OF", "OF"]``.

    Args:
        slots_dict: Mapping of slot-name to ``(count, eligible_codes)`` pairs.

    Returns:
        Flat list where each slot-name appears *count* times.
    """
    expanded: list[str] = []
    for slot, (count, _eligible_codes) in slots_dict.items():
        for _ in range(count):
            expanded.append(slot)
    return expanded

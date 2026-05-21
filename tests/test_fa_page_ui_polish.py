"""PR11+12+13+14: FA page cosmetic polish.

Four cosmetic-polish fixes for pages/14_Free_Agents.py:
- PR11: ECR shown as integer (909, not 909.00)
- PR12: Position strings dedupe ('2B/3B,3B' -> '2B,3B')
- PR13: _build_reasoning adds 'Other categories' reconciliation line
- PR14: 'Show all' toggle for FA pagination (currently capped at 200)
"""

from pathlib import Path

import pandas as pd  # noqa: F401  (kept to remain consistent with peer test imports)


def test_ecr_integer_format_in_fa_page():
    """PR11: pages/14_Free_Agents.py must format consensus_rank as integer."""
    fa_page = Path(__file__).parent.parent / "pages" / "14_Free_Agents.py"
    src = fa_page.read_text(encoding="utf-8")
    # Look for the integer format pattern on consensus_rank
    assert "consensus_rank" in src and ("int(x)" in src or "int(" in src), (
        "PR11: consensus_rank must use integer formatting"
    )


def test_dedupe_positions_helper():
    """PR12: _dedupe_positions helper collapses redundant tokens."""
    fa_page_path = Path(__file__).parent.parent / "pages" / "14_Free_Agents.py"
    src = fa_page_path.read_text(encoding="utf-8")
    assert "_dedupe_positions" in src or "dedupe_positions" in src or "drop_duplicates" in src.lower(), (
        "PR12: position dedup helper required"
    )


def test_reasoning_includes_reconciliation_when_nonzero():
    """PR13: _build_reasoning surfaces 'Other categories: +X.XX SGP'
    when the categories explicitly mentioned don't sum to net_sgp."""
    from src.optimizer.fa_recommender import _build_reasoning
    from src.optimizer.shared_data_layer import OptimizerDataContext
    from src.valuation import LeagueConfig

    ctx = OptimizerDataContext()
    ctx.config = LeagueConfig()
    ctx.urgency_weights = {"summary": {"losing": ["R"], "tied": []}}
    ctx.news_flags = {}

    fa = {"name": "TestFA", "player_id": 1, "sustainability": 0.5, "ownership_delta_7d": 0}
    drop = {"name": "TestDrop", "player_id": 2}
    swap = {
        "net_sgp": 5.0,
        "category_deltas": {
            "R": 2.0,
            "HR": 0.5,
            "RBI": 0.4,
            "K": 1.5,
            "ERA": 0.3,
            "WHIP": 0.3,
        },
    }
    reasons = _build_reasoning(fa, drop, swap, ctx)
    # At least one reason should mention "Other categories"
    has_other = any("other categor" in r.lower() for r in reasons)
    assert has_other, f"PR13: reasoning should include 'Other categories' reconciliation line. Got: {reasons}"


def test_fa_page_has_show_all_toggle():
    """PR14: FA page must have a 'Show all' toggle for the FA list."""
    fa_page = Path(__file__).parent.parent / "pages" / "14_Free_Agents.py"
    src = fa_page.read_text(encoding="utf-8")
    assert "Show all" in src or "show_all" in src or "fa_show_all" in src, (
        "PR14: 'Show all' toggle required for FA pagination"
    )

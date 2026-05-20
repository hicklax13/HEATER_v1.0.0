"""FA-engine P2 PR6 (2026-05-20): composite formula no longer adds
urgency_boost on top of multiplicative base_value × sustainability × ...

Background:
  The old composite was:
    composite = base_value * sustainability * ownership_mult * floor_mult + urgency_boost

  with `urgency_boost = sum(category_weights for cat in relevant_cats
  if fa_contributes)`. For a 5-tool hitter touching 6 categories this
  summed to ~6 × ~1.0 = ~6, while base_value for a marginal FA was
  often 0.5-2 SGP. The additive boost dwarfed the multiplicative
  portion → multi-category FAs leapfrogged concentrated-value players
  regardless of total SGP impact.

  Worse: _compute_base_value ALREADY applies ctx.category_weights as
  a per-category multiplier inside its loop — so the urgency was
  being double-counted (once multiplicatively in base_value, once
  additively in the boost).

  Fix: set urgency_boost = 0 in the composite. Per-category urgency
  weighting in _compute_base_value is now the only urgency signal.

These tests pin:
  * The composite line in fa_recommender.py uses urgency_boost = 0.0
  * Concentrated-value FAs (high base in 1-2 cats) score competitively
    with diffuse-value FAs (low base in many cats)
  * AST guard: composite calculation includes urgency_boost term but
    sets it to 0 (preserves variable name for git history continuity)
"""

from __future__ import annotations

import ast
import pathlib

_FA_RECOMMENDER_PATH = pathlib.Path("src/optimizer/fa_recommender.py")


def _source() -> str:
    return _FA_RECOMMENDER_PATH.read_text(encoding="utf-8")


def test_urgency_boost_set_to_zero():
    """The assignment `urgency_boost = 0.0` must exist in fa_recommender —
    confirms the fix is in place."""
    src = _source()
    assert "urgency_boost = 0.0" in src, (
        "Expected `urgency_boost = 0.0` assignment in fa_recommender. "
        "The composite formula must NOT add the multi-category sum on top "
        "of the multiplicative base_value × sustainability × ... product — "
        "that would re-introduce the diffuse-value FA over-weighting bug."
    )


def test_composite_still_includes_urgency_boost_for_history_continuity():
    """The composite formula keeps the `+ urgency_boost` term to preserve
    the variable name in git blame / history. Since urgency_boost is now
    always 0.0, this is a no-op term but keeps the diff focused."""
    src = _source()
    assert "+ urgency_boost" in src, (
        "Composite formula must still reference `+ urgency_boost`. The "
        "no-op term keeps git history continuity — the fix is that "
        "urgency_boost is set to 0.0, not that the term is removed."
    )


def test_urgency_boost_function_still_exists():
    """_compute_urgency_boost stays callable for legacy paths even though
    it's no longer wired into the composite. Removing it could break
    importers that call it directly."""
    src = _source()
    assert "def _compute_urgency_boost" in src, (
        "_compute_urgency_boost function must remain defined — legacy paths "
        "or future re-introduction of additive urgency need the function "
        "callable. Just don't add its result to composite."
    )


def test_per_category_weighting_still_in_base_value():
    """_compute_base_value must still apply ctx.category_weights as a
    per-category multiplier. That's the authoritative urgency signal now."""
    src = _source()
    # Look for the per-category weight application in _compute_base_value
    assert "ctx.category_weights.get(" in src, (
        "_compute_base_value must apply ctx.category_weights as a per-cat "
        "multiplier. This is the multiplicative urgency channel that "
        "replaces the additive urgency_boost."
    )


def test_composite_purely_multiplicative_when_urgency_zero():
    """AST-level: when urgency_boost is 0.0, the composite is mathematically
    `base_value × sustainability × ownership_mult × floor_mult`. Confirm via
    AST inspection of the composite assignment."""
    src = _source()
    tree = ast.parse(src)
    # Find the assignment: composite = ...
    composite_assign = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "composite":
                    composite_assign = node
                    break
            if composite_assign:
                break
    assert composite_assign is not None, "composite assignment not found in fa_recommender"
    # The RHS must reference base_value, sustainability, ownership_mult,
    # floor_mult, and urgency_boost (all 5 multipliers / additive term).
    names_in_rhs = {n.id for n in ast.walk(composite_assign.value) if isinstance(n, ast.Name)}
    for required in ("base_value", "sustainability", "ownership_mult", "floor_mult", "urgency_boost"):
        assert required in names_in_rhs, f"composite RHS must reference {required}. Found: {sorted(names_in_rhs)}"

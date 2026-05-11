"""SF-26: No hardcoded category lists in src/. Use LeagueConfig.

Validates that the canonical category lists, the inverse-stats set, and the
rate-stats set in modules listed below are derived from
``src.valuation.LeagueConfig`` rather than hand-rolled literals.

The check is intentionally narrow — only the *named module-level constants*
that this refactor migrated. Function-local hardcoded copies and unrelated
column-name lists (e.g. raw DB column names) are out of scope and are not
flagged.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

# Each entry: (file path, module-level constant name, "kind")
#   kind ∈ {"all_cats", "hit_cats", "pit_cats", "inverse", "rate"}
#
# A constant is considered LeagueConfig-derived when its right-hand side
# *does not* consist solely of string literals — i.e. it references
# ``LeagueConfig`` (directly or via an alias) somehow.
TARGETS: list[tuple[str, str, str]] = [
    ("src/engine/portfolio/valuation.py", "CATEGORIES", "all_cats"),
    ("src/engine/portfolio/valuation.py", "INVERSE_CATEGORIES", "inverse"),
    ("src/engine/portfolio/copula.py", "CATEGORIES", "all_cats"),
    ("src/engine/portfolio/copula.py", "INVERSE_CATEGORIES", "inverse"),
    ("src/engine/portfolio/copula.py", "CAT_ORDER", "all_cats"),
    ("src/engine/game_theory/sensitivity.py", "INVERSE_CATEGORIES", "inverse"),
    ("src/engine/game_theory/opponent_valuation.py", "CATEGORIES", "all_cats"),
    ("src/engine/game_theory/opponent_valuation.py", "INVERSE_CATEGORIES", "inverse"),
    ("src/optimizer/dual_objective.py", "INVERSE_CATS", "inverse"),
    ("src/optimizer/pipeline.py", "INVERSE_CATS", "inverse"),
    ("src/optimizer/scenario_generator.py", "INVERSE_CATS", "inverse"),
    ("src/optimizer/scenario_generator.py", "ALL_CATS", "all_cats"),
    ("src/optimizer/shared_data_layer.py", "_INVERSE_CATS", "inverse"),
    ("src/optimizer/projections.py", "RATE_CATS", "rate"),
    ("src/leaders.py", "INVERSE_CATS", "inverse"),
    ("src/player_databank.py", "HITTING_CATS", "hit_cats"),
    ("src/player_databank.py", "PITCHING_CATS", "pit_cats"),
    ("src/war_room.py", "INVERSE_CATS", "inverse"),
    ("src/war_room.py", "RATE_CATS", "rate"),
]


def _module_assignments(path: Path) -> dict[str, ast.AST]:
    """Return {target_name: rhs_node} for top-level assignments and AnnAssigns."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    out[tgt.id] = node.value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.value is not None:
                out[node.target.id] = node.value
    return out


def _is_pure_string_literal_collection(node: ast.AST) -> bool:
    """True iff node is a List/Tuple/Set/FrozenSet/Call(set) whose elements
    are entirely string Constant nodes (i.e. no LeagueConfig reference)."""
    if isinstance(node, ast.List | ast.Tuple | ast.Set):
        return all(isinstance(elt, ast.Constant) and isinstance(elt.value, str) for elt in node.elts)
    # set([...]) / frozenset([...]) form
    if isinstance(node, ast.Call):
        func_name = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else (node.func.attr if isinstance(node.func, ast.Attribute) else None)
        )
        if func_name in {"set", "frozenset"} and len(node.args) == 1:
            return _is_pure_string_literal_collection(node.args[0])
    return False


def test_named_constants_are_leagueconfig_derived():
    """Each listed module constant must be derived from ``LeagueConfig`` rather
    than a hand-rolled string-literal collection."""
    bad: list[str] = []
    missing: list[str] = []
    for f, name, _kind in TARGETS:
        p = Path(f)
        if not p.exists():
            missing.append(f)
            continue
        decls = _module_assignments(p)
        if name not in decls:
            missing.append(f"{f}::{name}")
            continue
        rhs = decls[name]
        if _is_pure_string_literal_collection(rhs):
            # Capture a short rendering of the literal for the failure message
            try:
                snippet = ast.unparse(rhs)[:80]
            except Exception:
                snippet = "<rhs>"
            bad.append(f"{f}::{name} is hardcoded literal -> {snippet}")
    msg_parts = []
    if missing:
        msg_parts.append("Missing target declarations:\n  " + "\n  ".join(missing))
    if bad:
        msg_parts.append("Hardcoded category literals (must derive from LeagueConfig):\n  " + "\n  ".join(bad))
    assert not (bad or missing), "\n\n".join(msg_parts)


# ── Secondary structural check: no full canonical inverse-stats *set literal*
# anywhere in the listed files (catches stray duplicates added later). ──────


SRC_FILES_FOR_INVERSE_SCAN: list[str] = [t[0] for t in TARGETS]


def test_no_full_inverse_set_literal_outside_leagueconfig():
    """Catches any literal ``{"L", "ERA", "WHIP"}`` set in scope. Comments
    referencing the inverse stats are stripped before matching."""
    pat = re.compile(r"\{\s*['\"][Ll]['\"]\s*,\s*['\"](?:ERA|era)['\"]\s*,\s*['\"](?:WHIP|whip)['\"]\s*\}")
    bad: list[str] = []
    for f in sorted(set(SRC_FILES_FOR_INVERSE_SCAN)):
        p = Path(f)
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        text_no_comments = re.sub(r"#.*$", "", text, flags=re.MULTILINE)
        for m in pat.finditer(text_no_comments):
            # Filter out matches inside docstrings/strings - approximated by
            # checking surrounding chars; cheap heuristic, good enough here.
            bad.append(f"{f} :: {m.group(0)}")
    assert bad == [], "Stray inverse-stats literal found:\n  " + "\n  ".join(bad)

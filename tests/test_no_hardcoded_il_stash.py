"""PR7: structural guard that src/alerts.py does NOT contain a hardcoded
IL_STASH_NAMES literal of {names}. The set must come from a function call
or DB query."""

import ast
from pathlib import Path


def _check_node_value(node_value, target_id):
    """Raise AssertionError if node_value is a hardcoded Set literal."""
    if isinstance(node_value, ast.Set):
        names = [elt.value for elt in node_value.elts if isinstance(elt, ast.Constant)]
        if names:
            raise AssertionError(
                f"src/alerts.py has hardcoded {target_id} = {{{', '.join(names)}}} "
                f"— PR7 requires this to be derived from league_rosters."
            )


def test_alerts_no_hardcoded_il_stash_literal():
    alerts_py = Path(__file__).parent.parent / "src" / "alerts.py"
    tree = ast.parse(alerts_py.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        # Plain assignment: IL_STASH_NAMES = {...}
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "IL_STASH_NAMES":
                    # Allow function call (e.g. = get_il_stash_names()) or attribute
                    # but NOT a Set literal with hardcoded strings
                    _check_node_value(node.value, target.id)
        # Annotated assignment: IL_STASH_NAMES: set[str] = {...}
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "IL_STASH_NAMES":
                if node.value is not None:
                    _check_node_value(node.value, node.target.id)

"""PR18: Structural guard ensuring pages/14_Free_Agents.py passes the
USER's roster slice to build_optimizer_context, not the full league
rosters. Regression of this caused all bad FA recs on 2026-05-20."""

import ast
from pathlib import Path


def test_fa_page_passes_user_roster_to_build_optimizer_context():
    """The FA page must call build_optimizer_context with roster=user_roster
    (or similar single-team slice), NOT roster=rosters (multi-team)."""
    fa_page = Path(__file__).parent.parent / "pages" / "14_Free_Agents.py"
    tree = ast.parse(fa_page.read_text(encoding="utf-8"))

    found_build_call = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = ""
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            if func_name == "build_optimizer_context":
                found_build_call = True
                roster_arg = None
                for kw in node.keywords:
                    if kw.arg == "roster":
                        roster_arg = kw.value
                assert roster_arg is not None, "build_optimizer_context call missing roster= kwarg"
                # Forbid `roster=rosters` (multi-team source).
                if isinstance(roster_arg, ast.Name):
                    assert roster_arg.id != "rosters", (
                        "pages/14_Free_Agents.py passes roster=rosters (full league). "
                        "Must pass user's slice (e.g., roster=user_roster) — root cause of "
                        "ctx.user_roster_ids bug from 2026-05-20."
                    )

    assert found_build_call, "FA page no longer calls build_optimizer_context — refactor needed"

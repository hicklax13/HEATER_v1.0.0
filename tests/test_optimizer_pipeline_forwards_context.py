"""BUG-011 fix: LineupOptimizerPipeline forwards confirmed_lineups,
recent_form, and team_strength to build_daily_dcv_table."""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE = REPO_ROOT / "src" / "optimizer" / "pipeline.py"


def test_pipeline_forwards_context_to_build_daily_dcv_table():
    """Parse src/optimizer/pipeline.py's AST and find every Call to
    build_daily_dcv_table. Each call must pass confirmed_lineups,
    recent_form, and team_strength as kwargs."""
    assert PIPELINE.exists()
    tree = ast.parse(PIPELINE.read_text(encoding="utf-8"))
    required_kwargs = {"confirmed_lineups", "recent_form", "team_strength"}

    found_calls = 0
    missing_per_call: list[tuple[int, set[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            # Match `build_daily_dcv_table(...)` direct call
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name != "build_daily_dcv_table":
                continue
            found_calls += 1
            kwarg_names = {kw.arg for kw in node.keywords if kw.arg is not None}
            missing = required_kwargs - kwarg_names
            if missing:
                missing_per_call.append((node.lineno, missing))

    assert found_calls > 0, (
        "Expected at least one build_daily_dcv_table call in src/optimizer/pipeline.py; did the file structure change?"
    )
    assert not missing_per_call, (
        f"BUG-011 regression: build_daily_dcv_table call(s) in "
        f"src/optimizer/pipeline.py do not forward all required kwargs. "
        f"Missing per call: {missing_per_call}. "
        f"Required: {required_kwargs}. Forward from kwargs.get(...) or ctx attr."
    )

"""Structural test: pool loading on 4 pages must go through @st.cache_data-decorated helpers.

Task 2.2 — Cache the pool load (BLOCKER/HIGH).

Each of the four pages must NOT call load_player_pool() (or load_databank(), which
internally calls load_player_pool()) as a bare module-level expression outside a cached
helper.  Instead each page must define a module-level @st.cache_data-decorated function
that wraps the expensive call.

The test is AST-based so it catches both patterns:
  1. `pool = load_player_pool()`  ← bare top-level call (BAD)
  2. `@st.cache_data(...)\ndef _get_pool(): ... load_player_pool() ...` ← correct
"""

import ast
import pathlib

PAGES_DIR = pathlib.Path(__file__).parent.parent / "pages"

# Map page stem → the expensive call(s) that must be wrapped
_PAGE_EXPENSIVE_CALLS: dict[str, set[str]] = {
    "14_Free_Agents": {"load_player_pool"},
    "17_Leaders": {"load_player_pool"},
    "19_Player_Databank": {"load_databank"},
    "5_Matchup_Planner": {"load_player_pool"},
}

# Helper name prefix each page must use (any _get_pool / _load_pool / _cached_pool / etc.)
# The test just verifies the cached-wrapper pattern — it doesn't enforce the exact name.


def _parse_page(stem: str) -> ast.Module:
    path = PAGES_DIR / f"{stem}.py"
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _get_cache_data_decorated_funcs(tree: ast.Module) -> set[str]:
    """Return names of module-level functions decorated with @st.cache_data."""
    names = set()
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for deco in node.decorator_list:
            # Matches both @st.cache_data and @st.cache_data(...)
            if isinstance(deco, ast.Attribute) and deco.attr == "cache_data":
                names.add(node.name)
                break
            if isinstance(deco, ast.Call) and isinstance(deco.func, ast.Attribute) and deco.func.attr == "cache_data":
                names.add(node.name)
                break
    return names


def _calls_in_func(func_node: ast.FunctionDef) -> set[str]:
    """Return the set of bare function-call names made anywhere inside a function body."""
    names = set()
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return names


def _bare_module_level_calls(tree: ast.Module, expensive: set[str]) -> set[str]:
    """
    Return any expensive call names that appear as bare module-level expressions
    (i.e. in an Assign/Expr at the top level, not inside any function/class).
    """
    found = set()
    for node in ast.iter_child_nodes(tree):
        # Skip function and class definitions — those are safe
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        # Check all Call nodes inside the module-level statement
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                name = None
                if isinstance(child.func, ast.Name):
                    name = child.func.id
                elif isinstance(child.func, ast.Attribute):
                    name = child.func.attr
                if name in expensive:
                    found.add(name)
    return found


def _cached_funcs_wrapping(tree: ast.Module, expensive: set[str]) -> set[str]:
    """
    Return the subset of expensive call names that are called inside at least one
    @st.cache_data-decorated module-level function.
    """
    cached_funcs = _get_cache_data_decorated_funcs(tree)
    wrapped = set()
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in cached_funcs:
            continue
        calls = _calls_in_func(node)
        for exp in expensive:
            if exp in calls:
                wrapped.add(exp)
    return wrapped


# ── Per-page tests ────────────────────────────────────────────────────────────


def test_free_agents_pool_is_cached():
    """pages/14_Free_Agents.py must obtain its pool through an @st.cache_data helper."""
    tree = _parse_page("14_Free_Agents")
    expensive = _PAGE_EXPENSIVE_CALLS["14_Free_Agents"]

    wrapped = _cached_funcs_wrapping(tree, expensive)
    missing = expensive - wrapped
    assert not missing, (
        f"14_Free_Agents.py: load_player_pool() must be called inside a "
        f"@st.cache_data-decorated helper, but these are not wrapped: {missing}"
    )


def test_leaders_pool_is_cached():
    """pages/17_Leaders.py must obtain its pool through an @st.cache_data helper."""
    tree = _parse_page("17_Leaders")
    expensive = _PAGE_EXPENSIVE_CALLS["17_Leaders"]

    wrapped = _cached_funcs_wrapping(tree, expensive)
    missing = expensive - wrapped
    assert not missing, (
        f"17_Leaders.py: load_player_pool() must be called inside a "
        f"@st.cache_data-decorated helper, but these are not wrapped: {missing}"
    )


def test_player_databank_pool_is_cached():
    """pages/19_Player_Databank.py must load data through an @st.cache_data helper."""
    tree = _parse_page("19_Player_Databank")
    expensive = _PAGE_EXPENSIVE_CALLS["19_Player_Databank"]

    wrapped = _cached_funcs_wrapping(tree, expensive)
    missing = expensive - wrapped
    assert not missing, (
        f"19_Player_Databank.py: load_databank() must be called inside a "
        f"@st.cache_data-decorated helper, but these are not wrapped: {missing}"
    )


def test_matchup_planner_pool_is_cached():
    """pages/5_Matchup_Planner.py must obtain its pool through an @st.cache_data helper."""
    tree = _parse_page("5_Matchup_Planner")
    expensive = _PAGE_EXPENSIVE_CALLS["5_Matchup_Planner"]

    wrapped = _cached_funcs_wrapping(tree, expensive)
    missing = expensive - wrapped
    assert not missing, (
        f"5_Matchup_Planner.py: load_player_pool() must be called inside a "
        f"@st.cache_data-decorated helper, but these are not wrapped: {missing}"
    )

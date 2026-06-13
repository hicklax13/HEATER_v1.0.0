"""Structural-invariant guards for Trade Analyzer performance fixes.

Tasks covered:
  2.1 — MC opt-in: default path uses enable_mc=False; opt-in checkbox gates MC.
  2.2-TA — Cached pool: pool loaded via @st.cache_data helper, not bare call.
  2.5-TA — Progress feedback: evaluate path wrapped in st.spinner.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

PAGE_PATH = Path(__file__).resolve().parent.parent / "pages" / "11_Trade_Analyzer.py"


@pytest.fixture(scope="module")
def source() -> str:
    return PAGE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tree(source: str) -> ast.Module:
    return ast.parse(source)


# ── Task 2.1: MC opt-in ──────────────────────────────────────────────────────


def test_default_evaluate_path_uses_enable_mc_false(source: str) -> None:
    """The evaluate_trade() call must pass enable_mc=False by default.

    MC runs ~44.8 s synchronously — it must be opt-in, not the default path.
    The value must be driven by a variable (the checkbox value), NOT the
    literal True.
    """
    # The literal `enable_mc=True` must NOT appear in the evaluate_trade call.
    # We allow enable_mc=True in comments and docstrings (not code).
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = ""
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            if func_name == "evaluate_trade":
                for kw in node.keywords:
                    if kw.arg == "enable_mc":
                        # The value must NOT be the literal True
                        assert not (isinstance(kw.value, ast.Constant) and kw.value.value is True), (
                            "evaluate_trade(enable_mc=True) is hardcoded. "
                            "MC must be opt-in — pass the checkbox bool variable instead."
                        )


def test_mc_optin_checkbox_present(source: str) -> None:
    """An opt-in MC checkbox must exist in the page source.

    Acceptable patterns: st.checkbox("Run deep risk analysis..." or similar).
    """
    # Must have a checkbox that controls MC
    assert re.search(r"st\.checkbox\(", source), "No st.checkbox() found. Task 2.1 requires an opt-in checkbox for MC."
    # The checkbox label (possibly on the next line after the call) must mention
    # the time cost or 'deep risk' / 'risk analysis' / 'Monte Carlo'.
    # Use DOTALL so multi-line label strings are matched.
    assert re.search(
        r"st\.checkbox\([^)]*?(deep risk|risk analysis|~45|45s|Monte Carlo)",
        source,
        re.IGNORECASE | re.DOTALL,
    ), (
        "The MC opt-in checkbox label must mention the time cost or "
        "'deep risk analysis' so users understand what they're opting into."
    )


def test_mc_checkbox_value_wired_to_evaluate_trade(source: str) -> None:
    """The checkbox return value must be passed as enable_mc= to evaluate_trade.

    A checkbox that isn't wired does nothing — we verify the enable_mc= kwarg
    is not the literal False either (it must be a variable).
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = ""
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            if func_name == "evaluate_trade":
                for kw in node.keywords:
                    if kw.arg == "enable_mc":
                        # Value must be a Name (variable), not a literal bool
                        assert isinstance(kw.value, ast.Name), (
                            f"enable_mc= must be a variable (checkbox value), not a literal. Got: {ast.dump(kw.value)}"
                        )
                        return  # found and validated
    pytest.fail("evaluate_trade() call with enable_mc= kwarg not found in page.")


def test_no_stale_200_simulations_tooltip(source: str) -> None:
    """The stale '200 simulations' tooltip/caption must be removed.

    The legacy progress text said '200 iterations' which was wrong.
    Any simulation-count mention in the evaluate path must reflect the
    actual sim count, not the old stale literal.
    """
    # The specific stale string that appeared in the legacy fallback path
    assert "200 iterations" not in source, (
        "Stale '200 iterations' tooltip/caption still present. Remove or correct it to the actual sim count."
    )
    # The old wrong hint: "Monte Carlo simulation (200 iterations)"
    assert re.search(r"200\s+simulations", source) is None or "200 simulations" not in source.split("enable_mc")[0], (
        "Stale '200 simulations' text must not appear in the primary evaluate path."
    )


# ── Task 2.2-TA: Cached pool ─────────────────────────────────────────────────


def test_pool_loaded_via_cache_data_helper(source: str, tree: ast.Module) -> None:
    """load_player_pool() must be wrapped in a @st.cache_data decorated helper.

    A bare `load_player_pool()` call in the page body re-runs on every Streamlit
    rerun (~4.3 s). The pool must be obtained through a @st.cache_data function.
    """
    # Find all function definitions decorated with st.cache_data
    cached_fns: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                dec_str = ast.dump(dec)
                if "cache_data" in dec_str:
                    cached_fns.add(node.name)

    assert cached_fns, (
        "No @st.cache_data-decorated helper found. Task 2.2-TA requires wrapping the pool load in a cached helper."
    )

    # Verify that load_player_pool is called INSIDE one of those cached fns
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in cached_fns:
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    fn = child.func
                    call_name = ""
                    if isinstance(fn, ast.Name):
                        call_name = fn.id
                    elif isinstance(fn, ast.Attribute):
                        call_name = fn.attr
                    if call_name == "load_player_pool":
                        return  # Found inside a cached function — pass

    pytest.fail(
        "load_player_pool() is not called inside any @st.cache_data function. "
        "Wrap it in a cached helper so it doesn't re-run on every rerun."
    )


def test_no_bare_load_player_pool_in_page_body(source: str, tree: ast.Module) -> None:
    """load_player_pool() must not be called bare at module level or in the
    page body outside a @st.cache_data helper.

    We allow it ONLY inside cache_data-decorated function definitions.
    """
    # Collect all cached function node ranges
    cached_fn_nodes: list[ast.FunctionDef] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                if "cache_data" in ast.dump(dec):
                    cached_fn_nodes.append(node)

    # Collect line ranges of cached functions
    cached_ranges: list[tuple[int, int]] = []
    for fn in cached_fn_nodes:
        start = fn.lineno
        end = fn.end_lineno or fn.lineno
        cached_ranges.append((start, end))

    def _in_cached(line: int) -> bool:
        return any(s <= line <= e for s, e in cached_ranges)

    # Find all calls to load_player_pool
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            call_name = ""
            if isinstance(fn, ast.Name):
                call_name = fn.id
            elif isinstance(fn, ast.Attribute):
                call_name = fn.attr
            if call_name == "load_player_pool":
                assert _in_cached(node.lineno), (
                    f"load_player_pool() at line {node.lineno} is called outside "
                    f"a @st.cache_data function. This re-runs every rerun (~4.3 s). "
                    f"Move it inside the cached helper."
                )


def test_health_adjusted_pool_also_cached(source: str, tree: ast.Module) -> None:
    """get_health_adjusted_pool() must also be inside or called by the cached helper.

    The health-adjusted pool adds ~0.5 s. Cache it alongside the base pool.
    Acceptable: it's called inside the same cached fn, or the cached fn returns
    the already-adjusted pool.
    """
    # Collect cached function nodes
    cached_fn_nodes: list[ast.FunctionDef] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                if "cache_data" in ast.dump(dec):
                    cached_fn_nodes.append(node)

    if not cached_fn_nodes:
        pytest.skip("No cached fns found — caught by test_pool_loaded_via_cache_data_helper")

    # Check if get_health_adjusted_pool is called inside any cached fn
    for fn in cached_fn_nodes:
        for child in ast.walk(fn):
            if isinstance(child, ast.Call):
                cn = child.func
                call_name = ""
                if isinstance(cn, ast.Name):
                    call_name = cn.id
                elif isinstance(cn, ast.Attribute):
                    call_name = cn.attr
                if call_name == "get_health_adjusted_pool":
                    return  # Found inside cached fn — pass

    # It's also acceptable if TRADE_INTEL_AVAILABLE is checked AND it's inside the cached fn
    # OR if the page no longer calls it at all outside
    # Check page body (outside cached fns)
    cached_ranges: list[tuple[int, int]] = []
    for fn in cached_fn_nodes:
        cached_ranges.append((fn.lineno, fn.end_lineno or fn.lineno))

    def _in_cached(line: int) -> bool:
        return any(s <= line <= e for s, e in cached_ranges)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn_attr = node.func
            cn = ""
            if isinstance(fn_attr, ast.Name):
                cn = fn_attr.id
            elif isinstance(fn_attr, ast.Attribute):
                cn = fn_attr.attr
            if cn == "get_health_adjusted_pool" and not _in_cached(node.lineno):
                pytest.fail(
                    f"get_health_adjusted_pool() at line {node.lineno} is called "
                    f"outside a @st.cache_data function. Move it inside the cached "
                    f"pool helper to avoid ~0.5 s per-rerun cost."
                )


# ── Task 2.5-TA: Progress feedback ───────────────────────────────────────────


def test_evaluate_path_wrapped_in_spinner(source: str) -> None:
    """The evaluate_trade() call must be wrapped in st.spinner.

    Without a spinner, the page appears frozen during the (potentially
    multi-second) deterministic phase.
    """
    assert "st.spinner" in source, (
        "No st.spinner() found. Task 2.5-TA requires wrapping the evaluate_trade() call in st.spinner(...)."
    )


def test_spinner_label_mentions_evaluating(source: str) -> None:
    """The spinner label (or a nearby variable) must be meaningful.

    Acceptable patterns:
      - st.spinner("Evaluating trade…")  — literal label
      - A variable assigned to "Evaluating trade…" used in st.spinner(var)
    """
    # Check for either a literal string spinner or a variable that contains
    # "Evaluat" being passed/assigned near st.spinner
    has_literal = bool(re.search(r'st\.spinner\(["\'].*?[Ee]valuat', source))
    has_variable_label = bool(
        re.search(r'"[^"]*[Ee]valuat[^"]*"', source)  # string somewhere referencing Evaluat
        and re.search(r"st\.spinner\(", source)
    )
    assert has_literal or has_variable_label, (
        "The page must have a spinner label mentioning 'Evaluating trade' or "
        "similar (either as a literal or via a variable) so the user knows "
        "what the app is computing."
    )


def test_mc_spinner_when_mc_enabled(source: str) -> None:
    """The page must include spinner/label text covering the MC time cost.

    Acceptable: a literal spinner label, or a string variable assigned to
    a value mentioning ~45s/risk analysis/Monte Carlo that is used in
    st.spinner(). Also acceptable: an st.info/st.caption near the checkbox
    warning about the time cost.
    """
    # Search for any string literal in the file that mentions MC cost
    mc_cost_mention = bool(
        re.search(r"45s|~45|risk analysis|Monte Carlo", source, re.IGNORECASE) and re.search(r"st\.spinner\(", source)
    )
    assert mc_cost_mention, (
        "No mention of MC time cost (~45s / 'risk analysis' / 'Monte Carlo') "
        "found alongside st.spinner(). "
        "Add a spinner or visible label for the opt-in MC path so users know why it's slow."
    )


# ── Syntax sanity ─────────────────────────────────────────────────────────────


def test_page_syntax_still_valid(source: str) -> None:
    """Page must remain valid Python after the edits."""
    ast.parse(source)

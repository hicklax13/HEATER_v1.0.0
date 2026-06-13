"""TDD tests for Pitcher Streaming page trust/comprehension fixes.

Tasks:
  3.1 — render_data_freshness_chip called near the header.
  3.3 — jargon_help / render_glossary_expander used for Stream Score, Net SGP,
        Opp wRC+, Conf columns.
  3.4+3.7 — "Top pick today" callout (naming #1 pitcher + score) appears above
            the board; Score moved to or near the first data column via a Rank
            column; a short scale note anchors the 0-100 Score.

AST-safety contract: no new BinOp arithmetic on *score*-named variables
(that remains guarded by test_stream_page_no_inline_scoring.py). The Rank
column must be derived from the engine-emitted Score column only (no new
arithmetic), and the callout must read the engine's top Score row directly.
"""

from __future__ import annotations

import ast
from pathlib import Path

_PAGE = Path(__file__).resolve().parents[1] / "pages" / "4_Pitcher_Streaming.py"


def _src() -> str:
    return _PAGE.read_text(encoding="utf-8")


# ── Task 3.1: freshness chip ─────────────────────────────────────────────────


def test_stream_freshness_chip_imported():
    """render_data_freshness_chip must be imported from src.ui_shared."""
    src = _src()
    assert "render_data_freshness_chip" in src, (
        "pages/4_Pitcher_Streaming.py must import render_data_freshness_chip from src.ui_shared"
    )


def test_stream_freshness_chip_called():
    """render_data_freshness_chip must be called (with any source) in the page."""
    src = _src()
    assert "render_data_freshness_chip(" in src, "pages/4_Pitcher_Streaming.py must call render_data_freshness_chip()"


# ── Task 3.3: jargon tooltips + glossary ─────────────────────────────────────


def test_stream_jargon_help_imported():
    """jargon_help must be imported from src.ui_shared."""
    src = _src()
    assert "jargon_help" in src, "pages/4_Pitcher_Streaming.py must import jargon_help from src.ui_shared"


def test_stream_jargon_help_stream_score():
    """jargon_help('Stream Score') must appear for the Score column tooltip."""
    src = _src()
    assert 'jargon_help("Stream Score")' in src or "jargon_help('Stream Score')" in src, (
        'pages/4_Pitcher_Streaming.py must reference jargon_help("Stream Score")'
    )


def test_stream_jargon_help_net_sgp():
    """jargon_help('Net SGP') must appear for the Net SGP column tooltip."""
    src = _src()
    assert 'jargon_help("Net SGP")' in src or "jargon_help('Net SGP')" in src, (
        'pages/4_Pitcher_Streaming.py must reference jargon_help("Net SGP")'
    )


def test_stream_glossary_expander_called_once():
    """render_glossary_expander must be called exactly once."""
    src = _src()
    count = src.count("render_glossary_expander(")
    assert count == 1, f"pages/4_Pitcher_Streaming.py must call render_glossary_expander exactly once, found {count}"


def test_stream_glossary_includes_key_terms():
    """render_glossary_expander must include Stream Score, Net SGP, Opp wRC+."""
    src = _src()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "render_glossary_expander"
        ):
            if node.args:
                first_arg = node.args[0]
                if isinstance(first_arg, ast.List):
                    terms = [
                        elt.value
                        for elt in first_arg.elts
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                    ]
                    assert "Stream Score" in terms, f"render_glossary_expander must include 'Stream Score', got {terms}"
                    assert "Net SGP" in terms, f"render_glossary_expander must include 'Net SGP', got {terms}"
                    # wRC+ is listed under 'wRC+' in JARGON
                    assert any("wRC" in t for t in terms), (
                        f"render_glossary_expander must include a wRC+ term, got {terms}"
                    )
                    return
    # If no list arg, just verify the call exists
    assert "render_glossary_expander(" in src


# ── Task 3.4+3.7: Top pick callout + Rank column + Scale note ────────────────


def test_stream_top_pick_callout_present():
    """A 'top pick' callout must appear above the board in the Stream Finder tab."""
    src = _src()
    # Accept 'top pick', 'Top pick', or 'Top stream' (case insensitive)
    lower = src.lower()
    has_top_pick = (
        "top pick" in lower or "top stream" in lower or "best pick" in lower or "#1" in lower or "no. 1" in lower
    )
    assert has_top_pick, "pages/4_Pitcher_Streaming.py must show a 'Top pick today' callout naming the #1 pick"


def test_stream_rank_column_present():
    """A 'Rank' column (or 'Rank' key) must be added to the stream board display."""
    src = _src()
    assert '"Rank"' in src or "'Rank'" in src, (
        "pages/4_Pitcher_Streaming.py must add a 'Rank' column to the stream board display"
    )


def test_stream_score_scale_note_present():
    """A short note anchoring the Score's 0-100 scale must appear in the page."""
    src = _src()
    lower = src.lower()
    has_scale = (
        "0-100" in src
        or "0 to 100" in lower
        or ("score" in lower and ("higher" in lower or "scale" in lower or "100" in src))
    )
    assert has_scale, "pages/4_Pitcher_Streaming.py must include a short note anchoring the 0-100 Score scale"


def test_stream_rank_not_arithmetic():
    """Rank column must be derived from position/row-number, not new score arithmetic.

    Specifically: no BinOp assignments to a 'rank'-named variable that contain
    'score' arithmetic — Rank can only be an index (range, enumerate, or .rank()).
    """
    tree = ast.parse(_src())
    forbidden: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            continue
        # Get target names
        targets: list[str] = []
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    targets.append(tgt.id)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            if isinstance(node.target, ast.Name):
                targets.append(node.target.id)

        for tname in targets:
            if "rank" not in tname.lower():
                continue
            val = getattr(node, "value", None)
            if val is None:
                continue
            # Check if RHS contains BinOp with a score-related Name load
            has_score_binop = False
            for n in ast.walk(val):
                if isinstance(n, ast.BinOp):
                    for child in ast.walk(n):
                        if isinstance(child, ast.Name) and "score" in child.id.lower():
                            has_score_binop = True
            if has_score_binop:
                forbidden.append(f"{tname} (line {node.lineno})")

    assert not forbidden, (
        f"Rank column must not be computed via score arithmetic: {forbidden}. "
        "Use range(1, len+1) or enumerate — Rank is a row position, not a derived score."
    )

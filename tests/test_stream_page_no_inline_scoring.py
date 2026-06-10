"""Structural guard: stream scoring lives in the engine, never in pages.

The Lineup Optimizer's Streaming tab grew a page-local ad-hoc scoring formula
(the `score = K*1.5 + W*0.3 - ...` block) outside CONSTANTS_REGISTRY — the
same duplication class the SGP consolidation eliminated. The Pitcher Streaming
page must never repeat it: every score/rank it renders arrives from
``src.optimizer.stream_analyzer`` / ``src.optimizer.fa_recommender``.

AST rule: no assignment to a ``*score*``-named variable whose right-hand side
is arithmetic (BinOp). Copying engine outputs into display columns is fine;
computing a score in the page is not.
"""

from __future__ import annotations

import ast
from pathlib import Path

_PAGE = Path(__file__).resolve().parents[1] / "pages" / "4_Pitcher_Streaming.py"


def _assign_targets(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Assign):
        names = []
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                names.append(tgt.id)
            elif isinstance(tgt, (ast.Tuple, ast.List)):
                names.extend(el.id for el in tgt.elts if isinstance(el, ast.Name))
        return names
    if isinstance(node, (ast.AnnAssign, ast.AugAssign)) and isinstance(node.target, ast.Name):
        return [node.target.id]
    return []


def test_page_exists():
    assert _PAGE.exists(), "pages/4_Pitcher_Streaming.py must exist (Phase 2)"


def test_page_imports_engine():
    src = _PAGE.read_text(encoding="utf-8")
    assert "build_stream_board" in src, "page must render scores from build_stream_board"
    assert "recommend_streaming_moves" in src, (
        "today's swap recommendations must come from the canonical fa_recommender.recommend_streaming_moves engine"
    )


def test_no_inline_score_arithmetic():
    tree = ast.parse(_PAGE.read_text(encoding="utf-8"))
    offenders: list[str] = []
    for node in ast.walk(tree):
        for name in _assign_targets(node):
            if "score" not in name.lower():
                continue
            value = getattr(node, "value", None)
            if value is not None and any(isinstance(n, ast.BinOp) for n in ast.walk(value)):
                offenders.append(f"{name} (line {node.lineno})")
    assert not offenders, (
        f"page computes a score with inline arithmetic — move it into src/optimizer/stream_analyzer.py: {offenders}"
    )

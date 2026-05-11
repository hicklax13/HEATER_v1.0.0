"""PuLP availability flag must have a single canonical definition.

Audit C5: Previously both ``src/lineup_optimizer.py`` and
``src/optimizer/advanced_lp.py`` independently performed
``try: import pulp ... PULP_AVAILABLE = True`` blocks. If the import
environment ever diverged (one module loaded before pulp was installed,
patched test mocks, etc.) the two flags would silently disagree.

The canonical source is ``src/lineup_optimizer.py``. All other modules
that need ``PULP_AVAILABLE`` should import it from there.
"""

from __future__ import annotations

import re
from pathlib import Path

CANONICAL_DEFINER = "src/lineup_optimizer.py"


def _normalize(path: Path) -> str:
    return str(path).replace("\\", "/")


def _strip_inline_comments(line: str) -> str:
    # Strip ``# ...`` while leaving ``"#"`` strings alone (heuristic; good
    # enough for our scan).
    if "#" in line and not line.lstrip().startswith("#"):
        in_str = False
        quote = ""
        for i, ch in enumerate(line):
            if in_str:
                if ch == quote and (i == 0 or line[i - 1] != "\\"):
                    in_str = False
            else:
                if ch in ('"', "'"):
                    in_str = True
                    quote = ch
                elif ch == "#":
                    return line[:i]
    if line.lstrip().startswith("#"):
        return ""
    return line


def _has_pulp_independence_block(text: str) -> bool:
    """Detects a 'try: import pulp ... PULP_AVAILABLE = True' block.

    This is the duplicate-determination pattern we want to forbid: a
    module that establishes PuLP availability INDEPENDENTLY by trying
    to import pulp itself and assigning the flag based on the result.

    Implementation: scans line-by-line for any ``try:`` opening, then
    looks ahead within the same indented suite for an ``import pulp``
    or ``from pulp import ...`` AND a ``PULP_AVAILABLE = True``.
    """
    lines = [_strip_inline_comments(ln) for ln in text.splitlines()]
    n = len(lines)
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "try:" or stripped.startswith("try :"):
            # Determine the body's indentation by finding the first non-blank
            # line after ``try:``.
            body_indent = None
            for j in range(idx + 1, n):
                if not lines[j].strip():
                    continue
                body_indent = len(lines[j]) - len(lines[j].lstrip())
                break
            if body_indent is None:
                continue
            # Walk subsequent lines until we leave the try-block (indent drops
            # below body_indent on a non-blank line) or hit the matching
            # except/finally at the original indentation.
            block_lines: list[str] = []
            try_indent = len(line) - len(line.lstrip())
            j = idx + 1
            while j < n:
                ln = lines[j]
                if not ln.strip():
                    j += 1
                    continue
                cur_indent = len(ln) - len(ln.lstrip())
                if cur_indent <= try_indent:
                    break  # left the try suite or hit except/finally
                block_lines.append(ln.strip())
                j += 1
            block_text = "\n".join(block_lines)
            # Need both: pulp imported AND PULP_AVAILABLE=True assigned.
            imports_pulp = bool(re.search(r"^(?:import\s+pulp|from\s+pulp)\b", block_text, re.MULTILINE))
            sets_true = bool(re.search(r"^PULP_AVAILABLE\s*=\s*True\b", block_text, re.MULTILINE))
            if imports_pulp and sets_true:
                return True
    return False


def test_pulp_independence_block_only_in_canonical_module():
    """No module other than src/lineup_optimizer.py should determine
    PuLP availability via its own ``try: import pulp`` block."""
    sources = list(Path("src").rglob("*.py"))
    offenders: list[str] = []
    for f in sources:
        text = f.read_text(encoding="utf-8")
        if _has_pulp_independence_block(text):
            offenders.append(_normalize(f))

    assert offenders == [CANONICAL_DEFINER], (
        f"PULP_AVAILABLE should only be INDEPENDENTLY determined in "
        f"{CANONICAL_DEFINER}; other modules should import it.\n"
        f"Offenders found: {offenders}"
    )


def test_advanced_lp_imports_canonical_flag():
    """src/optimizer/advanced_lp.py must import PULP_AVAILABLE from
    the canonical source rather than redefining it."""
    text = Path("src/optimizer/advanced_lp.py").read_text(encoding="utf-8")

    # Must import PULP_AVAILABLE from src.lineup_optimizer
    assert (
        re.search(
            r"from\s+src\.lineup_optimizer\s+import\s+[^\n]*\bPULP_AVAILABLE\b",
            text,
        )
        is not None
    ), "advanced_lp.py must import PULP_AVAILABLE from src.lineup_optimizer"

    # Must NOT have its own try-block that imports pulp AND sets
    # PULP_AVAILABLE = True (the duplicate pattern we removed).
    assert not _has_pulp_independence_block(text), (
        "advanced_lp.py still has a 'try: import pulp / PULP_AVAILABLE = True' "
        "block — should rely on the canonical flag from src.lineup_optimizer."
    )


def test_canonical_module_still_defines_flag():
    """Sanity: src/lineup_optimizer.py must still actually define
    PULP_AVAILABLE so dependent modules can import it."""
    text = Path("src/lineup_optimizer.py").read_text(encoding="utf-8")
    assert re.search(r"^[ \t]*PULP_AVAILABLE\s*=\s*True\b", text, re.MULTILINE), (
        "Canonical PULP_AVAILABLE = True assignment missing from src/lineup_optimizer.py"
    )
    assert re.search(r"^[ \t]*PULP_AVAILABLE\s*=\s*False\b", text, re.MULTILINE), (
        "Canonical PULP_AVAILABLE = False fallback missing from src/lineup_optimizer.py"
    )


def test_flag_value_is_consistent_across_modules():
    """At runtime, PULP_AVAILABLE imported from the two modules must
    refer to the same value (proof that consolidation works in practice)."""
    from src.lineup_optimizer import PULP_AVAILABLE as flag_canonical
    from src.optimizer.advanced_lp import PULP_AVAILABLE as flag_advanced

    assert flag_canonical == flag_advanced, (
        f"PULP_AVAILABLE diverged between modules: lineup_optimizer={flag_canonical!r} vs advanced_lp={flag_advanced!r}"
    )

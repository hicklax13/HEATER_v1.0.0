"""Permanent guard against BUG-014 regression at the caller site.

refresh_ecr_consensus must strip the '_rank' suffix from column names before
passing the sources dict to _compute_player_consensus — otherwise the
in-season source weights silently default to 1.0.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_ecr_strips_rank_suffix_before_consensus():
    """The call path that builds `sources` from `rank_cols` must strip the
    `_rank` suffix on the keys before invoking _compute_player_consensus.

    Heuristic check: in src/ecr.py, any dict-comprehension that consumes
    a `*_rank` column and feeds into _compute_player_consensus must call
    `.removesuffix("_rank")` (or equivalent) on the key.
    """
    ecr_path = REPO_ROOT / "src" / "ecr.py"
    text = ecr_path.read_text(encoding="utf-8")

    # Find every `_compute_player_consensus(` call and look at the few lines
    # immediately above it to find the dict-comprehension that builds `sources`.
    pattern = re.compile(r"sources\s*=\s*\{[^}]*\}", re.DOTALL)
    matches = list(pattern.finditer(text))
    assert matches, "Expected at least one `sources = {...}` construction in src/ecr.py; did the file structure change?"

    offending: list[str] = []
    for m in matches:
        snippet = m.group(0)
        # If this dict consumes a `*_rank` column...
        if "_rank" in snippet:
            # ...it MUST also call .removesuffix("_rank") on the key
            if "removesuffix" not in snippet and 'rstrip("_rank")' not in snippet:
                # Last resort: maybe a different transformation. But raw `c:` keys are the bug.
                # Look for the simplest bug pattern: `{c: row[c] for c in rank_cols`
                if re.search(r"\{\s*c\s*:\s*row\[c\]\s+for\s+c\s+in\s+rank_cols", snippet):
                    offending.append(snippet[:200])

    assert not offending, (
        "BUG-014 regression: dict comprehension consuming `*_rank` column names "
        "feeds into _compute_player_consensus without stripping the suffix. "
        'Weights silently default to 1.0. Add `.removesuffix("_rank")` to the key. '
        f"Offenders:\n{chr(10).join(offending)}"
    )

"""Deep per-team assertions for Free Agents (pages/14_Free_Agents.py).

Renders ONCE per team (module-scoped `results` fixture) and checks the FA board
is populated and plausible (ownership % in [0,100], ECR a positive integer, rate
stats in range, no NaN). Defensive style: checks fire only when data is present.

Run SERIALLY: .venv\\Scripts\\python.exe -m pytest tests/qa/test_page_14_free_agents.py -q
"""

from __future__ import annotations

import re

import pytest

PAGE = "pages/14_Free_Agents.py"
_BAD = ("nan", "none", "inf", "-inf", "")

# Minimum markdown length that indicates a page actually rendered content
# (section headers + context panel + at least a short status message).
_MIN_MARKDOWN_LEN = 200


def _is_bad_number(s) -> bool:
    return str(s).strip().lower() in _BAD


def _strip_html(text: str) -> str:
    """Remove HTML tags from a string so regex can match plain values."""
    return re.sub(r"<[^>]+>", " ", text)


@pytest.fixture(scope="module")
def results(run_page_as_team, team_names):
    return {t: run_page_as_team(PAGE, t, is_admin=False) for t in team_names}


# ── Core smoke ────────────────────────────────────────────────────────────────


def test_free_agents_renders_for_all_teams(results):
    """Every team renders without an unhandled exception or st.error call.

    A blank page (no dataframes, no metrics, very little markdown) is also
    treated as a failure — it means the page short-circuited before reaching
    any FA output.
    """
    problems = []
    for team, r in results.items():
        if not r.ran:
            problems.append((team, "did-not-run", r.exception))
            continue
        if r.exception:
            problems.append((team, "exception", r.exception))
            continue
        if r.errors:
            problems.append((team, "st.error", r.errors))
            continue
        # Accept the render if ANY of: dataframes, metrics, or substantial markdown
        has_content = r.dataframes or r.metrics or len(r.markdown) >= _MIN_MARKDOWN_LEN
        if not has_content:
            problems.append((team, "blank", "no dataframes/metrics/markdown"))
    assert not problems, "Free Agents problems:\n" + "\n".join(f"  {p}" for p in problems)


def test_free_agents_no_nan_metrics(results):
    """No metric value should be NaN, None, or inf for any team."""
    bad = []
    for team, r in results.items():
        if not r.ran or r.exception:
            continue
        for m in r.metrics:
            if _is_bad_number(m.get("value", "")):
                bad.append((team, m.get("label"), m.get("value")))
    assert not bad, "Free Agents NaN/None metrics:\n" + "\n".join(f"  {b}" for b in bad)


# ── Board populated ───────────────────────────────────────────────────────────


def test_free_agents_board_populated(results):
    """The FA board must produce a non-trivial result for every team.

    Accepts either:
      - At least one DataFrame with ≥ 1 row (the "All Free Agents" table),
      - OR markdown whose plain-text length exceeds a meaningful threshold
        (the page rendered a custom HTML table via render_compact_table).

    Skips teams that errored — those are caught in the smoke test.
    """
    empty_boards = []
    for team, r in results.items():
        if not r.ran or r.exception:
            continue
        # Check DataFrames first
        has_rows = any(len(df) >= 1 for df in r.dataframes)
        # Fall back to markdown length (custom HTML tables land here)
        plain_markdown = _strip_html(r.markdown)
        has_markdown = len(plain_markdown) >= _MIN_MARKDOWN_LEN
        if not has_rows and not has_markdown:
            empty_boards.append(team)
    assert not empty_boards, "FA board appears empty for teams: " + ", ".join(empty_boards)


# ── Numeric plausibility ──────────────────────────────────────────────────────


# Ownership % anchored to "owned"/"ownership" context (FA board renders the
# ownership-heat column as custom HTML via st.markdown → .dataframes is empty,
# so a text/markdown arm is needed for the normal render path).
_OWN_TEXT_RE = re.compile(r"(?:owned|ownership)[^0-9]{0,12}(\d{1,3}(?:\.\d+)?)\s*%?", re.IGNORECASE)
# ECR anchored to an "ECR" label.
_ECR_TEXT_RE = re.compile(r"\bECR\b[^0-9]{0,8}(\d{1,4})", re.IGNORECASE)


def test_free_agents_ownership_pct_in_range(results):
    """Any percent_owned / ownership % value found must be in [0, 100].

    The FA board renders via st.markdown (custom HTML table), so ``.dataframes``
    is empty on the normal render and the old dataframe-only scan was vacuous.
    This version adds a markdown/text arm that scans the unified ``r.text``
    corpus (HTML-stripped) for percent values anchored to ownership context, so
    a genuinely out-of-range ownership % (e.g. 137%) is now actually caught.
    The dataframe arm is kept for the degraded render path.  Only fires when a
    matching value is present.
    """
    out_of_range = []
    for team, r in results.items():
        if not r.ran or r.exception:
            continue

        # Markdown/text arm (normal render path).
        plain = _strip_html(r.text or "")
        for raw in _OWN_TEXT_RE.findall(plain):
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
            if not (0.0 <= val <= 100.0):
                out_of_range.append((team, "ownership(text)", val))

        # DataFrame arm (degraded render path).
        for df in r.dataframes:
            # Look for columns that clearly carry ownership percentage data
            own_cols = [c for c in df.columns if re.search(r"percent_owned|ownership|own_pct", c, re.I)]
            for col in own_cols:
                numeric = (
                    df[col]
                    .apply(lambda v: float(v) if str(v).replace(".", "", 1).replace("-", "", 1).isdigit() else None)
                    .dropna()
                )
                for val in numeric:
                    if not (0.0 <= val <= 100.0):
                        out_of_range.append((team, col, val))
    assert not out_of_range, "Ownership % out of [0,100]:\n" + "\n".join(
        f"  team={t} col={c} val={v}" for t, c, v in out_of_range
    )


def test_free_agents_ecr_plausible(results):
    """ECR values must be positive integers in [1, 2000].

    The FA board renders via st.markdown (custom HTML), so ``.dataframes`` is
    empty on the normal render and the old dataframe-only scan was vacuous.
    This version adds a markdown/text arm that scans the unified ``r.text``
    corpus (HTML-stripped) for integers anchored to an "ECR" label, so a
    genuinely implausible ECR (0, a 5-digit garbage value) is now actually
    caught.  The dataframe arm is kept for the degraded render path (the page
    formats ECR as a whole-number string, PR11).  Only fires when ECR data is
    present.
    """
    bad_ecr = []
    for team, r in results.items():
        if not r.ran or r.exception:
            continue

        # Markdown/text arm (normal render path).
        plain = _strip_html(r.text or "")
        for raw in _ECR_TEXT_RE.findall(plain):
            try:
                val = int(raw)
            except (ValueError, TypeError):
                continue
            if not (1 <= val <= 2000):
                bad_ecr.append((team, "ECR(text)", val))

        # DataFrame arm (degraded render path).
        for df in r.dataframes:
            ecr_cols = [c for c in df.columns if c in ("ECR", "consensus_rank")]
            for col in ecr_cols:
                for raw in df[col].dropna():
                    # Page formats as integer string; skip empty strings
                    s = str(raw).strip()
                    if not s:
                        continue
                    try:
                        val = int(s)
                    except (ValueError, TypeError):
                        continue  # Non-numeric — not an ECR value
                    if not (1 <= val <= 2000):
                        bad_ecr.append((team, col, val))
    assert not bad_ecr, "ECR out of expected range [1, 2000]:\n" + "\n".join(
        f"  team={t} col={c} val={v}" for t, c, v in bad_ecr
    )


def test_free_agents_no_nan_in_dataframes(results):
    """No captured DataFrame cell should contain a bad sentinel (NaN/None/inf).

    Scans every DataFrame returned for every team. Columns that are
    legitimately empty strings (e.g. formatted stats with no data) are
    excluded — only the Python/float sentinel values are flagged.
    """
    bad = []
    for team, r in results.items():
        if not r.ran or r.exception:
            continue
        for df_idx, df in enumerate(r.dataframes):
            for col in df.columns:
                for row_idx, val in enumerate(df[col]):
                    if _is_bad_number(val):
                        bad.append((team, df_idx, col, row_idx, val))
    assert not bad, "NaN/None/inf in Free Agents DataFrames:\n" + "\n".join(
        f"  team={t} df={di} col={c} row={ri} val={v}" for t, di, c, ri, v in bad
    )

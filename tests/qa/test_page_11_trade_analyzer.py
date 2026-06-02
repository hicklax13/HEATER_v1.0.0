"""Deep per-team assertions for Trade Analyzer (pages/11_Trade_Analyzer.py).

Renders ONCE per team (module-scoped `results` fixture).  The 6-phase engine
only runs after the user proposes a trade and clicks "Analyze Trade", so the
headless default render is checked for:

  - Clean load with no exception / st.error.
  - Non-trivial content: the page renders the "You Give" / "You Receive"
    multiselect builder UI plus context-panel HTML, so .markdown should be
    non-trivial even before any trade is proposed.
  - Builder subheaders present: "You Give" / "You Receive" are st.subheader
    calls → they live in .headings (NOT .markdown). Checked via .headings/.text.
  - "You Give" selector populated from the team's own roster: its option labels
    flow into .widget_options (NOT markdown); expect player-name–shaped entries.
  - No NaN/None/inf in any captured st.metric values or st.dataframe cells.
  - If any trade-value or SGP numbers happen to appear (e.g. from a cached
    prior render the harness captured), they must be in a sane band.

Do NOT assert that a grade, verdict, surplus_sgp, or confidence_pct exists —
the engine only produces those after an interactive evaluate click.

Run SERIALLY:
    .venv\\Scripts\\python.exe -m pytest tests/qa/test_page_11_trade_analyzer.py -q
"""

from __future__ import annotations

import re

import pytest

PAGE = "pages/11_Trade_Analyzer.py"
_BAD = ("nan", "none", "inf", "-inf", "")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _is_bad_number(s) -> bool:
    return str(s).strip().lower() in _BAD


def _strip_html(text: str) -> str:
    """Remove HTML tags so regex matching hits actual displayed text/numbers."""
    return re.sub(r"<[^>]+>", " ", text)


# Headings the page always renders in the two-column builder:
#   "You Give" (col1 subheader) and "You Receive" (col2 subheader)
_RE_YOU_GIVE = re.compile(r"\bYou\s+Give\b", re.IGNORECASE)
_RE_YOU_RECEIVE = re.compile(r"\bYou\s+Receive\b", re.IGNORECASE)

# "Trade Status" label from the context card rendered on every page load
_RE_TRADE_STATUS = re.compile(r"\bTrade\s+Status\b", re.IGNORECASE)

# A player-name–shaped widget option: two capitalised words separated by a space,
# anchored to the start of the option label so we validate the OPTION itself rather
# than matching an incidental name buried inside HTML prose.
# Allows initials/suffixes/hyphens: "Aaron Judge", "J.D. Martinez", "Ronald Acuna Jr."
_RE_PLAYER_NAME_OPTION = re.compile(r"^[A-Z][a-zA-Z.'-]+\s+[A-Z][a-zA-Z.'-]+")

# SGP values that can appear if the harness somehow renders a cached result:
# formatted by format_stat as "+N.NN" or "-N.NN"
_RE_SGP = re.compile(r"([+-]\d{1,3}\.\d{2})")

# Trade value 0-100 band (used in trade value chart tab, only if present)
_RE_TRADE_VALUE = re.compile(r"\bvalue[:\s]+(\d{1,3}(?:\.\d+)?)\b", re.IGNORECASE)


# ── Module-scoped fixture: render once per team ───────────────────────────────


@pytest.fixture(scope="module")
def results(run_page_as_team, team_names):
    """Render Trade Analyzer once per team; every assertion reuses this dict."""
    return {t: run_page_as_team(PAGE, t, is_admin=False) for t in team_names}


# ── Test 1: clean load for every team ────────────────────────────────────────


def test_trade_analyzer_renders_for_all_teams(results):
    """Every team: ran, no exception, no st.error, non-trivial content."""
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
        # Something should be rendered: either a dataframe, a metric, or
        # a non-trivial markdown blob (the context card + builder headings
        # emit at minimum ~200 chars of HTML even with no rosters loaded).
        if not r.dataframes and not r.metrics and len(r.markdown) < 50:
            problems.append((team, "blank", "no dataframes/metrics/markdown"))
    assert not problems, "Trade Analyzer problems:\n" + "\n".join(f"  {p}" for p in problems)


# ── Test 2: NaN/None/inf guard on all captured metrics ───────────────────────


def test_trade_analyzer_no_nan_metrics(results):
    """No captured st.metric value is NaN / None / inf for any team."""
    bad = []
    for team, r in results.items():
        if not r.ran or r.exception:
            continue
        for m in r.metrics:
            if _is_bad_number(m.get("value", "")):
                bad.append((team, m.get("label"), m.get("value")))
    assert not bad, "Trade Analyzer NaN/None metrics:\n" + "\n".join(f"  {b}" for b in bad)


# ── Test 3: builder UI headings present in HTML ───────────────────────────────


def test_trade_analyzer_builder_headings_present(results):
    """The 'You Give' and 'You Receive' subheaders appear in the rendered page.

    These are emitted by ``st.subheader("You Give")`` and
    ``st.subheader("You Receive")`` (pages/11_Trade_Analyzer.py L189 / L202),
    unconditionally inside the two-column builder once rosters load.  Subheaders
    are a SEPARATE Streamlit element type from st.markdown — they land in
    ``r.headings``, never in ``r.markdown`` — so we search the UNION of the
    headings blob and the unified ``r.text`` corpus.

    If they are absent the page either stopped early (empty pool → st.stop)
    or the builder failed to render.  Teams that legitimately rendered nothing
    (early st.stop on empty pool) are skipped.
    """
    missing = []
    for team, r in results.items():
        if not r.ran or r.exception or r.errors:
            continue
        # Union of headings (where subheaders live) + the unified text corpus.
        haystack = "\n".join(r.headings) + "\n" + (r.text or "")
        plain = _strip_html(haystack)
        # Skip teams whose render produced no substantive content (empty-pool
        # early st.stop): nothing in headings AND a trivial text corpus.
        if not r.headings and len(plain.strip()) < 50:
            continue
        if not _RE_YOU_GIVE.search(plain):
            missing.append((team, "missing 'You Give' heading"))
        if not _RE_YOU_RECEIVE.search(plain):
            missing.append((team, "missing 'You Receive' heading"))
    assert not missing, "Trade Analyzer missing builder headings:\n" + "\n".join(
        f"  team={m[0]}  reason={m[1]}" for m in missing
    )


# ── Test 4: roster player names in the "You Give" selector HTML ───────────────


def test_trade_analyzer_give_options_populated(results):
    """The "You Give" selector is populated from the team's own roster.

    pages/11_Trade_Analyzer.py builds ``give_options`` from the user's roster
    (L193) and feeds them to ``st.multiselect`` (L196).  Multiselect/selectbox
    option labels are captured by the harness into ``r.widget_options`` (they are
    NOT in markdown).  We assert that, for every team whose builder rendered,
    ``r.widget_options`` is non-empty AND contains at least a few player-name–
    shaped entries — proving the roster actually populated the selector rather
    than the selector being empty or carrying only sentinel labels.

    (The old version was vacuous: it matched card titles via a loose name regex
    over markdown and passed regardless of whether the selector populated.)
    """
    _MIN_NAME_OPTIONS = 3
    problems = []
    for team, r in results.items():
        if not r.ran or r.exception or r.errors:
            continue
        opts = list(r.widget_options or [])
        # Defensive skip: a team whose builder never rendered (empty-pool early
        # st.stop) produces no widget options AND no headings.
        if not opts and not r.headings:
            continue
        if not opts:
            problems.append((team, "widget_options empty (selector not populated)"))
            continue
        name_options = [o for o in opts if _RE_PLAYER_NAME_OPTION.match(str(o).strip())]
        if len(name_options) < _MIN_NAME_OPTIONS:
            problems.append(
                (
                    team,
                    f"only {len(name_options)} player-name-shaped options "
                    f"(need >= {_MIN_NAME_OPTIONS}); options sample={opts[:5]}",
                )
            )
    assert not problems, "Trade Analyzer 'You Give' selector not populated from roster:\n" + "\n".join(
        f"  team={p[0]}  reason={p[1]}" for p in problems
    )


# ── Test 5: any incidentally rendered SGP / trade-value numbers are sane ─────


def test_trade_analyzer_incidental_numbers_sane(results):
    """If SGP or trade-value numbers appear in HTML, they must be in a sane band.

    The default headless render does NOT click "Analyze Trade", so these
    numbers typically do NOT appear.  This test fires vacuously on a clean
    default render and acts as a regression guard for any future caching
    that might surface stale computed values.

    Sane bands (same as CLAUDE.md / structural-invariant documentation):
      SGP:          −30.0 ..  +30.0   (format_stat renders as "+N.NN" / "-N.NN")
      Trade value:    0.0 ..  100.0   (universal trade value chart 0-100)

    Also checks that no st.dataframe cell contains NaN/None/inf — the
    acceptance-analysis ADP detail table is the most likely dataframe on
    the default render (it only renders post-evaluate, so this fires
    vacuously but guards future changes).
    """
    bad = []
    for team, r in results.items():
        if not r.ran or r.exception:
            continue

        plain = _strip_html(r.markdown)

        # SGP values
        for m in _RE_SGP.finditer(plain):
            raw = m.group(1)
            try:
                val = float(raw)
            except ValueError:
                continue
            if not (-30.0 <= val <= 30.0):
                bad.append((team, "SGP", raw, "outside [-30, +30]"))

        # Trade-value numbers
        for m in _RE_TRADE_VALUE.finditer(plain):
            raw = m.group(1)
            try:
                val = float(raw)
            except ValueError:
                continue
            if not (0.0 <= val <= 100.0):
                bad.append((team, "trade_value", raw, "outside [0, 100]"))

        # Dataframe cell NaN guard
        for i, df in enumerate(r.dataframes):
            if df is None or not hasattr(df, "iteritems") and not hasattr(df, "items"):
                continue
            try:
                import pandas as pd

                for col in df.columns:
                    for cell in df[col]:
                        s = str(cell).strip().lower()
                        if s in _BAD:
                            bad.append((team, f"df[{i}].{col}", cell, "NaN/None/inf in cell"))
            except Exception:
                pass  # defensive: never fail due to introspection error

    assert not bad, "Trade Analyzer out-of-range / bad-cell values:\n" + "\n".join(
        f"  team={b[0]}  kind={b[1]}  value={b[2]}  reason={b[3]}" for b in bad
    )

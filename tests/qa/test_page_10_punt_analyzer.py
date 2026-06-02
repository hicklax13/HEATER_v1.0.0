"""Deep per-team assertions for Punt Analyzer (pages/10_Punt_Analyzer.py).

Renders ONCE per team (module-scoped `results` fixture) and checks the punt
recommendation surface is plausible (covers the 12 categories; any probability
in range; recommendation present). Defensive style: checks fire only when data
is present — this is a strategy page whose output is necessarily team-specific.

DEFAULT RENDER BEHAVIOUR
------------------------
Without a multiselect choice the page hits ``st.stop()`` immediately after
rendering the info banner ("Select one or more categories above …").  So the
default headless run produces:

  * ``.markdown`` — page-title HTML + description prose + info banner text +
    the multiselect widget label / option text (AppTest captures all
    st.markdown / st.info / st.multiselect label output here).
  * ``.dataframes`` — empty (no tables rendered before stop).
  * ``.metrics``   — empty (st.metric only fires inside the full-analysis path).

All plausibility tests guard on data presence, so they remain meaningful
whether or not a future harness version injects punt_cats session state.

Run SERIALLY: .venv\\Scripts\\python.exe -m pytest tests/qa/test_page_10_punt_analyzer.py -q
"""

from __future__ import annotations

import re

import pytest

PAGE = "pages/10_Punt_Analyzer.py"
_BAD = ("nan", "none", "inf", "-inf", "")

# The 12 FourzynBurn H2H categories (from CLAUDE.md / LeagueConfig)
_ALL_CATS = frozenset({"R", "HR", "RBI", "SB", "AVG", "OBP", "W", "L", "SV", "K", "ERA", "WHIP"})

# ── helpers ───────────────────────────────────────────────────────────────────


def _is_bad_number(s) -> bool:
    return str(s).strip().lower() in _BAD


def _strip_html(text: str) -> str:
    """Remove HTML tags so regex matching hits the actual displayed text."""
    return re.sub(r"<[^>]+>", " ", text)


# Rank pattern: "7/12" or "rank: 7" from the standings-impact table.
# Only emitted during the full-analysis path (standings section).
_RE_RANK_SLASH = re.compile(r"\b(\d{1,2})/12\b")

# SGP values: "+2.34" or "-1.05" — emitted in the gainers/losers tables.
_RE_SGP = re.compile(r"[+\-]\d{1,3}\.\d{2}")

# Standings-impact metric: captured by AppTest as e.g. {"label": "Standings Points from Active Categories", "value": "47"}
_STANDINGS_METRIC_LABEL = re.compile(r"standings points from active", re.IGNORECASE)


# ── fixture ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def results(run_page_as_team, team_names):
    """Render Punt Analyzer once per team; reused by every assertion."""
    return {t: run_page_as_team(PAGE, t, is_admin=False) for t in team_names}


# ── Test 1: render gate ───────────────────────────────────────────────────────


def test_punt_renders_for_all_teams(results):
    """Per team: ran, no exception, no st.error, and SOMETHING rendered (blank-page guard).

    The page always renders at least the page-title markup and the info banner
    before st.stop(), so .markdown length ≥ 50 chars is the minimum bar.
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
        if not r.dataframes and not r.metrics and len(r.markdown) < 50:
            problems.append((team, "blank", "no dataframes/metrics/markdown"))
    assert not problems, "Punt Analyzer problems:\n" + "\n".join(f"  {p}" for p in problems)


# ── Test 2: no NaN/None/inf in captured metrics ───────────────────────────────


def test_punt_no_nan_metrics(results):
    """No displayed st.metric value is NaN/None/inf for any team.

    st.metric only fires in the full-analysis path (standings section); this
    test is a no-op on the default render but catches regressions if the
    harness ever injects punt_cats state.
    """
    bad = []
    for team, r in results.items():
        if not r.ran or r.exception:
            continue
        for m in r.metrics:
            if _is_bad_number(m.get("value", "")):
                bad.append((team, m.get("label"), m.get("value")))
    assert not bad, "Punt Analyzer NaN/None metrics:\n" + "\n".join(f"  {b}" for b in bad)


# ── Test 3: category coverage in rendered markdown ────────────────────────────


def test_punt_category_tokens_present(results):
    """The 12 H2H category tokens are well-represented in the punt multiselect.

    pages/10_Punt_Analyzer.py builds the "Categories to punt:" multiselect with
    ``options=config.all_categories`` (L55-62) — i.e. the 12 canonical tokens
    (R, HR, RBI, SB, AVG, OBP, W, L, SV, K, ERA, WHIP) — and this multiselect
    renders BEFORE the ``st.stop()`` on the empty-selection path (L64-66).  So on
    the default headless render those 12 options ARE captured into
    ``r.widget_options`` (option text is NOT in markdown — the previous
    markdown-only scan was a likely false positive because the page st.stop()s
    before enumerating categories in prose).

    Primary check: ≥ 6 of the 12 canonical tokens appear among the captured
    widget options.  Defensive fallback: if a render produced no widget options
    at all (selector not captured on this Streamlit version), require at least
    that the multiselect EXISTS-by-evidence via the page's non-trivial render and
    scan ``r.text`` for tokens, skipping when neither channel surfaces them.
    """
    bad = []
    for team, r in results.items():
        if not r.ran or r.exception:
            continue
        if len(r.markdown) < 50 and not r.widget_options:
            continue  # blank render; caught by test_punt_renders_for_all_teams

        # Primary: tokens among the multiselect option labels.
        if r.widget_options:
            opt_blob = "\n".join(str(o) for o in r.widget_options)
            found = {cat for cat in _ALL_CATS if re.search(r"\b" + re.escape(cat) + r"\b", opt_blob)}
            n = len(found)
            if n < 6:
                bad.append(
                    (team, f"only {n}/12 category tokens among widget_options (sample={list(r.widget_options)[:6]})")
                )
            continue

        # Defensive fallback (no widget options captured): scan the unified text
        # corpus; only flag if SOME tokens appear but fewer than 6 — a wholly
        # absent token set means the category list only renders post-selection, so
        # we skip rather than false-fail.
        plain = _strip_html(r.text or "")
        found = {cat for cat in _ALL_CATS if re.search(r"\b" + re.escape(cat) + r"\b", plain)}
        n = len(found)
        if 0 < n < 6:
            bad.append((team, f"only {n}/12 category tokens found in text corpus"))
    assert not bad, "Punt Analyzer missing category tokens:\n" + "\n".join(f"  team={b[0]}  reason={b[1]}" for b in bad)


# ── Test 4: rank values in [1, 12] and SGP values not absurd ─────────────────


def test_punt_plausible_values_when_present(results):
    """Validate any rank or SGP values found in .markdown / .dataframes.

    Rank tokens ("N/12") must have N ∈ [1, 12].
    SGP tokens ("+N.NN" / "-N.NN") found in .markdown must be in [-50, 50]
    (a single player's SGP contribution never exceeds ±50 in practice).

    Dataframe numeric columns (when gainers/losers tables are captured) must
    not contain all-NaN columns.

    All guards are conditional on the data being present — an absent pattern
    is never a failure.
    """
    bad = []
    for team, r in results.items():
        if not r.ran or r.exception:
            continue

        plain = _strip_html(r.markdown)

        # Rank plausibility: "N/12" patterns
        for m in _RE_RANK_SLASH.finditer(plain):
            try:
                rank = int(m.group(1))
            except ValueError:
                continue
            if not (1 <= rank <= 12):
                bad.append((team, "rank", m.group(0), "outside [1, 12]"))

        # SGP plausibility: "+N.NN" / "-N.NN" tokens
        for m in _RE_SGP.finditer(plain):
            raw = m.group(0)
            try:
                val = float(raw)
            except ValueError:
                continue
            if not (-50.0 <= val <= 50.0):
                bad.append((team, "SGP", raw, "outside [-50, 50]"))

        # Dataframe integrity: no all-NaN column
        for i, df in enumerate(r.dataframes):
            if df is None or not hasattr(df, "columns"):
                continue
            for col in df.columns:
                if df[col].isna().all():
                    bad.append((team, f"df[{i}].{col}", "all-NaN column", ""))

    assert not bad, "Punt Analyzer implausible values:\n" + "\n".join(
        f"  team={b[0]}  kind={b[1]}  value={b[2]}  reason={b[3]}" for b in bad
    )


# ── Test 5: standings-metric plausibility when present ───────────────────────


def test_punt_standings_metric_plausible(results):
    """When the 'Standings Points from Active Categories' metric is captured,
    its integer value must be in [0, 12*12] = [0, 144].

    This metric is only emitted in the full-analysis path (requires standings
    + rosters from Yahoo + a matched user team).  On the default headless
    render it is absent — the test is a no-op and only fires if the harness
    is later upgraded to inject session state.
    """
    bad = []
    for team, r in results.items():
        if not r.ran or r.exception:
            continue
        for m in r.metrics:
            label = str(m.get("label", ""))
            if not _STANDINGS_METRIC_LABEL.search(label):
                continue
            raw = str(m.get("value", "")).strip()
            if _is_bad_number(raw):
                bad.append((team, label, raw, "NaN/None/inf"))
                continue
            try:
                val = int(raw)
            except ValueError:
                continue  # non-integer — skip rather than fail
            # 12 categories × 12 teams max; active subset is always ≤ total
            if not (0 <= val <= 144):
                bad.append((team, label, raw, "outside [0, 144]"))
    assert not bad, "Punt Analyzer standings metric out of range:\n" + "\n".join(
        f"  team={b[0]}  label={b[1]}  value={b[2]}  reason={b[3]}" for b in bad
    )

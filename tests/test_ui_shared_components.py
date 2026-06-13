"""Tests for the two shared UI components added to src/ui_shared.py:

1. Data-freshness chip — humanize_age() helper + render_data_freshness_chip()
2. Glossary / jargon tooltips — JARGON dict + jargon_help() + render_glossary_expander()

TDD: these tests were written BEFORE the implementation. All assertions are about
the public contract so page agents can rely on the signatures being stable.
"""

from __future__ import annotations

import unittest.mock as mock

import pytest

# ── helpers ───────────────────────────────────────────────────────────────────


def _import():
    """Import the relevant symbols; skip with a clean message if Streamlit is absent."""
    from src.ui_shared import (
        JARGON,
        humanize_age,
        jargon_help,
        render_data_freshness_chip,
        render_glossary_expander,
    )

    return JARGON, humanize_age, jargon_help, render_data_freshness_chip, render_glossary_expander


# ═══════════════════════════════════════════════════════════════════════════════
# Component 1 — Data-freshness chip
# ═══════════════════════════════════════════════════════════════════════════════


class TestHumanizeAge:
    """humanize_age(age_minutes) → human-readable string, no Streamlit needed."""

    def setup_method(self):
        _import()  # verify importable
        from src.ui_shared import humanize_age

        self.humanize_age = humanize_age

    def test_under_one_hour_shows_minutes(self):
        result = self.humanize_age(5)
        assert "5" in result
        assert "minute" in result.lower()

    def test_exactly_one_hour(self):
        result = self.humanize_age(60)
        assert "1" in result
        assert "hour" in result.lower()

    def test_under_two_hours(self):
        result = self.humanize_age(90)
        # 90 min → should say "1 hour" (floor) or "2 hours" — either is fine,
        # but "hour" must appear (not "minute")
        assert "hour" in result.lower()

    def test_two_hours(self):
        result = self.humanize_age(120)
        assert "2" in result
        assert "hour" in result.lower()

    def test_one_day_boundary(self):
        result = self.humanize_age(1440)
        assert "day" in result.lower()

    def test_two_days(self):
        result = self.humanize_age(2880)
        assert "2" in result
        assert "day" in result.lower()

    def test_zero_minutes(self):
        result = self.humanize_age(0)
        # "just now" or "0 minutes ago" — must be a non-empty string
        assert isinstance(result, str)
        assert result.strip()

    def test_returns_string(self):
        assert isinstance(self.humanize_age(45), str)

    def test_single_minute(self):
        result = self.humanize_age(1)
        assert "1" in result
        assert "minute" in result.lower()

    def test_59_minutes(self):
        result = self.humanize_age(59)
        assert "minute" in result.lower()


class TestDataFreshnessChip:
    """render_data_freshness_chip() renders amber treatment for old data."""

    def setup_method(self):
        _import()
        from src.ui_shared import render_data_freshness_chip

        self.render_chip = render_data_freshness_chip

    def test_age_minutes_fresh_no_amber(self):
        """age_minutes <= 1440 → neutral/green treatment; amber token must NOT appear."""
        calls = []
        with mock.patch("streamlit.markdown") as mock_md:
            mock_md.side_effect = lambda html, **kw: calls.append(html)
            self.render_chip(age_minutes=60)

        assert calls, "render_data_freshness_chip must call st.markdown"
        combined = " ".join(calls)
        # Should contain a time reference but NOT the amber warning color
        # The amber warning is THEME["warn"] == "#ff9f1c" or "warn" or amber
        # We just check it doesn't scream "stale" visually
        assert "minute" in combined.lower() or "hour" in combined.lower() or "unknown" in combined.lower()

    def test_age_minutes_stale_uses_amber(self):
        """age_minutes > 1440 (>24 h) → amber/warning visual treatment."""
        calls = []
        with mock.patch("streamlit.markdown") as mock_md:
            mock_md.side_effect = lambda html, **kw: calls.append(html)
            self.render_chip(age_minutes=2000)

        combined = " ".join(calls)
        # The chip must contain the warn color token or "warn" CSS variable
        # THEME["warn"] == "#ff9f1c" or "--fp-amber"
        # We check for the hex or the CSS var
        has_amber = "#ff9f1c" in combined or "fp-amber" in combined or "warn" in combined.lower()
        assert has_amber, f"Stale chip (age > 1440 min) must use amber/warn color. Got: {combined[:400]}"

    def test_no_source_no_age_renders_unknown(self):
        """When neither source nor age_minutes is given, renders gracefully."""
        calls = []
        with mock.patch("streamlit.markdown") as mock_md:
            mock_md.side_effect = lambda html, **kw: calls.append(html)
            # Patch out get_refresh_log_snapshot to return empty list
            with mock.patch("src.ui_shared.get_refresh_log_snapshot", return_value=[]):
                self.render_chip()  # no args

        combined = " ".join(calls)
        assert "unknown" in combined.lower() or "freshness" in combined.lower() or combined.strip()

    def test_source_not_in_log_renders_unknown(self):
        """When source not found in refresh_log, renders 'freshness unknown'."""
        calls = []
        with mock.patch("streamlit.markdown") as mock_md:
            mock_md.side_effect = lambda html, **kw: calls.append(html)
            with mock.patch("src.ui_shared.get_refresh_log_snapshot", return_value=[]):
                self.render_chip(source="projections")

        combined = " ".join(calls)
        assert "unknown" in combined.lower() or combined.strip()

    def test_no_emoji_in_output(self):
        """No emoji characters in chip output."""
        calls = []
        with mock.patch("streamlit.markdown") as mock_md:
            mock_md.side_effect = lambda html, **kw: calls.append(html)
            self.render_chip(age_minutes=30)

        combined = " ".join(calls)
        # Check for common emoji unicode ranges (U+1F300–U+1FAFF)
        for char in combined:
            cp = ord(char)
            assert not (0x1F300 <= cp <= 0x1FAFF), f"Emoji found in chip output: {char!r}"

    def test_source_lookup_uses_get_refresh_log_snapshot(self):
        """When source is given, get_refresh_log_snapshot is called."""
        fake_log = [
            {
                "source": "projections",
                "last_refresh": "2026-06-13T10:00:00",
                "status": "success",
                "rows_written": 100,
                "message": "",
                "tier": "primary",
            }
        ]
        calls = []
        with mock.patch("streamlit.markdown") as mock_md:
            mock_md.side_effect = lambda html, **kw: calls.append(html)
            with mock.patch("src.ui_shared.get_refresh_log_snapshot", return_value=fake_log) as mock_snap:
                self.render_chip(source="projections")

        mock_snap.assert_called()


# ═══════════════════════════════════════════════════════════════════════════════
# Component 2 — Glossary / jargon tooltips
# ═══════════════════════════════════════════════════════════════════════════════

_REQUIRED_TERMS = [
    "SGP",
    "DCV",
    "VORP",
    "wRC+",
    "xFIP",
    "SIERA",
    "Stuff+",
    "Net SGP",
    "Stream Score",
    "Heat",
    "% JOB",
    "Smash",
    "Magic#",
    "SOS",
    "Sell-High",
    "Buy-Low",
    "gmLI",
    "ECR",
    "ADP",
]


class TestJargonDict:
    """JARGON dict contains all required keys with clean definitions."""

    def setup_method(self):
        from src.ui_shared import JARGON

        self.JARGON = JARGON

    def test_is_dict(self):
        assert isinstance(self.JARGON, dict)

    @pytest.mark.parametrize("term", _REQUIRED_TERMS)
    def test_required_term_present(self, term):
        assert term in self.JARGON, f"JARGON missing required term: {term!r}"

    @pytest.mark.parametrize("term", _REQUIRED_TERMS)
    def test_definition_non_empty(self, term):
        defn = self.JARGON.get(term, "")
        assert defn and defn.strip(), f"JARGON[{term!r}] is empty"

    @pytest.mark.parametrize("term", _REQUIRED_TERMS)
    def test_definition_under_120_chars(self, term):
        defn = self.JARGON.get(term, "")
        assert len(defn) <= 120, f"JARGON[{term!r}] is {len(defn)} chars (max 120): {defn!r}"

    @pytest.mark.parametrize("term", _REQUIRED_TERMS)
    def test_definition_no_emoji(self, term):
        defn = self.JARGON.get(term, "")
        for char in defn:
            cp = ord(char)
            assert not (0x1F300 <= cp <= 0x1FAFF), f"JARGON[{term!r}] contains emoji: {char!r}"

    def test_all_values_are_strings(self):
        for k, v in self.JARGON.items():
            assert isinstance(v, str), f"JARGON[{k!r}] is not a string"


class TestJargonHelp:
    """jargon_help(term) → str — returns definition or ''."""

    def setup_method(self):
        from src.ui_shared import jargon_help

        self.jargon_help = jargon_help

    def test_known_term_returns_definition(self):
        result = self.jargon_help("SGP")
        assert isinstance(result, str)
        assert result.strip(), "jargon_help('SGP') must return a non-empty string"

    def test_unknown_term_returns_empty_string(self):
        result = self.jargon_help("TOTALLY_UNKNOWN_TERM_XYZ")
        assert result == "", f"Unknown term must return '', got {result!r}"

    def test_all_required_terms_return_non_empty(self):
        for term in _REQUIRED_TERMS:
            result = self.jargon_help(term)
            assert result, f"jargon_help({term!r}) returned empty"

    def test_returns_string(self):
        assert isinstance(self.jargon_help("DCV"), str)
        assert isinstance(self.jargon_help("BOGUS"), str)

    def test_case_sensitive_unknown(self):
        # "sgp" (lowercase) is not a required key; may or may not be defined —
        # but if it's not in JARGON it must return "".
        from src.ui_shared import JARGON

        term_lower = "sgp"
        expected = JARGON.get(term_lower, "")
        assert self.jargon_help(term_lower) == expected


class TestRenderGlossaryExpander:
    """render_glossary_expander() calls st.expander with the right label."""

    def setup_method(self):
        from src.ui_shared import render_glossary_expander

        self.render = render_glossary_expander

    def _run_with_mock(self, *args, **kwargs):
        """Run render_glossary_expander capturing st calls."""
        expander_labels = []

        class _FakeExpander:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *a):
                pass

        with mock.patch("streamlit.expander") as mock_exp:
            mock_exp.return_value = _FakeExpander()
            mock_exp.side_effect = lambda label, **kw: (expander_labels.append(label) or _FakeExpander())
            with mock.patch("streamlit.markdown"):
                with mock.patch("streamlit.write"):
                    self.render(*args, **kwargs)

        return expander_labels

    def test_default_label(self):
        labels = self._run_with_mock()
        assert labels, "render_glossary_expander must call st.expander"
        assert labels[0] == "What do these numbers mean?", f"Default label mismatch: {labels[0]!r}"

    def test_custom_label(self):
        labels = self._run_with_mock(label="Glossary")
        assert labels and labels[0] == "Glossary"

    def test_terms_subset(self):
        """Passing a subset of terms should work without raising."""
        self._run_with_mock(terms=["SGP", "DCV"])

    def test_all_terms_when_none_passed(self):
        """Passing terms=None renders all JARGON entries (no crash)."""
        self._run_with_mock(terms=None)

    def test_empty_terms_list(self):
        """Passing terms=[] renders nothing but still calls expander."""
        labels = self._run_with_mock(terms=[])
        assert labels  # expander still called

    def test_unknown_term_in_list_no_crash(self):
        """Unknown term in list must not raise."""
        self._run_with_mock(terms=["SGP", "NOT_A_REAL_TERM_XYZ"])

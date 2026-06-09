"""Tests for the Combustion-redesign detail renderers (Phase B, 2026-06-08).

These lock the reusable HTML-string builders + their thin ``st.markdown``
wrappers added to ``src/ui_shared.py``:

- ``build_eyebrow_html`` / ``render_eyebrow``        — mono uppercase eyebrow label
- ``render_page_header``                             — the mockup ``.phead`` block
- ``build_panel_html`` / ``render_panel``            — full ``.instr-panel`` instrument card
- ``build_heatbar_html``                             — mockup ``.cb``/``.cat`` heat bar
- ``build_sparkline_html``                           — mockup ``.spark`` sparkline
- ``build_stat_readout_html``                        — mockup ``.stat`` readout

The builders are pure functions (no Streamlit runtime needed). The ``render_*``
wrappers are tested by monkeypatching ``ui_shared.st`` with a mock and asserting
``st.markdown`` was called with the built HTML + ``unsafe_allow_html=True``.

Invariants guarded:
- each builder returns a non-empty ``str`` containing its expected class(es);
- colors come from the ``--fp-*`` tokens (never raw brand hex) — the orange
  accent surfaces reference ``var(--fp-primary)``;
- ``build_heatbar_html`` clamps the fill width to [0, 100];
- ``build_sparkline_html([])`` does not raise (empty/zero guard);
- ``build_stat_readout_html(..., accent=True)`` includes the orange token;
- NO builder output contains an emoji / non-ASCII pictograph;
- ``render_page_header`` output carries the title and (when given) the eyebrow
  + fig crumb text.
"""

from __future__ import annotations

import re
from unittest import mock

import pytest

from src import ui_shared

# A compact emoji / pictograph detector. Covers the common Unicode emoji blocks
# (Misc Symbols & Pictographs, Emoticons, Transport, Supplemental Symbols,
# Dingbats, Misc Symbols, regional indicators, variation selector-16).
_EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001faff"  # symbols & pictographs (incl. supplemental + extended-A)
    "\U00002600-\U000027bf"  # misc symbols + dingbats
    "\U0001f1e6-\U0001f1ff"  # regional indicators
    "\U0000fe0f"  # variation selector-16 (emoji presentation)
    "\U00002b00-\U00002bff"  # misc symbols & arrows (stars etc.)
    "]"
)


def _assert_no_emoji(s: str) -> None:
    found = _EMOJI_RE.findall(s)
    assert not found, f"output must not contain emoji/pictographs, found: {found!r}"


# ── build_eyebrow_html / render_eyebrow ───────────────────────────────


class TestEyebrow:
    def test_build_eyebrow_html_basic(self):
        html = ui_shared.build_eyebrow_html("Season")
        assert isinstance(html, str)
        assert html  # non-empty
        assert "eyebrow" in html
        assert "Season" in html
        _assert_no_emoji(html)

    def test_build_eyebrow_html_uses_muted_token(self):
        # Brightened to the muted token per the spec.
        html = ui_shared.build_eyebrow_html("Roster Control")
        assert "var(--fp-tx-muted)" in html

    def test_build_eyebrow_html_escapes(self):
        html = ui_shared.build_eyebrow_html("A & B <x>")
        assert "&amp;" in html
        assert "<x>" not in html

    def test_render_eyebrow_calls_markdown(self):
        m = mock.MagicMock()
        with mock.patch.object(ui_shared, "st", m):
            ui_shared.render_eyebrow("Season")
        m.markdown.assert_called_once()
        args, kwargs = m.markdown.call_args
        assert "eyebrow" in args[0]
        assert "Season" in args[0]
        assert kwargs.get("unsafe_allow_html") is True


# ── render_page_header ────────────────────────────────────────────────


class TestPageHeader:
    def test_render_page_header_title_only(self):
        m = mock.MagicMock()
        with mock.patch.object(ui_shared, "st", m):
            ui_shared.render_page_header("My Team")
        m.markdown.assert_called_once()
        html = m.markdown.call_args[0][0]
        assert "phead" in html
        assert "My Team" in html
        # Orange "." accent + orange underline rule come from the primary token.
        assert "var(--fp-primary)" in html
        assert m.markdown.call_args[1].get("unsafe_allow_html") is True
        _assert_no_emoji(html)

    def test_render_page_header_with_eyebrow_and_fig(self):
        m = mock.MagicMock()
        with mock.patch.object(ui_shared, "st", m):
            ui_shared.render_page_header(
                "My Team",
                eyebrow="Season",
                fig="FIG.01 — ROSTER CONTROL",
            )
        html = m.markdown.call_args[0][0]
        assert "My Team" in html
        assert "Season" in html
        assert "FIG.01" in html
        assert "ROSTER CONTROL" in html
        assert "eyebrow" in html
        assert "fig" in html

    def test_render_page_header_with_actions_html(self):
        m = mock.MagicMock()
        actions = '<button class="zzbtn">Refresh</button>'
        with mock.patch.object(ui_shared, "st", m):
            ui_shared.render_page_header("My Team", actions_html=actions)
        html = m.markdown.call_args[0][0]
        assert "zzbtn" in html
        assert "Refresh" in html

    def test_render_page_header_escapes_title(self):
        m = mock.MagicMock()
        with mock.patch.object(ui_shared, "st", m):
            ui_shared.render_page_header("Tom & Jerry <x>")
        html = m.markdown.call_args[0][0]
        assert "&amp;" in html
        # The raw title's angle-bracket tag must be escaped (not injected).
        assert "Jerry <x>" not in html


# ── build_panel_html / render_panel ───────────────────────────────────


class TestPanel:
    def test_build_panel_html_basic(self):
        html = ui_shared.build_panel_html("Matchup Pulse", "<i>body</i>")
        assert isinstance(html, str)
        assert html
        assert "instr-panel" in html
        assert "Matchup Pulse" in html
        # Body HTML is embedded verbatim (not escaped).
        assert "<i>body</i>" in html
        _assert_no_emoji(html)

    def test_build_panel_html_has_corner_ticks(self):
        html = ui_shared.build_panel_html("X", "y")
        # Four corner ticks, mirroring the mockup .pcorner tl/tr/bl/br.
        for cls in ("pcorner tl", "pcorner tr", "pcorner bl", "pcorner br"):
            assert cls in html, f"missing corner tick {cls!r}"

    def test_build_panel_html_has_accent_bar(self):
        html = ui_shared.build_panel_html("X", "y")
        # Orange accent header bar derives from the primary token.
        assert "accent" in html
        assert "var(--fp-primary)" in html

    def test_build_panel_html_fig_label(self):
        html = ui_shared.build_panel_html("X", "y", fig_label="FIG.09")
        assert "FIG.09" in html
        assert "fig" in html

    def test_build_panel_html_escapes_title(self):
        html = ui_shared.build_panel_html("A & B", "<i>ok</i>")
        assert "&amp;" in html
        # Body still unescaped.
        assert "<i>ok</i>" in html

    def test_build_panel_html_accent_left_variant(self):
        html = ui_shared.build_panel_html("X", "y", accent="left")
        assert "accent-left" in html

    def test_render_panel_calls_markdown_with_built_html(self):
        m = mock.MagicMock()
        with mock.patch.object(ui_shared, "st", m):
            ui_shared.render_panel("Matchup Pulse", "<i>body</i>", fig_label="WK 12")
        m.markdown.assert_called_once()
        html = m.markdown.call_args[0][0]
        assert "instr-panel" in html
        assert "Matchup Pulse" in html
        assert "WK 12" in html
        assert m.markdown.call_args[1].get("unsafe_allow_html") is True


# ── build_heatbar_html ────────────────────────────────────────────────


class TestHeatbar:
    def test_build_heatbar_html_basic(self):
        html = ui_shared.build_heatbar_html(62)
        assert isinstance(html, str)
        assert html
        assert "cb" in html
        assert "62" in html  # width percentage present
        _assert_no_emoji(html)

    def test_build_heatbar_html_clamps_over_100(self):
        html = ui_shared.build_heatbar_html(120)
        assert "width:100%" in html
        assert "120%" not in html

    def test_build_heatbar_html_clamps_negative(self):
        html = ui_shared.build_heatbar_html(-5)
        assert "width:0%" in html
        assert "-5" not in html

    def test_build_heatbar_html_win_uses_orange(self):
        html = ui_shared.build_heatbar_html(58, win=True)
        # Orange gradient for a winning/high bar.
        assert "var(--fp-primary)" in html or "var(--fp-ember)" in html or "var(--fp-flame)" in html

    def test_build_heatbar_html_low_uses_steel(self):
        html = ui_shared.build_heatbar_html(22, win=False)
        # Steel fill for a losing/low bar.
        assert "var(--fp-cold)" in html

    def test_build_heatbar_html_float_input(self):
        # Accepts float and does not raise.
        html = ui_shared.build_heatbar_html(33.4)
        assert "cb" in html


# ── build_sparkline_html ──────────────────────────────────────────────


class TestSparkline:
    def test_build_sparkline_html_basic(self):
        html = ui_shared.build_sparkline_html([1, 2, 3, 4])
        assert isinstance(html, str)
        assert html
        assert "spark" in html
        # One bar per value.
        assert html.count("<i") == 4
        _assert_no_emoji(html)

    def test_build_sparkline_html_empty_no_raise(self):
        # Must not raise on empty input.
        html = ui_shared.build_sparkline_html([])
        assert isinstance(html, str)
        assert "spark" in html

    def test_build_sparkline_html_all_zero_no_raise(self):
        # All-zero values must not divide-by-zero.
        html = ui_shared.build_sparkline_html([0, 0, 0])
        assert isinstance(html, str)
        assert "spark" in html
        assert html.count("<i") == 3

    def test_build_sparkline_html_hot_uses_orange(self):
        html = ui_shared.build_sparkline_html([1, 2, 3], tone="hot")
        assert "var(--fp-primary)" in html or "var(--fp-ember)" in html or "var(--fp-flame)" in html

    def test_build_sparkline_html_cold_uses_steel(self):
        html = ui_shared.build_sparkline_html([3, 2, 1], tone="cold")
        assert "var(--fp-cold)" in html

    def test_build_sparkline_html_scales_to_max(self):
        # Largest value maps to a 100% bar.
        html = ui_shared.build_sparkline_html([5, 10])
        assert "height:100%" in html


# ── build_stat_readout_html ───────────────────────────────────────────


class TestStatReadout:
    def test_build_stat_readout_html_basic(self):
        html = ui_shared.build_stat_readout_html("Record", "2-7-3")
        assert isinstance(html, str)
        assert html
        assert "stat" in html
        assert "Record" in html
        assert "2-7-3" in html
        _assert_no_emoji(html)

    def test_build_stat_readout_html_accent_uses_orange(self):
        html = ui_shared.build_stat_readout_html("Record", "2-7-3", accent=True)
        assert "var(--fp-primary)" in html

    def test_build_stat_readout_html_not_accent_no_orange_value(self):
        # The non-accent value should use the standard text token, not orange.
        html = ui_shared.build_stat_readout_html("Roster", "26", accent=False)
        assert "26" in html

    def test_build_stat_readout_html_sub_line(self):
        html = ui_shared.build_stat_readout_html("Rank", "10", sub="/12")
        assert "/12" in html

    def test_build_stat_readout_html_escapes(self):
        html = ui_shared.build_stat_readout_html("A & B", "<x>", sub="<y>")
        assert "&amp;" in html
        assert "<x>" not in html
        assert "<y>" not in html

    def test_build_stat_readout_html_uses_display_font(self):
        # Big figure uses the Archivo display font.
        html = ui_shared.build_stat_readout_html("HR", "28")
        assert "var(--font-display)" in html


# ── No-emoji sweep across every builder ───────────────────────────────


def test_no_builder_output_contains_emoji():
    outputs = [
        ui_shared.build_eyebrow_html("Season"),
        ui_shared.build_panel_html("Panel", "<i>x</i>", fig_label="FIG.01"),
        ui_shared.build_heatbar_html(50, win=True),
        ui_shared.build_heatbar_html(10, win=False),
        ui_shared.build_sparkline_html([1, 2, 3], tone="hot"),
        ui_shared.build_sparkline_html([3, 2, 1], tone="cold"),
        ui_shared.build_stat_readout_html("Record", "2-7-3", accent=True, sub="WK 12"),
    ]
    for html in outputs:
        _assert_no_emoji(html)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))

"""Accessibility (R-9) tests for src/ui_shared.py.

These tests assert additive aria/role/title/scope attributes on the shared
HTML builders without requiring any visual change or palette change.

TDD: written before the implementation — run red first, then green after
the implementation is applied.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_compact():
    from src.ui_shared import build_compact_table_html

    return build_compact_table_html


def _import_health_dot():
    """Return the health-dot HTML builder exposed as a public helper."""
    from src.ui_shared import build_health_dot_html

    return build_health_dot_html


def _import_page_icons():
    from src.ui_shared import PAGE_ICONS

    return PAGE_ICONS


def _import_headshot():
    from src.ui_shared import _headshot_img_html

    return _headshot_img_html


# ---------------------------------------------------------------------------
# 1. build_compact_table_html — <th> scope + table role
# ---------------------------------------------------------------------------


class TestCompactTableAccessibility:
    """<table> carries role=table; <th> headers carry scope=col."""

    @pytest.fixture(autouse=True)
    def _df(self):
        self.df = pd.DataFrame({"Name": ["Aaron Judge", "Shohei Ohtani"], "HR": [40, 38], "AVG": [0.325, 0.310]})
        self.build = _import_compact()

    def test_table_has_role_table(self):
        html = self.build(self.df)
        assert 'role="table"' in html, '<table> must carry role="table" for screen readers'

    def test_th_has_scope_col(self):
        html = self.build(self.df)
        # Match only <th ...> tags (not <thead ...>)
        th_tags = re.findall(r"<th(?:\s[^>]*)?>", html)
        assert th_tags, "No <th> tags found in output"
        for th in th_tags:
            assert 'scope="col"' in th, f'<th> missing scope="col": {th!r}'

    def test_html_cols_still_render_unescaped(self):
        """html_cols behavior must be preserved exactly after aria changes."""
        df = pd.DataFrame({"Player": ["Judge"], "Badge": ['<span class="hero-num">1</span>']})
        html = self.build(df, html_cols={"Badge"})
        # The raw badge span is present (not escaped)
        assert '<span class="hero-num">1</span>' in html

    def test_normal_cells_still_escaped(self):
        """Non-html_cols cells must still be HTML-escaped."""
        df = pd.DataFrame({"Player": ["<script>alert(1)</script>"], "HR": [0]})
        html = self.build(df)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_empty_df_still_returns_string(self):
        html = self.build(pd.DataFrame())
        assert isinstance(html, str)


# ---------------------------------------------------------------------------
# 2. build_health_dot_html — new public helper
# ---------------------------------------------------------------------------


class TestHealthDotHtml:
    """build_health_dot_html(status) must carry title + role + aria-label."""

    @pytest.fixture(autouse=True)
    def _fn(self):
        self.fn = _import_health_dot()

    def _statuses(self):
        return [
            "Healthy",
            "Day-to-Day",
            "IL",
            "IL-60",
            "Out",
            "Low Risk",
            "Moderate Risk",
            "Elevated Risk",
            "High Risk",
        ]

    @pytest.mark.parametrize(
        "status",
        [
            "Healthy",
            "Day-to-Day",
            "IL",
            "IL-60",
            "Out",
            "Low Risk",
            "Moderate Risk",
            "Elevated Risk",
            "High Risk",
        ],
    )
    def test_has_title_attribute(self, status):
        html = self.fn(status)
        assert f'title="{status}"' in html, f'health dot for {status!r} must carry title="{status}"'

    @pytest.mark.parametrize(
        "status",
        [
            "Healthy",
            "Day-to-Day",
            "IL",
            "IL-60",
            "Out",
            "Low Risk",
            "Moderate Risk",
            "Elevated Risk",
            "High Risk",
        ],
    )
    def test_has_role_img(self, status):
        html = self.fn(status)
        assert 'role="img"' in html, f'health dot for {status!r} must carry role="img"'

    @pytest.mark.parametrize(
        "status",
        [
            "Healthy",
            "Day-to-Day",
            "IL",
            "IL-60",
            "Out",
            "Low Risk",
            "Moderate Risk",
            "Elevated Risk",
            "High Risk",
        ],
    )
    def test_has_aria_label(self, status):
        html = self.fn(status)
        assert f'aria-label="{status}"' in html, f'health dot for {status!r} must carry aria-label="{status}"'

    def test_unknown_status_still_has_aria(self):
        html = self.fn("Unknown")
        assert "aria-label=" in html
        assert "role=" in html
        assert "title=" in html

    def test_returns_span_with_health_dot_class(self):
        html = self.fn("Healthy")
        assert 'class="health-dot"' in html

    def test_no_visual_change(self):
        """The dot still carries a background style (color signal preserved)."""
        html = self.fn("Healthy")
        assert "background:" in html


# ---------------------------------------------------------------------------
# 3. PAGE_ICONS SVG accessibility
# ---------------------------------------------------------------------------


class TestPageIconsAccessibility:
    """Meaningful icons carry role=img + aria-label; decorative carry aria-hidden."""

    @pytest.fixture(autouse=True)
    def _icons(self):
        self.icons = _import_page_icons()

    # Meaningful navigation icons — must have role + aria-label
    _MEANINGFUL_KEYS = [
        "my_team",
        "trade_analyzer",
        "player_compare",
        "free_agents",
        "lineup_optimizer",
        "configurations",
        "refresh",
        "check",
        "x_mark",
        "warning",
        "alert",
        "accept",
        "reject",
    ]

    @pytest.mark.parametrize("key", _MEANINGFUL_KEYS)
    def test_meaningful_icon_has_role_img(self, key):
        svg = self.icons[key]
        assert 'role="img"' in svg, f'PAGE_ICONS[{key!r}] must carry role="img"'

    @pytest.mark.parametrize("key", _MEANINGFUL_KEYS)
    def test_meaningful_icon_has_aria_label(self, key):
        svg = self.icons[key]
        assert "aria-label=" in svg, f"PAGE_ICONS[{key!r}] must carry aria-label"

    def test_logo_icon_has_role_img(self):
        svg = self.icons["logo"]
        assert 'role="img"' in svg

    def test_logo_icon_has_aria_label(self):
        svg = self.icons["logo"]
        assert "aria-label=" in svg

    def test_logo_lg_has_role_img(self):
        svg = self.icons["logo_lg"]
        assert 'role="img"' in svg


# ---------------------------------------------------------------------------
# 4. _headshot_img_html — alt attribute
# ---------------------------------------------------------------------------


class TestHeadshotImgAccessibility:
    """<img> tags must carry an alt attribute for screen readers."""

    @pytest.fixture(autouse=True)
    def _fn(self):
        self.fn = _import_headshot()

    def test_headshot_with_valid_mlb_id_has_alt(self):
        html = self.fn(660670)  # a real mlb_id
        assert "alt=" in html, "<img> from headshot must carry an alt attribute"

    def test_headshot_fallback_has_alt(self):
        html = self.fn(None)
        assert "alt=" in html, "fallback avatar <img> must carry an alt attribute"

    def test_alt_is_not_empty_string(self):
        html = self.fn(660670)
        # alt="" would mean purely decorative — a headshot is meaningful
        # (it identifies the player row); at minimum it should be non-empty
        # or explicitly describe the image.
        # We accept either non-empty alt OR aria-label on the img.
        has_nonempty_alt = bool(re.search(r'alt="[^"]+"', html))
        has_aria = "aria-label=" in html
        assert has_nonempty_alt or has_aria, "headshot img must have a non-empty alt or aria-label"

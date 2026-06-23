"""Guard for the matchup category-value sign-flip bug class (HEATER's FA-engine
history shows inverse-stat sign errors bite hard). ``_parse_cat_value`` MUST treat
the bare ``-``/empty as Yahoo's not-yet-played 0.0 while PRESERVING real negatives —
the old blanket ``str(x).replace("-", "0")`` turned ``"-5"`` into ``"05"`` = +5.0."""

import pytest

from api.services.matchup_service import _parse_cat_value


def test_bare_dash_is_zero():
    assert _parse_cat_value("-") == 0.0


def test_empty_is_zero():
    assert _parse_cat_value("") == 0.0


def test_real_negative_preserved_not_sign_flipped():
    # The old str.replace("-", "0") bug turned "-5" -> "05" = 5.0 (sign flip).
    assert _parse_cat_value("-5") == -5.0
    assert _parse_cat_value("-0.250") == pytest.approx(-0.250)


def test_positive_value_unchanged():
    assert _parse_cat_value("12") == 12.0
    assert _parse_cat_value(".284") == pytest.approx(0.284)


def test_garbage_is_zero_not_raise():
    # never-raise contract: unparseable -> 0.0
    assert _parse_cat_value("n/a") == 0.0
    assert _parse_cat_value(None) == 0.0

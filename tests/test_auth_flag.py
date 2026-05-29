"""multi_user_enabled() reads the MULTI_USER env var (default off)."""

import pytest

from src.auth import multi_user_enabled


@pytest.mark.parametrize("value", ["1", "true", "True", "yes", "on", "  1  "])
def test_enabled_truthy(monkeypatch, value):
    monkeypatch.setenv("MULTI_USER", value)
    assert multi_user_enabled() is True


@pytest.mark.parametrize("value", ["", "0", "false", "False", "no", "off"])
def test_disabled_falsey(monkeypatch, value):
    monkeypatch.setenv("MULTI_USER", value)
    assert multi_user_enabled() is False


def test_default_is_off(monkeypatch):
    monkeypatch.delenv("MULTI_USER", raising=False)
    assert multi_user_enabled() is False

"""src.version exposes a stable, env-overridable app version string."""

import importlib

import src.version


def test_app_version_is_nonempty_string():
    assert isinstance(src.version.APP_VERSION, str)
    assert src.version.APP_VERSION.strip()


def test_app_version_env_override(monkeypatch):
    monkeypatch.setenv("HEATER_APP_VERSION", "9.9.9-test")
    try:
        importlib.reload(src.version)
        assert src.version.APP_VERSION == "9.9.9-test"
    finally:
        # Restore the module's default so later tests see the real version.
        monkeypatch.delenv("HEATER_APP_VERSION", raising=False)
        importlib.reload(src.version)

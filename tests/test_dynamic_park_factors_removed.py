"""Test BUG-008 fix: _bootstrap_dynamic_park_factors is removed; not callable from bootstrap_all_data."""

import importlib


def test_dynamic_park_factors_function_removed():
    """The function _bootstrap_dynamic_park_factors should no longer exist."""
    import src.data_bootstrap as bootstrap_mod

    importlib.reload(bootstrap_mod)
    assert not hasattr(bootstrap_mod, "_bootstrap_dynamic_park_factors"), (
        "BUG-008: _bootstrap_dynamic_park_factors should be removed — it used "
        "team OPS+/wRC+ (a park-ADJUSTED metric) as a park factor proxy, "
        "silently corrupting correct Tier 1 / emergency factors every 7 days."
    )


def test_dynamic_park_factors_not_in_orchestrator_source_list():
    """bootstrap_all_data should not dispatch a 'dynamic_park_factors' or 'park_factors_dynamic' source."""
    from pathlib import Path

    src_text = Path("src/data_bootstrap.py").read_text(encoding="utf-8")
    forbidden = ["dynamic_park_factors", "park_factors_dynamic"]
    for name in forbidden:
        for lineno, line in enumerate(src_text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if f'"{name}"' in line:
                raise AssertionError(
                    f"BUG-008 regression: src/data_bootstrap.py:{lineno} references "
                    f'"{name}" — should have been removed entirely. Line: {line!r}'
                )

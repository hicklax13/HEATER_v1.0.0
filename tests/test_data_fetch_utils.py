"""Tests for the 3-tier fallback chain utility."""

import pytest


def test_fetch_with_fallback_tier1_success():
    from src.data_fetch_utils import fetch_with_fallback

    data, tier = fetch_with_fallback(
        "test_source",
        primary_fn=lambda: [1, 2, 3],
        fallback_fn=lambda: [4, 5],
        emergency_fn=lambda: [6],
    )
    assert data == [1, 2, 3]
    assert tier == "primary"


def test_fetch_with_fallback_tier1_fails_tier2_succeeds():
    from src.data_fetch_utils import fetch_with_fallback

    def fail():
        raise ConnectionError("API down")

    data, tier = fetch_with_fallback(
        "test_source",
        primary_fn=fail,
        fallback_fn=lambda: [4, 5],
        emergency_fn=lambda: [6],
    )
    assert data == [4, 5]
    assert tier == "fallback"


def test_fetch_with_fallback_tier1_empty_tier2_succeeds():
    from src.data_fetch_utils import fetch_with_fallback

    data, tier = fetch_with_fallback(
        "test_source",
        primary_fn=lambda: [],
        fallback_fn=lambda: [4, 5],
    )
    assert data == [4, 5]
    assert tier == "fallback"


def test_fetch_with_fallback_all_fail():
    from src.data_fetch_utils import fetch_with_fallback

    def fail():
        raise RuntimeError("down")

    data, tier = fetch_with_fallback(
        "test_source",
        primary_fn=fail,
        fallback_fn=fail,
        emergency_fn=fail,
    )
    assert data is None
    assert tier == "failed"


def test_fetch_with_fallback_none_returns_skip():
    from src.data_fetch_utils import fetch_with_fallback

    data, tier = fetch_with_fallback(
        "test_source",
        primary_fn=lambda: None,
        fallback_fn=None,
        emergency_fn=lambda: {"default": True},
    )
    assert data == {"default": True}
    assert tier == "emergency"


def test_fetch_with_fallback_dataframe():
    import pandas as pd

    from src.data_fetch_utils import fetch_with_fallback

    df = pd.DataFrame({"a": [1, 2]})
    data, tier = fetch_with_fallback(
        "test_source",
        primary_fn=lambda: df,
    )
    assert len(data) == 2
    assert tier == "primary"


def test_fetch_with_fallback_empty_dataframe_triggers_fallback():
    import pandas as pd

    from src.data_fetch_utils import fetch_with_fallback

    data, tier = fetch_with_fallback(
        "test_source",
        primary_fn=lambda: pd.DataFrame(),
        fallback_fn=lambda: pd.DataFrame({"a": [1]}),
    )
    assert len(data) == 1
    assert tier == "fallback"

"""Section 5 helper: position filter constants + pill-button helper.

The canonical position list and the pill-button widget are extracted from
inline implementations in Trade_Finder + Draft_Simulator (identical pattern)
and from Free_Agents (constant only). Leaders.py is NOT touched because
its prospect ordering [SS, OF, SP, 2B, 3B, 1B, C, RP] is intentionally
different.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_positions_constant_exists():
    """POSITIONS constant lives in ui_shared as the canonical roster order."""
    from src.ui_shared import POSITIONS

    assert POSITIONS == ["All", "C", "1B", "2B", "3B", "SS", "OF", "SP", "RP"]


def test_render_position_pills_returns_session_state_value():
    """When session has a stored value, that's returned."""
    from src.ui_shared import render_position_pills

    fake_state = {"my_filter": "OF"}
    with (
        patch("streamlit.session_state", new=fake_state),
        patch("streamlit.columns", return_value=[MagicMock() for _ in range(9)]),
        patch("streamlit.button", return_value=False),
    ):
        result = render_position_pills(key_prefix="my_pill", session_key="my_filter")
    assert result == "OF"


def test_render_position_pills_default_when_no_session_value():
    """If session_state has no key, returns the default."""
    from src.ui_shared import render_position_pills

    fake_state: dict = {}
    with (
        patch("streamlit.session_state", new=fake_state),
        patch("streamlit.columns", return_value=[MagicMock() for _ in range(9)]),
        patch("streamlit.button", return_value=False),
    ):
        result = render_position_pills(key_prefix="x", session_key="x_filter", default="SP")
    assert result == "SP"


def test_render_position_pills_creates_one_column_per_position():
    """The helper creates 9 columns (one per position including 'All')."""
    from src.ui_shared import POSITIONS, render_position_pills

    fake_state: dict = {}
    cols_mock = MagicMock(return_value=[MagicMock() for _ in range(len(POSITIONS))])
    with (
        patch("streamlit.session_state", new=fake_state),
        patch("streamlit.columns", cols_mock),
        patch("streamlit.button", return_value=False),
    ):
        render_position_pills(key_prefix="y", session_key="y_filter")
    cols_mock.assert_called_once_with(len(POSITIONS))


def test_render_position_pills_button_click_updates_session_and_reruns():
    """Clicking a button updates session_state and triggers st.rerun()."""
    from src.ui_shared import POSITIONS, render_position_pills

    fake_state: dict = {}
    # SS button (index 5) returns True; others False.
    click_pattern = [pos == "SS" for pos in POSITIONS]

    def _btn_side_effect(*args, **kwargs):
        return click_pattern.pop(0)

    rerun_mock = MagicMock()
    with (
        patch("streamlit.session_state", new=fake_state),
        patch("streamlit.columns", return_value=[MagicMock() for _ in range(len(POSITIONS))]),
        patch("streamlit.button", side_effect=_btn_side_effect),
        patch("streamlit.rerun", rerun_mock),
    ):
        # In production, st.rerun() raises to halt the script. With a mocked
        # rerun, execution continues — that's fine; we just check side effects.
        render_position_pills(key_prefix="z", session_key="z_filter")
    assert fake_state.get("z_filter") == "SS"
    rerun_mock.assert_called_once()

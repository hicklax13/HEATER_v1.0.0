"""render_feedback_widget is a no-op unless MULTI_USER is on AND a user exists."""

import pytest

import src.feedback as feedback


def test_widget_noop_when_multiuser_off(monkeypatch):
    monkeypatch.setattr(feedback, "multi_user_enabled", lambda: False)
    monkeypatch.setattr(feedback, "current_user", lambda: {"user_id": 1})
    # If the flag is off, the popover must never be reached.
    monkeypatch.setattr(feedback.st, "popover", lambda *a, **k: pytest.fail("popover called"))
    feedback.render_feedback_widget("My Team")  # must simply return


def test_widget_noop_when_no_user(monkeypatch):
    monkeypatch.setattr(feedback, "multi_user_enabled", lambda: True)
    monkeypatch.setattr(feedback, "current_user", lambda: None)
    monkeypatch.setattr(feedback.st, "popover", lambda *a, **k: pytest.fail("popover called"))
    feedback.render_feedback_widget("My Team")  # must simply return

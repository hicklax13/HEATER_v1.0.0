"""Member write-safety guards (2026-06-01 pre-launch audit).

Under MULTI_USER the background scheduler is the SOLE SQLite writer and member
sessions are read-only. These guard:
  - viewer_can_write(): who may trigger refresh/sync buttons (single-user v1, or
    admins under MULTI_USER) — the launch-blocker fix.
  - bump_activity(): the 60s heartbeat must never crash a member's page if the
    DB is momentarily locked.
  - enter_view_as(): the view-as preview must refuse admin targets (avoids audit
    mis-attribution).
"""

import src.auth as auth
import src.usage as usage

# ── viewer_can_write ──────────────────────────────────────────────────


def test_viewer_can_write_single_user_true(monkeypatch):
    monkeypatch.setattr(auth, "multi_user_enabled", lambda: False)
    assert auth.viewer_can_write() is True


def test_viewer_can_write_multiuser_admin_true(monkeypatch):
    monkeypatch.setattr(auth, "multi_user_enabled", lambda: True)
    monkeypatch.setattr(auth, "current_user", lambda: {"username": "a", "is_admin": 1})
    assert auth.viewer_can_write() is True


def test_viewer_can_write_multiuser_member_false(monkeypatch):
    monkeypatch.setattr(auth, "multi_user_enabled", lambda: True)
    monkeypatch.setattr(auth, "current_user", lambda: {"username": "m", "is_admin": 0})
    assert auth.viewer_can_write() is False


def test_viewer_can_write_multiuser_no_session_false(monkeypatch):
    monkeypatch.setattr(auth, "multi_user_enabled", lambda: True)
    monkeypatch.setattr(auth, "current_user", lambda: None)
    assert auth.viewer_can_write() is False


# ── bump_activity crash-safety ────────────────────────────────────────


def test_bump_activity_swallows_db_errors(monkeypatch):
    monkeypatch.setattr(usage, "multi_user_enabled", lambda: True)
    monkeypatch.setattr(usage, "current_user", lambda: {"user_id": 1, "username": "m"})
    monkeypatch.setattr(usage, "_session_state", lambda: {"_usage_session_id": "sess-1"})

    class _BoomConn:
        def execute(self, *a, **k):
            raise RuntimeError("database is locked")

        def commit(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr("src.database.get_connection", lambda: _BoomConn())
    # Must NOT raise — a locked DB on the heartbeat tick can't crash the page.
    usage.bump_activity()


# ── view-as refuses admin targets ─────────────────────────────────────


def test_enter_view_as_refuses_admin_target(monkeypatch):
    monkeypatch.setattr(auth, "multi_user_enabled", lambda: True)
    real_admin = {"username": "admin1", "is_admin": 1, "user_id": 1}
    state = {auth._SESSION_KEY: real_admin}
    monkeypatch.setattr(auth, "_session_state", lambda: state)
    monkeypatch.setattr(auth, "get_user", lambda u: {"username": u, "is_admin": 1, "user_id": 2})
    monkeypatch.setattr("src.audit.log_action", lambda *a, **k: None)

    auth.enter_view_as("admin2", admin_id=1)

    # Refused: session identity unchanged, no view-as stash created.
    assert state[auth._SESSION_KEY]["username"] == "admin1"
    assert auth._VIEW_AS_KEY not in state

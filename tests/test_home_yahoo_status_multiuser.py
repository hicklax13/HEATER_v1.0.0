"""BR-5 (2026-06-07): the draft/Home quick-stats card showed "Yahoo Not
Connected" + "Player Pool Loading..." under MULTI_USER even though the
scheduler keeps data live and read-only members never hold a Yahoo client.

The card derived its labels from per-session state (``yahoo_connected`` and the
``bootstrap_complete`` guard), both of which are inert for read-only members.
``app._home_quick_stats_labels`` is a pure helper that maps the inputs to the
correct display labels; under MULTI_USER it uses the scheduler-served
connection status instead of the per-session client flag.

v1 (flag off) must be byte-for-byte unchanged.
"""

from __future__ import annotations


def test_multiuser_server_status_shows_live_via_server():
    from app import _home_quick_stats_labels

    pool_label, conn_text = _home_quick_stats_labels(
        multi_user=True,
        bootstrap_done=False,  # members never run the per-session bootstrap
        pool_count=4200,
        yahoo_connected=False,  # members hold no live client
        conn_status="server",  # scheduler is serving fresh data
    )
    assert pool_label == "4,200", f"pool count should render even without bootstrap_complete, got {pool_label!r}"
    assert "Not Connected" not in conn_text
    assert "Live" in conn_text and "server" in conn_text.lower(), (
        f"MULTI_USER + server status should read 'Live (via server)', got {conn_text!r}"
    )


def test_multiuser_warming_status_shows_warming():
    from app import _home_quick_stats_labels

    pool_label, conn_text = _home_quick_stats_labels(
        multi_user=True,
        bootstrap_done=False,
        pool_count=0,
        yahoo_connected=False,
        conn_status="warming",
    )
    assert pool_label == "Loading..."
    assert "Not Connected" not in conn_text
    assert "arm" in conn_text.lower() or "Warming" in conn_text, f"warming should be surfaced, got {conn_text!r}"


def test_single_user_connected_unchanged():
    """v1: flag off → connected client reads 'Yahoo Connected' exactly."""
    from app import _home_quick_stats_labels

    pool_label, conn_text = _home_quick_stats_labels(
        multi_user=False,
        bootstrap_done=True,
        pool_count=4200,
        yahoo_connected=True,
        conn_status="connected",
    )
    assert pool_label == "4,200"
    assert conn_text == "Yahoo Connected"


def test_single_user_not_connected_unchanged():
    """v1: flag off + no client → 'Yahoo Not Connected' (byte-for-byte v1)."""
    from app import _home_quick_stats_labels

    pool_label, conn_text = _home_quick_stats_labels(
        multi_user=False,
        bootstrap_done=False,
        pool_count=0,
        yahoo_connected=False,
        conn_status="offline",
    )
    assert pool_label == "Loading..."
    assert conn_text == "Yahoo Not Connected"

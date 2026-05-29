"""The conftest network guard must block real outbound sockets.

This is the load-bearing fix for the Windows full-suite hang: pytest-timeout's
`thread` method cannot interrupt a blocking C-level ``socket.recv()``, so a
single unmocked live fetch wedges its xdist worker forever (~98%, "node down")
and the master never exits. The guard (tests/conftest.py) blocks any connect()
to a non-loopback host, turning an infinite hang into an instant exception that
HEATER's 3-tier fetch fallbacks already handle.
"""

import socket

import pytest

# RFC 5737 TEST-NET-1 — guaranteed non-routable, so the guard's host check
# fires BEFORE any real packet is sent. No network is touched by this test.
_TEST_NET_HOST = "192.0.2.1"


def test_guard_active_by_default():
    """With no opt-in marker, socket.connect is the guarded wrapper."""
    assert socket.socket.connect.__name__ == "_guarded_connect"
    assert socket.socket.connect_ex.__name__ == "_guarded_connect_ex"


def test_external_connect_is_blocked():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(RuntimeError, match="HEATER-TEST-NETWORK-BLOCKED"):
            s.connect((_TEST_NET_HOST, 80))
    finally:
        s.close()


def test_external_connect_ex_is_blocked():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(RuntimeError, match="HEATER-TEST-NETWORK-BLOCKED"):
            s.connect_ex((_TEST_NET_HOST, 80))
    finally:
        s.close()


def test_loopback_passes_the_guard():
    """Loopback is allowed through the guard — a refused connect surfaces as a
    normal OSError (ConnectionRefused/timeout), never NetworkBlockedError."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.25)
    try:
        with pytest.raises(OSError) as exc:
            s.connect(("127.0.0.1", 1))  # port 1: nothing listening → refused
        assert "HEATER-TEST-NETWORK-BLOCKED" not in str(exc.value)
    finally:
        s.close()


@pytest.mark.allow_network
def test_marker_lifts_guard_within_test():
    """@pytest.mark.allow_network restores the real connect for the test."""
    assert socket.socket.connect.__name__ != "_guarded_connect"

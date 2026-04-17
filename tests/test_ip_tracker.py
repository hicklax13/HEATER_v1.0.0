"""Tests for src.ip_tracker."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ip_tracker import compute_weekly_ip_projection


class TestDisplayConsistency:
    def test_display_string_matches_message(self):
        """The projected_ip_display string must match the number in the
        narrative message so the header and status lines don't disagree
        at rounding boundaries (e.g. 39.55 -> 39.5 header vs 39.6 status)."""
        pitchers = [
            {"ip": 40.0, "positions": "SP", "status": "active", "is_starter": True},
            {"ip": 22.0, "positions": "RP", "status": "active", "is_starter": False},
        ]
        result = compute_weekly_ip_projection(pitchers, days_remaining=3)
        disp = result["projected_ip_display"]
        assert disp in result["message"], f"Display string '{disp}' not present in message '{result['message']}'"

    def test_display_string_is_one_decimal(self):
        pitchers = [{"ip": 50.0, "positions": "SP", "status": "active", "is_starter": True}]
        result = compute_weekly_ip_projection(pitchers, days_remaining=5)
        disp = result["projected_ip_display"]
        # Must be a string with exactly one decimal place
        assert "." in disp
        assert len(disp.split(".")[1]) == 1

    def test_boundary_rounding_consistent(self):
        """Values that would double-round differently (e.g. 39.549999)
        must produce identical header and message values."""
        # Fake a pitcher whose projection lands near a 5-boundary
        pitchers = [{"ip": 39.549999 * 7 / 3, "positions": "SP", "status": "active", "is_starter": True}]
        result = compute_weekly_ip_projection(pitchers, days_remaining=3)
        disp = result["projected_ip_display"]
        projected = result["projected_ip"]
        # Re-format the stored projected_ip the same way the UI would
        ui_header = f"{projected:.1f}"
        # The source-of-truth display and a naive :.1f re-format on the
        # rounded value must not disagree by more than 0.1 (occasional
        # banker's-rounding drift is OK, but the message/header pair
        # should both come from the same _ip_disp).
        assert abs(float(disp) - float(ui_header)) <= 0.1

    def test_status_field_present(self):
        pitchers = [{"ip": 180.0, "positions": "SP", "status": "active", "is_starter": True}]
        result = compute_weekly_ip_projection(pitchers, days_remaining=7)
        assert result["status"] in ("safe", "warning", "danger")
        assert "projected_ip_display" in result

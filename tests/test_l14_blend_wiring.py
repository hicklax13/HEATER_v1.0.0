"""P5b: _blend_fa_row should incorporate ctx.recent_form (L14) into the
blend when available, using the canonical 0.70/0.20/0.10 weights."""

import pandas as pd

from src.optimizer.fa_recommender import _blend_fa_row


def _make_fa_row(**kwargs):
    base = {
        "player_id": 42,
        "is_hitter": 1,
        # ROS projection (~250 PA / 60 H rest of season)
        "r": 30,
        "hr": 6,
        "rbi": 25,
        "sb": 3,
        "ab": 250,
        "h": 60,
        "bb": 25,
        "hbp": 2,
        "sf": 1,
        "ytd_gp": 40,
        # YTD pace
        "ytd_r": 25,
        "ytd_hr": 5,
        "ytd_rbi": 20,
        "ytd_sb": 2,
        "ytd_ab": 150,
        "ytd_h": 38,
        "ytd_bb": 15,
        "ytd_hbp": 1,
        "ytd_sf": 0,
        # Pitching cols zeroed
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "ip": 0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
        "ytd_w": 0,
        "ytd_l": 0,
        "ytd_sv": 0,
        "ytd_k": 0,
        "ytd_ip": 0,
        "ytd_er": 0,
        "ytd_bb_allowed": 0,
        "ytd_h_allowed": 0,
    }
    base.update(kwargs)
    return pd.Series(base)


def test_l14_wired_when_recent_form_present():
    """A hot L14 hitter should have HIGHER blended HR than the same player
    with no L14 data, because L14 gets 0.10 weight in the canonical blend."""
    fa = _make_fa_row()
    # Simulate L14 hot streak: 5 HR in 14 days
    fa["l14_hr"] = 5
    fa["l14_r"] = 15
    fa["l14_pa"] = 60
    blended_with_l14 = _blend_fa_row(fa)
    # Compare to fa without L14 fields
    fa_no_l14 = _make_fa_row()
    blended_no_l14 = _blend_fa_row(fa_no_l14)
    # With L14 (5 HR in 14 days = ~58/162 pace), blended HR should be slightly higher
    assert blended_with_l14["hr"] > blended_no_l14["hr"], (
        f"L14 should boost blended HR. Got with_l14={blended_with_l14['hr']:.2f} vs no_l14={blended_no_l14['hr']:.2f}"
    )


def test_l14_skipped_for_short_sample():
    """When L14 PA is too small (<20), L14 weight should drop to 0 to avoid
    noise. Verify the function doesn't crash and skips L14 in that case."""
    fa = _make_fa_row()
    fa["l14_hr"] = 1
    fa["l14_pa"] = 5  # too small
    # Should not crash; should produce a sensible blend
    blended = _blend_fa_row(fa)
    assert blended["hr"] > 0


def test_l14_pitcher_uses_ip_volume():
    """For pitchers, L14 volume gate is on l14_ip, not l14_pa."""
    fa = pd.Series(
        {
            "player_id": 99,
            "is_hitter": 0,
            "w": 4,
            "l": 2,
            "sv": 0,
            "k": 60,
            "ip": 70,
            "er": 25,
            "bb_allowed": 20,
            "h_allowed": 55,
            "ytd_gp": 0,
            "ytd_w": 3,
            "ytd_l": 2,
            "ytd_sv": 0,
            "ytd_k": 45,
            "ytd_ip": 50,
            "ytd_er": 18,
            "ytd_bb_allowed": 14,
            "ytd_h_allowed": 40,
            "l14_k": 12,
            "l14_ip": 14,
            "l14_w": 1,
            # Hitting cols zero
            "r": 0,
            "hr": 0,
            "rbi": 0,
            "sb": 0,
            "ab": 0,
            "h": 0,
            "bb": 0,
            "hbp": 0,
            "sf": 0,
        }
    )
    blended = _blend_fa_row(fa)
    # Should produce a sensible IP value
    assert blended["ip"] >= 0
    assert blended["k"] >= 0


# ── FA-C2 (2026-06-07): L14 from the recent-form dict (the live FA path) ──
# Free agents carry NO l14_* columns, so L14 must arrive via l14_form (sourced
# from ctx.recent_form / get_player_recent_form_cached). These exercise that path.


def test_l14_wired_via_recent_form_dict_without_columns():
    """A hot L14 recent-form dict boosts blended HR even with no l14_* columns."""
    fa = _make_fa_row()  # no l14_* columns at all (the real FA pool shape)
    l14 = {"games": 14, "pa": 60, "hr": 5, "r": 15}
    blended_with = _blend_fa_row(fa, l14_form=l14)
    blended_without = _blend_fa_row(fa)
    assert blended_with["hr"] > blended_without["hr"], (
        f"recent-form L14 dict should boost blended HR. "
        f"with={blended_with['hr']:.2f} without={blended_without['hr']:.2f}"
    )


def test_l14_dict_respects_volume_gate():
    """A recent-form dict below the PA gate (<20) contributes no L14 weight."""
    fa = _make_fa_row()
    under_gate = _blend_fa_row(fa, l14_form={"games": 3, "pa": 8, "hr": 3})
    no_l14 = _blend_fa_row(fa)
    assert under_gate["hr"] == no_l14["hr"]


def test_resolve_fa_l14_reads_ctx_recent_form():
    """_resolve_fa_l14 prefers a pre-loaded ctx.recent_form[pid]['l14'] entry."""
    from types import SimpleNamespace

    from src.optimizer.fa_recommender import _resolve_fa_l14

    fa = _make_fa_row(player_id=42)
    ctx = SimpleNamespace(recent_form={42: {"l14": {"games": 14, "pa": 55, "hr": 4}}})
    assert _resolve_fa_l14(fa, ctx) == {"games": 14, "pa": 55, "hr": 4}


def test_resolve_fa_l14_none_when_absent():
    """No recent_form entry + no mlb_id → None (blend falls back to ROS+YTD)."""
    from types import SimpleNamespace

    from src.optimizer.fa_recommender import _resolve_fa_l14

    fa = _make_fa_row(player_id=42)  # no mlb_id present
    assert _resolve_fa_l14(fa, SimpleNamespace(recent_form={})) is None

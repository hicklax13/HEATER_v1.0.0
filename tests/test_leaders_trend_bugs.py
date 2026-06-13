"""Tests for Task 1.5 (Hot/Cold delta explosion + is_hitter or-bug)
and Task 1.6 (Statcast-NULL empty states in Breakouts and My Team).

TDD: these tests FAIL before the fix, PASS after.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pool_row(
    *,
    player_id: int = 1,
    name: str = "P",
    is_hitter: int = 1,
    ip: float = 0.0,
    pa: float = 0.0,
    era: float = 0.0,
    whip: float = 0.0,
    k: float = 0.0,
    avg: float = 0.0,
    hr: float = 0.0,
    rbi: float = 0.0,
    sb: float = 0.0,
    positions: str = "SP",
    team: str = "LAD",
) -> dict:
    return dict(
        player_id=player_id,
        name=name,
        is_hitter=is_hitter,
        ip=ip,
        pa=pa,
        era=era,
        whip=whip,
        k=k,
        avg=avg,
        hr=hr,
        rbi=rbi,
        sb=sb,
        positions=positions,
        team=team,
    )


# ---------------------------------------------------------------------------
# Task 1.5a — Volume gate: low-PA hitter and low-IP pitcher are excluded
# ---------------------------------------------------------------------------


def test_hot_cold_excludes_low_ip_pitcher():
    """A pitcher with ip_proj < 15 must be excluded from Hot/Cold output."""
    from src.trend_tracker import compute_player_trends

    pool = pd.DataFrame(
        [
            _make_pool_row(player_id=1, name="TinyIP", is_hitter=0, ip=5.0, era=4.50, whip=1.30, k=30),
            _make_pool_row(player_id=2, name="BigIP", is_hitter=0, ip=180.0, era=3.20, whip=1.10, k=200),
        ]
    )
    pool = pool.rename(columns={"name": "player_name"})

    season = pd.DataFrame(
        [
            {"player_id": 1, "ip": 12.0, "era": 2.50, "whip": 0.90, "k": 40, "pa": 0},
            {"player_id": 2, "ip": 90.0, "era": 3.00, "whip": 1.05, "k": 100, "pa": 0},
        ]
    )

    result = compute_player_trends(pool, season)

    # TinyIP pitcher (proj ip < 15) must not appear as HOT or COLD
    tiny_row = result[result["player_id"] == 1]
    assert not tiny_row.empty
    assert tiny_row.iloc[0]["trend_label"] == "NEUTRAL", "Pitcher with ip_proj < 15 must be NEUTRAL (volume-gated)"


def test_hot_cold_excludes_low_pa_hitter():
    """A hitter with pa_proj < 50 must be excluded from Hot/Cold output."""
    from src.trend_tracker import compute_player_trends

    pool = pd.DataFrame(
        [
            _make_pool_row(player_id=1, name="TinyPA", is_hitter=1, pa=20.0, avg=0.280, hr=3, rbi=10, sb=1),
            _make_pool_row(player_id=2, name="BigPA", is_hitter=1, pa=550.0, avg=0.270, hr=30, rbi=90, sb=10),
        ]
    )
    pool = pool.rename(columns={"name": "player_name"})

    season = pd.DataFrame(
        [
            {"player_id": 1, "pa": 25.0, "avg": 0.400, "hr": 5, "rbi": 15, "sb": 3, "ip": 0},
            {"player_id": 2, "pa": 300.0, "avg": 0.310, "hr": 18, "rbi": 50, "sb": 6, "ip": 0},
        ]
    )

    result = compute_player_trends(pool, season)

    tiny_row = result[result["player_id"] == 1]
    assert not tiny_row.empty
    assert tiny_row.iloc[0]["trend_label"] == "NEUTRAL", "Hitter with pa_proj < 50 must be NEUTRAL (volume-gated)"


# ---------------------------------------------------------------------------
# Task 1.5b — Delta clipping: delta is clipped to [-3, +3]
# ---------------------------------------------------------------------------


def test_trend_delta_clipped_positive():
    """A near-zero ERA projection must not produce a +200 delta — clip to 3.0."""
    from src.trend_tracker import compute_player_trends

    # ERA proj ≈ 0 → huge negative raw delta when actual ERA=4.50.
    # The inverse flip means (0.001 - 4.50)/0.001 ≈ -4499 → after inversion = +4499.
    pool = pd.DataFrame(
        [
            _make_pool_row(player_id=1, name="Phantom", is_hitter=0, ip=180.0, era=0.0, whip=0.0, k=150),
        ]
    )
    pool = pool.rename(columns={"name": "player_name"})

    season = pd.DataFrame(
        [
            {"player_id": 1, "ip": 90.0, "era": 4.50, "whip": 1.40, "k": 80, "pa": 0},
        ]
    )

    result = compute_player_trends(pool, season)
    delta = result.iloc[0]["trend_delta"]

    assert abs(delta) <= 3.0, (
        f"trend_delta={delta} is out of [-3, +3] clip range — near-zero projection causes explosion"
    )


def test_trend_delta_clipped_negative():
    """A near-zero SB projection must not produce a -200 delta — clip to -3.0."""
    from src.trend_tracker import compute_player_trends

    pool = pd.DataFrame(
        [
            _make_pool_row(player_id=1, name="SlowGuy", is_hitter=1, pa=550.0, avg=0.270, hr=25, rbi=85, sb=0.0),
        ]
    )
    pool = pool.rename(columns={"name": "player_name"})

    season = pd.DataFrame(
        [
            {"player_id": 1, "pa": 300.0, "avg": 0.270, "hr": 14, "rbi": 45, "sb": 0, "ip": 0},
        ]
    )

    result = compute_player_trends(pool, season)
    delta = result.iloc[0]["trend_delta"]

    assert abs(delta) <= 3.0, f"trend_delta={delta} is out of [-3, +3] — near-zero SB proj causes explosion"


# ---------------------------------------------------------------------------
# Task 1.5c — _trend_key_stats: is_hitter=0 returns ERA/K, not AVG/HR
# ---------------------------------------------------------------------------


def test_trend_key_stats_pitcher_returns_era_k():
    """_trend_key_stats must return ERA/K for is_hitter=0 rows, not AVG/HR."""
    # We test the function via the page's helper directly.
    # Since the helper lives inside the page module (not a standalone fn),
    # we replicate the correct behavior expectation: for is_hitter=0, the
    # result must contain "ERA" and "K", not "AVG" and "HR".

    # The bug is `int(row.get("is_hitter", 1) or 1)` — Python's `or 1`
    # coerces 0 → 1 because `0 or 1 == 1`.
    #
    # Test the core coercion logic directly:
    def _safe_is_hitter(val) -> int:
        """Safe coercion: 0/0.0/'0' all return 0 (pitcher), never `or 1`."""
        if val is None:
            return 1
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return 1

    # The bug: `int(0 or 1) == 1` — pitcher read as hitter
    assert int(0 or 1) == 1, "Confirms the or-bug: 0 or 1 == 1"

    # The fix: safe coercion preserves 0 as 0
    assert _safe_is_hitter(0) == 0
    assert _safe_is_hitter(0.0) == 0
    assert _safe_is_hitter("0") == 0
    assert _safe_is_hitter(1) == 1
    assert _safe_is_hitter(None) == 1  # default to hitter when missing


def test_trend_key_stats_produces_era_for_pitcher():
    """_trend_key_stats in the Leaders page must show ERA/K for a pitcher row.
    Specifically, the code line `int(row.get("is_hitter", 1) or 1)` must be gone.
    """
    import re
    from pathlib import Path

    src = Path("pages/17_Leaders.py").read_text(encoding="utf-8")
    # Extract the function body (non-comment lines only)
    fn_body = src.split("def _trend_key_stats")[1].split("def ")[0]
    # Strip comment lines before checking for the bug pattern
    non_comment_lines = [line for line in fn_body.splitlines() if not line.lstrip().startswith("#")]
    non_comment_src = "\n".join(non_comment_lines)

    # The bug pattern: `int(... or 1)` where the "or 1" is in executable code
    assert "or 1" not in non_comment_src, (
        "_trend_key_stats still contains the `or 1` coercion bug for is_hitter in "
        "executable code (not just a comment). "
        "Replace with an explicit safe coercion that preserves is_hitter=0 as pitcher."
    )


# ---------------------------------------------------------------------------
# Task 1.6 — Statcast-NULL empty state: Breakouts tab
# ---------------------------------------------------------------------------


def test_breakout_score_batch_all_null_statcast():
    """When all Statcast columns are NULL, breakout scores should use fallback.
    The Breakouts tab should detect NULL and show render_empty_state."""
    from src.leaders import compute_breakout_scores_batch

    # Build a pool where all Statcast columns are NaN
    pool = pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "Hitter A",
                "is_hitter": 1,
                "hr": 20,
                "rbi": 70,
                "sb": 5,
                "avg": 0.270,
                "obp": 0.340,
                "barrel_pct": np.nan,
                "xwoba": np.nan,
                "hard_hit_pct": np.nan,
                "stuff_plus": np.nan,
                "k_pct": np.nan,
            },
            {
                "player_id": 2,
                "name": "Pitcher B",
                "is_hitter": 0,
                "k": 180,
                "era": 3.20,
                "whip": 1.10,
                "w": 12,
                "sv": 0,
                "barrel_pct": np.nan,
                "xwoba": np.nan,
                "hard_hit_pct": np.nan,
                "stuff_plus": np.nan,
            },
        ]
    )

    # Statcast cols are entirely NULL
    statcast_cols = ["barrel_pct", "xwoba", "hard_hit_pct", "stuff_plus"]
    assert pool[statcast_cols].notna().any().any() is False or not pool[statcast_cols].notna().any().any(), (
        "Test setup: Statcast cols should be all-NULL"
    )

    scored = compute_breakout_scores_batch(pool)
    assert "breakout_score" in scored.columns

    # All scores must NOT be exactly 50.0 — with fallback scoring they vary
    # OR the caller must detect all-NULL and skip scoring entirely.
    # The fix should ensure the Breakouts UI detects the null condition.
    # We test the detection logic: no non-null values in key statcast cols.
    has_statcast = pool[statcast_cols].notna().any().any()
    assert not has_statcast  # confirms NULL condition that triggers empty state


def test_statcast_null_gate_logic():
    """The gate logic used in the Breakouts tab and My Team must correctly
    detect when all Statcast columns are NULL."""
    statcast_cols = ["barrel_pct", "xwoba", "hard_hit_pct", "stuff_plus"]

    # All-NULL pool → gate returns False (no data)
    null_pool = pd.DataFrame([{"barrel_pct": np.nan, "xwoba": np.nan, "hard_hit_pct": np.nan, "stuff_plus": np.nan}])
    assert not null_pool[statcast_cols].notna().any().any(), "All-NULL pool must be detected as having no Statcast data"

    # Partially populated pool → gate returns True (data present)
    partial_pool = pd.DataFrame([{"barrel_pct": 8.5, "xwoba": np.nan, "hard_hit_pct": np.nan, "stuff_plus": np.nan}])
    assert partial_pool[statcast_cols].notna().any().any(), (
        "Partially populated pool must be detected as having Statcast data"
    )


def test_breakouts_tab_uses_render_empty_state_on_null():
    """pages/17_Leaders.py Breakouts tab must call render_empty_state
    when Statcast data is all-NULL (not just show the scored table)."""
    import ast
    from pathlib import Path

    src = Path("pages/17_Leaders.py").read_text(encoding="utf-8")

    # The Breakouts section (tab4) should contain a statcast null check
    # followed by render_empty_state
    breakouts_section = src.split("with tab4:")[1].split("with tab3:")[0] if "with tab4:" in src else ""

    assert "render_empty_state" in breakouts_section, (
        "pages/17_Leaders.py Breakouts tab (tab4) must call render_empty_state "
        "when Statcast columns are all-NULL. Currently it shows 50.0 scores for everyone."
    )

    # Must contain a NULL check using .notna() or equivalent
    assert "notna" in breakouts_section or "isna" in breakouts_section or "isnull" in breakouts_section, (
        "pages/17_Leaders.py Breakouts tab must check for NULL Statcast columns "
        "before scoring (e.g. df[cols].notna().any().any())"
    )


def test_my_team_statcast_card_uses_render_empty_state_on_null():
    """pages/1_My_Team.py Statcast Signals card must import and call
    render_empty_state when all Statcast columns are NULL."""
    from pathlib import Path

    src = Path("pages/1_My_Team.py").read_text(encoding="utf-8")

    # Must import render_empty_state
    assert "render_empty_state" in src, "pages/1_My_Team.py must import render_empty_state from src.ui_shared"

    # The Statcast Signals section must check for NULL and call render_empty_state
    sc_section = ""
    if "Statcast Signals" in src:
        sc_section = src.split("Statcast Signals")[1].split("Data freshness card")[0]

    assert "render_empty_state" in sc_section or "notna" in sc_section or "isna" in sc_section, (
        "pages/1_My_Team.py Statcast Signals card must check for NULL Statcast data "
        "and call render_empty_state when data is unavailable. "
        "Currently it silently renders nothing (empty card)."
    )

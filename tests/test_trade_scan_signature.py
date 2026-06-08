"""#10 (2026-06-07): Trade Finder result caching.

The page reran the full ~16-75s scan on EVERY widget interaction. `trade_scan_signature`
yields a stable, order-independent key over the scan inputs so the page can memoize
results in session_state and only recompute when the inputs (rosters/totals/flags)
actually change — e.g. after a data refresh.
"""

from src.trade_finder import trade_scan_signature


def _inputs():
    return dict(
        user_team_name="Team Hickey",
        user_roster_ids=[3, 1, 2],
        league_rosters={"Team Hickey": [1, 2, 3], "Rivals": [4, 5, 6]},
        all_team_totals={"Team Hickey": {"R": 100.0, "HR": 40.0}, "Rivals": {"R": 90.0, "HR": 50.0}},
        top_partners=11,
        max_results=50,
        title_odds_enabled=False,
    )


def test_same_inputs_same_signature():
    assert trade_scan_signature(**_inputs()) == trade_scan_signature(**_inputs())


def test_signature_is_order_independent():
    a = _inputs()
    b = _inputs()
    # Reorder dict keys + list order — must not change the signature.
    b["league_rosters"] = {"Rivals": [6, 5, 4], "Team Hickey": [3, 2, 1]}
    b["user_roster_ids"] = [2, 3, 1]
    assert trade_scan_signature(**a) == trade_scan_signature(**b)


def test_roster_change_changes_signature():
    a = _inputs()
    b = _inputs()
    b["league_rosters"]["Rivals"] = [4, 5, 7]  # swapped a player
    assert trade_scan_signature(**a) != trade_scan_signature(**b)


def test_totals_change_changes_signature():
    a = _inputs()
    b = _inputs()
    b["all_team_totals"]["Rivals"]["HR"] = 51.0  # data refreshed
    assert trade_scan_signature(**a) != trade_scan_signature(**b)


def test_flag_change_changes_signature():
    a = _inputs()
    b = _inputs()
    b["title_odds_enabled"] = True
    assert trade_scan_signature(**a) != trade_scan_signature(**b)


def test_signature_is_hashable():
    sig = trade_scan_signature(**_inputs())
    assert hash(sig) is not None  # usable as a dict/session_state key


def test_trade_finder_page_memoizes_scan():
    """Structural guard: the page must cache the scan in session_state keyed on
    trade_scan_signature, so it doesn't re-run the 16-75s scan on every rerun."""
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "pages" / "12_Trade_Finder.py").read_text(encoding="utf-8")
    assert "trade_scan_signature" in src, "page must compute a scan signature"
    assert "_tf_scan_cache" in src, "page must memoize results in session_state"
    # The scan must be written into the cache (proves the miss-branch stores it).
    assert 'st.session_state["_tf_scan_cache"]' in src, "page must store scan results in the cache"

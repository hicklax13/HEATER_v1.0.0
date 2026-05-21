"""P5a: when 5 FAs all want to drop the same player, the engine should
surface multiple distinct recommendations by pairing each FA with its
OWN best valid drop, not just collapse to 1 rec via dedup."""

from src.optimizer.fa_recommender import _deduplicate_and_limit


def test_dedup_returns_multiple_when_diverse_pairs_available():
    """Given 5 swaps where 3 share the same drop and 2 use different drops,
    the result should include >= 3 distinct (add, drop) combos when max=5."""
    swaps = [
        {"add_id": 1, "drop_id": 100, "net_sgp_delta": 5.0, "add_name": "A", "drop_name": "D1"},
        {"add_id": 2, "drop_id": 100, "net_sgp_delta": 4.5, "add_name": "B", "drop_name": "D1"},
        {"add_id": 3, "drop_id": 101, "net_sgp_delta": 4.0, "add_name": "C", "drop_name": "D2"},
        {"add_id": 4, "drop_id": 100, "net_sgp_delta": 3.5, "add_name": "D", "drop_name": "D1"},
        {"add_id": 5, "drop_id": 102, "net_sgp_delta": 3.0, "add_name": "E", "drop_name": "D3"},
    ]
    result = _deduplicate_and_limit(swaps, max_moves=5)
    add_ids = {r["add_id"] for r in result}
    drop_ids = {r["drop_id"] for r in result}
    # Should surface at least 3 unique adds across 3 unique drops
    assert len(result) >= 3
    assert len(add_ids) == len(result)  # each add used at most once
    assert len(drop_ids) == len(result)  # each drop used at most once


def test_dedup_diversifies_when_top_swaps_share_drop():
    """5 swaps all sharing the same drop should reduce to 1 rec (no fallback
    since no alternative drops exist)."""
    swaps = [
        {
            "add_id": i,
            "drop_id": 100,
            "net_sgp_delta": 5.0 - i * 0.1,
            "add_name": f"FA{i}",
            "drop_name": "OnlyDrop",
        }
        for i in range(1, 6)
    ]
    result = _deduplicate_and_limit(swaps, max_moves=5)
    assert len(result) == 1

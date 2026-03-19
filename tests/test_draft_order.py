# tests/test_draft_order.py
import pytest

from src.draft_order import format_draft_order, generate_draft_order


def test_length_matches_input():
    names = ["A", "B", "C", "D"]
    assert len(generate_draft_order(names)) == 4


def test_contains_all_names():
    names = ["Alpha", "Beta", "Gamma"]
    result = generate_draft_order(names)
    assert set(result) == set(names)


def test_seed_reproducible():
    names = [f"Team {i}" for i in range(12)]
    assert generate_draft_order(names, seed=42) == generate_draft_order(names, seed=42)


def test_different_seeds_differ():
    names = [f"Team {i}" for i in range(12)]
    assert generate_draft_order(names, seed=1) != generate_draft_order(names, seed=2)


def test_single_team():
    assert generate_draft_order(["Solo"]) == ["Solo"]


def test_empty_list():
    assert generate_draft_order([]) == []


def test_format_numbered():
    order = ["A", "B", "C"]
    result = format_draft_order(order)
    assert "1. A" in result
    assert "2. B" in result
    assert "3. C" in result


def test_format_empty():
    assert format_draft_order([]) == ""

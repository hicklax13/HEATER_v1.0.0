# src/draft_order.py
"""Randomized draft order generator."""

from __future__ import annotations

import random


def generate_draft_order(team_names: list[str], seed: int | None = None) -> list[str]:
    """Return a shuffled copy of team_names. Seed for reproducibility."""
    if not team_names:
        return []
    rng = random.Random(seed)
    order = list(team_names)
    rng.shuffle(order)
    return order


def format_draft_order(order: list[str]) -> str:
    """Format as numbered list string."""
    if not order:
        return ""
    return "\n".join(f"{i}. {name}" for i, name in enumerate(order, 1))

"""Shared UI constants and helpers used by all pages."""

THEME = {
    "bg": "#0a0e1a",
    "card": "#1a1f2e",
    "card_h": "#252b3b",
    "amber": "#f59e0b",
    "amber_l": "#fbbf24",
    "teal": "#06b6d4",
    "ok": "#84cc16",
    "danger": "#f43f5e",
    "warn": "#fb923c",
    "tx": "#f0f0f0",
    "tx2": "#8b95a5",
    "tiers": [
        "#f59e0b",
        "#fbbf24",
        "#84cc16",
        "#06b6d4",
        "#8b5cf6",
        "#f97316",
        "#f43f5e",
        "#6b7280",
    ],
}

ROSTER_CONFIG = {
    "C": 1,
    "1B": 1,
    "2B": 1,
    "3B": 1,
    "SS": 1,
    "OF": 3,
    "Util": 2,
    "SP": 2,
    "RP": 2,
    "P": 4,
    "BN": 5,
}

HITTING_CATEGORIES = ["R", "HR", "RBI", "SB", "AVG"]
PITCHING_CATEGORIES = ["W", "SV", "K", "ERA", "WHIP"]
ALL_CATEGORIES = HITTING_CATEGORIES + PITCHING_CATEGORIES

T = THEME  # shorthand for f-strings

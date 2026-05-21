"""Basic news sentiment scoring for draft context.

Uses simple keyword matching -- no external NLP services required.
Provides a sentiment signal from -1.0 (very negative) to +1.0 (very positive)
based on common fantasy baseball news patterns.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

# ── Suspension parsing (P5d) ─────────────────────────────────────────
# Yahoo's news pipeline stores free-form text like "Suspended through 5/26"
# or "Suspended for 6 games". These patterns let us extract a structured
# `expected_return_days` value that `_il_weight_from_status` can feed
# into the return-date curve (instead of treating any suspension as
# season-ending). See `parse_suspension_days` below.
_SUSPENSION_THROUGH_PATTERN = re.compile(
    r"suspen[sd]\w*\s+through\s+(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?",
    re.IGNORECASE,
)
_SUSPENSION_N_GAMES_PATTERN = re.compile(
    r"suspen[sd]\w*\s+for\s+(\d+)\s+games?",
    re.IGNORECASE,
)


def parse_suspension_days(text: str | None, reference_date: datetime | None = None) -> int | None:
    """Parse a suspension end-date or duration from news headline text.

    Returns the number of days from ``reference_date`` until the player
    returns, clamped to >= 0 (in-the-past dates return 0). Returns None
    when no suspension pattern matches — caller falls through to other
    return-date sources (ESPN, status string).

    Patterns recognized:
      - 'Suspended through MM/DD' or 'MM/DD/YY' → date-based
      - 'Suspended for N games' → N-day estimate (typical 1-per-day cadence)
    """
    if not text or not isinstance(text, str):
        return None
    if reference_date is None:
        reference_date = datetime.now(UTC)

    # Try 'through DATE' pattern first (more specific than game count).
    m = _SUSPENSION_THROUGH_PATTERN.search(text)
    if m:
        try:
            month = int(m.group(1))
            day = int(m.group(2))
            year_str = m.group(3)
            year = int(year_str) if year_str else reference_date.year
            if year < 100:
                year += 2000
            end_date = datetime(year, month, day, tzinfo=UTC)
            delta = (end_date - reference_date).total_seconds() / 86400
            return max(0, int(delta))
        except (ValueError, AttributeError):
            pass

    # Try 'for N games' pattern (1 game/day approximation).
    m = _SUSPENSION_N_GAMES_PATTERN.search(text)
    if m:
        try:
            games = int(m.group(1))
            if 0 < games <= 365:
                return games
        except (ValueError, IndexError):
            pass

    return None


# ── Keyword Dictionaries ─────────────────────────────────────────────

# Positive signals (spring training performance, role upgrades)
POSITIVE_KEYWORDS: list[str] = [
    "breakout",
    "impressive",
    "dominant",
    "locked in",
    "everyday",
    "starting",
    "cleanup",
    "leadoff",
    "closer role",
    "promotion",
    "healthy",
    "full go",
    "ahead of schedule",
    "extension",
    "callup",
    "batting first",
    "power surge",
    "ace",
    "opening day",
    "no restrictions",
]

# Negative signals (injuries, role downgrades)
NEGATIVE_KEYWORDS: list[str] = [
    "injured",
    "IL stint",
    "setback",
    "surgery",
    "demotion",
    "platoon",
    "timeshare",
    "bullpen",
    "moved to bench",
    "limited",
    "sore",
    "tightness",
    "strain",
    "fracture",
    "shut down",
    "day-to-day",
    "out indefinitely",
    "torn",
    "inflammation",
    "optioned",
]

# High-impact keywords that count double
HIGH_IMPACT_POSITIVE: set[str] = {"breakout", "dominant", "closer role", "extension", "ace"}
HIGH_IMPACT_NEGATIVE: set[str] = {"surgery", "torn", "out indefinitely", "shut down", "fracture"}


# ── Data Classes ─────────────────────────────────────────────────────


@dataclass
class SentimentResult:
    """Structured sentiment analysis result."""

    score: float  # -1.0 to +1.0
    positive_count: int
    negative_count: int
    high_impact_flags: list[str]
    confidence: float  # 0.0 to 1.0 based on signal count


# ── Public API ───────────────────────────────────────────────────────


def compute_news_sentiment(news_items: list[str]) -> float:
    """Score sentiment from recent news items.

    Args:
        news_items: List of news headlines/blurbs about a player.

    Returns:
        Score from -1.0 (very negative) to +1.0 (very positive).
        Returns 0.0 when no news available.
    """
    if not news_items:
        return 0.0

    positive_count = 0
    negative_count = 0

    news_items = [item for item in news_items if isinstance(item, str)]
    if not news_items:
        return 0.0

    for item in news_items:
        item_lower = item.lower()
        for kw in POSITIVE_KEYWORDS:
            if kw in item_lower:
                if kw in HIGH_IMPACT_POSITIVE:
                    positive_count += 2
                else:
                    positive_count += 1
        for kw in NEGATIVE_KEYWORDS:
            if kw in item_lower:
                if kw in HIGH_IMPACT_NEGATIVE:
                    negative_count += 2
                else:
                    negative_count += 1

    total = positive_count + negative_count
    if total == 0:
        return 0.0

    raw_score = (positive_count - negative_count) / total
    return max(-1.0, min(1.0, raw_score))


def analyze_news_sentiment(news_items: list[str]) -> SentimentResult:
    """Detailed sentiment analysis with high-impact flag detection.

    More detailed than compute_news_sentiment() -- returns structured
    result with hit counts and high-impact keyword flags.

    Args:
        news_items: List of news headlines/blurbs about a player.

    Returns:
        SentimentResult with score, counts, flags, and confidence.
    """
    if not news_items:
        return SentimentResult(
            score=0.0,
            positive_count=0,
            negative_count=0,
            high_impact_flags=[],
            confidence=0.0,
        )

    news_items = [item for item in news_items if isinstance(item, str)]
    if not news_items:
        return SentimentResult(
            score=0.0,
            positive_count=0,
            negative_count=0,
            high_impact_flags=[],
            confidence=0.0,
        )

    positive_count = 0
    negative_count = 0
    high_impact_flags: list[str] = []

    for item in news_items:
        item_lower = item.lower()

        for kw in POSITIVE_KEYWORDS:
            if kw in item_lower:
                if kw in HIGH_IMPACT_POSITIVE:
                    positive_count += 2  # Double weight
                    high_impact_flags.append(f"+{kw}")
                else:
                    positive_count += 1

        for kw in NEGATIVE_KEYWORDS:
            if kw in item_lower:
                if kw in HIGH_IMPACT_NEGATIVE:
                    negative_count += 2  # Double weight
                    high_impact_flags.append(f"-{kw}")
                else:
                    negative_count += 1

    total = positive_count + negative_count
    if total == 0:
        return SentimentResult(
            score=0.0,
            positive_count=0,
            negative_count=0,
            high_impact_flags=high_impact_flags,
            confidence=0.0,
        )

    raw_score = (positive_count - negative_count) / total
    score = max(-1.0, min(1.0, raw_score))

    # Confidence scales with number of signals (caps at 1.0 for 10+ signals)
    confidence = min(1.0, total / 10.0)

    return SentimentResult(
        score=score,
        positive_count=positive_count,
        negative_count=negative_count,
        high_impact_flags=high_impact_flags,
        confidence=confidence,
    )


def sentiment_adjustment(score: float, weight: float = 0.05) -> float:
    """Convert sentiment score to a multiplicative adjustment factor.

    Maps [-1, +1] sentiment to a small multiplicative adjustment
    suitable for modifying draft value (pick_score).

    Args:
        score: Sentiment score from compute_news_sentiment().
        weight: Maximum adjustment magnitude. Default 0.05 means
            sentiment can adjust value by up to +/-5%.

    Returns:
        Multiplicative factor in [1.0 - weight, 1.0 + weight].
    """
    score = max(-1.0, min(1.0, score))
    return 1.0 + score * weight


def batch_sentiment(player_news: dict[int, list[str]]) -> dict[int, float]:
    """Compute sentiment scores for multiple players.

    Args:
        player_news: Dict mapping player_id to list of news items.

    Returns:
        Dict mapping player_id to sentiment score.
    """
    results: dict[int, float] = {}
    for player_id, news_items in player_news.items():
        results[player_id] = compute_news_sentiment(news_items)
    return results

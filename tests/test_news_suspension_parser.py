"""P5d: parse suspension end-dates from news text into expected_return_days."""

from datetime import UTC, datetime


def test_parse_suspension_through_date():
    """'Suspended through 5/26' → days from now to 5/26 (positive when in future)."""
    from src.news_sentiment import parse_suspension_days

    # Test that the function exists and handles a 'through DATE' pattern
    days = parse_suspension_days("Suspended through 5/26", reference_date=datetime(2026, 5, 20, tzinfo=UTC))
    # 5/20 to 5/26 = 6 days
    assert days == 6


def test_parse_suspension_for_n_games_no_date():
    """'Suspended for 6 games' without a specific date → estimate days
    based on typical 1-game-per-day cadence."""
    from src.news_sentiment import parse_suspension_days

    days = parse_suspension_days("Suspended for 6 games", reference_date=datetime(2026, 5, 20, tzinfo=UTC))
    # 6 games ≈ 6 days (approximation)
    assert 5 <= days <= 7


def test_parse_suspension_days_returns_none_when_no_match():
    """Non-suspension text → None (no parsing)."""
    from src.news_sentiment import parse_suspension_days

    assert parse_suspension_days("Hit a home run") is None
    assert parse_suspension_days("Placed on IL") is None  # IL not suspension
    assert parse_suspension_days("") is None
    assert parse_suspension_days(None) is None


def test_parse_suspension_already_returned():
    """When the through-date is in the past, return 0 (already back)."""
    from src.news_sentiment import parse_suspension_days

    days = parse_suspension_days("Suspended through 5/15", reference_date=datetime(2026, 5, 20, tzinfo=UTC))
    assert days == 0  # past date — back already

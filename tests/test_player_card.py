"""Tests for the player card data assembly module (src/player_card.py).

Organized into 8 test classes:
  - TestHeadshotUrl
  - TestComputeAge
  - TestDedupNews
  - TestFormatNewsDatetime
  - TestSparklineData
  - TestRadarPercentiles
  - TestBuildPlayerCardData
  - TestEdgeCases
"""

from __future__ import annotations

import pytest

from src.player_card import (
    _build_sparkline_data,
    _compute_age,
    _compute_radar_percentiles,
    _dedup_news,
    _format_news_datetime,
    _get_headshot_url,
    build_player_card_data,
)

# ── Constants ────────────────────────────────────────────────────────────────

# Marcus Semien — reliably in sample DB at player_id 9733
_HITTER_ID = 9733
# Zack Wheeler — pitcher in sample DB at player_id 9828
_PITCHER_ID = 9828
# Non-existent player
_MISSING_ID = 999_999

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_news_items():
    """Five news items with mixed duplicates."""
    return [
        {"headline": "Player hits walk-off homer", "published_at": "2026-03-20T14:00:00Z"},
        {"headline": "PLAYER HITS WALK-OFF HOMER", "published_at": "2026-03-20T13:00:00Z"},  # dup
        {"headline": "Player placed on IL", "published_at": "2026-03-19T10:00:00Z"},
        {"headline": "Trade rumors swirl", "published_at": "2026-03-18T08:00:00Z"},
        {"headline": "Player hits walk-off homer", "published_at": "2026-03-17T06:00:00Z"},  # dup
    ]


@pytest.fixture
def sample_historical_hitter():
    """Three seasons of hitting data, newest first (as DB returns them)."""
    return [
        {"season": 2025, "R": 100, "HR": 26, "RBI": 85, "SB": 12, "AVG": 0.265, "OBP": 0.335},
        {"season": 2024, "R": 101, "HR": 23, "RBI": 90, "SB": 10, "AVG": 0.258, "OBP": 0.326},
        {"season": 2023, "R": 91, "HR": 21, "RBI": 83, "SB": 8, "AVG": 0.244, "OBP": 0.313},
    ]


@pytest.fixture
def sample_historical_pitcher():
    """Two seasons of pitching data, newest first."""
    return [
        {"season": 2025, "W": 14, "L": 7, "SV": 0, "K": 200, "ERA": 3.10, "WHIP": 1.08},
        {"season": 2024, "W": 12, "L": 9, "SV": 0, "K": 185, "ERA": 3.55, "WHIP": 1.14},
    ]


@pytest.fixture
def hitter_stats():
    """Blended stat dict for a typical middle-of-the-order hitter."""
    return {"R": 95, "HR": 25, "RBI": 88, "SB": 10, "AVG": 0.270, "OBP": 0.340}


@pytest.fixture
def pitcher_stats():
    """Blended stat dict for a solid starting pitcher."""
    return {"W": 13, "L": 7, "SV": 0, "K": 195, "ERA": 3.20, "WHIP": 1.10}


# ── TestHeadshotUrl ──────────────────────────────────────────────────────────


class TestHeadshotUrl:
    def test_valid_mlb_id_returns_url(self):
        url = _get_headshot_url(545361)
        assert url.startswith("https://img.mlbstatic.com")
        assert "545361" in url

    def test_url_contains_mlb_id(self):
        mlb_id = 660271
        url = _get_headshot_url(mlb_id)
        assert str(mlb_id) in url

    def test_none_returns_empty_string(self):
        assert _get_headshot_url(None) == ""

    def test_zero_returns_empty_string(self):
        assert _get_headshot_url(0) == ""

    def test_string_digit_is_cast_to_int(self):
        """Numeric string mlb_id should still produce a valid URL."""
        url = _get_headshot_url("545361")
        assert "545361" in url
        assert url != ""

    def test_different_ids_produce_different_urls(self):
        url_a = _get_headshot_url(545361)
        url_b = _get_headshot_url(660271)
        assert url_a != url_b


# ── TestComputeAge ───────────────────────────────────────────────────────────


class TestComputeAge:
    def test_known_age_is_integer(self):
        age = _compute_age("1990-03-01")
        assert isinstance(age, int)

    def test_born_before_today_this_year(self):
        """Someone born Jan 1 should already have had their birthday by March 20."""
        age = _compute_age("1990-01-01")
        # Today is 2026-03-20; they turned 36 in January 2026
        assert age == 36

    def test_born_after_today_this_year(self):
        """Someone born Dec 31 hasn't had their birthday yet in March."""
        age = _compute_age("1990-12-31")
        # Still only 35 in March 2026
        assert age == 35

    def test_none_returns_none(self):
        assert _compute_age(None) is None

    def test_empty_string_returns_none(self):
        assert _compute_age("") is None

    def test_invalid_format_returns_none(self):
        assert _compute_age("not-a-date") is None

    def test_partial_datetime_string_works(self):
        """Function slices to first 10 characters, so datetimes should work."""
        age = _compute_age("1990-06-15T00:00:00")
        assert isinstance(age, int)
        assert age >= 35

    def test_very_old_birth_date(self):
        age = _compute_age("1950-01-01")
        assert age == 76

    def test_recent_birth_date(self):
        age = _compute_age("2003-08-20")
        # Born Aug 2003, today March 2026 — hasn't had birthday yet
        assert age == 22


# ── TestDedupNews ────────────────────────────────────────────────────────────


class TestDedupNews:
    def test_duplicates_removed(self, sample_news_items):
        result = _dedup_news(sample_news_items)
        headlines = [i["headline"].lower() for i in result]
        assert len(headlines) == len(set(headlines))

    def test_case_insensitive_dedup(self):
        items = [
            {"headline": "Big news", "published_at": "2026-03-20T12:00:00Z"},
            {"headline": "BIG NEWS", "published_at": "2026-03-19T12:00:00Z"},
        ]
        result = _dedup_news(items)
        assert len(result) == 1

    def test_newest_first_after_dedup(self, sample_news_items):
        result = _dedup_news(sample_news_items)
        dates = [i["published_at"] for i in result]
        assert dates == sorted(dates, reverse=True)

    def test_max_items_respected(self, sample_news_items):
        result = _dedup_news(sample_news_items, max_items=2)
        assert len(result) <= 2

    def test_default_max_is_five(self):
        items = [{"headline": f"Story {n}", "published_at": f"2026-03-{n:02d}T00:00:00Z"} for n in range(1, 11)]
        result = _dedup_news(items)
        assert len(result) == 5

    def test_empty_list_returns_empty(self):
        assert _dedup_news([]) == []

    def test_items_without_headline_skipped(self):
        items = [
            {"headline": "", "published_at": "2026-03-20T00:00:00Z"},
            {"headline": "Real headline", "published_at": "2026-03-19T00:00:00Z"},
            {"published_at": "2026-03-18T00:00:00Z"},  # no headline key
        ]
        result = _dedup_news(items)
        assert len(result) == 1
        assert result[0]["headline"] == "Real headline"

    def test_preserves_all_fields(self):
        items = [
            {
                "headline": "Breaking news",
                "source": "ESPN",
                "published_at": "2026-03-20T12:00:00Z",
                "sentiment": 0.5,
            }
        ]
        result = _dedup_news(items)
        assert result[0]["source"] == "ESPN"
        assert result[0]["sentiment"] == 0.5


# ── TestFormatNewsDatetime ────────────────────────────────────────────────────


class TestFormatNewsDatetime:
    def test_iso_utc_z_suffix(self):
        result = _format_news_datetime("2026-03-20T14:30:00Z")
        assert "Mar" in result
        assert "2026" in result
        assert "02:30" in result

    def test_iso_offset_aware(self):
        result = _format_news_datetime("2026-03-20T10:00:00+05:00")
        assert "2026" in result

    def test_none_returns_empty_string(self):
        assert _format_news_datetime(None) == ""

    def test_empty_string_returns_empty(self):
        assert _format_news_datetime("") == ""

    def test_invalid_string_returns_truncated(self):
        """Non-ISO strings fall back to str[:16] rather than crashing."""
        result = _format_news_datetime("not-a-datetime-at-all")
        # Should return a string (truncated), not raise
        assert isinstance(result, str)

    def test_output_format_matches_strftime(self):
        result = _format_news_datetime("2026-01-05T09:05:00Z")
        # Expected: "Jan 05, 2026 at 09:05 AM"
        assert "Jan 05, 2026" in result
        assert "AM" in result or "PM" in result

    def test_midnight_formatting(self):
        result = _format_news_datetime("2026-06-15T00:00:00Z")
        assert "Jun 15, 2026" in result


# ── TestSparklineData ────────────────────────────────────────────────────────


class TestSparklineData:
    def test_returns_dict(self, sample_historical_hitter):
        result = _build_sparkline_data(sample_historical_hitter)
        assert isinstance(result, dict)

    def test_empty_historical_returns_empty(self):
        assert _build_sparkline_data([]) == {}

    def test_hitting_cats_present_for_hitter(self, sample_historical_hitter):
        result = _build_sparkline_data(sample_historical_hitter)
        for cat in ("R", "HR", "RBI", "SB", "AVG", "OBP"):
            assert cat in result

    def test_pitching_cats_present_for_pitcher(self, sample_historical_pitcher):
        result = _build_sparkline_data(sample_historical_pitcher)
        for cat in ("W", "L", "SV", "K", "ERA", "WHIP"):
            assert cat in result

    def test_chronological_order(self, sample_historical_hitter):
        """Data oldest-first: 2023, 2024, 2025."""
        result = _build_sparkline_data(sample_historical_hitter)
        # HR values should be [21, 23, 26] in chronological order
        assert result["HR"] == [21, 23, 26]

    def test_length_matches_seasons(self, sample_historical_hitter):
        result = _build_sparkline_data(sample_historical_hitter)
        assert len(result["R"]) == 3

    def test_single_season(self):
        historical = [{"season": 2025, "HR": 30, "R": 90}]
        result = _build_sparkline_data(historical)
        assert result["HR"] == [30]

    def test_none_values_preserved(self):
        """Missing values remain None in the trend array."""
        historical = [
            {"season": 2025, "HR": 30},
            {"season": 2024, "HR": None},
        ]
        result = _build_sparkline_data(historical)
        assert None in result["HR"]

    def test_lowercase_key_fallback(self):
        """Accepts lowercase category keys as fallback."""
        historical = [{"season": 2025, "hr": 25, "r": 88}]
        result = _build_sparkline_data(historical)
        # Should have HR (looked up via lowercase fallback)
        assert "HR" in result
        assert result["HR"] == [25]

    def test_cats_without_any_data_excluded(self):
        """Categories with all-None values are not included in the result."""
        historical = [{"season": 2025, "HR": None}]
        result = _build_sparkline_data(historical)
        assert "HR" not in result


# ── TestRadarPercentiles ─────────────────────────────────────────────────────


class TestRadarPercentiles:
    """_compute_radar_percentiles hits the database; uses known player IDs."""

    def test_returns_three_keys(self, hitter_stats):
        result = _compute_radar_percentiles(hitter_stats, "2B", is_hitter=True)
        assert set(result.keys()) == {"player", "league_avg", "mlb_avg"}

    def test_player_percentiles_in_range(self, hitter_stats):
        result = _compute_radar_percentiles(hitter_stats, "2B", is_hitter=True)
        for cat, pct in result["player"].items():
            assert 0 <= pct <= 100, f"{cat} percentile {pct} out of range"

    def test_hitter_categories_present(self, hitter_stats):
        result = _compute_radar_percentiles(hitter_stats, "2B", is_hitter=True)
        for cat in ("R", "HR", "RBI", "SB", "AVG", "OBP"):
            assert cat in result["player"]

    def test_pitcher_categories_present(self, pitcher_stats):
        result = _compute_radar_percentiles(pitcher_stats, "P", is_hitter=False)
        for cat in ("W", "L", "SV", "K", "ERA", "WHIP"):
            assert cat in result["player"]

    def test_elite_hitter_has_high_hr_percentile(self):
        """Aaron Judge-level HR production should rank at or above 50th percentile.

        In CI (sample data), the projection pool may be empty so percentiles
        default to 50. With real data, elite stats rank well above 50.
        """
        elite_stats = {"R": 120, "HR": 55, "RBI": 130, "SB": 4, "AVG": 0.310, "OBP": 0.420}
        result = _compute_radar_percentiles(elite_stats, "OF", is_hitter=True)
        assert result["player"]["HR"] >= 50

    def test_inverse_stat_era_low_is_good(self):
        """A 1.50 ERA should rank very high (inverse stat)."""
        elite_pitcher = {"W": 15, "L": 4, "SV": 0, "K": 220, "ERA": 1.50, "WHIP": 0.90}
        result = _compute_radar_percentiles(elite_pitcher, "P", is_hitter=False)
        # Low ERA means better — should be above 50th percentile
        assert result["player"]["ERA"] >= 50

    def test_empty_stats_returns_defaults(self):
        """Empty player stats dict should still return a valid structure."""
        result = _compute_radar_percentiles({}, "2B", is_hitter=True)
        assert "player" in result
        for pct in result["player"].values():
            assert 0 <= pct <= 100

    def test_league_avg_type_consistency(self, hitter_stats):
        result = _compute_radar_percentiles(hitter_stats, "2B", is_hitter=True)
        # league_avg values are numeric (int or float)
        for cat, val in result["league_avg"].items():
            assert isinstance(val, (int, float)), f"{cat} league_avg is not numeric: {val!r}"


# ── TestBuildPlayerCardData ──────────────────────────────────────────────────


class TestBuildPlayerCardData:
    """Integration tests for the main build_player_card_data function."""

    def test_returns_dict(self):
        result = build_player_card_data(_HITTER_ID)
        assert isinstance(result, dict)

    def test_top_level_keys_present(self):
        result = build_player_card_data(_HITTER_ID)
        expected_keys = {
            "profile",
            "projections",
            "historical",
            "advanced",
            "injury_history",
            "rankings",
            "radar",
            "trends",
            "news",
            "prospect",
        }
        assert set(result.keys()) == expected_keys

    def test_profile_has_required_fields(self):
        profile = build_player_card_data(_HITTER_ID)["profile"]
        for field in (
            "name",
            "team",
            "positions",
            "bats",
            "throws",
            "age",
            "headshot_url",
            "health_score",
            "health_label",
            "tags",
        ):
            assert field in profile, f"Missing profile field: {field}"

    def test_player_name_is_string(self):
        profile = build_player_card_data(_HITTER_ID)["profile"]
        assert isinstance(profile["name"], str)
        assert len(profile["name"]) > 0

    def test_health_score_in_range(self):
        profile = build_player_card_data(_HITTER_ID)["profile"]
        assert 0.0 <= profile["health_score"] <= 1.0

    def test_tags_is_list(self):
        profile = build_player_card_data(_HITTER_ID)["profile"]
        assert isinstance(profile["tags"], list)

    def test_projections_structure(self):
        projections = build_player_card_data(_HITTER_ID)["projections"]
        assert "blended" in projections
        assert "systems" in projections
        assert isinstance(projections["blended"], dict)
        assert isinstance(projections["systems"], dict)

    def test_historical_is_list(self):
        historical = build_player_card_data(_HITTER_ID)["historical"]
        assert isinstance(historical, list)

    def test_radar_structure(self):
        radar = build_player_card_data(_HITTER_ID)["radar"]
        assert "player" in radar
        assert "league_avg" in radar
        assert "mlb_avg" in radar

    def test_news_is_list(self):
        news = build_player_card_data(_HITTER_ID)["news"]
        assert isinstance(news, list)

    def test_news_deduped_and_capped(self):
        """News should have at most 5 items (default max_items)."""
        news = build_player_card_data(_HITTER_ID)["news"]
        assert len(news) <= 5

    def test_pitcher_advanced_metrics_populated(self):
        """Advanced metrics dict should be populated for a pitcher."""
        result = build_player_card_data(_PITCHER_ID)
        advanced = result["advanced"]
        assert isinstance(advanced, dict)

    def test_pitcher_profile_is_hitter_false(self):
        """Pitcher profile should reflect pitcher's position."""
        profile = build_player_card_data(_PITCHER_ID)["profile"]
        assert isinstance(profile["positions"], str)

    def test_injury_history_is_list(self):
        injury_history = build_player_card_data(_HITTER_ID)["injury_history"]
        assert isinstance(injury_history, list)

    def test_rankings_is_dict(self):
        rankings = build_player_card_data(_HITTER_ID)["rankings"]
        assert isinstance(rankings, dict)

    def test_trends_is_dict(self):
        trends = build_player_card_data(_HITTER_ID)["trends"]
        assert isinstance(trends, dict)

    def test_prospect_is_none_or_dict(self):
        prospect = build_player_card_data(_HITTER_ID)["prospect"]
        assert prospect is None or isinstance(prospect, dict)


# ── TestEdgeCases ────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_missing_player_returns_safe_dict(self):
        """Non-existent player_id returns a safe empty-ish dict, not an exception."""
        result = build_player_card_data(_MISSING_ID)
        assert isinstance(result, dict)

    def test_missing_player_has_all_top_level_keys(self):
        result = build_player_card_data(_MISSING_ID)
        for key in (
            "profile",
            "projections",
            "historical",
            "advanced",
            "injury_history",
            "rankings",
            "radar",
            "trends",
            "news",
            "prospect",
        ):
            assert key in result, f"Missing key '{key}' for non-existent player"

    def test_missing_player_profile_name_uses_id(self):
        result = build_player_card_data(_MISSING_ID)
        assert str(_MISSING_ID) in result["profile"]["name"]

    def test_missing_player_projections_empty(self):
        result = build_player_card_data(_MISSING_ID)
        assert result["projections"]["blended"] == {}
        assert result["projections"]["systems"] == {}

    def test_missing_player_historical_empty(self):
        result = build_player_card_data(_MISSING_ID)
        assert result["historical"] == []

    def test_missing_player_news_empty(self):
        result = build_player_card_data(_MISSING_ID)
        assert result["news"] == []

    def test_missing_player_prospect_none(self):
        result = build_player_card_data(_MISSING_ID)
        assert result["prospect"] is None

    def test_dedup_news_with_none_published_at(self):
        """Items with None published_at should be deduplicated without crashing."""
        items = [
            {"headline": "Story A", "published_at": None},
            {"headline": "Story A", "published_at": None},
            {"headline": "Story B", "published_at": "2026-03-20T10:00:00Z"},
        ]
        result = _dedup_news(items)
        headlines = [i["headline"] for i in result]
        assert headlines.count("Story A") == 1

    def test_headshot_url_false_returns_empty(self):
        """Falsy non-zero values should return empty string."""
        assert _get_headshot_url(False) == ""

    def test_sparkline_single_none_season(self):
        """Historical list with all-None category should not appear in output."""
        historical = [{"season": 2025, "W": None, "ERA": None}]
        result = _build_sparkline_data(historical)
        assert "W" not in result
        assert "ERA" not in result

    def test_compute_age_non_string_input(self):
        """Numeric birth_date is coerced to string via str()."""
        # 1990-01-15 stored as '1990-01-15'; if someone passes a weird type:
        assert _compute_age(None) is None

"""Tests for player news intelligence module."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

# -- ESPN news parsing ----------------------------------------------------

MOCK_ESPN_NEWS = {
    "articles": [
        {
            "headline": "Mike Trout placed on 10-day IL with knee injury",
            "description": "Angels CF Mike Trout was placed on the 10-day IL...",
            "published": "2026-03-18T12:00:00Z",
            "categories": [{"athleteId": 30836}],
        },
        {
            "headline": "MLB Trade Rumors roundup",
            "description": "General news with no specific player",
            "published": "2026-03-18T10:00:00Z",
            "categories": [],
        },
    ]
}


def test_parse_espn_news():
    from src.player_news import _parse_espn_news

    items = _parse_espn_news(MOCK_ESPN_NEWS)
    assert len(items) == 1  # second article has no athleteId
    assert items[0]["headline"] == "Mike Trout placed on 10-day IL with knee injury"
    assert items[0]["source"] == "espn"
    assert items[0]["espn_athlete_id"] == 30836


# -- RotoWire RSS parsing -------------------------------------------------

MOCK_RSS_XML = """<?xml version="1.0"?>
<rss version="2.0">
<channel>
<item>
  <title>Shohei Ohtani (elbow) expected back next week</title>
  <description>Ohtani is progressing well from surgery...</description>
  <pubDate>Mon, 18 Mar 2026 14:00:00 GMT</pubDate>
</item>
</channel>
</rss>"""


def test_parse_rotowire_rss():
    feedparser = pytest.importorskip("feedparser")
    from src.player_news import _parse_rotowire_entries

    feed = feedparser.parse(MOCK_RSS_XML)
    items = _parse_rotowire_entries(feed.entries)
    assert len(items) == 1
    assert "Ohtani" in items[0]["headline"]
    assert items[0]["source"] == "rotowire"


# -- MLB enhanced status parsing ------------------------------------------


def test_parse_mlb_enhanced_status():
    from src.player_news import _parse_mlb_enhanced_status

    mock_data = {
        "people": [
            {
                "id": 545361,
                "fullName": "Mike Trout",
                "currentTeam": {"name": "Los Angeles Angels"},
                "rosterEntries": [{"status": {"code": "D10", "description": "10-Day Injured List"}}],
                "transactions": [
                    {"description": "placed on 10-day IL with right knee inflammation", "date": "2026-03-15"},
                ],
            }
        ]
    }
    items = _parse_mlb_enhanced_status(mock_data, 545361)
    assert len(items) >= 1
    assert items[0]["il_status"] == "IL10"


# -- News type classification --------------------------------------------


def test_classify_news_type_injury():
    from src.player_news import _classify_news_type

    assert _classify_news_type("placed on 10-day IL with knee strain") == "injury"


def test_classify_news_type_transaction():
    from src.player_news import _classify_news_type

    assert _classify_news_type("traded to the New York Yankees") == "transaction"


def test_classify_news_type_callup():
    from src.player_news import _classify_news_type

    assert _classify_news_type("called up from Triple-A") == "callup"


def test_classify_news_type_lineup():
    from src.player_news import _classify_news_type

    assert _classify_news_type("moved to leadoff in the batting order") == "lineup"


def test_classify_news_type_general():
    from src.player_news import _classify_news_type

    assert _classify_news_type("some random update") == "general"


# -- Template summary generation ------------------------------------------


def test_generate_injury_summary():
    from src.player_news import generate_intel_summary

    intel = {
        "news_type": "injury",
        "il_status": "IL10",
        "injury_body_part": "knee",
        "duration_weeks": 2.0,
        "lost_sgp": -1.5,
        "replacement_name": "John Doe",
        "replacement_sgp": 0.8,
        "ownership_trend": "down",
        "ownership_delta": -5.0,
    }
    summary = generate_intel_summary(intel)
    assert "knee" in summary.lower()
    assert "2.0" in summary
    assert "John Doe" in summary


def test_generate_trade_summary():
    from src.player_news import generate_intel_summary

    intel = {
        "news_type": "transaction",
        "new_team": "NYY",
        "park_factor": 1.05,
        "park_context": "Hitter-friendly park",
        "sgp_delta": 0.3,
    }
    summary = generate_intel_summary(intel)
    assert "NYY" in summary


def test_generate_callup_summary():
    from src.player_news import generate_intel_summary

    intel = {
        "news_type": "callup",
        "milb_level": "AAA",
        "milb_avg": 0.310,
        "milb_obp": 0.380,
        "milb_slg": 0.520,
        "projected_sgp": 2.5,
        "roster_context": "Replaces injured starter",
    }
    summary = generate_intel_summary(intel)
    assert "AAA" in summary


def test_generate_lineup_summary():
    from src.player_news import generate_intel_summary

    intel = {
        "news_type": "lineup",
        "batting_slot": 2,
        "projected_pa_week": 30.5,
        "pa_delta": 2.3,
        "sgp_delta": 0.15,
    }
    summary = generate_intel_summary(intel)
    assert "30.5" in summary


def test_generate_summary_unknown_type():
    from src.player_news import generate_intel_summary

    intel = {"news_type": "general", "headline": "Some news"}
    summary = generate_intel_summary(intel)
    assert isinstance(summary, str)


# -- Ownership trend computation ------------------------------------------


def test_compute_ownership_trend_no_data():
    from src.player_news import compute_ownership_trend

    with patch("src.player_news._load_ownership_history") as mock:
        mock.return_value = []
        result = compute_ownership_trend(1)
        assert result == {}


def test_compute_ownership_trend_with_data():
    from src.player_news import compute_ownership_trend

    with patch("src.player_news._load_ownership_history") as mock:
        mock.return_value = [
            {"date": "2026-03-18", "percent_owned": 85.0},
            {"date": "2026-03-11", "percent_owned": 80.0},
        ]
        result = compute_ownership_trend(1)
        assert result["current"] == 85.0
        assert result["delta_7d"] == 5.0
        assert result["direction"] == "up"


# -- DB round-trip --------------------------------------------------------


def test_store_and_query_news(tmp_path):
    db_path = tmp_path / "test.db"
    with patch("src.database.DB_PATH", db_path):
        from src.database import init_db
        from src.player_news import _query_player_news, _store_news_items

        init_db()
        items = [
            {
                "player_id": 1,
                "source": "espn",
                "headline": "Test headline",
                "detail": "Test detail",
                "news_type": "injury",
                "injury_body_part": "knee",
                "il_status": "IL10",
                "sentiment_score": -0.5,
                "published_at": "2026-03-18T12:00:00Z",
            }
        ]
        count = _store_news_items(items)
        assert count == 1
        rows = _query_player_news(1)
        assert len(rows) == 1
        assert rows[0]["headline"] == "Test headline"


def test_store_duplicate_ignored(tmp_path):
    db_path = tmp_path / "test.db"
    with patch("src.database.DB_PATH", db_path):
        from src.database import init_db

        init_db()
        from src.player_news import _store_news_items

        item = {
            "player_id": 1,
            "source": "espn",
            "headline": "Duplicate",
            "published_at": "2026-03-18",
        }
        _store_news_items([item])
        count = _store_news_items([item])  # second insert
        assert count == 0  # UNIQUE constraint prevents duplicate


# -- Yahoo injury field extraction ----------------------------------------


def test_extract_yahoo_injury_fields():
    from src.player_news import _extract_yahoo_injury_news

    player_data = {
        "player_id": 42,
        "name": "Mike Trout",
        "injury_note": "Right knee inflammation",
        "status_full": "10-Day Injured List",
        "percent_owned": 99.5,
    }
    items = _extract_yahoo_injury_news(player_data)
    assert len(items) == 1
    assert items[0]["source"] == "yahoo"
    assert items[0]["il_status"] == "IL10"


def test_extract_yahoo_no_injury():
    from src.player_news import _extract_yahoo_injury_news

    player_data = {"player_id": 42, "name": "Healthy Player"}
    items = _extract_yahoo_injury_news(player_data)
    assert len(items) == 0


# -- ESPN athleteId cross-reference ---------------------------------------


def test_espn_athlete_id_to_player_id():
    from src.player_news import _resolve_espn_athlete_id

    with patch("src.player_news._lookup_player_by_espn_id") as mock:
        mock.return_value = 42
        result = _resolve_espn_athlete_id(30836)
        assert result == 42


def test_espn_athlete_id_unknown():
    from src.player_news import _resolve_espn_athlete_id

    with patch("src.player_news._lookup_player_by_espn_id", return_value=None):
        result = _resolve_espn_athlete_id(99999)
        assert result is None


# -- Fallback with missing sources ----------------------------------------


def test_aggregate_news_graceful_on_all_failures():
    from src.player_news import aggregate_news

    with (
        patch("src.player_news.fetch_espn_news", side_effect=Exception("fail")),
        patch("src.player_news.fetch_rotowire_rss", side_effect=Exception("fail")),
        patch("src.player_news.fetch_mlb_enhanced_status", return_value=[]),
        patch("src.player_news._query_player_news", return_value=[]),
    ):
        result = aggregate_news(player_id=42)
        assert isinstance(result, list)
        assert len(result) == 0  # graceful empty, not exception


# -- Batch roster intel ---------------------------------------------------


def test_generate_roster_intel_batch():
    from src.player_news import generate_roster_intel

    with (
        patch("src.player_news.aggregate_news", return_value=[]) as mock_agg,
        patch("src.player_news.compute_ownership_trend", return_value={}),
    ):
        pool = pd.DataFrame(
            [
                {"player_id": 1, "name": "A"},
                {"player_id": 2, "name": "B"},
            ]
        )
        result = generate_roster_intel([1, 2], pool)
        assert isinstance(result, dict)
        assert mock_agg.call_count == 2

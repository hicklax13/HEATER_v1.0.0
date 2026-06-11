"""The schema card lists real tables/columns so the model targets valid SQL."""

from src.database import init_db


def test_schema_card_lists_core_tables():
    init_db()
    from src.ai.schema_card import build_schema_card

    card = build_schema_card()
    assert "players" in card
    assert "league_standings" in card
    # columns are included so the model doesn't hallucinate them
    assert "CREATE TABLE" in card or "(" in card


def test_schema_card_excludes_secret_tables():
    init_db()
    from src.ai.schema_card import build_schema_card

    card = build_schema_card()
    # never expose the key store or auth tokens to the model's SQL surface
    assert "ai_provider_keys" not in card
    assert "auth_tokens" not in card

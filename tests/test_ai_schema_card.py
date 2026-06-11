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
    from src.ai.schema_card import build_schema_card, excluded_tables

    card = build_schema_card()
    # Assert each excluded table's OWN definition is absent. We check the
    # "CREATE TABLE <name>" prefix (not a bare name substring) because excluded
    # table names legitimately appear inside other tables' FK clauses, e.g.
    # "REFERENCES users(user_id)" — that is a reference, not a schema leak.
    for name in excluded_tables():
        assert f"CREATE TABLE {name} " not in card, f"{name} schema must not leak"
        assert f"CREATE TABLE IF NOT EXISTS {name} " not in card, f"{name} schema must not leak"
    # explicit spot-checks on the most sensitive tables
    assert "CREATE TABLE ai_provider_keys" not in card
    assert "encrypted_key" not in card  # the key store's columns must never appear


def test_schema_card_excludes_chat_meta_tables():
    """The AI cannot read other users' conversations/usage via query_data."""
    init_db()
    from src.ai.schema_card import build_schema_card, excluded_tables

    card = build_schema_card()
    for t in ("ai_conversations", "ai_messages", "ai_usage_ledger", "forced_refresh_queue"):
        assert t in excluded_tables()
        assert f"CREATE TABLE {t}" not in card and f"CREATE TABLE IF NOT EXISTS {t}" not in card

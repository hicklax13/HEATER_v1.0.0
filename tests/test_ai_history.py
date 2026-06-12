"""Conversation CRUD, scoped to user_id."""

import pytest

from src.database import get_connection, init_db


@pytest.fixture(autouse=True)
def _db():
    init_db()
    conn = get_connection()
    try:
        conn.execute("DELETE FROM ai_messages")
        conn.execute("DELETE FROM ai_conversations")
        conn.commit()
    finally:
        conn.close()


def test_create_and_list():
    from src.ai.history import create_conversation, list_conversations

    cid = create_conversation(user_id=99, title="Trade talk", model="anthropic/claude-haiku-4-5")
    convos = list_conversations(99)
    assert len(convos) == 1
    assert convos[0]["id"] == cid
    assert convos[0]["title"] == "Trade talk"


def test_append_and_load_messages():
    from src.ai.history import append_message, create_conversation, load_messages

    cid = create_conversation(99, "x", "m")
    append_message(cid, "user", "hello")
    append_message(cid, "assistant", "hi", tokens_in=5, tokens_out=2, cost_usd=0.001)
    msgs = load_messages(cid)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["content"] == "hi"


def test_scoped_to_user():
    from src.ai.history import create_conversation, list_conversations

    create_conversation(1, "mine", "m")
    create_conversation(2, "theirs", "m")
    assert {c["title"] for c in list_conversations(1)} == {"mine"}


def test_rename_and_delete():
    from src.ai.history import create_conversation, delete_conversation, list_conversations, rename_conversation

    cid = create_conversation(99, "old", "m")
    rename_conversation(cid, "new")
    assert list_conversations(99)[0]["title"] == "new"
    delete_conversation(cid)
    assert list_conversations(99) == []


def test_user_id_scoping_blocks_cross_user_access():
    """Defense-in-depth: passing user_id scopes load/rename/delete to the owner."""
    from src.ai.history import (
        append_message,
        create_conversation,
        delete_conversation,
        list_conversations,
        load_messages,
        rename_conversation,
    )

    cid = create_conversation(1, "owner-only", "m")
    append_message(cid, "user", "secret")

    # a different user cannot read, rename, or delete it when scoped
    assert load_messages(cid, user_id=2) == []
    rename_conversation(cid, "hijacked", user_id=2)
    assert list_conversations(1)[0]["title"] == "owner-only"  # unchanged
    delete_conversation(cid, user_id=2)
    assert len(list_conversations(1)) == 1  # still there

    # the owner can
    assert [m["content"] for m in load_messages(cid, user_id=1)] == ["secret"]
    delete_conversation(cid, user_id=1)
    assert list_conversations(1) == []

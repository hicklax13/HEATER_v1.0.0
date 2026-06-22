"""PromptStore: in-memory fake + the SQLite impl over a tmp api_state.db."""

import pytest

from api.stores.prompt_store import InMemoryPromptStore, SqlitePromptStore


@pytest.fixture(params=["mem", "sqlite"])
def store(request, tmp_path):
    if request.param == "mem":
        return InMemoryPromptStore()
    return SqlitePromptStore(db_path=str(tmp_path / "api_state.db"))


def test_create_then_list_newest_first(store):
    store.create(7, "first", "text one")
    store.create(7, "second", "text two")
    names = [p.name for p in store.list(7)]
    assert names == ["second", "first"]  # newest first


def test_list_is_owner_scoped(store):
    store.create(1, "mine", "a")
    store.create(2, "theirs", "b")
    assert [p.name for p in store.list(1)] == ["mine"]


def test_delete_returns_true_then_false(store):
    p = store.create(5, "n", "t")
    assert store.delete(5, p.id) is True
    assert store.delete(5, p.id) is False  # already gone


def test_delete_is_owner_scoped(store):
    p = store.create(1, "n", "t")
    assert store.delete(2, p.id) is False  # owner 2 can't delete owner 1's prompt
    assert [x.id for x in store.list(1)] == [p.id]  # still there

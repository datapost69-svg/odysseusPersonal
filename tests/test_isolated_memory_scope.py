"""Regression tests for isolated per-chat memory retrieval."""
from types import SimpleNamespace

from src.memory import MemoryManager


def test_session_scoped_load_excludes_owner_wide_and_other_chat_memories(tmp_path):
    manager = MemoryManager(str(tmp_path))
    shared = manager.add_entry("User prefers concise answers", owner="alice")
    first = manager.add_entry("The spaceship is named Wayfarer", owner="alice")
    first["session_id"] = "chat-one"
    second = manager.add_entry("The spaceship is named Endeavour", owner="alice")
    second["session_id"] = "chat-two"
    manager.save([shared, first, second])

    visible = manager.load(owner="alice", session_id="chat-one")

    assert [entry["id"] for entry in visible] == [first["id"]]
    assert all(entry.get("session_id") == "chat-one" for entry in visible)


def test_unscoped_load_preserves_existing_shared_memory_behaviour(tmp_path):
    manager = MemoryManager(str(tmp_path))
    shared = manager.add_entry("User prefers concise answers", owner="alice")
    private = manager.add_entry("Private fact", owner="alice")
    private["session_id"] = "chat-one"
    manager.save([shared, private])

    visible = manager.load(owner="alice")

    assert {entry["id"] for entry in visible} == {shared["id"], private["id"]}

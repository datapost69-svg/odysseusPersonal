"""Regression tests for Ministral's strict tool-turn alternation."""
from src.llm_core import _apply_ministral_role_compat


def _roles(messages):
    return [m["role"] for m in messages]


def test_tool_round_followed_by_user_gets_neutral_assistant_continuation():
    messages = [
        {"role": "system", "content": "Study assistant"},
        {"role": "user", "content": "Create a document"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call12345", "type": "function",
             "function": {"name": "create_document", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "call12345",
         "content": "Permission denied"},
        {"role": "user", "content": "What went wrong?"},
    ]
    actual = _apply_ministral_role_compat(messages, "ministral3-local")
    assert _roles(actual) == [
        "system", "user", "assistant", "tool", "assistant", "user"
    ]
    assert actual[3] == messages[3]
    assert actual[4]["content"].lower().find("succeeded") == -1
    assert actual[5] == messages[4]
    assert messages[4]["role"] == "user"  # original not mutated


def test_normal_tool_continuation_is_unchanged():
    messages = [
        {"role": "user", "content": "Make notes"},
        {"role": "assistant", "tool_calls": [{"id": "call12345"}]},
        {"role": "tool", "tool_call_id": "call12345", "content": "done"},
        {"role": "assistant", "content": "The notes were saved."},
        {"role": "user", "content": "Thanks"},
    ]
    assert _apply_ministral_role_compat(messages, "ministral3-local") == messages


def test_other_models_are_unchanged():
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "tool_calls": [{"id": "call12345"}]},
        {"role": "tool", "tool_call_id": "call12345", "content": "failed"},
        {"role": "user", "content": "retry"},
    ]
    assert _apply_ministral_role_compat(messages, "qwen3.8:27b") is messages

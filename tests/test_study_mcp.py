import asyncio
import json

import pytest

from routes import study_routes


def test_study_mcp_tools_require_injected_owner():
    from mcp_servers import study_server

    with pytest.raises(ValueError, match="authenticated owner"):
        study_server._owner({})


def test_study_mcp_lists_only_authenticated_owner_modules(tmp_path, monkeypatch):
    from mcp_servers import study_server

    monkeypatch.setattr(study_routes, "DATA_DIR", str(tmp_path))
    alice_root = study_routes._owner_root("alice")
    bob_root = study_routes._owner_root("bob")
    study_routes._write_json(alice_root / "modules.json", [{"id": "law", "name": "Public Law"}])
    study_routes._write_json(bob_root / "modules.json", [{"id": "other", "name": "Private Library"}])

    result = asyncio.run(study_server.call_tool("study_list_modules", {"_odysseus_owner": "alice"}))
    payload = json.loads(result[0].text)

    assert payload["modules"] == [{"id": "law", "name": "Public Law"}]
    assert "Private Library" not in result[0].text


def test_study_mcp_search_is_module_scoped(tmp_path, monkeypatch):
    from mcp_servers import study_server

    monkeypatch.setattr(study_routes, "DATA_DIR", str(tmp_path))
    root = study_routes._owner_root("alice")
    modules = [
        {"id": "law", "name": "Public Law"},
        {"id": "history", "name": "History"},
    ]
    study_routes._write_json(root / "modules.json", modules)

    def fake_search(root_arg, modules_arg, query, module_id=None, limit=8):
        assert modules_arg == [modules[0]]
        assert module_id == "law"
        return [{"module": "Public Law", "source": "Lecture 1", "location": "p. 2", "snippet": "Judicial review"}]

    monkeypatch.setattr(study_routes, "_semantic_search", fake_search)
    result = asyncio.run(study_server.call_tool(
        "study_search_material",
        {"_odysseus_owner": "alice", "query": "judicial review", "module_id": "law"},
    ))
    payload = json.loads(result[0].text)
    assert payload["count"] == 1
    assert payload["results"][0]["module"] == "Public Law"


def test_study_mcp_transcript_paginates_timestamped_segments(tmp_path, monkeypatch):
    from mcp_servers import study_server

    monkeypatch.setattr(study_routes, "DATA_DIR", str(tmp_path))
    root = study_routes._owner_root("alice")
    module_id = "law"
    study_routes._write_json(root / "modules.json", [{"id": module_id, "name": "Public Law"}])
    transcript_dir = root / "modules" / module_id / "transcripts"
    transcript_dir.mkdir(parents=True)
    study_routes._write_json(transcript_dir / "lecture-1.json", {
        "id": "lecture-1", "title": "Lecture 1",
        "segments": [
            {"start": 0, "end": 4, "text": "First"},
            {"start": 5, "end": 9, "text": "Second"},
            {"start": 10, "end": 14, "text": "Third"},
        ],
    })

    result = asyncio.run(study_server.call_tool("study_get_transcript", {
        "_odysseus_owner": "alice", "module_id": module_id, "lecture_id": "lecture-1",
        "segment_offset": 1, "segment_limit": 1,
    }))
    payload = json.loads(result[0].text)

    assert payload["segments_total"] == 3
    assert payload["segments"][0]["text"] == "Second"
    assert payload["has_more_segments"] is True


def test_study_mcp_transcript_rejects_invalid_pagination(tmp_path, monkeypatch):
    from mcp_servers import study_server

    monkeypatch.setattr(study_routes, "DATA_DIR", str(tmp_path))
    root = study_routes._owner_root("alice")
    study_routes._write_json(root / "modules.json", [{"id": "law", "name": "Public Law"}])
    transcript_dir = root / "modules" / "law" / "transcripts"
    transcript_dir.mkdir(parents=True)
    study_routes._write_json(transcript_dir / "lecture-1.json", {"id": "lecture-1", "segments": []})

    result = asyncio.run(study_server.call_tool("study_get_transcript", {
        "_odysseus_owner": "alice", "module_id": "law", "lecture_id": "lecture-1",
        "segment_offset": True,
    }))

    assert "segment_offset must be a non-negative integer" in result[0].text




def test_study_mcp_saves_agent_authored_notes_to_the_configured_vault(tmp_path, monkeypatch):
    from mcp_servers import study_server

    monkeypatch.setattr(study_routes, "DATA_DIR", str(tmp_path))
    root = study_routes._owner_root("alice")
    study_routes._write_json(root / "modules.json", [{"id": "law", "name": "Public Law"}])
    vault = tmp_path / "vault"
    (vault / ".obsidian").mkdir(parents=True)
    study_routes._write_json(root / "obsidian.json", {"vault_path": str(vault)})

    markdown = "# Judicial Review\\n\\nThese are notes composed by Odysseus.\\n"
    result = asyncio.run(study_server.call_tool("study_save_notes", {
        "_odysseus_owner": "alice",
        "module_id": "law",
        "title": "../../Seminar: Judicial Review",
        "note_type": "targeted",
        "markdown": markdown,
    }))
    payload = json.loads(result[0].text)

    assert payload["saved"] is True
    saved_path = Path(payload["path"])
    assert saved_path.is_file()
    assert saved_path.parent == vault / "Public Law"
    assert saved_path.read_text(encoding="utf-8") == markdown
    assert saved_path.name.endswith(".md")
    assert "Seminar Judicial Review" in saved_path.name


def test_study_save_notes_does_not_overwrite_existing_notes(tmp_path, monkeypatch):
    monkeypatch.setattr(study_routes, "DATA_DIR", str(tmp_path))
    root = study_routes._owner_root("alice")
    study_routes._write_json(root / "modules.json", [{"id": "law", "name": "Public Law"}])
    vault = tmp_path / "vault"
    (vault / ".obsidian").mkdir(parents=True)
    study_routes._write_json(root / "obsidian.json", {"vault_path": str(vault)})

    payload = {
        "module_id": "law",
        "title": "Lecture Notes",
        "note_type": "general",
        "markdown": "# First version",
    }
    first = study_routes.save_notes_to_vault(payload, "alice")
    payload["markdown"] = "# Second version"
    second = study_routes.save_notes_to_vault(payload, "alice")

    assert Path(first["path"]).read_text(encoding="utf-8") == "# First version\\n"
    assert Path(second["path"]).read_text(encoding="utf-8") == "# Second version\\n"
    assert first["path"] != second["path"]


def test_study_mcp_errors_are_marked_as_tool_failures():
    from mcp_servers import study_server

    result = asyncio.run(study_server.call_tool("unknown_study_tool", {
        "_odysseus_owner": "alice",
    }))

    assert result[0].text.startswith("Error: ")
    assert "Unknown study tool" in result[0].text
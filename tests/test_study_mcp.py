import json

import pytest

from routes import study_routes


def test_study_mcp_tools_require_injected_owner():
    from mcp_servers import study_server

    with pytest.raises(ValueError, match="authenticated owner"):
        study_server._owner({})


@pytest.mark.asyncio
async def test_study_mcp_lists_only_authenticated_owner_modules(tmp_path, monkeypatch):
    from mcp_servers import study_server

    monkeypatch.setattr(study_routes, "DATA_DIR", str(tmp_path))
    alice_root = study_routes._owner_root("alice")
    bob_root = study_routes._owner_root("bob")
    study_routes._write_json(alice_root / "modules.json", [{"id": "law", "name": "Public Law"}])
    study_routes._write_json(bob_root / "modules.json", [{"id": "other", "name": "Private Library"}])

    result = await study_server.call_tool("study_list_modules", {"_odysseus_owner": "alice"})
    payload = json.loads(result[0].text)

    assert payload["modules"] == [{"id": "law", "name": "Public Law"}]
    assert "Private Library" not in result[0].text


@pytest.mark.asyncio
async def test_study_mcp_search_is_module_scoped(tmp_path, monkeypatch):
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
    result = await study_server.call_tool(
        "study_search_material",
        {"_odysseus_owner": "alice", "query": "judicial review", "module_id": "law"},
    )
    payload = json.loads(result[0].text)
    assert payload["count"] == 1
    assert payload["results"][0]["module"] == "Public Law"

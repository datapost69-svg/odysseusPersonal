import json
from pathlib import Path

from fastapi import HTTPException

from routes.study_routes import _owner_root, _read_modules, _write_json


def test_study_library_is_partitioned_by_owner(tmp_path, monkeypatch):
    monkeypatch.setattr("routes.study_routes.DATA_DIR", str(tmp_path))
    alice = _owner_root("alice")
    bob = _owner_root("bob")
    assert alice != bob
    _write_json(alice / "modules.json", [{"id": "a", "name": "History"}])
    assert _read_modules(alice) == [{"id": "a", "name": "History"}]
    assert _read_modules(bob) == []


def test_study_index_write_is_valid_json(tmp_path):
    path = Path(tmp_path) / "modules.json"
    _write_json(path, [{"id": "x", "name": "Politics"}])
    assert json.loads(path.read_text(encoding="utf-8"))[0]["name"] == "Politics"

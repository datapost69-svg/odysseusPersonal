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

def test_local_transcriber_returns_timestamped_segments(monkeypatch, tmp_path):
    import sys
    import types
    from routes.study_routes import _transcribe_local

    class Segment:
        start = 1.25
        end = 2.75
        text = "  A lecture sentence.  "

    class Info:
        language = "en"
        duration = 12.0

    class FakeModel:
        def transcribe(self, path, **kwargs):
            assert Path(path).is_file()
            assert kwargs["vad_filter"] is True
            return iter([Segment()]), Info()

    class FakeWhisperModel:
        def __init__(self, model_name, device, compute_type):
            assert device == "cpu"
            assert compute_type == "int8"
            self.model = FakeModel()

        def transcribe(self, path, **kwargs):
            return self.model.transcribe(path, **kwargs)

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeWhisperModel))
    monkeypatch.delattr(_transcribe_local, "_model", raising=False)
    audio = tmp_path / "lecture.wav"
    audio.write_bytes(b"placeholder")
    result = _transcribe_local(str(audio))
    assert result["language"] == "en"
    assert result["duration"] == 12.0
    assert result["segments"] == [{"start": 1.25, "end": 2.75, "text": "A lecture sentence."}]
    monkeypatch.delattr(_transcribe_local, "_model", raising=False)


def test_semantic_index_is_local_and_returns_source_locations(tmp_path, monkeypatch):
    import numpy as np
    import routes.study_routes as study

    class FakeEmbedder:
        def encode(self, texts, normalize_embeddings=True):
            vectors = []
            for text in texts:
                lowered = text.casefold()
                vector = [1.0, 0.0] if "democracy" in lowered else [0.0, 1.0]
                vectors.append(vector)
            return np.asarray(vectors, dtype="float32")

    monkeypatch.setattr(study, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(study, "_study_embedder", FakeEmbedder())
    root = study._owner_root("student")
    modules = [{"id": "politics", "name": "Politics"}, {"id": "history", "name": "History"}]
    study._index_records(root, "politics", [{
        "source_id": "lecture-1", "source_type": "lecture", "title": "Week 1",
        "location": "lecture 02:10–03:40", "text": "Democracy and political representation",
    }])
    study._index_records(root, "history", [{
        "source_id": "reading-1", "source_type": "document", "title": "Reading A",
        "location": "p. 12", "text": "Medieval trade routes and port cities",
    }])

    hits = study._semantic_search(root, modules, "democracy")
    assert len(hits) == 1
    assert hits[0]["module"] == "Politics"
    assert hits[0]["source"] == "Week 1"
    assert hits[0]["location"] == "lecture 02:10–03:40"
    assert hits[0]["source_type"] == "lecture"
    assert (root / "modules" / "politics" / "semantic_index.json").is_file()
    assert (root / "modules" / "history" / "semantic_index.json").is_file()


def test_semantic_search_can_be_limited_to_one_module(tmp_path, monkeypatch):
    import numpy as np
    import routes.study_routes as study

    class FakeEmbedder:
        def encode(self, texts, normalize_embeddings=True):
            return np.asarray([[1.0, 0.0] for _ in texts], dtype="float32")

    monkeypatch.setattr(study, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(study, "_study_embedder", FakeEmbedder())
    root = study._owner_root("student")
    modules = [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}]
    for module in modules:
        study._index_records(root, module["id"], [{
            "source_id": f"source-{module['id']}", "source_type": "document",
            "title": f"Source {module['id']}", "location": "p. 1", "text": "Shared concept",
        }])
    hits = study._semantic_search(root, modules, "concept", module_id="b")
    assert len(hits) == 1
    assert hits[0]["module_id"] == "b"


def test_semantic_search_backfills_preexisting_source_chunks(tmp_path, monkeypatch):
    import numpy as np
    import routes.study_routes as study

    class FakeEmbedder:
        def encode(self, texts, normalize_embeddings=True):
            return np.asarray([[1.0, 0.0] for _ in texts], dtype="float32")

    monkeypatch.setattr(study, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(study, "_study_embedder", FakeEmbedder())
    root = study._owner_root("student")
    module_id = "old-module"
    source_dir = root / "modules" / module_id / "sources"
    study._write_json(source_dir / "old-source.json", {
        "id": "old-source", "title": "Previously imported reading",
        "chunks": [{"location": "p. 9", "text": "The social contract legitimises political authority."}],
    })
    hits = study._semantic_search(root, [{"id": module_id, "name": "Political Theory"}], "political authority")
    assert hits[0]["source"] == "Previously imported reading"
    assert hits[0]["location"] == "p. 9"
    assert (root / "modules" / module_id / "semantic_index.json").is_file()

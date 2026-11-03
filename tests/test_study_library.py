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
